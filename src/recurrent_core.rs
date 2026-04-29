//! Recurrent core module.
//!
//! The recurrent core contains the block that is iteratively applied
//! during inference to refine the hidden state. It shares weights
//! across iterations. See `spec08_recurrent_core.md` for details on
//! stability, parameter sharing, and halting conditions.

use crate::config::{ModelConfig, RecurrentInputInjection};
use crate::stability::{has_non_finite, rms_norm};
use crate::transformer_block::TransformerBlock;
use crate::{CognexisError, Result};

/// Options controlling recurrent execution and state retention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecurrentOptions {
    pub loops: usize,
    pub retain_intermediate_states: bool,
}

impl Default for RecurrentOptions {
    fn default() -> Self {
        Self {
            loops: 1,
            retain_intermediate_states: false,
        }
    }
}

/// Loop-level recurrent execution stats.
#[derive(Debug, Clone, PartialEq)]
pub struct RecurrentLoopStats {
    pub requested_loops: usize,
    pub loops_executed: usize,
    pub mean_gate: Option<f32>,
    pub input_injection: RecurrentInputInjection,
}

/// Checked recurrent output with diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct RecurrentOutput {
    pub hidden: Vec<Vec<f32>>,
    pub intermediate_states: Vec<Vec<Vec<f32>>>,
    pub stats: RecurrentLoopStats,
}

/// Recurrent core consisting of one or more shared transformer blocks.
pub struct RecurrentCore {
    blocks: Vec<TransformerBlock>,
    pub min_loops: usize,
    pub max_loops: usize,
    pub gating: bool,
    pub input_injection: RecurrentInputInjection,
    pub input_injection_scale: f32,
    norm_epsilon: f32,
}

impl RecurrentCore {
    /// Initialize the recurrent core based on model configuration.
    pub fn new(config: &ModelConfig) -> Self {
        let blocks = (0..config.num_recurrent_blocks)
            .map(|_| TransformerBlock::with_residual_scale(config, config.recurrent_residual_scale))
            .collect();
        Self {
            blocks,
            min_loops: config.min_loop_count,
            max_loops: config.max_loop_count,
            gating: config.recurrent_gating,
            input_injection: config.recurrent_input_injection,
            input_injection_scale: config.recurrent_input_injection_scale,
            norm_epsilon: config.norm_epsilon,
        }
    }

    /// Perform iterative refinement of the hidden state while enforcing
    /// configured loop bounds.
    pub fn forward(&self, x: &[Vec<f32>], loops: usize) -> Vec<Vec<f32>> {
        self.forward_with_options(
            x,
            RecurrentOptions {
                loops,
                retain_intermediate_states: false,
            },
        )
        .map(|output| output.hidden)
        .unwrap_or_default()
    }

    /// Checked recurrent execution with gating, input injection, and diagnostics.
    pub fn forward_with_options(
        &self,
        x: &[Vec<f32>],
        options: RecurrentOptions,
    ) -> Result<RecurrentOutput> {
        validate_hidden(x)?;
        if self.min_loops == 0 || self.max_loops < self.min_loops {
            return Err(CognexisError::InvalidConfig(
                "recurrent loop bounds are invalid".to_string(),
            ));
        }

        let h0 = x.to_owned();
        let mut output = h0.clone();
        let iterations = options.loops.clamp(self.min_loops, self.max_loops);
        let mut intermediate_states = Vec::new();
        let mut gate_sum = 0.0f32;
        let mut gate_count = 0usize;

        for loop_index in 0..iterations {
            let previous = output.clone();
            let mut candidate = previous.clone();
            for block in &self.blocks {
                candidate = block.try_forward(&candidate)?;
            }
            apply_input_injection(
                &mut candidate,
                &h0,
                self.input_injection,
                self.input_injection_scale,
            )?;
            if self.gating {
                let (gated, sum, count) = apply_recurrent_gate(
                    &previous,
                    &candidate,
                    &h0,
                    loop_index,
                    self.input_injection,
                    self.norm_epsilon,
                )?;
                gate_sum += sum;
                gate_count += count;
                output = gated;
            } else {
                output = candidate;
            }

            if has_non_finite(&output) {
                return Err(CognexisError::Backend(format!(
                    "non-finite recurrent activation after loop {}",
                    loop_index + 1
                )));
            }
            if options.retain_intermediate_states {
                intermediate_states.push(output.clone());
            }
        }

        Ok(RecurrentOutput {
            hidden: output,
            intermediate_states,
            stats: RecurrentLoopStats {
                requested_loops: options.loops,
                loops_executed: iterations,
                mean_gate: (gate_count > 0).then_some(gate_sum / gate_count as f32),
                input_injection: self.input_injection,
            },
        })
    }
}

fn validate_hidden(x: &[Vec<f32>]) -> Result<()> {
    if x.iter()
        .any(|row| row.iter().any(|value| !value.is_finite()))
    {
        return Err(CognexisError::Backend(
            "recurrent input contains non-finite values".to_string(),
        ));
    }
    if let Some(width) = x.first().map(Vec::len) {
        if let Some((row_index, row)) = x.iter().enumerate().find(|(_, row)| row.len() != width) {
            return Err(CognexisError::ShapeMismatch {
                expected: format!("row width {width}"),
                actual: format!("row {row_index} width {}", row.len()),
            });
        }
    }
    Ok(())
}

fn apply_input_injection(
    candidate: &mut [Vec<f32>],
    h0: &[Vec<f32>],
    mode: RecurrentInputInjection,
    scale: f32,
) -> Result<()> {
    if mode == RecurrentInputInjection::None || mode == RecurrentInputInjection::GateCondition {
        return Ok(());
    }
    if candidate.len() != h0.len() {
        return Err(CognexisError::ShapeMismatch {
            expected: format!("{} recurrent rows", h0.len()),
            actual: format!("{} candidate rows", candidate.len()),
        });
    }
    for (row_index, (candidate_row, h0_row)) in candidate.iter_mut().zip(h0).enumerate() {
        if candidate_row.len() != h0_row.len() {
            return Err(CognexisError::ShapeMismatch {
                expected: format!("row {row_index} width {}", h0_row.len()),
                actual: format!("row {row_index} width {}", candidate_row.len()),
            });
        }
        for (candidate_value, h0_value) in candidate_row.iter_mut().zip(h0_row) {
            *candidate_value += scale * h0_value;
        }
    }
    Ok(())
}

fn apply_recurrent_gate(
    previous: &[Vec<f32>],
    candidate: &[Vec<f32>],
    h0: &[Vec<f32>],
    loop_index: usize,
    mode: RecurrentInputInjection,
    norm_epsilon: f32,
) -> Result<(Vec<Vec<f32>>, f32, usize)> {
    if previous.len() != candidate.len() || previous.len() != h0.len() {
        return Err(CognexisError::ShapeMismatch {
            expected: format!("{} rows", previous.len()),
            actual: format!("candidate rows {}, h0 rows {}", candidate.len(), h0.len()),
        });
    }

    let mut gate_sum = 0.0f32;
    let mut gate_count = 0usize;
    let mut output = Vec::with_capacity(previous.len());
    for (row_index, ((previous_row, candidate_row), h0_row)) in
        previous.iter().zip(candidate).zip(h0).enumerate()
    {
        if previous_row.len() != candidate_row.len() || previous_row.len() != h0_row.len() {
            return Err(CognexisError::ShapeMismatch {
                expected: format!("row {row_index} width {}", previous_row.len()),
                actual: format!(
                    "candidate width {}, h0 width {}",
                    candidate_row.len(),
                    h0_row.len()
                ),
            });
        }
        let normalized_previous = rms_norm(previous_row, norm_epsilon);
        let normalized_h0 = rms_norm(h0_row, norm_epsilon);
        let mut row = Vec::with_capacity(previous_row.len());
        for dim in 0..previous_row.len() {
            let conditioned = if mode == RecurrentInputInjection::GateCondition {
                0.25 * normalized_h0[dim]
            } else {
                0.0
            };
            let gate_logit =
                0.5 * normalized_previous[dim] + conditioned - 0.05 * loop_index as f32;
            let gate = sigmoid(gate_logit);
            gate_sum += gate;
            gate_count += 1;
            row.push(gate * candidate_row[dim] + (1.0 - gate) * previous_row[dim]);
        }
        output.push(row);
    }

    Ok((output, gate_sum, gate_count))
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}
