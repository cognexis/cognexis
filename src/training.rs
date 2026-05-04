//! Reference training-step utilities.
//!
//! The CPU reference model does not own mutable tensor parameters, but
//! training smoke tests still need the same batch validation, forward
//! pass, masked loss, loop accounting, and token counters used by a
//! full backend trainer.

use std::collections::BTreeMap;

use crate::config::ModelConfig;
use crate::data_loading::TrainingBatch;
use crate::recurrent_core::RecurrentOptions;
use crate::value_head::gain_targets_from_losses;
use crate::{CognexisError, CognexisModel, Result};

/// Options for one reference training step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainingStepOptions {
    pub loops: usize,
    pub gradient_clip_norm: Option<f32>,
    pub auxiliary_loss_weight: f32,
}

impl Default for TrainingStepOptions {
    fn default() -> Self {
        Self {
            loops: 1,
            gradient_clip_norm: None,
            auxiliary_loss_weight: 0.0,
        }
    }
}

impl TrainingStepOptions {
    pub fn validate(&self, model: &ModelConfig) -> Result<()> {
        if self.loops == 0 {
            return Err(CognexisError::InvalidConfig(
                "training step loops must be positive".to_string(),
            ));
        }
        if self.loops > model.max_loop_count {
            return Err(CognexisError::InvalidConfig(format!(
                "training step loops ({}) exceed model.max_loop_count ({})",
                self.loops, model.max_loop_count
            )));
        }
        if self
            .gradient_clip_norm
            .map(|norm| !norm.is_finite() || norm <= 0.0)
            .unwrap_or(false)
        {
            return Err(CognexisError::InvalidConfig(
                "gradient_clip_norm must be finite and positive when set".to_string(),
            ));
        }
        if !self.auxiliary_loss_weight.is_finite() || self.auxiliary_loss_weight < 0.0 {
            return Err(CognexisError::InvalidConfig(
                "auxiliary_loss_weight must be finite and non-negative".to_string(),
            ));
        }
        Ok(())
    }
}

/// Mean loss observed at one recurrent depth during a training step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainingDepthLoss {
    pub loops: usize,
    pub loss: f32,
}

/// Metrics emitted by a deterministic reference training step.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingStepMetrics {
    pub step: u64,
    pub loss: f32,
    pub active_target_weight: f64,
    pub recurrent_applications: u64,
    pub loops: usize,
    pub gradient_clip_norm: Option<f32>,
    pub auxiliary_loss_weight: f32,
    pub depth_losses: Vec<TrainingDepthLoss>,
    pub value_gain_targets: Vec<f32>,
}

/// Minimal reference trainer for smoke tests and integration harnesses.
pub struct ReferenceTrainer {
    pub model: CognexisModel,
    pub step: u64,
    pub training_tokens_seen: f64,
}

impl ReferenceTrainer {
    pub fn new(config: ModelConfig) -> Result<Self> {
        Ok(Self {
            model: CognexisModel::new(config)?,
            step: 0,
            training_tokens_seen: 0.0,
        })
    }

    /// Run a deterministic forward/loss step and update counters.
    pub fn train_step(
        &mut self,
        batch: &TrainingBatch,
        options: TrainingStepOptions,
    ) -> Result<TrainingStepMetrics> {
        options.validate(&self.model.config)?;
        validate_batch(batch)?;

        let loops = options.loops.clamp(
            self.model.config.min_loop_count,
            self.model.config.max_loop_count,
        );
        let mut weighted_loss = 0.0f64;
        let mut active_target_weight = 0.0f64;
        let mut active_input_tokens = 0u64;
        let mut depth_loss_accumulators: BTreeMap<usize, (f64, f64)> = BTreeMap::new();

        for row_index in 0..batch.batch_size() {
            let input_ids = &batch.input_ids[row_index];
            let target_ids = &batch.target_ids[row_index];
            let loss_mask = &batch.loss_mask[row_index];
            let row_target_weight = loss_mask.iter().map(|value| *value as f64).sum::<f64>();
            active_input_tokens += batch.attention_mask[row_index]
                .iter()
                .filter(|is_real| **is_real)
                .count() as u64;

            if row_target_weight == 0.0 {
                continue;
            }

            let retain_intermediate = batch.loop_metadata[row_index].retain_intermediate_states
                || options.auxiliary_loss_weight > 0.0;
            let row_depth_losses = self.row_depth_losses(
                input_ids,
                target_ids,
                loss_mask,
                loops,
                retain_intermediate,
            )?;
            for depth_loss in &row_depth_losses {
                let entry = depth_loss_accumulators
                    .entry(depth_loss.loops)
                    .or_insert((0.0, 0.0));
                entry.0 += depth_loss.loss as f64 * row_target_weight;
                entry.1 += row_target_weight;
            }
            let final_loss = row_depth_losses
                .last()
                .ok_or_else(|| {
                    CognexisError::Backend("row depth-loss collection was empty".to_string())
                })?
                .loss;
            let auxiliary_loss = if options.auxiliary_loss_weight > 0.0 {
                let auxiliary_depths = row_depth_losses.len().saturating_sub(1);
                if auxiliary_depths == 0 {
                    0.0
                } else {
                    row_depth_losses
                        .iter()
                        .take(auxiliary_depths)
                        .map(|depth_loss| depth_loss.loss)
                        .sum::<f32>()
                        / auxiliary_depths as f32
                }
            } else {
                0.0
            };
            let row_loss = final_loss + options.auxiliary_loss_weight * auxiliary_loss;
            weighted_loss += row_loss as f64 * row_target_weight;
            active_target_weight += row_target_weight;
        }

        if active_target_weight == 0.0 {
            return Err(CognexisError::InvalidConfig(
                "training batch contains no active target tokens".to_string(),
            ));
        }

        self.step += 1;
        self.training_tokens_seen += active_target_weight;
        let depth_losses = depth_loss_accumulators
            .into_iter()
            .filter_map(|(loops, (weighted, weight))| {
                (weight > 0.0).then_some(TrainingDepthLoss {
                    loops,
                    loss: (weighted / weight) as f32,
                })
            })
            .collect::<Vec<_>>();
        let value_gain_targets = gain_targets_from_losses(
            &depth_losses
                .iter()
                .map(|depth_loss| depth_loss.loss)
                .collect::<Vec<_>>(),
        );

        Ok(TrainingStepMetrics {
            step: self.step,
            loss: (weighted_loss / active_target_weight) as f32,
            active_target_weight,
            recurrent_applications: active_input_tokens * loops as u64,
            loops,
            gradient_clip_norm: options.gradient_clip_norm,
            auxiliary_loss_weight: options.auxiliary_loss_weight,
            depth_losses,
            value_gain_targets,
        })
    }

    fn row_depth_losses(
        &self,
        input_ids: &[u32],
        target_ids: &[u32],
        loss_mask: &[f32],
        loops: usize,
        retain_intermediate: bool,
    ) -> Result<Vec<TrainingDepthLoss>> {
        if !retain_intermediate {
            let logits = self.model.forward_logits(input_ids, loops)?;
            let loss = self
                .model
                .lm_head
                .cross_entropy_loss(&logits, target_ids, loss_mask)?;
            return Ok(vec![TrainingDepthLoss { loops, loss }]);
        }

        let embedded = self.model.embeddings.try_forward(input_ids)?;
        let prepared = self.model.prelude.forward(&embedded);
        let recurrent = self.model.recurrent.forward_with_options(
            &prepared,
            RecurrentOptions {
                loops,
                retain_intermediate_states: true,
            },
        )?;
        let mut losses = Vec::with_capacity(recurrent.intermediate_states.len());
        for (index, state) in recurrent.intermediate_states.iter().enumerate() {
            let hidden = self.model.coda.forward(state);
            let logits = self.model.lm_head.try_forward(&hidden)?;
            let loss = self
                .model
                .lm_head
                .cross_entropy_loss(&logits, target_ids, loss_mask)?;
            losses.push(TrainingDepthLoss {
                loops: index + 1,
                loss,
            });
        }
        Ok(losses)
    }
}

fn validate_batch(batch: &TrainingBatch) -> Result<()> {
    let batch_size = batch.batch_size();
    if batch_size == 0 {
        return Err(CognexisError::InvalidConfig(
            "training batch must contain at least one row".to_string(),
        ));
    }
    if batch.target_ids.len() != batch_size
        || batch.loss_mask.len() != batch_size
        || batch.attention_mask.len() != batch_size
        || batch.position_ids.len() != batch_size
        || batch.loop_metadata.len() != batch_size
    {
        return Err(CognexisError::ShapeMismatch {
            expected: format!("{batch_size} rows in every training batch tensor"),
            actual: format!(
                "targets={}, loss_mask={}, attention_mask={}, position_ids={}, loop_metadata={}",
                batch.target_ids.len(),
                batch.loss_mask.len(),
                batch.attention_mask.len(),
                batch.position_ids.len(),
                batch.loop_metadata.len()
            ),
        });
    }

    let seq_len = batch.seq_len();
    for row_index in 0..batch_size {
        let metadata = batch.loop_metadata[row_index];
        if metadata.min_loops == 0 || metadata.max_loops < metadata.min_loops {
            return Err(CognexisError::InvalidConfig(format!(
                "row {row_index} loop metadata must satisfy max_loops >= min_loops >= 1"
            )));
        }
        for (name, len) in [
            ("target_ids", batch.target_ids[row_index].len()),
            ("loss_mask", batch.loss_mask[row_index].len()),
            ("attention_mask", batch.attention_mask[row_index].len()),
            ("position_ids", batch.position_ids[row_index].len()),
        ] {
            if len != seq_len {
                return Err(CognexisError::ShapeMismatch {
                    expected: format!("row {row_index} {name} length {seq_len}"),
                    actual: format!("length {len}"),
                });
            }
        }
        if batch.loss_mask[row_index]
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(CognexisError::InvalidConfig(
                "training loss masks must be finite and non-negative".to_string(),
            ));
        }
    }

    if let Some(document_ids) = &batch.document_ids {
        if document_ids.len() != batch_size {
            return Err(CognexisError::ShapeMismatch {
                expected: format!("{batch_size} document-id rows"),
                actual: format!("{} document-id rows", document_ids.len()),
            });
        }
        for (row_index, row) in document_ids.iter().enumerate() {
            if row.len() != seq_len {
                return Err(CognexisError::ShapeMismatch {
                    expected: format!("row {row_index} document_ids length {seq_len}"),
                    actual: format!("length {}", row.len()),
                });
            }
        }
    }

    Ok(())
}
