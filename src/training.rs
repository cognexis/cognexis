//! Reference training-step utilities.
//!
//! The CPU reference model does not own mutable tensor parameters, but
//! training smoke tests still need the same batch validation, forward
//! pass, masked loss, loop accounting, and token counters used by a
//! full backend trainer.

use crate::config::ModelConfig;
use crate::data_loading::TrainingBatch;
use crate::{CognexisError, CognexisModel, Result};

/// Options for one reference training step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainingStepOptions {
    pub loops: usize,
    pub gradient_clip_norm: Option<f32>,
}

impl Default for TrainingStepOptions {
    fn default() -> Self {
        Self {
            loops: 1,
            gradient_clip_norm: None,
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
        Ok(())
    }
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

            let logits = self.model.forward_logits(input_ids, loops)?;
            let row_loss = self
                .model
                .lm_head
                .cross_entropy_loss(&logits, target_ids, loss_mask)?;
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

        Ok(TrainingStepMetrics {
            step: self.step,
            loss: (weighted_loss / active_target_weight) as f32,
            active_target_weight,
            recurrent_applications: active_input_tokens * loops as u64,
            loops,
            gradient_clip_norm: options.gradient_clip_norm,
        })
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
