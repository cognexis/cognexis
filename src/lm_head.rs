//! Language modeling head module.
//!
//! The language modeling head projects the final hidden state into
//! logits over the vocabulary. It optionally ties weights with the
//! token embedding matrix. See `spec10_lm_head.md` for discussion of
//! weight tying and output normalization.

use crate::config::ModelConfig;
use crate::embedding::reference_embedding_weight_value;
use crate::stability::rms_norm;
use crate::{CognexisError, Result};

/// Language modeling head.
#[derive(Debug, Clone, PartialEq)]
pub struct LMHead {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub tied_to_embeddings: bool,
    pub norm_epsilon: f32,
    pub bias: Option<Vec<f32>>,
}

impl LMHead {
    /// Create a new language modeling head from the model configuration.
    pub fn new(config: &ModelConfig) -> Self {
        Self {
            hidden_size: config.hidden_size,
            vocab_size: config.vocab_size,
            tied_to_embeddings: config.tie_embeddings,
            norm_epsilon: config.norm_epsilon,
            bias: None,
        }
    }

    /// Create an LM head with an explicit vocabulary bias.
    pub fn with_bias(config: &ModelConfig, bias: Vec<f32>) -> Result<Self> {
        if bias.len() != config.vocab_size {
            return Err(CognexisError::ShapeMismatch {
                expected: format!("bias length {}", config.vocab_size),
                actual: format!("bias length {}", bias.len()),
            });
        }
        if bias.iter().any(|value| !value.is_finite()) {
            return Err(CognexisError::InvalidConfig(
                "LM head bias values must be finite".to_string(),
            ));
        }

        Ok(Self {
            bias: Some(bias),
            ..Self::new(config)
        })
    }

    /// Compute deterministic reference logits for each token position.
    pub fn forward(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.try_forward(x).unwrap_or_default()
    }

    /// Checked LM-head path for callers that need structured errors.
    pub fn try_forward(&self, x: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if self.hidden_size == 0 || self.vocab_size == 0 {
            return Err(CognexisError::InvalidConfig(
                "LM head hidden_size and vocab_size must be positive".to_string(),
            ));
        }
        if let Some(bias) = &self.bias {
            if bias.len() != self.vocab_size {
                return Err(CognexisError::ShapeMismatch {
                    expected: format!("bias length {}", self.vocab_size),
                    actual: format!("bias length {}", bias.len()),
                });
            }
        }
        for (row_index, row) in x.iter().enumerate() {
            if row.len() != self.hidden_size {
                return Err(CognexisError::ShapeMismatch {
                    expected: format!("row width {}", self.hidden_size),
                    actual: format!("row {row_index} width {}", row.len()),
                });
            }
            if row.iter().any(|value| !value.is_finite()) {
                return Err(CognexisError::Backend(format!(
                    "LM head input row {row_index} contains non-finite values"
                )));
            }
        }

        let scale = 1.0 / (self.hidden_size as f32).sqrt();
        Ok(x.iter()
            .map(|row| {
                let row = rms_norm(row, self.norm_epsilon);
                (0..self.vocab_size)
                    .map(|token_id| {
                        let dot = row
                            .iter()
                            .enumerate()
                            .map(|(dim, value)| value * self.weight_value(token_id as u32, dim))
                            .sum::<f32>();
                        dot * scale + self.bias_value(token_id)
                    })
                    .collect()
            })
            .collect())
    }

    /// Compute logits only for the last active hidden position.
    pub fn logits_last(&self, x: &[Vec<f32>]) -> Result<Vec<f32>> {
        let last = x.last().ok_or_else(|| CognexisError::ShapeMismatch {
            expected: "at least one hidden row".to_string(),
            actual: "empty hidden sequence".to_string(),
        })?;
        let rows = self.try_forward(&[last.clone()])?;
        rows.into_iter()
            .next()
            .ok_or_else(|| CognexisError::Backend("LM head returned no last logits".to_string()))
    }

    /// Reference masked cross-entropy over materialized logits.
    pub fn cross_entropy_loss(
        &self,
        logits: &[Vec<f32>],
        targets: &[u32],
        loss_mask: &[f32],
    ) -> Result<f32> {
        if logits.len() != targets.len() || logits.len() != loss_mask.len() {
            return Err(CognexisError::ShapeMismatch {
                expected: format!("{} targets and mask values", logits.len()),
                actual: format!(
                    "{} targets and {} mask values",
                    targets.len(),
                    loss_mask.len()
                ),
            });
        }

        let mut weighted_nll = 0.0f64;
        let mut weight_sum = 0.0f64;
        for (row_index, ((row, &target), &mask)) in
            logits.iter().zip(targets).zip(loss_mask).enumerate()
        {
            if !mask.is_finite() || mask < 0.0 {
                return Err(CognexisError::InvalidConfig(
                    "loss mask values must be finite and non-negative".to_string(),
                ));
            }
            if mask == 0.0 {
                continue;
            }
            if row.len() != self.vocab_size {
                return Err(CognexisError::ShapeMismatch {
                    expected: format!("logit row width {}", self.vocab_size),
                    actual: format!("row {row_index} width {}", row.len()),
                });
            }
            if target as usize >= self.vocab_size {
                return Err(CognexisError::InvalidTokenId(target));
            }
            if row.iter().any(|value| !value.is_finite()) {
                return Err(CognexisError::Backend(format!(
                    "logit row {row_index} contains non-finite values"
                )));
            }

            weighted_nll += negative_log_likelihood(row, target as usize) * mask as f64;
            weight_sum += mask as f64;
        }

        if weight_sum == 0.0 {
            return Err(CognexisError::InvalidConfig(
                "loss mask contains no active targets".to_string(),
            ));
        }
        Ok((weighted_nll / weight_sum) as f32)
    }

    /// Fused-reference loss path: project hidden states and compute masked loss.
    pub fn cross_entropy_loss_from_hidden(
        &self,
        hidden: &[Vec<f32>],
        targets: &[u32],
        loss_mask: &[f32],
    ) -> Result<f32> {
        let logits = self.try_forward(hidden)?;
        self.cross_entropy_loss(&logits, targets, loss_mask)
    }

    fn weight_value(&self, token_id: u32, dim: usize) -> f32 {
        if self.tied_to_embeddings {
            reference_embedding_weight_value(token_id, dim)
        } else {
            untied_head_weight_value(token_id, dim)
        }
    }

    fn bias_value(&self, token_id: usize) -> f32 {
        self.bias
            .as_ref()
            .and_then(|bias| bias.get(token_id))
            .copied()
            .unwrap_or(0.0)
    }
}

fn untied_head_weight_value(token_id: u32, dim: usize) -> f32 {
    let mixed = (token_id as u64)
        .wrapping_mul(1_099_511_627_821)
        .wrapping_add((dim as u64).wrapping_mul(1_462_959_810_393_466_560));
    let bucket = ((mixed >> 8) & 0xffff) as f32 / 65_535.0;
    (bucket - 0.5) * 0.2
}

fn negative_log_likelihood(logits: &[f32], target: usize) -> f64 {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let shifted_sum = logits
        .iter()
        .map(|logit| (*logit - max_logit).exp() as f64)
        .sum::<f64>();
    let log_sum_exp = max_logit as f64 + shifted_sum.ln();
    log_sum_exp - logits[target] as f64
}
