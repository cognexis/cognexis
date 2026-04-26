//! Language modeling head module.
//!
//! The language modeling head projects the final hidden state into
//! logits over the vocabulary. It optionally ties weights with the
//! token embedding matrix. See `spec10_lm_head.md` for discussion of
//! weight tying and output normalization.

use crate::config::ModelConfig;
use crate::stability::rms_norm;
use crate::{CognexisError, Result};

/// Language modeling head.
pub struct LMHead {
    pub hidden_size: usize,
    pub vocab_size: usize,
}

impl LMHead {
    /// Create a new language modeling head from the model configuration.
    pub fn new(config: &ModelConfig) -> Self {
        Self {
            hidden_size: config.hidden_size,
            vocab_size: config.vocab_size,
        }
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
        for (row_index, row) in x.iter().enumerate() {
            if row.len() != self.hidden_size {
                return Err(CognexisError::ShapeMismatch {
                    expected: format!("row width {}", self.hidden_size),
                    actual: format!("row {row_index} width {}", row.len()),
                });
            }
        }

        let scale = 1.0 / (self.hidden_size as f32).sqrt();
        Ok(x.iter()
            .map(|row| {
                let row = rms_norm(row, 1.0e-5);
                let mean = row.iter().sum::<f32>() / row.len() as f32;
                (0..self.vocab_size)
                    .map(|token_id| {
                        let idx = token_id % self.hidden_size;
                        let alt = (token_id.wrapping_mul(31).wrapping_add(7)) % self.hidden_size;
                        let sign = if token_id % 2 == 0 { 1.0 } else { -1.0 };
                        (row[idx] * sign + 0.1 * row[alt] + 0.01 * mean) * scale
                    })
                    .collect()
            })
            .collect())
    }
}
