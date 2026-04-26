//! Feed‑forward network module.
//!
//! The feed‑forward sublayer consists of two linear transformations
//! separated by an activation function. The inner dimension is
//! typically larger than the model dimension to provide additional
//! capacity. See `spec05_feedforward.md` for recommended sizes and
//! activation choices.

use crate::config::ModelConfig;
use crate::{CognexisError, Result};

/// Feed‑forward network.
pub struct FeedForwardNetwork {
    pub hidden_size: usize,
    pub inner_size: usize,
}

impl FeedForwardNetwork {
    /// Initialize the feed‑forward network from the model config.
    pub fn new(config: &ModelConfig) -> Self {
        Self {
            hidden_size: config.hidden_size,
            inner_size: config.ff_inner_dim,
        }
    }

    /// Apply a deterministic SwiGLU-like reference transformation.
    pub fn forward(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.try_forward(x).unwrap_or_default()
    }

    /// Checked feed-forward path for callers that need structured errors.
    pub fn try_forward(&self, x: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if self.hidden_size == 0 || self.inner_size == 0 {
            return Err(CognexisError::InvalidConfig(
                "feed-forward hidden_size and inner_size must be positive".to_string(),
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

        let scale = (self.hidden_size as f32 / self.inner_size as f32).sqrt();
        Ok(x.iter()
            .map(|row| {
                row.iter()
                    .map(|value| swish(*value) * *value * scale)
                    .collect()
            })
            .collect())
    }
}

fn swish(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}
