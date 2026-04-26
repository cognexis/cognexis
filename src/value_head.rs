//! Value head module.
//!
//! In instruction‑tuned models it is common to attach a scalar value
//! head to predict rewards or confidence scores. This can be used for
//! DEI‑based scheduling, reinforcement learning, or fine‑grained
//! selection. See `spec19_value_head.md` for more details.

use crate::config::ModelConfig;

/// A simple linear value head mapping hidden states to scalar values.
pub struct ValueHead {
    pub hidden_size: usize,
}

impl ValueHead {
    pub fn new(config: &ModelConfig) -> Self {
        Self {
            hidden_size: config.hidden_size,
        }
    }

    /// Predict a scalar value for each token position. This stub
    /// returns zeros.
    pub fn forward(&self, x: &[Vec<f32>]) -> Vec<f32> {
        vec![0.0; x.len()]
    }
}