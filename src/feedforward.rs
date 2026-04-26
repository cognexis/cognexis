//! Feed‑forward network module.
//!
//! The feed‑forward sublayer consists of two linear transformations
//! separated by an activation function. The inner dimension is
//! typically larger than the model dimension to provide additional
//! capacity. See `spec05_feedforward.md` for recommended sizes and
//! activation choices.

use crate::config::ModelConfig;

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

    /// Apply the feed‑forward network to an input matrix. This skeleton
    /// returns the input unchanged.
    pub fn forward(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // TODO: Implement linear → activation → linear.
        let _ = x;
        x.to_owned()
    }
}