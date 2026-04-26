//! Transformer block module.
//!
//! A transformer block combines multi‑head attention and a feed‑forward
//! network with layer normalization and residual connections. See
//! `spec06_transformer_block.md` for discussion of normalization
//! strategies (RMSNorm), residual scaling, and block composition.

use crate::attention::MultiHeadAttention;
use crate::config::ModelConfig;
use crate::feedforward::FeedForwardNetwork;

/// A single transformer block.
pub struct TransformerBlock {
    attention: MultiHeadAttention,
    feedforward: FeedForwardNetwork,
}

impl TransformerBlock {
    /// Create a new transformer block based on the model config.
    pub fn new(config: &ModelConfig) -> Self {
        Self {
            attention: MultiHeadAttention::new(config),
            feedforward: FeedForwardNetwork::new(config),
        }
    }

    /// Apply the transformer block to the input tensor. This method
    /// combines attention and feed‑forward layers with residual
    /// connections. The skeleton returns the input unchanged.
    pub fn forward(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // TODO: Implement attention + residual + feedforward + residual.
        let _ = x;
        x.to_owned()
    }
}