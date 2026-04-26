//! Embedding module.
//!
//! The embedding layer maps discrete token IDs to dense vector
//! representations. It also optionally applies positional encoding to
//! capture sequence order. See `spec03_embedding.md` for the rationale
//! and configuration of embedding dimensions and positional encodings.

use crate::config::ModelConfig;

/// Embedding layer.
pub struct Embedding {
    /// The dimensionality of the output vectors.
    pub hidden_size: usize,
    /// The size of the token vocabulary.
    pub vocab_size: usize,
    // In a full implementation these fields would hold the learned
    // embedding table and positional encodings.
}

impl Embedding {
    /// Create a new embedding layer given a model configuration.
    pub fn new(config: &ModelConfig) -> Self {
        Self {
            hidden_size: config.hidden_size,
            vocab_size: config.vocab_size,
        }
    }

    /// Forward pass: convert a slice of token IDs into a matrix of
    /// embeddings. This skeleton returns a zero matrix of the proper
    /// shape.
    pub fn forward(&self, token_ids: &[u32]) -> Vec<Vec<f32>> {
        let seq_len = token_ids.len();
        let mut output = vec![vec![0.0; self.hidden_size]; seq_len];
        // TODO: lookup embeddings and add positional encodings.
        let _ = token_ids;
        output
    }
}