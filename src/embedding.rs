//! Embedding module.
//!
//! The embedding layer maps discrete token IDs to dense vector
//! representations. It also optionally applies positional encoding to
//! capture sequence order. See `spec03_embedding.md` for the rationale
//! and configuration of embedding dimensions and positional encodings.

use crate::config::ModelConfig;
use crate::{CognexisError, Result};

/// Embedding layer.
pub struct Embedding {
    /// The dimensionality of the output vectors.
    pub hidden_size: usize,
    /// The size of the token vocabulary.
    pub vocab_size: usize,
    /// Explicit scale applied after lookup.
    pub embedding_scale: f32,
}

impl Embedding {
    /// Create a new embedding layer given a model configuration.
    pub fn new(config: &ModelConfig) -> Self {
        Self {
            hidden_size: config.hidden_size,
            vocab_size: config.vocab_size,
            embedding_scale: config.embedding_scale,
        }
    }

    /// Forward pass: convert token IDs into deterministic reference
    /// embeddings with sinusoidal position information.
    pub fn forward(&self, token_ids: &[u32]) -> Vec<Vec<f32>> {
        self.try_forward(token_ids).unwrap_or_default()
    }

    /// Checked embedding path for callers that need structured errors.
    pub fn try_forward(&self, token_ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        if self.hidden_size == 0 || self.vocab_size == 0 {
            return Err(CognexisError::InvalidConfig(
                "embedding hidden_size and vocab_size must be positive".to_string(),
            ));
        }

        token_ids
            .iter()
            .enumerate()
            .map(|(position, &token_id)| {
                if token_id as usize >= self.vocab_size {
                    return Err(CognexisError::InvalidTokenId(token_id));
                }

                Ok((0..self.hidden_size)
                    .map(|dim| {
                        (token_feature(token_id, dim)
                            + position_feature(position, dim, self.hidden_size))
                            * self.embedding_scale
                    })
                    .collect())
            })
            .collect()
    }
}

fn token_feature(token_id: u32, dim: usize) -> f32 {
    let mixed = (token_id as u64)
        .wrapping_mul(1_315_423_911)
        .wrapping_add((dim as u64).wrapping_mul(2_654_435_761));
    let bucket = (mixed & 0xffff) as f32 / 65_535.0;
    (bucket - 0.5) * 0.2
}

fn position_feature(position: usize, dim: usize, hidden_size: usize) -> f32 {
    let pair = dim / 2;
    let exponent = (2 * pair) as f32 / hidden_size.max(1) as f32;
    let angle = position as f32 / 10_000_f32.powf(exponent);
    let value = if dim % 2 == 0 {
        angle.sin()
    } else {
        angle.cos()
    };
    value * 0.01
}
