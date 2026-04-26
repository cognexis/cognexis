//! Configuration definitions for the Cognexis model.
//!
//! This module defines the configuration structures used to parameterize
//! the Cognexis model. These structures mirror the fields described in
//! `spec11_config.md`. They can be serialized and deserialized via
//! `serde` to allow loading from JSON or other formats.

use serde::{Deserialize, Serialize};

/// Global configuration for the Cognexis model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Size of the vocabulary (number of tokens).
    pub vocab_size: usize,
    /// Dimensionality of token embeddings.
    pub hidden_size: usize,
    /// Number of transformer layers in the prelude.
    pub num_prelude_layers: usize,
    /// Number of unique recurrent blocks to iterate.
    pub num_recurrent_blocks: usize,
    /// Number of iterations (loops) during inference.
    pub max_loop_count: usize,
    /// Number of transformer layers in the coda.
    pub num_coda_layers: usize,
    /// Number of attention heads per block.
    pub num_attention_heads: usize,
    /// Dimension of each feed‑forward sublayer.
    pub ff_inner_dim: usize,
    /// Optional path to a tokenizer vocabulary file.
    pub tokenizer_path: Option<String>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50_000,
            hidden_size: 2_048,
            num_prelude_layers: 6,
            num_recurrent_blocks: 1,
            max_loop_count: 16,
            num_coda_layers: 6,
            num_attention_heads: 16,
            ff_inner_dim: 8_192,
            tokenizer_path: None,
        }
    }
}