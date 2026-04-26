//! Configuration definitions for the Cognexis model.
//!
//! This module defines the configuration structures used to parameterize
//! the Cognexis model. These structures mirror the fields described in
//! `spec11_config.md`. They can be serialized and deserialized via
//! `serde` to allow loading from JSON or other formats.

use serde::{Deserialize, Serialize};

use crate::{CognexisError, Result};

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
    /// Minimum number of recurrent iterations during inference.
    #[serde(default = "default_min_loop_count")]
    pub min_loop_count: usize,
    /// Number of iterations (loops) during inference.
    pub max_loop_count: usize,
    /// Number of transformer layers in the coda.
    pub num_coda_layers: usize,
    /// Number of attention heads per block.
    pub num_attention_heads: usize,
    /// Number of key/value heads for grouped query attention.
    #[serde(default = "default_num_kv_heads")]
    pub num_kv_heads: usize,
    /// Dimension of each feed‑forward sublayer.
    pub ff_inner_dim: usize,
    /// Epsilon used by RMS normalization.
    #[serde(default = "default_norm_epsilon")]
    pub norm_epsilon: f32,
    /// Residual scale used for recurrent transformer updates.
    #[serde(default = "default_recurrent_residual_scale")]
    pub recurrent_residual_scale: f32,
    /// Whether the LM head should be tied to embeddings in checkpointed models.
    #[serde(default = "default_tie_embeddings")]
    pub tie_embeddings: bool,
    /// Explicit embedding scale applied after lookup.
    #[serde(default = "default_embedding_scale")]
    pub embedding_scale: f32,
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
            min_loop_count: default_min_loop_count(),
            max_loop_count: 16,
            num_coda_layers: 6,
            num_attention_heads: 16,
            num_kv_heads: 16,
            ff_inner_dim: 8_192,
            norm_epsilon: default_norm_epsilon(),
            recurrent_residual_scale: default_recurrent_residual_scale(),
            tie_embeddings: default_tie_embeddings(),
            embedding_scale: default_embedding_scale(),
            tokenizer_path: None,
        }
    }
}

impl ModelConfig {
    /// Validate architecture invariants required by the specification.
    pub fn validate(&self) -> Result<()> {
        if self.vocab_size == 0 {
            return Err(CognexisError::InvalidConfig(
                "vocab_size must be positive".to_string(),
            ));
        }
        if self.hidden_size == 0 {
            return Err(CognexisError::InvalidConfig(
                "hidden_size must be positive".to_string(),
            ));
        }
        if self.num_attention_heads == 0 {
            return Err(CognexisError::InvalidConfig(
                "num_attention_heads must be positive".to_string(),
            ));
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(CognexisError::InvalidConfig(format!(
                "hidden_size ({}) must be divisible by num_attention_heads ({})",
                self.hidden_size, self.num_attention_heads
            )));
        }
        if self.num_kv_heads == 0 {
            return Err(CognexisError::InvalidConfig(
                "num_kv_heads must be positive".to_string(),
            ));
        }
        if self.num_attention_heads % self.num_kv_heads != 0 {
            return Err(CognexisError::InvalidConfig(format!(
                "num_attention_heads ({}) must be divisible by num_kv_heads ({})",
                self.num_attention_heads, self.num_kv_heads
            )));
        }
        if self.num_recurrent_blocks != 1 {
            return Err(CognexisError::InvalidConfig(
                "baseline Cognexis requires num_recurrent_blocks == 1".to_string(),
            ));
        }
        if self.min_loop_count == 0 {
            return Err(CognexisError::InvalidConfig(
                "min_loop_count must be at least 1".to_string(),
            ));
        }
        if self.max_loop_count < self.min_loop_count {
            return Err(CognexisError::InvalidConfig(format!(
                "max_loop_count ({}) must be >= min_loop_count ({})",
                self.max_loop_count, self.min_loop_count
            )));
        }
        if self.ff_inner_dim == 0 {
            return Err(CognexisError::InvalidConfig(
                "ff_inner_dim must be positive".to_string(),
            ));
        }
        if !self.norm_epsilon.is_finite() || self.norm_epsilon <= 0.0 {
            return Err(CognexisError::InvalidConfig(
                "norm_epsilon must be finite and positive".to_string(),
            ));
        }
        if !self.recurrent_residual_scale.is_finite() || self.recurrent_residual_scale < 0.0 {
            return Err(CognexisError::InvalidConfig(
                "recurrent_residual_scale must be finite and non-negative".to_string(),
            ));
        }
        if !self.embedding_scale.is_finite() || self.embedding_scale <= 0.0 {
            return Err(CognexisError::InvalidConfig(
                "embedding_scale must be finite and positive".to_string(),
            ));
        }
        Ok(())
    }

    /// Attention head dimension derived from the hidden size and query head count.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

const fn default_min_loop_count() -> usize {
    1
}

const fn default_num_kv_heads() -> usize {
    1
}

const fn default_norm_epsilon() -> f32 {
    1.0e-5
}

const fn default_recurrent_residual_scale() -> f32 {
    0.5
}

const fn default_tie_embeddings() -> bool {
    true
}

const fn default_embedding_scale() -> f32 {
    1.0
}
