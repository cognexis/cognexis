//! Attention module.
//!
//! This module contains the definitions for the multi‑head attention
//! mechanism used throughout Cognexis. The attention design is based
//! on scaled dot‑product attention with support for recurrent looping.
//! See `spec04_attention.md` for details on attention scaling, head
//! grouping (GQA), and masking strategies.

use crate::config::ModelConfig;
use crate::{CognexisError, Result};

/// Multi‑head attention layer.
pub struct MultiHeadAttention {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of grouped key/value heads.
    pub num_kv_heads: usize,
    /// Hidden size of the model.
    pub hidden_size: usize,
    /// Per-head dimension.
    pub head_dim: usize,
}

impl MultiHeadAttention {
    /// Initialize a new multi‑head attention layer from the model
    /// configuration.
    pub fn new(config: &ModelConfig) -> Self {
        Self {
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            hidden_size: config.hidden_size,
            head_dim: config.hidden_size / config.num_attention_heads.max(1),
        }
    }

    /// Apply causal scaled dot-product attention to query, key, and
    /// value tensors using identity projections.
    pub fn forward(&self, q: &[Vec<f32>], k: &[Vec<f32>], v: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.try_forward(q, k, v).unwrap_or_default()
    }

    /// Checked attention path for callers that need structured errors.
    pub fn try_forward(
        &self,
        q: &[Vec<f32>],
        k: &[Vec<f32>],
        v: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>> {
        self.validate_shapes(q, k, v)?;
        if q.is_empty() {
            return Ok(Vec::new());
        }

        let mut output = vec![vec![0.0; self.hidden_size]; q.len()];
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let decode_offset = k.len().saturating_sub(q.len());

        for (query_index, query) in q.iter().enumerate() {
            let absolute_query_index = decode_offset + query_index;
            let visible_keys = (absolute_query_index + 1).min(k.len());

            for head in 0..self.num_heads {
                let start = head * self.head_dim;
                let end = start + self.head_dim;
                let mut scores = Vec::with_capacity(visible_keys);

                for key in &k[..visible_keys] {
                    let dot = query[start..end]
                        .iter()
                        .zip(&key[start..end])
                        .map(|(a, b)| a * b)
                        .sum::<f32>();
                    scores.push(dot * scale);
                }

                let weights = stable_softmax(&scores);
                for (weight, value) in weights.iter().zip(&v[..visible_keys]) {
                    for dim in start..end {
                        output[query_index][dim] += weight * value[dim];
                    }
                }
            }
        }

        Ok(output)
    }

    fn validate_shapes(&self, q: &[Vec<f32>], k: &[Vec<f32>], v: &[Vec<f32>]) -> Result<()> {
        if self.num_heads == 0 || self.hidden_size == 0 {
            return Err(CognexisError::InvalidConfig(
                "attention heads and hidden size must be positive".to_string(),
            ));
        }
        if self.hidden_size % self.num_heads != 0 {
            return Err(CognexisError::InvalidConfig(format!(
                "hidden_size ({}) must be divisible by num_heads ({})",
                self.hidden_size, self.num_heads
            )));
        }
        if self.num_kv_heads == 0 || self.num_heads % self.num_kv_heads != 0 {
            return Err(CognexisError::InvalidConfig(format!(
                "num_heads ({}) must be divisible by num_kv_heads ({})",
                self.num_heads, self.num_kv_heads
            )));
        }
        if k.len() != v.len() {
            return Err(CognexisError::ShapeMismatch {
                expected: format!("k and v sequence lengths equal, got k={}", k.len()),
                actual: format!("v={}", v.len()),
            });
        }
        if !q.is_empty() && k.is_empty() {
            return Err(CognexisError::ShapeMismatch {
                expected: "non-empty key/value sequence for non-empty query".to_string(),
                actual: "empty key/value sequence".to_string(),
            });
        }

        for (name, matrix) in [("q", q), ("k", k), ("v", v)] {
            if let Some((row_index, row)) = matrix
                .iter()
                .enumerate()
                .find(|(_, row)| row.len() != self.hidden_size)
            {
                return Err(CognexisError::ShapeMismatch {
                    expected: format!("{name} row width {}", self.hidden_size),
                    actual: format!("row {row_index} width {}", row.len()),
                });
            }
        }

        Ok(())
    }
}

fn stable_softmax(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }

    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_scores = Vec::with_capacity(scores.len());
    let mut sum = 0.0;
    for score in scores {
        let exp = (score - max_score).exp();
        exp_scores.push(exp);
        sum += exp;
    }

    if sum == 0.0 || !sum.is_finite() {
        return vec![1.0 / scores.len() as f32; scores.len()];
    }

    exp_scores.into_iter().map(|value| value / sum).collect()
}
