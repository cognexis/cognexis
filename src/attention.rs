//! Attention module.
//!
//! This module contains the definitions for the multi‑head attention
//! mechanism used throughout Cognexis. The attention design is based
//! on scaled dot‑product attention with support for recurrent looping.
//! See `spec04_attention.md` for details on attention scaling, head
//! grouping (GQA), and masking strategies.

use crate::config::ModelConfig;
use crate::{CognexisError, Result};

/// Explicit attention execution context for masks, positions, and mode.
#[derive(Debug, Clone, Copy)]
pub struct AttentionContext<'a> {
    /// Real-key mask where `true` means a key/value position may be attended.
    pub key_padding_mask: Option<&'a [bool]>,
    /// Absolute or logical positions for query rows.
    pub query_position_ids: Option<&'a [u32]>,
    /// Absolute or logical positions for key rows.
    pub key_position_ids: Option<&'a [u32]>,
    /// Packed-document IDs for query rows.
    pub query_document_ids: Option<&'a [u32]>,
    /// Packed-document IDs for key rows.
    pub key_document_ids: Option<&'a [u32]>,
    /// Active query mask used by token-wise scheduling diagnostics.
    pub active_query_mask: Option<&'a [bool]>,
    /// Whether to apply decoder-only causal masking.
    pub enforce_causal: bool,
    /// Absolute offset of query row 0 when query is a suffix/decode chunk.
    pub decode_offset: Option<usize>,
    /// Whether this is a training call. The reference path has no dropout.
    pub training: bool,
}

impl<'a> Default for AttentionContext<'a> {
    fn default() -> Self {
        Self {
            key_padding_mask: None,
            query_position_ids: None,
            key_position_ids: None,
            query_document_ids: None,
            key_document_ids: None,
            active_query_mask: None,
            enforce_causal: true,
            decode_offset: None,
            training: false,
        }
    }
}

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
    /// Whether to apply rotary position encoding to Q/K heads.
    pub rope_enabled: bool,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// Maximum supported position ID without extrapolation.
    pub max_position_embeddings: usize,
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
            rope_enabled: config.rope_enabled,
            rope_theta: config.rope_theta,
            max_position_embeddings: config.max_position_embeddings,
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
        self.try_forward_with_context(q, k, v, AttentionContext::default())
    }

    /// Checked attention path with explicit masks and position context.
    pub fn try_forward_with_context(
        &self,
        q: &[Vec<f32>],
        k: &[Vec<f32>],
        v: &[Vec<f32>],
        context: AttentionContext<'_>,
    ) -> Result<Vec<Vec<f32>>> {
        self.validate_shapes(q, k, v)?;
        validate_context(q.len(), k.len(), context)?;
        if q.is_empty() {
            return Ok(Vec::new());
        }

        let mut output = vec![vec![0.0; self.hidden_size]; q.len()];
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let decode_offset = context
            .decode_offset
            .unwrap_or_else(|| k.len().saturating_sub(q.len()));

        for (query_index, query) in q.iter().enumerate() {
            if context
                .active_query_mask
                .map(|mask| !mask[query_index])
                .unwrap_or(false)
            {
                continue;
            }

            let query_position = query_position(query_index, decode_offset, context);
            self.validate_position_id(query_position)?;
            let query_document_id = query_document_id(query_index, decode_offset, k.len(), context);

            for head in 0..self.num_heads {
                let start = head * self.head_dim;
                let end = start + self.head_dim;
                let query_head = self.head_slice_with_rope(query, start, end, query_position);
                let mut key_indices = Vec::new();
                let mut scores = Vec::new();

                for (key_index, key) in k.iter().enumerate() {
                    if !key_is_visible(key_index, query_position, query_document_id, context) {
                        continue;
                    }
                    let key_position = context
                        .key_position_ids
                        .map(|ids| ids[key_index])
                        .unwrap_or(key_index as u32);
                    self.validate_position_id(key_position)?;
                    let key_head = self.head_slice_with_rope(key, start, end, key_position);
                    let dot = query_head
                        .iter()
                        .zip(&key_head)
                        .map(|(a, b)| a * b)
                        .sum::<f32>();
                    key_indices.push(key_index);
                    scores.push(dot * scale);
                }
                if scores.is_empty() {
                    continue;
                }

                let weights = stable_softmax(&scores);
                for (weight, key_index) in weights.iter().zip(key_indices) {
                    let value = &v[key_index];
                    for dim in start..end {
                        output[query_index][dim] += weight * value[dim];
                    }
                }
            }
        }

        Ok(output)
    }

    fn validate_position_id(&self, position_id: u32) -> Result<()> {
        if self.rope_enabled && position_id as usize >= self.max_position_embeddings {
            return Err(CognexisError::InvalidConfig(format!(
                "position_id {} exceeds max_position_embeddings {}",
                position_id, self.max_position_embeddings
            )));
        }
        Ok(())
    }

    fn head_slice_with_rope(
        &self,
        row: &[f32],
        start: usize,
        end: usize,
        position_id: u32,
    ) -> Vec<f32> {
        let mut head = row[start..end].to_vec();
        if !self.rope_enabled {
            return head;
        }

        let position = position_id as f32;
        let width = head.len().max(1) as f32;
        for pair_start in (0..head.len().saturating_sub(1)).step_by(2) {
            let exponent = pair_start as f32 / width;
            let angle = position / self.rope_theta.powf(exponent);
            let (sin, cos) = angle.sin_cos();
            let even = head[pair_start];
            let odd = head[pair_start + 1];
            head[pair_start] = even * cos - odd * sin;
            head[pair_start + 1] = even * sin + odd * cos;
        }
        head
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

fn validate_context(query_len: usize, key_len: usize, context: AttentionContext<'_>) -> Result<()> {
    for (name, len, expected) in [
        (
            "key_padding_mask",
            context.key_padding_mask.map(<[bool]>::len),
            key_len,
        ),
        (
            "query_position_ids",
            context.query_position_ids.map(<[u32]>::len),
            query_len,
        ),
        (
            "key_position_ids",
            context.key_position_ids.map(<[u32]>::len),
            key_len,
        ),
        (
            "query_document_ids",
            context.query_document_ids.map(<[u32]>::len),
            query_len,
        ),
        (
            "key_document_ids",
            context.key_document_ids.map(<[u32]>::len),
            key_len,
        ),
        (
            "active_query_mask",
            context.active_query_mask.map(<[bool]>::len),
            query_len,
        ),
    ] {
        if let Some(actual) = len {
            if actual != expected {
                return Err(CognexisError::ShapeMismatch {
                    expected: format!("{name} length {expected}"),
                    actual: format!("{name} length {actual}"),
                });
            }
        }
    }

    if context
        .query_position_ids
        .map(|ids| !is_monotonic(ids))
        .unwrap_or(false)
        || context
            .key_position_ids
            .map(|ids| !is_monotonic(ids))
            .unwrap_or(false)
    {
        return Err(CognexisError::InvalidConfig(
            "attention position_ids must be monotonically non-decreasing".to_string(),
        ));
    }

    Ok(())
}

fn query_position(query_index: usize, decode_offset: usize, context: AttentionContext<'_>) -> u32 {
    context
        .query_position_ids
        .map(|ids| ids[query_index])
        .unwrap_or((decode_offset + query_index) as u32)
}

fn query_document_id(
    query_index: usize,
    decode_offset: usize,
    key_len: usize,
    context: AttentionContext<'_>,
) -> Option<u32> {
    context
        .query_document_ids
        .map(|ids| ids[query_index])
        .or_else(|| {
            context.key_document_ids.map(|ids| {
                let key_index = (decode_offset + query_index).min(key_len.saturating_sub(1));
                ids[key_index]
            })
        })
}

fn key_is_visible(
    key_index: usize,
    query_position: u32,
    query_document_id: Option<u32>,
    context: AttentionContext<'_>,
) -> bool {
    if context
        .key_padding_mask
        .map(|mask| !mask[key_index])
        .unwrap_or(false)
    {
        return false;
    }

    let key_position = context
        .key_position_ids
        .map(|ids| ids[key_index])
        .unwrap_or(key_index as u32);
    if context.enforce_causal && key_position > query_position {
        return false;
    }

    if let (Some(query_document_id), Some(key_document_ids)) =
        (query_document_id, context.key_document_ids)
    {
        if key_document_ids[key_index] != query_document_id {
            return false;
        }
    }

    true
}

fn is_monotonic(ids: &[u32]) -> bool {
    ids.windows(2).all(|pair| pair[0] <= pair[1])
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
