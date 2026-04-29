//! Transformer block module.
//!
//! A transformer block combines multi‑head attention and a feed‑forward
//! network with layer normalization and residual connections. See
//! `spec06_transformer_block.md` for discussion of normalization
//! strategies (RMSNorm), residual scaling, and block composition.

use crate::attention::{AttentionContext, MultiHeadAttention};
use crate::config::ModelConfig;
use crate::feedforward::FeedForwardNetwork;
use crate::stability::rms_norm;
use crate::{CognexisError, Result};

/// Explicit transformer-block execution context.
#[derive(Debug, Clone, Copy, Default)]
pub struct BlockContext<'a> {
    pub layer_id: usize,
    pub attention: AttentionContext<'a>,
    /// Token positions that should receive the recurrent update.
    pub active_token_mask: Option<&'a [bool]>,
}

/// A single transformer block.
pub struct TransformerBlock {
    attention: MultiHeadAttention,
    feedforward: FeedForwardNetwork,
    norm_epsilon: f32,
    attention_residual_scale: f32,
    ffn_residual_scale: f32,
}

impl TransformerBlock {
    /// Create a new transformer block based on the model config.
    pub fn new(config: &ModelConfig) -> Self {
        Self::with_residual_scale(config, 1.0)
    }

    /// Create a block with explicit residual scaling.
    pub fn with_residual_scale(config: &ModelConfig, residual_scale: f32) -> Self {
        Self {
            attention: MultiHeadAttention::new(config),
            feedforward: FeedForwardNetwork::new(config),
            norm_epsilon: config.norm_epsilon,
            attention_residual_scale: residual_scale,
            ffn_residual_scale: residual_scale,
        }
    }

    /// Apply the transformer block using pre-norm attention and FFN
    /// residual paths.
    pub fn forward(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.try_forward(x).unwrap_or_default()
    }

    /// Checked transformer block path for callers that need structured errors.
    pub fn try_forward(&self, x: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        self.try_forward_with_context(x, BlockContext::default())
    }

    /// Checked transformer block path with explicit masks and instrumentation context.
    pub fn try_forward_with_context(
        &self,
        x: &[Vec<f32>],
        context: BlockContext<'_>,
    ) -> Result<Vec<Vec<f32>>> {
        self.validate_input(x)?;
        if let Some(active_mask) = context.active_token_mask {
            if active_mask.len() != x.len() {
                return Err(CognexisError::ShapeMismatch {
                    expected: format!("active_token_mask length {}", x.len()),
                    actual: format!("active_token_mask length {}", active_mask.len()),
                });
            }
        }

        let norm_attn = normalize_rows(x, self.norm_epsilon);
        let attn = self.attention.try_forward_with_context(
            &norm_attn,
            &norm_attn,
            &norm_attn,
            context.attention,
        )?;
        let after_attn = add_scaled(x, &attn, self.attention_residual_scale)?;

        let norm_ffn = normalize_rows(&after_attn, self.norm_epsilon);
        let ffn = self.feedforward.try_forward(&norm_ffn)?;
        let output = add_scaled(&after_attn, &ffn, self.ffn_residual_scale)?;
        Ok(match context.active_token_mask {
            Some(active_mask) => output
                .into_iter()
                .zip(x)
                .zip(active_mask)
                .map(|((updated_row, original_row), active)| {
                    if *active {
                        updated_row
                    } else {
                        original_row.clone()
                    }
                })
                .collect(),
            None => output,
        })
    }

    fn validate_input(&self, x: &[Vec<f32>]) -> Result<()> {
        for (row_index, row) in x.iter().enumerate() {
            if row.len() != self.attention.hidden_size {
                return Err(CognexisError::ShapeMismatch {
                    expected: format!("row width {}", self.attention.hidden_size),
                    actual: format!("row {row_index} width {}", row.len()),
                });
            }
        }
        Ok(())
    }
}

fn normalize_rows(x: &[Vec<f32>], epsilon: f32) -> Vec<Vec<f32>> {
    x.iter().map(|row| rms_norm(row, epsilon)).collect()
}

fn add_scaled(base: &[Vec<f32>], update: &[Vec<f32>], scale: f32) -> Result<Vec<Vec<f32>>> {
    if base.len() != update.len() {
        return Err(CognexisError::ShapeMismatch {
            expected: format!("{} rows", base.len()),
            actual: format!("{} rows", update.len()),
        });
    }

    base.iter()
        .zip(update)
        .enumerate()
        .map(|(row_index, (base_row, update_row))| {
            if base_row.len() != update_row.len() {
                return Err(CognexisError::ShapeMismatch {
                    expected: format!("row {row_index} width {}", base_row.len()),
                    actual: format!("row {row_index} width {}", update_row.len()),
                });
            }
            Ok(base_row
                .iter()
                .zip(update_row)
                .map(|(base_value, update_value)| base_value + scale * update_value)
                .collect())
        })
        .collect()
}
