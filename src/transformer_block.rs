//! Transformer block module.
//!
//! A transformer block combines multi‑head attention and a feed‑forward
//! network with layer normalization and residual connections. See
//! `spec06_transformer_block.md` for discussion of normalization
//! strategies (RMSNorm), residual scaling, and block composition.

use crate::attention::MultiHeadAttention;
use crate::config::ModelConfig;
use crate::feedforward::FeedForwardNetwork;
use crate::stability::rms_norm;
use crate::{CognexisError, Result};

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
        self.validate_input(x)?;

        let norm_attn = normalize_rows(x, self.norm_epsilon);
        let attn = self
            .attention
            .try_forward(&norm_attn, &norm_attn, &norm_attn)?;
        let after_attn = add_scaled(x, &attn, self.attention_residual_scale)?;

        let norm_ffn = normalize_rows(&after_attn, self.norm_epsilon);
        let ffn = self.feedforward.try_forward(&norm_ffn)?;
        add_scaled(&after_attn, &ffn, self.ffn_residual_scale)
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
