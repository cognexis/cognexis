//! Language modeling head module.
//!
//! The language modeling head projects the final hidden state into
//! logits over the vocabulary. It optionally ties weights with the
//! token embedding matrix. See `spec10_lm_head.md` for discussion of
//! weight tying and output normalization.

use crate::config::ModelConfig;

/// Language modeling head.
pub struct LMHead {
    pub hidden_size: usize,
    pub vocab_size: usize,
}

impl LMHead {
    /// Create a new language modeling head from the model configuration.
    pub fn new(config: &ModelConfig) -> Self {
        Self {
            hidden_size: config.hidden_size,
            vocab_size: config.vocab_size,
        }
    }

    /// Compute logits for each token position. This skeleton returns a
    /// zero matrix of shape (seq_len, vocab_size).
    pub fn forward(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let seq_len = x.len();
        vec![vec![0.0; self.vocab_size]; seq_len]
    }
}