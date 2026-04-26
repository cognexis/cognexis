//! Coda module.
//!
//! The coda is a stack of transformer blocks applied after the
//! recurrent core to finalize the hidden state before decoding. See
//! `spec09_coda.md` for the rationale on the number of layers and
//! post‑processing.

use crate::config::ModelConfig;
use crate::transformer_block::TransformerBlock;

/// Coda containing a series of transformer blocks.
pub struct Coda {
    layers: Vec<TransformerBlock>,
}

impl Coda {
    /// Construct the coda with the specified number of layers.
    pub fn new(config: &ModelConfig) -> Self {
        let layers = (0..config.num_coda_layers)
            .map(|_| TransformerBlock::new(config))
            .collect();
        Self { layers }
    }

    /// Apply all coda layers to the input.
    pub fn forward(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut output = x.to_owned();
        for layer in &self.layers {
            output = layer.forward(&output);
        }
        output
    }
}
