//! Prelude module.
//!
//! The prelude is the initial stack of transformer blocks that
//! process token embeddings prior to the recurrent core. It sets up
//! the context for iterative refinement. See `spec07_prelude.md` for
//! rationale on the number of layers and their initialization.

use crate::config::ModelConfig;
use crate::transformer_block::TransformerBlock;

/// Prelude containing a series of transformer blocks.
pub struct Prelude {
    layers: Vec<TransformerBlock>,
}

impl Prelude {
    /// Construct the prelude with the specified number of layers.
    pub fn new(config: &ModelConfig) -> Self {
        let layers = (0..config.num_prelude_layers)
            .map(|_| TransformerBlock::new(config))
            .collect();
        Self { layers }
    }

    /// Apply all prelude layers to the input.
    pub fn forward(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut output = x.to_owned();
        for layer in &self.layers {
            output = layer.forward(&output);
        }
        output
    }
}
