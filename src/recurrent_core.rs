//! Recurrent core module.
//!
//! The recurrent core contains the block that is iteratively applied
//! during inference to refine the hidden state. It shares weights
//! across iterations. See `spec08_recurrent_core.md` for details on
//! stability, parameter sharing, and halting conditions.

use crate::config::ModelConfig;
use crate::transformer_block::TransformerBlock;

/// Recurrent core consisting of one or more shared transformer blocks.
pub struct RecurrentCore {
    blocks: Vec<TransformerBlock>,
    pub min_loops: usize,
    pub max_loops: usize,
}

impl RecurrentCore {
    /// Initialize the recurrent core based on model configuration.
    pub fn new(config: &ModelConfig) -> Self {
        let blocks = (0..config.num_recurrent_blocks)
            .map(|_| TransformerBlock::with_residual_scale(config, config.recurrent_residual_scale))
            .collect();
        Self {
            blocks,
            min_loops: config.min_loop_count,
            max_loops: config.max_loop_count,
        }
    }

    /// Perform iterative refinement of the hidden state while enforcing
    /// configured loop bounds.
    pub fn forward(&self, x: &[Vec<f32>], loops: usize) -> Vec<Vec<f32>> {
        let mut output = x.to_owned();
        let iterations = loops.clamp(self.min_loops, self.max_loops);
        for _ in 0..iterations {
            for block in &self.blocks {
                output = block.forward(&output);
            }
        }
        output
    }
}
