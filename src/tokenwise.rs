//! Token‑wise scheduling module.
//!
//! Token‑wise scheduling allows the model to allocate different
//! numbers of loops to different positions in the sequence. See
//! `spec18_tokenwise_scheduling.md` for design considerations and
//! potential heuristics. This stub defines an interface for
//! specifying per‑token loop counts.

/// A schedule specifying loop counts for each token position.
#[derive(Debug)]
pub struct TokenwiseSchedule {
    pub loops_per_token: Vec<usize>,
}

impl TokenwiseSchedule {
    /// Create a schedule with a fixed number of loops per token.
    pub fn fixed(seq_len: usize, loops: usize) -> Self {
        Self {
            loops_per_token: vec![loops; seq_len],
        }
    }

    /// Create a schedule while clamping every token to configured bounds.
    pub fn bounded(loops_per_token: Vec<usize>, min_loops: usize, max_loops: usize) -> Self {
        let upper = max_loops.max(min_loops);
        Self {
            loops_per_token: loops_per_token
                .into_iter()
                .map(|loops| loops.clamp(min_loops, upper))
                .collect(),
        }
    }

    /// Get the number of loops for a specific token index.
    pub fn loops_for(&self, index: usize) -> usize {
        *self.loops_per_token.get(index).unwrap_or(&1)
    }

    /// Return a mask indicating which tokens are still active at a
    /// zero-based loop index.
    pub fn active_mask_for_loop(&self, loop_index: usize) -> Vec<bool> {
        self.loops_per_token
            .iter()
            .map(|loops| loop_index < *loops)
            .collect()
    }
}
