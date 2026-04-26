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
    /// Get the number of loops for a specific token index.
    pub fn loops_for(&self, index: usize) -> usize {
        *self.loops_per_token.get(index).unwrap_or(&1)
    }
}