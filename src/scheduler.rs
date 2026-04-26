//! Scheduler module.
//!
//! The scheduler determines how many recurrent loops to run during
//! inference. It may be fixed, adaptive, or token‑wise. See
//! `spec17_scheduler_design.md` and `spec18_tokenwise_scheduling.md`
//! for details on heuristics, DEI‑aware stopping criteria, and token
//! budget allocation.

/// Enumeration of loop scheduling strategies.
#[derive(Debug, Clone, Copy)]
pub enum LoopScheduling {
    Fixed(usize),
    Adaptive,
    TokenWise,
}

/// Compute the number of loops to run given the scheduling strategy
/// and optionally input complexity measures.
pub fn compute_loops(strategy: LoopScheduling, input_length: usize) -> usize {
    match strategy {
        LoopScheduling::Fixed(n) => n,
        LoopScheduling::Adaptive => {
            // TODO: Use DEI metrics or confidence scores to adapt loops.
            // Placeholder: scale loops linearly with input length.
            (input_length / 10).max(1)
        }
        LoopScheduling::TokenWise => {
            // TODO: Allocate loops per token based on difficulty.
            1
        }
    }
}