//! Loop scaling module.
//!
//! Routines for studying the effect of loop counts on performance and
//! compute cost. See `spec21_loop_scaling.md` for details on how to
//! run depth scaling experiments and plot accuracy vs. loops curves.

/// Placeholder function to generate a schedule of loop counts for
/// evaluation. Returns a vector of loop counts doubling each time.
pub fn loop_schedule(max_loops: usize) -> Vec<usize> {
    let mut schedule = vec![];
    let mut n = 1;
    while n <= max_loops {
        schedule.push(n);
        n *= 2;
    }
    schedule
}