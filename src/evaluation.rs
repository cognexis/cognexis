//! Evaluation module.
//!
//! This module contains stubs for evaluation routines. In practice
//! evaluation covers perplexity calculation, dataset benchmarks,
//! depth efficiency index (DEI) measurement, and other metrics. See
//! `spec20_evaluation_metrics.md` and `spec21_loop_scaling.md` for
//! definitions and recommended protocols.

/// Compute perplexity of model outputs against reference tokens. This
/// stub returns 0.0.
pub fn perplexity(_logits: &[Vec<f32>], _targets: &[u32]) -> f64 {
    // TODO: Compute log likelihood and exponentiate.
    0.0
}

/// Compute Depth Efficiency Index (DEI) given performance and compute.
pub fn depth_efficiency_index(delta_metric: f64, delta_compute: f64) -> f64 {
    if delta_compute == 0.0 {
        return 0.0;
    }
    delta_metric / delta_compute
}