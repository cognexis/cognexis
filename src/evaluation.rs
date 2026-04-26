//! Evaluation module.
//!
//! This module contains stubs for evaluation routines. In practice
//! evaluation covers perplexity calculation, dataset benchmarks,
//! depth efficiency index (DEI) measurement, and other metrics. See
//! `spec20_evaluation_metrics.md` and `spec21_loop_scaling.md` for
//! definitions and recommended protocols.

/// Compute perplexity of model outputs against reference tokens.
pub fn perplexity(logits: &[Vec<f32>], targets: &[u32]) -> f64 {
    if logits.is_empty() || targets.is_empty() {
        return f64::INFINITY;
    }

    let mut nll = 0.0;
    let mut count = 0usize;
    for (row, &target) in logits.iter().zip(targets) {
        if row.is_empty() || target as usize >= row.len() {
            return f64::INFINITY;
        }
        nll += negative_log_likelihood(row, target as usize);
        count += 1;
    }

    if count == 0 {
        return f64::INFINITY;
    }

    (nll / count as f64).exp()
}

/// Compute Depth Efficiency Index (DEI) given performance and compute.
pub fn depth_efficiency_index(delta_metric: f64, delta_compute: f64) -> f64 {
    if delta_compute == 0.0 {
        return 0.0;
    }
    delta_metric / delta_compute
}

fn negative_log_likelihood(logits: &[f32], target: usize) -> f64 {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum_exp = logits
        .iter()
        .map(|logit| (*logit as f64 - max_logit).exp())
        .sum::<f64>();
    let log_sum_exp = max_logit + sum_exp.ln();
    log_sum_exp - logits[target] as f64
}
