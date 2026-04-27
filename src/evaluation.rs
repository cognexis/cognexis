//! Evaluation module.
//!
//! This module contains reference evaluation routines for conventional
//! language-model metrics and recurrent-depth analysis. Benchmark
//! harnesses can build on these formulas while handling dataset I/O
//! and model execution separately.

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

/// Direction of a metric used by depth-aware comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricDirection {
    HigherIsBetter,
    LowerIsBetter,
}

/// A single quality/compute observation at a fixed loop count.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthPoint {
    pub loops: usize,
    pub metric: f64,
    pub compute: f64,
}

/// Compute DEI between two fixed-depth observations.
pub fn depth_efficiency_between(
    from: DepthPoint,
    to: DepthPoint,
    direction: MetricDirection,
) -> Option<f64> {
    let delta_compute = to.compute - from.compute;
    if !delta_compute.is_finite() || delta_compute <= 0.0 {
        return None;
    }

    let delta_metric = match direction {
        MetricDirection::HigherIsBetter => to.metric - from.metric,
        MetricDirection::LowerIsBetter => from.metric - to.metric,
    };
    Some(delta_metric / delta_compute)
}

/// Return the first depth where marginal return falls below `threshold`.
///
/// Points are interpreted by ascending loop count. If all positive
/// marginal returns stay above the threshold, the depth with the best
/// observed positive DEI is returned.
pub fn loop_saturation_point(
    points: &[DepthPoint],
    threshold: f64,
    direction: MetricDirection,
) -> Option<usize> {
    let points = sorted_depth_points(points)?;
    let mut best_positive = None;

    for window in points.windows(2) {
        let from = window[0];
        let to = window[1];
        let dei = depth_efficiency_between(from, to, direction)?;

        if dei > 0.0
            && best_positive
                .map(|(_, best_dei)| dei > best_dei)
                .unwrap_or(true)
        {
            best_positive = Some((to.loops, dei));
        }
        if dei < threshold {
            return Some(from.loops);
        }
    }

    best_positive.map(|(loops, _)| loops)
}

/// Return the first deeper loop count where quality degrades.
pub fn overthinking_threshold(
    points: &[DepthPoint],
    tolerance: f64,
    direction: MetricDirection,
) -> Option<usize> {
    let points = sorted_depth_points(points)?;
    for window in points.windows(2) {
        let from = window[0];
        let to = window[1];
        let harmed = match direction {
            MetricDirection::HigherIsBetter => to.metric < from.metric - tolerance,
            MetricDirection::LowerIsBetter => to.metric > from.metric + tolerance,
        };
        if harmed {
            return Some(to.loops);
        }
    }
    None
}

/// Compute depth gain ratio from a shallow baseline to a deeper point.
pub fn depth_gain_ratio(
    shallow_metric: f64,
    deep_metric: f64,
    direction: MetricDirection,
) -> Option<f64> {
    if shallow_metric == 0.0 || !shallow_metric.is_finite() || !deep_metric.is_finite() {
        return None;
    }

    let improvement = match direction {
        MetricDirection::HigherIsBetter => deep_metric - shallow_metric,
        MetricDirection::LowerIsBetter => shallow_metric - deep_metric,
    };
    Some(improvement / shallow_metric.abs())
}

/// Exact-match score with outer whitespace ignored.
pub fn exact_match(prediction: &str, reference: &str) -> bool {
    prediction.trim() == reference.trim()
}

/// Fraction of correct predictions. Empty inputs return 0.0.
pub fn accuracy(correct: usize, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    correct as f64 / total as f64
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

fn sorted_depth_points(points: &[DepthPoint]) -> Option<Vec<DepthPoint>> {
    if points.len() < 2 {
        return None;
    }
    if points
        .iter()
        .any(|point| !point.metric.is_finite() || !point.compute.is_finite())
    {
        return None;
    }

    let mut points = points.to_vec();
    points.sort_by_key(|point| point.loops);
    Some(points)
}
