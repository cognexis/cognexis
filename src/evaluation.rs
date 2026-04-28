//! Evaluation module.
//!
//! This module contains reference evaluation routines for conventional
//! language-model metrics and recurrent-depth analysis. Benchmark
//! harnesses can build on these formulas while handling dataset I/O
//! and model execution separately.

use serde::{Deserialize, Serialize};

use crate::{CognexisError, Result};

/// One machine-readable evaluation result row.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvaluationResultRow {
    pub checkpoint: String,
    pub tokenizer_checksum: Option<String>,
    pub dataset: String,
    pub split: String,
    pub loop_mode: String,
    pub loop_count: usize,
    pub metric_name: String,
    pub metric_value: f64,
    pub latency_ms_mean: Option<f64>,
    pub flops_mean: Option<f64>,
    pub hardware: Option<String>,
    pub dtype: Option<String>,
    pub seed: u64,
}

impl EvaluationResultRow {
    pub fn validate(&self) -> Result<()> {
        if self.checkpoint.trim().is_empty()
            || self.dataset.trim().is_empty()
            || self.split.trim().is_empty()
            || self.loop_mode.trim().is_empty()
            || self.metric_name.trim().is_empty()
        {
            return Err(CognexisError::InvalidConfig(
                "evaluation result identity fields must not be empty".to_string(),
            ));
        }
        if self.loop_count == 0 {
            return Err(CognexisError::InvalidConfig(
                "evaluation loop_count must be positive".to_string(),
            ));
        }
        if !self.metric_value.is_finite()
            || self
                .latency_ms_mean
                .map(|value| !value.is_finite() || value < 0.0)
                .unwrap_or(false)
            || self
                .flops_mean
                .map(|value| !value.is_finite() || value < 0.0)
                .unwrap_or(false)
        {
            return Err(CognexisError::InvalidConfig(
                "evaluation numeric fields must be finite and non-negative where applicable"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

/// Serialize result rows as JSON Lines.
pub fn results_to_jsonl(rows: &[EvaluationResultRow]) -> Result<String> {
    let mut output = String::new();
    for row in rows {
        row.validate()?;
        let encoded = serde_json::to_string(row).map_err(|error| {
            CognexisError::Backend(format!("result serialization failed: {error}"))
        })?;
        output.push_str(&encoded);
        output.push('\n');
    }
    Ok(output)
}

/// Parse JSON Lines result rows.
pub fn results_from_jsonl(jsonl: &str) -> Result<Vec<EvaluationResultRow>> {
    jsonl
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(line_index, line)| {
            let row = serde_json::from_str::<EvaluationResultRow>(line).map_err(|error| {
                CognexisError::InvalidConfig(format!(
                    "invalid evaluation JSONL line {}: {error}",
                    line_index + 1
                ))
            })?;
            row.validate()?;
            Ok(row)
        })
        .collect()
}

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

/// Unbiased pass@k estimator used by code-generation evaluation.
pub fn pass_at_k(total_samples: usize, correct_samples: usize, k: usize) -> f64 {
    if total_samples == 0 || k == 0 {
        return 0.0;
    }
    if correct_samples == 0 {
        return 0.0;
    }
    if k >= total_samples || total_samples - correct_samples < k {
        return 1.0;
    }

    let incorrect = total_samples - correct_samples;
    let mut probability_all_fail = 1.0;
    for i in 0..k {
        probability_all_fail *= (incorrect - i) as f64 / (total_samples - i) as f64;
    }
    1.0 - probability_all_fail
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
