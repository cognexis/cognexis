//! Evaluation module.
//!
//! This module contains reference evaluation routines for conventional
//! language-model metrics and recurrent-depth analysis. Benchmark
//! harnesses can build on these formulas while handling dataset I/O
//! and model execution separately.

use serde::{Deserialize, Serialize};

use crate::config::ModelConfig;
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
    #[serde(default)]
    pub scheduler_diagnostics: Option<SchedulerEvaluationDiagnostics>,
}

/// Scheduler observability attached to adaptive evaluation rows.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct SchedulerEvaluationDiagnostics {
    pub average_loops_used: Option<f64>,
    pub loop_count_histogram: Vec<usize>,
    pub halt_reasons: Vec<String>,
    pub scheduler_overhead_ms_mean: Option<f64>,
    pub budget_violation_count: usize,
}

impl SchedulerEvaluationDiagnostics {
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("average_loops_used", self.average_loops_used),
            (
                "scheduler_overhead_ms_mean",
                self.scheduler_overhead_ms_mean,
            ),
        ] {
            if value
                .map(|value| !value.is_finite() || value < 0.0)
                .unwrap_or(false)
            {
                return Err(CognexisError::InvalidConfig(format!(
                    "{name} must be finite and non-negative"
                )));
            }
        }
        if self
            .halt_reasons
            .iter()
            .any(|reason| reason.trim().is_empty())
        {
            return Err(CognexisError::InvalidConfig(
                "scheduler halt reasons must not be empty".to_string(),
            ));
        }
        Ok(())
    }
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
        if let Some(diagnostics) = &self.scheduler_diagnostics {
            diagnostics.validate()?;
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

/// Serialize result rows as CSV with stable column order.
pub fn results_to_csv(rows: &[EvaluationResultRow]) -> Result<String> {
    let mut output = String::from(
        "checkpoint,tokenizer_checksum,dataset,split,loop_mode,loop_count,metric_name,metric_value,latency_ms_mean,flops_mean,hardware,dtype,seed,average_loops_used,loop_count_histogram,halt_reasons,scheduler_overhead_ms_mean,budget_violation_count\n",
    );
    for row in rows {
        row.validate()?;
        let diagnostics = row.scheduler_diagnostics.as_ref();
        output.push_str(&csv_field(&row.checkpoint));
        output.push(',');
        output.push_str(&csv_field(row.tokenizer_checksum.as_deref().unwrap_or("")));
        output.push(',');
        output.push_str(&csv_field(&row.dataset));
        output.push(',');
        output.push_str(&csv_field(&row.split));
        output.push(',');
        output.push_str(&csv_field(&row.loop_mode));
        output.push(',');
        output.push_str(&row.loop_count.to_string());
        output.push(',');
        output.push_str(&csv_field(&row.metric_name));
        output.push(',');
        output.push_str(&row.metric_value.to_string());
        output.push(',');
        output.push_str(&optional_f64(row.latency_ms_mean));
        output.push(',');
        output.push_str(&optional_f64(row.flops_mean));
        output.push(',');
        output.push_str(&csv_field(row.hardware.as_deref().unwrap_or("")));
        output.push(',');
        output.push_str(&csv_field(row.dtype.as_deref().unwrap_or("")));
        output.push(',');
        output.push_str(&row.seed.to_string());
        output.push(',');
        output.push_str(&optional_f64(
            diagnostics.and_then(|diagnostics| diagnostics.average_loops_used),
        ));
        output.push(',');
        output.push_str(&csv_field(
            &diagnostics
                .map(|diagnostics| join_usize_values(&diagnostics.loop_count_histogram))
                .unwrap_or_default(),
        ));
        output.push(',');
        output.push_str(&csv_field(
            &diagnostics
                .map(|diagnostics| diagnostics.halt_reasons.join(";"))
                .unwrap_or_default(),
        ));
        output.push(',');
        output.push_str(&optional_f64(
            diagnostics.and_then(|diagnostics| diagnostics.scheduler_overhead_ms_mean),
        ));
        output.push(',');
        if let Some(diagnostics) = diagnostics {
            output.push_str(&diagnostics.budget_violation_count.to_string());
        }
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

/// Aggregate row-level evaluation outputs for report headers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvaluationSummary {
    pub row_count: usize,
    pub min_loop_count: usize,
    pub max_loop_count: usize,
    pub metric_value_mean: f64,
    pub latency_ms_mean: Option<f64>,
    pub flops_mean: Option<f64>,
}

/// Approximate forward-pass compute and memory estimate for reporting.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ComputeEstimate {
    pub sequence_len: usize,
    pub loop_count: usize,
    pub effective_depth: usize,
    pub token_updates: usize,
    pub prelude_flops: f64,
    pub recurrent_flops: f64,
    pub coda_flops: f64,
    pub lm_head_flops: f64,
    pub total_flops: f64,
    pub activation_memory_bytes: usize,
    pub kv_cache_memory_bytes: usize,
}

impl ComputeEstimate {
    pub fn validate(&self) -> Result<()> {
        if self.sequence_len == 0 {
            return Err(CognexisError::InvalidConfig(
                "compute estimate sequence_len must be positive".to_string(),
            ));
        }
        if self.effective_depth == 0 {
            return Err(CognexisError::InvalidConfig(
                "compute estimate effective_depth must be positive".to_string(),
            ));
        }
        for (name, value) in [
            ("prelude_flops", self.prelude_flops),
            ("recurrent_flops", self.recurrent_flops),
            ("coda_flops", self.coda_flops),
            ("lm_head_flops", self.lm_head_flops),
            ("total_flops", self.total_flops),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(CognexisError::InvalidConfig(format!(
                    "{name} must be finite and non-negative"
                )));
            }
        }
        Ok(())
    }
}

/// Estimate dense sequence-level forward FLOPs for a fixed loop count.
pub fn estimate_forward_compute(
    config: &ModelConfig,
    sequence_len: usize,
    loop_count: usize,
) -> Result<ComputeEstimate> {
    config.validate()?;
    if sequence_len == 0 {
        return Err(CognexisError::InvalidConfig(
            "sequence_len must be positive for compute estimation".to_string(),
        ));
    }

    let bounded_loops = loop_count.clamp(config.min_loop_count, config.max_loop_count);
    let block_flops = estimate_transformer_block_flops(config, sequence_len);
    let prelude_flops = block_flops * config.num_prelude_layers as f64;
    let recurrent_flops = block_flops * config.num_recurrent_blocks as f64 * bounded_loops as f64;
    let coda_flops = block_flops * config.num_coda_layers as f64;
    let lm_head_flops =
        2.0 * sequence_len as f64 * config.hidden_size as f64 * config.vocab_size as f64;
    let total_flops = prelude_flops + recurrent_flops + coda_flops + lm_head_flops;
    let estimate = ComputeEstimate {
        sequence_len,
        loop_count: bounded_loops,
        effective_depth: config.effective_depth(bounded_loops),
        token_updates: sequence_len * bounded_loops,
        prelude_flops,
        recurrent_flops,
        coda_flops,
        lm_head_flops,
        total_flops,
        activation_memory_bytes: sequence_len * config.hidden_size * std::mem::size_of::<f32>(),
        kv_cache_memory_bytes: estimate_kv_cache_memory_bytes(config, sequence_len),
    };
    estimate.validate()?;
    Ok(estimate)
}

/// Estimate dense masked token-wise forward cost.
///
/// The reference token-wise implementation is dense: it computes every
/// recurrent pass for every token and then masks halted token updates.
/// Therefore FLOPs are governed by the maximum token loop count, while
/// `token_updates` records the smaller number of actually applied token
/// updates for diagnostics.
pub fn estimate_tokenwise_forward_compute(
    config: &ModelConfig,
    token_loop_counts: &[usize],
) -> Result<ComputeEstimate> {
    if token_loop_counts.is_empty() {
        return Err(CognexisError::InvalidConfig(
            "token_loop_counts must not be empty".to_string(),
        ));
    }
    if token_loop_counts
        .iter()
        .any(|loops| *loops > config.max_loop_count)
    {
        return Err(CognexisError::InvalidConfig(format!(
            "token loop counts must be <= model max_loop_count {}",
            config.max_loop_count
        )));
    }
    let max_loop_count = token_loop_counts.iter().copied().max().unwrap_or(0);
    let mut estimate = estimate_forward_compute(config, token_loop_counts.len(), max_loop_count)?;
    estimate.token_updates = token_loop_counts.iter().sum();
    estimate.validate()?;
    Ok(estimate)
}

/// Summarize a homogeneous set of evaluation rows.
pub fn summarize_evaluation_results(rows: &[EvaluationResultRow]) -> Result<EvaluationSummary> {
    if rows.is_empty() {
        return Err(CognexisError::InvalidConfig(
            "cannot summarize empty evaluation results".to_string(),
        ));
    }
    for row in rows {
        row.validate()?;
    }

    let metric_value_mean =
        rows.iter().map(|row| row.metric_value).sum::<f64>() / rows.len() as f64;
    let min_loop_count = rows.iter().map(|row| row.loop_count).min().unwrap_or(0);
    let max_loop_count = rows.iter().map(|row| row.loop_count).max().unwrap_or(0);
    let latency_ms_mean = mean_present(rows.iter().filter_map(|row| row.latency_ms_mean));
    let flops_mean = mean_present(rows.iter().filter_map(|row| row.flops_mean));

    Ok(EvaluationSummary {
        row_count: rows.len(),
        min_loop_count,
        max_loop_count,
        metric_value_mean,
        latency_ms_mean,
        flops_mean,
    })
}

/// Compute perplexity of model outputs against reference tokens.
pub fn perplexity(logits: &[Vec<f32>], targets: &[u32]) -> f64 {
    let nll = negative_log_likelihood_for_targets(logits, targets);
    if !nll.is_finite() {
        return f64::INFINITY;
    }
    nll.exp()
}

/// Mean next-token negative log-likelihood for target IDs.
pub fn negative_log_likelihood_for_targets(logits: &[Vec<f32>], targets: &[u32]) -> f64 {
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
    nll / count as f64
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

/// One multiple-choice scoring example.
#[derive(Debug, Clone, PartialEq)]
pub struct MultipleChoiceExample {
    pub choice_scores: Vec<f64>,
    pub correct_index: usize,
}

/// Deterministic multiple-choice accuracy using argmax over choice scores.
pub fn multiple_choice_accuracy(examples: &[MultipleChoiceExample]) -> Result<f64> {
    if examples.is_empty() {
        return Ok(0.0);
    }
    let mut correct = 0usize;
    for (example_index, example) in examples.iter().enumerate() {
        if example.choice_scores.is_empty() {
            return Err(CognexisError::InvalidConfig(format!(
                "multiple-choice example {example_index} has no choices"
            )));
        }
        if example.correct_index >= example.choice_scores.len() {
            return Err(CognexisError::InvalidConfig(format!(
                "multiple-choice example {example_index} correct_index {} is out of range {}",
                example.correct_index,
                example.choice_scores.len()
            )));
        }
        if example.choice_scores.iter().any(|score| !score.is_finite()) {
            return Err(CognexisError::InvalidConfig(format!(
                "multiple-choice example {example_index} contains non-finite scores"
            )));
        }
        let predicted = example
            .choice_scores
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.partial_cmp(right)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
            .unwrap_or(0);
        if predicted == example.correct_index {
            correct += 1;
        }
    }
    Ok(accuracy(correct, examples.len()))
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

/// Corpus-level BLEU with up to 4-gram precision and brevity penalty.
pub fn bleu_score(predictions: &[&str], references: &[&str]) -> Result<f64> {
    if predictions.len() != references.len() {
        return Err(CognexisError::ShapeMismatch {
            expected: format!("{} references", predictions.len()),
            actual: format!("{} references", references.len()),
        });
    }
    if predictions.is_empty() {
        return Ok(0.0);
    }

    let mut clipped_matches = [0usize; 4];
    let mut predicted_counts = [0usize; 4];
    let mut prediction_len = 0usize;
    let mut reference_len = 0usize;
    for (prediction, reference) in predictions.iter().zip(references) {
        let prediction_tokens = tokenize_metric_text(prediction);
        let reference_tokens = tokenize_metric_text(reference);
        prediction_len += prediction_tokens.len();
        reference_len += reference_tokens.len();
        for n in 1..=4 {
            let prediction_ngrams = ngram_counts(&prediction_tokens, n);
            let reference_ngrams = ngram_counts(&reference_tokens, n);
            predicted_counts[n - 1] += prediction_ngrams
                .iter()
                .map(|(_, count)| *count)
                .sum::<usize>();
            for (ngram, count) in prediction_ngrams {
                let reference_count = reference_ngrams
                    .iter()
                    .find_map(|(candidate, reference_count)| {
                        (candidate == &ngram).then_some(*reference_count)
                    })
                    .unwrap_or(0);
                clipped_matches[n - 1] += count.min(reference_count);
            }
        }
    }

    if prediction_len == 0 {
        return Ok(0.0);
    }
    let precisions: Vec<f64> = (0..4)
        .map(|index| {
            if predicted_counts[index] == 0 {
                1.0
            } else {
                (clipped_matches[index] as f64 + 1.0) / (predicted_counts[index] as f64 + 1.0)
            }
        })
        .collect();
    let log_precision_mean = precisions.iter().map(|value| value.ln()).sum::<f64>() / 4.0;
    let brevity_penalty = if prediction_len > reference_len {
        1.0
    } else {
        (1.0 - reference_len as f64 / prediction_len as f64).exp()
    };
    Ok(brevity_penalty * log_precision_mean.exp())
}

/// Mean ROUGE-L F1 over prediction/reference pairs.
pub fn rouge_l_f1(predictions: &[&str], references: &[&str]) -> Result<f64> {
    if predictions.len() != references.len() {
        return Err(CognexisError::ShapeMismatch {
            expected: format!("{} references", predictions.len()),
            actual: format!("{} references", references.len()),
        });
    }
    if predictions.is_empty() {
        return Ok(0.0);
    }

    let mut total = 0.0;
    for (prediction, reference) in predictions.iter().zip(references) {
        let prediction_tokens = tokenize_metric_text(prediction);
        let reference_tokens = tokenize_metric_text(reference);
        if prediction_tokens.is_empty() || reference_tokens.is_empty() {
            continue;
        }
        let lcs = lcs_len(&prediction_tokens, &reference_tokens) as f64;
        let precision = lcs / prediction_tokens.len() as f64;
        let recall = lcs / reference_tokens.len() as f64;
        if precision + recall > 0.0 {
            total += 2.0 * precision * recall / (precision + recall);
        }
    }
    Ok(total / predictions.len() as f64)
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

fn estimate_transformer_block_flops(config: &ModelConfig, sequence_len: usize) -> f64 {
    let seq = sequence_len as f64;
    let hidden = config.hidden_size as f64;
    let heads = config.num_attention_heads as f64;
    let head_dim = config.head_dim() as f64;
    let kv_hidden = config.num_kv_heads as f64 * head_dim;
    let ffn = config.ff_inner_dim as f64;

    let q_projection = 2.0 * seq * hidden * hidden;
    let kv_projection = 4.0 * seq * hidden * kv_hidden;
    let output_projection = 2.0 * seq * hidden * hidden;
    let attention_scores = 2.0 * heads * seq * seq * head_dim;
    let attention_values = 2.0 * heads * seq * seq * head_dim;
    let ffn_flops = match config.ff_activation {
        crate::config::FeedForwardActivation::SwiGlu
        | crate::config::FeedForwardActivation::GeGlu => 6.0 * seq * hidden * ffn,
        crate::config::FeedForwardActivation::Gelu | crate::config::FeedForwardActivation::Relu => {
            4.0 * seq * hidden * ffn
        }
    };

    q_projection
        + kv_projection
        + output_projection
        + attention_scores
        + attention_values
        + ffn_flops
}

fn estimate_kv_cache_memory_bytes(config: &ModelConfig, sequence_len: usize) -> usize {
    let kv_width = config.num_kv_heads * config.head_dim();
    let recurrent_stages = config.max_loop_count * config.num_recurrent_blocks;
    let stages = config.num_prelude_layers + recurrent_stages + config.num_coda_layers;
    sequence_len * kv_width * 2 * stages * std::mem::size_of::<f32>()
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

fn optional_f64(value: Option<f64>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn csv_field(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') || value.contains('\r') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn join_usize_values(values: &[usize]) -> String {
    values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(";")
}

fn mean_present(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut total = 0.0;
    let mut count = 0usize;
    for value in values {
        total += value;
        count += 1;
    }
    (count > 0).then_some(total / count as f64)
}

fn tokenize_metric_text(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|token| token.to_lowercase())
        .collect()
}

fn ngram_counts(tokens: &[String], n: usize) -> Vec<(Vec<String>, usize)> {
    if n == 0 || tokens.len() < n {
        return Vec::new();
    }
    let mut counts: Vec<(Vec<String>, usize)> = Vec::new();
    for window in tokens.windows(n) {
        let ngram = window.to_vec();
        if let Some((_, count)) = counts.iter_mut().find(|(candidate, _)| candidate == &ngram) {
            *count += 1;
        } else {
            counts.push((ngram, 1));
        }
    }
    counts
}

fn lcs_len(left: &[String], right: &[String]) -> usize {
    let mut previous = vec![0usize; right.len() + 1];
    let mut current = vec![0usize; right.len() + 1];
    for left_token in left {
        for (right_index, right_token) in right.iter().enumerate() {
            current[right_index + 1] = if left_token == right_token {
                previous[right_index] + 1
            } else {
                current[right_index].max(previous[right_index + 1])
            };
        }
        std::mem::swap(&mut previous, &mut current);
        current.fill(0);
    }
    previous[right.len()]
}
