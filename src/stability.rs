//! Stability and normalization module.
//!
//! Recurrent models require careful handling of stability to ensure
//! convergent iterations. This module provides placeholders for
//! spectral normalization, RMSNorm, and dynamic scaling. See
//! `spec15_stability_normalization.md` for techniques and theory.

/// Summary statistics for hidden-state monitoring.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActivationSummary {
    pub rows: usize,
    pub cols: usize,
    pub elements: usize,
    pub non_finite_count: usize,
    pub mean_l2_norm: f32,
    pub max_l2_norm: f32,
    pub mean_abs: f32,
    pub max_abs: f32,
}

/// Apply RMS normalization to a vector.
pub fn rms_norm(x: &[f32], epsilon: f32) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let mean_square = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv_rms = 1.0 / (mean_square + epsilon).sqrt();
    x.iter().map(|v| v * inv_rms).collect()
}

/// Apply LayerNorm to a vector.
pub fn layer_norm(x: &[f32], epsilon: f32) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let mean = x.iter().sum::<f32>() / x.len() as f32;
    let variance = x
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f32>()
        / x.len() as f32;
    let inv_std = 1.0 / (variance + epsilon).sqrt();
    x.iter().map(|value| (value - mean) * inv_std).collect()
}

/// Summarize hidden activations without retaining raw content.
pub fn summarize_activations(hidden: &[Vec<f32>]) -> ActivationSummary {
    let rows = hidden.len();
    let cols = hidden.first().map_or(0, Vec::len);
    let mut elements = 0usize;
    let mut non_finite_count = 0usize;
    let mut norm_sum = 0.0;
    let mut max_l2_norm = 0.0;
    let mut abs_sum = 0.0;
    let mut max_abs = 0.0;

    for row in hidden {
        let mut finite_square_sum = 0.0;
        for value in row {
            elements += 1;
            if !value.is_finite() {
                non_finite_count += 1;
                continue;
            }
            let abs = value.abs();
            finite_square_sum += value * value;
            abs_sum += abs;
            if abs > max_abs {
                max_abs = abs;
            }
        }
        let norm = finite_square_sum.sqrt();
        norm_sum += norm;
        if norm > max_l2_norm {
            max_l2_norm = norm;
        }
    }

    let finite_count = elements.saturating_sub(non_finite_count);
    ActivationSummary {
        rows,
        cols,
        elements,
        non_finite_count,
        mean_l2_norm: if rows == 0 {
            0.0
        } else {
            norm_sum / rows as f32
        },
        max_l2_norm,
        mean_abs: if finite_count == 0 {
            0.0
        } else {
            abs_sum / finite_count as f32
        },
        max_abs,
    }
}

/// Return true when any activation is NaN or infinite.
pub fn has_non_finite(hidden: &[Vec<f32>]) -> bool {
    hidden.iter().flatten().any(|value| !value.is_finite())
}

/// Summarize `current - previous` for hidden-state delta monitoring.
pub fn summarize_delta(current: &[Vec<f32>], previous: &[Vec<f32>]) -> Option<ActivationSummary> {
    if current.len() != previous.len() {
        return None;
    }
    let mut delta = Vec::with_capacity(current.len());
    for (current_row, previous_row) in current.iter().zip(previous) {
        if current_row.len() != previous_row.len() {
            return None;
        }
        delta.push(
            current_row
                .iter()
                .zip(previous_row)
                .map(|(current, previous)| current - previous)
                .collect(),
        );
    }
    Some(summarize_activations(&delta))
}

/// Mean cosine similarity between matching rows.
pub fn mean_cosine_similarity(current: &[Vec<f32>], previous: &[Vec<f32>]) -> Option<f32> {
    if current.len() != previous.len() {
        return None;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for (current_row, previous_row) in current.iter().zip(previous) {
        if current_row.len() != previous_row.len() {
            return None;
        }
        let current_norm = l2_norm(current_row);
        let previous_norm = l2_norm(previous_row);
        if current_norm == 0.0 || previous_norm == 0.0 {
            continue;
        }
        let dot = current_row
            .iter()
            .zip(previous_row)
            .map(|(a, b)| a * b)
            .sum::<f32>();
        sum += dot / (current_norm * previous_norm);
        count += 1;
    }
    Some(if count == 0 { 0.0 } else { sum / count as f32 })
}

/// Clip a mutable gradient vector to a global L2 norm.
pub fn clip_global_norm(gradients: &mut [f32], max_norm: f32) -> f32 {
    let norm = l2_norm(gradients);
    if max_norm <= 0.0 || !max_norm.is_finite() || norm == 0.0 || norm <= max_norm {
        return norm;
    }
    let scale = max_norm / norm;
    for gradient in gradients {
        *gradient *= scale;
    }
    norm
}

/// Scale a matrix so its estimated spectral norm is at most 1.0.
pub fn spectral_normalize(weight: &[Vec<f32>]) -> Vec<Vec<f32>> {
    spectral_normalize_to(weight, 1.0)
}

/// Scale a matrix so its estimated spectral norm is at most `target_sigma`.
pub fn spectral_normalize_to(weight: &[Vec<f32>], target_sigma: f32) -> Vec<Vec<f32>> {
    if weight.is_empty() || weight[0].is_empty() || target_sigma <= 0.0 || !target_sigma.is_finite()
    {
        return weight.to_owned();
    }

    let cols = weight[0].len();
    if weight.iter().any(|row| row.len() != cols) {
        return weight.to_owned();
    }

    let sigma = estimate_spectral_norm(weight, 20);
    if !sigma.is_finite() || sigma <= target_sigma {
        return weight.to_owned();
    }

    let scale = target_sigma / sigma;
    weight
        .iter()
        .map(|row| row.iter().map(|value| value * scale).collect())
        .collect()
}

/// Estimate the largest singular value with power iteration.
pub fn estimate_spectral_norm(weight: &[Vec<f32>], iterations: usize) -> f32 {
    if weight.is_empty() || weight[0].is_empty() {
        return 0.0;
    }

    let cols = weight[0].len();
    if weight.iter().any(|row| row.len() != cols) {
        return f32::NAN;
    }

    let mut v = vec![1.0 / (cols as f32).sqrt(); cols];
    for _ in 0..iterations.max(1) {
        let u = normalize(&mat_vec(weight, &v));
        let wt_u = mat_t_vec(weight, &u);
        v = normalize(&wt_u);
    }

    l2_norm(&mat_vec(weight, &v))
}

fn mat_vec(weight: &[Vec<f32>], vector: &[f32]) -> Vec<f32> {
    weight
        .iter()
        .map(|row| row.iter().zip(vector).map(|(a, b)| a * b).sum())
        .collect()
}

fn mat_t_vec(weight: &[Vec<f32>], vector: &[f32]) -> Vec<f32> {
    let cols = weight[0].len();
    let mut result = vec![0.0; cols];
    for (row, value) in weight.iter().zip(vector) {
        for (col, weight_value) in row.iter().enumerate() {
            result[col] += weight_value * value;
        }
    }
    result
}

fn normalize(vector: &[f32]) -> Vec<f32> {
    let norm = l2_norm(vector);
    if norm == 0.0 || !norm.is_finite() {
        return vec![0.0; vector.len()];
    }
    vector.iter().map(|value| value / norm).collect()
}

fn l2_norm(vector: &[f32]) -> f32 {
    vector.iter().map(|value| value * value).sum::<f32>().sqrt()
}
