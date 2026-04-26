//! Stability and normalization module.
//!
//! Recurrent models require careful handling of stability to ensure
//! convergent iterations. This module provides placeholders for
//! spectral normalization, RMSNorm, and dynamic scaling. See
//! `spec15_stability_normalization.md` for techniques and theory.

/// Apply RMS normalization to a vector.
pub fn rms_norm(x: &[f32], epsilon: f32) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let mean_square = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv_rms = 1.0 / (mean_square + epsilon).sqrt();
    x.iter().map(|v| v * inv_rms).collect()
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
