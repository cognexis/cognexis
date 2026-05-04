//! Value head module.
//!
//! In instruction‑tuned models it is common to attach a scalar value
//! head to predict rewards or confidence scores. This can be used for
//! DEI‑based scheduling, reinforcement learning, or fine‑grained
//! selection. See `spec19_value_head.md` for more details.

use serde::{Deserialize, Serialize};

use crate::config::ModelConfig;
use crate::{CognexisError, Result};

/// Pooling mode used by the value head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValuePooling {
    TokenWise,
    SequenceMean,
}

/// Runtime features consumed by value-head prediction.
#[derive(Debug, Clone, PartialEq)]
pub struct ValueFeatures {
    pub loop_index: usize,
    pub max_loops: usize,
    pub confidence: Option<f32>,
    pub entropy: Option<f32>,
    pub hidden_delta: Option<f32>,
    pub loop_cost: f32,
    pub predicted_risk: f32,
    pub non_pad_mask: Option<Vec<bool>>,
    pub pooling: ValuePooling,
}

impl Default for ValueFeatures {
    fn default() -> Self {
        Self {
            loop_index: 0,
            max_loops: 1,
            confidence: None,
            entropy: None,
            hidden_delta: None,
            loop_cost: 1.0,
            predicted_risk: 0.0,
            non_pad_mask: None,
            pooling: ValuePooling::SequenceMean,
        }
    }
}

/// Tunable value-head output semantics.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct ValueHeadConfig {
    pub gain_threshold: f32,
    pub risk_weight: f32,
    pub latency_weight: f32,
}

impl Default for ValueHeadConfig {
    fn default() -> Self {
        Self {
            gain_threshold: 5.0e-4,
            risk_weight: 1.0,
            latency_weight: 0.0,
        }
    }
}

impl ValueHeadConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.gain_threshold.is_finite()
            || self.gain_threshold < 0.0
            || !self.risk_weight.is_finite()
            || self.risk_weight < 0.0
            || !self.latency_weight.is_finite()
            || self.latency_weight < 0.0
        {
            return Err(CognexisError::InvalidConfig(
                "value-head gain, risk, and latency weights must be finite and non-negative"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

/// Value-head predictions consumed by adaptive schedulers.
#[derive(Debug, Clone, PartialEq)]
pub struct ValuePrediction {
    pub predicted_gain: Vec<f32>,
    pub continue_logit: Vec<f32>,
    pub risk_adjusted_gain: Vec<f32>,
    pub uncertainty: Vec<f32>,
}

/// Calibration summary for validation traces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibrationReport {
    pub mean_absolute_error: f32,
    pub false_halt_rate: f32,
    pub false_continue_rate: f32,
}

/// A simple linear value head mapping hidden states to scalar values.
pub struct ValueHead {
    pub hidden_size: usize,
    pub config: ValueHeadConfig,
}

impl ValueHead {
    pub fn new(config: &ModelConfig) -> Self {
        Self {
            hidden_size: config.hidden_size,
            config: ValueHeadConfig::default(),
        }
    }

    pub fn with_config(model_config: &ModelConfig, value_config: ValueHeadConfig) -> Self {
        Self {
            hidden_size: model_config.hidden_size,
            config: value_config,
        }
    }

    /// Predict a deterministic scalar confidence proxy for each token position.
    pub fn forward(&self, x: &[Vec<f32>]) -> Vec<f32> {
        x.iter()
            .map(|row| {
                if row.is_empty() {
                    return 0.0;
                }
                let mean_abs = row.iter().map(|value| value.abs()).sum::<f32>() / row.len() as f32;
                mean_abs / (1.0 + mean_abs)
            })
            .collect()
    }

    /// Predict gain and scheduler-facing signals from hidden states and scalar features.
    pub fn predict(
        &self,
        hidden: &[Vec<f32>],
        features: &ValueFeatures,
    ) -> Result<ValuePrediction> {
        self.validate_inputs(hidden, features)?;
        let token_scores = self.forward(hidden);
        let active_scores: Vec<f32> = match features.pooling {
            ValuePooling::TokenWise => token_scores
                .iter()
                .enumerate()
                .map(|(index, score)| {
                    if mask_allows(features, index) {
                        *score
                    } else {
                        0.0
                    }
                })
                .collect(),
            ValuePooling::SequenceMean => vec![masked_mean(&token_scores, features)],
        };

        let loop_progress = if features.max_loops == 0 {
            1.0
        } else {
            features.loop_index as f32 / features.max_loops as f32
        }
        .clamp(0.0, 1.0);
        let confidence_penalty = features.confidence.unwrap_or(0.0).clamp(0.0, 1.0) * 0.15;
        let entropy_bonus = features.entropy.unwrap_or(0.0).max(0.0) * 0.01;
        let delta_bonus = features.hidden_delta.unwrap_or(0.0).max(0.0) * 0.25;

        let mut predicted_gain = Vec::with_capacity(active_scores.len());
        let mut continue_logit = Vec::with_capacity(active_scores.len());
        let mut risk_adjusted_gain = Vec::with_capacity(active_scores.len());
        let mut uncertainty = Vec::with_capacity(active_scores.len());

        for score in active_scores {
            let gain = (score * (1.0 - loop_progress) + entropy_bonus + delta_bonus
                - confidence_penalty)
                .max(-1.0);
            let adjusted = gain
                - self.config.risk_weight * features.predicted_risk.max(0.0)
                - self.config.latency_weight * features.loop_cost.max(0.0);
            predicted_gain.push(gain);
            continue_logit.push(adjusted - self.config.gain_threshold);
            risk_adjusted_gain.push(adjusted);
            uncertainty.push((1.0 - score.clamp(0.0, 1.0)).max(0.0));
        }

        Ok(ValuePrediction {
            predicted_gain,
            continue_logit,
            risk_adjusted_gain,
            uncertainty,
        })
    }

    fn validate_inputs(&self, hidden: &[Vec<f32>], features: &ValueFeatures) -> Result<()> {
        if self.hidden_size == 0 {
            return Err(CognexisError::InvalidConfig(
                "value head hidden_size must be positive".to_string(),
            ));
        }
        if features.max_loops == 0 {
            return Err(CognexisError::InvalidConfig(
                "value head max_loops must be positive".to_string(),
            ));
        }
        if !features.loop_cost.is_finite() || !features.predicted_risk.is_finite() {
            return Err(CognexisError::InvalidConfig(
                "value head scalar features must be finite".to_string(),
            ));
        }
        if let Some(mask) = &features.non_pad_mask {
            if mask.len() != hidden.len() {
                return Err(CognexisError::ShapeMismatch {
                    expected: format!("mask length {}", hidden.len()),
                    actual: format!("mask length {}", mask.len()),
                });
            }
        }
        for (row_index, row) in hidden.iter().enumerate() {
            if row.len() != self.hidden_size {
                return Err(CognexisError::ShapeMismatch {
                    expected: format!("hidden row width {}", self.hidden_size),
                    actual: format!("row {row_index} width {}", row.len()),
                });
            }
        }
        Ok(())
    }
}

/// Adjacent-loop gain targets for lower-is-better losses.
pub fn gain_targets_from_losses(losses: &[f32]) -> Vec<f32> {
    losses
        .windows(2)
        .map(|window| window[0] - window[1])
        .collect()
}

/// Huber loss for noisy value-head gain regression.
pub fn huber_loss(predictions: &[f32], targets: &[f32], delta: f32) -> Result<f32> {
    if predictions.len() != targets.len() {
        return Err(CognexisError::ShapeMismatch {
            expected: format!("{} targets", predictions.len()),
            actual: format!("{} targets", targets.len()),
        });
    }
    if predictions.is_empty() {
        return Ok(0.0);
    }
    let delta = delta.max(f32::EPSILON);
    let loss = predictions
        .iter()
        .zip(targets)
        .map(|(prediction, target)| {
            let error = (prediction - target).abs();
            if error <= delta {
                0.5 * error * error
            } else {
                delta * (error - 0.5 * delta)
            }
        })
        .sum::<f32>()
        / predictions.len() as f32;
    Ok(loss)
}

/// Calibration summary for continue/halt decisions at a gain threshold.
pub fn calibration_report(
    predictions: &[f32],
    targets: &[f32],
    gain_threshold: f32,
) -> Result<CalibrationReport> {
    if predictions.len() != targets.len() {
        return Err(CognexisError::ShapeMismatch {
            expected: format!("{} targets", predictions.len()),
            actual: format!("{} targets", targets.len()),
        });
    }
    if predictions.is_empty() {
        return Ok(CalibrationReport {
            mean_absolute_error: 0.0,
            false_halt_rate: 0.0,
            false_continue_rate: 0.0,
        });
    }

    let mut abs_error = 0.0;
    let mut false_halt = 0usize;
    let mut false_continue = 0usize;
    for (prediction, target) in predictions.iter().zip(targets) {
        abs_error += (prediction - target).abs();
        let predicted_continue = *prediction > gain_threshold;
        let actual_continue = *target > gain_threshold;
        match (predicted_continue, actual_continue) {
            (false, true) => false_halt += 1,
            (true, false) => false_continue += 1,
            _ => {}
        }
    }
    let count = predictions.len() as f32;
    Ok(CalibrationReport {
        mean_absolute_error: abs_error / count,
        false_halt_rate: false_halt as f32 / count,
        false_continue_rate: false_continue as f32 / count,
    })
}

fn masked_mean(scores: &[f32], features: &ValueFeatures) -> f32 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for (index, score) in scores.iter().enumerate() {
        if mask_allows(features, index) {
            sum += *score;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f32
    }
}

fn mask_allows(features: &ValueFeatures, index: usize) -> bool {
    features
        .non_pad_mask
        .as_ref()
        .and_then(|mask| mask.get(index))
        .copied()
        .unwrap_or(true)
}
