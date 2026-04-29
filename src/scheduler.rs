//! Scheduler module.
//!
//! The scheduler determines how many recurrent loops to run during
//! inference. It may be fixed, adaptive, or token‑wise. See
//! `spec17_scheduler_design.md` and `spec18_tokenwise_scheduling.md`
//! for details on heuristics, DEI‑aware stopping criteria, and token
//! budget allocation.

use crate::value_head::ValuePrediction;
use crate::{CognexisError, Result};

/// Enumeration of loop scheduling strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopScheduling {
    Fixed(usize),
    Adaptive,
    TokenWise,
}

/// Scheduler action for the next recurrent loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopAction {
    Continue,
    Halt,
}

/// Structured halt reasons emitted for diagnostics and monitoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HaltReason {
    MaxLoops,
    MinDelta,
    Confidence,
    ValueGain,
    Budget,
    Safety,
    Forced,
}

/// Observation consumed by a rule-based scheduler.
#[derive(Debug, Clone, Copy)]
pub struct SchedulerObservation {
    pub loops_executed: usize,
    pub min_loops: usize,
    pub max_loops: usize,
    pub hidden_delta: Option<f32>,
    pub confidence: Option<f32>,
    pub predicted_gain: Option<f32>,
    pub remaining_loop_budget: Option<usize>,
    pub safety_halt: bool,
}

/// Scalar value-head signals consumed by ACT-style scheduling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActObservation {
    pub loops_executed: usize,
    pub min_loops: usize,
    pub max_loops: usize,
    pub predicted_gain: f32,
    pub risk_adjusted_gain: f32,
    pub continue_logit: f32,
    pub uncertainty: f32,
    pub remaining_loop_budget: Option<usize>,
    pub safety_halt: bool,
}

/// ACT policy configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActSchedulerConfig {
    pub halting_threshold: f32,
    pub gain_threshold: f32,
    pub ponder_cost: f32,
    pub uncertainty_halt_threshold: f32,
}

impl Default for ActSchedulerConfig {
    fn default() -> Self {
        Self {
            halting_threshold: 0.5,
            gain_threshold: 5.0e-4,
            ponder_cost: 0.01,
            uncertainty_halt_threshold: 0.95,
        }
    }
}

impl ActSchedulerConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.halting_threshold.is_finite()
            || self.halting_threshold <= 0.0
            || self.halting_threshold >= 1.0
        {
            return Err(CognexisError::InvalidConfig(
                "ACT halting_threshold must be in (0, 1)".to_string(),
            ));
        }
        if !self.gain_threshold.is_finite()
            || !self.ponder_cost.is_finite()
            || self.ponder_cost < 0.0
            || !self.uncertainty_halt_threshold.is_finite()
            || self.uncertainty_halt_threshold < 0.0
        {
            return Err(CognexisError::InvalidConfig(
                "ACT gain, ponder, and uncertainty thresholds must be finite".to_string(),
            ));
        }
        Ok(())
    }
}

/// ACT decision with scheduler-internal scalar diagnostics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActSchedulerDecision {
    pub decision: SchedulerDecision,
    pub halt_probability: f32,
    pub ponder_penalty: f32,
    pub effective_gain: f32,
}

/// Adaptive Computation Time scheduler consuming value-head predictions.
#[derive(Debug, Clone, Copy, Default)]
pub struct ActScheduler {
    pub config: ActSchedulerConfig,
}

impl ActScheduler {
    pub fn decide(&self, observation: ActObservation) -> Result<ActSchedulerDecision> {
        self.config.validate()?;
        if observation.min_loops == 0 || observation.max_loops < observation.min_loops {
            return Err(CognexisError::InvalidConfig(
                "ACT observation loop bounds are invalid".to_string(),
            ));
        }
        if !observation.predicted_gain.is_finite()
            || !observation.risk_adjusted_gain.is_finite()
            || !observation.continue_logit.is_finite()
            || !observation.uncertainty.is_finite()
        {
            return Err(CognexisError::InvalidConfig(
                "ACT observation scalar values must be finite".to_string(),
            ));
        }

        let ponder_penalty = self.config.ponder_cost * (observation.loops_executed + 1) as f32;
        let effective_gain = observation.risk_adjusted_gain - ponder_penalty;
        let halt_probability = sigmoid(-observation.continue_logit);
        let decision = if observation.safety_halt {
            halt(HaltReason::Safety)
        } else if observation.remaining_loop_budget == Some(0) {
            halt(HaltReason::Budget)
        } else if observation.loops_executed >= observation.max_loops {
            halt(HaltReason::MaxLoops)
        } else if observation.loops_executed < observation.min_loops {
            continue_loop()
        } else if halt_probability >= self.config.halting_threshold
            || effective_gain <= self.config.gain_threshold
            || observation.uncertainty >= self.config.uncertainty_halt_threshold
        {
            halt(HaltReason::ValueGain)
        } else {
            continue_loop()
        };

        Ok(ActSchedulerDecision {
            decision,
            halt_probability,
            ponder_penalty,
            effective_gain,
        })
    }

    pub fn decide_from_value_prediction(
        &self,
        prediction: &ValuePrediction,
        prediction_index: usize,
        loops_executed: usize,
        min_loops: usize,
        max_loops: usize,
        remaining_loop_budget: Option<usize>,
        safety_halt: bool,
    ) -> Result<ActSchedulerDecision> {
        let predicted_gain = *prediction
            .predicted_gain
            .get(prediction_index)
            .ok_or_else(|| CognexisError::ShapeMismatch {
                expected: format!("prediction index {prediction_index}"),
                actual: format!("{} gain predictions", prediction.predicted_gain.len()),
            })?;
        let risk_adjusted_gain = *prediction
            .risk_adjusted_gain
            .get(prediction_index)
            .ok_or_else(|| CognexisError::ShapeMismatch {
                expected: format!("prediction index {prediction_index}"),
                actual: format!(
                    "{} risk-adjusted predictions",
                    prediction.risk_adjusted_gain.len()
                ),
            })?;
        let continue_logit = *prediction
            .continue_logit
            .get(prediction_index)
            .ok_or_else(|| CognexisError::ShapeMismatch {
                expected: format!("prediction index {prediction_index}"),
                actual: format!("{} continue logits", prediction.continue_logit.len()),
            })?;
        let uncertainty = *prediction
            .uncertainty
            .get(prediction_index)
            .ok_or_else(|| CognexisError::ShapeMismatch {
                expected: format!("prediction index {prediction_index}"),
                actual: format!("{} uncertainty predictions", prediction.uncertainty.len()),
            })?;

        self.decide(ActObservation {
            loops_executed,
            min_loops,
            max_loops,
            predicted_gain,
            risk_adjusted_gain,
            continue_logit,
            uncertainty,
            remaining_loop_budget,
            safety_halt,
        })
    }
}

/// Per-request scheduler setup and hard limits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedulerRequestContext {
    pub request_id: Option<String>,
    pub min_loops: usize,
    pub max_loops: usize,
    pub loop_budget: Option<usize>,
}

impl SchedulerRequestContext {
    pub fn validate(&self) -> Result<()> {
        if self.min_loops == 0 {
            return Err(CognexisError::InvalidConfig(
                "scheduler min_loops must be at least 1".to_string(),
            ));
        }
        if self.max_loops < self.min_loops {
            return Err(CognexisError::InvalidConfig(format!(
                "scheduler max_loops ({}) must be >= min_loops ({})",
                self.max_loops, self.min_loops
            )));
        }
        if self
            .request_id
            .as_deref()
            .map(|request_id| request_id.trim().is_empty())
            .unwrap_or(false)
        {
            return Err(CognexisError::InvalidConfig(
                "scheduler request_id must not be empty when provided".to_string(),
            ));
        }
        Ok(())
    }
}

/// Scheduler decision for sequence-level recurrent execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerDecision {
    pub action: LoopAction,
    pub halt_reason: Option<HaltReason>,
}

/// One scheduler observation/decision trace entry.
#[derive(Debug, Clone, PartialEq)]
pub struct SchedulerStepTrace {
    pub loops_executed: usize,
    pub decision: SchedulerDecision,
    pub hidden_delta: Option<f32>,
    pub confidence: Option<f32>,
    pub predicted_gain: Option<f32>,
    pub remaining_loop_budget: Option<usize>,
}

/// Lightweight diagnostics safe to emit in serving logs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SchedulerDiagnostics {
    pub request_id: Option<String>,
    pub decisions: Vec<SchedulerStepTrace>,
    pub loops_executed: usize,
    pub budget_consumed: usize,
    pub final_halt_reason: Option<HaltReason>,
}

/// Shared scheduler interface for runtime policies.
pub trait LoopScheduler {
    fn begin_request(&mut self, context: SchedulerRequestContext) -> Result<()>;
    fn observe(&mut self, observation: SchedulerObservation) -> Result<SchedulerDecision>;
    fn finish(&mut self) -> SchedulerDiagnostics;
}

/// Transparent rule-based scheduler for the CPU reference path.
#[derive(Debug, Clone)]
pub struct RuleBasedScheduler {
    pub min_delta: f32,
    pub confidence_threshold: f32,
    pub predicted_gain_threshold: f32,
    diagnostics: SchedulerDiagnostics,
    active_context: Option<SchedulerRequestContext>,
}

impl Default for RuleBasedScheduler {
    fn default() -> Self {
        Self {
            min_delta: 1.0e-4,
            confidence_threshold: 0.95,
            predicted_gain_threshold: 5.0e-4,
            diagnostics: SchedulerDiagnostics::default(),
            active_context: None,
        }
    }
}

impl RuleBasedScheduler {
    /// Decide whether to continue or halt while enforcing hard bounds.
    pub fn decide(&self, observation: SchedulerObservation) -> SchedulerDecision {
        if observation.safety_halt {
            return halt(HaltReason::Safety);
        }
        if observation.remaining_loop_budget == Some(0) {
            return halt(HaltReason::Budget);
        }
        if observation.loops_executed >= observation.max_loops {
            return halt(HaltReason::MaxLoops);
        }
        if observation.loops_executed < observation.min_loops {
            return continue_loop();
        }

        let low_delta = matches!(observation.hidden_delta, Some(delta) if delta <= self.min_delta);
        let low_gain = match observation.predicted_gain {
            Some(gain) => gain <= self.predicted_gain_threshold,
            None => true,
        };
        if low_delta && low_gain {
            return halt(if observation.predicted_gain.is_some() {
                HaltReason::ValueGain
            } else {
                HaltReason::MinDelta
            });
        }

        let high_confidence = matches!(
            observation.confidence,
            Some(confidence) if confidence >= self.confidence_threshold
        );
        if high_confidence && low_gain {
            return halt(HaltReason::Confidence);
        }

        continue_loop()
    }
}

impl LoopScheduler for RuleBasedScheduler {
    fn begin_request(&mut self, context: SchedulerRequestContext) -> Result<()> {
        context.validate()?;
        self.diagnostics = SchedulerDiagnostics {
            request_id: context.request_id.clone(),
            ..SchedulerDiagnostics::default()
        };
        self.active_context = Some(context);
        Ok(())
    }

    fn observe(&mut self, observation: SchedulerObservation) -> Result<SchedulerDecision> {
        if observation.min_loops == 0 || observation.max_loops < observation.min_loops {
            return Err(CognexisError::InvalidConfig(
                "scheduler observation loop bounds are invalid".to_string(),
            ));
        }

        let decision = self.decide(observation);
        if let Some(context) = &self.active_context {
            if let (Some(total), Some(remaining)) =
                (context.loop_budget, observation.remaining_loop_budget)
            {
                self.diagnostics.budget_consumed = self
                    .diagnostics
                    .budget_consumed
                    .max(total.saturating_sub(remaining));
            }
        }
        self.diagnostics.loops_executed = self
            .diagnostics
            .loops_executed
            .max(observation.loops_executed);
        if decision.action == LoopAction::Halt {
            self.diagnostics.final_halt_reason = decision.halt_reason;
        }
        self.diagnostics.decisions.push(SchedulerStepTrace {
            loops_executed: observation.loops_executed,
            decision,
            hidden_delta: observation.hidden_delta,
            confidence: observation.confidence,
            predicted_gain: observation.predicted_gain,
            remaining_loop_budget: observation.remaining_loop_budget,
        });

        Ok(decision)
    }

    fn finish(&mut self) -> SchedulerDiagnostics {
        let diagnostics = self.diagnostics.clone();
        self.diagnostics = SchedulerDiagnostics::default();
        self.active_context = None;
        diagnostics
    }
}

/// Compute the number of loops to run given the scheduling strategy
/// and optionally input complexity measures.
pub fn compute_loops(strategy: LoopScheduling, input_length: usize) -> usize {
    match strategy {
        LoopScheduling::Fixed(n) => n,
        LoopScheduling::Adaptive => (input_length / 10).clamp(1, 16),
        LoopScheduling::TokenWise => 1,
    }
}

/// Compute loops while enforcing configured bounds and an optional
/// remaining request budget.
pub fn compute_loops_bounded(
    strategy: LoopScheduling,
    input_length: usize,
    min_loops: usize,
    max_loops: usize,
    remaining_budget: Option<usize>,
) -> usize {
    let upper = max_loops.max(min_loops);
    let requested = compute_loops(strategy, input_length).clamp(min_loops, upper);
    match remaining_budget {
        Some(budget) => requested.min(budget),
        None => requested,
    }
}

fn halt(reason: HaltReason) -> SchedulerDecision {
    SchedulerDecision {
        action: LoopAction::Halt,
        halt_reason: Some(reason),
    }
}

fn continue_loop() -> SchedulerDecision {
    SchedulerDecision {
        action: LoopAction::Continue,
        halt_reason: None,
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}
