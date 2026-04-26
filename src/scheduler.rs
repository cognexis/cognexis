//! Scheduler module.
//!
//! The scheduler determines how many recurrent loops to run during
//! inference. It may be fixed, adaptive, or token‑wise. See
//! `spec17_scheduler_design.md` and `spec18_tokenwise_scheduling.md`
//! for details on heuristics, DEI‑aware stopping criteria, and token
//! budget allocation.

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

/// Scheduler decision for sequence-level recurrent execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerDecision {
    pub action: LoopAction,
    pub halt_reason: Option<HaltReason>,
}

/// Transparent rule-based scheduler for the CPU reference path.
#[derive(Debug, Clone, Copy)]
pub struct RuleBasedScheduler {
    pub min_delta: f32,
    pub confidence_threshold: f32,
    pub predicted_gain_threshold: f32,
}

impl Default for RuleBasedScheduler {
    fn default() -> Self {
        Self {
            min_delta: 1.0e-4,
            confidence_threshold: 0.95,
            predicted_gain_threshold: 5.0e-4,
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
