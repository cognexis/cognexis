//! Ablation testing module.
//!
//! Ablation tests remove or modify components of the model to study
//! their contributions. This module defines placeholders for running
//! such experiments. See `spec22_ablation.md` for recommended tests.

use serde::{Deserialize, Serialize};

use crate::config::CognexisConfig;
use crate::{CognexisError, Result};

/// Enumeration of ablation types.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AblationType {
    RemoveAttention,
    RemoveFeedForward,
    RemoveRecurrent,
    RemoveNormalization,
    RemovePrelude,
    RemoveCoda,
    DisableAdaptiveScheduling,
    RemoveValueHead,
    DisableTokenWiseScheduling,
    RemoveResidualScaling,
}

/// Component switches produced by an ablation selection.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct AblationPlan {
    pub use_attention: bool,
    pub use_feedforward: bool,
    pub use_recurrent: bool,
    pub use_normalization: bool,
    pub use_prelude: bool,
    pub use_coda: bool,
    pub use_adaptive_scheduling: bool,
    pub use_value_head: bool,
    pub use_tokenwise_scheduling: bool,
    pub use_residual_scaling: bool,
}

impl Default for AblationPlan {
    fn default() -> Self {
        Self {
            use_attention: true,
            use_feedforward: true,
            use_recurrent: true,
            use_normalization: true,
            use_prelude: true,
            use_coda: true,
            use_adaptive_scheduling: true,
            use_value_head: true,
            use_tokenwise_scheduling: true,
            use_residual_scaling: true,
        }
    }
}

/// Structured ablation experiment definition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AblationExperiment {
    pub name: String,
    pub ablation: AblationType,
    pub baseline_config: CognexisConfig,
    pub changed_factor: String,
    pub hypothesis: String,
    pub random_seeds: Vec<u64>,
    pub hardware: Option<String>,
    pub stopping_criteria: Option<String>,
}

impl AblationExperiment {
    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty()
            || self.changed_factor.trim().is_empty()
            || self.hypothesis.trim().is_empty()
        {
            return Err(CognexisError::InvalidConfig(
                "ablation name, changed_factor, and hypothesis must not be empty".to_string(),
            ));
        }
        if self.random_seeds.is_empty() {
            return Err(CognexisError::InvalidConfig(
                "ablation experiment must include at least one random seed".to_string(),
            ));
        }
        self.baseline_config.validate()
    }
}

/// Status of an ablation run.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AblationStatus {
    Planned,
    Completed,
    Failed,
    Unstable,
}

/// Machine-readable ablation result summary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AblationResult {
    pub experiment_name: String,
    pub plan: AblationPlan,
    pub status: AblationStatus,
    pub metric_delta: Option<f64>,
    pub compute_delta: Option<f64>,
    pub stability_non_finite_count: usize,
    pub parameter_count: Option<u64>,
    pub notes: Option<String>,
}

/// Convert an ablation type into explicit component switches.
pub fn plan_ablation(ablation: AblationType) -> AblationPlan {
    let mut plan = AblationPlan::default();
    match ablation {
        AblationType::RemoveAttention => plan.use_attention = false,
        AblationType::RemoveFeedForward => plan.use_feedforward = false,
        AblationType::RemoveRecurrent => plan.use_recurrent = false,
        AblationType::RemoveNormalization => plan.use_normalization = false,
        AblationType::RemovePrelude => plan.use_prelude = false,
        AblationType::RemoveCoda => plan.use_coda = false,
        AblationType::DisableAdaptiveScheduling => plan.use_adaptive_scheduling = false,
        AblationType::RemoveValueHead => plan.use_value_head = false,
        AblationType::DisableTokenWiseScheduling => plan.use_tokenwise_scheduling = false,
        AblationType::RemoveResidualScaling => plan.use_residual_scaling = false,
    }
    plan
}

/// Apply the ablation as structured config overrides.
pub fn apply_ablation_overrides(
    baseline: &CognexisConfig,
    ablation: AblationType,
) -> Result<CognexisConfig> {
    let mut config = baseline.clone();
    match ablation {
        AblationType::RemovePrelude => config.model.num_prelude_layers = 0,
        AblationType::RemoveCoda => config.model.num_coda_layers = 0,
        AblationType::RemoveRecurrent => {
            config.model.min_loop_count = 1;
            config.model.max_loop_count = 1;
            if let Some(inference) = &mut config.inference {
                inference.min_loops = 1;
                inference.max_loops = 1;
            }
            if let Some(safety) = &mut config.safety {
                safety.max_user_loops = 1;
            }
        }
        AblationType::DisableAdaptiveScheduling => {
            if let Some(inference) = &mut config.inference {
                inference.loop_mode = "fixed".to_string();
                inference.min_loops = inference.min_loops.max(1);
                inference.max_loops = inference.min_loops;
            }
        }
        AblationType::RemoveResidualScaling => config.model.recurrent_residual_scale = 1.0,
        AblationType::RemoveAttention
        | AblationType::RemoveFeedForward
        | AblationType::RemoveNormalization
        | AblationType::RemoveValueHead
        | AblationType::DisableTokenWiseScheduling => {}
    }
    config.validate()?;
    Ok(config)
}

/// Estimate parameter count for reporting and smoke tests.
pub fn estimate_parameter_count(config: &CognexisConfig, plan: AblationPlan) -> u64 {
    let hidden = config.model.hidden_size as u64;
    let vocab = config.model.vocab_size as u64;
    let attention = if plan.use_attention {
        4 * hidden * hidden
    } else {
        0
    };
    let ffn = if plan.use_feedforward {
        2 * hidden * config.model.ff_inner_dim as u64
    } else {
        0
    };
    let block = attention + ffn;
    let prelude = if plan.use_prelude {
        config.model.num_prelude_layers as u64 * block
    } else {
        0
    };
    let recurrent = if plan.use_recurrent { block } else { 0 };
    let coda = if plan.use_coda {
        config.model.num_coda_layers as u64 * block
    } else {
        0
    };
    let embeddings = vocab * hidden;
    let lm_head = if config.model.tie_embeddings {
        0
    } else {
        vocab * hidden
    };
    embeddings + prelude + recurrent + coda + lm_head
}

/// Materialize a planned ablation result without running model evaluation.
pub fn run_ablation(ablation: AblationType) -> AblationResult {
    let plan = plan_ablation(ablation);
    AblationResult {
        experiment_name: format!("{ablation:?}"),
        plan,
        status: AblationStatus::Planned,
        metric_delta: None,
        compute_delta: None,
        stability_non_finite_count: 0,
        parameter_count: None,
        notes: None,
    }
}

/// Aggregate completed/failed ablation results.
pub fn summarize_ablation_results(results: &[AblationResult]) -> AblationSummary {
    let completed = results
        .iter()
        .filter(|result| result.status == AblationStatus::Completed)
        .count();
    let unstable = results
        .iter()
        .filter(|result| result.status == AblationStatus::Unstable)
        .count();
    let failed = results
        .iter()
        .filter(|result| result.status == AblationStatus::Failed)
        .count();
    let best_metric_delta = results
        .iter()
        .filter_map(|result| result.metric_delta)
        .fold(None, |best: Option<f64>, value| {
            Some(best.map_or(value, |current| current.max(value)))
        });

    AblationSummary {
        runs: results.len(),
        completed,
        unstable,
        failed,
        best_metric_delta,
    }
}

/// Aggregated ablation status.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct AblationSummary {
    pub runs: usize,
    pub completed: usize,
    pub unstable: usize,
    pub failed: usize,
    pub best_metric_delta: Option<f64>,
}
