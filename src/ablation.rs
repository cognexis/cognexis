//! Ablation testing module.
//!
//! Ablation tests remove or modify components of the model to study
//! their contributions. This module defines placeholders for running
//! such experiments. See `spec22_ablation.md` for recommended tests.

/// Enumeration of ablation types.
#[derive(Debug, Clone, Copy)]
pub enum AblationType {
    RemoveAttention,
    RemoveFeedForward,
    RemoveRecurrent,
    RemoveNormalization,
}

/// Component switches produced by an ablation selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AblationPlan {
    pub use_attention: bool,
    pub use_feedforward: bool,
    pub use_recurrent: bool,
    pub use_normalization: bool,
}

impl Default for AblationPlan {
    fn default() -> Self {
        Self {
            use_attention: true,
            use_feedforward: true,
            use_recurrent: true,
            use_normalization: true,
        }
    }
}

/// Convert an ablation type into explicit component switches.
pub fn plan_ablation(ablation: AblationType) -> AblationPlan {
    let mut plan = AblationPlan::default();
    match ablation {
        AblationType::RemoveAttention => plan.use_attention = false,
        AblationType::RemoveFeedForward => plan.use_feedforward = false,
        AblationType::RemoveRecurrent => plan.use_recurrent = false,
        AblationType::RemoveNormalization => plan.use_normalization = false,
    }
    plan
}

/// Run an ablation experiment. In practice this would create a
/// modified model configuration and evaluate it on benchmark data. This
/// reference hook currently materializes the component plan.
pub fn run_ablation(ablation: AblationType) {
    let _ = plan_ablation(ablation);
}
