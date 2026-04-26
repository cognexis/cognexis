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

/// Run an ablation experiment. In practice this would create a
/// modified model configuration and evaluate it on benchmark data. This
/// stub does nothing.
pub fn run_ablation(_ablation: AblationType) {
    // TODO: Implement ablation logic.
}