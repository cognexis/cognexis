//! Safety and monitoring module.
//!
//! Large language models must be deployed with safeguards to prevent
//! harmful output and to monitor behavior. This module defines
//! placeholders for safety checks, logging, and telemetry. See
//! `spec24_safety_monitoring.md` for recommended practices.

/// Enumeration of possible safety concerns detected during inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyIssue {
    OffensiveLanguage,
    SensitiveInformation,
    Misinformation,
    Other,
}

/// Lightweight reference safety checks. Production deployments should
/// replace this with policy-specific classifiers and audit logging.
pub fn check_safety(generated: &str) -> Option<SafetyIssue> {
    let lower = generated.to_lowercase();
    if ["api_key", "password:", "secret_key", "ssn:"]
        .iter()
        .any(|pattern| lower.contains(pattern))
    {
        return Some(SafetyIssue::SensitiveInformation);
    }
    if ["kill yourself", "racial slur", "terrorist manifesto"]
        .iter()
        .any(|pattern| lower.contains(pattern))
    {
        return Some(SafetyIssue::OffensiveLanguage);
    }
    if ["vaccines always cause autism", "the earth is flat"]
        .iter()
        .any(|pattern| lower.contains(pattern))
    {
        return Some(SafetyIssue::Misinformation);
    }
    None
}

/// Log events or metrics during training and inference. In practice
/// this might send data to a monitoring system.
pub fn log_event(event: &str) {
    if !event.trim().is_empty() {
        eprintln!("[cognexis] {event}");
    }
}
