//! Safety and monitoring module.
//!
//! Large language models must be deployed with safeguards to prevent
//! harmful output and to monitor behavior. This module defines
//! placeholders for safety checks, logging, and telemetry. See
//! `spec24_safety_monitoring.md` for recommended practices.

use serde::{Deserialize, Serialize};

use crate::{CognexisError, Result};

/// Enumeration of possible safety concerns detected during inference.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SafetyIssue {
    OffensiveLanguage,
    SensitiveInformation,
    Misinformation,
    SpecialTokenInjection,
    BudgetExceeded,
    Other,
}

/// Safety policy behavior for a request.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PolicyMode {
    Enforce,
    AuditOnly,
}

/// Runtime compute budget enforced around generation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct ComputeBudget {
    pub max_prompt_tokens: usize,
    pub max_generated_tokens: usize,
    pub max_loops_per_token: usize,
    pub max_total_loops: Option<usize>,
}

impl Default for ComputeBudget {
    fn default() -> Self {
        Self {
            max_prompt_tokens: 8_192,
            max_generated_tokens: 512,
            max_loops_per_token: 16,
            max_total_loops: None,
        }
    }
}

impl ComputeBudget {
    pub fn validate(&self) -> Result<()> {
        if self.max_prompt_tokens == 0
            || self.max_generated_tokens == 0
            || self.max_loops_per_token == 0
        {
            return Err(CognexisError::InvalidConfig(
                "compute budget token and loop limits must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

/// Aggregated safety flags collected before and during generation.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SafetyFlags {
    pub issues: Vec<SafetyIssue>,
}

impl SafetyFlags {
    pub fn push_unique(&mut self, issue: SafetyIssue) {
        if !self.issues.contains(&issue) {
            self.issues.push(issue);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.issues.is_empty()
    }
}

/// Safety context passed through a serving request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SafetyContext {
    pub input_flags: SafetyFlags,
    pub output_flags: SafetyFlags,
    pub budget: ComputeBudget,
    pub policy_mode: PolicyMode,
}

impl Default for SafetyContext {
    fn default() -> Self {
        Self {
            input_flags: SafetyFlags::default(),
            output_flags: SafetyFlags::default(),
            budget: ComputeBudget::default(),
            policy_mode: PolicyMode::Enforce,
        }
    }
}

impl SafetyContext {
    pub fn inspect_input(&mut self, prompt: &str, prompt_tokens: usize) -> SafetyDecision {
        if prompt_tokens > self.budget.max_prompt_tokens {
            self.input_flags.push_unique(SafetyIssue::BudgetExceeded);
        }
        if contains_reserved_chat_token(prompt) {
            self.input_flags
                .push_unique(SafetyIssue::SpecialTokenInjection);
        }
        if let Some(issue) = check_safety(prompt) {
            self.input_flags.push_unique(issue);
        }
        decision_from_flags(&self.input_flags, self.policy_mode)
    }

    pub fn inspect_output(&mut self, output: &str, generated_tokens: usize) -> SafetyDecision {
        if generated_tokens > self.budget.max_generated_tokens {
            self.output_flags.push_unique(SafetyIssue::BudgetExceeded);
        }
        if let Some(issue) = check_safety(output) {
            self.output_flags.push_unique(issue);
        }
        decision_from_flags(&self.output_flags, self.policy_mode)
    }

    pub fn check_loop_budget(
        &mut self,
        loops_for_token: usize,
        total_loops: usize,
    ) -> SafetyDecision {
        if loops_for_token > self.budget.max_loops_per_token
            || self
                .budget
                .max_total_loops
                .map(|limit| total_loops > limit)
                .unwrap_or(false)
        {
            self.output_flags.push_unique(SafetyIssue::BudgetExceeded);
        }
        decision_from_flags(&self.output_flags, self.policy_mode)
    }
}

/// Action required by safety policy.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SafetyAction {
    Allow,
    Refuse,
    Audit,
}

/// Structured safety decision.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SafetyDecision {
    pub action: SafetyAction,
    pub issues: Vec<SafetyIssue>,
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

fn decision_from_flags(flags: &SafetyFlags, policy_mode: PolicyMode) -> SafetyDecision {
    let action = if flags.is_empty() {
        SafetyAction::Allow
    } else if policy_mode == PolicyMode::AuditOnly {
        SafetyAction::Audit
    } else {
        SafetyAction::Refuse
    };
    SafetyDecision {
        action,
        issues: flags.issues.clone(),
    }
}

fn contains_reserved_chat_token(text: &str) -> bool {
    [
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|tool|>",
        "<|end|>",
    ]
    .iter()
    .any(|token| text.contains(token))
}
