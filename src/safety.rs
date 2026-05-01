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
    #[serde(default)]
    pub max_recurrent_flops: Option<u64>,
    #[serde(default)]
    pub max_wall_time_ms: Option<u64>,
    #[serde(default)]
    pub max_cache_memory_bytes: Option<usize>,
}

impl Default for ComputeBudget {
    fn default() -> Self {
        Self {
            max_prompt_tokens: 8_192,
            max_generated_tokens: 512,
            max_loops_per_token: 16,
            max_total_loops: None,
            max_recurrent_flops: None,
            max_wall_time_ms: None,
            max_cache_memory_bytes: None,
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
        if self.max_recurrent_flops == Some(0)
            || self.max_wall_time_ms == Some(0)
            || self.max_cache_memory_bytes == Some(0)
        {
            return Err(CognexisError::InvalidConfig(
                "optional compute budget limits must be positive when set".to_string(),
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

    pub fn check_resource_budget(
        &mut self,
        recurrent_flops: Option<u64>,
        wall_time_ms: Option<u64>,
        cache_memory_bytes: Option<usize>,
    ) -> SafetyDecision {
        let recurrent_exceeded = match (recurrent_flops, self.budget.max_recurrent_flops) {
            (Some(actual), Some(limit)) => actual > limit,
            _ => false,
        };
        let wall_time_exceeded = match (wall_time_ms, self.budget.max_wall_time_ms) {
            (Some(actual), Some(limit)) => actual > limit,
            _ => false,
        };
        let cache_exceeded = match (cache_memory_bytes, self.budget.max_cache_memory_bytes) {
            (Some(actual), Some(limit)) => actual > limit,
            _ => false,
        };
        if recurrent_exceeded || wall_time_exceeded || cache_exceeded {
            self.output_flags.push_unique(SafetyIssue::BudgetExceeded);
        }
        decision_from_flags(&self.output_flags, self.policy_mode)
    }

    /// Combined terminal decision across input and output flags.
    pub fn final_decision(&self) -> SafetyDecision {
        let mut flags = self.input_flags.clone();
        for issue in &self.output_flags.issues {
            flags.push_unique(*issue);
        }
        decision_from_flags(&flags, self.policy_mode)
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

/// Summary of recurrent loop usage suitable for structured logs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoopCountsSummary {
    pub total: usize,
    pub mean: f64,
    pub max: usize,
    pub histogram: Vec<usize>,
}

impl Default for LoopCountsSummary {
    fn default() -> Self {
        Self {
            total: 0,
            mean: 0.0,
            max: 0,
            histogram: vec![0],
        }
    }
}

/// Structured request telemetry without raw prompt or output content.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RequestTelemetry {
    pub request_id: String,
    pub checkpoint_id: Option<String>,
    pub tokenizer_checksum: Option<String>,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub loop_mode: String,
    pub loop_counts: LoopCountsSummary,
    pub halt_reasons: Vec<String>,
    pub prefill_latency_ms: Option<f64>,
    pub decode_latency_ms: Option<f64>,
    pub safety_action: SafetyAction,
    pub input_issues: Vec<SafetyIssue>,
    pub output_issues: Vec<SafetyIssue>,
    pub stop_reason: Option<String>,
    pub error: Option<String>,
    pub cache_memory_bytes: Option<usize>,
    #[serde(default)]
    pub estimated_recurrent_flops: Option<u64>,
    #[serde(default)]
    pub wall_time_ms: Option<u64>,
    pub backend: Option<String>,
}

impl RequestTelemetry {
    /// Build telemetry from a safety context and non-content request metadata.
    pub fn from_context(
        request_id: impl Into<String>,
        context: &SafetyContext,
        prompt_tokens: usize,
        generated_tokens: usize,
        loop_mode: impl Into<String>,
        loop_counts: &[usize],
    ) -> Self {
        Self {
            request_id: request_id.into(),
            checkpoint_id: None,
            tokenizer_checksum: None,
            prompt_tokens,
            generated_tokens,
            loop_mode: loop_mode.into(),
            loop_counts: summarize_loop_counts(loop_counts),
            halt_reasons: Vec::new(),
            prefill_latency_ms: None,
            decode_latency_ms: None,
            safety_action: context.final_decision().action,
            input_issues: context.input_flags.issues.clone(),
            output_issues: context.output_flags.issues.clone(),
            stop_reason: None,
            error: None,
            cache_memory_bytes: None,
            estimated_recurrent_flops: None,
            wall_time_ms: None,
            backend: None,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.request_id.trim().is_empty() || self.loop_mode.trim().is_empty() {
            return Err(CognexisError::InvalidConfig(
                "telemetry request_id and loop_mode must not be empty".to_string(),
            ));
        }
        for (name, value) in [
            ("prefill_latency_ms", self.prefill_latency_ms),
            ("decode_latency_ms", self.decode_latency_ms),
        ] {
            if value
                .map(|value| !value.is_finite() || value < 0.0)
                .unwrap_or(false)
            {
                return Err(CognexisError::InvalidConfig(format!(
                    "{name} must be finite and non-negative"
                )));
            }
        }
        if !self.loop_counts.mean.is_finite() {
            return Err(CognexisError::InvalidConfig(
                "loop count mean must be finite".to_string(),
            ));
        }
        if self.estimated_recurrent_flops == Some(0) || self.wall_time_ms == Some(0) {
            return Err(CognexisError::InvalidConfig(
                "telemetry optional resource counters must be positive when set".to_string(),
            ));
        }
        Ok(())
    }

    /// Serialize as one JSONL record.
    pub fn to_jsonl(&self) -> Result<String> {
        self.validate()?;
        let encoded = serde_json::to_string(self).map_err(|error| {
            CognexisError::Backend(format!("telemetry serialization failed: {error}"))
        })?;
        Ok(format!("{encoded}\n"))
    }
}

/// Aggregate counters suitable for dashboards and alerts.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SafetyMetrics {
    pub request_count: u64,
    pub safety_refusals: u64,
    pub budget_exhaustions: u64,
    pub issue_counts: Vec<(SafetyIssue, u64)>,
}

impl SafetyMetrics {
    pub fn record(&mut self, telemetry: &RequestTelemetry) {
        self.request_count += 1;
        if telemetry.safety_action == SafetyAction::Refuse {
            self.safety_refusals += 1;
        }

        let mut issues = telemetry.input_issues.clone();
        for issue in &telemetry.output_issues {
            if !issues.contains(issue) {
                issues.push(*issue);
            }
        }
        if issues.contains(&SafetyIssue::BudgetExceeded) {
            self.budget_exhaustions += 1;
        }
        for issue in issues {
            increment_issue_count(&mut self.issue_counts, issue);
        }
    }

    pub fn issue_count(&self, issue: SafetyIssue) -> u64 {
        self.issue_counts
            .iter()
            .find_map(|(candidate, count)| (*candidate == issue).then_some(*count))
            .unwrap_or(0)
    }
}

/// Safety evaluation result for one loop depth.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SafetyDepthResult {
    pub loop_count: usize,
    pub evaluated_cases: usize,
    pub unsafe_outputs: usize,
    pub refusals: usize,
    pub budget_exhaustions: usize,
}

impl SafetyDepthResult {
    pub fn validate(&self) -> Result<()> {
        if self.loop_count == 0 {
            return Err(CognexisError::InvalidConfig(
                "safety depth loop_count must be positive".to_string(),
            ));
        }
        if self.unsafe_outputs > self.evaluated_cases
            || self.refusals > self.evaluated_cases
            || self.budget_exhaustions > self.evaluated_cases
        {
            return Err(CognexisError::InvalidConfig(
                "safety depth counts must not exceed evaluated_cases".to_string(),
            ));
        }
        Ok(())
    }

    pub fn unsafe_rate(&self) -> f64 {
        if self.evaluated_cases == 0 {
            return 0.0;
        }
        self.unsafe_outputs as f64 / self.evaluated_cases as f64
    }
}

/// Safety depth policy report.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SafetyDepthReport {
    pub baseline_loop_count: usize,
    pub baseline_unsafe_rate: f64,
    pub safe_loop_counts: Vec<usize>,
    pub restricted_loop_counts: Vec<usize>,
    pub worst_loop_count: Option<usize>,
}

/// Runtime policy derived from safety-depth evaluation.
///
/// The policy is intentionally conservative: serving code can only use
/// contiguous allowed loop ranges. If loop 3 is restricted, a later safe
/// result at loop 4 does not make loop 4 usable for adaptive scheduling
/// because the scheduler may halt at any intermediate depth.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LoopSafetyPolicy {
    pub allowed_loop_counts: Vec<usize>,
    pub restricted_loop_counts: Vec<usize>,
}

impl LoopSafetyPolicy {
    pub fn new(
        mut allowed_loop_counts: Vec<usize>,
        mut restricted_loop_counts: Vec<usize>,
    ) -> Result<Self> {
        allowed_loop_counts.sort_unstable();
        allowed_loop_counts.dedup();
        restricted_loop_counts.sort_unstable();
        restricted_loop_counts.dedup();
        let policy = Self {
            allowed_loop_counts,
            restricted_loop_counts,
        };
        policy.validate()?;
        Ok(policy)
    }

    pub fn from_depth_report(report: &SafetyDepthReport) -> Result<Self> {
        Self::new(
            report.safe_loop_counts.clone(),
            report.restricted_loop_counts.clone(),
        )
    }

    pub fn validate(&self) -> Result<()> {
        if self.allowed_loop_counts.is_empty() {
            return Err(CognexisError::InvalidConfig(
                "loop safety policy requires at least one allowed loop count".to_string(),
            ));
        }
        validate_loop_count_set("allowed_loop_counts", &self.allowed_loop_counts)?;
        validate_loop_count_set("restricted_loop_counts", &self.restricted_loop_counts)?;
        if self
            .allowed_loop_counts
            .iter()
            .any(|loops| self.restricted_loop_counts.contains(loops))
        {
            return Err(CognexisError::InvalidConfig(
                "loop safety policy counts cannot be both allowed and restricted".to_string(),
            ));
        }
        Ok(())
    }

    pub fn is_allowed(&self, loop_count: usize) -> bool {
        self.allowed_loop_counts.contains(&loop_count)
            && !self.restricted_loop_counts.contains(&loop_count)
    }

    pub fn safe_ceiling_from(
        &self,
        min_loop_count: usize,
        requested_max_loop_count: usize,
    ) -> Result<Option<usize>> {
        self.validate()?;
        if min_loop_count == 0 || requested_max_loop_count < min_loop_count {
            return Err(CognexisError::InvalidConfig(
                "loop safety policy bounds must satisfy requested_max >= min >= 1".to_string(),
            ));
        }

        let mut ceiling = None;
        for loop_count in min_loop_count..=requested_max_loop_count {
            if self.is_allowed(loop_count) {
                ceiling = Some(loop_count);
            } else {
                break;
            }
        }
        Ok(ceiling)
    }
}

/// Identify loop regimes that degrade safety relative to a baseline or absolute cap.
pub fn evaluate_safety_depth_regimes(
    results: &[SafetyDepthResult],
    baseline_loop_count: usize,
    max_unsafe_rate: f64,
    degradation_tolerance: f64,
) -> Result<SafetyDepthReport> {
    if results.is_empty() {
        return Err(CognexisError::InvalidConfig(
            "safety depth evaluation requires at least one result".to_string(),
        ));
    }
    if !max_unsafe_rate.is_finite()
        || !(0.0..=1.0).contains(&max_unsafe_rate)
        || !degradation_tolerance.is_finite()
        || degradation_tolerance < 0.0
    {
        return Err(CognexisError::InvalidConfig(
            "safety depth thresholds must be finite with max_unsafe_rate in [0, 1]".to_string(),
        ));
    }
    for result in results {
        result.validate()?;
    }
    let baseline = results
        .iter()
        .find(|result| result.loop_count == baseline_loop_count)
        .ok_or_else(|| {
            CognexisError::InvalidConfig(format!(
                "baseline loop count {baseline_loop_count} is not present in safety results"
            ))
        })?;
    let baseline_unsafe_rate = baseline.unsafe_rate();
    let mut safe_loop_counts = Vec::new();
    let mut restricted_loop_counts = Vec::new();
    let mut worst_loop_count = None;
    let mut worst_rate = f64::NEG_INFINITY;

    for result in results {
        let unsafe_rate = result.unsafe_rate();
        if unsafe_rate > worst_rate {
            worst_rate = unsafe_rate;
            worst_loop_count = Some(result.loop_count);
        }
        if unsafe_rate > max_unsafe_rate
            || unsafe_rate > baseline_unsafe_rate + degradation_tolerance
        {
            restricted_loop_counts.push(result.loop_count);
        } else {
            safe_loop_counts.push(result.loop_count);
        }
    }
    safe_loop_counts.sort_unstable();
    restricted_loop_counts.sort_unstable();

    Ok(SafetyDepthReport {
        baseline_loop_count,
        baseline_unsafe_rate,
        safe_loop_counts,
        restricted_loop_counts,
        worst_loop_count,
    })
}

/// Summarize per-token or per-step loop counts.
pub fn summarize_loop_counts(loop_counts: &[usize]) -> LoopCountsSummary {
    if loop_counts.is_empty() {
        return LoopCountsSummary::default();
    }

    let total = loop_counts.iter().sum::<usize>();
    let max = loop_counts.iter().copied().max().unwrap_or(0);
    let mut histogram = vec![0usize; max + 1];
    for &loops in loop_counts {
        histogram[loops] += 1;
    }

    LoopCountsSummary {
        total,
        mean: total as f64 / loop_counts.len() as f64,
        max,
        histogram,
    }
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
        "<s>",
        "</s>",
        "<pad>",
        "<unk>",
        "<eod>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|tool|>",
        "<|end|>",
    ]
    .iter()
    .any(|token| text.contains(token))
}

fn validate_loop_count_set(name: &str, counts: &[usize]) -> Result<()> {
    let mut previous = None;
    for &count in counts {
        if count == 0 {
            return Err(CognexisError::InvalidConfig(format!(
                "loop safety policy {name} must contain only positive counts"
            )));
        }
        if previous.map(|previous| count <= previous).unwrap_or(false) {
            return Err(CognexisError::InvalidConfig(format!(
                "loop safety policy {name} must be strictly sorted and unique"
            )));
        }
        previous = Some(count);
    }
    Ok(())
}

fn increment_issue_count(counts: &mut Vec<(SafetyIssue, u64)>, issue: SafetyIssue) {
    if let Some((_, count)) = counts.iter_mut().find(|(candidate, _)| *candidate == issue) {
        *count += 1;
    } else {
        counts.push((issue, 1));
        counts.sort_by_key(|(issue, _)| issue_order(*issue));
    }
}

fn issue_order(issue: SafetyIssue) -> u8 {
    match issue {
        SafetyIssue::OffensiveLanguage => 0,
        SafetyIssue::SensitiveInformation => 1,
        SafetyIssue::Misinformation => 2,
        SafetyIssue::SpecialTokenInjection => 3,
        SafetyIssue::BudgetExceeded => 4,
        SafetyIssue::Other => 5,
    }
}
