//! Top-level reference model and generation API.
//!
//! This module composes the tokenizer, embedding, Prelude, recurrent
//! core, Coda, and LM head into the public surface described by the
//! implementation outline. It is intentionally deterministic: sampling
//! controls filter logits, then the reference path selects the best
//! remaining token so tests are reproducible without an RNG backend.

use std::cmp::Ordering;
use std::fs;
use std::path::Path;
use std::time::Instant;

use crate::checkpoint::{load_optional_scheduler_state, CheckpointSchedulerState};
use crate::coda::Coda;
use crate::config::{CognexisConfig, InferenceConfig, ModelConfig, ServeConfig};
use crate::embedding::Embedding;
use crate::evaluation::estimate_forward_compute;
use crate::lm_head::LMHead;
use crate::prelude::Prelude;
use crate::recurrent_core::{RecurrentCore, RecurrentOptions};
use crate::safety::{
    check_safety, ComputeBudget, LoopSafetyPolicy, RequestTelemetry, SafetyAction, SafetyContext,
    SafetyDecision, SafetyIssue,
};
use crate::scheduler::{
    compute_loops_bounded, ActScheduler, HaltReason, LoopAction, LoopScheduling,
    RuleBasedScheduler, SchedulerObservation,
};
use crate::tokenizer::{
    DecodeOptions as TokenDecodeOptions, EncodeOptions as TokenEncodeOptions, TokenId, Tokenizer,
};
use crate::tokenwise::{apply_dense_masked_update, TokenLoopState, TokenwiseSchedule};
use crate::value_head::{ValueFeatures, ValueHead, ValuePooling};
use crate::{CognexisError, Result};

/// Request-time recurrent loop mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopMode {
    Fixed(usize),
    Adaptive { min_loops: usize, max_loops: usize },
    AdaptiveValue { min_loops: usize, max_loops: usize },
    TokenWise,
}

impl LoopMode {
    pub fn from_inference_config(config: &InferenceConfig) -> Result<Self> {
        let normalized = config
            .loop_mode
            .trim()
            .to_ascii_lowercase()
            .replace(['-', ' '], "_");
        match normalized.as_str() {
            "fixed" => Ok(Self::Fixed(config.max_loops)),
            "adaptive" | "adaptive_sequence" | "rule_based" => Ok(Self::Adaptive {
                min_loops: config.min_loops,
                max_loops: config.max_loops,
            }),
            "adaptive_value" | "value_head" | "hybrid" => Ok(Self::AdaptiveValue {
                min_loops: config.min_loops,
                max_loops: config.max_loops,
            }),
            "adaptive_token" | "tokenwise" | "token_wise" => Ok(Self::TokenWise),
            _ => Err(CognexisError::InvalidConfig(format!(
                "unsupported inference loop_mode {:?}",
                config.loop_mode
            ))),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Fixed(_) => "fixed",
            Self::Adaptive { .. } => "adaptive_sequence",
            Self::AdaptiveValue { .. } => "adaptive_value",
            Self::TokenWise => "adaptive_token",
        }
    }
}

/// Loop scheduling and hard-budget options for generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopOptions {
    pub mode: LoopMode,
    pub total_loop_budget: Option<usize>,
    pub max_prompt_tokens: Option<usize>,
}

impl Default for LoopOptions {
    fn default() -> Self {
        Self {
            mode: LoopMode::Fixed(1),
            total_loop_budget: None,
            max_prompt_tokens: None,
        }
    }
}

impl LoopOptions {
    pub fn from_inference_config(config: &InferenceConfig) -> Result<Self> {
        Ok(Self {
            mode: LoopMode::from_inference_config(config)?,
            total_loop_budget: None,
            max_prompt_tokens: Some(config.max_sequence_length),
        })
    }
}

/// Sampling controls applied after the LM head.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingOptions {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub eos_token_id: Option<TokenId>,
    pub stop_tokens: Vec<TokenId>,
}

impl Default for SamplingOptions {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            eos_token_id: Some(1),
            stop_tokens: Vec::new(),
        }
    }
}

impl SamplingOptions {
    pub fn from_inference_config(config: &InferenceConfig) -> Result<Self> {
        config.sampling.validate()?;
        Ok(Self {
            temperature: config.sampling.temperature,
            top_p: config.sampling.top_p,
            top_k: config.sampling.top_k,
            repetition_penalty: config.sampling.repetition_penalty,
            eos_token_id: config.sampling.eos_token_id,
            stop_tokens: config.sampling.stop_tokens.clone(),
        })
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    pub fn with_repetition_penalty(mut self, repetition_penalty: f32) -> Self {
        self.repetition_penalty = repetition_penalty;
        self
    }
}

/// Generation request matching the specification's inference contract.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationRequest {
    pub input_ids: Vec<TokenId>,
    pub max_new_tokens: usize,
    pub loop_options: LoopOptions,
    pub sampling: SamplingOptions,
}

impl GenerationRequest {
    pub fn new(input_ids: Vec<TokenId>, max_new_tokens: usize) -> Self {
        Self {
            input_ids,
            max_new_tokens,
            loop_options: LoopOptions::default(),
            sampling: SamplingOptions::default(),
        }
    }
}

/// Raw-text generation request. This is the safety-aware serving entry
/// point: callers provide user-visible text and the model applies the
/// checkpoint tokenizer and request safety context before recurrent compute.
#[derive(Debug, Clone, PartialEq)]
pub struct TextGenerationRequest {
    pub prompt: String,
    pub max_new_tokens: usize,
    pub loop_options: LoopOptions,
    pub sampling: SamplingOptions,
    pub safety_context: SafetyContext,
    pub loop_safety_policy: Option<LoopSafetyPolicy>,
}

impl TextGenerationRequest {
    pub fn new(prompt: impl Into<String>, max_new_tokens: usize) -> Self {
        Self {
            prompt: prompt.into(),
            max_new_tokens,
            loop_options: LoopOptions::default(),
            sampling: SamplingOptions::default(),
            safety_context: SafetyContext::default(),
            loop_safety_policy: None,
        }
    }

    pub fn from_inference_config(
        prompt: impl Into<String>,
        config: &InferenceConfig,
        safety_context: SafetyContext,
    ) -> Result<Self> {
        config.scheduler.validate()?;
        config.cache.validate()?;
        let mut safety_context = safety_context;
        merge_inference_budget(&mut safety_context.budget, config);
        Ok(Self {
            prompt: prompt.into(),
            max_new_tokens: config.max_new_tokens,
            loop_options: LoopOptions::from_inference_config(config)?,
            sampling: SamplingOptions::from_inference_config(config)?,
            safety_context,
            loop_safety_policy: None,
        })
    }
}

/// Final result for safety-aware raw-text generation.
#[derive(Debug, Clone, PartialEq)]
pub struct TextGenerationOutput {
    pub events: Vec<GenerationStepOutput>,
    pub generated_text: String,
    pub prompt_tokens: usize,
    pub safety_context: SafetyContext,
    pub estimated_recurrent_flops: Option<u64>,
    pub cache_memory_bytes: Option<usize>,
    pub wall_time_ms: Option<u64>,
}

impl TextGenerationOutput {
    /// Build content-free request telemetry for serving logs.
    pub fn telemetry(
        &self,
        request_id: impl Into<String>,
        loop_mode: impl Into<String>,
    ) -> RequestTelemetry {
        let loop_counts = self
            .events
            .iter()
            .filter_map(|event| (event.loop_count > 0).then_some(event.loop_count))
            .collect::<Vec<_>>();
        let generated_tokens = loop_counts.len();
        let mut telemetry = RequestTelemetry::from_context(
            request_id,
            &self.safety_context,
            self.prompt_tokens,
            generated_tokens,
            loop_mode,
            &loop_counts,
        );
        telemetry.halt_reasons = self
            .events
            .iter()
            .filter_map(|event| event.halt_reason)
            .map(halt_reason_label)
            .map(str::to_string)
            .collect();
        telemetry.stop_reason = self
            .events
            .iter()
            .rev()
            .find_map(|event| event.stop_reason)
            .map(stop_reason_label)
            .map(str::to_string);
        telemetry.estimated_recurrent_flops = self.estimated_recurrent_flops;
        telemetry.cache_memory_bytes = self.cache_memory_bytes;
        telemetry.wall_time_ms = self.wall_time_ms;
        telemetry
    }
}

/// Terminal reason for generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    MaxNewTokens,
    EosToken,
    StopToken,
    BudgetExhausted,
    Safety,
}

/// One generated-token event for streaming-style callers.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationStepOutput {
    pub token_id: TokenId,
    pub text_delta: String,
    pub loop_count: usize,
    pub effective_depth: usize,
    pub halt_reason: Option<HaltReason>,
    pub safety_issue: Option<SafetyIssue>,
    pub stop_reason: Option<StopReason>,
}

/// Scheduled forward pass output with loop diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct ForwardOutput {
    pub logits: Vec<Vec<f32>>,
    pub loop_count: usize,
    pub effective_depth: usize,
    pub halt_reason: Option<HaltReason>,
    pub token_loop_counts: Option<Vec<usize>>,
    pub token_halt_reasons: Option<Vec<Option<HaltReason>>>,
}

/// Deterministic CPU reference model.
pub struct CognexisModel {
    pub config: ModelConfig,
    pub embeddings: Embedding,
    pub prelude: Prelude,
    pub recurrent: RecurrentCore,
    pub coda: Coda,
    pub lm_head: LMHead,
    pub value_head: ValueHead,
    tokenizer: Tokenizer,
    scheduler: RuleBasedScheduler,
    act_scheduler: ActScheduler,
}

fn load_checkpoint_model_config(
    resolved_config_path: &Path,
) -> Result<(ModelConfig, Option<InferenceConfig>)> {
    let contents = fs::read_to_string(resolved_config_path).map_err(|error| {
        CognexisError::Backend(format!(
            "failed to read {}: {error}",
            resolved_config_path.display()
        ))
    })?;
    let value = serde_json::from_str::<serde_json::Value>(&contents).map_err(|error| {
        CognexisError::InvalidConfig(format!(
            "invalid resolved checkpoint config JSON {}: {error}",
            resolved_config_path.display()
        ))
    })?;

    let is_top_level_config = value
        .as_object()
        .map(|object| object.contains_key("model"))
        .unwrap_or(false);

    if is_top_level_config {
        let resolved = serde_json::from_value::<CognexisConfig>(value).map_err(|error| {
            CognexisError::InvalidConfig(format!("invalid resolved Cognexis config: {error}"))
        })?;
        resolved.validate()?;
        Ok((resolved.model, resolved.inference))
    } else {
        let config = serde_json::from_value::<ModelConfig>(value).map_err(|error| {
            CognexisError::InvalidConfig(format!("invalid resolved model config: {error}"))
        })?;
        config.validate()?;
        Ok((config, None))
    }
}

fn checkpoint_inference_with_scheduler_state(
    inference: Option<InferenceConfig>,
    config: &ModelConfig,
    scheduler_state: Option<&CheckpointSchedulerState>,
) -> Result<Option<InferenceConfig>> {
    let Some(scheduler_state) = scheduler_state else {
        return Ok(inference);
    };
    scheduler_state.validate()?;
    if let Some(inference) = &inference {
        if scheduler_state.scheduler != inference.scheduler {
            return Err(CognexisError::InvalidConfig(
                "checkpoint scheduler state does not match resolved inference scheduler config"
                    .to_string(),
            ));
        }
    }

    let mut inference = inference.unwrap_or_else(|| InferenceConfig {
        min_loops: config.min_loop_count,
        max_loops: config.max_loop_count,
        ..InferenceConfig::default()
    });
    inference.scheduler = scheduler_state.scheduler.clone();
    inference.validate(config)?;
    Ok(Some(inference))
}

impl CognexisModel {
    /// Build a reference model from a validated architecture config.
    pub fn new(config: ModelConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            embeddings: Embedding::new(&config),
            prelude: Prelude::new(&config),
            recurrent: RecurrentCore::new(&config),
            coda: Coda::new(&config),
            lm_head: LMHead::new(&config),
            value_head: ValueHead::new(&config),
            tokenizer: Tokenizer::new(),
            scheduler: RuleBasedScheduler::default(),
            act_scheduler: ActScheduler::default(),
            config,
        })
    }

    /// Apply request/runtime scheduler thresholds from inference config.
    pub fn with_inference_config(mut self, inference: &InferenceConfig) -> Result<Self> {
        self.apply_inference_config(inference)?;
        Ok(self)
    }

    pub fn apply_inference_config(&mut self, inference: &InferenceConfig) -> Result<()> {
        inference.validate(&self.config)?;
        self.scheduler.min_delta = inference.scheduler.min_delta;
        self.scheduler.confidence_threshold = inference.scheduler.confidence_threshold;
        self.scheduler.predicted_gain_threshold = inference.scheduler.predicted_gain_threshold;
        self.act_scheduler.config.gain_threshold = inference.scheduler.predicted_gain_threshold;
        self.act_scheduler.config.validate()?;
        Ok(())
    }

    /// Build a model from a checkpoint directory and serving config.
    ///
    /// The reference implementation accepts `config.resolved.json` when
    /// present and otherwise falls back to the supplied serve config.
    pub fn from_checkpoint(
        checkpoint_dir: impl AsRef<Path>,
        serve_config: &ServeConfig,
    ) -> Result<Self> {
        let resolved_config_path = checkpoint_dir.as_ref().join("config.resolved.json");
        let (config, checkpoint_inference) = if resolved_config_path.exists() {
            load_checkpoint_model_config(&resolved_config_path)?
        } else {
            serve_config.validate()?;
            (serve_config.model.clone(), None)
        };
        let checkpoint_scheduler_state = load_optional_scheduler_state(checkpoint_dir.as_ref())?;
        let checkpoint_inference = checkpoint_inference_with_scheduler_state(
            checkpoint_inference,
            &config,
            checkpoint_scheduler_state.as_ref(),
        )?;

        serve_config.validate_for_model(&config)?;
        let mut model = Self::new(config)?;
        if let Some(inference) = checkpoint_inference {
            model.apply_inference_config(&inference)?;
        }
        if let Some(scheduler_state) = checkpoint_scheduler_state {
            model.act_scheduler.config = scheduler_state.act;
            model.value_head.config = scheduler_state.value_head;
        }
        Ok(model)
    }

    /// Compute logits for a token sequence at a fixed recurrent depth.
    pub fn forward_logits(&self, token_ids: &[TokenId], loops: usize) -> Result<Vec<Vec<f32>>> {
        let embedded = self.embeddings.try_forward(token_ids)?;
        let prepared = self.prelude.forward(&embedded);
        let refined = self
            .recurrent
            .forward_with_options(
                &prepared,
                RecurrentOptions {
                    loops,
                    retain_intermediate_states: false,
                },
            )?
            .hidden;
        let hidden = self.coda.forward(&refined);
        self.lm_head.try_forward(&hidden)
    }

    /// Compute logits using the same loop scheduler semantics as generation.
    pub fn forward_with_loop_options(
        &self,
        token_ids: &[TokenId],
        loop_options: &LoopOptions,
    ) -> Result<ForwardOutput> {
        self.forward_scheduled(token_ids, loop_options, None)
    }

    /// Generate a deterministic sequence and return streaming-style events.
    pub fn generate_streaming(
        &self,
        request: GenerationRequest,
    ) -> Result<Vec<GenerationStepOutput>> {
        self.generate_streaming_with_wall_clock(request, None, None)
    }

    fn generate_streaming_with_wall_clock(
        &self,
        request: GenerationRequest,
        started_at: Option<Instant>,
        max_wall_time_ms: Option<u64>,
    ) -> Result<Vec<GenerationStepOutput>> {
        self.validate_generation_request(&request)?;

        let mut context = request.input_ids.clone();
        let mut generated = Vec::with_capacity(request.max_new_tokens);
        let mut events = Vec::with_capacity(request.max_new_tokens);
        let mut remaining_budget = request.loop_options.total_loop_budget;
        let mut decoder = self
            .tokenizer
            .streaming_decoder(TokenDecodeOptions::default());

        for _ in 0..request.max_new_tokens {
            if wall_clock_budget_exceeded(started_at, max_wall_time_ms) {
                mark_or_push_budget_stop(&mut events);
                break;
            }

            let forward =
                self.forward_scheduled(&context, &request.loop_options, remaining_budget)?;
            if forward.loop_count == 0 {
                mark_or_push_budget_stop(&mut events);
                break;
            }
            if let Some(budget) = &mut remaining_budget {
                if *budget < forward.loop_count {
                    mark_or_push_budget_stop(&mut events);
                    break;
                }
                *budget -= forward.loop_count;
            }

            let next_logits = forward.logits.last().ok_or_else(|| {
                CognexisError::Backend("LM head returned no logits for generation".to_string())
            })?;
            let token_id = select_token(next_logits, &request.sampling, &context)?;
            let text_delta = decoder.push(token_id)?;

            context.push(token_id);
            generated.push(token_id);

            let generated_text = self
                .tokenizer
                .decode_with_options(&generated, TokenDecodeOptions::default())?;
            let safety_issue = check_safety(&generated_text);
            let stop_reason = stop_reason_for_token(token_id, &request.sampling, safety_issue);

            events.push(GenerationStepOutput {
                token_id,
                text_delta,
                loop_count: forward.loop_count,
                effective_depth: forward.effective_depth,
                halt_reason: forward.halt_reason,
                safety_issue,
                stop_reason,
            });

            if stop_reason.is_some() {
                return Ok(events);
            }

            if wall_clock_budget_exceeded(started_at, max_wall_time_ms) {
                mark_or_push_budget_stop(&mut events);
                break;
            }
        }

        if let Some(last) = events.last_mut() {
            if last.stop_reason.is_none() {
                last.stop_reason = Some(StopReason::MaxNewTokens);
            }
        }

        Ok(events)
    }

    /// Blocking convenience wrapper returning decoded generated text.
    pub fn generate(&self, request: GenerationRequest) -> Result<String> {
        let events = self.generate_streaming(request)?;
        Ok(events
            .into_iter()
            .filter(|event| {
                !matches!(
                    event.stop_reason,
                    Some(StopReason::BudgetExhausted) if event.loop_count == 0
                )
            })
            .map(|event| event.text_delta)
            .collect())
    }

    /// Safety-aware raw-text streaming generation.
    pub fn generate_text_streaming(
        &self,
        request: TextGenerationRequest,
    ) -> Result<TextGenerationOutput> {
        if request.max_new_tokens == 0 {
            return Err(CognexisError::InvalidConfig(
                "max_new_tokens must be positive".to_string(),
            ));
        }
        let request_started_at = Instant::now();

        let TextGenerationRequest {
            prompt,
            max_new_tokens,
            loop_options,
            sampling,
            mut safety_context,
            loop_safety_policy,
        } = request;
        safety_context.budget.validate()?;

        let pre_tokenization_decision = safety_context.inspect_input(&prompt, 0);
        if should_stop_before_tokenization(&pre_tokenization_decision) {
            let stop_reason = input_decision_stop_reason(&pre_tokenization_decision);
            return Ok(terminal_text_output(
                safety_context,
                0,
                pre_tokenization_decision.issues.first().copied(),
                stop_reason,
            ));
        }

        let input_ids = self.tokenizer.encode_with_options(
            &prompt,
            TokenEncodeOptions {
                add_bos: true,
                allow_special: false,
                ..TokenEncodeOptions::default()
            },
        )?;
        let prompt_tokens = input_ids.len();
        let input_decision = safety_context.inspect_input(&prompt, prompt_tokens);
        if should_stop_after_tokenization(&input_decision) {
            let stop_reason = input_decision_stop_reason(&input_decision);
            return Ok(terminal_text_output(
                safety_context,
                prompt_tokens,
                input_decision.issues.first().copied(),
                stop_reason,
            ));
        }

        if max_new_tokens > safety_context.budget.max_generated_tokens {
            safety_context
                .output_flags
                .push_unique(SafetyIssue::BudgetExceeded);
            return Ok(terminal_text_output(
                safety_context,
                prompt_tokens,
                Some(SafetyIssue::BudgetExceeded),
                StopReason::BudgetExhausted,
            ));
        }

        let loop_options = if let Some(policy) = loop_safety_policy.as_ref() {
            match loop_options_with_safety_policy(loop_options, policy, &self.config)? {
                Some(loop_options) => loop_options,
                None => {
                    safety_context.output_flags.push_unique(SafetyIssue::Other);
                    return Ok(terminal_text_output(
                        safety_context,
                        prompt_tokens,
                        Some(SafetyIssue::Other),
                        StopReason::Safety,
                    ));
                }
            }
        } else {
            loop_options
        };

        let constrained_loop_options =
            loop_options_with_budget_constraints(loop_options, &safety_context.budget);
        let loop_ceiling = loop_options_max_loops(&constrained_loop_options, &self.config);
        let loop_decision = safety_context.check_loop_budget(loop_ceiling, loop_ceiling);
        if should_stop_for_compute_budget(&loop_decision) {
            return Ok(terminal_text_output(
                safety_context,
                prompt_tokens,
                Some(SafetyIssue::BudgetExceeded),
                StopReason::BudgetExhausted,
            ));
        }

        let resource_estimate = estimate_generation_resources(
            &self.config,
            prompt_tokens,
            max_new_tokens,
            loop_ceiling,
        )?;
        let resource_decision = safety_context.check_resource_budget(
            Some(resource_estimate.recurrent_flops),
            None,
            Some(resource_estimate.cache_memory_bytes),
        );
        if should_stop_for_compute_budget(&resource_decision) {
            return Ok(terminal_text_output_with_resources(
                safety_context,
                prompt_tokens,
                Some(SafetyIssue::BudgetExceeded),
                StopReason::BudgetExhausted,
                Some(resource_estimate.recurrent_flops),
                Some(resource_estimate.cache_memory_bytes),
                None,
            ));
        }

        let pre_generation_wall_time_ms = elapsed_wall_time_ms(request_started_at);
        let wall_time_decision =
            safety_context.check_resource_budget(None, Some(pre_generation_wall_time_ms), None);
        if should_stop_for_compute_budget(&wall_time_decision) {
            return Ok(terminal_text_output_with_resources(
                safety_context,
                prompt_tokens,
                Some(SafetyIssue::BudgetExceeded),
                StopReason::BudgetExhausted,
                Some(resource_estimate.recurrent_flops),
                Some(resource_estimate.cache_memory_bytes),
                Some(pre_generation_wall_time_ms),
            ));
        }

        let mut events = self.generate_streaming_with_wall_clock(
            GenerationRequest {
                input_ids,
                max_new_tokens,
                loop_options: constrained_loop_options,
                sampling,
            },
            Some(request_started_at),
            safety_context.budget.max_wall_time_ms,
        )?;

        let wall_time_ms = elapsed_wall_time_ms(request_started_at);
        let wall_time_decision =
            safety_context.check_resource_budget(None, Some(wall_time_ms), None);
        if should_stop_for_compute_budget(&wall_time_decision) {
            mark_or_push_budget_stop(&mut events);
        }

        let generated_text = events
            .iter()
            .map(|event| event.text_delta.as_str())
            .collect::<String>();
        let mut total_loops = 0usize;
        for event in &events {
            if let Some(issue) = event.safety_issue {
                safety_context.output_flags.push_unique(issue);
            }
            if event.loop_count > 0 {
                total_loops = total_loops.saturating_add(event.loop_count);
                let loop_decision = safety_context.check_loop_budget(event.loop_count, total_loops);
                if loop_decision.issues.contains(&SafetyIssue::BudgetExceeded) {
                    safety_context
                        .output_flags
                        .push_unique(SafetyIssue::BudgetExceeded);
                }
            }
            if event.stop_reason == Some(StopReason::BudgetExhausted) {
                safety_context
                    .output_flags
                    .push_unique(SafetyIssue::BudgetExceeded);
            }
        }

        let generated_tokens = events.iter().filter(|event| event.loop_count > 0).count();
        safety_context.inspect_output(&generated_text, generated_tokens);

        Ok(TextGenerationOutput {
            events,
            generated_text,
            prompt_tokens,
            safety_context,
            estimated_recurrent_flops: Some(resource_estimate.recurrent_flops),
            cache_memory_bytes: Some(resource_estimate.cache_memory_bytes),
            wall_time_ms: Some(wall_time_ms),
        })
    }

    /// Blocking raw-text convenience wrapper returning decoded generated text.
    pub fn generate_text(&self, request: TextGenerationRequest) -> Result<String> {
        self.generate_text_streaming(request)
            .map(|output| output.generated_text)
    }

    fn validate_generation_request(&self, request: &GenerationRequest) -> Result<()> {
        if request.input_ids.is_empty() {
            return Err(CognexisError::InvalidConfig(
                "generation input_ids must not be empty".to_string(),
            ));
        }
        if request.max_new_tokens == 0 {
            return Err(CognexisError::InvalidConfig(
                "max_new_tokens must be positive".to_string(),
            ));
        }
        if let Some(max_prompt_tokens) = request.loop_options.max_prompt_tokens {
            if request.input_ids.len() > max_prompt_tokens {
                return Err(CognexisError::InvalidConfig(format!(
                    "prompt has {} tokens but max_prompt_tokens is {}",
                    request.input_ids.len(),
                    max_prompt_tokens
                )));
            }
        }
        if !request.sampling.temperature.is_finite() || request.sampling.temperature < 0.0 {
            return Err(CognexisError::InvalidConfig(
                "sampling temperature must be finite and non-negative".to_string(),
            ));
        }
        if !request.sampling.top_p.is_finite()
            || request.sampling.top_p <= 0.0
            || request.sampling.top_p > 1.0
        {
            return Err(CognexisError::InvalidConfig(
                "top_p must be in the interval (0, 1]".to_string(),
            ));
        }
        if !request.sampling.repetition_penalty.is_finite()
            || request.sampling.repetition_penalty <= 0.0
        {
            return Err(CognexisError::InvalidConfig(
                "repetition_penalty must be finite and positive".to_string(),
            ));
        }
        Ok(())
    }

    fn forward_scheduled(
        &self,
        token_ids: &[TokenId],
        loop_options: &LoopOptions,
        remaining_budget: Option<usize>,
    ) -> Result<ForwardOutput> {
        match loop_options.mode {
            LoopMode::TokenWise => self.forward_tokenwise(token_ids, remaining_budget),
            LoopMode::AdaptiveValue {
                min_loops,
                max_loops,
            } => self.forward_value_adaptive(
                token_ids,
                min_loops.max(self.config.min_loop_count),
                max_loops
                    .min(self.config.max_loop_count)
                    .max(self.config.min_loop_count),
                remaining_budget,
            ),
            _ => {
                let scheduled =
                    self.schedule_loops(loop_options, token_ids.len(), remaining_budget);
                if scheduled.loop_count == 0 {
                    return Ok(ForwardOutput {
                        logits: Vec::new(),
                        loop_count: 0,
                        effective_depth: 0,
                        halt_reason: scheduled.halt_reason,
                        token_loop_counts: None,
                        token_halt_reasons: None,
                    });
                }
                let logits = self.forward_logits(token_ids, scheduled.loop_count)?;
                Ok(ForwardOutput {
                    logits,
                    loop_count: scheduled.loop_count,
                    effective_depth: self.config.effective_depth(scheduled.loop_count),
                    halt_reason: scheduled.halt_reason,
                    token_loop_counts: None,
                    token_halt_reasons: None,
                })
            }
        }
    }

    fn forward_value_adaptive(
        &self,
        token_ids: &[TokenId],
        min_loops: usize,
        max_loops: usize,
        remaining_budget: Option<usize>,
    ) -> Result<ForwardOutput> {
        let hard_max = remaining_budget.map_or(max_loops, |budget| budget.min(max_loops));
        if hard_max == 0 {
            return Ok(ForwardOutput {
                logits: Vec::new(),
                loop_count: 0,
                effective_depth: 0,
                halt_reason: Some(HaltReason::Budget),
                token_loop_counts: None,
                token_halt_reasons: None,
            });
        }

        let embedded = self.embeddings.try_forward(token_ids)?;
        let anchor = self.prelude.forward(&embedded);
        let mut current = anchor.clone();
        let mut previous: Option<Vec<Vec<f32>>> = None;
        let mut loops_executed = 0usize;

        loop {
            let remaining_loop_budget =
                remaining_budget.map(|budget| budget.saturating_sub(loops_executed));
            if loops_executed >= hard_max {
                return self.finish_scheduled_forward(
                    current,
                    loops_executed,
                    Some(HaltReason::MaxLoops),
                );
            }

            let hidden_delta = previous
                .as_ref()
                .map(|previous| mean_hidden_delta(previous, &current))
                .transpose()?;
            let interim_hidden = self.coda.forward(&current);
            let interim_logits = self.lm_head.try_forward(&interim_hidden)?;
            let (confidence, entropy) = interim_logits
                .last()
                .map(|logits| confidence_and_entropy(logits))
                .transpose()?
                .unwrap_or((None, None));
            let prediction = self.value_head.predict(
                &current,
                &ValueFeatures {
                    loop_index: loops_executed,
                    max_loops: hard_max,
                    confidence,
                    entropy,
                    hidden_delta,
                    loop_cost: 1.0,
                    predicted_risk: 0.0,
                    ..ValueFeatures::default()
                },
            )?;
            let decision = self.act_scheduler.decide_from_value_prediction(
                &prediction,
                0,
                loops_executed,
                min_loops.min(hard_max),
                hard_max,
                remaining_loop_budget,
                false,
            )?;
            if decision.decision.action == LoopAction::Halt {
                return self.finish_scheduled_forward(
                    current,
                    loops_executed,
                    decision.decision.halt_reason,
                );
            }

            let next = self
                .recurrent
                .forward_one_loop(&anchor, &current, loops_executed)?;
            previous = Some(current);
            current = next;
            loops_executed += 1;
        }
    }

    fn forward_tokenwise(
        &self,
        token_ids: &[TokenId],
        remaining_budget: Option<usize>,
    ) -> Result<ForwardOutput> {
        let hard_max = remaining_budget.map_or(self.config.max_loop_count, |budget| {
            budget.min(self.config.max_loop_count)
        });
        if hard_max == 0 {
            return Ok(ForwardOutput {
                logits: Vec::new(),
                loop_count: 0,
                effective_depth: 0,
                halt_reason: Some(HaltReason::Budget),
                token_loop_counts: Some(vec![0; token_ids.len()]),
                token_halt_reasons: Some(vec![Some(HaltReason::Budget); token_ids.len()]),
            });
        }

        let embedded = self.embeddings.try_forward(token_ids)?;
        let anchor = self.prelude.forward(&embedded);
        let mut current = anchor.clone();
        let mut state =
            TokenLoopState::new(TokenwiseSchedule::fixed(anchor.len(), hard_max), None)?;
        let min_loops = self.config.min_loop_count.min(hard_max);
        let mut dense_passes = 0usize;

        while state.any_active() && dense_passes < hard_max {
            let previous = current.clone();
            let candidate = self
                .recurrent
                .forward_one_loop(&anchor, &current, dense_passes)?;
            current = apply_dense_masked_update(&current, &candidate, &state.active)?;
            state.record_loop();
            dense_passes += 1;

            if !state.any_active() || dense_passes < min_loops {
                continue;
            }

            let deltas = per_token_hidden_delta(&previous, &current)?;
            let coda_hidden = self.coda.forward(&current);
            let logits = self.lm_head.try_forward(&coda_hidden)?;
            let (token_confidences, entropy_mean) = confidence_by_token(&logits)?;
            let prediction = self.value_head.predict(
                &current,
                &ValueFeatures {
                    loop_index: dense_passes,
                    max_loops: hard_max,
                    confidence: mean_present_f32(token_confidences.iter().copied().map(Some)),
                    entropy: entropy_mean,
                    hidden_delta: mean_present_f32(deltas.iter().copied().map(Some)),
                    loop_cost: 1.0,
                    predicted_risk: 0.0,
                    non_pad_mask: Some(state.active.clone()),
                    pooling: ValuePooling::TokenWise,
                },
            )?;
            let halt_reasons = tokenwise_halt_reasons(
                &state,
                &prediction.predicted_gain,
                &token_confidences,
                &deltas,
                self.scheduler.min_delta,
                self.scheduler.confidence_threshold,
                self.scheduler.predicted_gain_threshold,
                min_loops,
            )?;
            state.halt_where(&halt_reasons)?;
        }

        let hidden = self.coda.forward(&current);
        let logits = self.lm_head.try_forward(&hidden)?;
        let max_loops = state.max_loops();
        let halt_reason = if state
            .halt_reasons
            .iter()
            .any(|reason| *reason == Some(HaltReason::Budget))
        {
            Some(HaltReason::Budget)
        } else if state
            .halt_reasons
            .iter()
            .any(|reason| *reason == Some(HaltReason::MaxLoops))
        {
            Some(HaltReason::MaxLoops)
        } else {
            Some(HaltReason::ValueGain)
        };

        Ok(ForwardOutput {
            logits,
            loop_count: max_loops,
            effective_depth: self.config.effective_depth(max_loops),
            halt_reason,
            token_loop_counts: Some(state.loops),
            token_halt_reasons: Some(state.halt_reasons),
        })
    }

    fn finish_scheduled_forward(
        &self,
        recurrent_hidden: Vec<Vec<f32>>,
        loop_count: usize,
        halt_reason: Option<HaltReason>,
    ) -> Result<ForwardOutput> {
        let hidden = self.coda.forward(&recurrent_hidden);
        let logits = self.lm_head.try_forward(&hidden)?;
        Ok(ForwardOutput {
            logits,
            loop_count,
            effective_depth: self.config.effective_depth(loop_count),
            halt_reason,
            token_loop_counts: None,
            token_halt_reasons: None,
        })
    }

    fn schedule_loops(
        &self,
        loop_options: &LoopOptions,
        context_len: usize,
        remaining_budget: Option<usize>,
    ) -> ScheduledLoops {
        if remaining_budget == Some(0) {
            return ScheduledLoops {
                loop_count: 0,
                halt_reason: Some(HaltReason::Budget),
            };
        }

        let config_min = self.config.min_loop_count;
        let config_max = self.config.max_loop_count;
        match loop_options.mode {
            LoopMode::Fixed(loops) => {
                let bounded = compute_loops_bounded(
                    LoopScheduling::Fixed(loops),
                    context_len,
                    config_min,
                    config_max,
                    remaining_budget,
                );
                ScheduledLoops {
                    loop_count: bounded,
                    halt_reason: None,
                }
            }
            LoopMode::Adaptive {
                min_loops,
                max_loops,
            } => self.schedule_adaptive(
                min_loops.max(config_min),
                max_loops.min(config_max).max(config_min),
                context_len,
                remaining_budget,
            ),
            LoopMode::AdaptiveValue {
                min_loops,
                max_loops,
            } => self.schedule_adaptive(
                min_loops.max(config_min),
                max_loops.min(config_max).max(config_min),
                context_len,
                remaining_budget,
            ),
            LoopMode::TokenWise => {
                let bounded = compute_loops_bounded(
                    LoopScheduling::Adaptive,
                    context_len,
                    config_min,
                    config_max,
                    remaining_budget,
                );
                ScheduledLoops {
                    loop_count: bounded,
                    halt_reason: None,
                }
            }
        }
    }

    fn schedule_adaptive(
        &self,
        min_loops: usize,
        max_loops: usize,
        context_len: usize,
        remaining_budget: Option<usize>,
    ) -> ScheduledLoops {
        let hard_max = remaining_budget.map_or(max_loops, |budget| budget.min(max_loops));
        if hard_max == 0 {
            return ScheduledLoops {
                loop_count: 0,
                halt_reason: Some(HaltReason::Budget),
            };
        }

        let mut loops_executed = 0usize;
        loop {
            let remaining_loop_budget =
                remaining_budget.map(|budget| budget.saturating_sub(loops_executed));
            let depth = loops_executed.max(1) as f32;
            let complexity = context_len.max(1) as f32;
            let observation = SchedulerObservation {
                loops_executed,
                min_loops: min_loops.min(hard_max),
                max_loops: hard_max,
                hidden_delta: Some(1.0 / (complexity + depth)),
                confidence: Some((depth / (depth + 2.0)).min(0.99)),
                predicted_gain: Some(1.0 / ((loops_executed + 2) * (loops_executed + 2)) as f32),
                remaining_loop_budget,
                safety_halt: false,
            };
            let decision = self.scheduler.decide(observation);
            if decision.action == crate::scheduler::LoopAction::Halt {
                return ScheduledLoops {
                    loop_count: loops_executed,
                    halt_reason: decision.halt_reason,
                };
            }

            loops_executed += 1;
            if loops_executed >= hard_max {
                return ScheduledLoops {
                    loop_count: loops_executed,
                    halt_reason: Some(HaltReason::MaxLoops),
                };
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ScheduledLoops {
    loop_count: usize,
    halt_reason: Option<HaltReason>,
}

fn stop_reason_for_token(
    token_id: TokenId,
    sampling: &SamplingOptions,
    safety_issue: Option<SafetyIssue>,
) -> Option<StopReason> {
    if safety_issue.is_some() {
        return Some(StopReason::Safety);
    }
    if sampling.eos_token_id == Some(token_id) {
        return Some(StopReason::EosToken);
    }
    if sampling.stop_tokens.contains(&token_id) {
        return Some(StopReason::StopToken);
    }
    None
}

fn halt_reason_label(reason: HaltReason) -> &'static str {
    match reason {
        HaltReason::MaxLoops => "max_loops",
        HaltReason::MinDelta => "min_delta",
        HaltReason::Confidence => "confidence",
        HaltReason::ValueGain => "value_gain",
        HaltReason::Budget => "budget",
        HaltReason::Safety => "safety",
        HaltReason::Forced => "forced",
    }
}

fn stop_reason_label(reason: StopReason) -> &'static str {
    match reason {
        StopReason::MaxNewTokens => "max_new_tokens",
        StopReason::EosToken => "eos_token",
        StopReason::StopToken => "stop_token",
        StopReason::BudgetExhausted => "budget_exhausted",
        StopReason::Safety => "safety",
    }
}

fn elapsed_wall_time_ms(started_at: Instant) -> u64 {
    let elapsed = started_at.elapsed().as_millis();
    u64::try_from(elapsed).unwrap_or(u64::MAX).max(1)
}

fn wall_clock_budget_exceeded(started_at: Option<Instant>, max_wall_time_ms: Option<u64>) -> bool {
    match (started_at, max_wall_time_ms) {
        (Some(started_at), Some(limit)) => elapsed_wall_time_ms(started_at) > limit,
        _ => false,
    }
}

fn mark_or_push_budget_stop(events: &mut Vec<GenerationStepOutput>) {
    if let Some(last) = events.last_mut() {
        last.stop_reason = Some(StopReason::BudgetExhausted);
        return;
    }

    events.push(GenerationStepOutput {
        token_id: 0,
        text_delta: String::new(),
        loop_count: 0,
        effective_depth: 0,
        halt_reason: Some(HaltReason::Budget),
        safety_issue: None,
        stop_reason: Some(StopReason::BudgetExhausted),
    });
}

fn terminal_text_output(
    safety_context: SafetyContext,
    prompt_tokens: usize,
    issue: Option<SafetyIssue>,
    stop_reason: StopReason,
) -> TextGenerationOutput {
    terminal_text_output_with_resources(
        safety_context,
        prompt_tokens,
        issue,
        stop_reason,
        None,
        None,
        None,
    )
}

fn terminal_text_output_with_resources(
    safety_context: SafetyContext,
    prompt_tokens: usize,
    issue: Option<SafetyIssue>,
    stop_reason: StopReason,
    estimated_recurrent_flops: Option<u64>,
    cache_memory_bytes: Option<usize>,
    wall_time_ms: Option<u64>,
) -> TextGenerationOutput {
    TextGenerationOutput {
        events: vec![GenerationStepOutput {
            token_id: 0,
            text_delta: String::new(),
            loop_count: 0,
            effective_depth: 0,
            halt_reason: match stop_reason {
                StopReason::BudgetExhausted => Some(HaltReason::Budget),
                StopReason::Safety => Some(HaltReason::Safety),
                _ => None,
            },
            safety_issue: issue,
            stop_reason: Some(stop_reason),
        }],
        generated_text: String::new(),
        prompt_tokens,
        safety_context,
        estimated_recurrent_flops,
        cache_memory_bytes,
        wall_time_ms,
    }
}

fn should_stop_before_tokenization(decision: &SafetyDecision) -> bool {
    decision.action == SafetyAction::Refuse
        || decision
            .issues
            .contains(&SafetyIssue::SpecialTokenInjection)
}

fn should_stop_after_tokenization(decision: &SafetyDecision) -> bool {
    should_stop_before_tokenization(decision)
        || decision.issues.contains(&SafetyIssue::BudgetExceeded)
}

fn should_stop_for_compute_budget(decision: &SafetyDecision) -> bool {
    decision.action == SafetyAction::Refuse
        || decision.issues.contains(&SafetyIssue::BudgetExceeded)
}

fn input_decision_stop_reason(decision: &SafetyDecision) -> StopReason {
    if decision.issues.contains(&SafetyIssue::BudgetExceeded) {
        StopReason::BudgetExhausted
    } else {
        StopReason::Safety
    }
}

fn loop_options_max_loops(loop_options: &LoopOptions, config: &ModelConfig) -> usize {
    match loop_options.mode {
        LoopMode::Fixed(loops) => loops.max(config.min_loop_count).min(config.max_loop_count),
        LoopMode::Adaptive {
            min_loops,
            max_loops,
        }
        | LoopMode::AdaptiveValue {
            min_loops,
            max_loops,
        } => max_loops
            .max(min_loops)
            .max(config.min_loop_count)
            .min(config.max_loop_count),
        LoopMode::TokenWise => config.max_loop_count,
    }
}

fn loop_options_with_budget_constraints(
    mut loop_options: LoopOptions,
    budget: &ComputeBudget,
) -> LoopOptions {
    loop_options.max_prompt_tokens = Some(
        loop_options
            .max_prompt_tokens
            .map(|existing| existing.min(budget.max_prompt_tokens))
            .unwrap_or(budget.max_prompt_tokens),
    );
    loop_options.total_loop_budget = match (loop_options.total_loop_budget, budget.max_total_loops)
    {
        (Some(request), Some(policy)) => Some(request.min(policy)),
        (Some(request), None) => Some(request),
        (None, Some(policy)) => Some(policy),
        (None, None) => None,
    };
    loop_options
}

fn merge_inference_budget(budget: &mut ComputeBudget, config: &InferenceConfig) {
    budget.max_prompt_tokens = budget.max_prompt_tokens.min(config.max_sequence_length);
    budget.max_generated_tokens = budget.max_generated_tokens.min(config.max_new_tokens);
    budget.max_loops_per_token = budget.max_loops_per_token.min(config.max_loops);
    if let Some(configured) = config.compute_budget {
        budget.max_prompt_tokens = budget.max_prompt_tokens.min(configured.max_prompt_tokens);
        budget.max_generated_tokens = budget
            .max_generated_tokens
            .min(configured.max_generated_tokens);
        budget.max_loops_per_token = budget
            .max_loops_per_token
            .min(configured.max_loops_per_token);
        budget.max_total_loops =
            min_optional_usize(budget.max_total_loops, configured.max_total_loops);
        budget.max_recurrent_flops =
            min_optional_u64(budget.max_recurrent_flops, configured.max_recurrent_flops);
        budget.max_wall_time_ms =
            min_optional_u64(budget.max_wall_time_ms, configured.max_wall_time_ms);
        budget.max_cache_memory_bytes = min_optional_usize(
            budget.max_cache_memory_bytes,
            configured.max_cache_memory_bytes,
        );
    }
    budget.max_cache_memory_bytes = min_optional_usize(
        budget.max_cache_memory_bytes,
        config.cache.max_cache_memory_bytes,
    );
}

fn min_optional_usize(left: Option<usize>, right: Option<usize>) -> Option<usize> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}

fn min_optional_u64(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}

fn loop_options_with_safety_policy(
    mut loop_options: LoopOptions,
    policy: &LoopSafetyPolicy,
    config: &ModelConfig,
) -> Result<Option<LoopOptions>> {
    let requested_ceiling = loop_options_max_loops(&loop_options, config);
    let Some(safe_ceiling) = policy.safe_ceiling_from(config.min_loop_count, requested_ceiling)?
    else {
        return Ok(None);
    };

    loop_options.mode = match loop_options.mode {
        LoopMode::Fixed(loops) => {
            LoopMode::Fixed(loops.min(safe_ceiling).max(config.min_loop_count))
        }
        LoopMode::Adaptive {
            min_loops,
            max_loops,
        } => LoopMode::Adaptive {
            min_loops: min_loops.min(safe_ceiling).max(config.min_loop_count),
            max_loops: max_loops.min(safe_ceiling).max(config.min_loop_count),
        },
        LoopMode::AdaptiveValue {
            min_loops,
            max_loops,
        } => LoopMode::AdaptiveValue {
            min_loops: min_loops.min(safe_ceiling).max(config.min_loop_count),
            max_loops: max_loops.min(safe_ceiling).max(config.min_loop_count),
        },
        LoopMode::TokenWise => LoopMode::Adaptive {
            min_loops: config.min_loop_count,
            max_loops: safe_ceiling,
        },
    };
    Ok(Some(loop_options))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GenerationResourceEstimate {
    recurrent_flops: u64,
    cache_memory_bytes: usize,
}

fn estimate_generation_resources(
    config: &ModelConfig,
    prompt_tokens: usize,
    max_new_tokens: usize,
    loop_count: usize,
) -> Result<GenerationResourceEstimate> {
    let mut recurrent_flops = 0u64;
    let mut cache_memory_bytes = 0usize;
    for step in 0..max_new_tokens {
        let sequence_len = prompt_tokens + step;
        let estimate = estimate_forward_compute(config, sequence_len.max(1), loop_count)?;
        recurrent_flops =
            recurrent_flops.saturating_add(f64_to_u64_saturating(estimate.recurrent_flops));
        cache_memory_bytes = cache_memory_bytes.max(estimate.kv_cache_memory_bytes);
    }

    Ok(GenerationResourceEstimate {
        recurrent_flops,
        cache_memory_bytes,
    })
}

fn f64_to_u64_saturating(value: f64) -> u64 {
    if !value.is_finite() || value <= 0.0 {
        0
    } else if value >= u64::MAX as f64 {
        u64::MAX
    } else {
        value.ceil() as u64
    }
}

fn per_token_hidden_delta(previous: &[Vec<f32>], current: &[Vec<f32>]) -> Result<Vec<f32>> {
    if previous.len() != current.len() {
        return Err(CognexisError::ShapeMismatch {
            expected: format!("{} previous rows", previous.len()),
            actual: format!("{} current rows", current.len()),
        });
    }

    previous
        .iter()
        .zip(current)
        .enumerate()
        .map(|(row_index, (previous_row, current_row))| {
            if previous_row.len() != current_row.len() {
                return Err(CognexisError::ShapeMismatch {
                    expected: format!("row {row_index} width {}", previous_row.len()),
                    actual: format!("row {row_index} width {}", current_row.len()),
                });
            }
            let delta = previous_row
                .iter()
                .zip(current_row)
                .map(|(left, right)| {
                    let diff = right - left;
                    diff * diff
                })
                .sum::<f32>()
                .sqrt();
            if !delta.is_finite() {
                return Err(CognexisError::Backend(
                    "token hidden delta contains non-finite values".to_string(),
                ));
            }
            Ok(delta)
        })
        .collect()
}

fn mean_hidden_delta(previous: &[Vec<f32>], current: &[Vec<f32>]) -> Result<f32> {
    if previous.len() != current.len() {
        return Err(CognexisError::ShapeMismatch {
            expected: format!("{} previous rows", previous.len()),
            actual: format!("{} current rows", current.len()),
        });
    }
    if previous.is_empty() {
        return Ok(0.0);
    }

    let mut total = 0.0f32;
    for (row_index, (previous_row, current_row)) in previous.iter().zip(current).enumerate() {
        if previous_row.len() != current_row.len() {
            return Err(CognexisError::ShapeMismatch {
                expected: format!("row {row_index} width {}", previous_row.len()),
                actual: format!("row {row_index} width {}", current_row.len()),
            });
        }
        let row_norm = previous_row
            .iter()
            .zip(current_row)
            .map(|(left, right)| {
                let delta = right - left;
                delta * delta
            })
            .sum::<f32>()
            .sqrt();
        if !row_norm.is_finite() {
            return Err(CognexisError::Backend(
                "hidden delta contains non-finite values".to_string(),
            ));
        }
        total += row_norm;
    }
    Ok(total / previous.len() as f32)
}

fn confidence_by_token(logits: &[Vec<f32>]) -> Result<(Vec<f32>, Option<f32>)> {
    let mut confidences = Vec::with_capacity(logits.len());
    let mut entropies = Vec::with_capacity(logits.len());
    for row in logits {
        let (confidence, entropy) = confidence_and_entropy(row)?;
        confidences.push(confidence.unwrap_or(0.0));
        if let Some(entropy) = entropy {
            entropies.push(entropy);
        }
    }
    Ok((
        confidences,
        mean_present_f32(entropies.into_iter().map(Some)),
    ))
}

fn confidence_and_entropy(logits: &[f32]) -> Result<(Option<f32>, Option<f32>)> {
    if logits.is_empty() {
        return Ok((None, None));
    }
    if logits.iter().any(|logit| !logit.is_finite()) {
        return Err(CognexisError::Backend(
            "scheduler logits must be finite".to_string(),
        ));
    }

    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_scores: Vec<f32> = logits
        .iter()
        .map(|logit| (*logit - max_logit).exp())
        .collect();
    let total = exp_scores.iter().sum::<f32>();
    if total <= 0.0 || !total.is_finite() {
        return Ok((None, None));
    }

    let mut confidence = 0.0f32;
    let mut entropy = 0.0f32;
    for exp_score in exp_scores {
        let probability = exp_score / total;
        confidence = confidence.max(probability);
        if probability > 0.0 {
            entropy -= probability * probability.ln();
        }
    }
    Ok((Some(confidence), Some(entropy)))
}

fn tokenwise_halt_reasons(
    state: &TokenLoopState,
    predicted_gain: &[f32],
    confidence: &[f32],
    hidden_delta: &[f32],
    min_delta: f32,
    confidence_threshold: f32,
    gain_threshold: f32,
    min_loops: usize,
) -> Result<Vec<Option<HaltReason>>> {
    let len = state.active.len();
    if predicted_gain.len() != len || confidence.len() != len || hidden_delta.len() != len {
        return Err(CognexisError::ShapeMismatch {
            expected: format!("{len} token-wise scheduler signals"),
            actual: format!(
                "gain {}, confidence {}, delta {}",
                predicted_gain.len(),
                confidence.len(),
                hidden_delta.len()
            ),
        });
    }

    let mut decisions = vec![None; len];
    for index in 0..len {
        if !state.active[index] || state.loops[index] < min_loops {
            continue;
        }
        decisions[index] = if hidden_delta[index] <= min_delta {
            Some(HaltReason::MinDelta)
        } else if predicted_gain[index] <= gain_threshold {
            Some(HaltReason::ValueGain)
        } else if confidence[index] >= confidence_threshold {
            Some(HaltReason::Confidence)
        } else {
            None
        };
    }
    Ok(decisions)
}

fn mean_present_f32(values: impl Iterator<Item = Option<f32>>) -> Option<f32> {
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for value in values.flatten() {
        if value.is_finite() {
            sum += value;
            count += 1;
        }
    }
    (count > 0).then_some(sum / count as f32)
}

fn select_token(
    logits: &[f32],
    options: &SamplingOptions,
    previous_tokens: &[TokenId],
) -> Result<TokenId> {
    if logits.is_empty() {
        return Err(CognexisError::ShapeMismatch {
            expected: "non-empty logits".to_string(),
            actual: "empty logits".to_string(),
        });
    }
    if logits.iter().any(|logit| !logit.is_finite()) {
        return Err(CognexisError::Backend(
            "generation logits must be finite".to_string(),
        ));
    }

    let mut scores = logits.to_vec();
    if options.repetition_penalty != 1.0 {
        for &token_id in previous_tokens {
            if let Some(score) = scores.get_mut(token_id as usize) {
                if *score >= 0.0 {
                    *score /= options.repetition_penalty;
                } else {
                    *score *= options.repetition_penalty;
                }
            }
        }
    }

    if options.temperature > 0.0 && options.temperature != 1.0 {
        for score in &mut scores {
            *score /= options.temperature;
        }
    }

    let mut ranked: Vec<_> = scores
        .iter()
        .copied()
        .enumerate()
        .map(|(token_id, score)| (token_id as TokenId, score))
        .collect();
    ranked.sort_by(|a, b| compare_scores_desc(a.1, b.1).then_with(|| a.0.cmp(&b.0)));

    if options.top_k > 0 && options.top_k < ranked.len() {
        ranked.truncate(options.top_k);
    }
    if options.top_p < 1.0 {
        ranked = filter_top_p(ranked, options.top_p);
    }

    ranked
        .first()
        .map(|(token_id, _)| *token_id)
        .ok_or_else(|| CognexisError::Backend("sampling filters removed every token".to_string()))
}

fn filter_top_p(ranked: Vec<(TokenId, f32)>, top_p: f32) -> Vec<(TokenId, f32)> {
    if ranked.len() <= 1 {
        return ranked;
    }

    let max_score = ranked
        .iter()
        .map(|(_, score)| *score)
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_scores: Vec<f32> = ranked
        .iter()
        .map(|(_, score)| (*score - max_score).exp())
        .collect();
    let total = exp_scores.iter().sum::<f32>();
    if total == 0.0 || !total.is_finite() {
        return ranked.into_iter().take(1).collect();
    }

    let mut cumulative = 0.0;
    let mut kept = Vec::new();
    for ((token_id, score), exp_score) in ranked.into_iter().zip(exp_scores) {
        cumulative += exp_score / total;
        kept.push((token_id, score));
        if cumulative >= top_p {
            break;
        }
    }
    kept
}

fn compare_scores_desc(left: f32, right: f32) -> Ordering {
    right.partial_cmp(&left).unwrap_or(Ordering::Equal)
}
