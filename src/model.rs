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

use crate::coda::Coda;
use crate::config::{ModelConfig, ServeConfig};
use crate::embedding::Embedding;
use crate::lm_head::LMHead;
use crate::prelude::Prelude;
use crate::recurrent_core::RecurrentCore;
use crate::safety::{check_safety, SafetyIssue};
use crate::scheduler::{
    compute_loops_bounded, HaltReason, LoopScheduling, RuleBasedScheduler, SchedulerObservation,
};
use crate::tokenizer::{DecodeOptions as TokenDecodeOptions, TokenId, Tokenizer};
use crate::{CognexisError, Result};

/// Request-time recurrent loop mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopMode {
    Fixed(usize),
    Adaptive { min_loops: usize, max_loops: usize },
    TokenWise,
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
    pub halt_reason: Option<HaltReason>,
    pub safety_issue: Option<SafetyIssue>,
    pub stop_reason: Option<StopReason>,
}

/// Deterministic CPU reference model.
pub struct CognexisModel {
    pub config: ModelConfig,
    pub embeddings: Embedding,
    pub prelude: Prelude,
    pub recurrent: RecurrentCore,
    pub coda: Coda,
    pub lm_head: LMHead,
    tokenizer: Tokenizer,
    scheduler: RuleBasedScheduler,
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
            tokenizer: Tokenizer::new(),
            scheduler: RuleBasedScheduler::default(),
            config,
        })
    }

    /// Build a model from a checkpoint directory and serving config.
    ///
    /// The reference implementation accepts `config.resolved.json` when
    /// present and otherwise falls back to the supplied serve config.
    pub fn from_checkpoint(
        checkpoint_dir: impl AsRef<Path>,
        serve_config: &ServeConfig,
    ) -> Result<Self> {
        serve_config.validate()?;
        let resolved_config_path = checkpoint_dir.as_ref().join("config.resolved.json");
        let config = if resolved_config_path.exists() {
            let contents = fs::read_to_string(&resolved_config_path).map_err(|error| {
                CognexisError::Backend(format!(
                    "failed to read {}: {error}",
                    resolved_config_path.display()
                ))
            })?;
            serde_json::from_str::<ModelConfig>(&contents).map_err(|error| {
                CognexisError::InvalidConfig(format!("invalid resolved model config: {error}"))
            })?
        } else {
            serve_config.model.clone()
        };

        Self::new(config)
    }

    /// Compute logits for a token sequence at a fixed recurrent depth.
    pub fn forward_logits(&self, token_ids: &[TokenId], loops: usize) -> Result<Vec<Vec<f32>>> {
        let embedded = self.embeddings.try_forward(token_ids)?;
        let prepared = self.prelude.forward(&embedded);
        let refined = self.recurrent.forward(&prepared, loops);
        let hidden = self.coda.forward(&refined);
        self.lm_head.try_forward(&hidden)
    }

    /// Generate a deterministic sequence and return streaming-style events.
    pub fn generate_streaming(
        &self,
        request: GenerationRequest,
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
            let scheduled =
                self.schedule_loops(&request.loop_options, context.len(), remaining_budget);
            if scheduled.loop_count == 0 {
                mark_or_push_budget_stop(&mut events);
                break;
            }
            if let Some(budget) = &mut remaining_budget {
                if *budget < scheduled.loop_count {
                    mark_or_push_budget_stop(&mut events);
                    break;
                }
                *budget -= scheduled.loop_count;
            }

            let logits = self.forward_logits(&context, scheduled.loop_count)?;
            let next_logits = logits.last().ok_or_else(|| {
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
                loop_count: scheduled.loop_count,
                halt_reason: scheduled.halt_reason,
                safety_issue,
                stop_reason,
            });

            if stop_reason.is_some() {
                return Ok(events);
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

fn mark_or_push_budget_stop(events: &mut Vec<GenerationStepOutput>) {
    if let Some(last) = events.last_mut() {
        last.stop_reason = Some(StopReason::BudgetExhausted);
        return;
    }

    events.push(GenerationStepOutput {
        token_id: 0,
        text_delta: String::new(),
        loop_count: 0,
        halt_reason: Some(HaltReason::Budget),
        safety_issue: None,
        stop_reason: Some(StopReason::BudgetExhausted),
    });
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
