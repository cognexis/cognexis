//! Configuration definitions for the Cognexis model.
//!
//! This module defines the configuration structures used to parameterize
//! the Cognexis model. These structures mirror the fields described in
//! `spec11_config.md`. They can be serialized and deserialized via
//! `serde` to allow loading from JSON or other formats.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

use crate::tokenizer::TokenizerManifest;
use crate::{CognexisError, Result};

/// Global configuration for the Cognexis model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelConfig {
    /// Size of the vocabulary (number of tokens).
    pub vocab_size: usize,
    /// Dimensionality of token embeddings.
    pub hidden_size: usize,
    /// Number of transformer layers in the prelude.
    pub num_prelude_layers: usize,
    /// Number of unique recurrent blocks to iterate.
    pub num_recurrent_blocks: usize,
    /// Minimum number of recurrent iterations during inference.
    #[serde(default = "default_min_loop_count")]
    pub min_loop_count: usize,
    /// Number of iterations (loops) during inference.
    pub max_loop_count: usize,
    /// Number of transformer layers in the coda.
    pub num_coda_layers: usize,
    /// Number of attention heads per block.
    pub num_attention_heads: usize,
    /// Number of key/value heads for grouped query attention.
    #[serde(default = "default_num_kv_heads")]
    pub num_kv_heads: usize,
    /// Whether rotary position encoding is applied in attention.
    #[serde(default = "default_rope_enabled")]
    pub rope_enabled: bool,
    /// RoPE base frequency.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// Maximum supported RoPE position without extrapolation.
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    /// Dimension of each feed‑forward sublayer.
    pub ff_inner_dim: usize,
    /// Feed-forward activation mode recorded for checkpoint compatibility.
    #[serde(default)]
    pub ff_activation: FeedForwardActivation,
    /// Epsilon used by RMS normalization.
    #[serde(default = "default_norm_epsilon")]
    pub norm_epsilon: f32,
    /// Residual scale used for recurrent transformer updates.
    #[serde(default = "default_recurrent_residual_scale")]
    pub recurrent_residual_scale: f32,
    /// Whether the recurrent wrapper blends candidate updates through a gate.
    #[serde(default = "default_recurrent_gating")]
    pub recurrent_gating: bool,
    /// Input-injection policy anchoring recurrent updates to Prelude output.
    #[serde(default)]
    pub recurrent_input_injection: RecurrentInputInjection,
    /// Scale for recurrent input injection.
    #[serde(default = "default_recurrent_input_injection_scale")]
    pub recurrent_input_injection_scale: f32,
    /// Whether the LM head should be tied to embeddings in checkpointed models.
    #[serde(default = "default_tie_embeddings")]
    pub tie_embeddings: bool,
    /// Explicit embedding scale applied after lookup.
    #[serde(default = "default_embedding_scale")]
    pub embedding_scale: f32,
    /// Optional path to a tokenizer vocabulary file.
    pub tokenizer_path: Option<String>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50_000,
            hidden_size: 2_048,
            num_prelude_layers: 6,
            num_recurrent_blocks: 1,
            min_loop_count: default_min_loop_count(),
            max_loop_count: 16,
            num_coda_layers: 6,
            num_attention_heads: 16,
            num_kv_heads: 16,
            rope_enabled: default_rope_enabled(),
            rope_theta: default_rope_theta(),
            max_position_embeddings: default_max_position_embeddings(),
            ff_inner_dim: 8_192,
            ff_activation: FeedForwardActivation::default(),
            norm_epsilon: default_norm_epsilon(),
            recurrent_residual_scale: default_recurrent_residual_scale(),
            recurrent_gating: default_recurrent_gating(),
            recurrent_input_injection: RecurrentInputInjection::default(),
            recurrent_input_injection_scale: default_recurrent_input_injection_scale(),
            tie_embeddings: default_tie_embeddings(),
            embedding_scale: default_embedding_scale(),
            tokenizer_path: None,
        }
    }
}

impl ModelConfig {
    /// Validate architecture invariants required by the specification.
    pub fn validate(&self) -> Result<()> {
        if self.vocab_size == 0 {
            return Err(CognexisError::InvalidConfig(
                "vocab_size must be positive".to_string(),
            ));
        }
        if self.hidden_size == 0 {
            return Err(CognexisError::InvalidConfig(
                "hidden_size must be positive".to_string(),
            ));
        }
        if self.num_attention_heads == 0 {
            return Err(CognexisError::InvalidConfig(
                "num_attention_heads must be positive".to_string(),
            ));
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(CognexisError::InvalidConfig(format!(
                "hidden_size ({}) must be divisible by num_attention_heads ({})",
                self.hidden_size, self.num_attention_heads
            )));
        }
        if self.num_kv_heads == 0 {
            return Err(CognexisError::InvalidConfig(
                "num_kv_heads must be positive".to_string(),
            ));
        }
        if self.num_attention_heads % self.num_kv_heads != 0 {
            return Err(CognexisError::InvalidConfig(format!(
                "num_attention_heads ({}) must be divisible by num_kv_heads ({})",
                self.num_attention_heads, self.num_kv_heads
            )));
        }
        if self.max_position_embeddings == 0 {
            return Err(CognexisError::InvalidConfig(
                "max_position_embeddings must be positive".to_string(),
            ));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 1.0 {
            return Err(CognexisError::InvalidConfig(
                "rope_theta must be finite and greater than 1".to_string(),
            ));
        }
        if self.num_recurrent_blocks != 1 {
            return Err(CognexisError::InvalidConfig(
                "baseline Cognexis requires num_recurrent_blocks == 1".to_string(),
            ));
        }
        if self.min_loop_count == 0 {
            return Err(CognexisError::InvalidConfig(
                "min_loop_count must be at least 1".to_string(),
            ));
        }
        if self.max_loop_count < self.min_loop_count {
            return Err(CognexisError::InvalidConfig(format!(
                "max_loop_count ({}) must be >= min_loop_count ({})",
                self.max_loop_count, self.min_loop_count
            )));
        }
        if self.ff_inner_dim == 0 {
            return Err(CognexisError::InvalidConfig(
                "ff_inner_dim must be positive".to_string(),
            ));
        }
        if !self.norm_epsilon.is_finite() || self.norm_epsilon <= 0.0 {
            return Err(CognexisError::InvalidConfig(
                "norm_epsilon must be finite and positive".to_string(),
            ));
        }
        if !self.recurrent_residual_scale.is_finite() || self.recurrent_residual_scale < 0.0 {
            return Err(CognexisError::InvalidConfig(
                "recurrent_residual_scale must be finite and non-negative".to_string(),
            ));
        }
        if !self.recurrent_input_injection_scale.is_finite()
            || self.recurrent_input_injection_scale < 0.0
        {
            return Err(CognexisError::InvalidConfig(
                "recurrent_input_injection_scale must be finite and non-negative".to_string(),
            ));
        }
        if !self.embedding_scale.is_finite() || self.embedding_scale <= 0.0 {
            return Err(CognexisError::InvalidConfig(
                "embedding_scale must be finite and positive".to_string(),
            ));
        }
        Ok(())
    }

    /// Attention head dimension derived from the hidden size and query head count.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Effective sequential transformer-block applications for a loop count.
    pub fn effective_depth(&self, loops: usize) -> usize {
        self.num_prelude_layers + loops * self.num_recurrent_blocks + self.num_coda_layers
    }

    /// Effective depth using the configured maximum loop count.
    pub fn max_effective_depth(&self) -> usize {
        self.effective_depth(self.max_loop_count)
    }

    /// Build one of the named reference Cognexis configurations.
    pub fn for_variant(variant: CognexisVariant) -> Self {
        let spec = variant.spec();
        Self {
            hidden_size: spec.hidden_size,
            num_attention_heads: spec.num_attention_heads,
            num_kv_heads: spec.num_attention_heads / 4,
            max_position_embeddings: 8_192,
            num_prelude_layers: spec.num_prelude_layers,
            num_recurrent_blocks: spec.num_recurrent_blocks,
            min_loop_count: 1,
            max_loop_count: spec.max_loop_count,
            num_coda_layers: spec.num_coda_layers,
            ff_inner_dim: spec.hidden_size * 4,
            recurrent_residual_scale: 1.0 / (spec.max_loop_count as f32).sqrt(),
            ..Self::default()
        }
    }

    /// Build a reference model config by variant name.
    pub fn from_variant_name(name: &str) -> Result<Self> {
        Ok(Self::for_variant(CognexisVariant::from_name(name)?))
    }
}

/// Named reference model variants from the Cognexis architecture table.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CognexisVariant {
    Cognexis8B,
    Cognexis64B,
    Cognexis256B,
    Cognexis1_28T,
}

impl CognexisVariant {
    pub fn from_name(name: &str) -> Result<Self> {
        let normalized = name.trim().to_ascii_lowercase().replace(['_', ' '], "-");
        match normalized.as_str() {
            "cognexis-8b" | "8b" => Ok(Self::Cognexis8B),
            "cognexis-64b" | "64b" => Ok(Self::Cognexis64B),
            "cognexis-256b" | "256b" => Ok(Self::Cognexis256B),
            "cognexis-1.28t" | "cognexis-1280b" | "1.28t" | "1280b" => Ok(Self::Cognexis1_28T),
            _ => Err(CognexisError::InvalidConfig(format!(
                "unknown Cognexis variant {name:?}"
            ))),
        }
    }

    pub fn spec(self) -> CognexisVariantSpec {
        match self {
            Self::Cognexis8B => CognexisVariantSpec {
                name: "Cognexis-8B",
                hidden_size: 4_096,
                num_attention_heads: 32,
                num_prelude_layers: 8,
                num_recurrent_blocks: 1,
                num_coda_layers: 8,
                max_loop_count: 12,
                parameter_label: "~8B",
            },
            Self::Cognexis64B => CognexisVariantSpec {
                name: "Cognexis-64B",
                hidden_size: 8_192,
                num_attention_heads: 64,
                num_prelude_layers: 10,
                num_recurrent_blocks: 1,
                num_coda_layers: 10,
                max_loop_count: 16,
                parameter_label: "~64B",
            },
            Self::Cognexis256B => CognexisVariantSpec {
                name: "Cognexis-256B",
                hidden_size: 12_288,
                num_attention_heads: 96,
                num_prelude_layers: 12,
                num_recurrent_blocks: 1,
                num_coda_layers: 12,
                max_loop_count: 20,
                parameter_label: "~256B",
            },
            Self::Cognexis1_28T => CognexisVariantSpec {
                name: "Cognexis-1.28T",
                hidden_size: 16_384,
                num_attention_heads: 128,
                num_prelude_layers: 16,
                num_recurrent_blocks: 1,
                num_coda_layers: 16,
                max_loop_count: 24,
                parameter_label: "~1.28T",
            },
        }
    }
}

/// Immutable model-variant table row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CognexisVariantSpec {
    pub name: &'static str,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_prelude_layers: usize,
    pub num_recurrent_blocks: usize,
    pub num_coda_layers: usize,
    pub max_loop_count: usize,
    pub parameter_label: &'static str,
}

/// Recurrent input injection mode.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RecurrentInputInjection {
    None,
    Residual,
    GateCondition,
}

impl Default for RecurrentInputInjection {
    fn default() -> Self {
        Self::Residual
    }
}

/// Feed-forward activation family.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FeedForwardActivation {
    SwiGlu,
    GeGlu,
    Gelu,
    Relu,
}

impl Default for FeedForwardActivation {
    fn default() -> Self {
        Self::SwiGlu
    }
}

/// Serving configuration used when constructing a model from a checkpoint.
/// Top-level resolved Cognexis configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CognexisConfig {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub run: RunConfig,
    #[serde(default = "TokenizerManifest::reference")]
    pub tokenizer: TokenizerManifest,
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default)]
    pub training: Option<TrainingConfig>,
    #[serde(default)]
    pub inference: Option<InferenceConfig>,
    #[serde(default)]
    pub evaluation: Option<EvaluationConfig>,
    #[serde(default)]
    pub safety: Option<SafetyConfig>,
    #[serde(default)]
    pub logging: Option<LoggingConfig>,
}

impl Default for CognexisConfig {
    fn default() -> Self {
        Self {
            schema_version: default_schema_version(),
            run: RunConfig::default(),
            tokenizer: TokenizerManifest::reference(),
            model: ModelConfig::default(),
            training: None,
            inference: Some(InferenceConfig::default()),
            evaluation: None,
            safety: Some(SafetyConfig::default()),
            logging: Some(LoggingConfig::default()),
        }
    }
}

impl CognexisConfig {
    /// Build a top-level resolved config for a named reference variant.
    pub fn for_variant(variant: CognexisVariant) -> Self {
        let tokenizer = TokenizerManifest::reference();
        let mut model = ModelConfig::for_variant(variant);
        model.vocab_size = tokenizer.vocab_size;
        Self {
            tokenizer,
            model,
            inference: Some(InferenceConfig {
                max_loops: variant.spec().max_loop_count,
                ..InferenceConfig::default()
            }),
            safety: Some(SafetyConfig {
                max_user_loops: variant.spec().max_loop_count,
                ..SafetyConfig::default()
            }),
            ..Self::default()
        }
    }

    /// Load a top-level config from JSON.
    pub fn load_json(path: impl AsRef<Path>) -> Result<Self> {
        let contents = fs::read_to_string(path.as_ref()).map_err(|error| {
            CognexisError::Backend(format!(
                "failed to read config {}: {error}",
                path.as_ref().display()
            ))
        })?;
        let config = serde_json::from_str::<Self>(&contents).map_err(|error| {
            CognexisError::InvalidConfig(format!("invalid config JSON: {error}"))
        })?;
        config.validate()?;
        Ok(config)
    }

    /// Serialize the fully resolved config as pretty JSON.
    pub fn resolved_json(&self) -> Result<String> {
        self.validate()?;
        serde_json::to_string_pretty(self).map_err(|error| {
            CognexisError::Backend(format!("config serialization failed: {error}"))
        })
    }

    /// Validate top-level and cross-section invariants.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != default_schema_version() {
            return Err(CognexisError::InvalidConfig(format!(
                "unsupported schema_version {}",
                self.schema_version
            )));
        }
        self.run.validate()?;
        self.tokenizer
            .validate()
            .map_err(|error| CognexisError::InvalidConfig(error.to_string()))?;
        self.model.validate()?;
        if self.tokenizer.vocab_size != self.model.vocab_size {
            return Err(CognexisError::InvalidConfig(format!(
                "tokenizer vocab_size ({}) must match model vocab_size ({})",
                self.tokenizer.vocab_size, self.model.vocab_size
            )));
        }
        if let Some(training) = &self.training {
            training.validate()?;
        }
        if let Some(inference) = &self.inference {
            inference.validate(&self.model)?;
        }
        if let Some(evaluation) = &self.evaluation {
            evaluation.validate()?;
        }
        if let Some(safety) = &self.safety {
            safety.validate(&self.model)?;
        }
        Ok(())
    }
}

/// Run metadata recorded in resolved configs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunConfig {
    #[serde(default = "default_run_name")]
    pub name: String,
    #[serde(default)]
    pub seed: u64,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            name: default_run_name(),
            seed: 0,
        }
    }
}

impl RunConfig {
    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            return Err(CognexisError::InvalidConfig(
                "run.name must not be empty".to_string(),
            ));
        }
        Ok(())
    }
}

/// Data-related training config.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TrainingConfig {
    pub train_manifest: Option<String>,
    pub validation_manifest: Option<String>,
    #[serde(default = "default_sequence_length")]
    pub sequence_length: usize,
    #[serde(default = "default_true")]
    pub packing: bool,
    #[serde(default)]
    pub document_boundary_attention: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            train_manifest: None,
            validation_manifest: None,
            sequence_length: default_sequence_length(),
            packing: true,
            document_boundary_attention: false,
        }
    }
}

impl TrainingConfig {
    pub fn validate(&self) -> Result<()> {
        if self.sequence_length < 2 {
            return Err(CognexisError::InvalidConfig(
                "training.sequence_length must be at least 2".to_string(),
            ));
        }
        Ok(())
    }
}

/// Inference defaults separate from architecture fields.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InferenceConfig {
    #[serde(default = "default_max_sequence_length")]
    pub max_sequence_length: usize,
    #[serde(default = "default_max_new_tokens")]
    pub max_new_tokens: usize,
    #[serde(default = "default_loop_mode")]
    pub loop_mode: String,
    #[serde(default = "default_min_loop_count")]
    pub min_loops: usize,
    #[serde(default = "default_inference_max_loops")]
    pub max_loops: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_sequence_length: default_max_sequence_length(),
            max_new_tokens: default_max_new_tokens(),
            loop_mode: default_loop_mode(),
            min_loops: default_min_loop_count(),
            max_loops: default_inference_max_loops(),
        }
    }
}

impl InferenceConfig {
    pub fn validate(&self, model: &ModelConfig) -> Result<()> {
        if self.max_sequence_length == 0 || self.max_new_tokens == 0 {
            return Err(CognexisError::InvalidConfig(
                "inference token limits must be positive".to_string(),
            ));
        }
        if self.min_loops == 0 || self.max_loops < self.min_loops {
            return Err(CognexisError::InvalidConfig(
                "inference loop bounds must satisfy max_loops >= min_loops >= 1".to_string(),
            ));
        }
        if self.max_loops > model.max_loop_count {
            return Err(CognexisError::InvalidConfig(format!(
                "inference.max_loops ({}) exceeds model.max_loop_count ({})",
                self.max_loops, model.max_loop_count
            )));
        }
        let loop_mode = normalize_loop_mode(&self.loop_mode);
        if !matches!(
            loop_mode.as_str(),
            "fixed"
                | "adaptive"
                | "adaptive_sequence"
                | "rule_based"
                | "adaptive_value"
                | "value_head"
                | "hybrid"
                | "adaptive_token"
                | "tokenwise"
                | "token_wise"
        ) {
            return Err(CognexisError::InvalidConfig(format!(
                "unsupported inference loop_mode {:?}",
                self.loop_mode
            )));
        }
        Ok(())
    }
}

/// Evaluation defaults and output destination.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvaluationConfig {
    #[serde(default)]
    pub loop_counts: Vec<usize>,
    pub output_path: Option<String>,
}

impl EvaluationConfig {
    pub fn validate(&self) -> Result<()> {
        if self.loop_counts.iter().any(|loops| *loops == 0) {
            return Err(CognexisError::InvalidConfig(
                "evaluation.loop_counts must contain only positive depths".to_string(),
            ));
        }
        Ok(())
    }
}

/// Safety-related serving config.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SafetyConfig {
    #[serde(default = "default_true")]
    pub input_filter: bool,
    #[serde(default = "default_true")]
    pub output_filter: bool,
    #[serde(default = "default_inference_max_loops")]
    pub max_user_loops: usize,
    #[serde(default = "default_true")]
    pub block_special_token_injection: bool,
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            input_filter: true,
            output_filter: true,
            max_user_loops: default_inference_max_loops(),
            block_special_token_injection: true,
        }
    }
}

impl SafetyConfig {
    pub fn validate(&self, model: &ModelConfig) -> Result<()> {
        if self.max_user_loops == 0 || self.max_user_loops > model.max_loop_count {
            return Err(CognexisError::InvalidConfig(format!(
                "safety.max_user_loops ({}) must be in 1..={}",
                self.max_user_loops, model.max_loop_count
            )));
        }
        Ok(())
    }
}

/// Logging config used by reference tooling.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LoggingConfig {
    #[serde(default = "default_log_level")]
    pub level: String,
    #[serde(default = "default_log_format")]
    pub format: String,
    #[serde(default = "default_true")]
    pub log_loop_stats: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            format: default_log_format(),
            log_loop_stats: true,
        }
    }
}

/// Serving configuration used when constructing a model from a checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ServeConfig {
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default = "default_max_sequence_length")]
    pub max_sequence_length: usize,
    #[serde(default = "default_max_new_tokens")]
    pub max_new_tokens: usize,
    #[serde(default)]
    pub max_user_loops: Option<usize>,
}

impl Default for ServeConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            max_sequence_length: default_max_sequence_length(),
            max_new_tokens: default_max_new_tokens(),
            max_user_loops: None,
        }
    }
}

impl ServeConfig {
    /// Load a serving config from JSON. If the file contains only a
    /// model config, it is accepted and wrapped in serving defaults.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let contents = fs::read_to_string(path.as_ref()).map_err(|error| {
            CognexisError::Backend(format!(
                "failed to read serve config {}: {error}",
                path.as_ref().display()
            ))
        })?;

        serde_json::from_str::<ServeConfig>(&contents)
            .or_else(|_| {
                serde_json::from_str::<ModelConfig>(&contents).map(|model| ServeConfig {
                    model,
                    ..ServeConfig::default()
                })
            })
            .map_err(|error| CognexisError::InvalidConfig(format!("invalid serve config: {error}")))
    }

    /// Validate serving and architecture limits together.
    pub fn validate(&self) -> Result<()> {
        self.model.validate()?;
        if self.max_sequence_length == 0 {
            return Err(CognexisError::InvalidConfig(
                "max_sequence_length must be positive".to_string(),
            ));
        }
        if let Some(max_user_loops) = self.max_user_loops {
            if max_user_loops == 0 {
                return Err(CognexisError::InvalidConfig(
                    "max_user_loops must be positive when set".to_string(),
                ));
            }
        }
        Ok(())
    }
}

const fn default_min_loop_count() -> usize {
    1
}

const fn default_schema_version() -> u32 {
    1
}

fn default_run_name() -> String {
    "cognexis".to_string()
}

const fn default_sequence_length() -> usize {
    2_048
}

const fn default_true() -> bool {
    true
}

fn default_loop_mode() -> String {
    "adaptive_sequence".to_string()
}

fn normalize_loop_mode(loop_mode: &str) -> String {
    loop_mode
        .trim()
        .to_ascii_lowercase()
        .replace(['-', ' '], "_")
}

const fn default_inference_max_loops() -> usize {
    8
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_log_format() -> String {
    "json".to_string()
}

const fn default_num_kv_heads() -> usize {
    1
}

const fn default_rope_enabled() -> bool {
    true
}

const fn default_max_position_embeddings() -> usize {
    8_192
}

const fn default_rope_theta() -> f32 {
    10_000.0
}

const fn default_norm_epsilon() -> f32 {
    1.0e-5
}

const fn default_recurrent_residual_scale() -> f32 {
    0.5
}

const fn default_recurrent_gating() -> bool {
    true
}

const fn default_recurrent_input_injection_scale() -> f32 {
    0.1
}

const fn default_tie_embeddings() -> bool {
    true
}

const fn default_embedding_scale() -> f32 {
    1.0
}

const fn default_max_sequence_length() -> usize {
    8_192
}

const fn default_max_new_tokens() -> usize {
    512
}
