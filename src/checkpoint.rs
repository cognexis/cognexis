//! Checkpoint metadata and manifest support.
//!
//! Tensor serialization is backend-specific, but checkpoint bundles
//! still need a stable manifest recording config, tokenizer, scheduler,
//! data, and build metadata. These helpers implement the JSON metadata
//! contract used by the CPU reference path.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::{CognexisConfig, InferenceSchedulerConfig};
use crate::scheduler::ActSchedulerConfig;
use crate::value_head::ValueHeadConfig;
use crate::{CognexisError, Result};

pub const CHECKPOINT_SCHEMA_VERSION: u32 = 1;
pub const SCHEDULER_STATE_SCHEMA_VERSION: u32 = 1;

/// Metadata persisted in `metadata.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CheckpointMetadata {
    pub schema_version: u32,
    pub checkpoint_step: u64,
    pub build_id: Option<String>,
    pub tensor_dtype: String,
    pub parameter_count: u64,
    pub training_tokens_seen: u64,
    pub data_manifest_checksum: Option<String>,
    pub tokenizer_checksum: Option<String>,
    pub backend: Option<String>,
}

impl CheckpointMetadata {
    pub fn new(checkpoint_step: u64, tensor_dtype: impl Into<String>) -> Self {
        Self {
            schema_version: CHECKPOINT_SCHEMA_VERSION,
            checkpoint_step,
            build_id: None,
            tensor_dtype: tensor_dtype.into(),
            parameter_count: 0,
            training_tokens_seen: 0,
            data_manifest_checksum: None,
            tokenizer_checksum: None,
            backend: None,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION {
            return Err(CognexisError::InvalidConfig(format!(
                "unsupported checkpoint schema_version {}",
                self.schema_version
            )));
        }
        if self.tensor_dtype.trim().is_empty() {
            return Err(CognexisError::InvalidConfig(
                "checkpoint tensor_dtype must not be empty".to_string(),
            ));
        }
        Ok(())
    }
}

/// One artifact listed in a checkpoint manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CheckpointArtifact {
    pub path: String,
    pub kind: CheckpointArtifactKind,
    pub required: bool,
}

/// Known artifact roles in a checkpoint bundle.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CheckpointArtifactKind {
    ModelWeights,
    OptimizerState,
    ResolvedConfig,
    TokenizerModel,
    TokenizerManifest,
    SchedulerState,
    CurriculumState,
    DataLoaderState,
    Metadata,
}

/// Machine-readable checkpoint manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CheckpointManifest {
    pub metadata: CheckpointMetadata,
    pub artifacts: Vec<CheckpointArtifact>,
}

impl CheckpointManifest {
    pub fn reference(metadata: CheckpointMetadata) -> Self {
        Self {
            metadata,
            artifacts: vec![
                artifact(
                    "model.safetensors",
                    CheckpointArtifactKind::ModelWeights,
                    false,
                ),
                artifact(
                    "config.resolved.json",
                    CheckpointArtifactKind::ResolvedConfig,
                    true,
                ),
                artifact(
                    "tokenizer.json",
                    CheckpointArtifactKind::TokenizerManifest,
                    false,
                ),
                artifact(
                    "scheduler.json",
                    CheckpointArtifactKind::SchedulerState,
                    false,
                ),
                artifact("metadata.json", CheckpointArtifactKind::Metadata, true),
            ],
        }
    }

    pub fn validate(&self, checkpoint_dir: impl AsRef<Path>) -> Result<()> {
        self.metadata.validate()?;
        let has_metadata = self
            .artifacts
            .iter()
            .any(|artifact| artifact.kind == CheckpointArtifactKind::Metadata);
        if !has_metadata {
            return Err(CognexisError::InvalidConfig(
                "checkpoint manifest must include metadata artifact".to_string(),
            ));
        }

        for artifact in &self.artifacts {
            if artifact.path.trim().is_empty() {
                return Err(CognexisError::InvalidConfig(
                    "checkpoint artifact path must not be empty".to_string(),
                ));
            }
            if artifact.required {
                let path = checkpoint_dir.as_ref().join(&artifact.path);
                if !path.exists() {
                    return Err(CognexisError::InvalidConfig(format!(
                        "required checkpoint artifact is missing: {}",
                        path.display()
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Loaded checkpoint bundle metadata and resolved configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointBundle {
    pub manifest: CheckpointManifest,
    pub metadata: CheckpointMetadata,
    pub resolved_config: CognexisConfig,
    pub scheduler_state: Option<CheckpointSchedulerState>,
}

/// Serializable scheduler/value-head state recorded as `scheduler.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CheckpointSchedulerState {
    pub schema_version: u32,
    pub scheduler: InferenceSchedulerConfig,
    pub act: ActSchedulerConfig,
    pub value_head: ValueHeadConfig,
}

impl CheckpointSchedulerState {
    pub fn from_config(config: &CognexisConfig) -> Result<Self> {
        let scheduler = config
            .inference
            .as_ref()
            .map(|inference| inference.scheduler.clone())
            .unwrap_or_default();
        Self::from_scheduler_config(scheduler)
    }

    pub fn from_scheduler_config(scheduler: InferenceSchedulerConfig) -> Result<Self> {
        scheduler.validate()?;
        let mut act = ActSchedulerConfig::default();
        act.gain_threshold = scheduler.predicted_gain_threshold;
        let mut value_head = ValueHeadConfig::default();
        value_head.gain_threshold = scheduler.predicted_gain_threshold;
        let state = Self {
            schema_version: SCHEDULER_STATE_SCHEMA_VERSION,
            scheduler,
            act,
            value_head,
        };
        state.validate()?;
        Ok(state)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != SCHEDULER_STATE_SCHEMA_VERSION {
            return Err(CognexisError::InvalidConfig(format!(
                "unsupported scheduler state schema_version {}",
                self.schema_version
            )));
        }
        self.scheduler.validate()?;
        self.act.validate()?;
        self.value_head.validate()?;
        Ok(())
    }
}

/// Save the validated resolved config as `config.resolved.json`.
pub fn save_resolved_config_atomic(
    checkpoint_dir: impl AsRef<Path>,
    config: &CognexisConfig,
) -> Result<PathBuf> {
    fs::create_dir_all(checkpoint_dir.as_ref()).map_err(|error| {
        CognexisError::Backend(format!(
            "failed to create checkpoint directory {}: {error}",
            checkpoint_dir.as_ref().display()
        ))
    })?;
    let final_path = checkpoint_dir.as_ref().join("config.resolved.json");
    let tmp_path = checkpoint_dir.as_ref().join("config.resolved.json.tmp");
    let encoded = config.resolved_json()?;
    fs::write(&tmp_path, encoded).map_err(|error| {
        CognexisError::Backend(format!("failed to write {}: {error}", tmp_path.display()))
    })?;
    fs::rename(&tmp_path, &final_path).map_err(|error| {
        CognexisError::Backend(format!(
            "failed to move {} to {}: {error}",
            tmp_path.display(),
            final_path.display()
        ))
    })?;
    Ok(final_path)
}

/// Load and validate the resolved checkpoint config.
pub fn load_resolved_config(checkpoint_dir: impl AsRef<Path>) -> Result<CognexisConfig> {
    CognexisConfig::load_json(checkpoint_dir.as_ref().join("config.resolved.json"))
}

/// Save metadata, resolved config, and manifest for a reference checkpoint bundle.
pub fn save_checkpoint_bundle_atomic(
    checkpoint_dir: impl AsRef<Path>,
    metadata: &CheckpointMetadata,
    config: &CognexisConfig,
) -> Result<CheckpointManifest> {
    save_resolved_config_atomic(checkpoint_dir.as_ref(), config)?;
    let scheduler_state = CheckpointSchedulerState::from_config(config)?;
    save_scheduler_state_atomic(checkpoint_dir.as_ref(), &scheduler_state)?;
    save_metadata_atomic(checkpoint_dir.as_ref(), metadata)?;
    let manifest = CheckpointManifest::reference(metadata.clone());
    save_manifest_atomic(checkpoint_dir.as_ref(), &manifest)?;
    Ok(manifest)
}

/// Load a checkpoint bundle and validate cross-artifact compatibility.
pub fn load_checkpoint_bundle(checkpoint_dir: impl AsRef<Path>) -> Result<CheckpointBundle> {
    let manifest = load_manifest(checkpoint_dir.as_ref())?;
    let metadata = load_metadata(checkpoint_dir.as_ref())?;
    if manifest.metadata != metadata {
        return Err(CognexisError::InvalidConfig(
            "checkpoint manifest metadata does not match metadata.json".to_string(),
        ));
    }

    let resolved_config = load_resolved_config(checkpoint_dir.as_ref())?;
    validate_checkpoint_config_compatibility(&metadata, &resolved_config)?;
    let scheduler_state =
        load_optional_scheduler_state_for_manifest(checkpoint_dir.as_ref(), &manifest)?;
    if let Some(scheduler_state) = &scheduler_state {
        validate_checkpoint_scheduler_compatibility(scheduler_state, &resolved_config)?;
    }

    Ok(CheckpointBundle {
        manifest,
        metadata,
        resolved_config,
        scheduler_state,
    })
}

/// Validate metadata fields that must agree with the resolved config.
pub fn validate_checkpoint_config_compatibility(
    metadata: &CheckpointMetadata,
    config: &CognexisConfig,
) -> Result<()> {
    metadata.validate()?;
    config.validate()?;
    if let (Some(expected), Some(actual)) = (
        metadata.tokenizer_checksum.as_deref(),
        config.tokenizer.checksum.as_deref(),
    ) {
        if expected != actual {
            return Err(CognexisError::InvalidConfig(format!(
                "checkpoint tokenizer checksum mismatch: metadata has {expected}, config has {actual}"
            )));
        }
    }
    Ok(())
}

/// Validate scheduler state fields that must agree with the resolved config.
pub fn validate_checkpoint_scheduler_compatibility(
    scheduler_state: &CheckpointSchedulerState,
    config: &CognexisConfig,
) -> Result<()> {
    scheduler_state.validate()?;
    config.validate()?;
    if let Some(inference) = &config.inference {
        if scheduler_state.scheduler != inference.scheduler {
            return Err(CognexisError::InvalidConfig(
                "checkpoint scheduler state does not match resolved inference scheduler config"
                    .to_string(),
            ));
        }
    }
    Ok(())
}

/// Save scheduler/value-head state atomically as `scheduler.json`.
pub fn save_scheduler_state_atomic(
    checkpoint_dir: impl AsRef<Path>,
    scheduler_state: &CheckpointSchedulerState,
) -> Result<PathBuf> {
    scheduler_state.validate()?;
    fs::create_dir_all(checkpoint_dir.as_ref()).map_err(|error| {
        CognexisError::Backend(format!(
            "failed to create checkpoint directory {}: {error}",
            checkpoint_dir.as_ref().display()
        ))
    })?;

    let final_path = checkpoint_dir.as_ref().join("scheduler.json");
    let tmp_path = checkpoint_dir.as_ref().join("scheduler.json.tmp");
    let encoded = serde_json::to_vec_pretty(scheduler_state).map_err(|error| {
        CognexisError::Backend(format!("scheduler state serialization failed: {error}"))
    })?;
    fs::write(&tmp_path, encoded).map_err(|error| {
        CognexisError::Backend(format!("failed to write {}: {error}", tmp_path.display()))
    })?;
    fs::rename(&tmp_path, &final_path).map_err(|error| {
        CognexisError::Backend(format!(
            "failed to move {} to {}: {error}",
            tmp_path.display(),
            final_path.display()
        ))
    })?;
    Ok(final_path)
}

/// Load `scheduler.json` from a checkpoint bundle.
pub fn load_scheduler_state(checkpoint_dir: impl AsRef<Path>) -> Result<CheckpointSchedulerState> {
    load_scheduler_state_from_path(&checkpoint_dir.as_ref().join("scheduler.json"))
}

/// Load `scheduler.json` when present; absence is accepted for legacy bundles.
pub fn load_optional_scheduler_state(
    checkpoint_dir: impl AsRef<Path>,
) -> Result<Option<CheckpointSchedulerState>> {
    let path = checkpoint_dir.as_ref().join("scheduler.json");
    if path.exists() {
        load_scheduler_state_from_path(&path).map(Some)
    } else {
        Ok(None)
    }
}

/// Save metadata atomically as `metadata.json`.
pub fn save_metadata_atomic(
    checkpoint_dir: impl AsRef<Path>,
    metadata: &CheckpointMetadata,
) -> Result<PathBuf> {
    metadata.validate()?;
    fs::create_dir_all(checkpoint_dir.as_ref()).map_err(|error| {
        CognexisError::Backend(format!(
            "failed to create checkpoint directory {}: {error}",
            checkpoint_dir.as_ref().display()
        ))
    })?;

    let final_path = checkpoint_dir.as_ref().join("metadata.json");
    let tmp_path = checkpoint_dir.as_ref().join("metadata.json.tmp");
    let encoded = serde_json::to_vec_pretty(metadata).map_err(|error| {
        CognexisError::Backend(format!("metadata serialization failed: {error}"))
    })?;
    fs::write(&tmp_path, encoded).map_err(|error| {
        CognexisError::Backend(format!("failed to write {}: {error}", tmp_path.display()))
    })?;
    fs::rename(&tmp_path, &final_path).map_err(|error| {
        CognexisError::Backend(format!(
            "failed to move {} to {}: {error}",
            tmp_path.display(),
            final_path.display()
        ))
    })?;
    Ok(final_path)
}

/// Load and validate `metadata.json` from a checkpoint bundle.
pub fn load_metadata(checkpoint_dir: impl AsRef<Path>) -> Result<CheckpointMetadata> {
    let path = checkpoint_dir.as_ref().join("metadata.json");
    let contents = fs::read_to_string(&path).map_err(|error| {
        CognexisError::Backend(format!("failed to read {}: {error}", path.display()))
    })?;
    let metadata = serde_json::from_str::<CheckpointMetadata>(&contents)
        .map_err(|error| CognexisError::InvalidConfig(format!("invalid metadata.json: {error}")))?;
    metadata.validate()?;
    Ok(metadata)
}

/// Save a manifest as `manifest.json`.
pub fn save_manifest_atomic(
    checkpoint_dir: impl AsRef<Path>,
    manifest: &CheckpointManifest,
) -> Result<PathBuf> {
    fs::create_dir_all(checkpoint_dir.as_ref()).map_err(|error| {
        CognexisError::Backend(format!(
            "failed to create checkpoint directory {}: {error}",
            checkpoint_dir.as_ref().display()
        ))
    })?;
    manifest.metadata.validate()?;
    let final_path = checkpoint_dir.as_ref().join("manifest.json");
    let tmp_path = checkpoint_dir.as_ref().join("manifest.json.tmp");
    let encoded = serde_json::to_vec_pretty(manifest).map_err(|error| {
        CognexisError::Backend(format!("manifest serialization failed: {error}"))
    })?;
    fs::write(&tmp_path, encoded).map_err(|error| {
        CognexisError::Backend(format!("failed to write {}: {error}", tmp_path.display()))
    })?;
    fs::rename(&tmp_path, &final_path).map_err(|error| {
        CognexisError::Backend(format!(
            "failed to move {} to {}: {error}",
            tmp_path.display(),
            final_path.display()
        ))
    })?;
    Ok(final_path)
}

/// Load and validate `manifest.json` from a checkpoint bundle.
pub fn load_manifest(checkpoint_dir: impl AsRef<Path>) -> Result<CheckpointManifest> {
    let path = checkpoint_dir.as_ref().join("manifest.json");
    let contents = fs::read_to_string(&path).map_err(|error| {
        CognexisError::Backend(format!("failed to read {}: {error}", path.display()))
    })?;
    let manifest = serde_json::from_str::<CheckpointManifest>(&contents)
        .map_err(|error| CognexisError::InvalidConfig(format!("invalid manifest.json: {error}")))?;
    manifest.validate(checkpoint_dir)?;
    Ok(manifest)
}

fn load_optional_scheduler_state_for_manifest(
    checkpoint_dir: &Path,
    manifest: &CheckpointManifest,
) -> Result<Option<CheckpointSchedulerState>> {
    let path = manifest
        .artifacts
        .iter()
        .find(|artifact| artifact.kind == CheckpointArtifactKind::SchedulerState)
        .map(|artifact| checkpoint_dir.join(&artifact.path))
        .unwrap_or_else(|| checkpoint_dir.join("scheduler.json"));
    if path.exists() {
        load_scheduler_state_from_path(&path).map(Some)
    } else {
        Ok(None)
    }
}

fn load_scheduler_state_from_path(path: &Path) -> Result<CheckpointSchedulerState> {
    let contents = fs::read_to_string(path).map_err(|error| {
        CognexisError::Backend(format!("failed to read {}: {error}", path.display()))
    })?;
    let scheduler_state =
        serde_json::from_str::<CheckpointSchedulerState>(&contents).map_err(|error| {
            CognexisError::InvalidConfig(format!("invalid scheduler.json: {error}"))
        })?;
    scheduler_state.validate()?;
    Ok(scheduler_state)
}

fn artifact(
    path: impl Into<String>,
    kind: CheckpointArtifactKind,
    required: bool,
) -> CheckpointArtifact {
    CheckpointArtifact {
        path: path.into(),
        kind,
        required,
    }
}
