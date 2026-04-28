//! Checkpoint metadata and manifest support.
//!
//! Tensor serialization is backend-specific, but checkpoint bundles
//! still need a stable manifest recording config, tokenizer, scheduler,
//! data, and build metadata. These helpers implement the JSON metadata
//! contract used by the CPU reference path.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{CognexisError, Result};

pub const CHECKPOINT_SCHEMA_VERSION: u32 = 1;

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
