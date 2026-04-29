//! Data loading module.
//!
//! This module provides structures for loading, packing, and batching
//! tokenized training data. File readers and asynchronous prefetching
//! can feed these reference types without changing the training loop
//! contract described in `spec12_data_loading.md`.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::tokenizer::{EncodeOptions, TokenId, Tokenizer};
use crate::{CognexisError, Result};

/// Sentinel document ID used for padding positions in packed batches.
pub const PAD_DOCUMENT_ID: u32 = u32::MAX;

/// A single training example consisting of input and target token IDs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TrainingExample {
    pub input_ids: Vec<u32>,
    pub target_ids: Vec<u32>,
}

/// Supported dataset shard encodings.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DataShardFormat {
    JsonlText,
    JsonlTokenIds,
    TokenIdsU32,
}

/// One dataset shard listed in a manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DataShardManifestEntry {
    pub path: String,
    pub format: DataShardFormat,
    pub num_documents: usize,
    pub num_tokens: usize,
    pub checksum: Option<String>,
    pub weight: f32,
    pub domain: Option<String>,
}

impl DataShardManifestEntry {
    pub fn validate(&self) -> Result<()> {
        if self.path.trim().is_empty() {
            return Err(CognexisError::InvalidConfig(
                "data shard path must not be empty".to_string(),
            ));
        }
        if !self.weight.is_finite() || self.weight <= 0.0 {
            return Err(CognexisError::InvalidConfig(
                "data shard weight must be finite and positive".to_string(),
            ));
        }
        Ok(())
    }
}

/// Dataset manifest, the unit of data reproducibility.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DatasetManifest {
    pub schema_version: u32,
    pub shards: Vec<DataShardManifestEntry>,
    pub checksum: Option<String>,
}

impl DatasetManifest {
    pub fn new(shards: Vec<DataShardManifestEntry>) -> Result<Self> {
        let mut manifest = Self {
            schema_version: 1,
            shards,
            checksum: None,
        };
        manifest.validate()?;
        manifest.checksum = Some(manifest_checksum(&manifest));
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1 {
            return Err(CognexisError::InvalidConfig(format!(
                "unsupported dataset manifest schema_version {}",
                self.schema_version
            )));
        }
        if self.shards.is_empty() {
            return Err(CognexisError::InvalidConfig(
                "dataset manifest must contain at least one shard".to_string(),
            ));
        }
        for shard in &self.shards {
            shard.validate()?;
        }
        if let Some(checksum) = &self.checksum {
            let expected = manifest_checksum(self);
            if checksum != &expected {
                return Err(CognexisError::InvalidConfig(format!(
                    "dataset manifest checksum mismatch: expected {expected}, got {checksum}"
                )));
            }
        }
        Ok(())
    }

    pub fn total_tokens(&self) -> usize {
        self.shards.iter().map(|shard| shard.num_tokens).sum()
    }

    pub fn total_documents(&self) -> usize {
        self.shards.iter().map(|shard| shard.num_documents).sum()
    }
}

/// Save a dataset manifest as pretty JSON.
pub fn save_dataset_manifest(path: impl AsRef<Path>, manifest: &DatasetManifest) -> Result<()> {
    manifest.validate()?;
    let encoded = serde_json::to_vec_pretty(manifest).map_err(|error| {
        CognexisError::Backend(format!("dataset manifest serialization failed: {error}"))
    })?;
    fs::write(path.as_ref(), encoded).map_err(|error| {
        CognexisError::Backend(format!(
            "failed to write dataset manifest {}: {error}",
            path.as_ref().display()
        ))
    })
}

/// Load and validate a dataset manifest.
pub fn load_dataset_manifest(path: impl AsRef<Path>) -> Result<DatasetManifest> {
    let contents = fs::read_to_string(path.as_ref()).map_err(|error| {
        CognexisError::Backend(format!(
            "failed to read dataset manifest {}: {error}",
            path.as_ref().display()
        ))
    })?;
    let manifest = serde_json::from_str::<DatasetManifest>(&contents).map_err(|error| {
        CognexisError::InvalidConfig(format!("invalid dataset manifest JSON: {error}"))
    })?;
    manifest.validate()?;
    Ok(manifest)
}

/// Corrupt-record handling policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorruptionPolicy {
    Fail,
    Skip,
    Quarantine { path: String },
}

/// Metadata for a skipped or quarantined corrupt record.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CorruptRecord {
    pub line_number: usize,
    pub reason: String,
}

/// Report emitted by JSONL loading.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct DataLoadReport {
    pub records_read: usize,
    pub records_loaded: usize,
    pub records_skipped: usize,
    pub corrupt_records: Vec<CorruptRecord>,
}

/// Tokenized documents loaded from JSONL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedDocuments {
    pub documents: Vec<Vec<TokenId>>,
    pub report: DataLoadReport,
}

/// Per-example recurrent-loop metadata attached before the forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopMetadata {
    pub min_loops: usize,
    pub max_loops: usize,
    pub retain_intermediate_states: bool,
}

impl Default for LoopMetadata {
    fn default() -> Self {
        Self {
            min_loops: 1,
            max_loops: 1,
            retain_intermediate_states: false,
        }
    }
}

/// Options controlling document packing into fixed-length token blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DocumentPackingOptions {
    /// Full block length before next-token shifting.
    pub sequence_length: usize,
    pub eod_token_id: u32,
    pub pad_token_id: u32,
    pub document_boundary_attention: bool,
    pub loop_metadata: LoopMetadata,
}

impl Default for DocumentPackingOptions {
    fn default() -> Self {
        Self {
            sequence_length: 2_048,
            eod_token_id: 4,
            pad_token_id: 2,
            document_boundary_attention: false,
            loop_metadata: LoopMetadata::default(),
        }
    }
}

/// Dense batch tensors used by the reference training path.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingBatch {
    pub input_ids: Vec<Vec<u32>>,
    pub target_ids: Vec<Vec<u32>>,
    pub loss_mask: Vec<Vec<f32>>,
    pub attention_mask: Vec<Vec<bool>>,
    pub position_ids: Vec<Vec<u32>>,
    pub document_ids: Option<Vec<Vec<u32>>>,
    pub loop_metadata: Vec<LoopMetadata>,
}

impl TrainingBatch {
    /// Build a padded batch from already-shifted training examples.
    pub fn from_examples(
        examples: &[TrainingExample],
        pad_token_id: u32,
        loop_metadata: LoopMetadata,
    ) -> Result<Self> {
        let max_len = examples
            .iter()
            .map(|example| {
                validate_shifted_example(example)?;
                Ok(example.input_ids.len())
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap_or(0);

        let mut input_ids = Vec::with_capacity(examples.len());
        let mut target_ids = Vec::with_capacity(examples.len());
        let mut loss_mask = Vec::with_capacity(examples.len());
        let mut attention_mask = Vec::with_capacity(examples.len());
        let mut position_ids = Vec::with_capacity(examples.len());

        for example in examples {
            let real_len = example.input_ids.len();
            let mut inputs = example.input_ids.clone();
            let mut targets = example.target_ids.clone();
            inputs.resize(max_len, pad_token_id);
            targets.resize(max_len, pad_token_id);

            input_ids.push(inputs);
            target_ids.push(targets);
            loss_mask.push(
                (0..max_len)
                    .map(|index| if index < real_len { 1.0 } else { 0.0 })
                    .collect(),
            );
            attention_mask.push((0..max_len).map(|index| index < real_len).collect());
            position_ids.push((0..max_len as u32).collect());
        }

        Ok(Self {
            input_ids,
            target_ids,
            loss_mask,
            attention_mask,
            position_ids,
            document_ids: None,
            loop_metadata: vec![loop_metadata; examples.len()],
        })
    }

    /// Number of rows in the batch.
    pub fn batch_size(&self) -> usize {
        self.input_ids.len()
    }

    /// Number of shifted token positions per row.
    pub fn seq_len(&self) -> usize {
        self.input_ids.first().map_or(0, Vec::len)
    }
}

/// Pack tokenized documents into fixed-length language-modeling blocks.
///
/// Each non-empty document is followed by an EOD token. Blocks are then
/// shifted into `input_ids` and `target_ids`; loss is masked off only
/// for padding targets. When document-boundary attention is enabled,
/// `document_ids` records the source document for each input position.
pub fn pack_documents(
    documents: &[Vec<u32>],
    options: DocumentPackingOptions,
) -> Result<TrainingBatch> {
    if options.sequence_length < 2 {
        return Err(CognexisError::InvalidConfig(
            "sequence_length must be at least 2 for shifted targets".to_string(),
        ));
    }

    let mut blocks = Vec::new();
    let mut block_document_ids = Vec::new();
    let mut current_tokens = Vec::with_capacity(options.sequence_length);
    let mut current_document_ids = Vec::with_capacity(options.sequence_length);

    for (document_index, document) in documents.iter().enumerate() {
        if document.is_empty() {
            continue;
        }

        let document_id = document_index as u32;
        for token_id in document
            .iter()
            .copied()
            .chain(std::iter::once(options.eod_token_id))
        {
            current_tokens.push(token_id);
            current_document_ids.push(document_id);

            if current_tokens.len() == options.sequence_length {
                blocks.push(std::mem::take(&mut current_tokens));
                block_document_ids.push(std::mem::take(&mut current_document_ids));
                current_tokens = Vec::with_capacity(options.sequence_length);
                current_document_ids = Vec::with_capacity(options.sequence_length);
            }
        }
    }

    if !current_tokens.is_empty() {
        current_tokens.resize(options.sequence_length, options.pad_token_id);
        current_document_ids.resize(options.sequence_length, PAD_DOCUMENT_ID);
        blocks.push(current_tokens);
        block_document_ids.push(current_document_ids);
    }

    let shifted_len = options.sequence_length - 1;
    let mut input_ids = Vec::with_capacity(blocks.len());
    let mut target_ids = Vec::with_capacity(blocks.len());
    let mut loss_mask = Vec::with_capacity(blocks.len());
    let mut attention_mask = Vec::with_capacity(blocks.len());
    let mut position_ids = Vec::with_capacity(blocks.len());
    let mut batch_document_ids = options
        .document_boundary_attention
        .then(|| Vec::with_capacity(blocks.len()));

    for (block, document_ids) in blocks.into_iter().zip(block_document_ids) {
        input_ids.push(block[..shifted_len].to_vec());
        target_ids.push(block[1..].to_vec());
        loss_mask.push(
            block[1..]
                .iter()
                .map(|&token_id| {
                    if token_id == options.pad_token_id {
                        0.0
                    } else {
                        1.0
                    }
                })
                .collect(),
        );
        attention_mask.push(
            block[..shifted_len]
                .iter()
                .map(|&token_id| token_id != options.pad_token_id)
                .collect(),
        );
        position_ids.push((0..shifted_len as u32).collect());

        if let Some(all_document_ids) = &mut batch_document_ids {
            all_document_ids.push(document_ids[..shifted_len].to_vec());
        }
    }

    let row_count = input_ids.len();
    Ok(TrainingBatch {
        input_ids,
        target_ids,
        loss_mask,
        attention_mask,
        position_ids,
        document_ids: batch_document_ids,
        loop_metadata: vec![options.loop_metadata; row_count],
    })
}

/// Load JSONL records containing either `{ "text": "..." }` or
/// `{ "token_ids": [1, 2, ...] }`.
pub fn load_jsonl_documents(
    path: impl AsRef<Path>,
    tokenizer: &Tokenizer,
    policy: CorruptionPolicy,
) -> Result<LoadedDocuments> {
    let contents = fs::read_to_string(path.as_ref()).map_err(|error| {
        CognexisError::Backend(format!(
            "failed to read JSONL data {}: {error}",
            path.as_ref().display()
        ))
    })?;

    let mut documents = Vec::new();
    let mut report = DataLoadReport::default();
    for (line_index, line) in contents.lines().enumerate() {
        let line_number = line_index + 1;
        if line.trim().is_empty() {
            continue;
        }
        report.records_read += 1;
        match parse_jsonl_document(line, tokenizer) {
            Ok(document) => {
                report.records_loaded += 1;
                documents.push(document);
            }
            Err(error) => handle_corrupt_record(
                &policy,
                CorruptRecord {
                    line_number,
                    reason: error.to_string(),
                },
                &mut report,
            )?,
        }
    }

    if let CorruptionPolicy::Quarantine { path } = &policy {
        write_quarantine(path, &report.corrupt_records)?;
    }

    Ok(LoadedDocuments { documents, report })
}

/// Data loader for deterministic in-memory iteration.
pub struct DataLoader {
    /// Collection of all training examples.
    pub examples: Vec<TrainingExample>,
    /// Batch size used for iteration.
    pub batch_size: usize,
    pub position: usize,
}

impl DataLoader {
    /// Create a new data loader from a list of examples and a batch size.
    pub fn new(examples: Vec<TrainingExample>, batch_size: usize) -> Self {
        Self {
            examples,
            batch_size: batch_size.max(1),
            position: 0,
        }
    }

    /// Fetch the next batch of examples. Returns `None` when no more
    /// batches are available.
    pub fn next_batch(&mut self) -> Option<Vec<TrainingExample>> {
        if self.position >= self.examples.len() {
            return None;
        }
        let end = (self.position + self.batch_size).min(self.examples.len());
        let batch = self.examples[self.position..end].to_vec();
        self.position = end;
        Some(batch)
    }

    /// Reset iteration back to the first example.
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Fetch the next batch as padded training tensors.
    pub fn next_training_batch(
        &mut self,
        pad_token_id: u32,
        loop_metadata: LoopMetadata,
    ) -> Result<Option<TrainingBatch>> {
        let Some(examples) = self.next_batch() else {
            return Ok(None);
        };

        TrainingBatch::from_examples(&examples, pad_token_id, loop_metadata).map(Some)
    }
}

/// Deterministically partition examples for a distributed data-parallel rank.
pub fn partition_for_rank(
    examples: &[TrainingExample],
    world_size: usize,
    rank: usize,
) -> Result<Vec<TrainingExample>> {
    if world_size == 0 {
        return Err(CognexisError::InvalidConfig(
            "world_size must be positive".to_string(),
        ));
    }
    if rank >= world_size {
        return Err(CognexisError::InvalidConfig(format!(
            "rank ({rank}) must be smaller than world_size ({world_size})"
        )));
    }

    Ok(examples
        .iter()
        .enumerate()
        .filter(|(index, _)| index % world_size == rank)
        .map(|(_, example)| example.clone())
        .collect())
}

/// Checkpointable in-memory loader state.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct DataLoaderState {
    pub position: usize,
    pub epoch: u64,
    pub consumed_examples: u64,
}

impl DataLoader {
    /// Export checkpointable loader state.
    pub fn state_dict(&self, epoch: u64) -> DataLoaderState {
        DataLoaderState {
            position: self.position,
            epoch,
            consumed_examples: self.position as u64,
        }
    }

    /// Restore checkpointed loader state.
    pub fn load_state_dict(&mut self, state: DataLoaderState) -> Result<()> {
        if state.position > self.examples.len() {
            return Err(CognexisError::InvalidConfig(format!(
                "loader position {} exceeds example count {}",
                state.position,
                self.examples.len()
            )));
        }
        self.position = state.position;
        Ok(())
    }
}

fn validate_shifted_example(example: &TrainingExample) -> Result<()> {
    if example.input_ids.len() != example.target_ids.len() {
        return Err(CognexisError::ShapeMismatch {
            expected: format!("target length {}", example.input_ids.len()),
            actual: format!("target length {}", example.target_ids.len()),
        });
    }
    Ok(())
}

fn parse_jsonl_document(line: &str, tokenizer: &Tokenizer) -> Result<Vec<TokenId>> {
    let value = serde_json::from_str::<serde_json::Value>(line)
        .map_err(|error| CognexisError::InvalidConfig(format!("invalid JSON: {error}")))?;
    if let Some(text) = value.get("text").and_then(serde_json::Value::as_str) {
        return tokenizer.encode_with_options(
            text,
            EncodeOptions {
                allow_special: false,
                ..EncodeOptions::default()
            },
        );
    }
    if let Some(token_ids) = value.get("token_ids").and_then(serde_json::Value::as_array) {
        let mut document = Vec::with_capacity(token_ids.len());
        for token_id in token_ids {
            let id = token_id.as_u64().ok_or_else(|| {
                CognexisError::InvalidConfig("token_ids must contain unsigned integers".to_string())
            })?;
            if id > u32::MAX as u64 {
                return Err(CognexisError::InvalidTokenId(u32::MAX));
            }
            document.push(id as TokenId);
        }
        if document.is_empty() {
            return Err(CognexisError::InvalidConfig(
                "token_ids record must not be empty".to_string(),
            ));
        }
        return Ok(document);
    }
    Err(CognexisError::InvalidConfig(
        "JSONL record must contain text or token_ids".to_string(),
    ))
}

fn handle_corrupt_record(
    policy: &CorruptionPolicy,
    record: CorruptRecord,
    report: &mut DataLoadReport,
) -> Result<()> {
    match policy {
        CorruptionPolicy::Fail => Err(CognexisError::InvalidConfig(format!(
            "corrupt record at line {}: {}",
            record.line_number, record.reason
        ))),
        CorruptionPolicy::Skip | CorruptionPolicy::Quarantine { .. } => {
            report.records_skipped += 1;
            report.corrupt_records.push(record);
            Ok(())
        }
    }
}

fn write_quarantine(path: &str, records: &[CorruptRecord]) -> Result<()> {
    let mut output = String::new();
    for record in records {
        let line = serde_json::to_string(record).map_err(|error| {
            CognexisError::Backend(format!("quarantine serialization failed: {error}"))
        })?;
        output.push_str(&line);
        output.push('\n');
    }
    fs::write(path, output).map_err(|error| {
        CognexisError::Backend(format!("failed to write quarantine file {path}: {error}"))
    })
}

fn manifest_checksum(manifest: &DatasetManifest) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for shard in &manifest.shards {
        hash_bytes(&mut hash, shard.path.as_bytes());
        hash_bytes(&mut hash, format!("{:?}", shard.format).as_bytes());
        hash_bytes(&mut hash, &shard.num_documents.to_le_bytes());
        hash_bytes(&mut hash, &shard.num_tokens.to_le_bytes());
        hash_bytes(
            &mut hash,
            shard.checksum.as_deref().unwrap_or("").as_bytes(),
        );
        hash_bytes(&mut hash, &shard.weight.to_bits().to_le_bytes());
        hash_bytes(&mut hash, shard.domain.as_deref().unwrap_or("").as_bytes());
    }
    format!("fnv64:{hash:016x}")
}

fn hash_bytes(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= *byte as u64;
        *hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
}
