//! Data loading module.
//!
//! This module provides structures for loading, packing, and batching
//! tokenized training data. File readers and asynchronous prefetching
//! can feed these reference types without changing the training loop
//! contract described in `spec12_data_loading.md`.

use crate::{CognexisError, Result};

/// Sentinel document ID used for padding positions in packed batches.
pub const PAD_DOCUMENT_ID: u32 = u32::MAX;

/// A single training example consisting of input and target token IDs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingExample {
    pub input_ids: Vec<u32>,
    pub target_ids: Vec<u32>,
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

fn validate_shifted_example(example: &TrainingExample) -> Result<()> {
    if example.input_ids.len() != example.target_ids.len() {
        return Err(CognexisError::ShapeMismatch {
            expected: format!("target length {}", example.input_ids.len()),
            actual: format!("target length {}", example.target_ids.len()),
        });
    }
    Ok(())
}
