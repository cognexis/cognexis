//! Prefill and decode module.
//!
//! This module provides placeholders for the logic that distinguishes
//! prefill (context encoding) and decode (autoregressive generation).
//! See `spec16_prefill_decode.md` for an explanation of KV caching
//! and incremental state updates.

use crate::coda::Coda;
use crate::config::ModelConfig;
use crate::embedding::Embedding;
use crate::prelude::Prelude;
use crate::recurrent_core::RecurrentCore;
use crate::{CognexisError, Result};

/// Cache stage ownership. Entries from one stage must not be reused as
/// if they came from another stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStage {
    Prelude,
    Recurrent { loop_index: usize },
    Coda,
}

/// One stage-owned cache entry in the CPU reference path.
#[derive(Debug, Clone, PartialEq)]
pub struct CacheEntry {
    pub stage: CacheStage,
    pub layer_id: usize,
    pub position_id: u32,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
}

/// Represents cached key/value tensors used during decoding.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct KvCache {
    /// Legacy full-sequence reference keys retained for backwards compatibility.
    pub keys: Vec<Vec<f32>>,
    /// Legacy full-sequence reference values retained for backwards compatibility.
    pub values: Vec<Vec<f32>>,
    /// Stage-owned entries for cache correctness tests and serving metadata.
    pub entries: Vec<CacheEntry>,
    /// Number of real sequence positions represented by the cache.
    pub sequence_len: usize,
    /// Maximum sequence length accepted by this cache.
    pub max_sequence_len: Option<usize>,
    /// Whether the cache has been released.
    pub released: bool,
}

impl KvCache {
    /// Create an empty cache with an optional capacity limit.
    pub fn with_capacity(max_sequence_len: Option<usize>) -> Self {
        Self {
            max_sequence_len,
            ..Self::default()
        }
    }

    /// Append one position's stage-owned entries.
    pub fn append_position(
        &mut self,
        position_id: u32,
        hidden: &[f32],
        recurrent_loops: usize,
    ) -> Result<()> {
        if self.released {
            return Err(CognexisError::Backend(
                "cannot append to a released KV cache".to_string(),
            ));
        }
        if let Some(limit) = self.max_sequence_len {
            if self.sequence_len >= limit {
                return Err(CognexisError::InvalidConfig(format!(
                    "KV cache sequence length {} exceeds max_sequence_len {}",
                    self.sequence_len + 1,
                    limit
                )));
            }
        }

        self.entries.push(CacheEntry {
            stage: CacheStage::Prelude,
            layer_id: 0,
            position_id,
            key: hidden.to_vec(),
            value: hidden.to_vec(),
        });
        for loop_index in 0..recurrent_loops {
            self.entries.push(CacheEntry {
                stage: CacheStage::Recurrent { loop_index },
                layer_id: 0,
                position_id,
                key: hidden.to_vec(),
                value: hidden.to_vec(),
            });
        }
        self.entries.push(CacheEntry {
            stage: CacheStage::Coda,
            layer_id: 0,
            position_id,
            key: hidden.to_vec(),
            value: hidden.to_vec(),
        });
        self.sequence_len += 1;
        Ok(())
    }

    /// Entries for a stage and layer.
    pub fn entries_for(&self, stage: CacheStage, layer_id: usize) -> Vec<&CacheEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.stage == stage && entry.layer_id == layer_id)
            .collect()
    }

    /// Approximate cache memory in bytes for f32 keys/values.
    pub fn memory_bytes(&self) -> usize {
        let legacy = self
            .keys
            .iter()
            .chain(&self.values)
            .map(|row| row.len() * std::mem::size_of::<f32>())
            .sum::<usize>();
        let structured = self
            .entries
            .iter()
            .map(|entry| (entry.key.len() + entry.value.len()) * std::mem::size_of::<f32>())
            .sum::<usize>();
        legacy + structured
    }

    /// Release all cache memory and mark this cache closed.
    pub fn release(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.entries.clear();
        self.sequence_len = 0;
        self.released = true;
    }
}

/// Options for checked prefill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillOptions {
    pub loops: usize,
    pub max_sequence_len: Option<usize>,
}

impl Default for PrefillOptions {
    fn default() -> Self {
        Self {
            loops: 1,
            max_sequence_len: None,
        }
    }
}

/// Checked prefill output.
#[derive(Debug, Clone, PartialEq)]
pub struct PrefillOutput {
    pub hidden: Vec<Vec<f32>>,
    pub cache: KvCache,
    pub position_ids: Vec<u32>,
}

/// Decode-step result for tests and serving composition.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeStepOutput {
    pub hidden: Vec<f32>,
    pub position_id: u32,
    pub cache_sequence_len: usize,
}

/// Perform the prefill phase: compute hidden states for the entire
/// input sequence and initialize the KV cache. Returns the hidden
/// states and a new cache.
pub fn prefill(config: &ModelConfig, token_ids: &[u32]) -> (Vec<Vec<f32>>, KvCache) {
    let output = prefill_checked(
        config,
        token_ids,
        PrefillOptions {
            loops: config.max_loop_count,
            max_sequence_len: None,
        },
    )
    .expect("legacy prefill uses internally valid options");
    (output.hidden, output.cache)
}

/// Checked prefill path with position IDs and structured cache entries.
pub fn prefill_checked(
    config: &ModelConfig,
    token_ids: &[u32],
    options: PrefillOptions,
) -> Result<PrefillOutput> {
    config.validate()?;
    if let Some(limit) = options.max_sequence_len {
        if token_ids.len() > limit {
            return Err(CognexisError::InvalidConfig(format!(
                "prompt length {} exceeds max_sequence_len {}",
                token_ids.len(),
                limit
            )));
        }
    }

    let embedding = Embedding::new(config);
    let prelude = Prelude::new(config);
    let recurrent = RecurrentCore::new(config);
    let coda = Coda::new(config);

    let embedded = embedding.try_forward(token_ids)?;
    let prepared = prelude.forward(&embedded);
    let loops = options
        .loops
        .clamp(config.min_loop_count, config.max_loop_count);
    let refined = recurrent.forward(&prepared, loops);
    let hidden = coda.forward(&refined);

    let position_ids = position_ids(token_ids.len(), 0);
    let mut cache = KvCache {
        keys: hidden.clone(),
        values: hidden.clone(),
        max_sequence_len: options.max_sequence_len,
        ..KvCache::default()
    };
    for (position_id, row) in position_ids.iter().copied().zip(&hidden) {
        cache.append_position(position_id, row, loops)?;
    }
    cache.sequence_len = hidden.len();

    Ok(PrefillOutput {
        hidden,
        cache,
        position_ids,
    })
}

/// Perform the decode phase: use a cached state to generate new
/// hidden states for the next token. Updates the KV cache in place.
pub fn decode(
    config: &ModelConfig,
    previous_hidden: &[Vec<f32>],
    next_token_id: u32,
    cache: &mut KvCache,
) -> Vec<f32> {
    decode_step(
        config,
        previous_hidden,
        next_token_id,
        config.min_loop_count,
        cache,
    )
    .map(|output| output.hidden)
    .unwrap_or_else(|_| vec![0.0; config.hidden_size])
}

/// Checked decode step with explicit loop count and cache metadata.
pub fn decode_step(
    config: &ModelConfig,
    previous_hidden: &[Vec<f32>],
    next_token_id: u32,
    loops: usize,
    cache: &mut KvCache,
) -> Result<DecodeStepOutput> {
    config.validate()?;
    if cache.released {
        return Err(CognexisError::Backend(
            "cannot decode with a released KV cache".to_string(),
        ));
    }
    let embedding = Embedding::new(config);
    let recurrent = RecurrentCore::new(config);
    let coda = Coda::new(config);

    let mut sequence = if cache.keys.is_empty() {
        previous_hidden.to_owned()
    } else {
        cache.keys.clone()
    };

    let position_id = decode_position_id(cache, previous_hidden);
    let next_embedding = embedding.try_forward(&[next_token_id])?;
    if let Some(row) = next_embedding.into_iter().next() {
        sequence.push(row);
    }

    let loops = loops.clamp(config.min_loop_count, config.max_loop_count);
    let refined = recurrent.forward(&sequence, loops);
    let hidden = coda.forward(&refined);
    cache.keys = hidden.clone();
    cache.values = hidden.clone();
    let next_hidden = hidden
        .last()
        .cloned()
        .ok_or_else(|| CognexisError::Backend("decode produced no hidden state".to_string()))?;
    cache.append_position(position_id, &next_hidden, loops)?;
    cache.sequence_len = hidden.len();

    Ok(DecodeStepOutput {
        hidden: next_hidden,
        position_id,
        cache_sequence_len: cache.sequence_len,
    })
}

/// Monotonic position IDs for a contiguous sequence.
pub fn position_ids(seq_len: usize, offset: u32) -> Vec<u32> {
    (0..seq_len).map(|index| offset + index as u32).collect()
}

/// Position IDs for left-padded batches where `true` indicates a real token.
pub fn position_ids_from_attention_mask(attention_mask: &[bool]) -> Vec<u32> {
    let mut next_position = 0u32;
    attention_mask
        .iter()
        .map(|is_real| {
            if *is_real {
                let position = next_position;
                next_position += 1;
                position
            } else {
                0
            }
        })
        .collect()
}

fn decode_position_id(cache: &KvCache, previous_hidden: &[Vec<f32>]) -> u32 {
    let prefix_len = if cache.sequence_len > 0 {
        cache.sequence_len
    } else if !cache.keys.is_empty() {
        cache.keys.len()
    } else {
        previous_hidden.len()
    };
    prefix_len as u32
}
