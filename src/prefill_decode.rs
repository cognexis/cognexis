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

/// Represents cached key/value tensors used during decoding.
#[derive(Debug, Default)]
pub struct KvCache {
    // In a full implementation these would be multi‑dimensional arrays
    // storing past key and value projections.
    pub keys: Vec<Vec<f32>>,
    pub values: Vec<Vec<f32>>,
}

/// Perform the prefill phase: compute hidden states for the entire
/// input sequence and initialize the KV cache. Returns the hidden
/// states and a new cache.
pub fn prefill(config: &ModelConfig, token_ids: &[u32]) -> (Vec<Vec<f32>>, KvCache) {
    let embedding = Embedding::new(config);
    let prelude = Prelude::new(config);
    let recurrent = RecurrentCore::new(config);
    let coda = Coda::new(config);

    let embedded = embedding.forward(token_ids);
    let prepared = prelude.forward(&embedded);
    let refined = recurrent.forward(&prepared, config.max_loop_count);
    let hidden = coda.forward(&refined);

    let cache = KvCache {
        keys: hidden.clone(),
        values: hidden.clone(),
    };
    (hidden, cache)
}

/// Perform the decode phase: use a cached state to generate new
/// hidden states for the next token. Updates the KV cache in place.
pub fn decode(
    config: &ModelConfig,
    previous_hidden: &[Vec<f32>],
    next_token_id: u32,
    cache: &mut KvCache,
) -> Vec<f32> {
    let embedding = Embedding::new(config);
    let recurrent = RecurrentCore::new(config);
    let coda = Coda::new(config);

    let mut sequence = if cache.keys.is_empty() {
        previous_hidden.to_owned()
    } else {
        cache.keys.clone()
    };

    let next_embedding = embedding.forward(&[next_token_id]);
    if let Some(row) = next_embedding.into_iter().next() {
        sequence.push(row);
    }

    let refined = recurrent.forward(&sequence, config.min_loop_count);
    let hidden = coda.forward(&refined);
    cache.keys = hidden.clone();
    cache.values = hidden.clone();
    hidden
        .last()
        .cloned()
        .unwrap_or_else(|| vec![0.0; config.hidden_size])
}
