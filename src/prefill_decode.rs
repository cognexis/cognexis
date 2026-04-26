//! Prefill and decode module.
//!
//! This module provides placeholders for the logic that distinguishes
//! prefill (context encoding) and decode (autoregressive generation).
//! See `spec16_prefill_decode.md` for an explanation of KV caching
//! and incremental state updates.

use crate::config::ModelConfig;

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
    let _ = (config, token_ids);
    // TODO: Run embedding → prelude → recurrent iterations with full
    // sequence attention and populate the KV cache.
    (vec![vec![0.0; config.hidden_size]; token_ids.len()], KvCache::default())
}

/// Perform the decode phase: use a cached state to generate new
/// hidden states for the next token. Updates the KV cache in place.
pub fn decode(
    config: &ModelConfig,
    previous_hidden: &[Vec<f32>],
    next_token_id: u32,
    cache: &mut KvCache,
) -> Vec<f32> {
    let _ = (config, previous_hidden, next_token_id, cache);
    // TODO: Run a single iteration of the recurrent core and coda,
    // incorporating KV cache for efficiency.
    vec![0.0; config.hidden_size]
}