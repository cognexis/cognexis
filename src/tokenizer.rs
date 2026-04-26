//! Tokenizer module.
//!
//! The tokenizer is responsible for converting raw text into a sequence
//! of token IDs and vice versa. It should be trained on a corpus of
//! domain‑specific text and persisted for reuse. The design described
//! in `spec02_tokenizer.md` recommends using a subword segmentation
//! algorithm (e.g. SentencePiece or BPE). This skeleton defines the
//! interfaces without implementing a specific algorithm.

use std::collections::HashMap;

/// A tokenized representation of input text.
pub type TokenId = u32;

/// Structure representing a tokenizer.
#[derive(Debug, Default)]
pub struct Tokenizer {
    /// Mapping from token IDs to string representations.
    id_to_token: HashMap<TokenId, String>,
    /// Mapping from string representations to token IDs.
    token_to_id: HashMap<String, TokenId>,
}

impl Tokenizer {
    /// Create a new tokenizer. In a full implementation this would
    /// construct or load vocabulary and training state. In this skeleton
    /// the vocabularies are empty.
    pub fn new() -> Self {
        Self {
            id_to_token: HashMap::new(),
            token_to_id: HashMap::new(),
        }
    }

    /// Tokenize a string into a vector of token IDs.
    pub fn encode(&self, text: &str) -> Vec<TokenId> {
        // TODO: Implement subword tokenization.
        // At present this returns an empty vector.
        let _ = text;
        vec![]
    }

    /// Detokenize a sequence of token IDs back into a string.
    pub fn decode(&self, tokens: &[TokenId]) -> String {
        // TODO: Implement detokenization.
        let _ = tokens;
        String::new()
    }
}