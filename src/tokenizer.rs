//! Tokenizer module.
//!
//! The production tokenizer is expected to load a SentencePiece/BPE
//! artifact. This module provides a deterministic byte-fallback
//! reference tokenizer with the same public semantics: explicit special
//! token handling, truncation policy, and checked decoding.

use std::collections::HashMap;

use crate::{CognexisError, Result};

/// A tokenized representation of input text.
pub type TokenId = u32;

const BOS_ID: TokenId = 0;
const EOS_ID: TokenId = 1;
const PAD_ID: TokenId = 2;
const UNK_ID: TokenId = 3;
const EOD_ID: TokenId = 4;
const SYSTEM_ID: TokenId = 5;
const USER_ID: TokenId = 6;
const ASSISTANT_ID: TokenId = 7;
const TOOL_ID: TokenId = 8;
const END_ID: TokenId = 9;
const BYTE_OFFSET: TokenId = 10;
const BYTE_VOCAB_SIZE: usize = 256;

/// Truncation policy applied after optional BOS/EOS insertion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncationPolicy {
    Error,
    Left,
    Right,
    Middle,
}

/// Encoding options matching the tokenizer contract in the spec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodeOptions {
    pub add_bos: bool,
    pub add_eos: bool,
    pub allow_special: bool,
    pub max_len: Option<usize>,
    pub truncation: TruncationPolicy,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            add_bos: false,
            add_eos: false,
            allow_special: false,
            max_len: None,
            truncation: TruncationPolicy::Error,
        }
    }
}

/// Decoding options matching the tokenizer contract in the spec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeOptions {
    pub stop_at_eos: bool,
    pub skip_padding: bool,
    pub show_special: bool,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            stop_at_eos: false,
            skip_padding: true,
            show_special: false,
        }
    }
}

/// Structure representing a tokenizer.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// Mapping from token IDs to string representations.
    id_to_token: HashMap<TokenId, String>,
    /// Mapping from string representations to token IDs.
    token_to_id: HashMap<String, TokenId>,
}

impl Tokenizer {
    /// Create the deterministic byte-fallback reference tokenizer.
    pub fn new() -> Self {
        let mut tokenizer = Self {
            id_to_token: HashMap::with_capacity(BYTE_VOCAB_SIZE + 10),
            token_to_id: HashMap::with_capacity(10),
        };

        for (id, token) in [
            (BOS_ID, "<s>"),
            (EOS_ID, "</s>"),
            (PAD_ID, "<pad>"),
            (UNK_ID, "<unk>"),
            (EOD_ID, "<eod>"),
            (SYSTEM_ID, "<|system|>"),
            (USER_ID, "<|user|>"),
            (ASSISTANT_ID, "<|assistant|>"),
            (TOOL_ID, "<|tool|>"),
            (END_ID, "<|end|>"),
        ] {
            tokenizer.id_to_token.insert(id, token.to_string());
            tokenizer.token_to_id.insert(token.to_string(), id);
        }

        tokenizer
    }

    /// Number of tokens in the reference tokenizer vocabulary.
    pub fn vocab_size(&self) -> usize {
        BYTE_OFFSET as usize + BYTE_VOCAB_SIZE
    }

    /// Token ID for the beginning-of-sequence marker.
    pub fn bos_id(&self) -> TokenId {
        BOS_ID
    }

    /// Token ID for the end-of-sequence marker.
    pub fn eos_id(&self) -> TokenId {
        EOS_ID
    }

    /// Token ID for the padding marker.
    pub fn pad_id(&self) -> TokenId {
        PAD_ID
    }

    /// Token ID for the unknown-token marker.
    pub fn unk_id(&self) -> TokenId {
        UNK_ID
    }

    /// Tokenize a string into token IDs using permissive legacy defaults.
    pub fn encode(&self, text: &str) -> Vec<TokenId> {
        self.encode_with_options(
            text,
            EncodeOptions {
                allow_special: true,
                ..EncodeOptions::default()
            },
        )
        .unwrap_or_default()
    }

    /// Tokenize a string into token IDs with explicit options.
    pub fn encode_with_options(&self, text: &str, options: EncodeOptions) -> Result<Vec<TokenId>> {
        let mut tokens = Vec::with_capacity(text.len() + 2);

        if options.add_bos {
            tokens.push(BOS_ID);
        }

        let special_tokens = self.special_tokens_by_length();
        let mut index = 0;
        while index < text.len() {
            let remaining = &text[index..];
            if let Some((token, id)) = special_tokens
                .iter()
                .find(|(token, _)| remaining.starts_with(token.as_str()))
            {
                if !options.allow_special {
                    return Err(CognexisError::Tokenizer(format!(
                        "special token {token:?} is not allowed in raw text"
                    )));
                }
                tokens.push(*id);
                index += token.len();
                continue;
            }

            let ch = remaining
                .chars()
                .next()
                .expect("index is inside a non-empty UTF-8 string");
            let mut buf = [0u8; 4];
            for byte in ch.encode_utf8(&mut buf).as_bytes() {
                tokens.push(BYTE_OFFSET + *byte as TokenId);
            }
            index += ch.len_utf8();
        }

        if options.add_eos {
            tokens.push(EOS_ID);
        }

        if let Some(max_len) = options.max_len {
            apply_truncation(&mut tokens, max_len, options.truncation)?;
        }

        Ok(tokens)
    }

    /// Detokenize a sequence of token IDs back into a string using
    /// display-oriented legacy defaults.
    pub fn decode(&self, tokens: &[TokenId]) -> String {
        self.decode_with_options(tokens, DecodeOptions::default())
            .unwrap_or_default()
    }

    /// Detokenize token IDs with explicit handling of EOS, PAD, and
    /// special-token display.
    pub fn decode_with_options(
        &self,
        tokens: &[TokenId],
        options: DecodeOptions,
    ) -> Result<String> {
        let mut output = String::new();
        let mut byte_buffer = Vec::new();

        for &id in tokens {
            if id == EOS_ID && options.stop_at_eos {
                break;
            }
            if id == PAD_ID && options.skip_padding {
                continue;
            }

            if let Some(byte) = byte_from_token_id(id) {
                byte_buffer.push(byte);
                continue;
            }

            flush_bytes(&mut output, &mut byte_buffer);

            match self.id_to_token.get(&id) {
                Some(token) if options.show_special => output.push_str(token),
                Some(_) => {}
                None => return Err(CognexisError::InvalidTokenId(id)),
            }
        }

        flush_bytes(&mut output, &mut byte_buffer);
        Ok(output)
    }

    fn special_tokens_by_length(&self) -> Vec<(String, TokenId)> {
        let mut tokens: Vec<_> = self
            .token_to_id
            .iter()
            .map(|(token, id)| (token.clone(), *id))
            .collect();
        tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.0.cmp(&b.0)));
        tokens
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

fn byte_from_token_id(id: TokenId) -> Option<u8> {
    let offset = id.checked_sub(BYTE_OFFSET)?;
    (offset < BYTE_VOCAB_SIZE as TokenId).then_some(offset as u8)
}

fn flush_bytes(output: &mut String, byte_buffer: &mut Vec<u8>) {
    if !byte_buffer.is_empty() {
        output.push_str(&String::from_utf8_lossy(byte_buffer));
        byte_buffer.clear();
    }
}

fn apply_truncation(
    tokens: &mut Vec<TokenId>,
    max_len: usize,
    policy: TruncationPolicy,
) -> Result<()> {
    if tokens.len() <= max_len {
        return Ok(());
    }

    match policy {
        TruncationPolicy::Error => Err(CognexisError::Tokenizer(format!(
            "encoded sequence length {} exceeds max_len {}",
            tokens.len(),
            max_len
        ))),
        TruncationPolicy::Left => {
            let start = tokens.len() - max_len;
            tokens.drain(0..start);
            Ok(())
        }
        TruncationPolicy::Right => {
            tokens.truncate(max_len);
            Ok(())
        }
        TruncationPolicy::Middle => {
            let left = max_len / 2;
            let right = max_len - left;
            let mut truncated = Vec::with_capacity(max_len);
            truncated.extend_from_slice(&tokens[..left]);
            truncated.extend_from_slice(&tokens[tokens.len() - right..]);
            *tokens = truncated;
            Ok(())
        }
    }
}
