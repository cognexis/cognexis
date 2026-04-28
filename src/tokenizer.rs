//! Tokenizer module.
//!
//! The production tokenizer is expected to load a SentencePiece/BPE
//! artifact. This module provides a deterministic byte-fallback
//! reference tokenizer with the same public semantics: explicit special
//! token handling, truncation policy, and checked decoding.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

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
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TruncationPolicy {
    Error,
    Left,
    Right,
    Middle,
}

/// Encoding options matching the tokenizer contract in the spec.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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

/// Tokenizer algorithm family recorded in checkpoint manifests.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TokenizerKind {
    ReferenceByteFallback,
    SentencePiece,
    Bpe,
}

/// One special-token declaration in a tokenizer manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpecialTokenManifest {
    pub name: String,
    pub token: String,
    pub id: TokenId,
}

/// Serializable tokenizer manifest used for checkpoint compatibility.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenizerManifest {
    pub schema_version: u32,
    pub kind: TokenizerKind,
    pub tokenizer_version: String,
    pub vocab_size: usize,
    pub allow_byte_fallback: bool,
    pub add_bos_default: bool,
    pub add_eos_default: bool,
    pub chat_template: Option<String>,
    pub checksum: Option<String>,
    pub special_tokens: Vec<SpecialTokenManifest>,
}

impl TokenizerManifest {
    /// Build the manifest for the deterministic reference tokenizer.
    pub fn reference() -> Self {
        Self {
            schema_version: 1,
            kind: TokenizerKind::ReferenceByteFallback,
            tokenizer_version: "reference-byte-v1".to_string(),
            vocab_size: BYTE_OFFSET as usize + BYTE_VOCAB_SIZE,
            allow_byte_fallback: true,
            add_bos_default: false,
            add_eos_default: false,
            chat_template: Some("chatml_v1".to_string()),
            checksum: Some(reference_tokenizer_checksum()),
            special_tokens: default_special_token_manifest(),
        }
    }

    /// Validate manifest self-consistency independent of tokenizer state.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1 {
            return Err(CognexisError::Tokenizer(format!(
                "unsupported tokenizer manifest schema_version {}",
                self.schema_version
            )));
        }
        if self.vocab_size == 0 {
            return Err(CognexisError::Tokenizer(
                "tokenizer vocab_size must be positive".to_string(),
            ));
        }
        if self.tokenizer_version.trim().is_empty() {
            return Err(CognexisError::Tokenizer(
                "tokenizer_version must not be empty".to_string(),
            ));
        }

        let mut names = HashSet::new();
        let mut ids = HashSet::new();
        let mut strings = HashSet::new();
        for special in &self.special_tokens {
            if special.name.trim().is_empty() || special.token.is_empty() {
                return Err(CognexisError::Tokenizer(
                    "special token names and strings must not be empty".to_string(),
                ));
            }
            if special.id as usize >= self.vocab_size {
                return Err(CognexisError::Tokenizer(format!(
                    "special token {} has id {} outside vocab_size {}",
                    special.name, special.id, self.vocab_size
                )));
            }
            if !names.insert(special.name.as_str())
                || !ids.insert(special.id)
                || !strings.insert(special.token.as_str())
            {
                return Err(CognexisError::Tokenizer(
                    "special token names, ids, and strings must be unique".to_string(),
                ));
            }
        }

        for required in ["bos", "eos", "pad", "unk", "eod"] {
            if !names.contains(required) {
                return Err(CognexisError::Tokenizer(format!(
                    "tokenizer manifest missing required special token {required:?}"
                )));
            }
        }
        if self.chat_template.as_deref() == Some("chatml_v1") {
            for required in ["system", "user", "assistant", "tool", "end"] {
                if !names.contains(required) {
                    return Err(CognexisError::Tokenizer(format!(
                        "chatml_v1 manifest missing role token {required:?}"
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Incremental decoder that preserves UTF-8 boundaries for streaming.
#[derive(Debug, Clone)]
pub struct StreamingDecoder {
    tokenizer: Tokenizer,
    options: DecodeOptions,
    byte_buffer: Vec<u8>,
    stopped: bool,
}

impl StreamingDecoder {
    /// Decode one token and return only newly completed text.
    pub fn push(&mut self, token_id: TokenId) -> Result<String> {
        if self.stopped {
            return Ok(String::new());
        }
        if token_id == EOS_ID && self.options.stop_at_eos {
            self.stopped = true;
            return Ok(self.finish());
        }
        if token_id == PAD_ID && self.options.skip_padding {
            return Ok(String::new());
        }
        if let Some(byte) = byte_from_token_id(token_id) {
            self.byte_buffer.push(byte);
            return Ok(flush_complete_utf8(&mut self.byte_buffer));
        }

        let mut output = self.finish();
        match self.tokenizer.id_to_token.get(&token_id) {
            Some(token) if self.options.show_special => output.push_str(token),
            Some(_) => {}
            None => return Err(CognexisError::InvalidTokenId(token_id)),
        }
        Ok(output)
    }

    /// Flush pending bytes at the end of a stream.
    pub fn finish(&mut self) -> String {
        if self.byte_buffer.is_empty() {
            return String::new();
        }
        let output = String::from_utf8_lossy(&self.byte_buffer).into_owned();
        self.byte_buffer.clear();
        output
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

    /// Token ID for the end-of-document marker.
    pub fn eod_id(&self) -> TokenId {
        EOD_ID
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

    /// Token ID for a named special token.
    pub fn special_token_id(&self, token: &str) -> Option<TokenId> {
        self.token_to_id.get(token).copied()
    }

    /// Return the checkpoint-compatible manifest for this tokenizer.
    pub fn manifest(&self) -> TokenizerManifest {
        TokenizerManifest::reference()
    }

    /// Validate that a manifest matches this tokenizer.
    pub fn validate_manifest(&self, manifest: &TokenizerManifest) -> Result<()> {
        manifest.validate()?;
        if manifest.vocab_size != self.vocab_size() {
            return Err(CognexisError::Tokenizer(format!(
                "manifest vocab_size ({}) does not match tokenizer vocab_size ({})",
                manifest.vocab_size,
                self.vocab_size()
            )));
        }
        for special in &manifest.special_tokens {
            match self.token_to_id.get(&special.token) {
                Some(id) if *id == special.id => {}
                Some(id) => {
                    return Err(CognexisError::Tokenizer(format!(
                        "manifest token {:?} id {} does not match tokenizer id {}",
                        special.token, special.id, id
                    )))
                }
                None => {
                    return Err(CognexisError::Tokenizer(format!(
                        "manifest token {:?} is not known to tokenizer",
                        special.token
                    )))
                }
            }
        }
        if let Some(checksum) = &manifest.checksum {
            let expected = reference_tokenizer_checksum();
            if checksum != &expected {
                return Err(CognexisError::Tokenizer(format!(
                    "tokenizer checksum mismatch: expected {expected}, got {checksum}"
                )));
            }
        }
        Ok(())
    }

    /// Save the reference tokenizer manifest as JSON.
    pub fn save_manifest(&self, path: impl AsRef<Path>) -> Result<()> {
        let manifest = self.manifest();
        manifest.validate()?;
        let encoded = serde_json::to_vec_pretty(&manifest).map_err(|error| {
            CognexisError::Backend(format!("tokenizer manifest serialization failed: {error}"))
        })?;
        fs::write(path.as_ref(), encoded).map_err(|error| {
            CognexisError::Backend(format!(
                "failed to write tokenizer manifest {}: {error}",
                path.as_ref().display()
            ))
        })
    }

    /// Load a tokenizer from a manifest JSON file.
    pub fn from_manifest(path: impl AsRef<Path>) -> Result<Self> {
        let manifest = load_tokenizer_manifest(path)?;
        let tokenizer = Self::new();
        tokenizer.validate_manifest(&manifest)?;
        Ok(tokenizer)
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

    /// Create an incremental decoder for streaming generation.
    pub fn streaming_decoder(&self, options: DecodeOptions) -> StreamingDecoder {
        StreamingDecoder {
            tokenizer: self.clone(),
            options,
            byte_buffer: Vec::new(),
            stopped: false,
        }
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

/// Load a tokenizer manifest JSON file.
pub fn load_tokenizer_manifest(path: impl AsRef<Path>) -> Result<TokenizerManifest> {
    let contents = fs::read_to_string(path.as_ref()).map_err(|error| {
        CognexisError::Backend(format!(
            "failed to read tokenizer manifest {}: {error}",
            path.as_ref().display()
        ))
    })?;
    let manifest = serde_json::from_str::<TokenizerManifest>(&contents).map_err(|error| {
        CognexisError::Tokenizer(format!("invalid tokenizer manifest: {error}"))
    })?;
    manifest.validate()?;
    Ok(manifest)
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

fn default_special_token_manifest() -> Vec<SpecialTokenManifest> {
    [
        ("bos", "<s>", BOS_ID),
        ("eos", "</s>", EOS_ID),
        ("pad", "<pad>", PAD_ID),
        ("unk", "<unk>", UNK_ID),
        ("eod", "<eod>", EOD_ID),
        ("system", "<|system|>", SYSTEM_ID),
        ("user", "<|user|>", USER_ID),
        ("assistant", "<|assistant|>", ASSISTANT_ID),
        ("tool", "<|tool|>", TOOL_ID),
        ("end", "<|end|>", END_ID),
    ]
    .into_iter()
    .map(|(name, token, id)| SpecialTokenManifest {
        name: name.to_string(),
        token: token.to_string(),
        id,
    })
    .collect()
}

fn reference_tokenizer_checksum() -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for special in default_special_token_manifest() {
        for byte in special.name.bytes().chain(special.token.bytes()) {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x1000_0000_01b3);
        }
        for byte in special.id.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x1000_0000_01b3);
        }
    }
    for byte in (BYTE_OFFSET as usize + BYTE_VOCAB_SIZE).to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    format!("fnv64:{hash:016x}")
}

fn flush_bytes(output: &mut String, byte_buffer: &mut Vec<u8>) {
    if !byte_buffer.is_empty() {
        output.push_str(&String::from_utf8_lossy(byte_buffer));
        byte_buffer.clear();
    }
}

fn flush_complete_utf8(byte_buffer: &mut Vec<u8>) -> String {
    let mut output = String::new();

    loop {
        if byte_buffer.is_empty() {
            break;
        }

        match std::str::from_utf8(byte_buffer) {
            Ok(text) => {
                output.push_str(text);
                byte_buffer.clear();
                break;
            }
            Err(error) if error.error_len().is_none() => {
                let valid_up_to = error.valid_up_to();
                if valid_up_to == 0 {
                    break;
                }
                output.push_str(
                    std::str::from_utf8(&byte_buffer[..valid_up_to])
                        .expect("valid_up_to marks a valid UTF-8 prefix"),
                );
                byte_buffer.drain(..valid_up_to);
                break;
            }
            Err(error) => {
                let valid_up_to = error.valid_up_to();
                if valid_up_to > 0 {
                    output.push_str(
                        std::str::from_utf8(&byte_buffer[..valid_up_to])
                            .expect("valid_up_to marks a valid UTF-8 prefix"),
                    );
                }
                let bad_len = error.error_len().unwrap_or(1);
                output.push('\u{FFFD}');
                byte_buffer.drain(..valid_up_to + bad_len);
            }
        }
    }

    output
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
