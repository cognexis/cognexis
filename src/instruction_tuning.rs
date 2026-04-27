//! Instruction tuning module.
//!
//! Instruction tuning fine‑tunes the base language model on a
//! collection of instruction/response pairs. It typically includes
//! reinforcement learning from human feedback or other reward models.
//! See `spec23_instruction_tuning.md` for suggestions on dataset
//! preparation and training procedures.

use crate::tokenizer::{EncodeOptions, TokenId, Tokenizer};
use crate::{CognexisError, Result};

/// A single instruction tuning example.
pub struct InstructionExample {
    pub prompt: String,
    pub response: String,
}

/// Structured chat role used by supervised instruction examples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

impl ChatRole {
    pub fn marker(self) -> &'static str {
        match self {
            ChatRole::System => "<|system|>",
            ChatRole::User => "<|user|>",
            ChatRole::Assistant => "<|assistant|>",
            ChatRole::Tool => "<|tool|>",
        }
    }
}

/// A single chat message before template rendering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

/// Tokenized SFT example with assistant-only loss masking.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderedChatExample {
    pub text: String,
    pub token_ids: Vec<TokenId>,
    pub loss_mask: Vec<f32>,
}

/// Summary of a supervised instruction batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InstructionBatchStats {
    pub examples: usize,
    pub total_prompt_chars: usize,
    pub total_response_chars: usize,
}

/// Validate and summarize instruction examples before a training loop
/// consumes them.
pub fn summarize_examples(examples: &[InstructionExample]) -> InstructionBatchStats {
    InstructionBatchStats {
        examples: examples.len(),
        total_prompt_chars: examples
            .iter()
            .map(|example| example.prompt.chars().count())
            .sum(),
        total_response_chars: examples
            .iter()
            .map(|example| example.response.chars().count())
            .sum(),
    }
}

/// Render messages with the reference ChatML-style template.
pub fn render_chat_template(messages: &[ChatMessage]) -> Result<String> {
    if messages.is_empty() {
        return Err(CognexisError::InvalidConfig(
            "chat examples must contain at least one message".to_string(),
        ));
    }

    let mut rendered = String::new();
    for message in messages {
        rendered.push_str(message.role.marker());
        rendered.push('\n');
        rendered.push_str(&message.content);
        rendered.push_str("<|end|>\n");
    }
    Ok(rendered)
}

/// Render and tokenize an SFT chat example while masking loss to
/// assistant content tokens only.
pub fn render_chat_for_sft(
    tokenizer: &Tokenizer,
    messages: &[ChatMessage],
) -> Result<RenderedChatExample> {
    if messages.is_empty() {
        return Err(CognexisError::InvalidConfig(
            "chat examples must contain at least one message".to_string(),
        ));
    }

    let mut text = String::new();
    let mut token_ids = Vec::new();
    let mut loss_mask = Vec::new();

    for message in messages {
        let prefix = format!("{}\n", message.role.marker());
        push_segment(
            tokenizer,
            &prefix,
            true,
            0.0,
            &mut text,
            &mut token_ids,
            &mut loss_mask,
        )?;

        let content_mask = if message.role == ChatRole::Assistant {
            1.0
        } else {
            0.0
        };
        push_segment(
            tokenizer,
            &message.content,
            false,
            content_mask,
            &mut text,
            &mut token_ids,
            &mut loss_mask,
        )?;

        push_segment(
            tokenizer,
            "<|end|>\n",
            true,
            0.0,
            &mut text,
            &mut token_ids,
            &mut loss_mask,
        )?;
    }

    Ok(RenderedChatExample {
        text,
        token_ids,
        loss_mask,
    })
}

/// Reference hook for instruction tuning. A production implementation
/// would tokenize, batch, optimize, checkpoint, and evaluate.
pub fn fine_tune(examples: &[InstructionExample]) {
    let _ = summarize_examples(examples);
}

fn push_segment(
    tokenizer: &Tokenizer,
    segment: &str,
    allow_special: bool,
    mask_value: f32,
    text: &mut String,
    token_ids: &mut Vec<TokenId>,
    loss_mask: &mut Vec<f32>,
) -> Result<()> {
    let encoded = tokenizer.encode_with_options(
        segment,
        EncodeOptions {
            allow_special,
            ..EncodeOptions::default()
        },
    )?;
    text.push_str(segment);
    loss_mask.extend(std::iter::repeat(mask_value).take(encoded.len()));
    token_ids.extend(encoded);
    Ok(())
}
