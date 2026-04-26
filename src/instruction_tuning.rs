//! Instruction tuning module.
//!
//! Instruction tuning fine‑tunes the base language model on a
//! collection of instruction/response pairs. It typically includes
//! reinforcement learning from human feedback or other reward models.
//! See `spec23_instruction_tuning.md` for suggestions on dataset
//! preparation and training procedures.

/// A single instruction tuning example.
pub struct InstructionExample {
    pub prompt: String,
    pub response: String,
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

/// Reference hook for instruction tuning. A production implementation
/// would tokenize, batch, optimize, checkpoint, and evaluate.
pub fn fine_tune(examples: &[InstructionExample]) {
    let _ = summarize_examples(examples);
}
