//! Cognexis Library
//!
//! This crate provides a skeleton implementation of the Cognexis large
//! language model. The purpose of this skeleton is to demonstrate
//! module boundaries and basic type definitions without implementing
//! full functionality. Each module corresponds to a section of the
//! specification contained in the accompanying documentation.

pub mod ablation;
pub mod attention;
pub mod checkpoint;
pub mod coda;
pub mod config;
pub mod curriculum;
pub mod data_loading;
pub mod distributed_training;
pub mod embedding;
pub mod error;
pub mod evaluation;
pub mod feedforward;
pub mod instruction_tuning;
pub mod lm_head;
pub mod loop_scaling;
pub mod model;
pub mod prefill_decode;
pub mod prelude;
pub mod recurrent_core;
pub mod safety;
pub mod scheduler;
pub mod stability;
pub mod tokenizer;
pub mod tokenwise;
pub mod training;
pub mod transformer_block;
pub mod value_head;

pub use config::ServeConfig;
pub use error::{CognexisError, Result};
pub use model::{
    CognexisModel, ForwardOutput, GenerationRequest, GenerationStepOutput, LoopMode, LoopOptions,
    SamplingOptions, StopReason, TextGenerationOutput, TextGenerationRequest,
};
