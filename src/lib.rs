//! Cognexis Library
//!
//! This crate provides a skeleton implementation of the Cognexis large
//! language model. The purpose of this skeleton is to demonstrate
//! module boundaries and basic type definitions without implementing
//! full functionality. Each module corresponds to a section of the
//! specification contained in the accompanying documentation.

pub mod config;
pub mod tokenizer;
pub mod embedding;
pub mod attention;
pub mod feedforward;
pub mod transformer_block;
pub mod prelude;
pub mod recurrent_core;
pub mod coda;
pub mod lm_head;
pub mod data_loading;
pub mod curriculum;
pub mod distributed_training;
pub mod stability;
pub mod prefill_decode;
pub mod scheduler;
pub mod tokenwise;
pub mod value_head;
pub mod evaluation;
pub mod loop_scaling;
pub mod ablation;
pub mod instruction_tuning;
pub mod safety;