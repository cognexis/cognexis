//! Shared error and result types for Cognexis.

/// Crate-wide result type.
pub type Result<T> = std::result::Result<T, CognexisError>;

/// Structured errors used for configuration, shape, tokenization, and
/// reference-backend failures.
#[derive(thiserror::Error, Debug, Clone, PartialEq)]
pub enum CognexisError {
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    #[error("shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: String, actual: String },

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("invalid token id: {0}")]
    InvalidTokenId(u32),

    #[error("backend error: {0}")]
    Backend(String),
}
