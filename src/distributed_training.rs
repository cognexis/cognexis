//! Distributed training module.
//!
//! This module provides stub definitions related to distributed
//! training. In practice, large language models are trained across
//! multiple GPUs and nodes using techniques like data parallelism,
//! pipeline parallelism, and parameter sharding. See
//! `spec14_distributed_training.md` for guidelines on partitioning the
//! model and synchronizing gradients.

/// Enumeration of distributed training strategies.
#[derive(Debug, Clone, Copy)]
pub enum TrainingStrategy {
    DataParallel,
    PipelineParallel,
    TensorParallel,
}

/// Structure representing distributed training configuration.
pub struct DistributedConfig {
    pub strategy: TrainingStrategy,
    pub world_size: usize,
    pub rank: usize,
}

impl DistributedConfig {
    pub fn new(strategy: TrainingStrategy, world_size: usize, rank: usize) -> Self {
        Self {
            strategy,
            world_size,
            rank,
        }
    }
}