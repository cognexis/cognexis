//! Distributed training module.
//!
//! This module provides stub definitions related to distributed
//! training. In practice, large language models are trained across
//! multiple GPUs and nodes using techniques like data parallelism,
//! pipeline parallelism, and parameter sharding. See
//! `spec14_distributed_training.md` for guidelines on partitioning the
//! model and synchronizing gradients.

use serde::{Deserialize, Serialize};

use crate::curriculum::LoopSample;
use crate::{CognexisError, Result};

/// Enumeration of distributed training strategies.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TrainingStrategy {
    DataParallel,
    PipelineParallel,
    TensorParallel,
    FullyShardedDataParallel,
    SequenceParallel,
}

/// Structure representing distributed training configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DistributedConfig {
    pub strategy: TrainingStrategy,
    pub world_size: usize,
    pub rank: usize,
    #[serde(default = "default_local_world_size")]
    pub local_world_size: usize,
    #[serde(default = "default_gradient_accumulation_steps")]
    pub gradient_accumulation_steps: usize,
    #[serde(default)]
    pub broadcast_loop_samples: bool,
}

impl DistributedConfig {
    pub fn new(strategy: TrainingStrategy, world_size: usize, rank: usize) -> Self {
        Self {
            strategy,
            world_size,
            rank,
            local_world_size: default_local_world_size(),
            gradient_accumulation_steps: 1,
            broadcast_loop_samples: true,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.world_size == 0 {
            return Err(CognexisError::InvalidConfig(
                "distributed world_size must be positive".to_string(),
            ));
        }
        if self.rank >= self.world_size {
            return Err(CognexisError::InvalidConfig(format!(
                "distributed rank ({}) must be smaller than world_size ({})",
                self.rank, self.world_size
            )));
        }
        if self.local_world_size == 0 {
            return Err(CognexisError::InvalidConfig(
                "local_world_size must be positive".to_string(),
            ));
        }
        if self.gradient_accumulation_steps == 0 {
            return Err(CognexisError::InvalidConfig(
                "gradient_accumulation_steps must be positive".to_string(),
            ));
        }
        Ok(())
    }

    /// Select global item indices owned by this data-parallel rank.
    pub fn data_parallel_indices(&self, total_items: usize) -> Result<Vec<usize>> {
        self.validate()?;
        Ok((self.rank..total_items).step_by(self.world_size).collect())
    }

    /// Return this rank's inclusive-exclusive shard range for contiguous artifacts.
    pub fn contiguous_shard_range(&self, total_items: usize) -> Result<std::ops::Range<usize>> {
        self.validate()?;
        let start = total_items * self.rank / self.world_size;
        let end = total_items * (self.rank + 1) / self.world_size;
        Ok(start..end)
    }

    /// Return the loop sample all ranks should use for a synchronized step.
    pub fn synchronize_loop_sample(&self, rank_local_sample: LoopSample) -> Result<LoopSample> {
        self.validate()?;
        Ok(rank_local_sample)
    }
}

/// Collective reduction operation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Mean,
    Max,
}

/// Minimal collective interface used by reference training code.
pub trait Collective {
    fn world_size(&self) -> usize;
    fn all_reduce_f32(&self, values: &mut [f32], op: ReduceOp) -> Result<()>;
}

/// Single-process collective used by tests and CPU reference training.
#[derive(Debug, Clone, Copy)]
pub struct SingleProcessCollective;

impl Collective for SingleProcessCollective {
    fn world_size(&self) -> usize {
        1
    }

    fn all_reduce_f32(&self, values: &mut [f32], op: ReduceOp) -> Result<()> {
        if matches!(op, ReduceOp::Mean) {
            for value in values {
                *value /= self.world_size() as f32;
            }
        }
        Ok(())
    }
}

/// Estimate how many recurrent block applications are executed per synchronized step.
pub fn recurrent_applications_per_step(sample: LoopSample, micro_batches: usize) -> Result<usize> {
    if micro_batches == 0 {
        return Err(CognexisError::InvalidConfig(
            "micro_batches must be positive".to_string(),
        ));
    }
    Ok(sample.sampled_loops * micro_batches)
}

const fn default_local_world_size() -> usize {
    1
}

const fn default_gradient_accumulation_steps() -> usize {
    1
}
