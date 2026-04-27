//! Token‑wise scheduling module.
//!
//! Token‑wise scheduling allows the model to allocate different
//! numbers of loops to different positions in the sequence. See
//! `spec18_tokenwise_scheduling.md` for design considerations and
//! potential heuristics. This module provides the dense masked
//! execution path recommended as the first implementation: compute a
//! candidate update for all tokens and apply it only to active tokens.

use crate::scheduler::HaltReason;
use crate::{CognexisError, Result};

/// A schedule specifying loop counts for each token position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenwiseSchedule {
    pub loops_per_token: Vec<usize>,
}

impl TokenwiseSchedule {
    /// Create a schedule with a fixed number of loops per token.
    pub fn fixed(seq_len: usize, loops: usize) -> Self {
        Self {
            loops_per_token: vec![loops; seq_len],
        }
    }

    /// Create a schedule while clamping every token to configured bounds.
    pub fn bounded(loops_per_token: Vec<usize>, min_loops: usize, max_loops: usize) -> Self {
        let upper = max_loops.max(min_loops);
        Self {
            loops_per_token: loops_per_token
                .into_iter()
                .map(|loops| loops.clamp(min_loops, upper))
                .collect(),
        }
    }

    /// Get the number of loops for a specific token index.
    pub fn loops_for(&self, index: usize) -> usize {
        *self.loops_per_token.get(index).unwrap_or(&1)
    }

    /// Number of token positions in the schedule.
    pub fn len(&self) -> usize {
        self.loops_per_token.len()
    }

    /// Whether the schedule has no token positions.
    pub fn is_empty(&self) -> bool {
        self.loops_per_token.is_empty()
    }

    /// Return a mask indicating which tokens are still active at a
    /// zero-based loop index.
    pub fn active_mask_for_loop(&self, loop_index: usize) -> Vec<bool> {
        self.loops_per_token
            .iter()
            .map(|loops| loop_index < *loops)
            .collect()
    }
}

/// Mutable token-wise loop state for dense masked recurrence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenLoopState {
    pub active: Vec<bool>,
    pub loops: Vec<usize>,
    pub target_loops: Vec<usize>,
    pub halt_reasons: Vec<Option<HaltReason>>,
}

impl TokenLoopState {
    /// Initialize token loop state. `padding_mask` uses `true` for real
    /// tokens and `false` for PAD positions, which halt immediately.
    pub fn new(schedule: TokenwiseSchedule, padding_mask: Option<&[bool]>) -> Result<Self> {
        if let Some(mask) = padding_mask {
            if mask.len() != schedule.len() {
                return Err(CognexisError::ShapeMismatch {
                    expected: format!("padding mask length {}", schedule.len()),
                    actual: format!("padding mask length {}", mask.len()),
                });
            }
        }

        let mut active = Vec::with_capacity(schedule.len());
        let mut halt_reasons = Vec::with_capacity(schedule.len());
        for (index, &target_loops) in schedule.loops_per_token.iter().enumerate() {
            let is_padding = padding_mask.map(|mask| !mask[index]).unwrap_or(false);
            if is_padding || target_loops == 0 {
                active.push(false);
                halt_reasons.push(Some(HaltReason::Forced));
            } else {
                active.push(true);
                halt_reasons.push(None);
            }
        }

        Ok(Self {
            active,
            loops: vec![0; schedule.len()],
            target_loops: schedule.loops_per_token,
            halt_reasons,
        })
    }

    /// Return whether any token still needs recurrent updates.
    pub fn any_active(&self) -> bool {
        self.active.iter().any(|active| *active)
    }

    /// Record completion of one dense recurrent pass for active tokens.
    pub fn record_loop(&mut self) {
        for index in 0..self.active.len() {
            if !self.active[index] {
                continue;
            }

            self.loops[index] += 1;
            if self.loops[index] >= self.target_loops[index] {
                self.active[index] = false;
                self.halt_reasons[index] = Some(HaltReason::MaxLoops);
            }
        }
    }

    /// Histogram where index `i` stores how many tokens ran `i` loops.
    pub fn loop_histogram(&self) -> Vec<usize> {
        let max_loops = self.loops.iter().copied().max().unwrap_or(0);
        let mut histogram = vec![0; max_loops + 1];
        for &loops in &self.loops {
            histogram[loops] += 1;
        }
        histogram
    }
}

/// Apply candidate recurrent output only to active token positions.
pub fn apply_dense_masked_update(
    current: &[Vec<f32>],
    candidate: &[Vec<f32>],
    active_mask: &[bool],
) -> Result<Vec<Vec<f32>>> {
    if current.len() != candidate.len() || current.len() != active_mask.len() {
        return Err(CognexisError::ShapeMismatch {
            expected: format!("{} rows and active flags", current.len()),
            actual: format!(
                "{} candidate rows and {} active flags",
                candidate.len(),
                active_mask.len()
            ),
        });
    }

    current
        .iter()
        .zip(candidate)
        .zip(active_mask)
        .enumerate()
        .map(|(row_index, ((current_row, candidate_row), active))| {
            if current_row.len() != candidate_row.len() {
                return Err(CognexisError::ShapeMismatch {
                    expected: format!("row {row_index} width {}", current_row.len()),
                    actual: format!("row {row_index} width {}", candidate_row.len()),
                });
            }
            if *active {
                Ok(candidate_row.clone())
            } else {
                Ok(current_row.clone())
            }
        })
        .collect()
}
