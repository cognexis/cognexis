//! Loop scaling module.
//!
//! Routines for studying the effect of loop counts on performance and
//! compute cost. See `spec21_loop_scaling.md` for details on how to
//! run depth scaling experiments and plot accuracy vs. loops curves.

use serde::{Deserialize, Serialize};

use crate::evaluation::{
    depth_efficiency_between, loop_saturation_point, overthinking_threshold, DepthPoint,
    MetricDirection,
};
use crate::{CognexisError, Result};

/// Summary of a loop-scaling sweep for one metric/task pair.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoopScalingSummary {
    pub depths: Vec<usize>,
    pub adjacent_dei: Vec<f64>,
    pub loop_saturation_point: Option<usize>,
    pub overthinking_threshold: Option<usize>,
}

/// Generate a depth grid by doubling and always including `max_loops`.
pub fn loop_schedule(max_loops: usize) -> Vec<usize> {
    if max_loops == 0 {
        return Vec::new();
    }

    let mut schedule = Vec::new();
    let mut n = 1;
    while n <= max_loops {
        schedule.push(n);
        n = n.saturating_mul(2);
        if n == 0 {
            break;
        }
    }
    if schedule.last().copied() != Some(max_loops) {
        schedule.push(max_loops);
    }
    schedule
}

/// Parse a comma-separated loop-depth grid such as `1,2,4,8`.
pub fn parse_depth_grid(input: &str) -> Result<Vec<usize>> {
    let mut depths = Vec::new();
    for part in input.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let depth = part.parse::<usize>().map_err(|error| {
            CognexisError::InvalidConfig(format!("invalid loop depth {part:?}: {error}"))
        })?;
        if depth == 0 {
            return Err(CognexisError::InvalidConfig(
                "loop depths must be positive".to_string(),
            ));
        }
        if !depths.contains(&depth) {
            depths.push(depth);
        }
    }
    if depths.is_empty() {
        return Err(CognexisError::InvalidConfig(
            "depth grid must contain at least one depth".to_string(),
        ));
    }
    depths.sort_unstable();
    Ok(depths)
}

/// Build loop-scaling summary metrics from fixed-depth result points.
pub fn summarize_loop_scaling(
    points: &[DepthPoint],
    direction: MetricDirection,
    saturation_threshold: f64,
    overthinking_tolerance: f64,
) -> Result<LoopScalingSummary> {
    if points.len() < 2 {
        return Err(CognexisError::InvalidConfig(
            "loop scaling summary requires at least two depth points".to_string(),
        ));
    }

    let mut points = points.to_vec();
    points.sort_by_key(|point| point.loops);
    let mut adjacent_dei = Vec::with_capacity(points.len().saturating_sub(1));
    for window in points.windows(2) {
        adjacent_dei.push(
            depth_efficiency_between(window[0], window[1], direction).ok_or_else(|| {
                CognexisError::InvalidConfig(
                    "loop scaling points must have increasing finite compute".to_string(),
                )
            })?,
        );
    }

    Ok(LoopScalingSummary {
        depths: points.iter().map(|point| point.loops).collect(),
        adjacent_dei,
        loop_saturation_point: loop_saturation_point(&points, saturation_threshold, direction),
        overthinking_threshold: overthinking_threshold(&points, overthinking_tolerance, direction),
    })
}
