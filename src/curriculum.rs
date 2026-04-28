//! Curriculum scheduling module.
//!
//! Curriculum learning gradually increases task difficulty or loop
//! counts during training to improve stability. See
//! `spec13_curriculum.md` for details on loop schedules and staged
//! training.

use serde::{Deserialize, Serialize};

use crate::{CognexisError, Result};

/// Defines a curriculum stage controlling loop counts during training.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CurriculumStage {
    /// Maximum number of loops allowed in this stage.
    pub max_loops: usize,
    /// Number of training steps for this stage.
    pub steps: usize,
}

/// Curriculum schedule consisting of multiple stages.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Curriculum {
    pub stages: Vec<CurriculumStage>,
}

impl Curriculum {
    /// Get the current stage based on training step.
    pub fn current_stage(&self, step: usize) -> Option<&CurriculumStage> {
        let mut accumulated = 0;
        for stage in &self.stages {
            accumulated += stage.steps;
            if step < accumulated {
                return Some(stage);
            }
        }
        None
    }
}

/// Ramp used to increase the sampled depth after warm-up.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RampKind {
    Linear,
    Cosine,
    Exponential,
}

/// Training-time loop sampling distribution.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SamplingDistribution {
    Fixed,
    Uniform,
    Geometric,
}

/// Configuration for checkpointable recurrent-depth sampling.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoopCurriculumConfig {
    pub min_loops: usize,
    pub initial_max_loops: usize,
    pub target_max_loops: usize,
    pub warmup_steps: u64,
    pub ramp_steps: u64,
    pub ramp: RampKind,
    pub distribution: SamplingDistribution,
    pub high_depth_fraction: f32,
    pub retain_intermediate: bool,
    pub seed: u64,
}

impl Default for LoopCurriculumConfig {
    fn default() -> Self {
        Self {
            min_loops: 1,
            initial_max_loops: 2,
            target_max_loops: 8,
            warmup_steps: 0,
            ramp_steps: 10_000,
            ramp: RampKind::Linear,
            distribution: SamplingDistribution::Uniform,
            high_depth_fraction: 0.0,
            retain_intermediate: false,
            seed: 0xC06E_5175,
        }
    }
}

impl LoopCurriculumConfig {
    pub fn validate(&self, architecture_max_loops: usize) -> Result<()> {
        if self.min_loops == 0 {
            return Err(CognexisError::InvalidConfig(
                "curriculum min_loops must be at least 1".to_string(),
            ));
        }
        if self.initial_max_loops < self.min_loops {
            return Err(CognexisError::InvalidConfig(format!(
                "initial_max_loops ({}) must be >= min_loops ({})",
                self.initial_max_loops, self.min_loops
            )));
        }
        if self.target_max_loops < self.initial_max_loops {
            return Err(CognexisError::InvalidConfig(format!(
                "target_max_loops ({}) must be >= initial_max_loops ({})",
                self.target_max_loops, self.initial_max_loops
            )));
        }
        if self.target_max_loops > architecture_max_loops {
            return Err(CognexisError::InvalidConfig(format!(
                "target_max_loops ({}) exceeds architecture max_loop_count ({})",
                self.target_max_loops, architecture_max_loops
            )));
        }
        if !self.high_depth_fraction.is_finite()
            || self.high_depth_fraction < 0.0
            || self.high_depth_fraction > 1.0
        {
            return Err(CognexisError::InvalidConfig(
                "high_depth_fraction must be in [0, 1]".to_string(),
            ));
        }
        Ok(())
    }
}

/// Sample emitted to the training loop for a synchronized global step.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct LoopSample {
    pub min_loops: usize,
    pub max_loops: usize,
    pub sampled_loops: usize,
    pub retain_intermediate: bool,
}

/// Serializable sampler state for checkpoint/resume.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct CurriculumState {
    pub rng_state: u64,
    pub samples_drawn: u64,
    pub last_step: u64,
    pub current_max_loops: usize,
}

/// Deterministic loop curriculum sampler.
#[derive(Debug, Clone)]
pub struct LoopCurriculumSampler {
    config: LoopCurriculumConfig,
    state: CurriculumState,
}

impl LoopCurriculumSampler {
    pub fn new(config: LoopCurriculumConfig, architecture_max_loops: usize) -> Result<Self> {
        config.validate(architecture_max_loops)?;
        Ok(Self {
            state: CurriculumState {
                rng_state: config.seed,
                samples_drawn: 0,
                last_step: 0,
                current_max_loops: config.initial_max_loops,
            },
            config,
        })
    }

    pub fn sample(&mut self, step: u64) -> LoopSample {
        let current_max = self.current_max_for_step(step);
        self.state.current_max_loops = current_max;
        self.state.last_step = step;

        let sampled_loops = if self.should_take_high_depth_sample(current_max) {
            self.sample_uniform(current_max.saturating_add(1), self.config.target_max_loops)
        } else {
            match self.config.distribution {
                SamplingDistribution::Fixed => current_max,
                SamplingDistribution::Uniform => {
                    self.sample_uniform(self.config.min_loops, current_max)
                }
                SamplingDistribution::Geometric => {
                    self.sample_geometric(self.config.min_loops, current_max)
                }
            }
        };

        self.state.samples_drawn += 1;
        LoopSample {
            min_loops: self.config.min_loops,
            max_loops: current_max,
            sampled_loops,
            retain_intermediate: self.config.retain_intermediate,
        }
    }

    pub fn state_dict(&self) -> CurriculumState {
        self.state
    }

    pub fn load_state_dict(&mut self, state: CurriculumState) -> Result<()> {
        if state.current_max_loops < self.config.min_loops
            || state.current_max_loops > self.config.target_max_loops
        {
            return Err(CognexisError::InvalidConfig(format!(
                "invalid curriculum current_max_loops {} for range {}..={}",
                state.current_max_loops, self.config.min_loops, self.config.target_max_loops
            )));
        }
        self.state = state;
        Ok(())
    }

    pub fn config(&self) -> &LoopCurriculumConfig {
        &self.config
    }

    fn current_max_for_step(&self, step: u64) -> usize {
        if step < self.config.warmup_steps {
            return self.config.initial_max_loops;
        }
        if self.config.ramp_steps == 0 {
            return self.config.target_max_loops;
        }

        let elapsed = step - self.config.warmup_steps;
        let progress = (elapsed as f64 / self.config.ramp_steps as f64).clamp(0.0, 1.0);
        let ramp_progress = match self.config.ramp {
            RampKind::Linear => progress,
            RampKind::Cosine => 0.5 - 0.5 * (std::f64::consts::PI * progress).cos(),
            RampKind::Exponential => {
                if progress == 0.0 {
                    0.0
                } else {
                    (2_f64.powf(6.0 * progress) - 1.0) / (2_f64.powf(6.0) - 1.0)
                }
            }
        };
        let span = self.config.target_max_loops - self.config.initial_max_loops;
        let offset = (span as f64 * ramp_progress).floor() as usize;
        (self.config.initial_max_loops + offset).min(self.config.target_max_loops)
    }

    fn should_take_high_depth_sample(&mut self, current_max: usize) -> bool {
        current_max < self.config.target_max_loops
            && self.config.high_depth_fraction > 0.0
            && self.next_unit_f32() < self.config.high_depth_fraction
    }

    fn sample_uniform(&mut self, min_loops: usize, max_loops: usize) -> usize {
        if max_loops <= min_loops {
            return min_loops;
        }
        let width = max_loops - min_loops + 1;
        min_loops + (self.next_u64() as usize % width)
    }

    fn sample_geometric(&mut self, min_loops: usize, max_loops: usize) -> usize {
        let mut loops = min_loops;
        while loops < max_loops && self.next_unit_f32() < 0.5 {
            loops += 1;
        }
        loops
    }

    fn next_unit_f32(&mut self) -> f32 {
        let value = self.next_u64() >> 40;
        value as f32 / (1u64 << 24) as f32
    }

    fn next_u64(&mut self) -> u64 {
        self.state.rng_state = self
            .state
            .rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state.rng_state
    }
}
