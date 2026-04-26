//! Curriculum scheduling module.
//!
//! Curriculum learning gradually increases task difficulty or loop
//! counts during training to improve stability. See
//! `spec13_curriculum.md` for details on loop schedules and staged
//! training.

/// Defines a curriculum stage controlling loop counts during training.
#[derive(Debug, Clone)]
pub struct CurriculumStage {
    /// Maximum number of loops allowed in this stage.
    pub max_loops: usize,
    /// Number of training steps for this stage.
    pub steps: usize,
}

/// Curriculum schedule consisting of multiple stages.
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
