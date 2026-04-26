//! Data loading module.
//!
//! This module provides structures and stubs for loading and batching
//! training data. In a real implementation this would include file
//! readers, preprocessing pipelines, and efficient batching. See
//! `spec12_data_loading.md` for guidelines on memory mapping and
//! streaming large datasets.

/// A single training example consisting of input and target token IDs.
pub struct TrainingExample {
    pub input_ids: Vec<u32>,
    pub target_ids: Vec<u32>,
}

/// Data loader stub.
pub struct DataLoader {
    /// Collection of all training examples.
    pub examples: Vec<TrainingExample>,
    /// Batch size used for iteration.
    pub batch_size: usize,
    pub position: usize,
}

impl DataLoader {
    /// Create a new data loader from a list of examples and a batch size.
    pub fn new(examples: Vec<TrainingExample>, batch_size: usize) -> Self {
        Self {
            examples,
            batch_size,
            position: 0,
        }
    }

    /// Fetch the next batch of examples. Returns `None` when no more
    /// batches are available.
    pub fn next_batch(&mut self) -> Option<Vec<TrainingExample>> {
        if self.position >= self.examples.len() {
            return None;
        }
        let end = (self.position + self.batch_size).min(self.examples.len());
        let batch = self.examples[self.position..end].to_vec();
        self.position = end;
        Some(batch)
    }
}