// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Distributed parallel state types.
//!
//! Manages tensor-parallel (TP), pipeline-parallel (PP), and data-parallel (DP)
//! group configurations for distributed inference.
//!
//! Port of: `vllm/distributed/parallel_state.py` (types only, not runtime state)

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// ParallelGroup
// ---------------------------------------------------------------------------

/// A distributed communication group.
///
/// Represents a subset of workers that participate in a collective operation
/// (e.g., all-reduce for tensor parallelism, point-to-point for pipeline
/// parallelism).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelGroup {
    /// Name of this group (e.g., "tp", "pp", "dp").
    pub name: String,
    /// Total number of ranks in this group.
    pub world_size: usize,
    /// This process's rank within the group.
    pub rank_in_group: usize,
    /// Global ranks of all members in this group.
    pub ranks: Vec<usize>,
}

impl ParallelGroup {
    /// Create a new parallel group.
    pub fn new(name: impl Into<String>, world_size: usize, rank_in_group: usize) -> Self {
        let ranks = (0..world_size).collect();
        Self {
            name: name.into(),
            world_size,
            rank_in_group,
            ranks,
        }
    }

    /// Create a group with explicit rank mapping.
    pub fn with_ranks(
        name: impl Into<String>,
        world_size: usize,
        rank_in_group: usize,
        ranks: Vec<usize>,
    ) -> Self {
        assert_eq!(ranks.len(), world_size);
        Self {
            name: name.into(),
            world_size,
            rank_in_group,
            ranks,
        }
    }

    /// Whether this is a single-rank (non-distributed) group.
    pub fn is_single(&self) -> bool {
        self.world_size == 1
    }

    /// The global rank of this process in this group.
    pub fn global_rank(&self) -> usize {
        self.ranks[self.rank_in_group]
    }

    /// Whether this process is the first rank in the group.
    pub fn is_first_rank(&self) -> bool {
        self.rank_in_group == 0
    }

    /// Whether this process is the last rank in the group.
    pub fn is_last_rank(&self) -> bool {
        self.rank_in_group == self.world_size - 1
    }

    /// The next rank in the group (wraps around).
    pub fn next_rank(&self) -> usize {
        self.ranks[(self.rank_in_group + 1) % self.world_size]
    }

    /// The previous rank in the group (wraps around).
    pub fn prev_rank(&self) -> usize {
        self.ranks[(self.rank_in_group + self.world_size - 1) % self.world_size]
    }
}

// ---------------------------------------------------------------------------
// ParallelConfig (runtime-resolved)
// ---------------------------------------------------------------------------

/// Resolved parallel configuration for a worker.
///
/// This is the runtime representation of the parallel layout, computed
/// from the user-specified `ParallelConfig` after determining the actual
/// number of nodes and devices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedParallelConfig {
    /// Total number of workers in the world.
    pub world_size: usize,
    /// This worker's global rank.
    pub rank: usize,
    /// This worker's local rank (device index on this node).
    pub local_rank: usize,
    /// Number of nodes.
    pub num_nodes: usize,
    /// Tensor-parallel group.
    pub tp_group: ParallelGroup,
    /// Pipeline-parallel group.
    pub pp_group: ParallelGroup,
    /// Data-parallel group.
    pub dp_group: ParallelGroup,
}

impl ResolvedParallelConfig {
    /// Create a single-GPU (non-distributed) configuration.
    pub fn single_gpu() -> Self {
        Self {
            world_size: 1,
            rank: 0,
            local_rank: 0,
            num_nodes: 1,
            tp_group: ParallelGroup::new("tp", 1, 0),
            pp_group: ParallelGroup::new("pp", 1, 0),
            dp_group: ParallelGroup::new("dp", 1, 0),
        }
    }

    /// Create a configuration for tensor parallelism only.
    pub fn tensor_parallel(tp_size: usize, rank: usize) -> Self {
        Self {
            world_size: tp_size,
            rank,
            local_rank: rank,
            num_nodes: 1,
            tp_group: ParallelGroup::new("tp", tp_size, rank),
            pp_group: ParallelGroup::new("pp", 1, 0),
            dp_group: ParallelGroup::new("dp", 1, 0),
        }
    }

    /// Create a configuration for tensor + pipeline parallelism.
    ///
    /// Workers are laid out as: [TP0_PP0, TP1_PP0, ..., TP0_PP1, TP1_PP1, ...]
    /// So for TP=2, PP=2 with 4 workers:
    ///   rank 0: TP=0, PP=0 (first TP rank of first PP stage)
    ///   rank 1: TP=1, PP=0 (second TP rank of first PP stage)
    ///   rank 2: TP=0, PP=1 (first TP rank of second PP stage)
    ///   rank 3: TP=1, PP=1 (second TP rank of second PP stage)
    pub fn tensor_pipeline_parallel(tp_size: usize, pp_size: usize, rank: usize) -> Self {
        let world_size = tp_size * pp_size;
        assert!(rank < world_size, "rank {rank} >= world_size {world_size}");

        let tp_rank = rank % tp_size;
        let pp_rank = rank / tp_size;

        // TP group: all ranks in the same PP stage.
        let tp_ranks: Vec<usize> = (0..tp_size).map(|tp| pp_rank * tp_size + tp).collect();

        // PP group: all ranks with the same TP rank.
        let pp_ranks: Vec<usize> = (0..pp_size).map(|pp| pp * tp_size + tp_rank).collect();

        Self {
            world_size,
            rank,
            local_rank: rank,
            num_nodes: 1,
            tp_group: ParallelGroup::with_ranks("tp", tp_size, tp_rank, tp_ranks),
            pp_group: ParallelGroup::with_ranks("pp", pp_size, pp_rank, pp_ranks),
            dp_group: ParallelGroup::new("dp", 1, 0),
        }
    }

    /// Whether this is the driver rank (TP rank 0 of the first PP stage).
    pub fn is_driver(&self) -> bool {
        self.tp_group.is_first_rank() && self.pp_group.is_first_rank()
    }

    /// Whether this is the output rank (TP rank 0 of the last PP stage).
    ///
    /// Only this rank returns `ModelRunnerOutput` in pipeline-parallel mode.
    pub fn is_output_rank(&self) -> bool {
        self.tp_group.is_first_rank() && self.pp_group.is_last_rank()
    }

    /// Compute the output rank (first TP rank of last PP stage).
    pub fn output_rank(&self) -> usize {
        let tp_size = self.tp_group.world_size;
        let pp_size = self.pp_group.world_size;
        (pp_size - 1) * tp_size
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_group_single() {
        let group = ParallelGroup::new("tp", 1, 0);
        assert!(group.is_single());
        assert!(group.is_first_rank());
        assert!(group.is_last_rank());
        assert_eq!(group.global_rank(), 0);
    }

    #[test]
    fn test_parallel_group_multi() {
        let group = ParallelGroup::new("tp", 4, 1);
        assert!(!group.is_single());
        assert!(!group.is_first_rank());
        assert!(!group.is_last_rank());
        assert_eq!(group.global_rank(), 1);
        assert_eq!(group.next_rank(), 2);
        assert_eq!(group.prev_rank(), 0);
    }

    #[test]
    fn test_parallel_group_wrap_around() {
        let group = ParallelGroup::new("pp", 3, 2);
        assert!(group.is_last_rank());
        assert_eq!(group.next_rank(), 0); // wraps
        assert_eq!(group.prev_rank(), 1);
    }

    #[test]
    fn test_parallel_group_custom_ranks() {
        let group = ParallelGroup::with_ranks("tp", 2, 0, vec![0, 2]);
        assert_eq!(group.global_rank(), 0);
        assert_eq!(group.next_rank(), 2);
    }

    #[test]
    fn test_single_gpu_config() {
        let config = ResolvedParallelConfig::single_gpu();
        assert_eq!(config.world_size, 1);
        assert_eq!(config.rank, 0);
        assert!(config.is_driver());
        assert!(config.is_output_rank());
        assert_eq!(config.output_rank(), 0);
    }

    #[test]
    fn test_tensor_parallel_config() {
        // TP=4, rank 0
        let config0 = ResolvedParallelConfig::tensor_parallel(4, 0);
        assert_eq!(config0.world_size, 4);
        assert!(config0.is_driver());
        assert!(config0.is_output_rank());
        assert_eq!(config0.tp_group.world_size, 4);
        assert_eq!(config0.tp_group.rank_in_group, 0);
        assert_eq!(config0.pp_group.world_size, 1);

        // TP=4, rank 2
        let config2 = ResolvedParallelConfig::tensor_parallel(4, 2);
        assert!(!config2.is_driver());
        assert!(!config2.is_output_rank());
        assert_eq!(config2.tp_group.rank_in_group, 2);
    }

    #[test]
    fn test_tp_pp_parallel_config() {
        // TP=2, PP=2 → 4 workers
        // rank 0: TP0_PP0 (driver)
        // rank 1: TP1_PP0
        // rank 2: TP0_PP1 (output rank)
        // rank 3: TP1_PP1

        let c0 = ResolvedParallelConfig::tensor_pipeline_parallel(2, 2, 0);
        assert_eq!(c0.world_size, 4);
        assert_eq!(c0.tp_group.world_size, 2);
        assert_eq!(c0.tp_group.rank_in_group, 0);
        assert_eq!(c0.pp_group.world_size, 2);
        assert_eq!(c0.pp_group.rank_in_group, 0);
        assert!(c0.is_driver());
        assert!(!c0.is_output_rank());

        // TP group for PP stage 0 = {0, 1}
        assert_eq!(c0.tp_group.ranks, vec![0, 1]);
        // PP group for TP rank 0 = {0, 2}
        assert_eq!(c0.pp_group.ranks, vec![0, 2]);

        let c2 = ResolvedParallelConfig::tensor_pipeline_parallel(2, 2, 2);
        assert_eq!(c2.tp_group.rank_in_group, 0);
        assert_eq!(c2.pp_group.rank_in_group, 1);
        assert!(!c2.is_driver());
        assert!(c2.is_output_rank());
        assert_eq!(c2.output_rank(), 2);

        // TP group for PP stage 1 = {2, 3}
        assert_eq!(c2.tp_group.ranks, vec![2, 3]);

        let c3 = ResolvedParallelConfig::tensor_pipeline_parallel(2, 2, 3);
        assert_eq!(c3.tp_group.rank_in_group, 1);
        assert_eq!(c3.pp_group.rank_in_group, 1);
        assert!(!c3.is_driver());
        assert!(!c3.is_output_rank());
    }

    #[test]
    fn test_tp_pp_output_rank() {
        // TP=8, PP=4 → 32 workers
        // Output rank = (4-1) * 8 = 24
        let c = ResolvedParallelConfig::tensor_pipeline_parallel(8, 4, 0);
        assert_eq!(c.output_rank(), 24);

        let c_out = ResolvedParallelConfig::tensor_pipeline_parallel(8, 4, 24);
        assert!(c_out.is_output_rank());
    }

    #[test]
    fn test_parallel_group_serde() {
        let group = ParallelGroup::new("tp", 4, 1);
        let json = serde_json::to_string(&group).unwrap();
        let deserialized: ParallelGroup = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "tp");
        assert_eq!(deserialized.world_size, 4);
        assert_eq!(deserialized.rank_in_group, 1);
    }

    #[test]
    fn test_resolved_config_serde() {
        let config = ResolvedParallelConfig::tensor_pipeline_parallel(2, 2, 1);
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ResolvedParallelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.world_size, 4);
        assert_eq!(deserialized.rank, 1);
        assert_eq!(deserialized.tp_group.rank_in_group, 1);
        assert_eq!(deserialized.pp_group.rank_in_group, 0);
    }
}
