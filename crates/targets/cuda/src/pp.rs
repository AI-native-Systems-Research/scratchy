// SPDX-License-Identifier: Apache-2.0
//! Pipeline parallelism utilities.
//!
//! Port of `vllm.distributed.utils.get_pp_indices`.

/// Compute (start_layer, end_layer) for a PP rank.
///
/// Exact port of Python's `vllm.distributed.utils.get_pp_indices`.
/// Tries to evenly distribute layers across partitions. When layers don't
/// divide evenly, remainder layers go to the **second-to-last** partitions
/// (not first or last), because:
/// - First stage has embedding overhead
/// - Last stage has lm_head + norm overhead
///
/// Returns a half-open range `[start, end)` of layer indices.
pub fn get_pp_indices(num_layers: usize, pp_rank: usize, pp_size: usize) -> (usize, usize) {
    assert!(pp_size > 0, "pp_size must be > 0");
    assert!(pp_rank < pp_size, "pp_rank {pp_rank} >= pp_size {pp_size}");
    assert!(
        num_layers >= pp_size,
        "num_layers {num_layers} < pp_size {pp_size}"
    );

    let layers_per_partition = num_layers / pp_size;
    let mut partitions = vec![layers_per_partition; pp_size];

    let remaining = num_layers % pp_size;
    if remaining > 0 {
        // Python: for i in range(2, remaining + 2): partitions[-i] += 1
        // This distributes remainder to partitions[-2], [-3], ..., [-(remaining+1)]
        // i.e., middle partitions (excluding last, and first when possible).
        for i in 2..remaining + 2 {
            partitions[pp_size - i] += 1;
        }
    }

    let start = partitions[..pp_rank].iter().sum::<usize>();
    let end = start + partitions[pp_rank];
    (start, end)
}

/// PP configuration passed to model loading.
#[derive(Debug, Clone, Copy)]
pub struct PpConfig {
    pub pp_rank: usize,
    pub pp_size: usize,
    /// First layer index this stage is responsible for (inclusive).
    pub start_layer: usize,
    /// Last layer index this stage is responsible for (exclusive).
    pub end_layer: usize,
}

impl PpConfig {
    /// Create a PpConfig from model parameters.
    pub fn new(num_layers: usize, pp_rank: usize, pp_size: usize) -> Self {
        let (start_layer, end_layer) = get_pp_indices(num_layers, pp_rank, pp_size);
        Self {
            pp_rank,
            pp_size,
            start_layer,
            end_layer,
        }
    }

    /// Whether this is the first PP stage (responsible for embeddings).
    pub fn is_first_stage(&self) -> bool {
        self.pp_rank == 0
    }

    /// Whether this is the last PP stage (responsible for lm_head + norm).
    pub fn is_last_stage(&self) -> bool {
        self.pp_rank == self.pp_size - 1
    }

    /// Number of layers on this stage.
    pub fn num_layers(&self) -> usize {
        self.end_layer - self.start_layer
    }

    /// No-op config for single-GPU (pp_size=1).
    pub fn single() -> Self {
        // Placeholder — caller should use PpConfig::new() with real num_layers.
        Self {
            pp_rank: 0,
            pp_size: 1,
            start_layer: 0,
            end_layer: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pp_size_1() {
        // PP=1: all layers on single stage.
        assert_eq!(get_pp_indices(24, 0, 1), (0, 24));
        assert_eq!(get_pp_indices(32, 0, 1), (0, 32));
        assert_eq!(get_pp_indices(1, 0, 1), (0, 1));
    }

    #[test]
    fn test_even_split() {
        // 24 layers, PP=2 → 12/12
        assert_eq!(get_pp_indices(24, 0, 2), (0, 12));
        assert_eq!(get_pp_indices(24, 1, 2), (12, 24));

        // 24 layers, PP=4 → 6/6/6/6
        assert_eq!(get_pp_indices(24, 0, 4), (0, 6));
        assert_eq!(get_pp_indices(24, 1, 4), (6, 12));
        assert_eq!(get_pp_indices(24, 2, 4), (12, 18));
        assert_eq!(get_pp_indices(24, 3, 4), (18, 24));
    }

    #[test]
    fn test_remainder_pp2() {
        // 25 layers, PP=2 → remainder=1
        // Python: partitions = [12, 12], then for i in 2..3: partitions[-2] += 1
        // → [13, 12]
        assert_eq!(get_pp_indices(25, 0, 2), (0, 13));
        assert_eq!(get_pp_indices(25, 1, 2), (13, 25));
    }

    #[test]
    fn test_remainder_pp3() {
        // 25 layers, PP=3 → base=8, remainder=1
        // partitions = [8, 8, 8], for i in 2..3: partitions[-2] += 1
        // → [8, 9, 8]
        assert_eq!(get_pp_indices(25, 0, 3), (0, 8));
        assert_eq!(get_pp_indices(25, 1, 3), (8, 17));
        assert_eq!(get_pp_indices(25, 2, 3), (17, 25));
    }

    #[test]
    fn test_remainder_pp4() {
        // 26 layers, PP=4 → base=6, remainder=2
        // partitions = [6, 6, 6, 6], for i in 2..4: partitions[-2] += 1, partitions[-3] += 1
        // → [6, 7, 7, 6]
        assert_eq!(get_pp_indices(26, 0, 4), (0, 6));
        assert_eq!(get_pp_indices(26, 1, 4), (6, 13));
        assert_eq!(get_pp_indices(26, 2, 4), (13, 20));
        assert_eq!(get_pp_indices(26, 3, 4), (20, 26));
    }

    #[test]
    fn test_remainder_pp4_three_extra() {
        // 27 layers, PP=4 → base=6, remainder=3
        // partitions = [6, 6, 6, 6], for i in 2..5: [-2] += 1, [-3] += 1, [-4] += 1
        // → [7, 7, 7, 6]
        assert_eq!(get_pp_indices(27, 0, 4), (0, 7));
        assert_eq!(get_pp_indices(27, 1, 4), (7, 14));
        assert_eq!(get_pp_indices(27, 2, 4), (14, 21));
        assert_eq!(get_pp_indices(27, 3, 4), (21, 27));
    }

    #[test]
    fn test_all_layers_covered() {
        // Verify all layers are covered (no gaps, no overlaps) for many combos.
        for num_layers in 1..=48 {
            for pp_size in 1..=num_layers.min(8) {
                let mut prev_end = 0;
                for pp_rank in 0..pp_size {
                    let (start, end) = get_pp_indices(num_layers, pp_rank, pp_size);
                    assert_eq!(
                        start, prev_end,
                        "gap at num_layers={num_layers}, pp_size={pp_size}, pp_rank={pp_rank}"
                    );
                    assert!(
                        end > start,
                        "empty partition at num_layers={num_layers}, pp_size={pp_size}, pp_rank={pp_rank}"
                    );
                    prev_end = end;
                }
                assert_eq!(
                    prev_end, num_layers,
                    "not all layers covered: num_layers={num_layers}, pp_size={pp_size}"
                );
            }
        }
    }

    #[test]
    fn test_last_partition_not_largest() {
        // Remainder goes to middle partitions, not last.
        // For PP=4 with remainder, last partition should be <= first partition.
        for num_layers in 25..=31 {
            let pp_size = 4;
            let (_, end_first) = get_pp_indices(num_layers, 0, pp_size);
            let first_count = end_first;
            let (start_last, end_last) = get_pp_indices(num_layers, pp_size - 1, pp_size);
            let last_count = end_last - start_last;
            assert!(
                last_count <= first_count,
                "last partition larger than first: num_layers={num_layers}, first={first_count}, last={last_count}"
            );
        }
    }

    #[test]
    fn test_pp_config() {
        let pp = PpConfig::new(24, 0, 2);
        assert!(pp.is_first_stage());
        assert!(!pp.is_last_stage());
        assert_eq!(pp.num_layers(), 12);
        assert_eq!(pp.start_layer, 0);
        assert_eq!(pp.end_layer, 12);

        let pp = PpConfig::new(24, 1, 2);
        assert!(!pp.is_first_stage());
        assert!(pp.is_last_stage());
        assert_eq!(pp.num_layers(), 12);
        assert_eq!(pp.start_layer, 12);
        assert_eq!(pp.end_layer, 24);
    }
}
