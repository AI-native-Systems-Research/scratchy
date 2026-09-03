// SPDX-License-Identifier: Apache-2.0
//! Process group trait for distributed collective communication.
//!
//! Defines the `ProcessGroup` trait used by tensor-parallel workers
//! to broadcast config/metadata across ranks during init.
//!
//! GPU-level collectives (all-reduce, all-gather) live in the backend
//! crates (e.g. `NcclGroup` in scratchy-serving-cuda) operating on native tensor types.

/// Abstraction over collective communication for multi-process init.
///
/// Workers hold an `Option<Arc<dyn ProcessGroup>>` — when `None` (TP=1),
/// no communication is performed.
pub trait ProcessGroup: Send + Sync + std::fmt::Debug {
    /// This rank's index (0-based).
    fn rank(&self) -> usize;

    /// Total number of ranks.
    fn world_size(&self) -> usize;

    /// Broadcast a byte buffer from `root` to all ranks (collective).
    /// On root: sends `data`. On non-root: returns received data.
    /// Default: no-op passthrough (single-node).
    fn broadcast_bytes(&self, data: &[u8], _root: usize) -> Result<Vec<u8>, String> {
        Ok(data.to_vec())
    }
}
