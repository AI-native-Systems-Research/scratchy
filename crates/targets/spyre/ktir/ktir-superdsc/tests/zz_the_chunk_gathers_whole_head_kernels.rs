// SPDX-License-Identifier: Apache-2.0
//! The split-KV chunk's head-dim admission. (Its gather-layout pins — the entry orders the deleted
//! copy-gather composers shared with the host's index table — went with those composers.)

use ktir_superdsc::sdsc_abstract::{ChunkScratch, PagedKvPool};

const NKVH: usize = 8;

/// A head dim whose stick blocks straddle V's slabs has no slab-major order, so the chunk is refused —
/// the model keeps the per-page fold, as a build-time fact.
#[test]
fn a_head_dim_past_the_page_is_refused() {
    assert!(ChunkScratch::of_chunk(PagedKvPool::new(NKVH as u32, 512), 2).is_none());
}
