// SPDX-License-Identifier: Apache-2.0
//! ⭐ THE BATCH MASK's ROW LAW, at granite-3.1-8b's hd=128 geometry — the host's mask block is the
//! group block's `gqa·B` rows, and row `g·B + r` of every kv head's block is request `r`, the
//! re-laid query's destination order.
//!
//! (The emitter-side half of this file — the per-op counts and shapes of the group-major chunk —
//! was pinned through the deleted hand-written attention emitter and went with it; when the live
//! spliced kernels need the same pin, mint it from the splice, not from a copy of the old emitter.)

use ktir_superdsc::sdsc_abstract::{BatchChunk, ChunkScratch, PagedKvPool};

const NQH: u32 = 32;
const NKVH: u32 = 8;
const GQA: u32 = NQH / NKVH;
const HD: u32 = 128;

#[test]
fn the_batch_mask_rows_are_the_relaid_group_rows() {
    let pool = PagedKvPool::new(NKVH, HD);
    for seqs in PagedKvPool::BATCH_RUNGS {
        let t = ChunkScratch::ladder_for(pool, seqs)[0];
        let bc = BatchChunk::of(ChunkScratch::of_chunk(pool, t).unwrap(), seqs, GQA).unwrap();
        assert_eq!(bc.mask_rows(), GQA * seqs, "seqs={seqs}");
        for g in 0..GQA {
            for r in 0..seqs {
                assert_eq!(
                    bc.row_request(g * seqs + r),
                    r,
                    "seqs={seqs} head {g} request {r}"
                );
            }
        }
    }
}
