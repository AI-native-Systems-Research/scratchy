// SPDX-License-Identifier: Apache-2.0
//! The worker's KV pool rule on lowered Gemma tapes, with no CUDA toolchain: the tapes one worker
//! is handed share one pool exactly when `KvTape::same_pool` says so (`CudaWorker::new`'s
//! `kv_geometry`, which is nvcc-only), and the per-class layout is then the same for each.

// `pub`: each test binary uses a different part of `common`, and nothing in it is dead code.
pub mod common {
    pub mod chains;
    pub mod gemma_attn;
}

use common::gemma_attn::{GEMMA_ATTN_FACTS, gemma_attn, gemma_attn_global, gemma_attn_two_classes};
use scratchy_subtile::handoff::LoweredDecode;
use scratchy_target_cuda::tape::{CudaTape, KvClass, KvTape};
use scratchy_target_cuda_compiler::canonical::{CanonicalAt, lower_canonical};
use scratchy_target_cuda_compiler::emit::static_tape;

/// `lower(m)` at m = 1 and 512 as CUDA's compiler lowers it.
fn lowered(lower: fn(u32) -> LoweredDecode) -> Vec<CudaTape> {
    [1u32, 512]
        .iter()
        .map(|&m| {
            let at = CanonicalAt {
                stem: "kv_pool",
                m: u64::from(m),
            };
            let c = lower_canonical(&lower(m), GEMMA_ATTN_FACTS, at)
                .unwrap_or_else(|e| panic!("lowers at m={m}: {e}"));
            static_tape(&c)
        })
        .collect()
}

/// `t` with every KV class unwindowed, as `tape_player`'s `unwindowed` makes #922's twins.
fn unwindowed(t: &CudaTape) -> CudaTape {
    let kv = t.kv.map(|kv| {
        let classes: Vec<KvClass> = (kv.classes.iter())
            .map(|c| KvClass { window: None, ..*c })
            .collect();
        KvTape {
            classes: Box::leak(classes.into_boxed_slice()),
            ..kv
        }
    });
    CudaTape { kv, ..*t }
}

/// #923's GKE regression (`KvGeometry { bucket: 2 }` in #922's sliding-layer tests), host side:
/// the sliding layer at m = 1 and 512 and their unwindowed twins (`gemma_attn_buckets`' buckets
/// 0 to 3) share one pool, with one layout. The twins differ only in the class's window, which
/// lays out no cache.
#[test]
fn gemmas_sliding_layer_and_its_unwindowed_twin_share_one_pool() {
    let tapes = lowered(gemma_attn);
    let all: Vec<CudaTape> = (tapes.iter().cloned())
        .chain(tapes.iter().map(unwindowed))
        .collect();
    let first = all[0].kv.expect("a KV tape");
    for (b, t) in all.iter().enumerate() {
        let kv = t.kv.expect("a KV tape");
        assert!(first.same_pool(&kv), "bucket {b}: {kv:?}");
        assert_eq!(kv.pool_layout(48), first.pool_layout(48), "bucket {b}");
    }
}

/// The global layer alone and the two-class chain share a pool across their own buckets, and
/// neither shares one with the other, nor with the sliding layer: their layers' classes differ.
#[test]
fn the_global_and_two_class_tapes_share_a_pool_only_with_their_own_buckets() {
    let sets = [
        lowered(gemma_attn),
        lowered(gemma_attn_global),
        lowered(gemma_attn_two_classes),
    ];
    for (i, a) in sets.iter().enumerate() {
        for (j, b) in sets.iter().enumerate() {
            for (ta, tb) in a.iter().zip(b) {
                let (ka, kb) = (ta.kv.expect("a KV tape"), tb.kv.expect("a KV tape"));
                assert_eq!(ka.same_pool(&kb), i == j, "sets {i} and {j}");
            }
        }
    }
    let two = sets[2][0].kv.expect("a KV tape");
    let (_, bytes) = two.pool_layout(48).expect("the pool fits");
    assert_eq!(bytes, (8 * 256 + 2 * 512) * 2 * 2 * 48 * 64);
}
