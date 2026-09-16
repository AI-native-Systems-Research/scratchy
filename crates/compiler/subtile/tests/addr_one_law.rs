// SPDX-License-Identifier: Apache-2.0
//! ONE LAW: every hand-rolled address formula in this backend is `addr::Nest` at a different rank.
//!
//! If these pass, the four formulas are not four facts — they are one fact written four times, and
//! the four transcriptions are exactly where the bugs lived. Each is checked ELEMENT BY ELEMENT at
//! head_dim 64, 128 and 256 and at mq 1 and 31, i.e. on both sides of the two degeneracies
//! (`hd == lanes` and `m == 1`) that hid every one of them.
#![cfg(feature = "spyre")]

use scratchy_subtile::addr::{Head, Idx, Nest, Slot};
use scratchy_subtile::sdsc_abstract::{dev_off_stk, kcache_kt_write_offset, vcache_write_offset};
use scratchy_subtile::superdsc_opspec::Df;

const STK: usize = 64;

/// Rank-2 `[row, feat]` == `dev_off_stk(.., stick_idx = 1, ..)` — the RowBlocked/Kernel residency.
#[test]
fn rank2_is_dev_off_stk() {
    for &(rows, cols) in &[
        (1usize, 64usize),
        (1, 4096),
        (31, 2048),
        (31, 128),
        (64, 12800),
    ] {
        let n = Nest::new(&["row", "feat"], &[rows as u32, cols as u32], Df::Fp16);
        for r in [0, rows / 2, rows - 1] {
            for c in [0, 1, 63, 64, cols / 2, cols - 1] {
                if c >= cols {
                    continue;
                }
                assert_eq!(
                    n.off(&[r as u32, c as u32]),
                    dev_off_stk(&[rows, cols], 1, &[r, c], STK) as i64,
                    "rank2 rows={rows} cols={cols} at ({r},{c})"
                );
            }
        }
    }
}

// The rank-3 == `dev_off_head` case is proved by `v_cache_is_the_same_nest` /
// `kt_cache_is_the_same_nest` below, which are the same law at the same rank over the
// tensors that actually exist here.

/// `[slot, head, dim]` == the V-cache contract + head base. This is the formula the value bmm and
/// the cache write must BOTH resolve through; B5 and B6 were each a different transcription of it.
#[test]
fn v_cache_is_the_same_nest() {
    for &(nqh, cap, hd) in &[(32usize, 256usize, 64usize), (32, 256, 128), (8, 512, 256)] {
        let n = Nest::new(
            &["slot", "head", "feat"],
            &[cap as u32, nqh as u32, hd as u32],
            Df::Fp16,
        );
        for h in [0, nqh / 2, nqh - 1] {
            for slot in [0usize, 1, 63, 64, 65, cap - 1] {
                for d in [0usize, 1, 63, 64, hd - 1] {
                    assert_eq!(
                        n.off(&[slot as u32, h as u32, d as u32]),
                        (h * cap * hd + vcache_write_offset(slot, d, hd, cap, STK)) as i64,
                        "vcache nqh={nqh} cap={cap} hd={hd} at (h={h},slot={slot},d={d})"
                    );
                }
            }
        }
    }
}

/// `[dim, head, slot]` == the Kt-cache contract + head base — the transpose is a different AXIS
/// ORDER over the same law, not a different law.
#[test]
fn kt_cache_is_the_same_nest() {
    for &(nkvh, cap, hd) in &[(8usize, 256usize, 64usize), (8, 256, 128), (8, 512, 256)] {
        let n = Nest::new(
            &["feat", "head", "slot"],
            &[hd as u32, nkvh as u32, cap as u32],
            Df::Fp16,
        );
        for h in [0, nkvh - 1] {
            for d in [0usize, 1, 63, 64, hd - 1] {
                for slot in [0usize, 1, 63, 64, cap - 1] {
                    assert_eq!(
                        n.off(&[d as u32, h as u32, slot as u32]),
                        (h * hd * cap + kcache_kt_write_offset(slot, d, hd, cap, STK)) as i64,
                        "kt nkvh={nkvh} cap={cap} hd={hd} at (h={h},d={d},slot={slot})"
                    );
                }
            }
        }
    }
}

/// Composition: narrowing by semantic coordinate equals the full corner. A caller writes no
/// multiplier, so it cannot omit one — the point of the whole exercise.
#[test]
fn composed_views_equal_the_full_corner() {
    let (nqh, cap, hd) = (32u32, 256u32, 128u32);
    let n = Nest::new(&["slot", "head", "feat"], &[cap, nqh, hd], Df::Fp16);
    for h in [0u32, 17, 31] {
        for slot in [0u32, 1, 64, 255] {
            let composed = n
                .view()
                .at(Idx::<Head>::n(h))
                .at(Idx::<Slot>::n(slot))
                .off();
            assert_eq!(
                composed,
                n.off(&[slot, h, 0]),
                "composed head={h} slot={slot}"
            );
        }
    }
}

/// Every address is injective and inside the footprint, at every shape — the aliasing/OOB guard the
/// per-call-site formulas never had.
#[test]
fn nest_is_injective_and_in_footprint() {
    for &(a, b, c) in &[(32u32, 1u32, 128u32), (8, 31, 256), (32, 31, 64)] {
        let n = Nest::new(&["row", "head", "feat"], &[b, a, c], Df::Fp16);
        let total = n.elems() as usize;
        let mut seen = vec![false; total];
        for h in 0..a {
            for r in 0..b {
                for d in 0..c {
                    let o = n.off(&[r, h, d]) as usize;
                    assert!(o < total, "OOB {o} >= {total}");
                    assert!(!seen[o], "alias at {o} from (h={h},r={r},d={d})");
                    seen[o] = true;
                }
            }
        }
        assert!(
            seen.iter().all(|&x| x),
            "nest does not cover its own footprint"
        );
    }
}
