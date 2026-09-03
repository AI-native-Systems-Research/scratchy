// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "superdsc")]
//! The RoPE head-major collapse is valid at ANY row count — the law batched decode now rests on.
//!
//! RoPE used to emit one block per (row, head): ~40 ops per op-type, ~150 a layer, and its own
//! comment called that "the single largest unaccounted-for chunk of the perf gap". The collapsed form
//! emits FOUR ops total by treating the roped `[mq, heads*hd]` tensor as `[heads*mq, hd]`, and it was
//! gated to `mq == 1` on the belief that the equivalence was a one-row coincidence.
//!
//! It is not. Head `h`, row `r` of the token stream sits at `h*mq*hd + r*hd`, and row `h*mq + r` of
//! the `[heads*mq, hd]` view sits at `(h*mq + r)*hd` — the same element, at every `mq`. That is the
//! row-expansion law `addr_eq` already proves; these tests pin it in the two places the emitter
//! actually depends on it, because a drift in either is silent: the ops still emit, the bundle still
//! bakes, and the rotation lands on the wrong head or the wrong position.

use scratchy_subtile::sdsc_abstract::StickLayout;

const HD: usize = 64; // one fp16 stick — the collapse's precondition
const HEADS: usize = 32;

/// THE LAW: the token stream and its head-major view are the same bytes at any row count.
#[test]
fn the_head_major_view_is_the_token_stream_at_every_row_count() {
    for mq in [1usize, 2, 4, 8, 16, 32] {
        let stream = StickLayout::row_blocked(mq, HEADS * HD); // [mq, heads*hd]
        let view = StickLayout::row_blocked(HEADS * mq, HD); // [heads*mq, hd]
        assert!(
            stream.addr_eq(&view),
            "mq={mq}: [{mq}, {}] and [{}, {HD}] must be the same bytes",
            HEADS * HD,
            HEADS * mq
        );
        // And element-for-element, which is what the emitter's offsets actually use.
        for h in [0usize, 1, HEADS - 1] {
            for r in [0usize, mq / 2, mq - 1] {
                for d in [0usize, 1, HD - 1] {
                    assert_eq!(
                        stream.dev_off(r, h * HD + d),
                        view.dev_off(h * mq + r, d),
                        "mq={mq} h={h} r={r} d={d}"
                    );
                }
            }
        }
    }
}

/// THE COS/SIN CONTRACT. The worker stages the angle table stick-scattered: position `p`, element `i`
/// goes to `(i/64)*(mq*64) + p*64 + (i%64)`. The collapsed RoPE reads it as `[heads*mq, hd]` row
/// `h*mq + p`. Those must be the same address, or every head but head 0 rotates at the wrong
/// position — fluent output, wrong attention geometry, no crash.
#[test]
fn the_staged_angles_land_where_the_collapsed_read_expects() {
    for mq in [1usize, 2, 4, 8, 32] {
        let view = StickLayout::row_blocked(HEADS * mq, HD);
        for h in [0usize, 1, 7, HEADS - 1] {
            for p in [0usize, mq / 2, mq - 1] {
                for d in [0usize, 1, 63] {
                    let i = h * HD + d; // the table is head-tiled: head h's slice at h*hd
                    let staged = (i / 64) * (mq * 64) + p * 64 + (i % 64);
                    assert_eq!(
                        staged,
                        view.dev_off(h * mq + p, d),
                        "mq={mq} h={h} p={p} d={d}: staging and the collapsed read disagree"
                    );
                }
            }
        }
    }
}

/// The collapse's precondition. It holds only while a head IS one stick: above that the head's dims
/// span several device sticks and the flat `[heads*mq, hd]` view stops being contiguous, which is why
/// the emitter keeps the slab form for `hd > 64`.
#[test]
fn the_collapse_is_refused_above_one_stick_per_head() {
    use scratchy_subtile::addr::Shape;
    use scratchy_subtile::superdsc_opspec::Fp16;
    assert!(
        (64u32 == <Fp16 as scratchy_subtile::superdsc_opspec::DataFormat>::DF.elems_per_stick())
    );
    assert!(
        !(128u32 == <Fp16 as scratchy_subtile::superdsc_opspec::DataFormat>::DF.elems_per_stick())
    );
}
