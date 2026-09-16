// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "spyre")]
//! DOES THE `y`-BATCHED ROTATE ADDRESS THE SAME BYTES AS THE PER-HEAD ONE? Differenced out of the
//! EMITTED per-core start addresses, never predicted — the rule this file exists to obey.
//!
//! The per-head form (one `[mq,hd]` P matmul per head, `hd`-wide contraction) is CARD-PROVEN coherent
//! on granite-8b. The `y`-batched form (one op per slab, heads on `y`, one-stick contraction, the ±I
//! block read out of P at its own (in-slab, out-slab) coordinate) measured 4% FASTER and produced
//! output in which the model says its own prompt looks jumbled. So one of them addresses different
//! bytes, and this says which — at hd=128, where a wrong slab or a transposed kernel coordinate is
//! expressible, and at hd=64, where it is not.

use scratchy_subtile::addr::{DevOff, Feat, Idx, Nest, Row};
use scratchy_subtile::superdsc_opspec::Df;

const HD: u32 = 128;
const STK: u32 = 64;
const HEADS: u32 = 32;
const MQ: u32 = 21;

/// P's `[hd,hd]` kernel nest, as both the emitter and the worker's `stage_2d` see it: `row` = the
/// contraction index `inn`, `feat` = the output index `o` (the stick axis).
fn p_nest() -> Nest {
    Nest::new(&["row", "feat"], &[HD, HD], Df::Fp16)
}

/// THE WORKER'S OWN FILL LAW, so the test compares against what is actually in the buffer:
/// `stage_2d(StickLayout::kernel(hd,hd), |inn, o| rope_p_entry(hd, inn, o))`.
fn p_value_at(inn: u32, o: u32) -> i8 {
    scratchy_subtile::sdsc_abstract::rope_p_entry(HD as usize, inn as usize, o as usize)
}

/// ⭐ THE BLOCK THE `y`-BATCHED FORM READS for output slab `s`, and what is really in it. If the
/// (in-slab, out-slab) coordinate is transposed, this finds it: P is NOT symmetric — its transpose is
/// the INVERSE rotation, so a swap flips the sign of both halves and the rotation silently becomes
/// `-rotate_half`, which is fluent-but-wrong output, not a crash.
#[test]
fn the_block_the_slab_form_reads_is_plus_or_minus_identity() {
    let n_slabs = HD / STK;
    for s in 0..n_slabs {
        let p_in = if s < n_slabs / 2 {
            s + n_slabs / 2
        } else {
            s - n_slabs / 2
        };
        // The emitter's own coordinate for the block.
        let base = p_nest()
            .view()
            .at(Idx::<Row>::n(p_in * STK))
            .slab(s)
            .dev()
            .into_raw_elems();
        // Read the 64x64 window the op declares at that base, through the [64,64] KERNEL law
        // (`in*64 + out`), and check it is exactly ±I — the whole premise of the collapse.
        let want_sign: i8 = if s < n_slabs / 2 { -1 } else { 1 };
        for i in 0..STK {
            for o in 0..STK {
                let flat = base + i * STK + o;
                // Invert the [hd,hd] kernel law to learn which (inn, o') that byte really holds.
                let (grp, rem) = (flat / (HD * STK), flat % (HD * STK));
                let (inn, lane) = (rem / STK, rem % STK);
                let o_real = grp * STK + lane;
                let got = p_value_at(inn, o_real);
                let expect = if i == o { want_sign } else { 0 };
                assert_eq!(
                    got, expect,
                    "s={s} p_in={p_in} block[{i},{o}] -> P[{inn},{o_real}] = {got}, want {expect} \
                     (the block must be {want_sign}*I; a transposed coordinate flips both signs)"
                );
            }
        }
    }
}

/// AND THE ROTATE THE TWO FORMS COMPUTE MUST AGREE, element for element: the per-head form contracts
/// the WHOLE head dim against P, the slab form contracts ONE stick against a ±I block. Same answer, or
/// the collapse is not a collapse.
#[test]
fn both_forms_compute_the_same_rotate() {
    let n_slabs = HD / STK;
    // x[h][d] = a value distinguishable per (head, dim), so a wrong head or slab shows up.
    let x = |h: u32, d: u32| (h * HD + d) as i32;
    for h in [0u32, 1, HEADS - 1] {
        for o in 0..HD {
            // PER-HEAD: rot[o] = sum_inn x[inn] * P[inn, o]
            let per_head: i32 = (0..HD)
                .map(|inn| x(h, inn) * i32::from(p_value_at(inn, o)))
                .sum();
            // SLAB FORM: output slab s = o/64 draws ONLY from input slab p_in through ±I.
            let s = o / STK;
            let p_in = if s < n_slabs / 2 {
                s + n_slabs / 2
            } else {
                s - n_slabs / 2
            };
            let sign = i32::from(if s < n_slabs / 2 { -1 } else { 1 });
            let slab_form = sign * x(h, p_in * STK + o % STK);
            assert_eq!(
                per_head, slab_form,
                "h={h} o={o}: the slab form must reproduce the per-head rotate"
            );
        }
    }
}

/// THE OPERAND OFFSETS the slab form uses, differenced through the nest — the source reads input slab
/// `p_in` and writes output slab `s`, both in the `[mq, heads*hd]` token stream, and both must step one
/// stick PLANE per slab and `mq*hd` per head.
#[test]
fn the_slab_form_steps_planes_and_heads_through_the_token_stream() {
    let nest = Nest::new(&["row", "feat"], &[MQ, HEADS * HD], Df::Fp16);
    let at = |h: u32, s: u32| {
        nest.view()
            .at(Idx::<Feat>::n(h * HD + s * STK))
            .dev()
            .into_raw_elems()
    };
    for h in [0u32, 1, HEADS - 1] {
        assert_eq!(
            at(h, 1) - at(h, 0),
            MQ * STK,
            "h={h}: one slab is one stick plane"
        );
    }
    for s in 0..HD / STK {
        assert_eq!(at(1, s) - at(0, s), MQ * HD, "s={s}: one head is mq*hd");
    }
    // And the base the op is given for (head 0, slab s) is exactly `s` planes in.
    for s in 0..HD / STK {
        assert_eq!(at(0, s), s * MQ * STK, "s={s}: the op's own base");
    }
    let _ = DevOff::ZERO;
}
