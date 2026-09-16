// SPDX-License-Identifier: Apache-2.0
//! THE KV CACHE WRITE HAS NO KV-HEAD LOOP — pinned off the EMITTED descriptor fields, at every head
//! dim, plus the pitch relation that licenses the `y` axis it rides on.
//!
//! What this replaces: `cachewr_kc#s#_r#` / `cachewr_vc#s#_r#` were `nkvh * nslab * rows` pointwise
//! copies each. `y` now walks the kv heads of one slab, so each family is `nslab * rows` — an 8x cut
//! at granite's nkvh=8, flat in the head COUNT, at hd=64 and hd=512 alike.
//!
//! ⛔⛔⛔ AND THE SLAB DOES NOT JOIN THAT AXIS — REFUTED ON CARD, WITH THE ARITHMETIC INTACT. Walking
//! `y` over `(kv head, slab)` PAIRS folds both loops into one axis: `h*nslab + s` steps exactly one
//! stick plane on both operands, every builder check passes at hd 64/128/256/512, a conservation law
//! differenced out of both nests holds, and two Kani harnesses discharged coverage + injectivity.
//! granite-8b at hd=128 then emitted GARBAGE from the first token, 10/10 runs ("The is a 1") at an
//! ITL of 78-82 ms against the head-walked form's 80-85 on the same bake — the recorded tell, a
//! FASTER ITL with degenerate output. granite-2b could not have caught it: at nslab == 1 the two
//! forms are the same emission. So the pins below are on the HEAD-walked form, and the pair walk is
//! not to be retried on the strength of its arithmetic, which was correct and insufficient.
//!
//! ⛔ NOR IS IT AN ITL WIN: granite-8b measured 12.2-12.9 tok/s / ITL 80.0-84.8 ms over 10 runs
//! against 12.1-12.2 / 82.4 for the loops. Removing seven eighths of this family's ops moved nothing,
//! because a request's K and V writes already shared ONE launch group. The value here is the op count
//! at large head dims and one fewer loop — not latency.
//!
//! ⭐ THE EMITTER'S OWN ANSWER, not a re-derivation: every walk assertion reads `layoutDimOrder_` /
//! `maxDimSizes_` / the per-core start addresses out of the op the builder emitted. A lock that
//! recomputes the address it checks is testing its own arithmetic.

use ktir_superdsc::emit;
use ktir_superdsc::ir::bridge::tiled_op_sdsc_op::{SharedKernelBmmForm, assemble_matmul_placed};
use scratchy_subtile::addr::{DevOff, Head, Idx, Nest, Row, Shape, Slabs};
use scratchy_subtile::sdsc_abstract::{
    KernelTag, Lanes, MatK, MatM, MatN, MatY, OperandPlacement, PagedKvPool, QueryRowCount,
    RowBlockedTag, StickLayout, Stk,
};
use scratchy_subtile::superdsc_opspec::Df;

const NKVH: u32 = 8;
const STICK: u32 = 64;
const PLANE_SLOTS: u32 = PagedKvPool::PLANE_SLOTS as u32;

fn slabs(hd: u32) -> Slabs {
    Shape::<0, 0, 0, 0>::slabs_of(hd, Df::Fp16)
}

fn src_nest(mq: u32, hd: u32) -> Nest {
    Nest::new(&["row", "head", "feat"], &[mq, NKVH, hd], Df::Fp16)
}
fn dst_nest(hd: u32) -> Nest {
    Nest::new(&["slot", "feat"], &[PLANE_SLOTS, hd], Df::Fp16)
}

/// The cache write EXACTLY as `lower_attn_node` emits it: one op per (request, slab, tensor), `y`
/// over this slab's kv heads, one-stick identity contraction, head-outermost walk.
fn cachewr_op(mq: u32, hd: u32, per_request: bool, req: u32, s: u32) -> emit::EmittedOp {
    let (src, dst) = (src_nest(mq, hd), dst_nest(hd));
    let n = slabs(hd);
    let a_place = OperandPlacement::of_plane_walk_by_head(&src, Idx::<Row>::n(req), s, n);
    let o_place = OperandPlacement::of_plane_walk_by_head_block_base(&dst, s, n);
    let mut sym = 0i64;
    assemble_matmul_placed(
        "cachewr_pin",
        MatM::of_query_rows(QueryRowCount::of_mq(if per_request { 1 } else { mq })),
        MatN::one_stick(Lanes::FP16),
        MatK::one_stick(Lanes::FP16),
        MatY::of_gqa_group(NKVH, a_place, o_place),
        SharedKernelBmmForm::of_kv_cache_write(),
        &Stk::<RowBlockedTag>::new(
            "t_newk",
            StickLayout::row_blocked(mq as usize, (NKVH * hd) as usize),
        )
        .unwrap(),
        a_place,
        &Stk::<KernelTag>::kernel(STICK as usize, STICK as usize, "t_ident"),
        DevOff::ZERO,
        &Stk::<KernelTag>::kernel(PLANE_SLOTS as usize, hd as usize, "t_kc"),
        o_place,
        &mut sym,
        None,
    )
}

fn layout_dim_order(e: &emit::EmittedOp, arg: usize) -> Vec<String> {
    let v = serde_json::to_value(&e.op).unwrap();
    v["dscs_"][0]["cachewr_pin"]["scheduleTree_"][arg]["layoutDimOrder_"]
        .as_array()
        .expect("declared order")
        .iter()
        .map(|x| x.as_str().unwrap().to_string())
        .collect()
}

fn max_dim_sizes(e: &emit::EmittedOp, arg: usize) -> Vec<i64> {
    let v = serde_json::to_value(&e.op).unwrap();
    v["dscs_"][0]["cachewr_pin"]["scheduleTree_"][arg]["maxDimSizes_"]
        .as_array()
        .expect("declared walk")
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

/// The op's own base address for one operand, in elements — the minimum over the per-core start
/// addresses, so it is the emitter's answer and not a re-derivation of the placement.
fn base_addr_elems(e: &emit::EmittedOp, arg: usize) -> i64 {
    let v = serde_json::to_value(&e.op).unwrap();
    v["dscs_"][0]["cachewr_pin"]["scheduleTree_"][arg]["startAddressCoreCorelet_"]["data_"]
        .as_object()
        .expect("per-core start addresses")
        .values()
        .map(|a| a.as_str().unwrap().parse::<i64>().unwrap() / 2)
        .min()
        .expect("at least one core")
}

/// THE PITCH RELATION `y` RESTS ON, at every head dim: a head is `nslab` stick planes, so a `y` step
/// of `pitch * stick` with `pitch = rows * nslab` lands on the next head — on BOTH operands, whose
/// row depths differ (`mq` vs the physical plane's slots).
#[test]
fn a_head_is_nslab_planes_and_the_pitch_reaches_it() {
    for hd in [64u32, 128, 256, 512] {
        for mq in [1u32, 4, 8, 31, 32] {
            let (src, dst) = (src_nest(mq, hd), dst_nest(hd));
            let n = slabs(hd).get();
            let head = src.span(Idx::<Head>::n(0), Idx::<Head>::n(1)).elems();
            assert_eq!(
                head,
                mq * n * STICK,
                "hd={hd} mq={mq}: the stream's head in planes"
            );
            assert_eq!(
                dst.block_span().elems(),
                PLANE_SLOTS * n * STICK,
                "hd={hd}: the pool's kv-head block in planes"
            );
            // ⛔ THE TWO PITCHES ARE DIFFERENT NUMBERS — the reason each operand declares its own, and
            // the reason one shared `mb` extent made a one-stick contraction look impossible.
            assert_eq!(
                head / STICK,
                mq * n,
                "hd={hd} mq={mq}: the source's pitch in rows"
            );
            assert_ne!(
                mq * n,
                PLANE_SLOTS * n,
                "hd={hd} mq={mq}: and it is not the cache's"
            );
        }
    }
}

/// THE EMITTED WALK: head-OUTERMOST on both operands, at every head dim. Under the batch-inner order
/// (`[mb, y, in]`) `mb` sits outside `y`, so `y` would stride exactly one stick whatever pitch an
/// operand declares — 64 against a real head stride of `mq*hd` / `PLANE_SLOTS*hd`.
#[test]
fn the_write_declares_the_head_outermost_walk() {
    for hd in [64u32, 128, 256, 512] {
        for mq in [1u32, 4, 32] {
            for s in 0..slabs(hd).get() {
                let e = cachewr_op(mq, hd, true, 0, s);
                assert_eq!(
                    layout_dim_order(&e, 0),
                    vec!["y", "mb", "in"],
                    "hd={hd} mq={mq} s={s}: the input walk is head-OUTERMOST"
                );
                assert_eq!(
                    layout_dim_order(&e, 2),
                    vec!["y", "mb", "out"],
                    "hd={hd} mq={mq} s={s}: the output walk is head-OUTERMOST"
                );
            }
        }
    }
}

/// THE DECLARED EXTENTS ARE EACH OPERAND'S OWN: `nkvh` heads of `mq*nslab`-deep source planes against
/// `nkvh` heads of `PLANE_SLOTS*nslab`-deep cache planes. One shared `mb` extent cannot state both.
#[test]
fn each_operand_declares_its_own_head_depth() {
    for hd in [64u32, 128, 256, 512] {
        for mq in [1u32, 4, 32] {
            let n = slabs(hd).get();
            let e = cachewr_op(mq, hd, true, 0, 0);
            assert_eq!(
                max_dim_sizes(&e, 0),
                vec![i64::from(NKVH), i64::from(mq * n), i64::from(STICK)],
                "hd={hd} mq={mq}: the source's head is mq*nslab rows deep"
            );
            assert_eq!(
                max_dim_sizes(&e, 2),
                vec![
                    i64::from(NKVH),
                    i64::from(PLANE_SLOTS * n),
                    i64::from(STICK)
                ],
                "hd={hd} mq={mq}: the cache's head is PLANE_SLOTS*nslab rows deep"
            );
        }
    }
}

/// THE SLAB IS A COORDINATE OF BOTH OFFSETS, and it moves them by ONE PLANE — the term that is 0 at
/// hd=64 and is the whole hd>64 bug class everywhere else. Read off the emitted addresses.
#[test]
fn the_slab_steps_one_plane_on_each_operand() {
    for hd in [128u32, 256, 512] {
        for mq in [1u32, 32] {
            for s in 1..slabs(hd).get() {
                let (lo, hi) = (
                    cachewr_op(mq, hd, true, 0, s - 1),
                    cachewr_op(mq, hd, true, 0, s),
                );
                assert_eq!(
                    base_addr_elems(&hi, 0) - base_addr_elems(&lo, 0),
                    i64::from(mq * STICK),
                    "hd={hd} mq={mq} s={s}: the source slab step is one plane"
                );
                assert_eq!(
                    base_addr_elems(&hi, 2) - base_addr_elems(&lo, 2),
                    i64::from(PLANE_SLOTS * STICK),
                    "hd={hd} mq={mq} s={s}: the cache slab step is one plane"
                );
            }
        }
    }
}

/// A PROMPT CHUNK writes all its rows in one op per (slab, tensor) — rows are consecutive positions
/// of ONE sequence, which `m = mq` plus the runtime's single slot-shift expresses exactly.
#[test]
fn a_prompt_chunk_writes_all_its_rows_in_one_op() {
    for hd in [64u32, 128] {
        for mq in [4u32, 31, 63] {
            let e = cachewr_op(mq, hd, false, 0, 0);
            let n = slabs(hd).get();
            assert_eq!(
                max_dim_sizes(&e, 0),
                vec![i64::from(NKVH), i64::from(mq * n), i64::from(STICK)],
                "hd={hd} mq={mq}: a chunk sweeps all its rows under one head walk"
            );
        }
    }
}

/// THE REQUEST MOVES THE SOURCE ROW AND NOTHING ELSE: the cache side never moves, because the page
/// and the slot are the LAUNCH's (`kv_request` → the host's block table). A per-request term in the
/// cache ADDRESS would be double-counting the row — which is what the address carried before, and
/// what the pool's page-per-row map replaced.
#[test]
fn the_request_moves_the_source_row_and_nothing_else() {
    for hd in [64u32, 128, 512] {
        let mq = 32;
        let (mut src_prev, mut dst_prev) = (None, None);
        for req in 0..mq {
            let e = cachewr_op(mq, hd, true, req, 0);
            let (a, o) = (base_addr_elems(&e, 0), base_addr_elems(&e, 2));
            if let (Some(pa), Some(po)) = (src_prev, dst_prev) {
                assert_eq!(
                    a - pa,
                    i64::from(STICK),
                    "hd={hd} req={req}: one row is one stick"
                );
                assert_eq!(
                    o - po,
                    0,
                    "hd={hd} req={req}: the cache address carries NO request term"
                );
            }
            (src_prev, dst_prev) = (Some(a), Some(o));
        }
    }
}
