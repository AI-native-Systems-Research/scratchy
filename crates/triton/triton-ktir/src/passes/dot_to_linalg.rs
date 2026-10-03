// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! Ported from `DotToLinalg.cpp`. `tt.dot` -> `linalg.matmul`.
//!
//! `tt.dot`'s `d = a*b + c` maps directly: `linalg.matmul ins(a, b) outs(c)`
//! computes `outs += a*b`, so the accumulator `c` is the DPS init.
//!
//! THE SEPARATION OF CONCERNS THE C++ MAKES, kept exactly, because it is what
//! lets flash attention through at all:
//!
//! * LOWERING (`tt.dot` -> `linalg.matmul`) is a local, always-valid rewrite.
//! * ROUTING (deciding the whole kernel IS a standalone matmul the shape-keyed
//!   single-program builder can emit, and saying so with
//!   `spyre.canonical_verified`) is a whole-kernel contract.
//!
//! The discriminator is the dot COUNT. Exactly one dot -> run the whole-kernel
//! contract and RED-stop any deviation. More than one -> it cannot be the
//! single-program matmul kernel BY DEFINITION, so lower each dot locally and
//! leave it UNTAGGED. Untagged is the fail-closed part: the emitter REQUIRES the
//! tag, so a multi-dot kernel is refused at the stage that actually cannot
//! represent it. Attention has four dots (two legs x two causal stages) and must
//! not be refused here.

use crate::ir::*;
use crate::passes::walk::{self, OpPath};
use crate::{Refusal, Result};

const PASS: &str = "spyre-dot-to-linalg";

fn refuse(why: impl std::fmt::Display) -> Refusal {
    Refusal::new(
        PASS,
        format!(
            "matmul is not the canonical faithfully-lowerable kernel, so it is \
             refused rather than silently miscompiled (review P1.0): {why}"
        ),
    )
}

/// THE `indexing_maps` A TRANSPOSED DOT WEIGHT IS SPENT IN — A as `(m, k)`, W as `(n, k)`, out as
/// `(m, n)`. `linalg.matmul`'s own DEFAULT is `[(d0,d2), (d2,d1), (d0,d1)]`, i.e. W as `(k, n)`, and
/// an ABSENT `indexing_maps` IS that default; these three are what say the weight is indexed the
/// other way round. See the long note at the rewrite below for why the transposition is spent here
/// rather than left for the access tile.
///
/// ⭐ A NAMED CONST BECAUSE THERE IS A READER, AND TWO MATCHERS FOR ONE FACT IS THE DEFECT FAMILY
/// THIS TREE KEEPS PAYING FOR. `plan_corelets::recover_matmul_shape` has to know WHICH AXIS of the
/// weight's memory view is `N` before it can derive the work division, and the only thing that says
/// so is this attribute. Spelling the maps a second time there would let the writer and the reader
/// drift silently — and the failure mode is not a compile error, it is a work division derived from
/// the wrong `N`. `Attr::AffineMap`'s own doc says maps are "kept as text ON PURPOSE" because "NONE
/// of them inspects a map's interior"; the reader does not inspect one either, it compares against
/// this literal.
pub const TRANSPOSED_B_MAPS: [&str; 3] = [
    "(d0, d1, d2) -> (d0, d2)",
    "(d0, d1, d2) -> (d1, d2)",
    "(d0, d1, d2) -> (d0, d1)",
];

/// Is this `linalg.matmul`'s WEIGHT indexed `(n, k)` rather than `linalg`'s default `(k, n)`?
///
/// The op's own `indexing_maps`, compared against [`TRANSPOSED_B_MAPS`]. `None`/absent is
/// `linalg.matmul`'s default, which is `(k, n)` — so `false`, and that is MLIR's semantics rather
/// than a convention chosen here.
pub(crate) fn weight_is_n_by_k(op: &Op) -> bool {
    match op.attr(&AttrKey::IndexingMaps) {
        Some(Attr::AffineMapList(m)) => {
            m.len() == 3 && m.iter().zip(TRANSPOSED_B_MAPS.iter()).all(|(a, b)| a == b)
        }
        _ => false,
    }
}

pub fn run(module: &mut Module) -> Result<()> {
    // ANTI-FORGERY (review P1.0). `spyre.canonical_verified` is TRUST that THIS
    // pass verified the matmul, and an IRSource author can hand-write the
    // attribute into the input. Strip any pre-existing tag -- module and every op
    // -- BEFORE anything else, so the only tags that reach the emitter are ones
    // set below on matmuls actually verified this run.
    module.attrs.retain(|(k, _)| *k != AttrKey::CanonicalVerified);
    walk::for_each_mut(module, |op| op.remove_attr(&AttrKey::CanonicalVerified));

    let dots: Vec<OpPath> = walk::paths(module)
        .into_iter()
        .filter(|p| walk::at(module, p).map(|o| o.kind == OpKind::TtDot).unwrap_or(false))
        .collect();
    if dots.is_empty() {
        return Ok(());
    }
    let single_dot_kernel = dots.len() == 1;

    for path in &dots {
        let dot = walk::at(module, path).expect("collected path").clone();
        let a_ty = dot
            .operands
            .first()
            .and_then(|v| module.type_of(*v))
            .ok_or_else(|| Refusal::new(PASS, "expected 2-D ranked tensor operands"))?;
        let b_ty = dot
            .operands
            .get(1)
            .and_then(|v| module.type_of(*v))
            .ok_or_else(|| Refusal::new(PASS, "expected 2-D ranked tensor operands"))?;
        let d_ty = dot
            .result_type()
            .cloned()
            .ok_or_else(|| Refusal::new(PASS, "expected 2-D ranked tensor operands"))?;

        let two_d = |t: &IrType| matches!(t, IrType::Tensor { dims, .. } if dims.len() == 2);
        if !two_d(&a_ty) || !two_d(&b_ty) || !two_d(&d_ty) {
            return Err(Refusal::new(PASS, "expected 2-D ranked tensor operands"));
        }
        // f16-only: the device accumulates in DL16-f16. ⭐ THE ONE EXCEPTION IS THE fp8 W8A8
        // WEIGHT: the kernel WIDENS it with `.to(tl.float16)` (the frontend refuses a mixed
        // `tl.dot`, "Both operands must be same dtype"), so the dot's operands are all f16
        // here and the fp8-ness is carried by the WEIGHT LOAD's descriptor elem -- read below,
        // not assumed. An f32 operand is still refused: nothing widens that away.
        let mut is_fp8_kernel = false;
        if a_ty.elem() != Some(DType::F16)
            || b_ty.elem() != Some(DType::F16)
            || d_ty.elem() != Some(DType::F16)
        {
            return Err(Refusal::new(
                PASS,
                "only f16 tt.dot is supported (device accumulates in DL16-f16); use an \
                 f16 accumulator. f32-accumulator folding is a follow-on.",
            ));
        }

        // ⭐⭐⭐ THE fp8 W8A8 WEIGHT LEG: a chain `extf`/`trans` between the weight load and
        // the dot. The widening is a NO-OP AT THE KTIR LEVEL (`ktdp.load` over an fp8 view
        // already reads f16 values; scratchy's own producer types even the fp8 weight load's
        // result F16), and the transpose is spent in the indexing maps exactly as an f16
        // weight's is. So BOTH are peeled here, down to the direct `tt.descriptor_load` --
        // but only when that load's descriptor elem is fp8 (`f8E4M3FN`), which is what makes
        // this the fp8 form rather than a typo'd f16 kernel with a stray cast. Either order
        // is accepted (`extf(trans(load))` and `trans(extf(load))`); anything else on the
        // chain -- a reshape, a second op -- is not a load with a spelling on it and falls
        // through to the direct-load guard below, which refuses by name.
        let fp8_b_load = {
            let mut v = dot.operands[1];
            let mut peeled = 0;
            loop {
                match module.def_of(v).filter(|o| {
                    o.kind == OpKind::ArithExtf || o.kind == OpKind::TtTrans
                }) {
                    Some(o) if peeled < 2 => {
                        v = o.operands[0];
                        peeled += 1;
                    }
                    _ => break,
                }
            }
            // The peeled chain is the fp8 leg ONLY IF it lands on an fp8-typed load.
            let fp8 = module
                .def_of(v)
                .filter(|o| o.kind == OpKind::TtDescriptorLoad)
                .and_then(|ld| ld.operands.first().copied())
                .and_then(|desc| module.type_of(desc))
                .is_some_and(|t| t.elem() == Some(DType::Fp8E4m3));
            if fp8 {
                is_fp8_kernel = true;
                Some(v)
            } else {
                None
            }
        };

        // ⭐⭐⭐ RUNG 4 — THE GATHERED-B LEG: `tt.descriptor_gather` on B, with or without
        // a `tt.trans` over it. The rows the contraction's K axis reads are chosen AT
        // RUNTIME by an index vector read from device memory (rung 4 of the
        // address-provenance ladder), so B is an IN-REGISTER COMPUTED VALUE, not a
        // presented buffer. The no-fold law above applies unchanged: `b_trans_src`
        // requires a direct `tt.descriptor_load` under the trans, a gather fails that
        // filter, and a trans STAYS -- never folded into the maps. The paged verifier
        // then DISCRIMINATES the two forms: a direct gather is the V leg (contracted
        // where it lies) and is admitted; a transposed gather is the K leg, which needs
        // a relayout this arch's dxp refuses, and is refused there BY NAME. A gather
        // anywhere else keeps the direct-load guard's refusal.
        let gathered_b = {
            let mut v = dot.operands[1];
            if let Some(t) = module.def_of(v).filter(|o| o.kind == OpKind::TtTrans) {
                v = t.operands[0];
            }
            module
                .def_of(v)
                .filter(|o| o.kind == OpKind::TtDescriptorGather)
                .map(|_| v)
        };

        // The FIDELITY GATE, for the ROUTING decision only. Skipped for a multi-dot
        // kernel, which cannot be that kernel shape at all. THE fp8 ARM VERIFIES ITS OWN
        // CONTRACT: `{1 dot, 3 loads, 1 mulf chain, 1 store}` with the scale load traced to
        // its own parameter -- the f16 verifier's counts would refuse it (it expects 2
        // loads and no epilogue), so the fp8 form is verified by its own function rather
        // than by bending the f16 one. SO DOES THE GATHERED ARM (rung 4): its B is a
        // transposed gather, which the f16 counts would refuse the same way.
        let canonical = if single_dot_kernel {
            if is_fp8_kernel {
                // The PEELED LOAD, not `dot.operands[1]` -- this runs BEFORE the rewire
                // below, so the dot's operand is still the extf/trans chain. The peel in
                // `run` checked the fp8 elem; this function checks everything else.
                let b_load = fp8_b_load.expect("is_fp8_kernel implies the peeled load");
                verify_canonical_fp8_matmul_kernel(module, path, b_load)?;
            } else if let Some(g) = gathered_b {
                verify_canonical_paged_matmul_kernel(module, path, g)?;
            } else {
                verify_canonical_matmul_kernel(module, path)?;
            }
            true
        } else {
            false
        };

        // ⭐ A TRANSPOSED B IS SPENT HERE, IN THE INDEXING MAPS -- NOT LEFT FOR THE TILE.
        //
        // `ktir-superdsc`'s `matmul` reads the weight as `[n, k]`, and scratchy's producer states
        // exactly that with NO transpose anywhere: `KtirFunc::matmul` builds
        // `view_shaped(w.tensor, n, kdim)`, tiles it `[n, kdim]` with an identity order, and carries
        // the transposition in `indexing_maps = [[0, 2], [1, 2], [0, 1]]` -- A as `(m, k)`, W as
        // `(n, k)`, out as `(m, n)`. Standard `linalg.matmul` is `[[0, 2], [2, 1], [0, 1]]`.
        //
        // MEASURED, and this is why the transpose cannot be left downstream: with a `tt.trans` still
        // on B, `fold_trans_into_access_tile_order` sets a transposing `CoordinateOrder` AND rewrites
        // the tile's TYPE to the post-transpose shape. `regions()` reads that type, so the weight
        // reaches `matmul` as `[k, n]` and is refused ("W cols 256 != A cols (K) 128") no matter how
        // the kernel declares the buffer -- the kernel's `[n, k]` view is correct and the tile
        // contradicts it.
        //
        // So B is REWIRED past the transpose to the value the transpose reads, and the maps say the
        // weight is indexed `(n, k)`. The `tt.trans` is then dead and `canonicalize` removes it. The
        // result is the shape scratchy's producer emits, reached from Triton's `tl.dot(x, w.T)`.
        //
        // ⛔⛔⛔ AND IT IS SPENT HERE **ONLY FOR A PRESENTED BUFFER**, BECAUSE ONLY A PRESENTED
        // BUFFER HAS ANYBODY TO SPEND IT. Stating the maps LABELS which axis of B is `k`; it does not
        // PLACE it. The device's kernel slot reads ONE residency and it is `[k, n]` --
        // `ktir-superdsc`'s `matmul` declares `StickLayout::kernel(k_in, n_out)`, whose address law is
        // `dev_off_stk`'s rank-2 stick-blocked form with `dims[0] = k` as the ROW count -- so a
        // transpose-B `[n, k]` operand is bytes in the WRONG order and somebody has to move them:
        //
        //   * a PRESENTED weight (`desc_w.load(..).T`) is moved ONCE by the HOST stage
        //     (`stage_2d(&StickLayout::kernel(k, n), ..)`, `sdsc_abstract.rs:2124`), at bake time,
        //     for free at inference. That is this fold, and it is measured byte-identical.
        //   * a CACHE is moved by the device writing it already transposed
        //     (`kcache_kt_write_offset`), which is why the shipped attention keeps a third physical
        //     Kᵀ plane at all.
        //   * an IN-REGISTER COMPUTED VALUE has NEITHER. Nothing places its bytes, and nothing in
        //     `ktir-superdsc` reads an access tile's `CoordinateOrder` (`grep -rn CoordinateOrder
        //     src/` finds it inside one comment and nowhere else), so the transposing order this
        //     path can set is never honoured. Folding it into the maps therefore DROPPED it:
        //     `decoder_block.py`'s `tl.dot(q1r, k1r.T)` contracted as `q1r @ k1r`, invisible to every
        //     extent guard because `[M, HALF]` is 64x64 there.
        //
        // So the discriminator is whether the transposed value is a DIRECT `tt.descriptor_load` --
        // the only form whose bytes the host stage places. Anything else keeps its `tt.trans`, which
        // `to_ktir::convert_trans` turns into a REAL `linalg.transpose` (a `Program::Transpose`
        // beside the matmul), the matmul then states no maps at all, and MLIR's default `[k, n]` is
        // read off a buffer that is physically `[k, n]`. That costs a data movement and says so at
        // `convert_trans`; it does not cost a wrong answer.
        //
        // ⛔ CONSERVATIVE ON PURPOSE: a load with anything in between (a reshape, a second
        // transpose) is not a direct load either, and folding for it would be wrong for the same
        // reason -- the host stages the buffer, not the reshaped view of it.
        let b = dot.operands[1];
        // ⭐ THE fp8 LEG'S PEELED LOAD was computed above (`b` is already the peeled value
        // when the chain was an fp8 widening). Its transpose -- when the kernel spells `.T`
        // -- is folded into the maps exactly as an f16 weight's is: the presented buffer's
        // bytes are placed by the host stage either way.
        let fp8_peeled = fp8_b_load.filter(|&src| src != dot.operands[1]);
        let b_trans_src = module
            .def_of(b)
            .filter(|o| o.kind == OpKind::TtTrans)
            .and_then(|o| o.operands.first().copied())
            .filter(|src| {
                module.def_of(*src).is_some_and(|d| d.kind == OpKind::TtDescriptorLoad)
            })
            .or(fp8_peeled);

        // Rewrite in place: same results, same operands minus the accumulator's
        // position (linalg.matmul takes ins(a, b) outs(c), which is the same three
        // values in the same order).
        let mm = {
            let op = walk::at_mut(module, path).expect("collected path");
            op.kind = OpKind::LinalgMatmul;
            if is_fp8_kernel {
                // THE fp8 REWIRE: B becomes the WEIGHT LOAD's own result (fp8-typed on its
                // descriptor, f16 on the value), and the orientation follows the peeled
                // chain -- a `.T` in the kernel means transpose-B maps, a plain load means
                // MLIR's default. The maps are what `matmul_b_orientation` reads downstream,
                // so the framing is stated HERE and proven there, never defaulted.
                if let Some(src) = fp8_peeled {
                    op.operands[1] = src;
                    op.set_attr(
                        AttrKey::IndexingMaps,
                        Attr::AffineMapList(
                            TRANSPOSED_B_MAPS.iter().map(|m| (*m).to_string()).collect(),
                        ),
                    );
                } else if let Some(src) = fp8_b_load {
                    op.operands[1] = src;
                    op.remove_attr(&AttrKey::IndexingMaps);
                }
            } else if let Some(src) = b_trans_src {
                op.operands[1] = src;
                op.set_attr(
                    AttrKey::IndexingMaps,
                    Attr::AffineMapList(
                        TRANSPOSED_B_MAPS.iter().map(|m| (*m).to_string()).collect(),
                    ),
                );
            }
            if canonical {
                op.set_attr(AttrKey::CanonicalVerified, Attr::Unit);
            }
            op.clone()
        };
        let _ = mm;
        if canonical {
            module.attrs.push((AttrKey::CanonicalVerified, Attr::Unit));
        }
    }
    Ok(())
}

/// The whole-kernel matmul contract (review P1.0).
///
/// The multicore matmul emitter is a shape-keyed TEMPLATE: it derives M/N/K from
/// the memory-view SIZES and emits a full-K, contiguous, single-matmul kernel. It
/// IGNORES strides, load/store offsets and whole-function cardinality, so a
/// non-standard stride, a shifted offset, an extra store, an extra dot, or any
/// other compute would be SILENTLY DROPPED. So recognize ONLY the exact canonical
/// pattern and RED-stop every deviation.
fn verify_canonical_matmul_kernel(module: &Module, dot_path: &OpPath) -> Result<()> {
    let dot = walk::at(module, dot_path).expect("path");
    let a = dot.operands[0];
    let b = dot.operands[1];
    let c = dot.operands[2];
    let d = dot.result().ok_or_else(|| refuse("the dot defines no result"))?;

    // A and B must be DIRECT descriptor loads: no prologue or reshape on the
    // operands. This is the guard the swiglu fixture's docstring quotes -- the
    // tutorial's pointer-arithmetic load form is refused right here.
    //
    // ⭐ B's ONE permitted wrapper is a `tt.trans` DIRECTLY over the load -- the
    // `.T` form, which `dot_to_linalg` folds into the transpose-B `indexing_maps`
    // (B's region `[n, k]`, the bytes as a presented weight binds them). The
    // canonical form is then the transposed mirror of the plain one, checked with
    // the same discipline below: B's descriptor is `[n, k]` blocking `[bn, bk]`
    // and its load offset is `[pid, k]` (an n ROW selected by pid, k in place).
    // Anything else between the load and the dot -- a reshape, a second trans, a
    // gather -- is refused here exactly as before.
    let b_trans = module.def_of(b).filter(|o| o.kind == OpKind::TtTrans);
    let b_load_val = match b_trans {
        Some(t) => t.operands.first().copied(),
        None => Some(b),
    };
    let a_ld = module.def_of(a).filter(|o| o.kind == OpKind::TtDescriptorLoad);
    let b_ld = b_load_val.and_then(|v| module.def_of(v).filter(|o| o.kind == OpKind::TtDescriptorLoad));
    let (Some(a_ld), Some(b_ld)) = (a_ld, b_ld) else {
        return Err(refuse("A and B must be direct tt.descriptor_load results"));
    };

    let a_ty = module.type_of(a).ok_or_else(|| refuse("A has no type"))?;
    let b_ty = module.type_of(b).ok_or_else(|| refuse("B has no type"))?;
    let (bm, bk) = (a_ty.dims().unwrap()[0], a_ty.dims().unwrap()[1]);
    let bn = b_ty.dims().unwrap()[1];

    let a_desc = descriptor_of(module, a_ld)?;
    let b_desc = descriptor_of(module, b_ld)?;
    let (m, k) = desc_shape2(module, a_desc)
        .ok_or_else(|| refuse("A/B descriptor M/N/K are not compile-time constants"))?;
    // B's descriptor extents, framed by the orientation: plain B is `[k, ns]`, a `.T`
    // B is the presented weight's own `[ns, k]` (the fold in `run` states the
    // transpose-B maps over exactly this view). The axis that equals A's k names the
    // reduction dim; the other one is n. (A square `[k, k]` satisfies both readings
    // and gives the same `ns` either way.)
    let ns = match desc_shape2(module, b_desc) {
        Some((d0, d1)) if d0 == k => d1,
        Some((d0, d1)) if d1 == k => d0,
        Some((d0, d1)) => {
            return Err(refuse(format!(
                "B's descriptor [{d0},{d1}] contracts no axis of A's K={k} -- the \
                 weight extents do not state a matmul"
            )));
        }
        None => {
            return Err(refuse("A/B descriptor M/N/K are not compile-time constants"));
        }
    };

    // Single-program tile: the emitter emits ONE M x N program and DISCARDS the
    // block size, so a multi-block tiling would be silently collapsed.
    if bm != m || bn != ns {
        return Err(refuse(format!(
            "multi-block matmul is not supported: the block (BM={bm}, BN={bn}) must \
             equal the full (M={m}, N={ns}); the emitter discards the block size and \
             assumes one M x N program"
        )));
    }

    // Cardinality + no-other-compute.
    let func = module.kernel()?;
    let all = func.ops_deep();
    let count = |k: OpKind| all.iter().filter(|o| o.kind == k).count();
    let n_dot = count(OpKind::TtDot);
    let n_load = count(OpKind::TtDescriptorLoad);
    let n_store = count(OpKind::TtDescriptorStore);
    let n_pid = count(OpKind::TtGetProgramId);
    let n_for = count(OpKind::ScfFor);

    for op in &all {
        let structural = matches!(
            op.kind,
            OpKind::TtDot
                | OpKind::TtDescriptorLoad
                | OpKind::TtDescriptorStore
                | OpKind::TtMakeTensorDescriptor
                | OpKind::TtGetProgramId
                | OpKind::ScfFor
                | OpKind::TtFunc
                | OpKind::TtReturn
                | OpKind::ScfYield
                | OpKind::ArithConstant
                | OpKind::ArithIndexCast
                // The `.T` form's ONE wrapper (the trans the fold consumes).
                | OpKind::TtTrans
        );
        if !structural {
            return Err(refuse(
                "kernel contains an operation the matmul template does not model (only \
                 one matmul feeding one store, with no other compute or side-effecting \
                 op, is representable)",
            ));
        }
    }
    if n_dot != 1 || n_load != 2 || n_store != 1 {
        return Err(refuse(format!(
            "kernel is not exactly {{1 tt.dot, 2 tt.descriptor_load, 1 \
             tt.descriptor_store}} (got dot={n_dot} load={n_load} store={n_store}); \
             extra loads/stores/dots would be silently dropped"
        )));
    }
    if n_pid != 1 {
        return Err(refuse("kernel does not have exactly one tt.get_program_id"));
    }
    let pid_op = all
        .iter()
        .find(|o| o.kind == OpKind::TtGetProgramId)
        .expect("counted one");
    if pid_op.attr(&AttrKey::Axis).and_then(|a| a.as_int()).unwrap_or(0) != 0 {
        return Err(refuse("tt.get_program_id is not axis x (0)"));
    }
    let pid = pid_op.result().ok_or_else(|| refuse("program id defines no value"))?;

    // Descriptor shapes/strides/blocks: contiguous row-major, canonical layout. B is
    // framed by the orientation: plain `[k, ns]` blocking `[bk, bn]`, or the `.T`
    // form's presented `[ns, k]` blocking `[bn, bk]`.
    check_desc(module, a_desc, "A", m, k, bm, bk)?;
    if b_trans.is_some() {
        check_desc(module, b_desc, "B", ns, k, bn, bk)?;
    } else {
        check_desc(module, b_desc, "B", k, ns, bk, bn)?;
    }
    let store = all
        .iter()
        .find(|o| o.kind == OpKind::TtDescriptorStore)
        .expect("counted one");
    let c_desc = descriptor_of(module, store)?;
    check_desc(module, c_desc, "C", m, ns, bm, bn)?;

    // Base-pointer provenance: the emitted DFIR has NO pointer arguments -- buffers
    // are bound POSITIONALLY -- so A/B/C MUST read args 0/1/2 IN THAT ORDER. That
    // also makes them pairwise distinct.
    let args: Vec<Ssa> = func.regions[0].args.iter().map(|(v, _)| *v).collect();
    if args.len() < 3 {
        return Err(refuse("kernel has fewer than 3 arguments (need distinct A/B/C buffers)"));
    }
    let base = |d: &Op| d.operands.first().copied();
    if base(a_desc) != Some(args[0])
        || base(b_desc) != Some(args[1])
        || base(c_desc) != Some(args[2])
    {
        return Err(refuse(
            "A/B/C descriptors do not read the function's first three arguments in order \
             (A=arg0, B=arg1, C=arg2); the emitter binds buffers positionally, so a \
             swapped/aliased base pointer would be silently miscompiled",
        ));
    }

    // Full-K accumulation loop (0..K step BK, zero-init) carrying the dot result.
    let enclosing_for = enclosing_scf_for(module, dot_path);
    let (k_iv, stored) = match enclosing_for {
        Some(for_path) => {
            let forr = walk::at(module, &for_path).expect("path");
            // iter_args start at operand 3 (lb, ub, step first); the region args are
            // [iv, ...iter_args].
            let iter_args: Vec<Ssa> =
                forr.regions[0].args[1..].iter().map(|(v, _)| *v).collect();
            let idx = iter_args
                .iter()
                .position(|v| *v == c)
                .ok_or_else(|| refuse("accumulator is not the enclosing scf.for iter-arg"))?;
            let yieldd = forr
                .regions[0]
                .ops
                .last()
                .filter(|o| o.kind == OpKind::ScfYield)
                .ok_or_else(|| refuse("the K-loop has no scf.yield"))?;
            if yieldd.operands.get(idx) != Some(&d) {
                return Err(refuse("K-loop does not carry the matmul result"));
            }
            let init = forr.operands[3 + idx];
            if !is_zero_const(module, init) {
                return Err(refuse(
                    "matmul accumulator is not zero-initialized (a bias/fused init would \
                     be dropped)",
                ));
            }
            let lo = const_int(module, forr.operands[0]);
            let hi = const_int(module, forr.operands[1]);
            let st = const_int(module, forr.operands[2]);
            if lo != Some(0) || st != Some(bk) || hi != Some(k) {
                return Err(refuse(format!(
                    "K-loop does not contract the full K: expected 0..{k} step {bk}"
                )));
            }
            // BK must divide K exactly, or the last tile over-reads past K.
            if bk == 0 || k % bk != 0 {
                return Err(refuse(format!(
                    "K ({k}) is not a multiple of the K-loop step BK ({bk}); the last \
                     tile over-reads"
                )));
            }
            (Some(forr.regions[0].args[0].0), forr.results[idx])
        }
        None => {
            if bk != k {
                return Err(refuse(format!(
                    "single-tile matmul does not contract the full K (tile K={bk} != full \
                     K={k})"
                )));
            }
            if !is_zero_const(module, c) {
                return Err(refuse("single-tile matmul accumulator is not zero"));
            }
            (None, d)
        }
    };

    // Canonical offsets: A[pid,k], B[k,pid], C[pid,pid]. The K index is the loop IV
    // (looped) OR a constant 0 (single tile at K-start 0). Checked in BOTH branches
    // -- the no-loop branch previously skipped it.
    let is_k_idx = |v: Ssa| match k_iv {
        Some(iv) => v == iv,
        None => const_int(module, v) == Some(0),
    };
    let a_idx = &a_ld.operands[1..];
    let b_idx = &b_ld.operands[1..];
    let s_idx = &store.operands[1..];
    if a_idx.len() != 2 || a_idx[0] != pid || !is_k_idx(a_idx[1]) {
        return Err(refuse("A load offset is not the canonical [pid, k]"));
    }
    // B's offset, framed by the orientation: plain B reads `[k, pid]` out of its
    // `[k, n]` region; the `.T` form loads its `[n, k]` slice at `[pid, k]` (an n ROW
    // selected by pid, k in place) and the trans produces the `[bk, bn]` operand.
    if b_trans.is_some() {
        if b_idx.len() != 2 || b_idx[0] != pid || !is_k_idx(b_idx[1]) {
            return Err(refuse("B load offset is not the canonical transposed [pid, k]"));
        }
    } else if b_idx.len() != 2 || !is_k_idx(b_idx[0]) || b_idx[1] != pid {
        return Err(refuse("B load offset is not the canonical [k, pid]"));
    }
    // The store's operands are [desc, indices..., src]; drop the trailing src.
    let s_idx = &s_idx[..s_idx.len().saturating_sub(1)];
    if s_idx.len() != 2 || s_idx[0] != pid || s_idx[1] != pid {
        return Err(refuse("C store offset is not the canonical [pid, pid]"));
    }

    // No extra control flow: the ONLY loop may be the K-loop.
    if n_for != usize::from(k_iv.is_some()) {
        return Err(refuse(
            "kernel has a loop other than the single K-loop; an outer loop would gate \
             the matmul/store yet be silently dropped",
        ));
    }
    // The store must sit at the function top level.
    if !func.regions[0].ops.iter().any(|o| std::ptr::eq(o, *store)) {
        return Err(refuse(
            "output store is not at the function top level (it is nested in an extra \
             loop/region that would be silently dropped)",
        ));
    }
    // The store's SOURCE must BE the matmul/loop result.
    if store.operands.last() != Some(&stored) {
        return Err(refuse(
            "the descriptor_store source is not the matmul result (a different tensor \
             is stored, so the matmul would be discarded)",
        ));
    }
    // No epilogue: the loop/dot result feeds ONLY the store, as its src.
    for op in &all {
        if op.operands.contains(&stored)
            && !(op.kind == OpKind::TtDescriptorStore && op.operands.last() == Some(&stored))
        {
            return Err(refuse(
                "matmul result is consumed by a non-store op (a fused epilogue would be \
                 silently dropped)",
            ));
        }
    }
    Ok(())
}

/// ⭐ THE fp8 W8A8 WHOLE-KERNEL CONTRACT (review P1.0, the fp8 arm).
///
/// The lowering this admits is `ktir-superdsc`'s arity-3 `matmul_oriented` door: A, W_fp8
/// and w_scale read from the node's FULL parameter list, with the device quantizing the
/// activation and applying BOTH dequant scales (`a_scale[m] · w_scale[n]`) whatever the
/// kernel says. That door never inspects the compute ops -- so THIS verifier is the only
/// thing that makes the kernel's spelled `* w_scale` binding rather than decorative: it
/// requires the dot's result to feed EXACTLY one `arith.mulf` whose other operand is the
/// w_scale LOAD (through at most a `tt.broadcast` -- the `[1, N]` row sprayed over the m
/// product rows, which is the `In::mb` the device's dequant states), and THAT mulf's result
/// to be the store's source.
///
/// ⛔ WHY THE mulf MUST BE SPELLED, on the record. The per-`Program` door is
/// op-blind (`emit_regions` reads only the region list and the B orientation), so a kernel
/// that forgot the multiply would still lower, still run, and still get the door's dequant --
/// computing the right function from a program that states a different one. It would also be
/// refused at the handoff for an unrelated reason (an unloaded `desc_ws` is DCE'd and
/// `every_parameter_states_its_width` red-stops on the hole), which is luck, not design.
/// This check replaces the luck: the epilogue the door assumes and the epilogue the kernel
/// spells are pinned together HERE.
///
/// The counts follow the f16 verifier's discipline -- the emitter is a shape-keyed TEMPLATE
/// that would silently drop anything it does not model, so exactly
/// `{1 tt.dot, 3 tt.descriptor_load, 1 arith.mulf, 1 tt.descriptor_store}` is admitted and
/// every deviation is refused by name.
fn verify_canonical_fp8_matmul_kernel(
    module: &Module,
    dot_path: &OpPath,
    b_load: Ssa,
) -> Result<()> {
    let dot = walk::at(module, dot_path).expect("path");
    let a = dot.operands[0];
    let c = dot.operands[2];
    let d = dot.result().ok_or_else(|| refuse("the dot defines no result"))?;

    // A is a DIRECT f16 load, as in the f16 contract.
    let a_ld = module.def_of(a).filter(|o| o.kind == OpKind::TtDescriptorLoad);
    let Some(a_ld) = a_ld else {
        return Err(refuse("A must be a direct tt.descriptor_load result (fp8 affects W only)"));
    };
    // B IS THE fp8 LOAD -- `run` peeled it and hands it in (`b_load`), because this runs
    // BEFORE the rewire and the dot's second operand is still the extf/trans chain. Its
    // VALUE type is f16 (the spelled `.to(tl.float16)`); its DESCRIPTOR's elem is what
    // says fp8, and the peel in `run` checked exactly that.
    let b_ld = module.def_of(b_load).filter(|o| o.kind == OpKind::TtDescriptorLoad);
    let Some(b_ld) = b_ld else {
        return Err(refuse(
            "the fp8 weight must reach the dot as a direct tt.descriptor_load (the widening \
             cast and the transpose are folded away by this pass)",
        ));
    };
    let b = b_load;

    let a_ty = module.type_of(a).ok_or_else(|| refuse("A has no type"))?;
    let b_ty = module.type_of(b).ok_or_else(|| refuse("B has no type"))?;
    let (bm, bk) = (a_ty.dims().unwrap()[0], a_ty.dims().unwrap()[1]);
    // ⭐ THE LOAD's OWN SHAPE, not the dot's post-transpose operand: `b` is the weight
    // load's result, whose layout is `[n, k]` (the `.T` that swapped the axes is on the
    // peeled chain). The descriptor and the block are checked against THESE extents.
    let bn = b_ty.dims().unwrap()[0];

    let a_desc = descriptor_of(module, a_ld)?;
    let b_desc = descriptor_of(module, b_ld)?;
    let (m, k) = desc_shape2(module, a_desc)
        .ok_or_else(|| refuse("A/B descriptor M/N/K are not compile-time constants"))?;
    let (w_r, w_c) = desc_shape2(module, b_desc)
        .ok_or_else(|| refuse("A/B descriptor M/N/K are not compile-time constants"))?;

    // ⛔ THE WEIGHT VIEW IS `[N, K]` AND THE KERNEL'S BLOCK IS THE WHOLE TILE. The fp8 door
    // reads W through `rb(&w_name, ...)` with extents it never checks on ITS side -- the
    // `w_k != k` guard in the f16 arm is what catches a mis-framed weight, and this is its
    // fp8 twin: the descriptor must state the checkpoint's `[out, in]` with `K` the second
    // extent, and the block must cover it (single-tile contraction, `BLOCK_K == K`).
    if w_r != bn || w_c != k {
        return Err(refuse(format!(
            "the fp8 weight descriptor is [{w_r}, {w_c}] but the contraction needs [{bn}, {k}] \
             (the [n, k] = [out_features, in_features] checkpoint layout; the host stages the \
             kernel slot's device order)"
        )));
    }
    if bm != m || bn != w_r {
        return Err(refuse("multi-block fp8 matmul is not supported (the emitter emits one \
             M x N program and discards the block size)"));
    }
    if bk != k {
        return Err(refuse(format!(
            "fp8 matmul does not tile K here: the tile K={bk} != the full K={k}; a K-looped \
             fp8 form is a follow-on, not this contract"
        )));
    }

    // Zero-init accumulator, as in the f16 contract (a bias/fused init would be dropped).
    if !is_zero_const(module, c) {
        return Err(refuse("fp8 matmul accumulator is not zero"));
    }

    // THE COUNTS. Exactly {1 dot, 3 loads, 1 mulf, 1 store}; structural ops only besides
    // those (the f16 contract's list plus the mulf and the broadcast that carries the
    // scale row).
    let func = module.kernel()?;
    let all = func.ops_deep();
    let count = |k: OpKind| all.iter().filter(|o| o.kind == k).count();
    let n_dot = count(OpKind::TtDot);
    let n_load = count(OpKind::TtDescriptorLoad);
    let n_store = count(OpKind::TtDescriptorStore);
    let n_mulf = count(OpKind::ArithMulf);
    let n_bcast = count(OpKind::TtBroadcast);
    let n_pid = count(OpKind::TtGetProgramId);
    let n_for = count(OpKind::ScfFor);

    for op in &all {
        let structural = matches!(
            op.kind,
            OpKind::TtDot
                | OpKind::TtDescriptorLoad
                | OpKind::TtDescriptorStore
                | OpKind::TtMakeTensorDescriptor
                | OpKind::TtGetProgramId
                | OpKind::ScfFor
                | OpKind::TtFunc
                | OpKind::TtReturn
                | OpKind::ScfYield
                | OpKind::ArithConstant
                | OpKind::ArithIndexCast
                | OpKind::ArithExtf
                | OpKind::TtTrans
                | OpKind::ArithMulf
                | OpKind::TtBroadcast
                | OpKind::ArithMuli
        );
        if !structural {
            return Err(refuse(format!(
                "fp8 kernel contains an operation the matmul template does not model ({:?}; \
                 only one dot, its three loads, the scale multiply and one store are \
                 representable)",
                op.kind
            )));
        }
    }
    if n_dot != 1 || n_load != 3 || n_store != 1 || n_mulf != 1 {
        return Err(refuse(format!(
            "fp8 kernel is not exactly {{1 tt.dot, 3 tt.descriptor_load, 1 arith.mulf, 1 \
             tt.descriptor_store}} (got dot={n_dot} load={n_load} mulf={n_mulf} \
             store={n_store}); the W8A8 door assumes the spelled dequant, so anything else \
             would be silently reinterpreted",
        )));
    }
    if n_bcast > 1 {
        return Err(refuse(format!(
            "fp8 kernel has {n_bcast} tt.broadcast ops; at most one (the [1, N] scale row \
             over the product rows) is part of this contract",
        )));
    }
    if n_pid != 1 {
        return Err(refuse("kernel does not have exactly one tt.get_program_id"));
    }
    let pid_op = all
        .iter()
        .find(|o| o.kind == OpKind::TtGetProgramId)
        .expect("counted one");
    if pid_op.attr(&AttrKey::Axis).and_then(|a| a.as_int()).unwrap_or(0) != 0 {
        return Err(refuse("tt.get_program_id is not axis x (0)"));
    }
    let pid = pid_op.result().ok_or_else(|| refuse("program id defines no value"))?;

    // No loops at all: this contract is the single-tile form (a K-loop is the f16 arm's
    // follow-on too).
    if n_for != 0 {
        return Err(refuse(
            "fp8 kernel has a loop; the single-tile contract admits none (a K-looped fp8 \
             form is a follow-on)",
        ));
    }

    // Descriptor shapes/strides: contiguous row-major, canonical layout -- A [m,k], W
    // [n,k], C [m,n].
    check_desc(module, a_desc, "A", m, k, bm, bk)?;
    check_desc(module, b_desc, "W", bn, k, bn, bk)?;
    let store = all
        .iter()
        .find(|o| o.kind == OpKind::TtDescriptorStore)
        .expect("counted one");
    let c_desc = descriptor_of(module, store)?;
    check_desc(module, c_desc, "C", m, bn, bm, bn)?;

    // ⭐⭐⭐ POSITIONAL BINDING: A=arg0, W=arg1, ws=arg2, C=arg3. The emitted DFIR binds
    // buffers positionally, and the fp8 door reads the THIRD input as the scale -- so a
    // swapped ws/C would silently dequant by the output and store into the scale row.
    // Both are f16 pointers, so nothing type-checks the difference; only position does.
    let args: Vec<Ssa> = func.regions[0].args.iter().map(|(v, _)| *v).collect();
    if args.len() != 4 {
        return Err(refuse(format!(
            "fp8 kernel takes {} arguments; the W8A8 door binds FOUR positionally (A, W_fp8, \
             w_scale, output) and a trailing hole or extra pointer would renumber every \
             buffer after it",
            args.len()
        )));
    }
    let base = |d: &Op| d.operands.first().copied();
    if base(a_desc) != Some(args[0]) || base(b_desc) != Some(args[1]) {
        return Err(refuse(
            "A/W descriptors do not read the function's first two arguments in order \
             (A=arg0, W=arg1); the emitter binds buffers positionally",
        ));
    }

    // ⭐ THE SCALE LOAD: the third descriptor_load, over arg2, shaped [1, N]. Its window is
    // the whole row (the door reads the buffer as `[1, n]`, byte-identical to the loaded
    // `[1, n]`). Identified by the VALUE IT DEFINES (`a` and `b_load` are those loads'
    // results), not by pointer: `def_of` and `ops_deep` borrow the same ops through
    // different paths.
    let ws_ld = all
        .iter()
        .find(|o| {
            o.kind == OpKind::TtDescriptorLoad
                && o.result() != Some(a)
                && o.result() != Some(b_load)
        })
        .expect("counted three loads");
    let ws_desc = descriptor_of(module, ws_ld)?;
    if base(ws_desc) != Some(args[2]) {
        return Err(refuse(
            "the scale load does not read arg2; the W8A8 door binds w_scale as the THIRD \
             buffer, positionally",
        ));
    }
    let (ws_r, ws_c) = desc_shape2(module, ws_desc)
        .ok_or_else(|| refuse("the scale descriptor has non-constant shape/strides"))?;
    if ws_r != 1 || ws_c != bn {
        return Err(refuse(format!(
            "the w_scale descriptor is [{ws_r}, {ws_c}] but the dequant needs the one row \
             [1, {bn}] (per output channel)",
        )));
    }
    if base(c_desc) != Some(args[3]) {
        return Err(refuse(
            "the output store does not read arg3; the W8A8 door binds the output as the \
             FOURTH buffer, positionally",
        ));
    }

    // Canonical offsets: A[pid,0], W[0,0], ws[0,0], C[pid,0] -- the single-tile form of the
    // f16 contract's [pid,k]/[k,pid] pair with the k index pinned at 0. The row offset is
    // the IDIOMATIC `offs_m = pid * BLOCK_M` (`arith.muli`), admitted with its own guard:
    // one operand must BE the pid and the other the constant BLOCK_M -- anything else
    // scales the row by a value the emitter never reads and would silently misplace.
    let is_zero_idx = |v: Ssa| const_int(module, v) == Some(0);
    let is_row_idx = |v: Ssa| -> bool {
        if v == pid {
            return true;
        }
        match module.def_of(v).filter(|o| o.kind == OpKind::ArithMuli) {
            Some(mu) => {
                (mu.operands[0] == pid && const_int(module, mu.operands[1]).is_some())
                    || (mu.operands[1] == pid && const_int(module, mu.operands[0]).is_some())
            }
            None => false,
        }
    };
    let a_idx = &a_ld.operands[1..];
    if a_idx.len() != 2 || !is_row_idx(a_idx[0]) || !is_zero_idx(a_idx[1]) {
        return Err(refuse("A load offset is not the canonical [pid*BLOCK_M, 0]"));
    }
    let b_idx = &b_ld.operands[1..];
    if b_idx.len() != 2 || !is_zero_idx(b_idx[0]) || !is_zero_idx(b_idx[1]) {
        return Err(refuse("W load offset is not the canonical [0, 0]"));
    }
    let ws_idx = &ws_ld.operands[1..];
    if ws_idx.len() != 2 || !is_zero_idx(ws_idx[0]) || !is_zero_idx(ws_idx[1]) {
        return Err(refuse("the scale load offset is not the canonical [0, 0]"));
    }
    // The store's operands are [desc, indices..., src]: skip the desc, drop the src.
    let s_idx = &store.operands[1..store.operands.len().saturating_sub(1)];
    if s_idx.len() != 2 || !is_row_idx(s_idx[0]) || !is_zero_idx(s_idx[1]) {
        return Err(refuse("C store offset is not the canonical [pid*BLOCK_M, 0]"));
    }

    // ⭐⭐⭐ THE mulf CHAIN: `mulf(dot_or_bcast(dot), bcast_or_load(ws))` -- either operand
    // order, at most one broadcast on the scale side. The mulf's OTHER operand must be the
    // dot's result (possibly through the same broadcast); anything else is a kernel
    // computing a different function than the door lowers.
    let mulf = all
        .iter()
        .find(|o| o.kind == OpKind::ArithMulf)
        .expect("counted one");
    let mulf_res = mulf.result().ok_or_else(|| refuse("the scale mulf defines no result"))?;
    // Trace one value back through at most one tt.broadcast.
    let through_bcast = |v: Ssa| -> Ssa {
        match module.def_of(v).filter(|o| o.kind == OpKind::TtBroadcast) {
            Some(bc) => bc.operands[0],
            None => v,
        }
    };
    let (m0, m1) = (mulf.operands[0], mulf.operands[1]);
    let dot_side = through_bcast(m0);
    let scale_side = through_bcast(m1);
    // The dot's result must be one side; the ws load's result the other (through their
    // broadcasts).
    let ws_res = ws_ld
        .result()
        .ok_or_else(|| refuse("the scale load defines no value"))?;
    let dot_ok = dot_side == d;
    let scale_ok = scale_side == ws_res;
    if !(dot_ok && scale_ok) {
        // Try the mirror order before refusing: `mulf(ws_bcast, dot)`.
        let dot_ok_m = through_bcast(m1) == d;
        let scale_ok_m = through_bcast(m0) == ws_res;
        if !(dot_ok_m && scale_ok_m) {
            return Err(refuse(
                "the arith.mulf does not multiply the dot result by the w_scale load -- the \
                 W8A8 door applies that dequant whatever the kernel says, so a kernel \
                 spelling a different epilogue is computing a different function and is \
                 refused rather than reinterpreted",
            ));
        }
    }

    // The store must sit at the function top level and its source must BE the mulf's
    // result.
    if !func.regions[0].ops.iter().any(|o| std::ptr::eq(o, *store)) {
        return Err(refuse(
            "output store is not at the function top level (it is nested in an extra \
             loop/region that would be silently dropped)",
        ));
    }
    if store.operands.last() != Some(&mulf_res) {
        return Err(refuse(
            "the descriptor_store source is not the scale-multiply result (the W8A8 door \
             dequantizes into the output, so storing anything else would silently discard \
             it)",
        ));
    }
    // No other compute reads the mulf or the dot's result.
    for op in &all {
        if op.kind == OpKind::TtDescriptorStore || std::ptr::eq(op, mulf) {
            continue;
        }
        if op.operands.contains(&mulf_res) {
            return Err(refuse(
                "the scale-multiply result is consumed by something other than the store",
            ));
        }
    }
    Ok(())
}

/// ⭐⭐⭐ RUNG 4 — THE PAGED/GATHERED MATMUL CONTRACT (the gathered-B arm).
///
/// `dot(A, gather(V))`: the contraction's K axis reads rows of V chosen AT RUNTIME by
/// an index vector loaded from device memory (rung 4 of the address-provenance
/// ladder). This is the V leg of paged attention (`v_desc.gather(ids, 0)` then
/// `tl.dot(p, v)`) and it is the form IBM's own paged attention lowers the same way:
/// the gathered rows are MATERIALIZED by a KERNEL-less identity copy into a contiguous
/// scratch and the matmul reads the scratch -- never a gather-carrying matmul, which
/// the device refuses at build time (`emit_sdsc` refuses a gather on a KERNEL-bearing
/// op because `hasDimensionReuse` would fire the reuse explorer on gathered addresses).
///
/// ⛔ THE GATHERED B IS CONTRACTED WHERE IT LIES. Gathered rows `[n, hd]` in ROW-BLOCKED
/// order are already the kernel slot's `[k, n]` residency (`StickLayout::addr_eq`:
/// "RowBlocked vs Kernel is the identical stick-blocked formula"; the shipped attention
/// contracts the V cache in place exactly so). So the matmul states NO maps of its own
/// -- MLIR's default `[(d0,d2), (d2,d1), (d0,d1)]` reads B as `[k, n]`, which is the
/// gathered rows with the row count as K. NO RELAYOUT, NO TRANSDUCE.
///
/// ⛔⛔ AND THAT IS WHY A `tt.trans` OVER THE GATHER IS REFUSED, NOT FOLDED. The K leg
/// (`tl.dot(q, k_rows.T)`) needs the gathered rows transposed, and nothing on this arch
/// can do it: a separate `Program::Transpose` is a relayout op, and dxp refuses those
/// on `SENARCH=MPW4` ("Implicit syncs not available for architectures prior to
/// RCUDD1A", measured at 16 cores AND at 1). The no-fold law at `run` already keeps
/// the trans as a real `linalg.transpose`; this contract refuses it BY NAME rather
/// than let the whole-function door meet it at its own guard. The K leg's answer is a
/// Kᵀ-ordered table presented by the host (the shipped attention's third physical
/// plane), which is a buffer-layout decision, not a kernel op.
///
/// The counts follow the sibling contracts' discipline -- the whole-function door is a
/// template that would silently drop anything it does not model, so exactly
/// `{1 tt.dot, 2 tt.descriptor_load (A, ids), 1 tt.descriptor_gather, 1
/// tt.descriptor_store}` is admitted and every deviation is refused by name.
fn verify_canonical_paged_matmul_kernel(
    module: &Module,
    dot_path: &OpPath,
    gathered_b: Ssa,
) -> Result<()> {
    let dot = walk::at(module, dot_path).expect("path");
    let a = dot.operands[0];
    let c = dot.operands[2];
    let d = dot.result().ok_or_else(|| refuse("the dot defines no result"))?;

    // A is a DIRECT f16 load, exactly as in the f16 contract.
    let a_ld = module.def_of(a).filter(|o| o.kind == OpKind::TtDescriptorLoad);
    let Some(a_ld) = a_ld else {
        return Err(refuse(
            "paged matmul: A must be a direct tt.descriptor_load result (the gather is B's)",
        ));
    };
    let gather = module
        .def_of(gathered_b)
        .filter(|o| o.kind == OpKind::TtDescriptorGather)
        .ok_or_else(|| refuse("paged matmul: B's gather is not a tt.descriptor_gather"))?;
    // ⛔ THE TRANS-CARRYING K LEG IS REFUSED HERE, BY NAME, with the arch reason. The
    // detection in `run` peeled an optional trans to find the gather; if one was there,
    // this is `tl.dot(q, k_rows.T)` and the transposition has no lowering on this arch.
    if dot.operands[1] != gathered_b {
        return Err(refuse(
            "paged matmul: the gathered B reaches the dot through a `tt.trans` (the K leg, \
             `tl.dot(q, k_rows.T)`). The gathered rows are contracted where they lie \
             ([n, head] row-blocked IS the kernel slot's [k, n] residency), so a \
             transposed gather needs a relayout -- and dxp refuses relayout ops on this \
             arch (\"Implicit syncs not available for architectures prior to RCUDD1A\"). \
             Present the table already Kᵀ-ordered on the host (the shipped attention's \
             third physical plane) and gather its rows, rather than transposing in the \
             kernel",
        ));
    }

    let a_ty = module.type_of(a).ok_or_else(|| refuse("A has no type"))?;
    let (bm, bk) = (a_ty.dims().unwrap()[0], a_ty.dims().unwrap()[1]);
    // The gather's result is `[rows, HEAD_DIM]` -- the contraction's K rows gathered,
    // each a full head. This is B's own shape: no trans sits between it and the dot.
    let g_ty = module
        .type_of(gathered_b)
        .ok_or_else(|| refuse("the gathered B has no type"))?;
    let g_dims = g_ty
        .dims()
        .ok_or_else(|| refuse("the gathered B is not a ranked tensor"))?;
    if g_dims.len() != 2 {
        return Err(refuse(format!(
            "paged matmul: the gathered B is rank {} (a 2-D rows x head-dim tile is the \
             form the slot's [k, n] residency addresses)",
            g_dims.len()
        )));
    }
    let (rows, hd) = (g_dims[0], g_dims[1]);

    let a_desc = descriptor_of(module, a_ld)?;
    let v_desc = descriptor_of(module, gather)?;
    let (m, a_cols) = desc_shape2(module, a_desc)
        .ok_or_else(|| refuse("A descriptor M/K are not compile-time constants"))?;
    // The V table is `[V, HEAD_DIM]`: V rows in the pool, HEAD_DIM the contraction's N.
    let (v_full, head_dim) = desc_shape2(module, v_desc)
        .ok_or_else(|| refuse("the gather's descriptor shape is not compile-time"))?;
    if head_dim != hd {
        return Err(refuse(format!(
            "paged matmul: the V table is [{v_full}, {head_dim}] but the gathered rows are \
             [{rows}, {hd}] — each gathered row is one full table row",
        )));
    }
    if bm != m || bk != a_cols {
        return Err(refuse(format!(
            "paged matmul: A's block [{bm}, {bk}] does not cover the whole tile [{m}, \
             {a_cols}] (single-tile only; the emitter discards the block size)",
        )));
    }
    // ⭐ THE CONTRACTION: A's columns are the gathered ROW COUNT (K = rows), and the dot's
    // result is [m, head_dim]. This is the plain-B framing — B as [k, n] with k = the
    // row count — and it is checked, not assumed, because a [m, hd] A against a [rows,
    // hd] gather would contract the wrong axis with matching element types and no shape
    // error anywhere.
    let d_ty = dot
        .result_type()
        .ok_or_else(|| refuse("the dot defines no result type"))?;
    let d_dims = d_ty
        .dims()
        .ok_or_else(|| refuse("the dot result is not a ranked tensor"))?;
    if a_cols != rows {
        return Err(refuse(format!(
            "paged matmul: A is [{m}, {a_cols}] but {rows} rows were gathered — the \
             contraction's K is the gathered row count (plain-B), so A's columns must \
             equal it",
        )));
    }
    if d_dims.len() != 2 || d_dims[0] != m || d_dims[1] != hd {
        return Err(refuse(format!(
            "paged matmul: the dot's result is {d_dims:?}, not the [m, head] = [{m}, {hd}] \
             the plain-B contraction produces",
        )));
    }

    // Zero-init accumulator, as in both sibling contracts (a bias/fused init would be
    // silently dropped).
    if !is_zero_const(module, c) {
        return Err(refuse("paged matmul accumulator is not zero"));
    }

    // THE COUNTS: exactly {1 dot, 2 loads, 1 gather, 1 store} — and NO trans. The ids
    // load is the second load; the V table is NOT loaded directly (that is the whole
    // point -- a direct `v_desc.load` beside the gather would be the wrong-rows trap
    // the whole-function door's gather guard exists to refuse, and this counts gate
    // catches it here first, by name).
    let func = module.kernel()?;
    let all = func.ops_deep();
    let count = |k: OpKind| all.iter().filter(|o| o.kind == k).count();
    let n_dot = count(OpKind::TtDot);
    let n_load = count(OpKind::TtDescriptorLoad);
    let n_gather = count(OpKind::TtDescriptorGather);
    let n_trans = count(OpKind::TtTrans);
    let n_store = count(OpKind::TtDescriptorStore);

    for op in &all {
        let structural = matches!(
            op.kind,
            OpKind::TtDot
                | OpKind::TtDescriptorLoad
                | OpKind::TtDescriptorStore
                | OpKind::TtDescriptorGather
                | OpKind::TtMakeTensorDescriptor
                | OpKind::TtGetProgramId
                | OpKind::ScfFor
                | OpKind::TtFunc
                | OpKind::TtReturn
                | OpKind::ScfYield
                | OpKind::ArithConstant
                | OpKind::ArithIndexCast
        );
        if !structural {
            return Err(refuse(format!(
                "paged matmul kernel contains an operation the template does not model \
                 ({:?}; only one dot, its two loads, the gather and one store are \
                 representable)",
                op.kind
            )));
        }
    }
    if n_dot != 1 || n_load != 2 || n_gather != 1 || n_trans != 0 || n_store != 1 {
        return Err(refuse(format!(
            "paged matmul kernel is not exactly {{1 tt.dot, 2 tt.descriptor_load, 1 \
             tt.descriptor_gather, 1 tt.descriptor_store}} and no tt.trans (got \
             dot={n_dot} load={n_load} gather={n_gather} trans={n_trans} \
             store={n_store}); anything else would be silently dropped",
        )));
    }

    // ⭐ THE INDEX LOAD: the second `tt.descriptor_load`, whose result IS the gather's
    // `x_offsets` operand. Identified by the value it defines (the same discipline as the
    // fp8 arm's scale load), not by pointer: `def_of` and `ops_deep` borrow the same ops
    // through different paths.
    let ids_ld = all
        .iter()
        .find(|o| o.kind == OpKind::TtDescriptorLoad && o.result() != Some(a))
        .expect("counted two loads");
    if gather.operands.get(1).copied() != ids_ld.result() {
        return Err(refuse(
            "paged matmul: the gather's x_offsets is not the ids load's result (the ids \
             must be read through a descriptor in the same function; a computed index \
             vector has no view to point the indirect tile at)",
        ));
    }

    // Descriptor shapes/strides: contiguous row-major -- A [m, rows], V [V, head], C
    // [m, head]. The gather's block is one row (`[1, head]`), which the descriptor's own
    // block type states.
    check_desc(module, a_desc, "A", m, a_cols, bm, bk)?;
    check_desc(module, v_desc, "V", v_full, head_dim, 1, head_dim)?;
    let store = all
        .iter()
        .find(|o| o.kind == OpKind::TtDescriptorStore)
        .expect("counted one");
    let c_desc = descriptor_of(module, store)?;
    check_desc(module, c_desc, "C", m, hd, bm, hd)?;

    // ⭐⭐⭐ POSITIONAL BINDING: A=arg0, V=arg1, ids=arg2, O=arg3. The emitted DFIR binds
    // buffers positionally and the whole-function door numbers its windows by parameter
    // position, so a swapped ids/V would gather rows out of the ids buffer's own bytes --
    // both are pointers, only position says which is which.
    let args: Vec<Ssa> = func.regions[0].args.iter().map(|(v, _)| *v).collect();
    if args.len() != 4 {
        return Err(refuse(format!(
            "paged matmul kernel takes {} arguments; the door binds FOUR positionally \
             (A, V, ids, output) and a trailing hole or extra pointer would renumber \
             every buffer after it",
            args.len()
        )));
    }
    let base = |d: &Op| d.operands.first().copied();
    let ids_desc = descriptor_of(module, ids_ld)?;
    if base(a_desc) != Some(args[0])
        || base(v_desc) != Some(args[1])
        || base(ids_desc) != Some(args[2])
        || base(c_desc) != Some(args[3])
    {
        return Err(refuse(
            "A/V/ids/O descriptors do not read the function's four arguments in order \
             (A=arg0, V=arg1, ids=arg2, O=arg3); the emitter binds buffers positionally, \
             so a swapped/aliased base pointer would be silently miscompiled",
        ));
    }
    // The ids are a 1-D contiguous i32 vector of exactly the gathered-row count.
    let ids_ty = ids_ld
        .result_type()
        .ok_or_else(|| refuse("the ids load defines no type"))?;
    if ids_ty.elem() != Some(DType::I32) {
        return Err(refuse("paged matmul: the ids load's element is not i32 (the indirect tile indexes \
             by a 32-bit row number)".to_string()));
    }
    let ids_dims = ids_ty
        .dims()
        .ok_or_else(|| refuse("the ids load is not a ranked tensor"))?;
    if ids_dims.len() != 1 || ids_dims[0] != rows {
        return Err(refuse(format!(
            "paged matmul: the ids tile is {ids_dims:?}, not the one-D [{rows}] vector of \
             gathered-row indices",
        )));
    }

    // Canonical offsets, the single-tile grid-1 form: every load/store/gather corner is
    // the constant 0. A pid-scaled offset is the f16/fp8 arms' form and admits a follow-
    // on; here a nonzero corner would be silently dropped by the whole-function door.
    let is_zero_idx = |v: Ssa| const_int(module, v) == Some(0);
    let a_idx = &a_ld.operands[1..];
    if a_idx.len() != 2 || !is_zero_idx(a_idx[0]) || !is_zero_idx(a_idx[1]) {
        return Err(refuse("paged matmul: A load offset is not the canonical [0, 0]"));
    }
    let ids_idx = &ids_ld.operands[1..];
    if ids_idx.len() != 1 || !is_zero_idx(ids_idx[0]) {
        return Err(refuse("paged matmul: ids load offset is not the canonical [0]"));
    }
    // The gather's own y_offset is operand 2 (desc, x_offsets, y_offset).
    if gather.operands.len() != 3 || !is_zero_idx(gather.operands[2]) {
        return Err(refuse("paged matmul: the gather's y_offset is not the constant 0"));
    }
    // The store's operands are [desc, indices..., src]; drop the desc and the src.
    let s_idx = &store.operands[1..store.operands.len().saturating_sub(1)];
    if s_idx.len() != 2 || !is_zero_idx(s_idx[0]) || !is_zero_idx(s_idx[1]) {
        return Err(refuse("paged matmul: C store offset is not the canonical [0, 0]"));
    }

    // The store must sit at the function top level and its source must BE the dot's
    // result -- a gather result stored would discard the contraction.
    if !func.regions[0].ops.iter().any(|o| std::ptr::eq(o, *store)) {
        return Err(refuse(
            "paged matmul: output store is not at the function top level (it is nested \
             in an extra loop/region that would be silently dropped)",
        ));
    }
    if store.operands.last() != Some(&d) {
        return Err(refuse(
            "paged matmul: the descriptor_store source is not the dot result",
        ));
    }
    // No other compute reads the dot's result or the gather's result: the chain is
    // load-gather-dot-store and nothing else.
    for op in &all {
        if op.kind == OpKind::TtDescriptorStore {
            continue;
        }
        if op.operands.contains(&d) {
            return Err(refuse(
                "paged matmul: the dot result is consumed by something other than the store",
            ));
        }
        if op.operands.contains(&gathered_b) && !std::ptr::eq(*op, dot) {
            return Err(refuse(
                "paged matmul: the gather result is consumed by something other than the \
                 dot",
            ));
        }
    }
    Ok(())
}

/// A contiguous row-major 2-D `tt.make_tensor_descriptor` of the given full shape
/// and block. The emitter derives the layout from the shape ALONE and assumes
/// contiguous row-major, so a non-standard stride would be silently ignored.
fn check_desc(
    module: &Module,
    d: &Op,
    nm: &str,
    d0: i64,
    d1: i64,
    b0: i64,
    b1: i64,
) -> Result<()> {
    let (s0, s1) = desc_shape2(module, d)
        .ok_or_else(|| refuse(format!("{nm} has non-constant shape/strides")))?;
    if s0 != d0 || s1 != d1 {
        return Err(refuse(format!(
            "{nm} shape [{s0},{s1}] != expected [{d0},{d1}]"
        )));
    }
    let (t0, t1) = desc_strides2(module, d)
        .ok_or_else(|| refuse(format!("{nm} has non-constant shape/strides")))?;
    if t0 != d1 || t1 != 1 {
        return Err(refuse(format!(
            "{nm} is not contiguous row-major (strides [{t0},{t1}] != [{d1},1]) -- a \
             non-standard stride would be silently ignored"
        )));
    }
    let blk = d
        .result_type()
        .and_then(|t| t.dims().map(|x| x.to_vec()))
        .ok_or_else(|| refuse(format!("{nm} is not 2-D")))?;
    if blk.len() != 2 || blk[0] != b0 || blk[1] != b1 {
        return Err(refuse(format!("{nm} block shape mismatch")));
    }
    Ok(())
}

/// The `tt.make_tensor_descriptor` behind an access op's `desc` operand.
fn descriptor_of<'m>(module: &'m Module, access: &Op) -> Result<&'m Op> {
    let desc = access
        .operands
        .first()
        .copied()
        .ok_or_else(|| refuse("an access op with no descriptor operand"))?;
    module
        .def_of(desc)
        .filter(|o| o.kind == OpKind::TtMakeTensorDescriptor)
        .ok_or_else(|| refuse("A/B descriptors are not 2-D tt.make_tensor_descriptor"))
}

/// `tt.make_tensor_descriptor %base, [%s0, %s1], [%t0, %t1]` -- the operands are
/// base, then the shape values, then the stride values, so the split is at the
/// declared rank (from the block type).
fn desc_shape_strides(module: &Module, d: &Op) -> Option<(Vec<i64>, Vec<i64>)> {
    let rank = d.result_type()?.rank();
    if rank == 0 || d.operands.len() < 1 + 2 * rank {
        return None;
    }
    let mut shape = Vec::new();
    for i in 0..rank {
        shape.push(const_int(module, d.operands[1 + i])?);
    }
    let mut strides = Vec::new();
    for i in 0..rank {
        strides.push(const_int(module, d.operands[1 + rank + i])?);
    }
    Some((shape, strides))
}

fn desc_shape2(module: &Module, d: &Op) -> Option<(i64, i64)> {
    let (s, _) = desc_shape_strides(module, d)?;
    if s.len() != 2 {
        return None;
    }
    Some((s[0], s[1]))
}

fn desc_strides2(module: &Module, d: &Op) -> Option<(i64, i64)> {
    let (_, t) = desc_shape_strides(module, d)?;
    if t.len() != 2 {
        return None;
    }
    Some((t[0], t[1]))
}

/// Compile-time int from an `arith.constant` SSA value. `constInt`.
pub fn const_int(module: &Module, v: Ssa) -> Option<i64> {
    module
        .def_of(v)
        .filter(|o| o.kind == OpKind::ArithConstant)
        .and_then(|o| o.attr(&AttrKey::Value))
        .and_then(|a| a.as_int())
}

/// True iff `v` is a compile-time zero (splat dense-fp or scalar float 0.0).
/// `isZeroConst`.
pub fn is_zero_const(module: &Module, v: Ssa) -> bool {
    module
        .def_of(v)
        .filter(|o| o.kind == OpKind::ArithConstant)
        .and_then(|o| o.attr(&AttrKey::Value))
        .map(|a| match a {
            Attr::SplatFloat(f) | Attr::Float(f) => f.is_zero(),
            Attr::Int(i) => *i == 0,
            _ => false,
        })
        .unwrap_or(false)
}

/// The innermost enclosing `scf.for`, or `None`.
fn enclosing_scf_for(module: &Module, path: &OpPath) -> Option<OpPath> {
    let mut p = path.parent();
    while let Some(cur) = p {
        if walk::at(module, &cur).map(|o| o.kind == OpKind::ScfFor).unwrap_or(false) {
            return Some(cur);
        }
        p = cur.parent();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::parse;

    /// RUNG 4 — the paged V-leg fixture's own shape, as `make_ttir` prints it (the pod's
    /// `TTIR_DUMP=1` of `test/fixtures/paged_vmatmul.py`, SSA names shortened). The
    /// SCRATCHY FORM: `dot(p, v_rows)` with the rows gathered directly — contracted
    /// where they lie, no transpose anywhere. `K = 64` gathered rows because TWO
    /// floors meet there: the frontend requires `tl.dot`'s K >= 16, and the vendor
    /// matmul emitter refuses a K that is not a whole 64-element stick
    /// ("K={k} not a multiple of the 64-fp16 stick" — `assemble_matmul` emits no
    /// `coordinateMasking_`), so 64 is the smallest K both doors admit. Shared by
    /// the gathered-arm tests: the positive asserts the lowering and the tag, the
    /// controls each break ONE load-bearing fact and assert the refusal that
    /// names it.
    const PAGED: &str = "\
module {
  tt.func public @paged_vmatmul_fwd(%desc_p: !tt.ptr<f16>, %desc_v: !tt.ptr<f16>, %desc_ids: !tt.ptr<i32>, %desc_o: !tt.ptr<f16>) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<8x64xf16>
    %z = arith.constant 0 : i32
    %v_full = arith.constant 128 : i32
    %m = arith.constant 8 : i32
    %k = arith.constant 64 : i32
    %head = arith.constant 64 : i32
    %head64 = arith.constant 64 : i64
    %one = arith.constant 1 : i64
    %stride64 = arith.constant 64 : i64
    %p_desc = tt.make_tensor_descriptor %desc_p, [%m, %k], [%stride64, %one] : <f16>, <8x64xf16>
    %v_desc = tt.make_tensor_descriptor %desc_v, [%v_full, %head], [%head64, %one] : <f16>, <1x64xf16>
    %ids_desc = tt.make_tensor_descriptor %desc_ids, [%k], [%one] : <i32>, <64xsi32>
    %o_desc = tt.make_tensor_descriptor %desc_o, [%m, %head], [%stride64, %one] : <f16>, <8x64xf16>
    %p = tt.descriptor_load %p_desc[%z, %z] : !tt.tensordesc<8x64xf16> -> tensor<8x64xf16>
    %ids = tt.descriptor_load %ids_desc[%z] : !tt.tensordesc<64xsi32> -> tensor<64xi32>
    %rows = tt.descriptor_gather %v_desc[%ids, %z] : (!tt.tensordesc<1x64xf16>, tensor<64xi32>, i32) -> tensor<64x64xf16>
    %o = tt.dot %p, %rows, %cst : tensor<8x64xf16>
    tt.descriptor_store %o_desc[%z, %z], %o : !tt.tensordesc<8x64xf16>, tensor<8x64xf16>
    tt.return
  }
}
";

    /// The fixture lowers: the dot becomes `linalg.matmul` with NO maps of its own
    /// (MLIR's default `[k, n]` framing — the gathered rows contracted where they lie),
    /// NO trans is present, and both the matmul and the module carry the trust tag the
    /// emitter requires.
    #[test]
    fn the_paged_v_leg_fixture_lowers_and_is_tagged() {
        let mut m = parse::parse(PAGED).unwrap();
        run(&mut m).expect("the paged V-leg fixture must lower, not be refused");
        let c = m.census();
        let get = |n: &str| c.iter().find(|(k, _)| k == n).map(|(_, v)| *v).unwrap_or(0);
        assert_eq!(get("linalg.matmul"), 1, "the dot lowered");
        assert_eq!(get("tt.dot"), 0, "no dot survives");
        assert_eq!(get("tt.trans"), 0, "no trans -- the V leg is contracted where it lies");
        assert!(
            m.attrs.iter().any(|(k, _)| *k == AttrKey::CanonicalVerified),
            "the module carries the trust tag"
        );
        for op in m.ops_deep() {
            if op.kind == OpKind::LinalgMatmul {
                assert!(
                    op.attr(&AttrKey::IndexingMaps).is_none(),
                    "the matmul states NO maps -- a plain-B contraction over the gathered \
                     rows, never a folded transpose-B"
                );
                assert!(
                    op.attr(&AttrKey::CanonicalVerified).is_some(),
                    "the matmul is tagged"
                );
            }
        }
    }

    /// CONTROL 1 — the wrong-rows trap: a SECOND, direct load of the V table beside the
    /// gather. The counts gate must refuse it by name, because a program that both
    /// gathers and directly loads the table it gathers from is one stray rewire away
    /// from the direct read the whole-function door's own guard exists to refuse.
    #[test]
    fn a_direct_v_load_beside_the_gather_is_refused_by_the_counts() {
        let src = PAGED.replacen(
            "    %ids = tt.descriptor_load %ids_desc[%z] : !tt.tensordesc<64xsi32> -> tensor<64xi32>",
            "    %ids = tt.descriptor_load %ids_desc[%z] : !tt.tensordesc<64xsi32> -> tensor<64xi32>\n    %stray = tt.descriptor_load %v_desc[%z, %z] : !tt.tensordesc<1x64xf16> -> tensor<1x64xf16>",
            1,
        );
        let mut m = parse::parse(&src).unwrap();
        let e = run(&mut m).unwrap_err();
        assert!(
            e.message.contains("paged matmul kernel is not exactly"),
            "the counts refusal must fire -- got {e}"
        );
    }

    /// CONTROL 2 — a computed index vector: the gather's x_offsets is NOT the ids load's
    /// result. There is no memory view for the indirect tile to point at, so this is
    /// refused by name rather than pointed at some invented view.
    #[test]
    fn a_gather_whose_offsets_are_not_the_ids_load_is_refused() {
        let src = PAGED.replacen(
            "    %rows = tt.descriptor_gather %v_desc[%ids, %z]",
            "    %rows = tt.descriptor_gather %v_desc[%p, %z]",
            1,
        );
        let mut m = parse::parse(&src).unwrap();
        let e = run(&mut m).unwrap_err();
        assert!(
            e.message.contains("the gather's x_offsets is not the ids load's result"),
            "the offsets refusal must fire -- got {e}"
        );
    }

    /// CONTROL 3 — THE K LEG, refused with the arch reason. `tl.dot(q, k_rows.T)` (the
    /// score leg) needs the gathered rows transposed, and nothing on this arch can do
    /// it: a relayout op is dxp-refused ("Implicit syncs not available for
    /// architectures prior to RCUDD1A"), and scratchy's shipped attention keeps a third
    /// physical Kᵗ plane precisely so its kernels never transpose in-graph. This is the
    /// ORIGINAL rung-4 fixture's form (`paged_score.py`, `tl.dot(q, rows.T)`), kept as
    /// the control so the refusal it measured on the pod stays asserted.
    #[test]
    fn the_k_leg_transposed_gather_is_refused_with_the_arch_reason() {
        // At K=64=HEAD_DIM the K-leg shape is square, so only the trans itself differs
        // from the positive: the same gather, its result handed to the dot through a
        // `tt.trans`.
        let src = PAGED
            .replacen(
                "    %rows = tt.descriptor_gather %v_desc[%ids, %z] : (!tt.tensordesc<1x64xf16>, tensor<64xi32>, i32) -> tensor<64x64xf16>",
                "    %rows = tt.descriptor_gather %v_desc[%ids, %z] : (!tt.tensordesc<1x64xf16>, tensor<64xi32>, i32) -> tensor<64x64xf16>\n    %rowsT = tt.trans %rows : tensor<64x64xf16> {order = array<i32: 1, 0>}",
                1,
            )
            .replacen(
                "    %o = tt.dot %p, %rows, %cst : tensor<8x64xf16>",
                "    %o = tt.dot %p, %rowsT, %cst : tensor<8x64xf16>",
                1,
            );
        let mut m = parse::parse(&src).unwrap();
        let e = run(&mut m).unwrap_err();
        assert!(
            e.message.contains("reaches the dot through a `tt.trans`"),
            "the K-leg refusal must fire -- got {e}"
        );
        assert!(
            e.message.contains("RCUDD1A"),
            "the refusal must state the ARCH reason, not a style preference -- got {e}"
        );
    }

    /// CONTROL 4 — swapped bindings: ids and V read each other's arguments. Both are
    /// pointers, so nothing type-checks the difference; only the positional check can
    /// catch a program that would gather rows out of the ids buffer's own bytes.
    #[test]
    fn a_swapped_v_and_ids_binding_is_refused_by_position() {
        let src = PAGED
            .replacen(
                "tt.make_tensor_descriptor %desc_v, [%v_full, %head]",
                "tt.make_tensor_descriptor %desc_ids, [%v_full, %head]",
                1,
            )
            .replacen(
                "tt.make_tensor_descriptor %desc_ids, [%k]",
                "tt.make_tensor_descriptor %desc_v, [%k]",
                1,
            );
        let mut m = parse::parse(&src).unwrap();
        let e = run(&mut m).unwrap_err();
        assert!(
            e.message.contains("A/V/ids/O descriptors do not read the function's four arguments"),
            "the positional refusal must fire -- got {e}"
        );
    }

    /// The attention shape: FOUR dots, so the whole-kernel contract must NOT run
    /// and every dot must still lower -- untagged.
    #[test]
    fn a_multi_dot_kernel_lowers_every_dot_and_is_left_untagged() {
        let src = "\
module {
  tt.func public @attn(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %z = arith.constant dense<0.000000e+00> : tensor<64x64xf16>
    %a = arith.constant dense<1.000000e+00> : tensor<64x128xf16>
    %b = arith.constant dense<1.000000e+00> : tensor<128x64xf16>
    %d0 = tt.dot %a, %b, %z : tensor<64x128xf16> * tensor<128x64xf16> -> tensor<64x64xf16>
    %d1 = tt.dot %a, %b, %z : tensor<64x128xf16> * tensor<128x64xf16> -> tensor<64x64xf16>
    tt.return
  }
}
";
        let mut m = parse::parse(src).unwrap();
        run(&mut m).expect("two dots must lower, not be refused");
        let c = m.census();
        let get = |n: &str| c.iter().find(|(k, _)| k == n).map(|(_, v)| *v).unwrap_or(0);
        assert_eq!(get("linalg.matmul"), 2, "both dots lowered");
        assert_eq!(get("tt.dot"), 0, "no dot survives");
        assert!(
            !m.attrs.iter().any(|(k, _)| *k == AttrKey::CanonicalVerified),
            "a multi-dot kernel is NOT tagged -- the emitter refuses it there"
        );
        for op in m.ops_deep() {
            assert!(
                op.attr(&AttrKey::CanonicalVerified).is_none(),
                "no matmul is tagged in a multi-dot kernel"
            );
        }
    }

    #[test]
    fn a_forged_incoming_tag_is_stripped_before_anything_else() {
        // An IRSource author can hand-write the attribute; it must not survive.
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %z = arith.constant dense<0.000000e+00> : tensor<64x64xf16>
    %a = arith.constant dense<1.000000e+00> : tensor<64x128xf16>
    %b = arith.constant dense<1.000000e+00> : tensor<128x64xf16>
    %d0 = tt.dot %a, %b, %z {spyre.canonical_verified} : tensor<64x128xf16> * tensor<128x64xf16> -> tensor<64x64xf16>
    %d1 = tt.dot %a, %b, %z : tensor<64x128xf16> * tensor<128x64xf16> -> tensor<64x64xf16>
    tt.return
  }
}
";
        let mut m = parse::parse(src).unwrap();
        run(&mut m).unwrap();
        for op in m.ops_deep() {
            assert!(
                op.attr(&AttrKey::CanonicalVerified).is_none(),
                "the forged tag must be gone"
            );
        }
    }

    #[test]
    fn an_f32_accumulator_is_refused_by_name() {
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %z = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
    %a = arith.constant dense<1.000000e+00> : tensor<64x128xf16>
    %b = arith.constant dense<1.000000e+00> : tensor<128x64xf16>
    %d0 = tt.dot %a, %b, %z : tensor<64x128xf16> * tensor<128x64xf16> -> tensor<64x64xf32>
    %d1 = tt.dot %a, %b, %z : tensor<64x128xf16> * tensor<128x64xf16> -> tensor<64x64xf32>
    tt.return
  }
}
";
        let mut m = parse::parse(src).unwrap();
        let e = run(&mut m).unwrap_err();
        assert!(e.message.contains("only f16 tt.dot is supported"), "got {e}");
    }

    #[test]
    fn a_single_dot_that_is_not_a_descriptor_load_is_refused_by_name() {
        // The exact refusal `swiglu_mlp.py`'s delta 3 quotes.
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %z = arith.constant dense<0.000000e+00> : tensor<64x64xf16>
    %a = arith.constant dense<1.000000e+00> : tensor<64x128xf16>
    %b = arith.constant dense<1.000000e+00> : tensor<128x64xf16>
    %d0 = tt.dot %a, %b, %z : tensor<64x128xf16> * tensor<128x64xf16> -> tensor<64x64xf16>
    tt.return
  }
}
";
        let mut m = parse::parse(src).unwrap();
        let e = run(&mut m).unwrap_err();
        assert!(
            e.message.contains("A and B must be direct tt.descriptor_load results"),
            "got {e}"
        );
    }
}
