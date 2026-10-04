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

//! Ported from `KTDF/Transforms/PlanCorelets.cpp` (SP-E3-03). The KTIR -> KTDF
//! stage: which coordination pattern the two corelets use, and their fields.
//!
//! ```text
//!   linalg.matmul / tt.dot        -> independent_subtile
//!   linalg.reduce / tt.reduce     -> partial_combine
//!   elementwise arith.*/math.*    -> split
//!   matmul AND a last-axis reduce -> independent_rows
//! ```
//!
//! THE PASS OWNS THE HARDWARE KNOWLEDGE (the 1p0 constants) so the emitter stays
//! generation-agnostic: the emitter READS the plan rather than re-deriving the
//! split.
//!
//! EVERY BOUND IS DERIVED FROM THE LOWERED KTIR, never a baked-in literal -- the
//! stick count from `ktdp.construct_access_tile`, N from the `linalg.matmul`
//! output, the row count from the first reduction's operand. A value that cannot
//! be derived is a RED-stop, because baking in a magic number is exactly the bug.
//!
//! WHERE `vector_add` DIES, and it is not here: this pass PLANS it as `split` with
//! one stick, giving `[0, 0]` and `[0, 1]`, and the `ktdf.corelet_plan` VERIFIER
//! then refuses those bounds by name. The refusal is reproduced in [`verify_plan`]
//! for that reason -- a port that accepts `vector_add` is wrong.

use crate::ir::*;
use crate::passes::distribute_work::work_loops;
use crate::passes::walk::{self, OpPath};
use crate::{Refusal, Result};

const PASS: &str = "PlanCorelets";

//===----------------------------------------------------------------------===//
// 1p0 hardware constants -- owned by the planning pass.
//===----------------------------------------------------------------------===//

/// 1p0 has exactly two corelets per compute tile.
const NUM_CORELETS: i64 = 2;
/// A stick is 64 elements wide.
const STICK_WIDTH: i64 = 64;
/// Each corelet has its OWN 8 PT rows -- the same inclusive [0, 7] range while
/// they compute different `output_partition`s.
const MATMUL_PT_ROWS_HI: i64 = 7;
const MATMUL_XRF_CAPACITY_KB: i64 = 64;
/// CL0 reduces its partition and SENDS its partial; CL1 reduces, RECEIVES, and
/// combines -- so CL1 produces the final result. `ccw` is the 1p0 SFPRING
/// direction. These are the pass's choices so the emitter never hardcodes "CL1
/// always combines".
const ROLE_PARTIAL: &str = "partial";
const ROLE_COMBINE: &str = "combine";
const RING_DIRECTION: &str = "ccw";

/// The coordination pattern, as a type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pattern {
    /// An INDIVISIBLE tile -- fewer sticks than corelets -- assigned WHOLE to corelet 0.
    ///
    /// Never returned by `classify_region`: it is a RE-PATTERNING of `Split` or
    /// `PartialCombine` once the stick count is known, exactly as the C++ does it
    /// (`PlanCorelets.cpp:744`).
    SingleCorelet,
    Split,
    IndependentSubtile,
    PartialCombine,
    IndependentRows,
}

impl Pattern {
    pub fn spelling(self) -> &'static str {
        match self {
            Pattern::SingleCorelet => "single_corelet",
            Pattern::Split => "split",
            Pattern::IndependentSubtile => "independent_subtile",
            Pattern::PartialCombine => "partial_combine",
            Pattern::IndependentRows => "independent_rows",
        }
    }
}

pub fn run(module: &mut Module) -> Result<()> {
    // Collect the work loops first, then plan, so insertion does not perturb the
    // walk.
    for path in work_loops(module) {
        plan_loop(module, &path)?;
    }
    Ok(())
}

/// Already-planned loops carry a `ktdf.corelet_plan` in their body. Idempotency:
/// skip them so a second run does not insert a duplicate plan.
fn already_planned(module: &Module, path: &OpPath) -> bool {
    walk::at(module, path)
        .map(|l| l.ops_deep().iter().any(|o| o.kind == OpKind::KtdfCoreletPlan))
        .unwrap_or(false)
}

fn plan_loop(module: &mut Module, path: &OpPath) -> Result<()> {
    if already_planned(module, path) {
        return Ok(());
    }
    let loopp = walk::at(module, path).expect("path").clone();

    let pattern = classify_region(module, &loopp).ok_or_else(|| {
        Refusal::new(
            PASS,
            "cannot classify the per-core work region -- it is either (a) COMPOSITE, \
             mixing a matmul/dot with a reduction in one fused body (attention: QK^T + \
             masked online softmax + P*V), which none of the 3 hardcoded 1p0 patterns \
             coordinates, or (b) not tile compute at all: neither linalg.matmul/tt.dot \
             (independent_subtile), linalg.reduce/tt.reduce (partial_combine), nor \
             elementwise arith.*/math.* on a tensor (split). A region with only \
             index/scalar (address/loop) arithmetic and no tile-level tensor compute is \
             data movement, not compute, and must not be planned as split. Picking one \
             archetype for a composite body yields a plausible-looking WRONG plan that \
             the emitter then honours, so it is refused here. Adding a pattern or a \
             heuristic beyond the 3 hardcoded 1p0 patterns is a scope/architecture \
             change (ping @rganti).",
        )
    })?;

    // Pattern-specific bounds are DERIVED up front, so a RED-stop happens before
    // any IR mutation.
    let corelets: Vec<Op>;
    let mut work_division: Option<Vec<i64>> = None;
    // MUTABLE, because `Split`/`PartialCombine` can RE-PATTERN once the stick count is
    // known -- see the `single_corelet` arm below. The plan must then be verified and
    // labelled with the pattern it CLAIMS, not the one classification guessed.
    let mut pattern = pattern;

    match pattern {
        // NOT REACHABLE from classification -- `single_corelet` is only ever arrived at by
        // the re-patterning inside the `Split`/`PartialCombine` arm below, so a plan that
        // starts here means somebody made `classify_region` return it without deciding what
        // its stick count is. Refuse rather than fall through to a `[0, 0]` bound.
        Pattern::SingleCorelet => {
            return Err(Refusal::new(
                PASS,
                "single_corelet is a RE-PATTERNING of split/partial_combine once the tile's                  stick count is known, not a classification. Reaching it directly means the                  stick count was never derived, and the plan's data_bounds would be                  fabricated.",
            ))
        }
        Pattern::Split | Pattern::PartialCombine => {
            let sticks = recover_tile_sticks(&loopp).ok_or_else(|| {
                Refusal::new(
                    PASS,
                    format!(
                        "{} pattern cannot recover the tile stick count from a \
                         ktdp.construct_access_tile in the per-core work region (needs a \
                         static !ktdp.access_tile whose innermost/stick dimension is a \
                         multiple of the 64-element stick width). Deriving the partition \
                         from the KTDP tile metadata is required; baking in a magic stick \
                         count is not (ping @rganti).",
                        pattern.spelling()
                    ),
                )
            })?;
            // FEWER STICKS THAN CORELETS HAS NOTHING TO DIVIDE, so re-pattern.
            //
            // A STICK IS THE TRANSFER AND COMPUTE GRANULE, so there is no sub-stick
            // partition to hand a second corelet: assigning the whole tile to ONE is the
            // correct plan, not a workaround. The floor split computes
            // `k = tile_sticks / 2`, which for one stick gives corelet 0 the EMPTY range
            // `[0, 0]` and corelet 1 `[0, 1]` -- malformed, and rejected by name.
            //
            // THIS PORT PREVIOUSLY REPRODUCED THAT REFUSAL, AND WAS RIGHT TO:
            // `test/goldens/ktir/{vector_add,mul}/refusal.txt` recorded the C++ saying the
            // same thing, and a port that ACCEPTED what the C++ refused would have been the
            // defect. The C++ has since fixed it (`PlanCorelets.cpp:456-479` and `:744`), so
            // reproducing the refusal became reproducing a BUG -- the one way a faithfulness
            // doctrine can point the wrong way, which is worth saying here and not only in a
            // commit message. `vector_add`, `mul` and `embedding_granite` at BLOCK_M=64 all
            // reach KTIR now.
            //
            // WHAT WAS **NOT** RELAXED: the two-corelet coherence rule in [`verify_plan`] is
            // unchanged and still rejects both the old malformed shape and a `split` plan
            // carrying one corelet. The fix is a new plan SHAPE, not a weaker verifier.
            if sticks < NUM_CORELETS {
                pattern = Pattern::SingleCorelet;
            }
            corelets = match pattern {
                Pattern::SingleCorelet => fill_single_corelet(sticks),
                Pattern::Split => fill_split(sticks),
                _ => fill_partial_combine(sticks),
            };
        }
        Pattern::IndependentRows => {
            let rows = recover_row_count(module, &loopp).ok_or_else(|| {
                Refusal::new(
                    PASS,
                    "independent_rows pattern cannot read a static, even row count from \
                     the fused body's reductions (the M/2 partition must tile the rows \
                     with no overlap). Deriving it from the lowered KTIR is required; \
                     rounding an odd or dynamic count is not.",
                )
            })?;
            corelets = fill_independent_rows(rows);
        }
        Pattern::IndependentSubtile => {
            let n = recover_matmul_n(&loopp).ok_or_else(|| {
                Refusal::new(
                    PASS,
                    "independent_subtile pattern cannot read a static, even output N from \
                     the region's linalg.matmul output (the N/2 partition must tile N with \
                     no overlap). Reading N from the lowered KTIR is required; hardcoding \
                     the fixture's literal N is not (ping @rganti).",
                )
            })?;
            corelets = fill_independent_subtile(n);
            // P2.4: the whole-matmul WorkDivision makes the plan LOAD-BEARING -- the
            // multicore emitter DERIVES its 32-core form from this rather than
            // re-deriving from the memory views.
            if let Some([m, n_full, k]) = recover_matmul_shape(module, &loopp) {
                work_division = Some(matmul_work_division(m, n_full, k).to_vec());
            }
        }
    }

    // THE PLAN MUST VERIFY. This is the SP-E3-02 contract and it is where
    // `vector_add` and `mul` die.
    verify_plan(pattern, &corelets)?;

    let mut plan = Op::new(OpKind::KtdfCoreletPlan)
        .with_attr(AttrKey::Pattern, Attr::Str(pattern.spelling().to_string()));
    if let Some(wd) = work_division {
        plan.set_attr(AttrKey::WorkDivision, Attr::IntList(wd));
    }
    plan.regions.push(Region { args: vec![], ops: corelets });

    // Insert the plan at the TOP of the loop body so it precedes the compute it
    // describes.
    let loop_op = walk::at_mut(module, path).expect("path");
    loop_op.regions[0].ops.insert(0, plan);
    Ok(())
}

//===----------------------------------------------------------------------===//
// Classification
//===----------------------------------------------------------------------===//

/// Is `op` an elementwise TILE compute op the `split` pattern handles?
///
/// Classified by dialect so the set is not a brittle allowlist, but the op must
/// PRODUCE A SHAPED RESULT. Loop-index and address arithmetic live in the same
/// dialects but operate on scalars -- they are data-movement plumbing, not compute.
/// Without the shaped-result gate a pure data-movement region is misclassified
/// `split` and gets a bogus plan instead of a clean RED-stop.
fn is_elementwise_compute(op: &Op) -> bool {
    let s = op.kind.spelling();
    if !(s.starts_with("arith.") || s.starts_with("math.")) {
        return false;
    }
    op.result_types.iter().any(|t| t.dims().is_some())
}

/// Does every reduction in this body reduce along its operand's LAST axis?
///
/// This is the test for ROW INDEPENDENCE, and it is what lets a fused
/// matmul+reduction body be planned as disjoint row ranges with no cross-corelet
/// combine. A last-axis reduction consumes one row and produces that row's scalar,
/// so no output row depends on another; a reduction along any earlier axis mixes
/// rows and needs a real combine.
///
/// Read from the IR, never assumed. A reduction whose axis cannot be determined
/// counts as NOT last-axis, so an unrecognised form fails closed.
fn all_reductions_are_along_last_axis(module: &Module, loopp: &Op) -> bool {
    let mut all_last = true;
    for op in loopp.ops_deep() {
        match op.kind {
            OpKind::TtReduce => {
                let axis = op.attr(&AttrKey::Axis).and_then(|a| a.as_int());
                let rank = op
                    .operands
                    .first()
                    .and_then(|v| module.type_of(*v))
                    .map(|t| t.rank() as i64);
                match (axis, rank) {
                    (Some(a), Some(r)) if r >= 1 && a == r - 1 => {}
                    _ => all_last = false,
                }
            }
            OpKind::LinalgReduce => {
                let dims = op.attr(&AttrKey::Dimensions).and_then(|a| a.as_int_list());
                let rank = op
                    .operands
                    .first()
                    .and_then(|v| module.type_of(*v))
                    .map(|t| t.rank() as i64);
                match (dims, rank) {
                    (Some(d), Some(r)) if d.len() == 1 && r >= 1 && d[0] == r - 1 => {}
                    _ => all_last = false,
                }
            }
            _ => {}
        }
    }
    all_last
}

/// Classify a per-core work region by the op type of its compute body.
///
/// PRECEDENCE is only sound when the extra ops are incidental to one archetype (a
/// matmul's epilogue add, a reduce's scale). A COMPOSITE body -- matmul AND
/// reduction fused -- is NOT "whichever archetype wins": picking one produced a
/// plausible-looking WRONG plan the emitter then honoured. But it is not
/// automatically ambiguous either: when EVERY reduction runs along the last axis,
/// each output row depends only on that row, so the corelets take disjoint ROW
/// ranges and never communicate. That is a statement about the dependence
/// structure, checked from the IR.
///
/// `None` is the RED-stop.
fn classify_region(module: &Module, loopp: &Op) -> Option<Pattern> {
    let mut saw_matmul = false;
    let mut saw_reduce = false;
    let mut saw_elementwise = false;
    for op in loopp.ops_deep() {
        match op.kind {
            // TRITON-SHAPED REDUCTIONS AND DOTS COUNT TOO. Testing only the linalg
            // forms made every Triton reduction INVISIBLE here, so a body containing a
            // whole softmax fell through to `Split` on its elementwise ops alone -- and
            // the elementwise emitter then dropped the reduction on the floor.
            OpKind::LinalgMatmul | OpKind::TtDot => saw_matmul = true,
            OpKind::LinalgReduce | OpKind::TtReduce => saw_reduce = true,
            _ if is_elementwise_compute(op) => saw_elementwise = true,
            _ => {}
        }
    }
    if saw_matmul && saw_reduce {
        return if all_reductions_are_along_last_axis(module, loopp) {
            Some(Pattern::IndependentRows)
        } else {
            None // a reduction ACROSS rows needs a real combine: ambiguous
        };
    }
    if saw_matmul {
        return Some(Pattern::IndependentSubtile);
    }
    if saw_reduce {
        return Some(Pattern::PartialCombine);
    }
    if saw_elementwise {
        return Some(Pattern::Split);
    }
    None
}

//===----------------------------------------------------------------------===//
// Bound recovery -- from the IR, never baked in
//===----------------------------------------------------------------------===//

/// The tile's static stick count, from the region's `ktdp.construct_access_tile`.
///
/// The count comes from the STICK dim (the innermost, last dim), not from a bare
/// `numElements / 64` over all dims. That innermost dim must be exactly
/// stick-aligned; a tile whose innermost contiguous axis is NOT a stick boundary
/// RED-stops rather than silently miscounting.
/// # A KNOWN C++ FAIL-OPEN, REPRODUCED DELIBERATELY AND NAMED SO IT DOES NOT LOOK INTENDED
///
/// This takes the **FIRST** `ktdp.construct_access_tile` in the region, exactly as
/// `recoverTileSticks` does. For a region with one compute tile that is right. For the
/// EMBEDDING it is not: the first tile is the one-stick i32 **index** tile, not the
/// `128x4096` f16 tile the multiply runs on. So in
/// `test/experiment1/ktir/embedding_granite_bm128_control.ktir.mlir` the plan reads TWO
/// sticks where the compute is 8192 -- under-claiming by three orders of magnitude, and it
/// always has.
///
/// IT IS REPRODUCED RATHER THAN FIXED because this crate's job is to agree with the C++ KTIR
/// field by field, and `tests/pure_rust_ktir.rs` diffs against goldens the C++ produced with
/// this behaviour. Fixing it here would make this port DISAGREE with the oracle, and the
/// disagreement would be reported against the diff rather than against the bug.
///
/// WHAT WOULD FIX IT, on the C++ side, so this comment is actionable rather than a shrug: the
/// stick count must come from the tile the region's COMPUTE reads, not from the first tile
/// constructed -- for the embedding, the `ktdp.load` whose result feeds the `arith.mulf`.
/// That moves a checked-in golden, which is why it is reported and not done here.
fn recover_tile_sticks(loopp: &Op) -> Option<i64> {
    for op in loopp.ops_deep() {
        if op.kind != OpKind::KtdpConstructAccessTile {
            continue;
        }
        let Some(dims) = op.result_type().and_then(|t| t.dims()) else { continue };
        if dims.is_empty() || dims.contains(&DYNAMIC) {
            continue;
        }
        let stick_dim = *dims.last().unwrap();
        if stick_dim <= 0 || stick_dim % STICK_WIDTH != 0 {
            continue;
        }
        let n: i64 = dims.iter().product();
        if n <= 0 {
            continue;
        }
        return Some(n / STICK_WIDTH);
    }
    None
}

/// The matmul output's static N (last) dim. A non-static or ODD N is a RED-stop:
/// the N/2 partition must tile N exactly.
fn recover_matmul_n(loopp: &Op) -> Option<i64> {
    for op in loopp.ops_deep() {
        if op.kind != OpKind::LinalgMatmul || op.results.len() != 1 {
            continue;
        }
        let Some(dims) = op.result_type().and_then(|t| t.dims()) else { continue };
        if dims.is_empty() || dims.contains(&DYNAMIC) {
            continue;
        }
        let n = *dims.last().unwrap();
        if n > 0 && n % NUM_CORELETS == 0 {
            return Some(n);
        }
    }
    None
}

/// Row count of a fused body, from the FIRST reduction's operand leading dim.
///
/// Every reduction in an `independent_rows` body reduces along the last axis, so
/// they all share the same leading (row) extent; the matmul output agrees with it
/// by construction. `None` when it is not static, even and positive -- the halves
/// must tile the rows with no overlap, and an odd or dynamic count is a spec delta
/// rather than something to round.
fn recover_row_count(module: &Module, loopp: &Op) -> Option<i64> {
    for op in loopp.ops_deep() {
        let in_ty = match op.kind {
            OpKind::TtReduce | OpKind::LinalgReduce => {
                op.operands.first().and_then(|v| module.type_of(*v))
            }
            _ => None,
        };
        let Some(t) = in_ty else { continue };
        let Some(dims) = t.dims() else { continue };
        if dims.len() < 2 {
            continue;
        }
        let m = dims[0];
        if m > 0 && m != DYNAMIC && m % NUM_CORELETS == 0 {
            return Some(m);
        }
    }
    None
}

/// The matmul's FULL static (M, N, K).
///
/// CRITICAL: after the K-loop is tiled, the `linalg.matmul` OPERAND is a K-TILE, so
/// reading K off the operand tensor type gives the tile BK, not the full K. The
/// FULL shape lives on the KTDP memory view, so trace each operand exactly as the
/// emitter does:
///
/// ```text
///   linalg.matmul operand -> ktdp.load -> construct_access_tile(base)
///                         -> construct_memory_view.sizes
/// ```
fn recover_matmul_shape(module: &Module, loopp: &Op) -> Option<[i64; 3]> {
    let full_shape = |v: Ssa| -> Option<(i64, i64)> {
        let load = module.def_of(v).filter(|o| o.kind == OpKind::KtdpLoad)?;
        let at = module
            .def_of(load.operands.first().copied()?)
            .filter(|o| o.kind == OpKind::KtdpConstructAccessTile)?;
        let mv = module
            .def_of(at.operands.first().copied()?)
            .filter(|o| o.kind == OpKind::KtdpConstructMemoryView)?;
        let s = mv.attr(&AttrKey::Shape)?.as_int_list()?;
        if s.len() != 2 || s.contains(&DYNAMIC) {
            return None;
        }
        Some((s[0], s[1]))
    };
    for op in loopp.ops_deep() {
        if op.kind != OpKind::LinalgMatmul || op.operands.len() < 2 {
            continue;
        }
        let a = full_shape(op.operands[0])?; // [M, K] from A's view
        let b = full_shape(op.operands[1])?; // B's view, in the order the op's maps state
        // ⛔⛔⛔ WHICH AXIS OF B IS `N` IS STATED BY THE OP, NOT DEDUCED FROM WHICH EXTENT FITS.
        //
        // This read was `let (m, k, n) = (a.0, a.1, b.1)` with a `a.1 == b.0` sanity check, i.e. it
        // assumed B's view is `[K, N]` — `linalg.matmul`'s default indexing. That was true until
        // `dot_to_linalg` began spending a transposed dot weight in the `indexing_maps` instead of
        // leaving a `tt.trans` for the access tile: the weight's view is then the `[N, K]` the kernel
        // declares, `a.1 == b.0` is `128 == 256`, and this function fell off the end and returned
        // `None`.
        //
        // ⛔ AND THE CONSEQUENCE WAS SILENT, WHICH IS WHY THE READ IS BEING FIXED RATHER THAN THE
        // CHECK RELAXED. The caller is `if let Some(...) = recover_matmul_shape(...)`, so `None`
        // simply OMITS `work_division` — an attribute this pass's own comment four lines up calls
        // LOAD-BEARING ("the multicore emitter DERIVES its 32-core form from this rather than
        // re-deriving from the memory views"). MEASURED against IBM's C++ goldens: all three
        // `swiglu_mlp_*` configurations lost the attribute entirely, reported by
        // `tests/pure_rust_ktir.rs` as `ktdf.corelet_plan: attributes differs / golden:
        // {pattern=\"independent_subtile\", work_division=array<i64: 64, 256, 128, ...>} / ours:
        // {pattern=\"independent_subtile\"}`. Reading the orientation restores it BYTE-IDENTICALLY on
        // all three — `[64, 256, 128, 64, 4, 8, 32, 16, 8, 1]` for `small` and `tiled_k`,
        // `[64, 12800, 4096, 64, 16, 2, 32, 16, 4, 12]` for `granite` — which is the check that this
        // reads the same `N` the C++ did and not merely a different number that parses.
        //
        // The orientation predicate lives beside the WRITER (`dot_to_linalg::weight_is_n_by_k` over
        // `TRANSPOSED_B_MAPS`) so the maps are spelled once in the tree.
        let (k_of_b, n) = if crate::passes::dot_to_linalg::weight_is_n_by_k(op) {
            (b.1, b.0) // W as (n, k)
        } else {
            (b.0, b.1) // W as (k, n) — `linalg.matmul`'s default, and what an absent map means
        };
        let (m, k) = (a.0, a.1);
        if m > 0 && n > 0 && k > 0 && k == k_of_b {
            return Some([m, n, k]);
        }
    }
    None
}

/// The whole-matmul work division (P2.4) = f(shape, arch).
///
/// Order: `[M, N, K, S, OUT, IN, numCores, mLoop, D2, nBlocks]`. `S` = f16 stick
/// width; `effN` = N > numCores*S ? 1024 : N; `OUT` = effN/S output-stick tiles;
/// `IN` = numCores*S/effN K-splits filling the 32-core array; `numCores` = OUT*IN;
/// `mLoop` = M/4; `D2` = min(2*IN, 8) L0 streaming depth; `nBlocks` = N > 2048 ?
/// N/1024 : 1. Mirrors `EmitDFIRPhysical.cpp deriveWorkDivision` exactly, so the
/// emit is byte-identical for the supported shapes.
fn matmul_work_division(m: i64, n: i64, k: i64) -> [i64; 10] {
    const S: i64 = 64; // f16 stick width
    const ARRAY_N_LANE_WIDTH: i64 = 32 * S; // = 2048
    let (n_blocks, eff_n) = if n > ARRAY_N_LANE_WIDTH { (n / 1024, 1024) } else { (1, n) };
    let out = eff_n / S;
    let inn = ARRAY_N_LANE_WIDTH / eff_n;
    let num_cores = out * inn;
    let m_loop = m / 4;
    let d2 = (2 * inn).min(8);
    [m, n, k, S, out, inn, num_cores, m_loop, d2, n_blocks]
}

//===----------------------------------------------------------------------===//
// Per-corelet field fill
//===----------------------------------------------------------------------===//

fn corelet(index: i64) -> Op {
    Op::new(OpKind::KtdfCorelet).with_attr(AttrKey::Index, Attr::Int(index))
}

/// `split` (elementwise): partition the tile's sticks -- CL0 the lower half
/// `[0, k]`, CL1 `[k, T]`, disjoint + contiguous + covering the whole tile. 1p0:
/// `k = T / 2`, the `sticks_per_corelet` constant the pass owns. A non-even count
/// gives CL1 the remainder (floor split), so the bounds still cover `[0, T]`.
/// `single_corelet`: one `ktdf.corelet` at index 0, `data_bounds = [0, tile_sticks]`.
///
/// WHY A PLAN AT ALL, rather than emitting none: absence is indistinguishable from "the
/// planning pass never ran", which an emitter reads as "derive it from the shape" -- so a
/// region whose derived stick count is 1 but whose compute tile is wider would emit a
/// single-stick kernel with NO diagnostic. A stated plan fails closed instead: a consumer
/// refuses a pattern it does not implement, by name.
fn fill_single_corelet(tile_sticks: i64) -> Vec<Op> {
    vec![corelet(0).with_attr(AttrKey::DataBounds, Attr::IntList(vec![0, tile_sticks]))]
}

/// `fill_split`, exposed so a test can assert the two-corelet coherence rule STILL rejects
/// the shape a one-stick split produces. That rule is what `single_corelet` routes around
/// rather than weakens, and a control has to be able to show it is intact.
pub fn corelets_for_test(tile_sticks: i64) -> Vec<Op> {
    fill_split(tile_sticks)
}

fn fill_split(tile_sticks: i64) -> Vec<Op> {
    let k = tile_sticks / NUM_CORELETS;
    vec![
        corelet(0).with_attr(AttrKey::DataBounds, Attr::IntList(vec![0, k])),
        corelet(1).with_attr(AttrKey::DataBounds, Attr::IntList(vec![k, tile_sticks])),
    ]
}

/// `independent_rows`: each corelet runs the WHOLE fused body for a disjoint half
/// of the output ROWS. No ring fields and no combine, because every reduction runs
/// along the last axis.
fn fill_independent_rows(rows: i64) -> Vec<Op> {
    let h = rows / NUM_CORELETS;
    vec![
        corelet(0).with_attr(AttrKey::DataBounds, Attr::IntList(vec![0, h])),
        corelet(1).with_attr(AttrKey::DataBounds, Attr::IntList(vec![h, rows])),
    ]
}

/// `independent_subtile` (matmul): each corelet computes a disjoint half of the
/// output N, with its OWN PT rows and a 64 KB XRF budget. There is NO
/// cross-corelet partial-sum merge on 1p0, so no ring fields are emitted.
fn fill_independent_subtile(out_n: i64) -> Vec<Op> {
    let half = out_n / NUM_CORELETS;
    let one = |lo: i64, hi: i64, idx: i64| {
        corelet(idx)
            .with_attr(AttrKey::OutputPartition, Attr::IntList(vec![lo, hi]))
            .with_attr(AttrKey::PtRows, Attr::IntList(vec![0, MATMUL_PT_ROWS_HI]))
            .with_attr(AttrKey::XrfCapacity, Attr::Int(MATMUL_XRF_CAPACITY_KB))
    };
    vec![one(0, half, 0), one(half, out_n, 1)]
}

/// `partial_combine` (reduction): the same even floor-split as `split`, then CL0
/// reduces its partition and SENDS its partial over SFPRING; CL1 reduces, RECEIVES,
/// and combines -- so CL1 produces the final result, no LX round-trip.
fn fill_partial_combine(tile_sticks: i64) -> Vec<Op> {
    let k = tile_sticks / NUM_CORELETS;
    vec![
        corelet(0)
            .with_attr(AttrKey::DataBounds, Attr::IntList(vec![0, k]))
            .with_attr(AttrKey::Role, Attr::Str(ROLE_PARTIAL.into()))
            .with_attr(AttrKey::RingSend, Attr::Str(RING_DIRECTION.into())),
        corelet(1)
            .with_attr(AttrKey::DataBounds, Attr::IntList(vec![k, tile_sticks]))
            .with_attr(AttrKey::Role, Attr::Str(ROLE_COMBINE.into()))
            .with_attr(AttrKey::RingRecv, Attr::Str(RING_DIRECTION.into())),
    ]
}

//===----------------------------------------------------------------------===//
// The verifier -- ported from KTDFOps.cpp
//===----------------------------------------------------------------------===//

/// The `ktdf.corelet_plan` verifier's cross-corelet coherence rules.
///
/// PORTED HERE ON PURPOSE, because it is where `vector_add` and `mul` are refused
/// and the refusal is part of the behaviour being ported. BLOCK=64 f16 is ONE
/// stick, an even two-corelet split of one stick is `[0, 0]` plus `[0, 1]`, and the
/// verifier rejects that by name. A port that accepts `vector_add` where the C++
/// refuses it is WRONG, and this function is the test.
pub fn verify_plan(pattern: Pattern, corelets: &[Op]) -> Result<()> {
    // `single_corelet` IS THE ONE PATTERN WITH ONE CORELET, and its shape is still checked:
    // exactly one corelet, `data_bounds` present, two elements, non-empty, starting at 0.
    // The two-corelet coherence rule below does not apply because there is no second range
    // to be disjoint FROM -- which is why this is a separate arm rather than a relaxation of
    // that rule. The rule itself is byte-unchanged.
    if pattern == Pattern::SingleCorelet {
        if corelets.len() != 1 {
            return Err(Refusal::new(
                "ktdf.corelet_plan",
                format!(
                    "op single_corelet expects exactly 1 ktdf.corelet, got {}",
                    corelets.len()
                ),
            ));
        }
        let b = corelets[0]
            .attr(&AttrKey::DataBounds)
            .and_then(|a| a.as_int_list())
            .map(|s| s.to_vec());
        let Some(b) = b else {
            return Err(Refusal::new(
                "ktdf.corelet_plan",
                "op single_corelet's ktdf.corelet must carry a 2-element data_bounds",
            ));
        };
        if b.len() != 2 || b[0] != 0 || b[1] <= b[0] {
            return Err(Refusal::new(
                "ktdf.corelet_plan",
                format!(
                    "op single_corelet's data_bounds must be [0, tile_sticks] with                      tile_sticks > 0, got {b:?}"
                ),
            ));
        }
        return Ok(());
    }
    if corelets.len() != NUM_CORELETS as usize {
        return Err(Refusal::new(
            "ktdf.corelet_plan",
            format!("op expects exactly {NUM_CORELETS} ktdf.corelet ops, got {}", corelets.len()),
        ));
    }
    let bounds_key = match pattern {
        Pattern::IndependentSubtile => AttrKey::OutputPartition,
        _ => AttrKey::DataBounds,
    };
    let get = |c: &Op| -> Option<Vec<i64>> {
        c.attr(&bounds_key).and_then(|a| a.as_int_list()).map(|s| s.to_vec())
    };
    let (Some(a), Some(b)) = (get(&corelets[0]), get(&corelets[1])) else {
        return Err(Refusal::new(
            "ktdf.corelet_plan",
            format!("op each ktdf.corelet must carry a 2-element {}", bounds_key.spelling()),
        ));
    };
    if a.len() != 2 || b.len() != 2 {
        return Err(Refusal::new(
            "ktdf.corelet_plan",
            format!("op each {} must have exactly 2 elements", bounds_key.spelling()),
        ));
    }
    // Disjoint, contiguous and NON-EMPTY: each lo < hi, and one's hi == the other's
    // lo.
    let ok = a[0] < a[1] && b[0] < b[1] && (a[1] == b[0] || b[1] == a[0]);
    if !ok {
        return Err(Refusal::new(
            "ktdf.corelet_plan",
            format!(
                "op the two {} ranges must be disjoint and contiguous and non-empty (each \
                 lo < hi, one's hi == the other's lo), but got [{}, {}] and [{}, {}]",
                bounds_key.spelling(),
                a[0],
                a[1],
                b[0],
                b[1]
            ),
        ));
    }
    if pattern == Pattern::PartialCombine {
        fn role(c: &Op) -> &str {
            c.attr(&AttrKey::Role).and_then(|x| x.as_str()).unwrap_or("")
        }
        if !(role(&corelets[0]) == ROLE_PARTIAL && role(&corelets[1]) == ROLE_COMBINE) {
            return Err(Refusal::new(
                "ktdf.corelet_plan",
                "op partial_combine needs exactly one `partial` and one `combine` role",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::parse;

    /// THE `vector_add` REFUSAL, reproduced by name. One 64-element f16 stick means
    /// `tile_sticks == 1`, so the floor split is `[0, 0]` and `[0, 1]`.
    #[test]
    fn one_stick_cannot_be_split_across_two_corelets() {
        let corelets = fill_split(1);
        let e = verify_plan(Pattern::Split, &corelets).unwrap_err();
        assert_eq!(
            e.message,
            "op the two data_bounds ranges must be disjoint and contiguous and non-empty \
             (each lo < hi, one's hi == the other's lo), but got [0, 0] and [0, 1]",
            "this is the C++ verifier's message, verbatim"
        );
        assert_eq!(e.pass, "ktdf.corelet_plan", "the VERIFIER refuses it, not the pass");
    }

    #[test]
    fn two_sticks_split_cleanly() {
        verify_plan(Pattern::Split, &fill_split(2)).expect("[0,1] and [1,2] are legal");
    }

    #[test]
    fn a_fused_matmul_plus_last_axis_reduction_is_independent_rows() {
        // The attention body's shape. A last-axis reduce means the two corelets take
        // disjoint ROW halves and never communicate.
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %tid = ktdp.get_compute_tile_id : index
    %z = arith.constant dense<0.000000e+00> : tensor<64x64xf16>
    %a = arith.constant dense<1.000000e+00> : tensor<64x128xf16>
    %b = arith.constant dense<1.000000e+00> : tensor<128x64xf16>
    scf.for %i = %tid to %c8 step %c32 {
      %qk = linalg.matmul ins(%a, %b : tensor<64x128xf16>, tensor<128x64xf16>) outs(%z : tensor<64x64xf16>) -> tensor<64x64xf16>
      %m = \"tt.reduce\"(%qk) <{axis = 1 : i32}> ({
      ^bb0(%x: f16, %y: f16):
        %c = arith.maxnumf %x, %y : f16
        tt.reduce.return %c : f16
      }) : (tensor<64x64xf16>) -> tensor<64xf16>
    }
    tt.return
  }
}
";
        let mut m = parse::parse(src).unwrap();
        run(&mut m).expect("independent_rows, not ambiguous");
        let plan = m
            .ops_deep()
            .into_iter()
            .find(|o| o.kind == OpKind::KtdfCoreletPlan)
            .expect("a plan is inserted");
        assert_eq!(
            plan.attr(&AttrKey::Pattern).and_then(|a| a.as_str()),
            Some("independent_rows")
        );
        // Rows = 64 from the reduce operand's leading dim -> [0,32] and [32,64].
        let cs = &plan.regions[0].ops;
        assert_eq!(cs.len(), 2);
        assert_eq!(cs[0].attr(&AttrKey::DataBounds), Some(&Attr::IntList(vec![0, 32])));
        assert_eq!(cs[1].attr(&AttrKey::DataBounds), Some(&Attr::IntList(vec![32, 64])));
        // And the plan is the FIRST op in the loop body.
        let forr = m.ops_deep().into_iter().find(|o| o.kind == OpKind::ScfFor).unwrap();
        assert_eq!(forr.regions[0].ops[0].kind, OpKind::KtdfCoreletPlan);
    }

    #[test]
    fn a_reduction_across_rows_stays_ambiguous_and_red_stops() {
        // axis 0 on a rank-2 operand is NOT the last axis, so the fused body needs a
        // real combine and must be refused.
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %tid = ktdp.get_compute_tile_id : index
    %z = arith.constant dense<0.000000e+00> : tensor<64x64xf16>
    %a = arith.constant dense<1.000000e+00> : tensor<64x128xf16>
    %b = arith.constant dense<1.000000e+00> : tensor<128x64xf16>
    scf.for %i = %tid to %c8 step %c32 {
      %qk = linalg.matmul ins(%a, %b : tensor<64x128xf16>, tensor<128x64xf16>) outs(%z : tensor<64x64xf16>) -> tensor<64x64xf16>
      %m = \"tt.reduce\"(%qk) <{axis = 0 : i32}> ({
      ^bb0(%x: f16, %y: f16):
        %c = arith.maxnumf %x, %y : f16
        tt.reduce.return %c : f16
      }) : (tensor<64x64xf16>) -> tensor<64xf16>
    }
    tt.return
  }
}
";
        let mut m = parse::parse(src).unwrap();
        let e = run(&mut m).unwrap_err();
        assert!(e.message.contains("COMPOSITE"), "got {e}");
        assert!(e.message.contains("ping @rganti"), "got {e}");
    }

    #[test]
    fn a_data_movement_only_region_red_stops_rather_than_being_planned_split() {
        // Index arithmetic is plumbing, not compute. Without the shaped-result gate
        // this is misclassified `split` and gets a bogus plan.
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %tid = ktdp.get_compute_tile_id : index
    scf.for %i = %tid to %c8 step %c32 {
      %a = arith.muli %i, %c8 : index
      %b = arith.addi %a, %c32 : index
    }
    tt.return
  }
}
";
        let mut m = parse::parse(src).unwrap();
        let e = run(&mut m).unwrap_err();
        assert!(e.message.contains("data movement, not compute"), "got {e}");
    }

    #[test]
    fn the_work_division_matches_the_emitters_derivation() {
        // M7's shape, and the two branches of the N-block rule.
        assert_eq!(matmul_work_division(64, 128, 128), [64, 128, 128, 64, 2, 16, 32, 16, 8, 1]);
        // N = 4096 > 2048 -> nBlocks = 4, effN = 1024, OUT = 16, IN = 2.
        assert_eq!(
            matmul_work_division(64, 4096, 128),
            [64, 4096, 128, 64, 16, 2, 32, 16, 4, 4]
        );
    }

    #[test]
    fn the_plan_is_idempotent() {
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %tid = ktdp.get_compute_tile_id : index
    %a = arith.constant dense<1.000000e+00> : tensor<128xf16>
    scf.for %i = %tid to %c8 step %c32 {
      %t = ktdp.construct_access_tile %q[%i] {access_tile_order = affine_map<(d0) -> (d0)>} : memref<1024xf16> -> !ktdp.access_tile<128xindex>
      %b = arith.addf %a, %a : tensor<128xf16>
    }
    tt.return
  }
}
";
        let mut m = parse::parse(src).unwrap();
        run(&mut m).unwrap();
        run(&mut m).unwrap();
        let plans = m.ops_deep().iter().filter(|o| o.kind == OpKind::KtdfCoreletPlan).count();
        assert_eq!(plans, 1, "a second run must not insert a duplicate plan");
    }
}
