// SPDX-License-Identifier: Apache-2.0
//
// ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ CRUSTIFY BRIDGE-2 CAMPAIGN — READ THIS BEFORE YOU FILL AN ANCHOR IN THIS FILE.               ║
// ║ Full brief: crustify-bridge2/AGENT-BRIEF.md   ·   campaign statement: crustify-bridge2/TASK.md║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
//
// 1. THE AUTHORITY IS THE C++ TREE, NOT THE EXTRACT.
//       /Users/nickm/git/deeptools-src/<file>:<line>        (deeptools @ a0d29abbed — repo_info.txt)
//    That is the revision every citation below resolves against. `crustify-bridge2/source/bridge2.cpp`
//    says WHICH functions are in scope and IN WHAT ORDER; ⛔ its bodies are TRUNCATED AT THE TAIL —
//    366 of the 384 end in a blank line and bare closing braces, and a 48-entry sample against the
//    authority found 21 that had lost real trailing statements (a `return success();`, a
//    `return rhs;`, an entire `} else { … }` branch, an `initMASData(...)` call). Port from the
//    authority file at the cited line. ⛔ /Users/nickm/git/deeptools is a DIFFERENT revision.
//    ⛔ The pod (/project_src/deeptools) is NOT reachable from this host — use the mirror above.
//
// 2. PORTED MEANS THE WHOLE FUNCTION INCLUDING ITS EMISSION. The op a function emits IS the
//    function — its exact attribute names and values, branch order and early returns. A documented
//    predicate that emits nothing is NOT a port (that is how the previous attempt failed). What you
//    MAY drop is only the mechanism for REACHING operands: use-walks, memoising by
//    (core, corelet, component), positioning an OpBuilder. If the target IR cannot express a
//    function's input, ADD THE OP to `src/islands/{sentient,dataflow_ir}/` — never decide the
//    function is unnecessary.
//
// 3. THIS IS A PURE-LOGIC PORT WITH NO C ANYWHERE. Whatever the generic C-to-Rust conventions say:
//    ❌ no bindgen/allowlist/-sys, ❌ no `ffi::`/`mod ffi_export`/`#[unsafe(no_mangle)] extern "C"`,
//    ❌ no `CRUSTIFY_<FILE>` switch, ❌ no `Foo`/`FooRef`/`FooMut` layout triple, ❌ no `unsafe`,
//    ❌ no sanitizers and no C-vs-Rust equivalence harness (there is no C to call).
//
// 4. CRATE RULES BIND YOU — `crates/compiler/deeptools/CLAUDE.md`, read it in full.
//    🛑 NEVER RUNTIME REFUSE: no `Result`, no `Err(`, no `.ok_or`, no `assert!`, no `debug_assert!`
//    (frozen at zero by crates/targets/spyre/tests/dfir_never_runtime_refuses.rs). A closed set is
//    an `enum`; an invariant is a TYPE. `todo!("<op> …")` is tolerated, capped and ratcheted down —
//    and ⛔ never substitute a stand-in op to dodge one. Newtypes, never raw scalars. No strings for
//    closed sets. `Arch`/`Model`/`Workload` flow through as const generics.
//
// 5. ANCHORS: each `// crustify:todo: e<NNN>_<name>` below is one scheduled unit. Replace it with
//    the ported function carrying the doc anchor `/// Replaces: e<NNN>_<name>` on the item itself.
//    A surviving TODO is open work; the TODO must not survive beside the filled anchor.
//
// 6. TESTS: `#[cfg(test)] mod unit_tests` beside the code. 668 of the authority tree's 825
//    `dcc/test/**/*.mlir` cases carry `CHECK-SENT-IR` expectations — port the EXPECTATION, build the
//    typed input in Rust (this crate has no MLIR parser and must not get one).
//    `crates/compiler/deeptools/tests/sentient_corpus/` is the answer key (our DataflowIR beside the
//    reference's SentientIR for the same program).
//
// 7. GATE: `cargo check -p deeptools` and `cargo test -p deeptools`. ⛔ NEVER run the workspace or
//    acceptance build in an agent worktree — ~6 GB of target/ each and <50 GB free on this host.
//
// 8. `dataflow_ir_to_sentient/agen_to_sentient.rs` is the EARLIER PARTIAL ATTEMPT (predicates, no
//    emission, called by nothing). Nothing in it counts as ported; reuse what is right, but every
//    unit gets its own anchored item here.

//! `VectorChainToSentientPT.cpp` — 5 of bridge 2's 384 functions (dependency level(s) [0, 6, 7, 8]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e094_computeUnitPrecision` | 094/384 | 9 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:31` |
//! | `e346_lowerDanglingNonComputeOps` | 346/384 | 88 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:882` |
//! | `e368_fuseNonComputeOps` | 368/384 | 191 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:46` |
//! | `e369_fuseComputeOps` | 369/384 | 628 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:245` |
//! | `e379_runOnOperation` | 379/384 | 54 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:975` |

use super::vc_loop_mask_tree::MaskedColumns;
use super::vc_lowering_xrf::XrfPtrMap;
use super::vc_operand_reuse::{DataId, OperandReuse};
use super::vc_vector_chain_helper::{input_precision_from_operand, redefine_constant_vectors};
use super::vc_vector_chain_to_sentient_pesfp::{
    FoldModeAttr, fold_mode_attr_for_operation, walk_positions,
};
use super::vc_vector_operands::{
    ComputeComp, OpId, VectorOperand, erase_op, origin_val, use_positions,
};
use crate::arch::Arch;
use crate::islands::dataflow_ir::dialects::vectorchain as vc;
use crate::islands::dataflow_ir::dialects::{Op as DfirOp, Val, agen, dataflow, dbg_name, vector};
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::dataflow_ir::{self as dfir, Values};
use crate::islands::sentient::dialects::{self as sen, sentient};
use crate::units::DfirUnit;

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 094/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e094_computeUnitPrecision
///
/// # THE PRECISION A PT UNIT'S COMPUTES ARE EMITTED AT
///
/// ```cpp
/// std::string VectorChainToSentientPTLoweringPass::computeUnitPrecision(
///     dataflow::ProgramUnitOp &unit, const SenComponents &comp) {
///   DT_CHECK(comp == PT);
///   DT_CHECK_MSG(unit.getPrecision().has_value(),
///                "Precision attribute for PT is expected");
///   std::string precision = unit.getPrecision().value().str();
///   // Currently, we use fp80 type in MLIR to represent fp8.
///   if (precision == "fp80") return "fp8";
///
///   return precision;
/// }
/// ```
/// (`dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:30-41`)
///
/// # ⭐⭐ WHAT IT PRODUCES IS THE MAC'S `ComputePrecision`
///
/// Its one caller passes the answer straight into the compute lowering, and the vendor's own golden
/// shows where it lands: a unit declared `dataflow.program_unit … {precision = "mxfp4"}` yields
/// `sentient.vector_mac {ComputePrecision = #sentient<precision mxfp4>, …}` at all 24 of its MACs
/// (`dcc/test/Conversion/VectorChainToSentientPT/xrf_increments.mlir:374`). So this is a
/// `dataflow`-rung spelling crossing to the `sentient`-rung enum — which is why the port's signature
/// is [`dataflow::Precision`] in, [`sentient::Precision`] out, rather than `String` to `String`.
///
/// # ⛔⛔ ONE NON-IDENTITY ENTRY, AND IT IS THE WHOLE FUNCTION
///
/// `fp80 -> fp8`. Everything else passes through. The reference states the reason in a comment —
/// *"Currently, we use fp80 type in MLIR to represent fp8"* — and
/// [`dataflow::Precision::Fp80`] records that the spelling appears NOWHERE in the authority tree's
/// `dcc/test`, so this branch is defensive. It is still the only content this function has: a port
/// that dropped it would be `identity` with a citation attached, which is exactly the failure mode
/// the campaign brief names.
///
/// # THE TWO `DT_CHECK`s
///
/// * `DT_CHECK(comp == PT)` — ⛔ **UNREPRESENTABLE HERE.** The component parameter's only use in the
///   body is this comparison. Dropping it removes the way to call this with anything else: the
///   function names the PT lowering in its module and takes no component, so there is no value to
///   compare and no comparison to fail.
/// * `DT_CHECK_MSG(unit.getPrecision().has_value(), "Precision attribute for PT is expected")` —
///   ⛔ **DISCHARGED BY CONSTRUCTION.** The parameter is a [`dataflow::Precision`], not an
///   `Optional`. The island's `ProgramUnit::precision` IS an `Option` (a `dataflow.program_unit` may
///   legitimately carry no attribute, and
///   [`crate::islands::dataflow_ir::dialects::dataflow::Precision`] documents why), so the absent
///   case is a fact about the UNIT that its reader states — and this function is only reachable once
///   that reader has one, which is what taking the value rather than the option means.
#[must_use]
pub const fn compute_unit_precision(precision: dataflow::Precision) -> sentient::Precision {
    match precision {
        // ⭐ THE ONE REMAP.
        dataflow::Precision::Fp80 | dataflow::Precision::Fp8 => sentient::Precision::Fp8,

        // ── `return precision` — the same spelling, at the sentient rung ─────────────────────────
        dataflow::Precision::Int8 => sentient::Precision::Int8,
        dataflow::Precision::Int4 => sentient::Precision::Int4,
        dataflow::Precision::Fp4 => sentient::Precision::Fp4,
        dataflow::Precision::Fp16 => sentient::Precision::Fp16,
        dataflow::Precision::Fp32 => sentient::Precision::Fp32,
        dataflow::Precision::Bf16 => sentient::Precision::Bf16,
        dataflow::Precision::Mxfp4 => sentient::Precision::Mxfp4,
        dataflow::Precision::Mxfp8 => sentient::Precision::Mxfp8,
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 379/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e379_runOnOperation
///
/// `VectorChainToSentientPTLoweringPass::runOnOperation` (`VectorChainToSentientPT.cpp:975`) —
/// redefine the module's constant vectors once, then for each PT row unit: create the XRF index
/// modification ops, build the loop mask tree, build the reuse analysis, reset any existing sentient
/// FMAs, fuse non-compute then compute then dangling non-compute ops, validate, and finally insert
/// the PT mask ops.
///
/// ⛔ `redefineConstantVectors` IS OUTSIDE THE WALK HERE (`:983`) — the PESFP pass calls the same
/// helper from INSIDE its walk (`VectorChainToSentientPESFP.cpp:1385`), so this one runs once per
/// module and that one runs once per PE/SFP unit.
/// ⛔ `WalkResult::interrupt()` AT `:1019` IS A DISCARDED TEMPORARY — it is not returned, so a
/// failed validation signals pass failure and then STILL falls through to `insertPTMaskOps(:1024)`
/// and returns `skip()`. Treating it as a `break` would drop the mask ops of the failing unit.
/// ⛔ THE XRF MAP AND THE MASK TREE ARE BUILT BEFORE THE REUSE ANALYSIS (`:995-1003` before `:1006`)
/// and all three fusion steps take all three, so none of them can be defaulted away.
/// ⭐ THE MASK TREE IS POPULATED HERE, NOT IN THE FUSIONS — *"PT masking relies on loops being
/// stable"* (`:1000-1001`): it is read during fusion and only spent by `insertPTMaskOps`.
pub fn run_on_operation<A: Arch>(program: &mut dfir::Program<A>, values: &mut Values) {
    // `:983` — entry 067, once for the whole module.
    redefine_constant_vectors(program, values);

    // `:985` — the walk over the module's program units.
    for unit in program.units.iter() {
        // `:990` — `if (unit_comp != PT) return WalkResult::skip();`. A PT unit is one PT ROW here:
        // the island names the row the unit runs on, which is what `getUnitType` returns for each of
        // the rows the reference walks.
        let comp = unit.on.kind();
        if !matches!(comp, DfirUnit::PtRow(_)) {
            continue;
        }

        // `:995-997` — `LoweringXRF::createXrfIndexModifOps(unit, unit_comp)`, whose map every step
        // below takes.
        todo!(
            "e367_createXrfIndexModifOps is unported, so e368_fuseNonComputeOps, \
             e369_fuseComputeOps, e346_lowerDanglingNonComputeOps, \
             e228_validateLoweringAndSetMissingParameters and e239_insertPTMaskOps cannot run on the \
             {:?} unit (e227_OperandReuse and the LoopMaskTree of `:1003` are unported too)",
            comp
        );
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 346/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ONE DUMMY MAC THE PT DANGLING LOWERING EMITS, and where the op it replaces stood.
///
/// ⛔ THE MASK-TREE ENTRY TRAVELS WITH IT. `updateLoopMaskTreeForConstantMask(pt_masking_tree, op,
/// &mac_op, 0)` (`VectorChainToSentientPT.cpp:933`, `:947`) keys the tree by the MAC's own position,
/// which it does not have until the caller places it — so the constant mask is carried here and
/// [`update_loop_mask_tree_for_constant_mask`](super::vc_lowering_pt_masks::update_loop_mask_tree_for_constant_mask)
/// is called with it then. ⭐ IT IS THE LITERAL `0` IN BOTH ARMS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PtDummyMac {
    /// The position of the dangling op this replaces.
    pub at: OpId,
    /// The `sentient.scalar_constant` then the `sentient.vector_mac`, in emission order.
    pub ops: Vec<sen::Op>,
    /// The `mask_val` its mask-tree node carries.
    pub mask: MaskedColumns,
}

/// EVERY WAY `lowerDanglingNonComputeOps` CALLS `signalPassFailure()` — one per offending op.
///
/// ⛔ THE WALK DOES NOT STOP AT ONE. The reference's lambda returns `void`, so its `return;` skips
/// the rest of THAT op and the walk carries on — unlike the PESFP twin, which interrupts
/// ([`DanglingOutcome`](super::vc_vector_chain_to_sentient_pesfp::DanglingOutcome)).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PtDangling {
    /// *"There is still a receive or load with uses, that shouldn't happen"* (`:890-893`).
    UsedLoad(OpId),
    /// *"Dangling non-compute op has no use"* (`:950`).
    NoAbsorption(OpId),
    /// *"There is still a create_affine_mask with uses - that shouldn't happen"* (`:960-963`).
    UsedMask(OpId),
    /// `DT_CHECK_MSG(from.orig_precision_ == from.on_the_fly_conv_precision_, "Expecting no on the
    /// fly conversions in PT")` (`:911-912`).
    OnTheFlyConversion(OpId),
    /// The aborts that carry no message: `getOperand(..).value()`, `symbolize…().value()`, the unit's
    /// missing precision attribute, and `XrfPtrMap::at` on an unrecorded `agen.vector_load`.
    Unrepresentable(OpId),
}

/// WHAT ONE DANGLING PT OP BECOMES — a MAC, or the `signalPassFailure()` it earned.
///
/// ⭐ AN OUTCOME ENUM, NOT A `Result`: neither arm is an error the caller recovers from, the refusal
/// is a recorded pass failure ([`PtDangling`]) that the walk carries on past.
#[derive(Debug, Clone, PartialEq, Eq)]
enum PtMac {
    /// `:906-948` ran to the end.
    Emitted(PtDummyMac),
    /// One of the aborts on the way, named.
    Refused(PtDangling),
}

/// WHAT THE PT DANGLING LOWERING PRODUCES — the MACs it emits beside every op it refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PtDanglingLowering {
    /// One per dangling op that was used but not absorbed, in walk order.
    pub macs: Vec<PtDummyMac>,
    /// `signalPassFailure()`, once per offending op.
    pub refusals: Vec<PtDangling>,
}

/// Replaces: e346_lowerDanglingNonComputeOps
///
/// **346/384** `VectorChainToSentientPTLoweringPass::lowerDanglingNonComputeOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:882` (88L).
///
/// ⛔ THE PT MAC IS NOT THE PESFP MAC: it takes **no mask** (the `sentient.scalar_constant` is still
/// emitted, and `nullptr` is passed — `:920`, `:935`), all FOUR precisions are the operand's (`:925-928`,
/// no `"none"` marker), the non-west arm is `opA = from, opB = one, opC = zero` (`:937-939`) and there
/// is NO `DataTransferOnly` attribute. ⛔ AND `pointers` IS SET FOR AN `agen.vector_load` ONLY
/// (`:903-905`) — `at(0)` is the ARGUMENT pair, so the MAC binds both xrf pointers.
/// ⛔ AN OFFENDING OP IS NOT ERASED and does not stop the walk; see [`PtDangling`].
pub fn lower_dangling_non_compute_ops<A: Arch>(
    unit: &mut dfir::ProgramUnit<A>,
    comp: ComputeComp,
    vector_op_to_xrfptr_map: &XrfPtrMap,
    reuse: &OperandReuse,
    values: &mut Values,
) -> PtDanglingLowering {
    let mut tobe_deleted: Vec<OpId> = Vec::new();
    let mut lowering = PtDanglingLowering {
        macs: Vec::new(),
        refusals: Vec::new(),
    };
    // `computeUnitPrecision(unit, comp)` (entry 094) — one answer for every MAC in the unit.
    let compute_precision = unit.precision.map(compute_unit_precision);

    {
        let scope: &[DfirOp] = &unit.body;
        walk_positions(scope, &[], 0, &mut |op, at| -> Option<()> {
            if matches!(
                op,
                DfirOp::Dataflow(dataflow::Op::Receive { .. })
                    | DfirOp::Vector(vector::Op::Load { .. })
                    | DfirOp::Agen(agen::Op::VectorLoad { .. })
            ) {
                if !use_positions(&at, scope).is_empty() {
                    lowering.refusals.push(PtDangling::UsedLoad(at));
                    return None;
                }
                let Some(from) = VectorOperand::operand::<A>(&at, comp, true, scope) else {
                    lowering.refusals.push(PtDangling::Unrepresentable(at));
                    return None;
                };
                let keyed = origin_val(&from.op, scope)
                    .and_then(|val| reuse.absorbtion_flag(val).map(|flag| (val, flag)));
                match keyed {
                    Some((origin, false)) => {
                        // *"it has been used but not absorbed by some other op that is already
                        // lowered"*.
                        match pt_dummy_mac(
                            op,
                            &at,
                            &from,
                            origin,
                            comp,
                            compute_precision,
                            vector_op_to_xrfptr_map,
                            reuse,
                            values,
                        ) {
                            PtMac::Emitted(mac) => lowering.macs.push(mac),
                            PtMac::Refused(refusal) => {
                                lowering.refusals.push(refusal);
                                return None;
                            }
                        }
                    }
                    None => {
                        lowering
                            .refusals
                            .push(PtDangling::NoAbsorption(from.op.clone()));
                        return None;
                    }
                    Some((_, true)) => {}
                }
                tobe_deleted.push(at);
            } else if matches!(op, DfirOp::VectorChain(vc::Op::CreateAffineMask { .. })) {
                // *"All CreateAffineMaskOps should be connected to other operations that were
                // already lowered."*
                if !use_positions(&at, scope).is_empty() {
                    lowering.refusals.push(PtDangling::UsedMask(at));
                    return None;
                }
                tobe_deleted.push(at);
            }
            None
        });
    }

    // `for (auto op : tobe_deleted) VectorOperand::eraseOp(op);` — ⛔ DEEPEST FIRST, positions being
    // what names an op here.
    tobe_deleted.sort_unstable();
    tobe_deleted.dedup();
    for position in tobe_deleted.iter().rev() {
        erase_op(position, &mut unit.body);
    }
    lowering
}

/// The `sentient.scalar_constant` and `sentient.vector_mac` one dangling PT op becomes
/// (`VectorChainToSentientPT.cpp:906-948`).
#[allow(clippy::too_many_arguments)]
fn pt_dummy_mac(
    op: &DfirOp,
    at: &OpId,
    from: &VectorOperand,
    origin: Val,
    comp: ComputeComp,
    compute_precision: Option<sentient::Precision>,
    vector_op_to_xrfptr_map: &XrfPtrMap,
    reuse: &OperandReuse,
    values: &mut Values,
) -> PtMac {
    // `pointers = vector_op_to_xrfptr_map.at(op).at(0)` for an `agen.vector_load` and nothing else.
    let (xrf_write_ptr, xrf_read_ptr) = if matches!(op, DfirOp::Agen(agen::Op::VectorLoad { .. })) {
        match vector_op_to_xrfptr_map.get(at) {
            Some(ptrs) => (Some(ptrs.argument.write), Some(ptrs.argument.read)),
            None => return PtMac::Refused(PtDangling::Unrepresentable(at.clone())),
        }
    } else {
        (None, None)
    };

    // *"Expecting no on the fly conversions in PT"*.
    if from.orig_precision != from.on_the_fly_conv_precision {
        return PtMac::Refused(PtDangling::OnTheFlyConversion(at.clone()));
    }
    let (Some(precision), Some(compute_precision), Some(port)) = (
        input_precision_from_operand(from),
        compute_precision,
        from.name(),
    ) else {
        return PtMac::Refused(PtDangling::Unrepresentable(at.clone()));
    };
    let fold_mode = match fold_mode_attr_for_operation(op, comp) {
        FoldModeAttr::Absent => None,
        FoldModeAttr::Present(mode) => Some(mode),
        FoldModeAttr::Unsupported => {
            return PtMac::Refused(PtDangling::Unrepresentable(at.clone()));
        }
    };
    let id = match reuse.id(origin) {
        DataId::Unassigned => None,
        assigned => Some(assigned.attribute()),
    };

    // ⭐ THE MASK CONSTANT IS STILL BUILT, AND STILL NOT USED — *"PT masking is not attached to
    // operations"* (`:936`), the mask tree carrying it instead.
    let mask_const = sen::Op::Sentient(sentient::Op::ScalarConstant {
        value: 0,
        result: values.mint(),
        reg_locale: sentient::RegType::Imm,
        ty: ScalarTy::Index,
        is_symbol: false,
    });
    let west_receive = port == sentient::Port::West
        && matches!(op, DfirOp::Dataflow(dataflow::Op::Receive { .. }));
    let ((a, b, c), (id_a, id_b, id_c)) = if west_receive {
        (
            (sentient::Port::One, sentient::Port::Zero, port),
            (None, None, id),
        )
    } else {
        (
            (port, sentient::Port::One, sentient::Port::Zero),
            (id, None, None),
        )
    };
    let operand = |port: sentient::Port, data_id: Option<i32>| sentient::Operand {
        precision,
        data_id,
        ..sentient::Operand::from(port)
    };
    let mac = sen::Op::Sentient(sentient::Op::VectorMac {
        mask: None,
        xrf_write_ptr,
        xrf_read_ptr,
        results: Vec::new(),
        op_a: operand(a, id_a),
        op_b: operand(b, id_b),
        op_c: operand(c, id_c),
        result: sentient::ResultPorts {
            precision,
            ..sentient::ResultPorts::default()
        },
        mode: sentient::FmaMode::FusedMulAdd,
        compute_precision,
        fold_mode,
        unroll_factor: sentient::UnrollFactor::X1,
        xrf_read_incr: 0,
        xrf_write_incr: 0,
        data_transfer_only: false,
        dbg_name: dbg_name(op).map(str::to_owned),
    });
    PtMac::Emitted(PtDummyMac {
        at: at.clone(),
        ops: vec![mask_const, mac],
        mask: MaskedColumns(0),
    })
}

#[cfg(test)]
mod unit_tests {
    use super::{
        ComputeComp, MaskedColumns, OpId, VectorOperand, XrfPtrMap, compute_unit_precision,
        lower_dangling_non_compute_ops, run_on_operation,
    };
    use crate::arch::Target;
    use crate::bridges::dataflow_ir_to_sentient::vc_lowering_xrf::{XrfPtrPair, XrfPtrs};
    use crate::bridges::dataflow_ir_to_sentient::vc_operand_reuse::OperandReuse;
    use crate::bridges::dataflow_ir_to_sentient::vc_vector_operands::{
        OperandValue, VectorOperandType,
    };
    use crate::islands::dataflow_ir::dialects::{Index, Op as DfirOp, Val, agen, dataflow};
    use crate::islands::dataflow_ir::ty::{AffineExpr, AffineMap, ElemType, MemRef, Vector};
    use crate::islands::dataflow_ir::{self as dfir, ProgramUnit, Units, Values};
    use crate::islands::sentient::dialects::{Op as SenOp, sentient};
    use crate::units::{DfirUnit, Row};

    /// One `ptrow0` unit at `fp16` holding `body` — what this pass lowers.
    fn pt_unit(body: Vec<DfirOp>) -> ProgramUnit<Target> {
        let Some(row) = Row::checked(0) else {
            unreachable!("this arch has a row zero")
        };
        ProgramUnit {
            on: Units::one(DfirUnit::PtRow(row), Val(0)),
            precision: Some(dataflow::Precision::Fp16),
            body,
            arch: core::marker::PhantomData,
        }
    }

    /// `agen.vector_load %view[0, 0] : memref<64x64xf16>, vector<64xf16>` binding `result`.
    fn xrf_load(result: Val, view: Val) -> DfirOp {
        DfirOp::Agen(agen::Op::VectorLoad {
            dbg_name: None,
            result,
            view,
            indices: vec![Index::Const(0), Index::Const(0)],
            view_ty: MemRef {
                shape: vec![64, 64],
                elem: ElemType::F16,
            },
            ty: Vector {
                len: 64,
                elem: ElemType::F16,
            },
            multicast_info: None,
        })
    }

    /// One program holding one unit on `on` — what this pass walks.
    fn program_of(on: DfirUnit) -> dfir::Program<Target> {
        use crate::generated::OpFunc;
        use crate::islands::dataflow_ir::{
            Grid, GroupId, OpIndex, ProgramName, ProgramUnit, ProgramUnits, Units,
        };
        dfir::Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            grid: Grid::single(),
            preamble: Vec::new(),
            units: ProgramUnits::of(
                ProgramUnit {
                    on: Units::one(on, Val(0)),
                    precision: None,
                    body: Vec::new(),
                    arch: core::marker::PhantomData,
                },
                Vec::new(),
            ),
            arch: core::marker::PhantomData,
        }
    }

    /// 🎯 094/384 — THE ONE NON-IDENTITY ENTRY.
    ///
    /// `if (precision == "fp80") return "fp8";`
    /// (`VectorChainToSentientPT.cpp:37-38`) — the whole reason this function is not the identity.
    #[test]
    fn fp80_is_the_mlir_spelling_of_fp8() {
        assert_eq!(
            compute_unit_precision(dataflow::Precision::Fp80),
            sentient::Precision::Fp8
        );
    }

    /// 🎯 094/384 — AND EVERY OTHER SPELLING SURVIVES ITSELF.
    ///
    /// ⭐ THE FOUR MEASURED SPELLINGS ARE ALL HERE. A census of `precision = "…"` across the
    /// authority tree's `dcc/test` gives `int8` 500, `fp16` 492, `mxfp8` 4, `mxfp4` 4, `bf16` 3,
    /// `fp8` 2, `fp32` 2, `int4` 1 — and `xrf_increments.mlir:374` pins the `mxfp4` answer against
    /// `ComputePrecision = #sentient<precision mxfp4>` in its own `CHECK-SENT-IR`.
    #[test]
    fn every_other_precision_passes_through_unchanged() {
        for (from, to) in [
            (dataflow::Precision::Int8, sentient::Precision::Int8),
            (dataflow::Precision::Int4, sentient::Precision::Int4),
            (dataflow::Precision::Fp4, sentient::Precision::Fp4),
            (dataflow::Precision::Fp8, sentient::Precision::Fp8),
            (dataflow::Precision::Fp16, sentient::Precision::Fp16),
            (dataflow::Precision::Fp32, sentient::Precision::Fp32),
            (dataflow::Precision::Bf16, sentient::Precision::Bf16),
            (dataflow::Precision::Mxfp4, sentient::Precision::Mxfp4),
            (dataflow::Precision::Mxfp8, sentient::Precision::Mxfp8),
        ] {
            assert_eq!(compute_unit_precision(from), to, "{}", from.spelling());
            // ⭐⭐ AND THE ANSWER SPELLS ITSELF THE SAME. `return precision` returns the STRING, which
            // the caller then symbolizes — so a mapping that changed the spelling would change the
            // attribute, and this is the test that would catch it.
            assert_eq!(from.spelling(), to.spelling(), "{}", from.spelling());
        }
    }

    /// 🎯 379/384 — A PT ROW UNIT IS CLAIMED AND REACHES THE XRF PREPROCESSING. `redefine_constant_vectors`
    /// (entry 067) has already run for the whole module by then (`:983`), before the walk starts.
    #[test]
    #[should_panic(expected = "e367_createXrfIndexModifOps")]
    fn a_pt_row_unit_is_claimed_by_this_pass() {
        let Some(row) = Row::checked(0) else {
            unreachable!()
        };
        let mut program = program_of(DfirUnit::PtRow(row));
        run_on_operation(&mut program, &mut Values::default());
    }

    /// 🎯⛔ AND A NON-PT UNIT IS SKIPPED (`:990`) — the SFP is the PESFP pass's (entry 378), and a
    /// gate that let it through would lower it twice.
    #[test]
    fn a_non_pt_unit_is_left_to_the_pesfp_pass() {
        let mut program = program_of(DfirUnit::Sfp);
        run_on_operation(&mut program, &mut Values::default());
    }

    // ── 346/384 ───────────────────────────────────────────────────────────────────────────────

    /// 🎯 346/384 — TWO DANGLING `agen.vector_load`s ON A `ptxrf` VIEW (`xrf_increments.mlir:388`,
    /// `:414`), one of them absorbed. The unabsorbed one becomes a MAC that reads
    /// `xrf`/`one`/`zero`, binds the map's **argument** pointers and carries NO mask; the absorbed
    /// one emits nothing. ⛔ BOTH ARE ERASED.
    #[test]
    fn an_unabsorbed_xrf_load_becomes_a_pointer_bound_unmasked_mac_and_both_loads_go() {
        let mut vals = Values::default();
        let (handle, view, write, read) = (vals.mint(), vals.mint(), vals.mint(), vals.mint());
        let loaded = [vals.mint(), vals.mint()];
        let mut unit = pt_unit(vec![
            DfirOp::Dataflow(dataflow::Op::GetLocalUnit {
                result: handle,
                of: Val(0),
                which: dataflow::LocalUnit::PtXrf,
            }),
            DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                result: view,
                from: handle,
                start: Val(0),
                layout: AffineMap {
                    dims: 2,
                    syms: 0,
                    results: vec![AffineExpr::dim(1).times(64).plus(AffineExpr::dim(0))],
                },
                ty: MemRef {
                    shape: vec![64, 64],
                    elem: ElemType::F16,
                },
            }),
            xrf_load(loaded[0], view),
            xrf_load(loaded[1], view),
        ]);
        // The reuse pass latches the first of two loads reading the same `xrf` name, which leaves it
        // UNABSORBED, and marks the second absorbed.
        let mut reuse = OperandReuse::default();
        let mut operands: Vec<VectorOperand> = (2..4)
            .map(|at| {
                VectorOperand::new(
                    VectorOperandType::Xrf,
                    OperandValue::Port(sentient::Port::Xrf),
                    OpId::at(&[at]),
                )
            })
            .collect();
        reuse.set_reuse_information(&OpId::at(&[2]), &mut operands, &unit.body);
        let ptrs = XrfPtrMap::from([(
            OpId::at(&[2]),
            XrfPtrs {
                argument: XrfPtrPair { write, read },
                results: XrfPtrPair {
                    write: Val(90),
                    read: Val(91),
                },
            },
        )]);

        let lowering =
            lower_dangling_non_compute_ops(&mut unit, ComputeComp::Pt, &ptrs, &reuse, &mut vals);

        assert_eq!(lowering.refusals, Vec::new());
        let [mac] = lowering.macs.as_slice() else {
            panic!("one dummy MAC, not {:?}", lowering.macs)
        };
        assert_eq!((&mac.at, mac.mask), (&OpId::at(&[2]), MaskedColumns(0)));
        // `sentient.scalar_constant {value = 0 : si64} : index` — built, and NOT attached (`:936`).
        assert!(matches!(
            &mac.ops[0],
            SenOp::Sentient(sentient::Op::ScalarConstant { value: 0, .. })
        ));
        let SenOp::Sentient(sentient::Op::VectorMac {
            mask,
            xrf_write_ptr,
            xrf_read_ptr,
            op_a,
            op_b,
            op_c,
            result,
            compute_precision,
            fold_mode,
            data_transfer_only,
            ..
        }) = &mac.ops[1]
        else {
            panic!("a mac, not {:?}", mac.ops[1])
        };
        let reads = |port, data_id| sentient::Operand {
            precision: sentient::Precision::Fp16,
            data_id,
            ..sentient::Operand::from(port)
        };
        assert_eq!(
            (op_a, op_b, op_c),
            (
                &reads(sentient::Port::Xrf, Some(0)),
                &reads(sentient::Port::One, None),
                &reads(sentient::Port::Zero, None)
            )
        );
        assert_eq!(
            (*mask, *xrf_write_ptr, *xrf_read_ptr),
            (None, Some(write), Some(read))
        );
        // ⛔ ALL FOUR PRECISIONS ARE THE OPERAND'S — no `none` marker, and no `DataTransferOnly`.
        assert_eq!(
            (
                result.precision,
                *compute_precision,
                *fold_mode,
                *data_transfer_only
            ),
            (
                sentient::Precision::Fp16,
                sentient::Precision::Fp16,
                None,
                false
            )
        );
        assert_eq!(unit.body.len(), 2);
    }
}

// ⛔ RE-CREATED ANCHORS. These units' `crustify:todo:` markers were deleted without a
// `/// Replaces:` ever appearing, which removed them from every later schedule and let the
// driver report the campaign DONE. Outstanding work is now computed from UNITS.tsv.
// crustify:todo: e368_fuseNonComputeOps
// crustify:todo: e369_fuseComputeOps
