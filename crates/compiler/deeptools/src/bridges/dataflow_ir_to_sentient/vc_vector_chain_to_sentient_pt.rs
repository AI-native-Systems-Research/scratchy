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

use std::collections::BTreeMap;

use super::vc_helper::{EnclosingLoop, MaskValue, PtUnit, get_mask_value_for_pt};
use super::vc_loop_mask_tree::{MaskIncrement, MaskedColumns};
use super::vc_lowering_xrf::{
    DummyMacPtrs, MacXrfIncrements, XrfPtrMap, set_sentient_mac_xrf_reg_incr_attr,
};
use super::vc_operand_reuse::{DataId, OperandReuse};
use super::vc_vector_chain_helper::{
    FusionAnalysis, analyze_and_fill_operand_forwarding, analyze_and_fill_result_forwarding,
    analyze_non_compute_ops_for_fusion, input_precision_from_operand,
    is_sentient_binary_logical_op, redefine_constant_vectors, result_precision_from_operands,
    vector_binary_to_sentient_binary,
};
use super::vc_vector_chain_to_sentient_pesfp::{
    FoldModeAttr, fold_mode_attr_for_operation, walk_positions,
};
use super::vc_vector_operands::{
    ComputeComp, OpId, VectorOperand, VectorOperandType, defining_position, erase_op,
    erase_op_recording, erase_operands_recording, is_arith_constant, is_intermediate, op_at,
    origin_val, remove_at, same_block, use_positions,
};
use crate::arch::{Arch, IsaGen};
use crate::islands::dataflow_ir::dialects::vectorchain as vc;
use crate::islands::dataflow_ir::dialects::{
    Op as DfirOp, Val, agen, dataflow, dbg_name, operands as op_operands, vector,
};
use crate::islands::dataflow_ir::ty::{ScalarTy, Vector};
use crate::islands::dataflow_ir::{self as dfir, Values};
use crate::islands::sentient::dialects::{self as sen, Definitions, sentient};
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

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 368/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHETHER AN OP IS ONE OF THE THREE THIS FUSION CLAIMS — `isa<dataflow::SendOp, vector::StoreOp,
/// agen::VectorStoreOp>` (`VectorChainToSentientPT.cpp:56-57`, `:65-66`).
const fn is_send_or_store(op: &DfirOp) -> bool {
    matches!(
        op,
        DfirOp::Dataflow(dataflow::Op::Send { .. })
            | DfirOp::Vector(vector::Op::Store { .. })
            | DfirOp::Agen(agen::Op::VectorStore { .. })
    )
}

/// ONE FUSED PT MAC — the send or store it replaces, and the two ops that go in its place.
///
/// ⛔ THE MASK-TREE NODE AND THE MAC-MAP ENTRY BOTH TRAVEL WITH IT, for the reason [`PtDummyMac`]
/// gives: `updateLoopMaskTreeForConstantMask` (`:212`) and `replaceAndEraseDummyMacOps` (`:236`) key
/// by the MAC's own position, which it does not have until the caller places it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PtFusedMac {
    /// `OpBuilder builder(op)` — the position of the fused send or store.
    pub at: OpId,
    /// The `sentient.scalar_constant` then the `sentient.vector_mac`, in emission order.
    pub ops: Vec<sen::Op>,
    /// `mask_val` — ⭐ the literal `0`, and the MAC itself carries no mask (`:199`, `:212`).
    pub mask: MaskedColumns,
    /// `mac_op_to_xrfptr_map[mac_op]`, filled only when the xrf map knew `vector_op` (`:216-220`).
    pub xrf: Option<DummyMacPtrs>,
}

/// THE `sentient.set_send_dst` THE SFPRING DESTINATION EMITS — *"In case of SFPRing, only FMA result
/// can be sent to data fifo"* (`:173-183`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PtSendDst {
    /// `builder.setInsertionPointAfter(to.value().op_)` — ⛔ AFTER THE DESTINATION, not after the
    /// fused op, and ⛔ IT MOVES THE BUILDER for everything emitted later in the same call: the mask
    /// constant and the MAC then land after that destination instead of before `op`.
    pub after: OpId,
    /// `sentient::SetSendDestinationOp::create(builder, op->getLoc(), send_op.getToUnit())`.
    pub op: sen::Op,
}

/// EVERY WAY `fuseNonComputeOps` STOPS — ⛔ AND BOTH STOP THE WHOLE FUNCTION, not just one op.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PtFusionRefusal {
    /// `emitError("All to operands should be in the same block in order to be fused.")` +
    /// `signalPassFailure()` + `return` (`:229-233`) — ⛔ so `replaceAndEraseDummyMacOps` never runs,
    /// unlike the PESFP twin whose failure arm `continue`s.
    NotSameBlock(OpId),
    /// The unguarded `.value()`s, each of which aborts the compiler where it stands: an absent
    /// to-operand (`:151`), a port or precision that does not symbolize (`:198`, `:203-206`), a unit
    /// with no precision attribute (entry 094), and `dyn_cast<SendOp>(to.value().op_)` on the
    /// sfpring path (`:177-179`).
    Unrepresentable(OpId),
}

/// WHAT THE PT NON-COMPUTE FUSION PRODUCES.
///
/// ⭐ `dummy_macs` IS `mac_op_to_xrfptr_map`, AND IT IS RETURNED RATHER THAN APPLIED:
/// [`replace_and_erase_dummy_mac_ops`](super::vc_lowering_xrf::replace_and_erase_dummy_mac_ops)
/// (entry 093, `:236`) rewrites the SentientIR body the placeholder pointers live in, which does not
/// exist until the caller places these MACs. ⛔ IT MUST NOT RUN WHEN `refusal` IS SET.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PtNonComputeFusion {
    /// One per fused send or store, in walk order.
    pub macs: Vec<PtFusedMac>,
    /// The sfpring destinations, in the order their iterations reached them.
    pub send_dsts: Vec<PtSendDst>,
    /// `mac_op_to_xrfptr_map`, ready for entry 093.
    pub dummy_macs: Vec<DummyMacPtrs>,
    /// The one stop this function can take.
    pub refusal: Option<PtFusionRefusal>,
}

/// Replaces: e368_fuseNonComputeOps
///
/// **368/384** `VectorChainToSentientPTLoweringPass::fuseNonComputeOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:46` (191L).
///
/// FUSE EVERY SEND AND STORE INTO ONE `sentient.vector_mac` PER SOURCE, the destinations becoming its
/// `ResultForwarding` and the source its operand A or C.
///
/// ⛔ THE THREE `std::swap`s RUN ONCE PER DESTINATION AND ARE CUMULATIVE (`:152-154`), so two
/// destinations swap back; `opA_forwarding` and `opC_forwarding` are declared and NEVER pushed to
/// (`:148-149`); `from_ID[1]` is never written; and `op1` is read off the ORIGINAL `from`, not the
/// copy `setReuseInformation` may have re-valued to `latch`.
/// ⛔ `DT_CHECK(comp == PT)` IS UNREPRESENTABLE HERE, exactly as in [`compute_unit_precision`].
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn fuse_non_compute_ops<A: Arch>(
    unit: &mut dfir::ProgramUnit<A>,
    comp: ComputeComp,
    vector_op_to_xrfptr_map: &XrfPtrMap,
    reuse: &mut OperandReuse,
    values: &mut Values,
) -> PtNonComputeFusion {
    // `computeUnitPrecision(unit, comp)` (entry 094) — re-read per fused op there, one answer here.
    let unit_precision = unit.precision.map(compute_unit_precision);
    let mut out = PtNonComputeFusion {
        macs: Vec::new(),
        send_dsts: Vec::new(),
        dummy_macs: Vec::new(),
        refusal: None,
    };
    // ⛔ ONE LIST FOR THE WHOLE LOOP, REMOVED AT THE END DESCENDING. The reference erases inside the
    // loop, which an `Operation *` survives and a POSITION does not — the obligation
    // [`cleanup`](super::vc_vector_chain_to_sentient_pesfp::cleanup) carries for the same reason.
    // ⭐ AND IT IS STILL APPLIED AFTER A REFUSAL, because the reference's own earlier erases had
    // already happened when its `return` fired.
    let mut erased_list: Vec<OpId> = Vec::new();

    {
        let scope: &[DfirOp] = &unit.body;

        // `unit.walk<WalkOrder::PreOrder>(…)` (`:55-61`) — ⛔ THE WHOLE LIST IS SEEDED BEFORE ANY OP
        // IS FUSED, so entry 341 can mark a sibling this loop has not reached yet.
        let mut ops_list: Vec<OpId> = Vec::new();
        let mut is_visited: BTreeMap<OpId, bool> = BTreeMap::new();
        walk_positions(scope, &[], 0, &mut |op, at| -> Option<()> {
            if is_send_or_store(op) {
                ops_list.push(at.clone());
                is_visited.insert(at, false);
            }
            None
        });

        for at in &ops_list {
            // `if (!is_visited[op] && (isa<…>(op))) { is_visited[op] = true; … }`
            let Some(op) = op_at(at, scope) else { continue };
            if is_visited.get(at).copied().unwrap_or(false) || !is_send_or_store(op) {
                continue;
            }
            is_visited.insert(at.clone(), true);

            // `send_op.getSendData()` / `store_op.getValueToStore()` — the same value all three arms
            // resolve `from` through (`:90-107`).
            let Some(data) = (match op {
                DfirOp::Dataflow(dataflow::Op::Send { data, .. })
                | DfirOp::Vector(vector::Op::Store { value: data, .. })
                | DfirOp::Agen(agen::Op::VectorStore { value: data, .. }) => Some(*data),
                // `} else { is_fusion_respected = false; }` (`:109-110`) — unreachable under the
                // `isa<>` above, and the reference then falls straight into the `from` test.
                _ => None,
            }) else {
                continue;
            };
            let defining = defining_position(data, scope);

            // `Operation *vector_op = op;` and the send's look-past (`:70-81`): the FIRST user of the
            // send data's defining op that the xrf map knows, `break`ing on it. ⛔ THE SEARCH IS OVER
            // THE USERS, not over the map — the map's iteration order does not decide this.
            let mut vector_op = at.clone();
            if matches!(op, DfirOp::Dataflow(dataflow::Op::Send { .. }))
                && let Some(defining) = &defining
            {
                vector_op = defining.clone();
                for user in use_positions(defining, scope) {
                    if vector_op_to_xrfptr_map.contains_key(&user) {
                        vector_op = user;
                        break;
                    }
                }
            }
            // `if (count(vector_op) > 0) { pointers = …at(0); result_types = {Index, Index}; }`
            // (`:83-86`) — ⛔ `at(0)` IS THE ARGUMENT PAIR; the two index results are the MAC's own.
            let xrf = vector_op_to_xrfptr_map.get(&vector_op);

            // `VectorOperand::getOperandWithPrecision(ctx, …, comp, is_precision_converted)`, whose
            // `traverse_upwards` defaults to true. ⛔ `is_precision_converted_global` IS OR'd IN THREE
            // TIMES AND NEVER READ (`:88`, `:96`, `:102`, `:108`) — a dead out-parameter.
            let Some(from) = defining
                .as_ref()
                .and_then(|def| VectorOperand::operand::<A>(def, comp, true, scope))
            else {
                continue;
            };

            // `if (from.has_value() && isa<ReceiveOp, vector::LoadOp, agen::VectorLoadOp,
            //  arith::ConstantOp>(from.value().op_))` (`:113-116`).
            let fusible = matches!(
                op_at(&from.op, scope),
                Some(
                    DfirOp::Dataflow(dataflow::Op::Receive { .. })
                        | DfirOp::Vector(vector::Op::Load { .. })
                        | DfirOp::Agen(agen::Op::VectorLoad { .. })
                )
            ) || is_arith_constant(&from.op, scope);
            if !fusible {
                continue;
            }

            // `analyzeNonComputeOpsForFusion(unit, …)` (entry 341) — ⭐ `unit` IS THE SCOPE'S OWN
            // ROOT, so the containment test is the empty prefix.
            let mut analysis = FusionAnalysis::default();
            analyze_non_compute_ops_for_fusion::<A>(
                &OpId::at(&[]),
                at,
                &from,
                comp,
                &mut is_visited,
                scope,
                &mut analysis,
            );
            // `if (!is_fusion_respected) continue;` — ⭐ ONE OP, not the pass.
            if !analysis.is_fusion_respected {
                continue;
            }

            // `from_operands.push_back(from); reuse_info.setReuseInformation(op, from_operands);` —
            // ⛔ BY REFERENCE, and it can re-value the copy to `latch`, which is what `eraseOperands`
            // reads to decide the source op stays.
            let mut from_operands = vec![from.clone()];
            reuse.set_reuse_information(at, &mut from_operands, scope);

            let result_precision = result_precision_from_operands(&analysis.to_operands);
            // `DT_CHECK_MSG(from_operands.size() == 1, "Non compute ops expected to have one input
            //  operand")` — the vector was just built with one element.
            let mut op_a_precision = input_precision_from_operand(&from_operands[0]);
            // `opB_precision = opC_precision = compute_precision;`
            let op_b_precision = unit_precision;
            let mut op_c_precision = unit_precision;

            // `if (VectorOperand::sameBlock(op, to_operands).succeeded())` — the LIST overload
            // (`VectorOperands.cpp:670-685`): an absent entry fails it, and an empty list passes.
            if !analysis
                .to_operands
                .iter()
                .all(|to| same_block(at, to.as_ref(), scope))
            {
                out.refusal = Some(PtFusionRefusal::NotSameBlock(at.clone()));
                break;
            }

            // `std::string op1 = from.value().getName(), op2 = "one", op3 = "zero";`
            let Some(from_port) = from.name() else {
                out.refusal = Some(PtFusionRefusal::Unrepresentable(at.clone()));
                break;
            };
            let mut op1 = from_port;
            let mut op2 = sentient::Port::One;
            let mut op3 = sentient::Port::Zero;

            // `int from_ID[3] = {reuse_info.getId(from.value().op_).value(), -1, -1};` — ⛔ `[1]` IS
            // NEVER WRITTEN, and `None` is this island's spelling of that -1.
            let mut from_id: [Option<i32>; 3] = [
                match origin_val(&from.op, scope).map_or(DataId::Unassigned, |val| reuse.id(val)) {
                    DataId::Unassigned => None,
                    assigned => Some(assigned.attribute()),
                },
                None,
                None,
            ];

            let mut result_forwarding: Vec<sentient::Port> = Vec::new();
            let mut aborted = false;
            for to in &analysis.to_operands {
                // `std::string dest = to.value().getName();` — ⛔ BOTH `.value()`s ARE UNGUARDED and
                // entry 342 pushes an absent operand.
                let Some(dest) = to.as_ref().and_then(VectorOperand::name) else {
                    aborted = true;
                    break;
                };

                // *"Use the accumulation part if plan to use FMA"* (`:150-154`) — ⛔ once per
                // destination, cumulatively.
                core::mem::swap(&mut op1, &mut op3);
                core::mem::swap(&mut op_a_precision, &mut op_c_precision);
                from_id.swap(0, 2);
                result_forwarding.push(dest);

                // *"the hardware uses the src0 port value to transform resultant FP8 data into FP9
                // and store into XRF"* (`:156-169`) — `getArch() <= RCUDD1A_ISA`, read against the
                // ports AS THEY STAND AFTER this iteration's swap.
                if A::GEN <= IsaGen::Rcudd1a
                    && unit_precision == Some(sentient::Precision::Fp8)
                    && op1 == sentient::Port::Zero
                    && op2 == sentient::Port::One
                    && op3 == sentient::Port::North
                    && matches!(
                        to.as_ref().map(|to| to.kind),
                        Some(
                            VectorOperandType::Lrf
                                | VectorOperandType::Xrf
                                | VectorOperandType::Link
                        )
                    )
                {
                    op2 = sentient::Port::North;
                }

                // `if (dest == "sfpring")` (`:171-183`) — ⛔ `dyn_cast<SendOp>` IS NOT TESTED before
                // `getToUnit()`, so a destination named `sfpring` that is not a send aborts there.
                if dest == sentient::Port::SfpRing {
                    let after = to.as_ref().map(|to| to.op.clone());
                    let units = after.as_ref().and_then(|after| match op_at(after, scope) {
                        Some(DfirOp::Dataflow(dataflow::Op::Send { to, .. })) => Some(*to),
                        _ => None,
                    });
                    let (Some(after), Some(units)) = (after, units) else {
                        aborted = true;
                        break;
                    };
                    out.send_dsts.push(PtSendDst {
                        after,
                        op: sen::Op::Sentient(sentient::Op::SetSendDst { units }),
                    });
                }
            }

            let fold_mode = match fold_mode_attr_for_operation(op, comp) {
                FoldModeAttr::Absent => None,
                FoldModeAttr::Present(mode) => Some(mode),
                FoldModeAttr::Unsupported => {
                    aborted = true;
                    None
                }
            };
            // `symbolizeSentientPrecision(..).value()` on all five (`:203-207`).
            let (
                Some(op_a_precision),
                Some(op_b_precision),
                Some(op_c_precision),
                Some(result_precision),
                Some(compute_precision),
            ) = (
                op_a_precision,
                op_b_precision,
                op_c_precision,
                result_precision,
                unit_precision,
            )
            else {
                out.refusal = Some(PtFusionRefusal::Unrepresentable(at.clone()));
                break;
            };
            if aborted {
                out.refusal = Some(PtFusionRefusal::Unrepresentable(at.clone()));
                break;
            }

            // `sentient::ConstantOp::create(builder, loc, getIndexType(), 0)` (`:185-186`) — ⭐ STILL
            // BUILT AND STILL NOT ATTACHED: *"PT does not associate mask directly to operations"*.
            let mask_const = sen::Op::Sentient(sentient::Op::ScalarConstant {
                value: 0,
                result: values.mint(),
                reg_locale: sentient::RegType::Imm,
                ty: ScalarTy::Index,
                is_symbol: false,
            });

            // `TypeRange(result_types)` — the two index results exist exactly when the xrf map knew
            // `vector_op`, and they are what [`DummyMacPtrs`] reads back.
            let results: Vec<Val> = if xrf.is_some() {
                vec![values.mint(), values.mint()]
            } else {
                Vec::new()
            };
            let operand = |port, precision, data_id| sentient::Operand {
                precision,
                data_id,
                ..sentient::Operand::from(port)
            };
            let mut mac = sen::Op::Sentient(sentient::Op::VectorMac {
                // `nullptr` (`:199`).
                mask: None,
                xrf_write_ptr: xrf.map(|ptrs| ptrs.argument.write),
                xrf_read_ptr: xrf.map(|ptrs| ptrs.argument.read),
                results,
                op_a: operand(op1, op_a_precision, from_id[0]),
                op_b: operand(op2, op_b_precision, from_id[1]),
                op_c: operand(op3, op_c_precision, from_id[2]),
                // ⛔ THREE OF THE FOUR FORWARDING LISTS ARE EMPTY BY CONSTRUCTION — `opA_forwarding`
                // and `opC_forwarding` are never pushed to and opB's is a literal `{}` (`:202-205`),
                // so only the result's carries the destinations. [`sentient::Operand::from`] gives
                // the other three theirs empty.
                result: sentient::ResultPorts {
                    forwarding: result_forwarding,
                    precision: result_precision,
                    unroll_incr: false,
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

            let mut dummy: Option<DummyMacPtrs> = None;
            // `if (count(vector_op) > 0) { setSentientMacXrfRegIncrAttr(&mac_op, &builder,
            //  to_operands[0].value().orig_precision_, ctx); mac_op_to_xrfptr_map[mac_op] = …; }`
            // (`:216-220`) — ⭐ THE FIRST DESTINATION'S ORIGINAL PRECISION, not the unit's.
            if let Some(ptrs) = xrf {
                let Some(orig) = analysis
                    .to_operands
                    .first()
                    .and_then(|to| to.as_ref())
                    .and_then(|to| to.orig_precision)
                else {
                    out.refusal = Some(PtFusionRefusal::Unrepresentable(at.clone()));
                    break;
                };
                if let sen::Op::Sentient(inner) = &mut mac
                    && let Some(incr) = MacXrfIncrements::of(inner)
                {
                    set_sentient_mac_xrf_reg_incr_attr::<A>(incr, orig);
                }
                dummy = DummyMacPtrs::of(&mac, *ptrs);
                out.dummy_macs.extend(dummy.clone());
            }

            out.macs.push(PtFusedMac {
                at: at.clone(),
                ops: vec![mask_const, mac],
                mask: MaskedColumns(0),
                xrf: dummy,
            });

            // `VectorOperand::eraseOperands(to_operands);` and, only when nothing dangles,
            // `eraseOperands(from_operands)` (`:222-226`).
            erase_operands_recording(&analysis.to_operands, scope, &mut erased_list);
            if !analysis.is_dangling_ops_present_after_fusion {
                let from_operands: Vec<Option<VectorOperand>> =
                    from_operands.into_iter().map(Some).collect();
                erase_operands_recording(&from_operands, scope, &mut erased_list);
            }
        }
    }

    erased_list.sort_unstable();
    erased_list.dedup();
    for position in erased_list.iter().rev() {
        remove_at(position.path(), &mut unit.body);
    }
    out
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 369/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// THE THREE OPS THE COMPUTE WALK COLLECTS — `isa<vectorchain::MultiplyAndAccumulateOp>() ||
/// isa<vectorchain::MultiplyOp>() || isa<vectorchain::BinaryOp>()`
/// (`VectorChainToSentientPT.cpp:252-260`).
const fn is_pt_compute(op: &DfirOp) -> bool {
    matches!(
        op,
        DfirOp::VectorChain(
            vc::Op::MultiplyAccumulate { .. } | vc::Op::Multiply { .. } | vc::Op::Binary { .. }
        )
    )
}

/// `op->getResult(0).getType()` for the three ops [`is_pt_compute`] admits — the row
/// [`get_mask_value_for_pt`] measures the mask against.
const fn compute_result_ty(op: &DfirOp) -> Option<Vector> {
    match op {
        DfirOp::VectorChain(
            vc::Op::Multiply { ty, .. }
            | vc::Op::MultiplyAccumulate { ty, .. }
            | vc::Op::Binary { ty, .. },
        ) => Some(*ty),
        _ => None,
    }
}

/// ONE OPERAND SLOT AS THE EMITTED OP CARRIES IT — `symbolizeSentientComputePort(…getName()).value()`
/// with its forwarding list, precision and `reuse_info.getId(…op_).value()`.
fn pt_operand(
    port: sentient::Port,
    precision: sentient::Precision,
    data_id: Option<i32>,
    forwarding: Vec<sentient::Port>,
) -> sentient::Operand {
    sentient::Operand {
        forwarding,
        precision,
        data_id,
        ..sentient::Operand::from(port)
    }
}

/// `symbolizeSentientComputePort(operand.getName()).value()` AND `reuse_info.getId(operand.op_)` —
/// [`None`] where the reference calls `.value()` on an empty optional, and an unassigned id is the
/// `.td`'s own -1 (the precedent [`super::vc_vector_chain_to_sentient_pesfp`] states).
fn port_and_id(
    operand: Option<&VectorOperand>,
    reuse: &OperandReuse,
    scope: &[DfirOp],
) -> Option<(sentient::Port, Option<i32>)> {
    let operand = operand?;
    let port = operand.name()?;
    let id = match origin_val(&operand.op, scope).map_or(DataId::Unassigned, |val| reuse.id(val)) {
        DataId::Unassigned => None,
        assigned => Some(assigned.attribute()),
    };
    Some((port, id))
}

/// THE MASK NODE ONE LOWERED PT MAC OWES THE LOOP MASK TREE — ⛔ DEFERRED FOR THE SAME REASON
/// [`PtFusedMac::mask`] is: `updateLoopMaskTreeFor{Constant,Dynamic}Mask` keys by the MAC's own
/// position in the SentientIR body, which it does not have until the caller places it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtMacMask {
    /// `updateLoopMaskTreeForConstantMask(tree, op, &mac_op, <columns>)` — the literal `0` for a
    /// `multiply` and for a MAC with no mask operand, the constant's own value otherwise.
    Constant(MaskedColumns),
    /// `updateLoopMaskTreeForDynamicMask(tree, mask, &mac_op, 0, 1)` — *"Mask will start at 0,
    /// increment by 1 each loop iteration"*.
    Dynamic {
        /// The loop induction variable [`MaskValue::LoopIterator`] answered.
        iterator: Val,
        /// `start_val` — always the literal `0` here.
        start: MaskedColumns,
        /// `increment` — always [`MaskIncrement::PerParentLoopIteration`] here.
        increment: MaskIncrement,
    },
}

/// ONE LOWERED PT COMPUTE.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PtLoweredCompute {
    /// `VectorOperand::eraseOp(op)` — the `vectorchain` op this replaces.
    pub replaces: OpId,
    /// Where [`Self::ops`] go: `OpBuilder builder(op)`, or `to_operands[0].value().op_` when the xrf
    /// map knew `vector_op` (`:378-383`) — ⛔ so the MAC lands before the TRANSFER, not before `op`.
    pub insert_before: OpId,
    /// The `sentient.scalar_constant` [`get_mask_value_for_pt`] emitted — ⛔ AT [`Self::replaces`],
    /// because its builder was constructed before the move above, and DEAD on a MAC, which reads no
    /// mask.
    pub mask_ops: Vec<sen::Op>,
    /// The `sentient.vector_mac` or `sentient.vector_binary`.
    pub ops: Vec<sen::Op>,
    /// The `sentient.set_send_dst` ops entry 342 emitted for an `sfpring` destination, which it
    /// leaves to the caller to position after the send.
    pub send_dsts: Vec<sen::Op>,
    /// The mask node, absent for a `vector_binary` — only MACs control PT masking.
    pub mask: Option<PtMacMask>,
    /// `mac_op_to_xrfptr_map[mac_op]`.
    pub xrf: Option<DummyMacPtrs>,
}

/// EVERY WAY `fuseComputeOps` STOPS — ⛔ AND BOTH STOP THE WHOLE FUNCTION, so
/// `replaceAndEraseDummyMacOps` (`:875`) never runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PtComputeRefusal {
    /// `if (!mask_val.has_value()) { signalPassFailure(); return; }` (`:347-350`) — entry 089
    /// declined the mask.
    MaskUnsupported(OpId),
    /// The unguarded `.value()`s and `DT_CHECK_MSG`s, each of which aborts the compiler where it
    /// stands: an absent operand or a port that does not symbolize (`:396-402`), a precision that
    /// does not (`:404-408`), a `LogicalResultForwarding` in a PT unit (`:361-362`), a mask on a
    /// `multiply` (`:391-392`), a MISSING mask on a `binary` (`:489`), and a unit with no precision
    /// attribute (entry 094).
    Unrepresentable(OpId),
}

/// WHAT THE PT COMPUTE FUSION PRODUCES — ⭐ `dummy_macs` IS RETURNED RATHER THAN APPLIED, exactly as
/// in [`PtNonComputeFusion`], and ⛔ MUST NOT BE APPLIED WHEN `refusal` IS SET.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PtComputeFusion {
    /// One per lowered compute, in walk order.
    pub computes: Vec<PtLoweredCompute>,
    /// `mac_op_to_xrfptr_map`, ready for entry 093.
    pub dummy_macs: Vec<DummyMacPtrs>,
    /// The one stop this function can take.
    pub refusal: Option<PtComputeRefusal>,
}

/// Replaces: e369_fuseComputeOps
///
/// **369/384** `VectorChainToSentientPTLoweringPass::fuseComputeOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:245` (628L).
///
/// LOWER EVERY `vectorchain.multiply`, `multiply_and_accumulate` and `binary` IN A PT UNIT TO ITS
/// `sentient.vector_mac` or `sentient.vector_binary`, then erase the chain that fed it.
///
/// ⛔ **THE DISPATCH BEYOND THOSE THREE IS DEAD CODE.** The walk collects only those three classes
/// (`:252-260`), yet the body then branches on `ElementWiseCompareOp`, `ElementWiseSelectionOp`,
/// `ShuffleOp`, `ScanWithGapOp`, seven estimate ops and `PackOp` (`:519-869`), and MLIR's `isa<>` is
/// concrete-class matching with no hierarchy — none of them can arrive. The leading
/// `dyn_cast<ShuffleOp>` skip (`:264-271`) is dead for the same reason.
/// ⛔ THE `binary` ARM DISCARDS the `opA_forwarding`/`opB_forwarding` it just computed and passes two
/// literal `{}`s (`:508-509`).
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn fuse_compute_ops<A: Arch>(
    unit: &mut dfir::ProgramUnit<A>,
    comp: ComputeComp,
    vector_op_to_xrfptr_map: &XrfPtrMap,
    reuse: &mut OperandReuse,
    definitions: Definitions<'_>,
    loops: &[EnclosingLoop<'_>],
    values: &mut Values,
) -> PtComputeFusion {
    // `computeUnitPrecision(unit, comp)` (entry 094, `:365`) — re-read per op there, one answer here.
    let unit_precision = unit.precision.map(compute_unit_precision);
    let mut out = PtComputeFusion {
        computes: Vec::new(),
        dummy_macs: Vec::new(),
        refusal: None,
    };
    // ⛔ ONE LIST FOR THE WHOLE LOOP, REMOVED DESCENDING AT THE END — the reference erases inside its
    // loop, which an `Operation *` survives and an [`OpId`] does not. See [`fuse_non_compute_ops`].
    let mut erased_list: Vec<OpId> = Vec::new();

    {
        let scope: &[DfirOp] = &unit.body;

        // `unit.walk<WalkOrder::PreOrder>(…)` (`:252-260`).
        let mut ops_list: Vec<OpId> = Vec::new();
        walk_positions(scope, &[], 0, &mut |op, at| -> Option<()> {
            if is_pt_compute(op) {
                ops_list.push(at);
            }
            None
        });

        for at in &ops_list {
            let Some(op) = op_at(at, scope) else { continue };
            let reads = op_operands(op);

            // `int numOperands = 2; if (isa<MultiplyAndAccumulateOp>(op)) numOperands = 3;`
            let num_operands = usize::from(matches!(
                op,
                DfirOp::VectorChain(vc::Op::MultiplyAccumulate { .. })
            )) + 2;

            // `for (int i = 0; i < numOperands; i++)` (`:283-296`) — ⛔ THE LAST MATCHING OPERAND
            // WINS: `vector_op` is assigned inside the loop with no `break`.
            let mut vector_op: Option<OpId> = None;
            let mut from_operands: Vec<Option<VectorOperand>> = Vec::new();
            for read in reads.iter().take(num_operands) {
                let operand_op = defining_position(*read, scope);
                if let Some(operand_op) = &operand_op
                    && vector_op_to_xrfptr_map.contains_key(operand_op)
                    && matches!(
                        op_at(operand_op, scope),
                        Some(
                            DfirOp::Agen(agen::Op::VectorLoad { .. })
                                | DfirOp::Vector(vector::Op::Load { .. })
                        )
                    )
                {
                    vector_op = Some(operand_op.clone());
                }
                from_operands.push(operand_op.and_then(|operand_op| {
                    VectorOperand::operand::<A>(&operand_op, comp, true, scope)
                }));
            }

            // `if (!vector_op) { for (auto user : op->getUsers()) { … } }` (`:298-315`) — the store
            // side, *"There could be an intermediate operations between the operand and the store
            // op"*, and the FIRST map hit wins (`break`).
            if vector_op.is_none() {
                for mut user in use_positions(at, scope) {
                    while is_intermediate(&user, scope) {
                        let mut users = use_positions(&user, scope);
                        // `user->hasOneUse()`
                        if users.len() != 1 {
                            break;
                        }
                        user = users.remove(0);
                    }
                    if vector_op_to_xrfptr_map.contains_key(&user)
                        && matches!(
                            op_at(&user, scope),
                            Some(
                                DfirOp::Vector(vector::Op::Store { .. })
                                    | DfirOp::Agen(agen::Op::VectorStore { .. })
                            )
                        )
                    {
                        vector_op = Some(user);
                        break;
                    }
                }
            }
            // ⭐ `count(vector_op)` WITH A NULL `vector_op` IS 0, so every later test on it reads as
            // "the map knew it" — which is what the [`Option`] says outright.
            let xrf = vector_op
                .as_ref()
                .and_then(|vector_op| vector_op_to_xrfptr_map.get(vector_op));

            // `if (op->getNumOperands() == numOperands + 1) mask_operand = op->getOperand(
            //  numOperands).getDefiningOp();` (`:317-322`) — ⭐ ONLY A `binary` CAN HAVE ONE in this
            // island: `Multiply` and `MultiplyAccumulate` carry no mask field at all.
            let mask_operand = if reads.len() == num_operands + 1 {
                defining_position(reads[num_operands], scope)
            } else {
                None
            };

            // `reuse_info.setReuseInformation(op, from_operands);` (`:324`) — ⛔ BY REFERENCE, so the
            // latch re-valuing IS what every `from_operands[i].value().getName()` below reads. Its
            // own `if (!operands[i].has_value()) return nullopt;` makes it a no-op when one is
            // absent, which is what skipping the call is.
            if let Some(mut operands) = from_operands
                .iter()
                .cloned()
                .collect::<Option<Vec<VectorOperand>>>()
            {
                reuse.set_reuse_information(at, &mut operands, scope);
                from_operands = operands.into_iter().map(Some).collect();
            }

            // `analyzeAndFillOperandForwarding(…)` on 0, then 1 and 2 if the list is that long
            // (`:326-339`) — entry 340.
            let mut op_a_forwarding: Vec<sentient::Port> = Vec::new();
            let mut op_b_forwarding: Vec<sentient::Port> = Vec::new();
            let mut op_c_forwarding: Vec<sentient::Port> = Vec::new();
            analyze_and_fill_operand_forwarding::<A>(
                comp,
                from_operands[0].as_ref(),
                scope,
                &mut op_a_forwarding,
            );
            if from_operands.len() > 1 {
                analyze_and_fill_operand_forwarding::<A>(
                    comp,
                    from_operands[1].as_ref(),
                    scope,
                    &mut op_b_forwarding,
                );
            }
            if from_operands.len() > 2 {
                analyze_and_fill_operand_forwarding::<A>(
                    comp,
                    from_operands[2].as_ref(),
                    scope,
                    &mut op_c_forwarding,
                );
            }

            // `if (mask_operand.has_value()) { mask_val = getMaskValueForPT(…); if (!mask_val
            //  .has_value()) { signalPassFailure(); return; } }` (`:341-351`) — *"PT does not
            //  associate mask directly to operations"*, entry 089.
            let mut mask_ops: Vec<sen::Op> = Vec::new();
            let mut mask_val: Option<MaskValue> = None;
            if let Some(mask_at) = &mask_operand {
                let (Some(pt), Some(masked), Some(DfirOp::VectorChain(mask_op))) = (
                    // `DT_CHECK_MSG(comp == PT, …)` on entry — the witness, not a comparison.
                    match comp {
                        ComputeComp::Pt => Some(PtUnit),
                        ComputeComp::Pe | ComputeComp::Sfp => None,
                    },
                    compute_result_ty(op),
                    op_at(mask_at, scope),
                ) else {
                    out.refusal = Some(PtComputeRefusal::Unrepresentable(at.clone()));
                    break;
                };
                match get_mask_value_for_pt::<A>(pt, masked, mask_op, definitions, loops, values) {
                    Some(value) => {
                        // ⭐ THE CONSTANT ARM EMITS ITS OP EVEN WHERE NOTHING READS IT.
                        if let MaskValue::Constant { op, .. } = &value {
                            mask_ops.push(op.clone());
                        }
                        mask_val = Some(value);
                    }
                    None => {
                        out.refusal = Some(PtComputeRefusal::MaskUnsupported(at.clone()));
                        break;
                    }
                }
            }

            // `analyzeAndFillResultForwarding(…)` (`:353-360`) — entry 342.
            let mut to_operands: Vec<Option<VectorOperand>> = Vec::new();
            let mut result_forwarding: Vec<sentient::Port> = Vec::new();
            let mut logical_result_forwarding: Option<sentient::Port> = None;
            let mut send_dsts: Vec<sen::Op> = Vec::new();
            analyze_and_fill_result_forwarding::<A>(
                at,
                comp,
                scope,
                &mut to_operands,
                &mut result_forwarding,
                &mut logical_result_forwarding,
                &mut send_dsts,
            );
            // `DT_CHECK_MSG(!logical_result_forwarding, "LogicalResultForwarding should not appear in
            //  PT units")` — ⛔ AN ABORT, not a skip: entry 342 splits off an `istate` destination.
            if logical_result_forwarding.is_some() {
                out.refusal = Some(PtComputeRefusal::Unrepresentable(at.clone()));
                break;
            }

            // `:364-374` — entry 061, entry 094, then entry 060 per slot. ⛔ AN ABSENT B OR C SLOT
            // CARRIES THE **COMPUTE** PRECISION, not nothing.
            let result_precision = result_precision_from_operands(&to_operands);
            let op_a_precision = from_operands[0]
                .as_ref()
                .and_then(input_precision_from_operand);
            let op_b_precision = match from_operands.get(1) {
                Some(operand) => operand.as_ref().and_then(input_precision_from_operand),
                None => unit_precision,
            };
            let op_c_precision = match from_operands.get(2) {
                Some(operand) => operand.as_ref().and_then(input_precision_from_operand),
                None => unit_precision,
            };

            // `getSentientFoldModeAttrForOperation(op, comp)` (`:376`).
            let (fold_mode, fold_supported) = match fold_mode_attr_for_operation(op, comp) {
                FoldModeAttr::Absent => (None, true),
                FoldModeAttr::Present(mode) => (Some(mode), true),
                FoldModeAttr::Unsupported => (None, false),
            };
            let op_dbg_name = dbg_name(op).map(str::to_owned);

            // `symbolizeSentientPrecision(..).value()` on all five, and every arm reads all of them.
            let (
                Some(op_a_precision),
                Some(op_b_precision),
                Some(op_c_precision),
                Some(result_precision),
                Some(compute_precision),
            ) = (
                op_a_precision,
                op_b_precision,
                op_c_precision,
                result_precision,
                unit_precision,
            )
            else {
                out.refusal = Some(PtComputeRefusal::Unrepresentable(at.clone()));
                break;
            };
            if !fold_supported {
                out.refusal = Some(PtComputeRefusal::Unrepresentable(at.clone()));
                break;
            }

            // `if (count(vector_op) > 0) builder.setInsertionPoint(to_operands[0].value().op_);`
            // (`:378-383`) — *"XRF-related scalar adds were already created, right before the
            // transfer's SendOp/StoreOp"*.
            let insert_before = if xrf.is_some() {
                match to_operands.first().and_then(|to| to.as_ref()) {
                    Some(to) => to.op.clone(),
                    None => {
                        out.refusal = Some(PtComputeRefusal::Unrepresentable(at.clone()));
                        break;
                    }
                }
            } else {
                at.clone()
            };

            // `TypeRange(result_types)` — the two index results exist exactly when the xrf map knew
            // `vector_op`, and they are what [`DummyMacPtrs`] reads back. ⛔ A `binary` gets none:
            // `sentient::BinaryOp::create` takes no result types and no pointers.
            let mac_results: Vec<Val> = if xrf.is_some() {
                vec![values.mint(), values.mint()]
            } else {
                Vec::new()
            };

            let (Some((port_a, id_a)), Some((port_b, id_b))) = (
                port_and_id(from_operands[0].as_ref(), reuse, scope),
                port_and_id(from_operands.get(1).and_then(Option::as_ref), reuse, scope),
            ) else {
                out.refusal = Some(PtComputeRefusal::Unrepresentable(at.clone()));
                break;
            };

            let mut lowered = match op {
                // ── `isa<vectorchain::MultiplyOp>(op)` (`:386-421`) ──────────────────────────────
                DfirOp::VectorChain(vc::Op::Multiply { .. }) => {
                    // `DT_CHECK_MSG(comp == PT && !mask_val.has_value(), "expecting PT with no mask
                    //  op")` — and this island cannot give a `multiply` a mask operand at all.
                    if comp != ComputeComp::Pt || mask_val.is_some() {
                        out.refusal = Some(PtComputeRefusal::Unrepresentable(at.clone()));
                        break;
                    }
                    PtLoweredCompute {
                        replaces: at.clone(),
                        insert_before,
                        mask_ops,
                        ops: vec![sen::Op::Sentient(sentient::Op::VectorMac {
                            // `nullptr` — *"MACOps in PT units do not have/use mask operand"*.
                            mask: None,
                            xrf_write_ptr: xrf.map(|ptrs| ptrs.argument.write),
                            xrf_read_ptr: xrf.map(|ptrs| ptrs.argument.read),
                            results: mac_results,
                            op_a: pt_operand(port_a, op_a_precision, id_a, op_a_forwarding),
                            op_b: pt_operand(port_b, op_b_precision, id_b, op_b_forwarding),
                            // `SentientComputePort::zero` — ⛔ AND NO THIRD DATA ID IS PASSED, so
                            // operand C keeps the `.td`'s -1.
                            op_c: pt_operand(
                                sentient::Port::Zero,
                                op_c_precision,
                                None,
                                op_c_forwarding,
                            ),
                            result: sentient::ResultPorts {
                                forwarding: result_forwarding,
                                precision: result_precision,
                                unroll_incr: false,
                            },
                            // ⭐ NO `mode` ARGUMENT — the `.td`'s default.
                            mode: sentient::FmaMode::FusedMulAdd,
                            compute_precision,
                            fold_mode,
                            unroll_factor: sentient::UnrollFactor::X1,
                            xrf_read_incr: 0,
                            xrf_write_incr: 0,
                            data_transfer_only: false,
                            dbg_name: op_dbg_name,
                        })],
                        send_dsts,
                        // `updateLoopMaskTreeForConstantMask(pt_masking_tree, op, &mac_op, 0)`.
                        mask: Some(PtMacMask::Constant(MaskedColumns(0))),
                        xrf: None,
                    }
                }

                // ── `isa<vectorchain::MultiplyAndAccumulateOp>(op)` (`:423-487`) ─────────────────
                DfirOp::VectorChain(vc::Op::MultiplyAccumulate { .. }) => {
                    let Some((port_c, id_c)) =
                        port_and_id(from_operands.get(2).and_then(Option::as_ref), reuse, scope)
                    else {
                        out.refusal = Some(PtComputeRefusal::Unrepresentable(at.clone()));
                        break;
                    };
                    // `if ((neg_input0 || neg_input1) && !(neg_input0 && neg_input1))` — ⛔ EXACTLY
                    // ONE of the two, which is why two negations cancel back to `fused_mul_add`.
                    // ⭐ AND IT READS THE **RAW** OPERANDS, not the folded `from_operands`, so a
                    // negation that entry 320 absorbed still decides the mode.
                    let negated = |read: Val| {
                        defining_position(read, scope).is_some_and(|def| {
                            matches!(
                                op_at(&def, scope),
                                Some(DfirOp::VectorChain(vc::Op::Neg { .. }))
                            )
                        })
                    };
                    let mode = if negated(reads[0]) != negated(reads[1]) {
                        sentient::FmaMode::FusedNegMulSub
                    } else {
                        sentient::FmaMode::FusedMulAdd
                    };

                    PtLoweredCompute {
                        replaces: at.clone(),
                        insert_before,
                        mask_ops,
                        ops: vec![sen::Op::Sentient(sentient::Op::VectorMac {
                            mask: None,
                            xrf_write_ptr: xrf.map(|ptrs| ptrs.argument.write),
                            xrf_read_ptr: xrf.map(|ptrs| ptrs.argument.read),
                            results: mac_results,
                            op_a: pt_operand(port_a, op_a_precision, id_a, op_a_forwarding),
                            op_b: pt_operand(port_b, op_b_precision, id_b, op_b_forwarding),
                            op_c: pt_operand(port_c, op_c_precision, id_c, op_c_forwarding),
                            result: sentient::ResultPorts {
                                forwarding: result_forwarding,
                                precision: result_precision,
                                unroll_incr: false,
                            },
                            mode,
                            compute_precision,
                            fold_mode,
                            unroll_factor: sentient::UnrollFactor::X1,
                            xrf_read_incr: 0,
                            xrf_write_incr: 0,
                            data_transfer_only: false,
                            dbg_name: op_dbg_name,
                        })],
                        send_dsts,
                        // `:461-478` — *"Only multiply_and_accumulate ops control this masking."*
                        // ⛔ `DT_CHECK_MSG(isa<BlockArgument>(mask), "expecting a block arg mask")`
                        // is the [`MaskValue::LoopIterator`] arm: entry 089 answers a loop induction
                        // variable or a constant it minted, and nothing else.
                        mask: Some(match &mask_val {
                            Some(MaskValue::Constant { columns, .. }) => {
                                PtMacMask::Constant(*columns)
                            }
                            Some(MaskValue::LoopIterator(iterator)) => PtMacMask::Dynamic {
                                iterator: *iterator,
                                start: MaskedColumns(0),
                                increment: MaskIncrement::PerParentLoopIteration,
                            },
                            None => PtMacMask::Constant(MaskedColumns(0)),
                        }),
                        xrf: None,
                    }
                }

                // ── `dyn_cast<vectorchain::BinaryOp>(op)` (`:489-517`) ───────────────────────────
                DfirOp::VectorChain(vc::Op::Binary { binary_op, .. }) => {
                    // `DT_CHECK_MSG(mask_val.has_value(), "expecting valid mask value")` — ⛔ AN
                    // UNMASKED `vectorchain.binary` ABORTS HERE, unlike the PE/SFP twin, which
                    // substitutes a `sentient.constant 0` (entry 365).
                    let Some(mask) = mask_val.as_ref().map(|mask| match mask {
                        MaskValue::Constant { value, .. } => *value,
                        MaskValue::LoopIterator(iterator) => *iterator,
                    }) else {
                        out.refusal = Some(PtComputeRefusal::Unrepresentable(at.clone()));
                        break;
                    };

                    let sentient_binary = vector_binary_to_sentient_binary(*binary_op);
                    // *"Bitwise logical operation should have compute operand precision set to
                    // none."*
                    let compute_precision = if is_sentient_binary_logical_op(sentient_binary) {
                        sentient::Precision::None
                    } else {
                        compute_precision
                    };

                    PtLoweredCompute {
                        replaces: at.clone(),
                        insert_before,
                        mask_ops,
                        ops: vec![sen::Op::Sentient(sentient::Op::VectorBinary {
                            mask,
                            // ⛔ `ArrayAttr::get(context, {})` TWICE — the forwarding lists computed
                            // above are DISCARDED on this path.
                            op_a: pt_operand(port_a, op_a_precision, id_a, Vec::new()),
                            op_b: pt_operand(port_b, op_b_precision, id_b, Vec::new()),
                            // `nullptr` for `LogicalResultForwarding`, which the `DT_CHECK` above
                            // already established is absent.
                            binary_op: sentient::Binary::Plain(sentient_binary),
                            result: sentient::ResultPorts {
                                forwarding: result_forwarding,
                                precision: result_precision,
                                unroll_incr: false,
                            },
                            compute_precision,
                            fold_mode,
                            unroll_factor: sentient::UnrollFactor::X1,
                            dbg_name: op_dbg_name,
                        })],
                        send_dsts,
                        // A `vector_binary` carries its mask directly; it owes the tree nothing.
                        mask: None,
                        xrf: None,
                    }
                }

                // ⛔ THE TWELVE DEAD ARMS. `isa<>` at `:252-260` admits three classes and MLIR has
                // no op-class hierarchy, so `ShuffleOp`, `ElementWiseCompareOp`,
                // `ElementWiseSelectionOp`, `ScanWithGapOp`, `FastExpOp`, `ExpEstimateOp`,
                // `RecEstimateOp`, `LnEstimateOp`, `RsqrtEstimateOp`, `SigmoidEstimateOp`,
                // `TanhEstimateOp` and `PackOp` cannot reach `:519-869`. ⭐ AND NOTHING FALLS
                // THROUGH: an op that matched none of the three still runs the three erases below.
                _ => {
                    erase_operands_recording(&to_operands, scope, &mut erased_list);
                    erase_op_recording(at, scope, &mut erased_list);
                    erase_operands_recording(&from_operands, scope, &mut erased_list);
                    continue;
                }
            };

            // `if (count(vector_op) > 0) { setSentientMacXrfRegIncrAttr(&mac_op, &builder,
            //  to_operands[0].value().orig_precision_, ctx); mac_op_to_xrfptr_map[mac_op] = …; }`
            // (`:415-421`, `:480-486`) — ⭐ THE FIRST DESTINATION'S ORIGINAL PRECISION, not the
            // unit's. ⛔ The `binary` arm has no such block, and `mac_results` left it no results to
            // hang pointers on either.
            if let Some(ptrs) = xrf
                && let Some(mac) = lowered.ops.first_mut()
            {
                let Some(orig) = to_operands
                    .first()
                    .and_then(|to| to.as_ref())
                    .and_then(|to| to.orig_precision)
                else {
                    out.refusal = Some(PtComputeRefusal::Unrepresentable(at.clone()));
                    break;
                };
                if let sen::Op::Sentient(inner) = mac
                    && let Some(incr) = MacXrfIncrements::of(inner)
                {
                    set_sentient_mac_xrf_reg_incr_attr::<A>(incr, orig);
                }
                lowered.xrf = DummyMacPtrs::of(mac, *ptrs);
                out.dummy_macs.extend(lowered.xrf.clone());
            }

            // `VectorOperand::eraseOperands(to_operands); VectorOperand::eraseOp(op);
            //  VectorOperand::eraseOperands(from_operands);` (`:871-873`) — ⛔ THE OP GOES BETWEEN
            // the two operand erases, which is what lets the from-side see it already unused.
            erase_operands_recording(&to_operands, scope, &mut erased_list);
            erase_op_recording(at, scope, &mut erased_list);
            erase_operands_recording(&from_operands, scope, &mut erased_list);

            out.computes.push(lowered);
        }
    }

    erased_list.sort_unstable();
    erased_list.dedup();
    for position in erased_list.iter().rev() {
        remove_at(position.path(), &mut unit.body);
    }
    out
}

#[cfg(test)]
mod unit_tests {
    use super::{
        ComputeComp, MaskedColumns, OpId, PtMacMask, VectorOperand, XrfPtrMap,
        compute_unit_precision, fuse_compute_ops, fuse_non_compute_ops,
        lower_dangling_non_compute_ops, run_on_operation,
    };
    use crate::arch::Target;
    use crate::bridges::dataflow_ir_to_sentient::vc_lowering_xrf::{XrfPtrPair, XrfPtrs};
    use crate::bridges::dataflow_ir_to_sentient::vc_operand_reuse::OperandReuse;
    use crate::bridges::dataflow_ir_to_sentient::vc_vector_operands::{
        OperandValue, VectorOperandType,
    };
    use crate::islands::dataflow_ir::dialects::vectorchain as vc;
    use crate::islands::dataflow_ir::dialects::{Index, Op as DfirOp, Val, agen, dataflow};
    use crate::islands::dataflow_ir::link::{CrossPtnLink, L0lu, Link, Pe, PtRowUnit, Sfp};
    use crate::islands::dataflow_ir::ty::{AffineExpr, AffineMap, ElemType, MemRef, Vector};
    use crate::islands::dataflow_ir::{self as dfir, ProgramUnit, Units, Values};
    use crate::islands::sentient::dialects::{Definitions, Op as SenOp, sentient};
    use crate::units::{DfirUnit, Residency, Row};

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

    /// The vector every transfer in the fusion fixtures moves.
    const V: Vector = Vector {
        len: 64,
        elem: ElemType::F16,
    };

    /// `dataflow.get_unit` binding one transfer peer — ⛔ without it in the body, neither end of the
    /// transfer resolves to a port.
    fn unit_handle(result: Val, unit: DfirUnit) -> DfirOp {
        DfirOp::Dataflow(dataflow::Op::GetUnit {
            result,
            residency: Residency::Global,
            unit,
            num_folds: None,
        })
    }

    /// One operand slot at the unit's own `fp16`, with the reuse id the fusion assigned it.
    fn reads(port: sentient::Port, data_id: Option<i32>) -> sentient::Operand {
        sentient::Operand {
            precision: sentient::Precision::Fp16,
            data_id,
            ..sentient::Operand::from(port)
        }
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

    // ── 368/384 ───────────────────────────────────────────────────────────────────────────────

    /// 🎯 368/384 — THE CUMULATIVE SWAP, ONCE. One destination swaps operand A with operand C, so a
    /// `dataflow.send` fed by a `dataflow.receive` fuses into a MAC reading `zero`/`one`/the
    /// receive's own compass port — with the data id carried onto C by the same swap — and forwards
    /// its result to the send's port. ⛔ BOTH TRANSFERS ARE ERASED and the MAC carries NO mask.
    #[test]
    fn a_send_fed_by_a_receive_fuses_into_a_mac_whose_a_and_c_swapped_once() {
        let mut vals = Values::default();
        let (sfp, pe, data) = (Val(10), Val(11), Val(1));
        let from = Link::<Sfp, PtRowUnit<0>>::between(sfp, Val(0)).ends().1;
        let to = Link::<PtRowUnit<0>, Pe>::between(Val(0), pe).ends().0;
        let mut unit = pt_unit(vec![
            unit_handle(sfp, DfirUnit::Sfp),
            unit_handle(pe, DfirUnit::Pe),
            DfirOp::Dataflow(dataflow::Op::Receive {
                result: data,
                from,
                ty: V,
            }),
            DfirOp::Dataflow(dataflow::Op::Send { to, data, ty: V }),
        ]);

        let fusion = fuse_non_compute_ops(
            &mut unit,
            ComputeComp::Pt,
            &XrfPtrMap::new(),
            &mut OperandReuse::default(),
            &mut vals,
        );

        assert_eq!(fusion.refusal, None);
        assert_eq!(fusion.send_dsts, Vec::new());
        assert_eq!(fusion.dummy_macs, Vec::new());
        let [fused] = fusion.macs.as_slice() else {
            panic!("one fused MAC, not {:?}", fusion.macs)
        };
        assert_eq!(
            (&fused.at, fused.mask, &fused.xrf),
            (&OpId::at(&[3]), MaskedColumns(0), &None)
        );
        let SenOp::Sentient(sentient::Op::VectorMac {
            mask,
            op_a,
            op_b,
            op_c,
            result,
            mode,
            compute_precision,
            fold_mode,
            ..
        }) = &fused.ops[1]
        else {
            panic!("a mac, not {:?}", fused.ops[1])
        };
        assert_eq!(
            (op_a, op_b, op_c),
            (
                &reads(sentient::Port::Zero, None),
                &reads(sentient::Port::One, None),
                &reads(sentient::Port::North, Some(0))
            )
        );
        assert_eq!(
            (
                *mask,
                &result.forwarding,
                result.precision,
                *mode,
                *compute_precision,
                *fold_mode
            ),
            (
                None,
                &vec![sentient::Port::South],
                sentient::Precision::Fp16,
                sentient::FmaMode::FusedMulAdd,
                sentient::Precision::Fp16,
                None
            )
        );
        // ⛔ THE RECEIVE AND THE SEND BOTH GO; the two `get_unit`s are not this walk's business.
        assert_eq!(unit.body.len(), 2);
    }

    // ── 369/384 ───────────────────────────────────────────────────────────────────────────────

    /// 🎯 369/384 — A `vectorchain.multiply_and_accumulate` BECOMES ONE `sentient.vector_mac` READING
    /// ITS THREE PORTS IN OPERAND ORDER, each with its own reuse id, and it still owes the mask tree
    /// the literal `0` even though the op carries no mask operand.
    ///
    /// ⛔ NEITHER OPERAND IS A `vectorchain.neg`, so the mode is `fused_mul_add` — the neg test is
    /// exactly-one-of, not either-of.
    #[test]
    fn a_multiply_and_accumulate_becomes_a_three_port_mac_owing_a_zero_mask() {
        let mut vals = Values::default();
        let (l0lu, sfp, cross, pe) = (Val(10), Val(11), Val(12), Val(13));
        let (a, b, acc, product) = (Val(1), Val(2), Val(3), Val(4));
        fn recv<K: dfir::link::UnitKind>(peer: Val, result: Val) -> DfirOp {
            DfirOp::Dataflow(dataflow::Op::Receive {
                result,
                from: Link::<K, PtRowUnit<0>>::between(peer, Val(0)).ends().1,
                ty: V,
            })
        }
        let mut unit = pt_unit(vec![
            unit_handle(l0lu, DfirUnit::L0lu),
            unit_handle(sfp, DfirUnit::Sfp),
            unit_handle(cross, DfirUnit::CrossPtnLink),
            unit_handle(pe, DfirUnit::Pe),
            recv::<L0lu>(l0lu, a),
            recv::<Sfp>(sfp, b),
            recv::<CrossPtnLink>(cross, acc),
            DfirOp::VectorChain(vc::Op::MultiplyAccumulate {
                result: product,
                a,
                b,
                acc,
                reduction_map: AffineMap::unary(AffineExpr::Dim(0)),
                operand_ty: V,
                ty: V,
            }),
            DfirOp::Dataflow(dataflow::Op::Send {
                to: Link::<PtRowUnit<0>, Pe>::between(Val(0), pe).ends().0,
                data: product,
                ty: V,
            }),
        ]);

        let fusion = fuse_compute_ops(
            &mut unit,
            ComputeComp::Pt,
            &XrfPtrMap::new(),
            &mut OperandReuse::default(),
            Definitions::from_innermost(&[]),
            &[],
            &mut vals,
        );

        assert_eq!(fusion.refusal, None);
        assert_eq!(fusion.dummy_macs, Vec::new());
        let [lowered] = fusion.computes.as_slice() else {
            panic!("one lowered compute, not {:?}", fusion.computes)
        };
        assert_eq!(
            (&lowered.replaces, &lowered.insert_before, &lowered.xrf),
            (&OpId::at(&[7]), &OpId::at(&[7]), &None)
        );
        // ⭐ NO MASK OPERAND, SO NOTHING WAS BUILT — and the tree is still owed the literal `0`.
        assert_eq!(lowered.mask_ops, Vec::new());
        assert_eq!(lowered.send_dsts, Vec::new());
        assert_eq!(lowered.mask, Some(PtMacMask::Constant(MaskedColumns(0))));
        let SenOp::Sentient(sentient::Op::VectorMac {
            mask,
            results,
            op_a,
            op_b,
            op_c,
            result,
            mode,
            compute_precision,
            fold_mode,
            ..
        }) = &lowered.ops[0]
        else {
            panic!("a mac, not {:?}", lowered.ops[0])
        };
        assert_eq!(
            (op_a, op_b, op_c),
            (
                &reads(sentient::Port::West, Some(0)),
                &reads(sentient::Port::North, Some(1)),
                &reads(sentient::Port::CrossPtNorthLink, Some(2))
            )
        );
        assert_eq!(
            (
                *mask,
                results.as_slice(),
                &result.forwarding,
                result.precision,
                *mode,
                *compute_precision,
                *fold_mode
            ),
            (
                None,
                [].as_slice(),
                &vec![sentient::Port::South],
                sentient::Precision::Fp16,
                sentient::FmaMode::FusedMulAdd,
                sentient::Precision::Fp16,
                None
            )
        );
        // ⛔ THE THREE RECEIVES, THE COMPUTE AND THE SEND ALL GO; the four `get_unit`s stay.
        assert_eq!(unit.body.len(), 4);
    }
}
