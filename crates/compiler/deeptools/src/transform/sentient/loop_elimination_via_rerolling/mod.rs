// SPDX-License-Identifier: Apache-2.0
//
// ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ CRUSTIFY SENTIENT-PASSES CAMPAIGN — READ THIS BEFORE YOU FILL AN ANCHOR IN THIS FILE.        ║
// ║ Campaign statement: crustify-senpass/TASK.md   ·   worklist: crustify-senpass/UNITS.tsv      ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
//
// 1. THE AUTHORITY IS THE C++ TREE, NOT THE EXTRACT.
//       /Users/nickm/git/deeptools-src/<file>:<line>     (deeptools|master|a0d29abbed — repo_info.txt)
//    Every citation below resolves against that revision. `crustify-senpass/cpp/sentient.cpp` says
//    WHICH functions are in scope and IN WHAT ORDER; its bodies were verified byte-identical to the
//    authority (656/656, 962,619 bytes, two negative controls), so either may be read — but the
//    authority file is the one that carries the surrounding declarations you will need.
//    ⛔ /Users/nickm/git/deeptools is a DIFFERENT revision. ⛔ The pod is not reachable from here.
//
// 2. THESE PASSES REWRITE SentientIR IN PLACE. They are NOT a conversion between rungs like bridges
//    1–4: input and output are both `src/islands/sentient/`. Expect to EXTEND that island — ops
//    gaining an assigned register, a pinned address, a rolled loop — not to emit into a new one.
//    WHY IT MATTERS: bridge 2's emission assigns NO registers (`index: None` in 39 of 41 sites),
//    faithfully, because the reference's SentientIR carries `regIndex = -1 : i32` in all 54
//    occurrences of the committed golden corpus. THESE passes turn -1 into a real register file and
//    index. Without them ProgIR gets -1 where an instruction needs a register and the backend
//    refuses with `Register initialization out of boundary` (observed on lxsu0:LRF0, l3lu:LBR2).
//
// 3. PORTED MEANS THE WHOLE FUNCTION INCLUDING ITS EFFECT. For an in-place pass the effect IS the
//    port: WHICH ops are rewritten, WHICH attributes are set to WHAT, and IN WHAT ORDER. A hand
//    attempt on bridge 2 extracted each function's decision rule into a documented predicate,
//    omitted the part that changed the IR, and reported it done — nothing called any of it.
//    Droppable: only the mechanism for REACHING operands (walking uses, memoising, positioning a
//    builder). ⛔ If the island cannot express a result, EXTEND THE ISLAND. Deciding a function is
//    unnecessary is NOT the porter's call, and a predicate is not a port.
//
// 4. HARD CAPS — MEASURED, AND THEY BIND. Bridge 2's first 144 functions cost 43 h wall clock and
//    586 M cache-read tokens; of 30,991 lines produced only 5,306 were implementation (12,333 doc
//    comments, 12,575 tests). Per ported function:
//      • 8 DOC LINES: the `/// Replaces: eNNN_name` anchor, one line of what it does, any TRAP.
//        No tutorials, no restating the C++, no design essays.
//      • ONE TEST. Two only where the vendor's own case AND a negative both apply.
//      • `cargo check -p deeptools` + `cargo test -p deeptools` ONCE PER BATCH, not per function.
//      • Do NOT re-verify citations — the review pass owns that.
//      • Do NOT grep the crate to discover types; the anchor names what you need.
//    NOT capped: correctness, and the emission.
//
// 5. THIS IS A PURE-LOGIC PORT WITH NO C ANYWHERE. Whatever generic C-to-Rust conventions say:
//    ❌ no bindgen/allowlist/-sys, ❌ no `ffi::`/`extern "C"`, ❌ no `CRUSTIFY_<FILE>` switch,
//    ❌ no `Foo`/`FooRef`/`FooMut` triple, ❌ no `unsafe`, ❌ no C-vs-Rust equivalence harness.
//
// 6. CRATE RULES BIND YOU — `crates/compiler/deeptools/CLAUDE.md`, read it in full.
//    🛑 NEVER RUNTIME REFUSE: no `Result`, no `Err(`, no `.ok_or`, no `assert!`, no `debug_assert!`.
//    A closed set is an `enum`; an invariant is a TYPE; newtypes, never raw scalars. No strings for
//    closed sets. `Arch`/`Model`/`Workload` flow through. `todo!` NAMING UNPORTED WORK IS ALLOWED
//    here — and ⛔ never substitute a stand-in op to dodge one.
//
// 7. ANCHORS: each `// crustify:todo: eNNN_name` below is one scheduled unit. Replace it with the
//    ported item carrying the doc anchor `/// Replaces: eNNN_name`. ⛔ NEVER DELETE AN ANCHOR YOU
//    DID NOT PORT — on bridge 2 a deleted anchor was indistinguishable from a finished unit and the
//    campaign reported DONE having silently lost 149 of 384 functions, the biggest in its span.
//
// 8. GATE: `cargo check -p deeptools` and `cargo test -p deeptools`. ⛔ NEVER run the workspace or
//    the acceptance build in an agent worktree — ~6 GB of target/ each, and this host is tight.
//
// 9. 43 OF THE 52 PASSES CONSUME AN ANALYSIS THAT IS OUT OF SCOPE (122 of the 656 units name one):
//    `Analyses/` (Liveness, PropagationAnalysis, GraphColoring, ExpressionEvaluatorUtils,
//    InstructionEstimation, TimeStamps, RegisterPressureAnalysis, AddressPinningScheme,
//    XRFRegisterAnalyzer, CorrelationAnalysis, RedundantDefinitionEliminationTree, …) and
//    `RegisterInitialization/` (Collector, Evaluator, Selector, Transformer, UniformGrouper) —
//    ~11,700 lines NOT in this campaign. Per pass: `crustify-senpass/OUTSIDE-DEPS.tsv`; per unit:
//    `OUTSIDE-UNITS.tsv`. ⭐ When a unit needs one, port the part that is present and
//    `todo!("<Analysis>::<method> — out of campaign scope")` for the part that is not, leaving the
//    anchor FILLED so the unit is not lost. ⛔ DO NOT INVENT THE ANALYSIS and do not substitute a
//    constant for its result. Extending `src/islands/sentient/` is a different case and IS expected.

//! `LoopEliminationViaRerolling.cpp` — 7 of the campaign's 656 units (dependency level(s) [0, 1, 6, 7]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e073_getNontrivialOpsInLoop` | 073 | 0 | 9 | `dcc/src/Transform/Sentient/LoopEliminationViaRerolling.cpp:59` |
//! | `e074_checkOperandsOfComputeOp` | 074 | 0 | 39 | `dcc/src/Transform/Sentient/LoopEliminationViaRerolling.cpp:74` |
//! | `e313_isCandidate` | 313 | 1 | 52 | `dcc/src/Transform/Sentient/LoopEliminationViaRerolling.cpp:117` |
//! | `e314_removeLoopsContainingLoadSendOrReceiveStore` | 314 | 1 | 65 | `dcc/src/Transform/Sentient/LoopEliminationViaRerolling.cpp:175` |
//! | `e315_removeLoopsContainingComputeOp` | 315 | 1 | 80 | `dcc/src/Transform/Sentient/LoopEliminationViaRerolling.cpp:248` |
//! | `e626_runOn` | 626 | 6 | 79 | `dcc/src/Transform/Sentient/LoopEliminationViaRerolling.cpp:330` |
//! | `e641_runOnOperation` | 641 | 7 | 8 | `dcc/src/Transform/Sentient/LoopEliminationViaRerolling.cpp:410` |

use std::num::NonZeroU32;

use crate::arch::Elements;
use crate::islands::dataflow_ir::Values;
use crate::islands::sentient::dialects::{Definitions, Op, Val, replace_all_uses_with, sentient};
use crate::transform::sentient::utils::{SenTarget, round_down_unroll_factor};
use crate::units::DfirUnit;

/// WHERE AN OP SITS IN A LOOP BODY — one entry of the reference's `std::vector<Operation *>`.
///
/// ⛔ A POSITION AND NOT A BORROW, because every caller of [`get_nontrivial_ops_in_loop`] REWRITES
/// what it finds: e314 sets `burst_size` on the op it names and clones it, so a `&Op` would be the
/// one thing that cannot be handed on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InBody(pub usize);

/// A `sentient.for`, AS ITS BODY — `DT_CHECK_MSG(sentient_for, "Expect a valid for op.")` as a type.
///
/// ⭐ ONE WITNESS DISCHARGES BOTH OF THIS FILE'S COPIES OF THAT CHECK (`:61` and `:119`): a loop that
/// is not a `sentient.for` is a value this type cannot hold rather than one it aborts on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SentientFor<'a> {
    /// `getBody(0)->getOperations()` — the loop's single region.
    body: &'a [Op],
    /// `getBound()` — the trip count, which e314 and e315 both require to be a constant.
    bound: Val,
    /// `getRegionIterArgs()` PAIRED WITH `getInits()`, which is how e314 rewires them (`:217-223`).
    carried: &'a [sentient::Carried],
}

impl<'a> SentientFor<'a> {
    /// The loop `op` is, or nothing when it is not a `sentient.for`.
    #[must_use]
    pub fn of(op: &'a Op) -> Option<SentientFor<'a>> {
        let Op::Sentient(sentient::Op::For {
            bound, carried, body, ..
        }) = op
        else {
            return None;
        };
        Some(SentientFor {
            body,
            bound: *bound,
            carried,
        })
    }

    /// Its body, in block order.
    #[must_use]
    pub const fn body(self) -> &'a [Op] {
        self.body
    }

    /// Its trip count.
    #[must_use]
    pub const fn bound(self) -> Val {
        self.bound
    }

    /// The values it carries — `getNumRegionIterArgs()` is this many.
    #[must_use]
    pub const fn carried(self) -> &'a [sentient::Carried] {
        self.carried
    }
}

/// Replaces: e073_getNontrivialOpsInLoop
///
/// Every op in the loop's body except the `sentient.yield` and the constants, in block order.
///
/// ⛔ TRAP: `sentient::ConstantOp` IS `sentient.scalar_constant` (`SentientOps.td:848`), NOT
/// `arith.constant`. An `arith.constant` left in the body is nontrivial here, and every caller tests
/// `size() != 1`, so widening this filter would make loops candidates that the reference refuses.
#[must_use]
pub fn get_nontrivial_ops_in_loop(sentient_for: SentientFor<'_>) -> Vec<InBody> {
    sentient_for
        .body()
        .iter()
        .enumerate()
        .filter(|(_, op)| {
            !matches!(
                op,
                Op::Sentient(sentient::Op::Yield { .. } | sentient::Op::ScalarConstant { .. })
            )
        })
        .map(|(at, _)| InBody(at))
        .collect()
}

/// THE OPERAND A UNIT'S RE-ROLLING FORBIDS — `StringRef invalid_operand = is_pt ? "xrf" : "nfwd"`
/// (`:147`), which is a closed set of two and so an enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidOperand {
    /// `"xrf"` — PT units.
    Xrf,
    /// `"nfwd"` — PE and SFP units.
    Nfwd,
}

impl InvalidOperand {
    /// `stringifySentientComputePort(port).contains(invalid_operand)`.
    ///
    /// ⛔ TRAP: A SUBSTRING TEST, NOT EQUALITY, and that is load-bearing — `nfwd` matches BOTH
    /// `nfwd0` and `nfwd2` (`SentientTypes.td:213-214`), which is the only reason the reference
    /// spells the needle without an index.
    #[must_use]
    pub fn names(self, port: sentient::Port) -> bool {
        port.spelling().contains(match self {
            InvalidOperand::Xrf => "xrf",
            InvalidOperand::Nfwd => "nfwd",
        })
    }
}

/// A COMPUTE OP AS THE FIELDS THIS PASS READS — `llvm_unreachable("expect a compute op")` (`:110`)
/// as a type.
///
/// ⭐ THE FOUR ARMS COLLAPSE TO ONE LIST because they differ only in HOW MANY operands they declare:
/// each checks `unrollIncrOp<X>` and the port of every operand it has, plus `unrollIncrResult`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeOp<'a> {
    /// `opA`, `opB`, `opC` — exactly the ones the op declares.
    operands: Vec<&'a sentient::Operand>,
    /// `getUnrollIncrResult()`.
    unroll_incr_result: bool,
}

impl<'a> ComputeOp<'a> {
    /// The compute `op` is, or nothing for anything the reference's `llvm_unreachable` would meet.
    #[must_use]
    pub fn of(op: &'a Op) -> Option<ComputeOp<'a>> {
        let (operands, result) = match op {
            Op::Sentient(
                sentient::Op::VectorMac {
                    op_a,
                    op_b,
                    op_c,
                    result,
                    ..
                }
                | sentient::Op::VectorTernary {
                    op_a,
                    op_b,
                    op_c,
                    result,
                    ..
                },
            ) => (vec![op_a, op_b, op_c], result),
            Op::Sentient(sentient::Op::VectorBinary {
                op_a, op_b, result, ..
            }) => (vec![op_a, op_b], result),
            Op::Sentient(sentient::Op::VectorUnary { op_a, result, .. }) => (vec![op_a], result),
            _ => return None,
        };
        Some(ComputeOp {
            operands,
            unroll_incr_result: result.unroll_incr,
        })
    }
}

/// Replaces: e074_checkOperandsOfComputeOp
///
/// The compute may be re-rolled only when neither it nor any operand advances per unrolled copy and
/// no operand reads the unit's forbidden port.
///
/// ⛔ TRAP: `unrollIncrLogicalResult` ON `sentient.vector_ternary` IS A DIFFERENT ATTRIBUTE AND IS
/// NOT READ HERE — all four arms call `getUnrollIncrResult()`, so a ternary forwarding its logical
/// result stays a candidate.
#[must_use]
pub fn check_operands_of_compute_op(
    compute_op: &ComputeOp<'_>,
    invalid_operand: InvalidOperand,
) -> bool {
    if compute_op.unroll_incr_result {
        return false;
    }
    !compute_op
        .operands
        .iter()
        .any(|operand| operand.unroll_incr || invalid_operand.names(operand.port))
}

/// WHERE A LOOP SITS IN THE UNIT'S BLOCK — e314 and e315 both rewrite the loop's body and then the
/// uses of its results, so a `&Op` is again the one handle that cannot be passed on ([`InBody`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InBlock(pub usize);

/// A MEMORY OP AS THE FIELDS THIS PASS READS — `LoadAndSendOp` and `ReceiveAndStoreOp`, which declare
/// every one of them at the same type, so the reference's two `dyn_cast` arms are one pattern here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MemoryOp {
    mutable_addr: Val,
    interleaved_group: Elements,
    burst_size: Elements,
    result: Val,
}

/// `dyn_cast<LoadAndSendOp>(op)` or `dyn_cast<ReceiveAndStoreOp>(op)`.
fn memory_op(op: &Op) -> Option<MemoryOp> {
    let Op::Sentient(
        sentient::Op::LoadAndSend {
            mutable_addr,
            interleaved_group,
            extent,
            result,
            ..
        }
        | sentient::Op::ReceiveAndStore {
            mutable_addr,
            interleaved_group,
            extent,
            result,
            ..
        },
    ) = op
    else {
        return None;
    };
    Some(MemoryOp {
        mutable_addr: *mutable_addr,
        interleaved_group: *interleaved_group,
        burst_size: extent.burst_size,
        result: *result,
    })
}

/// Replaces: e313_isCandidate
///
/// Whether the loop is exactly one re-rollable op: on a compute unit one clean compute and no carried
/// value, on a memory unit one un-interleaved memory op walking the single carried address.
///
/// ⛔ TRAP: THE MEMORY ARM'S CARRIED VALUE MUST BE THE OP'S OWN `mutable_addr` (`:156`, `:161`) — a
/// loop carrying some OTHER address is refused, because folding the trip into the burst is only sound
/// when the loop's whole per-iteration state is the address the burst itself advances.
#[must_use]
pub fn is_candidate(sentient_for: SentientFor<'_>, unit_comp: DfirUnit) -> bool {
    // `is_any_of(unit_comp, SFP, PE, PT)` (`:139-140`) and `unit_comp == PT` (`:141`).
    let is_compute_unit = matches!(
        unit_comp,
        DfirUnit::Sfp | DfirUnit::Pe | DfirUnit::PtRow(_)
    );
    let is_pt = matches!(unit_comp, DfirUnit::PtRow(_));

    let nontrivial_ops_in_body = get_nontrivial_ops_in_loop(sentient_for);
    let iter_args = sentient_for.carried().len();
    if (!is_compute_unit && iter_args != 1)
        || (is_compute_unit && iter_args != 0)
        || nontrivial_ops_in_body.len() != 1
    {
        return false;
    }
    let target_op = &sentient_for.body()[nontrivial_ops_in_body[0].0];
    if is_compute_unit {
        let invalid_operand = if is_pt {
            InvalidOperand::Xrf
        } else {
            InvalidOperand::Nfwd
        };
        return ComputeOp::of(target_op)
            .is_some_and(|compute_op| check_operands_of_compute_op(&compute_op, invalid_operand));
    }
    let Some(memory) = memory_op(target_op) else {
        return false;
    };
    memory.mutable_addr == sentient_for.carried()[0].arg && memory.interleaved_group == Elements(0)
}

/// WHETHER THE LOOP WAS REWRITTEN AWAY — the reference's `bool`, whose meaning is *"the
/// transformation was applied and \p sentient_for is to be removed"* (`:172-173`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rerolled {
    /// `true` — the body op now covers the whole trip; `runOn` erases the loop.
    LoopIsToBeRemoved,
    /// `false` — not profitable, and the loop stands.
    Untouched,
}

/// `memory_op->setAttr("burst_size", ..)`, the residual clone's `getMutableAddrMutable().assign(..)`
/// and `updateDbgName(prefix, op, ")")` — which writes ONLY when the op already has a name
/// (`dcc/src/Utils/Utils.cpp:493-496, 511-515`).
fn rewrite_memory_op(op: &mut Op, burst: Elements, mutable_addr: Option<Val>, prefix: &str) {
    let Op::Sentient(
        sentient::Op::LoadAndSend {
            mutable_addr: addr,
            extent,
            dbg_name,
            ..
        }
        | sentient::Op::ReceiveAndStore {
            mutable_addr: addr,
            extent,
            dbg_name,
            ..
        },
    ) = op
    else {
        return;
    };
    extent.burst_size = burst;
    if let Some(new_addr) = mutable_addr {
        *addr = new_addr;
    }
    if let Some(name) = dbg_name {
        *name = format!("{prefix}{name})");
    }
}

/// A CLONE BINDS FRESH RESULTS — `builder.clone(*op)` gives the copy its own values, and re-minting is
/// how that reads at this rung.
fn cloned_with_fresh_results(op: &Op, values: &mut Values) -> Op {
    let mut clone = op.clone();
    if let Op::Sentient(inner) = &mut clone {
        for result in sentient::results_mut(inner) {
            *result = values.mint();
        }
    }
    clone
}

/// Replaces: e314_removeLoopsContainingLoadSendOrReceiveStore
///
/// Folds a memory loop's trip count into its memory op's burst size, cloning a second op beside it for
/// the residual burst, and rewires the loop's carried address and its result.
///
/// ⛔ `max_burst` IS A PARAMETER: `getMaxBurstSize(unit)` (`:183`) reads the per-unit table on
/// `dcc_ext_ctx_`, which is in `Analyses/` and out of campaign scope; its
/// `DT_CHECK_MSG(max_burst != -1)` is [`Elements`] having no negative value.
///
/// ⛔ TRAP: `DT_CHECK_MSG(target == residual_burst)` (`:200-201`) IS DISCHARGED BY THE REFUSAL ABOVE IT
/// — `target <= max_burst * 2` leaves at most one more burst after the first bite, so it is a subtraction.
pub fn remove_loops_containing_load_send_or_receive_store(
    block: &mut [Op],
    at: InBlock,
    max_burst: Elements,
    values: &mut Values,
) -> Rerolled {
    // Everything the decision needs, read before anything is rewritten.
    let (memory_at, carried, memory_result, new_burst, residual_burst) = {
        let scope: &[Op] = block;
        let Some(sentient_for) = scope.get(at.0).and_then(SentientFor::of) else {
            return Rerolled::Untouched;
        };
        let nontrivial_ops_in_body = get_nontrivial_ops_in_loop(sentient_for);
        let Some(memory_at) = nontrivial_ops_in_body.first().copied() else {
            return Rerolled::Untouched;
        };
        // `isCandidate` (`:151-165`) already proved this op is one of the two.
        let Some(memory) = memory_op(&sentient_for.body()[memory_at.0]) else {
            return Rerolled::Untouched;
        };
        // `dyn_cast<sentient::ConstantOp>(getBound().getDefiningOp())` (`:189-193`), then
        // `loop_count == 0` (`:196`). A NEGATIVE bound is not a trip count and the same refusal covers
        // it — the burst it would become has no negative value.
        let defs = Definitions::from_innermost(core::slice::from_ref(&scope));
        let Some(Op::Sentient(sentient::Op::ScalarConstant { value, .. })) =
            defs.of(sentient_for.bound())
        else {
            return Rerolled::Untouched;
        };
        if *value <= 0 {
            return Rerolled::Untouched;
        }
        let loop_count = value.unsigned_abs();
        // `int target = (burst_size > 0) ? loop_count * burst_size : loop_count;` (`:195`).
        let target = if memory.burst_size == Elements(0) {
            loop_count
        } else {
            loop_count.saturating_mul(memory.burst_size.0)
        };
        if target > max_burst.0.saturating_mul(2) {
            return Rerolled::Untouched;
        }
        let new_burst = target.min(max_burst.0);
        (
            memory_at,
            sentient_for.carried().to_vec(),
            memory.result,
            Elements(new_burst),
            Elements(target - new_burst),
        )
    };

    let last_result = {
        let Some(Op::Sentient(sentient::Op::For { body, .. })) = block.get_mut(at.0) else {
            return Rerolled::Untouched;
        };
        // `iter_arg.replaceAllUsesWith(starting_val)` over every carried pair (`:217-223`).
        for pair in &carried {
            replace_all_uses_with(body, pair.arg, pair.init);
        }
        let mut last_result = memory_result;
        if residual_burst > Elements(0) {
            // The residual copy goes AFTER the memory op (`:214`, `:227`) and reads the address the
            // first one left behind (`:233-236`).
            let mut clone = cloned_with_fresh_results(&body[memory_at.0], values);
            rewrite_memory_op(&mut clone, residual_burst, Some(memory_result), "LEVR-resid(");
            if let Some(residual) = memory_op(&clone) {
                last_result = residual.result;
            }
            body.insert(memory_at.0 + 1, clone);
        }
        // ⭐ AFTER the clone, so the residual's name wraps the ORIGINAL name and not `LEVR(..)`.
        rewrite_memory_op(&mut body[memory_at.0], new_burst, None, "LEVR(");
        last_result
    };
    // `sentient_for.getResults()[0].replaceAllUsesWith(last_memory_op->getResult(0))` (`:238-239`).
    if let Some(first) = carried.first() {
        replace_all_uses_with(block, first.result, last_result);
    }
    Rerolled::LoopIsToBeRemoved
}

/// `getUnrollFactor()` on any of the four compute ops.
fn unroll_factor_of(op: &Op) -> Option<sentient::UnrollFactor> {
    let Op::Sentient(
        sentient::Op::VectorMac { unroll_factor, .. }
        | sentient::Op::VectorBinary { unroll_factor, .. }
        | sentient::Op::VectorUnary { unroll_factor, .. }
        | sentient::Op::VectorTernary { unroll_factor, .. },
    ) = op
    else {
        return None;
    };
    Some(*unroll_factor)
}

/// `compute_op->setAttr("unrollFactor", ..)` plus `updateDbgName(prefix, op, ")")`.
fn rewrite_compute_op(op: &mut Op, unroll: sentient::UnrollFactor, prefix: &str) {
    let Op::Sentient(
        sentient::Op::VectorMac {
            unroll_factor,
            dbg_name,
            ..
        }
        | sentient::Op::VectorBinary {
            unroll_factor,
            dbg_name,
            ..
        }
        | sentient::Op::VectorUnary {
            unroll_factor,
            dbg_name,
            ..
        }
        | sentient::Op::VectorTernary {
            unroll_factor,
            dbg_name,
            ..
        },
    ) = op
    else {
        return;
    };
    *unroll_factor = unroll;
    if let Some(name) = dbg_name {
        *name = format!("{prefix}{name})");
    }
}

/// Replaces: e315_removeLoopsContainingComputeOp
///
/// Folds a compute loop's trip count into its compute op's unroll factor, cloning a second op beside
/// it for the residual unroll.
///
/// ⛔ TRAP: TWO UNROLL FIELDS OR NOTHING. `target_unroll != residual_unroll_val` (`:288`) refuses
/// whenever one rounding-down plus one more does not cover `loop_count * unroll_factor` EXACTLY —
/// a third copy is not profitable, so an unrollable-but-not-in-two loop stands.
///
/// ⛔ `sen_target` IS A PARAMETER: it is `dccExtContext().dsc_global_->backend` (`:283`), and
/// [`round_down_unroll_factor`] answers differently for a reduction on the card.
pub fn remove_loops_containing_compute_op(
    block: &mut [Op],
    at: InBlock,
    sen_target: SenTarget,
    values: &mut Values,
) -> Rerolled {
    let (compute_at, new_unroll, residual_unroll) = {
        let scope: &[Op] = block;
        let Some(sentient_for) = scope.get(at.0).and_then(SentientFor::of) else {
            return Rerolled::Untouched;
        };
        let nontrivial_ops_in_body = get_nontrivial_ops_in_loop(sentient_for);
        let Some(compute_at) = nontrivial_ops_in_body.first().copied() else {
            return Rerolled::Untouched;
        };
        let compute_op = &sentient_for.body()[compute_at.0];
        // `DT_ERROR("no match operation")` (`:266`) — `isCandidate` (`:143-149`) already proved this
        // op is one of the four, so there is nothing left here to abort on.
        let Some(unroll) = unroll_factor_of(compute_op) else {
            return Rerolled::Untouched;
        };
        // `loop_count` STAYS 0 when the bound is not a constant (`:277-280`), and `loop_count == 0`
        // refuses (`:281`) — as does anything that is not the reference's `unsigned`.
        let defs = Definitions::from_innermost(core::slice::from_ref(&scope));
        let Some(Op::Sentient(sentient::Op::ScalarConstant { value, .. })) =
            defs.of(sentient_for.bound())
        else {
            return Rerolled::Untouched;
        };
        let Some(target_unroll) = u32::try_from(*value)
            .ok()
            .and_then(|count| NonZeroU32::new(count.saturating_mul(unroll.count())))
        else {
            return Rerolled::Untouched;
        };
        let new_unroll = round_down_unroll_factor(target_unroll, compute_op, sen_target);
        let remaining = target_unroll.get() - new_unroll.count();
        let residual_unroll = NonZeroU32::new(remaining)
            .map(|left| round_down_unroll_factor(left, compute_op, sen_target));
        if remaining != residual_unroll.map_or(0, sentient::UnrollFactor::count) {
            return Rerolled::Untouched;
        }
        (compute_at, new_unroll, residual_unroll)
    };

    let Some(Op::Sentient(sentient::Op::For { body, .. })) = block.get_mut(at.0) else {
        return Rerolled::Untouched;
    };
    if let Some(residual_unroll) = residual_unroll {
        let mut clone = cloned_with_fresh_results(&body[compute_at.0], values);
        rewrite_compute_op(&mut clone, residual_unroll, "LEVR-resid(");
        body.insert(compute_at.0 + 1, clone);
    }
    // ⭐ AFTER the clone, so the residual's name wraps the ORIGINAL name.
    rewrite_compute_op(&mut body[compute_at.0], new_unroll, "LEVR(");
    Rerolled::LoopIsToBeRemoved
}

// crustify:todo: e626_runOn
//   authority : dcc/src/Transform/Sentient/LoopEliminationViaRerolling.cpp:330  (79 body lines, level 6)
//   original  : void runOn(dataflow::ProgramUnitOp unit)
//   calls     : e313_isCandidate, e314_removeLoopsContainingLoadSendOrReceiveStore, e315_removeLoopsContainingComputeOp, e597_runLightWeightSimplifications

// crustify:todo: e641_runOnOperation
//   authority : dcc/src/Transform/Sentient/LoopEliminationViaRerolling.cpp:410  (8 body lines, level 7)
//   original  : void runOnOperation()
//   calls     : e626_runOn

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::formats::Bits;
    use crate::islands::dataflow_ir::link::SendEnd;
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::units::Row;

    /// `sentient.scalar_constant` — one of the two ops e073 filters out.
    fn scalar_constant(result: Val) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value: 0,
            result,
            reg_locale: sentient::RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    /// `sentient.nop` — an op that is neither a yield, a constant nor a compute.
    fn nop() -> Op {
        Op::Sentient(sentient::Op::Nop { dbg_name: None })
    }

    /// `sentient.for` over `body`.
    fn sentient_for(body: Vec<Op>) -> Op {
        Op::Sentient(sentient::Op::For {
            iv: Val(0),
            bound: Val(1),
            carried: Vec::new(),
            dbg_name: None,
            body,
        })
    }

    /// `sentient.vector_mac` reading `op_a` from `port`, with the given unroll increments.
    fn mac(port: sentient::Port, operand_incr: bool, result_incr: bool) -> Op {
        let mut op_a = sentient::Operand::from(port);
        op_a.unroll_incr = operand_incr;
        Op::Sentient(sentient::Op::VectorMac {
            mask: None,
            xrf_write_ptr: None,
            xrf_read_ptr: None,
            results: Vec::new(),
            op_a,
            op_b: sentient::Operand::from(sentient::Port::Lrf(sentient::LrfIndex::L0)),
            op_c: sentient::Operand::from(sentient::Port::Latch),
            result: sentient::ResultPorts {
                forwarding: Vec::new(),
                precision: sentient::Precision::Fp16,
                unroll_incr: result_incr,
            },
            mode: sentient::FmaMode::FusedMulAdd,
            compute_precision: sentient::Precision::Fp16,
            fold_mode: None,
            unroll_factor: sentient::UnrollFactor::X1,
            xrf_read_incr: 0,
            xrf_write_incr: 0,
            data_transfer_only: false,
            is_data_weight: None,
            dbg_name: None,
        })
    }

    /// `sentient.vector_unary` reading `op_a` from `port` — the one-operand arm.
    fn unary(port: sentient::Port) -> Op {
        Op::Sentient(sentient::Op::VectorUnary {
            mask: Val(2),
            op_a: sentient::Operand::from(port),
            unary_op: sentient::UnaryOp::Rec,
            result: sentient::ResultPorts::default(),
            compute_precision: sentient::Precision::Fp16,
            fold_mode: None,
            unroll_factor: sentient::UnrollFactor::X1,
            dbg_name: None,
        })
    }

    /// The shape every caller tests for — one real op beside the constants and the yield — and the
    /// yield/constant filter, which is the whole of e073.
    #[test]
    fn e073_keeps_only_the_ops_that_are_neither_a_yield_nor_a_scalar_constant() {
        let loop_op = sentient_for(vec![
            scalar_constant(Val(3)),
            nop(),
            scalar_constant(Val(4)),
            Op::Sentient(sentient::Op::Yield {
                results: Vec::new(),
            }),
        ]);
        let sentient_for = SentientFor::of(&loop_op).expect("a sentient.for");
        assert_eq!(get_nontrivial_ops_in_loop(sentient_for), vec![InBody(1)]);
        // The `DT_CHECK` this witness replaces: anything that is not a loop has no body to walk.
        assert!(SentientFor::of(&nop()).is_none());
    }

    /// The three refusals — an operand's increment, the result's, and the unit's forbidden port —
    /// against a compute that is re-rollable on both units.
    #[test]
    fn e074_refuses_an_unroll_increment_or_the_units_own_forbidden_port() {
        let clean = mac(sentient::Port::Lrf(sentient::LrfIndex::L1), false, false);
        let clean = ComputeOp::of(&clean).expect("a compute op");
        assert!(check_operands_of_compute_op(&clean, InvalidOperand::Xrf));
        assert!(check_operands_of_compute_op(&clean, InvalidOperand::Nfwd));

        let operand_incr = mac(sentient::Port::Lrf(sentient::LrfIndex::L1), true, false);
        let operand_incr = ComputeOp::of(&operand_incr).expect("a compute op");
        assert!(!check_operands_of_compute_op(
            &operand_incr,
            InvalidOperand::Xrf
        ));

        let result_incr = mac(sentient::Port::Lrf(sentient::LrfIndex::L1), false, true);
        let result_incr = ComputeOp::of(&result_incr).expect("a compute op");
        assert!(!check_operands_of_compute_op(
            &result_incr,
            InvalidOperand::Xrf
        ));

        // `xrf` is forbidden on PT and allowed elsewhere; `nfwd2` is matched by the un-indexed
        // needle, which is the substring test the reference relies on.
        let xrf = mac(sentient::Port::Xrf, false, false);
        let xrf = ComputeOp::of(&xrf).expect("a compute op");
        assert!(!check_operands_of_compute_op(&xrf, InvalidOperand::Xrf));
        assert!(check_operands_of_compute_op(&xrf, InvalidOperand::Nfwd));
        let nfwd = unary(sentient::Port::Nfwd2);
        let nfwd = ComputeOp::of(&nfwd).expect("a compute op");
        assert!(!check_operands_of_compute_op(&nfwd, InvalidOperand::Nfwd));
        assert!(check_operands_of_compute_op(&nfwd, InvalidOperand::Xrf));

        // The `llvm_unreachable` this witness replaces.
        assert!(ComputeOp::of(&nop()).is_none());
    }

    /// `sentient.for` over `body`, carrying `carried` and bounded by `bound`.
    fn sentient_for_carrying(bound: Val, carried: Vec<sentient::Carried>, body: Vec<Op>) -> Op {
        Op::Sentient(sentient::Op::For {
            iv: Val(0),
            bound,
            carried,
            dbg_name: None,
            body,
        })
    }

    /// One carried address, unassigned.
    fn carried(init: Val, arg: Val, result: Val) -> sentient::Carried {
        sentient::Carried {
            init,
            arg,
            result,
            reg: sentient::Reg {
                locale: sentient::RegType::Lar,
                index: None,
            },
            program_header: false,
            element_size: None,
        }
    }

    /// `sentient.load_and_send` walking `mutable_addr`, with the `.td`'s own extent defaults.
    fn load_and_send(mutable_addr: Val, result: Val, burst_size: Elements, group: Elements) -> Op {
        Op::Sentient(sentient::Op::LoadAndSend {
            mutable_addr,
            immutable_addr: Val(6),
            increment: Val(7),
            consumer: SendEnd::to_self(Val(8)),
            result,
            extent: sentient::Extent {
                burst_size,
                ..sentient::Extent::of(Elements(64), Bits(32))
            },
            interleaved_group: group,
            rotate_val: None,
            dir: None,
            shuffle_mode: sentient::ShuffleMode::NoShuffle,
            reg: sentient::Reg {
                locale: sentient::RegType::Lar,
                index: None,
            },
            dbg_name: Some("LS".to_owned()),
        })
    }

    /// `sentient.yield` returning `results`.
    fn yield_op(results: Vec<Val>) -> Op {
        Op::Sentient(sentient::Op::Yield { results })
    }

    /// Both arms of the candidate test, and the two refusals that are unit-specific: `xrf` on PT, and
    /// an interleaved memory op — plus the iter-arg counts, which are opposite on the two unit kinds.
    #[test]
    fn e313_admits_one_clean_op_per_unit_kind_and_nothing_else() {
        let clean = sentient_for(vec![
            mac(sentient::Port::Lrf(sentient::LrfIndex::L1), false, false),
            yield_op(Vec::new()),
        ]);
        let clean = SentientFor::of(&clean).expect("a sentient.for");
        assert!(is_candidate(clean, DfirUnit::Pe));
        assert!(is_candidate(clean, DfirUnit::PtRow(Row::checked(0).expect("PT row 0 exists"))));
        // ⛔ A COMPUTE LOOP ON A MEMORY UNIT NEEDS ONE ITER ARG AND HAS NONE.
        assert!(!is_candidate(clean, DfirUnit::Lxlu));

        let xrf = sentient_for(vec![mac(sentient::Port::Xrf, false, false), yield_op(Vec::new())]);
        let xrf = SentientFor::of(&xrf).expect("a sentient.for");
        assert!(is_candidate(xrf, DfirUnit::Pe));
        assert!(!is_candidate(xrf, DfirUnit::PtRow(Row::checked(0).expect("PT row 0 exists"))));

        let memory = sentient_for_carrying(
            Val(1),
            vec![carried(Val(10), Val(11), Val(12))],
            vec![
                load_and_send(Val(11), Val(13), Elements(4), Elements(0)),
                yield_op(vec![Val(13)]),
            ],
        );
        let memory_for = SentientFor::of(&memory).expect("a sentient.for");
        assert!(is_candidate(memory_for, DfirUnit::Lxlu));
        assert!(!is_candidate(memory_for, DfirUnit::Pe));

        let interleaved = sentient_for_carrying(
            Val(1),
            vec![carried(Val(10), Val(11), Val(12))],
            vec![
                load_and_send(Val(11), Val(13), Elements(4), Elements(2)),
                yield_op(vec![Val(13)]),
            ],
        );
        let interleaved = SentientFor::of(&interleaved).expect("a sentient.for");
        assert!(!is_candidate(interleaved, DfirUnit::Lxlu));

        // The carried address must be the op's own.
        let other_addr = sentient_for_carrying(
            Val(1),
            vec![carried(Val(10), Val(11), Val(12))],
            vec![
                load_and_send(Val(99), Val(13), Elements(4), Elements(0)),
                yield_op(vec![Val(13)]),
            ],
        );
        let other_addr = SentientFor::of(&other_addr).expect("a sentient.for");
        assert!(!is_candidate(other_addr, DfirUnit::Lxlu));
    }

    /// Three iterations of a 4-element burst against a max of 8: the op takes 8, a residual clone
    /// takes the remaining 4 from the address the first left, and the loop's result is rewired to the
    /// clone. A trip that needs three bursts is refused instead.
    #[test]
    fn e314_folds_the_trip_into_one_burst_and_a_residual_clone() {
        let memory_loop = || {
            vec![
                scalar_constant_of(Val(1), 3),
                sentient_for_carrying(
                    Val(1),
                    vec![carried(Val(10), Val(11), Val(12))],
                    vec![
                        load_and_send(Val(11), Val(13), Elements(4), Elements(0)),
                        yield_op(vec![Val(13)]),
                    ],
                ),
                load_and_send(Val(12), Val(14), Elements(0), Elements(0)),
            ]
        };
        let mut block = memory_loop();
        let mut values = Values::default();
        for _ in 0..20 {
            let _ = values.mint();
        }
        assert_eq!(
            remove_loops_containing_load_send_or_receive_store(
                &mut block,
                InBlock(1),
                Elements(8),
                &mut values,
            ),
            Rerolled::LoopIsToBeRemoved
        );
        let sentient_for = SentientFor::of(&block[1]).expect("a sentient.for");
        let first = memory_op(&sentient_for.body()[0]).expect("the original memory op");
        let residual = memory_op(&sentient_for.body()[1]).expect("the residual clone");
        assert_eq!(first.burst_size, Elements(8));
        assert_eq!(residual.burst_size, Elements(4));
        // The iter arg became the loop's initial value, and the clone walks on from the original.
        assert_eq!(first.mutable_addr, Val(10));
        assert_eq!(residual.mutable_addr, Val(13));
        assert_eq!(residual.result, Val(20));
        // `LEVR(` wraps the original name; the residual wraps it BEFORE that rename.
        assert_eq!(
            sentient_for.body().iter().filter_map(dbg_name_of).collect::<Vec<_>>(),
            ["LEVR(LS)", "LEVR-resid(LS)"]
        );
        // The loop's result is now the residual clone's.
        assert_eq!(
            memory_op(&block[2]).expect("the reader").mutable_addr,
            Val(20)
        );

        // `target > max_burst * 2` — three bursts of 4 need three ops, which is not profitable.
        let mut refused = memory_loop();
        assert_eq!(
            remove_loops_containing_load_send_or_receive_store(
                &mut refused,
                InBlock(1),
                Elements(3),
                &mut values,
            ),
            Rerolled::Untouched
        );
    }

    /// A trip of 3 over an `x2` compute is a target of 6: `x4` on the op and an `x2` clone beside it.
    /// A target of 3 is refused, because `x2` plus `x1` does not cover it in two fields.
    #[test]
    fn e315_folds_the_trip_into_two_unroll_fields_or_declines() {
        let unrolled = |unroll| {
            let mut op = mac(sentient::Port::Lrf(sentient::LrfIndex::L1), false, false);
            rewrite_compute_op(&mut op, unroll, "");
            op
        };
        let mut block = vec![
            scalar_constant_of(Val(1), 3),
            sentient_for_carrying(
                Val(1),
                Vec::new(),
                vec![unrolled(sentient::UnrollFactor::X2), yield_op(Vec::new())],
            ),
        ];
        let mut values = Values::default();
        for _ in 0..20 {
            let _ = values.mint();
        }
        assert_eq!(
            remove_loops_containing_compute_op(
                &mut block,
                InBlock(1),
                SenTarget::Sentient,
                &mut values,
            ),
            Rerolled::LoopIsToBeRemoved
        );
        let sentient_for = SentientFor::of(&block[1]).expect("a sentient.for");
        assert_eq!(
            sentient_for
                .body()
                .iter()
                .filter_map(unroll_factor_of)
                .collect::<Vec<_>>(),
            [sentient::UnrollFactor::X4, sentient::UnrollFactor::X2]
        );

        // `target_unroll != residual_unroll_val`: 3 rounds down to 2, and 1 is not the missing 1... it
        // is — so take a target of 7 instead, which is 4 + 2 and leaves 1 uncovered.
        let mut refused = vec![
            scalar_constant_of(Val(1), 7),
            sentient_for_carrying(
                Val(1),
                Vec::new(),
                vec![unrolled(sentient::UnrollFactor::X1), yield_op(Vec::new())],
            ),
        ];
        assert_eq!(
            remove_loops_containing_compute_op(
                &mut refused,
                InBlock(1),
                SenTarget::Sentient,
                &mut values,
            ),
            Rerolled::Untouched
        );
    }

    /// `sentient.scalar_constant` binding `result` to `value`.
    fn scalar_constant_of(result: Val, value: i64) -> Op {
        Op::Sentient(sentient::Op::ScalarConstant {
            value,
            result,
            reg_locale: sentient::RegType::Imm,
            ty: ScalarTy::Index,
            is_symbol: false,
        })
    }

    /// The `dbgName` of a memory op, for the two `updateDbgName` calls.
    fn dbg_name_of(op: &Op) -> Option<&str> {
        let Op::Sentient(
            sentient::Op::LoadAndSend { dbg_name, .. }
            | sentient::Op::ReceiveAndStore { dbg_name, .. },
        ) = op
        else {
            return None;
        };
        dbg_name.as_deref()
    }
}
