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

use crate::islands::sentient::dialects::{Op, sentient};

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
}

impl<'a> SentientFor<'a> {
    /// The loop `op` is, or nothing when it is not a `sentient.for`.
    #[must_use]
    pub fn of(op: &'a Op) -> Option<SentientFor<'a>> {
        let Op::Sentient(sentient::Op::For { body, .. }) = op else {
            return None;
        };
        Some(SentientFor { body })
    }

    /// Its body, in block order.
    #[must_use]
    pub const fn body(self) -> &'a [Op] {
        self.body
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

// crustify:todo: e313_isCandidate
//   authority : dcc/src/Transform/Sentient/LoopEliminationViaRerolling.cpp:117  (52 body lines, level 1)
//   original  : bool isCandidate(Operation *for_op, SenComponents unit_comp) const
//   calls     : e073_getNontrivialOpsInLoop, e074_checkOperandsOfComputeOp, e252_size

// crustify:todo: e314_removeLoopsContainingLoadSendOrReceiveStore
//   authority : dcc/src/Transform/Sentient/LoopEliminationViaRerolling.cpp:175  (65 body lines, level 1)
//   original  : bool removeLoopsContainingLoadSendOrReceiveStore(sentient::ForOp sentient_for, dataflow::ProgramUnitOp unit, SenComponents unit_comp)
//   calls     : e073_getNontrivialOpsInLoop

// crustify:todo: e315_removeLoopsContainingComputeOp
//   authority : dcc/src/Transform/Sentient/LoopEliminationViaRerolling.cpp:248  (80 body lines, level 1)
//   original  : bool removeLoopsContainingComputeOp(sentient::ForOp sentient_for, SenComponents unit_comp)
//   calls     : e073_getNontrivialOpsInLoop, e243_roundDownUnrollFactor

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
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::Val;

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
}
