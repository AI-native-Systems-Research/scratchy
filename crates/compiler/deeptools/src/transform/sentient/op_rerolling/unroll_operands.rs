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

//! `OpRerolling.cpp` — 12 of the campaign's 656 units (dependency level(s) [0, 1]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e113_isXrfRdOp` | 113 | 0 | 11 | `dcc/src/Transform/Sentient/OpRerolling.cpp:58` |
//! | `e114_isXrfWtOp` | 114 | 0 | 9 | `dcc/src/Transform/Sentient/OpRerolling.cpp:69` |
//! | `e115_reset` | 115 | 0 | 31 | `dcc/src/Transform/Sentient/OpRerolling.cpp:439` |
//! | `e116_getUnrollOperandUsingPort` | 116 | 0 | 19 | `dcc/src/Transform/Sentient/OpRerolling.cpp:471` |
//! | `e117_fillMemOpAttrs` | 117 | 0 | 9 | `dcc/src/Transform/Sentient/OpRerolling.cpp:492` |
//! | `e118_areOperandsMatching` | 118 | 0 | 59 | `dcc/src/Transform/Sentient/OpRerolling.cpp:668` |
//! | `e119_createOperand` | 119 | 0 | 11 | `dcc/src/Transform/Sentient/OpRerolling.cpp:926` |
//! | `e120_createForwardingArray` | 120 | 0 | 17 | `dcc/src/Transform/Sentient/OpRerolling.cpp:939` |
//! | `e121_dump` | 121 | 0 | 27 | `dcc/src/Transform/Sentient/OpRerolling.cpp:966` |
//! | `e335_fill` | 335 | 1 | 165 | `dcc/src/Transform/Sentient/OpRerolling.cpp:502` |
//! | `e336_match` | 336 | 1 | 149 | `dcc/src/Transform/Sentient/OpRerolling.cpp:731` |
//! | `e337_updateUnrollInfo` | 337 | 1 | 44 | `dcc/src/Transform/Sentient/OpRerolling.cpp:881` |

// ⛔ NOTHING IN THE CRATE CALLS THIS FILE UNTIL `e607_runOnOperation` (level 5) LANDS, and CI runs
// clippy with `-D warnings`. ⭐ REMOVE THIS WITH e607.
#![allow(dead_code)]

use crate::islands::sentient::dialects::sentient as sen;


// crustify:todo: e113_isXrfRdOp
//   authority : dcc/src/Transform/Sentient/OpRerolling.cpp:58  (11 body lines, level 0)
//   original  : bool UnrollOperands::isXrfRdOp()

// crustify:todo: e114_isXrfWtOp
//   authority : dcc/src/Transform/Sentient/OpRerolling.cpp:69  (9 body lines, level 0)
//   original  : bool UnrollOperands::isXrfWtOp()

// crustify:todo: e115_reset
//   authority : dcc/src/Transform/Sentient/OpRerolling.cpp:439  (31 body lines, level 0)
//   original  : void UnrollOperands::reset()

// crustify:todo: e116_getUnrollOperandUsingPort
//   authority : dcc/src/Transform/Sentient/OpRerolling.cpp:471  (19 body lines, level 0)
//   original  : OperandName UnrollOperands::getUnrollOperandUsingPort( SenComponents type, SentientComputePort port_id)

// crustify:todo: e117_fillMemOpAttrs
//   authority : dcc/src/Transform/Sentient/OpRerolling.cpp:492  (9 body lines, level 0)
//   original  : void UnrollOperands::fillMemOpAttrs(Operation *op)

// crustify:todo: e118_areOperandsMatching
//   authority : dcc/src/Transform/Sentient/OpRerolling.cpp:668  (59 body lines, level 0)
//   original  : bool UnrollOperands::areOperandsMatching( const StringRef &ref_operand, const bool &is_ref_unrolled, const StringRef &curr_operand, const bool &is_curr_unrolled, const UnrollOperands &curr_operand_list, const bool is_xrf_read)

// crustify:todo: e119_createOperand
//   authority : dcc/src/Transform/Sentient/OpRerolling.cpp:926  (11 body lines, level 0)
//   original  : StringAttr UnrollOperands::createOperand(OperandName op_name, unsigned new_unroll_size)

// crustify:todo: e120_createForwardingArray
//   authority : dcc/src/Transform/Sentient/OpRerolling.cpp:939  (17 body lines, level 0)
//   original  : SmallVector<Attribute, 4> UnrollOperands::createForwardingArray( OperandName op_name, unsigned new_unroll_size)

/// WHICH OPERAND SLOT OF A COMPUTE OP — `enum OperandName` (`OpRerolling.hpp:30-36`), the key of
/// every one of [`UnrollOperands`]'s three maps.
///
/// ⛔ NOT THE IR'S `opA`/`opB`/`opC` ([`sen::Port::OpA`]), which name a forwarding TARGET. These are
/// containers: `e335_fill` puts an op's operand under the slot its PORT ID selects (`:523-531`), so
/// `op_a` here is "whatever arrives on port0" and not the op's own `opA` attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum OperandName {
    /// `op_a = 0`.
    OpA,
    /// `op_b = 1`.
    OpB,
    /// `op_c = 2`.
    OpC,
    /// `result = 3`.
    Result,
    /// `logical_result = 4`.
    LogicalResult,
}

impl OperandName {
    /// ⭐ THE ITERATION ORDER OF EVERY `std::map` KEYED BY THIS ENUM, and so the order e121 prints:
    /// `std::map` walks sorted by key, and the key's order is its `= 0..4` (`OpRerolling.hpp:30-36`).
    pub(crate) const ALL: [Self; 5] = [
        Self::OpA,
        Self::OpB,
        Self::OpC,
        Self::Result,
        Self::LogicalResult,
    ];

    /// Which entry of [`UnrollOperands`]'s per-slot arrays this name is.
    #[must_use]
    pub(crate) const fn slot(self) -> usize {
        self as usize
    }

    /// What e121's `toString` lambda gives (`:967-982`) — the C++ ENUMERATOR spellings, `op_a` and
    /// not the IR's `opA`.
    #[must_use]
    pub(crate) const fn spelling(self) -> &'static str {
        match self {
            Self::OpA => "op_a",
            Self::OpB => "op_b",
            Self::OpC => "op_c",
            Self::Result => "result",
            Self::LogicalResult => "logical_result",
        }
    }
}

/// ONE COMPUTE OR MEMORY OP'S UNROLL-RELEVANT FIELDS — `class UnrollOperands`
/// (`OpRerolling.hpp:43-164`), filled from one op by `e335_fill` and compared against another by
/// `e336_match`.
///
/// ⭐ AN ARRAY PER SLOT, NOT A MAP: the reference's `std::map<OperandName, …>` always holds exactly
/// the five keys — the constructor inserts all five (`OpRerolling.hpp:85-100`) and `e115_reset` only
/// overwrites them (`:440-451`) — so "every slot has an entry" is the array's length here.
///
/// ⚠️ ONLY THE TWO MEMBERS e121 DUMPS ARE PRESENT. `forwarding_list_`, `precisions_`, `unroll_size_`,
/// the xrf increments and the memory-unit members arrive with the units that read them.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct UnrollOperands {
    /// `operand_list_` (`OpRerolling.hpp:57`) — where each slot's operand comes from.
    ///
    /// ⛔ `None` IS THE REFERENCE'S `"NA"` (`:442`, `OpRerolling.hpp:86-90`) AND NOT A PORT: every
    /// value it ever holds otherwise is a `stringifySentientComputePort` result (`:529`, `:571`), so
    /// the sentinel cannot be a [`sen::Port`] case. `Default` is therefore the constructor's state.
    pub(crate) operand_list: [Option<sen::Port>; OperandName::ALL.len()],
    /// `is_field_unroll_` (`OpRerolling.hpp:59`) — whether that slot's register index advances with
    /// the unroll factor.
    pub(crate) is_field_unroll: [bool; OperandName::ALL.len()],
}

impl UnrollOperands {
    /// Replaces: e121_dump
    ///
    /// The debugger's view of one `UnrollOperands`: a header, then all five operands, then all five
    /// unroll flags, each block in [`OperandName::ALL`] order.
    ///
    /// ⭐ A RETURNED `String` FOR `llvm::outs()`, as e014_dump and e065_dump did. Nothing in the
    /// reference tree calls `dump()` at all — it is reached from a debugger — so no output the
    /// reference produced is lost by not writing it here.
    ///
    /// ⭐ `1`/`0` FOR THE FLAG, NOT `true`/`false`: `raw_ostream` has no `bool` overload, so `bool`
    /// promotes to `operator<<(int)` (`raw_ostream.h:284`) and the reference prints the digit.
    ///
    /// ⛔ THE LAMBDA'S `default: "unknown operand name"` (`:979-980`) IS UNREACHABLE AND IS NOT PORTED:
    /// the switch already covers all five cases the enum declares, and the arms come from the keys
    /// of maps that only ever hold those five.
    #[must_use]
    pub(crate) fn dump(&self) -> String {
        let mut out = String::from("dumping:\n");
        for name in OperandName::ALL {
            let operand =
                self.operand_list[name.slot()].map_or_else(|| "NA".to_owned(), sen::Port::spelling);
            out.push_str(&format!("{}: {operand}\n", name.spelling()));
        }
        for name in OperandName::ALL {
            let flag = u8::from(self.is_field_unroll[name.slot()]);
            out.push_str(&format!("{}: {flag}\n", name.spelling()));
        }
        out
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    /// Both blocks in key order over a mixed state: a port-backed slot, the `"NA"` sentinel for the
    /// untouched ones, and the flag as a digit.
    #[test]
    fn dump_prints_five_operands_then_five_flags_in_key_order() {
        let mut ops = UnrollOperands::default();
        ops.operand_list[OperandName::OpA.slot()] = Some(sen::Port::Lrf(sen::LrfIndex::L3));
        ops.operand_list[OperandName::Result.slot()] =
            Some(sen::Port::IState(sen::IStateIndex::S2));
        ops.is_field_unroll[OperandName::Result.slot()] = true;

        assert_eq!(
            ops.dump(),
            concat!(
                "dumping:\n",
                "op_a: lrf3\n",
                "op_b: NA\n",
                "op_c: NA\n",
                "result: istate2\n",
                "logical_result: NA\n",
                "op_a: 0\n",
                "op_b: 0\n",
                "op_c: 0\n",
                "result: 1\n",
                "logical_result: 0\n",
            )
        );
    }
}

// crustify:todo: e335_fill
//   authority : dcc/src/Transform/Sentient/OpRerolling.cpp:502  (165 body lines, level 1)
//   original  : void UnrollOperands::fill(Operation *op, SenComponents type)
//   calls     : e115_reset, e116_getUnrollOperandUsingPort, e117_fillMemOpAttrs

// crustify:todo: e336_match
//   authority : dcc/src/Transform/Sentient/OpRerolling.cpp:731  (149 body lines, level 1)
//   original  : bool UnrollOperands::match(UnrollOperands &new_operand_list)
//   calls     : e118_areOperandsMatching, e252_size

// crustify:todo: e337_updateUnrollInfo
//   authority : dcc/src/Transform/Sentient/OpRerolling.cpp:881  (44 body lines, level 1)
//   original  : void UnrollOperands::updateUnrollInfo(UnrollOperands &new_operand_list)
//   calls     : e252_size

