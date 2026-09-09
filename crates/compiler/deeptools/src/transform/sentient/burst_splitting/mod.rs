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

//! `BurstSplitting.cpp` — 4 of the campaign's 656 units (dependency level(s) [0, 1, 2, 3]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e021_splitBurst` | 021 | 0 | 47 | `dcc/src/Transform/Sentient/BurstSplitting.cpp:112` |
//! | `e285_runOn` | 285 | 1 | 29 | `dcc/src/Transform/Sentient/BurstSplitting.cpp:161` |
//! | `e430_runOn` | 430 | 2 | 21 | `dcc/src/Transform/Sentient/BurstSplitting.cpp:191` |
//! | `e495_runOnOperation` | 495 | 3 | 5 | `dcc/src/Transform/Sentient/BurstSplitting.cpp:213` |

use crate::arch::{Arch, Elements};
use crate::bridges::dataflow_ir_to_sentient::vc_vector_operands::OpId;
use crate::islands::sentient::ProgramUnit;
use crate::islands::sentient::dialects::{Op, sentient};
use crate::units::DfirUnit;

/// A MEMORY OP THIS PASS MAY SPLIT, WITH THE BURST IT CARRIES.
///
/// ⭐ THE `DT_CHECK` AT `BurstSplitting.cpp:114-117` AS A TYPE. Only `sentient.load_and_send`,
/// `sentient.receive_and_store` and `sentient.load_and_store` carry a burst this pass understands,
/// and "Unsupported operation for burst splitting" is then a state [`split_burst`] cannot be reached
/// in rather than one it aborts on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BurstCandidate {
    /// Where the op is, so the split can be inserted after it.
    at: OpId,
    /// `op->getAttr("burst_size")`.
    burst_size: Elements,
}

impl BurstCandidate {
    /// The candidate at `at`, or nothing when that op has no splittable burst.
    #[must_use]
    pub fn of(at: OpId, op: &Op) -> Option<BurstCandidate> {
        let Op::Sentient(
            sentient::Op::LoadAndSend { extent, .. }
            | sentient::Op::ReceiveAndStore { extent, .. }
            | sentient::Op::LoadAndStore { extent, .. },
        ) = op
        else {
            return None;
        };
        Some(BurstCandidate {
            at,
            burst_size: extent.burst_size,
        })
    }

    /// Where it is.
    #[must_use]
    pub fn at(&self) -> &OpId {
        &self.at
    }

    /// The burst it carries.
    #[must_use]
    pub fn burst_size(&self) -> Elements {
        self.burst_size
    }
}

/// Replaces: e021_splitBurst
///
/// Costs the split in IBuff — one entry when more than one full burst fits and so a `sentient.for` is
/// needed, one more for a residual op — then hands the rewrite itself to `burst_utils`.
/// ⛔ `getMaxBurstSize` IS A PARAMETER, NOT A LOOKUP: it reads the per-unit table on
/// `dcc_ext_ctx_`, which is outside this campaign, and `DT_CHECK_MSG(max_burst != -1)` (`:130`)
/// becomes [`Elements`] having no negative value.
/// ⭐ THE TWO IBUFF WARNINGS COST THIS PORT NOTHING: `haveIbuffSpace` and `getRemainingIbuffSpace`
/// only feed `LLVM_DEBUG` (`:123-127`, `:147-151`) and change no IR, so `InstructionEstimator` being
/// out of scope removes no effect.
pub fn split_burst<A: Arch>(
    unit: &mut ProgramUnit<A>,
    candidate: &BurstCandidate,
    comp: DfirUnit,
    max_burst: Elements,
) -> ! {
    // `DT_CHECK_MSG(burst_size > max_burst)` — and `Elements(0)`, this island's "unbursted", is where
    // the reference's `-1` sentinel for an undeterminable max burst lands.
    if max_burst == Elements(0) || candidate.burst_size <= max_burst {
        todo!(
            "splitBurst: expected burst_size {:?} to be larger than max_burst {max_burst:?} \
             (BurstSplitting.cpp:130-135)",
            candidate.burst_size
        )
    }
    let full_iterations = candidate.burst_size.0 / max_burst.0;
    let residual = candidate.burst_size.0 % max_burst.0;
    let required_ibuff = u32::from(full_iterations > 1) + u32::from(residual != 0);
    todo!(
        "burst_utils::processBurstSplitOrInterleave (Analyses/BurstUtils.cpp, out of campaign \
         scope) — splitting {:?} on {:?}/{comp:?} into {full_iterations} bursts of {max_burst:?} \
         plus a residual of {residual} needs {required_ibuff} additional IBuff \
         (BurstSplitting.cpp:157-158)",
        candidate.at,
        unit.on.kind()
    )
}

// crustify:todo: e285_runOn
//   authority : dcc/src/Transform/Sentient/BurstSplitting.cpp:161  (29 body lines, level 1)
//   original  : void runOn(dataflow::ProgramUnitOp unit, SenComponents comp)
//   calls     : e021_splitBurst

// crustify:todo: e430_runOn
//   authority : dcc/src/Transform/Sentient/BurstSplitting.cpp:191  (21 body lines, level 2)
//   original  : void runOn(ModuleOp module_op)
//   calls     : e285_runOn

// crustify:todo: e495_runOnOperation
//   authority : dcc/src/Transform/Sentient/BurstSplitting.cpp:213  (5 body lines, level 3)
//   original  : void runOnOperation()
//   calls     : e285_runOn, e430_runOn

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Dd2;
    use crate::islands::dataflow_ir::Units;
    use crate::islands::sentient::dialects::Val;

    /// The file header's own example: a burst of 264 where 64 is the maximum needs a loop of four
    /// full bursts AND a residual of 8, so both extra IBuff entries (`BurstSplitting.cpp:16-35`).
    #[test]
    #[should_panic(expected = "needs 2 additional IBuff")]
    fn e021_costs_the_vendors_264_over_64_example_at_two_ibuff() {
        let mut unit = ProgramUnit::<Dd2> {
            on: Units::one(DfirUnit::L3lu, Val(0)),
            precision: None,
            body: Vec::new(),
            arch: core::marker::PhantomData,
        };
        let candidate = BurstCandidate {
            at: OpId::at(&[0]),
            burst_size: Elements(264),
        };
        split_burst(&mut unit, &candidate, DfirUnit::L3lu, Elements(64));
    }
}
