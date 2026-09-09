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

//! `LocalRegionSplittingForValueCommoning.cpp` — 1 of the campaign's 656 units (dependency level(s) [0]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e065_dump` | 065 | 0 | 6 | `dcc/src/Transform/Sentient/LocalRegionSplittingForValueCommoning.cpp:468` |

use super::local_region::{Indent, LocalRegion};
use crate::islands::sentient::dialects::{Definitions, Val};

/// THE `uniform.uniformize_regions` A [`UniformRegion`] WAS BUILT FROM — `original_uro_` (`:180`),
/// named by the block argument of its FIRST region.
///
/// ⛔ NOT BY ITS RESULTS: `-> ()` in 305 of the 351 of them under `dcc/test`
/// ([`uniform::Op::UniformizeRegions`](crate::islands::sentient::dialects::uniform::Op)), so a result
/// list cannot name one. The op's verifier gives it at least one region (`Uniform.cpp:125`) and every
/// region binds its own argument, so the first region's argument is a name it always has.
/// ⭐ IDENTITY, NOT A BORROW — see [`crate::transform::sentient::ForRef`] and
/// [`super::local_region::OriginalRegion`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UniformizeRegions(pub Val);

/// THE UNIFORMIZED REGION THIS PASS WANTS, BEFORE ANY OP EXISTS FOR IT — `lrs::UniformRegion`
/// (`:154-181`). `analyze` (e444) fills it, `transform` (e505) builds the op from it; the reference
/// keeps it out of the IR because *"MLIR does not allow us to add regions to an existing op"*
/// (`:157-158`).
///
/// ⛔ NO `Clone`: `UniformRegion(const UniformRegion &) = delete` (`:160`).
#[derive(Debug, PartialEq, Eq)]
pub struct UniformRegion {
    /// `local_regions_`, in the order [`UniformRegion::add_local_region`] appended them — which is the
    /// region order `transform` emits (`:347`).
    ///
    /// ⭐ `Vec` FOR `std::deque` (`:178`): nothing holds a reference to an element across a push, so
    /// the deque's stable element addresses are mechanism a port may drop.
    pub local_regions: Vec<LocalRegion>,
    /// `original_uro_`.
    pub original_uro: UniformizeRegions,
}

impl UniformRegion {
    /// Replaces: e065_dump
    ///
    /// The header naming how many local regions it holds, then each of them indented by 2, then the
    /// closing brace.
    ///
    /// ⭐ A RETURNED `String` FOR `llvm::dbgs()`, as e014_dump did: the only caller wraps the whole
    /// call in `LLVM_DEBUG` (`:266`), so nothing the reference gated becomes ungated.
    #[must_use]
    pub fn dump(&self, defs: Definitions<'_>) -> String {
        let mut out = format!(
            "uniform region (with {} local regions) {{\n",
            self.local_regions.len()
        );
        for local_region in &self.local_regions {
            out.push_str(&local_region.dump(defs, Indent(2)));
        }
        out.push_str("}\n");
        out
    }
}

#[cfg(test)]
mod unit_tests {
    use super::super::local_region::OriginalRegion;
    use super::*;

    #[test]
    fn dump_names_the_local_region_count_and_closes_the_brace() {
        let regions: [&[crate::islands::sentient::dialects::Op]; 0] = [];
        let defs = Definitions::from_innermost(&regions);
        let ur = UniformRegion {
            local_regions: Vec::new(),
            original_uro: UniformizeRegions(Val(3)),
        };
        assert_eq!(
            ur.dump(defs),
            "uniform region (with 0 local regions) {\n}\n"
        );
    }

    /// e064 HAS LANDED, so the seam that used to `todo!` here is a real call: one local region is one
    /// indented line between the header and the brace.
    #[test]
    fn dump_indents_each_local_region_by_two() {
        let regions: [&[crate::islands::sentient::dialects::Op]; 0] = [];
        let defs = Definitions::from_innermost(&regions);
        let ur = UniformRegion {
            local_regions: vec![LocalRegion {
                units: Vec::new(),
                original_region: OriginalRegion(Val(10)),
            }],
            original_uro: UniformizeRegions(Val(3)),
        };
        assert_eq!(
            ur.dump(defs),
            "uniform region (with 1 local regions) {\n  local region units() {...}\n}\n"
        );
    }
}
