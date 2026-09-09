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
//! | `e064_dump` | 064 | 0 | 10 | `dcc/src/Transform/Sentient/LocalRegionSplittingForValueCommoning.cpp:457` |

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET, so every item below is reachable only from this
// file's own tests until `e624_runOnOperation` (level 6) lands and something calls it. CI runs clippy
// with `-D warnings`, so without this the first ported leaf of a 9-unit module fails the gate.
// ⭐ REMOVE THIS WITH e624: at that point an unused item here is a real defect again.
#![allow(dead_code)]

use crate::islands::sentient::dialects::{Definitions, Op, Val, dataflow};
use crate::units::{NumFolds, Residency};

/// THE REGION OF THE ORIGINAL `uniform.uniformize_regions` A LOCAL REGION CAME FROM —
/// `const Region &original_region_` (`:150`), named by the block argument that region binds.
///
/// ⛔⛔ AN IDENTITY, NOT A BORROW, for the reason [`crate::transform::sentient::ForRef`] gives: this
/// pass rewrites the very op these regions hang off, so a `&` into it is unusable. Each region of a
/// `uniform.uniformize_regions` binds its own argument
/// ([`uniform::LocalRegion::arg`](crate::islands::sentient::dialects::uniform::LocalRegion::arg),
/// `Uniform.td:96`), so that argument names exactly one region.
///
/// ⚠️ AND THE REGION'S OPS ARE STILL NOT REACHABLE THROUGH IT HERE — but the island gap is CLOSED:
/// [`Op::UniformRegions`](crate::islands::sentient::dialects::Op::UniformRegions) is the sentient-rung
/// `uniform.uniformize_regions`, whose regions hold THIS rung's ops. `e444_analyze` and
/// `e505_transform` are the units that must take this identity to it; until then
/// [`super::collect_uniform_maps`] is handed the region's ops directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OriginalRegion(pub Val);

/// ONE TRANSFORMED LOCAL REGION — `lrs::LocalRegion` (`:137-151`): which units it now represents, and
/// where it came from, *"without having to create an actual local region in MLIR"* (`:134-136`).
///
/// ⛔ NO `Clone`: `LocalRegion(const LocalRegion &) = delete` (`:142`).
#[derive(Debug, PartialEq, Eq)]
pub struct LocalRegion {
    /// `units_` — in [`super::uniform_region::UniformRegion::add_local_region`]'s order.
    pub units: Vec<Val>,
    /// `original_region_`.
    pub original_region: OriginalRegion,
}

/// HOW FAR A DUMP IS INDENTED — `llvm::raw_ostream::indent(int)`'s argument (`:458`).
///
/// ⛔ A TYPE AND NOT A BARE COUNT, so a caller cannot pass the region's index, its unit count or any
/// other number that happens to be in scope: [`super::uniform_region::UniformRegion::dump`] indents
/// every local region by 2 (`:471`), and
/// [`RegionInfo::dump`](crate::transform::sentient::sink_scalar_copy::RegionInfo::dump) is the other
/// `raw_ostream::indent` this crate ports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Indent(pub usize);

impl LocalRegion {
    /// Replaces: e064_dump
    ///
    /// The one line a local region contributes to [`super::uniform_region::UniformRegion::dump`]:
    /// the indent, then its units' hardware names, then the placeholder body.
    ///
    /// ⭐ ITS OWN TEXT, VERBATIM: `interleaveComma` puts `", "` BETWEEN names only, and `") {...}\n"`
    /// is literally what the reference prints — the braces are its placeholder, not an elision here.
    #[must_use]
    pub fn dump(&self, defs: Definitions<'_>, indent: Indent) -> String {
        let names = self
            .units
            .iter()
            .filter_map(|unit| unit_name(*unit, defs))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "{:width$}local region units({names}) {{...}}\n",
            "",
            width = indent.0
        )
    }
}

/// `dcc::utils::getUnitNameAsString` (`Analyses/Utils.cpp:635-641`) — `<type>core<N>corelet<N>folds<N>`
/// for the `dataflow.get_unit` behind a unit handle.
///
/// ⛔ `None` IS `dyn_cast_or_null<dataflow::GetUnitOp>` FAILING, which the reference `DT_CHECK`s on
/// (`:461-462`) — it cannot be a type here because `units_` is the cluster's `ArrayRef<Value>` in the
/// reference too, so a value that is not a unit handle simply has no hardware name and contributes
/// nothing to this trace.
///
/// ⛔ AND `-1` IS AN ABSENT ATTRIBUTE, NOT A UNIT: `dcc::getCoreId`/`getCoreletId` return `-1` where
/// the op carries no `core`/`corelet` (`Utils/DccExtContext.cpp:78-125`), which is exactly the
/// residencies whose emitter prints neither ([`dataflow::Op::GetUnit`]). `getNumResults()` is the
/// `num_folds` attribute (`Transform/Dataflow/UnitFiltering.cpp:279, 296`), one result when absent.
///
/// ⚠️ CITED FROM THE OUT-OF-SCOPE `Analyses/` DIRECTORY, and ported rather than `todo!`d because it is
/// a formatter over the op's own attributes and computes no analysis result.
fn unit_name(val: Val, defs: Definitions<'_>) -> Option<String> {
    let Op::Dataflow(dataflow::Op::GetUnit {
        residency,
        unit,
        num_folds,
        ..
    }) = defs.of(val)?
    else {
        return None;
    };
    // `std::to_string(dcc::getCoreId(unit))` — the attribute, or `-1` where the op carries none.
    let attr = |value: Option<u32>| match value {
        Some(value) => value.to_string(),
        None => "-1".to_owned(),
    };
    let (core, corelet) = match residency {
        Residency::Global => (None, None),
        Residency::Scratchpad { core } => (Some(core.get()), None),
        Residency::CoreWide { core } => (Some(core.get()), Some(0)),
        Residency::Corelet { core, corelet } => (Some(core.get()), Some(corelet.get())),
    };
    let folds = num_folds.unwrap_or(NumFolds::ONE).0;
    Some(format!(
        "{}core{}corelet{}folds{folds}",
        unit.spelling(),
        attr(core),
        attr(corelet)
    ))
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::units::{Core, Corelet, DfirUnit};

    /// e064_dump — the reference's indent-then-interleave shape, over one corelet unit and one global
    /// one so both halves of `getUnitNameAsString`'s `-1` are exercised.
    #[test]
    fn dump_indents_then_interleaves_the_units_hardware_names() {
        let ops = vec![
            Op::Dataflow(dataflow::Op::GetUnit {
                reg_locale: None,
                result: Val(7),
                residency: Residency::Corelet {
                    core: Core::checked(0).expect("core 0 exists"),
                    corelet: Corelet::checked(1).expect("corelet 1 exists"),
                },
                unit: DfirUnit::Lxlu,
                num_folds: Some(NumFolds(2)),
            }),
            Op::Dataflow(dataflow::Op::GetUnit {
                reg_locale: None,
                result: Val(9),
                residency: Residency::Global,
                unit: DfirUnit::Hbm,
                num_folds: None,
            }),
        ];
        let regions: [&[Op]; 1] = [&ops];
        let defs = Definitions::from_innermost(&regions);
        let region = LocalRegion {
            units: vec![Val(7), Val(9)],
            original_region: OriginalRegion(Val(3)),
        };

        assert_eq!(
            region.dump(defs, Indent(2)),
            "  local region units(lxlucore0corelet1folds2, hbmcore-1corelet-1folds1) {...}\n"
        );
    }
}
