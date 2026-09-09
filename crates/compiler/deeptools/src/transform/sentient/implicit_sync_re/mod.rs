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

//! `ImplicitSyncRE.cpp` — 7 of the campaign's 656 units (dependency level(s) [0, 2, 3, 4]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e047_isEqual` | 047 | 0 | 7 | `dcc/src/Transform/Sentient/ImplicitSyncRE.hpp:28` |
//! | `e048_copyTo` | 048 | 0 | 7 | `dcc/src/Transform/Sentient/ImplicitSyncRE.hpp:35` |
//! | `e049_print` | 049 | 0 | 5 | `dcc/src/Transform/Sentient/ImplicitSyncRE.hpp:49` |
//! | `e050_isOperationAUse` | 050 | 0 | 5 | `dcc/src/Transform/Sentient/ImplicitSyncRE.hpp:66` |
//! | `e438_runOn` | 438 | 2 | 18 | `dcc/src/Transform/Sentient/ImplicitSyncRE.cpp:58` |
//! | `e501_runOn` | 501 | 3 | 4 | `dcc/src/Transform/Sentient/ImplicitSyncRE.cpp:77` |
//! | `e559_runOnOperation` | 559 | 4 | 5 | `dcc/src/Transform/Sentient/ImplicitSyncRE.cpp:82` |

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET, so `ImplicitSyncGenValue` is reachable only
// from `implicit_sync_rde_tree` and its tests until `e559_runOnOperation` (level 4) lands. CI runs
// clippy with `-D warnings`, so without this the first ported leaf of the module fails the gate.
// ⭐ REMOVE THIS WITH e559: at that point an unused item here is a real defect again.
#![allow(dead_code)]

use core::num::NonZeroU32;

use crate::islands::sentient::dialects::Op;

pub(crate) mod implicit_sync_rde_tree;

/// `ImplicitSyncGenValue::tile_size_` WHEN IT IS KNOWN.
///
/// ⛔ `isUnknownValue()` IS `tile_size_ < 0` (`ImplicitSyncRE.hpp:24`) and the constructor's default
/// is `-1`, so the two states are a `NonZeroU32` and its ABSENCE — never a sentinel in the number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TileSize(NonZeroU32);

impl TileSize {
    /// The boundary a defining `sentient.sync` carries.
    #[must_use]
    pub(crate) const fn of(size: NonZeroU32) -> TileSize {
        TileSize(size)
    }

    /// `getTileSize()` (`ImplicitSyncRE.hpp:26`).
    #[must_use]
    pub(crate) const fn get(self) -> NonZeroU32 {
        self.0
    }
}

/// `ImplicitSyncGenValue` (`ImplicitSyncRE.hpp:19`) — the dataflow definition an RDE node generates.
///
/// ⭐ DECLARED HERE, WHERE ITS OWN METHODS BELONG: e047-e049 are `isEqual`/`copyTo`/`print` on this
/// class and are scheduled into this file. e045 (in [`implicit_sync_rde_tree`]) constructs it, which
/// is why the type lands first — a batch filling those anchors should UNION with this, not duplicate.
#[derive(Debug, Clone, Default)]
pub(crate) struct ImplicitSyncGenValue {
    /// `tile_size_` — `None` is the constructor's `-1`, i.e. `isUnknownValue()`.
    tile_size: Option<TileSize>,
    /// `op_` — absent for the default-constructed unknown value.
    op: Option<Op>,
}

impl ImplicitSyncGenValue {
    /// `ImplicitSyncGenValue()` — the unknown value.
    #[must_use]
    pub(crate) fn unknown() -> ImplicitSyncGenValue {
        ImplicitSyncGenValue::default()
    }

    /// `ImplicitSyncGenValue(tile_size, op)`.
    #[must_use]
    pub(crate) fn of(tile_size: TileSize, op: Op) -> ImplicitSyncGenValue {
        ImplicitSyncGenValue {
            tile_size: Some(tile_size),
            op: Some(op),
        }
    }

    /// `isUnknownValue()` (`ImplicitSyncRE.hpp:24`).
    #[must_use]
    pub(crate) const fn is_unknown_value(&self) -> bool {
        self.tile_size.is_none()
    }

    /// `getTileSize()`, absent when the value is unknown.
    #[must_use]
    pub(crate) const fn tile_size(&self) -> Option<TileSize> {
        self.tile_size
    }

    /// The op that generated it.
    #[must_use]
    pub(crate) const fn op(&self) -> Option<&Op> {
        self.op.as_ref()
    }
}

// crustify:todo: e047_isEqual
//   authority : dcc/src/Transform/Sentient/ImplicitSyncRE.hpp:28  (7 body lines, level 0)
//   original  : bool isEqual(const DataFlowDefinitionBase &rhs) const final override

// crustify:todo: e048_copyTo
//   authority : dcc/src/Transform/Sentient/ImplicitSyncRE.hpp:35  (7 body lines, level 0)
//   original  : void copyTo(DataFlowDefinitionBase &lhs) const final override

// crustify:todo: e049_print
//   authority : dcc/src/Transform/Sentient/ImplicitSyncRE.hpp:49  (5 body lines, level 0)
//   original  : void print(raw_ostream &OS) const final override

// crustify:todo: e050_isOperationAUse
//   authority : dcc/src/Transform/Sentient/ImplicitSyncRE.hpp:66  (5 body lines, level 0)
//   original  : bool isOperationAUse(const Operation &op) const final override

// crustify:todo: e438_runOn
//   authority : dcc/src/Transform/Sentient/ImplicitSyncRE.cpp:58  (18 body lines, level 2)
//   original  : void runOn(dataflow::ProgramUnitOp unit)
//   calls     : e378_optimize

// crustify:todo: e501_runOn
//   authority : dcc/src/Transform/Sentient/ImplicitSyncRE.cpp:77  (4 body lines, level 3)
//   original  : void runOn(ModuleOp module_op)
//   calls     : e438_runOn

// crustify:todo: e559_runOnOperation
//   authority : dcc/src/Transform/Sentient/ImplicitSyncRE.cpp:82  (5 body lines, level 4)
//   original  : void runOnOperation()
//   calls     : e438_runOn, e501_runOn
