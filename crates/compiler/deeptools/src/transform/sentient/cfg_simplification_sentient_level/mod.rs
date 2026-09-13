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

//! `CFGSimplificationSentientLevel.cpp` — 12 of the campaign's 656 units (dependency level(s) [0, 1, 5]).
//!
//! | unit | entry | level | lines | authority path:line |
//! |---|---|---|---|---|
//! | `e023_incrementLength` | 023 | 0 | 1 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:293` |
//! | `e024_incrementIntervalMarker` | 024 | 0 | 5 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:297` |
//! | `e025_getIntervalStride` | 025 | 0 | 5 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:302` |
//! | `e026_changeToDefaultValue` | 026 | 0 | 6 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:316` |
//! | `e027_print` | 027 | 0 | 11 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:323` |
//! | `e028_getPrevVal` | 028 | 0 | 4 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:380` |
//! | `e029_getEV` | 029 | 0 | 6 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:473` |
//! | `e030_isFull` | 030 | 0 | 5 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:527` |
//! | `e286_recomputeAsDefaultVals` | 286 | 1 | 58 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:396` |
//! | `e287_getTableEntryAtIdx` | 287 | 1 | 4 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:512` |
//! | `e288_createTableEntryAtIdx` | 288 | 1 | 4 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:517` |
//! | `e594_runOnOperation` | 594 | 5 | 201 | `dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:713` |

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET: `e594_runOnOperation` is ported, but nothing calls
// it, so every item below is reachable only from this file's own tests. CI runs clippy with
// `-D warnings`, so without this the module fails the gate.
// ⭐ THE CONDITION IS PIPELINE WIRING, NOT e594 — the note here used to say e594 would discharge it,
// and it does not: every sibling pass module whose entry is ported (`remove_redundant_conditionals`
// included) still carries this allow for the same reason.
#![allow(dead_code)]

use core::num::NonZeroI64;
use core::ops::{AddAssign, Mul, Sub, SubAssign};

pub(crate) mod pattern_simplification_manager;

use crate::arch::Arch;
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::Program;
use crate::islands::sentient::dialects::{self as ir, Op, Val, sentient};
use crate::model::Model;
use crate::workload::Workload;
use pattern_simplification_manager::{
    CondNode, EncodingIfOpCount, IvDim, IvValuesAndFilters, LeafSink, LoopCloning, LoopInfo, Mark,
    Marks, OpPath, PatternSimplificationManager, TreeSeam,
};
use std::collections::BTreeMap;

/// `EnableNonZeroStrideSeqSimplifications` (`:76-80`) — *"Allow replacing fixed non-zero stride
/// sequences of values yielded by conditionals with iterator arguments."*, `cl::init(false)`.
///
/// ⛔ A `dcc-opt` COMMAND-LINE FLAG, NOT A PROGRAM PROPERTY, and this crate has no flags — the same
/// reading [`super::loop_splitting_and_unrolling::is_ok_to_unroll`] made of `DisableLoopUnroll`.
/// ⭐ A NAMED `const` RATHER THAN A FOLDED LITERAL BECAUSE IT DECIDES WHICH SEQUENCE KINDS EXIST: at
/// `false` `padTableSlice` pads to the slice's own maximum instead of 3 (`:2422-2423`) and never asks
/// for a monotone slot (`:2437-2440`), so both branches fold away rather than going unread.
pub(crate) const ENABLE_NON_ZERO_STRIDE_SEQ_SIMPLIFICATIONS: bool = false;

/// `DisableThisPass` (`:66-69`) — `cl::init(false)`, so the pass SHIPS LIVE.
pub(crate) const DISABLE_THIS_PASS: bool = false;

/// `EnableDeadBranchRemoval` (`:83-88`) — `cl::init(false)`, so a node that is not a candidate for
/// RESULT-based simplification gets no table at all and its dead branches are never removed
/// (`:750-752`). ⭐ IT DOES NOT DISABLE THE CLEANUP: a branch e031 marks dead on a node that IS a
/// candidate is still erased at the end of the round.
pub(crate) const ENABLE_DEAD_BRANCH_REMOVAL: bool = false;

// ── THE FILE'S VALUE VOCABULARY ─────────────────────────────────────────────────────────────────
//
// `dcc::WidestIntType` is `int64_t` (`Analyses/CFGSSentientLevelConditionalTree.hpp:24`) and this file
// spends it on four unrelated quantities. Each is its own newtype, so passing a table index where a
// free-IV marker belongs is an E0308.

/// A COUNT OF TABLE ENTRIES — `Sequence::length_` (`:233`), `TableSlice::length_` (`:346`) and every
/// increment applied to either.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub(crate) struct Entries(pub(crate) i64);

impl AddAssign for Entries {
    fn add_assign(&mut self, rhs: Entries) {
        self.0 += rhs.0;
    }
}

/// THE LAST FREE-IV VALUE A SEQUENCE COVERS — `Sequence::interval_marker_` (`:246`). The code
/// generated for the sequence is `if (free iv <= interval_marker_) yield <iter arg>` (`:244-245`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct IntervalMarker(pub(crate) i64);

/// HOW FAR ONE TABLE ENTRY MOVES THE FREE IV — `Sequence::interval_stride_` (`:249`).
///
/// ⭐ NON-ZERO BY CONSTRUCTION, WHICH IS WHERE `getIntervalStride`'s
/// `DT_CHECK_MSG(interval_stride_ != 0, "Interval stride should never be zero.")` (`:303-304`) WENT —
/// a build-time guard rather than a run-time one, so no caller can be handed a zero to divide by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IntervalStride(NonZeroI64);

impl IntervalStride {
    /// The stride, whose non-zeroness the caller has already proved.
    pub(crate) const fn new(stride: NonZeroI64) -> IntervalStride {
        IntervalStride(stride)
    }

    /// The signed stride — its SIGN is what picks `sle` over `sge` when the interval test is emitted
    /// (`:2753`, `:2835`).
    #[must_use]
    pub(crate) const fn get(self) -> NonZeroI64 {
        self.0
    }

    /// HOW FAR ONE ENTRY MOVES THE FREE IV, as a distance — `:2676-2678` tests whether two interval
    /// markers are exactly one entry apart, and a stride is not an [`IntervalDelta`] until it is
    /// multiplied by a count.
    #[must_use]
    pub(crate) const fn one_entry(self) -> IntervalDelta {
        IntervalDelta(self.0.get())
    }
}

/// A SIGNED DISTANCE ALONG THE FREE IV — `interval_stride_ * increment` (`:300`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IntervalDelta(i64);

impl Mul<Entries> for IntervalStride {
    type Output = IntervalDelta;

    fn mul(self, rhs: Entries) -> IntervalDelta {
        IntervalDelta(self.0.get() * rhs.0)
    }
}

impl AddAssign<IntervalDelta> for IntervalMarker {
    fn add_assign(&mut self, rhs: IntervalDelta) {
        self.0 += rhs.0;
    }
}

/// `m1 - first sequence stride * first sequence length` (`:2405-2407`) — the marker of the dead
/// interval `padTableSlice` puts in front of the first sequence.
impl SubAssign<IntervalDelta> for IntervalMarker {
    fn sub_assign(&mut self, rhs: IntervalDelta) {
        self.0 -= rhs.0;
    }
}

/// The distance between two markers — what `:2676-2678` compares against ONE entry's stride.
impl Sub for IntervalMarker {
    type Output = IntervalDelta;

    fn sub(self, rhs: IntervalMarker) -> IntervalDelta {
        IntervalDelta(self.0 - rhs.0)
    }
}

impl Sub<IntervalDelta> for IntervalMarker {
    type Output = IntervalMarker;

    /// `seq->getIntervalMarker().value() - interval_step * (length_remaining - 1)` (`:427-429`) — the
    /// marker of the FIRST element of a monotone sequence, the stored one being the LAST's.
    fn sub(self, rhs: IntervalDelta) -> IntervalMarker {
        IntervalMarker(self.0 - rhs.0)
    }
}

/// AN INDEX INTO THE TABLE'S 1-D ARRAY — `TableSlice::start_idx_` (`:341`), the entry for
/// `(i1,..,in) = (c1,..,cn)` being at `c1(i2*..*in) + .. + cn-1(in) + cn` (`:486-487`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct TableIndex(pub(crate) i64);

/// HOW MANY ENTRIES A WHOLE TABLE HAS — `table_size` (`:500`).
///
/// ⭐ A LENGTH, NOT AN [`Entries`] DELTA, and that is the guard: `std::vector<TableEntry *>(table_size,
/// nullptr)` (`:506`) cannot be asked for a negative count, so the one place the reference's
/// `WidestIntType` would have to be narrowed has nothing to narrow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct TableSize(pub(crate) usize);

/// THE TABLE-INDEX STEP BETWEEN ONE SLICE'S ENTRIES — `TableSlice::idx_stride_` (`:344`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IndexStride(pub(crate) i64);

/// A VALUE THE OUT-OF-SCOPE `ExpressionEvaluator` OWNS, HELD BY IDENTITY — as the reference holds it.
///
/// Every `EvaluatedValue` field in this file is a non-owning `const EvaluatedValue *` into the
/// evaluator's arena (`Analyses/ExpressionEvaluatorUtils.h:48`), and that whole analysis is outside
/// this campaign (`crustify-senpass/OUTSIDE-DEPS.tsv`).
///
/// ⛔ `==` IS POINTER IDENTITY, NOT `EvaluatedValue::operator==`. Content equality — what
/// `*val == *(*it_next)->getLB()` (`:438`) asks for — belongs to the analysis and is out of scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct EvaluatedValueId(u32);

impl EvaluatedValueId {
    /// The id the evaluator handed back for one value.
    pub(crate) const fn new(id: u32) -> EvaluatedValueId {
        EvaluatedValueId(id)
    }

    /// `raw_ostream &operator<<(raw_ostream &, const EvaluatedValue &)`
    /// (`Analyses/ExpressionEvaluatorUtils.h:405`) — the analysis's own rendering, and out of campaign
    /// scope. Reached only from [`Sequence::print`], which is a `DEBUG_WITH_TYPE` trace.
    fn write(self, _out: &mut String) {
        todo!(
            "EvaluatedValue::operator<< (Analyses/ExpressionEvaluatorUtils.h:405) — out of campaign scope"
        )
    }
}

/// THE TWO KINDS OF SEQUENCE — `enum SequenceKind` (`:223`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SequenceKind {
    /// `kDefaultValue` — a constant sequence `D D ... D` (`:220`).
    DefaultValue,
    /// `kMonotoneSequence` — `a1 ... ak` with fixed stride `|ai+1 - ai| = x` (`:221-222`).
    MonotoneSequence,
}

/// A SEQUENCE FOUND WITHIN ONE SLICE OF THE TABLE — `class Sequence` (`:216`).
///
/// ⛔ `ExpressionEvaluator &evaluator_` (`:252`) IS NOT A FIELD. The evaluator is out of campaign
/// scope, and a back-reference in every sequence would put its lifetime into every type in this file;
/// the units that do `EvaluatedValue` arithmetic take it as a parameter instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Sequence {
    /// `kind_` (`:224`).
    pub(crate) kind: SequenceKind,
    /// `lb_` (`:231`) — `None` is the reference's `nullptr`, which is what makes a "dummy" sequence
    /// carrying only the fields consistent across table slices (`:227-229`).
    pub(crate) lb: Option<EvaluatedValueId>,
    /// `stride_` (`:232`).
    pub(crate) stride: Option<EvaluatedValueId>,
    /// `length_` (`:233`).
    pub(crate) length: Entries,
    /// `shifted_lb_` (`:240`) — the LB shifted by (#previous entries in the slice) * `stride_`, and
    /// the starting value of the iterator argument created for a monotone sequence (`:235-239`).
    pub(crate) shifted_lb: Option<EvaluatedValueId>,
    /// `interval_marker_` (`:246`) — `None` after `setIntervalMarker(std::nullopt)` clears it on a
    /// template sequence whose members disagree (`:2138-2139`).
    pub(crate) interval_marker: Option<IntervalMarker>,
    /// `interval_stride_` (`:249`), read through [`Sequence::interval_stride`].
    interval_stride: IntervalStride,
}

impl Sequence {
    /// `Sequence(const EvaluatedValue &lb, const EvaluatedValue &stride, ...)` (`:256-267`) —
    /// `shifted_lb_` starts null (`:264`).
    ///
    /// ⛔ THE CONSTANT-STRIDE OVERLOAD (`:269-280`) IS NOT HERE: it differs only by
    /// `stride_(&evaluator.getConstant(stride))` (`:275`), which is the out-of-scope evaluator's.
    pub(crate) const fn new(
        lb: EvaluatedValueId,
        stride: EvaluatedValueId,
        length: Entries,
        interval_marker: IntervalMarker,
        interval_stride: IntervalStride,
        kind: SequenceKind,
    ) -> Sequence {
        Sequence {
            kind,
            lb: Some(lb),
            stride: Some(stride),
            length,
            shifted_lb: None,
            interval_marker: Some(interval_marker),
            interval_stride,
        }
    }

    /// Replaces: e023_incrementLength
    ///
    /// `length_ += increment` (`:293`) — the sequence now covers `increment` more table entries.
    pub(crate) fn increment_length(&mut self, increment: Entries) {
        self.length += increment;
    }

    /// Replaces: e025_getIntervalStride
    ///
    /// `interval_stride_` (`:302-306`). Its `DT_CHECK_MSG(interval_stride_ != 0, ..)` is discharged by
    /// [`IntervalStride`] being non-zero by construction, so this cannot stop.
    #[must_use]
    pub(crate) const fn interval_stride(&self) -> IntervalStride {
        self.interval_stride
    }

    /// THE PROOF `incrementIntervalMarker` DEMANDS — `interval_marker_.has_value()` (`:298`), made
    /// once here instead of asserted on every call.
    pub(crate) fn marked_mut(&mut self) -> Option<MarkedSequence<'_>> {
        let interval_stride = self.interval_stride;
        self.interval_marker.as_mut().map(|marker| MarkedSequence {
            marker,
            interval_stride,
        })
    }

    /// THE PROOF `changeToDefaultValue` DEMANDS — `kind_ == kMonotoneSequence` (`:317`), made once
    /// here instead of asserted on every call.
    pub(crate) fn as_monotone_mut(&mut self) -> Option<MonotoneSequence<'_>> {
        match self.kind {
            SequenceKind::MonotoneSequence => Some(MonotoneSequence(self)),
            SequenceKind::DefaultValue => None,
        }
    }

    /// Replaces: e027_print
    ///
    /// One sequence as debug text (`:323-333`). Its only callers are
    /// `DEBUG_WITH_TYPE(VerboseDebug, seq->print(llvm::dbgs().indent(2)))` (`:1148`, `:1289`, `:2044`)
    /// and `raw_ostream::indent` writes its spaces before returning the stream — so the indent is the
    /// CALLER's and this writes bare, exactly as the reference does.
    ///
    /// ⛔ TRAP: RENDERING AN `EvaluatedValue` IS OUT OF CAMPAIGN SCOPE, so any branch that reaches an
    /// `lb_` or a `stride_` stops in [`EvaluatedValueId::write`]. The dummy-with-neither branch is
    /// complete. See [`write_interval_marker`] for the one spelling that is this port's choice.
    pub(crate) fn print(&self, out: &mut String) {
        // `if (!lb_ || !stride_)` (`:324`) — either one missing makes it a dummy.
        match (self.lb, self.stride) {
            (Some(lb), Some(stride)) => {
                // `:330-332`, one statement, and note the two-space continuation indents.
                out.push_str("Sequence with lb: ");
                lb.write(out);
                out.push_str("\n  stride: ");
                stride.write(out);
                out.push_str("\n  length: ");
                out.push_str(&self.length.0.to_string());
                out.push_str("\n  interval marker: ");
                write_interval_marker(self.interval_marker, out);
                out.push('\n');
            }
            _ => {
                // `:325-328` — and `"interval marker:"` here has NO space after the colon, unlike the
                // full sequence's `"\n  interval marker: "`.
                out.push_str("Dummy sequence with ");
                out.push_str("interval marker:");
                write_interval_marker(self.interval_marker, out);
                out.push('\n');
                if let Some(lb) = self.lb {
                    out.push_str("lb ");
                    lb.write(out);
                    out.push('\n');
                }
                if let Some(stride) = self.stride {
                    out.push_str("stride ");
                    stride.write(out);
                    out.push('\n');
                }
            }
        }
    }
}

/// `OS << interval_marker_` (`:326`, `:332`) — a `std::optional<dcc::WidestIntType>` through LLVM's
/// own `operator<<`.
///
/// ⛔ TRAP: THE EMPTY SPELLING IS THIS PORT'S CHOICE, NOT A READING. No LLVM header ships in
/// `/Users/nickm/git/deeptools-src`, so how the reference renders an empty `std::optional` could not
/// be read; the present case is exact. Debug trace only — no golden compares these bytes.
fn write_interval_marker(marker: Option<IntervalMarker>, out: &mut String) {
    match marker {
        Some(marker) => out.push_str(&marker.0.to_string()),
        None => out.push_str("nullopt"),
    }
}

/// A [`Sequence`] WHOSE INTERVAL MARKER IS PRESENT — the witness for
/// `DT_CHECK(interval_marker_.has_value() && "Can only increment valid interval marker")` (`:298-299`),
/// handed out by [`Sequence::marked_mut`].
pub(crate) struct MarkedSequence<'a> {
    marker: &'a mut IntervalMarker,
    interval_stride: IntervalStride,
}

impl MarkedSequence<'_> {
    /// Replaces: e024_incrementIntervalMarker
    ///
    /// `interval_marker_.value() += interval_stride_ * increment` (`:300`) — called when the sequence
    /// is extended to the RIGHT; extending it to the LEFT leaves the marker alone (`:294-296`).
    pub(crate) fn increment_interval_marker(&mut self, increment: Entries) {
        *self.marker += self.interval_stride * increment;
    }
}

/// A [`Sequence`] PROVED `kMonotoneSequence` — the witness for
/// `DT_CHECK_MSG(kind_ == kMonotoneSequence, "Should not try to change kDefaultValue sequences.")`
/// (`:317-318`), handed out by [`Sequence::as_monotone_mut`].
pub(crate) struct MonotoneSequence<'a>(&'a mut Sequence);

impl MonotoneSequence<'_> {
    /// Replaces: e026_changeToDefaultValue
    ///
    /// `kind_ = kDefaultValue; std::swap(lb_, shifted_lb_)` (`:319-320`) — the SHIFTED LB becomes the
    /// value the sequence yields, since no iterator argument starts at it any more (`:314-315`).
    /// Consumes the witness: the sequence is not monotone afterwards.
    pub(crate) fn change_to_default_value(self) {
        self.0.kind = SequenceKind::DefaultValue;
        core::mem::swap(&mut self.0.lb, &mut self.0.shifted_lb);
    }
}

/// ONE SLICE OF THE TABLE — `class TableSlice` (`:337`).
///
/// ⛔ `std::vector<Sequence *> sequences_` (`:350`) IS OWNING — `~TableSlice` deletes every element
/// (`:375-377`) — so it is a `Vec<Sequence>` and `Drop` is the destructor. `evaluator_` (`:362`) is a
/// parameter here for the same reason it is on [`Sequence`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TableSlice {
    /// `start_idx_` (`:341`) — table index of this slice's first entry.
    pub(crate) start_idx: TableIndex,
    /// `idx_stride_` (`:344`).
    pub(crate) idx_stride: IndexStride,
    /// `length_` (`:346`).
    pub(crate) length: Entries,
    /// `prev_val_` (`:348`), read through [`TableSlice::prev_val`].
    prev_val: EvaluatedValueId,
    /// `sequences_` (`:350`) — a slice consists of one or more sequences (`:349`).
    pub(crate) sequences: Vec<Sequence>,
    /// `has_inserted_monotone_seq_before_last_` (`:354`).
    pub(crate) has_inserted_monotone_seq_before_last: bool,
    /// `is_1D_rep_of_table_` (`:359`) — always true when the table is 1-D to start with (`:358`).
    pub(crate) is_1d_rep_of_table: bool,
}

impl TableSlice {
    /// `TableSlice(start_idx, idx_stride, length, evaluator, is_1D_rep_of_table)` (`:365-374`).
    ///
    /// ⛔ `prev_val_(&evaluator.getConstant(0))` (`:371`) IS A PARAMETER: `getConstant` belongs to the
    /// out-of-scope evaluator, so the caller passes the id it returned for zero.
    pub(crate) const fn new(
        start_idx: TableIndex,
        idx_stride: IndexStride,
        length: Entries,
        zero: EvaluatedValueId,
        is_1d_rep_of_table: bool,
    ) -> TableSlice {
        TableSlice {
            start_idx,
            idx_stride,
            length,
            prev_val: zero,
            sequences: Vec::new(),
            has_inserted_monotone_seq_before_last: false,
            is_1d_rep_of_table,
        }
    }

    /// Replaces: e028_getPrevVal
    ///
    /// The previous value seen in this slice (`:347`, `:380-383`). `DT_CHECK(prev_val_)` (`:381`) is
    /// discharged by construction: the constructor sets it to the evaluator's constant zero (`:371`)
    /// and `setPrevVal` takes a reference (`:384`), so the reference's pointer is never null.
    #[must_use]
    pub(crate) const fn prev_val(&self) -> EvaluatedValueId {
        self.prev_val
    }

    /// `void setPrevVal(const EvaluatedValue &prev_val)` (`:384`) — an excluded field accessor, but the
    /// field stays private so that [`TableSlice::prev_val`]'s discharged `DT_CHECK` cannot be undone by
    /// a caller storing nothing there.
    pub(crate) const fn set_prev_val(&mut self, prev_val: EvaluatedValueId) {
        self.prev_val = prev_val;
    }

    /// Replaces: e286_recomputeAsDefaultVals
    ///
    /// Every sequence of this slice re-expressed as default values: padding (length 0) is dropped, a
    /// monotone sequence of length 1 or stride 0 is simply relabelled (`:405-410`), and any other
    /// monotone sequence becomes one length-1 default sequence PER ELEMENT.
    ///
    /// ⛔ THE LAST ELEMENT IS ABSORBED INTO THE **NEXT** SEQUENCE, NOT THE NEW LIST (`:436-440`): when
    /// its value equals that sequence's LB the next sequence grows LEFTWARD, which is why its interval
    /// marker is deliberately left alone and why the walk must be able to mutate a sequence it has not
    /// reached yet. ⭐ AND THE MARKERS RUN FORWARDS FROM `marker - step * (length - 1)`: the stored
    /// marker is the LAST element's (`:425-429`).
    pub(crate) fn recompute_as_default_vals(&mut self, evaluator: &mut impl ExpressionEvaluator) {
        let zero = evaluator.get_constant(0);
        let mut new_sequences: Vec<Sequence> = Vec::new();
        let mut index = 0usize;
        while index < self.sequences.len() {
            let mut seq = self.sequences[index];
            // `if (seq->getLength() == 0)` (`:402-405`) — a padding sequence is dropped.
            if seq.length == Entries(0) {
                index += 1;
                continue;
            }
            // `*seq->getStride() == evaluator_.getConstant(0)` (`:408-409`) — a CONTENT comparison,
            // so it is the analysis's `operator==` and not this file's handle identity.
            let stride_is_zero = match seq.stride {
                Some(stride) => evaluator.equal(stride, zero),
                None => false,
            };
            if seq.kind == SequenceKind::MonotoneSequence
                && (seq.length == Entries(1) || stride_is_zero)
            {
                seq.kind = SequenceKind::DefaultValue;
            }
            if seq.kind == SequenceKind::DefaultValue {
                new_sequences.push(seq);
                index += 1;
                continue;
            }
            // `DT_CHECK(stride && ..)` and `DT_CHECK(getIntervalMarker().has_value() && ..)`
            // (`:418-423`) — a non-template sequence has both.
            let (Some(mut val), Some(stride), Some(marker)) =
                (seq.lb, seq.stride, seq.interval_marker)
            else {
                panic!(
                    "DT_CHECK(Expect non-template sequences to have a valid stride and interval \
                     marker) (`CFGSimplificationSentientLevel.cpp:418-423`): {seq:?}"
                );
            };
            let interval_step = seq.interval_stride();
            let mut interval_marker = marker - interval_step * Entries(seq.length.0 - 1);
            for _ in 0..seq.length.0 - 1 {
                new_sequences.push(Sequence::new(
                    val,
                    zero,
                    Entries(1),
                    interval_marker,
                    interval_step,
                    SequenceKind::DefaultValue,
                ));
                val = evaluator.evaluate_sum(val, stride);
                interval_marker += interval_step * Entries(1);
            }
            // `auto it_next = std::next(it_seq, 1)` (`:437`) — the sequence AFTER this one, in the
            // list still being walked, which is why this grows `self.sequences[index + 1]`.
            let absorbed = match self.sequences.get(index + 1).and_then(|next| next.lb) {
                Some(next_lb) => evaluator.equal(val, next_lb),
                None => false,
            };
            if absorbed {
                self.sequences[index + 1].increment_length(Entries(1));
            } else {
                new_sequences.push(Sequence::new(
                    val,
                    zero,
                    Entries(1),
                    interval_marker,
                    interval_step,
                    SequenceKind::DefaultValue,
                ));
            }
            index += 1;
        }
        self.sequences = new_sequences;
    }
}

/// WHICH ARM OF AN `if` A LEAF IS — `isThenNode()` / `isElseNode()` (`Analysis/ConditionalTree.hpp:98`,
/// `:100`), whose third case is `llvm_unreachable("should only be called from Then/Else node")`
/// (`:103`) and therefore not a state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IfRegion {
    /// `getParentNode()->getThenRegion()`.
    Then,
    /// `getParentNode()->getElseRegion()`.
    Else,
}

/// THE BLOCK OF CODE AT A LEAF — `CondNode::getBlock()` (`Analysis/ConditionalTree.hpp:96-106`).
///
/// ⭐ IT IS NOT AN ARBITRARY BLOCK. `getBlock()` returns `&getParentNode()->getThenRegion().front()`
/// or the else region's front, so an if-op path plus which arm names it exactly — and this island has
/// no `Block *` to hold instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BlockPath {
    /// The `sentient.if` owning the region — `getParentNode()`'s operation.
    pub(crate) if_op: OpPath,
    /// Which of its two regions.
    pub(crate) region: IfRegion,
}

/// WHAT `createTableEntryAtIdx` READS OFF A `dcc::CFGSCondNode *leaf` — and nothing else.
///
/// ⛔ `Analyses/CFGSSentientLevelConditionalTree.*` IS OUT OF CAMPAIGN SCOPE, so this is the leaf's
/// PROJECTION rather than the node: exactly `getResults()` (`:469-471`) and `getBlock()` (`:472`),
/// which is every field of the entry the reference builds from it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Leaf {
    /// `getResults()` (`Analyses/CFGSSentientLevelConditionalTree.hpp:77`) — what this leaf yields.
    pub(crate) results: Vec<Val>,
    /// `getBlock()` — `None` for a node with children, which the reference gives `nullptr`
    /// (`Analysis/ConditionalTree.hpp:97`).
    pub(crate) block: Option<BlockPath>,
}

/// ONE LEAF'S ENTRY IN THE TABLE — `class TableEntry` (`:456`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TableEntry {
    /// `ev_` (`:458`) — `None` is the reference's dummy `nullptr` for a conditional with no results,
    /// which still gets a table so dead branches can be removed (`:463-467`).
    ev: Option<EvaluatedValueId>,
    /// `block_` (`:460`) — the block of code at this leaf, read through `getBlock` (`:479`, an
    /// excluded field accessor).
    pub(crate) block: Option<BlockPath>,
}

impl TableEntry {
    /// `TableEntry(dcc::CFGSCondNode *leaf, ExpressionEvaluator &evaluator)` (`:468-472`), whose
    /// `ev_` is `nullptr` when the leaf yields nothing and `evaluator.evaluateValue(..)` otherwise —
    /// both the leaf and the evaluator being out of campaign scope, that choice is the caller's.
    pub(crate) const fn new(ev: Option<EvaluatedValueId>, block: Option<BlockPath>) -> TableEntry {
        TableEntry { ev, block }
    }

    /// Replaces: e029_getEV
    ///
    /// The yielded value (`:473-478`). `DT_CHECK_MSG(ev_, "Should only read value of TableEntry if
    /// corresponding leaf yields results")` becomes the `Option`: a result-less leaf's entry has no
    /// value to read, and no caller can reach one without saying what it does about that.
    #[must_use]
    pub(crate) const fn ev(&self) -> Option<EvaluatedValueId> {
        self.ev
    }
}

/// THE TABLE OF VALUES FOR EVERY TUPLE OF IV VALUES IN THE CURRENT NESTED CONDITIONAL —
/// `class Table` (`:484`), as a 1-D array (`:485-487`).
///
/// ⭐ `const Type predicate_type_` (`:491`) AND `Type entry_type_` (`:494`) ARE BOTH [`ScalarTy`]:
/// each is handed straight to a `sentient.scalar_constant` (`:2564`, `:2822`) or names a
/// `sentient.if`'s single result type (`:2731`, `:2833`), which the island spells with its scalar
/// type and nothing wider. `evaluator_` (`:497`) is a parameter, as on [`Sequence`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Table {
    /// `table_entries_` (`:488`) — `std::vector<TableEntry *>(table_size, nullptr)` (`:506`), owning:
    /// `~Table` deletes every element (`:508-510`), so `Drop` is the destructor.
    entries: Vec<Option<TableEntry>>,
    /// `predicate_type_` (`:491`) — the type of the predicates in this nested conditional.
    predicate_type: ScalarTy,
    /// `entry_type_` (`:494`) — the type of the entries in the table.
    entry_type: ScalarTy,
}

impl Table {
    /// `Table(table_size, predicate_type, entry_type, evaluator)` (`:500-506`) — every entry starts
    /// null.
    pub(crate) fn new(
        table_size: TableSize,
        predicate_type: ScalarTy,
        entry_type: ScalarTy,
    ) -> Table {
        Table {
            entries: vec![None; table_size.0],
            predicate_type,
            entry_type,
        }
    }

    /// `int64_t getSize() const` (`:525`) — an excluded field accessor.
    #[must_use]
    pub(crate) const fn size(&self) -> TableSize {
        TableSize(self.entries.len())
    }

    /// `Type getTableEntryType() const` (`:522`) — an excluded field accessor.
    #[must_use]
    pub(crate) const fn table_entry_type(&self) -> ScalarTy {
        self.entry_type
    }

    /// `Type getPredicateType() const` (`:524`) — an excluded field accessor.
    #[must_use]
    pub(crate) const fn predicate_type(&self) -> ScalarTy {
        self.predicate_type
    }

    /// Replaces: e030_isFull
    ///
    /// Whether every index of the table has an entry (`:527-531`) — a table with a hole means the
    /// nested conditional did not cover every tuple of IV values, so it is not simplifiable as one.
    #[must_use]
    pub(crate) fn is_full(&self) -> bool {
        self.entries.iter().all(Option::is_some)
    }

    /// Replaces: e287_getTableEntryAtIdx
    ///
    /// The entry at `idx`, `None` where the reference returns its `nullptr` — a tuple of IV values no
    /// leaf covered (`:512-515`).
    ///
    /// ⛔ THE OUTER `None` AND THE INNER ONE ARE DIFFERENT ANSWERS and only one is a state: an index
    /// past the end is `DT_CHECK_MSG(idx < table_entries_.size(), "Expect valid index to table.")`
    /// (`:513`), which stops, while a hole is what every caller's own `DT_CHECK_MSG` tests (`:2187`).
    #[must_use]
    pub(crate) fn table_entry_at_idx(&self, idx: TableIndex) -> Option<&TableEntry> {
        match usize::try_from(idx.0)
            .ok()
            .filter(|at| *at < self.entries.len())
        {
            Some(at) => self.entries[at].as_ref(),
            None => panic!(
                "DT_CHECK(idx < table_entries_.size()) Expect valid index to table. \
                 (`CFGSimplificationSentientLevel.cpp:513`): {idx:?} of {}",
                self.entries.len()
            ),
        }
    }

    /// Replaces: e288_createTableEntryAtIdx
    ///
    /// The entry for `leaf` at `idx` (`:517-521`) — its value is the evaluation of the leaf's FIRST
    /// result, and a leaf yielding nothing gets the reference's dummy (`:469-471`).
    ///
    /// ⛔ THE BOUNDS `DT_CHECK` COMES FIRST, BEFORE THE ENTRY IS BUILT (`:518`), so an out-of-range
    /// index never reaches the evaluator; and this OVERWRITES, exactly as `table_entries_[idx] = new
    /// TableEntry(..)` does — the reference leaks the old entry rather than refusing.
    pub(crate) fn create_table_entry_at_idx(
        &mut self,
        idx: TableIndex,
        leaf: &Leaf,
        evaluator: &mut impl ExpressionEvaluator,
    ) {
        let Some(at) = usize::try_from(idx.0)
            .ok()
            .filter(|at| *at < self.entries.len())
        else {
            panic!(
                "DT_CHECK(idx < table_entries_.size()) Expect valid index to table. \
                 (`CFGSimplificationSentientLevel.cpp:518`): {idx:?} of {}",
                self.entries.len()
            )
        };
        let ev = leaf
            .results
            .first()
            .map(|result| evaluator.evaluate_value(*result));
        self.entries[at] = Some(TableEntry::new(ev, leaf.block.clone()));
    }
}

/// THE OUT-OF-SCOPE `ExpressionEvaluator` (`Analyses/ExpressionEvaluatorUtils.h:196`) AS A SEAM —
/// the reference's `ExpressionEvaluator &evaluator_`, which every type in this file holds a
/// back-reference to and this port takes as a parameter.
///
/// ⛔ NO IMPLEMENTATION SHIPS IN THIS CAMPAIGN AND NONE MAY: `Analyses/` is outside its scope
/// (`crustify-senpass/OUTSIDE-DEPS.tsv`). The trait declares exactly the calls the ported bodies
/// make, so the boundary is a PARAMETER THE CALLER SUPPLIES rather than a guess at what the analysis
/// would have answered — which is why it is a trait and not the `todo!` this file's
/// `pattern_simplification_manager::step_matches_target` uses for a question no ported body can ask a
/// caller to answer.
///
/// ⛔ A SECOND SPELLING OF THE SAME SEAM: [`super::analyses::ExpressionEvaluator`] states it over
/// [`super::analyses::EvaluatedValue`], but every evaluator-owned field in THIS file is already an
/// [`EvaluatedValueId`] whose `==` is documented as pointer identity, so the content equality
/// `:2676` needs has to be a method here. Converging the two means converting this file's whole
/// handle vocabulary, which is a decision for a review pass and not for one batch.
pub(crate) trait ExpressionEvaluator {
    /// `const EvaluatedValue &getConstant(int64_t c)` (`ExpressionEvaluatorUtils.h:335`).
    fn get_constant(&mut self, value: i64) -> EvaluatedValueId;

    /// `const EvaluatedValue &evaluateSub(const EvaluatedValue &, const EvaluatedValue &)` (`:260`).
    fn evaluate_sub(&mut self, lhs: EvaluatedValueId, rhs: EvaluatedValueId) -> EvaluatedValueId;

    /// `const EvaluatedValue &evaluateSum(const EvaluatedValue &, const EvaluatedValue &)` (`:242`) —
    /// `*val = &evaluator_.evaluateSum(*val, *seq->getStride())` (`:433`), one step along a monotone
    /// sequence being expanded.
    fn evaluate_sum(&mut self, lhs: EvaluatedValueId, rhs: EvaluatedValueId) -> EvaluatedValueId;

    /// `const EvaluatedValue &evaluateValue(Value)` (`:219`) — an SSA value's evaluated form, which is
    /// what a table entry holds for its leaf's result (`:470`).
    fn evaluate_value(&mut self, value: Val) -> EvaluatedValueId;

    /// `const EvaluatedValue &evaluateMultiplyByConst(const EvaluatedValue &, int64_t)` (`:285`) —
    /// the factor is always a count of table entries at these call sites.
    fn evaluate_multiply_by_const(
        &mut self,
        value: EvaluatedValueId,
        factor: Entries,
    ) -> EvaluatedValueId;

    /// `EvaluatedValue::operator==` — ⛔ CONTENT EQUALITY, which [`EvaluatedValueId`]'s own `==` is
    /// deliberately NOT (see its note).
    fn equal(&self, lhs: EvaluatedValueId, rhs: EvaluatedValueId) -> bool;

    /// `Value EvaluatedValue::buildOffsetValue(const_builder, query_map_builder, loc, type)`
    /// (`ExpressionEvaluatorUtils.cpp:148`) — materialises the value, appending whatever constants
    /// and per-unit mappings it needs to the two blocks those builders are anchored in.
    fn build_offset_value(
        &mut self,
        value: EvaluatedValueId,
        ty: ScalarTy,
        builders: &mut Builders<'_>,
    ) -> Val;
}

/// THE TWO INSERTION POINTS THE MANAGER CARRIES, PLUS THE SSA MINTER — `OpBuilder &const_builder_,
/// &query_map_builder_` (`:538`), both anchored in the block CONTAINING the program unit rather than
/// in the block being rewritten.
///
/// ⛔ A PARAMETER, NOT A FIELD: a `&mut Vec<Op>` held by the manager cannot coexist with the
/// `getDefiningOp` lookups its own methods make over the same program — the convention this module
/// tree already states for `LoopRollingManager`.
pub(crate) struct Builders<'a> {
    /// Where `const_builder_` inserts.
    pub(crate) consts: &'a mut Vec<Op>,
    /// Where `query_map_builder_` inserts.
    pub(crate) query_maps: &'a mut Vec<Op>,
    /// Every created op's result comes from here.
    pub(crate) values: &'a mut Values,
}

/// ONE CANDIDATE NODE, AS MUCH OF `dcc::CFGSCondNode` AS THE WALK READS.
///
/// ⛔ `Analyses/CFGSSentientLevelConditionalTree.*` IS OUT OF CAMPAIGN SCOPE — `OUTSIDE-UNITS.tsv`
/// names it for this very unit — so every predicate the tree computes arrives ANSWERED rather than
/// guessed at, the same reading [`ExpressionEvaluator`] and [`Leaf`] already make of `Analyses/`.
/// ⭐ `isLeaf()`, `isThenNode()` AND `isElseNode()` (`:743-745`) NEED NO FIELD: in this projection a
/// then/else node IS a `Branch` and a leaf is a `Branch` with no child, so being a [`CondNode`] at all
/// is what those three tests select.
pub(crate) struct Candidate {
    /// The node — `getLhs()`, `getRhsVal()`, `getOperation()` and its two branches.
    pub(crate) node: CondNode,
    /// `n->isValid() && n->isCandidateForProcessing()` (`:742-743`).
    pub(crate) valid_candidate_for_processing: bool,
    /// `n->isCandidateForSimplification()` (`:752`, `:843`).
    pub(crate) candidate_for_simplification: bool,
    /// `n->isAncestorLHSInTheSubtree()` (`:758`).
    pub(crate) ancestor_lhs_in_the_subtree: bool,
    /// `n->doesTableExceedMaxSize()` (`:766`).
    pub(crate) table_exceeds_max_size: bool,
    /// `n->getLhsInSubtree()` (`:792`), in the tree's own order.
    pub(crate) lhs_in_subtree: Vec<Val>,
    /// `n->getResultTypes()` (`:833`) — EMPTY IS A CONDITIONAL THAT YIELDS NOTHING, which still gets a
    /// table so its dead branches can be removed.
    pub(crate) result_types: Vec<ScalarTy>,
    /// `n->getLhs().getType()` (`:839`) — the table's predicate type.
    pub(crate) predicate_type: ScalarTy,
    /// `n->getNumLeavesInSubtree()`, which [`TreeSeam`] carries into e555.
    pub(crate) num_leaves_in_subtree: i64,
}

/// WHAT ONE `tree.compute()` ANSWERS FOR ONE PROGRAM UNIT.
pub(crate) struct ComputedTree {
    /// The nodes `CFGSCondNode::walk<kPreOrder>(tree.getRoot(), ..)` visits (`:855`), OUTERMOST FIRST.
    /// ⭐ EMPTY IS `tree.empty()` (`:729`).
    pub(crate) candidates: Vec<Candidate>,
    /// `tree.getLhsToForOpOrNull()` (`:837`) — ⭐ AND `getOrCreateTupleForIV`'s answer (`:794`, `:820`),
    /// which is the map that call populates.
    pub(crate) lhs_to_for_op_or_null: BTreeMap<Val, LoopInfo>,
}

/// THE OUT-OF-SCOPE CONDITIONAL TREE AS A SEAM — a struct of closures for the same reason
/// [`TreeSeam`] and [`LoopCloning`] are: the caller supplies the analysis's answers, and nothing here
/// computes them.
pub(crate) struct ConditionalTree<'a> {
    /// `CFGSSentientLevelConditionalTree tree(*unit_op); tree.compute(); tree.reorderTreeIfLinear();`
    /// plus the pre-order walk of `tree.getRoot()` (`:727-736`, `:855`).
    pub(crate) compute: &'a mut dyn FnMut(&[Op]) -> ComputedTree,
    /// `tree.replaceIfOpWithOneBranch(if_op, region)` (`:875`, `:881`) — inlines that region where the
    /// conditional was, leaving the now-unread `sentient.if` for this pass to erase.
    pub(crate) replace_if_op_with_one_branch: &'a mut dyn FnMut(&mut Vec<Op>, &OpPath, IfRegion),
    /// `tree.compute(); tree.removeDuplicateConditionals()` on a tree REBUILT after the last round
    /// (`:893-896`).
    pub(crate) remove_duplicate_conditionals: &'a mut dyn FnMut(&mut Vec<Op>),
    /// `n->setNoCandidatesForAllSubtreeNodes()`, on the candidate at that path.
    pub(crate) set_no_candidates: &'a mut dyn FnMut(&OpPath),
    /// `simplifySubtree(n, ..)`, answering how many leaves that root has left.
    pub(crate) simplify_subtree: &'a mut dyn FnMut(&mut Vec<Op>, &OpPath) -> i64,
    /// `tree.isForOpToAvoidNewIterArgs(for_op)`.
    pub(crate) for_ops_to_avoid_new_iter_args: &'a [OpPath],
    /// `createSentientForOpWithAdditionalIterArgs` (e393, not ported) — see [`LoopCloning`].
    pub(crate) clone_for_op: &'a mut dyn FnMut(&mut Vec<Op>, &OpPath, &[(Val, Val)]) -> OpPath,
}

/// Replaces: e594_runOnOperation
///
/// THE PASS ENTRY (`:713-913`): per program unit, rounds of — build the conditional tree, populate one
/// candidate subtree's table of yielded values, simplify the patterns found in it, then erase every
/// conditional the round marked dead — until a round clones no outer loop.
///
/// ⚠️ TRAP: `Mark::Processed` OUTLIVES A ROUND (set at `:849`, removed only at `:906`), which is what
/// stops a later round redoing a node. ⚠️ DIVERGENCE: the walk is restarted over a recomputed tree
/// whenever a rewrite moved the positions the remaining candidates are named by — see the body.
pub(crate) fn run_on_operation<A: Arch, M: Model, W: Workload, E: ExpressionEvaluator>(
    program: &mut Program<A, M, W>,
    values: &mut Values,
    query_maps: &mut Vec<Op>,
    tree: &mut ConditionalTree<'_>,
    fresh_evaluator: &mut impl FnMut() -> E,
) {
    if DISABLE_THIS_PASS {
        return;
    }
    // `OpBuilder const_builder(unit_op); const_builder.setInsertionPointToStart(unit_op->getBlock())`
    // (`:718-719`) — the block the `dataflow.program_unit` SITS IN, which is the preamble here.
    // `query_map_builder` is `getLocalOrGlobalRegion(n->getOperation())` (`:835-836`), which CAN be a
    // region of the body being rewritten; one `&mut` cannot be handed out twice, so it is the
    // caller's — exactly what `analyses::OffsetSites::query_maps` states for the same pair.
    let Program {
        preamble, units, ..
    } = program;
    'units: for unit in units.iter_mut() {
        // The op markers of `:42-50` — pass-local, and they outlive the manager instance that set them.
        let mut marks = Marks::default();
        // `int encoding_if_op_count_ = 0` (`:710`) — THE PASS'S counter, borrowed by every instance.
        let mut encoding_if_op_count = EncodingIfOpCount(0);
        // `do { .. } while (for_loop_cloned);` (`:722-892`).
        loop {
            // `ExpressionEvaluator evaluator;` (`:724`) — A FRESH ONE PER ROUND: the last round
            // rewrote the body, so anything it memoised about a value is stale.
            let mut evaluator = fresh_evaluator();
            let mut for_loop_cloned = false;
            let mut computed = (tree.compute)(&unit.body);
            if computed.candidates.is_empty() {
                // `if (tree.empty()) return;` (`:729`) returns from the WALK LAMBDA, so it abandons
                // this unit's remaining rounds, its cleanup AND its flag removal.
                continue 'units;
            }
            // `CFGSCondNode::walk<kPreOrder>(tree.getRoot(), computeTableAndSimplify)` (`:855`).
            'walk: loop {
                let mut rewritten = false;
                {
                    let ComputedTree {
                        candidates,
                        lhs_to_for_op_or_null,
                    } = &mut computed;
                    for candidate in candidates.iter_mut() {
                        // `if (for_loop_cloned || !n->isValid() || !n->isCandidateForProcessing() ||
                        // n->isLeaf() || n->isThenNode() || n->isElseNode()) return nullptr;`
                        // (`:741-745`). ⭐ `dcc::CFGSCondNode *parent = n->getParentNode();` (`:746`)
                        // IS DEAD IN THE REFERENCE — assigned and never read.
                        if for_loop_cloned || !candidate.valid_candidate_for_processing {
                            continue;
                        }
                        // `!EnableDeadBranchRemoval && !n->isCandidateForSimplification()` (`:750-752`).
                        if !ENABLE_DEAD_BRANCH_REMOVAL && !candidate.candidate_for_simplification {
                            continue;
                        }
                        // `n->isAncestorLHSInTheSubtree()` (`:758`) — the table would be missing what
                        // the ancestor predicates fixed, so it may be wrong.
                        if candidate.ancestor_lhs_in_the_subtree {
                            continue;
                        }
                        let at = candidate.node.op.clone();
                        // `hasAttr(PROCESSED_SIMPLIFICATIONS) || hasAttr(TO_DELETE)` (`:760-764`).
                        if marks.has(Mark::Processed, at.path())
                            || marks.has(Mark::ToDelete, at.path())
                        {
                            continue;
                        }
                        // `n->doesTableExceedMaxSize()` (`:766-769`).
                        if candidate.table_exceeds_max_size {
                            continue;
                        }
                        // `ivs_to_values_and_filters` (`:786-789`), a `std::map` ordered by
                        // `compare_values` (`:771-781`): the IVs' loops outermost first.
                        // ⛔ THE COMPARATOR IS THE KEY, so two IVs whose loops sit at the SAME nest
                        // level compare EQUIVALENT — `insert` (`:812-813`) drops the second while
                        // `table_size` (`:815`) still multiplies by its trip count, leaving a table
                        // with holes that `is_full` then refuses.
                        let mut dims: Vec<(i64, Val, i64)> = Vec::new();
                        let mut table_size: i64 = 1;
                        let mut one_iteration = false;
                        for &lhs in &candidate.lhs_in_subtree {
                            // `DT_CHECK_MSG(num_iterations > 0, ..)` (`:794-796`) over
                            // `getOrCreateTupleForIV`, whose miss is `{nullptr,-1,-1,-1,-1}`.
                            let Some(info) = lhs_to_for_op_or_null
                                .get(&lhs)
                                .filter(|info| info.iterations > 0)
                            else {
                                panic!(
                                    "DT_CHECK(num_iterations > 0) Expect a nonzero number of \
                                     iterations of loop. \
                                     (`CFGSimplificationSentientLevel.cpp:794`): {lhs:?}"
                                )
                            };
                            // `if (num_iterations == 1) return nullptr;` (`:797-806`) — one possible
                            // value of the IV, and a conditional this pass simplifies has to have at
                            // least two branches.
                            if info.iterations == 1 {
                                one_iteration = true;
                                break;
                            }
                            let level = nest_level(&unit.body, info.for_op.as_ref());
                            if !dims.iter().any(|&(seen, _, _)| seen == level) {
                                dims.push((level, lhs, info.iterations));
                            }
                            table_size *= info.iterations;
                        }
                        if one_iteration {
                            continue;
                        }
                        dims.sort_by_key(|&(level, _, _)| level);
                        // `ivs_dimensions_multipliers` (`:817-830`): each dimension's multiplier is the
                        // product of the trip counts INSIDE it.
                        let mut multiplier = table_size;
                        let mut ivs_dimensions_multipliers = Vec::new();
                        for &(_, iv, iterations) in &dims {
                            multiplier /= iterations;
                            ivs_dimensions_multipliers.push(IvDim {
                                iv,
                                dimension: iterations,
                                multiplier,
                            });
                        }
                        // `n->getResultTypes().empty() ? const_builder.getIndexType() : ..front()`
                        // (`:832-834`).
                        let table_type = candidate
                            .result_types
                            .first()
                            .copied()
                            .unwrap_or(ScalarTy::Index);
                        let mut table = Table::new(
                            TableSize(usize::try_from(table_size).unwrap_or_default()),
                            candidate.predicate_type,
                            table_type,
                        );
                        let mut ivs = IvValuesAndFilters::of(
                            dims.iter().map(|&(_, iv, iterations)| (iv, iterations)),
                        );
                        // `PatternSimplificationManager instance(..)` (`:837-842`).
                        let mut instance = PatternSimplificationManager {
                            lhs_to_for_op_or_null: lhs_to_for_op_or_null.clone(),
                            ivs_dimensions_multipliers: ivs_dimensions_multipliers.clone(),
                            marks: core::mem::take(&mut marks),
                            encoding_if_op_count,
                            ..PatternSimplificationManager::default()
                        };
                        // `instance.populateTable(n, ivs_to_values_and_filters)` (`:843-847`), whose
                        // leaf handler is e431 writing into the table this instance then owns.
                        {
                            let mut sink = LeafSink {
                                ivs_dimensions_multipliers: &ivs_dimensions_multipliers,
                                lhs_to_for_op_or_null,
                                table: &mut table,
                            };
                            instance.populate_table(
                                &mut candidate.node,
                                &mut ivs,
                                &mut |leaf, ivs| {
                                    sink.process_leaf(leaf, ivs, 0, &mut Vec::new(), &mut evaluator)
                                },
                            );
                        }
                        instance.table = Some(table);
                        // `n->getOperation()->setAttr(PROCESSED_SIMPLIFICATIONS, i32 1)` (`:849-850`) —
                        // BEFORE any simplification, and it outlives the round.
                        instance.marks.set(Mark::Processed, at.path());
                        // `if (n->isCandidateForSimplification()) if (instance.hasFullTable())` (`:852`)
                        // — `hasFullTable()` IS `table_->isFull()` (`:1093`), and a table with a hole
                        // means the predicates did not cover the IVs' whole range.
                        if candidate.candidate_for_simplification
                            && instance.table.as_ref().is_some_and(Table::is_full)
                        {
                            let before = unit.body.clone();
                            let mut set_no_candidates = || (tree.set_no_candidates)(&at);
                            let mut seam = TreeSeam {
                                num_leaves_in_subtree: candidate.num_leaves_in_subtree,
                                set_no_candidates: &mut set_no_candidates,
                                simplify_subtree: &mut *tree.simplify_subtree,
                            };
                            let mut cloning = LoopCloning {
                                for_ops_to_avoid_new_iter_args: tree.for_ops_to_avoid_new_iter_args,
                                clone: &mut *tree.clone_for_op,
                            };
                            let mut builders = Builders {
                                consts: &mut *preamble,
                                query_maps: &mut *query_maps,
                                values: &mut *values,
                            };
                            for_loop_cloned = instance.find_patterns_and_simplify(
                                &mut unit.body,
                                &candidate.node,
                                &mut seam,
                                &mut cloning,
                                &mut builders,
                                &mut evaluator,
                            );
                            rewritten = unit.body != before;
                        }
                        marks = core::mem::take(&mut instance.marks);
                        encoding_if_op_count = instance.encoding_if_op_count;
                        if rewritten {
                            break;
                        }
                    }
                }
                if for_loop_cloned || !rewritten {
                    break 'walk;
                }
                // ⚠️ DIVERGENCE: THE WALK IS RESTARTED OVER A RECOMPUTED TREE. The reference's nodes
                // are `Operation *` and survive a sibling insertion; a candidate here is a POSITION,
                // and e555's rewrite moved every position after the one it rewrote. `Mark::Processed`
                // is already set on every node this walk handled, so the restarted walk skips exactly
                // those and reaches the same set of nodes the reference's single walk does.
                computed = (tree.compute)(&unit.body);
                if computed.candidates.is_empty() {
                    break 'walk;
                }
            }
            // `unit_op.walk<PreOrder>` collecting every marked `sentient.if` (`:857-863`), then
            // erasing them IN REVERSE so that no position still to be erased has moved (`:865-889`).
            let doomed: Vec<OpPath> = if_paths(&unit.body)
                .into_iter()
                .filter(|at| {
                    marks.has(Mark::ToDelete, at.path())
                        || marks.has(Mark::DeadThenBranch, at.path())
                        || marks.has(Mark::DeadElseBranch, at.path())
                })
                .collect();
            for at in doomed.iter().rev() {
                if marks.has(Mark::DeadThenBranch, at.path()) {
                    (tree.replace_if_op_with_one_branch)(&mut unit.body, at, IfRegion::Else);
                } else if marks.has(Mark::DeadElseBranch, at.path()) {
                    (tree.replace_if_op_with_one_branch)(&mut unit.body, at, IfRegion::Then);
                }
                erase_conditional(&mut unit.body, at);
            }
            if !for_loop_cloned {
                break;
            }
        }
        // `tree.compute(); tree.removeDuplicateConditionals()` (`:893-896`) — duplicate side-effect-free
        // siblings, on a tree rebuilt after the last round.
        (tree.remove_duplicate_conditionals)(&mut unit.body);
        // `if_op->removeAttr(PROCESSED_SIMPLIFICATIONS); if_op->removeAttr(TABLE_TOO_LARGE)`
        // (`:904-908`). ⭐ THE MARKS ARE PASS-LOCAL (see [`Mark`]) and go with the unit, so this states
        // the reference's effect rather than being the only thing that discharges it.
        for at in if_paths(&unit.body) {
            marks.take(Mark::Processed, at.path());
            marks.take(Mark::TableTooLarge, at.path());
        }
    }
}

/// Every `sentient.if` of a body, PRE-ORDER — `unit_op.walk<WalkOrder::PreOrder>(..)` (`:858`, `:905`).
fn if_paths(body: &[Op]) -> Vec<OpPath> {
    fn collect(body: &[Op], at: &mut Vec<(u32, u32)>, region: u32, out: &mut Vec<OpPath>) {
        for (index, op) in body.iter().enumerate() {
            at.push((region, index as u32));
            if matches!(op, Op::Sentient(sentient::Op::If { .. })) {
                out.push(OpPath::at(at));
            }
            for (sub_region, sub) in ir::regions_ref(op).into_iter().enumerate() {
                collect(sub, at, sub_region as u32, out);
            }
            at.pop();
        }
    }
    let mut out = Vec::new();
    collect(body, &mut Vec::new(), 0, &mut out);
    out
}

/// `dcc::utils::getLoopNestLevel<sentient::ForOp>(for_op)` (`Utils/Utils.cpp:211-220`) over a POSITION:
/// how many `sentient.for`s enclose the loop, the outermost answering 0.
///
/// ⛔ THE `None`s ARE ALL ONE STOP — `DT_CHECK_MSG(loop_op && isa<LoopTy>(loop_op), "expected valid
/// loop")` (`:213`), which is what a null tuple entry or a path naming something else reaches.
fn nest_level(body: &[Op], for_op: Option<&OpPath>) -> i64 {
    fn level_of(body: &[Op], steps: &[(u32, u32)]) -> Option<i64> {
        let mut scope = body;
        let mut level = 0;
        for (depth, &(_, index)) in steps.iter().enumerate() {
            let op = scope.get(index as usize)?;
            let is_loop = matches!(op, Op::Sentient(sentient::Op::For { .. }));
            if depth + 1 == steps.len() {
                return is_loop.then_some(level);
            }
            // ⭐ ONLY A `sentient.for` COUNTS: `getParentOfType<LoopTy>` skips every other enclosing
            // op, `affine.for` and `uniform.uniformize_regions` included.
            level += i64::from(is_loop);
            scope = ir::regions_ref(op)
                .into_iter()
                .nth(steps[depth + 1].0 as usize)?;
        }
        None
    }
    match for_op.and_then(|for_op| level_of(body, for_op.path())) {
        Some(level) => level,
        None => panic!(
            "DT_CHECK(loop_op && isa<LoopTy>(loop_op)) expected valid loop \
             (`dcc/src/Utils/Utils.cpp:213`): {for_op:?} names no `sentient.for` of this unit"
        ),
    }
}

/// `DT_CHECK_MSG(if_op.use_empty(), ..); if_op->erase()` (`:885-888`) — the conditional whose surviving
/// branch has just been inlined in its place.
fn erase_conditional(body: &mut Vec<Op>, at: &OpPath) {
    let read = op_ref(body, at.path())
        .map(ir::results)
        .is_some_and(|results| results.iter().any(|result| is_read(body, *result)));
    if read {
        panic!(
            "DT_CHECK(if_op.use_empty()) Expect conditional marked for deletion to have no uses. \
             (`CFGSimplificationSentientLevel.cpp:885`): {at:?} is still read"
        )
    }
    if let Some((block, index)) = block_of(body, at)
        && index < block.len()
    {
        block.remove(index);
    }
}

/// The op those steps name — `Operation *` as the position this module's [`OpPath`] states.
fn op_ref<'a>(body: &'a [Op], steps: &[(u32, u32)]) -> Option<&'a Op> {
    let mut scope = body;
    for (depth, &(_, index)) in steps.iter().enumerate() {
        let op = scope.get(index as usize)?;
        if depth + 1 == steps.len() {
            return Some(op);
        }
        scope = ir::regions_ref(op)
            .into_iter()
            .nth(steps[depth + 1].0 as usize)?;
    }
    None
}

/// The block an op sits in and its position in it — `Operation::getBlock()`, for an erase.
fn block_of<'a>(body: &'a mut Vec<Op>, at: &OpPath) -> Option<(&'a mut Vec<Op>, usize)> {
    let (&(region, index), under) = at.path().split_last()?;
    if under.is_empty() {
        return Some((body, index as usize));
    }
    let mut scope: &mut Vec<Op> = body;
    for (depth, &(_, step)) in under.iter().enumerate() {
        let op = scope.get_mut(step as usize)?;
        let next = if depth + 1 == under.len() {
            region as usize
        } else {
            under[depth + 1].0 as usize
        };
        scope = ir::regions_mut(op).into_iter().nth(next)?;
    }
    Some((scope, index as usize))
}

/// `Value::use_empty()` for one result of the op being erased.
fn is_read(body: &[Op], val: Val) -> bool {
    body.iter().any(|op| {
        ir::operands(op).contains(&val)
            || ir::regions_ref(op)
                .into_iter()
                .any(|region| is_read(region, val))
    })
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::ty::ScalarTy;

    fn ev(id: u32) -> EvaluatedValueId {
        EvaluatedValueId::new(id)
    }

    /// THE OUT-OF-SCOPE EVALUATOR AS AN ARENA THAT NEVER REUSES AN ENTRY — which is the point:
    /// `getConstant(0)` and a sum that happens to BE zero get different ids, so `equal` answers about
    /// contents and [`EvaluatedValueId`]'s own `==` about identity.
    ///
    /// ⛔ TEST ONLY, like `pattern_simplification_manager`'s `FakeEvaluator`: `Analyses/` is outside
    /// this campaign and no implementation of [`ExpressionEvaluator`] may ship in it.
    #[derive(Debug, Default)]
    struct StatedEvaluator {
        arena: Vec<i64>,
    }

    impl StatedEvaluator {
        fn intern(&mut self, value: i64) -> EvaluatedValueId {
            self.arena.push(value);
            EvaluatedValueId::new(
                u32::try_from(self.arena.len() - 1).expect("a fixture arena is small"),
            )
        }

        fn value_of(&self, id: EvaluatedValueId) -> i64 {
            self.arena[id.0 as usize]
        }
    }

    impl ExpressionEvaluator for StatedEvaluator {
        fn get_constant(&mut self, value: i64) -> EvaluatedValueId {
            self.intern(value)
        }

        fn evaluate_sum(
            &mut self,
            lhs: EvaluatedValueId,
            rhs: EvaluatedValueId,
        ) -> EvaluatedValueId {
            let sum = self.value_of(lhs) + self.value_of(rhs);
            self.intern(sum)
        }

        fn evaluate_value(&mut self, value: Val) -> EvaluatedValueId {
            self.intern(i64::from(value.0))
        }

        fn equal(&self, lhs: EvaluatedValueId, rhs: EvaluatedValueId) -> bool {
            self.value_of(lhs) == self.value_of(rhs)
        }

        fn evaluate_sub(
            &mut self,
            _lhs: EvaluatedValueId,
            _rhs: EvaluatedValueId,
        ) -> EvaluatedValueId {
            todo!("no unit of this batch subtracts two evaluated values")
        }

        fn evaluate_multiply_by_const(
            &mut self,
            _value: EvaluatedValueId,
            _factor: Entries,
        ) -> EvaluatedValueId {
            todo!("no unit of this batch scales an evaluated value")
        }

        fn build_offset_value(
            &mut self,
            _value: EvaluatedValueId,
            _ty: ScalarTy,
            _builders: &mut Builders<'_>,
        ) -> Val {
            todo!("no unit of this batch materialises an offset")
        }
    }

    fn stride(step: i64) -> IntervalStride {
        IntervalStride::new(NonZeroI64::new(step).expect("a test interval stride is never zero"))
    }

    /// `lb_` is `ev(1)` and `stride_` is `ev(2)`; `shifted_lb_` starts null, as the constructor leaves it.
    fn monotone(length: i64, marker: i64, step: i64) -> Sequence {
        Sequence::new(
            ev(1),
            ev(2),
            Entries(length),
            IntervalMarker(marker),
            stride(step),
            SequenceKind::MonotoneSequence,
        )
    }

    /// e023 — `length_ += increment`.
    #[test]
    fn increment_length_adds_entries() {
        let mut seq = monotone(3, 10, 2);
        seq.increment_length(Entries(2));
        assert_eq!(seq.length, Entries(5));
    }

    /// e024 — the marker moves by `interval_stride_ * increment` (10 + 4*2), and the sequence
    /// `setIntervalMarker(std::nullopt)` emptied hands out no witness to move at all.
    #[test]
    fn increment_interval_marker_steps_by_the_interval_stride() {
        let mut seq = monotone(3, 10, 4);
        seq.marked_mut()
            .expect("constructed with a marker")
            .increment_interval_marker(Entries(2));
        assert_eq!(seq.interval_marker, Some(IntervalMarker(18)));

        seq.interval_marker = None;
        assert!(seq.marked_mut().is_none());
    }

    /// e025 — the stride comes back signed, since its sign picks `sle` over `sge` downstream.
    #[test]
    fn interval_stride_keeps_its_sign() {
        assert_eq!(monotone(1, 0, -3).interval_stride().get().get(), -3);
    }

    /// e026 — the kind flips and the two lower bounds swap, so the sequence now yields the SHIFTED
    /// one; afterwards it is `kDefaultValue` and offers no witness to change again.
    #[test]
    fn change_to_default_value_swaps_the_lower_bounds() {
        let mut seq = monotone(4, 12, 3);
        seq.shifted_lb = Some(ev(9));

        seq.as_monotone_mut()
            .expect("constructed monotone")
            .change_to_default_value();

        assert_eq!(seq.kind, SequenceKind::DefaultValue);
        assert_eq!(seq.lb, Some(ev(9)));
        assert_eq!(seq.shifted_lb, Some(ev(1)));
        assert!(seq.as_monotone_mut().is_none());
    }

    /// e027 — the one branch that reaches no `EvaluatedValue`: a dummy with neither bound. Note the
    /// missing space after `"interval marker:"`, which the full-sequence branch has.
    #[test]
    fn print_writes_the_dummy_form_when_neither_bound_is_known() {
        let mut seq = monotone(3, 7, 1);
        seq.lb = None;
        seq.stride = None;

        let mut out = String::new();
        seq.print(&mut out);
        assert_eq!(out, "Dummy sequence with interval marker:7\n");
    }

    /// e028 — the constructor's `getConstant(0)` id, then whatever `setPrevVal` last stored.
    #[test]
    fn prev_val_tracks_the_last_value_seen_in_the_slice() {
        let mut slice = TableSlice::new(TableIndex(0), IndexStride(1), Entries(4), ev(0), true);
        assert_eq!(slice.prev_val(), ev(0));

        slice.set_prev_val(ev(6));
        assert_eq!(slice.prev_val(), ev(6));
    }

    /// e029 — a leaf that yields results has a value; the dummy entry a result-less conditional gets
    /// (so its dead branches can still be removed) has none.
    #[test]
    fn ev_is_absent_for_a_leaf_that_yields_nothing() {
        assert_eq!(TableEntry::new(Some(ev(5)), None).ev(), Some(ev(5)));
        assert_eq!(TableEntry::new(None, None).ev(), None);
    }

    /// e030 — full means EVERY index has an entry. A result-less entry still counts: the reference
    /// tests the `TableEntry *`, not its `ev_`.
    #[test]
    fn is_full_only_when_every_index_has_an_entry() {
        let mut table = Table::new(TableSize(2), ScalarTy::Int(1), ScalarTy::Index);
        assert!(!table.is_full());

        table.entries[0] = Some(TableEntry::new(None, None));
        assert!(!table.is_full());

        table.entries[1] = Some(TableEntry::new(Some(ev(1)), None));
        assert!(table.is_full());
    }

    /// e286 — a length-0 pad is dropped, a monotone run of 3 from 10 by 5 becomes `10` and `15` in the
    /// new list while its LAST element (20) is ABSORBED by the following default sequence, whose length
    /// grows to 3 and whose marker does not move. The markers run FORWARDS from `30 - 2*(3-1)`.
    #[test]
    fn e286_expands_a_monotone_run_and_absorbs_its_last_element_leftward() {
        let mut evaluator = StatedEvaluator::default();
        let ten = evaluator.get_constant(10);
        let five = evaluator.get_constant(5);
        let twenty = evaluator.get_constant(20);

        let mut slice = TableSlice::new(TableIndex(0), IndexStride(1), Entries(6), ten, true);
        slice.sequences = vec![
            Sequence::new(
                ten,
                five,
                Entries(0),
                IntervalMarker(0),
                stride(2),
                SequenceKind::MonotoneSequence,
            ),
            Sequence::new(
                ten,
                five,
                Entries(3),
                IntervalMarker(30),
                stride(2),
                SequenceKind::MonotoneSequence,
            ),
            Sequence::new(
                twenty,
                five,
                Entries(2),
                IntervalMarker(40),
                stride(2),
                SequenceKind::DefaultValue,
            ),
        ];

        slice.recompute_as_default_vals(&mut evaluator);

        assert_eq!(slice.sequences.len(), 3);
        assert!(
            slice
                .sequences
                .iter()
                .all(|seq| seq.kind == SequenceKind::DefaultValue)
        );
        // The two expanded elements: values 10 and 15, markers 26 then 28.
        let lbs: Vec<i64> = slice
            .sequences
            .iter()
            .map(|seq| evaluator.value_of(seq.lb.expect("every new sequence has an lb")))
            .collect();
        assert_eq!(lbs, vec![10, 15, 20]);
        assert_eq!(
            slice
                .sequences
                .iter()
                .map(|seq| seq.interval_marker)
                .collect::<Vec<_>>(),
            vec![
                Some(IntervalMarker(26)),
                Some(IntervalMarker(28)),
                Some(IntervalMarker(40)),
            ]
        );
        assert_eq!(
            slice
                .sequences
                .iter()
                .map(|seq| seq.length)
                .collect::<Vec<_>>(),
            vec![Entries(1), Entries(1), Entries(3)]
        );
    }

    /// e287 — a filled index answers with its entry, an index no leaf covered answers `None`, and an
    /// index past the end is the reference's `DT_CHECK`.
    #[test]
    fn e287_answers_a_hole_with_none_and_stops_past_the_end() {
        let mut table = Table::new(TableSize(2), ScalarTy::Int(1), ScalarTy::Index);
        table.entries[1] = Some(TableEntry::new(Some(ev(3)), None));

        assert_eq!(table.table_entry_at_idx(TableIndex(0)), None);
        assert_eq!(
            table
                .table_entry_at_idx(TableIndex(1))
                .and_then(TableEntry::ev),
            Some(ev(3))
        );
    }

    /// e287 — the bounds check, which stops rather than answering.
    #[test]
    #[should_panic(expected = "Expect valid index to table.")]
    fn e287_stops_on_an_index_past_the_end() {
        let _ = Table::new(TableSize(2), ScalarTy::Int(1), ScalarTy::Index)
            .table_entry_at_idx(TableIndex(2));
    }

    /// e288 — the entry holds the evaluation of the leaf's FIRST result and the leaf's block; a leaf
    /// yielding nothing gets the reference's dummy value and still records its block, so its dead
    /// branch can be removed.
    #[test]
    fn e288_evaluates_the_first_result_and_keeps_the_leaf_block() {
        let mut evaluator = StatedEvaluator::default();
        let mut table = Table::new(TableSize(2), ScalarTy::Int(1), ScalarTy::Index);
        let block = BlockPath {
            if_op: OpPath::at(&[(0, 0)]),
            region: IfRegion::Then,
        };

        table.create_table_entry_at_idx(
            TableIndex(0),
            &Leaf {
                results: vec![Val(7), Val(8)],
                block: Some(block.clone()),
            },
            &mut evaluator,
        );
        table.create_table_entry_at_idx(
            TableIndex(1),
            &Leaf {
                results: Vec::new(),
                block: Some(block.clone()),
            },
            &mut evaluator,
        );

        let first = table
            .table_entry_at_idx(TableIndex(0))
            .expect("just created");
        // `evaluateValue(getResults().front())` — the FIRST result, so `Val(7)` and not `Val(8)`.
        assert_eq!(
            evaluator.value_of(first.ev().expect("the leaf yields results")),
            7
        );
        assert_eq!(first.block, Some(block.clone()));
        let second = table
            .table_entry_at_idx(TableIndex(1))
            .expect("just created");
        assert_eq!(second.ev(), None);
        assert_eq!(second.block, Some(block));
        assert!(table.is_full());
    }

    /// e594 — one round over two candidates: the 2-iteration one is analysed, its dead then-branch is
    /// marked by `populate_table` and the round's cleanup erases the conditional through the tree
    /// seam; the 1-iteration one bails at `:797` and is left exactly as it was.
    #[test]
    fn e594_erases_the_marked_conditional_and_leaves_the_one_iteration_candidate_alone() {
        use crate::arch::Dd2;
        use crate::generated::OpFunc;
        use crate::islands::dataflow_ir::{GroupId, OpIndex, ProgramName, Units};
        use crate::islands::sentient::{ProgramUnit, ProgramUnits};
        use crate::units::DfirUnit;
        use pattern_simplification_manager::Branch;

        #[derive(Debug, Clone, PartialEq, Eq)]
        struct AnyModel;
        impl Model for AnyModel {
            const QUERY_HEADS: u32 = 32;
            const KV_HEADS: u32 = 8;
            const HEAD_DIM: u32 = 64;
            const HIDDEN: u32 = 2048;
            const LAYERS: u32 = 40;
            const FFN: u32 = 8192;
            const VOCAB: u32 = 49152;
        }

        #[derive(Debug, Clone, PartialEq, Eq)]
        struct AnyRung;
        impl Workload for AnyRung {
            const ROWS: u32 = 1;
            const ACTIVE_CAP: u32 = 64;
        }

        fn conditional(lhs: Val, rhs: Val) -> Op {
            Op::Sentient(sentient::Op::If {
                predicate: sentient::CmpPredicate::Eq,
                lhs,
                rhs,
                yielded: Vec::new(),
                dbg_name: None,
                then_body: Vec::new(),
                else_body: Vec::new(),
            })
        }

        fn loop_over(iv: Val, bound: Val, body: Vec<Op>) -> Op {
            Op::Sentient(sentient::Op::For {
                iv,
                bound,
                bound_reg: None,
                carried: Vec::new(),
                dbg_name: None,
                body,
            })
        }

        let (iv_a, iv_b, rhs) = (Val(10), Val(11), Val(12));
        // `sentient.for %iv { sentient.if %iv == %rhs }`, twice.
        let mut program: Program<Dd2, AnyModel, AnyRung> = Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            preamble: Vec::new(),
            units: ProgramUnits::of(
                ProgramUnit {
                    on: Units::one(DfirUnit::Lxlu, Val(0)),
                    precision: None,
                    body: vec![
                        loop_over(iv_a, Val(1), vec![conditional(iv_a, rhs)]),
                        loop_over(iv_b, Val(2), vec![conditional(iv_b, rhs)]),
                    ],
                    arch: core::marker::PhantomData,
                },
                Vec::new(),
            ),
            bound: core::marker::PhantomData,
        };
        let at_a = OpPath::at(&[(0, 0), (0, 0)]);
        let at_b = OpPath::at(&[(0, 1), (0, 0)]);
        let info = |for_op: &[(u32, u32)], iterations| LoopInfo {
            for_op: Some(OpPath::at(for_op)),
            lb: 0,
            ub: iterations,
            step: NonZeroI64::new(1).expect("one is not zero"),
            iterations,
        };
        // ⭐ A's then-branch ARRIVES DEAD, which is `populate_table`'s `setAttr(DEAD_THEN_BRANCH)` at
        // `:947` and the only thing the cleanup needs; B's loop runs ONCE, which is the `:797` bail.
        let candidate = |lhs: Val, at: &OpPath, dead_then: bool| Candidate {
            node: CondNode {
                lhs,
                rhs_val: 0,
                op: at.clone(),
                then_node: Branch {
                    dead: dead_then,
                    ..Branch::default()
                },
                else_node: Branch::default(),
            },
            valid_candidate_for_processing: true,
            candidate_for_simplification: true,
            ancestor_lhs_in_the_subtree: false,
            table_exceeds_max_size: false,
            lhs_in_subtree: vec![lhs],
            result_types: Vec::new(),
            predicate_type: ScalarTy::Int(1),
            num_leaves_in_subtree: 2,
        };
        let computed = || ComputedTree {
            candidates: vec![candidate(iv_b, &at_b, true), candidate(iv_a, &at_a, false)],
            lhs_to_for_op_or_null: [(iv_a, info(&[(0, 0)], 1)), (iv_b, info(&[(0, 1)], 2))]
                .into_iter()
                .collect(),
        };
        let mut replaced: Vec<(OpPath, IfRegion)> = Vec::new();
        let mut deduplicated = 0_u32;
        let mut tree = ConditionalTree {
            compute: &mut |_| computed(),
            replace_if_op_with_one_branch: &mut |_, at, region| {
                replaced.push((at.clone(), region));
            },
            remove_duplicate_conditionals: &mut |_| deduplicated += 1,
            set_no_candidates: &mut |_| panic!("no candidate is simplified here"),
            simplify_subtree: &mut |_, _| panic!("no subtree is simplified here"),
            for_ops_to_avoid_new_iter_args: &[],
            clone_for_op: &mut |_, at, _| at.clone(),
        };
        let mut values = Values::default();
        let mut query_maps = Vec::new();
        run_on_operation(
            &mut program,
            &mut values,
            &mut query_maps,
            &mut tree,
            &mut StatedEvaluator::default,
        );

        // B's dead then-branch was replaced by its else region and the conditional erased; A's
        // 1-iteration loop was never touched, so its conditional is still there.
        assert_eq!(replaced, vec![(at_b, IfRegion::Else)]);
        assert_eq!(deduplicated, 1);
        let body = &program.units.iter_mut().next().expect("one unit").body;
        assert_eq!(if_paths(body), vec![at_a]);
    }
}
