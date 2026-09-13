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

// ⛔ THE PASS IS NOT WIRED INTO THE PIPELINE YET, so every item below is reachable only from this
// file's own tests until `e594_runOnOperation` (level 5) lands and something calls it. CI runs clippy
// with `-D warnings`, so without this the first ported leaf of a 32-unit module fails the gate.
// ⭐ REMOVE THIS WITH e594: at that point an unused item here is a real defect again.
#![allow(dead_code)]

use core::num::NonZeroI64;
use core::ops::{AddAssign, Mul, Sub, SubAssign};

pub(crate) mod pattern_simplification_manager;

use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::{Op, Val};
use pattern_simplification_manager::OpPath;

/// `EnableNonZeroStrideSeqSimplifications` (`:76-80`) — *"Allow replacing fixed non-zero stride
/// sequences of values yielded by conditionals with iterator arguments."*, `cl::init(false)`.
///
/// ⛔ A `dcc-opt` COMMAND-LINE FLAG, NOT A PROGRAM PROPERTY, and this crate has no flags — the same
/// reading [`super::loop_splitting_and_unrolling::is_ok_to_unroll`] made of `DisableLoopUnroll`.
/// ⭐ A NAMED `const` RATHER THAN A FOLDED LITERAL BECAUSE IT DECIDES WHICH SEQUENCE KINDS EXIST: at
/// `false` `padTableSlice` pads to the slice's own maximum instead of 3 (`:2422-2423`) and never asks
/// for a monotone slot (`:2437-2440`), so both branches fold away rather than going unread.
pub(crate) const ENABLE_NON_ZERO_STRIDE_SEQ_SIMPLIFICATIONS: bool = false;

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

// crustify:todo: e594_runOnOperation
//   authority : dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:713  (201 body lines, level 5)
//   original  : void CFGSimplificationSentientLevelPass::runOnOperation()
//   calls     : e027_print, e278_isValid, e422_insert, e515_getOrCreateTupleForIV, e555_findPatternsAndSimplify

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
}
