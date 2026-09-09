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
use core::ops::{AddAssign, Mul};

pub(crate) mod pattern_simplification_manager;

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
}

/// ONE LEAF'S ENTRY IN THE TABLE — `class TableEntry` (`:456`).
///
/// ⛔ `Block *block_` (`:460`) IS NOT DECLARED YET, and neither is `getBlock` (`:479`, an excluded
/// field accessor). It is `leaf->getBlock()` on a `dcc::CFGSCondNode` (`:472`), which arrives with
/// `e288_createTableEntryAtIdx`; declare it there rather than guess an island spelling for it now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TableEntry {
    /// `ev_` (`:458`) — `None` is the reference's dummy `nullptr` for a conditional with no results,
    /// which still gets a table so dead branches can be removed (`:463-467`).
    ev: Option<EvaluatedValueId>,
}

impl TableEntry {
    /// `TableEntry(dcc::CFGSCondNode *leaf, ExpressionEvaluator &evaluator)` (`:468-472`), whose
    /// `ev_` is `nullptr` when the leaf yields nothing and `evaluator.evaluateValue(..)` otherwise —
    /// both the leaf and the evaluator being out of campaign scope, that choice is the caller's.
    pub(crate) const fn new(ev: Option<EvaluatedValueId>) -> TableEntry {
        TableEntry { ev }
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
/// ⛔ `const Type predicate_type_` (`:491`) AND `Type entry_type_` (`:494`) ARE NOT DECLARED YET.
/// They are MLIR `Type`s reached through `getPredicateType`/`getTableEntryType` (`:522-524`, both
/// excluded field accessors) and first consumed by `e292_generateEncodingTypes`; declare them there
/// rather than guess which of the island's type spellings they are. `evaluator_` (`:497`) is a
/// parameter, as on [`Sequence`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Table {
    /// `table_entries_` (`:488`) — `std::vector<TableEntry *>(table_size, nullptr)` (`:506`), owning:
    /// `~Table` deletes every element (`:508-510`), so `Drop` is the destructor.
    entries: Vec<Option<TableEntry>>,
}

impl Table {
    /// `Table(table_size, predicate_type, entry_type, evaluator)` (`:500-506`) — every entry starts
    /// null.
    pub(crate) fn new(table_size: TableSize) -> Table {
        Table {
            entries: vec![None; table_size.0],
        }
    }

    /// Replaces: e030_isFull
    ///
    /// Whether every index of the table has an entry (`:527-531`) — a table with a hole means the
    /// nested conditional did not cover every tuple of IV values, so it is not simplifiable as one.
    #[must_use]
    pub(crate) fn is_full(&self) -> bool {
        self.entries.iter().all(Option::is_some)
    }
}

// crustify:todo: e286_recomputeAsDefaultVals
//   authority : dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:396  (58 body lines, level 1)
//   original  : void recomputeAsDefaultVals()
//   calls     : e023_incrementLength, e025_getIntervalStride

// crustify:todo: e287_getTableEntryAtIdx
//   authority : dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:512  (4 body lines, level 1)
//   original  : TableEntry *getTableEntryAtIdx(dcc::WidestIntType idx) const
//   calls     : e252_size

// crustify:todo: e288_createTableEntryAtIdx
//   authority : dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:517  (4 body lines, level 1)
//   original  : void createTableEntryAtIdx(dcc::WidestIntType idx, dcc::CFGSCondNode *leaf)
//   calls     : e252_size

// crustify:todo: e594_runOnOperation
//   authority : dcc/src/Transform/Sentient/CFGSimplificationSentientLevel.cpp:713  (201 body lines, level 5)
//   original  : void CFGSimplificationSentientLevelPass::runOnOperation()
//   calls     : e027_print, e278_isValid, e422_insert, e515_getOrCreateTupleForIV, e555_findPatternsAndSimplify

#[cfg(test)]
mod unit_tests {
    use super::*;

    fn ev(id: u32) -> EvaluatedValueId {
        EvaluatedValueId::new(id)
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
        assert_eq!(TableEntry::new(Some(ev(5))).ev(), Some(ev(5)));
        assert_eq!(TableEntry::new(None).ev(), None);
    }

    /// e030 — full means EVERY index has an entry. A result-less entry still counts: the reference
    /// tests the `TableEntry *`, not its `ev_`.
    #[test]
    fn is_full_only_when_every_index_has_an_entry() {
        let mut table = Table::new(TableSize(2));
        assert!(!table.is_full());

        table.entries[0] = Some(TableEntry::new(None));
        assert!(!table.is_full());

        table.entries[1] = Some(TableEntry::new(Some(ev(1))));
        assert!(table.is_full());
    }
}
