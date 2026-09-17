// SPDX-License-Identifier: Apache-2.0
//
// ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ CRUSTIFY ddc / L3-SCHEDULER CAMPAIGN — READ THIS BEFORE YOU FILL AN ANCHOR IN THIS FILE.     ║
// ║ Campaign statement: crustify-ddc/TASK.md   ·   worklist: crustify-ddc/UNITS.tsv              ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
//
// 1. THE AUTHORITY IS THE C++ TREE, NOT THE EXTRACT.
//       /Users/nickm/git/deeptools-src/<file>:<line>     (revision a0d29abbed)
//    Every citation below resolves against that revision. `crustify-ddc/cpp/{l3,ddc,ddl,dcg}.cpp`
//    say WHICH functions are in scope and IN WHAT ORDER; their bodies were verified byte-identical
//    to the authority (382/382, 1,069,773 bytes, 2 of 2 negative controls DETECTED), so either
//    may be read — but the authority file is the one that carries the surrounding declarations you
//    will need. ⛔ The other local deeptools checkout is a DIFFERENT revision; the pod is not
//    reachable from here.
//
// 2. WHAT THIS STAGE IS. `dbo-opt` is the binary scratchy shells out to; its per-program pipeline
//    runs `runDdc` for every program (dbo/src/Transforms/sdsc_bundle/RunSchedulerOnSdsc.cpp:145).
//    ⭐ `dbo/src/Utils/sdsc_bundle/SchedulerStages.cpp` — 60 lines — IS THE SPEC FOR THIS WHOLE
//    JOB. READ IT FIRST. Four stages:
//      1.  sbf::doCoreletSplitSdsc     ⛔ EXCLUDED: SchedulerStages.cpp:25 returns early unless
//          numCoreletsPerCore == 2, and scratchy emits numCoreletsUsed_ = 1 in all 313 sampled
//          SuperDSCs. See crustify-ddc/EXCLUSIONS.tsv.
//      2a. L3DlOpsScheduler(dscGlobal, memTrackers, {executionStep}, verbose).run(sdsc)
//      2b. ddc::Ddc(dscGlobal, ..).run_v1(sdsc)      entry: ddc/ddcv1.cpp:3695
//      3.  DcgManager::runDcgForDlOpsStandalone(sdsc)   dcg/dcg_manager/dcg_manager.cpp:449
//          ⛔ THAT branch, NOT `runDcg`: SchedulerStages.cpp:53-57 picks it whenever `dscs_` is
//          non-empty, always true for scratchy's input. ⚠️ Its body delegates almost entirely to
//          `dcg_fe/pcfg_gen/` and `dcg_be/`, both OUT of scope — `todo!` NAMING the missing
//          translator is correct there; ⛔ do NOT invent a PCFG.
//    `runDdc` raises when DDC finds no mapping ("Scheduler failed to find a suitable op mapping"),
//    so the mapping is not optional.
//
// 3. WHY IT MATTERS. `ddc.run_v1` is what PLACES ADDRESSES, and the L3 scheduler's `run` commits
//    LX allocations then calls `fillAllocationStartAddrAndOffset` — literally "Set start address,
//    offset in allocations" (dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:8000). Our emitted views
//    have printed `start_address = 0` where the reference states a placed base; the backend has
//    refused with `Register initialization out of boundary`; and `src/reginit.rs` (1,569 lines, on
//    the integ branch) HAND-COMPUTES placement from ddc/ddcv1.cpp:132-360. This campaign replaces
//    that guesswork with the real thing.
//
// 4. PORTED MEANS THE WHOLE FUNCTION INCLUDING ITS EFFECT. For this stage the effect is WHAT IS
//    WRITTEN INTO THE `SuperDsc`: addresses, mappings, symbol definitions, fold state, schedule
//    steps. A hand attempt on bridge 2 extracted each function's decision rule into a documented
//    predicate, omitted the part that changed the IR, and reported it done — nothing called any of
//    it. A PREDICATE IS NOT A PORT. Droppable: only the mechanism for REACHING operands (walking
//    uses, memoising, positioning a builder). ⛔ If a TYPE cannot express a result, EXTEND THE
//    TYPE. Deciding a function is unnecessary is NOT the porter's call.
//
// 5. HARD CAPS — MEASURED, AND THEY BIND. Bridge 2's first 144 functions cost 43 h wall clock and
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
// 6. THIS IS A PURE-LOGIC PORT WITH NO C ANYWHERE. Whatever generic C-to-Rust conventions say:
//    ❌ no bindgen/allowlist/-sys, ❌ no `ffi::`/`extern "C"`, ❌ no `CRUSTIFY_<FILE>` switch,
//    ❌ no `Foo`/`FooRef`/`FooMut` triple, ❌ no `unsafe`, ❌ no sanitizers, ❌ no C-vs-Rust harness.
//
// 7. CRATE RULES BIND YOU — `crates/compiler/deeptools/CLAUDE.md`, read it in full.
//    🛑 NEVER RUNTIME REFUSE: no `Result`, no `Err(`, no `.ok_or`, no `assert!`, no `debug_assert!`.
//    NEWTYPES, NEVER RAW SCALARS — an address, an offset, a core index and an element count must
//    not be interchangeable `u64`s; transposing two must be E0308. A closed set is an `enum`, never
//    a string. `Arch`/`Model`/`Workload` flow through. `todo!` NAMING UNPORTED WORK IS ALLOWED here
//    — and ⛔ never substitute a stand-in op, or a fabricated address, to dodge one. A fabricated
//    placement to avoid a stop is worse than the stop.
//
// 8. ⚠️ THE L3 SCHEDULER MAY OR MAY NOT APPLY TO US — DO NOT DECIDE THIS, PORT IT. Scratchy states
//    its own schedule (our SuperDSC writes `coreIdToDscSchedule`) and L3DlOpsScheduler.cpp:415-425
//    READS that field. That question is the USER'S, not the porter's. Measured while scoping: the
//    field occurs in that file ONLY as a read, never a write, so it is an INPUT to both stages —
//    what the L3 scheduler PRODUCES is the dsc2 schedule tree, the LX buffer type, the committed
//    LX allocations and their start addresses.
//
// 9. ANCHORS: each `// crustify:todo: eNNN_name` below is one scheduled unit. Replace it with the
//    ported item carrying the doc anchor `/// Replaces: eNNN_name`. ⛔ NEVER DELETE AN ANCHOR YOU
//    DID NOT PORT — on bridge 2 a deleted anchor was indistinguishable from a finished unit and the
//    campaign reported DONE having silently lost 149 of 384 functions, the biggest in its span.
//
// 10. GATE: `cargo check -p deeptools` and `cargo test -p deeptools`. ⛔ NEVER run the workspace or
//     the acceptance build in an agent worktree — ~6 GB of target/ each, and this host is tight.
//
// 11. ⭐⭐ THE ACCEPTANCE CRITERION — WHAT THESE STAGES PRODUCE. They mutate the `SuperDsc` IN
//     PLACE, and the observable result is that THE `ScheduleNode` TREE GAINS ITS LOOP, TRANSFER,
//     SYNC AND CONDITION NODES. Scratchy's SuperDSC today has only ALLOCATE nodes (one construction
//     site: `crates/targets/spyre/src/lower_subtile_tape_to_superdsc.rs:5127`) plus a flat
//     `computeOp_` list — which matches torch-spyre's own `generate_sdsc`, and is exactly why
//     `dxp_standalone` works and the Rust `sdscToDataflowIR` port yields nothing.
//     The REPRESENTATIVE MINTING SITE to model is `ddc/ddl/ddl_conversion.cpp:1065`: it mints a
//     `dsc2::LoopNode`, takes its dims from the DDL, names it `loop_ds<num>_ds<den>`, and registers
//     it in `ddlInterface.loop_labels_`.
//     ⛔ A PORT THAT DOCUMENTS THE SCHEDULING DECISION WITHOUT ADDING NODES TO THAT TREE IS NOT A
//     PORT.
//
// 12. ⭐⭐ THE TARGET VOCABULARY IS ALREADY TYPED — a wrong shape must be a COMPILE ERROR, not a
//     judgement call. Emit into these EXISTING types; do not invent any:
//       `src/bridges/superdsc_to_dataflow_ir/driver.rs:467`        `Statement`
//       `driver.rs:1152`                                           `Scheduled`
//       `driver.rs:1756-1790`                                      `Viewed` / `Viewing`
//       `driver.rs:1539`                                           `ScheduleView::roots`
//       `driver.rs:1788`                                           `Dsc`
//     The single function it all plugs into is `Schedule::roots` in
//     `crates/targets/spyre/src/lower_superdsc_to_dataflow_ir.rs` — committed, compiles, and
//     currently yields nothing. Both the component and the DSC are already in hand there.
//
// 13. ALREADY PORTED, DO NOT DUPLICATE — all four in
//     `src/bridges/superdsc_to_dataflow_ir/shape_constraints.rs`, following the same
//     `/// Replaces: eNNN_name` convention so cross-referencing works:
//       `e001_checkConstraints` (ddc/ddcv1.cpp:792)   `e002_createDataConnectMetadata` (:3283)
//       `e041_getStickSizes`    `e071_getCumulativeStickSizes`  (both `DesignSpaceConfig` methods
//       in `dsc/dsc2.cpp`, OUTSIDE this campaign's file list — your units CALL them.)
//     `checkConstraints` is a LAMBDA inside `Ddc::exploreAssignDataStages`, so porting that unit
//     means CALLING the existing port, not writing a second constraint checker. Units carrying such
//     a constraint have a ⛔ NOTE on the anchor itself.

//! `ddc/ddc_metadata.h` — 9 of the campaign's 382 units (dependency level(s) [0]).
//!
//! | unit | entry | level | lines | class | authority path:line |
//! |---|---|---|---|---|---|
//! | `e096_updateMin` | 096 | 0 | 3 | `Constraints` | `ddc/ddc_metadata.h:40` |
//! | `e097_updateMax` | 097 | 0 | 3 | `Constraints` | `ddc/ddc_metadata.h:43` |
//! | `e098_updateValues` | 098 | 0 | 3 | `Constraints` | `ddc/ddc_metadata.h:46` |
//! | `e099_dump` | 099 | 0 | 23 | `Constraints` | `ddc/ddc_metadata.h:49` |
//! | `e100_insertProducer` | 100 | 0 | 5 | `DataConnect` | `ddc/ddc_metadata.h:147` |
//! | `e101_insertConsumer` | 101 | 0 | 5 | `DataConnect` | `ddc/ddc_metadata.h:153` |
//! | `e102_print` | 102 | 0 | 11 | `DataConnect` | `ddc/ddc_metadata.h:166` |
//! | `e103_getLoops` | 103 | 0 | 13 | `DataConnect` | `ddc/ddc_metadata.h:179` |
//! | `e104_clear` | 104 | 0 | 4 | `Metadata` | `ddc/ddc_metadata.h:225` |

use super::fold::{AllocId, BlockId, ConstIdx, NodeId, PadType};
use crate::arch::Elements;
use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{
    AbsoluteMin, Constraint, ConstraintKind, DimSet, NoEpilogueDimKind, PrimaryDim,
};
use crate::generated::{DataConnect, MaxUnroll, OpaqueReg, RegName, Strategy};
use crate::schedule::dsc2::{LdsIdx, NodeName};
use std::collections::{BTreeMap, BTreeSet};
use sys_arch_spec::arch_enums::{OpFunc, SenComponent};

// ═══ `Constraints` — THE FIELDS `dump` OBSERVES, AND THE THREE UPDATERS ══════════════════════════

/// WHICH LOOP EXTENT A DATASTAGE CONSTRAINT NAMES — `MetaDimKind` (`dsc/dims.h:59`) less its `Count`
/// terminator, which is the field's UNSET sentinel (`ddc/ddc_metadata.h:36`) and so is absence here.
///
/// ⛔ THE LABELS ARE `stringToMetaDimKind`'s (`dsc/dims.cpp:50-57`), which `e184_setMetaDimKind`
/// parses back. `Count`'s own label there is `"undefined"` — a string [`Constraint::dump`] never
/// prints, because it prints `NOT_SET` for that case instead (`ddc/ddc_metadata.h:51-56`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MetaDimKind {
    /// `unpadded`.
    Unpadded,
    /// `padded`.
    Padded,
    /// `pad_front`.
    PadFront,
    /// `pad_back`.
    PadBack,
    /// `pad_valid`.
    PadValid,
    /// `window`.
    WindowDim,
    /// `stride`.
    Stride,
    /// `dilation`.
    Dilation,
}

impl MetaDimKind {
    /// `EnumsConversion::metaDimKindToString.at(kind)` (`dsc/dims.cpp:50-57`).
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            MetaDimKind::Unpadded => "unpadded",
            MetaDimKind::Padded => "padded",
            MetaDimKind::PadFront => "pad_front",
            MetaDimKind::PadBack => "pad_back",
            MetaDimKind::PadValid => "pad_valid",
            MetaDimKind::WindowDim => "window",
            MetaDimKind::Stride => "stride",
            MetaDimKind::Dilation => "dilation",
        }
    }
}

impl NoEpilogueDimKind {
    /// The `MetaDimKind` a no-epilogue kind IS — the three of nine that arm accepts.
    #[must_use]
    pub fn dim_kind(self) -> MetaDimKind {
        match self {
            NoEpilogueDimKind::Unpadded => MetaDimKind::Unpadded,
            NoEpilogueDimKind::Padded => MetaDimKind::Padded,
            NoEpilogueDimKind::WindowDim => MetaDimKind::WindowDim,
        }
    }
}

/// `mustBeMultiple_` AND `loopDimKind_` ON A RELATIVE CONSTRAINT (`ddc/ddc_metadata.h:34-36`).
///
/// ⛔⛔ `loopDimKind_` OUTLIVES `mustBeMultiple_`, WHICH IS WHY `Off` STILL CARRIES A KIND:
/// `ddc/ddc_transformation.cpp:968-1035` sets `Unpadded` on constraints that are not multiples, and
/// `dump` prints the kind unconditionally — a fold that dropped it could not state that constraint.
///
/// ⛔ THE TWO `DT_ERROR`s ARE THE RELATIVE ARM'S ALONE: `mustBeMultiple_` with no kind
/// (`ddc/ddcv1.cpp:857-859`) and with a kind outside `{Unpadded, Padded, WindowDim}` (`:881-886`) are
/// both raised where the constraint is CHECKED against a reference stage — so on the ABSOLUTE arm,
/// whose kind is never read, [`Self::Unkinded`] is the state entry 307 legitimately stores: it keys
/// the constraint absolute (`:1119`) and then sets `mustBeMultiple_` with no kind (`:1222-1226`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopMultiple {
    /// `mustBeMultiple_` false, with whatever `loopDimKind_` holds — [`None`] for `Count`.
    Off(Option<MetaDimKind>),
    /// `mustBeMultiple_` with `loopDimKind_ == Count`, which only an ABSOLUTE constraint may hold.
    Unkinded,
    /// `mustBeMultiple_`, with the kind its no-epilogue divisibility test reads.
    NoEpilogue(NoEpilogueDimKind),
}

impl LoopMultiple {
    /// `mustBeMultiple_ = must_be_multiple; loopDimKind_ = dim_kind` as one value, and [`None`] where
    /// a must-be-multiple kind is one of the six the no-epilogue check has no arm for.
    #[must_use]
    pub fn of(must_be_multiple: bool, dim_kind: Option<MetaDimKind>) -> Option<Self> {
        if !must_be_multiple {
            return Some(Self::Off(dim_kind));
        }
        match dim_kind {
            None => Some(Self::Unkinded),
            Some(MetaDimKind::Unpadded) => Some(Self::NoEpilogue(NoEpilogueDimKind::Unpadded)),
            Some(MetaDimKind::Padded) => Some(Self::NoEpilogue(NoEpilogueDimKind::Padded)),
            Some(MetaDimKind::WindowDim) => Some(Self::NoEpilogue(NoEpilogueDimKind::WindowDim)),
            Some(_) => None,
        }
    }

    /// `mustBeMultiple_`.
    #[must_use]
    pub const fn must_be_multiple(self) -> bool {
        !matches!(self, Self::Off(_))
    }

    /// `loopDimKind_`, [`None`] for its `Count` sentinel.
    #[must_use]
    pub fn dim_kind(self) -> Option<MetaDimKind> {
        match self {
            Self::Off(dim_kind) => dim_kind,
            Self::Unkinded => None,
            Self::NoEpilogue(kind) => Some(kind.dim_kind()),
        }
    }
}

impl<S> Constraint<'_, S> {
    /// Replaces: e096_updateMin
    ///
    /// TIGHTENS THE LOWER BOUND — `min_ = min_ ? std::max(*min_, newVal) : newVal`
    /// (`ddc/ddc_metadata.h:40`).
    ///
    /// ⛔ `max` TIGHTENS A *MIN*: the stricter of two lower bounds is the larger. And on the absolute
    /// arm the bound doubles as the multiple ([`AbsoluteMin`]), so raising it raises both.
    pub fn update_min(&mut self, new_val: f32) {
        match &mut self.kind {
            ConstraintKind::Absolute { min, .. } => {
                *min = match *min {
                    AbsoluteMin::Unset => AbsoluteMin::Bound(new_val),
                    AbsoluteMin::Bound(min) => AbsoluteMin::Bound(stricter_min(min, new_val)),
                    AbsoluteMin::Multiple(min) => AbsoluteMin::Multiple(stricter_min(min, new_val)),
                };
            }
            ConstraintKind::Relative { min, .. } => {
                *min = Some(min.map_or(new_val, |min| stricter_min(min, new_val)));
            }
        }
    }

    /// Replaces: e097_updateMax
    ///
    /// TIGHTENS THE UPPER BOUND — `max_ = max_ ? std::min(*max_, newVal) : newVal`
    /// (`ddc/ddc_metadata.h:43`). ⛔ `min` TIGHTENS A *MAX*.
    pub fn update_max(&mut self, new_val: f32) {
        self.max = Some(self.max.map_or(new_val, |max| stricter_max(max, new_val)));
    }

    /// Replaces: e098_updateValues
    ///
    /// INTERSECTS THE PERMITTED RATIOS — `values_ = values_ ? set_intersect(*values_, newVals) :
    /// newVals` (`ddc/ddc_metadata.h:46`, `util/utils.h:112`).
    ///
    /// ⛔⛔ THE FIRST CALL ADOPTS, IT DOES NOT INTERSECT — an ABSENT `values_` is unconstrained, where
    /// an intersection may leave it ENGAGED AND EMPTY, which no size satisfies. Two different states,
    /// and only the second is a contradiction.
    pub fn update_values(&mut self, new_vals: &[f32]) {
        intersect_values(&mut self.values, new_vals);
    }

    /// Replaces: e099_dump
    ///
    /// THE CONSTRAINT AS `std::cerr` STATES IT — `ddc/ddc_metadata.h:49`, one trailing-newline line,
    /// returned rather than written because the caller owns the stream.
    ///
    /// ⛔ AN UNSET `loopDimKind_` PRINTS `NOT_SET`, not the conversion map's `"undefined"` (`:51-56`),
    /// and an unset bound prints `-inf` / `inf` — the bound it is ABSENT of, not one it holds.
    /// ⛔ AND `values_` IS A `std::set<float>` (`:38`): ascending, once each, observable ONLY here.
    #[must_use]
    pub fn dump(&self) -> String {
        let (must_be_multiple, dim_kind, min) = match self.kind {
            ConstraintKind::Absolute { dim_kind, min, .. } => (
                matches!(min, AbsoluteMin::Multiple(_)),
                dim_kind,
                match min {
                    AbsoluteMin::Unset => None,
                    AbsoluteMin::Bound(min) | AbsoluteMin::Multiple(min) => Some(min),
                },
            ),
            ConstraintKind::Relative { multiple, min, .. } => {
                (multiple.must_be_multiple(), multiple.dim_kind(), min)
            }
        };

        let mut out = String::from("mustBeMultiple_= ");
        out.push_str(if must_be_multiple { "T " } else { "F " });
        out.push_str("loopDimKind_= ");
        out.push_str(dim_kind.map_or("NOT_SET", MetaDimKind::label));
        out.push_str(" , min_= ");
        match min {
            Some(min) => out.push_str(&format!("{} ", stream_float(min))),
            None => out.push_str("-inf "),
        }
        out.push_str(", max_= ");
        match self.max {
            Some(max) => out.push_str(&format!("{} ", stream_float(max))),
            None => out.push_str("inf "),
        }
        out.push_str(", values_= {");
        if let Some(values) = &self.values {
            let mut ascending = values.clone();
            ascending.sort_by(f32::total_cmp);
            ascending.dedup();
            for value in ascending {
                out.push_str(&format!("{} ", stream_float(value)));
            }
        }
        out.push_str("}\n");
        out
    }
}

/// `std::max` ON TWO LOWER BOUNDS — spelled as the comparison it is, because `f32::max` prefers the
/// non-NaN operand where `std::max(a, b)` returns `a` whenever the comparison is false.
pub(crate) fn stricter_min(held: f32, new_val: f32) -> f32 {
    if held < new_val { new_val } else { held }
}

/// `std::min` ON TWO UPPER BOUNDS — likewise `(b < a) ? b : a`.
pub(crate) fn stricter_max(held: f32, new_val: f32) -> f32 {
    if new_val < held { new_val } else { held }
}

/// `values_ = values_ ? set_intersect(*values_, newVals) : newVals` — the SHARED body of entry 098,
/// because [`StoredConstraint`] holds the same `std::set<float>` and takes the same update.
pub(crate) fn intersect_values(values: &mut Option<Vec<f32>>, new_vals: &[f32]) {
    let mut incoming = new_vals.to_vec();
    incoming.sort_by(f32::total_cmp);
    incoming.dedup();
    *values = Some(match values {
        Some(values) => values
            .iter()
            .copied()
            .filter(|value| incoming.contains(value))
            .collect(),
        None => incoming,
    });
}

/// `operator<<(std::ostream&, float)` AT THE DEFAULT PRECISION 6 — `%g`: six significant digits,
/// trailing zeros dropped, scientific form outside `[1e-4, 1e6)` with a signed two-digit exponent.
///
/// ⛔ RUST'S `{}` IS NOT THIS. It prints the shortest decimal that round-trips, so a reciprocal like
/// `1.0 / 3.0` (`ddc/ddcv1.cpp:759-765`) would come out `0.33333334` where the reference says
/// `0.333333`.
fn stream_float(value: f32) -> String {
    if value.is_nan() {
        return String::from("nan");
    }
    if value.is_infinite() {
        return String::from(if value < 0.0 { "-inf" } else { "inf" });
    }
    // Six significant digits, whose exponent is the one `%g` chooses the form by — already rounded,
    // so a value that carries to the next power of ten picks the form its printed digits deserve.
    let scientific = format!("{value:.5e}");
    let (mantissa, exponent) = scientific
        .split_once('e')
        .unwrap_or((scientific.as_str(), "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    if (-4..6).contains(&exponent) {
        let decimals = usize::try_from(5 - exponent).unwrap_or(0);
        drop_trailing_zeros(format!("{value:.decimals$}"))
    } else {
        let sign = if exponent < 0 { '-' } else { '+' };
        let digits = drop_trailing_zeros(mantissa.to_string());
        format!("{digits}e{sign}{:02}", exponent.abs())
    }
}

/// `%g`'s trailing-zero removal, and the point with them.
fn drop_trailing_zeros(mut digits: String) -> String {
    if digits.contains('.') {
        while digits.ends_with('0') {
            digits.pop();
        }
        if digits.ends_with('.') {
            digits.pop();
        }
    }
    digits
}

// ═══ `DataConnect` — THE TWO ENDS OF ONE `data_connect=` ════════════════════════════════════════

/// WHERE A SCHEDULE NODE SITS IN THE WALK — its identity in the producer and consumer lists.
///
/// ⭐ POSITIONAL, BECAUSE THE REFERENCE'S IS A POINTER. `insertProducer(node)` stores the
/// `ScheduleNode*` and deduplicates on it (`ddc/ddc_metadata.h:147-157`); an index into the DFS order
/// is that identity without the walk, which is the mechanism for reaching the nodes rather than the
/// census.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct NodeIndex(pub usize);

/// ONE OF THOSE NODES THAT IS A `dsc2::LoopNode` — what `getOwnerLoop` hands back
/// (`dsc/dsc2.cpp:1896`), still an index because a loop's own owner walk continues from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct LoopIndex(pub NodeIndex);

impl LoopIndex {
    /// The loop as a plain node.
    #[must_use]
    pub fn node(self) -> NodeIndex {
        self.0
    }
}

/// THE NAME A SCHEDULE NODE PRINTS UNDER — [`NodeName`] resolved from the census' positional
/// identity, because the census projection carries no `name_` of its own.
pub trait NodeNames {
    /// `node->name_`.
    fn name(&self, node: NodeIndex) -> &NodeName;
}

/// THE INNERMOST LOOP ENCLOSING A SCHEDULE NODE — `getOwnerLoop` walks `prev_` until `nodeType_ ==
/// LOOP` (`dsc/dsc2.cpp:1896`), and [`None`] where that walk runs off the top of the tree.
pub trait OwnerLoops {
    /// `node->getOwnerLoop()`.
    fn owner_loop(&self, node: NodeIndex) -> Option<LoopIndex>;
}

/// ONE DATA CONNECT'S TWO ENDS — `ddc::Metadata::DataConnect`'s `producers_` and `consumers_`
/// (`ddc/ddc_metadata.h:143-145`), each deduplicated and in first-touch order as `insertProducer` and
/// `insertConsumer` keep them.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Ends {
    producers: Vec<NodeIndex>,
    consumers: Vec<NodeIndex>,
}

impl Ends {
    /// The nodes that write this connect — NON-EMPTY in every completed census, which
    /// `DT_ERROR("Illegal DDL: data_connect ... does not have any producer.")`
    /// (`ddc/ddcv1.cpp:3322-3325`) enforces after the walk and entry 002 answers as `NoProducer`.
    #[must_use]
    pub fn producers(&self) -> &[NodeIndex] {
        &self.producers
    }

    /// The nodes that read it, which may be none: a connect nothing consumes is legal.
    #[must_use]
    pub fn consumers(&self) -> &[NodeIndex] {
        &self.consumers
    }

    /// Replaces: e100_insertProducer
    ///
    /// RECORDS A WRITER OF THIS CONNECT, ONCE — `if (!is_any_of(node, producers_))
    /// producers_.push_back(node)` (`ddc/ddc_metadata.h:147`).
    pub fn insert_producer(&mut self, node: NodeIndex) {
        if !self.producers.contains(&node) {
            self.producers.push(node);
        }
    }

    /// Replaces: e101_insertConsumer
    ///
    /// RECORDS A READER OF THIS CONNECT, ONCE — `ddc/ddc_metadata.h:153`, the same test on the other
    /// list.
    pub fn insert_consumer(&mut self, node: NodeIndex) {
        if !self.consumers.contains(&node) {
            self.consumers.push(node);
        }
    }

    /// Replaces: e102_print
    ///
    /// BOTH LISTS AS THE REFERENCE WRITES THEM — `ddc/ddc_metadata.h:166`, returned rather than
    /// streamed because the caller owns `outs`.
    ///
    /// ⛔ CONSUMERS COME FIRST, and every name is PRECEDED by a space, so an empty list prints `[]`
    /// and a one-name list prints `[ name]`.
    #[must_use]
    pub fn print<N: NodeNames + ?Sized>(&self, names: &N) -> String {
        let mut out = String::from(" Consumers= [");
        for consumer in &self.consumers {
            out.push(' ');
            out.push_str(&names.name(*consumer).0);
        }
        out.push_str("] Producers= [");
        for producer in &self.producers {
            out.push(' ');
            out.push_str(&names.name(*producer).0);
        }
        out.push_str("]\n");
        out
    }

    /// Replaces: e103_getLoops
    ///
    /// EVERY LOOP ENCLOSING ANY OF `base_nodes`, ANCESTORS INCLUDED — `ddc/ddc_metadata.h:179`.
    ///
    /// ⛔ THE `break` ON AN ALREADY-RECORDED LOOP LOSES NOTHING: that loop's own ancestors went in
    /// when it did, so the walk it cuts short is one already taken. It is ALSO what terminates a
    /// `prev_` cycle.
    ///
    /// ⭐ FIRST-TOUCH ORDER where the reference's `unordered_set` has none, which BOTH its consumers
    /// (`ddc/ddc_transformation_util.cpp:411,446` and `ddc/ddc_transformation.cpp:1514`, all
    /// `.count()` membership tests) cannot tell apart.
    fn loops<T: OwnerLoops + ?Sized>(base_nodes: &[NodeIndex], tree: &T) -> Vec<LoopIndex> {
        let mut loops: Vec<LoopIndex> = Vec::new();
        for base_node in base_nodes {
            let mut owner = tree.owner_loop(*base_node);
            while let Some(enclosing) = owner {
                if loops.contains(&enclosing) {
                    break;
                }
                loops.push(enclosing);
                owner = tree.owner_loop(enclosing.node());
            }
        }
        loops
    }

    /// `getProducerLoops()` (`ddc/ddc_metadata.h:159`) — a 3-line accessor recorded in
    /// `EXCLUSIONS.tsv`, supplied because `getLoops` is private in the reference too.
    #[must_use]
    pub fn producer_loops<T: OwnerLoops + ?Sized>(&self, tree: &T) -> Vec<LoopIndex> {
        Ends::loops(&self.producers, tree)
    }

    /// `getConsumerLoops()` (`ddc/ddc_metadata.h:162`) — likewise.
    #[must_use]
    pub fn consumer_loops<T: OwnerLoops + ?Sized>(&self, tree: &T) -> Vec<LoopIndex> {
        Ends::loops(&self.consumers, tree)
    }
}
// ⭐ USES FOR ENTRY 104 ARE UNIONED INTO THIS FILE'S TOP BLOCK. Entries 096-103 landed first, so
// their `Constraint`/`Ends`/`MetaDimKind` vocabulary is REUSED below, never restated.

// ════════════════════════════════════════════════════════════════════════════════════════════════
// THE DDC METADATA — the state entry 104 resets (`ddc/ddc_metadata.h:31-239`).
//
// ⭐ THE FIELD LIST *IS* THIS UNIT'S CONTENT. The reference spells the reset
// `this->~Metadata(); new (this) Metadata();` (:225) because `Metadata` carries TWO `const int`
// members (`core_dstgid`, `chunk_dstgid`, :211-212) and therefore has no assignment operator at
// all. Here those two are associated consts rather than state, so the reset is ONE assignment and
// every field a later batch adds is reset by construction.
//
// ⚠️ A SCHEDULE NODE HAS TWO SPELLINGS IN THIS CRATE TODAY: [`NodeIndex`] above (entries 100-103's
// positional census identity) and [`NodeId`] from `super::fold` (entries 086-093'). The maps below
// key by `NodeId`; unifying the two belongs to whichever batch first owns a schedule-tree type.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// A DATASTAGE'S INDEX — a loop's `numId_` / `denId_` and the key of `datastages_`
/// (`ddc/ddc_metadata.h:82`, `ddc/ddcv1.cpp:607-609`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DatastageId(pub u32);

/// ONE STORED `Metadata::Datastage::Constraints` (`ddc/ddc_metadata.h:33`), as the header declares
/// it — [`Constraint`] is the SAME constraint at CHECK time, where entries 096-099 fold `min_` into
/// [`ConstraintKind`] and borrow the reference stage the outer key names here.
///
/// ⛔ RATIOS STAY `f32`, exactly as [`Constraint`] and [`AbsoluteMin`] keep them: these are
/// `size / refSize` ratios compared for exact equality, and a second spelling of one number is what
/// lets two spellings disagree.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredConstraint {
    /// `mustBeMultiple_` and `loopDimKind_` — one field, because [`LoopMultiple`] is what those two
    /// jointly say.
    pub multiple: LoopMultiple,
    /// `min_`.
    pub min: Option<f32>,
    /// `max_`.
    pub max: Option<f32>,
    /// `values_` — absent is UNCONSTRAINED, and an empty set admits no size at all.
    pub values: Option<Vec<f32>>,
    /// `cannotBeSymbolic_`.
    pub cannot_be_symbolic: bool,
}

impl StoredConstraint {
    /// [`Constraint::update_min`] over the stored spelling, where `min_`'s double role as the
    /// multiple is already one field and so needs no [`AbsoluteMin`] fold.
    pub fn update_min(&mut self, new_val: f32) {
        self.min = Some(self.min.map_or(new_val, |min| stricter_min(min, new_val)));
    }

    /// [`Constraint::update_max`] over the stored spelling.
    pub fn update_max(&mut self, new_val: f32) {
        self.max = Some(self.max.map_or(new_val, |max| stricter_max(max, new_val)));
    }

    /// `updateValues` ON THE STORED SIDE (`ddc/ddc_metadata.h:46`) — the same intersect entry 098
    /// performs, reached through [`Datastage::constraint_mut`] by the transformations that WRITE
    /// `constraints_` rather than check it.
    pub fn update_values(&mut self, new_vals: &[f32]) {
        intersect_values(&mut self.values, new_vals);
    }
}

impl Default for StoredConstraint {
    fn default() -> Self {
        Self {
            multiple: LoopMultiple::Off(None),
            min: None,
            max: None,
            values: None,
            cannot_be_symbolic: false,
        }
    }
}

/// ONE DATASTAGE'S EXPLORATION STATE — `Metadata::Datastage` (`ddc/ddc_metadata.h:32`).
#[derive(Debug, Clone, PartialEq)]
pub struct Datastage {
    /// `constraints_` (:73-76) — outer key is the REFERENCE datastage, [`None`] for the `-1`
    /// absolute constraints; the inner `std::map` key is a [`DimSet`], which has no ordering of its
    /// own here.
    ///
    /// ⛔ [`None`] IS THE EMPTY DIM KEY, which `constraints_[refDsId][{}]` states whenever every dim
    /// the op named was dropped or was a direct meta value: [`check_constraints`] can never read one
    /// (it wants `dims.count(dim)`), and only `dump` observes it.
    pub constraints: BTreeMap<Option<DatastageId>, Vec<(Option<DimSet>, StoredConstraint)>>,
    /// `strategyMinimize_` (:77), whose `// false == maximize` comment is this enum.
    pub strategy: Strategy,
    /// `allowEpilogue_` (:78).
    pub allow_epilogue: bool,
    /// `relevantDimsAndNumerator_` (:79) — per dim, the numerator datastage of the loop carrying it.
    pub relevant_dims_and_numerator: BTreeMap<PrimaryDim, DatastageId>,
    /// `nearestNumeratorIdx_` (:80) — `-1` is none (`ddc/ddcv1.cpp:609`, read as a datastage index
    /// at `:940`).
    pub nearest_numerator_idx: Option<DatastageId>,
}

impl Default for Datastage {
    fn default() -> Self {
        Self {
            constraints: BTreeMap::new(),
            strategy: Strategy::Minimize,
            allow_epilogue: false,
            relevant_dims_and_numerator: BTreeMap::new(),
            nearest_numerator_idx: None,
        }
    }
}

impl Datastage {
    /// `constraints_[reference][dims]` — `std::map::operator[]`, which DEFAULT-CONSTRUCTS the
    /// constraint when that dim set is not yet keyed (`ddc/ddc_transformation.cpp:968`).
    ///
    /// ⛔ THE INNER KEY IS MATCHED WHOLE, AND AS A SET: `{x, y}` and `{y, x}` are ONE entry while
    /// `{x}` is another. [`DimSet::of`] sorts and deduplicates, so equality of the key IS equality of
    /// the set the reference's `std::set<DimId>` compares, and [`None`] is the empty key.
    pub fn constraint_mut(
        &mut self,
        reference: Option<DatastageId>,
        dims: Option<DimSet>,
    ) -> &mut StoredConstraint {
        let entries = self.constraints.entry(reference).or_default();
        let at = match entries.iter().position(|(keyed, _)| *keyed == dims) {
            Some(at) => at,
            None => {
                entries.push((dims, StoredConstraint::default()));
                entries.len() - 1
            }
        };
        &mut entries[at].1
    }
}

/// WHICH DESTINATION of a multi-destination transfer — an index into `dstLdsAndLoopOffsets_`
/// (`ddc/ddcv1.cpp:2948-2957`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DestIdx(pub u32);

/// HOW ONE DIM IS READ AND WRITTEN — the reference's `pair<PadType, PadType>` typedef
/// (`ddc/ddc_metadata.h:84`), printed `<first>-to-<second>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TransferAccessPattern {
    /// `.first` — how the source is padded.
    pub from: PadType,
    /// `.second` — how the destination is.
    pub to: PadType,
}

/// ONE TRANSFER'S DDC STATE — `Metadata::DataTransfer` (`ddc/ddc_metadata.h:88`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataTransfer {
    /// `apply_row_offset_src_`.
    pub apply_row_offset_src: bool,
    /// `apply_row_offset_dst_`.
    pub apply_row_offset_dst: bool,
    /// `apply_pe_sfp_split_offset_src_`.
    pub apply_pe_sfp_split_offset_src: bool,
    /// `apply_pe_sfp_split_offset_dest_` — which destinations take the split offset
    /// (`ddc/ddc_transformation_util.cpp:1642`).
    pub apply_pe_sfp_split_offset_dest: Vec<DestIdx>,
    /// `replicated_`. ⛔ NOTHING IN THE AUTHORITY TREE WRITES THIS FIELD: its ONE read is the `else
    /// if` at `ddc/ddcv1.cpp:2913`, so on this revision the whole arm it gates is unreachable — and
    /// that arm opens with a `DT_CHECK_MSG` (`:2914-2916`) nothing could have satisfied.
    pub replicated: bool,
    /// `offset_src_`. ⛔ NOR IS THIS FIELD OR `offset_dest_` EVER WRITTEN, and both are read ONLY
    /// from inside that dead arm: `offset_src_ > 0` (`ddc/ddcv1.cpp:2914`, `:2933`),
    /// `offset_dest_.size() > 0` (`:2915`), and a walk of `offset_dest_` (`:2948`) whose own `> 0`
    /// is on the offset each entry carries (`:2950`).
    pub offset_src: Elements,
    /// `offset_dest_` — per destination, its offset.
    pub offset_dest: BTreeMap<DestIdx, Elements>,
    /// `force_num_elements_`, whose `-1` default is absence — ⛔ AND SO IS `0`: both readers test
    /// `> 0` (`ddc/ddcv1.cpp:458`, `ddc/ddl/ddl_conversion.cpp:3217`) while the one writer
    /// (`ddc/ddl/ddl_conversion.cpp:1237`) copies the DDL's attribute through unchecked.
    pub force_num_elements: Option<Elements>,
    /// `accessPatternPerDim_`, which the DDL's `access_pattern_style=` fills
    /// (`ddc/ddl/ddl_conversion.cpp:1219-1226`).
    pub access_pattern_per_dim: BTreeMap<PrimaryDim, TransferAccessPattern>,
}

impl Default for DataTransfer {
    fn default() -> Self {
        Self {
            apply_row_offset_src: false,
            apply_row_offset_dst: false,
            apply_pe_sfp_split_offset_src: false,
            apply_pe_sfp_split_offset_dest: Vec::new(),
            replicated: false,
            offset_src: Elements(0),
            offset_dest: BTreeMap::new(),
            force_num_elements: None,
            access_pattern_per_dim: BTreeMap::new(),
        }
    }
}

/// WHAT DDC NEWLY ALLOCATED IN ONE MEMORY — `Metadata::Allocation` (`ddc/ddc_metadata.h:121`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Allocation {
    /// `ldsIdxAndAllocNode`.
    pub lds_idx_and_alloc_node: BTreeMap<LdsIdx, AllocId>,
    /// `consIdAndAllocNode` — keyed by the allocation's `constIdx_`
    /// (`ddc/ddc_transformation_util.cpp:1356`), not by a consumer.
    pub cons_id_and_alloc_node: BTreeMap<ConstIdx, AllocId>,
    /// `compAndAllocNode`.
    pub comp_and_alloc_node: BTreeMap<NodeId, AllocId>,
}

/// A TRANSFER NODE THE METADATA OWNS OUTRIGHT — `unique_ptr<TransferNode>`
/// (`ddc/ddc_metadata.h:131`). ⛔ NEITHER `Copy` NOR `Clone`: entry 104 destroys it, and a second
/// handle would be exactly the reference's dangling pointer.
#[derive(Debug, PartialEq, Eq)]
pub struct OwnedTransferNode(pub NodeId);

/// AN ALLOCATE NODE THE METADATA OWNS OUTRIGHT — `unique_ptr<AllocateNode>` (:132), same contract.
#[derive(Debug, PartialEq, Eq)]
pub struct OwnedAllocateNode(pub AllocId);

/// A TRANSFER HELD OUTSIDE THE SCHEDULE TREE — `Metadata::ExternalTransfer`
/// (`ddc/ddc_metadata.h:130`).
#[derive(Debug, PartialEq, Eq)]
pub struct ExternalTransfer {
    /// `transfer_`.
    pub transfer: OwnedTransferNode,
    /// `allocate_`.
    pub allocate: OwnedAllocateNode,
}

/// WHICH END of a transfer's datastreams a data connect is filled through — the source's, or the
/// FIRST destination's when the source is external (`ddc/ddcv1.cpp:2312-2322`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TransferEnd {
    /// `&tn->srcLdsAndLoopOffsets_.dataConnect_`.
    Src,
    /// `&tn->dstLdsAndLoopOffsets_[0].dataConnect_`.
    FirstDst,
}

/// A `data_connect=` SLOT STILL TO BE FILLED — the reference's `std::string*`
/// (`ddc/ddc_metadata.h:138`). ⛔ NON-OWNING: entry 104 drops the locator and destroys nothing,
/// while the DDL writes through it later (`*prefilledIt->second = ...`,
/// `ddc/ddl/ddl_conversion.cpp:939`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DataConnectSlot {
    /// The transfer node holding the slot.
    pub transfer: NodeId,
    /// Which of its ends.
    pub end: TransferEnd,
}

/// A STORAGE AN EXTERNAL TRANSFER MAY BE PREFILLED IN — `is_any_of(storage, LX, PTXRF, L3LUIBR)`,
/// the guard every insert into that map passes (`ddc/ddcv1.cpp:2331`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExternalStorage {
    /// `LX`.
    Lx,
    /// `PTXRF`.
    PtxRf,
    /// `L3LUIBR`.
    L3LuIbr,
}

impl ExternalStorage {
    /// `is_any_of(storage, LX, PTXRF, L3LUIBR)` as the storage it then is, and [`None`] for every
    /// other component — which is entry 309's `DT_ERROR("External transfer node improperly set")`.
    #[must_use]
    pub const fn of(storage: SenComponent) -> Option<Self> {
        match storage {
            SenComponent::Lx => Some(Self::Lx),
            SenComponent::Ptxrf => Some(Self::PtxRf),
            SenComponent::L3luibr => Some(Self::L3LuIbr),
            _ => None,
        }
    }
}

/// A MEMORY DDC ALLOCATES IN — the key of `newAllocations_` (`ddc/ddc_metadata.h:127`), which every
/// writer takes from an allocate node's `component_` (`ddc/ddc_transformation_util.cpp:59`,
/// `ddc/ddl/ddl_conversion.cpp:820`).
///
/// ⛔ NOT `ddc::memories` (`ddc/ddc_metadata.h:20-21`): that set is eight components with no
/// `L0_SCALE`, and its one use tree-wide is `ddc/ddc_fold.cpp:1752`, which is not this map.
/// ⛔ NOT [`crate::generated::Memory`] either: that is the `ddl.allocate` census and has neither
/// `HBM` nor `PTIRF`, so it cannot spell this key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DdcMemory {
    /// `LX`.
    Lx,
    /// `L0`.
    L0,
    /// `L0_SCALE` — the scale half of a scaled L0 allocation, which entry 258 spreads over the same
    /// cores as `L0` (`ddc/ddcv1.cpp:191-192`) and, above `RCUDD1A`, over its corelets too (`:202`).
    L0Scale,
    /// `PELRF`.
    PeLrf,
    /// `SFPLRF`.
    SfpLrf,
    /// `PTARF`.
    PtaRf,
    /// `PTXRF`.
    PtxRf,
    /// `PTIRF`.
    PtiRf,
    /// `HBM`.
    Hbm,
}

/// ONE OPAQUE COMPUTE'S REGISTERS — `Metadata::OpaqueOp` (`ddc/ddc_metadata.h:196`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpaqueOp {
    /// `inOutRegAllocs_` — which allocation supplies each in/out register's address
    /// (`ddc/ddl/ddl_conversion.cpp:1676-1690`).
    pub in_out_reg_allocs: BTreeMap<RegName, AllocId>,
    /// `internalRegs_`.
    pub internal_regs: Vec<OpaqueReg>,
    /// `internalRegAlloc_`.
    pub internal_reg_alloc: Option<AllocId>,
    /// `max_unroll_`, whose default is ONE and not zero.
    pub max_unroll: MaxUnroll,
    /// `ldsIdx_` — `-1` is none.
    pub lds_idx: Option<LdsIdx>,
}

impl Default for OpaqueOp {
    fn default() -> Self {
        Self {
            in_out_reg_allocs: BTreeMap::new(),
            internal_regs: Vec::new(),
            internal_reg_alloc: None,
            max_unroll: MaxUnroll(1),
            lds_idx: None,
        }
    }
}

impl OpaqueOp {
    /// `internalRegsWithUnroll_` (`ddc/ddc_metadata.h:199`) — the reference keeps a counter bumped
    /// as each `_unroll` register is pushed (`ddc/ddl/ddl_conversion.cpp:1663-1664`);
    /// [`OpaqueReg::unrolled`] already carries that fact, so it is derived here and cannot disagree
    /// with the list it counts.
    #[must_use]
    pub fn internal_regs_with_unroll(&self) -> usize {
        self.internal_regs.iter().filter(|reg| reg.unrolled).count()
    }
}

/// `Metadata::DDCTransformationConfigT` (`ddc/ddc_metadata.h:230`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransformationConfig {
    /// `enableMovingDataTransfer`, whose default is TRUE — `run_v1` hoists transfers for reuse
    /// unless it is off (`ddc/ddcv1.cpp:3757-3759`).
    pub enable_moving_data_transfer: bool,
}

impl Default for TransformationConfig {
    fn default() -> Self {
        Self {
            enable_moving_data_transfer: true,
        }
    }
}

/// DDC'S WHOLE PER-DSC STATE — `ddc::Metadata` (`ddc/ddc_metadata.h:31`).
#[derive(Debug, Default, PartialEq)]
pub struct Metadata {
    /// `datastages_` (:82).
    pub datastages: BTreeMap<DatastageId, Datastage>,
    /// `datatransfers_` (:119), keyed by the transfer node.
    pub datatransfers: BTreeMap<NodeId, DataTransfer>,
    /// `newAllocations_` (:127).
    pub new_allocations: BTreeMap<DdcMemory, Allocation>,
    /// `shadowAllocations_` (:128).
    pub shadow_allocations: Vec<Vec<AllocId>>,
    /// `externalTransfers_` (:137) — the one OWNING field, so the one entry 104 frees through.
    pub external_transfers: Vec<ExternalTransfer>,
    /// `prefilledExternalTransferToDataConnectToFill_` (:138), keyed by labeled DS and storage.
    ///
    /// ⛔ THE LDS IS NOT OPTIONAL: entry 309's `ldsIdx >= labeledDs_.size()` (`ddc/ddcv1.cpp:2330`)
    /// compares an `int` against a `size_t`, which promotes `-1` to `SIZE_MAX`, so an end naming no
    /// labelled DS is refused and never reaches this map.
    pub prefilled_external_transfer_data_connects:
        BTreeMap<(LdsIdx, ExternalStorage), DataConnectSlot>,
    /// `externalNodes_` (:140).
    pub external_nodes: BTreeSet<NodeId>,
    /// `TransferNodesInterSliceTranspose_` (:141).
    pub transfer_nodes_inter_slice_transpose: BTreeSet<NodeId>,
    /// `dataConnects_` (:194) — keyed by the connect's NAME, holding the [`Ends`] entries 100-103
    /// maintain.
    pub data_connects: BTreeMap<DataConnect, Ends>,
    /// `opaqueOps_` (:204), keyed by the compute node.
    pub opaque_ops: BTreeMap<NodeId, OpaqueOp>,
    /// `implicitSyncs_` (:206) — the allocation each implicit sync stands for.
    pub implicit_syncs: BTreeMap<NodeId, AllocId>,
    /// `dimToCoreChunkLoops_` (:208-209).
    pub dim_to_core_chunk_loops: BTreeMap<PrimaryDim, Vec<NodeId>>,
    /// `rowSplitDim` (:213) — `PrimaryDimTypesCount` is UNSET, which is not dimension zero.
    pub row_split_dim: Option<PrimaryDim>,
    /// `clSplitDims_` (:214).
    pub cl_split_dims: BTreeSet<PrimaryDim>,
    /// `peSfpSplitDims_` (:215).
    pub pe_sfp_split_dims: BTreeSet<PrimaryDim>,
    /// `nodeCloningMap_` (:216-217).
    pub node_cloning_map: BTreeMap<NodeId, Vec<NodeId>>,
    /// `discardAboveLxSchedule_` (:218).
    pub discard_above_lx_schedule: bool,
    /// `belowLxScheduleInsertBlock` (:219) — the block named `lx_below_schedule` where the tree has
    /// one (`ddc/ddcv1.cpp:2342-2345`).
    pub below_lx_schedule_insert_block: Option<BlockId>,
    /// `opFuncBackup_` (:222) — `OpFuncs::NONE` is none.
    ///
    /// ⛔ THE FULL `OpFuncs`, NOT [`crate::generated::OpFunc`]: entry 308 backs up whatever name sits
    /// on `computeOp_.at(0)` under its `EXX2` gate (`ddc/ddcv1.cpp:2064-2078`), and the DDL census
    /// carries no `EXX2` at all, so the censused enum cannot express what entry 132 puts back.
    /// ⛔ AND `EXX2` IS NOT THE ONLY NAME IT HOLDS: the `opConsts` arm (`:2073-2077`) has no `else`,
    /// so where the `constantInfo_` arm (`:2065-2072`) already swapped in `EXX2_ZEROMEAN` the second
    /// backup re-reads that (`:2075`) and `restoreDsc` puts `EXX2_ZEROMEAN` back.
    pub op_func_backup: Option<OpFunc>,
    /// `transformationConfig_` (:232).
    pub transformation_config: TransformationConfig,
    /// `ldsIdxAfterDdc` (:235) — each labeled DS index before DDC to its index after.
    pub lds_idx_after_ddc: BTreeMap<LdsIdx, LdsIdx>,
    /// `intermLdsIdxToExtLds` (:238).
    pub interm_lds_idx_to_ext_lds: BTreeMap<LdsIdx, LdsIdx>,
}

impl Metadata {
    /// `core_dstgid` (`ddc/ddc_metadata.h:211`) — a `const int` in the reference, so not state.
    pub const CORE_DSTGID: DatastageId = DatastageId(0);
    /// `chunk_dstgid` (:212).
    pub const CHUNK_DSTGID: DatastageId = DatastageId(1);

    /// Replaces: e104_clear
    ///
    /// REINITIALIZES THE WHOLE METADATA (`ddc/ddc_metadata.h:225`). `run_v1` calls it once per DSC
    /// before working on that DSC (`ddc/ddcv1.cpp:3706`) — the ONLY callsite in the tree.
    ///
    /// ⛔ THE RESET IS NOT A MEMSET: `strategyMinimize_`, `enableMovingDataTransfer` and
    /// `max_unroll_` come back true/true/1, and `rowSplitDim`, `nearestNumeratorIdx_`,
    /// `force_num_elements_` and `ldsIdx_` come back UNSET rather than zero.
    /// ⛔ AND IT FREES: `externalTransfers_`'s `unique_ptr`s are destroyed here, while the
    /// non-owning `std::string*` slots beside them are only dropped.
    /// ⚠️ 31 OTHER UNITS LIST `e104_clear` AS A CALLEE IN `UNITS.tsv`; every one of those is a
    /// container `.clear()` resolved by name, not this method — `metadata.clear()` occurs ONCE in
    /// the whole tree.
    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{
        Ends, LoopIndex, LoopMultiple, MetaDimKind, NodeIndex, NodeName, NodeNames, OwnerLoops,
    };
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{
        AbsoluteMin, Constraint, ConstraintKind, NoEpilogueDimKind,
    };

    /// An absolute constraint with no dim kind — the arm where `min_` doubles as the multiple.
    fn absolute(min: AbsoluteMin) -> Constraint<'static, ()> {
        Constraint {
            kind: ConstraintKind::Absolute {
                cannot_be_symbolic: false,
                dim_kind: None,
                min,
            },
            max: None,
            values: None,
        }
    }

    /// A relative constraint against `reference`, whose `min_` is a bare bound.
    fn relative(min: Option<f32>, reference: &(), multiple: LoopMultiple) -> Constraint<'_, ()> {
        Constraint {
            kind: ConstraintKind::Relative {
                reference,
                min,
                multiple,
            },
            max: None,
            values: None,
        }
    }

    /// A schedule tree as two tables — a name per node, and each node's `prev_` walk already resolved
    /// to the loop it lands on.
    struct Tree {
        names: Vec<NodeName>,
        unnamed: NodeName,
        owner: Vec<Option<usize>>,
    }

    impl NodeNames for Tree {
        fn name(&self, node: NodeIndex) -> &NodeName {
            self.names.get(node.0).unwrap_or(&self.unnamed)
        }
    }

    impl OwnerLoops for Tree {
        fn owner_loop(&self, node: NodeIndex) -> Option<LoopIndex> {
            self.owner
                .get(node.0)
                .copied()
                .flatten()
                .map(|at| LoopIndex(NodeIndex(at)))
        }
    }

    /// 🎯 096/382 THE LARGER LOWER BOUND WINS, AND ON THE ABSOLUTE ARM IT IS ALSO THE MULTIPLE —
    /// `ddc/ddc_metadata.h:40`.
    #[test]
    fn a_second_min_keeps_the_larger_and_the_first_one_adopts() {
        let mut unset = absolute(AbsoluteMin::Unset);
        unset.update_min(4.0);
        assert_eq!(
            unset.kind,
            ConstraintKind::Absolute {
                cannot_be_symbolic: false,
                dim_kind: None,
                min: AbsoluteMin::Bound(4.0),
            }
        );
        // ⛔ A LOOSER BOUND CHANGES NOTHING, and the multiple relationship survives the raise.
        let mut multiple = absolute(AbsoluteMin::Multiple(4.0));
        multiple.update_min(2.0);
        multiple.update_min(8.0);
        assert_eq!(
            multiple.kind,
            ConstraintKind::Absolute {
                cannot_be_symbolic: false,
                dim_kind: None,
                min: AbsoluteMin::Multiple(8.0),
            }
        );
        let reference = ();
        // ⛔ AND ON THE RELATIVE ARM `min_` IS A BARE BOUND: raising it leaves `mustBeMultiple_` and
        // its dim kind exactly where they were.
        let mut ratio = relative(
            Some(0.5),
            &reference,
            LoopMultiple::NoEpilogue(NoEpilogueDimKind::Padded),
        );
        ratio.update_min(0.25);
        assert_eq!(
            ratio.kind,
            ConstraintKind::Relative {
                reference: &reference,
                min: Some(0.5),
                multiple: LoopMultiple::NoEpilogue(NoEpilogueDimKind::Padded),
            }
        );
    }

    /// 🎯 097/382 THE SMALLER UPPER BOUND WINS — `ddc/ddc_metadata.h:43`.
    #[test]
    fn a_second_max_keeps_the_smaller_and_the_first_one_adopts() {
        let mut constraint = absolute(AbsoluteMin::Unset);
        constraint.update_max(4.0);
        assert_eq!(constraint.max, Some(4.0));
        constraint.update_max(8.0);
        assert_eq!(constraint.max, Some(4.0));
        constraint.update_max(1.0);
        assert_eq!(constraint.max, Some(1.0));
    }

    /// 🎯 098/382 ⛔ THE FIRST SET IS ADOPTED WHOLE AND LATER ONES INTERSECT, WHICH CAN LEAVE THE SET
    /// ENGAGED AND EMPTY — `ddc/ddc_metadata.h:46`.
    #[test]
    fn the_first_values_are_adopted_and_an_intersection_may_empty_them() {
        let mut constraint = absolute(AbsoluteMin::Unset);
        // ⛔ NOT AN INTERSECTION WITH `None`: absent means unconstrained, so this keeps all three.
        constraint.update_values(&[4.0, 1.0, 2.0, 1.0]);
        assert_eq!(constraint.values, Some(vec![1.0, 2.0, 4.0]));
        constraint.update_values(&[2.0, 4.0, 8.0]);
        assert_eq!(constraint.values, Some(vec![2.0, 4.0]));
        // ⛔ ENGAGED AND EMPTY IS UNSATISFIABLE, and it is NOT the same state as absent.
        constraint.update_values(&[16.0]);
        assert_eq!(constraint.values, Some(vec![]));
    }

    /// 🎯 099/382 THE LINE `std::cerr` GETS, INCLUDING `%g`'s SIX SIGNIFICANT DIGITS AND THE
    /// `NOT_SET` / `-inf` / `inf` ABSENCES — `ddc/ddc_metadata.h:49`.
    #[test]
    fn a_dumped_constraint_states_every_field_the_way_the_stream_does() {
        let mut constraint = absolute(AbsoluteMin::Multiple(1.0 / 3.0));
        constraint.max = Some(1e7);
        constraint.values = Some(vec![0.5, 64.0, 1e-5]);
        // ⛔ `0.333333`, NOT Rust's round-tripping `0.33333334`; `%g` leaves `[1e-4, 1e6)` at both
        // ends of the value list; and the values come out ASCENDING, not in the order given.
        assert_eq!(
            constraint.dump(),
            "mustBeMultiple_= T loopDimKind_= NOT_SET , min_= 0.333333 , max_= 1e+07 , \
             values_= {1e-05 0.5 64 }\n"
        );
        // A kind on a constraint that is NOT a multiple is exactly what `ddc_transformation.cpp:968`
        // sets, and both bounds absent print as the infinities they are.
        let reference = ();
        let plain = relative(
            None,
            &reference,
            LoopMultiple::Off(Some(MetaDimKind::Unpadded)),
        );
        assert_eq!(
            plain.dump(),
            "mustBeMultiple_= F loopDimKind_= unpadded , min_= -inf , max_= inf , values_= {}\n"
        );
    }

    /// 🎯 100/382 · 101/382 EACH SIDE RECORDS A NODE ONCE, IN FIRST-TOUCH ORDER, AND THE TWO LISTS ARE
    /// INDEPENDENT — `ddc/ddc_metadata.h:147-157`.
    #[test]
    fn a_node_inserted_twice_on_a_side_is_one_entry_there_and_still_free_on_the_other() {
        let mut ends = Ends::default();
        ends.insert_producer(NodeIndex(2));
        ends.insert_producer(NodeIndex(0));
        ends.insert_producer(NodeIndex(2));
        assert_eq!(ends.producers(), [NodeIndex(2), NodeIndex(0)]);
        // ⛔ A NODE THAT PRODUCES AND CONSUMES THE SAME CONNECT IS ON BOTH LISTS: the test is
        // per-list.
        ends.insert_consumer(NodeIndex(2));
        assert_eq!(ends.consumers(), [NodeIndex(2)]);
    }

    /// 🎯 102/382 CONSUMERS FIRST, EACH NAME PRECEDED BY A SPACE, AND AN EMPTY SIDE PRINTS `[]` —
    /// `ddc/ddc_metadata.h:166`.
    #[test]
    fn a_printed_data_connect_leads_with_its_consumers() {
        let named = |name: &str| NodeName(name.to_string());
        let tree = Tree {
            names: vec![named("ht_out"), named("pe_mac"), named("sfp_act")],
            unnamed: named(""),
            owner: vec![],
        };
        let mut ends = Ends::default();
        ends.insert_producer(NodeIndex(0));
        ends.insert_consumer(NodeIndex(1));
        ends.insert_consumer(NodeIndex(2));
        assert_eq!(
            ends.print(&tree),
            " Consumers= [ pe_mac sfp_act] Producers= [ ht_out]\n"
        );
        assert_eq!(
            Ends::default().print(&tree),
            " Consumers= [] Producers= []\n"
        );
    }

    /// 🎯 103/382 THE ANCESTOR CHAIN OF EVERY END, WITH THE SHARED TAIL WALKED ONCE — and the `break`
    /// that makes a `prev_` cycle terminate (`ddc/ddc_metadata.h:179`).
    #[test]
    fn the_loops_of_an_end_are_its_ancestors_and_a_shared_tail_is_not_rewalked() {
        // 0 and 4 are leaves; loops 3 -> 2 -> 1 nest, and 1 is outermost. Loop 5 owns itself.
        let tree = Tree {
            names: vec![],
            unnamed: NodeName(String::new()),
            owner: vec![Some(3), None, Some(1), Some(2), Some(2), Some(5)],
        };
        let mut ends = Ends::default();
        ends.insert_producer(NodeIndex(0));
        ends.insert_producer(NodeIndex(4));
        // ⛔ NODE 4's OWNER 2 IS ALREADY RECORDED, so its walk stops there — and loses nothing,
        // because 1 went in behind 2 the first time.
        assert_eq!(
            ends.producer_loops(&tree),
            [
                LoopIndex(NodeIndex(3)),
                LoopIndex(NodeIndex(2)),
                LoopIndex(NodeIndex(1)),
            ]
        );
        // ⛔ A SELF-OWNING LOOP TERMINATES: node 5's owner is 5, recorded once and then broken on.
        ends.insert_consumer(NodeIndex(5));
        assert_eq!(ends.consumer_loops(&tree), [LoopIndex(NodeIndex(5))]);
    }
}

// ⭐ TESTS FOR ENTRY 104. Union this module with this file's other test modules when they land.
#[cfg(test)]
mod tests_e104 {
    use super::{
        AllocId, DataConnectSlot, DataTransfer, Datastage, DestIdx, ExternalStorage,
        ExternalTransfer, LdsIdx, Metadata, NodeId, OpaqueOp, OwnedAllocateNode, OwnedTransferNode,
        PrimaryDim, StoredConstraint, TransferEnd,
    };
    use crate::arch::Elements;
    use crate::generated::{MaxUnroll, Strategy};

    /// A metadata dirtied in the fields whose construction default is NOT the zero value comes back
    /// to exactly the constructed state — true, true, 1 and unset, not false, false, 0 and zero.
    #[test]
    fn clear_restores_the_constructed_defaults_and_not_zeros() {
        let mut md = Metadata::default();
        md.datastages.insert(
            Metadata::CORE_DSTGID,
            Datastage {
                strategy: Strategy::Maximize,
                nearest_numerator_idx: Some(Metadata::CHUNK_DSTGID),
                ..Datastage::default()
            },
        );
        md.datatransfers.insert(
            NodeId(4),
            DataTransfer {
                force_num_elements: Some(Elements(64)),
                offset_dest: [(DestIdx(1), Elements(8))].into_iter().collect(),
                ..DataTransfer::default()
            },
        );
        md.opaque_ops.insert(
            NodeId(5),
            OpaqueOp {
                max_unroll: MaxUnroll(4),
                lds_idx: Some(LdsIdx(2)),
                ..OpaqueOp::default()
            },
        );
        md.external_transfers.push(ExternalTransfer {
            transfer: OwnedTransferNode(NodeId(7)),
            allocate: OwnedAllocateNode(AllocId(9)),
        });
        // The non-owning slot beside them, which the reset drops without destroying anything.
        md.prefilled_external_transfer_data_connects.insert(
            (LdsIdx(3), ExternalStorage::Lx),
            DataConnectSlot {
                transfer: NodeId(7),
                end: TransferEnd::Src,
            },
        );
        md.row_split_dim = Some(PrimaryDim::Y);
        md.discard_above_lx_schedule = true;
        md.transformation_config.enable_moving_data_transfer = false;
        md.cl_split_dims.insert(PrimaryDim::X);

        md.clear();

        assert_eq!(md, Metadata::default());
        assert!(md.transformation_config.enable_moving_data_transfer);
        assert_eq!(md.row_split_dim, None);
        assert!(md.external_transfers.is_empty());
        assert_eq!(Datastage::default().strategy, Strategy::Minimize);
        assert_eq!(Datastage::default().nearest_numerator_idx, None);
        assert_eq!(OpaqueOp::default().max_unroll, MaxUnroll(1));
        assert_eq!(DataTransfer::default().offset_src, Elements(0));
        assert_eq!(DataTransfer::default().force_num_elements, None);
        assert_eq!(StoredConstraint::default().min, None);
        // The two `const int` members are consts here, so the reset cannot lose them.
        assert_eq!(Metadata::CORE_DSTGID.0, 0);
        assert_eq!(Metadata::CHUNK_DSTGID.0, 1);
    }
}

// `FailedAlloc` (`ddc/ddc_metadata.h:24`) IS [`crate::schedule::ddc::v1::TrackerSite`], field for
// field, and the anchor lives there because that is where the four values are load-bearing. Its
// members are the tracker site an allocation was refused at — the same four `getTracker` is keyed by
// — and its only container in the reference is read solely as `size() == 0` (`ddc/ddcv1.cpp:378`,
// `:436`), so a second struct here would carry nothing that `success` does not already say.
