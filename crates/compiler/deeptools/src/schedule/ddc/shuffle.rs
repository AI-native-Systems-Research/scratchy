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

//! `ddc/transformations/automatic_shuffle/shuffle.cpp`, `ddc/transformations/automatic_shuffle/shuffle.h` — 50 of the campaign's 382 units (dependency level(s) [0, 1, 2, 3, 4, 5]).
//!
//! | unit | entry | level | lines | class | authority path:line |
//! |---|---|---|---|---|---|
//! | `e137_int_log2` | 137 | 0 | 9 | — | `ddc/transformations/automatic_shuffle/shuffle.cpp:95` |
//! | `e138_swap` | 138 | 0 | 1 | `SwapBuffer` | `ddc/transformations/automatic_shuffle/shuffle.cpp:121` |
//! | `e139_packmerge_psuedostring` | 139 | 0 | 9 | — | `ddc/transformations/automatic_shuffle/shuffle.cpp:124` |
//! | `e140_repeat_over_dims` | 140 | 0 | 26 | `ComputationOp` | `ddc/transformations/automatic_shuffle/shuffle.cpp:178` |
//! | `e141_act` | 141 | 0 | 16 | `MergeAction` | `ddc/transformations/automatic_shuffle/shuffle.cpp:249` |
//! | `e142_cost` | 142 | 0 | 6 | `MergeAction` | `ddc/transformations/automatic_shuffle/shuffle.cpp:266` |
//! | `e143_get_shuffle_indices` | 143 | 0 | 29 | `MergeAction` | `ddc/transformations/automatic_shuffle/shuffle.cpp:275` |
//! | `e144_act` | 144 | 0 | 10 | `PackAction` | `ddc/transformations/automatic_shuffle/shuffle.cpp:351` |
//! | `e145_get_indices` | 145 | 0 | 12 | `PackAction` | `ddc/transformations/automatic_shuffle/shuffle.cpp:366` |
//! | `e146_act` | 146 | 0 | 13 | `ShiftLeftAction` | `ddc/transformations/automatic_shuffle/shuffle.cpp:409` |
//! | `e147_cost` | 147 | 0 | 4 | `ShiftLeftAction` | `ddc/transformations/automatic_shuffle/shuffle.cpp:423` |
//! | `e148_act` | 148 | 0 | 10 | `Pack8Action` | `ddc/transformations/automatic_shuffle/shuffle.cpp:476` |
//! | `e149_act` | 149 | 0 | 8 | `Pack9Action` | `ddc/transformations/automatic_shuffle/shuffle.cpp:523` |
//! | `e150_cost` | 150 | 0 | 4 | `Pack9Action` | `ddc/transformations/automatic_shuffle/shuffle.cpp:532` |
//! | `e151_act` | 151 | 0 | 11 | `Pack24Action` | `ddc/transformations/automatic_shuffle/shuffle.cpp:573` |
//! | `e152__out_format` | 152 | 0 | 8 | `GCVTF16F8PackAction` | `ddc/transformations/automatic_shuffle/shuffle.cpp:598` |
//! | `e153_act` | 153 | 0 | 7 | `GCVTF16F8PackAction` | `ddc/transformations/automatic_shuffle/shuffle.cpp:620` |
//! | `e154__out_format` | 154 | 0 | 8 | `GCVTF16F8MergeAction` | `ddc/transformations/automatic_shuffle/shuffle.cpp:644` |
//! | `e155_act` | 155 | 0 | 6 | `GCVTF16F8MergeAction` | `ddc/transformations/automatic_shuffle/shuffle.cpp:666` |
//! | `e156_contains` | 156 | 0 | 5 | `AbstractLayout` | `ddc/transformations/automatic_shuffle/shuffle.cpp:684` |
//! | `e157_get_node` | 157 | 0 | 11 | `AutoShuffler` | `ddc/transformations/automatic_shuffle/shuffle.cpp:717` |
//! | `e158_make_stick_number_key` | 158 | 0 | 14 | — | `ddc/transformations/automatic_shuffle/shuffle.cpp:730` |
//! | `e159_all_one` | 159 | 0 | 3 | — | `ddc/transformations/automatic_shuffle/shuffle.cpp:1043` |
//! | `e160_update_worklist` | 160 | 0 | 7 | `AutoShuffler` | `ddc/transformations/automatic_shuffle/shuffle.cpp:1139` |
//! | `e161_codegen_psuedocode` | 161 | 0 | 42 | `AutoShuffler` | `ddc/transformations/automatic_shuffle/shuffle.cpp:1224` |
//! | `e162_getDefaultSymbols` | 162 | 0 | 8 | `DimSymbol` | `ddc/transformations/automatic_shuffle/shuffle.h:48` |
//! | `e163_constexpr` | 163 | 0 | 3 | `std` | `ddc/transformations/automatic_shuffle/shuffle.h:160` |
//! | `e164_insert_before` | 164 | 0 | 7 | `DataEdge` | `ddc/transformations/automatic_shuffle/shuffle.h:175` |
//! | `e264_bin_op` | 264 | 1 | 19 | — | `ddc/transformations/automatic_shuffle/shuffle.cpp:137` |
//! | `e265_unary_op` | 265 | 1 | 17 | — | `ddc/transformations/automatic_shuffle/shuffle.cpp:160` |
//! | `e266_canonicalize_layout` | 266 | 1 | 18 | `AutoShuffler` | `ddc/transformations/automatic_shuffle/shuffle.cpp:690` |
//! | `e267_reset_graph` | 267 | 1 | 6 | `AutoShuffler` | `ddc/transformations/automatic_shuffle/shuffle.cpp:710` |
//! | `e268_sort_by_stick_key` | 268 | 1 | 15 | — | `ddc/transformations/automatic_shuffle/shuffle.cpp:746` |
//! | `e269_check_single_inorder_accesses` | 269 | 1 | 17 | — | `ddc/transformations/automatic_shuffle/shuffle.cpp:767` |
//! | `e270_inferLayouts` | 270 | 1 | 90 | `AutoShuffler` | `ddc/transformations/automatic_shuffle/shuffle.cpp:1047` |
//! | `e310_add_valid_actions` | 310 | 2 | 11 | `MergeAction` | `ddc/transformations/automatic_shuffle/shuffle.cpp:226` |
//! | `e311_enumerate_stick_computations` | 311 | 2 | 15 | `MergeAction` | `ddc/transformations/automatic_shuffle/shuffle.cpp:305` |
//! | `e312_add_valid_actions` | 312 | 2 | 12 | `PackAction` | `ddc/transformations/automatic_shuffle/shuffle.cpp:332` |
//! | `e313_enumerate_stick_computations` | 313 | 2 | 4 | `PackAction` | `ddc/transformations/automatic_shuffle/shuffle.cpp:379` |
//! | `e314_add_valid_actions` | 314 | 2 | 6 | `ShiftLeftAction` | `ddc/transformations/automatic_shuffle/shuffle.cpp:398` |
//! | `e315_enumerate_stick_computations` | 315 | 2 | 15 | `ShiftLeftAction` | `ddc/transformations/automatic_shuffle/shuffle.cpp:428` |
//! | `e316_add_valid_actions` | 316 | 2 | 20 | `Pack8Action` | `ddc/transformations/automatic_shuffle/shuffle.cpp:451` |
//! | `e317_add_valid_actions` | 317 | 2 | 17 | `Pack9Action` | `ddc/transformations/automatic_shuffle/shuffle.cpp:501` |
//! | `e318_add_valid_actions` | 318 | 2 | 20 | `Pack24Action` | `ddc/transformations/automatic_shuffle/shuffle.cpp:548` |
//! | `e319_add_valid_actions` | 319 | 2 | 8 | `GCVTF16F8PackAction` | `ddc/transformations/automatic_shuffle/shuffle.cpp:607` |
//! | `e320_add_valid_actions` | 320 | 2 | 8 | `GCVTF16F8MergeAction` | `ddc/transformations/automatic_shuffle/shuffle.cpp:653` |
//! | `e342_codegen_generic` | 342 | 3 | 123 | `AutoShuffler` | `ddc/transformations/automatic_shuffle/shuffle.cpp:910` |
//! | `e343_get_legal_transforms` | 343 | 3 | 12 | `AutoShuffler` | `ddc/transformations/automatic_shuffle/shuffle.cpp:1147` |
//! | `e362_get_shuffle` | 362 | 4 | 61 | `AutoShuffler` | `ddc/transformations/automatic_shuffle/shuffle.cpp:1161` |
//! | `e371_replace_assign` | 371 | 5 | 119 | `AutoShuffler` | `ddc/transformations/automatic_shuffle/shuffle.cpp:788` |

use crate::arch::Sticks;
use crate::generated::DataType;
use std::collections::{BTreeMap, BTreeSet};

// ───────────────────────────────────────────────────────────────────────────────────────────────
// THE LAYOUT VOCABULARY THESE UNITS ACT ON — `shuffle.h:26-110`, `shuffle.cpp:19-122`.
//
// Declarations, not units. This file's units are the anchors; the members that ARE units
// (`AbstractLayout::contains` e156, `DimSymbol::getDefaultSymbols` e162) stay unfilled below.
// ───────────────────────────────────────────────────────────────────────────────────────────────

/// A 2-ELEMENT (SUB)DIMENSION of a slice or of a stick array (`shuffle.h:26`).
///
/// Symbol 0 is the dummy: a place in the layout no live dimension claims.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DimSymbol(i32);

impl DimSymbol {
    /// `DimSymbol::getDummy()` (`shuffle.h:44`).
    pub const DUMMY: Self = Self(0);
    /// `DimSymbol::getDefaultSymbol()` (`shuffle.h:45`).
    pub const DEFAULT: Self = Self(1);

    /// `is_dummy()` (`shuffle.h:35`).
    #[must_use]
    pub const fn is_dummy(self) -> bool {
        self.0 == Self::DUMMY.0
    }

    /// `next()` (`shuffle.h:36`) — the next symbol in a minting run.
    #[must_use]
    pub const fn next(self) -> Self {
        Self(self.0 + 1)
    }

    /// `getID()` (`shuffle.h:43`).
    #[must_use]
    pub const fn id(self) -> i32 {
        self.0
    }
}

/// WHICH HALF of a split dimension a stick holds — the reference's `bool high` (`shuffle.h:60`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Half {
    /// `high == false`.
    Low,
    /// `high == true`.
    High,
}

/// A (PARTIAL) STICK INDEX — `std::set<DimIndex, DimIndexDimComparator>` (`shuffle.h:63-69`).
///
/// ⛔ KEYED BY DIMENSION ALONE, so inserting a dimension already present is a NO-OP that KEEPS the
/// half already recorded — `std::set::insert` does not overwrite, and `repeat_over_dims` inserts
/// into indices it has already filled.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct StickIndex(BTreeMap<DimSymbol, Half>);

impl StickIndex {
    /// The empty index — `emplace_back()` on an `inputs` list (`shuffle.cpp:162`).
    #[must_use]
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }

    /// `insert({dim, half})`, which keeps the existing half for a dimension already indexed.
    pub fn insert(&mut self, dim: DimSymbol, half: Half) {
        self.0.entry(dim).or_insert(half);
    }

    /// The indexed dimensions and their halves, in dimension order.
    pub fn iter(&self) -> impl Iterator<Item = (DimSymbol, Half)> + '_ {
        self.0.iter().map(|(dim, half)| (*dim, *half))
    }

    /// Which half this index takes of `dim`, absent when it does not index it.
    #[must_use]
    pub fn half_of(&self, dim: DimSymbol) -> Option<Half> {
        self.0.get(&dim).copied()
    }
}

impl FromIterator<(DimSymbol, Half)> for StickIndex {
    fn from_iter<I: IntoIterator<Item = (DimSymbol, Half)>>(iter: I) -> Self {
        let mut index = Self::new();
        for (dim, half) in iter {
            index.insert(dim, half);
        }
        index
    }
}

/// HOW MANY SUBDIMENSIONS ONE SLICE HAS — `dims_per_slice = 6 == log2(bits_per_slice)`
/// (`shuffle.cpp:90-91`).
pub const DIMS_PER_SLICE: usize = 6;

/// WHICH SUBDIMENSION OF A SLICE — the reference's `dim_2bit`..`dim_64bit` (`shuffle.cpp:78-83`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SliceDim {
    /// `dim_2bit = 0`.
    Bit2,
    /// `dim_4bit = 1`.
    Bit4,
    /// `dim_8bit = 2`.
    Bit8,
    /// `dim_16bit = 3`.
    Bit16,
    /// `dim_32bit = 4`.
    Bit32,
    /// `dim_64bit = 5`.
    Bit64,
}

impl SliceDim {
    /// Its position in the slice, which is the integer the reference indexes with.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// A LAYOUT WITH ITS DIMENSIONS AS SYMBOLS — stick dimensions unordered, slice dimensions ordered
/// by significance (`shuffle.h:81`).
///
/// ⛔ THE SLICE IS EXACTLY SIX SUBDIMENSIONS, AS AN ARRAY. The reference `DT_CHECK`s that length
/// (`shuffle.cpp:692-693`, `:1165-1166`) and every `act` restores it after its erase/insert pair,
/// so it is an invariant of the type here rather than a check someone remembers.
///
/// ⛔ FIELD ORDER IS LOAD-BEARING: `operator<` compares format, then stick dims, then slice dims
/// (`shuffle.h:105-109`), and that is what the derived `Ord` reproduces.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AbstractLayout {
    /// The element format the sticks are in.
    pub format: DataType,
    /// The dimensions taken across sticks. Order carries no meaning in the abstraction.
    pub stick_dims: BTreeSet<DimSymbol>,
    /// The dimensions within a slice, least significant first.
    pub slice_dims: [DimSymbol; DIMS_PER_SLICE],
}

impl AbstractLayout {
    /// `AbstractLayout(stick, slice, format)` (`shuffle.h:90`).
    #[must_use]
    pub fn new(
        stick_dims: BTreeSet<DimSymbol>,
        slice_dims: [DimSymbol; DIMS_PER_SLICE],
        format: DataType,
    ) -> Self {
        Self {
            format,
            stick_dims,
            slice_dims,
        }
    }

    /// The dimension occupying one slice subdimension — `sliceDims()[i]`.
    #[must_use]
    pub const fn slice_dim(&self, at: SliceDim) -> DimSymbol {
        self.slice_dims[at.index()]
    }

    /// `numSticks() = 1u << stick_dims.size()` (`shuffle.h:96`). The reference's own cap on that
    /// size is 32 (`make_stick_number_key`, `shuffle.cpp:731`).
    #[must_use]
    pub fn num_sticks(&self) -> Sticks {
        Sticks(1u64 << self.stick_dims.len())
    }
}

/// WHAT AN ACTION COSTS — Dijkstra's edge weight, the reference's `double` (`shuffle.h:230`).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct ShuffleCost(pub f64);

/// ONE LANE SELECTOR of a `packmerge` index table; `-1` selects nothing (`shuffle.cpp:19-76`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShuffleIndex(pub i32);

/// The vendored tables as written, so a transcription error is visible against `shuffle.cpp`.
const fn indices<const N: usize>(raw: [i32; N]) -> [ShuffleIndex; N] {
    let mut out = [ShuffleIndex(0); N];
    let mut i = 0;
    while i < N {
        out[i] = ShuffleIndex(raw[i]);
        i += 1;
    }
    out
}

/// `merge8h` (`shuffle.cpp:30`).
const MERGE8H: [ShuffleIndex; 16] =
    indices([1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31]);
/// `merge8l` (`shuffle.cpp:32`).
const MERGE8L: [ShuffleIndex; 16] =
    indices([0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30]);
/// `merge16h` (`shuffle.cpp:48`).
const MERGE16H: [ShuffleIndex; 8] = indices([1, 9, 3, 11, 5, 13, 7, 15]);
/// `merge16l` (`shuffle.cpp:49`).
const MERGE16L: [ShuffleIndex; 8] = indices([0, 8, 2, 10, 4, 12, 6, 14]);
/// `merge32h` (`shuffle.cpp:50`).
const MERGE32H: [ShuffleIndex; 8] = indices([2, 3, 10, 11, 6, 7, 14, 15]);
/// `merge32l` (`shuffle.cpp:51`).
const MERGE32L: [ShuffleIndex; 8] = indices([0, 1, 8, 9, 4, 5, 12, 13]);
/// `merge64h` (`shuffle.cpp:52`).
const MERGE64H: [ShuffleIndex; 8] = indices([4, 5, 6, 7, 12, 13, 14, 15]);
/// `merge64l` (`shuffle.cpp:53`).
const MERGE64L: [ShuffleIndex; 8] = indices([0, 1, 2, 3, 8, 9, 10, 11]);

/// A PSEUDOCODE REGISTER NAME — `r0`, `r1`, .. as `codegen_psuedocode` mints them
/// (`shuffle.cpp:1229-1234`).
///
/// ⛔ NOT [`crate::generated::RegName`], which is a template's declared register set.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PseudoReg(pub String);

/// ONE STICK COMPUTATION — which (partial) stick indices it reads, and which it writes
/// (`shuffle.h:199`).
///
/// The two `codegen` closures the reference carries are attached by e264/e265, which build the
/// `packmerge` ops; `repeat_over_dims` copies whatever the op holds.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ComputationOp {
    /// The (partial) stick indices for the inputs.
    pub inputs: Vec<StickIndex>,
    /// The (partial) stick index for the output. Several ops are used for several outputs.
    pub output: StickIndex,
    /// Whether the op reads one stick more than once (`unary_op`, `shuffle.cpp:163`).
    pub reuses_sticks: bool,
}

/// Replaces: e137_int_log2
///
/// Floor of log2, and `-1` for `x <= 0` — the reference's own documented answer
/// (`shuffle.cpp:94`), which its callers rely on for a `stickSize_` of 0.
#[must_use]
pub const fn int_log2(x: i32) -> i32 {
    if x <= 0 {
        -1
    } else {
        (i32::BITS - 1 - x.leading_zeros()) as i32
    }
}

/// A PAIR OF SLOTS WITH ONE ACTIVE — `SwapBuffer<T>` (`shuffle.cpp:109-122`), which `codegen_generic`
/// uses to route this action's output edges into the next action's inputs.
///
/// ⛔ THE DEFAULT IS `flag = true`, NOT `false` (`shuffle.cpp:112`): a derived `Default` would hand
/// back `b` where the reference hands back `a`, silently transposing the first action's edges.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwapBuffer<T> {
    flag: bool,
    a: T,
    b: T,
}

impl<T: Default> Default for SwapBuffer<T> {
    fn default() -> Self {
        Self {
            flag: true,
            a: T::default(),
            b: T::default(),
        }
    }
}

impl<T> SwapBuffer<T> {
    /// `SwapBuffer(a, b)` (`shuffle.cpp:118`).
    pub const fn new(a: T, b: T) -> Self {
        Self { flag: true, a, b }
    }

    /// `first()` (`shuffle.cpp:119`).
    pub const fn first(&self) -> &T {
        if self.flag { &self.a } else { &self.b }
    }

    /// `first()` as the reference's mutable reference.
    pub const fn first_mut(&mut self) -> &mut T {
        if self.flag { &mut self.a } else { &mut self.b }
    }

    /// `second()` (`shuffle.cpp:120`).
    pub const fn second(&self) -> &T {
        if self.flag { &self.b } else { &self.a }
    }

    /// `second()` as the reference's mutable reference.
    pub const fn second_mut(&mut self) -> &mut T {
        if self.flag { &mut self.b } else { &mut self.a }
    }

    /// Replaces: e138_swap
    ///
    /// Exchanges the two slots, so what `second()` answered is what `first()` answers next.
    pub const fn swap(&mut self) {
        self.flag = !self.flag;
    }
}

/// Replaces: e139_packmerge_psuedostring
///
/// One pseudocode line: `<out> = packmerge <in1> <in2> [ i i .. ]`, one space after every index.
#[must_use]
pub fn packmerge_psuedostring(
    indices: &[ShuffleIndex],
    in_reg_1: &PseudoReg,
    in_reg_2: &PseudoReg,
    out_reg_name: &PseudoReg,
) -> String {
    let mut line = format!(
        "{} = packmerge {} {} [ ",
        out_reg_name.0, in_reg_1.0, in_reg_2.0
    );
    for index in indices {
        line.push_str(&format!("{} ", index.0));
    }
    line.push(']');
    line
}

impl ComputationOp {
    /// Replaces: e140_repeat_over_dims
    ///
    /// Appends this op once per assignment of the `dims` it does not already index — 2^N copies for
    /// N absent dimensions, each copy differing in which half of one dimension it takes.
    ///
    /// ⛔ THE ABSENT SET IS COMPUTED FROM `inputs` ALONE (`shuffle.cpp:184-188`), so a dimension the
    /// OUTPUT already indexes is still split; the insert that would collide is the no-op
    /// [`StickIndex::insert`] documents.
    pub fn repeat_over_dims(&self, dims: &[DimSymbol], append_to: &mut Vec<Self>) {
        let original_size = append_to.len();
        append_to.push(self.clone());
        let mut dim_set: BTreeSet<DimSymbol> = BTreeSet::new();
        for input in &self.inputs {
            for (dim, _) in input.iter() {
                dim_set.insert(dim);
            }
        }
        for &dim in dims {
            if dim_set.contains(&dim) {
                continue;
            }
            let dim_stop = append_to.len();
            for i in original_size..dim_stop {
                let mut high_copy = append_to[i].clone();
                for input in &mut append_to[i].inputs {
                    input.insert(dim, Half::Low);
                }
                append_to[i].output.insert(dim, Half::Low);
                for input in &mut high_copy.inputs {
                    input.insert(dim, Half::High);
                }
                high_copy.output.insert(dim, Half::High);
                append_to.push(high_copy);
            }
        }
    }
}

/// WHICH SUBDIMENSION A MERGE ACTS ON — the four the ISA has a merge instruction for.
///
/// ⛔ THIS IS THE REFERENCE'S `DT_ERROR("Tried to use illegal merge instruction")`
/// (`shuffle.cpp:287`, `:300`) MADE UNREPRESENTABLE. Its only construction site loops from
/// `max(dim_8bit, bitwidth_to_idx(bw))` (`shuffle.cpp:230-232`), so a `MergeAction` never holds the
/// 2- or 4-bit slot and `get_shuffle_indices` needs no failing arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MergeDim {
    /// `dim_8bit`, using `merge8l`/`merge8h`.
    Bit8,
    /// `dim_16bit`, using `merge16l`/`merge16h`.
    Bit16,
    /// `dim_32bit`, using `merge32l`/`merge32h`.
    Bit32,
    /// `dim_64bit`, using `merge64l`/`merge64h`.
    Bit64,
}

impl MergeDim {
    /// The slice subdimension it names.
    #[must_use]
    pub const fn slice(self) -> SliceDim {
        match self {
            Self::Bit8 => SliceDim::Bit8,
            Self::Bit16 => SliceDim::Bit16,
            Self::Bit32 => SliceDim::Bit32,
            Self::Bit64 => SliceDim::Bit64,
        }
    }
}

/// WHAT BECOMES OF THE DIMENSION A MERGE DISPLACES — the reference's `bool swap`.
///
/// ⛔ `swap` IS NEVER SET WITH A DUMMY DISPLACED DIMENSION: its one construction site computes
/// `swap = !input.sliceDims()[i].is_dummy()` (`shuffle.cpp:233`), which is exactly the pair of
/// `DT_CHECK`s at `:250` and `:259`. Both are structural here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeMode {
    /// The slot held a dummy: the stick dimension moves in and nothing comes out.
    Fill,
    /// The slot held this live dimension, which is extracted into the sticks.
    Extract(DimSymbol),
}

/// MOVES A STICK DIMENSION INTO ONE SLICE SUBDIMENSION, optionally extracting the dimension it
/// displaces back into the sticks — `merge8L`/`merge8H` and friends (`shuffle.cpp:207-219`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeAction {
    stick_dim: DimSymbol,
    slice: MergeDim,
    mode: MergeMode,
}

impl MergeAction {
    /// `MergeAction(stick_dim, slice_dim, slice_idx, extract_slice_to_stick)` (`shuffle.cpp:240`),
    /// taking the displaced dimension as the construction site reads it (`shuffle.cpp:233-235`).
    #[must_use]
    pub const fn new(stick_dim: DimSymbol, slice: MergeDim, displaced: DimSymbol) -> Self {
        Self {
            stick_dim,
            slice,
            mode: if displaced.is_dummy() {
                MergeMode::Fill
            } else {
                MergeMode::Extract(displaced)
            },
        }
    }

    /// The stick dimension being merged in.
    #[must_use]
    pub const fn stick_dim(&self) -> DimSymbol {
        self.stick_dim
    }

    /// Which subdimension it merges on.
    #[must_use]
    pub const fn slice(&self) -> MergeDim {
        self.slice
    }

    /// Whether it also extracts the displaced dimension, and which one.
    #[must_use]
    pub const fn mode(&self) -> MergeMode {
        self.mode
    }

    /// Replaces: e141_act
    ///
    /// The stick dimension takes the slice slot; in [`MergeMode::Extract`] the dimension it
    /// displaced joins the stick dimensions, otherwise that slot's dummy is dropped.
    #[must_use]
    pub fn act(&self, input: &AbstractLayout) -> AbstractLayout {
        let mut output = input.clone();
        if !self.stick_dim.is_dummy() {
            output.stick_dims.remove(&self.stick_dim);
        }
        let evicted_dim = output.slice_dim(self.slice.slice());
        output.slice_dims[self.slice.slice().index()] = self.stick_dim;
        if matches!(self.mode, MergeMode::Extract(_)) {
            output.stick_dims.insert(evicted_dim);
        }
        output
    }

    /// Replaces: e142_cost
    ///
    /// One instruction per two input sticks, or one per stick when the merge also extracts.
    ///
    /// ⛔ INTEGER DIVISION, AS THE REFERENCE HAS IT (`shuffle.cpp:268`): a one-stick layout costs
    /// 0, not 0.5.
    #[must_use]
    pub fn cost(&self, input: &AbstractLayout) -> ShuffleCost {
        let sticks = input.num_sticks().0;
        match self.mode {
            MergeMode::Fill => ShuffleCost((sticks / 2) as f64),
            MergeMode::Extract(_) => ShuffleCost(sticks as f64),
        }
    }

    /// Replaces: e143_get_shuffle_indices
    ///
    /// The vector-shuffle table for the low or the high half of this merge.
    #[must_use]
    pub const fn get_shuffle_indices(&self, half: Half) -> &'static [ShuffleIndex] {
        match (self.slice, half) {
            (MergeDim::Bit8, Half::Low) => &MERGE8L,
            (MergeDim::Bit8, Half::High) => &MERGE8H,
            (MergeDim::Bit16, Half::Low) => &MERGE16L,
            (MergeDim::Bit16, Half::High) => &MERGE16H,
            (MergeDim::Bit32, Half::Low) => &MERGE32L,
            (MergeDim::Bit32, Half::High) => &MERGE32H,
            (MergeDim::Bit64, Half::Low) => &MERGE64L,
            (MergeDim::Bit64, Half::High) => &MERGE64H,
        }
    }
}

/// WHICH SUBDIMENSION A PACK INSERTS AT — the three `pack25`/`pack26`/`pack27` reach.
///
/// ⛔ THE REFERENCE'S `DT_ERROR("Tried to use illegal pack action")` (`shuffle.cpp:375`) MADE
/// UNREPRESENTABLE: `add_valid_actions` loops `dim_16bit..=dim_64bit` (`shuffle.cpp:341`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PackDim {
    /// `dim_16bit`, using `pack27`.
    Bit16,
    /// `dim_32bit`, using `pack26`.
    Bit32,
    /// `dim_64bit`, using `pack25`.
    Bit64,
}

impl PackDim {
    /// The slice subdimension it names.
    #[must_use]
    pub const fn slice(self) -> SliceDim {
        match self {
            Self::Bit16 => SliceDim::Bit16,
            Self::Bit32 => SliceDim::Bit32,
            Self::Bit64 => SliceDim::Bit64,
        }
    }
}

/// INSERTS A STICK DIMENSION INTO THE SLICE AND SLIDES THE DIMENSIONS BELOW IT RIGHT — `pack25`,
/// `pack26`, `pack27` (`shuffle.cpp:322-326`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackAction {
    stick_dim: DimSymbol,
    slice: PackDim,
}

impl PackAction {
    /// `PackAction(stick_dim, slice_idx)` (`shuffle.cpp:347`).
    #[must_use]
    pub const fn new(stick_dim: DimSymbol, slice: PackDim) -> Self {
        Self { stick_dim, slice }
    }

    /// The stick dimension being packed in.
    #[must_use]
    pub const fn stick_dim(&self) -> DimSymbol {
        self.stick_dim
    }

    /// Which subdimension it inserts above.
    #[must_use]
    pub const fn slice(&self) -> PackDim {
        self.slice
    }

    /// Replaces: e144_act
    ///
    /// The stick dimension lands at `slice`, the subdimensions from the 8-bit slot up to it shift
    /// one place down, and what the 8-bit slot held is dropped.
    ///
    /// ⛔ THAT IS THE REFERENCE'S `insert(begin() + slice_idx + 1)` FOLLOWED BY
    /// `erase(begin() + dim_8bit)` (`shuffle.cpp:357-358`) — a pair whose net effect on the
    /// six-slot slice is this shift, which is why the slice length is an invariant and not a check.
    #[must_use]
    pub fn act(&self, input: &AbstractLayout) -> AbstractLayout {
        let mut output = input.clone();
        if !self.stick_dim.is_dummy() {
            output.stick_dims.remove(&self.stick_dim);
        }
        let to = self.slice.slice().index();
        for i in SliceDim::Bit8.index()..to {
            output.slice_dims[i] = output.slice_dims[i + 1];
        }
        output.slice_dims[to] = self.stick_dim;
        output
    }
}

// crustify:todo: e145_get_indices
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:366  (12 body lines, level 0)
//   class     : PackAction
//   original  : std::vector<int> get_indices()
//   extract   : crustify-ddc/cpp/ddc.cpp:2763-2775

// crustify:todo: e146_act
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:409  (13 body lines, level 0)
//   class     : ShiftLeftAction
//   original  : AbstractLayout act(const AbstractLayout& input) override
//   extract   : crustify-ddc/cpp/ddc.cpp:2785-2798

// crustify:todo: e147_cost
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:423  (4 body lines, level 0)
//   class     : ShiftLeftAction
//   original  : double cost(const AbstractLayout& input) override
//   extract   : crustify-ddc/cpp/ddc.cpp:2808-2812

// crustify:todo: e148_act
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:476  (10 body lines, level 0)
//   class     : Pack8Action
//   original  : AbstractLayout act(const AbstractLayout& input) override
//   extract   : crustify-ddc/cpp/ddc.cpp:2822-2832

// crustify:todo: e149_act
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:523  (8 body lines, level 0)
//   class     : Pack9Action
//   original  : AbstractLayout act(const AbstractLayout& input) override
//   extract   : crustify-ddc/cpp/ddc.cpp:2842-2850

// crustify:todo: e150_cost
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:532  (4 body lines, level 0)
//   class     : Pack9Action
//   original  : double cost(const AbstractLayout& input) override
//   extract   : crustify-ddc/cpp/ddc.cpp:2860-2864

// crustify:todo: e151_act
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:573  (11 body lines, level 0)
//   class     : Pack24Action
//   original  : AbstractLayout act(const AbstractLayout& input) override
//   extract   : crustify-ddc/cpp/ddc.cpp:2874-2885

// crustify:todo: e152__out_format
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:598  (8 body lines, level 0)
//   class     : GCVTF16F8PackAction
//   original  : static std::optional<DataFormats> _out_format(DataFormats in, const AbstractLayout& goal)
//   extract   : crustify-ddc/cpp/ddc.cpp:2895-2904

// crustify:todo: e153_act
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:620  (7 body lines, level 0)
//   class     : GCVTF16F8PackAction
//   original  : AbstractLayout act(const AbstractLayout& input) override
//   extract   : crustify-ddc/cpp/ddc.cpp:2914-2921

// crustify:todo: e154__out_format
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:644  (8 body lines, level 0)
//   class     : GCVTF16F8MergeAction
//   original  : static std::optional<DataFormats> _out_format(DataFormats in, const AbstractLayout& goal)
//   extract   : crustify-ddc/cpp/ddc.cpp:2931-2940

// crustify:todo: e155_act
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:666  (6 body lines, level 0)
//   class     : GCVTF16F8MergeAction
//   original  : AbstractLayout act(const AbstractLayout& input) override
//   extract   : crustify-ddc/cpp/ddc.cpp:2950-2956

// crustify:todo: e156_contains
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:684  (5 body lines, level 0)
//   class     : AbstractLayout
//   original  : bool AbstractLayout::contains(DimSymbol dim) const
//   extract   : crustify-ddc/cpp/ddc.cpp:2966-2971

// crustify:todo: e157_get_node
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:717  (11 body lines, level 0)
//   class     : AutoShuffler
//   original  : std::shared_ptr<GraphNode> AutoShuffler::get_node( const AbstractLayout& layout)
//   extract   : crustify-ddc/cpp/ddc.cpp:2981-2993

// crustify:todo: e158_make_stick_number_key
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:730  (14 body lines, level 0)
//   original  : auto make_stick_number_key(const std::vector<DimSymbol>& stick_ordering)
//   extract   : crustify-ddc/cpp/ddc.cpp:3002-3016

// crustify:todo: e159_all_one
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:1043  (3 body lines, level 0)
//   original  : bool all_one(std::vector<int> vec)
//   extract   : crustify-ddc/cpp/ddc.cpp:3025-3028

// crustify:todo: e160_update_worklist
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:1139  (7 body lines, level 0)
//   class     : AutoShuffler
//   original  : void AutoShuffler::update_worklist(std::shared_ptr<GraphNode> node)
//   extract   : crustify-ddc/cpp/ddc.cpp:3038-3045

// crustify:todo: e161_codegen_psuedocode
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:1224  (42 body lines, level 0)
//   class     : AutoShuffler
//   original  : std::vector<std::string> AutoShuffler::codegen_psuedocode( const ConcreteLayout& input, const ConcreteLayout& output, const std::vector<std::shared_ptr<GraphNode>>& shuffle)
//   extract   : crustify-ddc/cpp/ddc.cpp:3055-3099

// crustify:todo: e162_getDefaultSymbols
//   authority : ddc/transformations/automatic_shuffle/shuffle.h:48  (8 body lines, level 0)
//   class     : DimSymbol
//   original  : static std::vector<DimSymbol> getDefaultSymbols(int n)
//   extract   : crustify-ddc/cpp/ddc.cpp:3109-3117

// crustify:todo: e163_constexpr
//   authority : ddc/transformations/automatic_shuffle/shuffle.h:160  (3 body lines, level 0)
//   class     : std
//   original  : if constexpr (sizeof(size_t) < sizeof(uint64_t))
//   extract   : crustify-ddc/cpp/ddc.cpp:3127-3130

// crustify:todo: e164_insert_before
//   authority : ddc/transformations/automatic_shuffle/shuffle.h:175  (7 body lines, level 0)
//   class     : DataEdge
//   original  : void insert_before(dsc2::BlockNode* parent, dsc2::ScheduleNode* insert_point)
//   extract   : crustify-ddc/cpp/ddc.cpp:3140-3148

// crustify:todo: e264_bin_op
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:137  (19 body lines, level 1)
//   original  : ComputationOp bin_op(DimSymbol dim, const std::vector<int>& indices, bool expand_indices = true)
//   extract   : crustify-ddc/cpp/ddc.cpp:7856-7876
//   calls     : e139_packmerge_psuedostring

// crustify:todo: e265_unary_op
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:160  (17 body lines, level 1)
//   original  : ComputationOp unary_op(const std::vector<int>& indices)
//   extract   : crustify-ddc/cpp/ddc.cpp:7885-7902
//   calls     : e139_packmerge_psuedostring

// crustify:todo: e266_canonicalize_layout
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:690  (18 body lines, level 1)
//   class     : AutoShuffler
//   original  : AbstractLayout AutoShuffler::canonicalize_layout(const AbstractLayout& layout, const AbstractLayout& goal)
//   extract   : crustify-ddc/cpp/ddc.cpp:7912-7931
//   calls     : e156_contains

// crustify:todo: e267_reset_graph
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:710  (6 body lines, level 1)
//   class     : AutoShuffler
//   original  : void AutoShuffler::reset_graph()
//   extract   : crustify-ddc/cpp/ddc.cpp:7941-7947
//   calls     : e104_clear

// crustify:todo: e268_sort_by_stick_key
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:746  (15 body lines, level 1)
//   original  : template <bool sort_inputs> void sort_by_stick_key(const std::function<uint32_t(const StickIndex&)> key, std::vector<ComputationOp>& ops)
//   extract   : crustify-ddc/cpp/ddc.cpp:7956-7973
//   calls     : e163_constexpr

// crustify:todo: e269_check_single_inorder_accesses
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:767  (17 body lines, level 1)
//   original  : template <bool check_inputs> bool check_single_inorder_accesses( const std::function<uint32_t(const StickIndex&)> key, const std::vector<ComputationOp>& ops)
//   extract   : crustify-ddc/cpp/ddc.cpp:7982-8002
//   calls     : e163_constexpr

// crustify:todo: e270_inferLayouts
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:1047  (90 body lines, level 1)
//   class     : AutoShuffler
//   original  : std::pair<ConcreteLayout, ConcreteLayout> AutoShuffler::inferLayouts( const PrimaryDsInfo& in, const PrimaryDsInfo& out)
//   extract   : crustify-ddc/cpp/ddc.cpp:8012-8103
//   calls     : e137_int_log2, e159_all_one

// crustify:todo: e310_add_valid_actions
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:226  (11 body lines, level 2)
//   class     : MergeAction
//   original  : static void add_valid_actions(ActionList& actions, const AbstractLayout& input, const AbstractLayout& goal)
//   extract   : crustify-ddc/cpp/ddc.cpp:11521-11534
//   calls     : e232_reset

// crustify:todo: e311_enumerate_stick_computations
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:305  (15 body lines, level 2)
//   class     : MergeAction
//   original  : std::vector<ComputationOp> enumerate_stick_computations() override
//   extract   : crustify-ddc/cpp/ddc.cpp:11544-11559
//   calls     : e143_get_shuffle_indices, e264_bin_op

// crustify:todo: e312_add_valid_actions
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:332  (12 body lines, level 2)
//   class     : PackAction
//   original  : static void add_valid_actions(ActionList& actions, const AbstractLayout& input, const AbstractLayout& goal)
//   extract   : crustify-ddc/cpp/ddc.cpp:11569-11583
//   calls     : e232_reset

// crustify:todo: e313_enumerate_stick_computations
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:379  (4 body lines, level 2)
//   class     : PackAction
//   original  : std::vector<ComputationOp> enumerate_stick_computations() override
//   extract   : crustify-ddc/cpp/ddc.cpp:11593-11597
//   calls     : e145_get_indices, e264_bin_op

// crustify:todo: e314_add_valid_actions
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:398  (6 body lines, level 2)
//   class     : ShiftLeftAction
//   original  : static void add_valid_actions(ActionList& actions, const AbstractLayout& input, const AbstractLayout& goal)
//   extract   : crustify-ddc/cpp/ddc.cpp:11607-11615
//   calls     : e232_reset

// crustify:todo: e315_enumerate_stick_computations
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:428  (15 body lines, level 2)
//   class     : ShiftLeftAction
//   original  : std::vector<ComputationOp> enumerate_stick_computations() override
//   extract   : crustify-ddc/cpp/ddc.cpp:11625-11640
//   calls     : e265_unary_op

// crustify:todo: e316_add_valid_actions
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:451  (20 body lines, level 2)
//   class     : Pack8Action
//   original  : static void add_valid_actions(ActionList& actions, const AbstractLayout& input, const AbstractLayout& goal)
//   extract   : crustify-ddc/cpp/ddc.cpp:11650-11672
//   calls     : e232_reset

// crustify:todo: e317_add_valid_actions
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:501  (17 body lines, level 2)
//   class     : Pack9Action
//   original  : static void add_valid_actions(ActionList& actions, const AbstractLayout& input, const AbstractLayout& goal)
//   extract   : crustify-ddc/cpp/ddc.cpp:11682-11701
//   calls     : e232_reset

// crustify:todo: e318_add_valid_actions
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:548  (20 body lines, level 2)
//   class     : Pack24Action
//   original  : static void add_valid_actions(ActionList& actions, const AbstractLayout& input, const AbstractLayout& goal)
//   extract   : crustify-ddc/cpp/ddc.cpp:11711-11733
//   calls     : e232_reset

// crustify:todo: e319_add_valid_actions
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:607  (8 body lines, level 2)
//   class     : GCVTF16F8PackAction
//   original  : static void add_valid_actions(ActionList& actions, const AbstractLayout& input, const AbstractLayout& goal)
//   extract   : crustify-ddc/cpp/ddc.cpp:11743-11753
//   calls     : e152__out_format, e154__out_format, e232_reset

// crustify:todo: e320_add_valid_actions
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:653  (8 body lines, level 2)
//   class     : GCVTF16F8MergeAction
//   original  : static void add_valid_actions(ActionList& actions, const AbstractLayout& input, const AbstractLayout& goal)
//   extract   : crustify-ddc/cpp/ddc.cpp:11763-11773
//   calls     : e152__out_format, e154__out_format, e232_reset

// crustify:todo: e342_codegen_generic
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:910  (123 body lines, level 3)
//   class     : AutoShuffler
//   original  : template <typename EdgeType> void AutoShuffler::codegen_generic( const ConcreteLayout& input_layout, const ConcreteLayout& output_layout, const std::vector<std::shared_ptr<GraphNode>>& shuffle, const std::vector<EdgeType>& input_edges, const std::function<std::vector<EdgeType>(std::shared_ptr<GraphN
//   extract   : crustify-ddc/cpp/ddc.cpp:12356-12488
//   calls     : e104_clear, e138_swap, e140_repeat_over_dims, e158_make_stick_number_key, e311_enumerate_stick_computations, e313_enumerate_stick_computations, e315_enumerate_stick_computations

// crustify:todo: e343_get_legal_transforms
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:1147  (12 body lines, level 3)
//   class     : AutoShuffler
//   original  : std::vector<std::shared_ptr<ShuffleAction>> AutoShuffler::get_legal_transforms( const AbstractLayout& layout, const AbstractLayout& goal)
//   extract   : crustify-ddc/cpp/ddc.cpp:12498-12511
//   calls     : e310_add_valid_actions, e312_add_valid_actions, e314_add_valid_actions, e316_add_valid_actions, e317_add_valid_actions, e318_add_valid_actions, e319_add_valid_actions, e320_add_valid_actions

// crustify:todo: e362_get_shuffle
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:1161  (61 body lines, level 4)
//   class     : AutoShuffler
//   original  : std::vector<std::shared_ptr<GraphNode>> AutoShuffler::get_shuffle( const AbstractLayout& input, const AbstractLayout& output)
//   extract   : crustify-ddc/cpp/ddc.cpp:14312-14374
//   calls     : e141_act, e142_cost, e144_act, e146_act, e147_cost, e148_act, e149_act, e150_cost, e151_act, e153_act, e155_act, e157_get_node, e160_update_worklist, e266_canonicalize_layout …

// crustify:todo: e371_replace_assign
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:788  (119 body lines, level 5)
//   class     : AutoShuffler
//   original  : bool AutoShuffler::replace_assign(DesignSpaceConfig* dsc, ComputationBuilder& builder, ComputeNode* assign)
//   extract   : crustify-ddc/cpp/ddc.cpp:14744-14865
//   calls     : e270_inferLayouts, e362_get_shuffle

// ⭐ TESTS FOR ENTRIES 137-144. Union this module with this file's other test modules when they land.
#[cfg(test)]
mod tests_e137_e144 {
    use super::{
        AbstractLayout, ComputationOp, DIMS_PER_SLICE, DimSymbol, Half, MergeAction, MergeDim,
        PackAction, PackDim, PseudoReg, ShuffleCost, ShuffleIndex, StickIndex, SwapBuffer,
        int_log2, packmerge_psuedostring,
    };
    use crate::generated::DataType;
    use std::collections::BTreeSet;

    /// Symbols 1, 2, 3, 4 — `getDefaultSymbol()` and its successors.
    const A: DimSymbol = DimSymbol::DEFAULT;
    const B: DimSymbol = A.next();
    const C: DimSymbol = B.next();
    const D: DimSymbol = C.next();
    const DUMMY: DimSymbol = DimSymbol::DUMMY;

    /// A layout with the given stick dimensions and slice, in `Senint8`.
    fn layout(sticks: &[DimSymbol], slice: [DimSymbol; DIMS_PER_SLICE]) -> AbstractLayout {
        AbstractLayout::new(
            sticks.iter().copied().collect::<BTreeSet<_>>(),
            slice,
            DataType::Senint8,
        )
    }

    /// e137: the reference's documented `-1` for `x <= 0`, and the floor for everything else — the
    /// `stickSize_` powers of two `inferLayouts` feeds it (`shuffle.cpp:1072`) plus a non-power case.
    #[test]
    fn int_log2_floors_and_answers_minus_one_below_one() {
        for (x, want) in [
            (0, -1),
            (-8, -1),
            (1, 0),
            (2, 1),
            (5, 2),
            (8, 3),
            (128, 7),
            (i32::MAX, 30),
        ] {
            assert_eq!(int_log2(x), want, "int_log2({x})");
        }
    }

    /// e138: the two slots exchange, and a default-constructed buffer starts on `a` — the `flag =
    /// true` a derived `Default` would get backwards.
    #[test]
    fn swap_exchanges_the_two_slots() {
        let mut buffer: SwapBuffer<u8> = SwapBuffer::default();
        assert_eq!((*buffer.first(), *buffer.second()), (0, 0));

        let mut buffer = SwapBuffer::new("a", "b");
        assert_eq!((*buffer.first(), *buffer.second()), ("a", "b"));
        buffer.swap();
        assert_eq!((*buffer.first(), *buffer.second()), ("b", "a"));
        buffer.swap();
        assert_eq!((*buffer.first(), *buffer.second()), ("a", "b"));
    }

    /// e139: the line as the reference's stream writes it, including the space before `]` and the
    /// `-1` a don't-care lane prints.
    #[test]
    fn the_packmerge_line_is_written_as_the_reference_streams_it() {
        let indices = [ShuffleIndex(0), ShuffleIndex(16), ShuffleIndex(-1)];
        assert_eq!(
            packmerge_psuedostring(
                &indices,
                &PseudoReg("r0".to_owned()),
                &PseudoReg("r1".to_owned()),
                &PseudoReg("r2".to_owned()),
            ),
            "r2 = packmerge r0 r1 [ 0 16 -1 ]"
        );
    }

    /// e140: one absent dimension doubles the list, low half on the original and high on the copy;
    /// a dimension the OUTPUT already indexes is still split, and the insert that would overwrite
    /// its half does not.
    #[test]
    fn repeat_over_dims_doubles_per_absent_dimension() {
        let op = ComputationOp {
            inputs: vec![[(A, Half::Low)].into_iter().collect::<StickIndex>()],
            output: StickIndex::new(),
            reuses_sticks: false,
        };
        let mut appended = Vec::new();
        op.repeat_over_dims(&[A, B], &mut appended);

        assert_eq!(appended.len(), 2, "A is already indexed, B is not");
        assert_eq!(appended[0].inputs[0].half_of(B), Some(Half::Low));
        assert_eq!(appended[0].output.half_of(B), Some(Half::Low));
        assert_eq!(appended[1].inputs[0].half_of(B), Some(Half::High));
        assert_eq!(appended[1].output.half_of(B), Some(Half::High));
        assert_eq!(appended[0].inputs[0].half_of(A), Some(Half::Low));

        // The absent set comes from `inputs` alone, so B splits even though the output indexes it —
        // and both copies keep the High the output already carried.
        let with_output = ComputationOp {
            inputs: vec![[(A, Half::Low)].into_iter().collect::<StickIndex>()],
            output: [(B, Half::High)].into_iter().collect::<StickIndex>(),
            reuses_sticks: false,
        };
        let mut appended = Vec::new();
        with_output.repeat_over_dims(&[B], &mut appended);
        assert_eq!(appended.len(), 2);
        assert_eq!(appended[0].output.half_of(B), Some(Half::High));
        assert_eq!(appended[1].output.half_of(B), Some(Half::High));
        assert_eq!(appended[0].inputs[0].half_of(B), Some(Half::Low));
    }

    /// e141: the stick dimension takes the slot; `Extract` puts what it displaced into the sticks,
    /// `Fill` drops the dummy that was there.
    #[test]
    fn merge_act_swaps_the_stick_dimension_for_the_slice_slot() {
        let input = layout(&[A, B], [DUMMY, DUMMY, DUMMY, C, DUMMY, D]);

        let extract = MergeAction::new(A, MergeDim::Bit64, D);
        assert_eq!(
            extract.act(&input),
            layout(&[B, D], [DUMMY, DUMMY, DUMMY, C, DUMMY, A])
        );

        let fill = MergeAction::new(A, MergeDim::Bit32, DUMMY);
        assert_eq!(
            fill.act(&input),
            layout(&[B], [DUMMY, DUMMY, DUMMY, C, A, D])
        );
    }

    /// e142: half a stick per stick, doubled when it extracts — and 0 for a single stick, which is
    /// the reference's integer division and not 0.5.
    #[test]
    fn merge_cost_is_integer_halves_of_the_stick_count() {
        let four_sticks = layout(&[A, B], [DUMMY; DIMS_PER_SLICE]);
        let one_stick = layout(&[], [DUMMY; DIMS_PER_SLICE]);

        assert_eq!(
            MergeAction::new(A, MergeDim::Bit64, DUMMY).cost(&four_sticks),
            ShuffleCost(2.0)
        );
        assert_eq!(
            MergeAction::new(A, MergeDim::Bit64, D).cost(&four_sticks),
            ShuffleCost(4.0)
        );
        assert_eq!(
            MergeAction::new(A, MergeDim::Bit64, DUMMY).cost(&one_stick),
            ShuffleCost(0.0)
        );
    }

    /// e143: `merge8h` and `merge64l` as `shuffle.cpp:30` and `:53` write them, so a transposed
    /// table is a failing test rather than a wrong instruction.
    #[test]
    fn the_shuffle_tables_are_the_vendors_own() {
        let merge8 = MergeAction::new(A, MergeDim::Bit8, DUMMY);
        assert_eq!(
            merge8
                .get_shuffle_indices(Half::High)
                .iter()
                .map(|i| i.0)
                .collect::<Vec<_>>(),
            vec![1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31]
        );
        let merge64 = MergeAction::new(A, MergeDim::Bit64, DUMMY);
        assert_eq!(
            merge64
                .get_shuffle_indices(Half::Low)
                .iter()
                .map(|i| i.0)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 8, 9, 10, 11]
        );
        assert_eq!(
            merge64.get_shuffle_indices(Half::High),
            [4, 5, 6, 7, 12, 13, 14, 15].map(ShuffleIndex).as_slice()
        );
    }

    /// e144: the slot at 8 bits goes, everything up to the insert point slides one down, and the
    /// stick dimension lands at the insert point.
    #[test]
    fn pack_act_slides_the_slice_down_into_the_eight_bit_slot() {
        let input = layout(&[A, B], [C, D, DUMMY, A, B, C]);

        assert_eq!(
            PackAction::new(A, PackDim::Bit16).act(&input),
            layout(&[B], [C, D, A, A, B, C])
        );
        assert_eq!(
            PackAction::new(A, PackDim::Bit64).act(&input),
            layout(&[B], [C, D, A, B, C, A])
        );
        assert_eq!(
            PackAction::new(DUMMY, PackDim::Bit32).act(&input),
            layout(&[A, B], [C, D, A, B, DUMMY, C])
        );
    }
}
