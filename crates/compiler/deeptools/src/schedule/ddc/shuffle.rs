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
use crate::formats::DataFormat;
use crate::schedule::dsc2::{AllocateNode, ComputeNode, DataInfo};
use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::hash::{Hash, Hasher};
use sys_arch_spec::arch_enums::SenComponent;

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

    /// Replaces: e162_getDefaultSymbols
    ///
    /// `n` FRESH SYMBOLS, `1..=n` (`shuffle.h:48`) — unique within the run and, as the reference's
    /// own comment says, free to collide with symbols minted by any other call.
    ///
    /// ⛔ NONE OF THEM IS THE DUMMY, and `n <= 0` mints none: the loop starts at 1, so a caller
    /// wanting a vacant slot asks [`Self::DUMMY`] for it (`shuffle_standalone.cpp:74`).
    #[must_use]
    pub fn default_symbols(n: i32) -> Vec<Self> {
        (1..=n).map(Self).collect()
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
/// A LAYOUT WITH ITS STICK DIMENSIONS ORDERED — `ConcreteLayout` (`shuffle.h:70`), which is what
/// `inferLayouts` (e270) states and what the codegen walk numbers its sticks from.
///
/// ⛔ THE STICK ORDER IS THE WHOLE DIFFERENCE FROM [`AbstractLayout`]: `make_stick_number_key` gives
/// position `i` bit `i`, so a concrete layout says WHICH stick an index denotes, where the
/// abstraction says only which dimensions are taken across sticks.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConcreteLayout {
    /// `stick_dims` — the numbering order.
    pub stick_dims: Vec<DimSymbol>,
    /// `slice_dims`, ordered by significance.
    pub slice_dims: Vec<DimSymbol>,
}

impl ConcreteLayout {
    /// `int_pow2(stick_dims.size())` (`shuffle.cpp:105`) — the stick count `codegen_generic`
    /// `DT_CHECK`s its input edge list against (`shuffle.cpp:929-930`).
    #[must_use]
    pub fn num_sticks(&self) -> Sticks {
        Sticks(1u64 << self.stick_dims.len())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct AbstractLayout {
    /// The element format the sticks are in.
    pub format: DataFormat,
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
        format: DataFormat,
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

    /// Replaces: e156_contains
    ///
    /// WHETHER THIS LAYOUT HOLDS `dim` AT ALL, in the sticks or anywhere in the slice — what
    /// `canonicalize_layout` (e266) asks a goal before keeping one of the input's dimensions.
    ///
    /// ⛔ THE DUMMY IS CONTAINED whenever any slice slot is vacant, and every layout with a vacant
    /// slot answers `true` for it. `canonicalize_layout` never asks about the dummy.
    #[must_use]
    pub fn contains(&self, dim: DimSymbol) -> bool {
        self.stick_dims.contains(&dim) || self.slice_dims.contains(&dim)
    }

    /// `std::hash<AbstractLayout>::operator()` (`shuffle.h:135`) — FNV-1a over the stick symbols in
    /// set order and then the six slice symbols, a byte of each `int` at a time, low byte first.
    ///
    /// ⛔ THE FORMAT IS NOT IN IT, while `operator==` compares it (`shuffle.h:101`), so two layouts
    /// differing only in format hash equal. Carried as the reference has it: a collision is legal
    /// for a hash, and adding the format would move every layout to a different bucket.
    /// ⛔ AND IT HAS NO USER IN THIS REVISION — `layout_to_nodes` is a `std::map` and the only
    /// unordered container in the file is keyed on [`DimSymbol`] (`shuffle.cpp:732`).
    #[must_use]
    pub fn fnv1a(&self) -> usize {
        /// `const uint64_t offset` (`shuffle.h:139`).
        const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        /// `const uint64_t prime` (`shuffle.h:140`).
        const PRIME: u64 = 0x0000_0100_0000_01b3;

        let mut hash = OFFSET;
        for symbol in self.stick_dims.iter().chain(&self.slice_dims) {
            // `hash_int` — FNV-1a wants bytes, so each symbol goes in a byte at a time.
            for byte in symbol.0.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(PRIME);
            }
        }
        narrow_to_usize(hash)
    }
}

/// Replaces: e163_constexpr
///
/// FOLDS THE 64-BIT FNV RESULT INTO A `size_t` (`shuffle.h:160`): on a narrow host the high half is
/// xored down first so the whole hash's entropy survives the truncation, and on a wide one the value
/// passes through.
///
/// ⭐ STILL A COMPILE-TIME CHOICE — `size_of` is const exactly as `sizeof` is, so one arm folds away
/// rather than being tested per hash.
const fn narrow_to_usize(hash: u64) -> usize {
    if size_of::<usize>() < size_of::<u64>() {
        (hash ^ (hash >> 32)) as usize
    } else {
        hash as usize
    }
}

/// `std::hash<AbstractLayout>` (`shuffle.h:135`) as Rust states it. The C++ RETURNS the `size_t`;
/// Rust feeds a hasher, so [`AbstractLayout::fnv1a`] is written whole and there is one definition of
/// the layout hash rather than two.
impl Hash for AbstractLayout {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self.fnv1a());
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
/// `pack25` (`shuffle.cpp:42`).
const PACK25: [ShuffleIndex; 16] =
    indices([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]);
/// `pack26` (`shuffle.cpp:44`).
const PACK26: [ShuffleIndex; 16] =
    indices([0, 2, 4, 6, 16, 18, 20, 22, 8, 10, 12, 14, 24, 26, 28, 30]);
/// `pack27` (`shuffle.cpp:46`).
const PACK27: [ShuffleIndex; 16] =
    indices([0, 2, 16, 18, 4, 6, 20, 22, 8, 10, 24, 26, 12, 14, 28, 30]);

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
    /// `codegen_psuedocode`, the `std::function` (`shuffle.h:213`), AS DATA: both setters emit ONE
    /// `packmerge` line from an index table and differ only in whether the second input register is
    /// the first one again (`shuffle.cpp:147-153`, `:169-175`). Absent until e264/e265 set it.
    pub packmerge_indices: Option<Vec<ShuffleIndex>>,
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

    /// `op.codegen_psuedocode(in_reg_names, out_reg_name)` (`shuffle.h:213`) — the one `packmerge`
    /// line this op's table writes, and nothing at all for an op that has no table yet.
    ///
    /// ⛔ THE SECOND REGISTER IS THE FIRST ONE AGAIN for a one-input op, which is `unary_op`'s own
    /// `insert_packmerge(input[0], input[0], ..)` (`shuffle.cpp:174`): the reference's two arity
    /// `DT_CHECK`s ARE that choice, so neither is a stop here.
    #[must_use]
    pub fn codegen_psuedocode(&self, in_regs: &[PseudoReg], out_reg: &PseudoReg) -> Vec<String> {
        match (&self.packmerge_indices, in_regs) {
            (Some(indices), [first, rest @ ..]) => vec![packmerge_psuedostring(
                indices,
                first,
                rest.first().unwrap_or(first),
                out_reg,
            )],
            _ => vec![],
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

    /// Replaces: e145_get_indices
    ///
    /// The `packmerge` table this pack drives the vector unit with, one per reachable
    /// subdimension — `pack25` at the 64-bit slot, `pack26` at the 32-bit, `pack27` at the 16-bit.
    ///
    /// ⛔ THE REFERENCE'S `DT_ERROR("Tried to use illegal pack action")` (`shuffle.cpp:375`) HAS NO
    /// ARM HERE BECAUSE [`PackDim`] CANNOT NAME THE THREE ILLEGAL SLOTS.
    #[must_use]
    pub const fn get_indices(&self) -> &'static [ShuffleIndex] {
        match self.slice {
            PackDim::Bit64 => &PACK25,
            PackDim::Bit32 => &PACK26,
            PackDim::Bit16 => &PACK27,
        }
    }
}

/// SHIFTS THE SLICE DIMENSIONS ONE PLACE UP, dropping the 64-bit subdimension or extracting it back
/// into the sticks — `pack12` and `pack13` (`shuffle.cpp:385-392`).
///
/// ⛔ `extract_dim` IS THE INPUT'S OWN 64-BIT SLOT. Its one construction site passes
/// `input.sliceDims()[dim_64bit]` (`shuffle.cpp:404`), which is what the `DT_CHECK` at `:410-411`
/// restates; [`Self::act`] therefore reads that slot instead of trusting the field to match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShiftLeftAction {
    extract_dim: DimSymbol,
}

impl ShiftLeftAction {
    /// `ShiftLeftAction(extract_dim)` (`shuffle.cpp:407`).
    #[must_use]
    pub const fn new(extract_dim: DimSymbol) -> Self {
        Self { extract_dim }
    }

    /// The dimension it lifts out of the 64-bit slot — dummy when that slot holds nothing live, in
    /// which case the shift discards it instead.
    #[must_use]
    pub const fn extract_dim(&self) -> DimSymbol {
        self.extract_dim
    }

    /// Replaces: e146_act
    ///
    /// Every subdimension from the 8-bit slot up moves one place up, the 8-bit slot becomes a dummy,
    /// and the dimension pushed out of the 64-bit slot joins the stick dimensions when it is live.
    ///
    /// ⛔ THAT IS `erase(begin() + dim_64bit)` FOLLOWED BY `insert(begin() + dim_8bit, getDummy())`
    /// (`shuffle.cpp:418-419`): the pair restores the six-slot length, so this shifts rather than
    /// resizes. ⛔ AND THE STICK GAINS THE SLOT'S DIMENSION, not `extract_dim` — the same symbol by
    /// the construction site above, and read from the layout so it stays so.
    #[must_use]
    pub fn act(&self, input: &AbstractLayout) -> AbstractLayout {
        let mut output = input.clone();
        if !self.extract_dim.is_dummy() {
            output.stick_dims.insert(input.slice_dim(SliceDim::Bit64));
        }
        for i in (SliceDim::Bit8.index() + 1..=SliceDim::Bit64.index()).rev() {
            output.slice_dims[i] = output.slice_dims[i - 1];
        }
        output.slice_dims[SliceDim::Bit8.index()] = DimSymbol::DUMMY;
        output
    }

    /// Replaces: e147_cost
    ///
    /// One instruction per input stick, DOUBLED when the shift also extracts — it then writes two
    /// output sticks per input.
    ///
    /// ⛔ NO DIVISION, UNLIKE EVERY OTHER ACTION'S COST (`shuffle.cpp:423-425`). A one-stick layout
    /// costs 1 or 2 here, where a merge or a pack costs 0.
    #[must_use]
    pub fn cost(&self, input: &AbstractLayout) -> ShuffleCost {
        let sticks = input.num_sticks().0;
        if self.extract_dim.is_dummy() {
            ShuffleCost(sticks as f64)
        } else {
            ShuffleCost((sticks * 2) as f64)
        }
    }
}

/// THE `pack8` INSTRUCTION — takes a stick dimension into the slice, drops the 4- and 8-bit
/// subdimensions and slides the rest down two places (`shuffle.cpp:446`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pack8Action {
    dim: DimSymbol,
}

impl Pack8Action {
    /// `Pack8Action(stick_dim)` (`shuffle.cpp:474`).
    #[must_use]
    pub const fn new(dim: DimSymbol) -> Self {
        Self { dim }
    }

    /// The stick dimension being packed in.
    #[must_use]
    pub const fn dim(&self) -> DimSymbol {
        self.dim
    }

    /// Replaces: e148_act
    ///
    /// The 16-, 32- and 64-bit dimensions drop two slots, `dim` lands in the 32-bit slot, and the
    /// 64-bit slot becomes a dummy. The 2-bit slot is untouched.
    ///
    /// ⛔ THE TWO ERASES RUN DESCENDING — `{dim_8bit, dim_4bit}` (`shuffle.cpp:480`) — on a
    /// shrinking vector, so they drop exactly those two originals; ascending would have taken the
    /// 8-bit slot and then whatever slid into the 4-bit one. `insert(end(), {dim, getDummy()})`
    /// restores the six-slot length.
    #[must_use]
    pub fn act(&self, input: &AbstractLayout) -> AbstractLayout {
        let mut output = input.clone();
        output.stick_dims.remove(&self.dim);
        output.slice_dims[SliceDim::Bit4.index()] = input.slice_dim(SliceDim::Bit16);
        output.slice_dims[SliceDim::Bit8.index()] = input.slice_dim(SliceDim::Bit32);
        output.slice_dims[SliceDim::Bit16.index()] = input.slice_dim(SliceDim::Bit64);
        output.slice_dims[SliceDim::Bit32.index()] = self.dim;
        output.slice_dims[SliceDim::Bit64.index()] = DimSymbol::DUMMY;
        output
    }
}

/// THE `pack9` INSTRUCTION — takes a stick dimension into the 4-bit subdimension in place
/// (`shuffle.cpp:496`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pack9Action {
    dim: DimSymbol,
}

impl Pack9Action {
    /// `Pack9Action(stick_dim)` (`shuffle.cpp:521`) — its construction site passes the GOAL's 4-bit
    /// dimension, having checked the sticks hold it (`shuffle.cpp:512-518`).
    #[must_use]
    pub const fn new(dim: DimSymbol) -> Self {
        Self { dim }
    }

    /// The stick dimension being packed in.
    #[must_use]
    pub const fn dim(&self) -> DimSymbol {
        self.dim
    }

    /// Replaces: e149_act
    ///
    /// `dim` takes the 4-bit slot and the 8-bit slot becomes a dummy.
    ///
    /// ⛔ NO SLIDE, UNLIKE `pack8` AND `pack24`: this is an assignment pair on the existing slots
    /// (`shuffle.cpp:527-528`), so the dimensions above the 8-bit slot keep their places.
    #[must_use]
    pub fn act(&self, input: &AbstractLayout) -> AbstractLayout {
        let mut output = input.clone();
        output.stick_dims.remove(&self.dim);
        output.slice_dims[SliceDim::Bit4.index()] = self.dim;
        output.slice_dims[SliceDim::Bit8.index()] = DimSymbol::DUMMY;
        output
    }

    /// Replaces: e150_cost
    ///
    /// One instruction per two input sticks.
    ///
    /// ⛔ SPELLED `int_pow2(input.stickDims().size())` (`shuffle.cpp:533`) WHERE EVERY OTHER COST
    /// SPELLS THE SAME NUMBER `numSticks()` — both are `1 << |stick dims|`, so this is not a second
    /// quantity. ⛔ AND THE DIVISION IS INTEGER: a one-stick layout costs 0, not 0.5.
    #[must_use]
    pub fn cost(&self, input: &AbstractLayout) -> ShuffleCost {
        ShuffleCost((input.num_sticks().0 / 2) as f64)
    }
}

/// THE `pack24` INSTRUCTION — takes a stick dimension into the slice, drops the 2-, 4- and 8-bit
/// subdimensions and slides the rest down three places (`shuffle.cpp:543`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pack24Action {
    dim: DimSymbol,
}

impl Pack24Action {
    /// `Pack24Action(stick_dim)` (`shuffle.cpp:571`).
    #[must_use]
    pub const fn new(dim: DimSymbol) -> Self {
        Self { dim }
    }

    /// The stick dimension being packed in.
    #[must_use]
    pub const fn dim(&self) -> DimSymbol {
        self.dim
    }

    /// Replaces: e151_act
    ///
    /// The 16-, 32- and 64-bit dimensions drop three slots to the bottom of the slice, `dim` lands
    /// in the 16-bit slot, and the 32- and 64-bit slots become dummies.
    ///
    /// ⛔ THE THREE ERASES RUN DESCENDING — `{dim_8bit, dim_4bit, dim_2bit}` (`shuffle.cpp:577`) —
    /// so they drop exactly the bottom three originals, and `insert(end(), {dim, getDummy(),
    /// getDummy()})` restores the six-slot length. ⛔ TWO dummies, not one: `pack24` frees three
    /// slots and fills only one.
    #[must_use]
    pub fn act(&self, input: &AbstractLayout) -> AbstractLayout {
        let mut output = input.clone();
        output.stick_dims.remove(&self.dim);
        output.slice_dims[SliceDim::Bit2.index()] = input.slice_dim(SliceDim::Bit16);
        output.slice_dims[SliceDim::Bit4.index()] = input.slice_dim(SliceDim::Bit32);
        output.slice_dims[SliceDim::Bit8.index()] = input.slice_dim(SliceDim::Bit64);
        output.slice_dims[SliceDim::Bit16.index()] = self.dim;
        output.slice_dims[SliceDim::Bit32.index()] = DimSymbol::DUMMY;
        output.slice_dims[SliceDim::Bit64.index()] = DimSymbol::DUMMY;
        output
    }
}

/// THE `gcvt` PACK — the one action family that CHANGES THE FORMAT, converting fp16 sticks to the
/// goal's fp8 while packing a stick dimension into the slice (`shuffle.cpp:593`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GCVTF16F8PackAction {
    dim: DimSymbol,
}

impl GCVTF16F8PackAction {
    /// `GCVTF16F8PackAction(stick_dim)` (`shuffle.cpp:618`).
    #[must_use]
    pub const fn new(dim: DimSymbol) -> Self {
        Self { dim }
    }

    /// The stick dimension being packed in.
    #[must_use]
    pub const fn dim(&self) -> DimSymbol {
        self.dim
    }

    /// Replaces: e152__out_format
    ///
    /// THE FORMAT THIS ACTION LEAVES THE STICKS IN — the goal's own fp8 format when the input is an
    /// fp16 and the goal an fp8, and nothing at all otherwise, which is how `add_valid_actions`
    /// (e319) decides whether to offer the action.
    ///
    /// ⛔ BOTH SIDES ARE TWO FORMATS WIDE (`is_any_of`, `shuffle.cpp:600-602`), and TWO OF THE FOUR
    /// ARE OUTSIDE OUR TEMPLATE CENSUS — `IEEE_FP16` and `SEN152_FP8` have no
    /// [`crate::generated::DataType`], which is why the layout carries a
    /// [`DataFormat`](crate::formats::DataFormat). Narrowing to the census would silently stop
    /// offering the conversion for a torch-fp16 input.
    #[must_use]
    pub const fn out_format(input: DataFormat, goal: &AbstractLayout) -> Option<DataFormat> {
        match (input, goal.format) {
            (
                DataFormat::IeeeFp16 | DataFormat::Sen169Fp16,
                DataFormat::Sen143Fp8 | DataFormat::Sen152Fp8,
            ) => Some(goal.format),
            _ => None,
        }
    }

    /// Replaces: e153_act
    ///
    /// The 8-bit slot is dropped, the dimensions above it slide one place down, and `dim` lands in
    /// the freed 64-bit slot.
    ///
    /// ⛔ `act` DOES NOT CONVERT THE FORMAT — [`Self::out_format`] is the separate answer its caller
    /// applies, and the output layout here keeps the input's format (`shuffle.cpp:620-625`).
    #[must_use]
    pub fn act(&self, input: &AbstractLayout) -> AbstractLayout {
        let mut output = input.clone();
        output.stick_dims.remove(&self.dim);
        for i in SliceDim::Bit8.index()..SliceDim::Bit64.index() {
            output.slice_dims[i] = output.slice_dims[i + 1];
        }
        output.slice_dims[SliceDim::Bit64.index()] = self.dim;
        output
    }
}

// ═══ e153..e160 — THE GCVT MERGE, THE MEMOISED GRAPH AND ITS WORKLIST ════════════════════════════

/// THE `gcvt` MERGE — converts fp16 sticks to the goal's fp8 while taking a stick dimension into
/// the 8-bit subdimension, moving nothing else (`shuffle.cpp:638`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GCVTF16F8MergeAction {
    dim: DimSymbol,
}

impl GCVTF16F8MergeAction {
    /// `GCVTF16F8MergeAction(stick_dim)` (`shuffle.cpp:661`).
    #[must_use]
    pub const fn new(dim: DimSymbol) -> Self {
        Self { dim }
    }

    /// The stick dimension being merged in.
    #[must_use]
    pub const fn dim(&self) -> DimSymbol {
        self.dim
    }

    /// Replaces: e154__out_format
    ///
    /// THE FORMAT THIS ACTION LEAVES THE STICKS IN — the goal's own fp8 when the input is an fp16
    /// and the goal an fp8, and nothing otherwise, which is how `add_valid_actions` (e320) decides
    /// whether to offer the action.
    ///
    /// ⛔ CHARACTER-FOR-CHARACTER THE PACK ACTION'S PREDICATE (`shuffle.cpp:644-651` against
    /// `:598-605`), and a separate `static` in the reference because each action overrides its own
    /// virtual `out_format`. Kept as a second function so a divergence upstream stays expressible.
    #[must_use]
    pub const fn out_format(input: DataFormat, goal: &AbstractLayout) -> Option<DataFormat> {
        match (input, goal.format) {
            (
                DataFormat::IeeeFp16 | DataFormat::Sen169Fp16,
                DataFormat::Sen143Fp8 | DataFormat::Sen152Fp8,
            ) => Some(goal.format),
            _ => None,
        }
    }

    /// Replaces: e155_act
    ///
    /// `dim` OVERWRITES the 8-bit slot and every other slot keeps its place.
    ///
    /// ⛔ THAT ONE ASSIGNMENT IS THE WHOLE DIFFERENCE FROM [`GCVTF16F8PackAction::act`], which
    /// erases the slot and slides (`shuffle.cpp:666-670`). Its `add_valid_actions` requires the slot
    /// to be a dummy, so nothing live is overwritten. ⛔ AND IT DOES NOT CONVERT THE FORMAT.
    #[must_use]
    pub fn act(&self, input: &AbstractLayout) -> AbstractLayout {
        let mut output = input.clone();
        output.stick_dims.remove(&self.dim);
        output.slice_dims[SliceDim::Bit8.index()] = self.dim;
        output
    }
}

/// WHICH NODE OF THE SHUFFLE GRAPH — an index into [`AutoShuffler`]'s arena, standing in for the
/// reference's `std::shared_ptr<GraphNode>` (`shuffle.h:264`).
///
/// ⛔ A HANDLE, NOT AN OWNER: the reference hands the same `shared_ptr` to the map, the worklist and
/// every node's `previous_node`, and relaxation MUTATES the node all three see. Copying a node into
/// the worklist would freeze the graph as it stood at the push.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GraphNodeId(usize);

/// ONE LAYOUT REACHED BY THE SEARCH, with Dijkstra's bookkeeping (`shuffle.h:238-256`).
///
/// ⛔ `prev_action` IS REDUCED HERE: the reference's `std::shared_ptr<ShuffleAction>` needs the
/// action family as one type, which e343 mints. Nothing this batch fills reads it.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphNode {
    /// The layout this node stands for.
    pub layout: AbstractLayout,
    /// The node this one was reached from, absent at the origin.
    pub previous_node: Option<GraphNodeId>,
    /// Best cost known from the origin to here — `INFINITY` until first relaxed.
    pub cost_origin_to_here: ShuffleCost,
    /// The A* heuristic's estimate from here to the goal.
    pub heuristic_cost_here_to_goal: ShuffleCost,
    /// Whether the node has been pulled from the worklist and settled.
    pub finalized: bool,
}

impl GraphNode {
    /// `GraphNode(layout)` (`shuffle.h:247`) — unreached, so its cost is infinite.
    #[must_use]
    pub const fn new(layout: AbstractLayout) -> Self {
        Self {
            layout,
            previous_node: None,
            cost_origin_to_here: ShuffleCost(f64::INFINITY),
            heuristic_cost_here_to_goal: ShuffleCost(0.0),
            finalized: false,
        }
    }

    /// `estimated_cost()` (`shuffle.h:253`) — A*'s `f = g + h`.
    #[must_use]
    pub fn estimated_cost(&self) -> ShuffleCost {
        ShuffleCost(self.cost_origin_to_here.0 + self.heuristic_cost_here_to_goal.0)
    }
}

/// WHEN AN ENTRY WAS PUSHED — the reference's monotonic `worklist_counter` (`shuffle.h:268`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct WorklistSeq(u64);

/// ONE QUEUED VISIT — `std::tuple<double, uint64_t, shared_ptr<GraphNode>>` (`shuffle.h:266`).
///
/// ⛔ ORDERED BY `total_cmp` ON THE FROZEN COST, THEN BY SEQUENCE, BECAUSE THE COSTS INCLUDE
/// `INFINITY`: `f64`'s `PartialOrd` is not a total order, and a [`std::collections::BinaryHeap`]
/// silently misbehaves on one. The reference's `std::greater<tuple>` compares the `double` directly
/// and gets away with it only because no NaN reaches the queue.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorklistEntry {
    /// The cost as it stood when the entry was pushed.
    pub cost: ShuffleCost,
    /// The push order, which breaks cost ties deterministically.
    pub seq: WorklistSeq,
    /// The node to visit.
    pub node: GraphNodeId,
}

impl Eq for WorklistEntry {}

impl Ord for WorklistEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost
            .0
            .total_cmp(&other.cost.0)
            .then(self.seq.cmp(&other.seq))
    }
}

impl PartialOrd for WorklistEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// DIJKSTRA OVER LAYOUTS — the search that turns one abstract layout into another
/// (`shuffle.h:258`).
#[derive(Debug, Default)]
pub struct AutoShuffler {
    /// The node arena. `GraphNodeId` indexes it, and nodes are never removed.
    nodes: Vec<GraphNode>,
    /// `layout_to_nodes` — ordered, as the reference notes, for deterministic iteration.
    layout_to_nodes: BTreeMap<AbstractLayout, GraphNodeId>,
    /// The min-priority worklist. `Reverse` makes the max-heap a min-heap.
    worklist: BinaryHeap<Reverse<WorklistEntry>>,
    /// `worklist_counter` — the next sequence number to hand out.
    worklist_counter: WorklistSeq,
}

impl AutoShuffler {
    /// `AutoShuffler() = default` (`shuffle.h:318`).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// One node of the graph.
    #[must_use]
    pub fn node(&self, id: GraphNodeId) -> &GraphNode {
        &self.nodes[id.0]
    }

    /// One node of the graph, to relax.
    pub fn node_mut(&mut self, id: GraphNodeId) -> &mut GraphNode {
        &mut self.nodes[id.0]
    }

    /// How many distinct layouts the search has manifested.
    #[must_use]
    pub fn graph_len(&self) -> usize {
        self.nodes.len()
    }

    /// The cheapest queued entry, by frozen cost then push order.
    pub fn pop_worklist(&mut self) -> Option<WorklistEntry> {
        self.worklist.pop().map(|Reverse(entry)| entry)
    }

    /// How many entries are queued — several per node, as the reference's own note explains.
    #[must_use]
    pub fn worklist_len(&self) -> usize {
        self.worklist.len()
    }

    /// Replaces: e157_get_node
    ///
    /// THE NODE FOR THIS LAYOUT, manifesting it on first sight and returning the existing one after
    /// — the memoisation that makes the layout, not the path, the graph's identity.
    ///
    /// ⛔ NEVER RESETS A NODE IT FINDS. A layout reached a second time keeps the cost, predecessor
    /// and `finalized` flag its earlier relaxations wrote; re-manifesting it would lose the search's
    /// whole state for that layout (`shuffle.cpp:717-727`).
    pub fn get_node(&mut self, layout: &AbstractLayout) -> GraphNodeId {
        if let Some(&id) = self.layout_to_nodes.get(layout) {
            return id;
        }
        let id = GraphNodeId(self.nodes.len());
        self.nodes.push(GraphNode::new(layout.clone()));
        self.layout_to_nodes.insert(layout.clone(), id);
        id
    }

    /// Replaces: e160_update_worklist
    ///
    /// QUEUES A VISIT AT THE NODE'S COST AS IT STANDS NOW, with the next sequence number.
    ///
    /// ⛔ THE COST IS COPIED, NOT READ THROUGH THE NODE, and the reference says why: relaxation
    /// mutates the node after the push, and the queue's order has to stay the order it was built
    /// with (`shuffle.cpp:1139-1145`). ⛔ AND IT NEVER REPLACES AN ENTRY — with no `reduce_key` a
    /// node is queued once per relaxation, and `finalized` is what discards the stale visits.
    pub fn update_worklist(&mut self, node: GraphNodeId) {
        let cost = self.node(node).estimated_cost();
        let seq = self.worklist_counter;
        self.worklist_counter = WorklistSeq(seq.0 + 1);
        self.worklist
            .push(Reverse(WorklistEntry { cost, seq, node }));
    }
}

/// WHICH BIT OF A STICK NUMBER ONE DIMENSION OWNS — the reference's `uint32_t` mask value
/// (`shuffle.cpp:732`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StickMask(u32);

/// WHICH STICK OF A LAYOUT — the number a [`StickIndex`] denotes under one stick ordering
/// (`shuffle.cpp:736-741`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StickNumber(pub u32);

/// THE NUMBERING ITSELF — the reference's captured `masks` closure, as a value (`shuffle.cpp:730`).
pub struct StickNumberKey {
    masks: BTreeMap<DimSymbol, StickMask>,
}

impl StickNumberKey {
    /// The stick this index denotes: the OR of the bits of the dimensions it takes the HIGH half of.
    ///
    /// ⛔ ONE DELIBERATE DIVERGENCE: `masks.at(x.dim)` THROWS for a dimension outside the ordering,
    /// where this contributes no bit. Identical for every input meeting the reference's own
    /// precondition — the ordering is the layout's own stick dimensions — and no stop otherwise.
    #[must_use]
    pub fn key(&self, index: &StickIndex) -> StickNumber {
        let mut key = 0u32;
        for (dim, half) in index.iter() {
            if half == Half::High {
                if let Some(mask) = self.masks.get(&dim) {
                    key |= mask.0;
                }
            }
        }
        StickNumber(key)
    }
}

/// Replaces: e158_make_stick_number_key
///
/// NUMBERS THE STICK DIMENSIONS, position `i` of the ordering owning bit `i`, and hands back the
/// numbering that turns a [`StickIndex`] into a [`StickNumber`].
///
/// ⛔ THE REFERENCE'S `DT_CHECK(stick_ordering.size() < 32)` (`shuffle.cpp:731`) IS THE `zip`: a
/// 32nd dimension gets no bit rather than shifting a `1u` off the end of a `uint32_t`. ⛔ AND A
/// REPEATED DIMENSION KEEPS ITS LAST BIT, as `masks[dim] = ...` assigns rather than inserts.
#[must_use]
pub fn make_stick_number_key(stick_ordering: &[DimSymbol]) -> StickNumberKey {
    let mut masks = BTreeMap::new();
    for (&dim, bit) in stick_ordering.iter().zip(0u32..32) {
        masks.insert(dim, StickMask(1 << bit));
    }
    StickNumberKey { masks }
}

/// HOW MANY TIMES ONE STICK DIMENSION IS REPLICATED — `PrimaryDsInfo::stickRepl_`'s element, where
/// 1 means not replicated at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StickRepl(pub i32);

/// Replaces: e159_all_one
///
/// WHETHER NOTHING IS REPLICATED — what `inferLayouts` (e270) `DT_CHECK`s of both its ends before
/// it will describe them as layouts (`shuffle.cpp:1043-1045`).
///
/// ⛔ VACUOUSLY TRUE FOR AN EMPTY LIST, which is `std::all_of`'s answer too.
#[must_use]
pub fn all_one(stick_repl: &[StickRepl]) -> bool {
    stick_repl.iter().all(|repl| repl.0 == 1)
}

// ───────────────────────────────────────────────────────────────────────────────────────────────
// THE CODEGEN WALK AND ITS PSEUDOCODE INSTANTIATION — `shuffle.cpp:910-1040`, `:1224-1266`.
// ───────────────────────────────────────────────────────────────────────────────────────────────

/// WHICH STICK EACH OPERAND OF ONE COMPUTATION IS — `do_op_codegen`'s `input_output_stick_id`, a
/// `std::pair<std::vector<int>, int>` (`shuffle.cpp:1003`) whose ints are the keys
/// [`StickNumberKey::key`] hands back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StickIds {
    /// `.first`, one per input in the op's own input order.
    pub inputs: Vec<StickNumber>,
    /// `.second`.
    pub output: StickNumber,
}

/// THE TWO `std::function`s THE WALK IS DRIVEN BY, WITH ITS `EdgeType` — `codegen_generic`'s last
/// two parameters as one implementable thing (`shuffle.cpp:914-920`). [`Pseudocode`] is the
/// `std::string` instantiation; e371 `replace_assign` is the [`DataEdge`] one.
pub trait ShuffleCodegen {
    /// `EdgeType`.
    type Edge;

    /// `get_edges_for_node(node)` — the edges this node's output sticks are written to.
    ///
    /// ⭐ THE LAYOUT, NOT THE NODE: both instantiations read `node->layout` and nothing else of it
    /// (`shuffle.cpp:825`, `:1244`), so the hook does not need the node handle.
    fn edges_for_node(&mut self, layout: &AbstractLayout) -> Vec<Self::Edge>;

    /// `do_op_codegen(op, inputs, output, input_output_stick_id, output_added)`.
    fn op_codegen(
        &mut self,
        op: &ComputationOp,
        inputs: &[Self::Edge],
        output: &mut Self::Edge,
        sticks: &StickIds,
        output_added: bool,
    );
}

/// THE CODEGEN WALK — `AutoShuffler::codegen_generic<EdgeType>` (`shuffle.cpp:910`), which is
/// `e342_codegen_generic`, LEVEL 3 and not this batch, so it is NAMED here rather than called.
///
/// ⭐ A TRAIT BECAUSE THE REFERENCE'S IS A MEMBER TEMPLATE: e342 lands as `impl CodegenGeneric for
/// AutoShuffler` with `Node = GraphNodeId`, and [`Self::codegen_psuedocode`] — e161, one of the
/// template's two instantiations — becomes a method on the shuffler with no signature change.
pub trait CodegenGeneric {
    /// An element of `shuffle` — the reference's `std::shared_ptr<GraphNode>`, which is
    /// [`GraphNodeId`] now that the nodes live in [`AutoShuffler`]'s arena.
    type Node;

    /// `codegen_generic(input_layout, output_layout, shuffle, input_edges, get_edges_for_node,
    /// do_op_codegen)`.
    fn codegen_generic<C: ShuffleCodegen>(
        &mut self,
        input_layout: &ConcreteLayout,
        output_layout: &ConcreteLayout,
        shuffle: &[Self::Node],
        input_edges: &[C::Edge],
        codegen: &mut C,
    );

    /// Replaces: e161_codegen_psuedocode
    ///
    /// THE PSEUDOCODE INSTANTIATION OF THE WALK (`shuffle.cpp:1224`): one register per input stick,
    /// then the walk drives [`Pseudocode`], whose hooks mint each node's registers and collect each
    /// op's own line.
    ///
    /// ⛔ ONE COUNTER FOR ALL OF IT — both lambdas capture `reg_id` BY REFERENCE, so a node's
    /// registers continue the input registers' numbering instead of restarting at `r0`.
    fn codegen_psuedocode(
        &mut self,
        input: &ConcreteLayout,
        output: &ConcreteLayout,
        shuffle: &[Self::Node],
    ) -> Vec<String> {
        let mut codegen = Pseudocode::default();
        let input_regs: Vec<PseudoReg> = (0..input.num_sticks().0)
            .map(|_| codegen.regs.new_reg_name())
            .collect();
        self.codegen_generic(input, output, shuffle, &input_regs, &mut codegen);
        codegen.code_lines
    }
}

/// `codegen_psuedocode`'s `reg_id` — the counter its `get_new_reg_name` lambda captures by reference
/// (`shuffle.cpp:1228`).
#[derive(Debug, Default)]
struct RegNamer(u32);

impl RegNamer {
    /// `get_new_reg_name()` — `"r" << reg_id`, then `reg_id++`.
    fn new_reg_name(&mut self) -> PseudoReg {
        let name = PseudoReg(format!("r{}", self.0));
        self.0 += 1;
        name
    }
}

/// `create_node_allocations` AND `do_codegen` AS ONE VALUE — they share the register counter and the
/// line list, which is exactly why the reference captures both by reference (`shuffle.cpp:1240`,
/// `:1254`).
#[derive(Debug, Default)]
struct Pseudocode {
    /// the shared `reg_id`.
    regs: RegNamer,
    /// `code_lines`.
    code_lines: Vec<String>,
}

impl ShuffleCodegen for Pseudocode {
    type Edge = PseudoReg;

    fn edges_for_node(&mut self, layout: &AbstractLayout) -> Vec<PseudoReg> {
        let mut regs: Vec<PseudoReg> = (0..layout.num_sticks().0)
            .map(|_| self.regs.new_reg_name())
            .collect();
        // "Codegen pulls in order from the back. Register names are interchangeable for psuedocode,
        // but this order is easier to read." (`shuffle.cpp:1247-1248`)
        regs.reverse();
        regs
    }

    fn op_codegen(
        &mut self,
        op: &ComputationOp,
        inputs: &[PseudoReg],
        output: &mut PseudoReg,
        _sticks: &StickIds,
        _output_added: bool,
    ) {
        // ⛔ THE STICK IDS AND `output_reg_added` ARE THE OTHER INSTANTIATION'S: `do_codegen`
        // (`shuffle.cpp:1254`) takes both and reads neither — they carry the element offsets
        // `replace_assign` writes into its data edges (`shuffle.cpp:869-895`).
        self.code_lines
            .extend(op.codegen_psuedocode(inputs, output));
    }
}

// ───────────────────────────────────────────────────────────────────────────────────────────────
// THE DATA EDGE — `DataEdge` (`shuffle.h:171`), the real codegen's `EdgeType`.
// ───────────────────────────────────────────────────────────────────────────────────────────────

/// WHERE A NODE SITS IN A BLOCK'S CHILD LIST — one `BlockNode::addChildNode` call's contribution to
/// `next_` (`dsc/dsc2.cpp:2013`): the node itself, plus the nodes later inserted immediately BEFORE
/// it, in insertion order.
///
/// ⛔ THE PARENT IS NOT A PARAMETER, AND THAT IS THE POINT. `addChildNode` scans `parent->next_` for
/// the sibling pointer and `DT_ERROR`s when it is absent (`dsc2.cpp:2022`), so giving the insertion
/// point custody of what precedes it makes "Sibling reference node not found in parent node"
/// unspellable — and the flattened child order is the reference's, since `insert_packmerge`
/// (`ddc/ddc_transformation.cpp:1992`) adds the packmerge before `assign` and then the allocate
/// before the packmerge.
/// ⭐ A COMPUTE NODE BECAUSE THAT IS THE ONLY INSERT POINT THE REFERENCE HAS: the freshly minted
/// packmerge. A fuller `BlockNode` belongs with the tree-construction units in `schedule/dsc2.rs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InsertPoint {
    node: ComputeNode,
    before: Vec<AllocateNode>,
}

impl InsertPoint {
    /// The node as `addChildNode` placed it, with nothing before it yet.
    #[must_use]
    pub const fn new(node: ComputeNode) -> Self {
        Self {
            node,
            before: Vec::new(),
        }
    }

    /// `insert_point` itself.
    #[must_use]
    pub const fn node(&self) -> &ComputeNode {
        &self.node
    }

    /// The allocate nodes sitting immediately before it, in the order they were inserted.
    pub fn preceding(&self) -> impl Iterator<Item = &AllocateNode> {
        self.before.iter()
    }
}

/// AN EDGE BETWEEN TWO STICK COMPUTATIONS — `DataEdge` (`shuffle.h:171`): which data, on which
/// component, and the allocate node that backs it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataEdge {
    /// `dinfo`.
    pub dinfo: DataInfo,
    /// `component`.
    pub component: SenComponent,
    /// `allocation`.
    ///
    /// ⛔ ONE `Option` FOR THE REFERENCE'S TWO LEVELS: `std::optional<AllocateNode*>` is checked as
    /// `has_value() && value() != nullptr` (`shuffle.h:177`) because "absent" and "present and null"
    /// are both spellable there. Here [`None`] is both.
    pub allocation: Option<AllocateNode>,
    /// `alloc_added`.
    ///
    /// ⛔ NOT DERIVABLE FROM `allocation`, WHICH IS WHY IT IS A SEPARATE BIT: it means "already in
    /// the tree, possibly put there by someone else", and `replace_assign` flips it both ways from
    /// the already-added list and `getPrev()` (`shuffle.cpp:863-868`).
    pub alloc_added: bool,
}

impl DataEdge {
    /// Replaces: e164_insert_before
    ///
    /// PUTS MY ALLOCATE NODE IMMEDIATELY BEFORE `insert_point`, ONCE — `shuffle.h:175`. A no-op for
    /// an edge with no allocation, or one whose allocation is already in the tree.
    ///
    /// ⛔ THE LATCH IS NOT A MOVE: the allocation stays readable afterwards, because
    /// `insert_packmerge` reads it again for the next computation
    /// (`ddc/ddc_transformation.cpp:1982`) and the edge is copied per stick (`shuffle.cpp:821`).
    pub fn insert_before(&mut self, insert_point: &mut InsertPoint) {
        if let Some(allocation) = self.allocation.as_ref() {
            if !self.alloc_added {
                insert_point.before.push(allocation.clone());
                self.alloc_added = true;
            }
        }
    }
}

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
    use crate::formats::DataFormat;
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
            DataFormat::Senint8,
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
        let buffer: SwapBuffer<u8> = SwapBuffer::default();
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
            packmerge_indices: None,
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
            packmerge_indices: None,
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

// ═══ e145..e152 — THE PACK, SHIFT AND GCVT ACTIONS ═══════════════════════════════════════════════

#[cfg(test)]
mod tests_e145_e152 {
    use super::{
        AbstractLayout, DIMS_PER_SLICE, DimSymbol, GCVTF16F8PackAction, Pack8Action, Pack9Action,
        Pack24Action, PackAction, PackDim, ShiftLeftAction, ShuffleCost, ShuffleIndex,
    };
    use crate::formats::DataFormat;
    use std::collections::BTreeSet;

    /// The symbol with this id; `0` is the dummy, as the reference numbers them.
    fn sym(id: i32) -> DimSymbol {
        let mut symbol = DimSymbol::DUMMY;
        for _ in 0..id {
            symbol = symbol.next();
        }
        symbol
    }

    /// A layout from slice symbol ids (2-bit slot first) and stick symbol ids.
    fn layout(slice: [i32; DIMS_PER_SLICE], sticks: &[i32]) -> AbstractLayout {
        AbstractLayout::new(
            sticks.iter().copied().map(sym).collect(),
            slice.map(sym),
            DataFormat::Senint8,
        )
    }

    /// The slice as symbol ids, so a failure prints the layout rather than six `DimSymbol`s.
    fn slice_ids(of: &AbstractLayout) -> Vec<i32> {
        of.slice_dims.iter().map(|dim| dim.id()).collect()
    }

    /// The stick dimensions as symbol ids.
    fn stick_ids(of: &AbstractLayout) -> Vec<i32> {
        of.stick_dims.iter().map(|dim| dim.id()).collect()
    }

    fn raw(indices: &[ShuffleIndex]) -> Vec<i32> {
        indices.iter().map(|index| index.0).collect()
    }

    /// e145 — the three vendored tables, CARRIED AS THEIR VALUES: comparing `get_indices()` against
    /// the constants it returns would pass on any transcription error.
    #[test]
    fn a_pack_names_the_vendors_three_tables() {
        assert_eq!(
            raw(PackAction::new(sym(1), PackDim::Bit64).get_indices()),
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
            "pack25 (shuffle.cpp:42)"
        );
        assert_eq!(
            raw(PackAction::new(sym(1), PackDim::Bit32).get_indices()),
            [0, 2, 4, 6, 16, 18, 20, 22, 8, 10, 12, 14, 24, 26, 28, 30],
            "pack26 (shuffle.cpp:44)"
        );
        assert_eq!(
            raw(PackAction::new(sym(1), PackDim::Bit16).get_indices()),
            [0, 2, 16, 18, 4, 6, 20, 22, 8, 10, 24, 26, 12, 14, 28, 30],
            "pack27 (shuffle.cpp:46)"
        );
    }

    /// e146 — the shift moves the slice up from the 8-bit slot and dummies it, and the 64-bit
    /// dimension it displaces reaches the sticks ONLY when the action extracts.
    #[test]
    fn a_shift_left_slides_the_slice_up_and_dummies_the_8_bit_slot() {
        let input = layout([1, 2, 3, 4, 5, 6], &[7]);

        let extracted = ShiftLeftAction::new(sym(6)).act(&input);
        assert_eq!(slice_ids(&extracted), [1, 2, 0, 3, 4, 5]);
        assert_eq!(stick_ids(&extracted), [6, 7]);

        let dropped = ShiftLeftAction::new(DimSymbol::DUMMY).act(&input);
        assert_eq!(slice_ids(&dropped), [1, 2, 0, 3, 4, 5]);
        assert_eq!(stick_ids(&dropped), [7], "a dummy 64-bit slot is discarded");
    }

    /// e147 — one instruction per stick, two when it extracts. ⛔ AND NO DIVISION: the one-stick
    /// layout costs 1, where a merge or a pack costs 0.
    #[test]
    fn a_shift_left_costs_double_when_it_extracts() {
        let eight = layout([1, 2, 3, 4, 5, 6], &[7, 8, 9]);
        assert_eq!(ShiftLeftAction::new(sym(6)).cost(&eight), ShuffleCost(16.0));
        assert_eq!(
            ShiftLeftAction::new(DimSymbol::DUMMY).cost(&eight),
            ShuffleCost(8.0)
        );

        let one = layout([1, 2, 3, 4, 5, 6], &[]);
        assert_eq!(
            ShiftLeftAction::new(DimSymbol::DUMMY).cost(&one),
            ShuffleCost(1.0),
            "no division here, unlike every other cost"
        );
    }

    /// e148 — `pack8` drops the 4- and 8-bit slots, slides three dimensions down and dummies the
    /// 64-bit slot.
    #[test]
    fn pack8_drops_two_slots_and_lands_the_dim_in_the_32_bit_slot() {
        let packed = Pack8Action::new(sym(7)).act(&layout([1, 2, 3, 4, 5, 6], &[7, 8]));
        assert_eq!(slice_ids(&packed), [1, 4, 5, 6, 7, 0]);
        assert_eq!(stick_ids(&packed), [8]);
    }

    /// e149 — `pack9` assigns in place: the 4-bit slot takes the dimension, the 8-bit slot goes
    /// dummy, and NOTHING ABOVE THEM MOVES.
    #[test]
    fn pack9_assigns_the_4_bit_slot_without_sliding() {
        let packed = Pack9Action::new(sym(7)).act(&layout([1, 2, 3, 4, 5, 6], &[7]));
        assert_eq!(slice_ids(&packed), [1, 7, 0, 4, 5, 6]);
        assert!(stick_ids(&packed).is_empty());
    }

    /// e150 — half the input sticks, by INTEGER division: a one-stick layout costs 0.
    #[test]
    fn pack9_costs_half_the_sticks_and_rounds_down() {
        assert_eq!(
            Pack9Action::new(sym(9)).cost(&layout([1, 2, 3, 4, 5, 6], &[7, 8, 9])),
            ShuffleCost(4.0)
        );
        assert_eq!(
            Pack9Action::new(sym(9)).cost(&layout([1, 2, 3, 4, 5, 6], &[])),
            ShuffleCost(0.0),
            "1 / 2 is 0, not 0.5"
        );
    }

    /// e151 — `pack24` drops three slots, slides three dimensions to the bottom and leaves TWO
    /// dummies behind.
    #[test]
    fn pack24_drops_three_slots_and_leaves_two_dummies() {
        let packed = Pack24Action::new(sym(7)).act(&layout([1, 2, 3, 4, 5, 6], &[7]));
        assert_eq!(slice_ids(&packed), [4, 5, 6, 7, 0, 0]);
        assert!(stick_ids(&packed).is_empty());
    }

    /// e152 — both fp16 spellings convert, to both fp8 spellings, and nothing else converts at all.
    #[test]
    fn a_gcvt_pack_converts_only_fp16_to_the_goals_fp8() {
        let goal = |format| {
            AbstractLayout::new(BTreeSet::new(), [DimSymbol::DUMMY; DIMS_PER_SLICE], format)
        };

        for input in [DataFormat::IeeeFp16, DataFormat::Sen169Fp16] {
            for out in [DataFormat::Sen143Fp8, DataFormat::Sen152Fp8] {
                assert_eq!(
                    GCVTF16F8PackAction::out_format(input, &goal(out)),
                    Some(out),
                    "{input:?} -> {out:?}"
                );
            }
        }

        assert_eq!(
            GCVTF16F8PackAction::out_format(DataFormat::IeeeFp32, &goal(DataFormat::Sen143Fp8)),
            None,
            "the input has to be an fp16"
        );
        assert_eq!(
            GCVTF16F8PackAction::out_format(DataFormat::Sen169Fp16, &goal(DataFormat::Sen169Fp16)),
            None,
            "the goal has to be an fp8"
        );
    }
}

#[cfg(test)]
mod tests_e153_e160 {
    use super::{
        AbstractLayout, AutoShuffler, DimSymbol, GCVTF16F8MergeAction, GCVTF16F8PackAction, Half,
        ShuffleCost, SliceDim, StickIndex, StickNumber, StickRepl, all_one, make_stick_number_key,
    };
    use crate::formats::DataFormat;
    use std::collections::BTreeSet;

    /// The symbol with this id; `0` is the dummy, as the reference numbers them.
    fn sym(id: i32) -> DimSymbol {
        let mut symbol = DimSymbol::DUMMY;
        for _ in 0..id {
            symbol = symbol.next();
        }
        symbol
    }

    /// A layout in `format` whose slice is `[3, 4, dummy, 5, 6, 7]` and whose sticks are `{1, 2}` —
    /// one with the vacant 8-bit slot both GCVT actions require.
    fn layout(format: DataFormat) -> AbstractLayout {
        AbstractLayout::new(
            BTreeSet::from([sym(1), sym(2)]),
            [sym(3), sym(4), DimSymbol::DUMMY, sym(5), sym(6), sym(7)],
            format,
        )
    }

    /// The slice as symbol ids, so a failure prints the layout rather than six `DimSymbol`s.
    fn slice_ids(of: &AbstractLayout) -> Vec<i32> {
        of.slice_dims.iter().map(|dim| dim.id()).collect()
    }

    /// e153 — the pack drops the 8-bit slot, slides 16/32/64 down one and lands the stick dimension
    /// at the top of the slice, leaving the format alone.
    #[test]
    fn a_gcvt_pack_slides_the_slice_down_and_takes_the_top_slot() {
        let input = layout(DataFormat::Sen169Fp16);
        let output = GCVTF16F8PackAction::new(sym(2)).act(&input);

        assert_eq!(slice_ids(&output), [3, 4, 5, 6, 7, 2]);
        assert_eq!(output.stick_dims, BTreeSet::from([sym(1)]));
        assert_eq!(
            output.format,
            DataFormat::Sen169Fp16,
            "`act` does not convert"
        );
    }

    /// e154 — the merge's predicate is the pack's, so it is checked as the same answer over the
    /// whole format square rather than restating e152's own case.
    #[test]
    fn the_merges_out_format_is_the_packs_answer_everywhere() {
        for goal in [
            DataFormat::Sen143Fp8,
            DataFormat::Sen152Fp8,
            DataFormat::Sen169Fp16,
            DataFormat::Senint8,
        ] {
            let goal = layout(goal);
            for input in [
                DataFormat::IeeeFp16,
                DataFormat::Sen169Fp16,
                DataFormat::IeeeFp32,
                DataFormat::Senint8,
            ] {
                assert_eq!(
                    GCVTF16F8MergeAction::out_format(input, &goal),
                    GCVTF16F8PackAction::out_format(input, &goal),
                    "{input:?} -> {:?}",
                    goal.format
                );
            }
        }
        assert_eq!(
            GCVTF16F8MergeAction::out_format(
                DataFormat::Sen169Fp16,
                &layout(DataFormat::Sen143Fp8)
            ),
            Some(DataFormat::Sen143Fp8),
            "and that answer is the goal's own fp8"
        );
    }

    /// e155 — the merge overwrites the 8-bit slot and moves nothing else, which is the one
    /// difference from the pack action.
    #[test]
    fn a_gcvt_merge_lands_the_stick_dimension_in_the_eight_bit_slot() {
        let output = GCVTF16F8MergeAction::new(sym(2)).act(&layout(DataFormat::Sen169Fp16));

        assert_eq!(slice_ids(&output), [3, 4, 2, 5, 6, 7]);
        assert_eq!(output.stick_dims, BTreeSet::from([sym(1)]));
        assert_eq!(output.slice_dim(SliceDim::Bit8), sym(2));
    }

    /// e156 — containment reads from either side, and the dummy of a vacant slot counts.
    #[test]
    fn a_layout_contains_a_dimension_from_either_the_sticks_or_the_slice() {
        let layout = layout(DataFormat::Sen169Fp16);

        assert!(layout.contains(sym(1)), "a stick dimension");
        assert!(layout.contains(sym(5)), "a slice dimension");
        assert!(layout.contains(DimSymbol::DUMMY), "the vacant 8-bit slot");
        assert!(!layout.contains(sym(9)), "a dimension in neither");
    }

    /// e157 — the graph memoises on the layout, so a second visit is the same node and does not
    /// reset what the first relaxation wrote.
    #[test]
    fn get_node_manifests_a_layout_once() {
        let mut shuffler = AutoShuffler::new();
        let first = shuffler.get_node(&layout(DataFormat::Sen169Fp16));

        assert_eq!(shuffler.graph_len(), 1);
        assert!(shuffler.node(first).cost_origin_to_here.0.is_infinite());

        shuffler.node_mut(first).cost_origin_to_here = ShuffleCost(3.0);
        let again = shuffler.get_node(&layout(DataFormat::Sen169Fp16));
        assert_eq!(again, first);
        assert_eq!(shuffler.graph_len(), 1);
        assert_eq!(shuffler.node(again).cost_origin_to_here, ShuffleCost(3.0));

        // A different format is a different layout, so a different node.
        assert_ne!(shuffler.get_node(&layout(DataFormat::Sen143Fp8)), first);
        assert_eq!(shuffler.graph_len(), 2);
    }

    /// e158 — position `i` owns bit `i`, only the high halves contribute, and a dimension the
    /// ordering does not number contributes nothing where the reference's `.at()` would throw.
    #[test]
    fn a_stick_number_is_the_or_of_the_high_dimensions_bits() {
        let key = make_stick_number_key(&[sym(1), sym(2), sym(3)]);
        let index = |halves: [(i32, Half); 3]| {
            halves
                .into_iter()
                .map(|(id, half)| (sym(id), half))
                .collect::<StickIndex>()
        };

        assert_eq!(
            key.key(&index([(1, Half::Low), (2, Half::High), (3, Half::High)])),
            StickNumber(0b110)
        );
        assert_eq!(
            key.key(&index([(1, Half::Low), (2, Half::Low), (3, Half::Low)])),
            StickNumber(0)
        );

        // ⛔ A dimension outside the ordering is not part of the stick number.
        let mut unnumbered = StickIndex::new();
        unnumbered.insert(sym(9), Half::High);
        assert_eq!(key.key(&unnumbered), StickNumber(0));
    }

    /// e159 — unreplicated means every entry is exactly 1, and an empty list is vacuously so.
    #[test]
    fn all_one_holds_only_for_unreplicated_stick_dimensions() {
        assert!(all_one(&[StickRepl(1), StickRepl(1), StickRepl(1)]));
        assert!(all_one(&[]), "no stick dimensions replicate nothing");
        assert!(!all_one(&[StickRepl(1), StickRepl(2)]));
        assert!(!all_one(&[StickRepl(0)]));
    }

    /// e160 — the queued cost is frozen at push time and equal costs pop in push order, which is
    /// the whole reason the entry carries a cost of its own instead of reading the node.
    #[test]
    fn the_worklist_freezes_the_cost_and_breaks_ties_by_push_order() {
        let mut shuffler = AutoShuffler::new();
        let first = shuffler.get_node(&layout(DataFormat::Sen169Fp16));
        let second = shuffler.get_node(&layout(DataFormat::Sen143Fp8));
        shuffler.node_mut(first).cost_origin_to_here = ShuffleCost(4.0);
        shuffler.node_mut(second).cost_origin_to_here = ShuffleCost(4.0);

        shuffler.update_worklist(first);
        shuffler.update_worklist(second);
        assert_eq!(shuffler.worklist_len(), 2);

        shuffler.node_mut(first).cost_origin_to_here = ShuffleCost(1.0);

        let popped = shuffler.pop_worklist();
        assert_eq!(
            popped.map(|entry| (entry.node, entry.cost)),
            Some((first, ShuffleCost(4.0))),
            "equal costs pop in push order, at the cost they were pushed with"
        );
        assert_eq!(
            shuffler.pop_worklist().map(|entry| entry.node),
            Some(second)
        );
        assert!(shuffler.pop_worklist().is_none());
    }

    /// ⛔ AND AN INFINITE COST STILL ORDERS: an unreached node sorts after every reached one instead
    /// of making the heap's comparator partial.
    #[test]
    fn an_unreached_node_queues_behind_every_reached_one() {
        let mut shuffler = AutoShuffler::new();
        let unreached = shuffler.get_node(&layout(DataFormat::Sen169Fp16));
        let reached = shuffler.get_node(&layout(DataFormat::Sen143Fp8));
        shuffler.node_mut(reached).cost_origin_to_here = ShuffleCost(9.0);

        shuffler.update_worklist(unreached);
        shuffler.update_worklist(reached);

        assert_eq!(
            shuffler.pop_worklist().map(|entry| entry.node),
            Some(reached)
        );
        assert_eq!(
            shuffler.pop_worklist().map(|entry| entry.node),
            Some(unreached)
        );
    }
}

#[cfg(test)]
mod tests_e161_e164 {
    use super::{
        AbstractLayout, CodegenGeneric, ComputationOp, ConcreteLayout, DIMS_PER_SLICE, DataEdge,
        DimSymbol, InsertPoint, ShuffleCodegen, ShuffleIndex, StickIds, StickIndex, StickNumber,
        narrow_to_usize,
    };
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::PrimaryDim;
    use crate::formats::DataFormat;
    use crate::generated::ComputeType;
    use crate::schedule::dsc2::{
        AllocLayout, AllocateNode, ComputeNode, DataInfo, MaxDimSize, NodeName, StartAddress,
    };
    use crate::units::NumFolds;
    use std::collections::BTreeSet;
    use sys_arch_spec::arch_enums::SenComponent;

    /// Symbols 1, 2, 3 — `getDefaultSymbol()` and its successors.
    const A: DimSymbol = DimSymbol::DEFAULT;
    const B: DimSymbol = A.next();
    const C: DimSymbol = B.next();
    const DUMMY: DimSymbol = DimSymbol::DUMMY;

    /// A stand-in for `e342_codegen_generic` (`shuffle.cpp:910`, level 3, NOT this batch): visits
    /// each node once and drives one `packmerge` op into the node's first edge.
    struct StubWalk;

    impl CodegenGeneric for StubWalk {
        type Node = AbstractLayout;

        fn codegen_generic<C: ShuffleCodegen>(
            &mut self,
            _input_layout: &ConcreteLayout,
            _output_layout: &ConcreteLayout,
            shuffle: &[AbstractLayout],
            input_edges: &[C::Edge],
            codegen: &mut C,
        ) {
            let op = ComputationOp {
                inputs: Vec::new(),
                output: StickIndex::new(),
                reuses_sticks: false,
                packmerge_indices: Some(vec![ShuffleIndex(0), ShuffleIndex(16)]),
            };
            for node in shuffle {
                let mut edges = codegen.edges_for_node(node);
                codegen.op_codegen(
                    &op,
                    input_edges,
                    &mut edges[0],
                    &StickIds {
                        inputs: Vec::new(),
                        output: StickNumber(0),
                    },
                    false,
                );
            }
        }
    }

    /// e162: `1..=n`, so none of them is the dummy, and a non-positive `n` mints none.
    #[test]
    fn default_symbols_are_one_through_n_and_never_dummy() {
        let three = DimSymbol::default_symbols(3);
        assert_eq!(three, vec![A, B, C]);
        assert!(!three.iter().any(|symbol| symbol.is_dummy()));
        assert_eq!(DimSymbol::default_symbols(0), vec![]);
        assert_eq!(DimSymbol::default_symbols(-4), vec![]);
    }

    /// e163: the FNV walk against a value derived outside this crate, and the fold — on a narrow host
    /// a hash whose whole payload is in the high half must not truncate to zero.
    #[test]
    fn the_layout_hash_is_fnv1a_over_its_symbols_narrowed_to_usize() {
        let layout = AbstractLayout::new(
            [A, B].into_iter().collect::<BTreeSet<_>>(),
            [C, DUMMY, DUMMY, DUMMY, DUMMY, DUMMY],
            DataFormat::Senint8,
        );
        assert_eq!(layout.fnv1a(), narrow_to_usize(0x8e03_bd9a_478e_c4a5));
        assert_ne!(narrow_to_usize(0xffff_ffff_0000_0000), 0);
    }

    /// e161: one register counter across both lambdas, and a node's registers handed out from the
    /// back — one input stick dimension mints `r0`/`r1`, then a 2-dimension node mints `r2..r5` and
    /// yields `r5` first.
    #[test]
    fn pseudocode_shares_one_register_counter_and_reverses_node_registers() {
        let input = ConcreteLayout {
            stick_dims: vec![A],
            slice_dims: vec![B],
        };
        let output = ConcreteLayout {
            stick_dims: vec![B],
            slice_dims: vec![A],
        };
        let node = AbstractLayout::new(
            [A, B].into_iter().collect::<BTreeSet<_>>(),
            [DUMMY; DIMS_PER_SLICE],
            DataFormat::Senint8,
        );

        let lines = StubWalk.codegen_psuedocode(&input, &output, &[node]);

        assert_eq!(lines, vec!["r5 = packmerge r0 r1 [ 0 16 ]".to_owned()]);
    }

    /// e164: the allocation lands before the insert point exactly once, and an edge with none — or
    /// one already in the tree — inserts nothing.
    #[test]
    fn insert_before_adds_the_allocation_once() {
        let allocation = AllocateNode {
            name: NodeName("alloc".to_owned()),
            component: SenComponent::Ptrow0,
            lds: None,
            const_idx: None,
            temp_storage_for_compute: None,
            layout: AllocLayout::new((PrimaryDim::Out, MaxDimSize::Unset), Vec::new()),
            start_address: StartAddress::default(),
            gap_stick_spread: Default::default(),
            alloc_users: Vec::new(),
        };
        let mut point = InsertPoint::new(ComputeNode {
            name: NodeName("packmerge".to_owned()),
            op: ComputeType::Packmerge,
            ex_unit: SenComponent::Ptrow0,
            inputs: Vec::new(),
            outputs: Vec::new(),
            num_folds_engaged: NumFolds::ONE,
        });
        let mut edge = DataEdge {
            dinfo: DataInfo::default(),
            component: SenComponent::Ptrow0,
            allocation: Some(allocation.clone()),
            alloc_added: false,
        };

        edge.insert_before(&mut point);
        edge.insert_before(&mut point);
        assert_eq!(point.preceding().collect::<Vec<_>>(), vec![&allocation]);
        assert!(edge.alloc_added);
        assert_eq!(
            edge.allocation.as_ref(),
            Some(&allocation),
            "still readable"
        );

        let mut nothing_to_add = DataEdge {
            allocation: None,
            ..edge.clone()
        };
        let mut already_there = DataEdge {
            alloc_added: true,
            ..edge.clone()
        };
        nothing_to_add.insert_before(&mut point);
        already_there.insert_before(&mut point);
        assert_eq!(point.preceding().count(), 1);
        assert_eq!(point.node().name, NodeName("packmerge".to_owned()));
    }
}
