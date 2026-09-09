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


// crustify:todo: e137_int_log2
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:95  (9 body lines, level 0)
//   original  : int int_log2(int x)
//   extract   : crustify-ddc/cpp/ddc.cpp:2573-2582

// crustify:todo: e138_swap
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:121  (1 body lines, level 0)
//   class     : SwapBuffer
//   original  : void swap()
//   extract   : crustify-ddc/cpp/ddc.cpp:2592-2593

// crustify:todo: e139_packmerge_psuedostring
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:124  (9 body lines, level 0)
//   original  : std::string packmerge_psuedostring(const std::vector<int> indices, const std::string& in_reg_1, const std::string& in_reg_2, const std::string& out_reg_name)
//   extract   : crustify-ddc/cpp/ddc.cpp:2602-2614

// crustify:todo: e140_repeat_over_dims
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:178  (26 body lines, level 0)
//   class     : ComputationOp
//   original  : void ComputationOp::repeat_over_dims( const std::vector<DimSymbol>& dims, std::vector<ComputationOp>& append_to) const
//   extract   : crustify-ddc/cpp/ddc.cpp:2624-2652

// crustify:todo: e141_act
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:249  (16 body lines, level 0)
//   class     : MergeAction
//   original  : AbstractLayout act(const AbstractLayout& input) override
//   extract   : crustify-ddc/cpp/ddc.cpp:2662-2678

// crustify:todo: e142_cost
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:266  (6 body lines, level 0)
//   class     : MergeAction
//   original  : double cost(const AbstractLayout& input) override
//   extract   : crustify-ddc/cpp/ddc.cpp:2688-2694

// crustify:todo: e143_get_shuffle_indices
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:275  (29 body lines, level 0)
//   class     : MergeAction
//   original  : std::vector<int> get_shuffle_indices(bool high)
//   extract   : crustify-ddc/cpp/ddc.cpp:2704-2733

// crustify:todo: e144_act
//   authority : ddc/transformations/automatic_shuffle/shuffle.cpp:351  (10 body lines, level 0)
//   class     : PackAction
//   original  : AbstractLayout act(const AbstractLayout& input) override
//   extract   : crustify-ddc/cpp/ddc.cpp:2743-2753

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

