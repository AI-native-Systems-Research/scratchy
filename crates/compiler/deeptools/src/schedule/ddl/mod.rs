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

//! `ddc/ddl/ddl.cpp` — 5 of the campaign's 382 units (dependency level(s) [0, 1, 2, 3, 4]).
//!
//! | unit | entry | level | lines | class | authority path:line |
//! |---|---|---|---|---|---|
//! | `e171_performActions` | 171 | 0 | 22 | — | `ddc/ddl/ddl.cpp:36` |
//! | `e273_processBuffer` | 273 | 1 | 13 | — | `ddc/ddl/ddl.cpp:62` |
//! | `e321_DdlMain` | 321 | 2 | 16 | — | `ddc/ddl/ddl.cpp:78` |
//! | `e344_DdlMain` | 344 | 3 | 12 | — | `ddc/ddl/ddl.cpp:96` |
//! | `e363_parseDdl` | 363 | 4 | 9 | `DdlModuleOp` | `ddc/ddl/ddl.cpp:121` |

pub(crate) mod conversion;
pub(crate) mod ops;

use ops::{DdlOp, Dialect, Unverified, Value, Verified};

/// WHAT A GENERIC MLIR PARSE OF A `.ddl` YIELDS — the defining-op table and the ops that carry a
/// verifier. Everything else in the module is stated by the framework's own generic parse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedDdl {
    /// What defines each SSA value — all `getDefiningOp()` is ever asked for here.
    pub defining: Vec<(Value, DdlOp)>,
    /// Every op with `hasVerifier = 1`, in the order the module states them.
    pub verifiable: Vec<Unverified>,
}

/// A DDL SOURCE — a buffer `parseSourceFileForTool` can read (`ddl.cpp:54-55`).
///
/// ⭐ A TRAIT BECAUSE MLIR'S GENERIC PARSE IS THE FRAMEWORK'S, not this campaign's: `performActions`
/// states only the CONFIG it hands over, and our own text parse of the same templates is `build.rs`.
pub trait DdlSource {
    /// The module this source states, or [`None`] where it is not well-formed for `dialect`.
    fn parse(&self, dialect: &Dialect) -> Option<ParsedDdl>;
}

/// Replaces: e171_performActions
///
/// PARSE, THEN RUN EVERY VERIFIER — `ParserConfig(context, /*verifyAfterParse=*/true, ..)`
/// (`ddl.cpp:49-50`) is the flag that makes the four ported verifiers load-bearing instead of dead
/// predicates: one rejection anywhere and the parse yields nothing.
///
/// ⭐ THE THREADING SAVE / DISABLE / RESTORE (`:40-41`, `:56`) IS PERFORMANCE-ONLY — it drops
/// `MLIRContext` synchronisation for the duration — and there is no context here.
/// ⭐ `PassReproducerOptions` + `FallbackAsmResourceMap` (`:47-51`) make unhandled external
/// resources PASSTHROUGH; no vendored template states one.
#[must_use]
pub fn perform_actions(source: &(impl DdlSource + ?Sized), dialect: &Dialect) -> Option<Verified> {
    let parsed = source.parse(dialect)?;
    Verified::of(parsed.defining.as_slice(), parsed.verifiable)
}

/// Replaces: e273_processBuffer
///
/// PARSES ONE OWNED BUFFER — the buffer is CONSUMED (`std::move(ownedBuffer)` into the `SourceMgr`),
/// so a source reaching here cannot be parsed a second time from the same storage.
///
/// ⛔ `allowUnregisteredDialects(false)` IS DISCHARGED BY THE ARGUMENT: `dialect` is the registry, so
/// an op outside it is [`DdlSource::parse`]'s [`None`] rather than a permissive parse.
/// ⭐ THE THREAD POOL, `loadAllAvailableDialects` and `SourceMgrDiagnosticHandler` ARE AMBIENT — a
/// pool to share, every dialect the build linked, and where diagnostics are printed.
#[must_use]
pub fn process_buffer(owned_buffer: impl DdlSource, dialect: &Dialect) -> Option<Verified> {
    perform_actions(&owned_buffer, dialect)
}

// crustify:todo: e321_DdlMain
//   authority : ddc/ddl/ddl.cpp:78  (16 body lines, level 2)
//   original  : OwningOpRef<Operation*> DdlMain(std::unique_ptr<llvm::MemoryBuffer> buffer, MLIRContext* context)
//   extract   : crustify-ddc/cpp/ddl.cpp:1535-1552
//   calls     : e273_processBuffer

// crustify:todo: e344_DdlMain
//   authority : ddc/ddl/ddl.cpp:96  (12 body lines, level 3)
//   original  : OwningOpRef<Operation*> DdlMain(const char* input_filename, MLIRContext* context)
//   extract   : crustify-ddc/cpp/ddl.cpp:3772-3785
//   calls     : e321_DdlMain

// crustify:todo: e363_parseDdl
//   authority : ddc/ddl/ddl.cpp:121  (9 body lines, level 4)
//   class     : DdlModuleOp
//   original  : void DdlModuleOp::parseDdl(const char* input_file)
//   extract   : crustify-ddc/cpp/ddl.cpp:4297-4306
//   calls     : e321_DdlMain, e344_DdlMain

#[cfg(test)]
mod tests_e171 {
    use super::ops::{DdlOp, Dialect, StorageBits, Unverified, Value};
    use super::{DdlSource, ParsedDdl, perform_actions, process_buffer};

    /// A source that hands back a fixed module, standing in for `parseSourceFileForTool`.
    struct Fixed(ParsedDdl);

    impl DdlSource for Fixed {
        fn parse(&self, _dialect: &Dialect) -> Option<ParsedDdl> {
            Some(self.0.clone())
        }
    }

    fn module(bit_width: Option<StorageBits>) -> Fixed {
        Fixed(ParsedDdl {
            defining: vec![(Value::sole("l0_allocation"), DdlOp::Allocate)],
            verifiable: vec![
                Unverified::Datatype {
                    data_type: "SEN143_FP8".to_owned(),
                    bit_width,
                },
                Unverified::ImplicitSync {
                    allocate: Value::sole("l0_allocation"),
                },
            ],
        })
    }

    /// ⭐⭐ `verifyAfterParse=true` IS THE WHOLE POINT: the same module differing only in a
    /// `bit_width=` too narrow for its format yields NOTHING, so a rejected verifier is a rejected
    /// parse rather than a logged complaint.
    #[test]
    fn perform_actions_gates_the_module_on_every_verifier() {
        let dialect = Dialect::initialize();
        let ok =
            perform_actions(&module(Some(StorageBits(16))), &dialect).expect("both verifiers pass");
        assert_eq!(ok.ops().len(), 2);
        assert_eq!(
            perform_actions(&module(Some(StorageBits(4))), &dialect),
            None
        );
    }

    /// ⭐ THE BUFFER IS CONSUMED — `process_buffer` takes its source BY VALUE, so the same storage
    /// cannot be parsed twice — and the module it states is gated by the very same verifiers.
    #[test]
    fn process_buffer_consumes_its_source_and_gates_it() {
        let dialect = Dialect::initialize();
        assert_eq!(
            process_buffer(module(Some(StorageBits(16))), &dialect).map(|ok| ok.ops().len()),
            Some(2)
        );
        assert_eq!(process_buffer(module(Some(StorageBits(4))), &dialect), None);
    }
}
