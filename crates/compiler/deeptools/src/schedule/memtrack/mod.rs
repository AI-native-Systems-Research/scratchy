// SPDX-License-Identifier: Apache-2.0
//
// ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ CRUSTIFY LX-MEMORY-TRACKER CAMPAIGN — READ THIS BEFORE YOU FILL AN ANCHOR IN THIS FILE.       ║
// ║ Campaign statement: crustify-memtrack/TASK.md   ·   worklist: crustify-memtrack/UNITS.tsv     ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
//
// 1. WHAT THIS IS: THE LX MEMORY ALLOCATOR — THE SINGLE THING BLOCKING BRIDGE 1. Bridge 1's stage
//    2a (`L3DlOpsScheduler::run`) now runs on real bake data, and the measurement is unambiguous:
//    24,363 of 24,363 programs in scratchy's corpus convert and then STOP AT EXACTLY ONE PLACE,
//    with ZERO other refusals — `ExPhaseTrackers::backup`, i.e. THIS TRACKER. The stop holds across
//    `numCoresUsed_` ∈ {1,4,8,16,21,32} and all three `dsType_`s. The schedule tree grows 3.74×
//    (93,110 → 347,939 nodes) on the way there and then halts. `sync`/`compute` are 0 ONLY because
//    `create_synchronization` runs after this stop.
//    ⛔ VERIFIED IN ZERO EARLIER CAMPAIGNS: all six existing UNITS.tsv (bridge1-4, senpass, ddc)
//    were grepped for `DsTrackInMem`, `MemoryOrganizer`, `checkAndAddDs`, `allocateMemory`,
//    `getAllFreeBlocks`, `findFirstFittingMemory` — zero hits. Nothing here was ported before.
//
// 2. THE AUTHORITY IS THE C++ TREE, NOT THE EXTRACT.
//       /Users/nickm/git/deeptools-src/<file>:<line>     (revision a0d29abbed)
//    `crustify-memtrack/cpp/memtrack.cpp` says WHICH functions are in scope and IN WHAT ORDER; its
//    bodies were verified byte-identical to the authority (38/38 units, 24,071 bytes,
//    2 of 2 negative controls DETECTED), so either may be read — but the authority file carries the
//    surrounding declarations you will need. ⛔ The pod is not reachable from here, and the other
//    local deeptools checkout is a DIFFERENT revision.
//
// 3. THE THREE PAIRS ARE ONE LADDER, and mem_track alone would place nothing:
//       memtrack/bundle.rs   MemTrackBundle   sets every CAPACITY and GRANULARITY
//                            (`initMemTrack("lxCore<c>", lxCapacity, numSteps, bytesPerStick)`)
//                            and `getTracker(comp, core, corelet, row)` IS the seam's site key.
//       memtrack/tracker.rs  DsTrackInMem     the per-exphase free lists, margin, allocFromBack,
//                            strict mode, backup/restore.
//       memtrack/memory.rs   MemoryOrganizer  the actual block list — EVERY address comes from here
//                            (`getAllFreeBlocks` + `allocateMemory`).
//
// 4. ⭐⭐⭐ THE ORACLE — THE REFERENCE'S OWN PLACED ADDRESSES, ON DISK, FOR SCRATCHY'S OWN PROGRAMS.
//       /Users/nickm/tmp/bridge1-fixtures/g0/debug/sdsc_<N>/sdsc.json   187 PLACED programs
//       /Users/nickm/tmp/bridge1-fixtures/g0/sdsc_<N>.json              the UNSCHEDULED input
//    Census tool: `python3 crustify-memtrack/tools/oracle_census.py` (stdlib, no arguments).
//    ⛔ THE PLACED ADDRESS IS `startAddressCoreCorelet_.data_`. IT IS **NOT**
//    `bufferOffsetCoreCorelet_` — that is 256 on every LX node AND on all 32 cores, because it is
//    the DOUBLE-BUFFER STRIDE (`kv.second / numBuffers`, L3DlOpsScheduler.cpp:5657). A tracker that
//    allocated EVERYTHING AT 256 would pass a check written against it. ⭐ THE RULE THAT CAUGHT
//    THAT: before verifying anything against a fixture field, PRINT IT FOR SEVERAL NODES and
//    confirm it varies the way the quantity should. A constant where values ought to differ is a
//    stride, a default or a flag.
//    ⛔ ONLY LX IS PLACED HERE. HBM nodes carry scratchy's own addresses through unchanged
//    (`Tensor{0,1}_hbm` at 128·core, `Tensor2_hbm` at 6610944 + 128·core); the register-file and L0
//    nodes are placed by stage 2b. The L3 caller `DT_CHECK_MSG(component_ == LX, "Expect only LX.")`
//    at L3DlOpsScheduler.cpp:5548. A tracker that "places" HBM is wrong.
//    ⭐ MEASURED OVER ALL 187 (not the three anyone quotes): 1,899 allocate nodes, 588 LX, only TEN
//    distinct LX addresses, every one of the form 1625344 + k·512. Each node's address is identical
//    across every `[core, corelet, time]` coordinate — 0 exceptions. 88 of 187 DSCs hold LX nodes of
//    DIFFERING sizes, so the caller's descending-size sort IS load-bearing.
//    ⭐⭐ THE LAW: forward FIRST FIT over 2,031,616 bytes with [0, 1625344) already held by
//    `reserved-frontend`, each request rounded UP to 128, in rounded-size DESCENDING order with ties
//    in `ldsIdx` ASCENDING → 179 of 187 programs reproduced EXACTLY. The numbers:
//       lxCapacity       = 2·1024·1024 − 64·1024 = 2,031,616   sys-arch-spec/sysdef.cpp:211
//       allocGranularity = bytesPerStick = 128                 sysdef.cpp:206
//       the base         = reserveFrontendLx reserves (int64_t)(2031616 × 0.8) = 1,625,292 at
//                          address 0, which checkAndAddDsAtAddr rounds UP to 1,625,344 → the first
//                          real LX allocation lands at EXACTLY 1625344.
//                          dbo/src/Transforms/ProgramLayout.cpp:81-98
//    ⛔ REJECTED VARIANTS, so the law is PINNED and not merely consistent: back-fit from the top
//    0/187 · no front reservation 0/187 · DESCENDING-ldsIdx ties 5/187 · schedule-tree-order ties
//    61/187 (fails on 126).
//    ⛔ THE TIE-BREAK IS GROUNDED, NOT FITTED: `nodeAndSize` is BUILT by iterating
//    `allocMetadata.ldsIdxAndAllocNode`, a `std::map<int, dsc2::AllocateNode *>`
//    (L3DlOpsScheduler.h:162), so ldsIdx-ascending IS the construction order and the
//    size-descending `std::sort` (L3DlOpsScheduler.cpp:5594) leaves equal keys in it. ⚠️ Reported
//    honestly: `std::sort` is NOT stable, so tie order is formally implementation-defined; what is
//    MEASURED is that the reference's observed order equals its build order everywhere the law
//    applies. Port a STABLE descending sort over an ldsIdx-ascending input.
//    ⛔⛔ WHAT THE 187 DO **NOT** TEST — READ THIS BEFORE YOU BELIEVE A GREEN TEST. Every one packs
//    consecutively from 1625344 with ZERO gaps. The corpus NEVER exercises a fragmented free list,
//    NEVER `allocFromBack`, NEVER a non-zero `margin`, and never the `eps.size() > 10 && opt_frag_`
//    arm; the L3 caller passes ONE eps per call, so the multi-phase intersection is never exercised
//    either. PORT ALL THOSE ARMS ANYWAY — they are UNTESTED, not absent. ⛔ Reproducing
//    1625344/1625856/1626368 and calling the allocator verified is an INVENTED RULE.
//
// 5. THE SEAM ALREADY EXISTS AND IS ALREADY TYPED. `pub trait ExPhaseTrackers` at
//    `src/schedule/l3/dl_ops.rs:8967` declares exactly six methods, each naming its C++:
//       `ex_phases()` · `capacity(at)` = `memCapacity` · `backup(at)` = `backupEps` per phase,
//       IDEMPOTENT · `restore_all()` = `restoreEps` · `remove(at, name)` = `removeDs` ·
//       `check_and_add(at, phase, name, size)` = `checkAndAddDs`
//    `L3TrackerSite` is `MemTrackBundle::getTracker(comp, core, corelet, row)`; `None` from
//    `check_and_add` is the `EXISTS` answer.
//    ⛔ WRITING THAT `impl` IS INTEGRATION WIRING AND NOT YOUR JOB — the orchestrator owns it after
//    the campaign. YOUR job is that the ported types CAN carry it: the six behaviours reachable as
//    methods, with the same answers. Do NOT edit dl_ops.rs; do NOT add a second trait beside it.
//
// 6. PORTED MEANS THE WHOLE FUNCTION INCLUDING ITS EFFECT. Here the effect is THE MUTATION OF
//    TRACKER STATE: the `dsInMem_` entry, the `free_` decrement and the `occupied_.allocateMemory`
//    commit are ONE indivisible act (`addDsAtStartAddr`, mem_track.cpp:377-393). A port that
//    computes the address and records the decision without mutating all three HANDS OUT THE SAME
//    ADDRESS TWICE on the next call, and it compiles. A PREDICATE IS NOT A PORT. Droppable: only the
//    mechanism for reaching operands. ⛔ If a TYPE cannot express a result, EXTEND THE TYPE.
//
// 7. ⛔⛔ A TEST MUST ASSERT WHAT THE REFERENCE DOES, CITED — NEVER WHAT THE PORT CURRENTLY DOES.
//    A green test asserting entry 207's own refusal survived a full review pass and then killed
//    stage 2a on EVERY program: code and test agreed with each other and both were wrong. CARRY
//    VALUES, NOT "IT REFUSES". Assert the NUMBER: 1625292 at granularity 128 → 1625344; capacity
//    2031616 with [0,1625344) held and a 512 request → 1625344, then 1625856, then 1626368; an
//    empty organizer's free list is the single INCLUSIVE pair (0, capacity_ − 1). And where the
//    reference refuses, assert WHICH sentinel (EXISTS = −1, DOESNT_FIT = −2, INCOHERANT = −3,
//    MemoryOrganizer's own −1) — those are different answers to the same question.
//
// 8. HARD CAPS — MEASURED, AND THEY BIND. Bridge 2's first 144 functions cost 43 h wall clock and
//    586 M cache-read tokens; of 30,991 lines produced only 5,306 were implementation. Per function:
//      • 8 DOC LINES: the `/// Replaces: eNNN_name` anchor, one line of what it does, any TRAP.
//      • ONE TEST. Two only where the vendor's own case AND a negative both apply.
//      • `cargo check -p deeptools` + `cargo test -p deeptools` ONCE PER BATCH, not per function.
//      • Do NOT re-verify citations — the review pass owns that.
//      • Do NOT grep the crate to discover types; the anchor names what you need.
//    NOT capped: correctness, and the effect.
//    ⛔ NEVER run the workspace or the acceptance build in an agent worktree — ~6 GB of target/
//    each, and this host has under 50 GB free.
//
// 9. THIS IS A PURE-LOGIC PORT WITH NO C ANYWHERE. Whatever generic C-to-Rust conventions say:
//    ❌ no bindgen/allowlist/-sys, ❌ no `ffi::`/`extern "C"`, ❌ no `CRUSTIFY_<FILE>` switch,
//    ❌ no `Foo`/`FooRef`/`FooMut` triple, ❌ no `unsafe`, ❌ no sanitizers, ❌ no C-vs-Rust harness.
//
// 10. CRATE RULES BIND YOU — `crates/compiler/deeptools/CLAUDE.md`, in full.
//     🛑 NEVER RUNTIME REFUSE: no `Result`, no `Err(`, no `.ok_or`, no `assert!`, no
//     `debug_assert!`. `todo!` NAMING UNPORTED WORK IS ALLOWED here (the refusal ratchet is scoped
//     to one file and does not cover `src/schedule/`) — but ⛔ NEVER substitute a fabricated
//     address to dodge one. Inventing addresses is a NAMED PAST FAILURE of this very crate.
//     ⛔ NEWTYPES, NEVER RAW SCALARS, AND THIS CAMPAIGN IS THE WORST CASE FOR IT. An ADDRESS, an
//     OFFSET, a CAPACITY, a GRANULARITY, a SIZE-IN-BYTES, a STICK COUNT, a CORE, a CORELET, a ROW
//     and an EXECUTION PHASE are TEN quantities that are all `int64_t`/`int` in the reference.
//     Transposing two must be E0308. The most dangerous pair is (address, offset) — see §4.
//     ⛔ NO STRINGS FOR CLOSED SETS: `getTracker`'s 11 `SenComponents` arms are an enum; the three
//     sentinels are an enum, not integers; a ds NAME is the seam's `v1::StorageName`.
//
// 11. ANCHORS: each `// crustify:todo: eNNN_name` below is one scheduled unit. Replace it with the
//     ported item carrying the doc anchor `/// Replaces: eNNN_name`. ⛔ NEVER DELETE AN ANCHOR YOU
//     DID NOT PORT — on bridge 2 a deleted anchor was indistinguishable from a finished unit and the
//     campaign reported DONE having silently lost 149 of 384 functions. OUTSTANDING WORK IS JUDGED
//     FROM `crustify-memtrack/UNITS.tsv`, never from anchors in the tree.
//     ⛔ TWO PAIRS SHARE A C++ NAME and the ENTRY NUMBER disambiguates: `growExPhases`
//     (DsTrackInMem's and MemTrackBundle's) and `findMemoryBound` (MemoryOrganizer's and
//     DsTrackInMem's), plus the two `removeDs` OVERLOADS. Resolve through UNITS.tsv, never by name.
//
// 12. EXCLUSIONS ARE STATED, NOT SILENT: 49 definitions found, 38 in scope, 11 excluded with a
//     measured reason each in `crustify-memtrack/EXCLUSIONS.tsv` — NONE for being hard, and none by
//     structural heuristic. ⛔ One is a REFERENCE DEFECT you must not "fix": `checkAndAddDsWithOvl`
//     writes through `dt::SmallMap::at` on a key it just proved absent, and that `at`
//     (util/smallmap.hpp:223-230) CONSTRUCTS AND DISCARDS `std::out_of_range` instead of throwing,
//     then dereferences `vec_.end()`. There is no defined behaviour to port.

//! THE LX MEMORY ALLOCATOR — the single thing blocking bridge 1. All 24,363 of 24,363 programs in scratchy's corpus reach `ExPhaseTrackers::backup` and stop there, with zero other refusals. Three C++ pairs, ONE ladder: `bundle` → `tracker` → `memory`.
//!
//! 38 units across 3 file(s).

pub(crate) mod bundle;
pub(crate) mod memory;
pub(crate) mod tracker;

