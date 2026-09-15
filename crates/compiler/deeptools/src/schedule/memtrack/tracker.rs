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

//! `DsTrackInMem` — the per-execution-phase tracker: one free counter, one name→size map and one `MemoryOrganizer` PER PHASE, plus `margin`, `allocFromBack`, `strict` and the backup/restore pair. ⭐ `checkAndAddDs` (mem_track.cpp:395) is THE ENTRY POINT the oracle measures, and `backupEps` (:566) is where all 24,363 programs stop today.
//!
//! `util/memtracker/mem_track.cpp` — 24 of the campaign's 38 units (dependency level(s) [0, 1, 2, 3, 4]).
//!
//! | unit | entry | level | lines | class | authority path:line |
//! |---|---|---|---|---|---|
//! | `e003_getAllDsAndSizeAtEps` | 003 | 0 | 6 | `DsTrackInMem` | `util/memtracker/mem_track.cpp:97` |
//! | `e004_checkIfDsFits` | 004 | 0 | 23 | `DsTrackInMem` | `util/memtracker/mem_track.cpp:134` |
//! | `e005_addEpsBefore` | 005 | 0 | 16 | `DsTrackInMem` | `util/memtracker/mem_track.cpp:472` |
//! | `e006_removeEps` | 006 | 0 | 7 | `DsTrackInMem` | `util/memtracker/mem_track.cpp:489` |
//! | `e007_renameEps` | 007 | 0 | 13 | `DsTrackInMem` | `util/memtracker/mem_track.cpp:497` |
//! | `e008_getFreeCapAtEps` | 008 | 0 | 7 | `DsTrackInMem` | `util/memtracker/mem_track.cpp:511` |
//! | `e009_findMemoryBound` | 009 | 0 | 10 | `DsTrackInMem` | `util/memtracker/mem_track.cpp:555` |
//! | `e018_checkIfDsNotExists` | 018 | 1 | 17 | `DsTrackInMem` | `util/memtracker/mem_track.cpp:203` |
//! | `e019_removeDs` | 019 | 1 | 15 | `DsTrackInMem` | `util/memtracker/mem_track.cpp:439` |
//! | `e020_removeDs` | 020 | 1 | 15 | `DsTrackInMem` | `util/memtracker/mem_track.cpp:456` |
//! | `e021_getCapOfDs` | 021 | 1 | 11 | `DsTrackInMem` | `util/memtracker/mem_track.cpp:519` |
//! | `e022_getAddrOfDs` | 022 | 1 | 11 | `DsTrackInMem` | `util/memtracker/mem_track.cpp:531` |
//! | `e023_backupEps` | 023 | 1 | 13 | `DsTrackInMem` | `util/memtracker/mem_track.cpp:566` |
//! | `e027_growExPhases` | 027 | 2 | 7 | `DsTrackInMem` | `util/memtracker/mem_track.cpp:115` |
//! | `e028_formatMemTrack` | 028 | 2 | 10 | `DsTrackInMem` | `util/memtracker/mem_track.cpp:123` |
//! | `e029_checkIfDsExists` | 029 | 2 | 3 | `DsTrackInMem` | `util/memtracker/mem_track.cpp:222` |
//! | `e030_findCommonFittingMemory` | 030 | 2 | 76 | `DsTrackInMem` | `util/memtracker/mem_track.cpp:269` |
//! | `e031_addDsAtStartAddr` | 031 | 2 | 15 | `DsTrackInMem` | `util/memtracker/mem_track.cpp:377` |
//! | `e032_initMemTrack` | 032 | 3 | 8 | `DsTrackInMem` | `util/memtracker/mem_track.cpp:104` |
//! | `e033_getAddressOfDs` | 033 | 3 | 14 | `DsTrackInMem` | `util/memtracker/mem_track.cpp:159` |
//! | `e034_checkDsForStartAddr` | 034 | 3 | 24 | `DsTrackInMem` | `util/memtracker/mem_track.cpp:348` |
//! | `e035_restoreEps` | 035 | 3 | 16 | `DsTrackInMem` | `util/memtracker/mem_track.cpp:580` |
//! | `e037_checkAndAddDs` | 037 | 4 | 18 | `DsTrackInMem` | `util/memtracker/mem_track.cpp:395` |
//! | `e038_checkAndAddDsAtAddr` | 038 | 4 | 18 | `DsTrackInMem` | `util/memtracker/mem_track.cpp:417` |


use super::bundle::GrowExPhases;
use super::memory::{
    Address, Capacity, DsKey, FreeBlock, MemoryOrganizer, Placement, find_or_create_ds_key,
};
use crate::schedule::ddc::v1::StorageName;
use crate::schedule::l3::dl_ops::ExPhase;
use std::collections::BTreeMap;
use std::num::NonZeroI64;

// ───────────────────────────────────────────────────────────────────────────────────────────────
// THE PER-PHASE VOCABULARY — `util/memtracker/mem_track.h:17-54`. Declarations, not units: the
// members that ARE units keep their anchors below.
//
// ⛔ THE EXTENTS ARE THE BLOCK LIST'S OWN. `free_`, `memCapacity`, `cap` and `margin` are the same
// quantity as `MemoryOrganizer::capacity_` — bytes of space — so they reuse [`Capacity`] rather than
// mint a second extent type, which would need conversions the reference's bodies do not have.
// ⛔ A GRANULARITY IS NOT AN EXTENT: `allocGranularity` reaches only ONE expression, the round-up
// `((cap + allocGranularity - 1) / allocGranularity) * allocGranularity` (`mem_track.cpp:138`,
// `:407`, `:429`) — it is DIVIDED BY and MULTIPLIED BY there and is never added to, subtracted from
// or compared against a capacity anywhere. It is a plain `int64_t` defaulting to 1 in the reference
// (`mem_track.h:38`); here it carries a non-zero type, so a granularity of 0 is UNCONSTRUCTIBLE
// rather than the division fault the reference would take.
// ───────────────────────────────────────────────────────────────────────────────────────────────

/// THE LABEL ONE TRACKER CARRIES — `name` (`mem_track.h:36`), written by unit e032 and read by
/// nothing in scope, but read outside it (`DeviceMemAllocation.cpp:102`, and `ddrManagement.cpp:38`
/// hands it straight back to `initMemTrack`): `"lxCore" + std::to_string(c)` for the LX cores
/// (`mem_track_bundle.cpp:59`), `"HBM_" + std::to_string(seg)` for the segments
/// (`ProgramLayout.cpp:70`).
///
/// ⛔ NOT A DS NAME: this and the `ds` of `checkAndAddDs` are both `const std::string&` in the
/// reference, and a ds name is the seam's [`StorageName`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MemName(pub String);

/// THE UNIT EVERY REQUEST IS SNAPPED UP TO — `allocGranularity` (`mem_track.h:38`), which
/// `MemTrackBundle` sets to `bytesPerStick` = 128 for every LX tracker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocGranularity(pub NonZeroI64);

impl AllocGranularity {
    /// The reference's own initialiser, `allocGranularity = 1` (`mem_track.h:38`) — rounds nothing.
    pub const ONE: Self = Self(NonZeroI64::new(1).expect("one is not zero"));

    /// `((cap + allocGranularity - 1) / allocGranularity) * allocGranularity`
    /// (`mem_track.cpp:137-138`, `:406-407`, `:428-429`) — the request rounded UP, and IDEMPOTENT,
    /// which is the only reason `checkAndAddDs` rounding before
    /// [`DsTrackInMem::check_if_ds_fits`] rounds again is harmless.
    #[must_use]
    pub const fn round_up(self, cap: Capacity) -> Capacity {
        let granularity = self.0.get();
        Capacity(((cap.0 + granularity - 1) / granularity) * granularity)
    }
}

impl Default for AllocGranularity {
    fn default() -> Self {
        Self::ONE
    }
}

/// HEADROOM DEMANDED BUT NOT RESERVED — `margin` (`mem_track.h:66`, defaulted to 0 at every call
/// site). Its own type so `(cap, margin)` cannot be transposed: they are both `int64_t` in the
/// reference and they answer different questions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Margin(pub Capacity);

impl Margin {
    /// The default every caller takes, and the only one the 187-program corpus ever passes.
    pub const NONE: Self = Self(Capacity(0));
}

/// HOW MANY EXECUTION PHASES THIS TRACKER HOLDS — `exPhases` (`mem_track.h:39`).
///
/// ⛔ SIGNED, like the reference's own `int`: `removeEps` decrements it whenever a key is found,
/// and it is maintained separately from `epsToListIter`, so the two can disagree. An unsigned count
/// would panic where the reference simply goes negative.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExPhaseCount(pub i32);

/// WHAT ONE PHASE HOLDS, name → capacity — `DsInMem` (`mem_track.h:44`).
///
/// ⛔ A `dt::SmallMap` IS A SORTED VECTOR THAT MORPHS INTO A `std::map` PAST TEN ENTRIES
/// (`util/smallmap.hpp:17-22`, `:261-284`), and its order is the one `vec_find`'s `std::lower_bound`
/// binary-searches under (`:292-299`) — so the vector is not in insertion order. ⛔ AND THAT ORDER IS
/// NOT LEXICOGRAPHIC EITHER: `StringKey` declares no `operator<` (`util/stringkeyset.hpp:20-43`), so
/// `a.first < key` resolves through `operator const char*()` to a comparison of INTERNED POINTERS.
/// Nothing in scope reads it — `backupEps` (unit e023) walks it, and `restoreEps` replays each entry
/// at its own RECORDED address — so this port carries the phase's own vector order instead.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DsInMem(pub Vec<(DsKey, Capacity)>);

/// ONE EXECUTION PHASE'S STATE — `DsTrackInMem::Entry` (`mem_track.h:46-50`).
///
/// ⛔ THE THREE FIELDS MOVE TOGETHER. `addDsAtStartAddr` (unit e031) records the map entry,
/// decrements `free_` and commits to `occupied_` as ONE act; a port that updates two of the three
/// hands out the same address twice on the next call, and it compiles.
#[derive(Debug, Clone)]
pub struct Entry {
    /// `free_`.
    pub free: Capacity,
    /// `dsInMem_`.
    pub ds_in_mem: DsInMem,
    /// `occupied_`.
    pub occupied: MemoryOrganizer,
}

/// A POSITION IN THE ENTRY LIST THAT SURVIVES EVERY INSERTION AND ERASURE ELSEWHERE — the
/// `std::list<Entry>::iterator` that `epsToListIter` stores (`mem_track.h:53-54`).
///
/// ⛔ THE TRAP THIS TYPE EXISTS FOR: the reference keeps ITERATORS precisely because a `std::list`
/// never invalidates them, and its own header says so (`mem_track.h:28`, `:60-63`). A `Vec<Entry>`
/// index would be silently wrong the moment `addEpsBefore` inserted ahead of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntrySlot(usize);

/// THE ANSWER `addEpsBefore` AND `renameEps` GIVE — the reference's `int` return, whose values are
/// `0`, `EXISTS` = −1 and `INCOHERANT` = −3 (`mem_track.h:17-19`).
///
/// ⛔ `DOESNT_FIT` = −2 is NOT reachable from either of them; it belongs to the address-returning
/// units. ⛔ `Incoherant` keeps the reference's own spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpsEdit {
    /// `0` — the edit happened.
    Done,
    /// `EXISTS` — the phase asked FOR is already bound, and nothing was changed.
    Exists,
    /// `INCOHERANT` — the phase asked ABOUT is not bound.
    Incoherant,
}

/// THE TWO MODES OF A TRACKER — `strict` (`mem_track.h:41`), whose two values the reference's own
/// comment names (`:40`): address assignment, or capacity assessment only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TrackingMode {
    /// `strict == true` — the reference's initialiser, and every tracker on the bridge-1 path.
    /// `dsInMem_`, `free_` AND `occupied_` all move.
    #[default]
    AddressAssignment,
    /// `strict == false` — `dsInMem_` and `free_` move and the block list is NEVER touched, so every
    /// ds reads back [`DsAddress::Unplaced`] and `checkDsForStartAddr` answers a real 0 (unit e034).
    CapacityOnly,
}

/// WHETHER THIS TRACKER FIGHTS FRAGMENTATION — `opt_frag_` (`mem_track.h:42`), which reaches exactly
/// ONE expression, `eps.size() > 10 && opt_frag_` (`mem_track.cpp:327`).
///
/// ⛔ IT IS NOT A "TRY HARDER" KNOB: switching it ON for a request spanning more than ten phases
/// moves the answer from the FIRST fitting block to the LAST one, so it changes the address.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FragmentationOpt {
    /// `opt_frag_ == true` — the reference's own initialiser, and every tracker on this path.
    #[default]
    On,
    /// `opt_frag_ == false` — the wide-request back-fit is not taken.
    Off,
}

/// WHAT `findCommonFittingMemory` IS BEING ASKED FOR — the reference's `addrNeeded` (−1 by default)
/// and `allocFromBack` (`false` by default) at `mem_track.h:72-74`: TWO parameters spelling THREE
/// requests, of which `(addrNeeded >= 0, allocFromBack)` is a pair the reference never reads.
///
/// ⛔ NOT [`Placement`]: that type's `FirstFit` arm names the forward end of the space, and
/// [`Self::AnywhereFromBack`] is the SAME "any address" question answered from the other end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitRequest {
    /// `addrNeeded < 0 && !allocFromBack` — the FIRST fitting block (`mem_track.cpp:336-341`), which
    /// is the arm all 187 reference programs take, and the one `eps.size() > 10 && opt_frag_` turns
    /// into [`Self::AnywhereFromBack`].
    Anywhere,
    /// `allocFromBack == true` — the LAST fitting block, at `iblk.second - cap + 1`
    /// (`mem_track.cpp:328-334`). Untested by the corpus, not absent from the reference.
    AnywhereFromBack,
    /// `addrNeeded >= 0` — this exact address if a single free block holds it with room, and nothing
    /// otherwise (`mem_track.cpp:317-325`).
    At(Address),
}

/// WHICH END OF THE SPACE `checkAndAddDs` TAKES — the reference's `bool allocFromBack`
/// (`mem_track.h:81-82`), a closed pair of behaviours rather than a number.
///
/// ⛔ NOT [`FitRequest`]: this parameter CANNOT spell [`FitRequest::At`], because `checkAndAddDs`
/// hands `checkDsForStartAddr` a hardwired `addrNeeded = -1` (`mem_track.cpp:408-409`) — only unit
/// e038 reaches a pinned address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocEnd {
    /// `allocFromBack == false` — the default, and the arm all 187 reference programs take.
    Front,
    /// `allocFromBack == true` — the LAST fitting block. Untested by the corpus, not absent from the
    /// reference.
    Back,
}

/// A PHASE THIS TRACKER IS KNOWN TO HOLD — the witness `epsToListIter.at(eps)` needs, because
/// `std::unordered_map::at` THROWS on an unbound key (`mem_track.cpp:442`) where
/// `dt::SmallMap::at` merely discards its exception.
///
/// ⛔ MINTABLE ONLY THROUGH [`DsTrackInMem::bound_phase`]: `removeDs` returns `void`, so it has no
/// channel to report that throw with, and the phase list it walks must not be able to contain one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundPhase(ExPhase);

/// WHAT AN ADDRESS QUERY ANSWERS — the reference's `int64_t`: an address on the good path, and one of
/// two DIFFERENT negative sentinels otherwise.
///
/// ⛔ THE SENTINELS ARE ARMS, NOT NUMBERS. `INCOHERANT` = −3 and `MemoryOrganizer`'s own −1 are two
/// answers to one question, and −1 is also the NUMBER of `EXISTS` (`mem_track.h:17-19`), which unit
/// e034 answers with. Units e033 and e034 add their own arms here rather than a fourth integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DsAddress {
    /// A byte address — and address 0 is one of them, not a refusal.
    At(Address),
    /// `INCOHERANT` — the phase is not bound, OR it is and does not track the name.
    Incoherant,
    /// `MemoryOrganizer::findAddressByName`'s own −1 — the phase TRACKS the name but no block holds
    /// it, which is every ds of a [`TrackingMode::CapacityOnly`] tracker.
    Unplaced,
    /// `DOESNT_FIT` — `getAddressOfDs` (unit e033) answers −2 for a name its phase does not hold AND
    /// for a phase that is not bound, the two states unit e022 answers [`Self::Incoherant`] to.
    DoesntFit,
    /// `EXISTS` — `checkDsForStartAddr` (unit e034) alone gives this, and its −1 is a DIFFERENT
    /// answer from the organizer's own −1 that [`Self::Unplaced`] carries.
    Exists,
}

/// ONE BACKED-UP DS — `DsTrackInMem::DsMemInfo` (`mem_track.h:32-35`), and the whole of what
/// `restoreEps` replays.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsMemInfo {
    /// `ds` — the NAME and not the interned key, because `restoreEps` hands it back to
    /// `addDsAtStartAddr` and to `freeMemBlockByName`, both of which take the string.
    pub ds: StorageName,
    /// `cap` — the ROUNDED capacity `free_` was debited by.
    pub cap: Capacity,
    /// `startAddr`. ⛔ [`None`] IS THE ORGANIZER'S −1 AND IS NOT AN ADDRESS: `restoreEps` feeds it
    /// to `addDsAtStartAddr`, whose `DT_CHECK(startAddr >= 0)` is the only thing standing between a
    /// missing name and a committed negative address (`mem_track.cpp:380`).
    pub start: Option<Address>,
}

/// THE PER-EXECUTION-PHASE TRACKER — `DsTrackInMem` (`mem_track.h:24-105`).
///
/// ⛔ THE LIST AND THE INDEX ARE NOT THE SAME SET. `formatMemTrack` (unit e028) rebinds only
/// `[0, exPhases)` and never clears stale keys, and [`Self::find_memory_bound`] walks the LIST.
/// Collapsing the two into one map would change that unit's answer.
#[derive(Debug, Clone)]
pub struct DsTrackInMem {
    /// `name` (`mem_track.h:36`) — unit e032 is where it enters, and nothing in scope reads it back.
    pub name: MemName,
    /// `memCapacity` (`mem_track.h:37`) — the space EVERY phase this tracker mints gets its own copy
    /// of, and the `free_` a fresh phase starts at (`mem_track.cpp:117`, `:125`).
    pub mem_capacity: Capacity,
    /// `allocGranularity` (`mem_track.h:38`).
    pub alloc_granularity: AllocGranularity,
    /// `exPhases` (`mem_track.h:39`).
    pub ex_phases: ExPhaseCount,
    /// `strict` (`mem_track.h:41`).
    pub strict: TrackingMode,
    /// `opt_frag_` (`mem_track.h:42`).
    pub opt_frag: FragmentationOpt,
    /// `entries_` (`mem_track.h:52`), in list order.
    entries: Vec<EntrySlot>,
    /// What the slots name. `None` is an erased node, and a slot is never reissued.
    slots: Vec<Option<Entry>>,
    /// `epsToListIter` (`mem_track.h:54`). Ordered where the reference's `unordered_map` is not,
    /// which is a determinism choice and not a divergence: no unit reads its iteration order.
    phases: BTreeMap<ExPhase, EntrySlot>,
}

/// ⛔ `memCapacity` AND `exPhases` HAVE NO INITIALISERS IN THE REFERENCE (`mem_track.h:37`, `:39`) —
/// unit e032 is where both enter. A default tracker therefore describes a ZERO-byte space with no
/// phases, in which every phase [`DsTrackInMem::grow_ex_phases`] mints is already full. The reference
/// reads uninitialised memory there; this is the deterministic reading of the same state, and it is
/// not one any caller constructs on purpose.
impl Default for DsTrackInMem {
    fn default() -> Self {
        Self {
            name: MemName::default(),
            mem_capacity: Capacity(0),
            alloc_granularity: AllocGranularity::ONE,
            ex_phases: ExPhaseCount(0),
            strict: TrackingMode::AddressAssignment,
            opt_frag: FragmentationOpt::On,
            entries: Vec::new(),
            slots: Vec::new(),
            phases: BTreeMap::new(),
        }
    }
}

impl DsTrackInMem {
    /// `epsToListIter.at(eps)->` — the composite lookup every accessor here starts from
    /// (`mem_track.cpp:101`, `:142`, `:514`).
    fn entry(&self, at: ExPhase) -> Option<&Entry> {
        let slot = self.phases.get(&at)?;
        self.slots.get(slot.0)?.as_ref()
    }

    /// `entries_.insert(pos, entry)` (`mem_track.cpp:478`) — the list insertion, and the only thing
    /// that mints an [`EntrySlot`]. A position past the end appends, as `entries_.end()` does.
    fn insert_entry(&mut self, position: usize, entry: Entry) -> EntrySlot {
        let slot = EntrySlot(self.slots.len());
        self.slots.push(Some(entry));
        self.entries.insert(position.min(self.entries.len()), slot);
        slot
    }

    /// `entries_.erase(iter)` (`mem_track.cpp:491`) — the node leaves the list and its slot is
    /// tombstoned, so a surviving [`EntrySlot`] naming it reads as gone rather than as its successor.
    fn erase(&mut self, slot: EntrySlot) {
        if let Some(held) = self.slots.get_mut(slot.0) {
            *held = None;
        }
        self.entries.retain(|held| *held != slot);
    }

    /// Where `slot` sits in the list — what `std::list::insert` takes and what an `unordered_map`
    /// value cannot give.
    fn position_of(&self, slot: EntrySlot) -> Option<usize> {
        self.entries.iter().position(|held| *held == slot)
    }

    /// Replaces: e003_getAllDsAndSizeAtEps
    ///
    /// The name → capacity map of one phase (`mem_track.cpp:97-102`).
    ///
    /// ⛔ `None` IS `DT_ERROR("Request Ds map for an unknown eps")`: the reference stops there, and
    /// its own `epsToListIter.at(eps)` on the next line would throw regardless.
    #[must_use]
    pub fn all_ds_and_size_at_eps(&self, at: ExPhase) -> Option<&DsInMem> {
        Some(&self.entry(at)?.ds_in_mem)
    }

    /// Replaces: e004_checkIfDsFits
    ///
    /// Whether `cap` rounded up, plus `margin`, is within the free counter of EVERY phase asked for
    /// (`mem_track.cpp:134-157`).
    ///
    /// ⛔ CAPACITY ONLY, and the refusal is `free_ < capRound + margin`: an EXACT fill of the free
    /// counter FITS, and `margin` is headroom demanded but never reserved — it changes whether the
    /// allocation is allowed, never where it lands. ⛔ A phase that does not exist does not fit.
    #[must_use]
    pub fn check_if_ds_fits(&self, cap: Capacity, eps: &[ExPhase], margin: Margin) -> bool {
        let mut demanded = self.alloc_granularity.round_up(cap);
        demanded += margin.0;
        eps.iter()
            .all(|at| self.entry(*at).is_some_and(|held| held.free >= demanded))
    }

    /// Replaces: e005_addEpsBefore
    ///
    /// Puts a COPY of `cand`'s phase into the list immediately before it and binds `new_eps` to that
    /// copy, so the new phase starts out holding everything the candidate holds
    /// (`mem_track.cpp:472-487`).
    ///
    /// ⛔ TRAP: `DT_CHECK(copy)` — the reference's `bool copy` has exactly ONE legal value, so the
    /// parameter is gone rather than carried as a mode that cannot be selected. ⛔ The copy carries
    /// `free_`, `dsInMem_` AND the whole block list, and the two phases then diverge independently.
    pub fn add_eps_before(&mut self, new_eps: ExPhase, cand: ExPhase) -> EpsEdit {
        if self.phases.contains_key(&new_eps) {
            return EpsEdit::Exists;
        }
        let Some(cand_slot) = self.phases.get(&cand).copied() else {
            return EpsEdit::Incoherant;
        };
        // The invariant `insert_entry` and `erase` keep: a bound phase names a live node in the list.
        let (Some(position), Some(Some(copy))) =
            (self.position_of(cand_slot), self.slots.get(cand_slot.0).cloned())
        else {
            return EpsEdit::Incoherant;
        };
        let fresh = self.insert_entry(position, copy);
        self.phases.insert(new_eps, fresh);
        self.ex_phases.0 += 1;
        EpsEdit::Done
    }

    /// Replaces: e006_removeEps
    ///
    /// Drops a phase: its node leaves the list, its key leaves the index, and the phase count falls
    /// (`mem_track.cpp:489-495`).
    ///
    /// ⛔ A PHASE THAT IS NOT BOUND IS NOT AN ERROR here, and it does not move the count.
    pub fn remove_eps(&mut self, at: ExPhase) {
        if let Some(slot) = self.phases.remove(&at) {
            self.erase(slot);
            self.ex_phases.0 -= 1;
        }
    }

    /// Replaces: e007_renameEps
    ///
    /// Rebinds one phase's node to a new key, leaving the list itself untouched
    /// (`mem_track.cpp:497-509`).
    ///
    /// ⛔ IT WILL NOT OVERWRITE: an already-bound target answers `Exists` and changes nothing, which
    /// is exactly why `insertPsBeforeAndRename` walks phases DOWNWARD. ⛔ The phase COUNT does not
    /// move — one node keeps one key, under a new name.
    pub fn rename_eps(&mut self, old_eps: ExPhase, new_eps: ExPhase) -> EpsEdit {
        if self.phases.contains_key(&new_eps) {
            return EpsEdit::Exists;
        }
        let Some(slot) = self.phases.remove(&old_eps) else {
            return EpsEdit::Incoherant;
        };
        self.phases.insert(new_eps, slot);
        EpsEdit::Done
    }

    /// Replaces: e008_getFreeCapAtEps
    ///
    /// The free counter of one phase (`mem_track.cpp:511-517`).
    ///
    /// ⛔ ZERO FOR A PHASE THAT IS NOT BOUND, and the reference's caller cannot tell that from a
    /// phase with nothing left. The answer is the VALUE the reference gives, not a refusal.
    #[must_use]
    pub fn free_cap_at_eps(&self, at: ExPhase) -> Capacity {
        match self.entry(at) {
            None => Capacity(0),
            Some(held) => held.free,
        }
    }

    /// Replaces: e009_findMemoryBound
    ///
    /// The highest bound any phase's block list reaches — a maximum over the LIST, not over the
    /// phase index (`mem_track.cpp:555-564`).
    ///
    /// ⛔ `None` IS THE REFERENCE'S −1 SEED AND MEANS NO PHASES AT ALL: a phase whose organizer is
    /// empty answers address 0 (unit e014), so −1 is reachable no other way. ⛔ DISTINCT FROM
    /// `MemoryOrganizer::findMemoryBound` (unit e014) despite the shared C++ name.
    #[must_use]
    pub fn find_memory_bound(&self) -> Option<Address> {
        self.entries
            .iter()
            .filter_map(|slot| self.slots.get(slot.0)?.as_ref())
            .map(|held| held.occupied.find_memory_bound())
            .max()
    }

    /// `epsToListIter.at(eps)->` for a WRITE (`mem_track.cpp:442`; `kv.second->` at `:458`) — the
    /// mutable half of [`Self::entry`], and it reads a stale key as gone rather than as its
    /// successor.
    fn entry_mut(&mut self, at: ExPhase) -> Option<&mut Entry> {
        let slot = *self.phases.get(&at)?;
        self.slots.get_mut(slot.0)?.as_mut()
    }

    /// THE WITNESS `epsToListIter.at(eps)` NEEDS — [`None`] exactly where the reference's
    /// `std::unordered_map::at` would throw (`mem_track.cpp:442`).
    #[must_use]
    pub fn bound_phase(&self, at: ExPhase) -> Option<BoundPhase> {
        self.entry(at).map(|_| BoundPhase(at))
    }

    /// `dsInMem_.erase(dsKey)`, `free_ += cap` and, when strict, `occupied_.freeMemBlockByName(ds)`
    /// for ONE phase (`mem_track.cpp:443-451`) — the body both `removeDs` overloads share, as ONE
    /// act, and a no-op for a name the phase does not hold.
    fn remove_ds_from(&mut self, at: ExPhase, ds: &StorageName) {
        let strict = self.strict;
        let Some(held) = self.entry_mut(at) else {
            return;
        };
        let Some(position) = held.ds_in_mem.0.iter().position(|(key, _)| *key == *ds) else {
            return;
        };
        let (_, cap) = held.ds_in_mem.0.remove(position);
        held.free += cap;
        if strict == TrackingMode::AddressAssignment {
            held.occupied.free_mem_block_by_name(ds);
        }
    }

    /// Replaces: e018_checkIfDsNotExists
    ///
    /// `true` when `ds` is ABSENT from every phase in `eps` (`mem_track.cpp:203-220`).
    ///
    /// ⛔ TRAP: THE NAME IS INVERTED RELATIVE TO THE ANSWER, and AN UNBOUND PHASE READS AS "ALREADY
    /// EXISTS" (`else { exists = false; break; }`, `:214-217`) — which is what makes
    /// `checkDsForStartAddr` answer EXISTS for a phase that was never created (unit e034). Both
    /// senses are kept. ⛔ An EMPTY `eps` answers `true`: the loop never runs.
    #[must_use]
    pub fn check_if_ds_not_exists(&self, ds: &StorageName, eps: &[ExPhase]) -> bool {
        eps.iter().all(|at| {
            self.entry(*at)
                .is_some_and(|held| !held.ds_in_mem.0.iter().any(|(key, _)| *key == *ds))
        })
    }

    /// Replaces: e019_removeDs
    ///
    /// Drops `ds` from each phase asked for — the map entry, the `free_` credit and, when strict, the
    /// block (`mem_track.cpp:439-454`).
    ///
    /// ⛔ A NO-OP FOR A NAME THAT IS ABSENT, and the L3 caller RELIES on that: it removes every
    /// candidate before placing any (`L3DlOpsScheduler.cpp:5609-5612`) so a retry cannot collide with
    /// its own previous attempt. ⛔ Takes [`BoundPhase`] because the reference's `epsToListIter.at`
    /// throws for an unbound phase and this signature has nowhere to put that.
    pub fn remove_ds(&mut self, ds: &StorageName, eps: &[BoundPhase]) {
        for at in eps {
            self.remove_ds_from(at.0, ds);
        }
    }

    /// Replaces: e020_removeDs
    ///
    /// The same removal in EVERY bound phase — the overload that walks `epsToListIter` itself
    /// (`mem_track.cpp:456-470`).
    ///
    /// ⛔ DISTINCT UNIT, SAME C++ NAME: unit e019 is the one that takes a phase list. ⛔ It walks the
    /// INDEX and not the list, so a node whose key `renameEps` moved is visited once, under the new
    /// key, and a phase the index no longer names is not visited at all.
    pub fn remove_ds_everywhere(&mut self, ds: &StorageName) {
        for at in self.phases.keys().copied().collect::<Vec<_>>() {
            self.remove_ds_from(at, ds);
        }
    }

    /// Replaces: e021_getCapOfDs
    ///
    /// The capacity one phase tracks `ds` under (`mem_track.cpp:519-529`).
    ///
    /// ⛔ [`None`] IS `INCOHERANT` (−3) AND IT ANSWERS TWO QUESTIONS AT ONCE: the phase is not bound,
    /// or it is bound and does not track the name. The reference cannot tell them apart either.
    /// ⛔ A CAPACITY AND NOT AN ADDRESS — it is the ROUNDED size `free_` was debited by.
    #[must_use]
    pub fn cap_of_ds(&self, at: ExPhase, name: &StorageName) -> Option<Capacity> {
        self.entry(at)?
            .ds_in_mem
            .0
            .iter()
            .find(|(key, _)| *key == *name)
            .map(|(_, cap)| *cap)
    }

    /// Replaces: e022_getAddrOfDs
    ///
    /// Where this phase's block list holds `ds`, for a name the phase tracks
    /// (`mem_track.cpp:531-541`).
    ///
    /// ⛔ TRAP: TWO DIFFERENT ABSENT ANSWERS FROM ONE FUNCTION, and neither is an address —
    /// [`DsAddress::Incoherant`] (−3) for an unbound phase or an untracked name, and
    /// [`DsAddress::Unplaced`] for the organizer's own −1. `getAddressOfDs` (unit e033) answers
    /// DOESNT_FIT (−2) to the same question, which is why the sentinel is an arm and not an integer.
    #[must_use]
    pub fn addr_of_ds(&self, at: ExPhase, name: &StorageName) -> DsAddress {
        let Some(held) = self.entry(at) else {
            return DsAddress::Incoherant;
        };
        if !held.ds_in_mem.0.iter().any(|(key, _)| *key == *name) {
            return DsAddress::Incoherant;
        }
        match held.occupied.find_address_by_name(name) {
            None => DsAddress::Unplaced,
            Some(address) => DsAddress::At(address),
        }
    }

    /// Replaces: e023_backupEps
    ///
    /// The snapshot `restoreEps` (unit e035) replays: every ds this phase tracks, its capacity, and
    /// the address its block list holds it at (`mem_track.cpp:566-578`).
    ///
    /// ⭐ THIS IS WHAT `ExPhaseTrackers::backup` IS NOW WIRED TO (`stages/carriers.rs`), where all
    /// 24,363 programs of the corpus used to stop; THE
    /// CALLER'S IDEMPOTENCE IS LOAD-BEARING: `trackerBackups.try_emplace` snapshots a tracker ONCE
    /// per `allocAllMem` (`L3DlOpsScheduler.cpp:5537-5543`), so a second backup would capture
    /// post-allocation state and `restoreEps` would commit the trial placement.
    /// ⛔ EMPTY FOR AN UNBOUND PHASE — the same answer as a phase holding nothing.
    #[must_use]
    pub fn backup_eps(&self, at: ExPhase) -> Vec<DsMemInfo> {
        let Some(held) = self.entry(at) else {
            return Vec::new();
        };
        held.ds_in_mem
            .0
            .iter()
            .map(|(key, cap)| DsMemInfo {
                ds: key.name().clone(),
                cap: *cap,
                start: held.occupied.find_address_by_name(key.name()),
            })
            .collect()
    }

    /// One phase per name, keyed `0..names.len()`, each holding just that ds at address 0 — the
    /// shape `formatMemTrack` (unit e028) binds, minted here because no filled unit reaches a
    /// phase-bound tracker yet and a renaming is only observable if the phases are TELLABLE APART.
    #[cfg(test)]
    pub(super) fn with_phases_holding(
        names: &[&str],
        capacity: Capacity,
        granularity: AllocGranularity,
    ) -> Self {
        use super::memory::{MemBlock, find_or_create_ds_key};
        use crate::schedule::ddc::v1::StorageName;

        let mut track = Self {
            mem_capacity: capacity,
            alloc_granularity: granularity,
            ..Self::default()
        };
        for (name, at) in names.iter().zip(0_u32..) {
            let key = find_or_create_ds_key(&StorageName((*name).to_owned()));
            let size = Capacity(granularity.0.get());
            let mut occupied = MemoryOrganizer::new(capacity);
            occupied.add_mem_block(MemBlock {
                name: key.clone(),
                start: Address::ZERO,
                size,
            });
            let entry = Entry {
                free: Capacity(capacity.0 - size.0),
                ds_in_mem: DsInMem(vec![(key, size)]),
                occupied,
            };
            let position = track.entries.len();
            let slot = track.insert_entry(position, entry);
            track.phases.insert(ExPhase(at), slot);
            track.ex_phases.0 += 1;
        }
        track
    }
}

impl DsTrackInMem {
    /// `Entry{memCapacity, {}, MemoryOrganizer(memCapacity)}` (`mem_track.cpp:117`, `:125-126`) — one
    /// phase holding nothing, with a block list OF ITS OWN. The reference's `resize` copies one
    /// prototype, so a port sharing an organizer between phases would compile and place twice.
    fn fresh_phase(&self) -> Entry {
        Entry {
            free: self.mem_capacity,
            ds_in_mem: DsInMem::default(),
            occupied: MemoryOrganizer::new(self.mem_capacity),
        }
    }

    /// `entries_.push_back(...)` then `epsToListIter[phase] = std::prev(entries_.end())`
    /// (`mem_track.cpp:117-118`, `:128-131`) — the append units e027 and e028 are both built from.
    ///
    /// ⛔ A NEGATIVE `phase` IS REACHABLE: `removeEps` (unit e006) decrements `exPhases` for every key
    /// it drops, so `growExPhases` can be asked to start below 0, and the reference happily keys its
    /// `unordered_map<int, ...>` with it. [`ExPhase`] is unsigned, so such a node joins the LIST —
    /// which [`Self::find_memory_bound`] walks — and gets no key, rather than a fabricated one.
    fn append_phase(&mut self, phase: i32) {
        let entry = self.fresh_phase();
        let position = self.entries.len();
        let slot = self.insert_entry(position, entry);
        if let Ok(key) = u32::try_from(phase) {
            self.phases.insert(ExPhase(key), slot);
        }
    }

    /// Replaces: e027_growExPhases
    ///
    /// Appends phases up to `new_ex_phases`, each holding nothing over its own `memCapacity` bytes,
    /// and raises the count only when the argument is greater (`mem_track.cpp:115-121`).
    ///
    /// ⛔ IT KEEPS WHAT THE EXISTING PHASES HOLD, and the header says why: growing through
    /// `formatMemTrack` would discard everything tracked so far (`mem_track.h:60-63`). ⛔ A SMALLER
    /// ARGUMENT IS A NO-OP — `if (newExPhases > exPhases)` (`:120`) will not lower the count.
    pub fn grow_ex_phases(&mut self, new_ex_phases: ExPhaseCount) {
        for phase in self.ex_phases.0..new_ex_phases.0 {
            self.append_phase(phase);
        }
        if new_ex_phases > self.ex_phases {
            self.ex_phases = new_ex_phases;
        }
    }

    /// Replaces: e028_formatMemTrack
    ///
    /// Throws every phase away and rebinds `[0, exPhases)` to fresh nodes, each over its own
    /// `memCapacity`-byte space (`mem_track.cpp:123-132`).
    ///
    /// ⛔ IT CLEARS, and `initMemTrack` calls it: init-then-grow keeps state, init-twice destroys it.
    /// ⛔ AND IT NEVER CLEARS A STALE KEY — `:128-131` rebinds only `[0, exPhases)`, so a phase bound
    /// at or above the count keeps a key naming an erased node (a dangling iterator in the reference).
    /// ⛔ A NEGATIVE `exPhases` IS REACHABLE (`removeEps`, unit e006) and the reference does NOT
    /// survive it: `entries_.resize(exPhases, ...)` (`:125`) asks a `size_type` for `-1` nodes.
    /// Here it clears and binds nothing, which is the state the reference had reached at `:124`.
    pub fn format_mem_track(&mut self) {
        // `entries_.clear()` (`:124`). The slots are tombstoned rather than dropped, so a stale key
        // reads as gone instead of aliasing one of the nodes appended next.
        for slot in std::mem::take(&mut self.entries) {
            if let Some(held) = self.slots.get_mut(slot.0) {
                *held = None;
            }
        }
        for phase in 0..self.ex_phases.0 {
            self.append_phase(phase);
        }
    }

    /// Replaces: e029_checkIfDsExists
    ///
    /// Whether `ds` is present in EVERY phase asked for — the inversion of unit e018
    /// (`mem_track.cpp:222-225`).
    ///
    /// ⛔ TRAP: AN UNBOUND PHASE ANSWERS "EXISTS", and an EMPTY `eps` answers `false`. Both fall out
    /// of `checkIfDsNotExists`' own `else { exists = false; break; }` (`:214-217`) and its loop, and
    /// the first is what makes `checkDsForStartAddr` (unit e034) refuse a phase never created.
    #[must_use]
    pub fn check_if_ds_exists(&self, ds: &StorageName, eps: &[ExPhase]) -> bool {
        !self.check_if_ds_not_exists(ds, eps)
    }

    /// Replaces: e030_findCommonFittingMemory
    ///
    /// ⭐ WHERE THE ADDRESS IS CHOSEN: intersects every phase's free-block list, then picks one block
    /// (`mem_track.cpp:269-346`). [`None`] is the −1, the `DT_CHECK(eps.size() >= 1)` and the throw.
    ///
    /// ⛔ THREE ARMS, ALL 187 PROGRAMS TAKE THE FORWARD ONE: [`FitRequest::At`] pins (`:317-325`); a
    /// back request — or `eps.size() > 10 && opt_frag_`, the SAME arm (`:327`) — takes the LAST
    /// fitting block at `iblk.second - cap + 1` (`:330`); otherwise the FIRST, at `iblk.first` (`:338`).
    #[must_use]
    pub fn find_common_fitting_memory(
        &self,
        cap: Capacity,
        eps: &[ExPhase],
        want: FitRequest,
    ) -> Option<Address> {
        let (first, rest) = eps.split_first()?;
        let mut intersect = self.entry(*first)?.occupied.all_free_blocks();
        for at in rest {
            let theirs = self.entry(*at)?.occupied.all_free_blocks();
            let mut narrowed = Vec::new();
            for iblk in &intersect {
                // Every block of theirs this one touches, ascending — and the walk STOPS at the first
                // block that starts above this one's end (`:294-301`).
                let mut overlapping = Vec::new();
                for entry in &theirs {
                    if (iblk.first >= entry.first && iblk.first <= entry.last)
                        || (iblk.last >= entry.first && iblk.last <= entry.last)
                        || (iblk.first <= entry.first && iblk.last >= entry.last)
                    {
                        overlapping.push(*entry);
                    }
                    if iblk.last < entry.first {
                        break;
                    }
                }
                // ⛔ ONLY THE ENDS ARE CLIPPED (`:303-312`): the FIRST overlap starts no lower than
                // this block and the LAST ends no higher, and a lone overlap is both.
                let count = overlapping.len();
                for (position, mut clipped) in overlapping.into_iter().enumerate() {
                    if position == 0 {
                        clipped.first = clipped.first.max(iblk.first);
                    }
                    if position + 1 == count {
                        clipped.last = clipped.last.min(iblk.last);
                    }
                    narrowed.push(clipped);
                }
            }
            intersect = narrowed;
        }
        match want {
            // ⛔ NO `break` IN THE REFERENCE HERE (`:318-324`) — every block is tested and the answer
            // is always `addrNeeded` itself, so one witness is the whole arm.
            FitRequest::At(pinned) => intersect
                .iter()
                .any(|iblk| {
                    let above = FreeBlock {
                        first: pinned,
                        last: iblk.last,
                    };
                    iblk.first <= pinned && iblk.last >= pinned && above.extent() >= cap
                })
                .then_some(pinned),
            FitRequest::AnywhereFromBack => Self::last_fitting(&intersect, cap),
            FitRequest::Anywhere => {
                if eps.len() > 10 && self.opt_frag == FragmentationOpt::On {
                    Self::last_fitting(&intersect, cap)
                } else {
                    intersect
                        .iter()
                        .find(|iblk| iblk.extent() >= cap)
                        .map(|iblk| iblk.first)
                }
            }
        }
    }

    /// `stAddr = iblk.second - cap + 1` over `reverse(intersectBlks.back())`
    /// (`mem_track.cpp:328-334`) — the back-fit BOTH paths into that arm share, so the two cannot
    /// drift apart.
    fn last_fitting(intersect: &[FreeBlock], cap: Capacity) -> Option<Address> {
        intersect
            .iter()
            .rev()
            .find(|iblk| iblk.extent() >= cap)
            .map(|iblk| Address(iblk.last.0 - cap.0 + 1))
    }

    /// Replaces: e031_addDsAtStartAddr
    ///
    /// THE COMMIT, and the whole of it: the `dsInMem_` entry, the `free_` debit and — when strict —
    /// the block itself, as ONE act per phase (`mem_track.cpp:377-393`).
    ///
    /// ⛔ ALL THREE OR NOTHING: the next request reads exactly these three, so recording the decision
    /// without mutating all of them hands out the same address twice. ⛔ `DT_CHECK(startAddr >= 0)`
    /// (`:380`) is discharged by the parameter's type — [`DsMemInfo::start`]'s [`None`] carries the −1.
    pub fn add_ds_at_start_addr(
        &mut self,
        ds: &StorageName,
        cap: Capacity,
        eps: &[BoundPhase],
        start: Address,
    ) {
        let strict = self.strict;
        for at in eps {
            let Some(held) = self.entry_mut(at.0) else {
                continue;
            };
            // `dsInMem_.insert({findOrCreateDsKey(ds), cap})` (`:383`) — and a `dt::SmallMap`
            // below its threshold UPDATES an existing key rather than dropping the insert
            // (`it->second = v.second`, `util/smallmap.hpp:269-271`), which is how a re-commit
            // changes a recorded capacity. ⛔ PAST THE THRESHOLD IT WOULD NOT: `DsInMem` is
            // `SmallMap<..., 10>` (`mem_track.h:44`) and an eleventh entry morphs it into a
            // `std::map`, whose `insert` (`:282`) leaves an existing value alone. The two arms
            // cannot be told apart through the ported surface — `checkAndAddDs` and
            // `checkAndAddDsAtAddr` reach this only past `checkDsForStartAddr`'s EXISTS
            // (`mem_track.cpp:353-355`, `:410`, `:432`) and `restoreEps` removes every name
            // first (`:585-594`), so no reachable call re-inserts a key it already holds.
            match held.ds_in_mem.0.iter().position(|(key, _)| *key == *ds) {
                Some(position) => held.ds_in_mem.0[position].1 = cap,
                None => held.ds_in_mem.0.push((find_or_create_ds_key(ds), cap)),
            }
            held.free -= cap;
            if strict == TrackingMode::AddressAssignment {
                held.occupied.allocate_memory(cap, ds, Placement::At(start));
            }
        }
    }
}

/// The one instance of the bundle's tracker seam ([`GrowExPhases`]), which is what makes unit e001's
/// delegation reach unit e027 rather than a generic parameter nothing implements.
impl GrowExPhases for DsTrackInMem {
    fn grow_ex_phases(&mut self, new_ex_phases: ExPhaseCount) {
        // Rust resolves the INHERENT method before this trait one, so this forwards to unit e027 and
        // does not recurse.
        DsTrackInMem::grow_ex_phases(self, new_ex_phases);
    }
}

impl DsTrackInMem {
    /// Replaces: e032_initMemTrack
    ///
    /// WHERE A CAPACITY AND A GRANULARITY ENTER: the five members, then a format, so the tracker
    /// comes out with `[0, ts)` bound to empty phases of `memCap` bytes each
    /// (`mem_track.cpp:107-112`).
    ///
    /// ⛔ IT CLEARS THROUGH UNIT e028 — a second init drops everything the first tracked, which is
    /// why `MemTrackBundle` skips the WHOLE per-core family when one is already there, on
    /// `initLx = lxTrackPerCore.empty()` and not on any tracker's contents
    /// (`mem_track_bundle.cpp:50`). ⛔ THE DEFAULTS ARE `allocGran = 1` AND `isStrict = true`
    /// (`mem_track.h:57-58`): LX ON THIS PATH takes ("lxCore<c>", 2031616, numSteps, 128) strict
    /// (`mem_track_bundle.cpp:59`, `sysdef.cpp:206`, `:211`), while `dsm/dsmperf.cpp:2894` seeds one
    /// at granularity 1 with `isStrict=false` — the [`TrackingMode::CapacityOnly`] tracker unit
    /// e034's real-address-0 trap exists for.
    pub fn init_mem_track(
        &mut self,
        mem_name: MemName,
        mem_cap: Capacity,
        ts: ExPhaseCount,
        alloc_gran: AllocGranularity,
        is_strict: TrackingMode,
    ) {
        self.name = mem_name;
        self.mem_capacity = mem_cap;
        self.alloc_granularity = alloc_gran;
        self.ex_phases = ts;
        self.strict = is_strict;
        self.format_mem_track();
    }

    /// Replaces: e033_getAddressOfDs
    ///
    /// Where ONE phase's block list holds `ds` (`mem_track.cpp:161-171`).
    ///
    /// ⛔ TRAP: `DOESNT_FIT` (−2) FOR AN UNBOUND PHASE — unit e029 answers "exists" there, so it is
    /// the `epsToListIter.find` inside that `if` (`:163`) refusing it, and unit e022 gives
    /// `INCOHERANT` (−3) to those very same two states. ⛔ [`DsAddress::Unplaced`] is still the
    /// organizer's own −1.
    #[must_use]
    pub fn get_address_of_ds(&self, ds: &StorageName, curr_ep: ExPhase) -> DsAddress {
        if !self.check_if_ds_exists(ds, &[curr_ep]) {
            return DsAddress::DoesntFit;
        }
        let Some(held) = self.entry(curr_ep) else {
            return DsAddress::DoesntFit;
        };
        match held.occupied.find_address_by_name(ds) {
            Some(address) => DsAddress::At(address),
            None => DsAddress::Unplaced,
        }
    }

    /// Replaces: e034_checkDsForStartAddr
    ///
    /// THE THREE-WAY DECISION: `EXISTS` for a name already there, `DOESNT_FIT` when either the free
    /// counters or the free lists refuse it, else the address — and it COMMITS NOTHING
    /// (`mem_track.cpp:353-374`).
    ///
    /// ⛔ TRAP: A [`TrackingMode::CapacityOnly`] TRACKER ANSWERS A REAL ADDRESS 0 (`:371`), not a
    /// refusal, and it is indistinguishable from a strict placement at address 0. ⛔ THE TWO
    /// `DOESNT_FIT`s ARE DIFFERENT CHECKS: the free counter of every phase (unit e004) and then the
    /// intersected free lists (unit e030). ⛔ `EXISTS` also covers a phase that was never bound, out
    /// of unit e018's inverted `else`.
    #[must_use]
    pub fn check_ds_for_start_addr(
        &self,
        ds: &StorageName,
        cap: Capacity,
        eps: &[ExPhase],
        want: FitRequest,
        margin: Margin,
    ) -> DsAddress {
        if !self.check_if_ds_not_exists(ds, eps) {
            return DsAddress::Exists;
        }
        if !self.check_if_ds_fits(cap, eps, margin) {
            return DsAddress::DoesntFit;
        }
        match self.strict {
            TrackingMode::CapacityOnly => DsAddress::At(Address::ZERO),
            // `if (startAddr < 0) return DOESNT_FIT` (`:365-366`): every negative the reference can
            // reach here is unit e030's own −1, which that port already carries as [`None`].
            TrackingMode::AddressAssignment => match self.find_common_fitting_memory(cap, eps, want)
            {
                None => DsAddress::DoesntFit,
                Some(start) => DsAddress::At(start),
            },
        }
    }

    /// Replaces: e035_restoreEps
    ///
    /// THE UNDO HALF OF `backupEps`: clears the phase over a SNAPSHOT of its own names, then re-adds
    /// every backup entry at its RECORDED address — so a restore is EXACT and not a re-pack, and a
    /// failed trial allocation leaves no trace (`mem_track.cpp:582-595`).
    ///
    /// ⛔ A PHASE THAT IS NOT BOUND IS A NO-OP (`:582`) and does not even clear. ⛔ AN UNPLACED
    /// BACKUP ENTRY STOPS THE REPLAY EXACTLY WHERE `addDsAtStartAddr`'s `DT_CHECK(startAddr >= 0)`
    /// (`:380`) throws: every entry before it is back, that one AND EVERYTHING BEHIND IT is not,
    /// and no address is invented. ⛔ STATED DIVERGENCE, OFF THIS PATH: the reference throws out of
    /// its caller where this returns to it with the phase cleared, and only a
    /// [`TrackingMode::CapacityOnly`] tracker reaches it — unit e031 commits a block for every entry
    /// a strict tracker records, so a strict backup never carries [`None`].
    pub fn restore_eps(&mut self, at: ExPhase, all_info: &[DsMemInfo]) {
        let Some(bound) = self.bound_phase(at) else {
            return;
        };
        // The names are copied out FIRST (`:585-587`), because the removal walks the very map it is
        // erasing from.
        let old_ds: Vec<StorageName> = self
            .all_ds_and_size_at_eps(at)
            .map(|held| held.0.iter().map(|(key, _)| key.name().clone()).collect())
            .unwrap_or_default();
        for ds in &old_ds {
            self.remove_ds(ds, std::slice::from_ref(&bound));
        }
        for info in all_info {
            let Some(start) = info.start else {
                return;
            };
            self.add_ds_at_start_addr(&info.ds, info.cap, std::slice::from_ref(&bound), start);
        }
    }
}

impl DsTrackInMem {
    /// The decide-then-commit BOTH entry points share (`mem_track.cpp:406-414`, `:428-436`), so the
    /// two cannot drift: the request is rounded UP once and `capRound` — never `cap` — is committed,
    /// because `free_` and the block list track ROUNDED sizes.
    ///
    /// ⛔ [`DsAddress::At`] IS EXACTLY `startAddr != EXISTS && startAddr != DOESNT_FIT`: unit e034
    /// reaches no other arm, so both sentinels come back unchanged and neither commits anything.
    fn check_and_add(
        &mut self,
        ds: &StorageName,
        cap: Capacity,
        eps: &[ExPhase],
        want: FitRequest,
        margin: Margin,
    ) -> DsAddress {
        let cap_round = self.alloc_granularity.round_up(cap);
        let start = self.check_ds_for_start_addr(ds, cap_round, eps, want, margin);
        let DsAddress::At(start_addr) = start else {
            return start;
        };
        // `addDsAtStartAddr` reaches `epsToListIter.at(currEp)` (`:382`), which THROWS on an unbound
        // phase — and cannot be reached with one: `checkIfDsNotExists` (`:353`, unit e018) BREAKS OUT
        // OF ITS WALK WITH "exists" for a phase that is not in `epsToListIter` at all (`:214-217`), so
        // the line above answered [`DsAddress::Exists`] and returned. ⛔ NOT unit e029, which is only
        // `!checkIfDsNotExists` (`:222-224`) and is called from neither entry point. Keeping only the
        // bound phases therefore drops nothing the reference commits, and keeps duplicates.
        let bound: Vec<BoundPhase> = eps.iter().filter_map(|at| self.bound_phase(*at)).collect();
        self.add_ds_at_start_addr(ds, cap_round, &bound, start_addr);
        start
    }

    /// Replaces: e037_checkAndAddDs
    ///
    /// ⭐ THE ENTRY POINT THE ORACLE MEASURES: round up, choose an address at one end or the other,
    /// commit there (`mem_track.cpp:395-415`).
    ///
    /// ⛔ THE MEASURED LAW, 588 LX allocations over 187 reference programs: at granularity 128 over
    /// 2,031,616 bytes with `[0, 1625344)` held by `reserved-frontend`, forward FIRST FIT puts the
    /// first request at EXACTLY 1,625,344 and every corpus address at `1625344 + k*512` — back-fit
    /// reproduces 0 of 187 and no reservation 0 of 187.
    /// ⛔ A SENTINEL IS AS ORDINARY AN ANSWER HERE AS AN ADDRESS, and a refusal commits nothing.
    pub fn check_and_add_ds(
        &mut self,
        ds: &StorageName,
        cap: Capacity,
        eps: &[ExPhase],
        margin: Margin,
        from: AllocEnd,
    ) -> DsAddress {
        // The `-1` of `checkDsForStartAddr(ds, capRound, eps, -1, margin, allocFromBack)` (`:408-409`)
        // is "anywhere" and not an address, so the only choice this call makes is the end.
        let want = match from {
            AllocEnd::Front => FitRequest::Anywhere,
            AllocEnd::Back => FitRequest::AnywhereFromBack,
        };
        self.check_and_add(ds, cap, eps, want, margin)
    }

    /// Replaces: e038_checkAndAddDsAtAddr
    ///
    /// ⭐ THE CALL THAT SEEDS THE ORACLE — `reserveFrontendLx` pins `reserved-frontend` through this
    /// one (`mem_track.cpp:417-437`, `dbo/src/Transforms/ProgramLayout.cpp:93`).
    ///
    /// ⛔ IT ROUNDS THE PIN'S SIZE UP TOO: 1,625,292 bytes asked for at address 0 become
    /// `[0, 1625344)`, and that rounding is what puts every later LX address where the corpus has it.
    /// ⛔ THE REFERENCE SWITCHES ON THE SIGN OF `addrNeeded`, NOT ON WHETHER ONE WAS GIVEN (`:317`),
    /// and [`Address`] must stay signed because a zero-byte space's own free block is `Address(-1)` —
    /// so a NEGATIVE `Some` takes the SAME arm as [`None`]. That arm is [`FitRequest::Anywhere`], which
    /// is the FIRST fitting block only while `eps.size() > 10 && opt_frag_` is false and the LAST one
    /// when it is not (`:326-342`) — it is not unconditionally the front. Every authority call site
    /// passes a non-negative address, two of them behind their own `DT_CHECK(addrNeeded >= 0)`
    /// (`perfDscToSdsc.cpp:666`, `sengraphToPerfDscTranslator.cpp:80`), so the negative arm is untested.
    pub fn check_and_add_ds_at_addr(
        &mut self,
        ds: &StorageName,
        cap: Capacity,
        eps: &[ExPhase],
        addr_needed: Option<Address>,
        margin: Margin,
    ) -> DsAddress {
        // `if (addrNeeded >= 0)` (`:317`): a negative pin is NOT a pin, and `Some` is not the test.
        let want = match addr_needed {
            Some(pinned) if pinned >= Address::ZERO => FitRequest::At(pinned),
            Some(_) | None => FitRequest::Anywhere,
        };
        self.check_and_add(ds, cap, eps, want, margin)
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{
        AllocEnd, AllocGranularity, DsAddress, DsInMem, DsMemInfo, DsTrackInMem, Entry, EpsEdit,
        ExPhase, ExPhaseCount, FitRequest, FragmentationOpt, Margin, MemName, TrackingMode,
    };
    use super::super::memory::{Address, Capacity, MemBlock, MemoryOrganizer, find_or_create_ds_key};
    use crate::schedule::ddc::v1::StorageName;
    use std::num::NonZeroI64;

    /// LX's whole space — `lxCapacity = 2*1024*1024 - 64*1024` (`sys-arch-spec/sysdef.cpp:211`).
    const LX_CAPACITY: Capacity = Capacity(2_031_616);

    /// `allocGranularity` as `MemTrackBundle` hands it to every LX tracker: `bytesPerStick` = 128
    /// (`sys-arch-spec/sysdef.cpp:206`).
    const BYTES_PER_STICK: AllocGranularity =
        AllocGranularity(NonZeroI64::new(128).expect("a stick is not zero bytes"));

    /// What `reserveFrontendLx` asks for and what the tracker rounds it to:
    /// `(int64_t)(2031616 * (1 - 0.2))` = 1,625,292 → 1,625,344 (`ProgramLayout.cpp:81-98`).
    const RESERVED_FRONT_END: Capacity = Capacity(1_625_344);

    /// What is left above it — 2,031,616 − 1,625,344.
    const FREE_ABOVE_RESERVATION: Capacity = Capacity(406_272);

    /// An empty organizer of `capacity` bytes — what `MemoryOrganizer(int64_t cap)` seeds
    /// (`memory.cpp:21`, unit e024): `capacity_ = cap`, then `clear()`.
    fn space(capacity: Capacity) -> MemoryOrganizer {
        MemoryOrganizer::new(capacity)
    }

    /// One phase holding nothing, the way `growExPhases` builds its prototype:
    /// `Entry{memCapacity, {}, MemoryOrganizer(memCapacity)}` (`mem_track.cpp:117`, unit e027).
    fn phase(capacity: Capacity) -> Entry {
        Entry {
            free: capacity,
            ds_in_mem: DsInMem::default(),
            occupied: space(capacity),
        }
    }

    /// A tracker of `count` empty LX phases keyed `0..count`, as `formatMemTrack` binds them
    /// (`mem_track.cpp:123-132`, unit e028).
    fn lx_tracker(count: u32) -> DsTrackInMem {
        let mut track = DsTrackInMem {
            mem_capacity: LX_CAPACITY,
            alloc_granularity: BYTES_PER_STICK,
            ..DsTrackInMem::default()
        };
        for at in 0..count {
            let position = track.entries.len();
            let slot = track.insert_entry(position, phase(LX_CAPACITY));
            track.phases.insert(ExPhase(at), slot);
            track.ex_phases.0 += 1;
        }
        track
    }

    /// One LX phase that already holds `reserved-frontend` at [0, 1625344), which is the state every
    /// real placement starts from and the state the oracle's 588 LX addresses were measured in.
    fn lx_with_reserved_front_end() -> DsTrackInMem {
        let mut track = lx_tracker(1);
        record(&mut track, ExPhase(0), "reserved-frontend", 0, RESERVED_FRONT_END.0);
        track
    }

    /// Commits one block to a phase the way `addDsAtStartAddr` does — the map entry, the `free_`
    /// decrement and the `occupied_` block as one act (`mem_track.cpp:377-393`, unit e031).
    fn record(track: &mut DsTrackInMem, at: ExPhase, name: &str, start: i64, size: i64) {
        let Some(slot) = track.phases.get(&at).copied() else {
            return;
        };
        let Some(Some(held)) = track.slots.get_mut(slot.0) else {
            return;
        };
        let key = find_or_create_ds_key(&StorageName(name.to_owned()));
        held.ds_in_mem.0.push((key.clone(), Capacity(size)));
        held.free.0 -= size;
        held.occupied.add_mem_block(MemBlock {
            name: key,
            start: Address(start),
            size: Capacity(size),
        });
    }

    /// 🎯 003/38 A PHASE'S MAP IS WHAT THAT PHASE HOLDS, AND AN UNBOUND PHASE IS THE REFERENCE'S
    /// `DT_ERROR("Request Ds map for an unknown eps")` — `mem_track.cpp:98-101`.
    #[test]
    fn the_ds_map_is_the_asked_for_phases_own_and_an_unbound_phase_has_none() {
        let track = lx_with_reserved_front_end();

        let held = track.all_ds_and_size_at_eps(ExPhase(0));
        assert_eq!(
            held.map(|map| map.0.as_slice()),
            Some(
                [(
                    find_or_create_ds_key(&StorageName("reserved-frontend".to_owned())),
                    RESERVED_FRONT_END,
                )]
                .as_slice()
            ),
        );
        // ⛔ Phase 1 was never bound: the reference does not reach its `return` at all.
        assert_eq!(track.all_ds_and_size_at_eps(ExPhase(1)), None);
    }

    /// 🎯 004/38 THE PRECHECK ROUNDS UP TO GRANULARITY AND REFUSES ON `free_ < capRound + margin`,
    /// so an EXACT fill of the free counter FITS — `mem_track.cpp:137-143`.
    #[test]
    fn the_fit_check_rounds_up_takes_an_exact_fill_and_refuses_one_stick_more() {
        let track = lx_with_reserved_front_end();
        let phases = [ExPhase(0)];

        // 406,272 free, and 406,272 is already a whole number of 128-byte sticks: `<` takes it.
        assert!(track.check_if_ds_fits(FREE_ABOVE_RESERVATION, &phases, Margin::NONE));
        // ⛔ One byte more rounds UP to 406,400 — a whole stick more than is free.
        assert!(!track.check_if_ds_fits(Capacity(406_273), &phases, Margin::NONE));
        // ⛔ `margin` is demanded but never reserved: it turns the exact fill above into a refusal.
        assert!(!track.check_if_ds_fits(
            FREE_ABOVE_RESERVATION,
            &phases,
            Margin(Capacity(128)),
        ));
        // ⛔ A phase that is not bound does not fit, whatever is being asked for.
        assert!(!track.check_if_ds_fits(Capacity(0), &[ExPhase(0), ExPhase(1)], Margin::NONE));
    }

    /// 🎯 005/38 THE NEW PHASE IS A COPY OF THE CANDIDATE PLACED BEFORE IT — `entries_.insert(cand,
    /// *cand)` then `epsToListIter[newEps] = std::prev(cand)`, `exPhases++` (`mem_track.cpp:478-480`).
    #[test]
    fn a_phase_added_before_another_is_a_copy_of_it_and_the_count_rises() {
        let mut track = lx_with_reserved_front_end();

        assert_eq!(track.add_eps_before(ExPhase(1), ExPhase(0)), EpsEdit::Done);
        assert_eq!(track.ex_phases.0, 2);
        // ⛔ The copy carries `dsInMem_`, `free_` AND the block list, so both phases now hold the
        // reservation and both report the same bound.
        assert_eq!(
            track.all_ds_and_size_at_eps(ExPhase(1)),
            track.all_ds_and_size_at_eps(ExPhase(0)),
        );
        assert_eq!(track.free_cap_at_eps(ExPhase(1)), FREE_ABOVE_RESERVATION);
        // ⛔ EXISTS when the new key is bound; INCOHERANT when the candidate is not. Different answers.
        assert_eq!(track.add_eps_before(ExPhase(1), ExPhase(0)), EpsEdit::Exists);
        assert_eq!(track.add_eps_before(ExPhase(7), ExPhase(9)), EpsEdit::Incoherant);
        assert_eq!(track.ex_phases.0, 2);
    }

    /// 🎯 006/38 A REMOVED PHASE LEAVES BOTH THE LIST AND THE INDEX, AND AN UNBOUND ONE IS A NO-OP —
    /// `mem_track.cpp:490-494`.
    #[test]
    fn removing_a_phase_drops_its_node_and_removing_an_unbound_one_changes_nothing() {
        let mut track = lx_with_reserved_front_end();

        track.remove_eps(ExPhase(0));
        assert_eq!(track.ex_phases.0, 0);
        assert_eq!(track.all_ds_and_size_at_eps(ExPhase(0)), None);
        // ⛔ Its node left `entries_` too, so the tracker is back to the −1 seed of unit e009.
        assert_eq!(track.find_memory_bound(), None);
        // ⛔ No `DT_ERROR` on the way out, and the count does not follow the call.
        track.remove_eps(ExPhase(0));
        assert_eq!(track.ex_phases.0, 0);
    }

    /// 🎯 007/38 A RENAME MOVES THE KEY AND NOT THE NODE, WILL NOT OVERWRITE A BOUND TARGET, AND DOES
    /// NOT MOVE `exPhases` — `mem_track.cpp:499-507`.
    #[test]
    fn renaming_a_phase_moves_its_key_and_refuses_a_bound_target() {
        let mut track = lx_with_reserved_front_end();

        assert_eq!(track.rename_eps(ExPhase(0), ExPhase(5)), EpsEdit::Done);
        assert_eq!(track.free_cap_at_eps(ExPhase(5)), FREE_ABOVE_RESERVATION);
        assert_eq!(track.all_ds_and_size_at_eps(ExPhase(0)), None);
        // ⛔ One node, one key, renamed: the count is untouched where `addEpsBefore` raises it.
        assert_eq!(track.ex_phases.0, 1);
        // ⛔ EXISTS, and phase 5 still holds what it held — this is why the bundle walks DOWNWARD.
        assert_eq!(track.rename_eps(ExPhase(5), ExPhase(5)), EpsEdit::Exists);
        assert_eq!(track.free_cap_at_eps(ExPhase(5)), FREE_ABOVE_RESERVATION);
        assert_eq!(track.rename_eps(ExPhase(3), ExPhase(4)), EpsEdit::Incoherant);
    }

    /// 🎯 008/38 THE FREE COUNTER IS THE PHASE'S OWN, AND AN UNBOUND PHASE ANSWERS 0 RATHER THAN
    /// REFUSING — `mem_track.cpp:513-515`.
    #[test]
    fn the_free_capacity_is_the_phases_own_and_an_unbound_phase_answers_zero() {
        let track = lx_with_reserved_front_end();

        assert_eq!(track.free_cap_at_eps(ExPhase(0)), FREE_ABOVE_RESERVATION);
        // ⛔ 0 is the reference's answer for a phase it never bound, and its caller cannot tell that
        // from a phase with nothing left. Carried as the value, not as a refusal.
        assert_eq!(track.free_cap_at_eps(ExPhase(1)), Capacity(0));
        assert_eq!(lx_tracker(1).free_cap_at_eps(ExPhase(0)), LX_CAPACITY);
    }

    /// 🎯 009/38 THE TRACKER'S BOUND IS THE MAXIMUM OVER EVERY PHASE'S ORGANIZER, SEEDED AT −1 —
    /// `mem_track.cpp:557-562`, over `MemoryOrganizer::findMemoryBound` (unit e014).
    #[test]
    fn the_tracker_bound_is_the_highest_of_its_phases_and_minus_one_only_when_it_has_none() {
        // ⛔ −1 IS NOT 0: with no phases at all the fold never runs, while a phase holding nothing
        // answers address 0 (`memory.cpp:71`).
        assert_eq!(DsTrackInMem::default().find_memory_bound(), None);
        assert_eq!(lx_tracker(2).find_memory_bound(), Some(Address::ZERO));

        let mut track = lx_tracker(2);
        record(&mut track, ExPhase(0), "reserved-frontend", 0, RESERVED_FRONT_END.0);
        record(&mut track, ExPhase(0), "Tensor0_lx", 1_625_344, 512);
        record(&mut track, ExPhase(1), "reserved-frontend", 0, RESERVED_FRONT_END.0);
        // Phase 0 reaches 1,625,856; phase 1 stops at 1,625,344. The maximum is the tracker's bound.
        assert_eq!(track.find_memory_bound(), Some(Address(1_625_856)));
        // ⛔ It is a maximum over the LIST, so it does not fall when a lower phase is added first.
        assert_eq!(track.add_eps_before(ExPhase(2), ExPhase(1)), EpsEdit::Done);
        assert_eq!(track.find_memory_bound(), Some(Address(1_625_856)));
    }

    /// Records a ds in the map and the free counter with NO block, which is exactly what
    /// `addDsAtStartAddr` does when `strict` is false (`mem_track.cpp:388-392`, unit e031).
    fn record_capacity_only(track: &mut DsTrackInMem, at: ExPhase, name: &str, size: i64) {
        let Some(slot) = track.phases.get(&at).copied() else {
            return;
        };
        let Some(Some(held)) = track.slots.get_mut(slot.0) else {
            return;
        };
        let key = find_or_create_ds_key(&StorageName(name.to_owned()));
        held.ds_in_mem.0.push((key, Capacity(size)));
        held.free.0 -= size;
    }

    /// 🎯 018/38 IT ANSWERS `true` FOR A DS ABSENT FROM EVERY PHASE ASKED FOR, AND `false` — "ALREADY
    /// EXISTS" — FOR A PHASE THAT WAS NEVER BOUND: `mem_track.cpp:205-219`.
    #[test]
    fn the_absence_check_is_inverted_and_an_unbound_phase_reads_as_already_existing() {
        let track = lx_with_reserved_front_end();
        let reserved = StorageName("reserved-frontend".to_owned());
        let fresh = StorageName("Tensor0_lx".to_owned());

        assert!(track.check_if_ds_not_exists(&fresh, &[ExPhase(0)]));
        assert!(!track.check_if_ds_not_exists(&reserved, &[ExPhase(0)]));
        // ⛔ Phase 1 is not bound, and the answer is FALSE for a name nothing holds — the `else` arm
        // breaks with `exists = false`, which is what makes unit e034 answer EXISTS there.
        assert!(!track.check_if_ds_not_exists(&fresh, &[ExPhase(0), ExPhase(1)]));
        // ⛔ An empty phase list never enters the loop, so the `true` seed is the answer.
        assert!(track.check_if_ds_not_exists(&reserved, &[]));
    }

    /// 🎯 019/38 REMOVAL CREDITS `free_`, DROPS THE MAP ENTRY AND FREES THE BLOCK, IN THE PHASES ASKED
    /// FOR ALONE — `mem_track.cpp:441-452`.
    #[test]
    fn removing_a_ds_credits_free_and_frees_the_block_in_that_phase_alone() {
        let mut track = lx_tracker(2);
        record(&mut track, ExPhase(0), "reserved-frontend", 0, RESERVED_FRONT_END.0);
        record(&mut track, ExPhase(0), "Tensor0_lx", 1_625_344, 512);
        record(&mut track, ExPhase(1), "Tensor0_lx", 1_625_344, 512);
        let tensor = StorageName("Tensor0_lx".to_owned());
        let phase = [track.bound_phase(ExPhase(0)).expect("phase 0 is bound")];

        track.remove_ds(&tensor, &phase);

        // 405,760 + 512 = 406,272, the map entry gone, and phase 0's block list back to the
        // reservation alone.
        assert_eq!(track.free_cap_at_eps(ExPhase(0)), FREE_ABOVE_RESERVATION);
        assert_eq!(track.cap_of_ds(ExPhase(0), &tensor), None);
        assert_eq!(track.addr_of_ds(ExPhase(0), &tensor), DsAddress::Incoherant);
        // ⛔ PHASE 1 IS UNTOUCHED: it still holds the block, so the tracker's bound is still its.
        assert_eq!(track.addr_of_ds(ExPhase(1), &tensor), DsAddress::At(Address(1_625_344)));
        assert_eq!(track.find_memory_bound(), Some(Address(1_625_856)));
        // ⛔ A name that is absent is a NO-OP — the L3 caller removes every candidate before placing
        // any, so this path runs on every retry.
        track.remove_ds(&tensor, &phase);
        assert_eq!(track.free_cap_at_eps(ExPhase(0)), FREE_ABOVE_RESERVATION);
    }

    /// 🎯 020/38 THE OVERLOAD THAT WALKS THE INDEX CLEARS THE NAME FROM EVERY BOUND PHASE, AND THE
    /// `if (strict)` GUARD IS THE ONLY THING THAT TOUCHES THE BLOCK LIST — `mem_track.cpp:457-470`.
    #[test]
    fn removing_a_ds_everywhere_clears_every_phase_and_capacity_only_keeps_the_block() {
        let mut track = lx_tracker(2);
        record(&mut track, ExPhase(0), "Tensor0_lx", 1_625_344, 512);
        record(&mut track, ExPhase(1), "Tensor0_lx", 1_625_344, 512);
        let tensor = StorageName("Tensor0_lx".to_owned());

        track.remove_ds_everywhere(&tensor);

        assert_eq!(track.free_cap_at_eps(ExPhase(0)), LX_CAPACITY);
        assert_eq!(track.free_cap_at_eps(ExPhase(1)), LX_CAPACITY);
        assert_eq!(track.find_memory_bound(), Some(Address::ZERO));

        // ⛔ THE ARM THE 187-PROGRAM CORPUS NEVER TAKES: with `strict` false the map entry and `free_`
        // move and the block list does NOT. Reached here by flipping the mode on a placed phase,
        // which is the only state in which the guard is observable at all.
        let mut lenient = lx_tracker(1);
        record(&mut lenient, ExPhase(0), "Tensor0_lx", 1_625_344, 512);
        lenient.strict = TrackingMode::CapacityOnly;
        lenient.remove_ds_everywhere(&tensor);
        assert_eq!(lenient.free_cap_at_eps(ExPhase(0)), LX_CAPACITY);
        assert_eq!(lenient.cap_of_ds(ExPhase(0), &tensor), None);
        assert_eq!(lenient.find_memory_bound(), Some(Address(1_625_856)));
    }

    /// 🎯 021/38 THE CAPACITY IS THE PHASE'S OWN RECORD, AND `INCOHERANT` (−3) IS THE ANSWER TO BOTH
    /// ABSENCES — `mem_track.cpp:521-527`.
    #[test]
    fn the_ds_capacity_is_the_tracked_one_and_both_absences_are_incoherant() {
        let track = lx_with_reserved_front_end();
        let reserved = StorageName("reserved-frontend".to_owned());

        assert_eq!(track.cap_of_ds(ExPhase(0), &reserved), Some(RESERVED_FRONT_END));
        // ⛔ `None` IS −3, for a name this phase does not track...
        assert_eq!(track.cap_of_ds(ExPhase(0), &StorageName("Tensor0_lx".to_owned())), None);
        // ⛔ ...and for a phase that was never bound. The reference gives −3 to both.
        assert_eq!(track.cap_of_ds(ExPhase(1), &reserved), None);
    }

    /// 🎯 022/38 THE ADDRESS IS THE BLOCK LIST'S OWN, AND THE TWO ABSENT ANSWERS ARE DIFFERENT ARMS:
    /// INCOHERANT (−3) against the organizer's own −1 — `mem_track.cpp:533-540`.
    #[test]
    fn the_ds_address_is_the_block_lists_and_incoherant_is_not_the_organizers_minus_one() {
        let track = lx_with_reserved_front_end();
        let reserved = StorageName("reserved-frontend".to_owned());

        // Address 0 is a REAL address: `reserveFrontendLx` pins the reservation at 0.
        assert_eq!(track.addr_of_ds(ExPhase(0), &reserved), DsAddress::At(Address::ZERO));
        // ⛔ −3 for a phase that is not bound, and −3 again for a name it does not track.
        assert_eq!(track.addr_of_ds(ExPhase(1), &reserved), DsAddress::Incoherant);
        assert_eq!(
            track.addr_of_ds(ExPhase(0), &StorageName("Tensor0_lx".to_owned())),
            DsAddress::Incoherant,
        );
        // ⛔ THE OTHER SENTINEL: a `CapacityOnly` tracker tracks names no block holds, and the
        // organizer answers its own −1 there — a DIFFERENT answer that shares EXISTS's number.
        let mut lenient = lx_tracker(1);
        lenient.strict = TrackingMode::CapacityOnly;
        record_capacity_only(&mut lenient, ExPhase(0), "Tensor0_lx", 512);
        assert_eq!(
            lenient.addr_of_ds(ExPhase(0), &StorageName("Tensor0_lx".to_owned())),
            DsAddress::Unplaced,
        );
    }

    /// 🎯 023/38 THE BACKUP CARRIES NAME, CAPACITY AND RECORDED ADDRESS FOR EVERY DS OF ONE PHASE, AND
    /// AN UNBOUND PHASE BACKS UP NOTHING — `mem_track.cpp:568-577`.
    #[test]
    fn a_phase_backs_up_every_ds_with_its_capacity_and_its_recorded_address() {
        let mut track = lx_with_reserved_front_end();
        record(&mut track, ExPhase(0), "Tensor0_lx", 1_625_344, 512);

        let backup = track.backup_eps(ExPhase(0));

        // ⛔ MATCHED BY CONTENT AND NOT BY POSITION: the reference's own walk order is the interned
        // pointer order of `dt::SmallMap` (see [`DsInMem`]), and nothing in scope reads it.
        assert_eq!(backup.len(), 2);
        assert!(backup.contains(&DsMemInfo {
            ds: StorageName("reserved-frontend".to_owned()),
            cap: RESERVED_FRONT_END,
            start: Some(Address::ZERO),
        }));
        assert!(backup.contains(&DsMemInfo {
            ds: StorageName("Tensor0_lx".to_owned()),
            cap: Capacity(512),
            start: Some(Address(1_625_344)),
        }));
        // ⛔ Empty for a phase that is not bound, which the reference does not distinguish from a
        // phase holding nothing.
        assert!(track.backup_eps(ExPhase(1)).is_empty());
        // ⛔ `None` IS THE ORGANIZER'S −1 TRAVELLING INTO `DsMemInfo::startAddr`: a `CapacityOnly`
        // tracker backs up a ds with no address, and that is where unit e035's `DT_CHECK` stops.
        let mut lenient = lx_tracker(1);
        lenient.strict = TrackingMode::CapacityOnly;
        record_capacity_only(&mut lenient, ExPhase(0), "Tensor0_lx", 512);
        assert_eq!(
            lenient.backup_eps(ExPhase(0)),
            [DsMemInfo {
                ds: StorageName("Tensor0_lx".to_owned()),
                cap: Capacity(512),
                start: None,
            }],
        );
    }

    /// 🎯 027/38 GROWING APPENDS PHASES OVER THE WHOLE SPACE, KEEPS WHAT THE EXISTING PHASES HOLD, AND
    /// NEVER LOWERS THE COUNT — `mem_track.cpp:115-121`, `mem_track.h:60-63`.
    #[test]
    fn growing_appends_empty_phases_keeps_the_old_ones_and_never_lowers_the_count() {
        let mut track = lx_with_reserved_front_end();

        track.grow_ex_phases(ExPhaseCount(3));

        // ⛔ PHASE 0 IS UNTOUCHED — the whole reason `formatMemTrack` cannot be used to grow.
        assert_eq!(track.free_cap_at_eps(ExPhase(0)), FREE_ABOVE_RESERVATION);
        // `Entry{memCapacity, {}, MemoryOrganizer(memCapacity)}` (`:117`): free is the WHOLE space.
        assert_eq!(track.free_cap_at_eps(ExPhase(1)), LX_CAPACITY);
        assert_eq!(track.free_cap_at_eps(ExPhase(2)), LX_CAPACITY);
        assert_eq!(track.ex_phases, ExPhaseCount(3));

        // ⛔ `if (newExPhases > exPhases)` (`:120`): a smaller argument adds nothing, drops nothing and
        // does not move the count.
        track.grow_ex_phases(ExPhaseCount(1));
        assert_eq!(track.ex_phases, ExPhaseCount(3));
        assert_eq!(track.free_cap_at_eps(ExPhase(2)), LX_CAPACITY);
    }

    /// 🎯 028/38 FORMATTING CLEARS EVERY PHASE, REBINDS EXACTLY `[0, exPhases)` TO NODES WITH THEIR OWN
    /// BLOCK LISTS, AND LEAVES A STALE KEY BEHIND — `mem_track.cpp:123-132`.
    #[test]
    fn formatting_clears_every_phase_rebinds_the_first_ex_phases_and_strands_the_rest() {
        let mut track = lx_with_reserved_front_end();
        // A second phase, bound ABOVE `[0, exPhases)` and holding the same reservation (unit e005).
        assert_eq!(track.add_eps_before(ExPhase(7), ExPhase(0)), EpsEdit::Done);
        assert_eq!(track.ex_phases, ExPhaseCount(2));

        track.format_mem_track();

        // ⛔ IT CLEARS: phase 0 no longer tracks the reservation and its free counter is the whole
        // space again (`:124-126`).
        assert_eq!(track.free_cap_at_eps(ExPhase(0)), LX_CAPACITY);
        assert_eq!(
            track.all_ds_and_size_at_eps(ExPhase(0)).map(|held| held.0.len()),
            Some(0),
        );
        // `epsToListIter[i] = it++` for `i` in `[0, exPhases)` (`:128-131`) — so phase 1 is now bound.
        assert_eq!(track.free_cap_at_eps(ExPhase(1)), LX_CAPACITY);
        // ⛔ AND IT NEVER CLEARS A STALE KEY: phase 7's key survives, naming an erased node.
        assert_eq!(track.all_ds_and_size_at_eps(ExPhase(7)), None);

        // ⛔ EACH PHASE GOT ITS OWN `MemoryOrganizer`, not a copy of one prototype's: committing to
        // phase 0 moves phase 0's free list and leaves phase 1's whole space free.
        record(&mut track, ExPhase(0), "Tensor0_lx", 0, 512);
        assert_eq!(
            track.find_common_fitting_memory(Capacity(512), &[ExPhase(0)], FitRequest::Anywhere),
            Some(Address(512)),
        );
        assert_eq!(
            track.find_common_fitting_memory(Capacity(512), &[ExPhase(1)], FitRequest::Anywhere),
            Some(Address::ZERO),
        );
    }

    /// 🎯 029/38 EXISTENCE IS THE INVERSION OF UNIT e018, SO AN UNBOUND PHASE READS AS "ALREADY
    /// EXISTS" AND AN EMPTY PHASE LIST READS AS "DOES NOT" — `mem_track.cpp:222-225`, `:214-217`.
    #[test]
    fn existence_inverts_unit_e018_so_an_unbound_phase_reads_as_already_existing() {
        let track = lx_with_reserved_front_end();
        let held = StorageName("reserved-frontend".to_owned());
        let absent = StorageName("Tensor0_lx".to_owned());

        assert!(track.check_if_ds_exists(&held, &[ExPhase(0)]));
        assert!(!track.check_if_ds_exists(&absent, &[ExPhase(0)]));
        // ⛔ Phase 1 was never bound, and the reference answers EXISTS for it — which is what makes
        // `checkDsForStartAddr` (unit e034) refuse a phase that was never created.
        assert!(track.check_if_ds_exists(&absent, &[ExPhase(1)]));
        // ⛔ An EMPTY `eps` answers `false`: `checkIfDsNotExists`' loop never runs, so `exists` stays
        // `true` there and this inverts it.
        assert!(!track.check_if_ds_exists(&absent, &[]));
    }

    /// 🎯 030/38 THE COMMON FIT INTERSECTS EVERY PHASE'S FREE LIST CLIPPING ONLY THE ENDS, THEN PICKS
    /// FORWARD, FROM THE BACK, OR AT A PINNED ADDRESS — `mem_track.cpp:283-346`.
    #[test]
    fn the_common_fit_intersects_every_phase_then_picks_forward_backward_or_pinned() {
        // Two phases over 4,096 bytes, fragmented DIFFERENTLY. ⛔ The 187-program corpus never
        // fragments a free list, so these addresses are computed from the reference's own algorithm
        // rather than measured from its output.
        let mut track = DsTrackInMem {
            mem_capacity: Capacity(4096),
            ..DsTrackInMem::default()
        };
        track.grow_ex_phases(ExPhaseCount(2));
        record(&mut track, ExPhase(0), "a0", 0, 512);
        record(&mut track, ExPhase(0), "a1", 1024, 512);
        record(&mut track, ExPhase(1), "b0", 0, 256);
        record(&mut track, ExPhase(1), "b1", 768, 512);
        // Phase 0 leaves [512,1023] and [1536,4095]; phase 1 leaves [256,767] and [1280,4095]; the
        // intersection is [512,767] and [1536,4095].
        let both = [ExPhase(0), ExPhase(1)];

        // `stAddr = iblk.first` on the first block with room (`:338-339`).
        assert_eq!(
            track.find_common_fitting_memory(Capacity(256), &both, FitRequest::Anywhere),
            Some(Address(512)),
        );
        // ⛔ 384 does not fit the 256-byte first block, so the forward walk skips it.
        assert_eq!(
            track.find_common_fitting_memory(Capacity(384), &both, FitRequest::Anywhere),
            Some(Address(1536)),
        );
        // `stAddr = iblk.second - cap + 1` (`:331`): 4095 − 384 + 1.
        assert_eq!(
            track.find_common_fitting_memory(Capacity(384), &both, FitRequest::AnywhereFromBack),
            Some(Address(3712)),
        );
        // A pinned address gets only what is left ABOVE it in its own block (`:321`): 767 − 600 + 1.
        assert_eq!(
            track.find_common_fitting_memory(Capacity(168), &both, FitRequest::At(Address(600))),
            Some(Address(600)),
        );
        assert_eq!(
            track.find_common_fitting_memory(Capacity(169), &both, FitRequest::At(Address(600))),
            None,
        );
        // ⛔ 900 is free in phase 0 but held in phase 1 (inside `b1`, [768,1279]), so no
        // intersected block holds it.
        assert_eq!(
            track.find_common_fitting_memory(Capacity(1), &both, FitRequest::At(Address(900))),
            None,
        );
        // ⛔ `DT_CHECK(eps.size() >= 1)` (`:272`) and the `epsToListIter.at` throw (`:277`) are both
        // the reference's own −1 here.
        assert_eq!(
            track.find_common_fitting_memory(Capacity(1), &[], FitRequest::Anywhere),
            None,
        );
        assert_eq!(
            track.find_common_fitting_memory(Capacity(1), &[ExPhase(9)], FitRequest::Anywhere),
            None,
        );

        // ⛔ AN ELEVENTH PHASE FLIPS A FORWARD REQUEST INTO A BACK-FIT — `eps.size() > 10 &&
        // opt_frag_` (`:327`). Phases 1..=10 hold nothing, so the intersection is phase 0's own
        // [512,1023] and [1536,4095], and 256 bytes land at 4095 − 256 + 1.
        let mut wide = DsTrackInMem {
            mem_capacity: Capacity(4096),
            ..DsTrackInMem::default()
        };
        wide.grow_ex_phases(ExPhaseCount(11));
        record(&mut wide, ExPhase(0), "a0", 0, 512);
        record(&mut wide, ExPhase(0), "a1", 1024, 512);
        let eleven: Vec<ExPhase> = (0..11).map(ExPhase).collect();
        assert_eq!(
            wide.find_common_fitting_memory(Capacity(256), &eleven, FitRequest::Anywhere),
            Some(Address(3840)),
        );
        // ⛔ AND `opt_frag_ == false` PUTS IT BACK: the same eleven phases answer forward again.
        wide.opt_frag = FragmentationOpt::Off;
        assert_eq!(
            wide.find_common_fitting_memory(Capacity(256), &eleven, FitRequest::Anywhere),
            Some(Address(512)),
        );
    }

    /// 🎯 030/38 ONE BLOCK OVERLAPPING SEVERAL CARRIES ALL OF THEM THROUGH, WITH ONLY THE FIRST'S
    /// START AND THE LAST'S END CLIPPED — `mem_track.cpp:303-312`.
    #[test]
    fn the_intersection_keeps_every_overlap_of_a_spanning_block_not_just_the_first() {
        // ⛔ THE ARM THE 187-PROGRAM CORPUS NEVER REACHES: it never fragments a free list, so
        // `myOvlBlks` there always holds ONE block and `j == 0` and `j == size - 1` are the same
        // element. These addresses are computed from the reference's own algorithm (`:289-313`).
        let mut track = DsTrackInMem {
            mem_capacity: Capacity(4096),
            ..DsTrackInMem::default()
        };
        track.grow_ex_phases(ExPhaseCount(2));
        // Phase 0 leaves ONE gap, [256,4095]; phase 1 leaves THREE, [0,511], [768,1535], [1792,4095].
        record(&mut track, ExPhase(0), "a0", 0, 256);
        record(&mut track, ExPhase(1), "b0", 512, 256);
        record(&mut track, ExPhase(1), "b1", 1536, 256);
        let both = [ExPhase(0), ExPhase(1)];
        // So the single spanning block overlaps all three, and the intersection is [256,511],
        // [768,1535], [1792,4095] — the middle one pushed UNCLIPPED (`:308` is false for it) and
        // already inside the spanning block, which is why the asymmetric clip is not an off-by-one.

        // The first block's own extent, 511 − 256 + 1, is exactly 256 (`:338-339`).
        assert_eq!(
            track.find_common_fitting_memory(Capacity(256), &both, FitRequest::Anywhere),
            Some(Address(256)),
        );
        // ⛔ ONE BYTE MORE MOVES IT TO THE MIDDLE OVERLAP, so a port that kept only `myOvlBlks[0]`
        // answers 1792 here and a port that dropped the unclipped middles answers 1792 too.
        assert_eq!(
            track.find_common_fitting_memory(Capacity(257), &both, FitRequest::Anywhere),
            Some(Address(768)),
        );
        // `iblk.second - cap + 1` over the LAST intersected block (`:331`): 4095 − 257 + 1.
        assert_eq!(
            track.find_common_fitting_memory(Capacity(257), &both, FitRequest::AnywhereFromBack),
            Some(Address(3839)),
        );
        // A pin inside the middle overlap gets only what is above it THERE (`:321`): 1535 − 1000 + 1.
        assert_eq!(
            track.find_common_fitting_memory(Capacity(536), &both, FitRequest::At(Address(1000))),
            Some(Address(1000)),
        );
        assert_eq!(
            track.find_common_fitting_memory(Capacity(537), &both, FitRequest::At(Address(1000))),
            None,
        );
        // ⛔ AND 600 IS IN NO INTERSECTED BLOCK — phase 1 holds it in `b0` — so the spanning block's
        // own [256,4095] is not what the pin arm reads.
        assert_eq!(
            track.find_common_fitting_memory(Capacity(1), &both, FitRequest::At(Address(600))),
            None,
        );
    }

    /// 🎯 031/38 THE COMMIT MOVES THE MAP, THE FREE COUNTER AND THE BLOCK LIST TOGETHER, AND ONLY THE
    /// LAST OF THE THREE IS CONDITIONAL ON `strict` — `mem_track.cpp:377-393`.
    #[test]
    fn the_commit_moves_the_map_the_free_counter_and_the_block_list_together() {
        let mut track = lx_with_reserved_front_end();
        let name = StorageName("Tensor0_lx".to_owned());
        let eps = [track.bound_phase(ExPhase(0)).expect("phase 0 is bound")];

        // The oracle's FIRST measured LX address, 1,625,344 (`sdsc_<N>.json`,
        // `startAddressCoreCorelet_.data_`), at the rounded stride 512 every program uses.
        track.add_ds_at_start_addr(&name, Capacity(512), &eps, Address(1_625_344));

        // `dsInMem_.insert({findOrCreateDsKey(ds), cap})` (`:383`).
        assert_eq!(track.cap_of_ds(ExPhase(0), &name), Some(Capacity(512)));
        // `free_ -= cap` (`:388`): 406,272 − 512.
        assert_eq!(track.free_cap_at_eps(ExPhase(0)), Capacity(405_760));
        // `occupied_.allocateMemory(cap, ds, startAddr)` (`:390`).
        assert_eq!(
            track.addr_of_ds(ExPhase(0), &name),
            DsAddress::At(Address(1_625_344)),
        );
        // ⛔ ALL THREE OR THE NEXT REQUEST HANDS OUT THE SAME ADDRESS TWICE: the next 512 bytes fit at
        // 1,625,856, which is the oracle's SECOND measured LX address.
        assert_eq!(
            track.find_common_fitting_memory(Capacity(512), &[ExPhase(0)], FitRequest::Anywhere),
            Some(Address(1_625_856)),
        );

        // ⛔ `if (strict)` (`:389`): a CapacityOnly tracker moves the map and the counter and NEVER the
        // block list, so the same ds reads back `Unplaced` and its whole space stays free.
        let mut lenient = lx_tracker(1);
        lenient.strict = TrackingMode::CapacityOnly;
        let lenient_eps = [lenient.bound_phase(ExPhase(0)).expect("phase 0 is bound")];
        lenient.add_ds_at_start_addr(&name, Capacity(512), &lenient_eps, Address::ZERO);
        assert_eq!(lenient.cap_of_ds(ExPhase(0), &name), Some(Capacity(512)));
        assert_eq!(
            lenient.free_cap_at_eps(ExPhase(0)),
            Capacity(LX_CAPACITY.0 - 512),
        );
        assert_eq!(lenient.addr_of_ds(ExPhase(0), &name), DsAddress::Unplaced);
    }

    /// `reserved-frontend` — the ds `reserveFrontendLx` pins at address 0 and every unit below asks
    /// about (`ProgramLayout.cpp:93`).
    fn reserved_name() -> StorageName {
        StorageName("reserved-frontend".to_owned())
    }

    /// 🎯 032/38 INIT TAKES THE CAPACITY, THE GRANULARITY, THE PHASE COUNT AND THE MODE AND THEN
    /// FORMATS, SO `[0, ts)` COMES OUT EMPTY OVER `memCap` BYTES EACH — `mem_track.cpp:107-112`.
    #[test]
    fn init_takes_the_capacity_and_granularity_and_then_formats_the_phases() {
        let mut track = DsTrackInMem::default();

        // The arguments `MemTrackBundle` passes every LX tracker: ("lxCore<c>", lxCapacity = 2031616,
        // numSteps, bytesPerStick = 128), strict by default (`mem_track_bundle.cpp:59`,
        // `sysdef.cpp:206`, `:211`).
        track.init_mem_track(
            MemName("lxCore0".to_owned()),
            LX_CAPACITY,
            ExPhaseCount(2),
            BYTES_PER_STICK,
            TrackingMode::AddressAssignment,
        );

        assert_eq!(track.name, MemName("lxCore0".to_owned()));
        assert_eq!(track.mem_capacity, LX_CAPACITY);
        assert_eq!(track.ex_phases, ExPhaseCount(2));
        assert_eq!(track.strict, TrackingMode::AddressAssignment);
        // `formatMemTrack()` (`:112`) bound exactly `[0, 2)`, each over the WHOLE space.
        assert_eq!(track.free_cap_at_eps(ExPhase(0)), LX_CAPACITY);
        assert_eq!(track.free_cap_at_eps(ExPhase(1)), LX_CAPACITY);
        assert_eq!(track.all_ds_and_size_at_eps(ExPhase(2)), None);
        // The granularity that entered is the one the frontend reservation is rounded by: 1,625,292 →
        // 1,625,344 (`ProgramLayout.cpp:81-98`).
        assert_eq!(
            track.alloc_granularity.round_up(Capacity(1_625_292)),
            RESERVED_FRONT_END,
        );

        // ⛔ IT CLEARS: a second init over a tracker that already holds the reservation loses it —
        // and this 5-argument form is the only thing SHOWING `allocGran` and `isStrict` land
        // (`:109`, `:111`). `dsm/dsmperf.cpp:2894` seeds an LX tracker with exactly this pair.
        let mut placed = lx_with_reserved_front_end();
        placed.init_mem_track(
            MemName("lx".to_owned()),
            LX_CAPACITY,
            ExPhaseCount(1),
            AllocGranularity::ONE,
            TrackingMode::CapacityOnly,
        );
        assert_eq!(placed.free_cap_at_eps(ExPhase(0)), LX_CAPACITY);
        assert_eq!(placed.strict, TrackingMode::CapacityOnly);
        assert_eq!(placed.alloc_granularity, AllocGranularity::ONE);
    }

    /// 🎯 033/38 THE ADDRESS QUERY GIVES THE BLOCK LIST'S OWN ADDRESS, AND BOTH ABSENCES ANSWER
    /// `DOESNT_FIT` (−2) WHERE UNIT e022 ANSWERS `INCOHERANT` (−3) — `mem_track.cpp:161-171`.
    #[test]
    fn the_address_of_a_ds_is_its_block_and_both_absences_are_doesnt_fit() {
        let mut track = lx_with_reserved_front_end();
        record(&mut track, ExPhase(0), "Tensor0_lx", 1_625_344, 512);
        let reserved = StorageName("reserved-frontend".to_owned());
        let tensor = StorageName("Tensor0_lx".to_owned());

        // Address 0 and 1,625,344 — the reservation and the oracle's first measured LX address.
        assert_eq!(
            track.get_address_of_ds(&reserved, ExPhase(0)),
            DsAddress::At(Address::ZERO),
        );
        assert_eq!(
            track.get_address_of_ds(&tensor, ExPhase(0)),
            DsAddress::At(Address(1_625_344)),
        );
        // ⛔ −2 for a name this phase does not hold...
        assert_eq!(
            track.get_address_of_ds(&StorageName("Tensor1_lx".to_owned()), ExPhase(0)),
            DsAddress::DoesntFit,
        );
        // ...and −2 AGAIN for a phase that was never bound, which unit e022 answers −3 to. Two
        // different sentinels for one state, from two functions.
        assert_eq!(track.get_address_of_ds(&tensor, ExPhase(1)), DsAddress::DoesntFit);
        assert_eq!(track.addr_of_ds(ExPhase(1), &tensor), DsAddress::Incoherant);

        // ⛔ THE THIRD ANSWER: a `CapacityOnly` tracker TRACKS a name no block holds, so the
        // organizer's own −1 travels out (`memory.cpp:161-168`) — not −2.
        let mut lenient = lx_tracker(1);
        lenient.strict = TrackingMode::CapacityOnly;
        record_capacity_only(&mut lenient, ExPhase(0), "Tensor0_lx", 512);
        assert_eq!(lenient.get_address_of_ds(&tensor, ExPhase(0)), DsAddress::Unplaced);
    }

    /// 🎯 034/38 THE THREE-WAY DECISION — `EXISTS` (−1), `DOESNT_FIT` (−2) FROM EITHER CHECK, ELSE THE
    /// ADDRESS — AND IT COMMITS NOTHING: `mem_track.cpp:353-374`.
    #[test]
    fn the_start_address_decision_separates_the_sentinels_and_commits_nothing() {
        let track = lx_with_reserved_front_end();
        let phases = [ExPhase(0)];
        let tensor = StorageName("Tensor0_lx".to_owned());
        let ask =
            |ds: &StorageName, cap: Capacity, eps: &[ExPhase], want: FitRequest, margin: Margin| {
                track.check_ds_for_start_addr(ds, cap, eps, want, margin)
            };

        // ⭐ THE ORACLE: capacity 2,031,616 with [0, 1625344) held by `reserved-frontend`, a 512-byte
        // request lands at EXACTLY 1,625,344 — the first LX address of all 187 reference programs.
        assert_eq!(
            ask(&tensor, Capacity(512), &phases, FitRequest::Anywhere, Margin::NONE),
            DsAddress::At(Address(1_625_344)),
        );
        // ⛔ AND IT COMMITTED NOTHING: asking twice answers the SAME address, never the next one.
        assert_eq!(
            ask(&tensor, Capacity(512), &phases, FitRequest::Anywhere, Margin::NONE),
            DsAddress::At(Address(1_625_344)),
        );
        // `EXISTS` for a name the phase already holds — the reservation itself.
        assert_eq!(
            ask(&reserved_name(), Capacity(512), &phases, FitRequest::Anywhere, Margin::NONE),
            DsAddress::Exists,
        );
        // ⛔ AND `EXISTS` FOR A PHASE THAT WAS NEVER BOUND (unit e018's inverted `else`) — the state
        // unit e033 answers `DOESNT_FIT` to.
        assert_eq!(
            ask(&tensor, Capacity(512), &[ExPhase(1)], FitRequest::Anywhere, Margin::NONE),
            DsAddress::Exists,
        );
        // `DOESNT_FIT` FROM THE FREE COUNTER: 406,272 is free and 406,273 rounds up to 406,400.
        assert_eq!(
            ask(&tensor, Capacity(406_273), &phases, FitRequest::Anywhere, Margin::NONE),
            DsAddress::DoesntFit,
        );
        // ⛔ AND FROM `margin`, WHICH IS DEMANDED BUT NEVER RESERVED: the exact fill still fits, and
        // one stick of headroom turns it into a refusal.
        assert_eq!(
            ask(&tensor, FREE_ABOVE_RESERVATION, &phases, FitRequest::Anywhere, Margin::NONE),
            DsAddress::At(Address(1_625_344)),
        );
        assert_eq!(
            ask(
                &tensor,
                FREE_ABOVE_RESERVATION,
                &phases,
                FitRequest::Anywhere,
                Margin(Capacity(128)),
            ),
            DsAddress::DoesntFit,
        );
        // ⛔ AND FROM THE FREE LIST, WHICH IS THE OTHER CHECK ENTIRELY: 406,272 bytes are free, so the
        // counter takes 512, but 1,000,000 sits inside the reservation and no free block holds it.
        assert_eq!(
            ask(
                &tensor,
                Capacity(512),
                &phases,
                FitRequest::At(Address(1_000_000)),
                Margin::NONE,
            ),
            DsAddress::DoesntFit,
        );

        // ⛔ TRAP: A `CapacityOnly` TRACKER ANSWERS A REAL ADDRESS 0 (`:371`) — never a refusal, and
        // indistinguishable from a strict placement at address 0.
        let mut lenient = lx_tracker(1);
        lenient.strict = TrackingMode::CapacityOnly;
        assert_eq!(
            lenient.check_ds_for_start_addr(
                &tensor,
                Capacity(512),
                &phases,
                FitRequest::Anywhere,
                Margin::NONE,
            ),
            DsAddress::At(Address::ZERO),
        );
    }

    /// 🎯 035/38 RESTORE CLEARS THE PHASE AND REPLAYS EVERY BACKUP ENTRY AT ITS RECORDED ADDRESS, SO
    /// A TRIAL ALLOCATION LEAVES NO TRACE — `mem_track.cpp:582-595`.
    #[test]
    fn restoring_a_phase_replays_the_backup_at_its_recorded_addresses_and_not_re_packed() {
        let mut track = lx_with_reserved_front_end();
        record(&mut track, ExPhase(0), "Tensor0_lx", 1_625_344, 512);
        let backup = track.backup_eps(ExPhase(0));
        let trial = StorageName("Tensor1_lx".to_owned());
        let eps = [track.bound_phase(ExPhase(0)).expect("phase 0 is bound")];

        // A 16,384-byte trial above the two survivors — one of the oracle's own rounded sizes.
        track.add_ds_at_start_addr(&trial, Capacity(16_384), &eps, Address(1_625_856));
        assert_eq!(track.free_cap_at_eps(ExPhase(0)), Capacity(389_376));

        track.restore_eps(ExPhase(0), &backup);

        // The trial is gone from the map, the free counter and the block list: 406,272 − 512.
        assert_eq!(track.cap_of_ds(ExPhase(0), &trial), None);
        assert_eq!(track.free_cap_at_eps(ExPhase(0)), Capacity(405_760));
        // ...and the survivors are back at their RECORDED addresses.
        assert_eq!(
            track.get_address_of_ds(&reserved_name(), ExPhase(0)),
            DsAddress::At(Address::ZERO),
        );
        assert_eq!(
            track.get_address_of_ds(&StorageName("Tensor0_lx".to_owned()), ExPhase(0)),
            DsAddress::At(Address(1_625_344)),
        );
        // ⛔ EXACT AND NOT A RE-PACK: 1,625,856 is free again, which is where a re-pack would have put
        // the second survivor.
        assert_eq!(
            track.find_common_fitting_memory(Capacity(512), &[ExPhase(0)], FitRequest::Anywhere),
            Some(Address(1_625_856)),
        );

        // ⛔ AN UNPLACED FIRST ENTRY STOPS THE REPLAY WHERE `DT_CHECK(startAddr >= 0)` (`:380`)
        // THROWS, AND THE CLEAR HAS ALREADY HAPPENED: the reference's tracker holds NOTHING at that
        // point, so the entry BEHIND the unplaced one does not come back either — a skip would have
        // brought it, and no address is invented for either.
        let stopped = [
            DsMemInfo {
                ds: trial.clone(),
                cap: Capacity(512),
                start: None,
            },
            DsMemInfo {
                ds: reserved_name(),
                cap: RESERVED_FRONT_END,
                start: Some(Address::ZERO),
            },
        ];
        track.restore_eps(ExPhase(0), &stopped);
        assert_eq!(track.get_address_of_ds(&trial, ExPhase(0)), DsAddress::DoesntFit);
        assert_eq!(track.get_address_of_ds(&reserved_name(), ExPhase(0)), DsAddress::DoesntFit);
        assert_eq!(track.free_cap_at_eps(ExPhase(0)), LX_CAPACITY);

        // ⛔ A PHASE THAT IS NOT BOUND IS A NO-OP — it does not clear phase 0 on the way past.
        track.restore_eps(ExPhase(0), &backup);
        track.restore_eps(ExPhase(1), &[]);
        assert_eq!(track.free_cap_at_eps(ExPhase(0)), Capacity(405_760));
    }

    /// 🎯 037/38 THE MEASURED ORACLE: WITH `[0, 1625344)` RESERVED, FORWARD FIRST FIT AT GRANULARITY
    /// 128 PLACES THE LX ALLOCATIONS AT 1625344 / 1625856 / 1626368, AND WHAT IS COMMITTED IS
    /// `capRound` — `mem_track.cpp:406-414`, and 588 LX nodes over 187 reference programs.
    #[test]
    fn check_and_add_ds_fits_forward_from_the_reservation_and_commits_the_rounded_size() {
        let mut track = lx_with_reserved_front_end();
        let phases = [ExPhase(0)];
        let place = |track: &mut DsTrackInMem, name: &str, cap: i64| {
            track.check_and_add_ds(
                &StorageName(name.to_owned()),
                Capacity(cap),
                &phases,
                Margin::NONE,
                AllocEnd::Front,
            )
        };

        // The corpus's own first three addresses, and `1625344 + k*512` is every address it records.
        assert_eq!(place(&mut track, "Tensor0_lx", 512), DsAddress::At(Address(1_625_344)));
        assert_eq!(place(&mut track, "Tensor1_lx", 512), DsAddress::At(Address(1_625_856)));
        assert_eq!(place(&mut track, "Tensor2_lx", 512), DsAddress::At(Address(1_626_368)));
        // ⛔ ALL THREE PARTS OF THE COMMIT MOVED, or the second address would have repeated the first:
        // the block list answers, and `free_` fell by 3 × 512 from 406,272.
        assert_eq!(
            track.get_address_of_ds(&StorageName("Tensor1_lx".to_owned()), ExPhase(0)),
            DsAddress::At(Address(1_625_856)),
        );
        assert_eq!(track.free_cap_at_eps(ExPhase(0)), Capacity(404_736));

        // ⛔ IT COMMITS `capRound` AND NOT `cap` (`:412`): 300 bytes is recorded as 384, and the NEXT
        // address steps over 384 and not over 300.
        assert_eq!(place(&mut track, "Tensor3_lx", 300), DsAddress::At(Address(1_626_880)));
        assert_eq!(
            track.cap_of_ds(ExPhase(0), &StorageName("Tensor3_lx".to_owned())),
            Some(Capacity(384)),
        );
        assert_eq!(place(&mut track, "Tensor4_lx", 512), DsAddress::At(Address(1_627_264)));

        // ⛔ THE BACK ARM IS PORTED AND THE CORPUS NEVER TAKES IT: the last fitting block ends at
        // 2,031,615, so a 512-byte request lands at `2031615 - 512 + 1` and nowhere near the front.
        let mut from_back = lx_with_reserved_front_end();
        assert_eq!(
            from_back.check_and_add_ds(
                &StorageName("Tensor0_lx".to_owned()),
                Capacity(512),
                &phases,
                Margin::NONE,
                AllocEnd::Back,
            ),
            DsAddress::At(Address(2_031_104)),
        );
    }

    /// 🎯 037/38 THE TWO REFUSALS ARE DIFFERENT ANSWERS AND NEITHER COMMITS — `EXISTS` (−1) for a name
    /// the phase already holds, `DOESNT_FIT` (−2) when the space cannot take it (`mem_track.cpp:410`).
    #[test]
    fn check_and_add_ds_answers_exists_for_a_name_already_there_and_doesnt_fit_above_the_space() {
        let mut track = lx_with_reserved_front_end();
        let phases = [ExPhase(0)];

        // ⛔ `EXISTS` AND NOT AN ADDRESS, and not `DOESNT_FIT` either: 406,272 bytes are free, so the
        // reservation would have fitted a second time had the name not been there.
        assert_eq!(
            track.check_and_add_ds(
                &reserved_name(),
                Capacity(512),
                &phases,
                Margin::NONE,
                AllocEnd::Front,
            ),
            DsAddress::Exists,
        );
        // ⛔ `DOESNT_FIT` FOR ONE BYTE TOO MANY — 406,273 rounds UP to 406,400, a whole stick more than
        // is free.
        assert_eq!(
            track.check_and_add_ds(
                &StorageName("Tensor0_lx".to_owned()),
                Capacity(406_273),
                &phases,
                Margin::NONE,
                AllocEnd::Front,
            ),
            DsAddress::DoesntFit,
        );
        // Neither refusal touched the map, the free counter or the block list.
        assert_eq!(track.cap_of_ds(ExPhase(0), &reserved_name()), Some(RESERVED_FRONT_END));
        assert_eq!(
            track.cap_of_ds(ExPhase(0), &StorageName("Tensor0_lx".to_owned())),
            None,
        );
        assert_eq!(track.free_cap_at_eps(ExPhase(0)), FREE_ABOVE_RESERVATION);
    }

    /// 🎯 038/38 THE ORACLE'S SEED: `checkAndAddDsAtAddr("reserved-frontend", 1625292, {0}, 0)` PINS
    /// `[0, 1625344)` — the request ROUNDED UP — which is what puts the first real LX allocation at
    /// exactly 1,625,344 (`mem_track.cpp:428-436`, `dbo/src/Transforms/ProgramLayout.cpp:93`).
    #[test]
    fn check_and_add_ds_at_addr_pins_the_front_end_reservation_rounded_up_to_a_whole_stick() {
        let mut track = lx_tracker(1);
        let phases = [ExPhase(0)];

        // ⛔ ADDRESS 0 IS AN ADDRESS AND NOT A REFUSAL.
        assert_eq!(
            track.check_and_add_ds_at_addr(
                &reserved_name(),
                Capacity(1_625_292),
                &phases,
                Some(Address::ZERO),
                Margin::NONE,
            ),
            DsAddress::At(Address::ZERO),
        );
        // ⛔ AND THE BLOCK IS 1,625,344 BYTES, NOT THE 1,625,292 ASKED FOR: 52 bytes fewer would leave
        // every address in the corpus one stick low.
        assert_eq!(track.cap_of_ds(ExPhase(0), &reserved_name()), Some(RESERVED_FRONT_END));
        assert_eq!(track.free_cap_at_eps(ExPhase(0)), FREE_ABOVE_RESERVATION);
        assert_eq!(
            track.check_and_add_ds(
                &StorageName("Tensor0_lx".to_owned()),
                Capacity(512),
                &phases,
                Margin::NONE,
                AllocEnd::Front,
            ),
            DsAddress::At(Address(1_625_344)),
        );

        // ⛔ A PIN NO SINGLE FREE BLOCK HOLDS IS `DOESNT_FIT` (`mem_track.cpp:317-325`) — 1,625,344 is
        // taken now — and it commits nothing: 406,272 − 512 either way.
        assert_eq!(
            track.check_and_add_ds_at_addr(
                &StorageName("Tensor1_lx".to_owned()),
                Capacity(512),
                &phases,
                Some(Address(1_625_344)),
                Margin::NONE,
            ),
            DsAddress::DoesntFit,
        );
        assert_eq!(track.free_cap_at_eps(ExPhase(0)), Capacity(405_760));
    }

    /// 🎯 038/38 A NEGATIVE `addrNeeded` IS NOT A PIN — `if (addrNeeded >= 0)` (`mem_track.cpp:317`)
    /// sends it to the anywhere arm, so it answers the FIRST FITTING BLOCK's address and not a refusal.
    ///
    /// ⛔ REGRESSION: this port read [`Option::is_some`] as the reference's sign test and answered
    /// `DOESNT_FIT` (−2) to `Some(Address(-1))`, because no free block contains −1. The reference
    /// returns 1,625,344 — the same address [`FitRequest::Anywhere`] gives, and the same one
    /// `findCommonFittingMemory`'s own default `addrNeeded = -1` (`mem_track.h:73`) has always meant —
    /// a default `checkAndAddDsAtAddr` itself does NOT have (`mem_track.h:83-85`).
    #[test]
    fn check_and_add_ds_at_addr_treats_a_negative_address_as_anywhere_and_not_as_a_pin() {
        let mut track = lx_with_reserved_front_end();
        let phases = [ExPhase(0)];

        // ⛔ THE NUMBER, NOT "IT PLACED IT": forward first fit above the reservation, exactly as
        // `check_and_add_ds` reaches it.
        assert_eq!(
            track.check_and_add_ds_at_addr(
                &StorageName("Tensor0_lx".to_owned()),
                Capacity(512),
                &phases,
                Some(Address(-1)),
                Margin::NONE,
            ),
            DsAddress::At(Address(1_625_344)),
        );
        // ...and it COMMITTED there, so the next request steps over it rather than repeating it.
        assert_eq!(
            track.check_and_add_ds_at_addr(
                &StorageName("Tensor1_lx".to_owned()),
                Capacity(512),
                &phases,
                None,
                Margin::NONE,
            ),
            DsAddress::At(Address(1_625_856)),
        );
        assert_eq!(track.free_cap_at_eps(ExPhase(0)), Capacity(405_248));
    }
}
