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

//! `MemTrackBundle` — one tracker per (component, core, corelet, row), and the place every CAPACITY and GRANULARITY in the campaign is set. ⭐ `getTracker(comp, core, corelet, row)` IS the `L3TrackerSite` of the Rust seam, and `initializeMemoryTrackers` is what makes LX capacity 2,031,616 with granularity 128.
//!
//! `sys-arch-spec/memtracker/mem_track_bundle.cpp` — 4 of the campaign's 38 units (dependency level(s) [0, 1, 4]).
//!
//! | unit | entry | level | lines | class | authority path:line |
//! |---|---|---|---|---|---|
//! | `e001_growExPhases` | 001 | 0 | 21 | `MemTrackBundle` | `sys-arch-spec/memtracker/mem_track_bundle.cpp:16` |
//! | `e002_getTracker` | 002 | 0 | 28 | `MemTrackBundle` | `sys-arch-spec/memtracker/mem_track_bundle.cpp:174` |
//! | `e017_insertPsBeforeAndRename` | 017 | 1 | 84 | `MemTrackBundle` | `sys-arch-spec/memtracker/mem_track_bundle.cpp:204` |
//! | `e036_initializeMemoryTrackers` | 036 | 4 | 134 | `MemTrackBundle` | `sys-arch-spec/memtracker/mem_track_bundle.cpp:38` |


use std::collections::BTreeMap;

use sys_arch_spec::arch_enums::{LdsSegment, SenComponent};

use crate::units::{Core, Corelet, Row};

// HOW MANY EXECUTION PHASES A TRACKER HOLDS — `exPhases` (`util/memtracker/mem_track.h:39`), a COUNT
// and not one phase's index.
// ⛔ ONE REFERENCE FIELD, ONE TYPE: this IS `DsTrackInMem`'s own count, not a second one beside it.
// ⚠️ `MemTrackBundle::growExPhases` DOES NOT READ IT: it only forwards `newExPhases` to every tracker
// (`mem_track_bundle.cpp:16-36`), and the reads that decide anything are `DsTrackInMem`'s own
// (`mem_track.cpp:116`, `:120`). In THIS file it is read by `insertPsBeforeAndRename`, as the bound of
// its downward rename loop (`mem_track_bundle.cpp:207-208`), and by `initializeMemoryTrackers`, as the
// count it seeds every on-core tracker from (`:51-53`). It is SIGNED there because
// `DsTrackInMem::removeEps` can drive it below zero — an unsigned count would panic where the
// reference simply goes negative.
// ⛔ DISTINCT FROM `dl_ops::ExPhase`, which is an INDEX into this count; the reference spells both
// `int`, so transposing them here is E0308.
use super::tracker::{
    AllocGranularity, DsTrackInMem, EpsEdit, ExPhaseCount, MemName, TrackingMode,
};
use crate::schedule::l3::dl_ops::ExPhase;

// AN EXTENT IN BYTES — `memCapacity` (`util/memtracker/mem_track.h:37`).
// ⛔ ONE TYPE, TWO UNITS IN THIS FILE: bytes for the LX and L0 trackers, REGISTER BITS for the seven
// register-file families, because `initializeMemoryTrackers` passes `reg.maxNum * reg.bitSize` into
// the same parameter it passes `lxCapacity` into (`mem_track_bundle.cpp:80` against `:60`). See
// [`RegFileSize`].
use super::memory::Capacity;

/// WHAT THE BUNDLE ASKS OF ONE TRACKER — `DsTrackInMem::growExPhases`
/// (`util/memtracker/mem_track.cpp:115`), which is entry 027, two dependency levels above this file
/// and scheduled after it.
///
/// ⛔ THE BUNDLE IS GENERIC OVER ITS TRACKER FOR EXACTLY THAT REASON, not for reuse: entry 001's whole
/// effect IS this delegation, and a `todo!` in its place would drop the effect rather than name a gap.
/// `MemTrackBundle<DsTrackInMem>` is the one instance, once entry 027 lands.
pub(crate) trait GrowExPhases {
    /// `growExPhases(newExPhases)`.
    fn grow_ex_phases(&mut self, new_ex_phases: ExPhaseCount);
}

/// A TRACKER FAMILY KEYED BY `(core, corelet)` — the seven `getTracker` arms that read
/// `.at(core).at(corelet)` and IGNORE `row`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum PerCoreletTrack {
    /// `pelrfTrack` — `PELRF`.
    PeLrf,
    /// `sfplrfTrack` — `SFPLRF`.
    SfpLrf,
    /// `l0Track` — `L0`.
    L0,
    /// `l0ScaleTrack` — `L0_SCALE`.
    L0Scale,
    /// `sfpStateTrack` — `SFPSTATE`.
    SfpState,
    /// `peStateTrack` — `PESTATE`.
    PeState,
    /// `lxluScaleTrack` — `LXLUSCALEREG`.
    LxluScaleReg,
}

/// A TRACKER FAMILY KEYED BY `(core, corelet, row)` — the only three components whose `getTracker` arm
/// reads all four arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum PerRowTrack {
    /// `xrfTrack` — `PTXRF`.
    Ptxrf,
    /// `ptArfTrack` — `PTARF`.
    Ptarf,
    /// `ptIrfTrack` — `PTIRF`.
    Ptirf,
}

/// WHICH TRACKER OF THE BUNDLE — `getTracker`'s four arguments reduced to the coordinates each arm
/// actually reads, and the [`crate::schedule::l3::dl_ops::L3TrackerSite`] the seam keys by.
///
/// ⛔ THE ARITY IS A LIE FOR MOST COMPONENTS, and this is where it stops being one: `LX` reads only
/// `core`, seven families read `(core, corelet)`, three read `(core, corelet, row)`. Passing a corelet
/// to an LX site is now unspellable rather than ignored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum BundleSite {
    /// `lxTrackPerCore.at(core)`.
    Lx(Core),
    /// One of the seven per-corelet families.
    PerCorelet(PerCoreletTrack, Core, Corelet),
    /// One of the three per-row families.
    PerRow(PerRowTrack, Core, Corelet, Row),
}

impl BundleSite {
    /// Replaces: e002_getTracker
    ///
    /// WHICH SITE `(comp, core, corelet, row)` NAMES — the classification half of `getTracker`'s
    /// eleven-arm `if` chain (`sys-arch-spec/memtracker/mem_track_bundle.cpp:174`).
    ///
    /// ⛔ [`None`] IS THE REFERENCE'S `DT_ERROR("No tracker for component..")`, a hard stop and not a
    /// tracker that happens to be absent: the bundle holds no map for any other component.
    pub(crate) fn of(
        memory: SenComponent,
        core: Core,
        corelet: Corelet,
        row: Row,
    ) -> Option<BundleSite> {
        Some(match memory {
            SenComponent::Pelrf => Self::PerCorelet(PerCoreletTrack::PeLrf, core, corelet),
            SenComponent::Sfplrf => Self::PerCorelet(PerCoreletTrack::SfpLrf, core, corelet),
            SenComponent::L0 => Self::PerCorelet(PerCoreletTrack::L0, core, corelet),
            SenComponent::L0Scale => Self::PerCorelet(PerCoreletTrack::L0Scale, core, corelet),
            SenComponent::Ptxrf => Self::PerRow(PerRowTrack::Ptxrf, core, corelet, row),
            SenComponent::Ptarf => Self::PerRow(PerRowTrack::Ptarf, core, corelet, row),
            SenComponent::Lx => Self::Lx(core),
            SenComponent::Ptirf => Self::PerRow(PerRowTrack::Ptirf, core, corelet, row),
            SenComponent::Sfpstate => Self::PerCorelet(PerCoreletTrack::SfpState, core, corelet),
            SenComponent::Pestate => Self::PerCorelet(PerCoreletTrack::PeState, core, corelet),
            SenComponent::Lxluscalereg => {
                Self::PerCorelet(PerCoreletTrack::LxluScaleReg, core, corelet)
            }
            _ => return None,
        })
    }
}

/// EVERY MEMORY TRACKER OF ONE PROGRAM — `MemTrackBundle`
/// (`sys-arch-spec/memtracker/mem_track_bundle.h:17-43`), one tracker per (component, core, corelet,
/// row).
///
/// ⛔ THIRTEEN SPARSE MAPS, NOT DENSE ARRAYS: `initializeMemoryTrackers` reads
/// `lxTrackPerCore.empty()` and `hbmTrack.begin()->second.exPhases` (`mem_track_bundle.cpp:50-53`), so
/// both PRESENCE and ITERATION ORDER are load-bearing and a fully populated array states neither.
///
/// ⛔ GENERIC OVER THE TRACKER because `DsTrackInMem` is entry 027's class, at dependency level 2.
#[derive(Debug, Clone)]
pub(crate) struct MemTrackBundle<T> {
    /// `xrfTrack` — `(core, corelet, row)`.
    pub(crate) xrf_track: BTreeMap<Core, BTreeMap<Corelet, BTreeMap<Row, T>>>,
    /// `lxTrackStPinPerCore` — ⛔ NO `getTracker` ARM REACHES IT, and entry 036 never creates it.
    pub(crate) lx_track_st_pin_per_core: BTreeMap<Core, T>,
    /// `lxTrackPerCore` — the LX, and the only tracker the L3 scheduler places into.
    pub(crate) lx_track_per_core: BTreeMap<Core, T>,
    /// `hbmTrack` — one per virtual segment, created by `ProgramLayout.cpp:66-72`, not by entry 036.
    pub(crate) hbm_track: BTreeMap<LdsSegment, T>,
    /// `sfplrfTrack`.
    pub(crate) sfplrf_track: BTreeMap<Core, BTreeMap<Corelet, T>>,
    /// `pelrfTrack`.
    pub(crate) pelrf_track: BTreeMap<Core, BTreeMap<Corelet, T>>,
    /// `sfpStateTrack`.
    pub(crate) sfp_state_track: BTreeMap<Core, BTreeMap<Corelet, T>>,
    /// `peStateTrack`.
    pub(crate) pe_state_track: BTreeMap<Core, BTreeMap<Corelet, T>>,
    /// `l0Track`.
    pub(crate) l0_track: BTreeMap<Core, BTreeMap<Corelet, T>>,
    /// `l0ScaleTrack`.
    pub(crate) l0_scale_track: BTreeMap<Core, BTreeMap<Corelet, T>>,
    /// `lxluScaleTrack`.
    pub(crate) lxlu_scale_track: BTreeMap<Core, BTreeMap<Corelet, T>>,
    /// `ptArfTrack` — `(core, corelet, row)`.
    pub(crate) pt_arf_track: BTreeMap<Core, BTreeMap<Corelet, BTreeMap<Row, T>>>,
    /// `ptIrfTrack` — `(core, corelet, row)`.
    pub(crate) pt_irf_track: BTreeMap<Core, BTreeMap<Corelet, BTreeMap<Row, T>>>,
}

impl<T> Default for MemTrackBundle<T> {
    fn default() -> Self {
        Self {
            xrf_track: BTreeMap::new(),
            lx_track_st_pin_per_core: BTreeMap::new(),
            lx_track_per_core: BTreeMap::new(),
            hbm_track: BTreeMap::new(),
            sfplrf_track: BTreeMap::new(),
            pelrf_track: BTreeMap::new(),
            sfp_state_track: BTreeMap::new(),
            pe_state_track: BTreeMap::new(),
            l0_track: BTreeMap::new(),
            l0_scale_track: BTreeMap::new(),
            lxlu_scale_track: BTreeMap::new(),
            pt_arf_track: BTreeMap::new(),
            pt_irf_track: BTreeMap::new(),
        }
    }
}

/// `std::map::at` — the reference THROWS on a missing key, and nothing on this path catches it.
fn at<'m, K: Ord, V>(map: &'m BTreeMap<K, V>, key: &K) -> &'m V {
    map.get(key)
        .expect("the bundle holds a tracker at this site, as `std::map::at` requires")
}

/// `std::map::at`, for the mutable pointer `getTracker` hands back.
fn at_mut<'m, K: Ord, V>(map: &'m mut BTreeMap<K, V>, key: &K) -> &'m mut V {
    map.get_mut(key)
        .expect("the bundle holds a tracker at this site, as `std::map::at` requires")
}

impl<T> MemTrackBundle<T> {
    /// Replaces: e002_getTracker
    ///
    /// THE TRACKER AT ONE SITE, SHARED — the lookup half of `getTracker`
    /// (`sys-arch-spec/memtracker/mem_track_bundle.cpp:174`), reached through [`BundleSite::of`].
    ///
    /// ⭐ THE SHARED FORM EXISTS BECAUSE THE SEAM NEEDS IT: `ExPhaseTrackers::capacity` takes `&self`
    /// (`src/schedule/l3/dl_ops.rs:8975`), so a `DsTrackInMem*`-shaped accessor alone cannot carry it.
    pub(crate) fn tracker(&self, at_site: BundleSite) -> &T {
        match at_site {
            BundleSite::Lx(core) => at(&self.lx_track_per_core, &core),
            BundleSite::PerCorelet(family, core, corelet) => {
                at(at(self.per_corelet(family), &core), &corelet)
            }
            BundleSite::PerRow(family, core, corelet, row) => {
                at(at(at(self.per_row(family), &core), &corelet), &row)
            }
        }
    }

    /// Replaces: e002_getTracker
    ///
    /// THE TRACKER AT ONE SITE, EXCLUSIVE — the `DsTrackInMem*` the reference returns, which every
    /// caller writes through (`L3DlOpsScheduler.cpp:5538`).
    pub(crate) fn tracker_mut(&mut self, at_site: BundleSite) -> &mut T {
        match at_site {
            BundleSite::Lx(core) => at_mut(&mut self.lx_track_per_core, &core),
            BundleSite::PerCorelet(family, core, corelet) => {
                at_mut(at_mut(self.per_corelet_mut(family), &core), &corelet)
            }
            BundleSite::PerRow(family, core, corelet, row) => at_mut(
                at_mut(at_mut(self.per_row_mut(family), &core), &corelet),
                &row,
            ),
        }
    }

    /// Which of the seven per-corelet maps a family names.
    fn per_corelet(&self, family: PerCoreletTrack) -> &BTreeMap<Core, BTreeMap<Corelet, T>> {
        match family {
            PerCoreletTrack::PeLrf => &self.pelrf_track,
            PerCoreletTrack::SfpLrf => &self.sfplrf_track,
            PerCoreletTrack::L0 => &self.l0_track,
            PerCoreletTrack::L0Scale => &self.l0_scale_track,
            PerCoreletTrack::SfpState => &self.sfp_state_track,
            PerCoreletTrack::PeState => &self.pe_state_track,
            PerCoreletTrack::LxluScaleReg => &self.lxlu_scale_track,
        }
    }

    /// [`Self::per_corelet`], exclusively.
    fn per_corelet_mut(
        &mut self,
        family: PerCoreletTrack,
    ) -> &mut BTreeMap<Core, BTreeMap<Corelet, T>> {
        match family {
            PerCoreletTrack::PeLrf => &mut self.pelrf_track,
            PerCoreletTrack::SfpLrf => &mut self.sfplrf_track,
            PerCoreletTrack::L0 => &mut self.l0_track,
            PerCoreletTrack::L0Scale => &mut self.l0_scale_track,
            PerCoreletTrack::SfpState => &mut self.sfp_state_track,
            PerCoreletTrack::PeState => &mut self.pe_state_track,
            PerCoreletTrack::LxluScaleReg => &mut self.lxlu_scale_track,
        }
    }

    /// Which of the three per-row maps a family names.
    fn per_row(&self, family: PerRowTrack) -> &BTreeMap<Core, BTreeMap<Corelet, BTreeMap<Row, T>>> {
        match family {
            PerRowTrack::Ptxrf => &self.xrf_track,
            PerRowTrack::Ptarf => &self.pt_arf_track,
            PerRowTrack::Ptirf => &self.pt_irf_track,
        }
    }

    /// [`Self::per_row`], exclusively.
    fn per_row_mut(
        &mut self,
        family: PerRowTrack,
    ) -> &mut BTreeMap<Core, BTreeMap<Corelet, BTreeMap<Row, T>>> {
        match family {
            PerRowTrack::Ptxrf => &mut self.xrf_track,
            PerRowTrack::Ptarf => &mut self.pt_arf_track,
            PerRowTrack::Ptirf => &mut self.pt_irf_track,
        }
    }

    /// EVERY TRACKER OF THE BUNDLE, IN THE REFERENCE'S OWN GROUP ORDER — `growExPhases`' three plain
    /// loops, its seven `std::ref` per-corelet families and its three per-row ones, in that order
    /// (`sys-arch-spec/memtracker/mem_track_bundle.cpp:20-35`).
    ///
    /// ⛔ ALL THIRTEEN MAPS, INCLUDING THE TWO NO `getTracker` ARM REACHES.
    fn all_trackers_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.hbm_track
            .values_mut()
            .chain(self.lx_track_per_core.values_mut())
            .chain(self.lx_track_st_pin_per_core.values_mut())
            .chain(
                self.sfplrf_track
                    .values_mut()
                    .flat_map(BTreeMap::values_mut),
            )
            .chain(self.pelrf_track.values_mut().flat_map(BTreeMap::values_mut))
            .chain(
                self.sfp_state_track
                    .values_mut()
                    .flat_map(BTreeMap::values_mut),
            )
            .chain(
                self.pe_state_track
                    .values_mut()
                    .flat_map(BTreeMap::values_mut),
            )
            .chain(self.l0_track.values_mut().flat_map(BTreeMap::values_mut))
            .chain(
                self.l0_scale_track
                    .values_mut()
                    .flat_map(BTreeMap::values_mut),
            )
            .chain(
                self.lxlu_scale_track
                    .values_mut()
                    .flat_map(BTreeMap::values_mut),
            )
            .chain(
                self.xrf_track
                    .values_mut()
                    .flat_map(|per_corelet| per_corelet.values_mut())
                    .flat_map(BTreeMap::values_mut),
            )
            .chain(
                self.pt_arf_track
                    .values_mut()
                    .flat_map(|per_corelet| per_corelet.values_mut())
                    .flat_map(BTreeMap::values_mut),
            )
            .chain(
                self.pt_irf_track
                    .values_mut()
                    .flat_map(|per_corelet| per_corelet.values_mut())
                    .flat_map(BTreeMap::values_mut),
            )
    }
}

impl<T: GrowExPhases> MemTrackBundle<T> {
    /// Replaces: e001_growExPhases
    ///
    /// GROWS EVERY TRACKER IN THE BUNDLE TO `new_ex_phases`
    /// (`sys-arch-spec/memtracker/mem_track_bundle.cpp:16`).
    ///
    /// ⛔ ALL OF THEM TOGETHER, WHICH IS THE WHOLE POINT: the on-core trackers take their phase count
    /// from the HBM ones at creation (`mem_track_bundle.cpp:51-53`), so growing a subset leaves the
    /// rest silently on the old count. ⛔ Distinct unit from `DsTrackInMem::growExPhases`, same C++ name.
    pub(crate) fn grow_ex_phases(&mut self, new_ex_phases: ExPhaseCount) {
        for tracker in self.all_trackers_mut() {
            tracker.grow_ex_phases(new_ex_phases);
        }
    }
}

#[cfg(test)]
mod tests_e001 {
    use super::*;

    /// A tracker that only records what it was grown to.
    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    struct Grown(Vec<ExPhaseCount>);

    impl GrowExPhases for Grown {
        fn grow_ex_phases(&mut self, new_ex_phases: ExPhaseCount) {
            self.0.push(new_ex_phases);
        }
    }

    fn core() -> Core {
        Core::checked(0).expect("core 0")
    }

    /// EVERY ONE OF THE THIRTEEN MAPS IS GROWN, including `hbmTrack` and `lxTrackStPinPerCore`, which
    /// `getTracker` cannot reach — `mem_track_bundle.cpp:20-35` walks all thirteen.
    #[test]
    fn grows_every_tracker_in_the_bundle() {
        let corelet = Corelet::at::<0>();
        let row = Row::at::<0>();
        let mut bundle = MemTrackBundle::<Grown>::default();

        bundle.hbm_track.insert(LdsSegment::Const, Grown::default());
        bundle.lx_track_per_core.insert(core(), Grown::default());
        bundle
            .lx_track_st_pin_per_core
            .insert(core(), Grown::default());
        for family in [
            PerCoreletTrack::PeLrf,
            PerCoreletTrack::SfpLrf,
            PerCoreletTrack::L0,
            PerCoreletTrack::L0Scale,
            PerCoreletTrack::SfpState,
            PerCoreletTrack::PeState,
            PerCoreletTrack::LxluScaleReg,
        ] {
            bundle
                .per_corelet_mut(family)
                .entry(core())
                .or_default()
                .insert(corelet, Grown::default());
        }
        for family in [PerRowTrack::Ptxrf, PerRowTrack::Ptarf, PerRowTrack::Ptirf] {
            bundle
                .per_row_mut(family)
                .entry(core())
                .or_default()
                .entry(corelet)
                .or_default()
                .insert(row, Grown::default());
        }

        bundle.grow_ex_phases(ExPhaseCount(4));

        // Read back from the FIELDS, not from `all_trackers_mut`: a check driven by the same walk
        // under test would pass on a family the walk skips.
        let grown = vec![ExPhaseCount(4)];
        assert_eq!(bundle.hbm_track[&LdsSegment::Const].0, grown);
        assert_eq!(bundle.lx_track_per_core[&core()].0, grown);
        assert_eq!(bundle.lx_track_st_pin_per_core[&core()].0, grown);
        assert_eq!(bundle.pelrf_track[&core()][&corelet].0, grown);
        assert_eq!(bundle.sfplrf_track[&core()][&corelet].0, grown);
        assert_eq!(bundle.l0_track[&core()][&corelet].0, grown);
        assert_eq!(bundle.l0_scale_track[&core()][&corelet].0, grown);
        assert_eq!(bundle.sfp_state_track[&core()][&corelet].0, grown);
        assert_eq!(bundle.pe_state_track[&core()][&corelet].0, grown);
        assert_eq!(bundle.lxlu_scale_track[&core()][&corelet].0, grown);
        assert_eq!(bundle.xrf_track[&core()][&corelet][&row].0, grown);
        assert_eq!(bundle.pt_arf_track[&core()][&corelet][&row].0, grown);
        assert_eq!(bundle.pt_irf_track[&core()][&corelet][&row].0, grown);
    }
}

#[cfg(test)]
mod tests_e002 {
    use super::*;

    /// A tracker that is only its own identity, so the wrong map is a wrong number.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    struct Marked(u32);

    fn core() -> Core {
        Core::checked(0).expect("core 0")
    }

    /// EACH OF THE ELEVEN ARMS SELECTS THE MAP `mem_track_bundle.cpp:174-201` SELECTS — and `LX` reads
    /// `lxTrackPerCore`, NOT `lxTrackStPinPerCore`, which no arm reaches.
    #[test]
    fn each_component_selects_the_reference_s_map() {
        let corelet = Corelet::at::<0>();
        let row = Row::at::<0>();
        let mut bundle = MemTrackBundle::<Marked>::default();

        bundle.lx_track_per_core.insert(core(), Marked(1));
        bundle.lx_track_st_pin_per_core.insert(core(), Marked(99));
        for (mark, family) in [
            PerCoreletTrack::PeLrf,
            PerCoreletTrack::SfpLrf,
            PerCoreletTrack::L0,
            PerCoreletTrack::L0Scale,
            PerCoreletTrack::SfpState,
            PerCoreletTrack::PeState,
            PerCoreletTrack::LxluScaleReg,
        ]
        .into_iter()
        .enumerate()
        {
            bundle
                .per_corelet_mut(family)
                .entry(core())
                .or_default()
                .insert(corelet, Marked(10 + mark as u32));
        }
        for (mark, family) in [PerRowTrack::Ptxrf, PerRowTrack::Ptarf, PerRowTrack::Ptirf]
            .into_iter()
            .enumerate()
        {
            bundle
                .per_row_mut(family)
                .entry(core())
                .or_default()
                .entry(corelet)
                .or_default()
                .insert(row, Marked(20 + mark as u32));
        }

        let found: Vec<Marked> = [
            SenComponent::Lx,
            SenComponent::Pelrf,
            SenComponent::Sfplrf,
            SenComponent::L0,
            SenComponent::L0Scale,
            SenComponent::Sfpstate,
            SenComponent::Pestate,
            SenComponent::Lxluscalereg,
            SenComponent::Ptxrf,
            SenComponent::Ptarf,
            SenComponent::Ptirf,
        ]
        .into_iter()
        .map(|memory| {
            let site = BundleSite::of(memory, core(), corelet, row).expect("a tracked component");
            *bundle.tracker(site)
        })
        .collect();

        assert_eq!(
            found,
            vec![
                Marked(1),
                Marked(10),
                Marked(11),
                Marked(12),
                Marked(13),
                Marked(14),
                Marked(15),
                Marked(16),
                Marked(20),
                Marked(21),
                Marked(22),
            ],
        );
        assert_eq!(
            bundle.tracker_mut(BundleSite::of(SenComponent::Lx, core(), corelet, row).expect("LX")),
            &mut Marked(1),
        );
    }

    /// THE `else` ARM IS THE HARD ERROR `DT_ERROR("No tracker for component..")`
    /// (`mem_track_bundle.cpp:199-200`) — the bundle has no map for `HBM`, `LXLU` or `NO_COMPONENT`.
    #[test]
    fn an_untracked_component_has_no_site() {
        let corelet = Corelet::at::<0>();
        let row = Row::at::<0>();
        for memory in [
            SenComponent::Hbm,
            SenComponent::Lxlu,
            SenComponent::NoComponent,
        ] {
            assert_eq!(BundleSite::of(memory, core(), corelet, row), None);
        }
    }
}

/// WHETHER THE PHASE WENT IN, AND WHICH `DT_CHECK` STOPPED IT IF IT DID NOT — the three checks the
/// reference's `insertWithCopyAndShift` lambda makes (`mem_track_bundle.cpp:207`, `:210`, `:213`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PhaseInsertion {
    /// Every walked tracker renamed downward and then took the copy.
    Inserted,
    /// `DT_CHECK(orig_ps < tracker.exPhases)` — this tracker does not hold the phase asked for.
    PhaseOutOfRange,
    /// `DT_CHECK(success == 0)` after a `renameEps`.
    RenameRefused(EpsEdit),
    /// `DT_CHECK(success == 0)` after the `addEpsBefore`.
    AddRefused(EpsEdit),
}

/// WHAT `insertPsBeforeAndRename` ASKS OF ONE TRACKER — the three `DsTrackInMem` members its lambda
/// touches (`mem_track_bundle.cpp:206-214`), a trait for the same reason [`GrowExPhases`] is one:
/// `DsTrackInMem` is entry 027's class, two dependency levels above this file.
pub(crate) trait ShiftExPhases {
    /// `tracker.exPhases`.
    fn ex_phase_count(&self) -> ExPhaseCount;

    /// `tracker.renameEps(eps, eps + 1)` — entry 007.
    fn rename_eps(&mut self, old_eps: ExPhase, new_eps: ExPhase) -> EpsEdit;

    /// `tracker.addEpsBefore(orig_ps, orig_ps + 1, true)` — entry 005.
    fn add_eps_before(&mut self, new_eps: ExPhase, cand: ExPhase) -> EpsEdit;
}

impl ShiftExPhases for DsTrackInMem {
    fn ex_phase_count(&self) -> ExPhaseCount {
        self.ex_phases
    }

    // Both bodies reach the INHERENT method of the same name — method resolution finds an inherent
    // method before a trait one, so these forward to units e007 and e005 and do not recurse.
    fn rename_eps(&mut self, old_eps: ExPhase, new_eps: ExPhase) -> EpsEdit {
        DsTrackInMem::rename_eps(self, old_eps, new_eps)
    }

    fn add_eps_before(&mut self, new_eps: ExPhase, cand: ExPhase) -> EpsEdit {
        DsTrackInMem::add_eps_before(self, new_eps, cand)
    }
}

impl<T: ShiftExPhases> MemTrackBundle<T> {
    /// THE TWELVE MAPS `insertPsBeforeAndRename` WALKS, IN ITS OWN ORDER
    /// (`sys-arch-spec/memtracker/mem_track_bundle.cpp:216-286`).
    ///
    /// ⛔ TWELVE, NOT THIRTEEN, AND NOT [`Self::all_trackers_mut`]'s ORDER: `l0ScaleTrack` IS NOT
    /// WALKED HERE although `growExPhases` walks it (`:20-35`), and `pelrfTrack` comes before
    /// `sfplrfTrack` here and after it there. Reusing the grow walk would silently insert a phase
    /// into every L0-scale tracker too, and it would compile.
    fn shifted_trackers_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.hbm_track
            .values_mut()
            .chain(self.lx_track_per_core.values_mut())
            .chain(self.lx_track_st_pin_per_core.values_mut())
            .chain(self.pelrf_track.values_mut().flat_map(BTreeMap::values_mut))
            .chain(
                self.sfplrf_track
                    .values_mut()
                    .flat_map(BTreeMap::values_mut),
            )
            .chain(self.l0_track.values_mut().flat_map(BTreeMap::values_mut))
            .chain(
                self.sfp_state_track
                    .values_mut()
                    .flat_map(BTreeMap::values_mut),
            )
            .chain(
                self.pe_state_track
                    .values_mut()
                    .flat_map(BTreeMap::values_mut),
            )
            .chain(
                self.lxlu_scale_track
                    .values_mut()
                    .flat_map(BTreeMap::values_mut),
            )
            .chain(
                self.xrf_track
                    .values_mut()
                    .flat_map(|per_corelet| per_corelet.values_mut())
                    .flat_map(BTreeMap::values_mut),
            )
            .chain(
                self.pt_arf_track
                    .values_mut()
                    .flat_map(|per_corelet| per_corelet.values_mut())
                    .flat_map(BTreeMap::values_mut),
            )
            .chain(
                self.pt_irf_track
                    .values_mut()
                    .flat_map(|per_corelet| per_corelet.values_mut())
                    .flat_map(BTreeMap::values_mut),
            )
    }

    /// Replaces: e017_insertPsBeforeAndRename
    ///
    /// MAKES ROOM FOR A NEW EXECUTION PHASE AT `orig_ps` IN EVERY WALKED TRACKER: each renames its
    /// phases DOWNWARD from the top, so no rename ever lands on a live key, and then binds `orig_ps`
    /// to a COPY of what `orig_ps` used to hold (`mem_track_bundle.cpp:204-287`).
    ///
    /// ⛔ TWELVE MAPS, NOT `grow_ex_phases`' THIRTEEN — see [`Self::shifted_trackers_mut`].
    /// ⛔ THE REFERENCE'S `DT_CHECK(orig_ps >= 0)` IS THIS SIGNATURE: [`ExPhase`] is unsigned.
    /// ⛔ A refusal is a `DT_CHECK` abort there, so the trackers walked before it STAY SHIFTED — the
    /// reference has no rollback and inventing one would diverge.
    #[must_use]
    pub(crate) fn insert_ps_before_and_rename(&mut self, orig_ps: ExPhase) -> PhaseInsertion {
        for tracker in self.shifted_trackers_mut() {
            let count = tracker.ex_phase_count();
            if i64::from(orig_ps.0) >= i64::from(count.0) {
                return PhaseInsertion::PhaseOutOfRange;
            }
            // `for (eps = exPhases - 1; eps >= orig_ps; eps--)` — the guard above leaves the count
            // strictly above `orig_ps` and therefore positive, so the top of the range is `count-1`.
            for eps in (orig_ps.0..count.0.unsigned_abs()).rev() {
                match tracker.rename_eps(ExPhase(eps), ExPhase(eps + 1)) {
                    EpsEdit::Done => {}
                    refused => return PhaseInsertion::RenameRefused(refused),
                }
            }
            match tracker.add_eps_before(orig_ps, ExPhase(orig_ps.0 + 1)) {
                EpsEdit::Done => {}
                refused => return PhaseInsertion::AddRefused(refused),
            }
        }
        PhaseInsertion::Inserted
    }
}

#[cfg(test)]
mod tests_e017 {
    use super::super::memory::{Capacity, DsKey, find_or_create_ds_key};
    use super::super::tracker::AllocGranularity;
    use super::*;
    use crate::schedule::ddc::v1::StorageName;
    use std::num::NonZeroI64;

    /// LX's whole space — `lxCapacity` (`sys-arch-spec/sysdef.cpp:211`).
    const LX_CAPACITY: Capacity = Capacity(2_031_616);

    /// `bytesPerStick` (`sys-arch-spec/sysdef.cpp:206`), the granularity every LX tracker is given.
    const BYTES_PER_STICK: AllocGranularity =
        AllocGranularity(NonZeroI64::new(128).expect("a stick is not zero bytes"));

    fn core() -> Core {
        Core::checked(0).expect("core 0")
    }

    fn named(name: &str) -> DsKey {
        find_or_create_ds_key(&StorageName(name.to_owned()))
    }

    /// Which ds each of phases `0..4` holds, `None` where that phase is not bound at all.
    fn held(track: &DsTrackInMem) -> Vec<Option<DsKey>> {
        (0..4_u32)
            .map(|at| {
                track
                    .all_ds_and_size_at_eps(ExPhase(at))
                    .and_then(|map| map.0.first().map(|(key, _)| key.clone()))
            })
            .collect()
    }

    /// 🎯 017/38 THE PHASE AT `orig_ps` IS DUPLICATED AND EVERY PHASE FROM IT UPWARD SHIFTS UP ONE
    /// (`mem_track_bundle.cpp:206-214`) — AND `l0ScaleTrack` IS NOT WALKED: `:216-286` lists TWELVE
    /// maps where `growExPhases` (`:20-35`) lists thirteen, so its trackers keep the old numbering
    /// and the old count.
    #[test]
    fn a_phase_is_inserted_as_a_copy_and_the_l0_scale_trackers_are_left_behind() {
        let corelet = Corelet::at::<0>();
        let three =
            || DsTrackInMem::with_phases_holding(&["a", "b", "c"], LX_CAPACITY, BYTES_PER_STICK);
        let mut bundle = MemTrackBundle::<DsTrackInMem>::default();
        bundle.lx_track_per_core.insert(core(), three());
        bundle
            .l0_scale_track
            .entry(core())
            .or_default()
            .insert(corelet, three());

        assert_eq!(
            bundle.insert_ps_before_and_rename(ExPhase(1)),
            PhaseInsertion::Inserted,
        );

        let walked = &bundle.lx_track_per_core[&core()];
        assert_eq!(walked.ex_phases, ExPhaseCount(4));
        assert_eq!(
            held(walked),
            vec![
                Some(named("a")),
                Some(named("b")),
                Some(named("b")),
                Some(named("c")),
            ],
        );

        let skipped = &bundle.l0_scale_track[&core()][&corelet];
        assert_eq!(skipped.ex_phases, ExPhaseCount(3));
        assert_eq!(
            held(skipped),
            vec![Some(named("a")), Some(named("b")), Some(named("c")), None],
        );
    }
}

/// HOW MANY REGISTERS A FILE HOLDS — `reg.maxNum`, and the SAME quantity `FORCE_XRF_ROWS` overrides
/// (`mem_track_bundle.cpp:136` against `:146`), which is why it is ONE type and not two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RegCount(pub(crate) i64);

/// ONE REGISTER FILE AS `initializeMemoryTrackers` READS IT — `regInfoPerUnit.at(comp).at(reg)`
/// reduced to the two fields it touches (`mem_track_bundle.cpp:75-80`).
///
/// ⛔⛔ THE PRODUCT IS IN BITS. `reg.bitSize` is a register's WIDTH, so a register-file tracker's
/// capacity and granularity are denominated in BITS where LX's and L0's are in BYTES — one reference
/// field, two units (`sys-arch-spec/sysdef.cpp:367-368` gives LXLU's SCALE file ONE register of 1024
/// bits). Nothing here converts, and nothing may.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RegFileSize {
    /// `reg.maxNum`.
    pub(crate) max_num: RegCount,
    /// `reg.bitSize` — the granularity AND a factor of the capacity.
    pub(crate) bit_size: AllocGranularity,
}

impl RegFileSize {
    /// `reg.maxNum * reg.bitSize` (`mem_track_bundle.cpp:80`) — the whole file.
    fn capacity(self) -> Capacity {
        self.over(self.max_num)
    }

    /// The same product over a DIFFERENT count — `numForcedXRFRows * ptXrf.bitSize` (`:146`).
    fn over(self, count: RegCount) -> Capacity {
        Capacity(count.0 * self.bit_size.0.get())
    }
}

/// ⭐⭐ EVERY CAPACITY AND GRANULARITY IN THE CAMPAIGN — what `initializeMemoryTrackers` reads out of
/// `dscGlobal.sysDef` (`sys-arch-spec/memtracker/mem_track_bundle.cpp:38-171`), and so where LX
/// becomes 2,031,616 bytes at a granularity of 128.
///
/// ⛔ AN INPUT AND NOT A LOOKUP, because the mechanism for reaching the operands does not exist yet:
/// `sys_arch_spec::regfile::RegType` has no `SCALE` arm at all, so `regInfoPerUnit.at(LXLU).at(SCALE)`
/// (`:82-84`) is unreachable through the spec crate, and extending that crate is not this campaign's.
/// The twelve operands the reference reads out of `sysDef` are the twelve fields below and the
/// thirteenth field is its environment hack.
///
/// ⛔ THAT IS NOT EVERY `sysDef` READ, AND THE ONE MISSING FROM IT IS A DECISION RATHER THAN AN
/// OPERAND: `coreArch` (`:149-153`) picks `RegType::LRF` below RCUDD1A and `RegType::ARF` at or above
/// it, and this port RESOLVES THAT BRANCH IN ITS CALLER — [`Self::pt_arf`] arrives already chosen
/// (`sysdef.cpp:430-436`). ⛔ NOTE THE `<` IS STRICT THERE where XRF's is `<=` (`sysdef.cpp:437`), so
/// RCUDD1A takes ARF's 4 registers and XRF's 64. The three loop bounds are the index types' own
/// instead, per [`every_core`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TrackerSizes {
    /// `sysDef.bytesPerStick` = 128 (`sys-arch-spec/sysdef.cpp:206`) — LX's and L0's granularity, and
    /// the only granularity here that counts BYTES.
    pub(crate) bytes_per_stick: AllocGranularity,
    /// `sysDef.lxCapacity` = `lxCap - 64*1024` = 2,031,616 (`sysdef.cpp:211`).
    pub(crate) lx_capacity: Capacity,
    /// `sysDef.l0Capacity` — `64 * bytesPerStick` up to RCUDD1A, `128 *` above it
    /// (`sysdef.cpp:207-208`).
    pub(crate) l0_capacity: Capacity,
    /// `sysDef.l0ScaleCapacity` — ⛔ ZERO up to RCUDD1A (`sysdef.cpp:209-210`), and a tracker over a
    /// zero-byte space is a reference state rather than a missing one.
    pub(crate) l0_scale_capacity: Capacity,
    /// `regInfoPerUnit.at(SFP).at(LRF)`.
    pub(crate) sfp_lrf: RegFileSize,
    /// `regInfoPerUnit.at(SFP).at(STATE)`.
    pub(crate) sfp_state: RegFileSize,
    /// `regInfoPerUnit.at(PE).at(LRF)`.
    pub(crate) pe_lrf: RegFileSize,
    /// `regInfoPerUnit.at(PE).at(STATE)`.
    pub(crate) pe_state: RegFileSize,
    /// `regInfoPerUnit.at(LXLU).at(SCALE)`.
    pub(crate) lxlu_scale: RegFileSize,
    /// `regInfoPerUnit.at(PT).at(XRF)`.
    pub(crate) pt_xrf: RegFileSize,
    /// `regInfoPerUnit.at(PT).at(coreArch < RCUDD1A_ISA ? LRF : ARF)` (`mem_track_bundle.cpp:149-153`).
    ///
    /// ⛔ THE `LRF` ARM IS UNREACHABLE FROM THIS BUILD, and that is the enum's doing rather than an
    /// omission: `<` is STRICT, `DEFAULT_ISA = RCUDD1A_ISA` (`sys-arch-spec/isa/isa.hpp:31`) is the
    /// LOWEST generation [`crate::arch::IsaGen`] models, and MPW2/3/4 (`isa.hpp:26-28`) are left out
    /// of this compiler on purpose (`src/arch.rs:20-22`). So this field is the ARF file.
    pub(crate) pt_arf: RegFileSize,
    /// `regInfoPerUnit.at(PT).at(IRF)`.
    pub(crate) pt_irf: RegFileSize,
    /// `FORCE_XRF_ROWS`, as the CONST INPUT it has to be — [`None`] is the reference's
    /// `numForcedXRFRows = -1`, its "variable unset" (`mem_track_bundle.cpp:43-47`).
    ///
    /// ⛔ A DOCUMENTED VENDOR HACK, PORTED AS A PARAMETER: the reference reads the environment to cap
    /// AIU MPW4 at 8 XRFs per PT row (`:40-42`), and this repo never gates behaviour on the
    /// environment. ⛔ A value BELOW 1 creates nothing, because the reference's arm is `>= 1` (`:138`).
    pub(crate) forced_xrf_rows: Option<RegCount>,
}

/// WHAT `initializeMemoryTrackers` ASKS OF ONE TRACKER — the three members it touches
/// (`mem_track_bundle.cpp:51`, `:59-62`), a trait for the same reason [`GrowExPhases`] is one.
pub(crate) trait InitMemTrack {
    /// `tracker.exPhases` — read off the trackers a PREVIOUS stage created, never off the ones this
    /// unit is about to make (`mem_track_bundle.cpp:51-53`).
    fn ex_phases(&self) -> ExPhaseCount;

    /// `initMemTrack(memName, memCap, ts, allocGran)` — entry 032.
    fn init_mem_track(
        &mut self,
        mem_name: MemName,
        mem_cap: Capacity,
        ts: ExPhaseCount,
        alloc_gran: AllocGranularity,
    );

    /// `formatMemTrack()` — entry 028.
    fn format_mem_track(&mut self);
}

impl InitMemTrack for DsTrackInMem {
    fn ex_phases(&self) -> ExPhaseCount {
        self.ex_phases
    }

    fn init_mem_track(
        &mut self,
        mem_name: MemName,
        mem_cap: Capacity,
        ts: ExPhaseCount,
        alloc_gran: AllocGranularity,
    ) {
        // ⛔ EVERY CALL SITE IN THIS UNIT TAKES `isStrict`'s DEFAULT `true`
        // (`util/memtracker/mem_track.h:57-58`), so every tracker it makes assigns addresses rather
        // than only counting bytes.
        DsTrackInMem::init_mem_track(
            self,
            mem_name,
            mem_cap,
            ts,
            alloc_gran,
            TrackingMode::AddressAssignment,
        );
    }

    fn format_mem_track(&mut self) {
        DsTrackInMem::format_mem_track(self);
    }
}

/// `initMemTrack(...)` and then the EXPLICIT `formatMemTrack()` the reference writes after each of its
/// twelve calls (`mem_track_bundle.cpp:59-62`, and eleven more down to `:168`).
///
/// ⛔ THE SECOND FORMAT IS A NO-OP AND NOT A SECOND INIT: entry 032 already calls entry 028
/// (`mem_track.cpp:112`), which tombstones every phase and rebinds exactly `[0, exPhases)`, so running
/// it twice lands on the same state. A second `initMemTrack` would NOT.
fn seed<T: InitMemTrack>(
    tracker: &mut T,
    name: String,
    capacity: Capacity,
    steps: ExPhaseCount,
    granularity: AllocGranularity,
) {
    tracker.init_mem_track(MemName(name), capacity, steps, granularity);
    tracker.format_mem_track();
}

/// `sfpCore[cl]` — `std::map::operator[]`, which DEFAULT-CONSTRUCTS every level it walks instead of
/// throwing the way [`at`] does.
fn corelet_slot<V: Default>(
    map: &mut BTreeMap<Core, BTreeMap<Corelet, V>>,
    core: Core,
    corelet: Corelet,
) -> &mut V {
    map.entry(core).or_default().entry(corelet).or_default()
}

/// `xrfCore[cl][r]` — the same walk one level deeper.
fn row_slot<T: Default>(
    map: &mut BTreeMap<Core, BTreeMap<Corelet, BTreeMap<Row, T>>>,
    core: Core,
    corelet: Corelet,
    row: Row,
) -> &mut T {
    corelet_slot(map, core, corelet).entry(row).or_default()
}

/// `for (c = 0; c < dscGlobal.sysDef.numCores; c++)` (`mem_track_bundle.cpp:56`).
///
/// ⛔ THE BOUND IS THE TYPE'S AND NOT A COUNTER'S: [`Core`] is `CoreId<{Target::CORES}>`
/// (`src/units.rs:30`), so this loop cannot run past the build's core count and one arch's index
/// cannot key another's map — `numCores` (`sys-arch-spec/sysdef.h:95`) is not a value this unit reads.
fn every_core() -> impl Iterator<Item = Core> {
    (0..).map_while(Core::checked)
}

/// `for (cl = 0; cl < dscGlobal.sysDef.numCoreletsPerCore; cl++)` (`mem_track_bundle.cpp:74`).
fn every_corelet() -> impl Iterator<Item = Corelet> {
    (0..).map_while(Corelet::checked)
}

/// `for (r = 0; r < dscGlobal.sysDef.numPTRows; r++)` (`mem_track_bundle.cpp:128`).
fn every_row() -> impl Iterator<Item = Row> {
    (0..).map_while(Row::checked)
}

impl<T: InitMemTrack + Default> MemTrackBundle<T> {
    /// Replaces: e036_initializeMemoryTrackers
    ///
    /// ⭐⭐ WHERE EVERY CAPACITY AND GRANULARITY IN THE CAMPAIGN IS SET: one tracker per core, per
    /// `(core, corelet)` and per `(core, corelet, row)`, each over its own space at its own
    /// granularity (`sys-arch-spec/memtracker/mem_track_bundle.cpp:38-171`).
    ///
    /// ⛔ SKIP-IF-PRESENT, NOT OVERWRITE (`:50`, `:54`): a non-empty `lxTrackPerCore` or `xrfTrack`
    /// means a previous stage built it, and re-initialising would THROW AWAY what it tracks. NINE of
    /// the bundle's other eleven families are re-initialised unconditionally — `lxTrackStPinPerCore`
    /// and `hbmTrack` ARE NEVER WRITTEN HERE at all (nothing in `:38-171` assigns either), and
    /// `hbmTrack` is the one `numSteps` is READ off (`:53`), so re-initialising it would destroy the
    /// count this unit reads.
    /// ⛔ `numSteps` COMES FROM WHAT ALREADY EXISTS — LX's phase count if LX is there, else HBM's,
    /// else 1 (`:51-53`) — which is why growing a bundle has to walk every tracker together.
    /// ⛔ THE REGISTER FILES ARE DENOMINATED IN BITS and LX/L0 in bytes; see [`RegFileSize`].
    /// ⛔ THE PT IRF TRACKER IS NAMED `"ptArfCore{c}Cl{cl}Row{r}"` (`:165-166`) — the SAME string as
    /// the ARF tracker beside it, a reference copy-paste. Ported verbatim: a "fixed" name diverges.
    pub(crate) fn initialize_memory_trackers(&mut self, sizes: &TrackerSizes) {
        let init_lx = self.lx_track_per_core.is_empty();
        let init_xrf = self.xrf_track.is_empty();
        // `!initLx ? lxTrackPerCore.begin()->second.exPhases : hbmTrack.empty() ? 1 :
        // hbmTrack.begin()->second.exPhases` (`:51-53`) — `begin()` is the LOWEST key, which is what
        // makes a `BTreeMap` and not a hash map the reading of `std::map` here.
        let num_steps = match self.lx_track_per_core.first_key_value() {
            Some((_, existing)) => existing.ex_phases(),
            None => match self.hbm_track.first_key_value() {
                Some((_, existing)) => existing.ex_phases(),
                None => ExPhaseCount(1),
            },
        };

        for core in every_core() {
            let c = core.get();
            if init_lx {
                seed(
                    self.lx_track_per_core.entry(core).or_default(),
                    format!("lxCore{c}"),
                    sizes.lx_capacity,
                    num_steps,
                    sizes.bytes_per_stick,
                );
            }
            for corelet in every_corelet() {
                let cl = corelet.get();
                seed(
                    corelet_slot(&mut self.sfplrf_track, core, corelet),
                    format!("sfpCore{c}Cl{cl}"),
                    sizes.sfp_lrf.capacity(),
                    num_steps,
                    sizes.sfp_lrf.bit_size,
                );
                seed(
                    corelet_slot(&mut self.lxlu_scale_track, core, corelet),
                    // ⛔ NO `Core` IN THIS ONE'S NAME (`:87`), where every other name has it.
                    format!("lxluScale{c}Cl{cl}"),
                    sizes.lxlu_scale.capacity(),
                    num_steps,
                    sizes.lxlu_scale.bit_size,
                );
                seed(
                    corelet_slot(&mut self.sfp_state_track, core, corelet),
                    format!("sfpState{c}Cl{cl}"),
                    sizes.sfp_state.capacity(),
                    num_steps,
                    sizes.sfp_state.bit_size,
                );
                seed(
                    corelet_slot(&mut self.pelrf_track, core, corelet),
                    format!("peCore{c}Cl{cl}"),
                    sizes.pe_lrf.capacity(),
                    num_steps,
                    sizes.pe_lrf.bit_size,
                );
                seed(
                    corelet_slot(&mut self.pe_state_track, core, corelet),
                    format!("peState{c}Cl{cl}"),
                    sizes.pe_state.capacity(),
                    num_steps,
                    sizes.pe_state.bit_size,
                );
                seed(
                    corelet_slot(&mut self.l0_track, core, corelet),
                    format!("l0Core{c}Cl{cl}"),
                    sizes.l0_capacity,
                    num_steps,
                    sizes.bytes_per_stick,
                );
                seed(
                    corelet_slot(&mut self.l0_scale_track, core, corelet),
                    format!("l0ScaleCore{c}Cl{cl}"),
                    sizes.l0_scale_capacity,
                    num_steps,
                    sizes.bytes_per_stick,
                );
                // `ptArfCl`, `ptIrfCl` and `xrfCl` are `operator[]` too (`:125-127`): the per-corelet
                // level of all three per-row maps is created whether or not the row loop below writes
                // into it, and `xrfCl` is created even when no XRF row is.
                corelet_slot(&mut self.pt_arf_track, core, corelet);
                corelet_slot(&mut self.pt_irf_track, core, corelet);
                corelet_slot(&mut self.xrf_track, core, corelet);
                for row in every_row() {
                    let r = row.get();
                    // ⛔ THE XRF ROW ENTRY EXISTS ONLY INSIDE THESE TWO ARMS (`:129-148`): with
                    // `initXrf` false and no forced rows, `xrfCl[r]` is never named and that row gets
                    // no tracker at all.
                    if init_xrf {
                        seed(
                            row_slot(&mut self.xrf_track, core, corelet, row),
                            format!("xrfCore{c}Cl{cl}Row{r}"),
                            sizes.pt_xrf.capacity(),
                            num_steps,
                            sizes.pt_xrf.bit_size,
                        );
                    } else if let Some(forced) = sizes.forced_xrf_rows.filter(|rows| rows.0 >= 1) {
                        // The override RE-INITIALISES the trackers the skip above preserved, and what
                        // changes is the register COUNT — the granularity is `ptXrf.bitSize` either
                        // way (`:146`).
                        seed(
                            row_slot(&mut self.xrf_track, core, corelet, row),
                            format!("xrfCore{c}Cl{cl}Row{r}"),
                            sizes.pt_xrf.over(forced),
                            num_steps,
                            sizes.pt_xrf.bit_size,
                        );
                    }
                    seed(
                        row_slot(&mut self.pt_arf_track, core, corelet, row),
                        format!("ptArfCore{c}Cl{cl}Row{r}"),
                        sizes.pt_arf.capacity(),
                        num_steps,
                        sizes.pt_arf.bit_size,
                    );
                    seed(
                        row_slot(&mut self.pt_irf_track, core, corelet, row),
                        // ⛔ `"ptArfCore..."`, NOT `"ptIrfCore..."` (`:165-166`) — see this unit's
                        // anchor.
                        format!("ptArfCore{c}Cl{cl}Row{r}"),
                        sizes.pt_irf.capacity(),
                        num_steps,
                        sizes.pt_irf.bit_size,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests_e036 {
    use super::*;
    use std::num::NonZeroI64;

    /// `bytesPerStick` (`sys-arch-spec/sysdef.cpp:206`).
    const BYTES_PER_STICK: AllocGranularity =
        AllocGranularity(NonZeroI64::new(128).expect("a stick is not zero bytes"));

    fn core(index: u32) -> Core {
        Core::checked(index).expect("a core this build has")
    }

    fn corelet(index: u32) -> Corelet {
        Corelet::checked(index).expect("a corelet this build has")
    }

    fn row(index: u32) -> Row {
        Row::checked(index).expect("a PT row this build has")
    }

    /// EVERY OPERAND RCUDD1A GIVES THIS UNIT, cited one by one — the generation this build selects,
    /// and the source of the 32 cores, 2 corelets and 8 PT rows counted below (`src/arch.rs:287-293`).
    fn rcudd1a() -> TrackerSizes {
        let bits = |n: i64| AllocGranularity(NonZeroI64::new(n).expect("a register has a width"));
        let file = |max_num: i64, bit_size: i64| RegFileSize {
            max_num: RegCount(max_num),
            bit_size: bits(bit_size),
        };
        TrackerSizes {
            bytes_per_stick: BYTES_PER_STICK,
            lx_capacity: Capacity(2_031_616),  // `lxCap - 64*1024` (`sysdef.cpp:211`)
            l0_capacity: Capacity(8192),       // `64 * bytesPerStick` (`:207-208`)
            l0_scale_capacity: Capacity(0),    // ⛔ ZERO on this arch (`:209-210`)
            sfp_lrf: file(16, 128),            // `:400-401`
            sfp_state: file(1, 128),           // `:402-403`
            pe_lrf: file(16, 128),             // `:415-416`
            pe_state: file(1, 128),            // `:417-418`
            lxlu_scale: file(1, 1024),         // ⛔ 1024-BIT REGISTERS (`:367-368`)
            pt_xrf: file(64, 128),             // `:438-439`
            pt_arf: file(4, 128),              // the `else` of `coreArch < RCUDD1A` (`:430-436`)
            pt_irf: file(2, 128),              // `:444-445`
            forced_xrf_rows: None,             // `FORCE_XRF_ROWS` unset (`:43-47`)
        }
    }

    /// The three fields `initMemTrack` is handed, with the granularity as a plain number so a BIT
    /// count and a BYTE count are read side by side.
    fn seeded(track: &DsTrackInMem) -> (String, Capacity, i64) {
        (
            track.name.0.clone(),
            track.mem_capacity,
            track.alloc_granularity.0.get(),
        )
    }

    /// 🎯 036/38 ONE TRACKER PER CORE, PER CORELET AND PER ROW, EACH OVER ITS OWN SPACE AT ITS OWN
    /// GRANULARITY — and the register files' capacities are in BITS where LX's and L0's are in bytes
    /// (`mem_track_bundle.cpp:56-171` over the operands of [`rcudd1a`]).
    #[test]
    fn every_family_is_seeded_with_the_reference_s_own_name_capacity_and_granularity() {
        let mut bundle = MemTrackBundle::<DsTrackInMem>::default();

        bundle.initialize_memory_trackers(&rcudd1a());

        // LX: the whole 2,031,616-byte space at a stick, one per core, in address-assignment mode
        // because `initMemTrack`'s `isStrict` defaults to `true` (`mem_track.h:57-58`).
        assert_eq!(bundle.lx_track_per_core.len(), 32);
        let lx = &bundle.lx_track_per_core[&core(31)];
        assert_eq!(
            seeded(lx),
            ("lxCore31".to_owned(), Capacity(2_031_616), 128),
        );
        assert_eq!(lx.strict, TrackingMode::AddressAssignment);
        // ⛔ ON AN EMPTY BUNDLE `numSteps` IS 1, not 0 (`:52`) — a tracker with no phase would place
        // nothing.
        assert_eq!(lx.ex_phases, ExPhaseCount(1));
        // ⛔ AND `lxTrackStPinPerCore` IS NEVER CREATED HERE (nothing in `:38-171` names it).
        assert!(bundle.lx_track_st_pin_per_core.is_empty());

        // The seven per-corelet families, in the reference's own order (`:78-124`).
        let cl = corelet(1);
        let per_corelet = [
            (&bundle.sfplrf_track, "sfpCore0Cl1", Capacity(2048), 128),
            (&bundle.lxlu_scale_track, "lxluScale0Cl1", Capacity(1024), 1024),
            (&bundle.sfp_state_track, "sfpState0Cl1", Capacity(128), 128),
            (&bundle.pelrf_track, "peCore0Cl1", Capacity(2048), 128),
            (&bundle.pe_state_track, "peState0Cl1", Capacity(128), 128),
            (&bundle.l0_track, "l0Core0Cl1", Capacity(8192), 128),
            (&bundle.l0_scale_track, "l0ScaleCore0Cl1", Capacity(0), 128),
        ];
        for (family, name, capacity, granularity) in per_corelet {
            assert_eq!(family.len(), 32);
            assert_eq!(family[&core(0)].len(), 2);
            assert_eq!(
                seeded(&family[&core(0)][&cl]),
                (name.to_owned(), capacity, granularity),
            );
        }

        // The three per-row families: 8 rows apiece (`Target::PT_ROWS`), and the IRF tracker carries
        // the ARF's NAME (`:165-166`) with the IRF's own 2 × 128 bits.
        let per_row = [
            (&bundle.xrf_track, "xrfCore0Cl1Row7", Capacity(8192)),
            (&bundle.pt_arf_track, "ptArfCore0Cl1Row7", Capacity(512)),
            (&bundle.pt_irf_track, "ptArfCore0Cl1Row7", Capacity(256)),
        ];
        for (family, name, capacity) in per_row {
            assert_eq!(family[&core(0)][&cl].len(), 8);
            assert_eq!(
                seeded(&family[&core(0)][&cl][&row(7)]),
                (name.to_owned(), capacity, 128),
            );
        }
    }

    /// 🎯 036/38 SKIP-IF-PRESENT, AND `numSteps` COMES OFF WHAT SURVIVED — `initLx`/`initXrf` (`:50`,
    /// `:54`) leave a previous stage's trackers alone, `numSteps` is read from LX's phase count or else
    /// HBM's (`:51-53`), and the `FORCE_XRF_ROWS` arm is the ONLY thing that re-initialises a kept XRF
    /// row (`:138-147`).
    #[test]
    fn a_kept_tracker_is_not_reseeded_but_sets_num_steps_and_only_forced_rows_resize_the_xrf() {
        let three = || {
            DsTrackInMem::with_phases_holding(&["a", "b", "c"], Capacity(4096), BYTES_PER_STICK)
        };
        let mut bundle = MemTrackBundle::<DsTrackInMem>::default();
        bundle.lx_track_per_core.insert(core(0), three());
        // A per-corelet level with no row in it still makes `xrfTrack` non-empty, so `initXrf` is
        // false (`:54`).
        bundle
            .xrf_track
            .entry(core(0))
            .or_default()
            .entry(corelet(0))
            .or_default();

        bundle.initialize_memory_trackers(&rcudd1a());

        // ⛔ LX IS NOT RE-INITIALISED AND NOT EXTENDED TO THE OTHER 31 CORES: the whole `if (initLx)`
        // block is skipped (`:57-63`), so the kept tracker keeps the 4096-byte space it was given and
        // the empty name a defaulted tracker has — `initMemTrack` never ran on it.
        assert_eq!(bundle.lx_track_per_core.len(), 1);
        assert_eq!(
            seeded(&bundle.lx_track_per_core[&core(0)]),
            (String::new(), Capacity(4096), 128),
        );
        // ...and every tracker this unit DID make took ITS phase count, not 1 (`:51`).
        assert_eq!(bundle.l0_track[&core(31)][&corelet(1)].ex_phases, ExPhaseCount(3));
        // ⛔ WITH `initXrf` FALSE AND NO FORCED ROWS, NO XRF ROW IS CREATED ANYWHERE (`:129-148`) —
        // while the per-corelet level itself is, by `operator[]` (`:127`).
        assert_eq!(bundle.xrf_track[&core(31)].len(), 2);
        assert!(bundle.xrf_track[&core(31)][&corelet(1)].is_empty());
        assert!(bundle.xrf_track[&core(0)][&corelet(0)].is_empty());

        // Now with no LX at all, an HBM tracker to read the count off, one XRF row already built, and
        // the environment hack set to 8 rows.
        let mut forced = MemTrackBundle::<DsTrackInMem>::default();
        forced.hbm_track.insert(
            LdsSegment::Model,
            DsTrackInMem::with_phases_holding(&["a", "b"], Capacity(4096), BYTES_PER_STICK),
        );
        forced
            .xrf_track
            .entry(core(0))
            .or_default()
            .entry(corelet(0))
            .or_default()
            .insert(row(0), three());
        let sizes = TrackerSizes {
            forced_xrf_rows: Some(RegCount(8)),
            ..rcudd1a()
        };

        forced.initialize_memory_trackers(&sizes);

        // LX was empty, so it IS seeded — with HBM's 2 phases (`:52-53`).
        assert_eq!(forced.lx_track_per_core[&core(0)].ex_phases, ExPhaseCount(2));
        // ⛔ THE HACK RE-INITIALISES THE ROW THE SKIP HAD PRESERVED, at `numForcedXRFRows *
        // ptXrf.bitSize` = 8 × 128 rather than the file's own 64 × 128 (`:146` against `:136`).
        let rows = &forced.xrf_track[&core(0)][&corelet(0)];
        assert_eq!(rows.len(), 8);
        assert_eq!(
            seeded(&rows[&row(0)]),
            ("xrfCore0Cl0Row0".to_owned(), Capacity(1024), 128),
        );
        assert_eq!(
            seeded(&rows[&row(7)]),
            ("xrfCore0Cl0Row7".to_owned(), Capacity(1024), 128),
        );
    }
}
