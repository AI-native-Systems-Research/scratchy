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

//! `MemoryOrganizer` — THE BLOCK LIST, and where EVERY address actually comes from. A sorted-by-address `std::list<MemBlock>` plus a free-capacity counter. `getAllFreeBlocks` derives the gaps as INCLUSIVE `[start, end]` pairs and `allocateMemory` commits one. ⛔ The sorted-by-address invariant lives in `addMemBlock` and THREE other units silently depend on it.
//!
//! `util/memtracker/memory.cpp` — 10 of the campaign's 38 units (dependency level(s) [0, 1]).
//!
//! | unit | entry | level | lines | class | authority path:line |
//! |---|---|---|---|---|---|
//! | `e010_findOrCreateDsKey` | 010 | 0 | 4 | — | `util/memtracker/memory.cpp:16` |
//! | `e011_clear` | 011 | 0 | 4 | `MemoryOrganizer` | `util/memtracker/memory.cpp:28` |
//! | `e012_freeMemBlockByName` | 012 | 0 | 11 | `MemoryOrganizer` | `util/memtracker/memory.cpp:33` |
//! | `e013_addMemBlock` | 013 | 0 | 23 | `MemoryOrganizer` | `util/memtracker/memory.cpp:45` |
//! | `e014_findMemoryBound` | 014 | 0 | 6 | `MemoryOrganizer` | `util/memtracker/memory.cpp:69` |
//! | `e015_findFirstFittingMemory` | 015 | 0 | 39 | `MemoryOrganizer` | `util/memtracker/memory.cpp:76` |
//! | `e016_findAddressByName` | 016 | 0 | 8 | `MemoryOrganizer` | `util/memtracker/memory.cpp:161` |
//! | `e024_MemoryOrganizer` | 024 | 1 | 1 | `MemoryOrganizer` | `util/memtracker/memory.cpp:21` |
//! | `e025_allocateMemory` | 025 | 1 | 23 | `MemoryOrganizer` | `util/memtracker/memory.cpp:136` |
//! | `e026_getAllFreeBlocks` | 026 | 1 | 18 | `MemoryOrganizer` | `util/memtracker/memory.cpp:217` |

use crate::schedule::ddc::v1::StorageName;
use std::ops::{AddAssign, SubAssign};

// ───────────────────────────────────────────────────────────────────────────────────────────────
// THE BLOCK-LIST VOCABULARY — `util/memtracker/memory.h:18-49`. Declarations, not units: the
// members that ARE units keep their anchors below.
//
// ⛔ TWO QUANTITIES, NOT ONE `int64_t`. The reference mixes a POSITION and an EXTENT in the same
// comparison — `reqCap < memOccupied_.begin()->startAddress_` (`memory.cpp:90`) and
// `reqEndAddress < capacity_` (`:105`) — so transposing them has to be an E0308 here. `capacity_`,
// `freeCapacity_`, `size_` and `reqCap` are all ONE quantity in the reference (an extent in bytes)
// and stay one type; splitting those four further would invent conversions the bodies do not have.
// ⛔ SIGNED, like the reference's own `int64_t`: `allocateMemory` decrements `freeCapacity_`
// unconditionally on the PINNED path without ever checking the fit, and `getAllFreeBlocks` walks
// from `prevEndAddr = -1`. An unsigned extent would wrap there and hand out the whole space.
// ───────────────────────────────────────────────────────────────────────────────────────────────

/// A BYTE ADDRESS in one memory space — `MemBlock::startAddress_`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Address(pub i64);

/// AN EXTENT IN BYTES — the reference's `capacity_`, `freeCapacity_`, `size_` and `reqCap`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Capacity(pub i64);

impl Address {
    /// THE FIRST ADDRESS OF THE SPACE, and a real address rather than a sentinel.
    pub const ZERO: Self = Self(0);

    /// `startAddress_ + size_` (`memory.cpp:73`, `:95`) — the first address past a block of `cap`
    /// placed here.
    #[must_use]
    pub const fn past(self, cap: Capacity) -> Self {
        Self(self.0 + cap.0)
    }

    /// `reqStartAddress + reqCap - 1` (`memory.cpp:96`) — the LAST byte a block of `cap` placed here
    /// occupies, so an exact fill of a gap ends one byte BELOW the block that follows it.
    #[must_use]
    pub const fn last_byte_of(self, cap: Capacity) -> Self {
        Self(self.0 + cap.0 - 1)
    }

    /// THE EXTENT BELOW THIS ADDRESS — `[0, self)`, which is what the reference compares `reqCap`
    /// against at the head of memory (`memory.cpp:90`).
    #[must_use]
    pub const fn extent_below(self) -> Capacity {
        Capacity(self.0)
    }

    /// `entry.startAddress_ - 1` (`memory.cpp:225`, `:227`) — THE LAST BYTE BELOW THIS ADDRESS,
    /// where the gap in front of a block ends.
    #[must_use]
    pub const fn last_byte_below(self) -> Self {
        Self(self.0 - 1)
    }
}

impl Capacity {
    /// THE FIRST ADDRESS THE SPACE DOES NOT CONTAIN — `capacity_` read as a bound
    /// (`memory.cpp:105`).
    #[must_use]
    pub const fn limit(self) -> Address {
        Address(self.0)
    }

    /// `capacity_ - 1` (`memory.cpp:221`, `:231`) — THE LAST BYTE THE SPACE CONTAINS, and the highest
    /// `last` any [`FreeBlock`] can carry.
    ///
    /// ⛔ A SPACE OF 0 BYTES ANSWERS `Address(-1)`: that is the pair the reference itself pushes there
    /// (`memory.cpp:221`), and it is not an address.
    #[must_use]
    pub const fn last_byte(self) -> Address {
        Address(self.0 - 1)
    }
}

impl AddAssign for Capacity {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl SubAssign for Capacity {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

/// THE IDENTITY A TRACKED DS IS HELD UNDER — `StringKey<DsKey>` (`memory.h:18-19`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct DsKey(StorageName);

/// `StringKey::operator==(std::string_view)` (`util/stringkeyset.hpp:33`) — the comparison both
/// lookups in this file run, and it is by string CONTENT and not by interned identity.
impl PartialEq<StorageName> for DsKey {
    fn eq(&self, other: &StorageName) -> bool {
        self.0 == *other
    }
}

impl DsKey {
    /// `StringKey::str()` (`util/stringkeyset.hpp:34-37`) — the interned name back out, which is what
    /// `backupEps` (unit e023) copies into `DsMemInfo::ds` and hands to `restoreEps` (unit e035).
    #[must_use]
    pub fn name(&self) -> &StorageName {
        &self.0
    }
}

/// ONE OCCUPIED BLOCK — `MemBlock` (`memory.h:21-25`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemBlock {
    /// `name_`.
    pub name: DsKey,
    /// `startAddress_`.
    pub start: Address,
    /// `size_`.
    pub size: Capacity,
}

/// THE BLOCK LIST — `MemoryOrganizer` (`memory.h:28-49`), one per execution phase of a tracker.
///
/// ⛔ `memOccupied_` IS SORTED BY ADDRESS and [`MemoryOrganizer::add_mem_block`] is the only thing
/// that keeps it so. A `Vec` stands in for the reference's `std::list` because nothing here holds a
/// position across a mutation — `DsTrackInMem::epsToListIter` does, but that is over `entries_`.
///
/// ⛔ THE FIELDS ARE `pub(super)` FOR `tracker`'s PHASE FIXTURES ONLY. `DsTrackInMem::Entry` holds
/// one of these per execution phase, and [`MemoryOrganizer::new`] (unit e024) is the only sanctioned
/// way to reach a seeded one.
#[derive(Debug, Clone)]
pub struct MemoryOrganizer {
    /// `capacity_`.
    pub(super) capacity: Capacity,
    /// `freeCapacity_`.
    pub(super) free_capacity: Capacity,
    /// `memOccupied_`, ascending by [`MemBlock::start`].
    pub(super) occupied: Vec<MemBlock>,
}

/// Replaces: e010_findOrCreateDsKey
///
/// THE IDENTITY `memOccupied_` AND `dsInMem_` KEY ON. The reference interns the name in a
/// process-wide `StringKeySet<DsKey>` and hands back a cheap handle, but `StringKey::operator==`
/// compares the pointed-to STRING (`util/stringkeyset.hpp:32-33`) — so the table is a
/// pointer-sharing optimisation and the identity it provides is the name itself.
///
/// ⛔ NO GLOBAL MUTABLE INTERN TABLE: two keys of equal names are equal without one.
#[must_use]
pub fn find_or_create_ds_key(name: &StorageName) -> DsKey {
    DsKey(name.clone())
}

impl MemoryOrganizer {
    /// Replaces: e011_clear
    ///
    /// Empties the space — every block gone and the whole capacity free again (`memory.cpp:28-31`).
    ///
    /// ⛔ THE ONLY PLACE `freeCapacity_` IS SEEDED. The constructor (unit e024) reaches its initial
    /// state through here, and `formatMemTrack` clears every execution phase with it.
    pub fn clear(&mut self) {
        self.free_capacity = self.capacity;
        self.occupied.clear();
    }

    /// Replaces: e012_freeMemBlockByName
    ///
    /// Frees the FIRST block held under `name` and returns; a name held twice loses only one block
    /// (`memory.cpp:33-43`).
    ///
    /// ⛔ THE NOT-FOUND ARM IS NOT A REFUSAL — the reference prints to `std::cout` and returns
    /// normally, and its caller (`removeDs`) cannot tell. Faithful is: do nothing, report nothing.
    pub fn free_mem_block_by_name(&mut self, name: &StorageName) {
        let Some(at) = self.occupied.iter().position(|block| block.name == *name) else {
            return;
        };
        self.free_capacity += self.occupied.remove(at).size;
    }

    /// Replaces: e013_addMemBlock
    ///
    /// Inserts before the first block with a GREATER `startAddress_`, and pushes back when there is
    /// none (`memory.cpp:45-67`) — which is what keeps `memOccupied_` ascending by address.
    ///
    /// ⛔ THE SORTED-BY-ADDRESS INVARIANT LIVES HERE: `findFirstFittingMemory` walks the gaps in
    /// that order, `getAllFreeBlocks` derives them from it, `findMemoryBound` reads the LAST block.
    /// ⛔ The extent leaves `freeCapacity_` once in every arm, and NOTHING checks for an overlap.
    pub fn add_mem_block(&mut self, block: MemBlock) {
        let size = block.size;
        match self
            .occupied
            .iter()
            .position(|held| held.start > block.start)
        {
            Some(at) => self.occupied.insert(at, block),
            None => self.occupied.push(block),
        }
        self.free_capacity -= size;
    }

    /// Replaces: e014_findMemoryBound
    ///
    /// One past the LAST block — `back().startAddress_ + back().size_`, or address 0 when the space
    /// holds nothing (`memory.cpp:69-74`).
    ///
    /// ⛔ THAT 0 IS A REAL ADDRESS, not a sentinel; and this is only the bound because
    /// [`Self::add_mem_block`] keeps the list ascending.
    #[must_use]
    pub fn find_memory_bound(&self) -> Address {
        match self.occupied.last() {
            None => Address::ZERO,
            Some(last) => last.start.past(last.size),
        }
    }

    /// Replaces: e015_findFirstFittingMemory
    ///
    /// FORWARD FIRST FIT — the head gap, then the gap after each block, then the tail
    /// (`memory.cpp:76-114`). [`None`] is the organizer's own `-1`, which is NOT an address.
    ///
    /// ⛔ THE HEAD TEST IS STRICT (`reqCap < begin()->startAddress_`, `:90`): an EXACT fill of the
    /// leading gap is refused. But `reqEndAddress` is the LAST byte occupied (`:96`), so the strict
    /// tail test (`:105`) ADMITS a block ending on `capacity_ - 1` — ⚠️ UNITS.tsv reads it exclusive.
    #[must_use]
    pub fn find_first_fitting_memory(&self, req_cap: Capacity) -> Option<Address> {
        if req_cap > self.free_capacity {
            return None;
        }
        let Some(first) = self.occupied.first() else {
            return Some(Address::ZERO);
        };
        if req_cap < first.start.extent_below() {
            return Some(Address::ZERO);
        }
        for (at, block) in self.occupied.iter().enumerate() {
            let start = block.start.past(block.size);
            let last = start.last_byte_of(req_cap);
            let fits = match self.occupied.get(at + 1) {
                Some(next) => last < next.start,
                None => last < self.capacity.limit(),
            };
            if fits {
                return Some(start);
            }
        }
        None
    }

    /// Replaces: e016_findAddressByName
    ///
    /// The address this space holds `name` at, by linear scan (`memory.cpp:161-168`).
    ///
    /// ⛔ [`None`] IS THE REFERENCE'S `-1`, AND `-1` IS NOT AN ADDRESS: `backupEps` writes it
    /// straight into `DsMemInfo::startAddr`, from which `restoreEps` feeds `addDsAtStartAddr`, whose
    /// `DT_CHECK(startAddr >= 0)` is all that stands between a missing name and a committed
    /// negative address.
    #[must_use]
    pub fn find_address_by_name(&self, name: &StorageName) -> Option<Address> {
        self.occupied
            .iter()
            .find(|block| block.name == *name)
            .map(|block| block.start)
    }
}

/// WHERE A COMMIT GOES — the `staddr` parameter of `allocateMemory` (`memory.h:39`, defaulted to −1).
///
/// ⛔ THE TWO ARMS ARE NOT THE SAME OPERATION. [`Self::FirstFit`] asks the free list and can find
/// nothing; [`Self::At`] is taken ON TRUST with no fit and no overlap test, which is exactly how
/// `reserveFrontendLx` pins `[0, 1625344)` (`dbo/src/Transforms/ProgramLayout.cpp:93`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// `staddr < 0` — let [`MemoryOrganizer::find_first_fitting_memory`] choose.
    FirstFit,
    /// A pinned address, whatever it overlaps. ⛔ The reference reads ANY negative `staddr` as
    /// [`Self::FirstFit`] (`memory.cpp:140`), so a negative address is not a reference state here.
    At(Address),
}

/// ONE GAP, AS AN INCLUSIVE `[first, last]` PAIR OF ADDRESSES — the reference's
/// `std::pair<int64_t, int64_t>` (`memory.cpp:217-235`).
///
/// ⛔ `last` IS THE LAST BYTE FREE AND NOT ONE PAST IT: `findCommonFittingMemory` tests
/// `(iblk.second - iblk.first + 1) >= cap` and back-fits at `iblk.second - cap + 1`
/// (`mem_track.cpp:330-331`, `:338`). An exclusive-end pair would be off by one everywhere and would
/// still compile, so both ends are an [`Address`] and the extent is derived from them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FreeBlock {
    /// `first` — the lowest free address of the gap.
    pub first: Address,
    /// `second` — the HIGHEST free address, `capacity_ - 1` at the top of the space.
    pub last: Address,
}

impl FreeBlock {
    /// `iblk.second - iblk.first + 1` (`mem_track.cpp:330`, `:338`) — how many bytes an INCLUSIVE
    /// pair holds.
    #[must_use]
    pub const fn extent(self) -> Capacity {
        Capacity(self.last.0 - self.first.0 + 1)
    }
}

impl MemoryOrganizer {
    /// Replaces: e024_MemoryOrganizer
    ///
    /// A space of `capacity` bytes holding nothing — `capacity_(cap)` and then `clear()`
    /// (`memory.cpp:21`).
    ///
    /// ⛔ IT SEEDS `freeCapacity_`, AND THAT IS THE WHOLE POINT: [`Self::find_first_fitting_memory`]
    /// refuses every request while that counter is 0 (`memory.cpp:79-84`), so a `Default` skipping
    /// this would place nothing at all.
    #[must_use]
    pub fn new(capacity: Capacity) -> Self {
        let mut organizer = Self {
            capacity,
            free_capacity: Capacity(0),
            occupied: Vec::new(),
        };
        organizer.clear();
        organizer
    }

    /// Replaces: e025_allocateMemory
    ///
    /// THE COMMIT — chooses an address when asked to, then records the block (`memory.cpp:136-159`).
    /// [`None`] is the reference's own `-1` and leaves the space untouched.
    ///
    /// ⛔ THE PINNED ARM IS TAKEN ON TRUST: no fit test, no overlap test, and `freeCapacity_` falls
    /// anyway. ⛔ The block records `reqCap` UNROUNDED — rounding to `allocGranularity` belongs to
    /// the tracker above this layer.
    pub fn allocate_memory(
        &mut self,
        req_cap: Capacity,
        name: &StorageName,
        at: Placement,
    ) -> Option<Address> {
        let start = match at {
            Placement::At(pinned) => pinned,
            Placement::FirstFit => self.find_first_fitting_memory(req_cap)?,
        };
        self.add_mem_block(MemBlock {
            name: find_or_create_ds_key(name),
            start,
            size: req_cap,
        });
        Some(start)
    }

    /// Replaces: e026_getAllFreeBlocks
    ///
    /// EVERY GAP THE BLOCK LIST LEAVES, ascending, as inclusive pairs (`memory.cpp:217-235`).
    ///
    /// ⛔ A ZERO-LENGTH GAP IS NOT A PAIR — `entry.startAddress_ - 1 > prevEndAddr` (`:225`) is
    /// strict — while a ONE-BYTE gap does yield a pair with `first == last`.
    /// ⛔ THE EMPTY SPACE IS ITS OWN ARM and is not the tail arm: at `capacity_ == 0` the reference
    /// pushes `(0, -1)` there where the tail test would have pushed nothing.
    #[must_use]
    pub fn all_free_blocks(&self) -> Vec<FreeBlock> {
        if self.occupied.is_empty() {
            return vec![FreeBlock {
                first: Address::ZERO,
                last: self.capacity.last_byte(),
            }];
        }
        let mut free = Vec::new();
        // `prevEndAddr + 1` throughout, so the reference's `-1` seed — which is not an address — never
        // has to be spelled as one.
        let mut next_free = Address::ZERO;
        for held in &self.occupied {
            if held.start > next_free {
                free.push(FreeBlock {
                    first: next_free,
                    last: held.start.last_byte_below(),
                });
            }
            next_free = held.start.past(held.size);
        }
        if next_free < self.capacity.limit() {
            free.push(FreeBlock {
                first: next_free,
                last: self.capacity.last_byte(),
            });
        }
        free
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{
        Address, Capacity, FreeBlock, MemBlock, MemoryOrganizer, Placement, find_or_create_ds_key,
    };
    use crate::schedule::ddc::v1::StorageName;

    /// An empty space of `capacity` bytes, through the constructor that seeds `freeCapacity_`
    /// (`memory.cpp:21`, unit e024).
    fn space(capacity: i64) -> MemoryOrganizer {
        MemoryOrganizer::new(Capacity(capacity))
    }

    /// One `MemBlock`, reached the way `allocateMemory` builds it (`memory.cpp:147-149`).
    fn block(name: &str, start: i64, size: i64) -> MemBlock {
        MemBlock {
            name: find_or_create_ds_key(&StorageName(name.to_owned())),
            start: Address(start),
            size: Capacity(size),
        }
    }

    fn named(name: &str) -> StorageName {
        StorageName(name.to_owned())
    }

    fn starts(org: &MemoryOrganizer) -> Vec<Address> {
        org.occupied.iter().map(|held| held.start).collect()
    }

    /// 🎯 010/38 A KEY IS THE NAME'S IDENTITY, AND EQUALITY IS BY STRING CONTENT — the reference's
    /// `StringKey::operator==` compares `*str_ == *rhs.str_` (`util/stringkeyset.hpp:32`), so two
    /// keys minted from equal names are one key without any intern table.
    #[test]
    fn two_keys_of_equal_names_are_equal_and_a_key_equals_the_bare_name() {
        let reserved = named("reserved-frontend");
        assert_eq!(
            find_or_create_ds_key(&reserved),
            find_or_create_ds_key(&named("reserved-frontend"))
        );
        // ⛔ `operator==(std::string_view)` (`stringkeyset.hpp:33`) — the comparison both lookups run.
        assert!(find_or_create_ds_key(&reserved) == reserved);
        assert_ne!(
            find_or_create_ds_key(&reserved),
            find_or_create_ds_key(&named("lxCore0"))
        );
    }

    /// 🎯 011/38 CLEAR PUTS THE WHOLE CAPACITY BACK — `freeCapacity_ = capacity_` and an empty block
    /// list (`memory.cpp:28-31`), over the campaign's own LX space of 2,031,616 bytes
    /// (`sys-arch-spec/sysdef.cpp:211`).
    #[test]
    fn clear_restores_the_full_lx_capacity_and_drops_every_block() {
        let mut org = space(2_031_616);
        org.add_mem_block(block("reserved-frontend", 0, 1_625_344));
        assert_eq!(org.free_capacity, Capacity(406_272));
        org.clear();
        assert_eq!(org.free_capacity, Capacity(2_031_616));
        assert!(org.occupied.is_empty());
    }

    /// 🎯 012/38 THE FIRST BLOCK UNDER THAT NAME GOES AND THE SECOND STAYS — the reference returns
    /// from inside the loop (`memory.cpp:37-40`), so a name held twice loses exactly one block.
    #[test]
    fn freeing_a_name_held_twice_releases_only_the_first_block() {
        let mut org = space(4096);
        org.add_mem_block(block("twice", 0, 512));
        org.add_mem_block(block("twice", 512, 1024));
        assert_eq!(org.free_capacity, Capacity(2560));
        org.free_mem_block_by_name(&named("twice"));
        assert_eq!(starts(&org), vec![Address(512)]);
        assert_eq!(org.free_capacity, Capacity(3072));
        // ⛔ AN ABSENT NAME IS A NO-OP AND NOT A REFUSAL: the reference prints to `std::cout` and
        // returns normally (`memory.cpp:42`), and `removeDs` cannot tell.
        org.free_mem_block_by_name(&named("never-held"));
        assert_eq!(starts(&org), vec![Address(512)]);
        assert_eq!(org.free_capacity, Capacity(3072));
    }

    /// 🎯 013/38 THE LIST STAYS ASCENDING BY ADDRESS WHATEVER ORDER THE BLOCKS ARRIVE IN — insert
    /// before the first GREATER `startAddress_`, else push back (`memory.cpp:45-67`).
    #[test]
    fn blocks_added_out_of_order_are_held_ascending_by_address() {
        let mut org = space(2_031_616);
        org.add_mem_block(block("third", 1_626_368, 512));
        org.add_mem_block(block("first", 1_625_344, 512));
        org.add_mem_block(block("second", 1_625_856, 512));
        assert_eq!(
            starts(&org),
            vec![Address(1_625_344), Address(1_625_856), Address(1_626_368)]
        );
        // The extent leaves `freeCapacity_` once per block, in every arm.
        assert_eq!(org.free_capacity, Capacity(2_031_616 - 1536));
    }

    /// 🎯 014/38 THE BOUND IS ONE PAST THE LAST BLOCK, AND 0 ON AN EMPTY SPACE — `back()` plus its
    /// size (`memory.cpp:69-74`); with the front end reserved that bound is the corpus's own first
    /// real LX address, 1,625,344 (`dbo/src/Transforms/ProgramLayout.cpp:81-98`).
    #[test]
    fn the_memory_bound_is_zero_when_empty_and_one_past_the_last_block_after() {
        let mut org = space(2_031_616);
        assert_eq!(org.find_memory_bound(), Address(0));
        org.add_mem_block(block("reserved-frontend", 0, 1_625_344));
        assert_eq!(org.find_memory_bound(), Address(1_625_344));
    }

    /// 🎯 015/38 THE MEASURED LAW: capacity 2,031,616 with `[0, 1625344)` held by
    /// `reserved-frontend`, 512-byte requests land at 1,625,344, then 1,625,856, then 1,626,368 —
    /// the addresses the reference itself recorded over 588 LX allocations in 187 programs.
    #[test]
    fn forward_first_fit_above_the_reserved_front_end_yields_the_recorded_addresses() {
        let mut org = space(2_031_616);
        org.add_mem_block(block("reserved-frontend", 0, 1_625_344));
        assert_eq!(
            org.find_first_fitting_memory(Capacity(512)),
            Some(Address(1_625_344))
        );
        org.add_mem_block(block("lds0", 1_625_344, 512));
        assert_eq!(
            org.find_first_fitting_memory(Capacity(512)),
            Some(Address(1_625_856))
        );
        org.add_mem_block(block("lds1", 1_625_856, 512));
        assert_eq!(
            org.find_first_fitting_memory(Capacity(512)),
            Some(Address(1_626_368))
        );
        // ⛔ THE ORGANIZER'S OWN `-1` (`memory.cpp:79-84`) — a request past what is left is None, and
        // NOT address 0.
        assert_eq!(org.find_first_fitting_memory(Capacity(2_031_616)), None);
    }

    /// 🎯 015/38 THE TWO STRICT COMPARISONS ARE NOT THE SAME TEST. The head test is on an EXTENT
    /// (`reqCap < memOccupied_.begin()->startAddress_`, `memory.cpp:90`) so an exact fill of the
    /// leading gap is refused, while the tail test is on the LAST BYTE OCCUPIED
    /// (`reqEndAddress < capacity_`, `:96`, `:105`) so an exact fill of the tail is taken.
    #[test]
    fn an_exact_fill_of_the_head_gap_is_refused_where_an_exact_fill_of_the_tail_is_taken() {
        let mut head = space(4096);
        head.add_mem_block(block("held", 1024, 1024));
        // 1024 bytes below address 1024: `1024 < 1024` is false, so it lands AFTER the block. With
        // the dead `findAllFittingMemory`'s `<=` this would be 0.
        assert_eq!(
            head.find_first_fitting_memory(Capacity(1024)),
            Some(Address(2048))
        );
        // One byte less does take the head gap.
        assert_eq!(
            head.find_first_fitting_memory(Capacity(1023)),
            Some(Address(0))
        );
        let mut tail = space(2048);
        tail.add_mem_block(block("held", 512, 512));
        // Ends on 2047, the space's last byte: admitted.
        assert_eq!(
            tail.find_first_fitting_memory(Capacity(1024)),
            Some(Address(1024))
        );
        // ⛔ One byte more would end ON `capacity_`, and that is the byte the strict `<` excludes.
        assert_eq!(tail.find_first_fitting_memory(Capacity(1025)), None);
    }

    /// 🎯 016/38 A NAME PRESENT ANSWERS ITS ADDRESS AND A NAME ABSENT ANSWERS THE REFERENCE'S `-1`
    /// (`memory.cpp:161-168`), which is not an address and here cannot be used as one.
    #[test]
    fn an_address_lookup_answers_the_block_it_finds_and_none_for_a_name_never_held() {
        let mut org = space(2_031_616);
        org.add_mem_block(block("reserved-frontend", 0, 1_625_344));
        org.add_mem_block(block("lds0", 1_625_344, 512));
        assert_eq!(
            org.find_address_by_name(&named("reserved-frontend")),
            Some(Address(0))
        );
        assert_eq!(
            org.find_address_by_name(&named("lds0")),
            Some(Address(1_625_344))
        );
        assert_eq!(org.find_address_by_name(&named("absent")), None);
    }

    /// 🎯 024/38 THE CONSTRUCTOR SEEDS `freeCapacity_`, AND WITHOUT THAT SEED NOTHING IS EVER PLACED —
    /// `capacity_(cap)` then `clear()` (`memory.cpp:21`), over LX's own 2,031,616 bytes
    /// (`sys-arch-spec/sysdef.cpp:211`).
    #[test]
    fn a_new_space_is_wholly_free_and_first_fits_from_address_zero() {
        let org = MemoryOrganizer::new(Capacity(2_031_616));
        assert_eq!(org.capacity, Capacity(2_031_616));
        assert_eq!(org.free_capacity, Capacity(2_031_616));
        assert!(org.occupied.is_empty());
        assert_eq!(
            org.find_first_fitting_memory(Capacity(512)),
            Some(Address::ZERO)
        );
        // ⛔ THE CONSEQUENCE OF THE SEED: with `freeCapacity_` at 0 the organizer answers its own −1
        // to every request (`memory.cpp:79-84`), which is what a `Default` would have built.
        assert_eq!(
            MemoryOrganizer::new(Capacity(0)).find_first_fitting_memory(Capacity(512)),
            None
        );
    }

    /// 🎯 025/38 THE COMMIT REPRODUCES THE CORPUS'S OWN ADDRESSES: pin `[0, 1625344)` for
    /// `reserved-frontend` (`ProgramLayout.cpp:93`), after which 512-byte first fits land at
    /// 1,625,344, then 1,625,856, then 1,626,368 — the reference's own recorded LX addresses.
    #[test]
    fn allocation_pins_the_front_end_and_then_first_fits_the_recorded_lx_addresses() {
        let mut org = space(2_031_616);
        assert_eq!(
            org.allocate_memory(
                Capacity(1_625_344),
                &named("reserved-frontend"),
                Placement::At(Address::ZERO),
            ),
            Some(Address::ZERO)
        );
        assert_eq!(org.free_capacity, Capacity(406_272));

        let placed: Vec<Option<Address>> = ["lds0", "lds1", "lds2"]
            .into_iter()
            .map(|name| org.allocate_memory(Capacity(512), &named(name), Placement::FirstFit))
            .collect();
        assert_eq!(
            placed,
            vec![
                Some(Address(1_625_344)),
                Some(Address(1_625_856)),
                Some(Address(1_626_368)),
            ]
        );
        assert_eq!(org.free_capacity, Capacity(404_736));

        // ⛔ THE ORGANIZER'S OWN −1 (`memory.cpp:141-143`), and it commits NOTHING: the block list and
        // `freeCapacity_` are both where they were.
        assert_eq!(
            org.allocate_memory(Capacity(2_031_616), &named("too-big"), Placement::FirstFit),
            None
        );
        assert_eq!(starts(&org).len(), 4);
        assert_eq!(org.free_capacity, Capacity(404_736));

        // ⛔ THE PINNED ARM IS TAKEN ON TRUST: an address already occupied is accepted, the list holds
        // two blocks at 0, and `freeCapacity_` falls again.
        assert_eq!(
            org.allocate_memory(
                Capacity(512),
                &named("overlaps-the-front-end"),
                Placement::At(Address::ZERO),
            ),
            Some(Address::ZERO)
        );
        assert_eq!(starts(&org).len(), 5);
        assert_eq!(org.free_capacity, Capacity(404_224));
    }

    /// 🎯 026/38 THE GAPS ARE INCLUSIVE `[first, last]` PAIRS, `capacity_ - 1` is the last legal end,
    /// and an empty space is the single pair `(0, capacity_ - 1)` (`memory.cpp:217-235`).
    #[test]
    fn the_free_list_is_inclusive_pairs_and_touching_blocks_leave_no_pair_between_them() {
        let mut org = space(2_031_616);
        assert_eq!(
            org.all_free_blocks(),
            vec![FreeBlock {
                first: Address(0),
                last: Address(2_031_615),
            }]
        );
        assert_eq!(org.all_free_blocks()[0].extent(), Capacity(2_031_616));

        // ⛔ THE FRAGMENTED FREE LIST THE 187-PROGRAM CORPUS NEVER PRODUCES — the 512-byte hole left
        // between the reservation and a block placed one slot above it, then the whole tail.
        org.add_mem_block(block("reserved-frontend", 0, 1_625_344));
        org.add_mem_block(block("lds1", 1_625_856, 512));
        assert_eq!(
            org.all_free_blocks(),
            vec![
                FreeBlock {
                    first: Address(1_625_344),
                    last: Address(1_625_855),
                },
                FreeBlock {
                    first: Address(1_626_368),
                    last: Address(2_031_615),
                },
            ]
        );
        assert_eq!(org.all_free_blocks()[0].extent(), Capacity(512));

        // ⛔ `entry.startAddress_ - 1 > prevEndAddr` (`:225`) IS STRICT: touching blocks leave no
        // zero-length pair, and one free byte leaves a pair whose `first == last`.
        let mut packed = space(2048);
        packed.add_mem_block(block("a", 0, 512));
        packed.add_mem_block(block("b", 512, 512));
        packed.add_mem_block(block("c", 1025, 1023));
        assert_eq!(
            packed.all_free_blocks(),
            vec![FreeBlock {
                first: Address(1024),
                last: Address(1024),
            }]
        );
        assert_eq!(packed.all_free_blocks()[0].extent(), Capacity(1));
    }
}
