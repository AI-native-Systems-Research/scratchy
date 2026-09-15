// SPDX-License-Identifier: Apache-2.0
//! ⭐ THE FOUR SMALL CARRIERS — `P`, `M`, `K` and `S` in
//! [`crate::schedule::l3::dl_ops::run`]: the design space's placement, the memory trackers, the
//! `DataInfo` sink and the symbol table.
//!
//! ⛔⛔ ALL FOUR ARE CONSTRUCTION ARGUMENTS OF THE REFERENCE SCHEDULER, NOT FACTS OF THE SUPER-DSC.
//! `L3DlOpsScheduler(dscGlobal, memTrackers, {executionStep}, verbose)`
//! (`dbo/src/Utils/sdsc_bundle/SchedulerStages.cpp:29`) is handed `dscGlobal` — the design space
//! config and the system definition — and `memTrackers` — `ddc::DsTrackInMem`, the LX allocator's own
//! book. Neither is serialised into an SDSC, so a caller that has them fills these in and a caller
//! that does not gets a `todo!` NAMING the fact rather than a plausible byte count.
//!
//! ⛔ A FABRICATED CAPACITY IS THE ONE THING RANKED WORSE THAN A STOP by this crate's own
//! `CLAUDE.md`: `alloc_all_mem` would place every allocation against it and commit, and the
//! resulting program reads memory nothing filled.
//!
//! ⭐⭐ `M` IS NO LONGER ONE OF THEM. [`Trackers`] owns a real [`MemTrackBundle`] of ported
//! [`DsTrackInMem`] trackers, so every capacity it reports and every address it hands out comes off
//! `initMemTrack`'s own operands and `checkAndAddDs`' own block list — see [`Trackers::at_step`].

#[cfg(test)]
mod lx_oracle;

use std::collections::BTreeMap;
use std::num::NonZeroI64;

use crate::arch::{Arch, Bytes};
use crate::schedule::ddc::fold::{AllocId, ConstIdx, NodeId};
use crate::schedule::ddc::transformation::LoopId;
use crate::schedule::ddc::v1;
use crate::schedule::dsc2::{LdsIdx, StartAddress};
use crate::schedule::l3::dl_ops::{
    AddressFoldCoords, ExPhase, ExPhaseTrackers, L3DataInfoSink, L3Fill, L3Placement,
    L3TrackerSite, SymbolOp, SymbolOperand, SymbolTable, VariableSymbol,
};
use crate::schedule::memtrack::bundle::{BundleSite, MemTrackBundle};
use crate::schedule::memtrack::memory::{Address, Capacity};
use crate::schedule::memtrack::tracker::{
    AllocEnd, AllocGranularity, BoundPhase, DsAddress, DsMemInfo, DsTrackInMem, ExPhaseCount,
    Margin, MemName, TrackingMode,
};
use crate::units::{Core, Corelet, Row};

/// `P` — the design space's placement, which entry 222 sizes and names buffers through.
///
/// ⭐ THE FOLD SPACE IS REAL, READ OFF THE COORDINATES `run` IS ALREADY HANDED:
/// [`AddressFoldCoords::depth`] is `{coreFoldProp_, coreletFoldProp_} ++ sdscFoldProps_`'s size and
/// its own doc says so, and the coordinate count is that same list's length. Two answers from ONE
/// value, so the fold space this places along and the fold space entry 292 walks cannot disagree.
#[derive(Debug, Clone)]
pub struct Placement {
    coords: AddressFoldCoords,
}

impl Placement {
    /// The placement over the fold manager's own address coordinates — the same value
    /// [`crate::schedule::l3::dl_ops::L3RunInputs::coords`] carries.
    #[must_use]
    pub const fn of(coords: AddressFoldCoords) -> Self {
        Self { coords }
    }
}

impl L3Placement for Placement {
    /// ⛔ NEVER A CONSTANT. `getBufferCapacityForNode(node, lds, comp, corelet, row, bytesPerStick,
    /// forceEvenNumSticks=true)` (`dsc/dsc2.cpp:3977`) walks the allocate node's layout against the
    /// DSC's stick sizes and rounds to an EVEN stick count. It is a `dsc/` seam over the live
    /// super-DSC; `DesignSpaceConfig::lx_chunk_capacity` is the SAME call already made for the CHUNK
    /// stage, and reusing it for an arbitrary `(alloc, lds, corelet, row)` would answer a different
    /// question with the same number.
    fn buffer_capacity_even_sticks(
        &self,
        _alloc: AllocId,
        _lds: LdsIdx,
        _corelet: Corelet,
        _row: Row,
    ) -> Bytes {
        todo!(
            "L3Placement::buffer_capacity_even_sticks: wants \
             DesignSpaceConfig::getBufferCapacityForNode (dsc/dsc2.cpp:3977) over the live \
             super-DSC's stick sizes — a fabricated capacity would commit a fabricated placement"
        )
    }

    /// `{coreFoldProp_, coreletFoldProp_} ++ sdscFoldProps_`'s size, which is
    /// [`AddressFoldCoords::depth`] — the two struck axes plus each tail axis.
    fn address_fold_depth(&self) -> usize {
        self.coords.depth()
    }

    /// `getFlattenedCoordinates({{0, 0}, {1, 0}}).size()` — how many coordinates one
    /// `(core, corelet)` spreads its address list over, which is how many tails there are.
    fn address_fold_coords(&self) -> usize {
        self.coords.tails().count()
    }
}

impl v1::StorageNames for Placement {
    /// ⛔⛔ THE FIELD EXISTS NOW AND THE CARRIER STILL CANNOT SAY WHICH DSC'S. `dsName_` is
    /// [`crate::schedule::l3::dsc::LdsRecord::name`] and `name_` is
    /// [`crate::schedule::l3::dsc::ConstantInfo::name`] — [`super::Dsc2Reads`] answers both off them.
    /// What blocks it HERE is the SHAPE of the stage-2a carrier, not the projection:
    ///
    ///   * [`v1::StorageNames`] takes an [`LdsIdx`] and NO [`crate::schedule::l3::dsc::DscIdx`],
    ///     because the reference reads it off `currDsc_` — a member the scheduler re-points as it goes.
    ///   * `P` is ONE object for the WHOLE run: [`super::run_l3`] builds a single [`Placement`] and
    ///     [`crate::schedule::l3::dl_ops::run`] then loops `for dsc_idx in dsc_indices(sdsc)`
    ///     (`l3/dl_ops.rs:20712`) with that same `&P`.
    ///
    /// So an `LdsIdx` reaching this method names a position in SOME DSC's `labeledDs_` and the carrier
    /// cannot tell which. ⛔ ANSWERING OFF `dscs_.first()` WOULD BE THE FABRICATION: every name here
    /// goes STRAIGHT TO THE MEMORY TRACKER as a DS key (`ddc/ddcv1.cpp:280`, `:336`, `:341`), so a
    /// two-DSC super-DSC would place two different tensors against one tracker entry — and it would
    /// compile, because all 187 programs of `g0/` have exactly one DSC. Threading `currDsc` into this
    /// trait is the cross-entry work review 382 names, the same cut as [`super::Reads`]/[`super::Env`].
    fn lds_name(&self, _lds: LdsIdx) -> v1::StorageName {
        todo!(
            "v1::StorageNames::lds_name: wants currDsc_->labeledDs_.at(lds).dsName_ — the FIELD is \
             now l3::dsc::LdsRecord::name, but v1::StorageNames takes no DscIdx and `P` is ONE \
             carrier for a run that loops every DSC (l3/dl_ops.rs:20712), so this cannot say WHICH \
             DSC's labeledDs_ the index names; the name is a memory-tracker DS key"
        )
    }

    /// ⛔ THE SAME CUT — `constantInfo_.at(constant).name_` is
    /// [`crate::schedule::l3::dsc::ConstantInfo::name`] now, and `constantInfo_` is a PER-DSC table.
    fn constant_name(&self, _constant: ConstIdx) -> v1::StorageName {
        todo!(
            "v1::StorageNames::constant_name: wants currDsc_->constantInfo_.at(constant).name_ — the \
             FIELD is now l3::dsc::ConstantInfo::name, but constantInfo_ is PER DSC and \
             v1::StorageNames takes no DscIdx; see lds_name"
        )
    }
}

/// THE EXECUTION PHASE A PROGRAM IS ACCOUNTED TO — `runDdc`'s `int executionStep`
/// (`dbo/src/Utils/sdsc_bundle/SchedulerStages.cpp:29-30`), which it hands the scheduler as the
/// ONE-element list `{executionStep}`.
///
/// ⭐ ZERO IS THE REFERENCE'S OWN DEFAULT AND IT IS DOCUMENTED AS SUCH, not a value chosen here:
/// `dbo::execStepOf` (`dbo/src/ProgramAttrs.h:100-104`) reads the `sbf.exec_step` module attribute
/// and returns `0` when nothing stamped one, because *"a bundle whose SDSCs were not loaded has no
/// tree to take phases from, and one phase is what a single-phase run would have used anyway"*.
/// A scratchy bake is exactly that single-phase run.
///
/// ⛔ A NEWTYPE AND A PARAMETER, NOT A CONSTANT IN A METHOD BODY. It is a construction argument of
/// the stage; the caller states it, so a multi-phase bake cannot silently inherit phase 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExecutionStep(pub u32);

/// WHAT FRACTION OF EACH CORE'S LX A PROGRAM MAY ALLOCATE FROM — `DXP_LX_FRAC_AVAIL`, whose
/// `value_or` default is 0.2 (`dbo/src/Transforms/ProgramLayout.cpp:82-83`).
///
/// ⛔ A CONST INPUT AND NOT AN ENVIRONMENT READ, and an exact RATIONAL where the reference multiplies
/// by a `double`: `const int64_t reserved = lx_tracker.memCapacity * (1 - lx_avail_frac)` (`:92`)
/// truncates. Over LX's own 2,031,616 bytes both spellings give 1,625,292, which
/// [`DsTrackInMem::check_and_add_ds_at_addr`] then rounds UP to 1,625,344 — the address every LX
/// allocation of all 187 reference programs is placed at or above.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LxAvailFraction {
    /// How much of the space the program may allocate from, over [`Self::denominator`].
    numerator: i64,
    /// What that share is out of.
    denominator: NonZeroI64,
}

impl LxAvailFraction {
    /// `dtGetEnv<double>("DXP_LX_FRAC_AVAIL").value_or(0.2)` — one fifth
    /// (`ProgramLayout.cpp:82-83`).
    pub const DEFAULT: Self = Self {
        numerator: 1,
        denominator: NonZeroI64::new(5).expect("five is not zero"),
    };

    /// `memCapacity * (1 - lx_avail_frac)` (`ProgramLayout.cpp:92`) — what the front end holds,
    /// BEFORE the tracker rounds the request up to a stick.
    #[must_use]
    const fn reserved(self, capacity: Capacity) -> Capacity {
        let denominator = self.denominator.get();
        Capacity(capacity.0 * (denominator - self.numerator) / denominator)
    }
}

/// THE DS `reserveFrontendLx` PINS AT ADDRESS 0 — `"reserved-frontend"`
/// (`dbo/src/Transforms/ProgramLayout.cpp:93`).
///
/// ⛔ A TRACKER KEY AND NOT A TENSOR: it names no `labeledDs_` entry, which is why nothing else in
/// this stage can collide with it.
fn reserved_frontend() -> v1::StorageName {
    v1::StorageName("reserved-frontend".to_owned())
}

/// `M` — `memTrackers`, the run's own [`MemTrackBundle`], where entry 222 places each allocation per
/// execution phase.
///
/// ⭐⭐ THE LX FAMILY, WHICH IS THE ONLY ONE THIS STAGE PLACES INTO: entry 222's own
/// `DT_CHECK_MSG(allocNode->component_ == LX, "Expect only LX.")`
/// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5548`) refuses every other component BEFORE it asks
/// for a capacity or an address, and the reference's own 187-program output confirms it — the HBM
/// nodes carry scratchy's addresses through unchanged, and the register-file and L0 nodes are placed
/// by stage 2b.
///
/// ⛔ `initializeMemoryTrackers` (`sys-arch-spec/memtracker/mem_track_bundle.cpp:38`) IS NOT CALLED,
/// AND ONE MISSING OPERAND IS WHY. Its per-corelet loop reads
/// `regInfoPerUnit.at(LXLU).at(RegType::SCALE)` (`:82-84`, one 1024-bit register at
/// `sysdef.cpp:367-368`) and `sys_arch_spec::regfile::RegType` HAS NO `SCALE` ARM — the twelve
/// (component, file) tables are exhaustive `match`es over thirteen variants, so adding it is a
/// vendored-table change and not this wiring's. So [`Trackers::at_step`] seeds the LX family exactly
/// as that unit seeds it (`:57-63`) and a site in any other family is a `todo!` naming this gap.
#[derive(Debug, Clone)]
pub struct Trackers {
    /// The one phase this run places into — see [`ExecutionStep`].
    step: ExecutionStep,
    /// `memTrackers` — the bundle itself.
    bundle: MemTrackBundle<DsTrackInMem>,
    /// `trackerBackups` (`L3DlOpsScheduler.cpp:5518-5519`) — one `backupEps` snapshot per phase, per
    /// site, taken before a trial placement and replayed by [`ExPhaseTrackers::restore_all`].
    backups: BTreeMap<BundleSite, Vec<Vec<DsMemInfo>>>,
}

impl Trackers {
    /// ⭐⭐ THE BUNDLE ENTRY 222 PLACES INTO, BUILT THE WAY THE REFERENCE PIPELINE BUILDS IT: one LX
    /// tracker per core over `lxCapacity` bytes at a granularity of `bytesPerStick`
    /// (`mem_track_bundle.cpp:57-63`), grown to hold `step`'s phase
    /// (`dbo/src/Transforms/sdsc_bundle/MemTrackerInit.cpp:92`), with the front end's share of every
    /// core's LX pinned at address 0 across every phase (`:99`, `ProgramLayout.cpp:81-98`).
    ///
    /// ⛔ EVERY NUMBER IS AN `Arch` CONSTANT AND NONE IS A LITERAL HERE: [`Arch::LX_CAPACITY`] is
    /// `lxCap - 64*1024` (`sysdef.cpp:211`) and [`Arch::BYTES_PER_STICK`] is `bytesPerStick`
    /// (`sysdef.cpp:206`). The capacity a caller reads back is the tracker's own `memCapacity`.
    ///
    /// ⭐ THE PHASE COUNT IS `step + 1`, WHICH PLACES IDENTICALLY TO THE WHOLE BUNDLE'S `numSteps`:
    /// each phase owns its own `free_`, `dsInMem_` and block list (`mem_track.cpp:117`), the reserve
    /// covers `0..numSteps` uniformly, and entry 222 places into `{executionStep}` alone — so a
    /// bundle with more phases answers the same addresses for this program. A caller that holds the
    /// bundle's real `numSteps` states it through [`ExecutionStep`] of the LAST program instead.
    #[must_use]
    pub fn at_step<A: Arch>(step: ExecutionStep) -> Self {
        // `numSteps` (`MemTrackerInit.cpp:85`) — a phase count, not a phase index.
        let count = ExPhaseCount(
            i32::try_from(step.0)
                .expect("a bundle holds fewer SDSC nodes than i32::MAX")
                .saturating_add(1),
        );
        // `std::iota(phases.begin(), phases.end(), 0)` (`ProgramLayout.cpp:89-90`).
        let phases: Vec<ExPhase> = (0..count.0)
            .filter_map(|phase| u32::try_from(phase).ok())
            .map(ExPhase)
            .collect();
        let capacity = Capacity(
            i64::try_from(A::LX_CAPACITY.0).expect("lxCapacity is 2 MiB, far inside an i64"),
        );
        let granularity = AllocGranularity(
            NonZeroI64::new(
                i64::try_from(A::BYTES_PER_STICK.get())
                    .expect("bytesPerStick is 128, far inside an i64"),
            )
            .expect("a stick is not zero bytes"),
        );
        let mut bundle = MemTrackBundle::<DsTrackInMem>::default();
        for core in (0..).map_while(Core::checked) {
            let tracker = bundle.lx_track_per_core.entry(core).or_default();
            tracker.init_mem_track(
                MemName(format!("lxCore{}", core.get())),
                capacity,
                count,
                granularity,
                // `isStrict` defaults to `true` (`mem_track.h:57-58`): this tracker ASSIGNS
                // addresses rather than only counting bytes.
                TrackingMode::AddressAssignment,
            );
            // The explicit second format the reference writes after each `initMemTrack`
            // (`mem_track_bundle.cpp:62`) — a no-op, because unit e032 already calls unit e028.
            tracker.format_mem_track();
            // ⭐ THE BASE EVERY LX ADDRESS OF THE CORPUS IS MEASURED FROM. `reserveFrontendLx`
            // returns an ERROR STRING the pass turns into a `signalPassFailure` when this refuses
            // (`ProgramLayout.cpp:93-97`); it cannot refuse here, because `reserved` is
            // `memCapacity * 4/5` of a tracker that holds nothing else.
            let _reserved = tracker.check_and_add_ds_at_addr(
                &reserved_frontend(),
                LxAvailFraction::DEFAULT.reserved(capacity),
                &phases,
                Some(Address::ZERO),
                Margin::NONE,
            );
        }
        Self {
            step,
            bundle,
            backups: BTreeMap::new(),
        }
    }

    /// `memTrackers->getTracker(comp, core, corelet, row)` (`mem_track_bundle.cpp:174`) reduced to
    /// the coordinates the component's own arm reads.
    ///
    /// ⛔ THE `todo!` IS THE ONE GAP THIS CARRIER HAS LEFT, and it is not a placement: no site
    /// outside the LX family is reachable before entry 222's *"Expect only LX."* — see [`Trackers`].
    fn site(at: L3TrackerSite) -> BundleSite {
        match BundleSite::of(at.memory, at.core, at.corelet, at.row) {
            Some(site @ BundleSite::Lx(_)) => site,
            Some(_) | None => todo!(
                "ExPhaseTrackers: a {:?} site wants MemTrackBundle::initializeMemoryTrackers \
                 (mem_track_bundle.cpp:38), whose per-corelet loop reads \
                 regInfoPerUnit.at(LXLU).at(SCALE) — sys_arch_spec::regfile::RegType has no SCALE \
                 arm, so this bundle holds the LX family only",
                at.memory
            ),
        }
    }
}

impl ExPhaseTrackers for Trackers {
    /// ⭐ ONE PHASE, WHICH IS WHAT `{executionStep}` IS — a single-element initializer list
    /// (`SchedulerStages.cpp:30`), not a range. See [`ExecutionStep`] for why 0 is the default.
    fn ex_phases(&self) -> Vec<ExPhase> {
        vec![ExPhase(self.step.0)]
    }

    /// `myTracker->memCapacity` (`L3DlOpsScheduler.cpp:5617`) — the WHOLE space of this site, which
    /// entry 222 widens a streaming buffer's request to.
    ///
    /// ⛔ THE TRACKER'S OWN FIELD, NOT A CONSTANT HERE: it is what `initMemTrack` was handed
    /// (`mem_track.cpp:109`), so a caller cannot read a capacity the trackers were not built with.
    /// ⛔ NON-NEGATIVE BY CONSTRUCTION — [`Trackers::at_step`] sets it from [`Arch::LX_CAPACITY`].
    /// ⚠️ NOT EXERCISED BY THE CORPUS: all 588 LX allocations of the 187 reference programs carry
    /// `numBuffers_` ∈ {1, 2}, and only `-1` (streaming) reaches this call.
    fn capacity(&self, at: L3TrackerSite) -> Bytes {
        Bytes(
            self.bundle
                .tracker(Self::site(at))
                .mem_capacity
                .0
                .unsigned_abs(),
        )
    }

    /// `backupEps(exphase)` for every phase, once per site — `trackerBackups.try_emplace(myTracker)`
    /// then a `backupEps` per phase (`L3DlOpsScheduler.cpp:5539-5542`).
    ///
    /// ⛔⛔ THE SNAPSHOT IS RE-TAKEN, AND `try_emplace` IS WHY THAT IS THE SAME BEHAVIOUR:
    /// `trackerBackups` is a LOCAL of `allocAllMem` (`:5518-5519`), so its idempotence is scoped to
    /// ONE call — and within one call a site is visited at most once, because the `(comp, core)`
    /// pairs it loops over are distinct (`:5521-5537`) and no two components share a tracker family.
    /// Across calls re-taking it is REQUIRED: a snapshot that survived a committing call would let a
    /// later failing call's `restoreEps` roll a committed placement back.
    fn backup(&mut self, at: L3TrackerSite) {
        let site = Self::site(at);
        let tracker = self.bundle.tracker(site);
        let snapshot: Vec<Vec<DsMemInfo>> = self
            .ex_phases()
            .iter()
            .map(|phase| tracker.backup_eps(*phase))
            .collect();
        self.backups.insert(site, snapshot);
    }

    /// `restoreEps(exphases.at(i), backupInfo.at(i))` for every tracker backed up since
    /// (`L3DlOpsScheduler.cpp:5739-5742`) — the exact replay that leaves a failed trial placement
    /// with no trace.
    ///
    /// ⛔ IT ENDS THE TRANSACTION, which is what dropping `trackerBackups` at the close of
    /// `allocAllMem` does: a snapshot replayed twice would undo whatever was placed in between.
    fn restore_all(&mut self) {
        let phases = self.ex_phases();
        for (site, snapshot) in std::mem::take(&mut self.backups) {
            let tracker = self.bundle.tracker_mut(site);
            for (phase, info) in phases.iter().zip(&snapshot) {
                tracker.restore_eps(*phase, info);
            }
        }
    }

    /// `myTracker->removeDs(name, exphases)` (`L3DlOpsScheduler.cpp:5608-5610`) — every candidate is
    /// removed before any is placed, so a retry cannot collide with its own previous attempt.
    ///
    /// ⛔ A PHASE THIS TRACKER DOES NOT HOLD IS DROPPED HERE where the reference's
    /// `epsToListIter.at(eps)` throws (`mem_track.cpp:442`): [`BoundPhase`] is mintable only off a
    /// bound phase and `removeDs` returns `void`, so there is nowhere to put that throw. Unreachable
    /// from [`Trackers::at_step`], which binds every phase up to [`ExecutionStep`]'s own.
    fn remove(&mut self, at: L3TrackerSite, name: &v1::StorageName) {
        let phases = self.ex_phases();
        let tracker = self.bundle.tracker_mut(Self::site(at));
        let bound: Vec<BoundPhase> = phases
            .iter()
            .filter_map(|phase| tracker.bound_phase(*phase))
            .collect();
        tracker.remove_ds(name, &bound);
    }

    /// ⭐⭐ `myTracker->checkAndAddDs(name, mySize, {exphase})` (`L3DlOpsScheduler.cpp:5629-5631`) —
    /// THE CALL THAT DECIDES THE BYTE OFFSET, and it commits: the `dsInMem_` entry, the `free_` debit
    /// and the block are one act (`mem_track.cpp:377-393`).
    ///
    /// ⛔ THE THREE ANSWERS ARE THREE ARMS. `EXISTS` is [`None`] — the reference's
    /// `DT_CHECK(addr != EXISTS)` (`:5632`), a name already in the tracker meaning this set is being
    /// placed twice over itself; `DOESNT_FIT` is [`v1::Placed::DoesntFit`], its `return false`; an
    /// address is [`v1::Placed::At`]. [`DsAddress::Incoherant`] and [`DsAddress::Unplaced`] are
    /// unreachable from unit e037 (unit e034 answers only those three) and are the same `DT_CHECK`.
    /// ⛔ `margin` IS 0 AND THE END IS THE FRONT — both are `checkAndAddDs`' own defaults
    /// (`mem_track.h:81-82`), and the call site takes them.
    fn check_and_add(
        &mut self,
        at: L3TrackerSite,
        phase: ExPhase,
        name: &v1::StorageName,
        size: Bytes,
    ) -> Option<v1::Placed> {
        // `int64_t cap` — [`None`] is the `DT_CHECK` a request no `int64_t` can carry would take.
        let cap = Capacity(i64::try_from(size.0).ok()?);
        let placed = self.bundle.tracker_mut(Self::site(at)).check_and_add_ds(
            name,
            cap,
            &[phase],
            Margin::NONE,
            AllocEnd::Front,
        );
        match placed {
            // `DT_CHECK(startAddr >= 0)` (`mem_track.cpp:411`) is this conversion.
            DsAddress::At(address) => Some(v1::Placed::At(Bytes(u64::try_from(address.0).ok()?))),
            DsAddress::DoesntFit => Some(v1::Placed::DoesntFit),
            DsAddress::Exists | DsAddress::Incoherant | DsAddress::Unplaced => None,
        }
    }
}

/// `K` — where entry 333's `DataInfo` fills land, kept by the operand they landed on.
#[derive(Debug, Default)]
pub struct Sink {
    fills: BTreeMap<v1::OperandSite, L3Fill>,
    last_fusable_src: BTreeMap<NodeId, Option<LoopId>>,
    last_fusable_dsts: BTreeMap<NodeId, Vec<Option<LoopId>>>,
}

impl Sink {
    /// Every fill entry 333 wrote, by the operand it landed on.
    #[must_use]
    pub const fn fills(&self) -> &BTreeMap<v1::OperandSite, L3Fill> {
        &self.fills
    }
}

impl L3DataInfoSink for Sink {
    fn fill(&mut self, at: v1::OperandSite, filled: L3Fill) -> Option<()> {
        self.fills.insert(at, filled);
        Some(())
    }

    fn set_last_fusable_src(&mut self, node: NodeId, at: Option<LoopId>) -> Option<()> {
        self.last_fusable_src.insert(node, at);
        Some(())
    }

    fn set_last_fusable_dsts(&mut self, node: NodeId, at: Vec<Option<LoopId>>) -> Option<()> {
        self.last_fusable_dsts.insert(node, at);
        Some(())
    }
}

/// `S` — `dimToSymbolMapping_`'s table, which entry 333 defines variables in.
///
/// ⭐ EVERY ADDRESS THIS CALLER PLACES IS A BYTE COUNT, so the reference's symbolic arm is never
/// entered: [`v1::Symbols::divide_symbols`] is the identity the reference performs on a
/// non-symbolic address, which [`StartAddress::divided_by`] already is.
#[derive(Debug, Clone, Copy, Default)]
pub struct Symbols {
    next: u64,
}

impl v1::Symbols for Symbols {
    fn divide_symbols(&mut self, address: &StartAddress, by: std::num::NonZeroU64) -> StartAddress {
        address.divided_by(by)
    }
}

impl SymbolTable for Symbols {
    /// `addVar(op, operands)` — the symbol the definition is filed under, one per definition.
    fn add_var(&mut self, _op: SymbolOp, _operands: &[SymbolOperand]) -> VariableSymbol {
        self.next = self.next.saturating_add(1);
        VariableSymbol(self.next)
    }
}
