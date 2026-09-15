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

use std::collections::BTreeMap;

use crate::arch::Bytes;
use crate::schedule::ddc::fold::{AllocId, ConstIdx, NodeId};
use crate::schedule::ddc::transformation::LoopId;
use crate::schedule::ddc::v1;
use crate::schedule::dsc2::{LdsIdx, StartAddress};
use crate::schedule::l3::dl_ops::{
    AddressFoldCoords, ExPhase, ExPhaseTrackers, L3DataInfoSink, L3Fill, L3Placement,
    L3TrackerSite, SymbolOp, SymbolOperand, SymbolTable, VariableSymbol,
};
use crate::units::{Corelet, Row};

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

/// `M` — `memTrackers`, where entry 222 places each allocation, per execution phase.
#[derive(Debug, Clone, Copy, Default)]
pub struct Trackers {
    /// The one phase this run places into — see [`ExecutionStep`].
    pub step: ExecutionStep,
}

impl ExPhaseTrackers for Trackers {
    /// ⭐ ONE PHASE, WHICH IS WHAT `{executionStep}` IS — a single-element initializer list
    /// (`SchedulerStages.cpp:30`), not a range. See [`ExecutionStep`] for why 0 is the default.
    fn ex_phases(&self) -> Vec<ExPhase> {
        vec![ExPhase(self.step.0)]
    }

    /// ⛔⛔ WANTS `DsTrackInMem`, WHICH IS AN UNPORTED 703-LINE C++ ALLOCATOR — `util/memtracker/
    /// mem_track.{h,cpp}` (107 + 596 lines), OUTSIDE every campaign's file list. `checkAndAddDs`
    /// (`mem_track.cpp:395-415`) rounds the request to `allocGranularity`, asks
    /// `checkDsForStartAddr(ds, capRound, eps, -1, margin, allocFromBack)` for an address and records
    /// it with `addDsAtStartAddr` — with per-exphase free lists, a `margin`, an `allocFromBack`
    /// direction, overlap handling (`checkAndAddDsWithOvl`) and a `strict` mode carrying an
    /// `occupied_` block list. THIS IS A PORT, NOT INTEGRATION WIRING, and it needs its own campaign
    /// unit.
    ///
    /// ⛔⛔ DO NOT HAND-ROLL IT FROM THE FIXTURE. The verified oracle below is THREE consecutive
    /// allocations, and three points fit many laws — none of them exercise `margin`,
    /// `allocFromBack`, the per-exphase free lists, the overlap path or `strict`. Inventing a
    /// placement policy that reproduces three addresses is exactly the fabricated placement this
    /// crate ranks worse than a stop.
    ///
    /// ⭐⭐ THE ORACLE, MEASURED, for whoever ports it — `g0/debug/sdsc_0/sdsc.json`, DSC `rmsq_o728`,
    /// every value uniform across all 32 cores:
    ///
    /// | node | `startAddressCoreCorelet_.data_` | `numBuffers_` | `bufferOffsetCoreCorelet_` |
    /// |---|---|---|---|
    /// | `allocate_lds0_lx` | 1_625_344 | 2 | 256 |
    /// | `allocate_lds1_lx` | 1_625_856 | 2 | 256 |
    /// | `allocate_lds2_lx` | 1_626_368 | 2 | 256 |
    ///
    /// ⛔ AND `bufferOffsetCoreCorelet_` IS NOT THE PLACED ADDRESS — it is the DOUBLE-BUFFER STRIDE,
    /// identical (256) on all three nodes and on all 32 cores, which is why it cannot be the address
    /// of three distinct allocations. The placed address is `startAddressCoreCorelet_.data_`.
    /// Validating a tracker against the buffer offset would pass for one that allocated everything at
    /// 256. The consecutive delta is 512 = `numBuffers` x `bufferOffset`.
    ///
    /// ⭐ THE HBM NODES ARE NOT THE TRACKER'S: `allocate-Tensor{0,1}_hbm` sit at `128 * core` and
    /// `Tensor2_hbm` at `6_610_944 + 128 * core`, which is scratchy's OWN
    /// `startAddressCoreCorelet_` carried through unchanged. Only LX is placed here.
    fn capacity(&self, _at: L3TrackerSite) -> Bytes {
        todo!(
            "ExPhaseTrackers::capacity: wants memCapacity off DsTrackInMem — an UNPORTED 703-line \
             C++ allocator (util/memtracker/mem_track.{{h,cpp}}), outside every campaign's file \
             list. It is a PORT, not wiring; see this method's doc for the measured oracle."
        )
    }

    /// ⛔ Wants `backupEps(exphase)` on the live tracker.
    fn backup(&mut self, _at: L3TrackerSite) {
        todo!("ExPhaseTrackers::backup: wants backupEps(exphase) on ddc::DsTrackInMem")
    }

    /// ⛔ Wants `restoreEps(exphase, backupInfo)` on the live tracker.
    fn restore_all(&mut self) {
        todo!(
            "ExPhaseTrackers::restore_all: wants restoreEps(exphase, backupInfo) on ddc::DsTrackInMem"
        )
    }

    /// ⛔ Wants `removeDs(name, exphases)` on the live tracker.
    fn remove(&mut self, _at: L3TrackerSite, _name: &v1::StorageName) {
        todo!("ExPhaseTrackers::remove: wants removeDs(name, exphases) on ddc::DsTrackInMem")
    }

    /// ⛔ Wants `checkAndAddDs(name, size, {exphase})` — the call that DECIDES the byte offset. This
    /// is the placement authority itself; an invented [`v1::Placed`] is a fabricated address.
    fn check_and_add(
        &mut self,
        _at: L3TrackerSite,
        _phase: ExPhase,
        _name: &v1::StorageName,
        _size: Bytes,
    ) -> Option<v1::Placed> {
        todo!(
            "ExPhaseTrackers::check_and_add: wants checkAndAddDs(name, size, {{exphase}}) on \
             ddc::DsTrackInMem — this call IS the placement authority"
        )
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
