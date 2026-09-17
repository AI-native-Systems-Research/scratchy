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
//! that does not gets a RECORDED REFUSAL naming the fact rather than a plausible byte count.
//!
//! ⛔ A FABRICATED CAPACITY IS THE ONE THING RANKED WORSE THAN A STOP by this crate's own
//! `CLAUDE.md`: `alloc_all_mem` places every allocation against it and commits, and the
//! resulting program reads memory nothing filled.
//!
//! ⭐⭐ `P` NO LONGER REFUSES THE CAPACITY EITHER, AND THAT WAS THE LAST CARRIER REFUSAL ON THE HOT
//! PATH. [`Placement::buffer_capacity_even_sticks`] walks
//! [`crate::schedule::l3::capacity::buffer_capacity`] on the `&DesignSpaceConfig` the reference calls
//! `getBufferCapacityForNode` **on**, so `alloc_all_mem` commits and `DscState::refusals()` is EMPTY
//! for a program that reaches it. Its [`Option`] is still an [`Option`] and NOT a `todo!`: the stops
//! left inside the ported walk are on the hot path of all 24,363 programs, and a panic there unwinds
//! the [`DscState`] the whole node census is read off — `spyre`'s `run_stages` catches it and reports
//! `Ran::Stopped`, which carries no artifacts, so a panicking frontier costs the measurement too.
//!
//! ⭐⭐ `M` IS NO LONGER ONE OF THEM. [`Trackers`] owns a real [`MemTrackBundle`] of ported
//! [`DsTrackInMem`] trackers, so every capacity it reports and every address it hands out comes off
//! `initMemTrack`'s own operands and `checkAndAddDs`' own block list — see [`Trackers::at_step`].

#[cfg(test)]
mod lx_oracle;

use std::collections::BTreeMap;
use std::num::{NonZeroI64, NonZeroU64};

use crate::arch::{Arch, Bytes};
use crate::schedule::ddc::fold::{AllocId, NodeId};
use crate::schedule::ddc::transformation::LoopId;
use crate::schedule::ddc::v1;
use crate::schedule::dsc2::{LdsIdx, LoopNode, StartAddress};
use crate::schedule::l3::capacity::{
    AllocSizing, AncestorLoops, BytesForm, DscSizing, SampledBuffer, StickRounding, buffer_capacity,
};
use crate::schedule::l3::dsc::{DesignSpaceConfig, DscIdx};
use crate::schedule::l3::dl_ops::{
    AddressFoldCoords, ExPhase, ExPhaseTrackers, L3DataInfoSink, L3Fill, L3Placement,
    L3TrackerSite, SymbolOp, SymbolOperand, SymbolTable, VariableSymbol, parent_loop_nodes,
};
use crate::schedule::memtrack::bundle::{BundleSite, MemTrackBundle};
use crate::schedule::memtrack::memory::{Address, Capacity};
use crate::schedule::memtrack::tracker::{
    AllocEnd, AllocGranularity, BoundPhase, DsAddress, DsMemInfo, DsTrackInMem, ExPhaseCount,
    Margin, MemName, TrackingMode,
};
use crate::units::{Core, Corelet, Row};

use super::state::DscState;

/// `P` — the design space's placement, which entry 222 sizes and names buffers through.
///
/// ⭐ THE FOLD SPACE IS REAL, READ OFF THE COORDINATES `run` IS ALREADY HANDED:
/// [`AddressFoldCoords::depth`] is `{coreFoldProp_, coreletFoldProp_} ++ sdscFoldProps_`'s size and
/// its own doc says so, and the coordinate count is that same list's length. Two answers from ONE
/// value, so the fold space this places along and the fold space entry 292 walks cannot disagree.
///
/// ⭐⭐ IT HOLDS THE SAME `&'s DscState` [`super::Reads`] AND [`super::Env`] DO, so the allocate node
/// and the loop chain the capacity walk needs come off the very tree the growers minted — the one
/// cell per `memOrg_` entry, not a second map that could disagree with it.
#[derive(Debug, Clone)]
pub struct Placement<'s> {
    state: &'s DscState,
    coords: AddressFoldCoords,
    /// `dscGlobal.sysDef.bytesPerStick` (`L3DlOpsScheduler.cpp:5559`, `:5657`) — [`Arch::BYTES_PER_STICK`].
    bytes_per_stick: NonZeroU64,
}

impl<'s> Placement<'s> {
    /// The placement over the fold manager's own address coordinates — the same value
    /// [`crate::schedule::l3::dl_ops::L3RunInputs::coords`] carries — the state its refusals land in,
    /// and the stick width its LX capacities round to an EVEN count of.
    ///
    /// ⛔⛔ `bytes_per_stick` IS A CONSTRUCTION ARGUMENT AND NOT A TYPE PARAMETER ON THIS CARRIER.
    /// `dscGlobal.sysDef.bytesPerStick` is a member of the scheduler's OWN construction argument
    /// (`L3DlOpsScheduler(dscGlobal, memTrackers, {executionStep}, verbose)`), exactly like the
    /// [`Trackers::at_step`] granularity beside it — and a `Placement<'s, A>` cannot work here:
    /// `l3::dl_ops::run` takes `A` explicitly and `P` freely with nothing tying the two, so the arch
    /// is unconstrained at the call site (E0283) and a defaulted parameter would silently resolve to
    /// one arch's stick width on every target. The caller states the width, so a wrong one is a
    /// visible argument rather than an inferred fabrication.
    #[must_use]
    pub const fn of(
        state: &'s DscState,
        coords: AddressFoldCoords,
        bytes_per_stick: NonZeroU64,
    ) -> Self {
        Self {
            state,
            coords,
            bytes_per_stick,
        }
    }
}

/// THE ENCLOSING LOOPS OF ONE NODE AS `dsc2::LoopNode`s — `node->getOwnerLoop()` applied until the
/// loop that has no parent block, which is [`parent_loop_nodes`] (entry 015), each id resolved to the
/// loop the tree minted at it.
///
/// ⛔ THE TWO `LoopNode`s ARE TWO VOCABULARIES OF ONE C++ TYPE AND THE CONVERSION IS THE SAME ONE
/// [`super::ddc_store2`]'s `sched_node_of` ALREADY MAKES: the tree stores
/// [`crate::schedule::ddc::transformation_util::LoopNode`] (name, `numId_`, `denId_`, `dims_`) and
/// [`AncestorLoops`] reads `dsc2::LoopNode`, whose extra field is `isParametricLoop_` /
/// `parametricLdsIdx_`.
///
/// ⛔ `parametric_lds: None` IS THE AUTHORITY'S OWN MEMBER INITIALIZER AND NOT A DROPPED FACT
/// (`dsc/dsc2.h:617-618`). Exactly two things in the reference ever write it — the `ParametricLoopOp`
/// arm of the DDL conversion (`ddc/ddl/ddl_conversion.cpp:1126-1161`) and the JSON importer reading a
/// SERIALISED super-DSC back (`dsc/dsc2.cpp:1409-1416`) — and neither has run: every loop in this
/// tree was minted in memory by the L3 scheduler's own `createLoopNode`
/// (`L3DlOpsScheduler.cpp:625-645`), which writes `name_`, `numId_`, `denId_` and `dims_` and nothing
/// else. ⭐ THE ANSWER MATTERS: `getSizeDataStageForNode` sizes a parametric loop's dim by
/// `parametricStride` and every other loop's by its `denId_` datastage (`dsc/dsc2.cpp:3652-3668`), so
/// a wrongly parametric loop writes an invented extent straight into the buffer size.
///
/// ⭐ THE BLOCK IS THE REAL SUBTREE, not an empty stand-in: `head_block_of` materialises it, so
/// nothing here states a child list the tree does not have.
fn ancestor_loop_nodes(tree: &super::state::DscTree, node: NodeId) -> Vec<LoopNode> {
    parent_loop_nodes(tree, node)
        .into_iter()
        .filter_map(|at| {
            tree.with(|held| {
                let minted = held.loop_node(at)?;
                Some(LoopNode {
                    // ⭐ `dims_` NEEDS NO CONVERSION: `tu::PrimaryDimAndKind` IS
                    // [`crate::schedule::dsc2::LoopDim`], one Rust type for `dsc/dims.h:76`.
                    dims: minted.dims.iter().collect(),
                    num: Some(minted.num),
                    den: Some(minted.den),
                    ..LoopNode::bare(super::ddc_store2::head_block_of(held, at.0))
                })
            })
        })
        .collect()
}

impl L3Placement for Placement<'_> {
    /// ⛔ NEVER A CONSTANT. `getBufferCapacityForNode(node, lds, comp, corelet, row, bytesPerStick,
    /// forceEvenNumSticks=true)` (`dsc/dsc2.cpp:3977`) walks the allocate node's layout against the
    /// DSC's stick sizes and rounds to an EVEN stick count. It is a `dsc/` seam over the live
    /// super-DSC, and it is [`crate::schedule::l3::capacity::buffer_capacity`].
    /// ⛔⛔⛔ **THE CALLEE IS PORTED. THIS IS A THREADING GAP, AND THE THREE SENTENCES THAT USED TO
    /// STAND HERE SAYING OTHERWISE WERE FALSE** — they would send the next reader to re-port 1,881
    /// lines that already exist. `getBufferCapacityForNode` and its whole closure landed as
    /// [`crate::schedule::l3::capacity`]: `buffer_capacity` (e019), `buffer_capacity_per_dim` (e018),
    /// `buffer_capacity_per_dim_at` (e017, the ~210-line body), `size_data_stage_of_alloc_at_node`
    /// (e016) and `size_data_stage_for_node` (e015), and
    /// [`crate::schedule::l3::capacity::DscSizing`] answers its `SizeDsc` seam off a REAL
    /// [`crate::schedule::l3::dsc::DesignSpaceConfig`] — proved against `sdsc_1`'s own two exported
    /// buffer offsets, 256 and 4096.
    ///
    /// ⭐⭐ **IT ANSWERS.** The five reads it takes to ask are all off state this carrier already
    /// holds, and none of them is a derivation:
    ///
    /// * **`currDsc`** is the ARGUMENT, because the reference's call is a `DesignSpaceConfig` METHOD
    ///   (`currDsc->getBufferCapacityForNode(..)`, `L3DlOpsScheduler.cpp:5560`) and `try_alloc_l3`
    ///   already holds that borrow. It cannot be a FIELD here: a borrow of the super-DSC cannot live
    ///   across `l3::dl_ops::run`'s own `&mut SuperDsc`, and a CLONE would freeze `dataStageParam_` —
    ///   precisely the map `size_data_stage_for_node` reads each enclosing loop's `denId_` out of.
    /// * **the allocate node** is `TreeData::node_of_alloc(alloc)` → `DscTree::placed(node)`, the ONE
    ///   `labeledDs_.at(lds).memOrg_.at(storage).allocateNode_` cell entry 353's mint fills and
    ///   [`crate::schedule::l3::dl_ops::AllocationReads`] reads — so this and `try_alloc_l3`'s own
    ///   `node` cannot be two different nodes.
    /// * **the ancestor loops** are [`ancestor_loop_nodes`], which is entry 015's own
    ///   `parent_loop_nodes` walk, with [`crate::schedule::stages::DscTree::head_den`] as the head's
    ///   `denId_` — `ScheduleTree::head_` is itself a `LoopNode` and is the loop that ends the
    ///   reference's `getPrev() == nullptr` walk (`dsc/dsc2.h:623`, `:629`).
    /// * **`allocateCoordinates_`** is `TreeData::coordinate(node)`. ⭐ ITS ABSENCE IS THE
    ///   DEFAULT-CONSTRUCTED COORDINATE AND NOT A REFUSAL: the reference holds
    ///   `CoordinateType<CoordinateBaseType> allocateCoordinates_` BY VALUE (`dsc/dsc2.h:1007`), so a
    ///   node no coordinate pass has reached carries an empty one whose `foldConstructed()` is false —
    ///   which is exactly what `hasCoordinate` then reads (`dsc/dsc2.cpp:3777-3778`).
    ///   `sliceViewCoordinates_` is [`None`] because it is NEVER SERIALISED (`dsc/dsc2.cpp:1827` is a
    ///   bare `// TO DO: sliceview coordinate`).
    /// * **`bytesPerStick`** is [`Self::of`]'s construction argument, per that constructor's note.
    ///
    /// ⛔ THE COMPONENT IS THE NODE'S OWN `component_` AND NOT AN ARGUMENT: the reference passes
    /// `allocNode->component_` (`:5562`) and `try_alloc_l3` has already proved it is `LX` two
    /// statements above (`:5551`'s *"Expect only LX."*).
    /// ⛔ `corelet` AND `row` GO THROUGH AS THE `0`/`0` THE REFERENCE PASSES, not as `-1`:
    /// [`SampledBuffer::of`] applies the component's own forcing (`:3768-3772`), and doing it here
    /// instead would read `sliceViewCoordinates_` where the reference reads `allocateCoordinates_`.
    /// ⛔ `allowSymbolicVolumeLimit` IS ABSENT FROM [`BytesForm`] BY DESIGN — e019 hands its callee a
    /// HARDCODED `true` (`:3990`), so a caller able to state it would be stating something the
    /// reference overrides.
    /// ⛔ `nonUnifiedAllocInHBM_` (`dsc/dsc2.h:1004`) IS `false` TWICE OVER: no unit of the L3
    /// scheduler writes it (its only writers are `perfDscToSdsc.cpp:1870` and the SDSC parser
    /// `dsc/dsc2.cpp:1805`), and it is read ONLY inside the `component_ == HBM` arm
    /// (`dsc/dsc2.cpp:3626`), which the LX proof above makes unreachable.
    /// ⛔ `DesignSpaceConfig::lx_chunk_capacity` IS NOT THIS CALL: it is the same call already made
    /// for the CHUNK stage at one fixed site, so reusing it would answer a different question with the
    /// same number.
    /// ⛔⛔ AND [`None`] IS STILL NEVER A CAPACITY. It is a stop of the ported walk — a `.at` the
    /// reference throws from, or one of the four arms of
    /// `getBufferCapacityForNodePerDimCustomLocation` still carrying a `todo!` — propagated by
    /// `try_alloc_l3` unchanged. A fabricated byte count would commit a fabricated placement, which
    /// this crate ranks worse than any stop.
    fn buffer_capacity_even_sticks(
        &self,
        dsc: &DesignSpaceConfig,
        dsc_idx: DscIdx,
        alloc: AllocId,
        lds: LdsIdx,
        corelet: Corelet,
        row: Row,
    ) -> Option<Bytes> {
        let tree = self.state.dsc(dsc_idx)?;
        let node = tree.with(|held| held.node_of_alloc(alloc))?;
        // `allocNode` — the ddc view of that one `memOrg_` cell.
        let allocation = tree.placed(node)?;
        // The two `dsc2::AllocateNode` members `AllocSizing` needs that the ddc view has no slot for,
        // read off the L3 mint's own node rather than stated here — see
        // [`crate::schedule::l3::dl_ops::L3AllocateNode::back_gap_dims`].
        let (_, minted) = tree.with(|held| held.allocate(node))?;
        let coordinates = tree.with(|held| held.coordinate(node)).unwrap_or_default();
        let above = ancestor_loop_nodes(tree, node);
        let ancestors = AncestorLoops::of(above.iter().collect(), tree.head_den());
        let sizing = AllocSizing {
            allocate_coordinates: &coordinates,
            slice_view_coordinates: None,
            ignore_symbolic_volume_limits: minted.ignore_symbolic_volume_limits,
            indirect: minted.indirect,
            back_gap_dims: &minted.back_gap_dims,
        };
        let view = DscSizing::of(dsc)?;
        let at = SampledBuffer::of(
            lds,
            view.labeled_ds(lds)?,
            allocation.component,
            Some(corelet),
            Some(row),
        );
        buffer_capacity(
            &allocation,
            sizing,
            false,
            at,
            BytesForm {
                do_not_round: false,
                include_gaps: true,
                rounding: StickRounding::EvenSticks(self.bytes_per_stick),
            },
            &ancestors,
            &view,
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

// ⭐⭐ `Placement` NO LONGER ANSWERS [`v1::StorageNames`], AND THAT IS THE FIX RATHER THAN A GAP.
// The two names were never `P`'s to give: the L3 copy of entry 124 is
// `getLdsOrConstNameOfAllocNode(DesignSpaceConfig *currDsc, dsc2::AllocateNode *anode)`
// (`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5493-5494`) and reads
// `currDsc->labeledDs_.at(anode->ldsIdx_).dsName_` (`:5498`) and
// `currDsc->constantInfo_.at(anode->constIdx_).name_` (`:5500`) off the DSC POINTER
// `allocAllMem(mySDsc, currDsc, dscIdx, commitIfValid)` (`:5509-5511`) was handed — not off a member
// and not off the scheduler's construction arguments. Only the ddc copy reads a `currDsc_` member,
// with a ONE-argument call (`ddc/ddcv1.cpp:20-30`, `:280`, `:336`, `:341`), which is why
// [`v1::StorageNames`] itself keeps its DSC-less signature and [`super::Dsc2Reads`] still implements
// it. `try_alloc_l3` already holds the very `&DesignSpaceConfig` the reference passes, so the L3 side
// asks [`crate::schedule::l3::dl_ops::DscNames`] instead and a name can no longer come from the wrong
// DSC BY CONSTRUCTION — there is no index to get wrong and no `dscs_.first()` to reach for.

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
