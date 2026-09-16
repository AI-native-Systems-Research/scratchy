// SPDX-License-Identifier: Apache-2.0
//! ⭐ THE REST OF [`super::Dsc2Store`]'S THIRTY-FIVE SUPERTRAITS — the transformation surface.
//!
//! ⛔ SPLIT ACROSS TWO FILES ONLY FOR LENGTH; every impl here is on the SAME [`super::Dsc2Store`] and
//! reads the SAME state, so nothing here can hold a second view of `currDsc`.
//!
//! ⭐ WHAT IS ANSWERED HERE reads the schedule tree or the labelled DS list: the transfer walks and
//! their read-modify-write updates, the condition regions, the sync units under a loop, the corelet
//! and core sets, the layout scales, the fresh allocation identity.
//!
//! ⛔ WHAT IS NOT falls into the same three buckets [`super::ddc_store`]'s header names — no COMPUTE
//! arm, seven `l3::dsc`-dropped fields, and the `dsc/`/`util/foldManager/` seams — plus one more that
//! is worth stating on its own: **[`crate::generated::DataConnect`] IS A CLOSED GENERATED SET**
//! censused from `ddl_templates/*.ddl` at build time, so [`tu::MintedConnects::intern_connect`]
//! cannot mint a connect the templates did not already name.

use std::collections::{BTreeMap, BTreeSet};

use sys_arch_spec::arch_enums::SenComponent;

use crate::arch::Elements;
use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{
    PrimaryDim, ScheduleNode, StickDims,
};
use crate::formats::DataFormat;
use crate::generated::DataConnect;
use crate::schedule::ddc::fold::{AllocId, NodeId, NodeKind};
use crate::schedule::ddc::transformation as tr;
use crate::schedule::ddc::transformation_util as tu;
use crate::schedule::ddc::v1;
use crate::schedule::dsc2::{
    BlockNode, ComputeNode, DataInfo, LatchDataId, LdsIdx, NodeName, Operand, OperandPos, SyncUnits,
    TransferNode, WordLength,
};
use crate::units::{Core, Corelet};

use super::ddc_state;
use super::ddc_store::Dsc2Store;

// ════════════════════════════════════════════════════════════════════════════════════════════════
// ⭐⭐ THE TWO LOOKUPS SEVERAL IMPLS BELOW SHARE — reached through [`Dsc2Store::with_tree`] and the
// already-answered [`tu::ComponentAllocations`], so this file holds NO second view of `currDsc`.
// ════════════════════════════════════════════════════════════════════════════════════════════════

impl Dsc2Store<'_, '_> {
    /// `static_cast<dsc2::ComputeNode *>(node)` — the COMPUTE this node IS, [`None`] for any other
    /// `nodeType_`.
    ///
    /// ⭐⭐ THE ARM IS THERE NOW. [`super::tree::Kind::Compute`] holds the whole
    /// [`crate::schedule::dsc2::ComputeNode`] (landed in `c1f5c63fa`, filled by the DDL expansion's
    /// `conv::ScheduleWrites::add_compute`/`mint_compute` from `5ee670017` onward), so every reader
    /// below that used to refuse for *"no Compute arm"* and asks only for fields
    /// [`crate::schedule::dsc2::ComputeNode`] carries is answered from the live tree.
    fn compute_of(&self, node: NodeId) -> Option<ComputeNode> {
        self.with_tree(|tree| match tree.kind_of(node) {
            Some(super::tree::Kind::Compute(held)) => Some(held.clone()),
            _ => None,
        })
    }

    /// `getAllocation(di, storage, /*allowMissingAlloc=*/true)` (`dsc/dsc2.cpp:2586-2631`) — ⭐
    /// ANSWERED ON BOTH ARMS, and each `nullptr` path below is one of its four.
    ///
    /// ⛔⛔ THE CONSTANT ARM IS PROJECTED AFTER ALL. Two `todo!`s in this file used to refuse because
    /// *"its constant arm reads `constantInfo_`, which `l3::dsc::DesignSpaceConfig` does not
    /// project"* — it DOES: [`crate::schedule::l3::dsc::ConstantInfo::allocations`] is
    /// `allocations_` as arena handles, and [`tu::DscAllocations::allocation_in`] already answers
    /// the same call through it.
    ///
    /// ⛔ THE LABELLED DS WINS WHERE BOTH INDICES ARE SET, which is the reference's own closing
    /// ternary `di.myLdsIdx_ >= 0 ? labeledDs_... : constantInfo_...` (`:2629-2631`) and NOT an order
    /// chosen here.
    fn allocation_at(
        &self,
        data: DataInfo,
        storage: SenComponent,
    ) -> Option<crate::schedule::ddc::fold::AllocId> {
        // `dsc2::memories.count(storage) == 0` (`:2597-2603`).
        if !tu::is_memory(storage) {
            return None;
        }
        match (data.my_lds_idx, data.constant_id) {
            // `labeledDs_.at(myLdsIdx_).memOrg_.at(storage).allocateNode_` (`:2605-2615`).
            (Some(lds), _) => tu::ComponentAllocations::mem_org_allocation(self, lds, storage),
            // `constantInfo_.at(constantId_).allocations_.at(storage)` (`:2617-2625`).
            (None, Some(constant)) => self.dsc_facts().with_dsc(|dsc| {
                dsc.ddc
                    .constants
                    .get(&constant)?
                    .allocations
                    .get(&storage)
                    .copied()
            }),
            // *"One of myLdsIdx or constantId must be set"* (`:2589-2596`).
            (None, None) => None,
        }
    }

    /// `labeledDs_.at(ldsIdx)` NARROWED TO THE THREE FIELDS ENTRY 126 READS — `dsType_` as stick
    /// dims, `scale_`'s `-2` positions, and `dataFormat_`.
    ///
    /// ⛔ THE `-2` SET IS READ IN LAYOUT ORDER, which is what `is_any_of(-2, lds.scale_)` means:
    /// `scale_` is indexed by a dim's position in layout order (`getDimIndexInLayoutOrder`), so
    /// every entry of it names a layout dim — [`v1::SplatDims`](crate::schedule::ddc::v1::SplatDims)'
    /// own note.
    ///
    /// ⛔ TOTAL, AND BOTH STOPS ARE `.at()`s: an lds this list does not hold, and the
    /// `dataFormat_` of `INVALID` [`v1::TransferLds`](crate::schedule::ddc::v1::TransferLds) has no
    /// spelling for. See [`v1::Dsc2Store::transfer_operands`] for why neither may be an absent lds
    /// instead.
    fn transfer_lds(&self, lds: LdsIdx, transfer: NodeId) -> v1::TransferLds {
        self.dsc_facts().with_dsc(|dsc| {
            let held = dsc.labeled_ds.at(lds).unwrap_or_else(|| {
                panic!(
                    "v1::Dsc2Store::transfer_operands: labeledDs_.at({lds:?}) throws for the end \
                     {transfer:?} names"
                )
            });
            v1::TransferLds {
                stick: ddc_state::stick_dims_of(dsc, lds).unwrap_or_default(),
                splat_dims: crate::schedule::ddc::v1::SplatDims(
                    dsc.layout_dims
                        .get(&lds)
                        .map(|layout| {
                            layout
                                .iter()
                                .filter(|dim| held.scale(*dim) == Some(tr::Scale::StickDim))
                                .collect()
                        })
                        .unwrap_or_default(),
                ),
                format: held.record().data_format.unwrap_or_else(|| {
                    panic!(
                        "v1::Dsc2Store::transfer_operands: labeledDs_.at({lds:?}).dataFormat_ is \
                         DataFormats::INVALID and v1::TransferLds::format is not an Option — see \
                         that method's third stop"
                    )
                }),
            }
        })
    }

    /// `constantInfo_.at(constantId_)` AS A CONSTANT-TO-CONSTANT TRANSFER READS IT —
    /// `data_.getSingleData().size()` and `dataFormat_`.
    ///
    /// ⛔ TOTAL ON THREE COUNTS, all three of them `.at()`/divisor facts the type already carries: a
    /// `constantId_` the table does not hold, an EMPTY `data_` (the element count is the reference's
    /// `numElemInConst` DIVISOR at `ddc/ddcv1.cpp:456-458`, and
    /// [`v1::ConstantData`](crate::schedule::ddc::v1::ConstantData)`::elements` is a [`NonZeroU64`]
    /// *"both divisors non-zero by type"*), and a `dataFormat_` of `INVALID` — whose width is the
    /// reference's OTHER divisor and `dataFormatsToBitWidth.at(INVALID) == -1`.
    fn constant_data(
        &self,
        constant: crate::schedule::ddc::fold::ConstIdx,
        transfer: NodeId,
    ) -> v1::ConstantData {
        self.dsc_facts().with_dsc(|dsc| {
            let held = dsc.ddc.constants.get(&constant).unwrap_or_else(|| {
                panic!(
                    "v1::Dsc2Store::transfer_operands: constantInfo_.at({constant:?}) throws for \
                     the source {transfer:?} names"
                )
            });
            v1::ConstantData {
                elements: std::num::NonZeroU64::new(held.data.len() as u64).unwrap_or_else(|| {
                    panic!(
                        "v1::Dsc2Store::transfer_operands: \
                         constantInfo_.at({constant:?}).data_.getSingleData().size() is nought, and \
                         it is entry 126's replicationFactor_ divisor (ddc/ddcv1.cpp:456-458)"
                    )
                }),
                format: held.data_format.unwrap_or_else(|| {
                    panic!(
                        "v1::Dsc2Store::transfer_operands: \
                         constantInfo_.at({constant:?}).dataFormat_ is DataFormats::INVALID, whose \
                         dataFormatsToBitWidth.at() is -1 and is entry 126's other divisor"
                    )
                }),
            }
        })
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// THE LABELLED DS LIST AS THE TRANSFORMATIONS READ IT — ⭐ ANSWERED WHOLE.
// ════════════════════════════════════════════════════════════════════════════════════════════════

impl tr::LabeledDs for Dsc2Store<'_, '_> {
    /// `labeledDs_.at(lds).scale_`, one entry per layout dim — IN LAYOUT ORDER, which is what the
    /// reference's positional `scale_.at(i)` means.
    fn scale(&self, lds: LdsIdx) -> Vec<tr::Scale> {
        self.dsc_facts().with_dsc(|dsc| {
            let Some(held) = dsc.labeled_ds.at(lds) else {
                return Vec::new();
            };
            dsc.layout_dims
                .get(&lds)
                .map(|layout| layout.iter().filter_map(|dim| held.scale(dim)).collect())
                .unwrap_or_default()
        })
    }

    /// `labeledDs_.at(lds).dsType_` — ⛔ TOTAL, and the reference's `.at()` throws for an index the
    /// list does not hold.
    fn ds_type(&self, lds: LdsIdx) -> tr::DsType {
        self.dsc_facts()
            .with_dsc(|dsc| ddc_state::ds_type_of(dsc, lds))
            .unwrap_or_else(|| {
                panic!("tr::LabeledDs::ds_type: labeledDs_.at({lds:?}) throws for an absent index")
            })
    }
}

impl tr::DsSticks for Dsc2Store<'_, '_> {
    /// `labeledDs_.at(lds).dsType_`'s `stickDimOrder_` zipped with its `stickSize_`.
    fn ds_stick_dims(&self, lds: LdsIdx) -> StickDims {
        self.dsc_facts()
            .with_dsc(|dsc| ddc_state::stick_dims_of(dsc, lds))
            .unwrap_or_default()
    }
}

impl tu::TransferUnrolling for Dsc2Store<'_, '_> {
    /// `currDsc->getNonBroadcastLdsDims(lds)` — ⛔ [`None`] is its own `getLayoutDims` abort.
    fn non_broadcast_lds_dims(&self, lds: LdsIdx) -> Option<Vec<PrimaryDim>> {
        self.dsc_facts()
            .with_dsc(|dsc| dsc.non_broadcast_lds_dims(lds))
    }
}

impl tr::TransferLoads for Dsc2Store<'_, '_> {
    /// `0 .. numCoreletsUsed_DSC2_` as corelets.
    fn corelets(&self) -> Vec<Corelet> {
        self.dsc2_corelets()
    }

    /// ⛔ `getBlockTransferSize(*transferNode, src_.unit_, clId, false, true)` — a
    /// `DesignSpaceConfig` accessor over the transfer's layout and the DSC's stick sizes.
    fn block_transfer_loads(&self, _transfer: NodeId, _corelet: Corelet) -> tr::Loads {
        todo!(
            "tr::TransferLoads::block_transfer_loads: wants \
             getBlockTransferSize(transfer, src_.unit_, clId, false, true) — a DesignSpaceConfig \
             accessor over the live super-DSC's stick sizes, a `dsc/` seam"
        )
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// THE TRANSFER BODY, READ-MODIFY-WRITTEN — ⭐ ANSWERED, because a TRANSFER is a kind this tree holds
// and [`crate::schedule::dsc2::TransferNode`] is its whole body.
// ════════════════════════════════════════════════════════════════════════════════════════════════

impl tr::FixedSizeTransfers for Dsc2Store<'_, '_> {
    /// ⛔ `metadata.dataConnects_.find(dc)->second.consumers_` AS WHAT EACH CONSUMER IS — a
    /// [`tr::ScopeNode`], whose COMPUTE arm this tree has no node for.
    fn connect_consumers(&self, _connect: DataConnect) -> Vec<tr::ScopeNode> {
        todo!(
            "tr::FixedSizeTransfers::connect_consumers: wants each consumer of that data connect AS \
             A ScopeNode — its Compute arm needs a COMPUTE node, which super::tree::Kind has no arm \
             for"
        )
    }

    /// `transferNode->transferSize_.clear()` FOLLOWED BY one entry per stick dim — ⭐ ONE WRITE, so a
    /// cleared-but-unfilled `transferSize_` is unspellable.
    fn set_transfer_size(&mut self, transfer: NodeId, sizes: Vec<(PrimaryDim, Elements)>) {
        self.edit_transfer(transfer, |held| {
            held.transfer_size = sizes.into_iter().collect();
        });
    }
}

impl v1::LoopOffsets for Dsc2Store<'_, '_> {
    /// `0 .. numCoreletsUsed_DSC2_` as corelets.
    fn corelets(&self) -> Vec<Corelet> {
        self.dsc2_corelets()
    }

    /// `traverseTreeDFSMutable(nullptr, {TRANSFER})`.
    fn transfers(&self) -> Vec<NodeId> {
        tr::TransferWalk::transfers(self)
    }

    /// The transfer that node is — ⛔ TOTAL, as the reference's downcast is.
    fn transfer(&self, node: NodeId) -> TransferNode {
        tu::ScheduleSurgery::transfer(self, node)
    }

    /// `traverseTreeDFSMutable(nullptr, {COMPUTE})` — ⭐ EMPTY, and that is a reading: the tree holds
    /// none.
    fn computes(&self) -> Vec<NodeId> {
        tr::ComputeWalk::computes(self)
    }

    /// `static_cast<dsc2::ComputeNode *>(node)`, whose `type_`, `exUnit_` and `inputs_` zipped with
    /// `inputsLdsAndLoopOffsets_` are what entry 260 reads — ⭐ ANSWERED off
    /// [`super::tree::Kind::Compute`].
    ///
    /// ⛔ TOTAL, AS THE REFERENCE'S DOWNCAST IS: a node that is not a COMPUTE is its own undefined
    /// behaviour there, so the stop names the node exactly as
    /// [`tu::ScheduleSurgery::transfer`](crate::schedule::ddc::transformation_util::ScheduleSurgery::transfer)
    /// does for the transfer half.
    fn compute(&self, node: NodeId) -> ComputeNode {
        self.compute_of(node).unwrap_or_else(|| {
            panic!(
                "v1::LoopOffsets::compute: {node:?} is not a COMPUTE of this DSC's scheduleTree_ — \
                 the reference's static_cast there is undefined"
            )
        })
    }

    /// `getStickDims(lds)`.
    fn stick_dims(&self, lds: LdsIdx) -> Vec<PrimaryDim> {
        self.dsc_facts()
            .with_dsc(|dsc| ddc_state::stick_dims_of(dsc, lds))
            .map(|dims| dims.0.iter().map(|(dim, _)| *dim).collect())
            .unwrap_or_default()
    }

    /// `node->getOwnerLoop()`.
    fn owner_loop(&self, node: NodeId) -> Option<tr::LoopId> {
        tu::ScheduleSurgery::owner_loop(self, node)
    }

    /// ⛔⛔ `loopEleOffsets_` IS NOT ON THE NODE IN THIS PORT — AND THAT IS A FINDING, not a gap in
    /// this carrier. `dsc2::DataInfo` (`schedule/dsc2.rs`) projects FOUR fields of
    /// `LdsAndLoopOffsets` — `dataConnect_`, `myLdsIdx_`, `constantId_`, `latchDataId_` — and
    /// `loopEleOffsets_`/`constEleOffsets_` live instead on [`v1::DataInfoFill`]
    /// (`ddc/v1.rs:2121-2135`), which entry 260 WRITES INTO THE SINK rather than onto the transfer.
    /// So the reference's read-back of an offset it wrote earlier has no node field to come from, and
    /// answering EMPTY here would make entry 133's restickify fixup silently find no offsets to
    /// adjust.
    fn src_loop_ele_offsets(
        &self,
        _node: NodeId,
        _corelet: Corelet,
    ) -> Vec<(tr::LoopId, Vec<PrimaryDim>)> {
        todo!(
            "v1::LoopOffsets::src_loop_ele_offsets: wants \
             srcLdsAndLoopOffsets_.loopEleOffsets_.at(corelet) — dsc2::DataInfo projects only \
             dataConnect_/myLdsIdx_/constantId_/latchDataId_, and loopEleOffsets_ lives on \
             v1::DataInfoFill (ddc/v1.rs:2127), which entry 260 writes into the SINK and not onto \
             the transfer node"
        )
    }

    /// ⛔ THE WRITE HALF OF THE SAME SPLIT.
    fn set_src_loop_ele_offset(
        &mut self,
        _node: NodeId,
        _corelet: Corelet,
        _dim_loop: tr::LoopId,
        _dim: PrimaryDim,
        _offset: v1::LoopEleOffset,
    ) {
        todo!(
            "v1::LoopOffsets::set_src_loop_ele_offset: wants \
             srcLdsAndLoopOffsets_.loopEleOffsets_[cl][loop][dim] = offset — see \
             src_loop_ele_offsets: the map is on v1::DataInfoFill, not on dsc2::DataInfo"
        )
    }

    /// ⛔ No COMPUTE arm — `inputsLdsAndLoopOffsets_` lives on the compute node.
    fn set_input_loop_ele_offset(
        &mut self,
        _node: NodeId,
        _input: v1::InputIdx,
        _corelet: Corelet,
        _dim_loop: Option<tr::LoopId>,
        _dim: PrimaryDim,
        _offset: v1::LoopEleOffset,
    ) {
        todo!(
            "v1::LoopOffsets::set_input_loop_ele_offset: wants \
             inputsLdsAndLoopOffsets_.at(input).loopEleOffsets_[cl][dim_loop][dim] = offset — \
             super::tree::Kind has no Compute arm"
        )
    }
}

impl tu::FifoResults for Dsc2Store<'_, '_> {
    /// ⛔ `metadata.dataConnects_[connect].consumers_` as [`tu::FifoConsumer`]s — its compute arm
    /// again.
    fn connect_consumers(&self, _connect: Option<DataConnect>) -> Vec<tu::FifoConsumer> {
        todo!(
            "tu::FifoResults::connect_consumers: wants metadata.dataConnects_[connect].consumers_ as \
             FifoConsumers — the metadata's census, which no carrier owns, and whose compute arm \
             needs a COMPUTE node"
        )
    }

    /// ⛔ `computeNode->isOpaqueOp_`.
    fn is_opaque(&self, _compute: NodeId) -> bool {
        todo!(
            "tu::FifoResults::is_opaque: wants computeNode->isOpaqueOp_ (dsc/dsc2.h:941) — no \
             Compute arm"
        )
    }

    /// ⛔ The transfer's two ends AS ALLOCATION LOOKUPS — the lookup is
    /// `getMutableAllocation(operand, storage)`, which reads `constantInfo_` on a constant end.
    fn transfer_ends(&self, _transfer: NodeId) -> tu::TransferEnds {
        todo!(
            "tu::FifoResults::transfer_ends: wants the transfer's ends as allocation lookups — \
             getMutableAllocation(operand, storage) reads constantInfo_ on a constant end, which \
             l3::dsc::DesignSpaceConfig does not project"
        )
    }

    /// ⛔ A minted `ddc` allocate node — see `tu::AllocateCloning::allocate`.
    fn insert_allocate(&mut self, _alloc: AllocId, _node: tu::DdcAllocateNode, _at: tu::InsertionPoint) {
        todo!(
            "tu::FifoResults::insert_allocate: wants a minted DdcAllocateNode placed and registered \
             — super::tree::Kind::Allocate holds the L3 view, and the ddc view carries placed state \
             no unplaced allocation has"
        )
    }

    /// `transferNode->dstVias_[i].loc_.storage_ = storage` — ⭐ ANSWERED, by rebuilding
    /// [`crate::schedule::dsc2::Dsts`] with that one destination edited: the type keeps its
    /// non-emptiness by holding `first` and `rest` privately, so a write goes through the constructor.
    fn set_dst_storage(&mut self, transfer: NodeId, dst: usize, storage: SenComponent) {
        self.edit_dst(transfer, dst, move |operand| operand.storage = storage);
    }

    /// `transferConsumer->src_.storage_ = storage` — ⭐ ANSWERED.
    fn set_src_storage(&mut self, transfer: NodeId, storage: SenComponent) {
        self.edit_transfer(transfer, |held| held.src.storage = storage);
    }

    /// ⛔ `computeConsumer->inputs_[i] = unit`.
    fn set_compute_input_unit(&mut self, _compute: NodeId, _input: usize, _unit: SenComponent) {
        todo!(
            "tu::FifoResults::set_compute_input_unit: wants computeConsumer->inputs_[i] = unit \
             (dsc/dsc2.h:906) — no Compute arm"
        )
    }

    /// `allocNode->addAllocUser(user)` — ⭐ ANSWERED off the labelled DS's own `memOrg_`, which is
    /// where the users list lives.
    fn add_alloc_user(&mut self, alloc: AllocId, user: NodeId) {
        self.add_user_to(alloc, user);
    }
}

impl tu::SkipRegResults for Dsc2Store<'_, '_> {
    /// `transferNode->dstLdsAndLoopOffsets_.at(i).latchDataId_ = id` — ⭐ ANSWERED.
    fn set_dst_latch_data_id(&mut self, transfer: NodeId, dst: usize, id: LatchDataId) {
        self.edit_dst(transfer, dst, move |operand| {
            operand.data.latch_data_id = Some(id);
        });
    }

    /// `transferConsumer->srcLdsAndLoopOffsets_.latchDataId_ = id` — ⭐ ANSWERED.
    fn set_src_latch_data_id(&mut self, transfer: NodeId, id: LatchDataId) {
        self.edit_transfer(transfer, |held| held.src.data.latch_data_id = Some(id));
    }

    /// ⛔ No COMPUTE arm.
    fn set_compute_input_latch_data_id(
        &mut self,
        _compute: NodeId,
        _input: usize,
        _id: LatchDataId,
    ) {
        todo!(
            "tu::SkipRegResults::set_compute_input_latch_data_id: wants \
             inputsLdsAndLoopOffsets_.at(i).latchDataId_ = id — no Compute arm"
        )
    }

    /// ⛔ `allocNode->removeAllocUser(user)` (`dsc/dsc2.h:1024`) — [`super::tree::Org`] has
    /// `add_user` and no remove, and its abort *"Schedule node is not in the user list"* is
    /// [`tu::AllocationUse`] missing, which a silent no-op would swallow.
    fn remove_alloc_use(&mut self, _alloc_use: tu::AllocationUse) {
        todo!(
            "tu::SkipRegResults::remove_alloc_use: wants allocNode->removeAllocUser(user) \
             (dsc/dsc2.h:1024) — super::tree::Org has no remove, and its abort for a user not in the \
             list would be swallowed by a no-op"
        )
    }

    /// ⛔ No COMPUTE arm.
    fn set_sole_compute_output(&mut self, _compute: NodeId, _unit: SenComponent, _data: DataInfo) {
        todo!(
            "tu::SkipRegResults::set_sole_compute_output: wants computeNode->outputs_.resize(1) then \
             at(0) = (unit, data) — no Compute arm"
        )
    }

    /// ⛔ No COMPUTE arm.
    fn resize_compute_inputs_to(
        &mut self,
        _compute: NodeId,
        _input: tu::InputIdx,
        _unit: SenComponent,
        _data: DataInfo,
    ) {
        todo!(
            "tu::SkipRegResults::resize_compute_inputs_to: wants \
             computeNode->inputs_.resize(i + 1) then at(i) = (unit, data) — no Compute arm"
        )
    }
}

impl tu::MintedConnects for Dsc2Store<'_, '_> {
    /// ⛔⛔ [`DataConnect`] IS A GENERATED CLOSED SET. `build.rs` censuses every `data_connect=`
    /// spelling across `ddl_templates/*.ddl` into one enum, so there is no way to INTERN a connect
    /// the templates did not already name — and a minted one that collided with an existing variant
    /// would silently join that variant's producer/consumer census.
    fn intern_connect(&mut self, _connect: tu::MintedConnect) -> DataConnect {
        todo!(
            "tu::MintedConnects::intern_connect: DataConnect is a closed set generated from \
             ddl_templates/*.ddl by build.rs, so a minted connect has no variant to intern into — \
             and folding it onto an existing variant would join that variant's data-connect census"
        )
    }
}

impl tu::ExternalStreams for Dsc2Store<'_, '_> {
    /// ⛔ `storageOrDatastreamIsExternal(dataInfo, storage, isIncoming)` — reads
    /// `externalDataStreams_`, a `DesignSpaceConfig` field `l3::dsc` does not project.
    fn storage_or_datastream_is_external(
        &self,
        _data: DataInfo,
        _storage: SenComponent,
        _direction: tu::StreamDirection,
    ) -> bool {
        todo!(
            "tu::ExternalStreams::storage_or_datastream_is_external: wants \
             storageOrDatastreamIsExternal(dataInfo, storage, isIncoming), which reads \
             externalDataStreams_ — not projected onto l3::dsc::DesignSpaceConfig"
        )
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// THE CONDITION SURFACE ENTRY 249 MOVES TRANSFERS THROUGH.
// ════════════════════════════════════════════════════════════════════════════════════════════════

impl tu::TransferMoves for Dsc2Store<'_, '_> {
    /// `node->nodeType_ == CONDITION`.
    fn is_condition(&self, node: NodeId) -> bool {
        self.node_kind_of(node) == Some(NodeKind::Condition)
    }

    /// `condNode->getThenBranchNode()` — the FIRST region added, which is what
    /// [`super::tree::Cond::then_region`] holds in order.
    fn then_branch(&self, condition: NodeId) -> Option<NodeId> {
        self.then_region_of(condition)
    }

    /// `condNode->coreClCond_` — ⛔ TOTAL, and an EMPTY set is what a loop-guarded condition carries.
    fn core_cl_cond(&self, condition: NodeId) -> v1::CoreClSet {
        self.core_cl_cond_of(condition).unwrap_or_default()
    }

    /// `0 .. numCoreletsUsed_DSC2_` — the `allCorelets` universe.
    fn corelets_used(&self) -> BTreeSet<Corelet> {
        self.dsc2_corelets().into_iter().collect()
    }

    /// `currDsc->coreIdsUsed_`.
    fn core_ids_used(&self) -> Vec<Core> {
        self.dsc_facts()
            .with_dsc(|dsc| dsc.core_ids_used.iter().collect())
    }

    /// ⛔ `metadata.dataConnects_.at(connect).getProducerLoops()` — the METADATA's census, which no
    /// carrier owns; the trait's own doc says why it is asked through the seam.
    fn producer_loops(&self, _connect: DataConnect) -> Option<Vec<tr::LoopId>> {
        todo!(
            "tu::TransferMoves::producer_loops: wants \
             metadata.dataConnects_.at(connect).getProducerLoops() (ddc/ddc_metadata.h:159) — the \
             metadata's own census, which no carrier owns"
        )
    }

    /// `currDsc->getAllocation(dstLdsAndLoopOffsets_.at(i), dstVias_.at(i).loc_.storage_, true)`
    /// (`ddc/ddc_transformation.cpp:1487-1489`) AS THAT ALLOCATION'S SCHEDULE NODE — ⭐ ANSWERED
    /// through [`Dsc2Store::allocation_at`].
    ///
    /// ⛔⛔ THE `constantInfo_` OBJECTION WAS WRONG. This method used to refuse saying *"its constant
    /// arm reads `constantInfo_`, which `l3::dsc::DesignSpaceConfig` does not project"* — the
    /// projection is [`crate::schedule::l3::dsc::ConstantInfo::allocations`], and
    /// [`tu::DscAllocations::allocation_in`] beside it already answers the same `getAllocation` on
    /// both arms.
    ///
    /// ⛔ [`None`] IS THE REFERENCE'S OWN `nullptr` AND NOT A REFUSAL: the one caller takes it as
    /// *"this destination has no allocation to own a loop"* and skips the destination
    /// (`ddc/transformation.rs:2467-2469`), which is what `allowMissingAlloc = true` buys there.
    fn destination_allocation(&self, dst: &Operand) -> Option<NodeId> {
        let alloc = self.allocation_at(dst.data, dst.storage)?;
        self.with_tree(|tree| tree.node_of_alloc(alloc))
    }

    /// ⛔ `condNode->clone()` — a DETACHED copy with its regions still to be added, and
    /// [`super::tree::Cond`] holds its regions as links INTO this tree, so a detached clone of one
    /// has no state to carry.
    fn clone_condition(&mut self, _condition: NodeId) -> NodeId {
        todo!(
            "tu::TransferMoves::clone_condition: wants condNode->clone() as a DETACHED copy — \
             super::tree::Cond holds its then/else regions as NodeIds of this tree, so cloning one \
             means deciding what the copy's regions name, which is entry 249's decision and not this \
             carrier's"
        )
    }

    /// `new dsc2::ConditionNode()` with its `name_` and `coreClCond_`, and no loop condition — ⭐
    /// ANSWERED: that is exactly [`super::tree::Cond`]'s core/corelet-guarded shape.
    fn new_condition(&mut self, name: NodeName, core_cl: v1::CoreClSet) -> NodeId {
        self.new_core_cl_condition(name, core_cl)
    }

    /// `condNode->addThenRegion(block)`.
    fn add_then_region(&mut self, condition: NodeId, block: NodeId) {
        self.add_region_to(condition, block, true);
    }

    /// `condNode->addElseRegion(block)`.
    fn add_else_region(&mut self, condition: NodeId, block: NodeId) {
        self.add_region_to(condition, block, false);
    }
}

impl tr::HoistTransfers for Dsc2Store<'_, '_> {
    /// `currParent->nodeType_` AS THE ONE THREE-WAY DISPATCH ENTRY 301'S PARENT WALK MAKES — ⭐
    /// ANSWERED WHOLE off `nodeType_` alone.
    ///
    /// ⛔⛔ THIS DOC USED TO SAY *"its arms name the same COMPUTE node [`tr::ScopeTree::scope_node`]
    /// wants"* AND THE AUTHORITY REFUTES IT: [`tr::HoistParent`] HAS NO COMPUTE ARM, and neither does
    /// the walk it feeds. `hoistTransfersUpForReuse` tests exactly two `nodeType_`s on the way up —
    /// `if (currParent->nodeType_ == CONDITION) { collectLoopReferences(..); continue; }` then
    /// `if (currParent->nodeType_ != LOOP) { continue; }`
    /// (`ddc/ddc_transformation.cpp:1497-1505`) — so a COMPUTE parent takes the same `continue` a
    /// BLOCK, ALLOCATE, TRANSFER, SYNC or STICKMASK parent takes, which is
    /// [`tr::HoistParent::Other`]'s own *"every other kind, which the walk passes through"*.
    ///
    /// ⛔ [`tr::HoistParent::Other`] FOR A NODE THIS TREE DOES NOT HOLD, and that is not a fabricated
    /// answer: the only caller reaches this through [`tu::ScheduleSurgery::parent`]
    /// (`ddc/transformation.rs:2481-2482`), whose every answer is a node of this tree, and the
    /// reference's own loop condition ends at the `nullptr` parent rather than classifying one.
    fn hoist_parent(&self, node: NodeId) -> tr::HoistParent {
        match self.node_kind_of(node) {
            // `nodeType_ == CONDITION` — the walk collects the loops the condition names.
            Some(NodeKind::Condition) => tr::HoistParent::Condition,
            // `nodeType_ == LOOP` — `static_cast<dsc2::LoopNode *>(currParent)`, whose identity IS
            // the node's: [`tr::LoopId`] is a `NodeId` newtype and the kind is what proves it.
            Some(NodeKind::Loop) => tr::HoistParent::Loop(tr::LoopId(node)),
            _ => tr::HoistParent::Other,
        }
    }

    /// `dataStageParam_.at(core).ss_.paddingSizes_.at(dim).windowDim_` — ⭐ THE SHARED MAP, and
    /// [`None`] both where the map has no entry for `dim` AND for its `PrimaryDimTypesCount`, which is
    /// the reference's own *"no window dim"*.
    fn window_dim(&self, dim: PrimaryDim) -> Option<PrimaryDim> {
        self.core_stage_padding(dim)?.window_dim
    }

    /// `traverseTreeDFSMutable(base, {SYNC}, .., excludeList)` reduced to each SYNC node's `units_`
    /// — ⭐ ANSWERED: a SYNC is a kind this tree holds and `units_` is on it.
    fn sync_units_under(&self, base: tr::LoopId, exclude: Option<tr::LoopId>) -> Vec<SyncUnits> {
        self.sync_units_below(base, exclude)
    }

    /// The identity a freshly minted `dsc2::AllocateNode` takes — ⭐ ANSWERED: the tree's own next
    /// free id.
    fn free_alloc_id(&self) -> AllocId {
        self.next_free_alloc()
    }
}

impl tr::Splat4bRead for Dsc2Store<'_, '_> {
    /// `labeledDs_.at(lds).dataFormat_` — ⭐ ANSWERED off
    /// [`crate::schedule::l3::dsc::LdsRecord::data_format`]. ⛔ [`None`] IS `DataFormats::INVALID` or
    /// an lds the list does not hold, which the trait's own [`Option`] already spells.
    fn lds_data_format(&self, lds: LdsIdx) -> Option<DataFormat> {
        self.dsc_facts()
            .with_lds(lds, |held| held.record().data_format)
            .flatten()
    }

    /// `lds.scale_.at(getDimIndexInLayoutOrder(dsType_, dim))` for EVERY `labeledDs_` entry — ⭐
    /// ANSWERED, and ⛔ [`None`] IS the reference's own `dimIdx < 0`, its first `broadcastNeeded` arm.
    fn every_lds_scale_on(&self, dim: PrimaryDim) -> Vec<Option<tr::Scale>> {
        self.dsc_facts().with_dsc(|dsc| {
            dsc.labeled_ds
                .indexed()
                .map(|(at, held)| {
                    dsc.layout_dims
                        .get(&at)
                        .filter(|layout| layout.iter().any(|named| named == dim))
                        .and_then(|_| held.scale(dim))
                })
                .collect()
        })
    }

    /// ⛔ No COMPUTE arm.
    fn mint_compute(&mut self, _node: ComputeNode) -> NodeId {
        todo!(
            "tr::Splat4bRead::mint_compute: wants new dsc2::ComputeNode() unparented — \
             super::tree::Kind has no Compute arm"
        )
    }

    /// The tree's own next free allocation identity.
    fn free_alloc(&self) -> AllocId {
        self.next_free_alloc()
    }
}

impl tr::SpreadTransfers for Dsc2Store<'_, '_> {
    /// ⛔ `allocNode->gapStickSpread_.empty()` negated (`dsc/dsc2.h:995`) — a `ddc`-view
    /// `AllocateNode` field, and entry 300 (its only writer) has its single callsite inside `#if 0`
    /// (`ddc/ddcv1.cpp:3735-3752`), so nothing in this pipeline ever fills it.
    fn has_gap_stick_spread(&self, _alloc: AllocId) -> bool {
        todo!(
            "tr::SpreadTransfers::has_gap_stick_spread: wants allocNode->gapStickSpread_ \
             (dsc/dsc2.h:995) — a ddc-view AllocateNode field that l3::dl_ops::L3AllocateNode does \
             not carry; its only writer (entry 300) is behind `#if 0` at ddc/ddcv1.cpp:3735-3752"
        )
    }
}

impl tr::SymbolicTransfers for Dsc2Store<'_, '_> {
    /// `labeledDs_.at(lds).memOrg_`'s FIRST entry with an `allocateNode_` — ⭐ ANSWERED off the
    /// tree's `Org`, in `SenComponent` order, which is `std::map`'s.
    fn first_allocation_of(&self, lds: LdsIdx) -> Option<AllocId> {
        self.first_alloc_of(lds)
    }

    /// ⛔ `getSizeDataStageForNode(transfer, reference).ss_.symbolicDimInfo_` — WHICH datastage sizes
    /// that transfer is the same `getSizeDataStageForNode` seam
    /// [`v1::StageSizes::size_stage`](crate::schedule::ddc::v1::StageSizes::size_stage) names.
    fn symbolic_dims(
        &self,
        _transfer: NodeId,
        _reference: AllocId,
    ) -> BTreeMap<PrimaryDim, crate::schedule::l3::dsc::SymbolicDimInfo> {
        todo!(
            "tr::SymbolicTransfers::symbolic_dims: wants \
             getSizeDataStageForNode(transfer, reference).ss_.symbolicDimInfo_ \
             (dsc/designSpaceConfig.h:264) — the same size-datastage seam StageSizes::size_stage wants"
        )
    }
}

impl tr::PeSfpWorkSplit for Dsc2Store<'_, '_> {
    /// `dataStageParam_.at(core).ss_.peSfpSplit_.empty()` negated — ⭐ THE SHARED MAP.
    fn has_pe_sfp_split(&self) -> bool {
        self.core_stage_has_pe_sfp_split()
    }

    /// ⛔ `traverseTreeDFSMutable(nullptr, {ALLOCATE, TRANSFER, COMPUTE}, ..)` as
    /// [`tr::PeSfpSplitNode`]s — a PARTIAL walk that omitted the compute arm would silently split
    /// only half the nodes, so the absence is named rather than narrowed.
    fn split_candidates(&self) -> Vec<tr::PeSfpSplitNode> {
        todo!(
            "tr::PeSfpWorkSplit::split_candidates: wants the {{ALLOCATE, TRANSFER, COMPUTE}} walk as \
             PeSfpSplitNodes — super::tree::Kind has no Compute arm, and a walk that dropped that \
             arm would split only some of the nodes"
        )
    }
}

impl tr::OffsetAdjustment for Dsc2Store<'_, '_> {
    /// ⛔ `node->repetitionWithOffset_.forOutputs_` — a COMPUTE-node field.
    fn output_repetitions(&self, _node: NodeId) -> Vec<tr::Repetition> {
        todo!(
            "tr::OffsetAdjustment::output_repetitions: wants \
             node->repetitionWithOffset_.forOutputs_ (dsc/dsc2.h:950) — no Compute arm"
        )
    }

    /// ⛔ No COMPUTE arm.
    fn clone_compute_after(&mut self, _node: NodeId) -> NodeId {
        todo!(
            "tr::OffsetAdjustment::clone_compute_after: wants node->clone() placed after node — no \
             Compute arm"
        )
    }

    /// ⛔ No COMPUTE arm.
    fn set_output_repetition(&mut self, _node: NodeId, _idx: tr::OutputIdx, _reps: tr::Repetition) {
        todo!(
            "tr::OffsetAdjustment::set_output_repetition: wants \
             repetitionWithOffset_.forOutputs_.at(idx) = reps — no Compute arm"
        )
    }

    /// ⛔ `metadata.nodeCloningMap_[original].push_back(clone)` — the METADATA's map
    /// (`ddc/ddc_metadata.h:217`), which `run_v1` holds as a local and does not hand to this carrier.
    fn record_clone(&mut self, _original: NodeId, _clone: NodeId) {
        todo!(
            "tr::OffsetAdjustment::record_clone: wants metadata.nodeCloningMap_[original]\
             .push_back(clone) (ddc/ddc_metadata.h:217) — the Metadata is run_v1's own local and is \
             not one of Dsc2Carriers' ten borrows"
        )
    }

    /// `labeledDs_.at(outputsLdsAndLoopOffsets_.at(idx).myLdsIdx_).memOrg_.at(outputs_.at(idx))`
    /// `.allocateNode_` — ⭐ ANSWERED, both halves off [`super::tree::Kind::Compute`]'s own node:
    /// [`crate::schedule::dsc2::Operand`] zips `outputs_.at(idx)` (its `unit`) with
    /// `outputsLdsAndLoopOffsets_.at(idx)` (its `data`), so the two `.at()`s cannot select different
    /// positions.
    ///
    /// ⛔ THE `memOrg_` KEY IS THE OUTPUT'S **UNIT** AND NOT ITS `storage`, which is what
    /// `memOrg_.at(outputs_.at(idx))` spells: `outputs_` is a `std::vector<SenComponents>`
    /// (`dsc/dsc2.h:934`) and the trait's own doc names it.
    ///
    /// ⛔ EVERY [`None`] IS ALREADY THE PORT'S STATED DIVERGENCE, not one added here: entry 108's own
    /// doc says *"an output whose `memOrg_` entry is missing or whose `allocateNode_` is null makes
    /// the reference throw or dereference null (`ddc/ddc_transformation.cpp:1370-1373`); here the
    /// clone is still made and recorded and only the user bump and the spread are skipped"*.
    fn output_allocation(&self, node: NodeId, idx: tr::OutputIdx) -> Option<AllocId> {
        let held = self.compute_of(node)?;
        let output = held.outputs.get(idx.0)?;
        let lds = output.data.my_lds_idx?;
        tu::ComponentAllocations::mem_org_allocation(self, lds, output.unit)
    }

    /// `alloc->allocUsers_.push_back({user, 1})` — ⛔ THE RAW PUSH, not `addAllocUser`: a repeat user
    /// gets a SECOND entry at count 1, which is exactly what [`super::tree::Org::add_user`] does.
    fn add_alloc_user(&mut self, alloc: AllocId, user: NodeId) {
        self.add_user_to(alloc, user);
    }

    /// `alloc->layoutDimOrder_.at(0)` — ⭐ ANSWERED off the L3 allocate node's own layout; ⛔ TOTAL,
    /// because the reference `DT_CHECK`s that order non-empty wherever it derives one.
    fn alloc_outermost_layout_dim(&self, alloc: AllocId) -> PrimaryDim {
        self.alloc_layout_first(alloc).unwrap_or_else(|| {
            panic!(
                "tr::OffsetAdjustment::alloc_outermost_layout_dim: {alloc:?} names no ALLOCATE of \
                 this tree, or one whose layoutDimOrder_ is empty — which the reference DT_CHECKs \
                 against wherever it derives one"
            )
        })
    }

    /// ⛔ `alloc->gapStickSpread_[dim] = spread` — the same field
    /// [`tr::SpreadTransfers::has_gap_stick_spread`] names, whose only writer is behind `#if 0`.
    fn set_gap_stick_spread(&mut self, _alloc: AllocId, _dim: PrimaryDim, _spread: tr::StickSpread) {
        todo!(
            "tr::OffsetAdjustment::set_gap_stick_spread: wants allocNode->gapStickSpread_[dim] = \
             spread (dsc/dsc2.h:995) — a ddc-view field l3::dl_ops::L3AllocateNode does not carry"
        )
    }
}

impl tr::AutoShuffling for Dsc2Store<'_, '_> {
    /// `currDsc->labeledDs_[lds]`.
    fn lds_entry(&self, lds: LdsIdx) -> Option<Self::Entry> {
        v1::PrepDsc::lds_entry(self, lds)
    }

    /// `ds_info.dsName_ = name` — ⭐ ANSWERED through the shared `currDsc` cell. ⛔ AN ASSIGNMENT AND
    /// NOT AN APPEND, unlike [`v1::PrepDsc::append_lds_name`]: entry 372 NAMES its minted register
    /// `autoshuffle_reg_{n}` outright (`ddc/transformation.rs:3295`).
    fn set_lds_name(&mut self, lds: LdsIdx, name: v1::StorageName) {
        let _: Option<()> = self
            .dsc_facts()
            .with_lds_mut(lds, |held| held.set_name(name));
    }

    /// `ds_info.dataFormat_ = format` — ⭐ ANSWERED.
    fn set_lds_format(&mut self, lds: LdsIdx, format: DataFormat) {
        let _: Option<()> = self
            .dsc_facts()
            .with_lds_mut(lds, |held| held.set_data_format(format));
    }

    /// `ds_info.wordLength = word_length` — ⭐ ANSWERED.
    fn set_lds_word_length(&mut self, lds: LdsIdx, length: WordLength) {
        let _: Option<()> = self
            .dsc_facts()
            .with_lds_mut(lds, |held| held.set_word_length(length));
    }

    /// `currDsc->computeOp_.back()` — ⛔ [`None`] on any other size, which is the ONE call site's own
    /// `DT_CHECK(computeOp_.size() == 1)` and is answered from `computeOp_`.
    fn sole_compute_op(&self) -> Option<crate::schedule::ddl::conversion::ComputeOpIdx> {
        let ops = self.dsc_facts().ops();
        (ops.len() == 1).then_some(crate::schedule::ddl::conversion::ComputeOpIdx(0))
    }

    /// ⛔ `computeOp_.back().interimLabeledDs.push_back(..)` — `interimLabeledDs` is not a field of
    /// [`v1::DscComputeOp`], which projects the five entries 307/308 read.
    fn add_interim_lds(
        &mut self,
        _compute_op: crate::schedule::ddl::conversion::ComputeOpIdx,
        _lds: LdsIdx,
    ) {
        todo!(
            "tr::AutoShuffling::add_interim_lds: wants \
             computeOp_.back().interimLabeledDs.push_back(&labeledDs_[lds]) — interimLabeledDs is \
             not a field of v1::DscComputeOp"
        )
    }

    /// An [`AllocId`] no allocation carries yet.
    fn free_alloc(&self) -> AllocId {
        self.next_free_alloc()
    }

    /// ⛔ A minted `ddc` allocate node — see `tu::AllocateCloning::allocate`.
    fn insert_allocate(&mut self, _alloc: AllocId, _node: tu::DdcAllocateNode, _at: tu::InsertionPoint) {
        todo!(
            "tr::AutoShuffling::insert_allocate: wants the held DdcAllocateNode handed to the tree \
             (shuffle.h:175) — super::tree::Kind::Allocate holds the L3 view"
        )
    }

    /// `allocNode->addAllocUser(user)`.
    fn add_alloc_user(&mut self, alloc: AllocId, user: NodeId) {
        self.add_user_to(alloc, user);
    }

    /// ⛔ No COMPUTE arm.
    fn insert_compute_before(&mut self, _node: ComputeNode, _before: NodeId) -> NodeId {
        todo!(
            "tr::AutoShuffling::insert_compute_before: wants new dsc2::ComputeNode(*assign) placed \
             before `assign` — no Compute arm"
        )
    }

    /// `node->getMutableParent()->deleteChildNode(currDsc, node)`.
    fn delete_node(&mut self, node: NodeId) {
        v1::ConditionSimplification::delete_node(self, node);
    }

    /// `dataFormatsToBitWidth.at(labeledDs_[dinfo.myLdsIdx_].dataFormat_)` — ⭐ ANSWERED off
    /// [`crate::schedule::l3::dsc::LdsRecord::data_format`] and
    /// [`crate::formats::DataFormat::bits`], which IS that table
    /// (`util/sendefs/sendefs.cpp:129-141`).
    ///
    /// ⛔ TOTAL BY THE TRAIT, AND NEITHER ABSENCE HAS AN ANSWER IN THE REFERENCE, not a width. The
    /// expression is `dataFormatsToBitWidth.at(currDsc->labeledDs_[in1.dinfo.myLdsIdx_].dataFormat_)`
    /// (`ddc/ddc_transformation.cpp:1966-1968`), and the two halves fail DIFFERENTLY: the index is
    /// `operator[]`, so `myLdsIdx_ = -1` or an lds past the end is UNDEFINED BEHAVIOUR there rather
    /// than a throw — stated as a divergence, and a panic is the only honest reading of it — while the
    /// `.at()` on the width table IS the throw for a `dataFormat_` of `INVALID`. This is an ELEMENT
    /// WIDTH: a stand-in resizes every packmerge operand, so both stop.
    fn operand_element_bits(&self, dinfo: DataInfo) -> crate::formats::Bits {
        let lds = dinfo.my_lds_idx.unwrap_or_else(|| {
            panic!(
                "tr::AutoShuffling::operand_element_bits: labeledDs_[myLdsIdx_] on an operand whose \
                 myLdsIdx_ is the reference's -1"
            )
        });
        self.dsc_facts()
            .with_lds(lds, |held| held.record().data_format)
            .flatten()
            .unwrap_or_else(|| {
                panic!(
                    "tr::AutoShuffling::operand_element_bits: \
                     dataFormatsToBitWidth.at(labeledDs_[{lds:?}].dataFormat_) has no row for an \
                     absent lds or a dataFormat_ of INVALID"
                )
            })
            .bits()
    }

    /// ⛔⛔ THE `dataFormat_` HALF IS NOW CARRIED AND `stickRepl_` IS WHAT IS LEFT.
    /// [`crate::schedule::ddc::shuffle::OperandSticks::primary_stick_repl`] is
    /// `primaryDsInfo_.at(dsType_).stickRepl_` (`shuffle.cpp:805-813`), which
    /// [`crate::schedule::l3::dsc::PrimaryDsInfo`] projects as neither of its two fields and which
    /// scratchy emits nowhere — `primaryDsInfo_` on the wire is `layoutDimOrder_`/`stickDimOrder_`/
    /// `stickSize_` and nothing else.
    ///
    /// ⛔ AND AN EMPTY LIST WOULD NOT BE ITS ABSENCE: `all_one` (entry 159) is
    /// [`crate::schedule::ddc::shuffle::AutoShuffler::infer_layouts`]' own `DT_CHECK`
    /// (`shuffle.cpp:1043-1045`), which an empty `stickRepl_` PASSES — so a fresh empty vector here
    /// would assert *"nothing is replicated"* for an operand that may well be, and it would compile.
    fn operand_sticks(&self, _dinfo: DataInfo) -> Option<crate::schedule::ddc::shuffle::OperandSticks> {
        todo!(
            "tr::AutoShuffling::operand_sticks: wants primaryDsInfo_.at(dsType_).stickRepl_ \
             (shuffle.cpp:805-813), which l3::dsc::PrimaryDsInfo does not project and scratchy's \
             primaryDsInfo_ does not emit — an EMPTY stickRepl_ PASSES all_one's DT_CHECK, so it \
             would assert 'nothing is replicated' for an operand that may be"
        )
    }

    /// `currDsc->coreIdsUsed_`.
    fn cores_used(&self) -> v1::CoresUsed {
        self.cores_used_of()
    }

    /// `0 .. currDsc->numCoreletsUsed_` — ⛔ NOT `numCoreletsUsed_DSC2_`.
    fn corelets_used(&self) -> Vec<Corelet> {
        ddc_state::corelets_of(self.dsc_facts().with_dsc(|dsc| dsc.corelets_used.get()))
    }

    /// `allocNode->getPrev()` — whether that allocate node already has a preceding sibling.
    fn allocation_has_prev(&self, alloc: AllocId) -> bool {
        self.alloc_has_parent(alloc)
    }

    /// ⛔ No COMPUTE arm — `{inputs,outputs}LdsAndLoopOffsets_` live on the compute node.
    fn set_const_ele_offsets(
        &mut self,
        _node: NodeId,
        _at: OperandPos,
        _offsets: crate::schedule::ddc::shuffle::ConstEleOffsets,
    ) {
        todo!(
            "tr::AutoShuffling::set_const_ele_offsets: wants \
             node->{{inputs,outputs}}LdsAndLoopOffsets_[pos].constEleOffsets_ = offsets \
             (shuffle.cpp:881-896) — no Compute arm"
        )
    }
}

impl crate::schedule::ddc::fold::FoldConstruction for Dsc2Store<'_, '_> {
    /// ⛔ `buildAndPropagateFold()` — `util/foldManager/`, outside this campaign's file list, and the
    /// call that BUILDS every address fold. ⛔ Leaving it a no-op would let entry 375's capture run
    /// against an empty fold space and report coordinates nothing built.
    fn build_and_propagate_fold(&mut self) {
        todo!(
            "fold::FoldConstruction::build_and_propagate_fold: wants buildAndPropagateFold() — \
             util/foldManager/, outside this campaign's file list. A no-op here would let entry 375 \
             capture coordinates off a fold space nothing built."
        )
    }
}

impl crate::schedule::ddc::fold::CoordinateCapture for Dsc2Store<'_, '_> {
    /// ⛔ `traverseTreeDFSMutable(nullptr, {ALLOCATE, COMPUTE, TRANSFER})` as
    /// [`crate::schedule::ddc::fold::CapturedNode`]s — its compute arm again.
    fn captured_nodes(&self) -> Vec<crate::schedule::ddc::fold::CapturedNode> {
        todo!(
            "fold::CoordinateCapture::captured_nodes: wants the {{ALLOCATE, COMPUTE, TRANSFER}} walk \
             as CapturedNodes — no Compute arm, and a walk missing that arm would report a capture \
             over only part of the tree"
        )
    }

    /// ⛔ `coord.debugPrint(std::cout, printContent)` (`dsc/dsc2.h:361`) — a `dsc/` printer.
    fn coordinate_text(
        &self,
        _coord: &crate::schedule::dsc2::Coordinate,
        _content: crate::schedule::ddc::fold::CoordContent,
    ) -> String {
        todo!(
            "fold::CoordinateCapture::coordinate_text: wants coord.debugPrint(std::cout, \
             printContent) (dsc/dsc2.h:361) — a `dsc/` printer outside this campaign's file list"
        )
    }

    /// ⛔ `node->print(std::cout)` — three `dsc/` printers, one per node type.
    fn node_text(&self, _node: NodeId) -> String {
        todo!(
            "fold::CoordinateCapture::node_text: wants node->print(std::cout), which \
             AllocateNode/TransferNode/ComputeNode each override — `dsc/` printers outside this \
             campaign's file list"
        )
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// ⭐⭐ AND THE TRAIT ITSELF — the split, and the five facts none of its supertraits owns.
// ════════════════════════════════════════════════════════════════════════════════════════════════

impl<'s, 'l> v1::Dsc2Store for Dsc2Store<'s, 'l> {
    type Reads = super::ddc_reads::Dsc2Reads<'s, 'l>;
    type Tree = super::ddc_tree::Dsc2Tree<'s, 'l>;

    /// ⭐⭐ BOTH VIEWS OF ONE STORE AT ONCE — two disjoint FIELDS, each holding `&'s Dsc2State`, so
    /// the shared and the exclusive view name the same tree and the same `labeledDs_`.
    fn split(&mut self) -> (&Self::Reads, &mut Self::Tree) {
        self.halves()
    }

    /// `getTransferType()` (`dsc/dsc2.h:883-896`) WITH THE LABELLED DS THE MATCHING SIDE NAMES — ⭐
    /// ANSWERED WHOLE.
    ///
    /// ⛔⛔ THE `constantInfo_` OBJECTION WAS WRONG, exactly as it was for
    /// [`tu::TransferMoves::destination_allocation`]: `l3::dsc` DOES project it, and
    /// [`crate::schedule::l3::dsc::ConstantInfo`] carries both halves
    /// [`v1::ConstantData`](crate::schedule::ddc::v1::ConstantData) asks for — `data` for
    /// `data_.getSingleData().size()` and `data_format` for `dataFormat_`.
    ///
    /// ⛔ THE DESTINATION SIDE IS `dstLdsAndLoopOffsets_.front()` AND ITS EMPTINESS TEST IS
    /// DISCHARGED: `isDstLabeledDs()`/`isDstConstant()` are both guarded by
    /// `!dstLdsAndLoopOffsets_.empty()`, and [`crate::schedule::dsc2::Dsts`] is non-empty by type.
    ///
    /// ⛔ WHICH SIDE SUPPLIES THE LDS IS THE ARM'S OWN ANSWER and it is the reference's own ternary
    /// `((constantToTensor || noTransToTensor) ? dstLdsAndLoopOffsets_.front().myLdsIdx_ :
    /// srcLdsAndLoopOffsets_.myLdsIdx_)` (`ddc/ddcv1.cpp:481-484`). Each arm's own predicate proves
    /// that index PRESENT, so the [`Option`] the type keeps for the reference's `ldsIdx < 0` skip is
    /// filled on every arm here rather than being a place to put a different absence.
    ///
    /// # ⛔ THREE STOPS, AND EACH IS NAMED
    ///
    /// * **BOTH INDICES SET** is `DT_CHECK_MSG(!(myLdsIdx_ >= 0 && constantId_ >= 0), "Cannot be
    ///   both labeledDs and constant.")` (`dsc/dsc2.h:742`, `:747`) — `isLabeledDs()` and
    ///   `isConstant()` each assert it before answering, so this is the reference's abort and not one
    ///   added here.
    /// * **`INVALID_TRANSFER_TYPE`** has no arm, which is [`v1::TransferOperands`]' own statement, and
    ///   the one caller's `DT_CHECK_MSG(.., "Unexpected transfer type.")` (`ddc/ddcv1.cpp:479`) is
    ///   that same abort reached from the other side.
    /// * **`dataFormat_` OF `INVALID`** on the selected labelled DS cannot be spelled:
    ///   [`v1::TransferLds`](crate::schedule::ddc::v1::TransferLds)`::format` is a
    ///   [`DataFormat`] and not an [`Option`]. ⛔ AND [`None`] WOULD BE THE WRONG ANSWER, not a
    ///   conservative one: the consumer takes a [`None`] lds as *"skip this transfer"* and returns
    ///   having pushed NO `unitTimeTransferChunkSize_`, so a transfer the reference chunks would move
    ///   nothing. This is the same reading [`tr::AutoShuffling::operand_element_bits`] states, and a
    ///   ⚠️ DIVERGENCE: the reference reads `dataFormat_` only under `do2BSplat` and aborts there
    ///   (*"No system in place for splatting of data formats with <16b"*), so it processes an
    ///   INVALID-format transfer on every other path and this stops one.
    fn transfer_operands(&self, transfer: NodeId) -> v1::TransferOperands {
        let held = tu::ScheduleSurgery::transfer(self, transfer);
        let src = held.src.data;
        let dst = held.dsts.first().data;
        let both_set = |data: DataInfo| {
            assert!(
                !(data.my_lds_idx.is_some() && data.constant_id.is_some()),
                "v1::Dsc2Store::transfer_operands: {transfer:?} has an end that is both a labelled \
                 DS and a constant — DT_CHECK_MSG(\"Cannot be both labeledDs and constant.\") \
                 (dsc/dsc2.h:742)"
            );
        };
        both_set(src);
        both_set(dst);
        match (src.my_lds_idx, src.constant_id, dst.my_lds_idx, dst.constant_id) {
            // `isSrcConstant() && isDstConstant()`.
            (_, Some(constant), _, Some(_)) => {
                v1::TransferOperands::ConstantToConstant(self.constant_data(constant, transfer))
            }
            // `isSrcConstant() && isDstLabeledDs()`.
            (_, Some(_), Some(lds), _) => {
                v1::TransferOperands::ConstantToTensor(Some(self.transfer_lds(lds, transfer)))
            }
            // `isSrcLabeledDs() && isDstLabeledDs()`.
            (Some(lds), _, Some(_), _) => {
                v1::TransferOperands::TensorToTensor(Some(self.transfer_lds(lds, transfer)))
            }
            // `(!isSrcLabeledDs() && !isSrcConstant()) && isDstLabeledDs()`.
            (None, None, Some(lds), _) => {
                v1::TransferOperands::NoTransferToTensor(Some(self.transfer_lds(lds, transfer)))
            }
            // `isSrcLabeledDs() && (!isDstLabeledDs() && !isDstConstant())`.
            (Some(lds), _, None, None) => {
                v1::TransferOperands::NoTransferFromTensor(Some(self.transfer_lds(lds, transfer)))
            }
            // `INVALID_TRANSFER_TYPE` — the caller's own `DT_CHECK_MSG("Unexpected transfer type.")`.
            _ => panic!(
                "v1::Dsc2Store::transfer_operands: {transfer:?} is INVALID_TRANSFER_TYPE — neither \
                 end is a labelled DS or a constant, which is entry 126's own \
                 DT_CHECK_MSG(\"Unexpected transfer type.\") (ddc/ddcv1.cpp:479)"
            ),
        }
    }

    /// ⛔ `traverseTreeDFSMutable(nullptr, {COMPUTE, TRANSFER})` as entry 002 censuses it — the
    /// census keys every operand by its `dataConnect_`, and [`DataConnect`] is the generated closed
    /// set [`tu::MintedConnects::intern_connect`] names.
    fn census_nodes(&self) -> Vec<ScheduleNode> {
        todo!(
            "v1::Dsc2Store::census_nodes: wants the {{COMPUTE, TRANSFER}} walk as \
             shape_constraints::ScheduleNodes — every operand keyed by its dataConnect_, which is a \
             generated closed set no transfer of this tree carries a variant of yet, and its Compute \
             arm needs a COMPUTE node"
        )
    }

    /// ⭐⭐⭐ `scheduleTree_.getHead()` AS THE WHOLE BLOCK ONE `DdlConvertInterface` IS OPENED OVER —
    /// ANSWERED, and this is the seam stage 2b used to stop on (`run_v1`'s `ddc/v1.rs:6483`, one line
    /// before `select_and_parse_ddl_template`).
    ///
    /// ⛔⛔ THE `isParametricLoop_` OBJECTION WAS WRONG AND THE AUTHORITY SAYS SO. This method used to
    /// refuse because a materialised LOOP child needs
    /// [`crate::schedule::dsc2::LoopNode::parametric_lds`] and *"writing [`None`] there states 'this
    /// loop is not parametric' as a FACT"*. It IS the fact: `isParametricLoop_ = false` and
    /// `parametricLdsIdx_ = -1` are the field's own member initializers (`dsc/dsc2.h:617-618`), and
    /// EXACTLY TWO things in the whole reference ever write them —
    ///
    ///   * `ParametricLoopOp`'s arm of the DDL conversion, which calls `markAsParametricLoop()` then
    ///     `setParametricLdsIdx(ldsIdx)` on a loop IT mints (`ddc/ddl/ddl_conversion.cpp:1126-1161`);
    ///   * the JSON importer, reading `parametricLoop_`/`parametricLdsIdx_` back off a SERIALISED
    ///     super-DSC (`dsc/dsc2.cpp:1409-1416`).
    ///
    /// Neither has run: the tree here is the one stage 2a's growers minted in memory, and the DDL
    /// conversion is what this call OPENS. So every loop of it is non-parametric by the authority's own
    /// default, and [`tu::LoopNode`] — the view the tree stores — correspondingly has no such field to
    /// lose. ⭐ A loop the DDL conversion later mints as parametric carries the flag on ITS node, which
    /// is why the field stays on [`crate::schedule::dsc2::LoopNode`] rather than being dropped.
    ///
    /// ⛔ AN EMPTY-CHILDREN BLOCK IS STILL THE ONE THING THAT MUST NOT HAPPEN — it would say the DSC's
    /// schedule tree is empty and the conversion would splice the whole parsed template into nothing —
    /// so the walk below is over `tree.children` from the head, recursively, and every kind the tree
    /// holds has an arm.
    ///
    /// ⛔ TOTAL, BECAUSE `getHead()` IS: the reference reaches it unguarded and every tree
    /// [`super::DscState::seeded`] builds has a `root_level_operations` block at node `[0]`.
    fn schedule_head_block(&self) -> BlockNode {
        self.with_tree(|tree| {
            let head = tree.head().unwrap_or_else(|| {
                panic!(
                    "v1::Dsc2Store::schedule_head_block: scheduleTree_.getHead() on a tree with no \
                     head — DscState::seeded sets one root block per DSC"
                )
            });
            head_block_of(tree, head)
        })
    }

    /// ⛔ `dsc.setRelevantCompCoreCl()` (`dsc/dsc2.cpp:2647`) — outside this campaign's file list,
    /// and it is what FILLS the `relevantComps_`/`relevantCoreCl_` maps four other methods here want.
    fn set_relevant_comp_core_cl(&mut self) {
        todo!(
            "v1::Dsc2Store::set_relevant_comp_core_cl: wants dsc.setRelevantCompCoreCl() \
             (dsc/dsc2.cpp:2647) — outside this campaign's file list, and the call that fills the \
             relevantComps_/relevantCoreCl_ maps ScheduleNodes::is_relevant, next_view_len, \
             nodes_of_kind_under and ConditionSimplification::relevant_comps all want"
        )
    }

    /// ⛔ `dsc.finalizeScheduleTree(sdsc, dscGlobal.sysDef)` (`dsc/dsc2.cpp:2749`) — likewise a
    /// seam, and the reference calls it on `dsc` itself.
    fn finalize_schedule_tree(&mut self, _sdsc: &crate::schedule::l3::dsc::SuperDsc) {
        todo!(
            "v1::Dsc2Store::finalize_schedule_tree: wants \
             dsc.finalizeScheduleTree(sdsc, dscGlobal.sysDef) (dsc/dsc2.cpp:2749) — outside this \
             campaign's file list"
        )
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// ⭐⭐ `scheduleTree_.getHead()` MATERIALISED — the identity-keyed tree as ONE OWNED `dsc2::BlockNode`.
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// ONE BLOCK AND EVERY NODE UNDER IT, AS OWNED `dsc2::` NODES — `BlockNode::next_` is a vector of
/// `ScheduleNode*` there and a `Vec<SchedNode>` here, so *"the block"* is the whole subtree.
///
/// ⛔ THE ORDER IS `tree.children`'s, WHICH IS `getNextView(ALL)`: *"EVERY child in schedule order,
/// because the `ALL` arm of `isNodeRelevant` filters nothing at all"*
/// ([`crate::schedule::ddc::fold::ScheduleTree::children`]'s own note). A DDL conversion splices by
/// position, so a reordering here would move where the parsed template lands.
pub(super) fn head_block_of(tree: &super::tree::TreeData, block: NodeId) -> BlockNode {
    BlockNode {
        name: tree.name(block).unwrap_or_default(),
        children: tree
            .children(block)
            .into_iter()
            .map(|child| sched_node_of(tree, child))
            .collect(),
    }
}

/// ONE NODE AS THE `dsc2::SchedNode` ITS `nodeType_` MAKES IT — the same eight kinds
/// [`super::tree::Kind`] holds, arm for arm.
///
/// ⛔ EVERY BLOCK KIND RECURSES AND EVERY LEAF DOES NOT, which is `isBlockNode()`
/// (`dsc/dsc2.h:479`): a `BLOCK`, a `LOOP` and a `CONDITION` own children, and an `ALLOCATE`, a
/// `TRANSFER`, a `COMPUTE`, a `SYNC` and a `STICKMASK` do not.
///
/// ⛔ AN `ALLOCATE`, A `TRANSFER` AND A `COMPUTE` ARE [`crate::schedule::dsc2::SchedNode::Leaf`] —
/// NAME ONLY, by that type's own statement — and none of the three drops a field: the DDL conversion
/// reaches an allocation through `metadata.newAllocations_`, a transfer through the arena and a compute
/// through the arena too, all keyed by identity rather than through this block.
///
/// ⛔ THIS DOC USED TO SAY *"materialising this tree needs no COMPUTE arm"*, WHICH WAS TRUE ONLY
/// BECAUSE [`super::tree::Kind`] COULD NOT HOLD ONE. It can now, so the arm is here — and the two
/// kinds that DO carry their node rather than their name, `SYNC` and `STICKMASK`, are the two whose
/// minters cross-link what they minted.
///
/// ⛔ AND A `SYNC` CARRIES ITS WHOLE NODE, not its name: *"the sequences that mint syncs cross-link the
/// pair they minted and a bare name cannot be linked"*
/// ([`crate::schedule::dsc2::SchedNode::Sync`]).
/// ⭐⭐ IT MATCHES THE NODE'S OWN [`super::tree::Kind`] AND NOT ITS `nodeType_`, WHICH IS WHAT MAKES
/// "a `LOOP` without its `LoopNode`" UNSPELLABLE RATHER THAN A PANIC: `node_kind()` is DERIVED from
/// that same enum (`Kind::node_kind`), so a second lookup keyed by the derived answer could only ever
/// restate what the match already proved. Three stops came off this function that way.
fn sched_node_of(tree: &super::tree::TreeData, node: NodeId) -> crate::schedule::dsc2::SchedNode {
    use super::tree::Kind;
    use crate::schedule::dsc2::{LoopDim, SchedNode};

    match tree.kind_of(node) {
        Some(Kind::Block) => SchedNode::Block(head_block_of(tree, node)),
        Some(Kind::Loop(held)) => {
            SchedNode::Loop(Box::new(crate::schedule::dsc2::LoopNode {
                block: head_block_of(tree, node),
                // `dims_` — the SAME order [`tu::LoopDims`] states, which is the order the loop's own
                // name spells them in.
                dims: held
                    .dims
                    .iter()
                    .map(|entry| LoopDim {
                        dim: entry.dim,
                        kind: entry.kind,
                    })
                    .collect(),
                // ⭐ `numId_`/`denId_` ARE NOT OPTIONAL ON THE STORED VIEW — *"every callsite of the
                // constructor passes a real pair"* ([`tu::LoopNode`]) — so both are `Some` here and the
                // `-1` a `dsc2::LoopNode` admits is the DDL's parametric loop, not this one.
                num: Some(held.num),
                den: Some(held.den),
                // ⛔ `isParametricLoop_ = false` / `parametricLdsIdx_ = -1`, the member initializers of
                // `dsc/dsc2.h:617-618`. See [`v1::Dsc2Store::schedule_head_block`] for the two — and
                // only two — writers, neither of which has run.
                parametric_lds: None,
            }))
        }
        Some(Kind::Condition(held)) => {
            // ⭐ THE TWO REGIONS AS `addThenRegion`/`addElseRegion` FILLED THEM, each a list of BLOCKS
            // (`dsc/dsc2.cpp:2143`'s *"ConditionNode only accepts 2 BlockNodes as children"*).
            SchedNode::Guarded(Box::new(crate::schedule::dsc2::ConditionNode {
                name: tree.name(node).unwrap_or_default(),
                // ⛔ THE EMPTY COMPOSITE IS `hasCoreClCond()`, WHICH IS THE REFERENCE'S OWN TEST:
                // *"`loopCond_.twoLevelOrOfAnds_.empty()`… answers 'the core/corelet set is what guards
                // this'"* ([`crate::schedule::dsc2::ConditionNode`]), so a core/corelet-guarded
                // condition's `loopCond_` IS empty there too.
                loop_cond: held.loop_cond.clone().unwrap_or_default(),
                core_cl_cond: held
                    .cores
                    .as_ref()
                    .map(|set| set.0.clone())
                    .unwrap_or_default(),
                then_region: held
                    .then_region
                    .iter()
                    .map(|child| SchedNode::Block(head_block_of(tree, *child)))
                    .collect(),
                else_region: held
                    .else_region
                    .iter()
                    .map(|child| SchedNode::Block(head_block_of(tree, *child)))
                    .collect(),
            }))
        }
        Some(Kind::Sync(held)) => SchedNode::Sync(held.clone()),
        // ⭐ `STICKMASK` CARRIES ITS MASK, not its name — *"a leaf that carries its mask"*
        // ([`crate::schedule::dsc2::SchedNode::StickMask`]), the same reason a `SYNC` carries its whole
        // node.
        Some(Kind::StickMask(held)) => SchedNode::StickMask(Box::new(held.clone())),
        // `ALLOCATE`, `TRANSFER` and `COMPUTE` — `isBlockNode()` is false and none has children, which
        // is exactly the set [`crate::schedule::dsc2::SchedNode::Leaf`]'s own doc names.
        //
        // ⛔ A COMPUTE IS A **NAME** HERE AND ITS NODE IS NOT DROPPED. `Kind::Compute` holds the whole
        // `ComputeNode`, and this function materialises the BLOCK the DDL conversion is handed — which
        // reaches a compute through the arena by identity, not through this block, exactly as the
        // comment above says of an allocation and a transfer.
        Some(Kind::Allocate(..) | Kind::Transfer(_) | Kind::Compute(_)) => {
            SchedNode::Leaf(tree.name(node).unwrap_or_default())
        }
        None => panic!(
            "v1::Dsc2Store::schedule_head_block: {node:?} is a child of this tree with no \
             nodeType_ — the walk came out of tree.children, so this is a defect in the tree"
        ),
    }
}

