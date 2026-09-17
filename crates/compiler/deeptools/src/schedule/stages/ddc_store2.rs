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
//! ⛔ WHAT IS NOT falls into the same three buckets [`super::ddc_store`]'s header names — seven
//! `l3::dsc`-dropped fields and the `dsc/`/`util/foldManager/` seams — plus one more that is worth
//! stating on its own: **[`crate::generated::DataConnect`] IS A CLOSED GENERATED SET** censused from
//! `ddl_templates/*.ddl` at build time, so [`tu::MintedConnects::intern_connect`] cannot mint a
//! connect the templates did not already name.
//!
//! # ⛔⛔ *"NO COMPUTE ARM"* WAS THE THIRD BUCKET AND IT IS GONE — DO NOT WRITE IT AGAIN
//!
//! [`super::tree::Kind::Compute`] holds the whole [`crate::schedule::dsc2::ComputeNode`]
//! (`stages/tree.rs:96`, landed in `c1f5c63fa`), and since `5ee670017` the DDL expansion mints into
//! the **live** tree through [`super::ddc_sites`]' `add_compute`/`mint_compute`. Sixteen `todo!`s in
//! this file refused *"no Compute arm"*; the arm was already being MATCHED at the bottom of this same
//! file (`sched_node_of`). Every one of them has been reclassified in place. What is left splits into
//! exactly three, and each remaining message says which:
//!
//! 1. **⛔ A FIELD OUR `dsc2::ComputeNode` DROPPED** — `isOpaqueOp_` (`dsc/dsc2.h:941`),
//!    `repetitionWithOffset_` (`:954`), `inputCoordinates_`/`outputCoordinate_` (`:948-949`). The
//!    FIELD is the work; adding one is a `schedule/dsc2.rs` change, not a carrier change.
//!    ⭐ `dsc2::DataInfo`'s `loopEleOffsets_` (`:730-734`) and `constEleOffsets_` (`:727-729`) HAVE
//!    LEFT THIS CLASS: both are declared (`schedule/dsc2.rs:433`, `:428`), so those four `todo!`s are
//!    case 2 — a write door — and their messages say so.
//! 2. **⛔ THE ONE MISSING DOOR** — a `pub(super)` compute READ-MODIFY-WRITE on [`Dsc2Store`], the
//!    exact shape [`Dsc2Store::edit_transfer`] already is for a transfer, plus a mint. `with_tree_mut`
//!    is PRIVATE to [`super::ddc_store`] and [`Dsc2Store`]'s `state`/`dsc` fields are that module's,
//!    so this file can READ the tree ([`Dsc2Store::with_tree`] is `pub(super)`) and cannot write a
//!    compute into it. ⛔ THE FIX IS `edit_compute`/`mint_compute` BESIDE `edit_transfer`, NOT a
//!    second view of `currDsc` here and NOT a new trait: every write-blocked `todo!` below names this.
//! 3. **⛔ A SEAM** — `metadata.dataConnects_`/`externalNodes_`/`nodeCloningMap_` (the `Metadata` is
//!    `run_v1`'s own local and is not one of `Dsc2Carriers`' borrows), and the `dsc/`,
//!    `util/foldManager/` and `ddc/transformations/automatic_shuffle/` files outside this campaign's
//!    list.

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
    BlockNode, ComputeNode, DataInfo, LatchDataId, LdsIdx, NodeBase, NodeName, Operand, OperandPos,
    SyncUnits, TransferNode, WordLength,
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
        data: &DataInfo,
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
    /// ⛔ TOTAL, AND EVERY STOP IS A THROW OF THE REFERENCE'S OWN — ⛔⛔ NOT AN EMPTY, WHICH IS THE
    /// WHOLE POINT: entry 126 turns this value straight into `unitTimeTransferChunkSize_`, so a
    /// defaulted [`StickDims`] or an empty splat set does not narrow the answer, it EMITS A DIFFERENT
    /// TRANSFER — one that moves no elements, or one that is not a splat when it is. The four:
    ///
    ///   * `labeledDs_.at(ldsIdx)` — the throw entry 126 reaches unguarded (`ddc/ddcv1.cpp:490`);
    ///   * `getStickSizes(lds.dsType_, ..)` opens with `primaryDsInfo_.at(dsType)`
    ///     (`dsc/dsc2.cpp:4077`), so a `dsType_` that table has no row for throws THERE, and an empty
    ///     stick would instead push NO chunks at all;
    ///   * `getLayoutDims`' own `DT_CHECK` that the layout order is non-empty (`dsc/dsc2.cpp:4007`) —
    ///     needed here because [`crate::schedule::l3::dsc::LabeledDs`] exposes `scale(dim)` and no
    ///     whole-`scale_` accessor, so the layout order is the only way to enumerate the `-2` set that
    ///     `is_any_of(-2, lds.scale_)` walks positionally. ⭐ AN `l3::dsc` ACCESSOR FOR `scales`
    ///     WOULD REMOVE THIS ONE;
    ///   * a `dataFormat_` of `INVALID`, which
    ///     [`v1::TransferLds`](crate::schedule::ddc::v1::TransferLds)`::format` has no spelling for —
    ///     the ⚠️ DIVERGENCE [`v1::Dsc2Store::transfer_operands`]' second stop states.
    fn transfer_lds(&self, lds: LdsIdx, transfer: NodeId) -> v1::TransferLds {
        self.dsc_facts().with_dsc(|dsc| {
            let held = dsc.labeled_ds.at(lds).unwrap_or_else(|| {
                panic!(
                    "v1::Dsc2Store::transfer_operands: labeledDs_.at({lds:?}) throws for the end \
                     {transfer:?} names (ddc/ddcv1.cpp:490)"
                )
            });
            v1::TransferLds {
                stick: ddc_state::stick_dims_of(dsc, lds).unwrap_or_else(|| {
                    panic!(
                        "v1::Dsc2Store::transfer_operands: getStickSizes' own \
                         primaryDsInfo_.at(dsType_) (dsc/dsc2.cpp:4077) has no row for \
                         labeledDs_.at({lds:?})'s dsType_ — an empty stick would push NO \
                         unitTimeTransferChunkSize_ and move nothing"
                    )
                }),
                splat_dims: crate::schedule::ddc::v1::SplatDims(
                    dsc.layout_dims
                        .get(&lds)
                        .map(|layout| {
                            layout
                                .iter()
                                .filter(|dim| held.scale(*dim) == Some(tr::Scale::StickDim))
                                .collect()
                        })
                        .unwrap_or_else(|| {
                            panic!(
                                "v1::Dsc2Store::transfer_operands: getLayoutDims DT_CHECKs a \
                                 non-empty layout order (dsc/dsc2.cpp:4007) and this DSC states none \
                                 for {lds:?} — `scale_`'s -2 set cannot be enumerated without it, \
                                 and an EMPTY set would say 'not a splat' and change \
                                 replicationFactor_"
                            )
                        }),
                ),
                format: held.record().data_format.unwrap_or_else(|| {
                    panic!(
                        "v1::Dsc2Store::transfer_operands: labeledDs_.at({lds:?}).dataFormat_ is \
                         DataFormats::INVALID and v1::TransferLds::format is not an Option — see \
                         that method's second stop"
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
    ///
    /// ⛔⛔ ITS TWO EMPTIES ARE BOTH THROWS AND ARE NOW FILED. This returned `Vec::new()` for an lds
    /// the list does not hold — the `.at()` throw [`Self::ds_type`] PANICS on, one method below, for
    /// the very same index — and again for an lds with no layout order, which is `getLayoutDims`'
    /// own `DT_CHECK` that the order is non-empty (`dsc/dsc2.cpp:4007`).
    ///
    /// ⛔ AND AN EMPTY IS THE WRONG BRANCH, NOT A NARROWER ONE: every caller asks
    /// `scale(lds).contains(&Scale::StickDim)` and SKIPS the transfer when it does not
    /// (`ddc/transformation.rs:574-575`, `:2902-2903`), so an unanswerable position silently declines
    /// to pack a stick or to splat a 4-byte read. An empty `scale_` on a labelled DS that HAS a
    /// layout order stays a true empty.
    fn scale(&self, lds: LdsIdx) -> Vec<tr::Scale> {
        let held = self.dsc_facts().with_dsc(|dsc| {
            let held = dsc.labeled_ds.at(lds)?;
            let layout = dsc.layout_dims.get(&lds)?;
            Some(layout.iter().filter_map(|dim| held.scale(dim)).collect())
        });
        held.unwrap_or_else(|| {
            self.refuse(
                "LabeledDs::scale: labeledDs_.at(lds) throws for an index the list does not hold, \
                 and getLayoutDims DT_CHECKs a non-empty layoutDimOrder_ (dsc/dsc2.cpp:4007) — so \
                 scale_ is UNKNOWN and not unscaled, and every caller reads an empty as 'not a \
                 stick dim' and skips the transfer",
            )
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
    ///
    /// ⛔⛔ ITS EMPTY IS THE `primaryDsInfo_.at(dsType_)` THROW AND IS NOW FILED — the SAME throw
    /// [`Dsc2Store::transfer_lds`] in this file panics on (`dsc/dsc2.cpp:4077`), which is why the two
    /// must not answer one fact two ways. `stick_dims_of` is [`None`] for an lds the list does not
    /// hold OR a `dsType_` `primaryDsInfo_` has no row for; a labelled DS whose row IS there and
    /// carries no stick dim is a true empty and still returns one.
    ///
    /// ⛔ AN EMPTY IS AGAIN THE WRONG BRANCH: entry 372's 4-byte splat read matches
    /// `ds_stick_dims(lds).0.as_slice()` against exactly ONE `(dim, _)` and declines otherwise
    /// (`ddc/transformation.rs:2905-2907`), and entries 106/107 hand it to `stick_sizes(..,
    /// StickPart::Whole)` (`:1365`, `:1402-1403`), which then measures a stick of no elements.
    fn ds_stick_dims(&self, lds: LdsIdx) -> StickDims {
        let held = self
            .dsc_facts()
            .with_dsc(|dsc| ddc_state::stick_dims_of(dsc, lds));
        held.unwrap_or_else(|| {
            self.refuse(
                "DsSticks::ds_stick_dims: primaryDsInfo_.at(that labelled DS's dsType_) throws \
                 (dsc/dsc2.cpp:4077), so its stickDimOrder_/stickSize_ are UNKNOWN and not empty — \
                 an empty declines the 4B splat and measures a stick of no elements",
            )
            .unwrap_or_default()
        })
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

    /// ⛔ `getBlockTransferSize(*transferNode, src_.unit_, clId, false, true)`
    /// (`dsc/designSpaceConfig.h:225`) — a `DesignSpaceConfig` accessor over the transfer's layout and
    /// the DSC's stick sizes, and the SAME seam `conv::DdlSizes::block_transfer_size` and
    /// `v1::ExploreTree::block_transfer_sizes` stop on. ⛔ UNPORTED, and it belongs to crustify: it is
    /// a `dsc/` function outside this campaign's file list, not a fact this carrier is withholding.
    fn block_transfer_loads(&self, _transfer: NodeId, _corelet: Corelet) -> tr::Loads {
        todo!(
            "tr::TransferLoads::block_transfer_loads: wants \
             getBlockTransferSize(transfer, src_.unit_, clId, false, true) \
             (dsc/designSpaceConfig.h:225) — a `dsc/` accessor over the live super-DSC's stick \
             sizes, UNPORTED and outside this campaign's file list"
        )
    }
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
// THE TRANSFER BODY, READ-MODIFY-WRITTEN — ⭐ ANSWERED, because a TRANSFER is a kind this tree holds
// and [`crate::schedule::dsc2::TransferNode`] is its whole body.
// ════════════════════════════════════════════════════════════════════════════════════════════════

impl tr::FixedSizeTransfers for Dsc2Store<'_, '_> {
    /// ⛔ `metadata.dataConnects_.find(dc)->second.consumers_` AS WHAT EACH CONSUMER IS — a
    /// [`tr::ScopeNode`] per consumer.
    ///
    /// ⛔⛔ A RECORDED CORRECTION: THIS REFUSED BECAUSE *"its Compute arm needs a COMPUTE node, which
    /// `super::tree::Kind` has no arm for"*, AND THAT IS FALSE — [`super::tree::Kind::Compute`] holds
    /// the whole node and [`tr::ScopeTree::scope_node`] in the sibling file ALREADY answers that arm
    /// off it. Classifying a consumer is not what blocks this.
    ///
    /// ⛔ WHAT BLOCKS IT IS THE CENSUS: `metadata.dataConnects_` (`ddc/ddc_metadata.h:194`) is a map
    /// no carrier owns, and `consumers_` (`:145`) is the list inside it — the
    /// [`crate::schedule::ddc::metadata::Metadata`] is `run_v1`'s own local and is not one of
    /// `Dsc2Carriers`' borrows, the same seam [`tu::FifoResults::connect_consumers`] and
    /// [`tu::TransferMoves::producer_loops`] name. ⛔ AND AN EMPTY LIST IS NOT ITS ABSENCE: the caller
    /// sizes a transfer from what consumes its connect, so *"nothing consumes this"* is a different
    /// transfer, not a narrower one.
    fn connect_consumers(&self, _connect: DataConnect) -> Vec<tr::ScopeNode> {
        todo!(
            "tr::FixedSizeTransfers::connect_consumers: wants \
             metadata.dataConnects_.find(dc)->second.consumers_ as ScopeNodes \
             (ddc/ddc_metadata.h:194, the list at :145) — the Metadata's own census, which is \
             run_v1's local and no carrier's borrow. NOT the Compute arm: Kind::Compute holds the \
             node and ScopeTree::scope_node already classifies one"
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
    ///
    /// ⭐ AND IT IS THE *SAME* STOP — [`Dsc2Store::compute_body`], which
    /// [`tr::ComputeNodes::compute`] and [`tu::ComputeCloning::compute`] also read. This method used
    /// to spell its own `panic!` beside theirs; three statements of one `static_cast` are three
    /// places it can drift.
    fn compute(&self, node: NodeId) -> ComputeNode {
        self.compute_body(node)
    }

    /// `getStickDims(lds)` — ⛔ THE SAME `primaryDsInfo_.at(dsType_)` THROW
    /// [`tr::DsSticks::ds_stick_dims`] states, filed for the same reason: entry 264 matches this
    /// against exactly ONE `[dim]` to find the broadcast dim and answers [`None`] otherwise
    /// (`ddc/v1.rs:1145-1148`), and four other readers feed it to `cumulative_stick_sizes`, so an
    /// unanswerable position reads as *"this labelled DS has no sticks"*.
    fn stick_dims(&self, lds: LdsIdx) -> Vec<PrimaryDim> {
        let held = self
            .dsc_facts()
            .with_dsc(|dsc| ddc_state::stick_dims_of(dsc, lds));
        match held {
            Some(dims) => dims.0.iter().map(|(dim, _)| *dim).collect(),
            None => self
                .refuse(
                    "LoopOffsets::stick_dims: primaryDsInfo_.at(that labelled DS's dsType_) throws \
                     (dsc/dsc2.cpp:4077), so its stickDimOrder_ is UNKNOWN and not empty",
                )
                .unwrap_or_default(),
        }
    }

    /// `node->getOwnerLoop()`.
    fn owner_loop(&self, node: NodeId) -> Option<tr::LoopId> {
        tu::ScheduleSurgery::owner_loop(self, node)
    }

    /// ⛔⛔ THE FIELD IS DECLARED AND STILL NOTHING WRITES THE NODE'S COPY — AND THAT IS THE FINDING,
    /// not a gap in this carrier. `loopEleOffsets_` IS
    /// [`crate::schedule::dsc2::DataInfo::loop_ele_offsets`] (`schedule/dsc2.rs:433`), as
    /// `constEleOffsets_` is `:428`; what has no writer is the copy ON THE TRANSFER. Entry 260
    /// (`v1::fill_data_info`, `ddc/v1.rs:2535`) hands both maps to a SINK as a [`v1::DataInfoFill`]
    /// (`ddc/v1.rs:2203`) instead of onto the node, so the reference's read-back of an offset it
    /// wrote earlier still has nothing to read, and answering EMPTY here would make entry 133's
    /// restickify fixup silently find no offsets to adjust.
    fn src_loop_ele_offsets(
        &self,
        _node: NodeId,
        _corelet: Corelet,
    ) -> Vec<(tr::LoopId, Vec<PrimaryDim>)> {
        todo!(
            "v1::LoopOffsets::src_loop_ele_offsets: wants \
             srcLdsAndLoopOffsets_.loopEleOffsets_.at(corelet) (dsc/dsc2.h:730-734) — the field IS \
             dsc2::DataInfo::loop_ele_offsets (schedule/dsc2.rs:433) and NOTHING writes the node's \
             copy: entry 260 hands the map to a sink as v1::DataInfoFill (ddc/v1.rs:2203) instead \
             of onto the transfer"
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
             src_loop_ele_offsets: the map is DECLARED on dsc2::DataInfo (schedule/dsc2.rs:433) and \
             written only into a v1::DataInfoFill"
        )
    }

    /// ⛔ THE SAME SPLIT ON THE COMPUTE SIDE — `inputsLdsAndLoopOffsets_` (`dsc/dsc2.h:937`) IS
    /// [`crate::schedule::dsc2::ComputeNode::inputs`], so this is NOT *"no Compute arm"*, and it is
    /// no longer the field either: `loopEleOffsets_` (`dsc/dsc2.h:730-734`) is
    /// [`crate::schedule::dsc2::DataInfo::loop_ele_offsets`]. What is left is the WRITE DOOR alone.
    /// See [`Self::src_loop_ele_offsets`].
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
             loopEleOffsets_ (dsc/dsc2.h:730-734) IS dsc2::DataInfo::loop_ele_offsets and the \
             Compute arm is present; what is missing is the compute WRITE door"
        )
    }
}

impl tu::FifoResults for Dsc2Store<'_, '_> {
    /// ⛔ `metadata.dataConnects_[connect].consumers_` (`ddc/ddc_metadata.h:194`, the list at `:145`)
    /// as [`tu::FifoConsumer`]s — THE METADATA'S CENSUS, which `run_v1` holds as a local and does not
    /// hand to this carrier.
    ///
    /// ⛔⛔ A RECORDED CORRECTION: THIS ALSO SAID *"whose compute arm needs a COMPUTE node"*, AND IT
    /// DOES NOT — [`tu::FifoConsumer::Compute`] carries the node AND its whole
    /// [`crate::schedule::dsc2::ComputeNode`], and [`Dsc2Store::compute_of`] reads exactly that off
    /// [`super::tree::Kind::Compute`]. The classification is free; the map is the seam.
    ///
    /// ⛔ AND `operator[]` IS WHAT THE REFERENCE USES ON THE UNNAMED CONNECT, so `dataConnects_[""]`
    /// DEFAULT-CONSTRUCTS a real entry with no consumers — which is why this trait's own doc takes
    /// [`None`] as a connect and not as an error.
    fn connect_consumers(&self, _connect: Option<DataConnect>) -> Vec<tu::FifoConsumer> {
        todo!(
            "tu::FifoResults::connect_consumers: wants \
             metadata.dataConnects_[connect].consumers_ as FifoConsumers \
             (ddc/ddc_metadata.h:194, the list at :145) — the Metadata is run_v1's local and no \
             carrier's borrow. NOT the Compute arm: FifoConsumer::Compute carries the whole \
             dsc2::ComputeNode and Dsc2Store::compute_of reads it"
        )
    }

    /// ⛔ `computeNode->isOpaqueOp_` (`dsc/dsc2.h:941`) — ⛔⛔ A FIELD OUR
    /// [`crate::schedule::dsc2::ComputeNode`] DROPPED, and NOT *"no Compute arm"*: the arm holds the
    /// node, and this is the one bit of it this method wants.
    ///
    /// ⛔ AND IT MUST NOT BE INFERRED FROM `instrAttribute_`. An opaque body is the only thing that
    /// fills `input_data_connects`/`output_data_connects` and the two register maps, so those look
    /// like a proxy — but they are a CONSEQUENCE of opaqueness and `isOpaqueOp_` is the fact:
    /// `dsc/dsc2.h:941` is its own `bool` beside them, an opaque with no declared ports is
    /// spellable, and `false` here routes a FIFO result the reference repoints as an opaque input.
    /// ⭐ THE WORK IS THE FIELD, cited, plus `op_opaque`'s own `true`
    /// (`ddc/ddl/ddl_conversion.cpp` mints the node there).
    fn is_opaque(&self, _compute: NodeId) -> bool {
        todo!(
            "tu::FifoResults::is_opaque: wants computeNode->isOpaqueOp_ (dsc/dsc2.h:941) — a \
             dsc2::ComputeNode field OUR ComputeNode dropped; Kind::Compute holds the node. ⛔ Do \
             not infer it from instrAttribute_'s port lists: those are a consequence of opaqueness, \
             not the fact"
        )
    }

    /// ⛔ The transfer's two ends AS [`crate::schedule::ddc::fold::StoredStream`]s.
    ///
    /// ⛔⛔ THE STATED REASON WAS WRONG AND THE BLOCKER IS A TYPE, NOT A PROJECTION. This used to
    /// refuse because *"`getMutableAllocation(operand, storage)` reads `constantInfo_` … which
    /// `l3::dsc::DesignSpaceConfig` does not project"*: it DOES project it
    /// ([`crate::schedule::l3::dsc::ConstantInfo`]), [`Dsc2Store::allocation_at`] answers that call
    /// on both arms, and [`tu::TransferEnds`] does not perform the lookup at all — it carries the two
    /// ends and the ported unit looks them up.
    ///
    /// ⛔ WHAT ACTUALLY BLOCKS IT IS `StoredStream::storage`, WHICH IS A [`crate::units::DfirUnit`]
    /// WHILE AN [`Operand`]'s IS A `SenComponents` — the same mismatch
    /// [`crate::schedule::ddc::fold::Allocations::allocation`] names in the sibling file. [`DfirUnit`]
    /// is the DataflowIR-bound SUBSET and spells no register file: `PELRF`, `SFPLRF`, `PTARF`,
    /// `PTXRF`, `PTIRF` and `LRFREG` are all `dsc2::memories` a transfer end may live in and none has
    /// a `DfirUnit` arm — and a FIFO result is exactly the register case this unit is about, so the
    /// end that matters most is the one that cannot be spelled. `TransferEnds`' fields are not
    /// optional, so there is nowhere to put that absence either.
    fn transfer_ends(&self, _transfer: NodeId) -> tu::TransferEnds {
        todo!(
            "tu::FifoResults::transfer_ends: wants the transfer's ends as fold::StoredStreams, whose \
             `storage` is a units::DfirUnit while dsc2::Operand's is a SenComponent — DfirUnit \
             spells no register file (PELRF/SFPLRF/PTARF/PTXRF/PTIRF/LRFREG are dsc2::memories with \
             no arm), which is the very case a FIFO result lives in. NOT the constantInfo_ \
             projection: l3::dsc::ConstantInfo carries it and Dsc2Store::allocation_at reads it"
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

    /// ⛔ `computeConsumer->inputs_[i] = unit` (`dsc/dsc2.h:935`) — ⛔⛔ NOT *"no Compute arm"*, AND
    /// THE CITATION WAS 29 LINES OFF: this said `dsc/dsc2.h:906`, which is
    /// `InstrAttribute::indices_`. [`super::tree::Kind::Compute`] holds the node and
    /// [`crate::schedule::dsc2::ComputeNode::inputs`] IS `inputs_` zipped with
    /// `inputsLdsAndLoopOffsets_`, so the FACT is present and READABLE
    /// ([`Dsc2Store::compute_of`]).
    ///
    /// ⭐ ANSWERED through [`Dsc2Store::edit_compute`], the door this file's header used to call
    /// missing.
    ///
    /// ⛔ ONLY `inputs_` IS WRITTEN, WHICH IS WHY THE PARAMETER IS THE OPERAND'S `unit` AND NOT ITS
    /// `storage`: the reference assigns into a `std::vector<SenComponents>` and touches no
    /// `inputsLdsAndLoopOffsets_` entry here (`ddc/ddc_transformation_util.cpp:893`, `:1005`), so the
    /// paired [`Operand`]'s other two fields are carried over unchanged.
    ///
    /// ⛔ AN OUT-OF-RANGE `input` IS UNREACHABLE FROM THE PORT'S OWN CALLERS AND SO NOT A REFUSAL:
    /// both of them index by `compute.inputs.iter().enumerate()` on the very node they then write
    /// (`ddc/transformation_util.rs:1990-1997`, `:3325-3329`), which is the reference's own
    /// `for (i = 0, e = inputsLdsAndLoopOffsets_.size(); i < e; ++i)`. ⭐ AND THE THROW IT COULD
    /// OTHERWISE HAVE IS ALREADY UNSPELLABLE: `inputs_` and `inputsLdsAndLoopOffsets_` are two
    /// independent lengths there and ONE [`Operand`] list here, which is that type's own point.
    fn set_compute_input_unit(&mut self, compute: NodeId, input: usize, unit: SenComponent) {
        self.edit_compute(compute, |held| {
            if let Some(operand) = held.inputs.get_mut(input) {
                operand.unit = unit;
            }
        });
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

    /// `inputsLdsAndLoopOffsets_.at(i).latchDataId_ = id` (`dsc/dsc2.h:937`, `:725`) — ⭐ ANSWERED
    /// through [`Dsc2Store::edit_compute`].
    ///
    /// ⛔ THE `.at(i)` IS THE SAME INDEX [`tu::FifoResults::set_compute_input_unit`] JUST WROTE, from
    /// one enumeration of one operand list (`ddc/transformation_util.rs:1997-1999`), and the
    /// reference writes the two through the same `i` two statements apart
    /// (`ddc/ddc_transformation_util.cpp:1005-1007`) — so an absent position here is that method's
    /// unreachable case and not a second one.
    fn set_compute_input_latch_data_id(&mut self, compute: NodeId, input: usize, id: LatchDataId) {
        self.edit_compute(compute, |held| {
            if let Some(operand) = held.inputs.get_mut(input) {
                operand.data.latch_data_id = Some(id);
            }
        });
    }

    /// ⛔ `allocNode->removeAllocUser(user)` (`dsc/dsc2.h:1022` — ⛔ CITATION CORRECTED, `:1024` is a
    /// line INSIDE its body and `:1012` is `addAllocUser`) — [`super::tree::Org`] has `add_user` and
    /// no remove, and that function's abort *"RemoveAllocUser: Schedule node <n> is not in the user
    /// list of allocate node <a>"* (`:1032`) is [`tu::AllocationUse`] being unconstructible, which a
    /// silent no-op would swallow.
    fn remove_alloc_use(&mut self, _alloc_use: tu::AllocationUse) {
        todo!(
            "tu::SkipRegResults::remove_alloc_use: wants allocNode->removeAllocUser(user) \
             (dsc/dsc2.h:1022, its abort at :1032) — super::tree::Org has add_user and no remove, \
             and its abort for a user not in the list would be swallowed by a no-op"
        )
    }

    /// `computeNode->outputs_.resize(1)` (`dsc/dsc2.h:936`) THEN `at(0) = unit` alongside
    /// `outputsLdsAndLoopOffsets_.resize(1)` and its own `at(0) = data` (`:938`) — ⭐ ANSWERED
    /// through [`Dsc2Store::edit_compute`]; both halves are one
    /// [`crate::schedule::dsc2::ComputeNode::outputs`] entry, which is why the two `resize`s cannot
    /// leave the vectors different lengths (`ddc/ddc_transformation_util.cpp:1075-1077`, `:1086`).
    ///
    /// ⛔ THE OPERAND'S `storage` IS CARRIED OVER, NOT WRITTEN, AND THAT IS NOT A DROPPED EFFECT:
    /// `outputs_` is a `std::vector<SenComponents>` with NO storage half, so a compute operand's
    /// `storage` has no `ComputeNode` counterpart to be faithful to — the reference writes the
    /// TRANSFER destination's `loc_.storage_` into `outputs_`, i.e. into this operand's `unit`, which
    /// is what [`tu::SkipRegResults`]' own doc means by *"`unit` is an `outputs_` entry"*. A slot
    /// that did not exist takes `NO_COMPONENT`, the `DataLocation` default (`arch_enums.h:390-391`).
    fn set_sole_compute_output(&mut self, compute: NodeId, unit: SenComponent, data: DataInfo) {
        self.edit_compute(compute, |held| {
            // `resize(1)` — every output past the first is DROPPED, which is the trait's own ⚠️.
            held.outputs.truncate(1);
            match held.outputs.first_mut() {
                Some(operand) => {
                    operand.unit = unit;
                    operand.data = data;
                }
                None => held.outputs.push(Operand {
                    unit,
                    storage: SenComponent::NoComponent,
                    data,
                }),
            }
        });
    }

    /// `computeNode->inputs_.resize(i + 1)` (`dsc/dsc2.h:935`) THEN `at(i) = unit` with
    /// `inputsLdsAndLoopOffsets_.resize(i + 1)` and its `at(i) = data` (`:937`) — ⭐ ANSWERED, the
    /// same zipped pair (`ddc/ddc_transformation_util.cpp:1083-1088`).
    ///
    /// ⛔⛔ THE GAP FILL IS `HBM`, NOT `NO_COMPONENT` — A CORRECTION TO [`tu::SkipRegResults`]' OWN
    /// DOC, which says *"a gap below it is `NO_COMPONENT` against a default-constructed `DataInfo`"*.
    /// `std::vector<SenComponents>::resize` VALUE-INITIALISES the elements it appends, and
    /// `SenComponents : int` opens `NO_COMPONENT = -1, HBM = 0`
    /// (`sys-arch-spec/arch_enums.h:13-15`) — so a value-initialised entry is `HBM`, and
    /// `NO_COMPONENT` is the one value `resize` cannot produce. The [`DataInfo`] half of the doc is
    /// right: `std::vector<DataInfo>::resize` default-constructs, which is [`DataInfo::default`].
    ///
    /// ⛔ AND NO GAP IS EVER PRODUCED IN PRACTICE, which is why this is reproduced rather than
    /// refused: the reference's ONLY callsite passes `computeInputIndex = 0` on a `new
    /// dsc2::ComputeNode()` whose `inputs_` is empty (`ddc/ddc_transformation.cpp:1833-1841`), so the
    /// resize only ever grows 0 → 1. The port's caller passes the same `InputIdx(0)`
    /// (`ddc/transformation.rs:2991`).
    fn resize_compute_inputs_to(
        &mut self,
        compute: NodeId,
        input: tu::InputIdx,
        unit: SenComponent,
        data: DataInfo,
    ) {
        self.edit_compute(compute, |held| {
            // ⭐ THE SLOT'S OWN `storage` SURVIVES A RE-WRITE, for the reason
            // [`Self::set_sole_compute_output`] states: `inputs_` has no storage half to overwrite.
            let held_storage = held.inputs.get(input.0).map(|operand| operand.storage);
            // `resize(i + 1)`, both ways: inputs at and past `i` are dropped, a gap below it is
            // value-initialised.
            held.inputs.truncate(input.0);
            held.inputs.resize(
                input.0,
                Operand {
                    unit: SenComponent::Hbm,
                    storage: SenComponent::NoComponent,
                    data: DataInfo::default(),
                },
            );
            held.inputs.push(Operand {
                unit,
                storage: held_storage.unwrap_or(SenComponent::NoComponent),
                data,
            });
        });
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
    /// ⛔ `storageOrDatastreamIsExternal(dataInfo, storage, isIncoming)`
    /// (`ddc/ddc_transformation_util.cpp:1651-1683`).
    ///
    /// ⛔⛔ THERE IS NO `externalDataStreams_` — THAT FIELD DOES NOT EXIST. This used to refuse
    /// naming it as *"a `DesignSpaceConfig` field `l3::dsc` does not project"*; the spelling has ZERO
    /// occurrences in the whole reference. The function reads exactly two things, and neither is it:
    ///
    ///   * `getAllocation(dataInfo, storage, true)` then `isExternalNode(allocNode)` (`:1657-1658`) —
    ///     ⭐ THE FIRST HALF IS ANSWERED, by [`Dsc2Store::allocation_at`];
    ///   * `metadata.dataConnects_.at(dc)`'s `producers_` or `consumers_`, each tested against
    ///     `externalNodes_` (`:1668-1681`).
    ///
    /// ⛔ SO THE BLOCKER IS THE `Metadata`, AND IT IS ALREADY PORTED ELSEWHERE:
    /// [`tu::storage_or_datastream_is_external`] IS entry 120
    /// (`ddc/transformation_util.rs:1290`), it takes `&Metadata` and a
    /// [`tu::CensusNodes`], and this trait's own doc says a carrier *"states [`tu::CensusNodes`]
    /// beside its allocations and delegates"*. `Dsc2Store` is handed no `Metadata` — it is `run_v1`'s
    /// own local — so the one line that would answer this cannot be written yet. ⛔ AND A `false`
    /// HERE WOULD NOT BE CONSERVATIVE: entries 253-256/305/306/341/361 take it as *"this stream is
    /// internal, transform it"*, so it would transform the external nodes another stage already
    /// filled.
    fn storage_or_datastream_is_external(
        &self,
        _data: &DataInfo,
        _storage: SenComponent,
        _direction: tu::StreamDirection,
    ) -> bool {
        todo!(
            "tu::ExternalStreams::storage_or_datastream_is_external: entry 120 is ALREADY PORTED as \
             tu::storage_or_datastream_is_external (ddc/transformation_util.rs:1290) and this impl \
             need only delegate — but it takes &Metadata for externalNodes_/dataConnects_ \
             (ddc/ddc_transformation_util.cpp:1657-1681) and Dsc2Store is handed none; the Metadata \
             is run_v1's own local. NOT `externalDataStreams_`, which does not exist"
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
        let alloc = self.allocation_at(&dst.data, dst.storage)?;
        self.with_tree(|tree| tree.node_of_alloc(alloc))
    }

    /// `condNode->clone()` — a DETACHED copy with its regions still to be added. ⭐ ANSWERED through
    /// [`Dsc2Store::clone_condition_node`], which is the minter this note asked for.
    ///
    /// ⛔⛔ THE OBJECTION WAS WRONG AND THE AUTHORITY SETTLES IT OUTRIGHT. This used to refuse saying
    /// *"cloning one means deciding what the copy's regions name, which is entry 249's decision and
    /// not this carrier's"*. IT IS NOT A DECISION: `BlockNode::next_` is a
    /// `VectorOfChildren`, whose COPY CONSTRUCTOR IS `VectorOfChildren(const VectorOfChildren&) {}`
    /// — *"do nothing on purpose … when copying, it is up to the caller to manually insert copies of
    /// the children"* (`dsc/dsc2.h:529-537`). So `clone()` yields `loopCond_` and `coreClCond_`
    /// copied and `next_` EMPTY, and entry 249 fills both regions with fresh blocks immediately
    /// afterwards (`ddc/transformation_util.rs:3115-3119`).
    ///
    /// ⛔ WHAT BLOCKS IT IS THE MINTER, AND IT IS ONE LINE IN THE SIBLING FILE.
    /// [`Dsc2Store::new_core_cl_condition`] is the only `pub(super)` condition minter and it hardcodes
    /// `loop_cond: None` / `cores: Some(..)`. Every condition entry 249 clones is LOOP-guarded (they
    /// come out of its `loop_conditions` list and it negates their `loopCond_` two statements later),
    /// so the clone must carry `cores: None` — and `Some(CoreClSet::default())` is NOT the same
    /// state: [`v1::ConditionSimplification::has_core_cl_cond`] is `core_cl_cond(..).is_some()` in the
    /// sibling file, so an empty-but-present set would answer *"this is core/corelet-guarded"* for a
    /// loop-guarded clone. ⛔ THE FIX WAS A `clone_condition_node` BESIDE `new_core_cl_condition`
    /// that copies the whole [`super::tree::Cond`] with both regions empty, and that is what
    /// [`Dsc2Store::clone_condition_node`] now is — `cores` COPIED, so a loop-guarded clone keeps
    /// [`None`] there and [`v1::ConditionSimplification::has_core_cl_cond`] keeps answering `false`
    /// for it.
    fn clone_condition(&mut self, condition: NodeId) -> NodeId {
        self.clone_condition_node(condition)
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

    /// `new dsc2::ComputeNode()` UNPARENTED — ⭐ ANSWERED through [`Dsc2Store::mint_compute`], the
    /// same `TreeData::add`-with-no-parent [`super::ddc_sites`]' `conv::ScheduleWrites::mint_compute`
    /// makes for the DDL expansion (`stages/ddc_sites.rs:1424`).
    ///
    /// ⛔ DETACHED IS THE POINT AND NOT AN OMISSION: the reference mints the node, sets its
    /// `exUnit_`/`name_`/`type_`/`dataFormat_` and only then hands it to
    /// `insertComputeBetweenTransferAndReg`, which `DT_CHECK(computeNode->getPrev() == nullptr)`
    /// before splicing it (`ddc/ddc_transformation.cpp:1833-1841`,
    /// `ddc/ddc_transformation_util.cpp:1035`). That check is
    /// [`tu::UnplacedNode`](crate::schedule::ddc::transformation_util::UnplacedNode) here, which reads
    /// `parent(node).is_none()` — so linking it would make the witness unconstructible.
    fn mint_compute(&mut self, node: ComputeNode) -> NodeId {
        let name = node.name.clone();
        Dsc2Store::mint_compute(self, name, node, None)
    }

    /// The tree's own next free allocation identity.
    fn free_alloc(&self) -> AllocId {
        self.next_free_alloc()
    }
}

impl tr::SpreadTransfers for Dsc2Store<'_, '_> {
    /// ⛔ `allocNode->gapStickSpread_.empty()` negated (`dsc/dsc2.h:1006`) — a `ddc`-view
    /// `AllocateNode` field [`crate::schedule::l3::dl_ops::L3AllocateNode`] does not carry.
    ///
    /// ⛔⛔ TWO CORRECTIONS, AND THE SECOND ONE MATTERS: THE FIELD IS **NOT DEAD**. This used to say
    /// *"entry 300 (its only writer) has its single callsite inside `#if 0`
    /// (`ddc/ddcv1.cpp:3735-3752`), so nothing in this pipeline ever fills it"*. That `#if 0` block is
    /// real, but it wraps `packStickDim()` and the `peSfpSplit_` clearing — NOT this field's writer.
    /// `gapStickSpread_` has FOUR live writers:
    ///
    ///   * `ddc/ddc_transformation.cpp:1380` — entry 108's `alloc->gapStickSpread_[dim] = spread`,
    ///     reached from `cloneForOffsetAdjustment()` at `ddc/ddcv1.cpp:3732`, one line ABOVE the
    ///     `#if 0`; it is [`tr::OffsetAdjustment::set_gap_stick_spread`] in this same file;
    ///   * `ddc/ddc_transformation.cpp:1132-1135` — three more, on an intrinsic's in/out/internal
    ///     allocations;
    ///   * `ddc/ddcv1.cpp:1704` — `gapStickSpread_[layoutDimOrder_.at(0)] = 8`.
    ///
    /// ⛔ AND IT SCALES EXTENTS, so a dropped one is not a dropped annotation: `dsc/dsc2.cpp:3958-3961`
    /// DIVIDES each `sizePerDim` entry by the spread and `:2882-2892` MULTIPLIES
    /// `unitView.sizesNoGaps_` and each loop's `elemOffset_` by it, and `dsc/dsc2.cpp:926` puts it on
    /// the wire. ⛔ SO A `false` HERE WOULD BE A FABRICATED EXTENT: entry 341 reads it to decide
    /// whether a transfer's two ends agree, and answering *"no spread"* for an allocation entry 108
    /// spread would move the wrong element count.
    fn has_gap_stick_spread(&self, _alloc: AllocId) -> bool {
        todo!(
            "tr::SpreadTransfers::has_gap_stick_spread: wants allocNode->gapStickSpread_ \
             (dsc/dsc2.h:1006) — a ddc-view AllocateNode field l3::dl_ops::L3AllocateNode does not \
             carry, and it is LIVE: four writers (ddc/ddc_transformation.cpp:1380 is entry 108's, \
             reached from ddcv1.cpp:3732; :1132-1135; ddcv1.cpp:1704) and it scales extents at \
             dsc/dsc2.cpp:3958-3961 and :2882-2892"
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
    /// [`tr::PeSfpSplitNode`]s.
    ///
    /// ⛔⛔ THE COMPUTE ARM IS NO LONGER WHAT BLOCKS THIS — all three kinds are kinds this tree holds,
    /// and the walk is one DFS over [`super::tree::TreeData::dfs`]. What is missing is the ALLOCATE
    /// arm's SECOND field: [`tr::PeSfpSplitNode::Allocate`] carries a
    /// [`tr::SkipMetadataUpdate`], which is `allocNode->tempStorageForCompute_ != nullptr`
    /// (`dsc/dsc2.h:978`), and [`crate::schedule::l3::dl_ops::L3AllocateNode`] does not carry that
    /// pointer — the same field [`tu::AllocateCloning::set_temp_storage_for_compute`] stops on in the
    /// sibling file. ⛔ AND `SkipMetadataUpdate::No` WOULD NOT BE A SAFE DEFAULT: it is what makes
    /// entry 118 REGISTER the clone in `metadata.newAllocations_`, so guessing it for a temp-storage
    /// allocation would file an allocation the reference deliberately keeps out of the tracker.
    fn split_candidates(&self) -> Vec<tr::PeSfpSplitNode> {
        todo!(
            "tr::PeSfpWorkSplit::split_candidates: the {{ALLOCATE, TRANSFER, COMPUTE}} walk is now \
             one DFS over kinds this tree holds; what is missing is PeSfpSplitNode::Allocate's \
             SkipMetadataUpdate, which is allocNode->tempStorageForCompute_ != nullptr \
             (dsc/dsc2.h:978) and l3::dl_ops::L3AllocateNode does not carry it"
        )
    }
}

impl tr::OffsetAdjustment for Dsc2Store<'_, '_> {
    /// ⛔ `node->repetitionWithOffset_.forOutputs_` (`dsc/dsc2.h:954`, the struct at `:950-953`) — ⛔⛔
    /// A FIELD OUR [`crate::schedule::dsc2::ComputeNode`] DROPPED, and NOT *"no Compute arm"*: the
    /// arm holds the node and this is the one field of it entry 108 reads that is not there.
    ///
    /// ⛔ AN EMPTY VECTOR IS THE ONE ANSWER THAT MUST NOT BE GIVEN: entry 108 loops `1..spread` over
    /// each entry, so empty means *"no output repeats"* and the whole pass does nothing — no clones,
    /// no `gapStickSpread_`, and the spread that scales extents at `dsc/dsc2.cpp:3958` never
    /// happens. ⭐ THE WORK IS `RepetitionWithOffset` ON `dsc2::ComputeNode` (`forInputs_` and
    /// `forOutputs_`, both `std::vector<int>`), plus the DDL conversion's own default.
    fn output_repetitions(&self, _node: NodeId) -> Vec<tr::Repetition> {
        todo!(
            "tr::OffsetAdjustment::output_repetitions: wants \
             node->repetitionWithOffset_.forOutputs_ (dsc/dsc2.h:954) — a dsc2::ComputeNode field \
             OUR ComputeNode dropped; Kind::Compute holds the node. An empty vector would make \
             entry 108 do nothing at all"
        )
    }

    /// `static_cast<dsc2::ComputeNode *>(node->clone())` then
    /// `node->getMutableParent()->addChildNode(newNode, false, node)`
    /// (`ddc/ddc_transformation.cpp:1362-1364`) — ⭐ ANSWERED as TWO calls that already exist:
    /// [`tu::ComputeCloning::compute`] reads the body through the one downcast
    /// [`Dsc2Store::compute_body`] is, and [`tu::ComputeCloning::clone_compute_after`] mints and
    /// splices it. ⛔ NOTHING IS RE-DERIVED HERE, so this cannot clone a compute differently from the
    /// way entry 340's own cloner does.
    ///
    /// ⛔ A PARENTLESS `node` IS RECORDED: the reference dereferences `getMutableParent()` and this
    /// tree's `link` no-ops, which would leave the clone out of the tree while the caller goes on to
    /// set its repetition, record it under the original and bump the allocation's users
    /// (`ddc/transformation.rs:645-648`).
    fn clone_compute_after(&mut self, node: NodeId) -> NodeId {
        if tu::ScheduleSurgery::parent(self, node).is_none() {
            let _: Option<()> = self.refuse(
                "OffsetAdjustment::clone_compute_after: the compute has no parent, so \
                 node->getMutableParent()->addChildNode(clone, false, node) has no block to insert \
                 into — the clone stays OUT of the tree",
            );
        }
        let body = tu::ComputeCloning::compute(self, node);
        tu::ComputeCloning::clone_compute_after(self, node, body)
    }

    /// ⛔ `repetitionWithOffset_.forOutputs_.at(idx) = reps` (`dsc/dsc2.h:954`) — BOTH halves are
    /// missing: the field ([`Self::output_repetitions`]) and the write door.
    fn set_output_repetition(&mut self, _node: NodeId, _idx: tr::OutputIdx, _reps: tr::Repetition) {
        todo!(
            "tr::OffsetAdjustment::set_output_repetition: wants \
             repetitionWithOffset_.forOutputs_.at(idx) = reps (dsc/dsc2.h:954) — the field is not on \
             our dsc2::ComputeNode (see output_repetitions) and there is no compute WRITE door"
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
    /// (`dsc/dsc2.h:936` — ⛔ CITATION CORRECTED, `:934` is `dataFormat_`) and the trait's own doc
    /// names it.
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

    /// ⛔ `alloc->gapStickSpread_[dim] = spread` (`ddc/ddc_transformation.cpp:1380`,
    /// `dsc/dsc2.h:1006`) — ⭐ THIS IS THE LIVE WRITER
    /// [`tr::SpreadTransfers::has_gap_stick_spread`]'s corrected note names, reached from
    /// `cloneForOffsetAdjustment()` (`ddc/ddcv1.cpp:3732`) and NOT from anything behind `#if 0`.
    /// [`crate::schedule::l3::dl_ops::L3AllocateNode`] does not carry the map.
    fn set_gap_stick_spread(&mut self, _alloc: AllocId, _dim: PrimaryDim, _spread: tr::StickSpread) {
        todo!(
            "tr::OffsetAdjustment::set_gap_stick_spread: wants allocNode->gapStickSpread_[dim] = \
             spread (ddc/ddc_transformation.cpp:1380, dsc/dsc2.h:1006) — a ddc-view field \
             l3::dl_ops::L3AllocateNode does not carry, and this callsite IS live"
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

    /// `computeOp_.back().interimLabeledDs.push_back(&labeledDs_[lds])`
    /// (`ddc/ddc_transformation.cpp:1909-1910`) — ⭐ ANSWERED through the shared `currDsc` cell, now
    /// that [`v1::DscComputeOp::interim`] projects `interimLabeledDs` (`dsc/dscdefn.h:508`).
    ///
    /// ⛔⛔ THE FIELD WAS THE WHOLE BLOCKER AND IT IS LOAD-BEARING RATHER THAN BOOKKEEPING — four live
    /// readers: `ddc/ddc_transformation_util.cpp:1842` and `ddc/ddl/ddl_conversion.cpp:523` walk it,
    /// `dsc/superdsc.cpp:1585` feeds it to `insertLdsIdx`, and `dsc/designSpaceConfig.cpp:6738-6741`
    /// puts it ON THE WIRE (with `:7495-7496` reading it back).
    ///
    /// ⭐ IT IS A PUSH AND `interimLabeledDs` HOLDS THE INDEX, NOT THE POINTER, for the reason
    /// [`v1::DscComputeOp::inputs`] states: all three operand lists are
    /// `std::vector<LabeledDsInfo*>` (`dsc/dscdefn.h:506-510`) over one `labeledDs_`, so the index
    /// those pointers carry IS the entry.
    ///
    /// ⛔ `.back()` ON AN EMPTY `computeOp_` IS UNDEFINED THERE AND THE INDEX IS ALREADY PROVED HERE:
    /// [`Self::sole_compute_op`] hands out [`crate::schedule::ddl::conversion::ComputeOpIdx`] only
    /// when `ops.len() == 1`, which is the `DT_CHECK(computeOp_.size() == 1)` one line above the push
    /// (`:1908`). An index the list does not hold is therefore a defect in this port and is FILED,
    /// not dropped — a silent no-op would leave the minted `autoshuffle_reg_` tensor off the op and
    /// so out of the SDSC JSON entirely.
    fn add_interim_lds(
        &mut self,
        compute_op: crate::schedule::ddl::conversion::ComputeOpIdx,
        lds: LdsIdx,
    ) {
        let pushed = self.dsc_facts().with_ops_mut(|ops| {
            ops.get_mut(compute_op.0).map(|op| op.interim.push(lds))
        });
        if pushed.is_none() {
            let _: Option<()> = self.refuse(
                "AutoShuffling::add_interim_lds: computeOp_ holds no op at that index, so \
                 interimLabeledDs has nothing to push onto — .back() is undefined there and the \
                 interim tensor would never reach the wire",
            );
        }
    }

    /// An [`AllocId`] no allocation carries yet.
    fn free_alloc(&self) -> AllocId {
        self.next_free_alloc()
    }

    /// ⛔ A minted `ddc` allocate node — see `tu::AllocateCloning::allocate`.
    fn insert_allocate(&mut self, _alloc: AllocId, _node: tu::DdcAllocateNode, _at: tu::InsertionPoint) {
        todo!(
            "tr::AutoShuffling::insert_allocate: wants the held DdcAllocateNode handed to the tree \
             (ddc/transformations/automatic_shuffle/shuffle.h:175, DataEdge::insert_before) — \
             super::tree::Kind::Allocate holds the L3 view"
        )
    }

    /// `allocNode->addAllocUser(user)`.
    fn add_alloc_user(&mut self, alloc: AllocId, user: NodeId) {
        self.add_user_to(alloc, user);
    }

    /// `new dsc2::ComputeNode(*assign)` PLACED BEFORE `assign` — ⭐ ANSWERED through
    /// [`Dsc2Store::mint_compute`], because the caller HANDS the whole
    /// [`crate::schedule::dsc2::ComputeNode`] in and [`tu::InsertionPoint::Before`] is a placement
    /// [`super::tree::TreeData::link`] already makes.
    ///
    /// ⭐ THE PARENT IS THE SIBLING'S, which is what naming the sibling already says: the reference
    /// spells it `assign_parent->addChildNode(packmerge_node, /*addBefore=*/true, assign)`
    /// (`ddc/ddc_transformation.cpp:1992`).
    ///
    /// ⛔ A `before` WITH NO PARENT IS RECORDED RATHER THAN SILENTLY DROPPED. There
    /// `assign_parent` is a captured pointer and the insert happens regardless; here
    /// [`super::tree::TreeData::link`] resolves the parent THROUGH the sibling and no-ops when it has
    /// none, which would leave the packmerge minted, returned, given alloc users — and in no block.
    fn insert_compute_before(&mut self, node: ComputeNode, before: NodeId) -> NodeId {
        if tu::ScheduleSurgery::parent(self, before).is_none() {
            let _: Option<()> = self.refuse(
                "AutoShuffling::insert_compute_before: the node to insert before has no parent, so \
                 assign_parent->addChildNode(node, true, assign) has no block to insert into — the \
                 minted packmerge stays OUT of the tree",
            );
        }
        let name = node.name.clone();
        Dsc2Store::mint_compute(self, name, node, Some(tu::InsertionPoint::Before(before)))
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
    fn operand_element_bits(&self, dinfo: &DataInfo) -> crate::formats::Bits {
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
    /// `primaryDsInfo_.at(dsType_).stickRepl_`
    /// (`ddc/transformations/automatic_shuffle/shuffle.cpp:805-813`), which
    /// [`crate::schedule::l3::dsc::PrimaryDsInfo`] projects as neither of its two fields and which
    /// scratchy emits nowhere — `primaryDsInfo_` on the wire is `layoutDimOrder_`/`stickDimOrder_`/
    /// `stickSize_` and nothing else.
    ///
    /// ⛔ AND AN EMPTY LIST WOULD NOT BE ITS ABSENCE: `all_one` (entry 159) is
    /// [`crate::schedule::ddc::shuffle::AutoShuffler::infer_layouts`]' own `DT_CHECK`
    /// (`ddc/transformations/automatic_shuffle/shuffle.cpp:1043-1045`), which an empty `stickRepl_`
    /// PASSES — so a fresh empty vector here would assert *"nothing is replicated"* for an operand
    /// that may well be, and it would compile.
    fn operand_sticks(
        &self,
        _dinfo: &DataInfo,
    ) -> Option<crate::schedule::ddc::shuffle::OperandSticks> {
        todo!(
            "tr::AutoShuffling::operand_sticks: wants primaryDsInfo_.at(dsType_).stickRepl_ \
             (ddc/transformations/automatic_shuffle/shuffle.cpp:805-813), which \
             l3::dsc::PrimaryDsInfo does not project and scratchy's primaryDsInfo_ does not emit — \
             an EMPTY stickRepl_ PASSES all_one's DT_CHECK, so it would assert 'nothing is \
             replicated' for an operand that may be"
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

    /// ⛔ `node->{inputs,outputs}LdsAndLoopOffsets_[pos].constEleOffsets_ = offsets`
    /// (`ddc/transformations/automatic_shuffle/shuffle.cpp:881-896`) — ⛔⛔ NOT *"no Compute arm"*:
    /// the arm holds the node and `{inputs,outputs}LdsAndLoopOffsets_` (`dsc/dsc2.h:937-938`) are
    /// [`crate::schedule::dsc2::ComputeNode`]'s own `inputs`/`outputs`. AND NOT THE FIELD EITHER:
    /// `constEleOffsets_` (`dsc/dsc2.h:727-729`, per core AND corelet) IS
    /// [`crate::schedule::dsc2::DataInfo::const_ele_offsets`] (`schedule/dsc2.rs:428`). THE COMPUTE
    /// WRITE DOOR IS THE ONE HALF LEFT.
    fn set_const_ele_offsets(
        &mut self,
        _node: NodeId,
        _at: OperandPos,
        _offsets: crate::schedule::ddc::shuffle::ConstEleOffsets,
    ) {
        todo!(
            "tr::AutoShuffling::set_const_ele_offsets: wants \
             node->{{inputs,outputs}}LdsAndLoopOffsets_[pos].constEleOffsets_ = offsets \
             (ddc/transformations/automatic_shuffle/shuffle.cpp:881-896) — constEleOffsets_ \
             (dsc/dsc2.h:727-729) IS dsc2::DataInfo::const_ele_offsets (schedule/dsc2.rs:428) and \
             the Compute arm is present; what is missing is the compute WRITE door"
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
    /// [`crate::schedule::ddc::fold::CapturedNode`]s.
    ///
    /// ⛔⛔ NOT THE COMPUTE ARM — all three kinds are kinds this tree holds. What is missing is the
    /// COMPUTE arm's COORDINATES: it carries `inputs: Vec<Coordinate>` for `inputCoordinates_`
    /// (`dsc/dsc2.h:948`) and `output: Coordinate` for `outputCoordinate_` (`:949`), which are PER
    /// OPERAND, while [`super::tree::TreeData::coordinate`] holds exactly ONE coordinate per node —
    /// the shape an `allocateCoordinates_` or a `transferCoordinates_` needs. ⛔ A ONE-ENTRY `inputs`
    /// OR A DEFAULT `output` WOULD BE A FABRICATED FOLD COORDINATE, which is what entry 375 reports
    /// against.
    ///
    /// ⛔ AND THIS SITS BEHIND [`crate::schedule::ddc::fold::FoldConstruction::build_and_propagate_fold`]
    /// ANYWAY: nothing has built the fold space these coordinates would come out of.
    fn captured_nodes(&self) -> Vec<crate::schedule::ddc::fold::CapturedNode> {
        todo!(
            "fold::CoordinateCapture::captured_nodes: the {{ALLOCATE, COMPUTE, TRANSFER}} walk is \
             over kinds this tree holds; what is missing is the Compute arm's PER-OPERAND \
             inputCoordinates_ (dsc/dsc2.h:948) and outputCoordinate_ (:949) — TreeData::coordinate \
             holds one per NODE — and buildAndPropagateFold has not run"
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

    /// `getTransferType()` (`dsc/dsc2.h:884-896`) WITH THE LABELLED DS THE MATCHING SIDE NAMES — ⭐
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
    /// ⭐⭐ THE CLASSIFICATION IS NOT RE-DERIVED HERE — it is
    /// [`crate::schedule::dsc2::TransferNode::transfer_kind`], which IS `getTransferType()` already
    /// ported, five arms and its `Invalid` sixth. Restating those five predicates in this carrier
    /// would be a SECOND `getTransferType` that could disagree with the first; this method's own job
    /// is only *which side's labelled DS goes with the answer*, and the reference states that
    /// separately as the ternary `((constantToTensor || noTransToTensor) ?
    /// dstLdsAndLoopOffsets_.front().myLdsIdx_ : srcLdsAndLoopOffsets_.myLdsIdx_)`
    /// (`ddc/ddcv1.cpp:481-484`).
    ///
    /// ⛔ AND THE BOTH-INDICES-SET `DT_CHECK` IS DELIBERATELY NOT GUARDED, because
    /// [`crate::schedule::dsc2::DataInfo::is_labeled_ds`] has already settled it: *"the reference
    /// stops, we do not, and no caller distinguishes"*. A guard here would be this port disagreeing
    /// with itself about one field.
    ///
    /// # ⛔ TWO STOPS
    ///
    /// * **`INVALID_TRANSFER_TYPE`** has no arm, which is [`v1::TransferOperands`]' own statement
    ///   (*"five arms and no sixth"*), and entry 126's `DT_CHECK_MSG(.., "Unexpected transfer
    ///   type.")` (`ddc/ddcv1.cpp:479`) is that same abort reached from the other side. The
    ///   `ConstantToConstant`-without-a-`constantId_` case folds into it: `transfer_kind` answers
    ///   that arm only for `src.is_constant()`, so a missing id there is a defect in this port and
    ///   not an input.
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
        use crate::schedule::dsc2::TransferKind;

        let held = tu::ScheduleSurgery::transfer(self, transfer);
        let kind = held.transfer_kind();
        // ⭐ THE REFERENCE'S OWN TERNARY, AND NOTHING ELSE. Each arm's predicate already proved the
        // index it names PRESENT, so [`None`] here is unreachable for the four tensor arms — and it
        // is still the right spelling for them, because it is exactly the `ldsIdx < 0` skip
        // [`v1::TransferOperands`] keeps the [`Option`] for.
        let named = match kind {
            TransferKind::ConstantToTensor | TransferKind::NoTransferToTensor => {
                held.dsts.first().data.my_lds_idx
            }
            TransferKind::TensorToTensor | TransferKind::NoTransferFromTensor => {
                held.src.data.my_lds_idx
            }
            TransferKind::ConstantToConstant | TransferKind::Invalid => None,
        }
        .map(|lds| self.transfer_lds(lds, transfer));

        match (kind, held.src.data.constant_id) {
            (TransferKind::ConstantToConstant, Some(constant)) => {
                v1::TransferOperands::ConstantToConstant(self.constant_data(constant, transfer))
            }
            (TransferKind::ConstantToTensor, _) => v1::TransferOperands::ConstantToTensor(named),
            (TransferKind::TensorToTensor, _) => v1::TransferOperands::TensorToTensor(named),
            (TransferKind::NoTransferToTensor, _) => {
                v1::TransferOperands::NoTransferToTensor(named)
            }
            (TransferKind::NoTransferFromTensor, _) => {
                v1::TransferOperands::NoTransferFromTensor(named)
            }
            (TransferKind::Invalid | TransferKind::ConstantToConstant, _) => panic!(
                "v1::Dsc2Store::transfer_operands: {transfer:?} is INVALID_TRANSFER_TYPE — neither \
                 end names a labelled DS or a constant, which is entry 126's own \
                 DT_CHECK_MSG(\"Unexpected transfer type.\") (ddc/ddcv1.cpp:479); a \
                 CONSTANT_TO_CONSTANT reaching this arm means transfer_kind answered it for an \
                 unset src constantId_"
            ),
        }
    }

    /// ⛔ `traverseTreeDFSMutable(nullptr, {COMPUTE, TRANSFER})` as entry 002 censuses it
    /// (`ddc/ddcv1.cpp:3283-3317`).
    ///
    /// ⛔⛔ THE OLD *"no Compute arm"* REASON IS GONE: the arm exists, and the DDL conversion DOES
    /// fill [`crate::schedule::dsc2::DataInfo::data_connect`] — `ddl/conversion.rs:2929` writes
    /// `resolved.operand.data.data_connect = Some(data_connect)`.
    ///
    /// ⛔⛔ BUT THIS DOC ALSO CLAIMED *"[`crate::schedule::dsc2::InstrAttribute`] already carries both
    /// opaque port lists (`dsc/dsc2.h:936-939`) the Compute arm's `opaque_reads`/`opaque_writes`
    /// want"*, AND THAT IS FALSE ON BOTH HALVES — A SECOND BLOCKER, NOT A CLEARED ONE. Our
    /// `InstrAttribute` carries `unroll`, `precision`, `read_write_regs`, `read_only_regs`, `mode`,
    /// `compute_mask`, `repetition` and `indices`, and NEITHER `input_data_connects_` nor
    /// `output_data_connects_`; and `:936-939` names `outputs_`/`inputsLdsAndLoopOffsets_`/
    /// `outputsLdsAndLoopOffsets_`/`instrAttribute_`, while the two port lists are `:926-927` and
    /// `:928-929` INSIDE `InstrAttribute`. So the walk needs those two `Vec<DataConnect>`s projected
    /// as well, and an opaque compute censused with empty port lists declares an opaque body that
    /// reads and writes nothing.
    ///
    /// ⛔ THE OTHER BLOCKER IS THAT
    /// [`crate::bridges::superdsc_to_dataflow_ir::shape_constraints::Reads::data_connect`] IS A BARE
    /// [`DataConnect`] WHILE `dsc2::DataInfo`'S IS AN [`Option`], and the DDL's own
    /// `data_connect=` is optional too (`ddl/conversion.rs:4431-4432`). The reference keys the
    /// UNNAMED case: `dataConnect_` is a `std::string` defaulting to `""`, `dcMap[""]` is a real
    /// entry, and the census' closing check then reports
    /// `DT_ERROR("Illegal DDL: data_connect  does not have any producer.")` for it
    /// (`ddc/ddcv1.cpp:3319-3325`). ⛔ SO DROPPING THE UNNAMED OPERANDS IS NOT A NARROWING — it
    /// deletes the one diagnostic the census exists to produce, and
    /// [`crate::bridges::superdsc_to_dataflow_ir::shape_constraints::DataConnects::NoProducer`] has no
    /// value to name the offender with. ⭐ THE FIX IS `Reads::data_connect: Option<DataConnect>`
    /// (and the same on both `dsts`/`outputs` lists), in `shape_constraints.rs`.
    fn census_nodes(&self) -> Vec<ScheduleNode> {
        todo!(
            "v1::Dsc2Store::census_nodes: the {{COMPUTE, TRANSFER}} walk is over kinds this tree \
             holds and the DDL fills data_connect (ddl/conversion.rs:2929); TWO things block it. \
             (1) shape_constraints::Reads::data_connect is a bare DataConnect while dsc2::DataInfo's \
             is an Option — the reference keys dcMap[\"\"] and REPORTS on it \
             (ddc/ddcv1.cpp:3319-3325), so dropping unnamed operands deletes that diagnostic. \
             (2) dsc2::InstrAttribute projects NEITHER input_data_connects_ (dsc/dsc2.h:926-927) nor \
             output_data_connects_ (:928-929), which the Compute arm's opaque_reads/opaque_writes \
             are — empty lists there declare an opaque body that reads and writes nothing"
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
        base: NodeBase::named(tree.name(block).unwrap_or_default()),
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
    use crate::schedule::dsc2::{CondRegions, LeafKind, LeafNode, NodeBase, SchedNode};

    match tree.kind_of(node) {
        Some(Kind::Block) => SchedNode::Block(head_block_of(tree, node)),
        Some(Kind::Loop(held)) => {
            SchedNode::Loop(Box::new(crate::schedule::dsc2::LoopNode {
                // `dims_` — the SAME order [`tu::LoopDims`] states, which is the order the loop's own
                // name spells them in, AND NO CONVERSION: `tu::PrimaryDimAndKind` IS
                // [`crate::schedule::dsc2::LoopDim`], one Rust type for `dsc/dims.h:76`.
                dims: held.dims.iter().collect(),
                // ⭐ `numId_`/`denId_` ARE NOT OPTIONAL ON THE STORED VIEW — *"every callsite of the
                // constructor passes a real pair"* ([`tu::LoopNode`]) — so both are `Some` here and the
                // `-1` a `dsc2::LoopNode` admits is the DDL's parametric loop, not this one.
                num: Some(held.num),
                den: Some(held.den),
                // ⛔ `isParametricLoop_ = false` / `parametricLdsIdx_ = -1`, the member initializers of
                // `dsc/dsc2.h:617-618`. See [`v1::Dsc2Store::schedule_head_block`] for the two — and
                // only two — writers, neither of which has run. `loopCountSymbolIds_` (`:576`) is empty
                // for the same reason: its one writer is `finalizeScheduleTree`
                // (`dsc/dsc2.cpp:2999-3005`), which this path does not reach.
                ..crate::schedule::dsc2::LoopNode::bare(head_block_of(tree, node))
            }))
        }
        Some(Kind::Condition(held)) => {
            // ⭐ THE TWO REGIONS AS `addThenRegion`/`addElseRegion` FILLED THEM, each a BLOCK
            // (`dsc/dsc2.cpp:2143`'s *"ConditionNode only accepts 2 BlockNodes as children"*).
            SchedNode::Guarded(Box::new(crate::schedule::dsc2::ConditionNode {
                base: NodeBase::named(tree.name(node).unwrap_or_default()),
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
                // ⭐ `next_[0]` AND `next_[1]` — `getThenBranchNode()`/`getElseBranchNode()`
                // (`dsc/dsc2.h:707`, `:713`). AN ELSE WITH NO THEN NAMES NO REFERENCE STATE: it is the
                // arm `addElseRegion` itself `DT_ERROR`s on, *"ConditionNode does not have a 'then'
                // region"* (`dsc/dsc2.cpp:2163-2164`).
                next: match (held.then_region.first(), held.else_region.first()) {
                    (None, _) => CondRegions::Empty,
                    (Some(then), None) => CondRegions::Then(head_block_of(tree, *then)),
                    (Some(then), Some(otherwise)) => CondRegions::ThenElse([
                        head_block_of(tree, *then),
                        head_block_of(tree, *otherwise),
                    ]),
                },
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
        Some(Kind::Allocate(..)) => SchedNode::Leaf(LeafNode::new(
            LeafKind::Allocate,
            tree.name(node).unwrap_or_default(),
        )),
        Some(Kind::Transfer(_)) => SchedNode::Leaf(LeafNode::new(
            LeafKind::Transfer,
            tree.name(node).unwrap_or_default(),
        )),
        Some(Kind::Compute(_)) => SchedNode::Leaf(LeafNode::new(
            LeafKind::Compute,
            tree.name(node).unwrap_or_default(),
        )),
        None => panic!(
            "v1::Dsc2Store::schedule_head_block: {node:?} is a child of this tree with no \
             nodeType_ — the walk came out of tree.children, so this is a defect in the tree"
        ),
    }
}


// ════════════════════════════════════════════════════════════════════════════════════════════════
// ⭐⭐ THE FIVE ANSWERS THIS FILE GAINED, AGAINST THE REFERENCE'S OWN VALUES.
// ════════════════════════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod authority_tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::num::NonZeroU32;

    use sys_arch_spec::arch_enums::SenComponent;

    use crate::arch::Elements;
    use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{
        Extent, PrimaryDim, StickDims,
    };
    use crate::formats::DataFormat;
    use crate::schedule::ddc::fold::{ConstIdx, NodeId, NodeKind};
    use crate::schedule::ddc::transformation::{self as tr, DsType, Scale};
    use crate::schedule::ddc::transformation_util as tu;
    use crate::schedule::ddc::v1;
    use crate::schedule::ddl::conversion as conv;
    use crate::schedule::ddl::ops::DdlComputeType;
    use crate::schedule::dsc2::{
        ComputeNode, Coordinate, DataInfo, Dsts, InstrAttribute, LayoutDims, LdsIdx, NodeBase,
        NodeName, NumChunks, Operand, RepetitionWithOffset, ReplicationFactor, TransferNode,
        TransferPadding, TransferRepetition,
    };
    use crate::schedule::l3::dsc::{
        CoreIdsUsed, CoreletsUsed, DataStage, DataStages, DesignSpaceConfig, DscIdx, DscList,
        FilledDims, LabeledDs, LabeledDsList, NamedDims, Pinning, PrimaryDsInfo, StageDims,
        SuperDsc,
    };
    use crate::schedule::l3::dl_ops::AddressFoldCoords;
    use crate::units::{Core, NumFolds};

    use super::super::ddc_sites::Dsc2Ddl;
    use super::super::ddc_state::Dsc2State;
    use super::super::state::DscState;
    use super::Dsc2Store;

    /// `labeledDs_[0]` — an `INPUT`, HBM-PINNED so [`DscState::seeded`] mints its `ALLOCATE` and files
    /// it in `memOrg_[HBM]`, with `scale_` stating `-2` on `X` and a non-negative scale on `Y`.
    const TENSOR: LdsIdx = LdsIdx(0);
    /// `labeledDs_[1]` — a `KERNEL`, pinned NOWHERE and carrying a DIFFERENT `dataFormat_`, so *which
    /// side's labelled DS an arm names* is observable rather than a coincidence.
    const KERNEL: LdsIdx = LdsIdx(1);

    fn core(index: u32) -> Core {
        Core::checked(index).expect("a core in range")
    }

    /// One stage stating two extents — the shape [`FilledDims::of`] demands.
    fn a_stage() -> DataStage {
        let mut dims = StageDims::default();
        dims.extents.insert(PrimaryDim::X, Extent(4));
        dims.extents.insert(PrimaryDim::Y, Extent(2));
        let named = NamedDims {
            name: crate::schedule::ddc::transformation_util::StageName::default(),
            dims: FilledDims::of(dims).expect("a stage that states a dim"),
        };
        DataStage {
            ss: named.clone(),
            el: named,
        }
    }

    /// TWO LABELLED DSs, ONE HBM-PINNED, WITH DISTINCT STICKS AND DISTINCT FORMATS.
    fn a_dsc() -> DesignSpaceConfig {
        let hbm_pinned = Pinning {
            mem_org: BTreeMap::from([(SenComponent::Hbm, true)]),
            lx: false,
            lx_padded: false,
        };
        // `scale_`: `-2` on X (a STICK dim, entry 126's splat test) and `1.0` on Y.
        let mut tensor = LabeledDs::new(
            DsType::Input,
            vec![(PrimaryDim::X, Scale::StickDim), (PrimaryDim::Y, Scale::Sized(1.0))],
            TENSOR,
            hbm_pinned,
        );
        tensor.set_data_format(DataFormat::Sen169Fp16);
        let mut kernel = LabeledDs::new(
            DsType::Kernel,
            vec![(PrimaryDim::X, Scale::Sized(1.0))],
            KERNEL,
            Pinning::default(),
        );
        // ⭐ A DIFFERENT FORMAT FROM THE TENSOR'S — this is what makes "which side" a VALUE.
        kernel.set_data_format(DataFormat::Sen143Fp8);
        let two = CoreletsUsed::new(NonZeroU32::new(2).expect("two corelets"));
        DesignSpaceConfig {
            ddc: crate::schedule::l3::dsc::DdcFacts {
                // `constantInfo_[7]` — four elements of FP32, which is entry 126's
                // `numElemInConst`/`dataFormat_` pair.
                constants: BTreeMap::from([(
                    ConstIdx(7),
                    crate::schedule::l3::dsc::ConstantInfo {
                        name: v1::StorageName("k".to_owned()),
                        data_format: Some(DataFormat::IeeeFp32),
                        data: vec![1, 2, 3, 4],
                        is_data_symbolic: false,
                        allocations: BTreeMap::new(),
                    },
                )]),
                ..crate::schedule::l3::dsc::DdcFacts::default()
            },
            gtr_ids_used: BTreeSet::new(),
            corelets_used: two,
            corelets_used_dsc2: Some(two),
            corelet_shares: BTreeMap::new(),
            // `primaryDsInfo_.at(dsType_)` — `getStickSizes`' own `.at()` (`dsc/dsc2.cpp:4077`),
            // keyed by DS TYPE and not by lds, which is why the two entries differ.
            primary_ds_info: BTreeMap::from([
                (
                    DsType::Input,
                    PrimaryDsInfo {
                        layout: LayoutDims::new(PrimaryDim::X, vec![PrimaryDim::Y]),
                        stick: StickDims(vec![(PrimaryDim::X, Elements(64))]),
                    },
                ),
                (
                    DsType::Kernel,
                    PrimaryDsInfo {
                        layout: LayoutDims::new(PrimaryDim::X, vec![]),
                        stick: StickDims(vec![(PrimaryDim::X, Elements(32))]),
                    },
                ),
            ]),
            core_ids_used: CoreIdsUsed::new(core(0), vec![]),
            layout_dims: BTreeMap::from([
                (TENSOR, LayoutDims::new(PrimaryDim::X, vec![PrimaryDim::Y])),
                (KERNEL, LayoutDims::new(PrimaryDim::X, vec![])),
            ]),
            data_stages: DataStages::new(a_stage(), a_stage()),
            indirect_access_index_lds: BTreeSet::new(),
            lx_chunk_capacity: BTreeMap::new(),
            full_padding: BTreeMap::new(),
            labeled_ds: LabeledDsList::new(tensor, vec![kernel]),
        }
    }

    /// One end — `src_.unit_`/`dstVias_.at(i).loc_` zipped with its `..LdsAndLoopOffsets_`.
    fn end(unit: SenComponent, storage: SenComponent, data: DataInfo) -> Operand {
        Operand {
            unit,
            storage,
            data,
        }
    }

    /// `myLdsIdx_ = lds`, every other `DataInfo` field at its own initializer.
    fn of_lds(lds: LdsIdx) -> DataInfo {
        DataInfo {
            data_connect: None,
            my_lds_idx: Some(lds),
            constant_id: None,
            latch_data_id: None,
            ..DataInfo::EMPTY
        }
    }

    /// `constantId_ = id`, `myLdsIdx_ = -1`.
    fn of_constant(id: ConstIdx) -> DataInfo {
        DataInfo {
            data_connect: None,
            my_lds_idx: None,
            constant_id: Some(id),
            latch_data_id: None,
            ..DataInfo::EMPTY
        }
    }

    /// A fresh `dsc2::TransferNode` between two ends — every other field its own initializer.
    fn a_transfer(name: &str, src: Operand, dst: Operand) -> TransferNode {
        TransferNode {
            repetition: TransferRepetition::default(),
            last_fusable_parent_loop_src: None,
            last_fusable_parent_loop_dst: Vec::new(),
            unit_time_transfer_chunk_stride: Vec::new(),
            rotate_num_elements: None,
            corelet_views: BTreeMap::new(),
            transfer_coordinates: Coordinate::default(),
            name: NodeName(name.to_owned()),
            src,
            dsts: Dsts::new(dst, Vec::new()),
            replication_factor: ReplicationFactor(1),
            unit_time_transfer_chunk_size: Vec::new(),
            unit_time_transfer_num_chunks: NumChunks(1),
            padding: TransferPadding::default(),
            src_indirect: None,
            dst_indirect: None,
            core_id_to_gtr_info: BTreeMap::new(),
            transfer_size: BTreeMap::new(),
        }
    }

    /// ⭐⭐⭐ EVERY ANSWER THIS FILE GAINED, READ OFF A TREE THE DDL EXPANSION'S OWN WRITERS FILLED.
    ///
    /// ⛔ THE VALUES ARE THE REFERENCE'S, NOT OURS. The two labelled DSs carry DIFFERENT
    /// `dataFormat_`s and DIFFERENT stick extents, so every assertion that names one of them would
    /// FAIL if the arm took the other side — which is the whole content of `getTransferType`'s
    /// ternary (`ddc/ddcv1.cpp:481-484`).
    #[test]
    fn the_five_gained_answers_carry_the_references_own_values() {
        let sdsc = SuperDsc::new(
            DscList::new(a_dsc(), Vec::new()),
            BTreeMap::new(),
            BTreeMap::new(),
            BTreeMap::new(),
        );
        let l3 = DscState::seeded(&sdsc);
        // ⭐ THE SEED: the root block plus ONE `ALLOCATE`, because `TENSOR` is HBM-pinned and `KERNEL`
        // is pinned nowhere. That allocate node is what `destination_allocation` must find.
        assert_eq!(
            l3.node_count(),
            2,
            "root_level_operations plus the HBM allocation of the one pinned labelled DS: {:?}",
            l3.dsc(DscIdx(0)).expect("the one DSC").names()
        );
        let state = Dsc2State::seeded(&sdsc, &l3, &[Vec::new()], &[v1::StorageName("d".to_owned())]);

        // ── MINT INTO THE LIVE TREE through the DDL expansion's own writer ─────────────────────
        let head = l3
            .dsc(DscIdx(0))
            .expect("the one DSC's tree")
            .with(|tree| tree.head())
            .expect("a seeded tree has a head");
        let (tensor_to_tensor, constant_to_tensor, compute, loop_node, condition) = {
            let mut ddl = Dsc2Ddl::new(&state, DscIdx(0));
            let tensor_to_tensor = conv::ScheduleWrites::add_transfer(
                &mut ddl,
                head,
                a_transfer(
                    "t2t",
                    end(SenComponent::Lxlu, SenComponent::Lx, of_lds(TENSOR)),
                    end(SenComponent::Pe, SenComponent::Pelrf, of_lds(KERNEL)),
                ),
            )
            .expect("the tree took the transfer");
            // ⛔ SOURCE IS THE CONSTANT AND THE DESTINATION IS `KERNEL`, so the arm must name
            // `KERNEL`'s FP8 — naming the source's lds is unspellable and naming `TENSOR` is the bug
            // this catches.
            let constant_to_tensor = conv::ScheduleWrites::add_transfer(
                &mut ddl,
                head,
                a_transfer(
                    "c2t",
                    end(
                        SenComponent::Constant,
                        SenComponent::Lx,
                        of_constant(ConstIdx(7)),
                    ),
                    end(SenComponent::Pe, SenComponent::Hbm, of_lds(KERNEL)),
                ),
            )
            .expect("the tree took the transfer");
            let compute = conv::ScheduleWrites::add_compute(
                &mut ddl,
                head,
                ComputeNode {
                    is_opaque_op: false,
                    corelet_views: BTreeMap::new(),
                    input_coordinates: Vec::new(),
                    output_coordinate: Coordinate::default(),
                    repetition_with_offset: RepetitionWithOffset::default(),
                    name: NodeName("mm".to_owned()),
                    op: DdlComputeType::Fma16,
                    ex_unit: SenComponent::Pe,
                    inputs: vec![end(SenComponent::Lx, SenComponent::Lx, of_lds(KERNEL))],
                    // ⭐ THE OUTPUT'S **UNIT** IS `HBM`, which is the key
                    // `memOrg_.at(outputs_.at(idx))` uses — NOT its `storage`, which is `LX` here so
                    // that a port reading the wrong half finds nothing.
                    outputs: vec![end(SenComponent::Hbm, SenComponent::Lx, of_lds(TENSOR))],
                    num_folds_engaged: NumFolds::ONE,
                    data_format: Some(DataFormat::Sen169Fp16),
                    instr_attribute: InstrAttribute::default(),
                },
            )
            .expect("the tree took the compute");
            let loop_node = conv::ScheduleWrites::add_block(&mut ddl, head, NodeName("b".to_owned()))
                .expect("the tree took the block");
            let condition = conv::ScheduleWrites::add_condition(
                &mut ddl,
                head,
                crate::schedule::dsc2::ConditionNode {
                    base: NodeBase::named(NodeName("cond".to_owned())),
                    loop_cond: crate::schedule::dsc2::LoopCondComposite::default(),
                    core_cl_cond: BTreeMap::from([(
                        core(0),
                        BTreeSet::from([crate::units::Corelet::at::<0>()]),
                    )]),
                    next: crate::schedule::dsc2::CondRegions::Empty,
                },
            )
            .expect("the tree took the condition");
            (
                tensor_to_tensor,
                constant_to_tensor,
                compute,
                loop_node,
                condition,
            )
        };

        let store = Dsc2Store::new(&state, DscIdx(0), AddressFoldCoords::flat());

        // ── 1. `hoist_parent` — `nodeType_` AND NOTHING ELSE ──────────────────────────────────
        // ⛔ A COMPUTE PARENT IS `Other`, which is the `continue` at
        // `ddc/ddc_transformation.cpp:1503-1505`. A HoistParent with a Compute arm is what the old
        // refusal claimed the trait had.
        assert_eq!(
            tr::HoistTransfers::hoist_parent(&store, condition),
            tr::HoistParent::Condition,
            "nodeType_ == CONDITION takes collectLoopReferences' arm"
        );
        assert_eq!(
            tr::HoistTransfers::hoist_parent(&store, compute),
            tr::HoistParent::Other,
            "a COMPUTE parent takes the same `continue` a BLOCK does"
        );
        assert_eq!(
            tr::HoistTransfers::hoist_parent(&store, loop_node),
            tr::HoistParent::Other,
            "a BLOCK parent is Other"
        );
        assert_eq!(
            tr::HoistTransfers::hoist_parent(&store, head),
            tr::HoistParent::Other,
            "the head block is Other, not the walk's Loop arm"
        );

        // ── 2. `v1::LoopOffsets::compute` — the whole node off `Kind::Compute` ─────────────────
        let held = v1::LoopOffsets::compute(&store, compute);
        assert_eq!(held.op, DdlComputeType::Fma16, "type_");
        assert_eq!(held.ex_unit, SenComponent::Pe, "exUnit_");
        assert_eq!(
            held.inputs.first().map(|input| input.data.my_lds_idx),
            Some(Some(KERNEL)),
            "inputs_ zipped with inputsLdsAndLoopOffsets_"
        );

        // ── 3. `destination_allocation` — `getAllocation(dst, dst storage, true)` ──────────────
        let seeded = l3
            .dsc(DscIdx(0))
            .expect("the one DSC's tree")
            .with(|tree| tree.children(head).first().copied())
            .expect("the seeded HBM allocation is the head's first child");
        assert_eq!(
            tu::TransferMoves::destination_allocation(
                &store,
                &end(SenComponent::Pe, SenComponent::Hbm, of_lds(TENSOR)),
            ),
            Some(seeded),
            "memOrg_[HBM].allocateNode_ of the pinned labelled DS"
        );
        // ⛔ `allowMissingAlloc = true`'s `nullptr`: `KERNEL` is pinned nowhere, so its `memOrg_`
        // names no HBM entry.
        assert_eq!(
            tu::TransferMoves::destination_allocation(
                &store,
                &end(SenComponent::Pe, SenComponent::Hbm, of_lds(KERNEL)),
            ),
            None,
            "a labelled DS with no memOrg_ entry is the reference's nullptr, not a stop"
        );
        // ⛔ AND `dsc2::memories.count(storage) == 0` (`dsc/dsc2.cpp:2597-2603`): the PE is a unit,
        // not a memory.
        assert_eq!(
            tu::TransferMoves::destination_allocation(
                &store,
                &end(SenComponent::Pe, SenComponent::Pe, of_lds(TENSOR)),
            ),
            None,
            "a storage that is not a memory is the reference's nullptr"
        );

        // ── 4. `output_allocation` — keyed by the output's UNIT, not its storage ───────────────
        let alloc = crate::schedule::ddc::transformation_util::AllocationsByNode::allocation_of(
            &store, seeded,
        )
        .expect("the seeded node names an allocation");
        assert_eq!(
            tr::OffsetAdjustment::output_allocation(&store, compute, tr::OutputIdx(0)),
            Some(alloc),
            "labeledDs_.at(outputsLdsAndLoopOffsets_.at(0).myLdsIdx_).memOrg_.at(outputs_.at(0))"
        );
        // ⛔ PAST THE END OF `outputs_` is the reference's `.at()` and this port's [`None`], which
        // entry 108's own doc calls a stated divergence.
        assert_eq!(
            tr::OffsetAdjustment::output_allocation(&store, compute, tr::OutputIdx(1)),
            None,
            "an output index past the end"
        );

        // ── 5. `transfer_operands` — `getTransferType()` AND WHICH SIDE'S LDS ─────────────────
        // ⭐ TENSOR_TO_TENSOR NAMES THE **SOURCE**, so the stick is the INPUT's 64 and the format is
        // the INPUT's FP16 — the destination is `KERNEL`, whose stick is 32 and whose format is FP8.
        let said = v1::Dsc2Store::transfer_operands(&store, tensor_to_tensor);
        let v1::TransferOperands::TensorToTensor(Some(lds)) = said else {
            panic!("a labelled-DS-to-labelled-DS transfer is TENSOR_TO_TENSOR: {said:?}");
        };
        assert_eq!(
            lds.stick,
            StickDims(vec![(PrimaryDim::X, Elements(64))]),
            "getStickSizes(SOURCE lds.dsType_) — 32 here would mean the arm took the destination"
        );
        assert_eq!(
            lds.format,
            DataFormat::Sen169Fp16,
            "the SOURCE's dataFormat_ — FP8 would mean the arm took the destination"
        );
        assert_eq!(
            lds.splat_dims.0,
            BTreeSet::from([PrimaryDim::X]),
            "`is_any_of(-2, scale_)` as dims: X is Scale::StickDim and Y is not"
        );

        // ⭐ CONSTANT_TO_TENSOR NAMES THE **DESTINATION**, which is `KERNEL` — so the same two values
        // must come back as 32 and FP8. This pair is the ternary, carried as values.
        let said = v1::Dsc2Store::transfer_operands(&store, constant_to_tensor);
        let v1::TransferOperands::ConstantToTensor(Some(lds)) = said else {
            panic!("a constant-to-labelled-DS transfer is CONSTANT_TO_TENSOR: {said:?}");
        };
        assert_eq!(
            lds.stick,
            StickDims(vec![(PrimaryDim::X, Elements(32))]),
            "getStickSizes(DESTINATION lds.dsType_) — 64 would mean the arm took the source"
        );
        assert_eq!(
            lds.format,
            DataFormat::Sen143Fp8,
            "the DESTINATION's dataFormat_"
        );
        assert!(
            lds.splat_dims.0.is_empty(),
            "KERNEL's scale_ states no -2"
        );
    }

    /// ⭐ THE CONSTANT ARM, WHOSE WHOLE POINT IS THAT `l3::dsc` **DOES** PROJECT `constantInfo_` —
    /// `data_.getSingleData().size()` and `dataFormat_`, carried as the two divisors entry 126 forms
    /// `replicationFactor_` from (`ddc/ddcv1.cpp:456-458`).
    #[test]
    fn constant_to_constant_carries_the_constants_own_element_count_and_format() {
        let sdsc = SuperDsc::new(
            DscList::new(a_dsc(), Vec::new()),
            BTreeMap::new(),
            BTreeMap::new(),
            BTreeMap::new(),
        );
        let l3 = DscState::seeded(&sdsc);
        let state = Dsc2State::seeded(&sdsc, &l3, &[Vec::new()], &[v1::StorageName("d".to_owned())]);
        let head = l3
            .dsc(DscIdx(0))
            .expect("the one DSC's tree")
            .with(|tree| tree.head())
            .expect("a seeded tree has a head");
        let transfer = {
            let mut ddl = Dsc2Ddl::new(&state, DscIdx(0));
            conv::ScheduleWrites::add_transfer(
                &mut ddl,
                head,
                a_transfer(
                    "c2c",
                    end(
                        SenComponent::Constant,
                        SenComponent::Lx,
                        of_constant(ConstIdx(7)),
                    ),
                    end(
                        SenComponent::Constant,
                        SenComponent::Lx,
                        of_constant(ConstIdx(7)),
                    ),
                ),
            )
            .expect("the tree took the transfer")
        };
        let store = Dsc2Store::new(&state, DscIdx(0), AddressFoldCoords::flat());
        let said = v1::Dsc2Store::transfer_operands(&store, transfer);
        let v1::TransferOperands::ConstantToConstant(constant) = said else {
            panic!("a constant-to-constant transfer is CONSTANT_TO_CONSTANT: {said:?}");
        };
        assert_eq!(
            constant.elements.get(),
            4,
            "constantInfo_.at(7).data_.getSingleData().size()"
        );
        assert_eq!(
            constant.format,
            DataFormat::IeeeFp32,
            "constantInfo_.at(7).dataFormat_"
        );
    }

    /// ⛔ AND THE `nodeType_` DISPATCH IS TOTAL: a node this tree does not hold is
    /// [`tr::HoistParent::Other`], the arm the reference's own `continue` takes — NOT a stop.
    #[test]
    fn hoist_parent_of_a_node_this_tree_does_not_hold_is_other() {
        let sdsc = SuperDsc::new(
            DscList::new(a_dsc(), Vec::new()),
            BTreeMap::new(),
            BTreeMap::new(),
            BTreeMap::new(),
        );
        let l3 = DscState::seeded(&sdsc);
        let state = Dsc2State::seeded(&sdsc, &l3, &[Vec::new()], &[v1::StorageName("d".to_owned())]);
        let store = Dsc2Store::new(&state, DscIdx(0), AddressFoldCoords::flat());
        assert_eq!(
            tr::HoistTransfers::hoist_parent(&store, NodeId(9_999)),
            tr::HoistParent::Other,
            "an id this tree issued no node for"
        );
    }

    /// ⛔ A REFERENCE TO `NodeKind` KEEPS THIS MODULE'S IMPORT LIVE — the census kinds the mint above
    /// produces, asserted so the fixture cannot silently stop minting them.
    #[test]
    fn the_fixture_mints_the_kinds_the_answers_read() {
        let sdsc = SuperDsc::new(
            DscList::new(a_dsc(), Vec::new()),
            BTreeMap::new(),
            BTreeMap::new(),
            BTreeMap::new(),
        );
        let l3 = DscState::seeded(&sdsc);
        let state = Dsc2State::seeded(&sdsc, &l3, &[Vec::new()], &[v1::StorageName("d".to_owned())]);
        let head = l3
            .dsc(DscIdx(0))
            .expect("the one DSC's tree")
            .with(|tree| tree.head())
            .expect("a seeded tree has a head");
        {
            let mut ddl = Dsc2Ddl::new(&state, DscIdx(0));
            conv::ScheduleWrites::add_compute(
                &mut ddl,
                head,
                ComputeNode {
                    is_opaque_op: false,
                    corelet_views: BTreeMap::new(),
                    input_coordinates: Vec::new(),
                    output_coordinate: Coordinate::default(),
                    repetition_with_offset: RepetitionWithOffset::default(),
                    name: NodeName("mm".to_owned()),
                    op: DdlComputeType::Fma16,
                    ex_unit: SenComponent::Pe,
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                    num_folds_engaged: NumFolds::ONE,
                    data_format: None,
                    instr_attribute: InstrAttribute::default(),
                },
            )
            .expect("the tree took the compute");
        }
        // ⭐ `compute: 1` — the census that used to read `compute: 0` because the DDL walk minted into
        // a dropped copy (`5ee670017`).
        assert_eq!(
            l3.dsc(DscIdx(0)).expect("the one DSC").kinds().get(&NodeKind::Compute),
            Some(&1),
            "Kind::Compute reaches the LIVE tree"
        );
    }
}
