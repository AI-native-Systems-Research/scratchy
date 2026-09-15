// SPDX-License-Identifier: Apache-2.0
//! ⭐ ONE DSC'S SCHEDULE TREE AND ITS `memOrg_`s, CELL-BACKED — the ONE object [`super::Reads`] and
//! [`super::Env`] both name.
//!
//! Every tree trait in [`crate::schedule::l3::dl_ops`] is keyed by [`NodeId`] and answers by VALUE
//! ([`NodeParents::parent`], [`LoopStages::loop_dims`], `node_names`), which is exactly why a
//! [`RefCell`] interior can serve a `&self` reader and a `&mut self` writer of the same tree: no
//! method hands a borrow of the interior out, so no borrow can be held across a write.
//!
//! ⛔ THIS IS NOT [`crate::schedule::dsc2::ScheduleTree`]. That type nests owned `SchedNode`s and
//! cannot answer "the parent of node 7"; the ported stage reaches every node by identity.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

use sys_arch_spec::arch_enums::SenComponent;

use crate::arch::Elements;
use crate::bridges::superdsc_to_dataflow_ir::shape_constraints::{Extent, PrimaryDim};
use crate::schedule::ddc::fold::{AllocId, AllocLayout, NodeId, NodeKind};
use crate::schedule::ddc::metadata::DatastageId;
use crate::schedule::ddc::transformation::LoopId;
use crate::schedule::ddc::transformation_util::{InsertionPoint, LoopDims, LoopNode, PaddingForm};
use crate::schedule::ddc::v1;
use crate::schedule::dsc2::{
    AllocateNode, ComputeNode, Coordinate, LayoutDims, LdsIdx, LoopCondComposite, NodeName,
    StickMaskNode, SyncNode, TransferNode,
};
use crate::schedule::l3::dl_ops::{
    GtrGroupId, L3AllocateNode, L3Sync, L3WalkNode, LX_BELOW_BLOCK_NODE_NAME,
};
use crate::schedule::l3::dsc::{
    AddressCoord, BufferOffset, Buffering, ByteAddress, IndirectAlloc, L3Transfer, MemOrg,
    PlacedAllocation,
};
use crate::units::{Core, Corelet};

// ⭐⭐ THE READ SURFACE OF THIS FILE IS `pub`; EVERY WRITE STAYS `pub(super)`.
//
// The DataflowIR lowering has to walk the tree the stages GREW — `Schedule::roots` read
// `dsc.scheduleTree_`, the WIRE vec, which the stages never write and whose `AllocNode` cannot even
// represent a non-`allocate` node. So the typed tree is the truth, and a consumer outside `stages`
// needs to name [`TreeData`], [`Kind`] and [`Cond`] and to read nodes off them.
//
// ⛔ BUT ONLY TO READ. A target crate composing a lowering has no business mutating the scheduler's
// tree, and the asymmetry is the guard: `add`, `link`, `delete`, `move_children`, `set_*` and the L3
// walks stay `pub(super)`, so the seam is a VIEW. That is the capability twin of the path rule
// `crates/targets/spyre/tests/scratchy_knows_nothing_about_l3.rs` enforces — a target names
// `deeptools::sdsc` and, through it, can look at this tree and not change it.
//
// ⛔ AND THE SEAM IS NAMED, NOT WHOLESALE: nine reads and three types, which is what the census and
// the root walk need. Raising the other fifty would be publishing surface nothing asked for.

/// WHICH KIND OF NODE, CARRYING WHAT THAT KIND IS READ FOR — `dsc2::ScheduleNode::nodeType_` and the
/// downcast beside every use of it.
///
/// ⭐ THE VARIANTS ARE `dsc2::ScheduleNode::NodeType`'s OWN ORDER (`dsc/dsc2.h:446-456`: `INVALID`,
/// `BLOCK`, `LOOP`, `TRANSFER`, `COMPUTE`, `SYNC`, `CONDITION`, `ALLOCATE`, `STICKMASK`) less
/// `INVALID`, which is an absence and has no variant.
///
/// ⛔⛔ `Compute` AND `StickMask` EXIST HERE BUT NOTHING CONSTRUCTS THEM YET, AND THAT MUST NOT BE READ
/// AS "COMPUTES APPEAR". The type gap is closed — [`Self::node_kind`] can now return
/// `NodeKind::Compute`, which stage 2b already switches on (`schedule/ddc/v1.rs:2416`, `:4911`) — but
/// the DDL walk that mints computes writes into a **detached copy**: `ddc/v1.rs:6489` is
/// `DdlConversion::new(store.schedule_head_block())` and `schedule_head_block`
/// (`stages/ddc_store2.rs`) hands back a `BlockNode` **by value**, while the reference's
/// `DdlConversion` holds a `DesignSpaceConfig&` (`ddc/ddl/ddl_conversion.h:511`) and `parseDdl2Dsc`
/// starts from `dsc.scheduleTree_.getHeadMutable()` (`ddl_conversion.cpp:2774`). Until that seam
/// writes back, `op_compute`/`op_opaque` mint into a value that is dropped and this variant has no
/// producer.
///
/// ⛔ SO NOTHING HERE FABRICATES A NODE TO MOVE A CENSUS. The corpus compute count stays 0 after this,
/// and a stand-in node is the fabricated-placement failure this crate ranks worse than a stop.
///
/// ⛔ AND THE "+2,834 COMPUTES ACROSS 187 PROGRAMS" THIS DOC USED TO CITE IS **UNSOURCED** — it traces
/// to a comment and a build log, not to a census anyone ran, so it is not repeated as a number. What
/// IS measured is the scratchy side: `sync 0, compute 0` over 24,363 programs
/// (`crates/targets/spyre/src/superdsc_to_l3_sdsc.rs`'s corpus report).
#[derive(Debug, Clone)]
pub enum Kind {
    /// `BLOCK`.
    Block,
    /// `LOOP`, holding the node entry 217 and entry 018 minted.
    Loop(LoopNode),
    /// `TRANSFER`.
    Transfer(TransferNode),
    /// `COMPUTE` — `dsc2::ComputeNode` (`dsc/dsc2.h:900`).
    ///
    /// ⭐ THE WHOLE NODE, WHICH IS WHY NO OP-INDEX PAIRING IS PORTED WITH IT. `ComputeNode` has NO
    /// back-pointer to `computeOp_`: `opIdx_` appears ZERO times in `dsc2.h` and `computeOp_` is a
    /// `DesignSpaceConfig` member, not a node field. The reference's only reader of that pairing
    /// (`ddc/ddcv1.cpp:1092-1113`) takes `compute->exUnit_` and
    /// `compute->outputsLdsAndLoopOffsets_.at(i).myLdsIdx_` straight off the node — so holding the
    /// node IS the answer, and `stages/ddc_tree.rs`'s claim that the reference "holds the pairing on
    /// the node (`ComputeNode::opIdx_`)" describes a field that does not exist.
    Compute(ComputeNode),
    /// `SYNC`.
    Sync(SyncNode),
    /// `CONDITION` — a `loopCond_` guard, a `coreClCond_` guard, or neither.
    Condition(Cond),
    /// `ALLOCATE`, with the L3 view of the node and its allocation identity.
    Allocate(AllocId, L3AllocateNode),
    /// `STICKMASK` — `dsc2::StickMaskNode` (`dsc/dsc2.h:1059`), which `dsc2::SchedNode` already
    /// carries an arm for.
    StickMask(StickMaskNode),
}

impl Kind {
    /// `nodeType_`.
    fn node_kind(&self) -> NodeKind {
        match self {
            Self::Block => NodeKind::Block,
            Self::Loop(_) => NodeKind::Loop,
            Self::Transfer(_) => NodeKind::Transfer,
            Self::Compute(_) => NodeKind::Compute,
            Self::Sync(_) => NodeKind::Sync,
            Self::Condition(_) => NodeKind::Condition,
            Self::Allocate(..) => NodeKind::Allocate,
            Self::StickMask(_) => NodeKind::StickMask,
        }
    }
}

/// ONE CONDITION NODE'S GUARD AND ITS TWO REGIONS — `loopCond_` XOR `coreClCond_`, which is what
/// `hasCoreClCond()` distinguishes.
#[derive(Debug, Clone, Default)]
pub struct Cond {
    /// `loopCond_`, absent on a core/corelet-guarded condition.
    pub loop_cond: Option<LoopCondComposite>,
    /// `coreClCond_`, absent on a loop-guarded one.
    pub cores: Option<v1::CoreClSet>,
    /// The blocks added to the then-region, in order.
    pub then_region: Vec<NodeId>,
    /// The blocks added to the else-region, in order.
    pub else_region: Vec<NodeId>,
}

/// ONE NODE — its name, its `prev_` parent, its `next_` children and its kind.
#[derive(Debug, Clone)]
struct Entry {
    name: NodeName,
    parent: Option<NodeId>,
    children: Vec<NodeId>,
    kind: Kind,
}

/// ONE DSC'S `scheduleTree_` BY NODE ID.
///
/// ⭐⭐ THIS IS THE TREE THE STAGES ACTUALLY GROW, and the reason it is `pub`. The wire
/// `SuperDsc::scheduleTree_` is a `Vec<AllocNode>` whose every `nodeType_` is the string
/// `"allocate"` — it cannot represent a loop, a transfer or a sync at all — and nothing writes the
/// stages' nodes back into it. A consumer that censused the wire vec was therefore measuring an
/// unscheduled tree; this is the one to ask.
///
/// ⛔ READ-ONLY FROM OUTSIDE `stages`. See this file's header: the accessors are `pub`, every mutator
/// is `pub(super)`.
#[derive(Debug, Default)]
pub struct TreeData {
    nodes: BTreeMap<NodeId, Entry>,
    next_node: u32,
    next_alloc: u32,
    head: Option<NodeId>,
    /// `scheduleTree_.getHead()->denId_`, which serialises as `scheduleTreeHeadDenId_`.
    head_den: Option<DatastageId>,
    /// `dsc.gtrIdsUsed_` as entries 218 and 291 insert into it.
    gtr_ids: BTreeSet<GtrGroupId>,
    /// The `(alloc, node)` pairing every `AllocId` names, both ways.
    node_of_alloc: BTreeMap<AllocId, NodeId>,
    /// `allocNode->allocateCoordinates_`.
    coordinates: BTreeMap<NodeId, Coordinate>,
}

impl TreeData {
    /// A fresh node, linked to `parent` as its LAST child when one is given.
    pub(super) fn add(&mut self, name: NodeName, kind: Kind, parent: Option<NodeId>) -> NodeId {
        let id = NodeId(self.next_node);
        self.next_node = self.next_node.saturating_add(1);
        if let Kind::Allocate(alloc, _) = &kind {
            self.node_of_alloc.insert(*alloc, id);
        }
        self.nodes.insert(
            id,
            Entry {
                name,
                parent,
                children: Vec::new(),
                kind,
            },
        );
        if let Some(parent) = parent
            && let Some(entry) = self.nodes.get_mut(&parent)
        {
            entry.children.push(id);
        }
        id
    }

    /// `scheduleTree_.getHead()` — the root block, set once when the tree is seeded.
    pub(super) fn set_head(&mut self, head: NodeId) {
        self.head = Some(head);
    }

    /// `scheduleTree_.getHead()`.
    pub const fn head(&self) -> Option<NodeId> {
        self.head
    }

    /// `getHeadMutable()->denId_ = den`.
    pub(super) fn set_head_den(&mut self, den: DatastageId) {
        self.head_den = Some(den);
    }

    /// `getHead()->denId_`.
    pub(super) const fn head_den(&self) -> Option<DatastageId> {
        self.head_den
    }

    /// `dsc.gtrIdsUsed_.insert(group)`.
    pub(super) fn insert_gtr_id(&mut self, group: GtrGroupId) {
        self.gtr_ids.insert(group);
    }

    /// The identity `new dsc2::AllocateNode()` WOULD issue — a fresh heap pointer in the reference,
    /// and the next free id here, WITHOUT taking it. What `free_alloc_id`/`free_alloc` answer.
    pub(super) const fn peek_alloc(&self) -> AllocId {
        AllocId(self.next_alloc.saturating_add(1))
    }

    /// `condNode->getThenBranchNode()` (`dsc/dsc2.h:685`) — the FIRST block added to the then-region.
    pub(super) fn then_branch(&self, condition: NodeId) -> Option<NodeId> {
        match &self.nodes.get(&condition)?.kind {
            Kind::Condition(cond) => cond.then_region.first().copied(),
            _ => None,
        }
    }

    /// The identity `new dsc2::AllocateNode()` issues.
    pub(super) fn fresh_alloc(&mut self) -> AllocId {
        self.next_alloc = self.next_alloc.saturating_add(1);
        AllocId(self.next_alloc)
    }

    /// The node held under that allocation identity.
    pub(super) fn node_of_alloc(&self, alloc: AllocId) -> Option<NodeId> {
        self.node_of_alloc.get(&alloc).copied()
    }

    /// `allocNode->allocateCoordinates_ = coordinate`.
    pub(super) fn set_coordinate(&mut self, node: NodeId, coordinate: Coordinate) {
        self.coordinates.insert(node, coordinate);
    }

    /// `allocNode->allocateCoordinates_`.
    pub(super) fn coordinate(&self, node: NodeId) -> Option<Coordinate> {
        self.coordinates.get(&node).cloned()
    }

    /// `node->getPrev()` — the PARENT block, absent at the root.
    pub fn parent(&self, node: NodeId) -> Option<NodeId> {
        self.nodes.get(&node)?.parent
    }

    /// `next_` of that node, in order; EMPTY for a node the tree does not hold.
    pub fn children(&self, parent: NodeId) -> Vec<NodeId> {
        self.nodes
            .get(&parent)
            .map(|entry| entry.children.clone())
            .unwrap_or_default()
    }

    /// `node->name_`.
    pub fn name(&self, node: NodeId) -> Option<NodeName> {
        self.nodes.get(&node).map(|entry| entry.name.clone())
    }

    /// `node->name_ = name`.
    pub(super) fn set_name(&mut self, node: NodeId, name: NodeName) {
        if let Some(entry) = self.nodes.get_mut(&node) {
            entry.name = name;
        }
    }

    /// `sync->otherEndOfTheSignals_.push_back(other)`, held BY NAME as [`SyncNode`] does.
    pub(super) fn add_sync_other_end(&mut self, sync: NodeId, other: NodeName) {
        if let Some(Entry {
            kind: Kind::Sync(held),
            ..
        }) = self.nodes.get_mut(&sync)
        {
            held.other_ends.push(other);
        }
    }

    /// `traverseTreeDFSMutable(nullptr, {LOOP, TRANSFER, ALLOCATE})` PROJECTED OUT, in DFS order.
    pub(super) fn loops_transfers_and_allocates(&self) -> Vec<L3WalkNode> {
        self.dfs()
            .into_iter()
            .filter_map(|node| match &self.nodes.get(&node)?.kind {
                Kind::Loop(_) => Some(L3WalkNode::Loop(LoopId(node))),
                Kind::Transfer(_) => Some(L3WalkNode::Transfer(node)),
                Kind::Allocate(_, held) => Some(L3WalkNode::Allocate(node, held.clone())),
                _ => None,
            })
            .collect()
    }

    /// `traverseTreeDFSMutable(root, {CONDITION})` — every condition node below `root`.
    pub(super) fn conditions_under(&self, root: NodeId) -> Vec<NodeId> {
        let mut found = Vec::new();
        let mut stack = vec![root];
        while let Some(at) = stack.pop() {
            if matches!(
                self.nodes.get(&at).map(|entry| &entry.kind),
                Some(Kind::Condition(_))
            ) {
                found.push(at);
            }
            stack.extend(self.children(at).into_iter().rev());
        }
        found
    }

    /// `condNode->loopCond_`.
    pub(super) fn loop_cond(&self, condition: NodeId) -> Option<LoopCondComposite> {
        match &self.nodes.get(&condition)?.kind {
            Kind::Condition(cond) => cond.loop_cond.clone(),
            _ => None,
        }
    }

    /// `condNode->loopCond_ = cond` — a NO-OP on a node that is not a condition, which is the
    /// reference's own downcast of a non-`CONDITION` node.
    pub(super) fn set_loop_cond(&mut self, condition: NodeId, cond: LoopCondComposite) {
        if let Some(Entry {
            kind: Kind::Condition(held),
            ..
        }) = self.nodes.get_mut(&condition)
        {
            held.loop_cond = Some(cond);
        }
    }

    /// `condNode->coreClCond_`, absent on a loop-guarded condition and on every other kind.
    pub(super) fn core_cl_cond(&self, condition: NodeId) -> Option<v1::CoreClSet> {
        match &self.nodes.get(&condition)?.kind {
            Kind::Condition(cond) => cond.cores.clone(),
            _ => None,
        }
    }

    /// `node->nodeType_`.
    pub fn node_kind(&self, node: NodeId) -> Option<NodeKind> {
        self.nodes.get(&node).map(|entry| entry.kind.node_kind())
    }

    /// `node->nodeType_` **WITH THE NODE'S OWN PAYLOAD** — the one [`Kind`] the entry holds.
    ///
    /// ⭐⭐ ASKING FOR BOTH AT ONCE IS WHAT MAKES A `LOOP` WITHOUT ITS `LoopNode` UNSPELLABLE.
    /// [`Self::node_kind`] is DERIVED from this same enum ([`Kind::node_kind`]), so a caller that
    /// switched on the derived answer and then looked the payload up again had to state what to do
    /// when the two disagreed — which they cannot. See [`super::ddc_store2`]'s `sched_node_of`.
    pub fn kind_of(&self, node: NodeId) -> Option<&Kind> {
        self.nodes.get(&node).map(|entry| &entry.kind)
    }

    /// `node->getOwnerLoop()` — the nearest enclosing `LOOP`, walking `prev_`.
    pub(super) fn owner_loop(&self, node: NodeId) -> Option<LoopId> {
        let mut walk = self.nodes.get(&node)?.parent;
        while let Some(at) = walk {
            let entry = self.nodes.get(&at)?;
            if matches!(entry.kind, Kind::Loop(_)) {
                return Some(LoopId(at));
            }
            walk = entry.parent;
        }
        None
    }

    /// The loop node minted at that id.
    pub(super) fn loop_node(&self, at: LoopId) -> Option<&LoopNode> {
        match &self.nodes.get(&at.0)?.kind {
            Kind::Loop(node) => Some(node),
            _ => None,
        }
    }

    /// `loopNode->denId_ = den`.
    pub(super) fn set_loop_den(&mut self, at: LoopId, den: DatastageId) {
        if let Some(Entry {
            kind: Kind::Loop(node),
            ..
        }) = self.nodes.get_mut(&at.0)
        {
            node.den = den;
        }
    }

    /// `loopNode->dims_ = dims`.
    pub(super) fn set_loop_dims(&mut self, at: LoopId, dims: LoopDims) {
        if let Some(Entry {
            kind: Kind::Loop(node),
            ..
        }) = self.nodes.get_mut(&at.0)
        {
            node.dims = dims;
        }
    }

    /// `traverseTreeDFS()` from the head — every node in pre-order, PARENT BEFORE CHILDREN.
    pub fn dfs(&self) -> Vec<NodeId> {
        let mut order = Vec::new();
        let Some(head) = self.head else {
            return order;
        };
        let mut stack = vec![head];
        while let Some(at) = stack.pop() {
            order.push(at);
            let children = self.children(at);
            stack.extend(children.into_iter().rev());
        }
        order
    }

    /// Every node's name in DFS order.
    pub(super) fn node_names(&self) -> Vec<NodeName> {
        self.dfs()
            .into_iter()
            .filter_map(|node| self.name(node))
            .collect()
    }

    /// HOW MANY NODES THE TREE HOLDS — every node, linked or not, which is what a caller measuring
    /// the stage's effect counts.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// ⭐ HOW MANY NODES OF EACH `nodeType_` THE TREE HOLDS — the per-kind reading of [`Self::len`],
    /// which is the quantity `check.py census` states on the reference side.
    ///
    /// ⛔ EVERY NODE, LINKED OR NOT, EXACTLY AS [`Self::len`] COUNTS THEM, so the counts SUM to it. A
    /// per-kind census taken over [`Self::dfs`] instead would silently omit a minted node the stage
    /// had not linked yet, and the two totals would disagree with no way to tell which was short.
    pub fn node_kinds(&self) -> BTreeMap<NodeKind, usize> {
        let mut census = BTreeMap::new();
        for entry in self.nodes.values() {
            *census.entry(entry.kind.node_kind()).or_insert(0) += 1;
        }
        census
    }

    /// The FIRST node of that name in DFS order, which is how `getLxBelowBlockNode` finds its block.
    pub(super) fn find_named(&self, name: &str) -> Option<NodeId> {
        self.dfs()
            .into_iter()
            .find(|node| self.name(*node).is_some_and(|held| held.0 == name))
    }

    /// `getLxBelowBlockNode(scheduleTree_)`.
    pub(super) fn lx_below_block(&self) -> Option<NodeId> {
        self.find_named(LX_BELOW_BLOCK_NODE_NAME)
    }

    /// The transfer node held at that id.
    pub(super) fn transfer(&self, node: NodeId) -> Option<TransferNode> {
        match &self.nodes.get(&node)?.kind {
            Kind::Transfer(held) => Some(held.clone()),
            _ => None,
        }
    }

    /// The same node written back.
    pub(super) fn set_transfer(&mut self, node: NodeId, transfer: TransferNode) {
        if let Some(entry) = self.nodes.get_mut(&node) {
            entry.name = transfer.name.clone();
            entry.kind = Kind::Transfer(transfer);
        }
    }

    /// `traverseTreeDFS(nullptr, {TRANSFER})` reduced to what entry 208's filter reads.
    pub(super) fn transfers(&self) -> Vec<L3Transfer> {
        self.dfs()
            .into_iter()
            .filter_map(|node| {
                let held = self.transfer(node)?;
                Some(L3Transfer {
                    node,
                    name: held.name.clone(),
                    src: held.src.storage,
                    dst: held.dsts.first().storage,
                })
            })
            .collect()
    }

    /// `syncNode->units_`, [`None`] for a node that is not a `SYNC`.
    pub(super) fn sync_units(&self, node: NodeId) -> Option<crate::schedule::dsc2::SyncUnits> {
        match &self.nodes.get(&node)?.kind {
            Kind::Sync(held) => Some(held.units.clone()),
            _ => None,
        }
    }

    // ⛔ `sync_node`, `loop_at` AND `condition` LIVED HERE AND ARE GONE, SUPERSEDED BY
    // [`Self::kind_of`] — each was `nodes.get(node)?.kind` narrowed to ONE variant, which is what a
    // caller reached for after switching on the DERIVED `node_kind()`, and each therefore made every
    // caller state what to do when the two answers disagreed. They cannot: `kind_of` hands back the
    // one `Kind` and the match narrows it once. See `ddc_store2`'s `sched_node_of`.

    /// `traverseTreeDFSMutable(nullptr, {SYNC})` reduced to what entry 214's sweep reads.
    ///
    /// ⭐ THE OTHER ENDS ARE RESOLVED BY NAME, which is how [`SyncNode::other_ends`] holds them.
    pub(super) fn syncs(&self) -> Vec<L3Sync> {
        self.dfs()
            .into_iter()
            .filter_map(|node| {
                let Kind::Sync(held) = &self.nodes.get(&node)?.kind else {
                    return None;
                };
                let other_ends = held
                    .other_ends
                    .iter()
                    .filter_map(|name| {
                        let at = self.find_named(&name.0)?;
                        match &self.nodes.get(&at)?.kind {
                            Kind::Sync(other) => Some(other.units.clone()),
                            _ => None,
                        }
                    })
                    .collect();
                Some(L3Sync {
                    node,
                    units: held.units.clone(),
                    direction: held.direction,
                    other_ends,
                })
            })
            .collect()
    }

    /// `traverseTreeDFS(nullptr, {ALLOCATE})` as entry 055 reads it — the `ldsIdx_`, the
    /// `component_` and every `(core, address)` the node's `startAddressCoreCorelet_` states.
    pub(super) fn allocations(
        &self,
        placed: &BTreeMap<NodeId, AllocateNode>,
    ) -> Vec<PlacedAllocation> {
        self.dfs()
            .into_iter()
            .filter_map(|node| {
                let Kind::Allocate(_, held) = &self.nodes.get(&node)?.kind else {
                    return None;
                };
                let addresses = placed.get(&node).map_or_else(Vec::new, |node| {
                    node.start_address
                        .at_corelet(Corelet::at::<0>())
                        .into_iter()
                        .map(|(core, bytes)| (core, ByteAddress(bytes.0)))
                        .collect()
                });
                Some(PlacedAllocation {
                    lds: Some(held.lds),
                    component: held.component,
                    addresses,
                })
            })
            .collect()
    }

    /// The L3 view of the allocate node at that id.
    pub(super) fn allocate(&self, node: NodeId) -> Option<(AllocId, L3AllocateNode)> {
        match &self.nodes.get(&node)?.kind {
            Kind::Allocate(alloc, held) => Some((*alloc, held.clone())),
            _ => None,
        }
    }

    /// `allocNode->numBuffers_ = n`.
    pub(super) fn set_buffering(&mut self, node: NodeId, buffering: Buffering) {
        if let Some(Entry {
            kind: Kind::Allocate(_, held),
            ..
        }) = self.nodes.get_mut(&node)
        {
            held.buffering = buffering;
        }
    }

    /// `condition->addThenRegion(block)` / `addElseRegion(block)`, then the link that region is.
    pub(super) fn add_region(&mut self, condition: NodeId, block: NodeId, then_region: bool) {
        if let Some(Entry {
            kind: Kind::Condition(cond),
            ..
        }) = self.nodes.get_mut(&condition)
        {
            if then_region {
                cond.then_region.push(block);
            } else {
                cond.else_region.push(block);
            }
        }
        self.link(block, InsertionPoint::LastIn(condition));
    }

    /// `parent->unlinkChildNode(node)` — the node keeps its own children.
    pub(super) fn unlink(&mut self, node: NodeId) {
        let Some(parent) = self.nodes.get_mut(&node).and_then(|e| e.parent.take()) else {
            return;
        };
        if let Some(entry) = self.nodes.get_mut(&parent) {
            entry.children.retain(|child| *child != node);
        }
    }

    /// `parent->addChildNode(node, addBefore, sibling)`, whose four spellings are
    /// [`InsertionPoint`]'s.
    pub(super) fn link(&mut self, node: NodeId, at: InsertionPoint) {
        let Some((parent, index)) = self.position(at) else {
            return;
        };
        if let Some(entry) = self.nodes.get_mut(&parent) {
            let index = index.min(entry.children.len());
            entry.children.insert(index, node);
        } else {
            return;
        }
        if let Some(entry) = self.nodes.get_mut(&node) {
            entry.parent = Some(parent);
        }
    }

    /// Where an insertion point lands — the parent and the position among its children.
    fn position(&self, at: InsertionPoint) -> Option<(NodeId, usize)> {
        match at {
            InsertionPoint::Before(sibling) | InsertionPoint::After(sibling) => {
                let parent = self.nodes.get(&sibling)?.parent?;
                let position = self
                    .nodes
                    .get(&parent)?
                    .children
                    .iter()
                    .position(|child| *child == sibling)?;
                let after = usize::from(matches!(at, InsertionPoint::After(_)));
                Some((parent, position + after))
            }
            InsertionPoint::FirstIn(parent) => Some((parent, 0)),
            InsertionPoint::LastIn(parent) => {
                Some((parent, self.nodes.get(&parent)?.children.len()))
            }
        }
    }

    /// `node->getMutableParent()->deleteChildNode(&dsc, node)` — the node AND its subtree.
    pub(super) fn delete(&mut self, node: NodeId) {
        self.unlink(node);
        let mut doomed = vec![node];
        while let Some(at) = doomed.pop() {
            let Some(entry) = self.nodes.remove(&at) else {
                continue;
            };
            if let Kind::Allocate(alloc, _) = entry.kind {
                self.node_of_alloc.remove(&alloc);
            }
            self.coordinates.remove(&at);
            doomed.extend(entry.children);
        }
    }

    /// `moveChildren(from, to)` — every child re-parented, in order.
    pub(super) fn move_children(&mut self, from: NodeId, to: NodeId) {
        let children = self.children(from);
        if let Some(entry) = self.nodes.get_mut(&from) {
            entry.children.clear();
        }
        for child in children {
            if let Some(entry) = self.nodes.get_mut(&child) {
                entry.parent = Some(to);
            }
            if let Some(entry) = self.nodes.get_mut(&to) {
                entry.children.push(child);
            }
        }
    }
}

/// ONE LABELLED DS'S MEMORY ORGANISATION — `labeledDs_.at(lds).memOrg_`.
///
/// ⭐ ITS `allocateNode_` IS ONE NODE READ THROUGH TWO VOCABULARIES: [`L3AllocateNode`] is what
/// entry 353 mints and [`AllocateNode`] is what entries 219/220/292 place, and both live here so
/// that [`crate::schedule::l3::dl_ops::AllocationSites`] and [`MemOrg`] cannot disagree about the
/// address on it.
#[derive(Debug, Default)]
pub struct Org {
    data: RefCell<OrgData>,
}

/// What one `memOrg_` holds, per storage.
#[derive(Debug, Default)]
struct OrgData {
    /// `memOrg_[storage].isPresent` FUSED WITH its `allocateNode_` identity.
    nodes: BTreeMap<SenComponent, NodeId>,
    /// `allocateNode_->allocUsers_`.
    users: BTreeMap<SenComponent, Vec<NodeId>>,
    /// The L3 view of that node.
    minted: BTreeMap<SenComponent, L3AllocateNode>,
    /// The ddc view of the same node.
    placed: BTreeMap<SenComponent, AllocateNode>,
    /// `memOrg_[storage].isZeroPadded != NOZEROPAD`.
    zero_padded: BTreeSet<SenComponent>,
}

impl Org {
    /// The organisation of a labelled DS that names one storage already — the seed an input
    /// super-DSC's own allocate node is.
    pub(super) fn seeded(storage: SenComponent, node: NodeId, held: L3AllocateNode) -> Self {
        let org = Self::default();
        {
            let mut data = org.data.borrow_mut();
            data.nodes.insert(storage, node);
            data.minted.insert(storage, held);
        }
        org
    }

    /// `memOrg_[storage].isPresent = true; .allocateNode_ = node`.
    pub(super) fn set_node(&self, storage: SenComponent, node: NodeId, held: L3AllocateNode) {
        let mut data = self.data.borrow_mut();
        data.nodes.insert(storage, node);
        data.minted.insert(storage, held);
    }

    /// `memOrg_.at(storage).allocateNode_` BY IDENTITY.
    pub(super) fn node(&self, storage: SenComponent) -> Option<NodeId> {
        self.data.borrow().nodes.get(&storage).copied()
    }

    /// Which storage this organisation holds `node` at.
    pub(super) fn storage_of(&self, node: NodeId) -> Option<SenComponent> {
        self.data
            .borrow()
            .nodes
            .iter()
            .find(|(_, held)| **held == node)
            .map(|(storage, _)| *storage)
    }

    /// `allocNode->addAllocUser(user)`.
    pub(super) fn add_user(&self, storage: SenComponent, user: NodeId) {
        self.data
            .borrow_mut()
            .users
            .entry(storage)
            .or_default()
            .push(user);
    }

    /// The ddc view of the node at that storage.
    pub(super) fn placed(&self, storage: SenComponent) -> Option<AllocateNode> {
        self.data.borrow().placed.get(&storage).cloned()
    }

    /// The same, written back.
    pub(super) fn set_placed(&self, storage: SenComponent, node: AllocateNode) {
        self.data.borrow_mut().placed.insert(storage, node);
    }

    /// `getPageSize()` on the node at that storage — every layout dim the allocation BOUNDS.
    ///
    /// ⭐ AN UNBOUNDED `maxDimSizes_` ENTRY IS THE REFERENCE'S `-1` AND PAGES NOTHING, which is why
    /// a freshly minted allocation's page set is empty rather than absent.
    fn page_sizes(&self, storage: SenComponent) -> Option<BTreeMap<PrimaryDim, Extent>> {
        let data = self.data.borrow();
        let held = data.minted.get(&storage)?;
        Some(
            held.layout
                .0
                .iter()
                .filter_map(|(dim, max)| {
                    let max = i64::try_from((*max)?.0).ok()?;
                    Some((*dim, Extent(max)))
                })
                .collect(),
        )
    }
}

impl MemOrg for Org {
    fn hbm_pinned(&self) -> bool {
        self.data.borrow().nodes.contains_key(&SenComponent::Hbm)
    }

    fn lx_buffering(&self) -> Option<Buffering> {
        self.data
            .borrow()
            .minted
            .get(&SenComponent::Lx)
            .map(|held| held.buffering)
    }

    fn lx_start_address(&self, at: &AddressCoord) -> Option<ByteAddress> {
        let placed = self.placed(SenComponent::Lx)?;
        placed
            .start_address
            .at(at.core, at.corelet)
            .map(|bytes| ByteAddress(bytes.0))
    }

    fn lx_buffer_offset(&self, core: Core, corelet: Corelet) -> Option<BufferOffset> {
        let placed = self.placed(SenComponent::Lx)?;
        placed
            .placement
            .buffer_offset
            .get(&core)?
            .get(&corelet)
            .map(|bytes| BufferOffset(bytes.0))
    }

    fn hbm_indirection(&self) -> Option<IndirectAlloc> {
        self.data.borrow().minted.get(&SenComponent::Hbm)?.indirect
    }

    fn hbm_allocation(&self) -> Option<NodeName> {
        self.data
            .borrow()
            .minted
            .get(&SenComponent::Hbm)
            .map(|held| held.name.clone())
    }

    fn hbm_layout_dims(&self) -> Option<LayoutDims> {
        let data = self.data.borrow();
        let layout = &data.minted.get(&SenComponent::Hbm)?.layout;
        let (first, rest) = layout.0.split_first()?;
        Some(LayoutDims::new(
            first.0,
            rest.iter().map(|(dim, _)| *dim).collect(),
        ))
    }

    fn lx_zero_padded(&self) -> Option<bool> {
        Some(self.data.borrow().zero_padded.contains(&SenComponent::Lx))
    }

    fn hbm_page_sizes(&self) -> Option<BTreeMap<PrimaryDim, Extent>> {
        self.page_sizes(SenComponent::Hbm)
    }

    fn lx_padding(&self) -> Option<PaddingForm> {
        self.data
            .borrow()
            .minted
            .get(&SenComponent::Lx)
            .map(|held| held.padding.clone())
    }

    fn lx_page_sizes(&self) -> BTreeMap<PrimaryDim, Extent> {
        self.page_sizes(SenComponent::Lx).unwrap_or_default()
    }

    fn hbm_alloc_users(&self) -> Option<Vec<NodeId>> {
        let data = self.data.borrow();
        data.nodes.get(&SenComponent::Hbm)?;
        Some(
            data.users
                .get(&SenComponent::Hbm)
                .cloned()
                .unwrap_or_default(),
        )
    }

    fn lx_alloc_users(&self) -> Option<Vec<NodeId>> {
        let data = self.data.borrow();
        data.nodes.get(&SenComponent::Lx)?;
        Some(
            data.users
                .get(&SenComponent::Lx)
                .cloned()
                .unwrap_or_default(),
        )
    }
}

/// A fresh L3 allocate node over the layout dims a DSC states for that labelled DS, with every
/// `maxDimSizes_` entry UNBOUNDED — the reference's `resize(n, -1)` (`dsc/dsc2.h:982`).
pub(super) fn seed_allocate_node(
    name: NodeName,
    lds: LdsIdx,
    component: SenComponent,
    layout: &LayoutDims,
) -> L3AllocateNode {
    L3AllocateNode {
        name,
        lds,
        component,
        buffering: Buffering::None,
        layout: AllocLayout(
            layout
                .iter()
                .map(|dim| (dim, None::<Elements>))
                .collect::<Vec<_>>(),
        ),
        padding: PaddingForm::default(),
        indirect: None,
        related_indirect: None,
    }
}
