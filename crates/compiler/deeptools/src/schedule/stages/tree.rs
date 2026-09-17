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
use crate::schedule::ddl::conversion::LdsSlot;
use crate::schedule::dsc2::{
    AllocateNode, ComputeNode, Coordinate, Dsts, Hops, LayoutDims, LdsIdx, LoopCondComposite,
    NodeName, Operand, StickMaskNode, SyncNode, TransferNode,
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
/// ⛔⛔ `Compute` AND `StickMask` HAVE PRODUCERS — THE SENTENCE THAT STOOD HERE, *"NOTHING CONSTRUCTS
/// THEM YET"*, WAS STALE AND IS WITHDRAWN. `Kind::Compute` is constructed at four callsites, all of
/// them writing the tree this file holds: `add_compute` (`stages/ddc_sites.rs:1440`, parented under a
/// given block) and `mint_compute` (`:1448`, unparented), reached from the DDL walk's `op_compute`
/// (`schedule/ddl/conversion.rs:3541`) and `op_opaque` (`:3748`); and `mint_compute`
/// (`stages/ddc_store.rs:188`) with the clone beside it (`:1485`). `Kind::StickMask` is constructed by
/// `mint_sched_node` (`stages/ddc_tree.rs:701`). ⛔ `stages/ddc_store.rs:29-31` still carries the
/// withdrawn claim in its own header.
///
/// ⛔ WHAT REMAINS TRUE IS A READING AND NOT AN ABSENCE, AND IT IS THE SCRATCHY SIDE THAT IS MEASURED:
/// `sync 0, compute 0` over 24,363 programs
/// (`crates/targets/spyre/src/superdsc_to_l3_sdsc.rs`'s corpus report), so nothing on that corpus
/// reaches those sites. ⛔ AND NOTHING HERE FABRICATES A NODE TO MOVE A CENSUS: a stand-in node is the
/// fabricated-placement failure this crate ranks worse than a stop.
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
    /// ⭐⭐ `relevantComps_` (`dsc/dsc2.h:517`) — the per-node `comp -> core -> corelets` map, held
    /// BESIDE the entries because it is a field of EVERY `ScheduleNode` and of no one kind.
    ///
    /// ⛔⛔ AN ABSENT ENTRY IS `relevantComps_.empty()`, WHICH IS NOT THE SAME FACT AS "THIS
    /// COMPONENT IS NOT RELEVANT". The reference's map starts empty on every node and stays empty
    /// until `dsc.setRelevantCompCoreCl()` (`dsc/dsc2.cpp:2647-2712`) runs — the reference's own
    /// serialisation shows exactly that state, `"relevantComps_" : {}` on every one of the three
    /// nodes of `ddc/test/int64_arith/add_i32_to_i32/sdsc_alxs_input_Add.json`'s `scheduleTree_` —
    /// and `isNodeRelevant(comp, -1, -1)` answers FALSE for every component while it is
    /// (`dsc/dsc2.cpp:1921-1932`). [`Self::relevant_comps`] hands back [`None`] for a node in that
    /// state and `Some(map)` for one the pass has stated, so a reader can tell the two apart rather
    /// than reading a constant off either.
    ///
    /// ⛔ AND NEITHER CONSTANT IS A SAFE FILLER FOR THE MISSING PASS: `last_fusable_loop`
    /// (`ddc/v1.rs:3144`) turns `is_relevant`'s answer straight into which loop a transfer fuses
    /// under, so `true` and `false` each state a fact about every node in the tree.
    relevant_comps: BTreeMap<NodeId, BTreeMap<SenComponent, v1::CoreClSet>>,
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

    /// `scheduleTree_.getHead()->next_.at(0)` — the root block, set once when the tree is seeded.
    pub(super) fn set_head(&mut self, head: NodeId) {
        self.head = Some(head);
    }

    /// ⭐⭐ THE FIRST **REAL** NODE OF `scheduleTree_`, WHICH IS NOT `getHead()`. `ScheduleTree` holds
    /// `LoopNode head_` BY VALUE (`dsc/dsc2.h:622`) as an UNNAMED SENTINEL — `empty()` is
    /// `head_.next_.empty()` (`:626`), `getHead()` hands back `&head_` (`:636`), and its `name_` is
    /// `""`, which is why every top-level node serialises with `"prev_" : ""`
    /// (`dsc/designSpaceConfig.cpp:379-380`: `prevName = node->prev_ ? node->prev_->name_ : ""`).
    /// This id is the sentinel's first child — the `block "root_level_operations"` that is entry `[0]`
    /// of a scheduled `scheduleTree_` array.
    ///
    /// ⛔ THE SENTINEL HAS NO ENTRY IN THIS TREE AND NEEDS NONE: nothing reads a name, a kind, a
    /// parent or children off it, and its one field anything asks for is its `denId_`, which is
    /// [`Self::head_den`] — a field of the tree and not of a node.
    pub const fn head(&self) -> Option<NodeId> {
        self.head
    }

    /// `getHeadMutable()->denId_ = den` — the SENTINEL's `denId_`, per [`Self::head`].
    pub(super) fn set_head_den(&mut self, den: DatastageId) {
        self.head_den = Some(den);
    }

    /// `getHead()->denId_` — the SENTINEL's, which serialises as `scheduleTreeHeadDenId_`
    /// (`dsc/designSpaceConfig.cpp:368`) and which `ScheduleTree()` initialises to the core datastage
    /// (`dsc/dsc2.h:628`).
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

    /// ⭐⭐ `node->relevantComps_` (`dsc/dsc2.h:517`) — [`None`] where that map is EMPTY, which is
    /// every node of this tree until `dsc.setRelevantCompCoreCl()` (`dsc/dsc2.cpp:2647-2712`) has
    /// stated it. See the field for why the two are different answers and not one default.
    ///
    /// ⭐ EVERY QUESTION THE REFERENCE ASKS OF THAT MAP IS A READ OF THIS ONE VALUE, AND THAT IS WHY
    /// none of them is ported here: `isNodeRelevant(comp, clId, coreId)` (`dsc/dsc2.cpp:1916-1932`)
    /// is `get(comp)` and then `get(coreId)` and then `contains(clId)`; `getRelevantComps(coreId,
    /// clId)` (`:1947-1975`) is the components whose entry survives that same filter; and
    /// `getRelevantCoreCl(comp)` (`:1934-1945`) is the union of the entries it keeps. Each has its
    /// own `DT_ERROR` arms for the argument combinations its caller may not pass, so each belongs to
    /// the unit that calls it — this is the map they all read.
    pub fn relevant_comps(&self, node: NodeId) -> Option<BTreeMap<SenComponent, v1::CoreClSet>> {
        self.relevant_comps.get(&node).cloned()
    }

    /// `node->relevantComps_[comp] = core_cl` — the ONE write `setRelevantCompCoreCl`
    /// (`dsc/dsc2.cpp:2647-2712`) performs, per node per component.
    ///
    /// ⭐ ONE COMPONENT'S ENTRY, REPLACED, IS THE WHOLE WRITE SURFACE THAT PASS NEEDS. Its head seed
    /// (`:2655`), its `relevantComps_[NO_COMPONENT] = prev_->relevantComps_.at(NO_COMPONENT)`
    /// inheritance (`:2659-2661`), the `emplace`/`set_intersect` on the then arm and the
    /// `set_diff`/`erase` on the else arm (`:2666-2683`), and the `relCoreCls[core].insert(..)` union
    /// walked up `prev_` (`:2687-2697`) each take ONE component's `core -> corelets` map, change it
    /// and put it back — which [`Self::relevant_comps`] hands out by value and this puts back.
    ///
    /// ⛔ AND IT NEVER MINTS AN EMPTY MAP: a node becomes stated by naming a component, so
    /// [`Self::relevant_comps`]' [`None`] stays the single spelling of `relevantComps_.empty()`.
    pub(super) fn set_relevant_comps(
        &mut self,
        node: NodeId,
        comp: SenComponent,
        core_cl: v1::CoreClSet,
    ) {
        self.relevant_comps
            .entry(node)
            .or_default()
            .insert(comp, core_cl);
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

    /// `traverseTreeDFS()` — every node in pre-order, PARENT BEFORE CHILDREN.
    ///
    /// ⭐⭐ IT STARTS AT THE SAME NODE THE REFERENCE DOES, AND THE MATCH IS NOT A COINCIDENCE.
    /// `traverseTreeDFS(nullptr, ..)` seeds its queue from `head_.next_` and so **EXCLUDES** the head
    /// (`dsc/dsc2.cpp:2231-2234`), while this seeds from [`Self::head`] and INCLUDES it — and the two
    /// agree because that id is `head_.next_.at(0)` and not `getHead()`: the reference's head is the
    /// unnamed `LoopNode` sentinel this tree holds no entry for. Its ordering is the same too — a
    /// `deque` with the children `push_front`ed in reverse (`:2259-2261`) against this stack with the
    /// children pushed in reverse — so a node's subtree is walked before its next sibling on both.
    ///
    /// ⛔ WHAT THIS IS **NOT** IS `traverseTreeDFS`'S FILTERS. That body drops a node whose
    /// `isNodeRelevant(comp, clId, coreId)` is false and does not descend into it (`:2245-2247`), skips
    /// an `excludeList` member, keeps only the requested `nodeTypes`, and stops descending a `LOOP`
    /// past `maxLoopDepth` (`:2255-2256`). This is the unfiltered walk; each projection above states
    /// which of those it reduces to, and the relevance filter needs
    /// [`Self::relevant_comps`] — see that field.
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

    /// The compute node held at that id — `static_cast<dsc2::ComputeNode *>(node)`, [`None`] for
    /// any other `nodeType_`.
    pub(super) fn compute(&self, node: NodeId) -> Option<ComputeNode> {
        match &self.nodes.get(&node)?.kind {
            Kind::Compute(held) => Some(held.clone()),
            _ => None,
        }
    }

    /// The same node written back — `name_` follows the body exactly as [`Self::set_transfer`]'s
    /// does, because `name_` is ONE `ScheduleNode` field and not a second copy of it.
    pub(super) fn set_compute(&mut self, node: NodeId, compute: ComputeNode) {
        if let Some(entry) = self.nodes.get_mut(&node) {
            entry.name = compute.name.clone();
            entry.kind = Kind::Compute(compute);
        }
    }

    /// ⭐⭐ ONE OPERAND SLOT'S `myLdsIdx_`, WRITTEN — a COMPUTE's
    /// `inputsLdsAndLoopOffsets_.at(i).myLdsIdx_` / `outputsLdsAndLoopOffsets_.at(i).myLdsIdx_`
    /// (`dsc/dsc2.h:722` on `:935-938`), a TRANSFER's `srcLdsAndLoopOffsets_.myLdsIdx_` /
    /// `dstLdsAndLoopOffsets_.at(i).myLdsIdx_` (`:820`), and an ALLOCATE's `ldsIdx_` (`:979`) — which
    /// is the write half of [`super::Dsc2Ddl`]'s `slot_lds` (`stages/ddc_sites.rs:1111`).
    ///
    /// ⛔⛔ TOTAL OVER [`LdsSlot`], WHICH IS WHAT MAKES A PARTIAL RETAG UNSPELLABLE. Entry 173
    /// renumbers `labeledDs_` and then retags EVERY slot naming the old last index
    /// (`schedule/ddl/conversion.rs:445-457`); retagging only the transfer slots would leave every
    /// compute naming the old index against a renumbered list, which is a silently wrong operand and
    /// not a missing one. The match below is exhaustive on the enum, so a slot kind added later is an
    /// E0004 here and not a silent skip.
    ///
    /// ⛔ [`LdsSlot::OpaqueCompute`] IS THE ONE ARM THIS TREE DOES NOT HOLD, BY THE TRAIT'S OWN
    /// CONTRACT AND NOT BY OMISSION: it is `metadata_.opaqueOps_.at(cn).ldsIdx_`, `slot_lds` answers
    /// [`None`] for it, and its one caller retags it through `metadata.opaque_ops` itself and reaches
    /// this method only in the `else` arm (`schedule/ddl/conversion.rs:446-456`) — so no write is
    /// dropped here.
    ///
    /// ⛔ A NODE THIS TREE DOES NOT HOLD, A KIND THAT DOES NOT MATCH THE SLOT, AND AN INDEX PAST THE
    /// END ARE EACH NO WRITE — the reference reaches each slot from a walk of this same tree
    /// (`lds_slots`, `stages/ddc_sites.rs:1070`), so none of the three is constructible from one.
    pub(super) fn set_slot_lds(&mut self, slot: LdsSlot, lds: LdsIdx) {
        match slot {
            LdsSlot::OpaqueCompute(_) => {}
            LdsSlot::ComputeInput(node, at) => {
                if let Some(Entry {
                    kind: Kind::Compute(held),
                    ..
                }) = self.nodes.get_mut(&node)
                    && let Some(operand) = held.inputs.get_mut(at)
                {
                    operand.data.my_lds_idx = Some(lds);
                }
            }
            LdsSlot::ComputeOutput(node, at) => {
                if let Some(Entry {
                    kind: Kind::Compute(held),
                    ..
                }) = self.nodes.get_mut(&node)
                    && let Some(operand) = held.outputs.get_mut(at)
                {
                    operand.data.my_lds_idx = Some(lds);
                }
            }
            LdsSlot::TransferSrc(node) => {
                if let Some(Entry {
                    kind: Kind::Transfer(held),
                    ..
                }) = self.nodes.get_mut(&node)
                {
                    held.src.data.my_lds_idx = Some(lds);
                }
            }
            LdsSlot::TransferDst(node, at) => {
                if let Some(Entry {
                    kind: Kind::Transfer(held),
                    ..
                }) = self.nodes.get_mut(&node)
                {
                    set_dst_lds(&mut held.dsts, at, lds);
                }
            }
            LdsSlot::Allocate(node) => {
                if let Some(Entry {
                    kind: Kind::Allocate(_, held),
                    ..
                }) = self.nodes.get_mut(&node)
                {
                    held.lds = lds;
                }
            }
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
            // ⛔ `relevantComps_` IS A FIELD OF THE DELETED NODE AND GOES WITH IT — the reference's
            // `deleteChildNode` destroys the `unique_ptr` and the map inside it, so a later node
            // reissued at this id must not inherit a dead node's relevance.
            self.relevant_comps.remove(&at);
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
    /// `memOrg_[storage].isZeroPadded != ZpType::NOZEROPAD` (`dsc/dscdefn.h:232`, `:310`) — and
    /// **ABSENT WHERE NOTHING HAS STATED IT**, which is the whole point of the map.
    ///
    /// ⛔⛔ IT IS AN INPUT AND THIS BRIDGE HAS NO WRITER FOR IT. Every assignment to `isZeroPadded` in
    /// the reference is in `dsm/` — `dsm/dsm.cpp:6661`, `:6700`, `:7657`, `dsm/dsmperf.cpp:1143`,
    /// `dsm/translators/perfDscToSdsc/perfDscToSdsc.cpp:1255-1274` — or in the SDSC parse that reads
    /// it back (`dsc/designSpaceConfig.cpp:7092`, against the emit at `:6439`). `dcg/` and `ddc/` only
    /// READ it (`L3DlOpsScheduler.cpp:3854`, `:5324`). So it belongs to the parsed super-DSC,
    /// [`crate::schedule::l3::dsc::Pinning`] carries `isPadded` and not this, and a `false` filled in
    /// here would be a fact about every organisation that nothing stated.
    ///
    /// ⛔ THIS USED TO BE A `BTreeSet<SenComponent>` WITH NO WRITER ANYWHERE IN THE CRATE, so
    /// [`MemOrg::lx_zero_padded`] answered `Some(false)` for every organisation the stages build while
    /// `schedule/l3/dl_ops.rs:9301` and `:13597` branched on it — a fabricated value reaching live
    /// code. A [`bool`] per component makes "stated NOZEROPAD" and "nobody said" different answers;
    /// [`super::state::DscState::seeded`] states the ones the reference's own check pins and leaves the
    /// rest unstated.
    zero_padded: BTreeMap<SenComponent, bool>,
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

    /// `allocNode->allocUsers_.push_back({user, 1})` — the RAW PUSH its caller performs
    /// (`super::Dsc2Store`'s `add_alloc_user`, `stages/ddc_store2.rs:1026-1030`), so a repeat user is
    /// a SECOND entry and not a bumped count.
    ///
    /// ⭐ WHICH IS WHY THE COUNT IS MULTIPLICITY HERE. `allocUsers_` is
    /// `vector<pair<const ScheduleNode*, int>>` (`dsc/dsc2.h:1012`), and this list holds each
    /// reference as one element: `[{user, 1}, {user, 1}]` there is `[user, user]` here, and
    /// `addAllocUser`'s `refCount++` on an existing entry (`:1017-1024`) is the same one-more-element.
    /// Both readers of the field in scope — `getNextView`-free walks over `allocUsers_`
    /// (`L3DlOpsScheduler.cpp:5343`) and `hasAllocUsers()` — see the same thing either way.
    pub(super) fn add_user(&self, storage: SenComponent, user: NodeId) {
        self.data
            .borrow_mut()
            .users
            .entry(storage)
            .or_default()
            .push(user);
    }

    /// `allocNode->removeAllocUser(user)` (`dsc/dsc2.h:1022-1034`) — ONE reference dropped, and the
    /// user gone with its last one.
    ///
    /// ⭐ REMOVING ONE OCCURRENCE **IS** THAT DECREMENT under [`Self::add_user`]'s multiplicity
    /// encoding: the reference finds the FIRST entry naming the node, does `--(it->second)` and
    /// `erase`s it at zero, which on `[{user, 2}]` leaves `[{user, 1}]` and on
    /// `[{user, 1}, {user, 1}]` leaves `[{user, 1}]` — one reference fewer in both, which is one
    /// element fewer here.
    ///
    /// ⛔ ITS *"RemoveAllocUser: Schedule node <n> is not in the user list of allocate node <a>"* IS
    /// AN ABSENT ENTRY, AND THAT IS NO WRITE — the same reading
    /// [`crate::schedule::ddc::transformation_util::DdcAllocateNode::remove_alloc_user`] already
    /// states of the same body: a node that was never a user cannot lose a reference.
    pub(super) fn remove_alloc_user(&self, storage: SenComponent, user: NodeId) {
        let mut data = self.data.borrow_mut();
        if let Some(users) = data.users.get_mut(&storage)
            && let Some(at) = users.iter().position(|held| *held == user)
        {
            users.remove(at);
        }
    }

    /// `memOrg_[storage].isZeroPadded`, AS THE ONE QUESTION ITS READERS ASK — `!= ZpType::NOZEROPAD`
    /// (`L3DlOpsScheduler.cpp:3854`, `:5324`). ⛔ The three-way `ZpType` is deliberately not projected:
    /// `ALLZEROPAD` and `ROWZEROPAD` are told apart only in `dsi/` and `dm/`, neither of which is on
    /// this bridge, and a variant nothing here can construct would be a filler with no source.
    pub(super) fn set_zero_padded(&self, storage: SenComponent, zero_padded: bool) {
        self.data
            .borrow_mut()
            .zero_padded
            .insert(storage, zero_padded);
    }

    /// The ddc view of the node at that storage.
    pub(super) fn placed(&self, storage: SenComponent) -> Option<AllocateNode> {
        self.data.borrow().placed.get(&storage).cloned()
    }

    /// The same, written back.
    pub(super) fn set_placed(&self, storage: SenComponent, node: AllocateNode) {
        self.data.borrow_mut().placed.insert(storage, node);
    }

    /// EVERY PLACED NODE THIS ORGANISATION HOLDS, BY IDENTITY — the `(allocateNode_, its ddc view)`
    /// pairs [`super::state::DscTree::with_placed`] gathers so that a walk keyed by [`NodeId`] reads
    /// the SAME cell [`Self::placed`] answers from.
    ///
    /// ⛔ A STORAGE WITH A ddc VIEW BUT NO `allocateNode_` IS SKIPPED AND IS NOT A REFUSAL: it cannot
    /// arise — every write of one goes through [`Self::set_node`] first — and inventing an identity
    /// for it would name a node the tree does not hold.
    pub(super) fn placed_nodes(&self) -> Vec<(NodeId, AllocateNode)> {
        let data = self.data.borrow();
        data.placed
            .iter()
            .filter_map(|(storage, held)| Some((*data.nodes.get(storage)?, held.clone())))
            .collect()
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

    /// ⛔ [`None`] WHERE NOTHING HAS STATED `memOrg_.at(LX).isZeroPadded`, WHICH IS NOT `Some(false)`.
    /// The trait already gives [`None`] a meaning — the `DT_CHECK_MSG(isPadded, ..)` beside the read
    /// failing (`L3DlOpsScheduler.cpp:5325`) — and an unstated value lands on the same answer for the
    /// same reason: neither is a program this bridge can carry on lowering by guessing a flag for.
    /// See [`OrgData::zero_padded`] for why nothing here can state it, and
    /// [`super::state::DscState::seeded`] for the ones it can.
    fn lx_zero_padded(&self) -> Option<bool> {
        self.data
            .borrow()
            .zero_padded
            .get(&SenComponent::Lx)
            .copied()
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

/// `dstLdsAndLoopOffsets_.at(index).myLdsIdx_ = lds` — one destination of one transfer, in place, and
/// NO WRITE for an index past the end (which is [`Dsts::get`]'s own *"absent past the end"*).
///
/// ⭐ THROUGH A REBUILD BECAUSE THAT IS THIS CRATE'S ONE SPELLING FOR IT. [`Dsts`] holds its first
/// destination and the rest privately and exposes `first_mut()` and no `get_mut`, so
/// `clone_transfer_for_pe_sfp_work_split` (`schedule/ddc/transformation_util.rs:1668-1677`) already
/// writes every destination of a transfer by collecting `routes()` and `iter()`, changing the
/// operands and putting a fresh [`Dsts`] back — the same three lines, for one destination.
fn set_dst_lds(dsts: &mut Dsts, index: usize, lds: LdsIdx) {
    let hops: Vec<Hops> = dsts.routes().map(|(_, hops)| Hops(hops.to_vec())).collect();
    let mut operands: Vec<Operand> = dsts.iter().cloned().collect();
    let Some(operand) = operands.get_mut(index) else {
        return;
    };
    operand.data.my_lds_idx = Some(lds);
    if let Some((first, rest)) = operands.split_first() {
        *dsts = Dsts::new(first.clone(), rest.to_vec()).with_hops(hops);
    }
}

/// A fresh L3 allocate node over the layout dims a DSC states for that labelled DS, with every
/// `maxDimSizes_` entry UNBOUNDED — the reference's `resize(n, -1)` (`dsc/dsc2.h:982`).
///
/// ⛔ `ignoreSymbolicVolumeLimits_` AND `backGapCore_` ARE THE MEMBER INITIALIZERS AND NOT A CHOICE
/// MADE HERE (`dsc/dsc2.h:1002`, `:989`). This seed stands for the HBM allocate node an input
/// super-DSC already carries, and the SDSC parser is one of the three things in the reference that
/// EVER writes those two (`dsc/dsc2.cpp:1786`, `:1803`) — measured `{}` and `0` on all 1,899 allocate
/// nodes of `/Users/nickm/tmp/bridge1-fixtures/g0/debug/sdsc_*/sdsc.json`, so the wire states the
/// initializer on every one. ⭐ AND NO SEEDED NODE REACHES THE CAPACITY WALK ANYWAY: entry 222 sizes
/// only what `newAllocations_` holds, which `create_allocate_node` fills with nodes IT minted.
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
        ignore_symbolic_volume_limits: false,
        back_gap_dims: BTreeSet::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::ddl::ops::DdlComputeType;
    use crate::schedule::dsc2::{
        DataInfo, InstrAttribute, NumChunks, ReplicationFactor, TransferPadding,
    };
    use crate::units::NumFolds;

    fn core_zero() -> Core {
        Core::checked(0).expect("core 0")
    }

    fn operand(lds: Option<u32>) -> Operand {
        Operand {
            unit: SenComponent::Lx,
            storage: SenComponent::NoComponent,
            data: DataInfo {
                my_lds_idx: lds.map(LdsIdx),
                ..DataInfo::default()
            },
        }
    }

    fn seeded_root() -> (TreeData, NodeId) {
        let mut tree = TreeData::default();
        let root = tree.add(
            NodeName("root_level_operations".to_owned()),
            Kind::Block,
            None,
        );
        tree.set_head(root);
        (tree, root)
    }

    /// ⭐ THE READ THAT TELLS "NOBODY STATED IT" FROM "STATED, AND THIS COMPONENT IS NOT IN IT".
    ///
    /// The first is the reference's `relevantComps_.empty()` — the state every node of
    /// `ddc/test/int64_arith/add_i32_to_i32/sdsc_alxs_input_Add.json`'s `scheduleTree_` serialises as
    /// `"relevantComps_" : {}` — and the second is what `setRelevantCompCoreCl`'s head seed leaves
    /// (`dsc/dsc2.cpp:2653-2656`: `relevantComps_[NO_COMPONENT] = {core -> {0}}` at
    /// `numCoreletsUsed_DSC2_ == 1`). `isNodeRelevant(comp, -1, -1)` reads FALSE on both
    /// (`dsc/dsc2.cpp:1921-1926`), which is exactly why they must not be one answer here.
    #[test]
    fn an_unstated_relevance_is_not_a_stated_absence() {
        let (mut tree, root) = seeded_root();
        assert_eq!(tree.relevant_comps(root), None);

        let seed = v1::CoreClSet(BTreeMap::from([(
            core_zero(),
            BTreeSet::from([Corelet::at::<0>()]),
        )]));
        tree.set_relevant_comps(root, SenComponent::NoComponent, seed.clone());

        let stated = tree.relevant_comps(root);
        assert_eq!(
            stated
                .as_ref()
                .and_then(|held| held.get(&SenComponent::NoComponent)),
            Some(&seed)
        );
        // ⭐ `Some(false)` AND NOT `false`: the node IS stated, and the component it was asked about is
        // not in its map — which is the answer `None` above is not.
        assert_eq!(
            stated.map(|held| held.contains_key(&SenComponent::Lx)),
            Some(false)
        );
    }

    /// ⛔ `relevantComps_` DIES WITH ITS NODE — the reference's `deleteChildNode` destroys the
    /// `unique_ptr` and the map inside it, so a node reissued at this id must not inherit it.
    #[test]
    fn deleting_a_node_drops_its_relevance() {
        let (mut tree, root) = seeded_root();
        let child = tree.add(NodeName("b".to_owned()), Kind::Block, Some(root));
        tree.set_relevant_comps(child, SenComponent::Lx, v1::CoreClSet::default());
        assert!(tree.relevant_comps(child).is_some());

        tree.delete(child);
        assert_eq!(tree.relevant_comps(child), None);
        // The parent's own entry is untouched: only the deleted subtree loses its fields.
        assert_eq!(tree.relevant_comps(root), None);
    }

    /// ⭐⭐ EVERY SLOT KIND THIS TREE HOLDS IS RETAGGED BY ONE CALL — the whole point of
    /// [`TreeData::set_slot_lds`] being total, because entry 173 renumbers `labeledDs_` and a slot it
    /// misses names the OLD last index against the new list.
    ///
    /// ⭐ EACH ONE IS READ BACK THROUGH THE SAME FIELD `slot_lds` READS (`stages/ddc_sites.rs:1111`),
    /// and the fixture deliberately gives the compute TWO inputs and the transfer TWO destinations so
    /// that neither `at` is 0 and a first-only write would fail.
    #[test]
    fn every_slot_kind_is_retagged() {
        let (mut tree, root) = seeded_root();
        let old = LdsIdx(7);
        let new = LdsIdx(9);

        let compute = tree.add(
            NodeName("c".to_owned()),
            Kind::Compute(ComputeNode {
                name: NodeName("c".to_owned()),
                op: DdlComputeType::Macc,
                ex_unit: SenComponent::Ptrow0,
                inputs: vec![operand(None), operand(Some(old.0))],
                outputs: vec![operand(None), operand(Some(old.0))],
                num_folds_engaged: NumFolds::ONE,
                data_format: None,
                instr_attribute: InstrAttribute::default(),
            }),
            Some(root),
        );
        let transfer = tree.add(
            NodeName("t".to_owned()),
            Kind::Transfer(TransferNode {
                name: NodeName("t".to_owned()),
                src: operand(Some(old.0)),
                dsts: Dsts::new(operand(None), vec![operand(Some(old.0))])
                    .with_hops(vec![Hops(vec![SenComponent::L0]), Hops::default()]),
                replication_factor: ReplicationFactor::ONE,
                unit_time_transfer_chunk_size: Vec::new(),
                unit_time_transfer_num_chunks: NumChunks::ONE,
                padding: TransferPadding::default(),
                src_indirect: None,
                dst_indirect: None,
                core_id_to_gtr_info: BTreeMap::new(),
                transfer_size: BTreeMap::new(),
            }),
            Some(root),
        );
        let alloc = tree.fresh_alloc();
        let allocate = tree.add(
            NodeName("a".to_owned()),
            Kind::Allocate(
                alloc,
                seed_allocate_node(
                    NodeName("a".to_owned()),
                    old,
                    SenComponent::Lx,
                    &LayoutDims::new(PrimaryDim::Out, Vec::new()),
                ),
            ),
            Some(root),
        );

        for slot in [
            LdsSlot::ComputeInput(compute, 1),
            LdsSlot::ComputeOutput(compute, 1),
            LdsSlot::TransferSrc(transfer),
            LdsSlot::TransferDst(transfer, 1),
            LdsSlot::Allocate(allocate),
        ] {
            tree.set_slot_lds(slot, new);
        }

        // ⭐ EVERY ASSERTION BELOW IS A VALUE LIST AND NOT A GUARDED ARM: a kind that failed to match
        // reads as an EMPTY list here and fails, rather than skipping the assertions silently.
        let (inputs, outputs) = match tree.kind_of(compute) {
            Some(Kind::Compute(held)) => (
                held.inputs
                    .iter()
                    .map(|held| held.data.my_lds_idx)
                    .collect::<Vec<_>>(),
                held.outputs
                    .iter()
                    .map(|held| held.data.my_lds_idx)
                    .collect::<Vec<_>>(),
            ),
            _ => (Vec::new(), Vec::new()),
        };
        // ⛔ AND THE SLOT THAT NAMED NO LDS IS UNTOUCHED: the retag walk asks `slot_lds == old` first
        // (`schedule/ddl/conversion.rs:454`), so a write is per slot and not per node.
        assert_eq!(inputs, vec![None, Some(new)]);
        assert_eq!(outputs, vec![None, Some(new)]);

        let (src, dsts, hops) = match tree.kind_of(transfer) {
            Some(Kind::Transfer(held)) => (
                held.src.data.my_lds_idx,
                held.dsts
                    .iter()
                    .map(|held| held.data.my_lds_idx)
                    .collect::<Vec<_>>(),
                (0..held.dsts.len())
                    .map(|at| held.dsts.hops(at).to_vec())
                    .collect::<Vec<_>>(),
            ),
            _ => (None, Vec::new(), Vec::new()),
        };
        assert_eq!(src, Some(new));
        assert_eq!(dsts, vec![None, Some(new)]);
        // ⛔ THE REBUILD KEEPS EVERY OTHER FIELD OF THE DESTINATION LIST — `dstVias_.at(i).via_` above
        // all, which it carries through `routes()`.
        assert_eq!(hops, vec![vec![SenComponent::L0], Vec::new()]);

        assert_eq!(tree.allocate(allocate).map(|(_, held)| held.lds), Some(new));
    }

    /// ⭐ `removeAllocUser` IS ONE REFERENCE, NOT THE USER — `--(it->second)` with the `erase` only at
    /// zero (`dsc/dsc2.h:1022-1034`), against [`Org::add_user`]'s raw `push_back({user, 1})`, whose
    /// multiplicity IS that count.
    #[test]
    fn removing_an_alloc_user_drops_one_reference() {
        let org = Org::default();
        let (a, b) = (NodeId(1), NodeId(2));
        org.set_node(
            SenComponent::Hbm,
            NodeId(0),
            seed_allocate_node(
                NodeName("a".to_owned()),
                LdsIdx(0),
                SenComponent::Hbm,
                &LayoutDims::new(PrimaryDim::Out, Vec::new()),
            ),
        );
        org.add_user(SenComponent::Hbm, a);
        org.add_user(SenComponent::Hbm, a);
        org.add_user(SenComponent::Hbm, b);
        assert_eq!(MemOrg::hbm_alloc_users(&org), Some(vec![a, a, b]));

        org.remove_alloc_user(SenComponent::Hbm, a);
        assert_eq!(MemOrg::hbm_alloc_users(&org), Some(vec![a, b]));
        org.remove_alloc_user(SenComponent::Hbm, a);
        assert_eq!(MemOrg::hbm_alloc_users(&org), Some(vec![b]));

        // ⛔ *"RemoveAllocUser: Schedule node <n> is not in the user list"* IS NO WRITE.
        org.remove_alloc_user(SenComponent::Hbm, a);
        assert_eq!(MemOrg::hbm_alloc_users(&org), Some(vec![b]));
        // ⛔ AND IT NEVER REACHES ANOTHER STORAGE'S LIST.
        org.remove_alloc_user(SenComponent::Lx, b);
        assert_eq!(MemOrg::hbm_alloc_users(&org), Some(vec![b]));
    }

    /// ⛔⛔ AN ORGANISATION NOBODY STATED `isZeroPadded` FOR ANSWERS [`None`], NOT `Some(false)` —
    /// the defect this slot used to ship, with `schedule/l3/dl_ops.rs:9301` and `:13597` branching on
    /// the fabricated `false`.
    #[test]
    fn an_unstated_zero_pad_is_not_a_stated_no_zero_pad() {
        let org = Org::default();
        assert_eq!(MemOrg::lx_zero_padded(&org), None);

        org.set_zero_padded(SenComponent::Lx, false);
        assert_eq!(MemOrg::lx_zero_padded(&org), Some(false));

        // `ZpType::ALLZEROPAD`/`ROWZEROPAD` both reach the readers as the one question they ask.
        org.set_zero_padded(SenComponent::Lx, true);
        assert_eq!(MemOrg::lx_zero_padded(&org), Some(true));

        // ⛔ AND HBM'S ENTRY IS NOT LX'S: `memOrg_` states one per component.
        let other = Org::default();
        other.set_zero_padded(SenComponent::Hbm, true);
        assert_eq!(MemOrg::lx_zero_padded(&other), None);
    }
}
