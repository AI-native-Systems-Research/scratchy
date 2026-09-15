// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐ THE ONE STATE BOTH CARRIERS NAME — [`DscState`] holds every DSC's schedule tree and every
//! labelled DS's `memOrg_`, and [`super::Reads`] (`&'a F`) and [`super::Env`] (`&'a mut E`) each hold
//! a SHARED reference to it.
//!
//! ⛔⛔ WHY IT IS SHAPED THIS WAY — REVIEW 382'S OWN NOTE. `L3RunInputs::reads` is `&'a F` and
//! `L3RunSurgery::env` is `&'a mut E`, so the reference's single `this` is severed by the port's
//! signature (`l3/dl_ops.rs:20556-20560`), and the two chains that cross the cut are the PLACED
//! ADDRESS (`:5687` writes, `:4971`/`:5034` rewrite, `:5816` reads back) and the MINTED TREE
//! (entries 290/353/368 write `env`; 289/295/332 read `reads`).
//!
//! ⭐ TWO CARRIERS, ONE STATE, THROUGH A CELL. That is not a workaround that changes behaviour: it
//! makes both views name the SAME tree and the SAME `memOrg_` node, which is what the reference does
//! and what any faithful caller must do. Giving the two carriers separate trees would compile and
//! silently drop every minted node — the exact defect review 382 warns about.
//!
//! ⭐ IT IS SOUND BECAUSE EVERY TREE TRAIT ANSWERS BY VALUE. No method of [`NodeParents`],
//! [`LoopStages`], [`ScheduleNodes`] or [`MemOrg`](crate::schedule::l3::dsc::MemOrg) hands a borrow
//! of the interior out, so no [`RefCell`] borrow outlives the call that took it and none can overlap
//! a write.

use std::cell::RefCell;
use std::collections::BTreeMap;

use sys_arch_spec::arch_enums::SenComponent;

use crate::schedule::ddc::fold::{NodeId, NodeKind};
use crate::schedule::ddc::metadata::DatastageId;
use crate::schedule::ddc::transformation::LoopId;
use crate::schedule::ddc::transformation_util::{LoopDims, LoopNode};
use crate::schedule::ddc::v1;
use crate::schedule::dsc2::{AllocateNode, LdsIdx, LoopCondComposite, NodeName};
use crate::schedule::l3::dl_ops::{LoopNesting, LoopStages};
use crate::schedule::l3::dsc::{DesignSpaceConfig, DscIdx, NodeParents, ScheduleNodes, SuperDsc};

use super::tree::{Kind, Org, TreeData, seed_allocate_node};

/// `scheduleTree_.getHead()->name_` on scratchy's own staged bundle — node `[0]` of every scheduled
/// program in `/Users/nickm/tmp/bridge1-fixtures/g0/debug/sdsc_*/sdsc.json` is a `block` of this
/// name with `prev_ = ""`.
const ROOT_BLOCK_NAME: &str = "root_level_operations";

/// ⭐ ONE DSC'S SCHEDULE TREE AND `memOrg_`s — the object [`crate::schedule::l3::dl_ops::DscTrees`]
/// hands out as its `Tree`, and the object every `memOrg_` read reaches through.
#[derive(Debug, Default)]
pub struct DscTree {
    tree: RefCell<TreeData>,
    /// `labeledDs_`'s organisations, POSITIONALLY beside
    /// [`crate::schedule::l3::dsc::LabeledDsList::indexed`] — AND THE ONLY PLACE AN ALLOCATE NODE'S
    /// ddc VIEW LIVES, see [`Self::placed`].
    orgs: Vec<Org>,
}

impl DscTree {
    /// The tree, for the length of ONE question.
    /// ⭐ `pub` WHILE [`Self::with_mut`] STAYS `pub(super)` — the read/write asymmetry
    /// [`super::tree`]'s header states. A consumer outside `stages` may LOOK at the grown tree; the
    /// stages alone may change it.
    pub fn with<T>(&self, ask: impl FnOnce(&TreeData) -> T) -> T {
        ask(&self.tree.borrow())
    }

    /// The tree, for the length of ONE write.
    pub(super) fn with_mut<T>(&self, write: impl FnOnce(&mut TreeData) -> T) -> T {
        write(&mut self.tree.borrow_mut())
    }

    /// That labelled DS's organisation.
    pub(super) fn org(&self, lds: LdsIdx) -> Option<&Org> {
        self.orgs.get(usize::try_from(lds.0).ok()?)
    }

    /// Every organisation, in `labeledDs_` order.
    pub(super) fn orgs(&self) -> &[Org] {
        &self.orgs
    }

    /// Which `memOrg_` entry an allocate node hangs from — how `addAllocUser` reaches the users list
    /// of a node it was handed by identity.
    pub(super) fn home_of(&self, node: NodeId) -> Option<(LdsIdx, SenComponent)> {
        self.orgs.iter().enumerate().find_map(|(at, org)| {
            let storage = org.storage_of(node)?;
            Some((LdsIdx(u32::try_from(at).ok()?), storage))
        })
    }

    /// The ddc view of the allocate node at that id.
    ///
    /// ⛔⛔ A PROJECTION OF ONE `memOrg_` ENTRY, NOT A MAP OF ITS OWN — AND THAT IS THE WHOLE FIX.
    /// The reference holds ONE `dsc2::AllocateNode *` per `labeledDs_.at(lds).memOrg_.at(storage)`
    /// (`L3DlOpsScheduler.cpp:1585`, `:5811`, and thirty more), and this tree used to keep a SECOND
    /// map of the same nodes keyed by [`NodeId`] beside [`Org`]'s own. Entry 353's mint wrote only
    /// this one and [`crate::schedule::l3::dl_ops::AllocationSites::allocation`] read only the
    /// other, so every freshly minted LX allocation was invisible to the stage that places it —
    /// stage 2a stopped there for all 24,363 programs. One cell cannot disagree with itself.
    pub(super) fn placed(&self, node: NodeId) -> Option<AllocateNode> {
        let (lds, storage) = self.home_of(node)?;
        self.org(lds)?.placed(storage)
    }

    /// The same, written back — into that one `memOrg_` entry.
    pub(super) fn set_placed(&self, node: NodeId, held: AllocateNode) {
        let Some((lds, storage)) = self.home_of(node) else {
            return;
        };
        if let Some(org) = self.org(lds) {
            org.set_placed(storage, held);
        }
    }

    /// Every placed node, for one read, gathered BY IDENTITY out of the organisations that hold
    /// them — the shape [`crate::schedule::stages::tree::TreeData::allocations`] walks.
    pub(super) fn with_placed<T>(
        &self,
        ask: impl FnOnce(&BTreeMap<NodeId, AllocateNode>) -> T,
    ) -> T {
        let placed: BTreeMap<NodeId, AllocateNode> =
            self.orgs.iter().flat_map(Org::placed_nodes).collect();
        ask(&placed)
    }

    /// HOW MANY NODES THIS DSC'S TREE HOLDS — what a caller measuring stage 2a's effect counts.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.with(TreeData::len)
    }

    /// Every node's name in DFS order — the dump beside *"Invalid scheduleTree."*.
    #[must_use]
    pub fn names(&self) -> Vec<NodeName> {
        self.with(TreeData::node_names)
    }

    /// ⭐⭐ THE PER-`nodeType_` CENSUS OF THIS TREE — [`Self::node_count`] broken out by kind, which
    /// is what a caller measuring stage 2a's effect on real data reports.
    ///
    /// ⭐ A VALUE AND NOT A RATIO, and the counts SUM to [`Self::node_count`]: the reference side
    /// states the same quantity per kind (`check.py census`: transfer +3,102, compute +2,834, block
    /// +2,158, loop +2,043, sync +1,783, allocate +1,319, condition +892 over 187 programs), so a
    /// per-kind reading here is comparable against it kind for kind rather than as one total.
    #[must_use]
    pub fn kinds(&self) -> BTreeMap<NodeKind, usize> {
        self.with(TreeData::node_kinds)
    }

    /// `scheduleTree_.getHead()->denId_`, which serialises as `scheduleTreeHeadDenId_`
    /// (`dsc/dsc2.cpp:368`) — [`None`] before entry 217 states it.
    #[must_use]
    pub fn head_den(&self) -> Option<DatastageId> {
        self.with(TreeData::head_den)
    }

    /// `node->nodeType_`, [`None`] for a node this tree does not hold.
    #[must_use]
    pub fn node_kind(&self, node: NodeId) -> Option<NodeKind> {
        self.with(|tree| tree.node_kind(node))
    }

    /// `condNode->loopCond_` — [`None`] where the node is not a loop-guarded condition.
    #[must_use]
    pub fn loop_cond(&self, node: NodeId) -> Option<LoopCondComposite> {
        self.with(|tree| tree.loop_cond(node))
    }

    /// `condNode->coreClCond_` — [`None`] where the node is not a core/corelet-guarded condition,
    /// which is what `hasCoreClCond()` distinguishes.
    #[must_use]
    pub fn core_cl_cond(&self, node: NodeId) -> Option<v1::CoreClSet> {
        self.with(|tree| tree.core_cl_cond(node))
    }

    /// The loop minted at that id.
    ///
    /// ⛔ TOTAL BY THE TRAIT'S OWN STATEMENT — *"`numId_` and `denId_` — TOTAL: every loop node
    /// carries both"*. Every [`LoopId`] a ported unit holds came out of THIS tree's own walk
    /// ([`LoopNesting::owner_loop`], `loops_under`), so an id that names no loop here is a defect in
    /// the walk and not an answer this carrier can give.
    fn minted(&self, at: LoopId) -> LoopNode {
        self.with(|tree| {
            tree.loop_node(at)
                .cloned()
                .expect("a LoopId this tree's own walk produced names a loop of this tree")
        })
    }
}

impl NodeParents for DscTree {
    fn parent(&self, node: NodeId) -> Option<NodeId> {
        self.with(|tree| tree.parent(node))
    }

    fn children(&self, parent: NodeId) -> Vec<NodeId> {
        self.with(|tree| tree.children(parent))
    }
}

impl LoopNesting for DscTree {
    fn owner_loop(&self, node: NodeId) -> Option<LoopId> {
        self.with(|tree| tree.owner_loop(node))
    }

    fn has_parent(&self, node: LoopId) -> bool {
        self.with(|tree| tree.parent(node.0).is_some())
    }
}

impl LoopStages for DscTree {
    fn loop_num(&self, loop_node: LoopId) -> DatastageId {
        self.minted(loop_node).num
    }

    fn loop_den(&self, loop_node: LoopId) -> DatastageId {
        self.minted(loop_node).den
    }

    fn loop_dims(&self, loop_node: LoopId) -> LoopDims {
        self.minted(loop_node).dims
    }
}

impl ScheduleNodes for DscTree {
    fn node_names(&self) -> Vec<NodeName> {
        self.names()
    }
}

/// ⭐ EVERY DSC'S TREE AND `memOrg_`s IN ONE SUPER-DSC — what [`super::Reads`] and [`super::Env`]
/// both hold a shared reference to.
#[derive(Debug, Default)]
pub struct DscState {
    dscs: Vec<DscTree>,
    /// ⭐ EVERY PROVIDER METHOD THAT REFUSED, IN THE ORDER IT WAS ASKED — the FIRST entry is the one
    /// fact that decides what happens next, and it is recorded rather than printed so a caller can
    /// report it.
    ///
    /// ⛔ AN OBSERVER AND NOT A BEHAVIOUR: nothing in the port reads it, and a refusal is recorded
    /// on the way to answering [`None`], which is the ported units' own idiom.
    refusals: RefCell<Vec<&'static str>>,
}

impl DscState {
    /// ⭐ THE SEED — the schedule tree an input super-DSC ALREADY HAS, before stage 2a grows it: one
    /// root block per DSC, and one HBM `ALLOCATE` node per HBM-pinned labelled DS with that DS's own
    /// `layoutDimOrder_` on it, filed in its `memOrg_`.
    ///
    /// ⛔⛔ THE ONE THING HERE THAT IS NOT IN [`SuperDsc`] IS THE SEED NODE'S **NAME**. The reference
    /// names it `allocate-<dsName_>_hbm` (`allocate-Tensor0_hbm` in
    /// `/Users/nickm/tmp/bridge1-fixtures/g0/debug/sdsc_0/sdsc.json`), and `dsName_` is carried by
    /// neither [`crate::schedule::l3::dsc::LabeledDs`] nor [`DesignSpaceConfig`] — it reaches this
    /// stage only through [`crate::schedule::ddc::v1::StorageNames`], which is on the `P` carrier and
    /// not on the super-DSC. So the seed is named after the POSITION instead, in the same shape the
    /// reference's own minted LX allocations use (`allocate_lds0_lx`). ⛔ THAT IS AN INTERNAL
    /// IDENTITY AND NOTHING ELSE: no address, extent or capacity is derived from it, and every name
    /// stage 2a itself mints is minted by the ported units.
    ///
    /// ⛔ AND `maxDimSizes_` IS LEFT UNBOUNDED, which is the reference's own `resize(n, -1)`
    /// (`dsc/dsc2.h:982`) — so the seed pages nothing, rather than paging a size nobody stated.
    #[must_use]
    pub fn seeded(sdsc: &SuperDsc) -> Self {
        Self {
            dscs: sdsc.dscs().iter().map(seed_dsc).collect(),
            refusals: RefCell::new(Vec::new()),
        }
    }

    /// That DSC's tree, [`None`] for a `dscs_` position this state holds none for.
    ///
    /// ⭐ `pub` BECAUSE THE LOWERING IS PER-DSC. [`Self::dscs`] hands out the whole slice and
    /// [`Self::kinds`] sums every tree; a lowering walks ONE DSC at a time and needs it by
    /// [`DscIdx`], which is the index the super-DSC already keys its `dscs_` by.
    #[must_use]
    pub fn dsc(&self, at: DscIdx) -> Option<&DscTree> {
        self.dscs.get(usize::try_from(at.0).ok()?)
    }

    /// ⭐ A PROVIDER METHOD'S OWN REFUSAL, RECORDED AND THEN PROPAGATED — `<Trait>::<method>` plus
    /// the fact it wants and where that fact lives.
    pub(super) fn refuse<T>(&self, what: &'static str) -> Option<T> {
        self.refusals.borrow_mut().push(what);
        None
    }

    /// Every refusal so far, in the order it was made.
    #[must_use]
    pub fn refusals(&self) -> Vec<&'static str> {
        self.refusals.borrow().clone()
    }

    /// The FIRST refusal — the one fact that decides what happens next.
    #[must_use]
    pub fn first_refusal(&self) -> Option<&'static str> {
        self.refusals.borrow().first().copied()
    }

    /// Every DSC's tree, in `dscs_` order.
    #[must_use]
    pub fn dscs(&self) -> &[DscTree] {
        &self.dscs
    }

    /// HOW MANY NODES EVERY DSC'S TREE HOLDS TOGETHER — the whole super-DSC's node count, which is
    /// the number stage 2a moves.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.dscs.iter().map(DscTree::node_count).sum()
    }

    /// ⭐⭐ THE WHOLE SUPER-DSC'S PER-`nodeType_` CENSUS — [`DscTree::kinds`] summed over every DSC,
    /// so the counts sum to [`Self::node_count`].
    #[must_use]
    pub fn kinds(&self) -> BTreeMap<NodeKind, usize> {
        let mut census: BTreeMap<NodeKind, usize> = BTreeMap::new();
        for tree in &self.dscs {
            for (kind, count) in tree.kinds() {
                *census.entry(kind).or_insert(0) += count;
            }
        }
        census
    }
}

/// One DSC's seed — the root block and its HBM allocations.
fn seed_dsc(dsc: &DesignSpaceConfig) -> DscTree {
    let tree = TreeData::default();
    let mut tree = tree;
    let root = tree.add(NodeName(ROOT_BLOCK_NAME.to_owned()), Kind::Block, None);
    tree.set_head(root);
    let mut orgs = Vec::new();
    for (at, lds) in dsc.labeled_ds.indexed() {
        let seeded = lds
            .pinning()
            .hbm()
            .then(|| dsc.layout_dims.get(&at))
            .flatten()
            .map(|layout| {
                let name = NodeName(format!("allocate_lds{}_hbm", at.0));
                let held = seed_allocate_node(name.clone(), at, SenComponent::Hbm, layout);
                let alloc = tree.fresh_alloc();
                let node = tree.add(name, Kind::Allocate(alloc, held.clone()), Some(root));
                (node, held)
            });
        orgs.push(match seeded {
            Some((node, held)) => Org::seeded(SenComponent::Hbm, node, held),
            None => Org::default(),
        });
    }
    DscTree {
        tree: RefCell::new(tree),
        orgs,
    }
}
