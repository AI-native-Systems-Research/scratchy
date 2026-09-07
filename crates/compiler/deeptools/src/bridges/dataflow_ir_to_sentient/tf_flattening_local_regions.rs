// SPDX-License-Identifier: Apache-2.0
//
// ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ CRUSTIFY BRIDGE-2 CAMPAIGN — READ THIS BEFORE YOU FILL AN ANCHOR IN THIS FILE.               ║
// ║ Full brief: crustify-bridge2/AGENT-BRIEF.md   ·   campaign statement: crustify-bridge2/TASK.md║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
//
// 1. THE AUTHORITY IS THE C++ TREE, NOT THE EXTRACT.
//       /Users/nickm/git/deeptools-src/<file>:<line>        (deeptools @ a0d29abbed — repo_info.txt)
//    That is the revision every citation below resolves against. `crustify-bridge2/source/bridge2.cpp`
//    says WHICH functions are in scope and IN WHAT ORDER; ⛔ its bodies are TRUNCATED AT THE TAIL —
//    366 of the 384 end in a blank line and bare closing braces, and a 48-entry sample against the
//    authority found 21 that had lost real trailing statements (a `return success();`, a
//    `return rhs;`, an entire `} else { … }` branch, an `initMASData(...)` call). Port from the
//    authority file at the cited line. ⛔ /Users/nickm/git/deeptools is a DIFFERENT revision.
//    ⛔ The pod (/project_src/deeptools) is NOT reachable from this host — use the mirror above.
//
// 2. PORTED MEANS THE WHOLE FUNCTION INCLUDING ITS EMISSION. The op a function emits IS the
//    function — its exact attribute names and values, branch order and early returns. A documented
//    predicate that emits nothing is NOT a port (that is how the previous attempt failed). What you
//    MAY drop is only the mechanism for REACHING operands: use-walks, memoising by
//    (core, corelet, component), positioning an OpBuilder. If the target IR cannot express a
//    function's input, ADD THE OP to `src/islands/{sentient,dataflow_ir}/` — never decide the
//    function is unnecessary.
//
// 3. THIS IS A PURE-LOGIC PORT WITH NO C ANYWHERE. Whatever the generic C-to-Rust conventions say:
//    ❌ no bindgen/allowlist/-sys, ❌ no `ffi::`/`mod ffi_export`/`#[unsafe(no_mangle)] extern "C"`,
//    ❌ no `CRUSTIFY_<FILE>` switch, ❌ no `Foo`/`FooRef`/`FooMut` layout triple, ❌ no `unsafe`,
//    ❌ no sanitizers and no C-vs-Rust equivalence harness (there is no C to call).
//
// 4. CRATE RULES BIND YOU — `crates/compiler/deeptools/CLAUDE.md`, read it in full.
//    🛑 NEVER RUNTIME REFUSE: no `Result`, no `Err(`, no `.ok_or`, no `assert!`, no `debug_assert!`
//    (frozen at zero by crates/targets/spyre/tests/dfir_never_runtime_refuses.rs). A closed set is
//    an `enum`; an invariant is a TYPE. `todo!("<op> …")` is tolerated, capped and ratcheted down —
//    and ⛔ never substitute a stand-in op to dodge one. Newtypes, never raw scalars. No strings for
//    closed sets. `Arch`/`Model`/`Workload` flow through as const generics.
//
// 5. ANCHORS: each `// crustify:todo: e<NNN>_<name>` below is one scheduled unit. Replace it with
//    the ported function carrying the doc anchor `/// Replaces: e<NNN>_<name>` on the item itself.
//    A surviving TODO is open work; the TODO must not survive beside the filled anchor.
//
// 6. TESTS: `#[cfg(test)] mod unit_tests` beside the code. 668 of the authority tree's 825
//    `dcc/test/**/*.mlir` cases carry `CHECK-SENT-IR` expectations — port the EXPECTATION, build the
//    typed input in Rust (this crate has no MLIR parser and must not get one).
//    `crates/compiler/deeptools/tests/sentient_corpus/` is the answer key (our DataflowIR beside the
//    reference's SentientIR for the same program).
//
// 7. GATE: `cargo check -p deeptools` and `cargo test -p deeptools`. ⛔ NEVER run the workspace or
//    acceptance build in an agent worktree — ~6 GB of target/ each and <50 GB free on this host.
//
// 8. `dataflow_ir_to_sentient/agen_to_sentient.rs` is the EARLIER PARTIAL ATTEMPT (predicates, no
//    emission, called by nothing). Nothing in it counts as ported; reuse what is right, but every
//    unit gets its own anchored item here.

//! `FlatteningLocalRegions.cpp` — 16 of bridge 2's 384 functions (dependency level(s) [0, 1, 2, 3, 4]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e101_OperationNode` | 101/384 | 0 | `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:51` |
//! | `e102_getParentNode` | 102/384 | 2 | `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:53` |
//! | `e103_getFirstChild` | 103/384 | 2 | `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:56` |
//! | `e104_getNextSibling` | 104/384 | 2 | `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:59` |
//! | `e105_getPrevSibling` | 105/384 | 2 | `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:62` |
//! | `e106_OperationTreeBase` | 106/384 | 0 | `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:78` |
//! | `e107_getRoot` | 107/384 | 2 | `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:81` |
//! | `e108_partitionUnits` | 108/384 | 17 | `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:168` |
//! | `e179_clear` | 179/384 | 15 | `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:112` |
//! | `e180_traverseRegion` | 180/384 | 17 | `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:129` |
//! | `e181_inRegionEmpty` | 181/384 | 6 | `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:214` |
//! | `e182_cloneOpsForRegions` | 182/384 | 60 | `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:223` |
//! | `e246_FlatteningLocalRegionsTree` | 246/384 | 0 | `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:79` |
//! | `e247_compute` | 247/384 | 15 | `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:151` |
//! | `e287_flatten` | 287/384 | 70 | `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:384` |
//! | `e305_runOnOperation` | 305/384 | 17 | `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:459` |

use super::vc_loop_mask_tree::{OperationNodeId, OperationTreeBase};
use crate::islands::dataflow_ir::dialects::{Op as DfirOp, Val};

/// THE IDENTITY OF ONE NODE IN THE FLATTENING TREE — what a `LocalOpNode *` is in the C++.
///
/// # ⭐ AN INDEX WHERE THE REFERENCE HAS A `LocalOpNode *`
///
/// The reference's tree is a heap of individually `new`ed nodes wired by three raw pointers, freed by
/// a post-order walk in `clear()` (`FlatteningLocalRegions.cpp:112-127`). A node cannot hold a
/// `&LocalOpNode` to its sibling and be built by the same walk that appends to the storage, so the
/// links are identities into storage the tree owns — which is also what makes the reads below total
/// instead of unsafe.
///
/// # ⛔⛔ AND THAT STORAGE IS THE `mlir::OperationTreeBase` THAT IS ALREADY PORTED
///
/// `FlatteningLocalRegions.cpp:15` includes `Analysis/OperationTree.hpp` and `:47` declares
/// `class LocalOpNode : public OperationNode` — the SAME base the Loop Mask Tree derives from
/// (`LoopMaskTree.hpp:28`). The links, the sibling walks and `insertChildNode` are therefore not this
/// family's code at all: they are [`OperationTreeBase`], written with entries 081-088 in
/// [`super::vc_loop_mask_tree`] and given a payload parameter for exactly this moment — *"that batch
/// instantiates `OperationTreeBase<LocalOpNode>` from here instead of writing a second arena"*
/// (`vc_loop_mask_tree.rs:90-96`). Entry 106 is where that happens. A second arena would mean two
/// `prev_sibling` walks and, at entry 180, two `insertChildNode`s for one relation.
///
/// ⚠️ THE LAYER STILL WANTS HOISTING into a module of its own, as its own banner says; it is imported
/// rather than moved because moving it would rewrite a file whose remaining entries (076-078, 171,
/// 236-238, 281) belong to other batches landing in parallel.
///
/// ⭐ MINTING ONE **IS** THE `static_cast`, exactly as it is for
/// [`super::vc_loop_mask_tree::LoopMaskNodeId`]: `static_cast<LocalOpNode *>` is an unchecked
/// downcast, sound only because every node in this tree was `new`ed as a `LocalOpNode`
/// (`FlatteningLocalRegions.cpp:134`, `:392`) — and here that fact is the arena's payload type, so a
/// cast that could fail is unwritable rather than unchecked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocalOpNodeId(OperationNodeId);

/// WHICH REGION OF ITS PARENT AN OPERATION SITS IN — `is_in_region_num`.
///
/// ⭐ AN INDEX, NOT A COUNT AND NOT A FLAG. `traverseRegion` recurses with the loop counter of
/// `op.getNumRegions()` (`FlatteningLocalRegions.cpp:145-147`), so a node carries the index of the
/// PARENT'S region it was found in. ⚠️ The reference's field is an `int` and `compute` passes
/// `false` for the uniformized op's own regions (`:164`), which is 0; every other site passes a
/// non-negative counter, so `u32` is total over the values that can reach it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RegionNum(pub u32);

/// ONE OPERATION IN THE TREE `FlatteningLocalRegions` BUILDS OVER A `uniform.uniformize_regions`.
///
/// ⭐ WHAT `LocalOpNode` **ADDS** TO `OperationNode`, WHICH IS ALL IT IS. The base class holds the
/// operation and the three links (`dcc/src/Analysis/OperationTree.hpp:185-188`) and is
/// [`OperationTreeBase`]'s business; the derived class adds `units` and `is_in_region_num`
/// (`FlatteningLocalRegions.cpp:70-71`) — so this type is the arena's payload, `N`, and the
/// operation rides along with it because a payload is what the base is generic over.
///
/// # ⛔⛔ NEITHER `Clone` NOR `PartialEq`, AND BOTH ARE THE REFERENCE'S OWN DECISIONS
///
/// `OperationNode` deletes its copy constructors (`dcc/src/Analysis/OperationTree.hpp:30-31`): a
/// second node wrapping the same operation with the same links would appear twice in one sibling
/// chain and be deleted twice by `clear()`. And its `operator==` is `this == &n` (`:34`) — POINTER
/// identity, not structural equality, which is also how `cloneOpsForRegions` recognises a node's
/// operation (`std::find` over an `Operation *` list, `:233-234`). A derived `Clone` would reintroduce
/// the copy the reference forbids and a derived `PartialEq` would answer a different question from
/// the one the reference asks — the identity question is `==` on two [`LocalOpNodeId`]s — so this
/// type derives only `Debug`.
///
/// ⚠️ `op` IS ANY `DfirOp` BECAUSE ENTRIES 101-108 NEVER LOOK AT IT. The ops this tree actually holds
/// are `uniform.uniformize_regions` and `uniform.yield` and their contents, and the `uniform` dialect
/// is not in `islands/dataflow_ir/` yet — the first unit that asks an op WHICH op it is
/// (`isa<uniform::UniformizeRegionsOp>` in entries 180, 247, 182 and 287) is the one that has to add
/// it, per the campaign brief. A constructor, four link reads and a partition are opaque in the
/// operation: entry 108 compares op IDENTITY and never op contents.
#[derive(Debug)]
pub struct LocalOpNode<'p> {
    /// `operation_op_` — the operation this node stands for
    /// (`dcc/src/Analysis/OperationTree.hpp:29`), borrowed from the program being flattened.
    ///
    /// ⚠️ The reference reads it back through `OperationNode::getOperation()` (`:37`), a base-class
    /// accessor outside the 384; the member is exposed directly rather than porting an unscheduled
    /// unit to wrap it.
    pub op: &'p DfirOp,
    /// `std::vector<mlir::Value> units` — the units whose copy of the enclosing local region
    /// contains this operation (`FlatteningLocalRegions.cpp:70`).
    ///
    /// ⭐ EMPTY AT CONSTRUCTION AND FILLED BY THE WALK: `traverseRegion` pushes the unit list it was
    /// given, one entry per unit, for every op that is not itself a `uniform.uniformize_regions`
    /// (`:136-139`).
    pub units: Vec<Val>,
    /// `int is_in_region_num = 0` — see [`RegionNum`].
    ///
    /// ⚠️ ITS ONLY READER IN THE REFERENCE IS NEVER CALLED: `inRegionEmpty` (`:214-221`, entry 181)
    /// walks a sibling chain looking for this value, and the three mentions of it in the whole file
    /// are its declaration, its definition and nothing else.
    pub is_in_region_num: RegionNum,
}

impl<'p> LocalOpNode<'p> {
    /// Replaces: e101_OperationNode
    ///
    /// **101/384** `LocalOpNode::LocalOpNode` —
    /// `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:51` (0L).
    ///
    /// ```cpp
    /// class LocalOpNode : public OperationNode {
    ///   friend class FlatteningLocalRegionsTree;
    ///
    ///  public:
    ///   LocalOpNode(Operation *op) : OperationNode(op) {};
    ///   // ...
    ///   std::vector<mlir::Value> units;
    ///   int is_in_region_num = 0;
    /// };
    /// ```
    ///
    /// # ⭐ A ZERO-LINE CONSTRUCTOR STILL DECIDES FOUR THINGS
    ///
    /// It forwards the operation to `OperationNode(op) : operation_op_(op)`
    /// (`dcc/src/Analysis/OperationTree.hpp:29`) and then runs the member initialisers of both
    /// classes. What comes out is a node that is **detached** — `parent_operation_`, `first_child_`
    /// and `next_sibling_` are all null until `insertChildNode` wires it in — with **no units** and
    /// marked as being in **region 0**. `traverseRegion` overwrites the region number on the very
    /// next line (`FlatteningLocalRegions.cpp:134-135`), so the `= 0` is what a node built anywhere
    /// else keeps.
    ///
    /// ⭐ THE DETACHMENT IS THE BASE'S HALF AND IS NOT REPEATED HERE: the three `= nullptr`
    /// initialisers are `Links::UNLINKED`, applied by every path that puts a payload into the arena
    /// ([`OperationTreeBase::with_root`], [`OperationTreeBase::push_child`]). What this constructor
    /// decides is the two members the derived class adds, and it is a payload rather than a node
    /// because a detached node is not a thing the arena can hold.
    ///
    /// ⛔ AND `friend class FlatteningLocalRegionsTree` IS THE ONE THING WITH NO ANALOGUE: it lets the
    /// tree reach `is_in_region_num` and the base's links directly. Here the links are the base's own
    /// private state and the two members the friendship was for are public in the reference too.
    #[must_use]
    pub const fn new(op: &'p DfirOp) -> Self {
        Self {
            op,
            units: Vec::new(),
            is_in_region_num: RegionNum(0),
        }
    }
}

/// EVERY OPERATION A UNIT RUNS, PER UNIT, IN THE ORDER THE WALK FIRST SAW THE UNIT.
///
/// ```cpp
/// llvm::MapVector<mlir::Value, std::vector<mlir::Operation *>> unit_to_ops;
/// ```
/// (`FlatteningLocalRegions.cpp:76`.)
///
/// # ⛔⛔ A `MapVector`, AND ITS KEY ORDER IS THE PASS'S OUTPUT ORDER
///
/// `llvm::MapVector` keeps keys in first-insertion order, and this pass's own vendor case proves the
/// order is observable. `flatten_local_region.mlir`'s input declares its units `%0, %1, %2, %3` and
/// groups them two per region — `(%arg1 -> %0, %2)` at `:91` and `(%arg1 -> %1, %3)` at `:120` — and
/// the flattened output's four regions come out **`%0, %2, %1, %3`** (`CHECK-SENT-IR` `:19`, `:31`,
/// `:43`, `:55`, against the unit definitions at `:8-11`). That is the order the walk first reached
/// each unit, since `flatten` builds its region list by iterating the equivalence classes
/// (`:411-415`, `:430`). A hash map would have lost that order and a sorted map would have replaced
/// it, so the container is part of the semantics rather than a choice of container.
#[derive(Debug, Default)]
pub struct UnitToOps<'p>(Vec<(Val, Vec<&'p DfirOp>)>);

impl<'p> UnitToOps<'p> {
    /// AN EMPTY MAP — what the tree's constructor leaves behind (entry 106) and what `clear()`
    /// restores (`FlatteningLocalRegions.cpp:126`).
    #[must_use]
    pub const fn new() -> Self {
        Self(Vec::new())
    }

    /// `unit_to_ops[u].push_back(&op)` — what `traverseRegion` does once per unit for every operation
    /// it walks (`FlatteningLocalRegions.cpp:142`).
    ///
    /// ⭐ THE SUBSCRIPT IS AN INSERT: `MapVector::operator[]` default-constructs an empty vector for a
    /// key it has not seen and appends the key to its order, which is why a unit's position here is
    /// decided by the first operation attributed to it and never changes afterwards.
    pub fn push_op(&mut self, unit: Val, op: &'p DfirOp) {
        match self.0.iter_mut().find(|(u, _)| *u == unit) {
            Some((_, ops)) => ops.push(op),
            None => self.0.push((unit, vec![op])),
        }
    }

    /// THE ENTRIES IN KEY ORDER — iterating the `MapVector`, as `partitionUnits` does twice
    /// (`FlatteningLocalRegions.cpp:173`, `:177`).
    #[must_use]
    pub fn entries(&self) -> &[(Val, Vec<&'p DfirOp>)] {
        &self.0
    }
}

/// THE TREE `FlatteningLocalRegions` BUILDS OVER ONE `uniform.uniformize_regions`.
///
/// ```cpp
/// class FlatteningLocalRegionsTree : public OperationTreeBase {
///  public:
///   llvm::MapVector<mlir::Value, std::vector<mlir::Operation *>> unit_to_ops;
///   // ...
/// };
/// ```
/// (`FlatteningLocalRegions.cpp:74-99`.)
///
/// # ⛔⛔ THE BASE IS AN `Option`, AND THAT IS THIS FAMILY'S DIFFERENCE FROM THE LOOP MASK TREE
///
/// `OperationTreeBase` default-constructs with `root_ == nullptr` (`OperationTree.hpp:232`) and this
/// family really does observe that state: `flatten` starts by calling `clear()`, which sets
/// `root_ = nullptr` (`:124`), and then returns early without installing a root when the operation is
/// not a `uniform.uniformize_regions` (`:385-387`). The Loop Mask Tree never can — `computeLoops`
/// installs its root as its first act — which is why [`OperationTreeBase::with_root`] takes the root
/// payload and has no rootless state at all. Wrapping it is how a rootless tree stays expressible
/// here without reintroducing a null check inside the arena: `None` **is** `root_ == nullptr`, and it
/// is the first conjunct of `getRoot`'s `DT_CHECK` answered by the type.
///
/// ⭐ AND THE NODE STORAGE COMES WITH IT: the reference's per-node `new`/`delete`
/// (`:134`, `:121-123`) is one `Vec` inside the arena, so `~FlatteningLocalRegionsTree() { clear(); }`
/// (entry 246) has nothing to free — dropping the tree drops the nodes — and `clear()` (entry 179)
/// becomes setting this field back to `None`.
///
/// ⛔ NEITHER `Clone` NOR `Copy`: `OperationTreeBase` deletes both copy constructors
/// (`OperationTree.hpp:200-201`), because two trees sharing one node heap would free it twice.
#[derive(Debug)]
pub struct FlatteningLocalRegionsTree<'p> {
    /// `unit_to_ops` — public in the reference too (`FlatteningLocalRegions.cpp:76`), written by
    /// `traverseRegion` (`:142`, entry 180) and read by `partitionUnits` (entry 108) and by `flatten`
    /// when it fills each new region (`:448`).
    pub unit_to_ops: UnitToOps<'p>,
    /// The `OperationTreeBase` this class derives from, absent until a root is installed.
    base: Option<OperationTreeBase<LocalOpNode<'p>>>,
}

impl<'p> FlatteningLocalRegionsTree<'p> {
    /// Replaces: e106_OperationTreeBase
    ///
    /// **106/384** `FlatteningLocalRegionsTree::FlatteningLocalRegionsTree` —
    /// `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:78` (0L).
    ///
    /// ```cpp
    /// FlatteningLocalRegionsTree() : OperationTreeBase() {}
    /// ```
    ///
    /// # ⭐ THE LEDGER NAMES THIS ENTRY AFTER ITS MEM-INITIALISER
    ///
    /// The unit is `e106_OperationTreeBase` because the extract took the name from
    /// `: OperationTreeBase()`, which is the whole body: the derived constructor adds nothing, so what
    /// it does is run the base's default constructor and the member initialisers. That leaves
    /// `root_ == nullptr` (`OperationTree.hpp:232`) and an empty `unit_to_ops` — and those two facts
    /// are the entire semantics of this unit.
    ///
    /// ⛔⛔ IT IS ALSO WHERE THE FAMILY'S STORAGE IS DECIDED, which is why entries 101-105 could not
    /// decide it: `LocalOpNode`'s links have nowhere to live until the tree that owns them exists.
    /// The answer is [`OperationTreeBase`]`<`[`LocalOpNode`]`>` — the base class the C++ names right
    /// here — instantiated rather than reimplemented, so `getPrevSibling`'s walk (entry 105) and
    /// `insertChildNode` (entry 180) exist once for both derived families.
    ///
    /// ⚠️ THE DESTRUCTOR IS ENTRY 246 AND IS NOT THIS: `~FlatteningLocalRegionsTree() { clear(); }`
    /// (`:79`) is the RAII half, and `clear()` (entry 179) is what frees the nodes. Here dropping the
    /// tree drops the arena, so 246's body has nothing left to do — but the entry is not mine to fill
    /// and the `Drop`-freeness is deliberately not asserted here.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            unit_to_ops: UnitToOps::new(),
            base: None,
        }
    }

    /// Replaces: e107_getRoot
    ///
    /// **107/384** `FlatteningLocalRegionsTree::getRoot` —
    /// `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:81` (2L).
    ///
    /// ```cpp
    /// const LocalOpNode *getRoot() const {
    ///   return static_cast<const LocalOpNode *>(OperationTreeBase::getRoot());
    /// }
    /// ```
    ///
    /// # ⛔⛔ `nullptr` IS REACHABLE HERE, SO ONE CONJUNCT OF THE `DT_CHECK` SURVIVES AS AN `Option`
    ///
    /// The base's `getRoot` re-asks its invariant on every call:
    ///
    /// ```cpp
    /// DT_CHECK(root_ && root_->getNextSibling() == nullptr &&
    ///          root_->getParentNode() == nullptr && "invalid root");
    /// ```
    ///
    /// (`dcc/src/Analysis/OperationTree.hpp:205-206`.) The second and third conjuncts hold **by
    /// construction** — see [`OperationTreeBase::root`]: the only writer of links is `push_child`,
    /// which never touches the fields of the node it was handed as a parent, and the root is never
    /// anyone's child. The FIRST conjunct is a real state in this family, because `flatten` clears the
    /// tree before it decides whether the operation is a `uniform.uniformize_regions` at all
    /// (`:385-387`) — so `None` is `root_ == nullptr`, and a caller cannot dereference it by accident
    /// the way the C++ can when the `DT_CHECK` is compiled out.
    ///
    /// ⭐ ONE METHOD FOR THE TWO OVERLOADS. `:84-86` is the same body without the `const`s, and the
    /// base's pair is the same (`OperationTree.hpp:204-212`, the mutable one a `const_cast` of the
    /// other). A [`LocalOpNodeId`] is not a borrow, so mutability is the caller's business —
    /// `clear()` mutates through `const_cast<LocalOpNode *>(getRoot())` (`:116`) while `flatten` only
    /// reads it (`:447`).
    ///
    /// ⭐ ITS CALLERS ARE ALREADY VISIBLE: `clear()` starts its post-order delete from the root
    /// (`:116`, entry 179) and `flatten` hands `getRoot()->getFirstChild()` to `cloneOpsForRegions`
    /// (`:447`, entry 287) — the root itself stands for the `uniformize_regions` op being replaced,
    /// so what gets cloned is its children.
    #[must_use]
    pub fn root(&self) -> Option<LocalOpNodeId> {
        Some(LocalOpNodeId(self.base.as_ref()?.root()))
    }

    /// Replaces: e102_getParentNode
    ///
    /// **102/384** `LocalOpNode::getParentNode` —
    /// `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:53` (2L).
    ///
    /// ```cpp
    /// LocalOpNode *getParentNode() const {
    ///   return static_cast<LocalOpNode *>(OperationNode::getParentNode());
    /// }
    /// ```
    ///
    /// # ⛔⛔ THE WHOLE FUNCTION IS A DOWNCAST, AND THE DOWNCAST IS AN UNCHECKED ASSERTION
    ///
    /// The base returns an `OperationNode *` (`dcc/src/Analysis/OperationTree.hpp:50`); this narrows
    /// it with `static_cast`, which does no check — if the tree ever held a node of another derived
    /// class, every read through the returned pointer would be undefined. It is sound only because
    /// `FlatteningLocalRegionsTree` allocates nothing but `LocalOpNode`s
    /// (`FlatteningLocalRegions.cpp:134` and `:392`), and that is an invariant of a different file
    /// than the cast.
    ///
    /// ⭐ SO THE PORT OF THESE FOUR READS IS THAT THE CAST HAS NOTHING LEFT TO DO. One arena, one
    /// payload type, one [`LocalOpNodeId`]: the narrow type is what the id already means, and the
    /// assertion is discharged at compile time. ⚠️ THE READ IS ON THE TREE AND NOT ON THE NODE
    /// because that is where the links are — a `LocalOpNode` is the payload the derived class adds,
    /// and `getParentNode` is the base's field read (see [`LocalOpNodeId`]).
    ///
    /// ⛔ AND `nullptr` IS `None`, NOT A ROOT. `OperationTreeBase` gives the forest a synthetic root
    /// whose parent is null, so an absent parent is what identifies it — `getDepth`/`isOutermost`
    /// count on exactly that (`OperationTree.hpp:62-69`).
    #[must_use]
    pub fn parent_node(&self, node: LocalOpNodeId) -> Option<LocalOpNodeId> {
        Some(LocalOpNodeId(self.base.as_ref()?.parent_node(node.0)?))
    }

    /// Replaces: e103_getFirstChild
    ///
    /// **103/384** `LocalOpNode::getFirstChild` —
    /// `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:56` (2L).
    ///
    /// ```cpp
    /// LocalOpNode *getFirstChild() const {
    ///   return static_cast<LocalOpNode *>(OperationNode::getFirstChild());
    /// }
    /// ```
    ///
    /// The first child **in syntactic order** (`dcc/src/Analysis/OperationTree.hpp:53-54`) — the
    /// operations of a region keep the order they appear in, which is what makes the sibling chain a
    /// program rather than a set. See [`Self::parent_node`] for what the `static_cast` becomes.
    ///
    /// ⛔ `None` IS A LEAF, and the reference says so itself:
    /// `bool isLeaf() const { return getFirstChild() == nullptr; }` (`:70`). Every walk over this
    /// tree — `cloneOpsForRegions`' recursion into a nested `uniformize_regions` (entry 182,
    /// `:235-238`), the post-order delete in `clear()` (entry 179) — stops on it.
    #[must_use]
    pub fn first_child(&self, node: LocalOpNodeId) -> Option<LocalOpNodeId> {
        Some(LocalOpNodeId(self.base.as_ref()?.first_child(node.0)?))
    }

    /// Replaces: e104_getNextSibling
    ///
    /// **104/384** `LocalOpNode::getNextSibling` —
    /// `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:59` (2L).
    ///
    /// ```cpp
    /// LocalOpNode *getNextSibling() const {
    ///   return static_cast<LocalOpNode *>(OperationNode::getNextSibling());
    /// }
    /// ```
    ///
    /// The next operation in the same region (`dcc/src/Analysis/OperationTree.hpp:57-58`). ⭐ THIS IS
    /// THE ONE THE TRANSFORM ACTUALLY LOOPS ON: `while (node) { …; node = node->getNextSibling(); }` is
    /// how `cloneOpsForRegions` (`FlatteningLocalRegions.cpp:228-...`) and `inRegionEmpty` (`:216-219`)
    /// traverse a region, and `getPrevSibling` and `getLastChild` are both derived from it by walking
    /// forward from the parent's first child rather than being stored.
    ///
    /// ⛔ `None` IS THE END OF THE REGION, which is why those loops need no count.
    #[must_use]
    pub fn next_sibling(&self, node: LocalOpNodeId) -> Option<LocalOpNodeId> {
        Some(LocalOpNodeId(self.base.as_ref()?.next_sibling(node.0)?))
    }

    /// Replaces: e105_getPrevSibling
    ///
    /// **105/384** `LocalOpNode::getPrevSibling` —
    /// `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:62` (2L).
    ///
    /// ```cpp
    /// LocalOpNode *getPrevSibling() const {
    ///   return static_cast<LocalOpNode *>(OperationNode::getPrevSibling());
    /// }
    /// ```
    ///
    /// # ⛔⛔ THE ONLY ONE OF THE FOUR THAT IS NOT A FIELD READ
    ///
    /// There is no `prev_sibling_` field: the base WALKS the parent's chain to find it
    /// (`dcc/src/Analysis/OperationTree.cpp:37-46`), which makes this O(children) and needs the node
    /// to have a parent. Caching it would be a second source of truth for one relation, which is the
    /// same reason it is a walk in the C++ and a walk in [`OperationTreeBase::prev_sibling`] — where
    /// the `DT_CHECK_MSG(getParentNode(), "expected a parent")` becomes an answer rather than a
    /// refusal, because for the one node that can reach it (the synthetic root) `None` is the true
    /// answer and is already what a first child returns (`:40`).
    ///
    /// ⭐ THE WALK IS ENTRY 084'S CODE AND IS DELIBERATELY NOT WRITTEN TWICE. `LoopMaskNode` declares
    /// the identical one-line delegation (`LoopMaskTree.hpp:45`), and the thing both delegate to is
    /// `mlir::OperationNode::getPrevSibling` — one function in `dcc/src/Analysis/`, so one function
    /// here.
    ///
    /// ⚠️ AND ITS READER IN THIS FILE IS `unlink` — `OperationTree.hpp:135-143`, which needs the
    /// previous sibling to close the chain around a node it detaches. That is what `clear(start)` and
    /// `remove` are built on (`hpp:216`, entry 179's neighbourhood), not one of the 384 itself.
    #[must_use]
    pub fn prev_sibling(&self, node: LocalOpNodeId) -> Option<LocalOpNodeId> {
        Some(LocalOpNodeId(self.base.as_ref()?.prev_sibling(node.0)?))
    }

    /// ONE NODE'S PAYLOAD — the derived state the C++ reaches through the `static_cast`.
    ///
    /// ⚠️ NOT ONE OF THE 384: `OperationNode::getOperation` and the `units`/`is_in_region_num` members
    /// are on the exclusion list as field accessors, and this is the read they become. `None` is a
    /// tree with no root, which has no nodes for an id to name.
    #[must_use]
    pub fn node(&self, node: LocalOpNodeId) -> Option<&LocalOpNode<'p>> {
        Some(self.base.as_ref()?.payload(node.0))
    }

    /// Replaces: e108_partitionUnits
    ///
    /// **108/384** `FlatteningLocalRegionsTree::partitionUnits` —
    /// `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:168` (17L).
    ///
    /// ```cpp
    /// void FlatteningLocalRegionsTree::partitionUnits(
    ///     llvm::MapVector<mlir::Value, std::vector<mlir::Value>> &equivalence_classes) {
    ///   // if the target operations the same, puts the units into the same bucket.
    ///   std::vector<mlir::Value> visited;
    ///   for (auto unit_to_op0 : unit_to_ops) {
    ///     if (std::find(visited.begin(), visited.end(), unit_to_op0.first) != visited.end())
    ///       continue;
    ///     for (auto unit_to_op1 : unit_to_ops) {
    ///       if (std::find(visited.begin(), visited.end(), unit_to_op1.first) != visited.end())
    ///         continue;
    ///       if (unit_to_op0.second == unit_to_op1.second) {
    ///         equivalence_classes[unit_to_op0.first].push_back(unit_to_op1.first);
    ///         visited.push_back(unit_to_op1.first);
    ///       }
    ///     }
    ///   }
    /// }
    /// ```
    ///
    /// # ⭐⭐ WHICH UNITS CAN SHARE ONE LOCAL REGION — THE WHOLE POINT OF THE PASS
    ///
    /// `flatten` turns the answer straight into the new op: one region per class, the class's units as
    /// that region's unit list, and the class sizes as the `ArrayAttr` of list sizes
    /// (`:411-415`, consumed at `:420-423`). If the count comes back equal to the number of regions the op already has, the
    /// pass declines the rewrite (`:418`).
    ///
    /// # ⛔⛔ EVERY CLASS CONTAINS ITS OWN REPRESENTATIVE, BY WAY OF THE INNER LOOP
    ///
    /// The outer loop does NOT mark `unit_to_op0.first` visited itself; the inner loop reaches the
    /// same entry, compares it with itself, and pushes it. So a class is never empty and the
    /// representative is always its first member — which is what makes `unit_rep_order.at(i)` the key
    /// `flatten` looks the region's operation list up under (`:448`).
    ///
    /// # ⛔⛔ AND THE COMPARISON IS POINTER IDENTITY, NOT STRUCTURAL EQUALITY
    ///
    /// `unit_to_op0.second == unit_to_op1.second` compares two `std::vector<mlir::Operation *>`:
    /// elementwise, in order, by ADDRESS. Two units are equivalent exactly when the walk attributed
    /// **the same operation objects** to both — see [`runs_the_same_operations`]. Structurally
    /// identical bodies in two different regions are two sets of operations and stay in two classes,
    /// which is precisely what the vendor's `@diff_groups` case checks: four units whose four inner
    /// regions differ only in which unit they name come out as four separate regions
    /// (`flatten_local_region.mlir`, `CHECK-SENT-IR` `:19`-`:66`).
    ///
    /// ⭐ THE OUT-PARAMETER BECOMES THE RETURN VALUE. `flatten` declares the map empty immediately
    /// before the call (`:396-397`) and is the only caller, so the accumulator has exactly one filling
    /// and nothing depends on appending to a map that already holds classes.
    #[must_use]
    pub fn partition_units(&self) -> Vec<(Val, Vec<Val>)> {
        // `llvm::MapVector<mlir::Value, std::vector<mlir::Value>> &equivalence_classes` (`:169-170`).
        let mut equivalence_classes: Vec<(Val, Vec<Val>)> = Vec::new();
        // `std::vector<mlir::Value> visited;` (`:172`).
        let mut visited: Vec<Val> = Vec::new();
        for (unit0, ops0) in self.unit_to_ops.entries() {
            // `if (std::find(...) != visited.end()) continue;` (`:174-176`) — this unit already
            // belongs to a class, and a class is decided once.
            if visited.contains(unit0) {
                continue;
            }
            for (unit1, ops1) in self.unit_to_ops.entries() {
                // The same skip for the inner unit (`:178-180`).
                if visited.contains(unit1) {
                    continue;
                }
                if runs_the_same_operations(ops0, ops1) {
                    // `equivalence_classes[unit_to_op0.first].push_back(unit_to_op1.first);` (`:182`)
                    // — a `MapVector` subscript, so a key not yet present is appended in key order.
                    match equivalence_classes
                        .iter_mut()
                        .find(|(unit, _)| unit == unit0)
                    {
                        Some((_, class)) => class.push(*unit1),
                        None => equivalence_classes.push((*unit0, vec![*unit1])),
                    }
                    // `visited.push_back(unit_to_op1.first);` (`:183`).
                    visited.push(*unit1);
                }
            }
        }
        equivalence_classes
    }
}

impl<'p> Default for FlatteningLocalRegionsTree<'p> {
    /// The reference's only constructor is the default one (entry 106); this forwards to it so the two
    /// cannot disagree.
    fn default() -> Self {
        Self::new()
    }
}

/// DO TWO UNITS RUN THE SAME OPERATIONS — `unit_to_op0.second == unit_to_op1.second`
/// (`FlatteningLocalRegions.cpp:181`).
///
/// # ⛔⛔ ELEMENTWISE, IN ORDER, AND BY ADDRESS
///
/// `std::vector::operator==` compares length then elements pairwise, and the elements are
/// `mlir::Operation *`. So this asks *"were these two units walked over the same operation objects,
/// in the same order"*, and:
///
/// - two units of ONE region always match, because `traverseRegion` pushes the same `&op` for every
///   unit in the list it was given (`:140-143`);
/// - two units of two regions match only if a nested `uniform.uniformize_regions` attributed the same
///   inner operations to both, since each region owns its own operation objects;
/// - two units with no operations at all would match trivially, `{} == {}` — ⚠️ unreachable from the
///   walk, which only ever subscripts the map to push (`:142`), so an entry always has an operation.
///
/// ⛔ A STRUCTURAL `==` WOULD BE A DIFFERENT AND WEAKER PREDICATE, and it would merge regions the
/// reference keeps apart — `DfirOp` derives `PartialEq`, so the mistake is one character wide. The
/// reference is consistent about this: `OperationNode::operator==` is `this == &n`
/// (`dcc/src/Analysis/OperationTree.hpp:34`) and `cloneOpsForRegions` recognises an operation with
/// `std::find` over a list of pointers (`:233-234`).
fn runs_the_same_operations(lhs: &[&DfirOp], rhs: &[&DfirOp]) -> bool {
    lhs.len() == rhs.len() && lhs.iter().zip(rhs).all(|(l, r)| core::ptr::eq(*l, *r))
}


#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::dialects::{arith, scf};

    /// ONE OPERATION PER LINE OF A REGION — the ops the nodes below stand for.
    ///
    /// ⚠️ WHICH ops they are is immaterial to entries 101-108: the tree is opaque in the operation
    /// until entry 180 asks `isa<uniform::UniformizeRegionsOp>`. What matters is that they are
    /// DISTINCT objects, because that is what entry 108 compares.
    fn a_region() -> Vec<DfirOp> {
        vec![
            DfirOp::Scf(scf::Op::If {
                cond: Val(30),
                body: Vec::new(),
                else_body: Vec::new(),
            }),
            DfirOp::Arith(arith::Op::Constant {
                result: Val(1),
                value: 1,
            }),
            DfirOp::Arith(arith::Op::Constant {
                result: Val(2),
                value: 2,
            }),
            DfirOp::Arith(arith::Op::Constant {
                result: Val(3),
                value: 3,
            }),
        ]
    }

    /// A TREE WHOSE ROOT'S REGION HOLDS THE REMAINING OPERATIONS, IN ORDER.
    ///
    /// ```text
    ///   ops[0] ── first_child ─► ops[1] ─ next_sibling ─► ops[2] ─ next_sibling ─► ops[3]
    ///                               └────────── parent ─► ops[0] ◄──────────┘
    /// ```
    ///
    /// ⭐ BUILT THROUGH THE BASE'S OWN `insertChildNode`, which is what `traverseRegion` does
    /// (`FlatteningLocalRegions.cpp:134-137`) — so the links under test are wired by the code that
    /// wires them in the pass, not by the test. THREE children rather than two so that
    /// [`FlatteningLocalRegionsTree::prev_sibling`] has a chain to walk instead of a first-child
    /// special case.
    fn a_flattening_tree<'p>(ops: &'p [DfirOp]) -> (FlatteningLocalRegionsTree<'p>, Vec<LocalOpNodeId>) {
        // `auto new_node = new LocalOpNode(&op_); root_ = new_node;` (`:392-393`).
        let mut base = OperationTreeBase::with_root(LocalOpNode::new(&ops[0]));
        let root = base.root();
        let mut ids = vec![LocalOpNodeId(root)];
        for op in &ops[1..] {
            ids.push(LocalOpNodeId(base.push_child(root, LocalOpNode::new(op))));
        }
        let tree = FlatteningLocalRegionsTree {
            unit_to_ops: UnitToOps::new(),
            base: Some(base),
        };
        (tree, ids)
    }

    /// 🎯 106/384 — A FRESH TREE HAS NO ROOT AND NO ATTRIBUTED OPERATIONS.
    ///
    /// `: OperationTreeBase()` leaves `root_ == nullptr` (`dcc/src/Analysis/OperationTree.hpp:232`)
    /// and `unit_to_ops` default-constructed. ⭐ AND THAT STATE IS REACHABLE IN THE PASS, not just at
    /// construction: `flatten` calls `clear()` before it knows whether it will build anything
    /// (`FlatteningLocalRegions.cpp:385-387`), so a caller can hold a rootless tree.
    #[test]
    fn a_fresh_tree_has_no_root() {
        let tree = FlatteningLocalRegionsTree::new();
        assert_eq!(tree.root(), None, "`root_ = nullptr`");
        assert!(
            tree.unit_to_ops.entries().is_empty(),
            "`unit_to_ops` is default-constructed"
        );
        assert_eq!(
            tree.partition_units(),
            Vec::new(),
            "no units, so no equivalence classes"
        );
    }

    /// 🎯 106/384 — `Default` AND THE REFERENCE'S ONLY CONSTRUCTOR AGREE.
    #[test]
    fn the_default_tree_is_the_constructed_one() {
        let tree = FlatteningLocalRegionsTree::default();
        assert_eq!(tree.root(), None);
        assert!(tree.unit_to_ops.entries().is_empty());
    }

    /// 🎯 101/384 — A FRESH NODE CARRIES ITS OPERATION, NO UNITS, AND REGION 0.
    ///
    /// The constructor forwards the operation and runs the member initialisers, and nothing else:
    /// `traverseRegion` sets the region number on the line after the `new`
    /// (`FlatteningLocalRegions.cpp:134-135`) and pushes the units four lines later, so this is the
    /// state everything else starts from. ⭐ THE DETACHMENT IS NOW THE ARENA'S HALF — see
    /// [`the_root_is_the_operation_being_flattened`], which checks it where it lives.
    #[test]
    fn a_fresh_node_carries_its_operation_and_nothing_else() {
        let op = DfirOp::Scf(scf::Op::If {
            cond: Val(30),
            body: Vec::new(),
            else_body: Vec::new(),
        });
        let node = LocalOpNode::new(&op);
        assert!(
            core::ptr::eq(node.op, &op),
            "`OperationNode(op) : operation_op_(op)` stores the operation itself"
        );
        assert!(node.units.is_empty(), "`units` is default-constructed");
        assert_eq!(
            node.is_in_region_num,
            RegionNum(0),
            "`int is_in_region_num = 0`"
        );
    }

    /// 🎯 107/384 — THE ROOT IS THE NODE `flatten` INSTALLED, AND IT IS PARENTLESS AND SIBLINGLESS.
    ///
    /// Those are the second and third conjuncts of the `DT_CHECK` this entry inherits
    /// (`dcc/src/Analysis/OperationTree.hpp:205-206`), and the root stands for the
    /// `uniform.uniformize_regions` being replaced — which is why `flatten` hands
    /// `getRoot()->getFirstChild()` and not the root to `cloneOpsForRegions` (`:447`).
    #[test]
    fn the_root_is_the_operation_being_flattened() {
        let ops = a_region();
        let (tree, ids) = a_flattening_tree(&ops);
        let root = tree.root().expect("a tree built with a root has one");
        assert_eq!(root, ids[0]);
        assert!(core::ptr::eq(
            tree.node(root).expect("the root's payload").op,
            &ops[0]
        ));
        assert_eq!(tree.parent_node(root), None, "`getParentNode() == nullptr`");
        assert_eq!(tree.next_sibling(root), None, "`getNextSibling() == nullptr`");
    }

    /// 🎯 102/384 — EVERY OPERATION OF A REGION NAMES THE OPERATION THAT OWNS THE REGION.
    ///
    /// An absent parent is what identifies the synthetic root the tree collects its forest under
    /// (`dcc/src/Analysis/OperationTree.hpp:62-69`, `getDepth`/`isOutermost`).
    #[test]
    fn the_parent_of_a_region_is_the_op_that_owns_it() {
        let ops = a_region();
        let (tree, ids) = a_flattening_tree(&ops);
        for child in &ids[1..] {
            assert_eq!(tree.parent_node(*child), Some(ids[0]));
        }
        assert_eq!(tree.parent_node(ids[0]), None, "the root has no parent");
    }

    /// 🎯 103/384 — THE FIRST CHILD IS THE FIRST OPERATION IN SYNTACTIC ORDER, AND A LEAF HAS NONE.
    #[test]
    fn the_first_child_is_the_first_operation_in_the_region() {
        let ops = a_region();
        let (tree, ids) = a_flattening_tree(&ops);
        assert_eq!(tree.first_child(ids[0]), Some(ids[1]));
        for child in &ids[1..] {
            assert_eq!(
                tree.first_child(*child),
                None,
                "`isLeaf()` is `getFirstChild() == nullptr`"
            );
        }
    }

    /// 🎯 104/384 — THE SIBLING CHAIN WALKS THE REGION IN ORDER AND ENDS ON `None`.
    ///
    /// This is the loop every traversal in the file runs — `while (node) { …; node = getNextSibling(); }`
    /// (`FlatteningLocalRegions.cpp:216-219`, `:228` onwards) — so the test walks it the same way and
    /// collects what it visits, which is what tells the loop it may stop.
    #[test]
    fn the_sibling_chain_walks_the_region_in_order() {
        let ops = a_region();
        let (tree, ids) = a_flattening_tree(&ops);
        let mut visited = Vec::new();
        let mut cursor = tree.first_child(ids[0]);
        while let Some(node) = cursor {
            visited.push(node);
            cursor = tree.next_sibling(node);
        }
        assert_eq!(
            visited,
            ids[1..].to_vec(),
            "every operation of the region, in syntactic order"
        );
        assert_eq!(
            tree.next_sibling(ids[3]),
            None,
            "the end of the region needs no count"
        );
    }

    /// 🎯 105/384 — THE PREVIOUS SIBLING IS FOUND BY WALKING FORWARD FROM THE FIRST CHILD.
    ///
    /// ⭐ THE THIRD CHILD IS THE ONE THAT PROVES IT IS A WALK: reaching it takes two hops from the
    /// parent's `first_child_`, which is the loop at `dcc/src/Analysis/OperationTree.cpp:41-44`. The
    /// first child's `None` is the early return at `:40`, and the ROOT's `None` is the
    /// `DT_CHECK_MSG(getParentNode(), "expected a parent")` at `:38` answered rather than refused —
    /// the root has no siblings, so `None` is the true answer.
    #[test]
    fn the_previous_sibling_walks_back_up_the_region() {
        let ops = a_region();
        let (tree, ids) = a_flattening_tree(&ops);
        assert_eq!(tree.prev_sibling(ids[3]), Some(ids[2]), "two hops");
        assert_eq!(tree.prev_sibling(ids[2]), Some(ids[1]), "one hop");
        assert_eq!(
            tree.prev_sibling(ids[1]),
            None,
            "`if (prev == this) return nullptr`"
        );
        assert_eq!(
            tree.prev_sibling(ids[0]),
            None,
            "the root has no parent to walk and no sibling to find"
        );
    }

    /// 🎯 108/384 — THE VENDOR'S `@diff_groups`: FOUR UNITS, FOUR CLASSES, IN WALK ORDER.
    ///
    /// `flatten_local_region.mlir` declares four units and groups them two per region — `(%arg1 ->
    /// %0, %2)` at `:91` and `(%arg1 -> %1, %3)` at `:120` — each region holding a nested
    /// `uniform.uniformize_regions` with ONE region per unit (`:93`, `:105`, `:122`, `:134`). So the
    /// walk attributes each unit its OWN copy of the body, plus the enclosing region's shared
    /// `uniform.yield`; the four operation lists are therefore pairwise distinct, and:
    ///
    /// - the pass emits **four** regions, `%0`, `%2`, `%1`, `%3` — `CHECK-SENT-IR` `:19`, `:31`,
    ///   `:43`, `:55` against the unit definitions at `:8-11`;
    /// - four ≠ the two the op came in with, so `flatten` does NOT decline the rewrite (`:418`);
    /// - and the order is the `MapVector`'s key order, which is the order the walk first reached each
    ///   unit — `%2` before `%1` because the whole of the first region is walked first.
    ///
    /// ⛔ THE SHARED `uniform.yield` IS THE POINT: `%0` and `%2` both end their list with the SAME
    /// operation object and are still not merged, because the entries before it differ.
    #[test]
    fn the_vendor_diff_groups_case_yields_four_classes_in_walk_order() {
        let bodies = a_region();
        let yields = [
            DfirOp::Scf(scf::Op::Yield {
                operands: Vec::new(),
            }),
            DfirOp::Scf(scf::Op::Yield {
                operands: Vec::new(),
            }),
        ];
        let mut tree = FlatteningLocalRegionsTree::new();
        // The walk order of `@diff_groups`: the outer op's first region in full, then its second.
        for (unit, body, shared) in [
            (Val(0), &bodies[0], &yields[0]),
            (Val(2), &bodies[1], &yields[0]),
            (Val(1), &bodies[2], &yields[1]),
            (Val(3), &bodies[3], &yields[1]),
        ] {
            tree.unit_to_ops.push_op(unit, body);
            tree.unit_to_ops.push_op(unit, shared);
        }
        let classes = tree.partition_units();
        assert_eq!(
            classes,
            vec![
                (Val(0), vec![Val(0)]),
                (Val(2), vec![Val(2)]),
                (Val(1), vec![Val(1)]),
                (Val(3), vec![Val(3)]),
            ],
            "four singleton classes, keyed in first-insertion order"
        );
        assert_ne!(
            classes.len(),
            2,
            "`if (op_.getNumRegions() == num_of_regions) return false;` does not fire"
        );
    }

    /// 🎯 108/384 — UNITS WALKED OVER THE SAME OPERATIONS SHARE A CLASS, AND THE KEY IS THE FIRST OF
    /// THEM.
    ///
    /// This is the case the pass exists for: two units in ONE region are handed the same `&op` for
    /// every operation (`FlatteningLocalRegions.cpp:140-143`), so their lists are pointer-identical
    /// and they collapse into one region. ⭐ THE CLASS CONTAINS ITS OWN REPRESENTATIVE because the
    /// inner loop meets the outer entry before any other — which is what makes `unit_rep_order.at(i)`
    /// a key `unit_to_ops` can be looked up under (`:448`).
    #[test]
    fn units_running_the_same_operations_share_a_class() {
        let ops = a_region();
        let mut tree = FlatteningLocalRegionsTree::new();
        for op in [&ops[0], &ops[1]] {
            tree.unit_to_ops.push_op(Val(10), op);
            tree.unit_to_ops.push_op(Val(12), op);
        }
        tree.unit_to_ops.push_op(Val(11), &ops[0]);
        tree.unit_to_ops.push_op(Val(13), &ops[1]);
        assert_eq!(
            tree.partition_units(),
            vec![
                (Val(10), vec![Val(10), Val(12)]),
                (Val(11), vec![Val(11)]),
                (Val(13), vec![Val(13)]),
            ],
            "`%10` and `%12` ran the same two operations; `%11` and `%13` ran one each"
        );
    }

    /// 🎯 108/384 — TWO STRUCTURALLY IDENTICAL BUT DISTINCT OPERATIONS DO NOT MERGE THEIR UNITS.
    ///
    /// ⛔⛔ `std::vector<Operation *>::operator==` compares ADDRESSES, so this is the difference
    /// between the reference's predicate and the one a derived `PartialEq` on `DfirOp` would give.
    /// The test asserts the two operations ARE structurally equal first, so that its subject is
    /// identity and not a difference in the fixtures — and this is exactly what keeps
    /// `@diff_groups`' four regions apart, since its four bodies differ only in which unit they name.
    #[test]
    fn structurally_identical_operations_are_still_different_operations() {
        let lhs = DfirOp::Arith(arith::Op::Constant {
            result: Val(1),
            value: 1,
        });
        let rhs = DfirOp::Arith(arith::Op::Constant {
            result: Val(1),
            value: 1,
        });
        assert_eq!(lhs, rhs, "the fixtures are structurally equal");
        assert!(
            !runs_the_same_operations(&[&lhs], &[&rhs]),
            "and they are still two operations"
        );
        let mut tree = FlatteningLocalRegionsTree::new();
        tree.unit_to_ops.push_op(Val(20), &lhs);
        tree.unit_to_ops.push_op(Val(21), &rhs);
        assert_eq!(
            tree.partition_units(),
            vec![(Val(20), vec![Val(20)]), (Val(21), vec![Val(21)])],
            "two regions, not one"
        );
    }

    /// 🎯 108/384 — THE COMPARISON IS ORDERED: THE SAME OPERATIONS IN A DIFFERENT ORDER ARE A
    /// DIFFERENT PROGRAM.
    ///
    /// `std::vector::operator==` is elementwise in index order, and a region's list is in syntactic
    /// order, so two units cannot share a region merely by running the same set of operations.
    #[test]
    fn the_same_operations_in_a_different_order_do_not_merge() {
        let ops = a_region();
        let mut tree = FlatteningLocalRegionsTree::new();
        tree.unit_to_ops.push_op(Val(30), &ops[0]);
        tree.unit_to_ops.push_op(Val(30), &ops[1]);
        tree.unit_to_ops.push_op(Val(31), &ops[1]);
        tree.unit_to_ops.push_op(Val(31), &ops[0]);
        assert_eq!(
            tree.partition_units(),
            vec![(Val(30), vec![Val(30)]), (Val(31), vec![Val(31)])]
        );
    }

    /// 🎯 108/384 — LISTS OF DIFFERENT LENGTHS NEVER MATCH, EVEN WHEN ONE IS A PREFIX OF THE OTHER.
    ///
    /// The length test is `std::vector::operator==`'s first act, and it is what stops a unit whose
    /// region ended early from joining one that carried on.
    #[test]
    fn a_prefix_is_not_the_same_program() {
        let ops = a_region();
        let mut tree = FlatteningLocalRegionsTree::new();
        tree.unit_to_ops.push_op(Val(40), &ops[0]);
        tree.unit_to_ops.push_op(Val(41), &ops[0]);
        tree.unit_to_ops.push_op(Val(41), &ops[1]);
        assert_eq!(
            tree.partition_units(),
            vec![(Val(40), vec![Val(40)]), (Val(41), vec![Val(41)])]
        );
    }

    /// A UNIT'S OPERATIONS ARRIVE IN WALK ORDER AND THE KEY ORDER IS THE FIRST SIGHTING.
    ///
    /// `MapVector::operator[]` appends a key the first time it is subscripted
    /// (`FlatteningLocalRegions.cpp:142`), and that order is observable in the emitted region order —
    /// see [`the_vendor_diff_groups_case_yields_four_classes_in_walk_order`].
    #[test]
    fn the_unit_map_keeps_first_insertion_order() {
        let ops = a_region();
        let mut map = UnitToOps::new();
        map.push_op(Val(9), &ops[0]);
        map.push_op(Val(7), &ops[1]);
        map.push_op(Val(9), &ops[2]);
        let keys: Vec<Val> = map.entries().iter().map(|(unit, _)| *unit).collect();
        assert_eq!(keys, vec![Val(9), Val(7)], "`%9` was seen first");
        assert_eq!(map.entries()[0].1.len(), 2, "and it ran two operations");
    }
}
