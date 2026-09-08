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
use crate::islands::dataflow_ir::dialects::{self, Op as DfirOp, Val, uniform};
use crate::islands::dataflow_ir::{ValueMapping, Values};

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
/// `op.getNumRegions()` (`FlatteningLocalRegions.cpp:144-146`), so a node carries the index of the
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
/// ⚠️ `op` IS ANY `DfirOp` BECAUSE ENTRIES 101-108 NEVER LOOK AT IT. A constructor, four link reads
/// and a partition are opaque in the operation: entry 108 compares op IDENTITY and never op contents.
/// The ops this tree actually holds are `uniform.uniformize_regions` and `uniform.yield` and their
/// contents, and the first unit that asked an op WHICH op it is —
/// [`FlatteningLocalRegionsTree::traverse_region`], entry 180, on
/// `isa<uniform::UniformizeRegionsOp>` (`:137`) — is what added the dialect to
/// [`crate::islands::dataflow_ir::dialects::uniform`]. Entries 247 and 287 ask the same question.
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
    /// walks a sibling chain looking for this value, and the two mentions of it in the whole file
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

    /// Replaces: e179_clear
    ///
    /// **179/384** `FlatteningLocalRegionsTree::clear` —
    /// `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:112` (15L).
    ///
    /// ```cpp
    /// void FlatteningLocalRegionsTree::clear() {
    ///   if (!empty()) {
    ///     SmallVector<LocalOpNode *> to_be_deleted;
    ///     LocalOpNode::walk<OperationNode::WalkOrder::kPostOrder>(
    ///         const_cast<LocalOpNode *>(getRoot()),
    ///         [&](LocalOpNode *n) -> LocalOpNode * {
    ///           to_be_deleted.push_back(n);
    ///           return nullptr;
    ///         });
    ///     for (LocalOpNode *n : to_be_deleted) delete n;
    ///   } else if (root_)
    ///     delete root_;
    ///   root_ = nullptr;
    ///
    ///   unit_to_ops.clear();
    /// }
    /// ```
    ///
    /// # ⛔⛔ FOURTEEN OF ITS FIFTEEN LINES ARE THE `delete`, AND THE `delete` IS THE ARENA'S DROP
    ///
    /// The reference's tree is a heap of individually `new`ed nodes (`:134`, `:392`) wired by three raw
    /// pointers, so freeing it needs a walk that reaches every node and frees each exactly once —
    /// hence the two-phase shape: collect in a post-order walk, THEN delete, because deleting inside
    /// the walk would free the node whose `next_sibling_` the walk is about to read. Here the nodes
    /// live in one `Vec` inside [`OperationTreeBase`] (see [`LocalOpNodeId`]), so dropping the arena
    /// frees all of them, in one statement, with no walk to get wrong.
    ///
    /// ⭐ THE THREE STATES THE `if` DISTINGUISHES ALL COLLAPSE TO `None`. `empty()` is
    /// `!root_ || !root_->getFirstChild()` (`dcc/src/Analysis/OperationTree.hpp:214`), so the branches
    /// are: a root WITH children (walk and delete every node, the root included — `postOrderWalk`
    /// visits `n` itself after its descendants and never its siblings,
    /// `dcc/src/Analysis/OperationTree.cpp:145-153`); a root with NO children (`delete root_`, the one
    /// node); and NO root at all (nothing to free). The split exists because `getRoot()` re-asserts
    /// `root_ != nullptr` (`hpp:205-206`) and so cannot be called in the third state — it is a guard
    /// against the reference's own accessor, not a difference in what gets freed.
    ///
    /// ⛔ SO WHAT SURVIVES IS EXACTLY WHAT IS OBSERVABLE AFTERWARDS: `root_ = nullptr` (`:124`) and
    /// `unit_to_ops.clear()` (`:126`) — a tree with no root and no attributed operations, which is
    /// [`Self::new`]'s state. ⚠️ IT IS **NOT** WRITTEN AS `*self = Self::new()`, because that would
    /// make the two functions one and this one would stop being a port of `:112`; the two fields are
    /// reset in the reference's own order.
    ///
    /// # ⭐ ITS TWO CALLERS ARE THE DESTRUCTOR AND THE FIRST LINE OF `flatten`
    ///
    /// `~FlatteningLocalRegionsTree() { clear(); }` (`:79`, entry 246) is the RAII half — and it has
    /// nothing to do here, because dropping the tree already drops the arena. `flatten` calls it as
    /// its **first** statement (`:385`), before it knows whether the operation it was handed is a
    /// `uniform.uniformize_regions` at all (`:386-387`) — which is why the rootless state is
    /// reachable in the pass and not only at construction, and why [`Self::root`] returns an
    /// [`Option`].
    ///
    /// ⚠️ AND IT IS NOT THE SUBTREE OVERLOAD. `OperationTreeBase::clear(OperationNode *start)`
    /// (`hpp:230`, reached through `remove`, `hpp:216`) frees one subtree and keeps the tree; nothing
    /// in this file calls it.
    pub fn clear(&mut self) {
        // `root_ = nullptr;` (`:124`) — and with it every node, since the arena owns them.
        self.base = None;
        // `unit_to_ops.clear();` (`:126`).
        self.unit_to_ops = UnitToOps::new();
    }

    /// Replaces: e180_traverseRegion
    ///
    /// **180/384** `FlatteningLocalRegionsTree::traverseRegion` —
    /// `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:129` (17L).
    ///
    /// ```cpp
    /// void FlatteningLocalRegionsTree::traverseRegion(mlir::Region &region,
    ///                                                 LocalOpNode *parent_node,
    ///                                                 std::vector<mlir::Value> &units,
    ///                                                 int is_in_region_num) {
    ///   for (auto &op : region.getOps()) {
    ///     auto new_node = new LocalOpNode(&op);
    ///     new_node->is_in_region_num = is_in_region_num;
    ///     parent_node->insertChildNode(new_node);
    ///     if (isa<uniform::UniformizeRegionsOp>(op)) {
    ///       compute(new_node);
    ///     } else {
    ///       for (auto u : units) {
    ///         new_node->units.push_back(u);
    ///         unit_to_ops[u].push_back(&op);
    ///       }
    ///       for (int region_num = 0; region_num < op.getNumRegions(); region_num++) {
    ///         traverseRegion(op.getRegion(region_num), new_node, units, region_num);
    ///       }
    ///     }
    ///   }
    /// }
    /// ```
    ///
    /// # ⭐⭐ THE WALK THAT ATTRIBUTES EVERY OPERATION TO EVERY UNIT THAT RUNS IT
    ///
    /// It builds **two** things at once and the pass needs both: the node tree
    /// [`Self::clone_ops_for_regions`] later clones from, and [`Self::unit_to_ops`] — the map
    /// [`Self::partition_units`] turns into the equivalence classes that become the new op's regions.
    /// Nothing else in the file writes either.
    ///
    /// # ⛔⛔ THE UNIT LIST IS INHERITED BY NESTING, NOT RE-DERIVED
    ///
    /// `:145` passes the SAME `units` into every nested region, so an operation inside an `scf.if`
    /// inside a local region is attributed to every unit of that local region — the enclosing
    /// conditional does not narrow it. That is what makes the vendor's four-region case decidable: in
    /// `flatten_local_region4.mlir` all 32 units run the same outer `arith.cmpi`/`scf.if` chain and
    /// differ **only** in the three operations of the nested local regions, so the partition comes back
    /// as two classes of 16 (input `:737-765`, expectation `:325-395`).
    ///
    /// # ⛔⛔ A `uniform.uniformize_regions` GETS NO UNITS AND NO REGION WALK OF ITS OWN
    ///
    /// Its node is inserted and then handed to `compute` (`:137-138`), which re-derives a unit list
    /// **per region** from the op's own `$units`/`$list_sizes` and calls back in
    /// (`:151-166`). So the node itself stays with `units` empty — which is exactly the
    /// `(uniformize_regions1, {})` of the worked example in the file's own algorithm comment
    /// (`:303-382`) — and it is never subscripted into `unit_to_ops`. ⭐ THAT EMPTINESS IS LOAD-BEARING
    /// in [`Self::clone_ops_for_regions`]: an op that is in no unit's list fails the
    /// `std::find` at `:233` and takes the flatten-through branch at `:235-238`.
    ///
    /// ⛔ AND IT IS WHY `is_in_region_num` CANNOT IDENTIFY A LOCAL REGION. `compute` calls back with
    /// `false` for **every** region index (`:164`), so all of a uniformized op's children are stamped
    /// region 0 no matter which region they came from. Recovering the real one is a pointer comparison
    /// against the parent's region list, and [`Self::clone_ops_for_regions`] does it at `:246-256`
    /// because this walk did not.
    ///
    /// # ⭐ `compute` IS ENTRY 247/384 AND THE RECURSION THROUGH IT IS MUTUAL
    ///
    /// A NESTED `uniform.uniformize_regions` is handed to [`Self::compute`] — ported in this file —
    /// which walks each of its regions back into this function with that region's own unit list. The
    /// same file's `flatten` (287) and `runOnOperation` (305) are still open. ⚠️ A region with no
    /// nested local region never reaches it, which is every region of the vendor's cases 1-3 and the
    /// outer region of case 4.
    ///
    /// ⚠️ `region` IS A SLICE, NOT AN `mlir::Region`. A region here is the block's operation list
    /// (`islands/dataflow_ir/dialects` has no block type), and `region.getOps()` is iterating it —
    /// terminator included, which matters: `uniform.yield` and `scf.yield` DO get nodes and DO enter
    /// `unit_to_ops`, and `cloneOpsForRegions` skips the former by kind (`:229`) rather than by absence.
    pub fn traverse_region(
        &mut self,
        region: &'p [DfirOp],
        parent_node: LocalOpNodeId,
        units: &[Val],
        is_in_region_num: RegionNum,
    ) {
        // `for (auto &op : region.getOps())` (`:133`) — in syntactic order, which is the order the
        // sibling chain and `unit_to_ops` both come out in.
        for op in region {
            // `auto new_node = new LocalOpNode(&op); new_node->is_in_region_num = is_in_region_num;`
            // (`:134-135`).
            let mut new_node = LocalOpNode::new(op);
            new_node.is_in_region_num = is_in_region_num;
            // `isa<uniform::UniformizeRegionsOp>(op)` (`:137`) — asked once, because both branches
            // below need the answer.
            let uniformizes = matches!(op, DfirOp::Uniform(uniform::Op::UniformizeRegions { .. }));
            if !uniformizes {
                // `for (auto u : units) { new_node->units.push_back(u); unit_to_ops[u].push_back(&op); }`
                // (`:140-143`) — one map entry per unit, appended in unit order.
                //
                // ⚠️ HOISTED ABOVE THE INSERTION, which the reference does after it (`:136`): the arena
                // hands back an id rather than a reference, so the payload must be complete before it
                // goes in. Nothing reads the node in between, and neither the sibling order nor the
                // `unit_to_ops` order changes.
                for unit in units {
                    new_node.units.push(*unit);
                    self.unit_to_ops.push_op(*unit, op);
                }
            }
            // `parent_node->insertChildNode(new_node);` (`:136`) — appended after the parent's last
            // child (`dcc/src/Analysis/OperationTree.cpp:239-254`), so the chain keeps program order.
            //
            // ⛔ NO BASE MEANS NO TREE, so there is no parent for a child to be inserted under and
            // nothing to walk — the state `clear()` leaves and `flatten` returns early from
            // (`:385-387`). It is unreachable from `compute`, which is only called on a node.
            let Some(base) = self.base.as_mut() else {
                return;
            };
            let new_node = LocalOpNodeId(base.push_child(parent_node.0, new_node));
            if uniformizes {
                // `compute(new_node);` (`:138`) — the nested local op's own regions, each walked with
                // its own unit list. ⭐ MUTUAL RECURSION, as in the reference.
                self.compute(new_node);
            } else {
                // `for (int region_num = 0; region_num < op.getNumRegions(); region_num++)
                //    traverseRegion(op.getRegion(region_num), new_node, units, region_num);`
                // (`:144-146`) — the child's `is_in_region_num` is the index of the PARENT'S region it
                // sits in.
                //
                // ⛔ INSIDE THE `else`, AND THAT IS NOT COSMETIC: the reference does NOT walk a nested
                // `uniform.uniformize_regions`' own regions from here, because `compute` walks them
                // with the per-region unit lists instead. Nesting the loop is what keeps that true
                // now that the other arm calls `compute`.
                for (region_num, inner) in (0u32..).zip(dialects::regions(op)) {
                    self.traverse_region(inner, new_node, units, RegionNum(region_num));
                }
            }
        }
    }

    /// Replaces: e247_compute
    ///
    /// **247/384** `FlatteningLocalRegionsTree::compute` — `FlatteningLocalRegions.cpp:151` (15L):
    /// the per-region unit lists of a NESTED `uniform.uniformize_regions`, walked back into the tree.
    ///
    /// ⛔ EVERY REGION IS STAMPED REGION 0 — `traverseRegion(.., /*is_in_region_num=*/false)` for all
    /// of them (`:164`), which is why [`Self::clone_ops_for_regions`] recovers the real region by
    /// pointer comparison (`:246-256`). ⭐ `DT_CHECK(units_for_each_region.size() == getNumRegions())`
    /// (`:161`) is unwritable: [`uniform::LocalRegion`] pairs each region's ops with its own units.
    pub fn compute(&mut self, node: LocalOpNodeId) {
        // `auto uniform_op = dyn_cast<uniform::UniformizeRegionsOp>(node->getOperation());
        //  if (!uniform_op) return;` (`:152-154`) — the only caller has already asked the same
        // question (`:137`), so the null arm is the reference's own belt and braces.
        let op: &'p DfirOp = match self.node(node) {
            Some(payload) => payload.op,
            // ⚠️ A NODE THAT IS NOT IN THIS TREE, which the reference cannot spell: `node` is a
            // pointer there and an id here. Same answer as a failed `dyn_cast` — nothing to walk.
            None => return,
        };
        let DfirOp::Uniform(uniform::Op::UniformizeRegions { regions, .. }) = op else {
            return;
        };

        // `getUnitsPerRegionsAsVectorOfVector(uniform_op, units_for_each_region);` (`:157-158`) is
        // this iteration: one [`uniform::LocalRegion`] per region, carrying its own unit list.
        //
        // `for (int idx = 0; idx < getNumRegions(); idx++)
        //    traverseRegion(getRegion(idx), node, units_for_each_region.at(idx), false);` (`:162-165`)
        // — the node is the PARENT of every region's children, so all of them become its siblings in
        // one chain, and `false` is region 0 for every one of them.
        for region in regions {
            self.traverse_region(&region.body, node, &region.units, RegionNum(0));
        }
    }

    /// Replaces: e181_inRegionEmpty
    ///
    /// **181/384** `FlatteningLocalRegionsTree::inRegionEmpty` —
    /// `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:214` (6L).
    ///
    /// ```cpp
    /// bool FlatteningLocalRegionsTree::inRegionEmpty(int region_num,
    ///                                                LocalOpNode *node) {
    ///   while (node) {
    ///     if (node->is_in_region_num == region_num) return false;
    ///     node = node->getNextSibling();
    ///   }
    ///   return true;
    /// }
    /// ```
    ///
    /// # ⭐ DOES REGION `region_num` OF SOME PARENT HOLD NOTHING — ASKED OVER A SIBLING CHAIN
    ///
    /// `node` is one region's worth of operations reached as a chain, and every node in it carries the
    /// index of the parent's region it came from (`:145`). So the question is answered by looking for
    /// **any** sibling stamped `region_num`, and `true` means the parent's region `region_num`
    /// contributed no node at all.
    ///
    /// ⚠️ `node` IS AN [`Option`] BECAUSE THE REFERENCE IS CALLED WITH `getFirstChild()`, which is null
    /// for a leaf — `isLeaf()` is defined as exactly that (`dcc/src/Analysis/OperationTree.hpp:70`). A
    /// chain that starts nowhere is a parent with no children, and `true` is the right answer.
    ///
    /// ⚠️ IT WALKS FORWARD ONLY, so it answers about the whole chain only when handed its head. Handed
    /// a node in the middle it answers about the suffix, and nothing in the reference constrains that
    /// — it is the caller's business.
    ///
    /// # ⛔⛔ IT HAS NO CALLER ANYWHERE IN THE AUTHORITY TREE, AND THE REASON IS INSTRUCTIVE
    ///
    /// The two mentions of `inRegionEmpty` in `dcc/` are its declaration (`:88`), this definition and
    /// nothing else. ⭐ THE PLACE THAT WANTS IT IS `cloneOpsForRegions` AT `:269`, which needs to know
    /// whether region `rn` of a cloned op came out empty — and it answers that by BUILDING the block
    /// and calling `block.erase()` when it is still empty afterwards, a test on the RESULT rather than
    /// on `is_in_region_num`. That is not a stylistic difference: this predicate would have given the
    /// wrong answer there. `compute` stamps region 0 on the children of **every** region of a
    /// uniformized op (`:164`), so `inRegionEmpty(1, uniformize_node->first_child())` reports "empty"
    /// for a two-region local op whose second region is full — and `cloneOpsForRegions` also filters by
    /// unit class (`:233`), which this predicate cannot see at all.
    ///
    /// ⛔ PORTED ANYWAY, uncalled, because it is a scheduled unit — the same decision entry 042 records.
    /// Deciding it unnecessary is how the previous attempt failed.
    #[must_use]
    pub fn in_region_empty(&self, region_num: RegionNum, node: Option<LocalOpNodeId>) -> bool {
        let mut node = node;
        // `while (node)` (`:216`) — to the end of the sibling chain, which needs no count.
        while let Some(current) = node {
            // `if (node->is_in_region_num == region_num) return false;` (`:217`) — one operation from
            // that region is enough.
            if self
                .node(current)
                .is_some_and(|payload| payload.is_in_region_num == region_num)
            {
                return false;
            }
            // `node = node->getNextSibling();` (`:218`).
            node = self.next_sibling(current);
        }
        // `return true;` (`:220`).
        true
    }

    /// Replaces: e182_cloneOpsForRegions
    ///
    /// **182/384** `FlatteningLocalRegionsTree::cloneOpsForRegions` —
    /// `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:223` (60L).
    ///
    /// ```cpp
    /// void FlatteningLocalRegionsTree::cloneOpsForRegions(
    ///     LocalOpNode *node, const std::vector<mlir::Operation *> &op_list,
    ///     mlir::OpBuilder builder_region, int region_num, IRMapping &arg_map,
    ///     mlir::BlockArgument &block_arg) {
    ///   std::map<mlir::Operation *, mlir::Operation *> old_to_new_op_map;
    ///   while (node) {
    ///     if (isa<mlir::uniform::YieldOp>(node->getOperation())) {
    ///       node = node->getNextSibling();
    ///       continue;
    ///     }
    ///     if (std::find(op_list.begin(), op_list.end(), node->getOperation()) == op_list.end()) {
    ///       if (isa<uniform::UniformizeRegionsOp>(node->getOperation())) {
    ///         cloneOpsForRegions(node->getFirstChild(), op_list, builder_region,
    ///                            region_num, arg_map, block_arg);
    ///       }
    ///       node = node->getNextSibling();
    ///       continue;
    ///     }
    ///     if (node->is_in_region_num != region_num) {
    ///       node = node->getNextSibling();
    ///       continue;
    ///     }
    ///     if (auto parent_uniform_op = llvm::dyn_cast<uniform::UniformizeRegionsOp>(
    ///             node->getOperation()->getParentOp())) {
    ///       for (int region_idx = 0; region_idx < parent_uniform_op.getNumRegions(); region_idx++) {
    ///         if (&parent_uniform_op.getRegion(region_idx) ==
    ///             node->getOperation()->getParentRegion()) {
    ///           arg_map.map(parent_uniform_op.getRegion(region_idx).getArgument(0), block_arg);
    ///         }
    ///       }
    ///     }
    ///     auto new_op_ = builder_region.cloneWithoutRegions(*node->getOperation(), arg_map);
    ///     old_to_new_op_map[node->getOperation()] = new_op_;
    ///     if (new_op_->getNumRegions() > 0) {
    ///       DT_CHECK(new_op_->getNumRegions() == node->getOperation()->getNumRegions());
    ///       for (int rn = 0; rn < new_op_->getNumRegions(); rn++) {
    ///         OpBuilder builder_inner_region(new_op_->getRegion(rn));
    ///         auto &block = new_op_->getRegion(rn).emplaceBlock();
    ///         builder_inner_region.setInsertionPointToStart(&block);
    ///         cloneOpsForRegions(node->getFirstChild(), op_list, builder_inner_region,
    ///                            rn, arg_map, block_arg);
    ///         if (block.empty()) block.erase();
    ///       }
    ///     }
    ///     node = node->getNextSibling();
    ///   }
    ///   for (auto old_new_op : old_to_new_op_map) {
    ///     auto parent_region = old_new_op.second->getParentRegion();
    ///     old_new_op.first->replaceUsesWithIf(
    ///         old_new_op.second, [&](mlir::OpOperand &use) {
    ///           auto curr_region = use.getOwner()->getParentRegion();
    ///           while (!isa<dataflow::ProgramUnitOp>(curr_region->getParentOp())) {
    ///             if (curr_region == parent_region) return true;
    ///             curr_region = curr_region->getParentRegion();
    ///           }
    ///           return false;
    ///         });
    ///   }
    /// }
    /// ```
    ///
    /// # ⭐⭐ ONE NEW LOCAL REGION'S BODY: THE OPERATIONS OF **ONE UNIT CLASS**, THE NESTING FLATTENED
    ///
    /// `flatten` makes one region per equivalence class (`:430-450`) and calls this once per region with
    /// that class's representative's operation list (`:448`). Every node of the source tree is then
    /// filtered by three questions, in this order:
    ///
    /// 1. **is it a `uniform.yield`** (`:229-232`) — skipped, because `flatten` builds the new region's
    ///    terminator itself (`:450`) and cloning the old one would emit two;
    /// 2. **is it one of this class's operations** (`:233-241`) — by ADDRESS, see
    ///    [`runs_the_same_operations`]. A nested `uniform.uniformize_regions` is never in any list (the
    ///    walk gives its node no units, `:137-139`), so it takes the special branch and its CHILDREN are
    ///    cloned into the region being built — ⭐⭐ THAT IS THE FLATTENING: the inner local op disappears
    ///    and the operations of whichever of its regions belong to this class are spliced in where it
    ///    stood;
    /// 3. **is it in region `region_num` of its parent** (`:242-245`) — which is how one pass over the
    ///    parent's whole sibling chain fills one region of a cloned `scf.if` and a second pass fills the
    ///    other.
    ///
    /// # ⛔⛔ `:246-256` IS WHAT REBINDS THE UNIT, AND IT IS THE REASON THE PASS WORKS AT ALL
    ///
    /// A local region's block argument IS the unit the region runs on. When the operation about to be
    /// cloned sits directly inside a `uniform.uniformize_regions`, the region it sits in is found by
    /// comparing against the parent's region list, and that region's argument is mapped to the NEW
    /// region's block argument — so the clone reads the new binder. In the vendor's case 4 the nested
    /// op's second region has argument `%arg48` and its `uniform.query_map(map:%285, key:%arg48)`
    /// (`flatten_local_region4.mlir:757`) comes out as
    /// `uniform.query_map(map:%VAL_356, key:%VAL_351)` (`:355`), where `%VAL_351` is the new outer
    /// region's argument. Without this the flattened body would read a binder that no longer exists.
    ///
    /// ⛔ THE COMPARISON HAS TO BE POINTER IDENTITY, NOT `is_in_region_num`. `compute` stamps 0 on the
    /// children of EVERY region of a uniformized op (`:164`), so the field cannot say which local region
    /// an operation came from — see [`Self::in_region_empty`], whose whole defect is that. Here the
    /// region is found by asking which of the parent's bodies actually contains this operation, which is
    /// what `&parent_uniform_op.getRegion(i) == ...getParentRegion()` asks.
    ///
    /// # ⛔ THE REGIONS ARE FILLED FROM THE **ORIGINAL'S** CHILDREN, WHICH IS WHY THE CLONE IS SHALLOW
    ///
    /// `cloneWithoutRegions` (`:257-258`) copies the operation and its operands through `arg_map` and
    /// leaves the regions empty; each region `rn` of the copy is then filled by re-walking the
    /// ORIGINAL's children with `region_num = rn` (`:263-268`). A deep clone would have copied the
    /// children this filter is about to reject, and copied them with the wrong names — see
    /// [`Values::clone_without_regions`].
    ///
    /// ⭐ AND `if (block.empty()) block.erase()` (`:269`) IS AN EMPTY `Vec` HERE. MLIR distinguishes a
    /// region with no block from one holding an empty block, and this island records that distinction as
    /// an empty `else_body` — see [`crate::islands::dataflow_ir::dialects::scf::Op::If::else_body`]. So
    /// leaving the vector empty **is** the erase, and the vendor's own output proves the case is live:
    /// the flattened `scf.if %VAL_355 { .. } {dbgName = "condition__1"}` at
    /// `flatten_local_region4.mlir:353-357` prints no `else`, because the source conditional's `else`
    /// region held nothing belonging to either class.
    ///
    /// # ⚠️ THE TRAILING `replaceUsesWithIf` (`:274-285`) — SCOPED, AND ALREADY MOSTLY DONE
    ///
    /// MLIR's `cloneWithoutRegions(op, mapper)` records `old result -> new result` in the mapper, so an
    /// operation cloned later in the same pass already reads the earlier clone's names through
    /// `arg_map`: `%VAL_355 = arith.cmpi eq, %VAL_353, %VAL_4` (`:352`) reads the CLONE of the
    /// conditional. What this loop adds is the same substitution applied to uses that were emitted into
    /// the new region **without** going through the map, and its predicate bounds it to the region the
    /// clone was inserted into and everything nested below it — climbing from the use's own region and
    /// stopping at the enclosing `dataflow.program_unit`. That is a recursive rewrite of `into`.
    ///
    /// ⭐ THE `std::map` ITERATION ORDER IS UNOBSERVABLE, which is why a `Vec` of pairs replaces it: the
    /// keys are `Operation *` and the order is address order, but every `to` is a value this pass just
    /// minted and can therefore never be another pair's `from`, so the substitutions commute.
    ///
    /// ⚠️ `builder_region` BECOMES `into`, AND THE SHARING AT `:236` IS WHY. The reference passes its
    /// `OpBuilder` **by value** but the two calls differ: `:236-237` hands on the SAME builder, so a
    /// flattened-through operation lands in the caller's region, while `:264-267` builds a new one on
    /// the clone's own region. A `&mut Vec<DfirOp>` reproduces both exactly.
    ///
    /// ⚠️ `values` HAS NO COUNTERPART: MLIR's builder mints result names implicitly. It is threaded
    /// through because [`Values`] is this island's only source of a [`Val`].
    pub fn clone_ops_for_regions(
        &self,
        node: Option<LocalOpNodeId>,
        op_list: &[&DfirOp],
        into: &mut Vec<DfirOp>,
        region_num: RegionNum,
        arg_map: &mut ValueMapping,
        block_arg: Val,
        values: &mut Values,
    ) {
        // `std::map<mlir::Operation *, mlir::Operation *> old_to_new_op_map;` (`:227`) — kept as the
        // RESULT pairs, because `replaceUsesWithIf` reads nothing else out of it.
        let mut old_to_new: Vec<(Val, Val)> = Vec::new();
        let mut node = node;
        // `while (node)` (`:228`).
        while let Some(current) = node {
            // Every branch below ends in `node = node->getNextSibling();`, so it is done once here.
            node = self.next_sibling(current);
            let Some(payload) = self.node(current) else {
                continue;
            };
            // `if (isa<mlir::uniform::YieldOp>(..)) continue;` (`:229-232`).
            if matches!(payload.op, DfirOp::Uniform(uniform::Op::Yield { .. })) {
                continue;
            }
            // `if (std::find(op_list.begin(), op_list.end(), node->getOperation()) == op_list.end())`
            // (`:233-234`) — membership by ADDRESS, as everywhere in this file.
            if !op_list
                .iter()
                .any(|listed| core::ptr::eq(*listed, payload.op))
            {
                // `if (isa<uniform::UniformizeRegionsOp>(..)) cloneOpsForRegions(node->getFirstChild(),
                //     op_list, builder_region, region_num, arg_map, block_arg);` (`:235-238`).
                //
                // ⭐⭐ THE SAME `into` AND THE SAME `region_num`: the nested local operation is dropped
                // and its children are examined as if they were siblings of it, so those of them that
                // belong to this class land where it stood.
                if matches!(
                    payload.op,
                    DfirOp::Uniform(uniform::Op::UniformizeRegions { .. })
                ) {
                    self.clone_ops_for_regions(
                        self.first_child(current),
                        op_list,
                        into,
                        region_num,
                        arg_map,
                        block_arg,
                        values,
                    );
                }
                continue;
            }
            // `if (node->is_in_region_num != region_num) continue;` (`:242-245`).
            if payload.is_in_region_num != region_num {
                continue;
            }
            // `:246-256` — the operation sits directly inside a `uniform.uniformize_regions`, so the
            // unit binder of the region it sits in now stands for the NEW region's block argument.
            //
            // ⛔ THE REGION IS FOUND BY ASKING WHICH BODY HOLDS THIS OPERATION, because
            // `is_in_region_num` cannot say (`:164`). `core::ptr::eq` is
            // `&parent.getRegion(i) == op->getParentRegion()`.
            if let Some(parent) = self.parent_node(current)
                && let Some(DfirOp::Uniform(uniform::Op::UniformizeRegions { regions, .. })) =
                    self.node(parent).map(|parent| parent.op)
            {
                for local in regions {
                    if local
                        .body
                        .iter()
                        .any(|sibling| core::ptr::eq(sibling, payload.op))
                    {
                        arg_map.map(local.arg, block_arg);
                    }
                }
            }
            // `auto new_op_ = builder_region.cloneWithoutRegions(*node->getOperation(), arg_map);`
            // (`:257-258`).
            let mut clone = values.clone_without_regions(payload.op, arg_map);
            // `old_to_new_op_map[node->getOperation()] = new_op_;` (`:259`) — the results, pairwise, in
            // the order the op binds them.
            old_to_new.extend(
                dialects::results(payload.op)
                    .into_iter()
                    .zip(dialects::results(&clone)),
            );
            // `if (new_op_->getNumRegions() > 0) { DT_CHECK(..); for (int rn = 0; ..) { .. } }`
            // (`:260-271`) — the `> 0` guard is an empty loop and the `DT_CHECK` that the copy has as
            // many regions as the original is discharged by [`Values::clone_without_regions`], which
            // empties the regions rather than removing them.
            //
            // ⭐ `emplaceBlock()` THEN `if (block.empty()) block.erase()` (`:265`, `:269`) is a region
            // that stays an empty `Vec` when nothing was cloned into it — the state that prints no
            // `else` at all.
            for (rn, inner) in (0u32..).zip(dialects::regions_mut(&mut clone)) {
                self.clone_ops_for_regions(
                    self.first_child(current),
                    op_list,
                    inner,
                    RegionNum(rn),
                    arg_map,
                    block_arg,
                    values,
                );
            }
            into.push(clone);
        }
        // `for (auto old_new_op : old_to_new_op_map) { .. replaceUsesWithIf(..) }` (`:274-285`) — at or
        // below the region the clones were inserted into, which is `into` and everything nested in it.
        for (old, new) in old_to_new {
            for op in into.iter_mut() {
                replace_uses_at_or_below(op, old, new);
            }
        }
    }
}

impl<'p> Default for FlatteningLocalRegionsTree<'p> {
    /// The reference's only constructor is the default one (entry 106); this forwards to it so the two
    /// cannot disagree.
    fn default() -> Self {
        Self::new()
    }
}

/// Replaces: e246_FlatteningLocalRegionsTree
///
/// **246/384** `~FlatteningLocalRegionsTree() { clear(); }` — `FlatteningLocalRegions.cpp:79` (0L).
///
/// ⛔ THE PORT IS THE ABSENCE OF A `Drop` IMPL, AND LOSING THAT ABSENCE IS A BUILD ERROR. Both of
/// [`FlatteningLocalRegionsTree`]'s fields OWN what `clear()` frees (`:112-127`), so drop glue frees
/// the same nodes and the same map, and `root_ = nullptr` is unobservable on a dying object. The
/// destructuring below is legal only while no `Drop` impl exists (E0509) — entries 086/087's shape.
const _: () = {
    let _the_tree_adds_nothing_to_its_fields = |tree: FlatteningLocalRegionsTree<'_>| {
        let FlatteningLocalRegionsTree { unit_to_ops, base } = tree;
        drop(unit_to_ops);
        drop(base);
    };
};

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



/// `replaceUsesWithIf` WITH THIS PASS'S PREDICATE — one operation and everything nested inside it
/// (`FlatteningLocalRegions.cpp:276-284`).
///
/// The predicate climbs from the use's own region towards the enclosing `dataflow.program_unit` and
/// accepts as soon as it reaches the region the clone was inserted into, so a use is rewritten exactly
/// when it sits at or below that region. Descending instead of climbing asks the same question of a
/// tree that only has downward links.
///
/// ⭐ ONE ENTRY PER USE — [`dialects::replace_uses_of_with`] re-points every operand of one op, and this
/// adds the regions it does not reach.
fn replace_uses_at_or_below(op: &mut DfirOp, from: Val, to: Val) {
    dialects::replace_uses_of_with(op, from, to);
    for region in dialects::regions_mut(op) {
        for inner in region.iter_mut() {
            replace_uses_at_or_below(inner, from, to);
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    /// 🎯 181/384 — AN OPERATION STAMPED WITH THE REGION MEANS IT IS NOT EMPTY.
    ///
    /// `traverseRegion` stamps each child with the index of the parent's region it was found in
    /// (`FlatteningLocalRegions.cpp:145`), so a conditional whose `then` and `else` regions both hold an
    /// operation has both answers `false` — and asking about a region the parent does not have gets
    /// `true`, since no sibling can carry that index.
    #[test]
    fn a_region_holding_an_operation_is_not_empty() {
        let program = a_uniformize_over(
            vec![Val(10)],
            vec![DfirOp::Scf(scf::Op::If {
                cond: Val(282),
                results: Vec::new(),
                body: vec![DfirOp::Arith(arith::Op::Constant {
                    result: Val(285),
                    value: 1,
                })],
                else_body: vec![DfirOp::Arith(arith::Op::Constant {
                    result: Val(286),
                    value: 2,
                })],
                dbg_name: None,
            })],
        );
        let region = the_region(&program);
        let (mut tree, root) = a_rooted_tree(&program);
        tree.traverse_region(&region.body, root, &region.units, RegionNum(0));
        let conditional = tree.first_child(root).expect("the `scf.if`");
        let chain = tree.first_child(conditional);

        assert!(!tree.in_region_empty(RegionNum(0), chain), "the `then` region");
        assert!(!tree.in_region_empty(RegionNum(1), chain), "the `else` region");
        assert!(
            tree.in_region_empty(RegionNum(2), chain),
            "an `scf.if` has two regions, so nothing is stamped 2"
        );
    }

    /// 🎯 181/384 — AN `else` WITH NO BLOCK LEAVES ITS REGION EMPTY, AND A LEAF HAS NO CHAIN AT ALL.
    ///
    /// An `scf.if` always has two regions, and an absent `else` is a region with no operations — the
    /// state [`crate::islands::dataflow_ir::dialects::scf::Op::If::else_body`] records as an empty
    /// `Vec` and MLIR prints by printing no `else` (`dcc/test/PT/issue-236.mlir:65-71`). So the walk
    /// contributes no node stamped 1 and this predicate says so.
    ///
    /// ⚠️ AND `None` IS `getFirstChild()` ON A LEAF (`dcc/src/Analysis/OperationTree.hpp:70`): a chain
    /// that starts nowhere is empty for every region number.
    #[test]
    fn a_region_with_no_block_and_a_leaf_are_both_empty() {
        let program = a_uniformize_over(
            vec![Val(10)],
            vec![DfirOp::Scf(scf::Op::If {
                cond: Val(282),
                results: Vec::new(),
                body: vec![DfirOp::Arith(arith::Op::Constant {
                    result: Val(285),
                    value: 1,
                })],
                else_body: Vec::new(),
                dbg_name: None,
            })],
        );
        let region = the_region(&program);
        let (mut tree, root) = a_rooted_tree(&program);
        tree.traverse_region(&region.body, root, &region.units, RegionNum(0));
        let conditional = tree.first_child(root).expect("the `scf.if`");

        assert!(!tree.in_region_empty(RegionNum(0), tree.first_child(conditional)));
        assert!(
            tree.in_region_empty(RegionNum(1), tree.first_child(conditional)),
            "no `else` block, so no node carries region 1"
        );
        // The `arith.constant` in the `then` region is a leaf.
        let leaf = tree.first_child(conditional).expect("the constant");
        assert_eq!(tree.first_child(leaf), None, "a leaf has no children");
        assert!(tree.in_region_empty(RegionNum(0), tree.first_child(leaf)));
    }

    /// 🎯 181/384 — ⛔ AND ON A UNIFORMIZED OP'S CHILDREN IT IS WRONG, WHICH IS WHY NOTHING CALLS IT.
    ///
    /// `compute` calls back with `false` for **every** region index (`:164`), so all of a
    /// `uniform.uniformize_regions`' children are stamped region 0 whichever local region they came
    /// from. This test builds that stamping directly — entry 247 is the unit that would produce it —
    /// and shows the predicate reporting region 1 empty while it holds an operation. ⭐ THE PLACE THAT
    /// NEEDED THE ANSWER TAKES A DIFFERENT ROUTE: `cloneOpsForRegions` builds the block and erases it
    /// if it is still empty (`:265-269`).
    #[test]
    fn it_reports_a_full_local_region_empty_because_compute_stamps_zero() {
        let ops = vec![
            DfirOp::Arith(arith::Op::Constant {
                result: Val(1),
                value: 1,
            }),
            DfirOp::Arith(arith::Op::Constant {
                result: Val(2),
                value: 2,
            }),
        ];
        // `traverseRegion(uniform_op.getRegion(idx), node, .., false)` for idx 0 AND idx 1 (`:162-164`).
        let mut base = OperationTreeBase::with_root(LocalOpNode::new(&ops[0]));
        let root = base.root();
        for op in &ops {
            base.push_child(root, LocalOpNode::new(op));
        }
        let tree = FlatteningLocalRegionsTree {
            unit_to_ops: UnitToOps::new(),
            base: Some(base),
        };
        let root = LocalOpNodeId(root);

        assert!(!tree.in_region_empty(RegionNum(0), tree.first_child(root)));
        assert!(
            tree.in_region_empty(RegionNum(1), tree.first_child(root)),
            "`ops[1]` came from region 1 and is stamped 0, so the answer is a lie"
        );
    }
    use crate::generated::SyncSignal;
    use crate::islands::dataflow_ir::dialects::{arith, dataflow, scf};
    use crate::islands::dataflow_ir::print;
    use crate::islands::dataflow_ir::ty::ScalarTy;

    /// A COUNTER POSITIONED PAST EVERY VALUE IN A HAND-BUILT PROGRAM.
    ///
    /// ⛔ NOT A CONVENIENCE. In the pass the counter is the program's own — [`Values`] issued the names
    /// the program already holds, so the next one it mints cannot collide with them. A test that starts
    /// a fresh counter at zero re-issues names its own fixture is using, and the collision looks exactly
    /// like a clone reading the right value.
    fn values_past(highest: u32) -> Values {
        let mut values = Values::default();
        while values.issued() <= highest {
            values.mint();
        }
        values
    }

    /// THE VENDOR'S CASE 4, SCALED TO ONE UNIT PER CLASS.
    ///
    /// `dcc/test/Transform/FlatteningLocalRegions/flatten_local_region4.mlir:737-765` is one outer local
    /// region over 32 units (`%arg47`, `:738`) holding a condition chain and, inside a conditional, a
    /// NESTED `uniform.uniformize_regions` that splits those 32 into two classes of 16. Two units stand
    /// in for the two classes here; nothing in `cloneOpsForRegions` counts them.
    ///
    /// ⚠️ THE UNIT AND MAPPING VALUES ARE THE **EXPECTATION'S**, NOT THE INPUT'S. The two files number
    /// the same values differently (the expectation runs them through FileCheck's `VAL_` capture), and a
    /// mapping op's operands are units bound OUTSIDE the local region, so the clone carries them
    /// through unchanged — `[%10 -> %74]` here is `%[[VAL_10]] -> %[[VAL_74]]` at `:354`. Taking them
    /// from the expectation is what lets the pinned output be read against it line for line.
    ///
    /// ⚠️ THE FIXTURE'S `arith.cmpi` OPS ARE AN `arith.constant` AND AN `arith.subi`. This island has no
    /// integer comparison; what the test needs of the second one is that it READS the conditional's
    /// result, which `arith.subi` does — and `arith.subi` is in the vendor's own expectation
    /// (`:379-386`).
    fn a_case_four_program() -> DfirOp {
        let nested = DfirOp::Uniform(uniform::Op::UniformizeRegions {
            regions: vec![
                // THE INPUT'S **SECOND** REGION: `(%arg48 -> ..16 units..){ .. receive_cl1 }`
                // (`:755-760`), which `flatten` emits FIRST (`:344-359`) because the equivalence
                // classes come out in `unit_rep_order` (`:448`), not in region order.
                uniform::LocalRegion {
                    arg: Val(48),
                    units: vec![Val(10)],
                    body: vec![
                        DfirOp::Uniform(uniform::Op::DefImmutableMapping {
                            result: Val(285),
                            pairs: vec![(Val(10), Val(74))],
                        }),
                        DfirOp::Uniform(uniform::Op::QueryMap {
                            result: Val(286),
                            map: Val(285),
                            key: Val(48),
                        }),
                        DfirOp::Dataflow(dataflow::Op::SyncRecv {
                            from: Val(286),
                            signal: SyncSignal::InputToLxsuToLxluToSync,
                        }),
                        DfirOp::Uniform(uniform::Op::Yield {
                            operands: Vec::new(),
                        }),
                    ],
                },
                // AND THE INPUT'S **FIRST**: `(%arg48 -> ..16 units..){ .. receive_cl0 }`
                // (`:749-754`), emitted second (`:360-374`).
                //
                // ⚠️ `Val(49)`, THOUGH THE INPUT PRINTS `%arg48` HERE TOO. A block argument is
                // scoped to its own region in MLIR, so two regions may both name theirs
                // `%arg48`; this island has one flat value space and needs two numbers. The
                // rebinding at `:246-256` is what the test is about, and it is exactly this
                // value that has to disappear from the output.
                uniform::LocalRegion {
                    arg: Val(49),
                    units: vec![Val(11)],
                    body: vec![
                        DfirOp::Uniform(uniform::Op::DefImmutableMapping {
                            result: Val(295),
                            pairs: vec![(Val(11), Val(75))],
                        }),
                        DfirOp::Uniform(uniform::Op::QueryMap {
                            result: Val(296),
                            map: Val(295),
                            key: Val(49),
                        }),
                        DfirOp::Dataflow(dataflow::Op::SyncRecv {
                            from: Val(296),
                            signal: SyncSignal::InputToLxsuToLxluToSync,
                        }),
                        DfirOp::Uniform(uniform::Op::Yield {
                            operands: Vec::new(),
                        }),
                    ],
                },
            ],
            results: Vec::new(),
        });
        a_uniformize_over(
            vec![Val(10), Val(11)],
            vec![
                DfirOp::Arith(arith::Op::Constant {
                    result: Val(282),
                    value: 0,
                }),
                DfirOp::Scf(scf::Op::If {
                    cond: Val(282),
                    results: vec![Val(283)],
                    body: vec![
                        DfirOp::Arith(arith::Op::Constant {
                            result: Val(284),
                            value: 1,
                        }),
                        DfirOp::Scf(scf::Op::Yield {
                            operands: vec![Val(284)],
                        }),
                    ],
                    else_body: vec![DfirOp::Scf(scf::Op::Yield {
                        operands: vec![Val(4)],
                    })],
                    dbg_name: Some("condition__1".to_owned()),
                }),
                DfirOp::Arith(arith::Op::SubI(arith::IntBinary {
                    result: Val(290),
                    lhs: Val(283),
                    rhs: Val(4),
                    ty: ScalarTy::Index,
                })),
                DfirOp::Scf(scf::Op::If {
                    cond: Val(290),
                    results: Vec::new(),
                    body: vec![nested],
                    else_body: Vec::new(),
                    dbg_name: Some("condition__1".to_owned()),
                }),
                DfirOp::Uniform(uniform::Op::Yield {
                    operands: Vec::new(),
                }),
            ],
        )
    }

    /// THE TREE `traverseRegion` AND `compute` BUILD OVER [`a_case_four_program`], SPELLED OUT, WITH
    /// `unit_to_ops` AS THE WALK WOULD LEAVE IT.
    ///
    /// ⚠️ BUILT BY HAND, NOT WALKED: this states the expectation independently of
    /// [`FlatteningLocalRegionsTree::traverse_region`] and [`FlatteningLocalRegionsTree::compute`],
    /// which are the two it is used to check. What they together produce is exactly this:
    /// `is_in_region_num` is the parent's region index for ordinary nesting (`:145`) and **0 for both**
    /// of the nested op's regions, because `compute` passes `false` (`:164`); `units` is the enclosing
    /// region's list, except on the uniformize node itself, which gets none (`:137-139`).
    fn a_case_four_tree(program: &DfirOp) -> (FlatteningLocalRegionsTree<'_>, LocalOpNodeId) {
        let outer = the_region(program);
        let both = [Val(10), Val(11)];
        let mut base = OperationTreeBase::with_root(LocalOpNode::new(program));
        let root = base.root();
        // `new LocalOpNode(&op)`, stamped and attributed, then `parent_node->insertChildNode(new_node)`
        // (`:134-136`, `:140-141`).
        fn push<'p>(
            base: &mut OperationTreeBase<LocalOpNode<'p>>,
            parent: OperationNodeId,
            op: &'p DfirOp,
            rn: u32,
            units: &[Val],
        ) -> OperationNodeId {
            let mut node = LocalOpNode::new(op);
            node.is_in_region_num = RegionNum(rn);
            node.units = units.to_vec();
            base.push_child(parent, node)
        }

        // `%282 = arith.constant`.
        push(&mut base, root, &outer.body[0], 0, &both);
        // The condition-forwarding `scf.if` and its three inner operations, two in region 0 and one in
        // region 1.
        let forwarding = push(&mut base, root, &outer.body[1], 0, &both);
        let arms = dialects::regions(&outer.body[1]);
        push(&mut base, forwarding, &arms[0][0], 0, &both);
        push(&mut base, forwarding, &arms[0][1], 0, &both);
        push(&mut base, forwarding, &arms[1][0], 1, &both);
        // `%290 = arith.subi %283, %4`.
        push(&mut base, root, &outer.body[2], 0, &both);
        // The guarding `scf.if`, its nested local operation, and that operation's two regions' worth of
        // children — all stamped region 0, and each attributed to its own class's unit only.
        let guard = push(&mut base, root, &outer.body[3], 0, &both);
        let nested = &dialects::regions(&outer.body[3])[0][0];
        let nested_node = push(&mut base, guard, nested, 0, &[]);
        let inner = the_regions(nested);
        for (region, unit) in inner.iter().zip([Val(10), Val(11)]) {
            for op in &region.body {
                push(&mut base, nested_node, op, 0, &[unit]);
            }
        }
        // `uniform.yield`.
        push(&mut base, root, &outer.body[4], 0, &both);

        let mut tree = FlatteningLocalRegionsTree {
            unit_to_ops: UnitToOps::new(),
            base: Some(base),
        };
        // `unit_to_ops[u].push_back(&op)` in walk order (`:142`) — the nested operations before the
        // outer terminator, because the walk reaches them there.
        for unit in both {
            for op in [
                &outer.body[0],
                &outer.body[1],
                &arms[0][0],
                &arms[0][1],
                &arms[1][0],
                &outer.body[2],
                &outer.body[3],
            ] {
                tree.unit_to_ops.push_op(unit, op);
            }
        }
        for (region, unit) in inner.iter().zip([Val(10), Val(11)]) {
            for op in &region.body {
                tree.unit_to_ops.push_op(unit, op);
            }
        }
        for unit in both {
            tree.unit_to_ops.push_op(unit, &outer.body[4]);
        }
        (tree, LocalOpNodeId(root))
    }

    /// 🎯 182/384 — THE SECOND CLASS GETS THE **OTHER** LOCAL REGION'S BODY, AND ITS OWN BLOCK ARGUMENT.
    ///
    /// The same tree, the same outer condition chain, one operation different: this class's
    /// representative was attributed the nested op's SECOND region, so `:233-234` selects those three
    /// operations and rejects the first region's. That is the whole contrast the vendor's expectation
    /// draws between its two flattened regions — `..receive_cl1` at `flatten_local_region4.mlir:356`
    /// against `..receive_cl0` at `:372`, over identical condition chains at `:345-353` and `:361-369`.
    ///
    /// ⭐ AND `key:` IS **THIS** REGION'S ARGUMENT: `%VAL_357 = uniform.query_map(map:.., key:%VAL_351)`
    /// (`:355`) against `%VAL_364 = uniform.query_map(map:.., key:%VAL_358)` (`:371`). `flatten` mints one
    /// block argument per new region and calls this function once per region (`:431-449`), and `:246-256`
    /// maps whichever local region actually holds the operation to whichever argument it was handed.
    #[test]
    fn the_second_class_selects_the_other_local_region() {
        let program = a_case_four_program();
        let (tree, root) = a_case_four_tree(&program);
        let class = tree
            .unit_to_ops
            .entries()
            .iter()
            .find(|(unit, _)| *unit == Val(11))
            .map(|(_, ops)| ops.clone())
            .expect("unit %11 is the second class's representative");

        let mut values = values_past(300);
        let block_arg = values.mint();
        let mut arg_map = ValueMapping::new();
        arg_map.map(the_region(&program).arg, block_arg);
        let mut into = Vec::new();

        tree.clone_ops_for_regions(
            tree.first_child(root),
            &class,
            &mut into,
            RegionNum(0),
            &mut arg_map,
            block_arg,
            &mut values,
        );

        let mut got = String::new();
        for op in &into {
            print::emit(&mut got, op, 0);
        }
        assert_eq!(got, concat!(
            // The same condition chain the first class got, cloned again for this region
            // (`flatten_local_region4.mlir:361-369` against `:345-353`).
            "%302 = arith.constant 0 : index\n",
            "%303 = scf.if %302 -> (index) {\n",
            "  %304 = arith.constant 1 : index\n",
            "  scf.yield %304 : index\n",
            "} else {\n",
            "  scf.yield %4 : index\n",
            "} {dbgName = \"condition__1\"}\n",
            "%305 = arith.subi %303, %4 : index\n",
            "scf.if %305 {\n",
            // ⭐ `%11 -> %75`, NOT `%10 -> %74` — region ONE's mapping op, so `:233-234` really did
            // select this class's list and reject the other region's (`:370` against `:354`).
            "  %306 = uniform.def_immutable_mapping([%11 -> %75]):index\n",
            // ⭐ AND `key:%301` AGAIN — this call was handed its own region's argument, exactly as
            // `:371` reads `%VAL_358` where `:355` reads `%VAL_351`.
            "  %307 = uniform.query_map(map:%306, key:%301) : index\n",
            "  dataflow.sync_recv %307 {dbgName = \"input-lxsu-lxlu-sync\"} : index\n",
            "} {dbgName = \"condition__1\"}\n",
        ));
    }

    /// The regions of a `uniform.uniformize_regions`, for reaching a nested one's unit lists.
    fn the_regions(op: &DfirOp) -> &[uniform::LocalRegion] {
        match op {
            DfirOp::Uniform(uniform::Op::UniformizeRegions { regions, .. }) => regions,
            _ => unreachable!("only ever called on the nested local operation"),
        }
    }

    /// 🎯 182/384 — THE VENDOR'S CASE 4: ONE CLASS'S BODY, WITH THE NESTED LOCAL REGION FLATTENED AWAY
    /// AND ITS UNIT BINDER REBOUND TO THE NEW REGION'S ARGUMENT.
    ///
    /// This is `flatten_local_region4.mlir:344-358` — the first region of the flattened op — built from
    /// the input at `:737-765`. Four things have to come out right and each is a different branch:
    ///
    /// * the outer condition chain is cloned, and `%VAL_355 = arith.cmpi eq, %VAL_353, ..` (`:352`)
    ///   reads the CLONE of the conditional, not the original (`arg_map`, `:257-258`);
    /// * the conditional's regions are refilled from the ORIGINAL's children by `is_in_region_num`, so
    ///   the `then` region gets two operations and the `else` region one (`:263-268`, expectation
    ///   `:346-351`);
    /// * the nested `uniform.uniformize_regions` is GONE and the three operations of the region
    ///   belonging to this class stand where it was (`:235-238`, expectation `:354-356`);
    /// * `uniform.query_map(map:%285, key:%arg48)` becomes `key:%VAL_351` — the new region's block
    ///   argument, mapped by `:246-256` because the operation's parent is a local op.
    ///
    /// ⛔ AND THE GUARDING `scf.if` PRINTS NO `else`: its source `else` region held nothing, so the
    /// clone's stays an empty `Vec` — `if (block.empty()) block.erase()` (`:269`), expectation `:357`.
    #[test]
    fn the_vendor_case_four_first_region_is_one_class_with_the_nesting_flattened() {
        let program = a_case_four_program();
        let (tree, root) = a_case_four_tree(&program);
        let class = tree
            .unit_to_ops
            .entries()
            .iter()
            .find(|(unit, _)| *unit == Val(10))
            .map(|(_, ops)| ops.clone())
            .expect("unit %10 is the first class's representative");

        // `flatten`'s per-region setup: a fresh mapping, the new region's block argument, and the OLD
        // region's argument mapped to it (`:444-446`).
        let mut values = values_past(300);
        let block_arg = values.mint();
        let mut arg_map = ValueMapping::new();
        arg_map.map(the_region(&program).arg, block_arg);
        let mut into = Vec::new();

        tree.clone_ops_for_regions(
            tree.first_child(root),
            &class,
            &mut into,
            RegionNum(0),
            &mut arg_map,
            block_arg,
            &mut values,
        );

        let mut got = String::new();
        for op in &into {
            print::emit(&mut got, op, 0);
        }
        assert_eq!(
            got,
            concat!(
                // `%VAL_352 = arith.cmpi eq, %VAL_310, %VAL_8 : index` (`:345`).
                "%302 = arith.constant 0 : index\n",
                // `:346-351` — the result is bound, the `then` region gets both of its operations and
                // the `else` region its one, and the debug name prints after the regions.
                "%303 = scf.if %302 -> (index) {\n",
                "  %304 = arith.constant 1 : index\n",
                "  scf.yield %304 : index\n",
                "} else {\n",
                "  scf.yield %4 : index\n",
                "} {dbgName = \"condition__1\"}\n",
                // ⭐ `%VAL_355 = arith.cmpi eq, %VAL_353, %VAL_4 : i1` (`:352`) — the first operand is
                // the CLONE of the conditional, the second a value from outside the region and so
                // unmapped.
                "%305 = arith.subi %303, %4 : index\n",
                // ⛔ NO `else`: the source conditional's `else` region held nothing for this class, so
                // `if (block.empty()) block.erase()` (`:269`) leaves it empty — `:353-357`.
                "scf.if %305 {\n",
                // ⭐⭐ THE NESTED LOCAL OPERATION IS GONE AND ITS REGION-0 BODY STANDS HERE (`:354-356`),
                // with `key:` rebound from `%arg48` to the new region's block argument (`:246-256`).
                "  %306 = uniform.def_immutable_mapping([%10 -> %74]):index\n",
                "  %307 = uniform.query_map(map:%306, key:%301) : index\n",
                "  dataflow.sync_recv %307 {dbgName = \"input-lxsu-lxlu-sync\"} : index\n",
                "} {dbgName = \"condition__1\"}\n",
                // ⚠️ AND NO `uniform.yield`: `flatten` builds the new region's terminator itself
                // (`:450`), which is `:358` of the expectation.
            )
        );
    }

    /// 🎯 247/384 — THE VENDOR'S `@diff_groups`: A NESTED LOCAL OP RE-DERIVES THE UNIT LIST PER REGION,
    /// AND THAT IS WHAT SPLITS TWO UNITS THAT WERE IN ONE REGION.
    ///
    /// The outer region runs on `%0, %2` together; its nested `uniform.uniformize_regions` gives `%0`
    /// one region and `%2` another (`flatten_local_region.mlir:90-119`). `compute` walks each with its
    /// OWN list, so the two units are attributed DIFFERENT operation objects and
    /// [`FlatteningLocalRegionsTree::partition_units`] returns two classes — which is why the vendor's
    /// four units come out as four regions and not two (`CHECK-SENT-IR` `:19`, `:31`, `:43`, `:55`).
    /// ⭐ AND THE NESTED OP'S OWN NODE CARRIES NO UNITS, since `traverseRegion` skips the push for it.
    #[test]
    fn a_nested_local_op_splits_the_units_its_parent_shared() {
        let nested = DfirOp::Uniform(uniform::Op::UniformizeRegions {
            regions: vec![
                uniform::LocalRegion {
                    arg: Val(48),
                    units: vec![Val(0)],
                    body: vec![
                        DfirOp::Arith(arith::Op::Constant {
                            result: Val(9),
                            value: 0,
                        }),
                        DfirOp::Uniform(uniform::Op::Yield {
                            operands: Vec::new(),
                        }),
                    ],
                },
                uniform::LocalRegion {
                    arg: Val(48),
                    units: vec![Val(2)],
                    body: vec![
                        DfirOp::Arith(arith::Op::Constant {
                            result: Val(19),
                            value: 0,
                        }),
                        DfirOp::Uniform(uniform::Op::Yield {
                            operands: Vec::new(),
                        }),
                    ],
                },
            ],
            results: Vec::new(),
        });
        let program = a_uniformize_over(vec![Val(0), Val(2)], vec![nested]);
        let region = the_region(&program);
        let (mut tree, root) = a_rooted_tree(&program);

        tree.traverse_region(&region.body, root, &region.units, RegionNum(0));

        let inner_node = tree.first_child(root).expect("the nested local op");
        assert!(
            tree.node(inner_node).expect("its payload").units.is_empty(),
            "`traverseRegion` pushes no unit for a `uniform.uniformize_regions` (`:137-139`)"
        );
        assert_eq!(
            tree.partition_units(),
            vec![(Val(0), vec![Val(0)]), (Val(2), vec![Val(2)])],
            "each nested region attributed its own operations to its own unit"
        );
        let attributed: Vec<usize> = tree
            .unit_to_ops
            .entries()
            .iter()
            .map(|(_, ops)| ops.len())
            .collect();
        assert_eq!(
            attributed,
            vec![2, 2],
            "the constant and the `uniform.yield` of one region each — `compute` walked both (`:162-165`)"
        );
        // `traverseRegion(.., false)` for BOTH regions (`:164`), so nothing is stamped region 1.
        assert!(
            tree.in_region_empty(RegionNum(1), tree.first_child(inner_node)),
            "every child of a uniformized op is stamped region 0"
        );
    }

    /// A `uniform.uniformize_regions` OVER ONE REGION — the shape `compute` hands to
    /// [`FlatteningLocalRegionsTree::traverse_region`].
    ///
    /// `flatten` roots the tree at the operation itself (`FlatteningLocalRegions.cpp:392-393`) and
    /// `compute` then walks each of its regions with that region's own unit list (`:162-165`), so a
    /// test of the walk needs the op, not just the ops inside it.
    fn a_uniformize_over(units: Vec<Val>, body: Vec<DfirOp>) -> DfirOp {
        DfirOp::Uniform(uniform::Op::UniformizeRegions {
            regions: vec![uniform::LocalRegion {
                arg: Val(47),
                units,
                body,
            }],
            results: Vec::new(),
        })
    }

    /// A TREE WITH A ROOT AND NO CHILDREN — `root_ = new LocalOpNode(&op_)` and nothing else yet
    /// (`FlatteningLocalRegions.cpp:392-393`), which is the state `compute` is called in.
    fn a_rooted_tree<'p>(op: &'p DfirOp) -> (FlatteningLocalRegionsTree<'p>, LocalOpNodeId) {
        let base = OperationTreeBase::with_root(LocalOpNode::new(op));
        let root = LocalOpNodeId(base.root());
        let tree = FlatteningLocalRegionsTree {
            unit_to_ops: UnitToOps::new(),
            base: Some(base),
        };
        (tree, root)
    }

    /// The one region of [`a_uniformize_over`], for taking its body and units back out.
    fn the_region(op: &DfirOp) -> &uniform::LocalRegion {
        match op {
            DfirOp::Uniform(uniform::Op::UniformizeRegions { regions, .. }) => &regions[0],
            _ => unreachable!("built by `a_uniformize_over` one line above every caller"),
        }
    }

    /// 🎯 180/384 — EVERY UNIT OF A LOCAL REGION IS ATTRIBUTED EVERY OPERATION IN IT, NESTING INCLUDED.
    ///
    /// `traverseRegion` passes the SAME unit list into each nested region (`:145`), so the enclosing
    /// conditional does not narrow attribution. ⭐ THAT IS WHAT MAKES THE VENDOR'S CASE 4 DECIDABLE: all
    /// 32 units run one `arith.cmpi`/`scf.if` chain in the outer region
    /// (`dcc/test/Transform/FlatteningLocalRegions/flatten_local_region4.mlir:738-747`) and differ only
    /// inside the nested local regions, so the outer walk contributes an identical list to every unit
    /// and [`FlatteningLocalRegionsTree::partition_units`] has nothing to split on there.
    ///
    /// ⚠️ THE FIXTURE'S `arith.cmpi` IS AN `arith.constant` HERE. This island has no integer
    /// comparison, and the walk is blind to what an operation IS except for the single
    /// `isa<uniform::UniformizeRegionsOp>` at `:137` — so the substitution cannot change the answer.
    #[test]
    fn the_walk_attributes_nested_operations_to_every_unit_of_the_region() {
        let program = a_uniformize_over(
            vec![Val(10), Val(14)],
            vec![
                DfirOp::Arith(arith::Op::Constant {
                    result: Val(282),
                    value: 0,
                }),
                DfirOp::Scf(scf::Op::If {
                    cond: Val(282),
                    results: vec![Val(283)],
                    body: vec![DfirOp::Scf(scf::Op::Yield {
                        operands: vec![Val(285)],
                    })],
                    else_body: vec![DfirOp::Scf(scf::Op::Yield {
                        operands: vec![Val(4)],
                    })],
                    dbg_name: None,
                }),
                DfirOp::Uniform(uniform::Op::Yield {
                    operands: Vec::new(),
                }),
            ],
        );
        let region = the_region(&program);
        let (mut tree, root) = a_rooted_tree(&program);

        // `traverseRegion(uniform_op.getRegion(idx), node, units_for_each_region.at(idx), false)`
        // (`:163-164`) — `false` is region 0.
        tree.traverse_region(&region.body, root, &region.units, RegionNum(0));

        // Four operations reach the map: the constant, the conditional, and BOTH of its yields. The
        // `uniform.yield` reaches it too — the walk does not filter terminators, `cloneOpsForRegions`
        // does (`:229`).
        let expected: Vec<&DfirOp> = vec![
            &region.body[0],
            &region.body[1],
            &the_regions_of(&region.body[1])[0][0],
            &the_regions_of(&region.body[1])[1][0],
            &region.body[2],
        ];
        for (unit, ops) in tree.unit_to_ops.entries() {
            assert!(
                runs_the_same_operations(ops, &expected),
                "unit {unit:?} was walked over a different operation list"
            );
        }
        assert_eq!(
            tree.unit_to_ops
                .entries()
                .iter()
                .map(|(unit, _)| *unit)
                .collect::<Vec<_>>(),
            vec![Val(10), Val(14)],
            "`llvm::MapVector` keeps the units in the order the walk pushed them"
        );
    }

    /// The regions of one op, for naming a nested operation in a test expectation.
    fn the_regions_of(op: &DfirOp) -> Vec<&[DfirOp]> {
        dialects::regions(op)
    }

    /// 🎯 180/384 — A NODE CARRIES THE INDEX OF THE **PARENT'S** REGION IT WAS FOUND IN.
    ///
    /// `traverseRegion(op.getRegion(region_num), new_node, units, region_num)` (`:145`) stamps the loop
    /// counter, so the `then` region's operations are region 0 and the `else` region's are region 1 —
    /// the order [`dialects::regions`] indexes an `scf.if` in, and the distinction
    /// [`FlatteningLocalRegionsTree::clone_ops_for_regions`] filters on at `:242`.
    ///
    /// ⭐ AND THE TOP CALL STAMPS 0, because `compute` passes `false` (`:164`).
    #[test]
    fn a_nested_operation_carries_the_index_of_its_parents_region() {
        let program = a_uniformize_over(
            vec![Val(10)],
            vec![DfirOp::Scf(scf::Op::If {
                cond: Val(282),
                results: Vec::new(),
                body: vec![DfirOp::Arith(arith::Op::Constant {
                    result: Val(285),
                    value: 1,
                })],
                else_body: vec![DfirOp::Arith(arith::Op::Constant {
                    result: Val(286),
                    value: 2,
                })],
                dbg_name: None,
            })],
        );
        let region = the_region(&program);
        let (mut tree, root) = a_rooted_tree(&program);

        tree.traverse_region(&region.body, root, &region.units, RegionNum(0));

        let conditional = tree.first_child(root).expect("the `scf.if` is the root's child");
        assert_eq!(
            tree.node(conditional).map(|node| node.is_in_region_num),
            Some(RegionNum(0)),
            "the outer operation is stamped by `compute`'s `false` (`:164`)"
        );
        let then_op = tree
            .first_child(conditional)
            .expect("the `then` region's constant");
        let else_op = tree
            .next_sibling(then_op)
            .expect("the `else` region's constant, in the same sibling chain");
        assert_eq!(
            tree.node(then_op).map(|node| node.is_in_region_num),
            Some(RegionNum(0))
        );
        assert_eq!(
            tree.node(else_op).map(|node| node.is_in_region_num),
            Some(RegionNum(1)),
            "`getRegions()[1]` is the `else` region (`:802-810` of `dialects/mod.rs`)"
        );
        // ⛔ BOTH REGIONS' OPERATIONS ARE THE SAME UNIT'S — one sibling chain, two region numbers.
        assert_eq!(
            tree.node(else_op).map(|node| node.units.as_slice()),
            Some([Val(10)].as_slice())
        );
    }

    /// ONE OPERATION PER LINE OF A REGION — the ops the nodes below stand for.
    ///
    /// ⚠️ WHICH ops they are is immaterial to entries 101-108: the tree is opaque in the operation
    /// until entry 180 asks `isa<uniform::UniformizeRegionsOp>`. What matters is that they are
    /// DISTINCT objects, because that is what entry 108 compares.
    fn a_region() -> Vec<DfirOp> {
        vec![
            DfirOp::Scf(scf::Op::If {
                cond: Val(30),
                results: Vec::new(),
                body: Vec::new(),
                else_body: Vec::new(),
                dbg_name: None,
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
            results: Vec::new(),
            body: Vec::new(),
            else_body: Vec::new(),
            dbg_name: None,
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

    /// 🎯 179/384 — `clear()` PUTS BACK THE STATE `flatten` STARTS FROM.
    ///
    /// The reference frees every node and then sets `root_ = nullptr` and `unit_to_ops.clear()`
    /// (`FlatteningLocalRegions.cpp:112-127`); what a caller can observe afterwards is a tree with no
    /// root and no attributed operations — the same state as [`FlatteningLocalRegionsTree::new`], which
    /// is why `flatten` can call it before it decides anything (`:385`).
    #[test]
    fn clearing_a_built_tree_leaves_no_root_and_no_operations() {
        let ops = a_region();
        let (mut tree, ids) = a_flattening_tree(&ops);
        tree.unit_to_ops.push_op(Val(9), &ops[1]);
        tree.unit_to_ops.push_op(Val(7), &ops[2]);
        assert!(tree.root().is_some(), "the tree was built with a root");
        assert_eq!(tree.first_child(ids[0]), Some(ids[1]), "and with children");
        assert_eq!(tree.unit_to_ops.entries().len(), 2);

        tree.clear();

        assert_eq!(tree.root(), None, "`root_ = nullptr` (`:124`)");
        assert!(
            tree.unit_to_ops.entries().is_empty(),
            "`unit_to_ops.clear()` (`:126`)"
        );
    }

    /// 🎯 179/384 — AND ON A TREE THAT NEVER HAD A ROOT IT DOES NOTHING, WHICH IS THE THIRD BRANCH.
    ///
    /// `empty()` is `!root_ || !root_->getFirstChild()` (`dcc/src/Analysis/OperationTree.hpp:214`), so
    /// a rootless tree takes the `else if (root_)` branch and frees nothing. ⭐ THE PASS REACHES THIS:
    /// `flatten` clears, returns early when the operation is not a `uniform.uniformize_regions`
    /// (`:386-387`), and its destructor then clears the same rootless tree a second time (`:79`).
    #[test]
    fn clearing_a_rootless_tree_is_the_state_it_was_already_in() {
        let mut tree = FlatteningLocalRegionsTree::new();
        tree.clear();
        assert_eq!(tree.root(), None);
        assert!(tree.unit_to_ops.entries().is_empty());
    }
}




// ⛔ RE-CREATED ANCHORS. These units' `crustify:todo:` markers were deleted without a
// `/// Replaces:` ever appearing, which removed them from every later schedule and let the
// driver report the campaign DONE. Outstanding work is now computed from UNITS.tsv.
// crustify:todo: e287_flatten
// crustify:todo: e305_runOnOperation
