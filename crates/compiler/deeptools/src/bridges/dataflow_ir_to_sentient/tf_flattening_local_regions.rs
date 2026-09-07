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

use crate::islands::dataflow_ir::dialects::{Op as DfirOp, Val};

/// THE IDENTITY OF ONE NODE IN THE FLATTENING TREE.
///
/// # ⭐ AN INDEX WHERE THE REFERENCE HAS A `LocalOpNode *`
///
/// The reference's tree is a heap of individually `new`ed nodes wired by four raw pointers, freed by
/// a post-order walk in `clear()` (`FlatteningLocalRegions.cpp:112-127`). A node cannot hold a
/// `&LocalOpNode` to its sibling and be built by the same walk that appends to the storage, so the
/// links are identities into storage the tree owns — which is also what makes the three accessors
/// below total instead of unsafe.
///
/// ⚠️ THE STORAGE THAT MINTS THESE ARRIVES WITH `FlatteningLocalRegionsTree` (entries 106, 107 and
/// 246) AND ITS `clear()` (entry 179); the linking is `insertChildNode`, a base-class member of
/// `dcc/src/Analysis/OperationTree.hpp`, which contributes nothing to the 384 and so is written by
/// whichever unit first needs it (entry 180, `traverseRegion`). Entries 101-104 are one node's
/// constructor and its three link reads, and that is all this file claims today.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct LocalOpNodeId(pub u32);

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
/// # ⛔⛔ NEITHER `Clone` NOR `PartialEq`, AND BOTH ARE THE REFERENCE'S OWN DECISIONS
///
/// `OperationNode` deletes its copy constructors (`dcc/src/Analysis/OperationTree.hpp:30-31`): a
/// second node wrapping the same operation with the same links would appear twice in one sibling
/// chain and be deleted twice by `clear()`. And its `operator==` is `this == &n` (`:34`) — POINTER
/// identity, not structural equality, which is also how `cloneOpsForRegions` recognises a node's
/// operation (`std::find` over an `Operation *` list, `:230`). A derived `Clone` would reintroduce
/// the copy the reference forbids and a derived `PartialEq` would answer a different question from
/// the one the reference asks, so this type derives only `Debug`.
///
/// ⚠️ `op` IS ANY `DfirOp` BECAUSE ENTRIES 101-104 NEVER LOOK AT IT. The ops this tree actually holds
/// are `uniform.uniformize_regions` and `uniform.yield` and their contents, and the `uniform` dialect
/// is not in `islands/dataflow_ir/` yet — the first unit that asks an op WHICH op it is
/// (`isa<uniform::UniformizeRegionsOp>` in entries 180, 247, 182 and 287) is the one that has to add
/// it, per the campaign brief. A constructor and three link reads are opaque in the operation.
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
    /// `parent_operation_`.
    parent: Option<LocalOpNodeId>,
    /// `first_child_`.
    first_child: Option<LocalOpNodeId>,
    /// `next_sibling_`.
    next_sibling: Option<LocalOpNodeId>,
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
    /// ⛔ AND `friend class FlatteningLocalRegionsTree` IS THE ONE THING WITH NO ANALOGUE: it lets the
    /// tree reach `is_in_region_num` and the base's links directly. Here the links are private to
    /// this module for the same reason and the two public members are public in the reference too.
    #[must_use]
    pub const fn new(op: &'p DfirOp) -> Self {
        Self {
            op,
            units: Vec::new(),
            is_in_region_num: RegionNum(0),
            parent: None,
            first_child: None,
            next_sibling: None,
        }
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
    /// ⭐ SO THE PORT OF THESE THREE FUNCTIONS IS THAT THE CAST HAS NOTHING LEFT TO DO. One node
    /// type, one storage, one [`LocalOpNodeId`]: the narrow type is what the link already is, the
    /// assertion is discharged at compile time, and `getParentNode`, `getFirstChild` and
    /// `getNextSibling` are the field reads the base performs. ⚠️ The three sibling functions this
    /// class also declares are NOT here: `getPrevSibling` is entry 105 and computed rather than
    /// stored, and `walk` is a template the tree instantiates.
    ///
    /// ⛔ AND `nullptr` IS `None`, NOT A ROOT. `OperationTreeBase` gives the forest a synthetic root
    /// whose parent is null, so an absent parent is what identifies it — `getDepth`/`isOutermost`
    /// count on exactly that (`OperationTree.hpp:62-69`).
    #[must_use]
    pub const fn parent_node(&self) -> Option<LocalOpNodeId> {
        self.parent
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
    /// `:232-234`), the post-order delete in `clear()` (entry 179) — stops on it.
    #[must_use]
    pub const fn first_child(&self) -> Option<LocalOpNodeId> {
        self.first_child
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
    /// THE ONE THE TRANSFORM ACTUALLY LOOPS ON: `for (node; node; node = node->getNextSibling())` is
    /// how `cloneOpsForRegions` (`FlatteningLocalRegions.cpp:224-...`) and `inRegionEmpty` (`:214-221`)
    /// traverse a region, and `getPrevSibling` and `getLastChild` are both derived from it by walking
    /// forward from the parent's first child rather than being stored.
    ///
    /// ⛔ `None` IS THE END OF THE REGION, which is why those loops need no count.
    #[must_use]
    pub const fn next_sibling(&self) -> Option<LocalOpNodeId> {
        self.next_sibling
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::dialects::{arith, scf};

    /// 🎯 101/384 — A FRESH NODE IS DETACHED, CARRIES NO UNITS AND IS MARKED REGION 0.
    ///
    /// The constructor forwards the operation and runs the member initialisers, and nothing else:
    /// `traverseRegion` sets the region number on the line after the `new`
    /// (`FlatteningLocalRegions.cpp:134-135`) and pushes the units four lines later, so this is the
    /// state everything else starts from.
    #[test]
    fn a_fresh_node_is_detached() {
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
        assert_eq!(node.parent_node(), None);
        assert_eq!(node.first_child(), None);
        assert_eq!(node.next_sibling(), None);
        assert!(node.units.is_empty(), "`units` is default-constructed");
        assert_eq!(
            node.is_in_region_num,
            RegionNum(0),
            "`int is_in_region_num = 0`"
        );
    }

    /// THREE LINKED NODES: a parent whose region holds two operations in order.
    ///
    /// ```text
    ///   node 0  ── first_child ──►  node 1  ── next_sibling ──►  node 2
    ///                                  └── parent ──► node 0 ──┘
    /// ```
    ///
    /// ⚠️ WIRED FIELD BY FIELD BECAUSE `insertChildNode` IS NOT PORTED — it is a base-class member of
    /// `dcc/src/Analysis/OperationTree.hpp`, outside the 384, and arrives with entry 180.
    fn a_linked_region<'p>(ops: &'p [DfirOp]) -> Vec<LocalOpNode<'p>> {
        let mut nodes: Vec<LocalOpNode<'p>> = ops.iter().map(LocalOpNode::new).collect();
        nodes[0].first_child = Some(LocalOpNodeId(1));
        nodes[1].parent = Some(LocalOpNodeId(0));
        nodes[1].next_sibling = Some(LocalOpNodeId(2));
        nodes[1].is_in_region_num = RegionNum(0);
        nodes[2].parent = Some(LocalOpNodeId(0));
        nodes[2].is_in_region_num = RegionNum(0);
        nodes
    }

    /// THE THREE OPS THE NODES ABOVE STAND FOR.
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
        ]
    }

    /// 🎯 102/384 — BOTH OPERATIONS OF A REGION NAME THE SAME PARENT, AND THE PARENT NAMES NONE.
    ///
    /// An absent parent is what identifies the synthetic root the tree collects its forest under
    /// (`dcc/src/Analysis/OperationTree.hpp:62-69`, `getDepth`/`isOutermost`).
    #[test]
    fn the_parent_of_a_region_is_the_op_that_owns_it() {
        let ops = a_region();
        let nodes = a_linked_region(&ops);
        assert_eq!(nodes[1].parent_node(), Some(LocalOpNodeId(0)));
        assert_eq!(nodes[2].parent_node(), Some(LocalOpNodeId(0)));
        assert_eq!(nodes[0].parent_node(), None, "the root has no parent");
    }

    /// 🎯 103/384 — THE FIRST CHILD IS THE FIRST OPERATION IN SYNTACTIC ORDER, AND A LEAF HAS NONE.
    #[test]
    fn the_first_child_is_the_first_operation_in_the_region() {
        let ops = a_region();
        let nodes = a_linked_region(&ops);
        assert_eq!(nodes[0].first_child(), Some(LocalOpNodeId(1)));
        assert_eq!(
            nodes[1].first_child(),
            None,
            "`isLeaf()` is `getFirstChild() == nullptr`"
        );
        assert_eq!(nodes[2].first_child(), None);
    }

    /// 🎯 104/384 — THE SIBLING CHAIN WALKS THE REGION IN ORDER AND ENDS ON `None`.
    ///
    /// This is the loop every traversal in the file runs — `for (; node; node = getNextSibling())`
    /// (`FlatteningLocalRegions.cpp:214-220`, `:224` onwards) — so the test walks it the same way and
    /// counts what it visits, which is what tells the loop it may stop.
    #[test]
    fn the_sibling_chain_walks_the_region_in_order() {
        let ops = a_region();
        let nodes = a_linked_region(&ops);
        let mut visited = Vec::new();
        let mut cursor = nodes[0].first_child();
        while let Some(LocalOpNodeId(id)) = cursor {
            visited.push(id);
            cursor = nodes[id as usize].next_sibling();
        }
        assert_eq!(visited, vec![1, 2], "both operations, in syntactic order");
        assert_eq!(
            nodes[2].next_sibling(),
            None,
            "the end of the region needs no count"
        );
    }
}
