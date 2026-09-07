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

//! `LoopMaskTree.cpp` — 17 of bridge 2's 384 functions (dependency level(s) [0, 1, 2, 3]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e077_walk` | 077/384 | 2 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:132` |
//! | `e078_findNodeFromOp` | 078/384 | 4 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:167` |
//! | `e079_OperationNode` | 079/384 | 0 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:32` |
//! | `e080_LoopMaskNode` | 080/384 | 0 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:34` |
//! | `e081_getParentNode` | 081/384 | 2 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:36` |
//! | `e082_getFirstChild` | 082/384 | 2 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:39` |
//! | `e083_getNextSibling` | 083/384 | 2 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:42` |
//! | `e084_getPrevSibling` | 084/384 | 2 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:45` |
//! | `e085_getLastChild` | 085/384 | 2 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:48` |
//! | `e086_MaskNode` | 086/384 | 0 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:74` |
//! | `e087_LMTLoopNode` | 087/384 | 0 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:94` |
//! | `e088_getRoot` | 088/384 | 2 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:109` |
//! | `e171_isMaskEquivalentToNode` | 171/384 | 4 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:123` |
//! | `e236_addMaskNode` | 236/384 | 16 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:136` |
//! | `e237_updateNode` | 237/384 | 10 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:155` |
//! | `e238_computeLoops` | 238/384 | 18 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:173` |
//! | `e281_OperationTreeBase` | 281/384 | 2 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:105` |
//!
//! Original files homed here: `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp`, `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp`

use std::collections::VecDeque;

use super::vc_vector_operands::OpId;

// ══════════════════════════════════════════════════════════════════════════════════════════════════
// THE BASE LAYER — `mlir::OperationNode` AND `mlir::OperationTreeBase`
// (`dcc/src/Analysis/OperationTree.hpp`, `dcc/src/Analysis/OperationTree.cpp`)
// ══════════════════════════════════════════════════════════════════════════════════════════════════
//
// ⚠️ NOT A SCHEDULED UNIT, DELIBERATELY WRITTEN HERE. `src/Analysis/OperationTree.hpp` is outside
// bridge 2's 490-definition span, so no entry number can be spent on it — and entries 081-085 are
// each ONE LINE that delegates to it (`static_cast<LoopMaskNode *>(OperationNode::getFirstChild())`).
// There is no way to port a delegation without the thing it delegates to.
// `agen_access_details.rs`'s `AccessDetailsBase::new` is the same case and carries the same note.
//
// ⭐ AND A SECOND DERIVED FAMILY IS ALREADY WAITING FOR IT. `FlatteningLocalRegions.cpp:15` includes
// the same header and `:47` declares `class LocalOpNode : public OperationNode` with the identical
// five delegating accessors — entries 101-107, homed in `tf_flattening_local_regions.rs`. Hence the
// payload parameter `N`: that batch instantiates `OperationTreeBase<LocalOpNode>` from here instead
// of writing a second arena. ⚠️ WHEN IT DOES, THIS LAYER WANTS HOISTING into a module of its own
// beside the two files that share it; it sits in this file because this file is the one this batch
// owns, and a new `pub mod` line is a change to a file every parallel batch is also editing.

/// A NODE'S IDENTITY WITHIN ONE TREE — what an `OperationNode *` is in the C++.
///
/// # 🛑 AN INDEX, NOT A REFERENCE, AND THAT IS THE SHAPE OF THIS WHOLE PORT
///
/// ⛔⛔ THE C++ TREE IS INTRUSIVE AND CYCLIC. `parent_operation_`, `first_child_` and
/// `next_sibling_` (`OperationTree.hpp:186-188`) are raw `OperationNode *` into nodes `new`ed one at
/// a time and `delete`d by `OperationTreeBase::clear()` (`OperationTree.cpp:256`), so a child names
/// its parent while the parent names the child. `&`-references cannot express that at all, and
/// `Rc<RefCell<…>>` would only move the aliasing to run time and add a borrow that can fail. An
/// arena index is the same graph with the aliasing gone.
///
/// ⭐ AND IDENTITY COMES OUT RIGHT FOR FREE: `operator==` on a node is `this == &n`
/// (`OperationTree.hpp:33`) — pointer identity, not payload equality — which is exactly `==` on two
/// of these. `MaskNode::isMaskEquivalentToNode` (entry 171) compares `n->getParentNode() ==
/// getParentNode()`, and that is a comparison of two ids.
///
/// ⛔ IDS ARE MINTED ONLY BY THE TREE THAT OWNS THEM, which is why every read below indexes without
/// a bounds question: an `OperationNodeId` can only have come from [`OperationTreeBase::with_root`]
/// or one of its two inserts, the field is private, and nothing removes a node (the C++
/// `unlink`/`clear` are entry 179's, in the other family's file).
///
/// ⚠️ WHAT THAT DOES *NOT* RULE OUT is an id minted by one tree being read against another. It cannot
/// happen in this pass: the walk builds one `LoopMaskTree` per PT `dataflow::ProgramUnitOp` and
/// finishes with it before the next (`new LoopMaskTree(unit)`,
/// `VectorChainToSentientPT.cpp:1003`, threaded through `fuseNonComputeOps`/`fuseComputeOps`/
/// `lowerDanglingNonComputeOps` at `:1009-1014`), so two are never live at once — and the C++ raw
/// pointer has exactly the same hazard. If a second tree ever becomes reachable at the same time,
/// the answer is to brand the id with the tree's lifetime, not to add a bounds check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OperationNodeId(usize);

/// THE THREE LINKS A NODE CARRIES — `OperationTree.hpp:186-188`.
///
/// ⛔ THERE IS NO `prev_sibling_` AND NO `last_child_` FIELD, and that is load-bearing rather than an
/// omission to tidy up. `getPrevSibling` and `getLastChild` WALK the parent's chain
/// (`OperationTree.cpp:30-46`) — they are O(children) and derive their answer from the three links
/// alone. Caching either would be a second source of truth for one relation, and the insert
/// ([`OperationTreeBase::push_child`]) would then have two things to keep in step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Links {
    /// `OperationNode *parent_operation_ = nullptr;` (`hpp:186`).
    parent: Option<OperationNodeId>,
    /// `OperationNode *first_child_ = nullptr;` (`hpp:187`).
    first_child: Option<OperationNodeId>,
    /// `OperationNode *next_sibling_ = nullptr;` (`hpp:188`).
    next_sibling: Option<OperationNodeId>,
}

impl Links {
    /// THE `= nullptr` INITIALISERS (`hpp:186-188`) — a fresh node is linked to nothing, and
    /// `insertChildNode` (`cpp:239-254`) is what links it.
    const UNLINKED: Self = Self {
        parent: None,
        first_child: None,
        next_sibling: None,
    };
}

/// ONE NODE: ITS LINKS AND ITS PAYLOAD.
///
/// ⭐ THE PAYLOAD IS A TYPE PARAMETER BECAUSE THE C++ USES INHERITANCE FOR IT. `OperationNode` is
/// the base of `LoopMaskNode` (`LoopMaskTree.hpp:28`) and of `LocalOpNode`
/// (`FlatteningLocalRegions.cpp:47`); what each derived class adds is state, not overridden
/// structure — the links, the walks and `insertChildNode` are the base's and are never overridden.
///
/// ⛔ NOT `Copy`, AND THE FIELD BELOW IS WHY: an [`OpId`] owns its path. `Clone` survives only
/// because [`OperationTreeBase`] is `Clone` and a WHOLE-TREE clone is sound where a single-node copy
/// is not — the links are indices into the tree's own arena, so cloning every node together
/// reproduces the same graph, while cloning one node would put two nodes with the same links and the
/// same operation in one sibling chain. That single-node copy is what `OperationNode` deletes
/// (`OperationTree.hpp:30-31`) and what `LocalOpNode` refuses to derive
/// (`tf_flattening_local_regions.rs`); it is unreachable here because this type is private to this
/// module.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Node<N> {
    /// `Operation *operation_op_;` (`OperationTree.hpp:185`) — see [`Node::new`], entry 079.
    op: Option<OpId>,
    /// The three links.
    links: Links,
    /// What the derived class adds.
    payload: N,
}

impl<N> Node<N> {
    /// Replaces: e079_OperationNode
    ///
    /// **079/384** `LoopMaskNode::LoopMaskNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:32` (0L).
    ///
    /// ```cpp
    /// LoopMaskNode(Operation *op) : OperationNode(op) {};
    /// ```
    ///
    /// which is the base's constructor and its three member initialisers:
    ///
    /// ```cpp
    /// OperationNode(Operation *op) : operation_op_(op) {}
    /// // ...
    /// Operation *operation_op_;
    /// OperationNode *parent_operation_ = nullptr;
    /// OperationNode *first_child_ = nullptr;
    /// OperationNode *next_sibling_ = nullptr;
    /// ```
    ///
    /// # ⭐⭐ WHAT A ZERO-LINE CONSTRUCTOR DECIDES: A NODE **NAMES AN OPERATION** AND IS **DETACHED**
    ///
    /// Every one of the three links starts null and the insert (`insertChildNode`,
    /// `OperationTree.cpp:239-254`) is the only thing that wires them, so a node that was minted but
    /// never inserted is in no tree at all rather than in a broken one. That is why the mint sites
    /// here are the insert sites: [`OperationTreeBase::with_root`],
    /// [`OperationTreeBase::push_child`] and [`OperationTreeBase::push_named_child`].
    ///
    /// # 🛑 THE OPERATION IS AN [`OpId`] — A POSITION, NOT A BORROW
    ///
    /// ⛔⛔ A `&DfirOp` WOULD FREEZE THE PROGRAM AGAINST THE VERY REWRITES THIS TREE EXISTS FOR. The
    /// reference builds one tree per PT program unit and then threads it through three *mutating*
    /// passes — `fuseNonComputeOps`, `fuseComputeOps` and `lowerDanglingNonComputeOps`
    /// (`VectorChainToSentientPT.cpp:1003-1014`) — and `insertPTMaskOps` walks it while inserting
    /// `sentient.set_mask`/`incrmask` ops into the body (`LoweringPTMasks.cpp:74-100`). A shared
    /// borrow of an op held across those calls forbids them; an owned position does not.
    ///
    /// ⭐ AND THE REFERENCE ITSELF SAYS IDENTITY MUST SURVIVE REWRITING: `updateNode(from, to)`
    /// (entry 237, `LoopMaskTree.cpp:155-165`) exists precisely because a lowering REPLACES the op a
    /// node names, and it re-keys `op_to_node_` in the same breath. A borrow cannot express that; a
    /// re-assignable [`OpId`] can.
    ///
    /// ⚠️ SO THIS DIVERGES FROM `LocalOpNode::new`, WHICH TOOK `&'p DfirOp` (entry 101), and the
    /// difference is not taste: that node is built, read and dropped inside one non-mutating walk of
    /// `uniform.uniformize_regions`, and entries 101-104 never look at the op at all. This one is
    /// SEARCHED BY IDENTITY (`findNodeFromOp`, entry 078) and used as an INSERTION POINT
    /// (`n->getParentNode()->getOperation()`, `LoweringPTMasks.cpp:75`), and [`OpId`] — this crate's
    /// stand-in for `Operation *`, one ordinal per region level — answers both.
    ///
    /// ⛔ `None` IS THE SYNTHETIC ROOT AND ONLY THE ROOT. `new LoopMaskNode(nullptr)`
    /// (`LoopMaskTree.cpp:175`) is the one null the reference passes; `computeLoops` mints loop nodes
    /// over a `sentient::ForOp` (`:178`) and `addMaskNode` mask nodes over a compute (`:138`). That is
    /// enforced rather than documented: [`OperationTreeBase::push_named_child`], the insert this
    /// tree uses, takes an [`OpId`] by value rather than an `Option`, so the only way to reach `None`
    /// here is `with_root`. (The other insert, [`OperationTreeBase::push_child`], leaves it `None`
    /// for a tree that names its operations in the payload instead — entry 101's.)
    #[must_use]
    const fn new(op: Option<OpId>, payload: N) -> Self {
        Self {
            op,
            links: Links::UNLINKED,
            payload,
        }
    }
}

/// ONE OPERATION TREE — `class OperationTreeBase` (`OperationTree.hpp:201`).
///
/// ⭐⭐ THE ROOT IS SYNTHETIC AND THE HEADER SAYS SO: *"a program consists of a forest of operation
/// trees … The OperationTree class introduces a synthetic root node and collects all such trees
/// under that single root"* (`hpp:195-200`). So the top-level loops of a program unit are SIBLINGS
/// under one node that stands for no operation at all — which is why the vendor's own
/// `dynamic_pt_masking.mlir` case, with its two top-level `sentient.for` nests, exercises
/// `getNextSibling`/`getPrevSibling`/`getLastChild` on the root's children.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationTreeBase<N> {
    /// Every node, in creation order; an [`OperationNodeId`] indexes this.
    nodes: Vec<Node<N>>,
    /// `OperationNode *root_ = nullptr;` (`hpp:232`) — ⛔ NOT AN `Option`, see [`Self::root`].
    root: OperationNodeId,
}

impl<N> OperationTreeBase<N> {
    /// A TREE HOLDING NOTHING BUT ITS SYNTHETIC ROOT — `root_ = new LoopMaskNode(nullptr)`, the
    /// first act of `computeLoops` (`LoopMaskTree.cpp:175`).
    ///
    /// ⛔⛔ THE ROOT ARRIVES WITH THE TREE, AND THAT IS WHAT DISCHARGES `getRoot`'s CHECK. The C++
    /// default-constructs with `root_ == nullptr` (`hpp:232`) and every reader then re-asks whether
    /// it is set: `DT_CHECK(root_ && …)` in `getRoot` (`hpp:205-206`), `empty()` at `hpp:214`,
    /// `DT_CHECK_MSG(!root_ …)` at `LoopMaskTree.cpp:174`. Taking the root payload here makes a
    /// tree without a root unconstructible, so the question cannot be asked at run time.
    ///
    /// ⛔ AND THE ROOT NAMES NO OPERATION — `Node::new(None, …)`, the one null `Operation *` in this
    /// family (`LoopMaskTree.cpp:175`, and see [`Node::new`]). The header says why: the synthetic root
    /// is what *collects* the forest's trees, not one of them (`OperationTree.hpp:191-196`).
    pub fn with_root(root: N) -> Self {
        Self {
            nodes: vec![Node::new(None, root)],
            root: OperationNodeId(0),
        }
    }

    /// `OperationTreeBase::getRoot()` — `OperationTree.hpp:204-208`.
    ///
    /// ```cpp
    /// const OperationNode *getRoot() const {
    ///   DT_CHECK(root_ && root_->getNextSibling() == nullptr &&
    ///            root_->getParentNode() == nullptr && "invalid root");
    ///   return root_;
    /// }
    /// ```
    ///
    /// ⛔ ALL THREE CONJUNCTS HOLD BY CONSTRUCTION, which is why the `DT_CHECK` has no counterpart.
    /// `root_` is non-null because [`Self::with_root`] takes it; and the root's `parent` and
    /// `next_sibling` are `None` from [`Links::UNLINKED`] and stay so because the ONLY writer of
    /// links is the shared insert, which writes the parent of the node it just created and the
    /// `next_sibling` of an existing CHILD — never the fields of the node it was given as a parent.
    /// The root is never anyone's child, so nothing can reach either field.
    pub fn root(&self) -> OperationNodeId {
        self.root
    }

    /// A NODE'S PAYLOAD — the derived state the C++ reaches through the `static_cast`.
    pub fn payload(&self, n: OperationNodeId) -> &N {
        &self.nodes[n.0].payload
    }

    /// `OperationNode::getOperation()` — `OperationTree.hpp:37`, `return operation_op_;`.
    ///
    /// ⚠️ A FIELD ACCESSOR ON THE EXCLUDED LIST, written because the field it reads arrives with
    /// entry 079. ⛔ `None` IS THE SYNTHETIC ROOT — `print` branches on exactly that
    /// (`if (op) … else OS << " null, at depth: "`, `OperationTree.cpp:125-130`).
    pub fn operation(&self, n: OperationNodeId) -> Option<&OpId> {
        self.nodes[n.0].op.as_ref()
    }

    /// THE FIRST NODE NAMING `op` — the scan behind [`LoopMaskTree::find_node_from_op`] (entry 078).
    ///
    /// ⛔ THE ROOT CANNOT MATCH: its operation is `None` and this compares against `Some(op)`, which
    /// is the same exclusion the reference gets from never inserting the root into `op_to_node_`
    /// (`LoopMaskTree.cpp:175` inserts nothing; `:152` and `:179` are the only insertions).
    fn find_by_op(&self, op: &OpId) -> Option<OperationNodeId> {
        self.nodes
            .iter()
            .position(|node| node.op.as_ref() == Some(op))
            .map(OperationNodeId)
    }

    /// `OperationNode::breadthFirstWalk(n, action)` — `OperationTree.cpp:183-198`.
    ///
    /// ```cpp
    /// void OperationNode::breadthFirstWalk(OperationNode *n, ActionFuncTy action) {
    ///   DT_CHECK_MSG(n, "expected valid node");
    ///   std::queue<OperationNode *> queue;
    ///   queue.push(n);
    ///   while (!queue.empty()) {
    ///     OperationNode *curr_node = queue.front();
    ///     DT_CHECK_MSG(curr_node, "expected valid node");
    ///     (void)action(curr_node);
    ///     queue.pop();
    ///     OperationNode *child = curr_node->getFirstChild();
    ///     while (child) {
    ///       queue.push(child);
    ///       child = child->getNextSibling();
    ///     }
    ///   }
    /// }
    /// ```
    ///
    /// ⚠️ NOT A SCHEDULED UNIT — same standing as the rest of this base layer, and the body entry 077
    /// delegates to. `kBFS` is the one of the seven `WalkOrder`s this file needs
    /// (`LoopMaskTree.cpp:133`); the other six specializations are unscheduled and unwritten.
    ///
    /// ⭐⭐ `(void)action(curr_node)` IS WHY THE ACTION RETURNS NOTHING HERE. The C++ action type
    /// returns a node (`ActionFuncTy`, `hpp:77`) because the two *guided* walks steer by it; a BFS
    /// discards it. That has a consequence worth stating, because a reader of the one caller will
    /// otherwise get it wrong: `analyzeAndInsertMaskOps`'s `return nullptr` after
    /// `signalPassFailure()` (`LoweringPTMasks.cpp:197`) does **not** stop the walk — the remaining
    /// queue is still visited and further masks are still lowered. Only the pass failure is recorded.
    ///
    /// ⛔ BOTH `DT_CHECK_MSG(…, "expected valid node")`s ARE UNREPRESENTABLE: an
    /// [`OperationNodeId`] is never null, and the queue is fed only from `first_child`/`next_sibling`,
    /// which yield ids that exist.
    ///
    /// ⭐ AND THE ACTION TAKES `&mut` WHILE THE TREE IS BORROWED SHARED, which is exactly the C++'s
    /// contract for `kBFS`: *"The action must not remove itself, its siblings or any parent or
    /// ancestor"* (`hpp:96-99`). Here it cannot mutate the tree at all, so the one freedom the C++
    /// leaves — removing descendants mid-walk — is closed rather than merely documented. The one
    /// caller uses none of it: `analyzeAndInsertMaskOps` reads the tree and rewrites the IR
    /// (`LoweringPTMasks.cpp:164-203`).
    fn breadth_first_walk(&self, from: OperationNodeId, action: &mut impl FnMut(OperationNodeId)) {
        let mut queue: VecDeque<OperationNodeId> = VecDeque::new();
        queue.push_back(from);
        while let Some(curr_node) = queue.pop_front() {
            action(curr_node);
            let mut child = self.first_child(curr_node);
            while let Some(c) = child {
                queue.push_back(c);
                child = self.next_sibling(c);
            }
        }
    }

    /// The three links of one node.
    fn links(&self, n: OperationNodeId) -> Links {
        self.nodes[n.0].links
    }

    /// `OperationNode::getParentNode()` — `OperationTree.hpp:50`, `return parent_operation_;`.
    pub fn parent_node(&self, n: OperationNodeId) -> Option<OperationNodeId> {
        self.links(n).parent
    }

    /// `OperationNode::getFirstChild()` — `OperationTree.hpp:54`, `return first_child_;`.
    pub fn first_child(&self, n: OperationNodeId) -> Option<OperationNodeId> {
        self.links(n).first_child
    }

    /// `OperationNode::getNextSibling()` — `OperationTree.hpp:58`, `return next_sibling_;`.
    pub fn next_sibling(&self, n: OperationNodeId) -> Option<OperationNodeId> {
        self.links(n).next_sibling
    }

    /// `OperationNode::getLastChild()` — `OperationTree.cpp:30-35`.
    ///
    /// ```cpp
    /// OperationNode *OperationNode::getLastChild() const {
    ///   OperationNode *sibling = getFirstChild();
    ///   while (sibling && sibling->getNextSibling())
    ///     sibling = sibling->getNextSibling();
    ///   return sibling;
    /// }
    /// ```
    ///
    /// ⭐ THE `sibling &&` IS THE LEAF CASE: a node with no children returns null, and `nullptr` here
    /// means *no last child*, not *failure*. `isLeaf()` (`hpp:70`) asks the same question of
    /// `getFirstChild()`.
    pub fn last_child(&self, n: OperationNodeId) -> Option<OperationNodeId> {
        let mut sibling = self.first_child(n)?;
        while let Some(next) = self.next_sibling(sibling) {
            sibling = next;
        }
        Some(sibling)
    }

    /// `OperationNode::getPrevSibling()` — `OperationTree.cpp:37-46`.
    ///
    /// ```cpp
    /// OperationNode *OperationNode::getPrevSibling() const {
    ///   DT_CHECK_MSG(getParentNode(), "expected a parent");
    ///   OperationNode *prev = getParentNode()->getFirstChild();
    ///   if (prev == this) return nullptr;
    ///   while (prev) {
    ///     if (prev->getNextSibling() == this) break;
    ///     prev = prev->getNextSibling();
    ///   }
    ///   return prev;
    /// }
    /// ```
    ///
    /// ⛔⛔ THE `DT_CHECK` IS THE ROOT AND NOTHING ELSE, so it becomes an answer rather than a
    /// refusal. Every node but the synthetic root has a parent — the insert sets it — so the only
    /// call that can reach the check is `prev_sibling(root)`, and for the root `None` is not a
    /// degraded answer but the true one: the root has no siblings at all
    /// (`getRoot`'s own invariant, `hpp:205-206`). ⭐ AND THE FIRST-CHILD ARM IS ALREADY THIS SAME
    /// `None`: `if (prev == this) return nullptr` (`:40`).
    pub fn prev_sibling(&self, n: OperationNodeId) -> Option<OperationNodeId> {
        let parent = self.parent_node(n)?;
        let mut prev = self.first_child(parent);
        if prev == Some(n) {
            return None;
        }
        while let Some(p) = prev {
            if self.next_sibling(p) == Some(n) {
                break;
            }
            prev = self.next_sibling(p);
        }
        prev
    }

    /// `OperationNode::insertChildNode(child)` — `OperationTree.cpp:239-254`, with `pos` defaulted.
    ///
    /// ```cpp
    /// void OperationNode::insertChildNode(OperationNode *child, OperationNode *pos) {
    ///   DT_CHECK_MSG(child, "expected valid child");
    ///   if (isLeaf()) {
    ///     DT_CHECK_MSG(pos == nullptr, "position incorrectly specified");
    ///     setFirstChild(child);
    ///   } else if (pos) { … }
    ///   else
    ///     getLastChild()->setNextSibling(child);
    ///   child->setParentNode(this);
    /// }
    /// ```
    ///
    /// ⭐ ONE `last_child` CALL COVERS BOTH SURVIVING BRANCHES: it is `None` exactly when `isLeaf()`
    /// is true, since both read `first_child_`.
    ///
    /// ⚠️ THE `pos` BRANCH IS NOT WRITTEN, AND NOT BECAUSE IT IS HARD. Both callers in this family
    /// default it — `computeLoops` calls `parent_node->insertChildNode(new_node)`
    /// (`LoopMaskTree.cpp:189`) and `addMaskNode` calls `found_node->insertChildNode(mask_node)`
    /// (`:145`, `:148`) — so *append* is the whole of what the Loop Mask Tree uses, and the appended
    /// order is program order. The day a unit needs mid-list insertion it arrives with that unit,
    /// and it will take `pos` alone rather than a `(parent, pos)` pair: `pos->getParentNode()` IS
    /// the parent, which is what makes the C++'s second `DT_CHECK_MSG` (`:245-247`) true.
    ///
    /// ⛔ AND THE FIRST `DT_CHECK_MSG(child, …)` IS UNREPRESENTABLE HERE: this takes a payload by
    /// value and mints the node itself, so there is no null child to check for.
    ///
    /// ⭐ TWO PUBLIC INSERTS OVER ONE BODY, AND THE REFERENCE IS WHY THERE IS ONLY ONE THERE. In C++
    /// `operation_op_` is set by the node's own constructor, which the CALLER runs
    /// (`new LMTLoopNode(op)`, `LoopMaskTree.cpp:178`), so `insertChildNode` never sees an operation.
    /// Here the arena mints the node, so the two constructors that reach it reach it as two inserts:
    /// [`Self::push_named_child`] for a tree that names the operation in the base by [`OpId`] (entry
    /// 079, the Loop Mask Tree) and [`Self::push_child`] for one whose PAYLOAD names it (entry 101,
    /// `LocalOpNode`'s `&DfirOp`). The linking below is shared because the reference shares it.
    fn insert_child(
        &mut self,
        parent: OperationNodeId,
        op: Option<OpId>,
        node: N,
    ) -> OperationNodeId {
        let child = OperationNodeId(self.nodes.len());
        self.nodes.push(Node::new(op, node));
        match self.last_child(parent) {
            // `if (isLeaf()) setFirstChild(child)` (`:241-243`).
            None => self.nodes[parent.0].links.first_child = Some(child),
            // `else getLastChild()->setNextSibling(child)` (`:251-252`).
            Some(last) => self.nodes[last.0].links.next_sibling = Some(child),
        }
        // `child->setParentNode(this)` (`:253`) — last, as in the reference. ⭐ The new node is not in
        // any sibling chain until the `match` above runs, so `last_child(parent)` cannot see it.
        self.nodes[child.0].links.parent = Some(parent);
        child
    }

    /// [`Self::insert_child`] FOR A TREE WHOSE PAYLOAD NAMES THE OPERATION.
    ///
    /// ⭐ `FlatteningLocalRegions`' node keeps the operation in `N` as a `&DfirOp` (entry 101), so
    /// the base's own [`Node::op`] stays `None` for every node of that tree — including its root,
    /// which does name an op (`new LocalOpNode(&op_)`, `FlatteningLocalRegions.cpp:392`). ⛔ SO
    /// `None` HERE IS "NOT NAMED IN THE BASE", NOT "NAMES NOTHING": a tree built through this insert
    /// must not be searched with [`Self::find_by_op`], which would find nothing.
    pub fn push_child(&mut self, parent: OperationNodeId, node: N) -> OperationNodeId {
        self.insert_child(parent, None, node)
    }

    /// [`Self::insert_child`] FOR A TREE THAT NAMES THE OPERATION IN THE BASE, BY [`OpId`].
    ///
    /// ⭐ `op` IS NOT AN `Option`, AND THAT IS WHAT MAKES "ONLY THE ROOT NAMES NO OPERATION" A TYPE
    /// RATHER THAN A CLAIM FOR THE LOOP MASK TREE — see [`Node::new`]. Both C++ callers pass a real
    /// op: `new LMTLoopNode(op)` over a `sentient::ForOp` (`LoopMaskTree.cpp:178`) and
    /// `new MaskNode(mask_related_op, …)` over the compute the mask applies to (`:138`), and the
    /// one null the reference passes goes to the root (`:175`), which only [`Self::with_root`]
    /// builds.
    pub fn push_named_child(
        &mut self,
        parent: OperationNodeId,
        op: OpId,
        node: N,
    ) -> OperationNodeId {
        self.insert_child(parent, Some(op), node)
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════════
// THE LOOP MASK TREE — `LoopMaskTree.hpp` / `LoopMaskTree.cpp`
// ══════════════════════════════════════════════════════════════════════════════════════════════════

/// HOW MANY COLUMNS OF A PT ROW A MASK COVERS — the `int start_val` a `MaskNode` carries
/// (`LoopMaskTree.hpp:71`, `:87`).
///
/// ⭐⭐ IT IS COLUMNS, NOT LANES, AND THE REFERENCE NAMES IT ITSELF. `getMaskValueForPT` derives the
/// value it hands to `updateLoopMaskTreeForConstantMask` as
/// `masked_columns = num_masked_elems / sys_def.numSlicesPerStick` (`Helper.cpp:104-105`), the same
/// divisor it uses two lines earlier for `num_lanes_in_slice` (`:56`). The vendor's own case pins the
/// arithmetic: `#set2 = affine_set<(d0) : (d0 - 48 >= 0, -d0 + 63 >= 0)>` over a `vector<64xf16>`
/// gives `num_masked_elems = 64 - 48 = 16`, and `16 / 8` is the `sentient.scalar_constant {value = 2}`
/// the golden's `set_mask` takes (`dynamic_pt_masking.mlir:86-87`, `:211`).
///
/// ⛔ SO A RAW `int` WOULD BE THE THIRD THING IN THAT LINE WITH NO UNIT. Elements, slices and columns
/// all appear in one expression; the type is what stops a lane count reaching a `set_mask`.
///
/// ⭐ AND `u32` IS DELIBERATE OVER `u64`: `i64::from` a `u32` is total, so minting the
/// `sentient.scalar_constant` this becomes needs no fallible conversion. The C++ validates
/// non-negativity of the same quantity by hand (`Helper.cpp:106-111`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub struct MaskedColumns(pub u32);

/// WHETHER A MASK STANDS STILL OR ADVANCES WITH ITS LOOP — the `int increment` of a `MaskNode`
/// (`LoopMaskTree.hpp:71`, `:87`).
///
/// # 🛑 A CLOSED SET OF TWO, MEASURED AT BOTH WRITERS
///
/// ⭐⭐ ONLY 0 AND 1 EXIST. There are exactly two calls that build a mask node:
/// `updateLoopMaskTreeForConstantMask` passes `0` (`LoweringPTMasks.cpp:24-25`) and
/// `updateLoopMaskTreeForDynamicMask` is called only as `…, /*start_val*/ 0, /*increment*/ 1`
/// (`VectorChainToSentientPT.cpp:463-478`). The reader agrees: `insertMaskOps` branches
/// `if (increment == 0) … else { DT_CHECK_MSG(increment == 1, "increments > 1 not currently
/// supported"); … }` (`LoweringPTMasks.cpp:47`, `:70-71`).
///
/// ⛔ SO AN `int` HERE WOULD BE A FIELD WHOSE THIRD VALUE IS A RUN-TIME ABORT. The enum makes the
/// unsupported case unrepresentable instead — the crate's rule, and the one that keeps `insertMaskOps`
/// (entry 239) free of a `DT_CHECK` counterpart.
///
/// ⚠️ WHEN INCREMENTS > 1 LAND, THIS GAINS A PAYLOAD, not a raw `int`. The reference's own TODO says
/// what that costs: *"When we support increments greater than 1, we will need to wrap the incrmask op
/// with a loop to increment the correct number of times"* (`LoweringPTMasks.cpp:86-87`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaskIncrement {
    /// `increment == 0` — a CONSTANT MASK. `insertMaskOps` brackets the masked op itself: a
    /// `sentient.set_mask <start_val>` before it and a `sentient.set_mask 0` after it to reset
    /// (`LoweringPTMasks.cpp:48-68`).
    Constant,
    /// `increment == 1` — the mask advances by one column per iteration of the mask node's PARENT
    /// loop. `insertMaskOps` reads the parent to find it (`auto parent_loop =
    /// n->getParentNode()->getOperation()`, `LoweringPTMasks.cpp:74`) and brackets the LOOP rather
    /// than the op: `set_mask <start_val>` before the loop, `sentient.incrmask` at the loop's
    /// terminator, `set_mask 0` after the loop (`:72-102`).
    PerParentLoopIteration,
}

/// Replaces: e086_MaskNode
///
/// **086/384** `~MaskNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:74` (0L).
///
/// ```cpp
/// class MaskNode : public LoopMaskNode {
///  public:
///   MaskNode(Operation *op, int start_val, int increment)
///       : LoopMaskNode(op), start_val_(start_val), increment_(increment) {};
///   ~MaskNode() {}
///   …
///  private:
///   int start_val_, increment_;
/// };
/// ```
///
/// # 🛑 THE UNIT IS AN EMPTY DESTRUCTOR, AND EMPTY IS ITS CONTENT
///
/// ⭐⭐ WHAT `~MaskNode() {}` SAYS IS THAT DESTROYING A MASK NODE RELEASES NOTHING AND TOUCHES NO
/// OTHER NODE. Its own members are two `int`s (`hpp:87`); the tree's storage is freed elsewhere —
/// `OperationTreeBase::clear()` post-order-walks and `delete`s each node (`OperationTree.cpp:256`,
/// and `FlatteningLocalRegionsTree::clear` at `FlatteningLocalRegions.cpp:112` is the same shape.) A
/// destructor that deleted its children would double-free every one of them under that walk.
///
/// ⛔ SO THE PORT IS THE ABSENCE OF A `Drop` IMPL, AND THE ABSENCE IS CHECKED AT BUILD TIME — see the
/// `const` below. Writing `impl Drop for MaskNode {}` would be the opposite of this unit: it would
/// make the type non-`Copy`, forbid moving out of a field, and add a call where the C++ has an
/// empty one that the compiler removes.
///
/// ⛔ THE TWO GETTERS ARE PUBLIC FIELDS. `getStartVal` (`hpp:76`) and `getIncrement` (`hpp:77`) are
/// on the excluded list as field accessors — *there is no function to port* — and `isMaskEquivalentToNode`
/// (entry 171) reads both.
///
/// ⚠️ `operation_op_` IS NOT DECLARED IN THIS TYPE AND MUST NOT BE. The op a mask node stands for
/// belongs to the BASE (`OperationTree.hpp:185`), and the constructor that takes it is entry 079 with
/// entry 238 (`computeLoops`) as its caller — the two units that decide how this crate names an
/// operation. Declaring it here would both duplicate the base's member and pre-empt that decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaskNode {
    /// `int start_val_` (`hpp:87`) — where the mask starts.
    pub start_val: MaskedColumns,
    /// `int increment_` (`hpp:87`) — whether it advances.
    pub increment: MaskIncrement,
}

/// Replaces: e087_LMTLoopNode
///
/// **087/384** `~LMTLoopNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:94` (0L).
///
/// ```cpp
/// class LMTLoopNode : public LoopMaskNode {
///  public:
///   LMTLoopNode(Operation *op) : LoopMaskNode(op) {};
///   LMTLoopNode(const LMTLoopNode &n) : LoopMaskNode(n.getOperation()) {};
///   ~LMTLoopNode() {}
///
///   bool isLoopNode() const final override { return true; }
/// };
/// ```
///
/// ⭐⭐ THE CLASS ADDS NO STATE AT ALL, hence a unit struct. A loop node is a `sentient.for`
/// (`computeLoops` creates one per `isa<sentient::ForOp>`, `LoopMaskTree.cpp:177-178`) and everything
/// it carries — the op, the links — is the base's. What distinguishes it from a mask node is
/// `isLoopNode`/`isMaskNode`, and in [`LoopMaskNode`] that is the variant itself.
///
/// ⭐ SO `~LMTLoopNode() {}` MAKES THE SAME STATEMENT AS `~MaskNode() {}`: destroying a loop node
/// releases nothing, and in particular does not touch the loop's children — the nodes for the loops
/// nested inside it, which `clear()`'s post-order walk owns. Checked at build time below.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LMTLoopNode;

// ⛔ THE TWO EMPTY DESTRUCTORS, AS A BUILD-TIME GUARD RATHER THAN A RUN-TIME ONE (entries 086, 087).
//
// `needs_drop::<T>()` is `false` exactly when destroying a `T` runs no code — which is what
// `~MaskNode() {}` and `~LMTLoopNode() {}` state. The array length below is that answer as a
// `usize`, so a node type that acquired a destructor (a `Drop` impl, or a field that owns a heap
// allocation or another node) would make the initialiser `[(); 1]` and fail to compile against the
// declared `[(); 0]`. ⭐ THIS IS A TYPE MISMATCH AT BUILD TIME, not an `assert!` — the crate's
// never-runtime-refuse rule and `guard-every-crash-at-build-time` both point the same way.
//
// ⭐ AND IT BITES — FALSIFIED, NOT ASSUMED. Pointing either line at a type that owns something
// (`needs_drop::<String>()`) stops the build with *expected an array with a size of 0, found one with
// a size of 1*. A `Drop` impl on one of these types is caught one step earlier still, by the derived
// `Copy`: *the trait `Copy` cannot be implemented for this type; the type has a destructor* (E0184).
const _: [(); 0] = [(); core::mem::needs_drop::<MaskNode>() as usize];
const _: [(); 0] = [(); core::mem::needs_drop::<LMTLoopNode>() as usize];

/// ONE NODE OF THE LOOP MASK TREE — the `LoopMaskNode` hierarchy as the closed set it is.
///
/// # 🛑 THREE KINDS, AND THE REFERENCE'S OWN PRINTER PROVES THERE ARE NO MORE
///
/// ⭐⭐ `LoopMaskNode::print` DISPATCHES ON EXACTLY THIS SET AND CALLS THE FOURTH CASE AN ERROR:
/// `if (n->isLoopNode()) … else if (n->isMaskNode()) … else OS << "ERROR: non-loop, non-mask node
/// detected "` (`LoopMaskTree.cpp:102-111`). The one node that legitimately reaches that arm is the
/// synthetic root, `new LoopMaskNode(nullptr)` (`:175`) — the only plain base instance this family
/// creates, and the only node whose `getOperation()` is null.
///
/// ⛔ SO THE ROOT IS A VARIANT, NOT AN `Option<…>` AROUND THE OTHERS. "No operation" and "neither a
/// loop nor a mask" are the same fact about the same single node; splitting them would leave every
/// reader unwrapping an op that is absent for exactly one node it can name.
///
/// ⭐ AND VIRTUAL DISPATCH IS NOT LOST, IT IS INVERTED. `isLoopNode`/`isMaskNode` (`hpp:54-55`,
/// overridden at `:79` and `:96`) are on the excluded list as accessors: in Rust the question is a
/// `match` on this enum, which a new kind cannot silently answer `false` to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoopMaskNode {
    /// `new LoopMaskNode(nullptr)` (`LoopMaskTree.cpp:175`) — the synthetic root the header
    /// describes (`hpp:99-102`), standing for no operation, holding the program unit's top-level
    /// loops as its children.
    SyntheticRoot,
    /// A `sentient.for` — [`LMTLoopNode`], entry 087.
    Loop(LMTLoopNode),
    /// A mask on a compute — [`MaskNode`], entry 086.
    Mask(MaskNode),
}

/// Replaces: e080_LoopMaskNode
///
/// **080/384** `~LoopMaskNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:34` (0L).
///
/// ```cpp
/// virtual ~LoopMaskNode() {}
/// ```
///
/// # ⭐⭐ THE BODY IS EMPTY AND THE `virtual` IS THE WHOLE CONTENT
///
/// ⛔ A NODE OWNS NOTHING — not its children, not its operation. `clear()` deletes every node in a
/// post-order walk of the tree (`OperationTree.cpp:256`), so a destructor that freed its children
/// would double-free them; and the operation belongs to the MLIR context, not to the node. That much
/// this shares with `~MaskNode` and `~LMTLoopNode` (entries 086, 087, guarded above), and the guard
/// below is the same statement: destroying one runs no code.
///
/// ⭐ WHAT IS DIFFERENT IS THAT THIS ONE IS **`virtual`** AND THOSE TWO ARE NOT. Every node in this
/// family is reached through a `LoopMaskNode *` — the five `static_cast`s of entries 081-085 return
/// one, `clear()` deletes through the base pointer — so without the vtable slot declared *here*,
/// `delete` on a base pointer to a `MaskNode` would be undefined behaviour and the derived
/// destructor would never run. The `virtual` is what makes the two non-virtual destructors below it
/// safe.
///
/// ⭐⭐ AND IN RUST THAT SLOT IS NOT NEEDED, BECAUSE THE HIERARCHY IS ONE TYPE. [`LoopMaskNode`] is an
/// enum: dropping one drops the right variant because the discriminant is in the value, so there is
/// no base pointer through which the wrong destructor could be selected. The three-kind `match`
/// replaced the vtable for `isLoopNode`/`isMaskNode` and it replaces it here too.
///
/// ⛔ THIS GUARDS THE PAYLOAD, NOT THE ARENA NODE. `needs_drop::<Node<LoopMaskNode>>()` is `true` and
/// must be — a `Node` carries an [`OpId`], which owns its path (see [`Node::new`]). What the
/// reference's destructor is about is the node's OWN state, and in this port that is the payload.
const _: [(); 0] = [(); core::mem::needs_drop::<LoopMaskNode>() as usize];

/// A NODE'S IDENTITY IN A LOOP MASK TREE — the `LoopMaskNode *` that the five `static_cast`s of
/// entries 081-085 produce.
///
/// # 🛑 MINTING ONE **IS** THE `static_cast`
///
/// ⛔⛔ `static_cast<LoopMaskNode *>(OperationNode::getFirstChild())` IS AN UNCHECKED DOWNCAST — no
/// `dyn_cast`, no null test, no RTTI. It is sound in the C++ only because of a fact about the whole
/// tree: every node in a `LoopMaskTree` was `new`ed as a `LMTLoopNode`, a `MaskNode` or the root's
/// plain `LoopMaskNode` (`LoopMaskTree.cpp:175`, `:178`, `:138`), so a base pointer out of this tree
/// always points at a derived object.
///
/// ⭐ HERE THAT FACT IS THE TYPE. The arena's payload IS [`LoopMaskNode`], so wrapping an
/// [`OperationNodeId`] that came from `self.base` cannot be wrong, and there is nothing to check at
/// run time. A cast that could fail is unwritable rather than unchecked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LoopMaskNodeId(OperationNodeId);

/// THE LOOP MASK TREE — `class LoopMaskTree : public OperationTreeBase`
/// (`LoopMaskTree.hpp:103-138`).
///
/// ⛔ FILLED WAVE BY WAVE, LIKE `AccessDetailsBase`. What is declared here is what entries 077-088
/// need.
///
/// ⭐⭐ AND THE PRIVATE `DenseMap<Operation *, LoopMaskNode *> op_to_node_` (`hpp:137`) IS STILL
/// ABSENT — ON PURPOSE, NOW THAT ITS READER IS WRITTEN. The map is a MEMO over one predicate: *which
/// node names this operation*. Entry 078 answers that by scanning the arena
/// ([`LoopMaskTree::find_node_from_op`]), so the map buys speed and nothing else — and it would cost
/// a second source of truth that its two writers (entries 236 `addMaskNode` and 237 `updateNode`)
/// have to keep in step with the nodes, which is exactly the bug its own
/// `DT_CHECK_MSG(op_to_node_.find(mask_related_op) == op_to_node_.end(), "op already in map - should
/// not happen")` (`LoopMaskTree.cpp:150-151`) exists to catch. The campaign brief lets a port drop
/// the mechanism for REACHING an operand; a memo is that mechanism.
///
/// ⚠️ WHAT THAT TRADES IS COMPLEXITY, AND THE SIZE IS KNOWN: the tree has one node per
/// `sentient.for` plus one per masked compute — 15 for the vendor's own two-nest case
/// (`dynamic_pt_masking.mlir:215-317`) — and `addMaskNode` is the only per-compute caller. The day a
/// profile says otherwise, the index belongs beside its writers in the same batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopMaskTree {
    /// The `OperationTreeBase` this derives from.
    base: OperationTreeBase<LoopMaskNode>,
}

impl LoopMaskTree {
    /// Replaces: e088_getRoot
    ///
    /// **088/384** `getRoot` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:109` (2L).
    ///
    /// ```cpp
    /// const LoopMaskNode *getRoot() const {
    ///   return static_cast<const LoopMaskNode *>(OperationTreeBase::getRoot());
    /// }
    /// ```
    ///
    /// ⭐ ONE METHOD FOR THE TWO OVERLOADS. `hpp:112-114` is the same body without the `const`s, and
    /// the base's pair does the same (`OperationTree.hpp:204-212`, the mutable one a `const_cast` of
    /// the other). An [`LoopMaskNodeId`] is not a borrow, so mutability is the caller's business:
    /// `addMaskNode` mutates through `getRoot()` (`LoopMaskTree.cpp:147-148`) and `insertPTMaskOps`
    /// only reads.
    ///
    /// ⛔ THE `DT_CHECK` INSIDE THE BASE'S `getRoot` IS DISCHARGED BY CONSTRUCTION — see
    /// [`OperationTreeBase::root`].
    ///
    /// ⭐ ITS CALLERS ARE ALREADY VISIBLE: `addMaskNode` reaches the root's first child and uses the
    /// root as the parent for a mask with no enclosing loop (`:142`, `:147`), `computeLoops` makes it
    /// the default parent of a top-level loop (`:181`), and `walk` starts the BFS from it (`:133`).
    pub fn root(&self) -> LoopMaskNodeId {
        LoopMaskNodeId(self.base.root())
    }

    /// Replaces: e081_getParentNode
    ///
    /// **081/384** `getParentNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:36` (2L).
    ///
    /// ```cpp
    /// LoopMaskNode *getParentNode() const {
    ///   return static_cast<LoopMaskNode *>(OperationNode::getParentNode());
    /// }
    /// ```
    ///
    /// ⭐⭐ FOR A MASK NODE THE PARENT IS THE LOOP THAT CONTROLS THE MASK, and that is not a
    /// bookkeeping detail — it is where `insertMaskOps` puts its ops: `auto parent_loop =
    /// n->getParentNode()->getOperation()` (`LoweringPTMasks.cpp:74`), then `set_mask` before that
    /// loop, `incrmask` at its terminator and `set_mask 0` after it. `addMaskNode` is what
    /// establishes the relation, attaching a mask node under the `sentient.for` whose induction
    /// variable drives it (`LoopMaskTree.cpp:143-145`).
    ///
    /// ⛔ `None` IS THE SYNTHETIC ROOT AND ONLY THE ROOT. The insert gives every other node a
    /// parent, so `nullptr` here is not a "not found" — the C++ readers rely on that:
    /// `isMaskEquivalentToNode` compares two parents without a null test (`:124`) and
    /// `insertMaskOps` dereferences the result directly (`LoweringPTMasks.cpp:74`).
    pub fn parent_node(&self, n: LoopMaskNodeId) -> Option<LoopMaskNodeId> {
        self.base.parent_node(n.0).map(LoopMaskNodeId)
    }

    /// Replaces: e082_getFirstChild
    ///
    /// **082/384** `getFirstChild` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:39` (2L).
    ///
    /// ```cpp
    /// LoopMaskNode *getFirstChild() const {
    ///   return static_cast<LoopMaskNode *>(OperationNode::getFirstChild());
    /// }
    /// ```
    ///
    /// ⭐ THE CHILD LIST IS IN PROGRAM ORDER, LOOPS BEFORE MASKS. `computeLoops` builds every loop
    /// node first, in a pre-order walk of the unit (`LoopMaskTree.cpp:176-190`); mask nodes are
    /// appended later, one per compute, as the lowering meets them (`:136-152`). So a loop that
    /// contains both a nested loop and a masked compute lists the nested loop first however the two
    /// appear in the source — which is exactly what the vendor's `dynamic_pt_masking.mlir` tree does
    /// (`for(4900)`'s children are the inner `for(8)` and then two mask nodes).
    ///
    /// ⛔ `None` IS A LEAF, NOT AN ERROR — `isLeaf()` is this same read (`OperationTree.hpp:70`).
    /// `insertPTMaskOps`' `verifyLoopNest` walks from here and stops on null
    /// (`LoweringPTMasks.cpp:115-117`).
    pub fn first_child(&self, n: LoopMaskNodeId) -> Option<LoopMaskNodeId> {
        self.base.first_child(n.0).map(LoopMaskNodeId)
    }

    /// Replaces: e083_getNextSibling
    ///
    /// **083/384** `getNextSibling` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:42` (2L).
    ///
    /// ```cpp
    /// LoopMaskNode *getNextSibling() const {
    ///   return static_cast<LoopMaskNode *>(OperationNode::getNextSibling());
    /// }
    /// ```
    ///
    /// ⭐ THIS IS HOW EVERY CHILD LIST IS READ: `verifyLoopNest` iterates `curr_node =
    /// curr_node->getNextSibling()` over a loop's children (`LoweringPTMasks.cpp:116-130`), and the
    /// base's own `getLastChild`, `getPrevSibling`, `getNumberOfChildren` and BFS all walk this link.
    ///
    /// ⛔ `None` ON THE ROOT IS PART OF `getRoot`'s INVARIANT (`OperationTree.hpp:205-206`), which is
    /// why the root can be the head of no list.
    pub fn next_sibling(&self, n: LoopMaskNodeId) -> Option<LoopMaskNodeId> {
        self.base.next_sibling(n.0).map(LoopMaskNodeId)
    }

    /// Replaces: e084_getPrevSibling
    ///
    /// **084/384** `getPrevSibling` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:45` (2L).
    ///
    /// ```cpp
    /// LoopMaskNode *getPrevSibling() const {
    ///   return static_cast<LoopMaskNode *>(OperationNode::getPrevSibling());
    /// }
    /// ```
    ///
    /// ⛔⛔ THE ONLY ONE OF THE FIVE THAT IS NOT A FIELD READ. The base walks the parent's chain to
    /// find it (`OperationTree.cpp:37-46`) because no `prev_sibling_` field exists — so this is
    /// O(children), it needs the node to have a parent, and it answers `None` both for a first child
    /// and for the root. See [`OperationTreeBase::prev_sibling`] for what became of the
    /// `DT_CHECK_MSG(getParentNode(), "expected a parent")`.
    pub fn prev_sibling(&self, n: LoopMaskNodeId) -> Option<LoopMaskNodeId> {
        self.base.prev_sibling(n.0).map(LoopMaskNodeId)
    }

    /// Replaces: e085_getLastChild
    ///
    /// **085/384** `getLastChild` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:48` (2L).
    ///
    /// ```cpp
    /// LoopMaskNode *getLastChild() const {
    ///   return static_cast<LoopMaskNode *>(OperationNode::getLastChild());
    /// }
    /// ```
    ///
    /// ⭐ ALSO A WALK, AND IT IS WHAT MAKES APPENDING WORK: `insertChildNode` with no position calls
    /// `getLastChild()->setNextSibling(child)` (`OperationTree.cpp:251-252`), so every mask node the
    /// lowering adds lands at the end of its parent's list through this.
    ///
    /// ⛔ `None` IS A LEAF — the C++ returns the null `getFirstChild()` unchanged (`cpp:31-34`), and a
    /// caller that appends has already taken the `isLeaf()` branch instead.
    pub fn last_child(&self, n: LoopMaskNodeId) -> Option<LoopMaskNodeId> {
        self.base.last_child(n.0).map(LoopMaskNodeId)
    }

    /// WHICH KIND OF NODE THIS IS — `isLoopNode()`/`isMaskNode()` (`hpp:54-55`, `:79`, `:96`) and the
    /// `static_cast<MaskNode *>` that follows them (`LoopMaskTree.cpp:105`), as one `match`.
    ///
    /// ⛔ THE PAIR OF PREDICATES IS ON THE EXCLUDED LIST as field accessors; this is the read they
    /// become. Every C++ caller asks the question and then downcasts —
    /// `verifyLoopNest`'s `DT_CHECK_MSG(node->isLoopNode(), …)` (`LoweringPTMasks.cpp:115`),
    /// `print`'s three-way branch (`LoopMaskTree.cpp:102-111`) — and here the answer carries the
    /// payload with it.
    pub fn node(&self, n: LoopMaskNodeId) -> &LoopMaskNode {
        self.base.payload(n.0)
    }

    /// THE OPERATION A NODE NAMES — `LoopMaskNode::getOperation()`, the base's accessor
    /// (`OperationTree.hpp:37`).
    ///
    /// ⛔ `None` IS THE SYNTHETIC ROOT AND ONLY THE ROOT (see [`Node::new`], entry 079). Its readers
    /// treat it that way: `insertMaskOps` dereferences `n->getParentNode()->getOperation()` with no
    /// null test to get the loop it emits around (`LoweringPTMasks.cpp:74`), and it only ever asks
    /// that of a mask node's parent.
    pub fn operation(&self, n: LoopMaskNodeId) -> Option<&OpId> {
        self.base.operation(n.0)
    }

    /// Replaces: e078_findNodeFromOp
    ///
    /// **078/384** `findNodeFromOp` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:167` (4L).
    ///
    /// ```cpp
    /// LoopMaskNode *LoopMaskTree::findNodeFromOp(Operation *op) {
    ///   DT_CHECK_MSG(op, "valid op expected");
    ///   if (op_to_node_.find(op) == op_to_node_.end()) return nullptr;
    ///   return op_to_node_[op];
    /// }
    /// ```
    ///
    /// # ⭐⭐ THE MAP IS A MEMO, SO THE PORT ASKS THE QUESTION DIRECTLY
    ///
    /// A node NAMES its operation (entry 079), so *which node names this op* is answerable from the
    /// nodes alone — and `op_to_node_` is a cache of that answer, populated in lockstep with the two
    /// insertions that build the tree (`:152`, `:179`). Scanning the arena returns what the lookup
    /// returns, without a second structure to keep in step; see [`LoopMaskTree`] for why that
    /// trade is deliberate and what it costs.
    ///
    /// ⭐ THE TWO ANSWERS COINCIDE EXACTLY, and it is worth spelling out why rather than asserting it.
    /// The map holds an entry for every node the tree creates except the root — `computeLoops`
    /// inserts each loop node (`:179`) and `addMaskNode` each mask node (`:152`), while
    /// `new LoopMaskNode(nullptr)` (`:175`) inserts nothing — and the scan skips the root for the same
    /// reason, its operation being `None`. So *found in the map* and *found in the arena* are the same
    /// set.
    ///
    /// ⛔ `DT_CHECK_MSG(op, "valid op expected")` IS UNREPRESENTABLE: this takes `&OpId`, so there is
    /// no null to check. ⭐ AND THE `nullptr` RETURN IS A GENUINE ANSWER, NOT A FAILURE — `addMaskNode`
    /// checks it (`DT_CHECK_MSG(found_node, "could not locate loop_op in tree")`, `:144`), so it is
    /// `Option`, not a refusal.
    ///
    /// ⚠️ THE FIRST MATCH WINS, AND UNIQUENESS IS THE WRITER'S OBLIGATION. `addMaskNode`'s own
    /// `DT_CHECK_MSG(op_to_node_.find(mask_related_op) == op_to_node_.end(), "op already in map -
    /// should not happen")` (`:150-151`) is where the reference states that one op gets one node; that
    /// check belongs to entry 236, which is the unit that inserts.
    pub fn find_node_from_op(&self, op: &OpId) -> Option<LoopMaskNodeId> {
        self.base.find_by_op(op).map(LoopMaskNodeId)
    }

    /// Replaces: e077_walk
    ///
    /// **077/384** `walk` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:132` (2L).
    ///
    /// ```cpp
    /// void LoopMaskTree::walk(LoopMaskNode::ActionFuncTy action) {
    ///   LoopMaskNode::walk<OperationNode::WalkOrder::kBFS>(getRoot(), action);
    /// }
    /// ```
    ///
    /// # ⭐⭐ BREADTH-FIRST FROM THE ROOT, AND THE ORDER IS THE POINT
    ///
    /// The one caller is `insertPTMaskOps`, whose action `analyzeAndInsertMaskOps` looks at ONE node's
    /// children at a time and decides what mask ops to emit around it (`LoweringPTMasks.cpp:164-205`).
    /// Breadth-first is what makes that sound: a loop is visited before anything nested inside it, so
    /// when `verifyLoopNest` re-walks the nest FROM the node itself, once per loop child, to prove no
    /// second different mask lives there (`:114-138`), the nodes it inspects have not yet been lowered.
    /// ⚠️ Its `WalkResult::interrupt()` at `:128` is a DISCARDED temporary — `found_mask_node` at `:134`
    /// is what stops the sibling loop. The `visited_nodes` set at `:173` records what this order decides.
    ///
    /// # ⭐ THE `LoopMaskNode::walk<kBFS>` SPECIALIZATION IS THE CLOSURE BELOW
    ///
    /// The seven specializations at `LoopMaskTree.cpp:25-94` exist for one reason the header states
    /// outright — *"Callables cannot be type casted"* (`hpp:57-60`) — and each is the same three lines:
    /// wrap the derived action in a lambda that downcasts the node, then call the base's walk. Here
    /// `&mut |n| action(LoopMaskNodeId(n))` **is** that `action_wrapper` (`:68-71`), with the
    /// `static_cast<LoopMaskNode *>` being the newtype wrap. ⛔ The other six orders are unscheduled
    /// and unwritten: this file needs `kBFS` and nothing else.
    ///
    /// ⛔ THE ACTION RETURNS NOTHING BECAUSE `kBFS` DISCARDS THE RETURN, and one caller's apparent
    /// early exit is therefore not one — see [`OperationTreeBase::breadth_first_walk`].
    ///
    /// ⭐ AND IT TAKES `&self`, NOT `&mut self`, which is why an action may freely READ the tree it is
    /// walking: `analyzeAndInsertMaskOps` calls `getFirstChild`, `getNextSibling`, `getRoot` and
    /// `getStartVal` on it (`:166-180`). Two shared borrows coexist; a mutating action would need the
    /// C++'s own restriction (*the action can only remove descendants*) to become a type, and no
    /// caller wants one.
    pub fn walk(&self, action: &mut impl FnMut(LoopMaskNodeId)) {
        self.base
            .breadth_first_walk(self.base.root(), &mut |n| action(LoopMaskNodeId(n)));
    }

    /// A NODE'S MASK, IF IT HAS ONE — `n->isMaskNode()` followed by `static_cast<MaskNode *>(n)`,
    /// the pair every C++ caller writes (`LoweringPTMasks.cpp:123-125`, `:174-178`,
    /// `LoopMaskTree.cpp:104-105`).
    ///
    /// ⛔⛔ THIS IS WHERE THE DOWNCAST HAPPENS, ONCE, AND IT IS THE ONLY PLACE IT CAN. Minting a
    /// [`MaskNodeId`] copies the payload out at the same moment it classifies, so
    /// [`Self::is_mask_equivalent_to_node`] — whose C++ signature takes `MaskNode *` on both sides
    /// (`LoopMaskTree.hpp:84`) — has nothing left to check and no arm to answer for a node that turned
    /// out not to be a mask. ⭐ `Option` HERE IS THE PREDICATE, NOT A CHECKED NARROWING: `isMaskNode`
    /// is a question with two honest answers, and `None` is the one the reference spells `false`.
    #[must_use]
    pub fn mask_node(&self, n: LoopMaskNodeId) -> Option<MaskNodeId> {
        match self.node(n) {
            LoopMaskNode::Mask(mask) => Some(MaskNodeId {
                node: n,
                mask: *mask,
            }),
            LoopMaskNode::SyntheticRoot | LoopMaskNode::Loop(LMTLoopNode) => None,
        }
    }

    /// Replaces: e171_isMaskEquivalentToNode
    ///
    /// **171/384** `MaskNode::isMaskEquivalentToNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:123` (4L).
    ///
    /// ```cpp
    /// bool MaskNode::isMaskEquivalentToNode(MaskNode *n) {
    ///   return (n->getParentNode() == getParentNode() &&
    ///           n->getStartVal() == getStartVal() &&
    ///           n->getIncrement() == getIncrement());
    /// }
    /// ```
    ///
    /// # 🛑 THREE CLAUSES, AND THE FIRST ONE IS THE ONE THAT DECIDES
    ///
    /// ⭐⭐ "EQUIVALENT" MEANS **SAME PARENT LOOP** AND SAME MASK, WHICH MAKES IT A TEST FOR SIBLINGS.
    /// The reference's own worked example proves the parent clause is not redundant
    /// (`LoweringPTMasks.cpp:152-163`):
    ///
    /// ```text
    ///             A:[root]
    ///                /  \
    ///       B:[loop i]  C:[mask {2,0}]
    ///          /    \
    ///  D:[loop j]   E:[mask {0,1}]
    ///       |
    ///  F:[mask {0,1}]
    /// ```
    ///
    /// E and F carry `{0, 1}` each — identical on both of the fields `:125-126` compares — and the
    /// comment above the diagram still says *"We find another mask node so the program is not
    /// supported and we signal pass failure"* (`:160-163`). Only the parent comparison can produce that
    /// answer: E hangs off B and F off D. A port that compared the two masks alone would call them
    /// equivalent, and `insertMaskOps` would then emit ONE `set_mask`/`incrmask`/`set_mask 0` trio for
    /// two nests that each need their own.
    ///
    /// ⭐ AND THE CASE IT *DOES* ALLOW IS THE VENDOR'S OWN. `dynamic_pt_masking.mlir` has two macs
    /// under `for %arg4`, both masked by `%arg4` and so both `{0, 1}` with the SAME parent node — the
    /// reason the golden output brackets that loop once (`:38`, `:53`, `:56`) rather than twice. That is
    /// what *"If the mask is equivalent, it is allowed"* (`:124`) is for.
    ///
    /// ⛔ THE PARENTS ARE COMPARED AS IDENTITIES, NOT AS VALUES. The C++ `==` is on
    /// `LoopMaskNode *` — pointer equality — so two DIFFERENT loop nodes standing for the same
    /// `sentient.for` would still compare unequal. [`LoopMaskNodeId`] is that identity; comparing the
    /// operations instead would fuse nodes the tree keeps apart.
    ///
    /// ⭐ NEITHER PARENT NEEDS A NULL TEST, AND THE REFERENCE TAKES NONE. A mask node always has a
    /// parent — `addMaskNode` attaches it under a loop, or under the root when no loop encloses it
    /// (`LoopMaskTree.cpp:142-147`) — so `Option<LoopMaskNodeId>` here is compared, never unwrapped,
    /// and `None == None` is the two-root case the reference's `nullptr == nullptr` also calls equal.
    ///
    /// ⛔ THE RECEIVER AND THE ARGUMENT ARE INTERCHANGEABLE and the relation is an equivalence — all
    /// three clauses are symmetric. `verifyLoopNest` relies on it, calling
    /// `m->isMaskEquivalentToNode(mask_node)` with `m` from the walk and `mask_node` from the caller
    /// (`LoweringPTMasks.cpp:126`); which of the two is the receiver is not a decision anyone made.
    #[must_use]
    pub fn is_mask_equivalent_to_node(&self, this: MaskNodeId, other: MaskNodeId) -> bool {
        // `n->getParentNode() == getParentNode() &&`
        self.parent_node(other.node) == self.parent_node(this.node)
            // `n->getStartVal() == getStartVal() &&`
            && other.mask.start_val == this.mask.start_val
            // `n->getIncrement() == getIncrement());`
            && other.mask.increment == this.mask.increment
    }
}

/// A NODE KNOWN TO CARRY A MASK — the `MaskNode *` that `isMaskEquivalentToNode` takes on both sides
/// (`LoopMaskTree.hpp:84`), and that `insertMaskOps` takes as its only argument
/// (`LoweringPTMasks.cpp:45`, entry 239).
///
/// # 🛑 THE WITNESS AND ITS PAYLOAD TRAVEL TOGETHER
///
/// ⛔⛔ A `MaskNode *` IN THE C++ IS A CLAIM SOMEONE ELSE ALREADY CHECKED. Both call sites reach one
/// the same way — `if (n->isMaskNode()) { auto m = static_cast<MaskNode *>(n); … }`
/// (`LoweringPTMasks.cpp:123-125`, `:174-178`) — and after that the two `int`s are read with no
/// further test. Here [`LoopMaskTree::mask_node`] performs that pair once and copies the
/// [`MaskNode`] out with the identity, so no reader downcasts a second time and none needs an
/// `unreachable!` arm for a node that is not a mask. ⭐ A PAIRING IS A WITNESS CONSUMED ONCE, which
/// is this crate's rule and the reason the payload is a field rather than a lookup.
///
/// ⛔ THE IDENTITY IS STILL CARRIED, because the mask alone does not answer the question: the parent
/// clause of [`LoopMaskTree::is_mask_equivalent_to_node`] needs the node, and `insertMaskOps` needs
/// both the node's own operation and its parent's (`LoweringPTMasks.cpp:74`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaskNodeId {
    /// Which node it is — the `LoopMaskNode *` the `static_cast` started from.
    node: LoopMaskNodeId,
    /// `start_val_` and `increment_`, copied at the moment of the cast (`LoopMaskTree.hpp:87`).
    mask: MaskNode,
}

impl MaskNodeId {
    /// The node's identity, for the accessors [`LoopMaskTree`] takes a [`LoopMaskNodeId`] for.
    #[must_use]
    pub const fn node(self) -> LoopMaskNodeId {
        self.node
    }

    /// `getStartVal()` and `getIncrement()` (`LoopMaskTree.hpp:76-77`) — the two excluded field
    /// accessors, already downcast.
    #[must_use]
    pub const fn mask(self) -> MaskNode {
        self.mask
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{
        LMTLoopNode, LoopMaskNode, LoopMaskNodeId, LoopMaskTree, MaskIncrement, MaskNode,
        MaskNodeId, MaskedColumns, Node, OpId, OperationTreeBase,
    };

    /// HOW DEEP A NODE SITS — `OperationNode::getDepth()` (`OperationTree.hpp:62-67`), which is
    /// unscheduled and unported; the tests that need it count parents themselves.
    fn depth(tree: &LoopMaskTree, mut n: LoopMaskNodeId) -> usize {
        let mut d = 0;
        while let Some(parent) = tree.parent_node(n) {
            d += 1;
            n = parent;
        }
        d
    }

    /// A `sentient.for`'s node — [`LMTLoopNode`], as `computeLoops` mints it
    /// (`LoopMaskTree.cpp:178`).
    fn loop_node() -> LoopMaskNode {
        LoopMaskNode::Loop(LMTLoopNode)
    }

    /// A DYNAMIC MASK'S NODE — `addMaskNode(parent_op, *mac_op, 0, 1)`, the only shape
    /// `updateLoopMaskTreeForDynamicMask` is ever called with
    /// (`VectorChainToSentientPT.cpp:463-478`).
    fn dynamic_mask() -> LoopMaskNode {
        LoopMaskNode::Mask(MaskNode {
            start_val: MaskedColumns(0),
            increment: MaskIncrement::PerParentLoopIteration,
        })
    }

    /// A CONSTANT MASK'S NODE — `addMaskNode(parent_loop, *mac_op, mask_val, 0)`
    /// (`LoweringPTMasks.cpp:24-25`).
    fn constant_mask(columns: u32) -> LoopMaskNode {
        LoopMaskNode::Mask(MaskNode {
            start_val: MaskedColumns(columns),
            increment: MaskIncrement::Constant,
        })
    }

    /// THE VENDOR'S OWN LOOP MASK TREE, NODE BY NODE.
    ///
    /// ⭐⭐ THIS IS `@non_default_ops_to_insert` FROM
    /// `dcc/test/Conversion/VectorChainToSentientPT/dynamic_pt_masking.mlir:215-317`, whose
    /// `CHECK-SENT-IR` block (`:13-109`) is the reference's own output for it. The case is worth
    /// building whole because of what its shape contains: **two** top-level `sentient.for` nests in
    /// one `dataflow.program_unit`, so the synthetic root has two children and every sibling link is
    /// exercised; a loop with three children (a nested loop and two mask nodes); and both kinds of
    /// mask.
    ///
    /// ```text
    /// root                                     — new LoopMaskNode(nullptr)
    /// ├── n1_l1  sentient.for %arg1 = 2
    /// │   └── n1_l2  for %arg2 = 7
    /// │       └── n1_l3  for %arg3 = 4
    /// │           └── n1_l4  for %arg4 = 4900
    /// │               ├── n1_l5  for %arg5 = 8
    /// │               ├── n1_m1  MASK {start: 0, incr: 1}   — the mac in n1_l4's body
    /// │               └── n1_m2  MASK {start: 0, incr: 1}   — the mac inside n1_l5, masked by %arg4
    /// └── n2_l1  sentient.for %arg1 = 2
    ///     └── n2_l2  for %arg2 = 7
    ///         └── n2_l3  for %arg3 = 4
    ///             └── n2_l4  for %arg4 = 4900
    ///                 └── n2_l5  for %arg5 = 8
    ///                     ├── n2_l6  for %arg6 = 8
    ///                     │   └── n2_m_dyn  MASK {start: 0, incr: 1}
    ///                     └── n2_m_const    MASK {start: 2, incr: 0}
    /// ```
    ///
    /// ⛔ THE BUILD ORDER IS THE REFERENCE'S ORDER, AND IT DECIDES EVERY CHILD LIST. `computeLoops`
    /// creates all eleven loop nodes first, in a pre-order walk of the program unit
    /// (`LoopMaskTree.cpp:176-190`); the four mask nodes are appended one at a time as the lowering
    /// reaches each compute (`:136-152`). That is why `n1_l4`'s children are `[n1_l5, n1_m1, n1_m2]`
    /// and `n2_l5`'s are `[n2_l6, n2_m_const]` — the nested loop precedes a mask that the source
    /// wrote first.
    ///
    /// ⛔ AND THE MASK PARENTS ARE THE REFERENCE'S, NOT THE SOURCE'S NESTING. `n1_m2` hangs off
    /// `n1_l4` although its `vector_mac` sits inside `n1_l5`, because the mask is driven by `%arg4`
    /// and `updateLoopMaskTreeForDynamicMask` attaches the node to the loop that owns the block
    /// argument (`LoweringPTMasks.cpp:28-38`, `dynamic_pt_masking.mlir:260`). The vendor output
    /// confirms the consequence: ONE `set_mask`/`incrmask`/`set_mask 0` trio around `for %arg4`
    /// (`:38`, `:53`, `:56`) rather than two.
    struct Vendor {
        tree: LoopMaskTree,
        root: LoopMaskNodeId,
        n1_l1: LoopMaskNodeId,
        n1_l2: LoopMaskNodeId,
        n1_l3: LoopMaskNodeId,
        n1_l4: LoopMaskNodeId,
        n1_l5: LoopMaskNodeId,
        n1_m1: LoopMaskNodeId,
        n1_m2: LoopMaskNodeId,
        n2_l1: LoopMaskNodeId,
        n2_l2: LoopMaskNodeId,
        n2_l3: LoopMaskNodeId,
        n2_l4: LoopMaskNodeId,
        n2_l5: LoopMaskNodeId,
        n2_l6: LoopMaskNodeId,
        n2_m_const: LoopMaskNodeId,
        n2_m_dyn: LoopMaskNodeId,
    }

    impl Vendor {
        fn build() -> Self {
            let mut base = OperationTreeBase::with_root(LoopMaskNode::SyntheticRoot);
            let root = base.root();

            // ── `computeLoops`: every `sentient.for`, in pre-order ────────────────────────────────
            //
            // ⭐⭐ THE PATHS ARE THE VENDOR CASE'S OWN POSITIONS, counted inside the
            // `dataflow.program_unit` body (`dynamic_pt_masking.mlir:225-315`), one ordinal per region
            // level. The first nest's `sentient.for %arg1` is the third op of the body — two `arith`
            // ops set up its bound first (`:226-228`) — and the second nest's is the sixth (`:268-270`).
            let n1_l1 = base.push_named_child(root, OpId::at(&[2]), loop_node());
            let n1_l2 = base.push_named_child(n1_l1, OpId::at(&[2, 3]), loop_node());
            let n1_l3 = base.push_named_child(n1_l2, OpId::at(&[2, 3, 3]), loop_node());
            let n1_l4 = base.push_named_child(n1_l3, OpId::at(&[2, 3, 3, 3]), loop_node());
            // ⭐ THE FIFTEENTH OP OF `%arg4`'s BODY: eight setup ops, two receives, a select, a mask, the
            // mac and a store come first (`:241-254`), then `sentient.for %arg5` (`:255`).
            let n1_l5 = base.push_named_child(n1_l4, OpId::at(&[2, 3, 3, 3, 14]), loop_node());
            let n2_l1 = base.push_named_child(root, OpId::at(&[5]), loop_node());
            let n2_l2 = base.push_named_child(n2_l1, OpId::at(&[5, 3]), loop_node());
            let n2_l3 = base.push_named_child(n2_l2, OpId::at(&[5, 3, 3]), loop_node());
            let n2_l4 = base.push_named_child(n2_l3, OpId::at(&[5, 3, 3, 3]), loop_node());
            let n2_l5 = base.push_named_child(n2_l4, OpId::at(&[5, 3, 3, 3, 8]), loop_node());
            let n2_l6 = base.push_named_child(n2_l5, OpId::at(&[5, 3, 3, 3, 8, 9]), loop_node());

            // ── then `addMaskNode`, once per masked compute, in lowering order ────────────────────
            //
            // ⛔ A MASK NODE NAMES THE COMPUTE, NOT THE LOOP. `addMaskNode(loop_op, mask_related_op,
            // …)` builds the node over the SECOND argument (`LoopMaskTree.cpp:138`) and attaches it
            // under the first — `updateLoopMaskTreeForDynamicMask` passes the
            // `vectorchain.multiply_and_accumulate` (`VectorChainToSentientPT.cpp:463-478`).
            let n1_m1 = base.push_named_child(n1_l4, OpId::at(&[2, 3, 3, 3, 12]), dynamic_mask());
            // ⭐ ITS MAC IS INSIDE `%arg5` (`:261`) WHILE ITS PARENT NODE IS `%arg4` — the mask is
            // driven by `%14`, which `%arg4` owns.
            let n1_m2 = base.push_named_child(n1_l4, OpId::at(&[2, 3, 3, 3, 14, 5]), dynamic_mask());
            // ⭐ `#set2 = affine_set<(d0) : (d0 - 48 >= 0, -d0 + 63 >= 0)>` over `vector<64xf16>`:
            // `(64 - 48) / 8 = 2`, the `scalar_constant {value = 2}` the golden's `set_mask` takes.
            let n2_m_const = base.push_named_child(n2_l5, OpId::at(&[5, 3, 3, 3, 8, 5]), constant_mask(2));
            let n2_m_dyn =
                base.push_named_child(n2_l6, OpId::at(&[5, 3, 3, 3, 8, 9, 5]), dynamic_mask());

            Self {
                tree: LoopMaskTree { base },
                root: LoopMaskNodeId(root),
                n1_l1: LoopMaskNodeId(n1_l1),
                n1_l2: LoopMaskNodeId(n1_l2),
                n1_l3: LoopMaskNodeId(n1_l3),
                n1_l4: LoopMaskNodeId(n1_l4),
                n1_l5: LoopMaskNodeId(n1_l5),
                n1_m1: LoopMaskNodeId(n1_m1),
                n1_m2: LoopMaskNodeId(n1_m2),
                n2_l1: LoopMaskNodeId(n2_l1),
                n2_l2: LoopMaskNodeId(n2_l2),
                n2_l3: LoopMaskNodeId(n2_l3),
                n2_l4: LoopMaskNodeId(n2_l4),
                n2_l5: LoopMaskNodeId(n2_l5),
                n2_l6: LoopMaskNodeId(n2_l6),
                n2_m_const: LoopMaskNodeId(n2_m_const),
                n2_m_dyn: LoopMaskNodeId(n2_m_dyn),
            }
        }
    }

    /// 🎯 088 — `getRoot` NAMES THE SYNTHETIC ROOT, AND THE `DT_CHECK`'S THREE CONJUNCTS HOLD.
    ///
    /// `DT_CHECK(root_ && root_->getNextSibling() == nullptr && root_->getParentNode() == nullptr)`
    /// (`OperationTree.hpp:205-206`) is the invariant the C++ re-tests on every call; here it is
    /// asserted once because nothing can break it. The root is also the one node that is neither a
    /// loop nor a mask — `print`'s third arm (`LoopMaskTree.cpp:109-111`).
    #[test]
    fn the_root_is_the_synthetic_node_and_has_no_parent_and_no_sibling() {
        let v = Vendor::build();
        let root = v.tree.root();

        assert_eq!(
            root, v.root,
            "the root is the node the tree was built around"
        );
        assert_eq!(v.tree.node(root), &LoopMaskNode::SyntheticRoot);
        assert_eq!(
            v.tree.parent_node(root),
            None,
            "root_->getParentNode() == nullptr"
        );
        assert_eq!(
            v.tree.next_sibling(root),
            None,
            "root_->getNextSibling() == nullptr"
        );
        // ⛔ AND THE ROOT IS NOT A LOOP OR A MASK, so a reader that assumed two kinds would read the
        // program unit itself as a loop.
        assert!(!matches!(
            v.tree.node(root),
            LoopMaskNode::Loop(_) | LoopMaskNode::Mask(_)
        ));
    }

    /// 🎯 081 — THE PARENT OF A MASK NODE IS THE LOOP THAT CONTROLS THE MASK.
    ///
    /// This is the read `insertMaskOps` makes before it emits anything —
    /// `n->getParentNode()->getOperation()` (`LoweringPTMasks.cpp:74`) — so `n1_m2`'s parent being
    /// `n1_l4` rather than `n1_l5` is what puts the `set_mask` outside `for %arg4` in the vendor
    /// output (`dynamic_pt_masking.mlir:38`).
    #[test]
    fn a_mask_nodes_parent_is_the_loop_that_drives_it() {
        let v = Vendor::build();

        assert_eq!(v.tree.parent_node(v.n1_m1), Some(v.n1_l4));
        assert_eq!(
            v.tree.parent_node(v.n1_m2),
            Some(v.n1_l4),
            "driven by %arg4, not %arg5"
        );
        assert_eq!(v.tree.parent_node(v.n2_m_const), Some(v.n2_l5));
        assert_eq!(v.tree.parent_node(v.n2_m_dyn), Some(v.n2_l6));

        // ⭐ AND THE LOOP CHAIN IS THE SOURCE NESTING, top-level loops parented to the root.
        assert_eq!(v.tree.parent_node(v.n1_l5), Some(v.n1_l4));
        assert_eq!(v.tree.parent_node(v.n1_l4), Some(v.n1_l3));
        assert_eq!(v.tree.parent_node(v.n1_l3), Some(v.n1_l2));
        assert_eq!(v.tree.parent_node(v.n1_l2), Some(v.n1_l1));
        assert_eq!(v.tree.parent_node(v.n1_l1), Some(v.root));
        assert_eq!(v.tree.parent_node(v.n2_l1), Some(v.root));
    }

    /// 🎯 082 — THE FIRST CHILD OF A LOOP IS ITS NESTED LOOP, NOT ITS MASK.
    ///
    /// ⛔ THE ORDER IS NOT A TIE TO BREAK: `computeLoops` runs to completion before any mask node
    /// exists (`LoopMaskTree.cpp:176-190` at construction, `:136-152` during lowering), so a mask can
    /// never precede a loop in a child list. `verifyLoopNest` walks from here expecting exactly that
    /// (`LoweringPTMasks.cpp:115-130`).
    #[test]
    fn the_first_child_is_the_nested_loop_and_a_leaf_has_none() {
        let v = Vendor::build();

        assert_eq!(
            v.tree.first_child(v.root),
            Some(v.n1_l1),
            "the first nest, in program order"
        );
        assert_eq!(
            v.tree.first_child(v.n1_l4),
            Some(v.n1_l5),
            "the loop, ahead of two masks"
        );
        assert_eq!(
            v.tree.first_child(v.n2_l5),
            Some(v.n2_l6),
            "the loop, ahead of the const mask"
        );
        assert_eq!(
            v.tree.first_child(v.n2_l6),
            Some(v.n2_m_dyn),
            "a loop whose only child is a mask"
        );

        // ⛔ `isLeaf()` IS THIS SAME READ (`OperationTree.hpp:70`): the innermost loop of nest 1 holds
        // no mask node of its own, because its compute's mask is driven from outside it.
        assert_eq!(v.tree.first_child(v.n1_l5), None);
        assert_eq!(
            v.tree.first_child(v.n1_m1),
            None,
            "a mask node never has children"
        );
    }

    /// 🎯 083 — THE SIBLING CHAIN IS THE CHILD LIST, AND IT ENDS IN `None`.
    ///
    /// `verifyLoopNest` reads a loop's children exactly this way (`LoweringPTMasks.cpp:116-130`), and
    /// the two top-level nests being siblings is what the synthetic root exists for
    /// (`OperationTree.hpp:195-200`).
    #[test]
    fn the_next_sibling_chain_walks_a_child_list_to_its_end() {
        let v = Vendor::build();

        // The root's children: the two top-level `sentient.for` nests.
        assert_eq!(v.tree.next_sibling(v.n1_l1), Some(v.n2_l1));
        assert_eq!(v.tree.next_sibling(v.n2_l1), None);

        // `for %arg4 = 4900`'s three children, in insertion order.
        assert_eq!(v.tree.next_sibling(v.n1_l5), Some(v.n1_m1));
        assert_eq!(v.tree.next_sibling(v.n1_m1), Some(v.n1_m2));
        assert_eq!(v.tree.next_sibling(v.n1_m2), None);

        // An only child has no sibling at all.
        assert_eq!(v.tree.next_sibling(v.n1_l2), None);
        assert_eq!(v.tree.next_sibling(v.n2_m_dyn), None);
    }

    /// 🎯 084 — `getPrevSibling` WALKS, AND ANSWERS `None` FOR A FIRST CHILD AND FOR THE ROOT.
    ///
    /// The C++ has no `prev_sibling_` field: it re-derives the answer from the parent's chain
    /// (`OperationTree.cpp:37-46`). ⛔ `n1_m2`'s predecessor is found only after two hops through
    /// that chain, which is the loop this port has to reproduce rather than a field to read.
    #[test]
    fn the_prev_sibling_is_found_by_walking_the_parents_chain() {
        let v = Vendor::build();

        // Two iterations of the `while (prev)` loop: first_child is `n1_l5`, then `n1_m1`.
        assert_eq!(v.tree.prev_sibling(v.n1_m2), Some(v.n1_m1));
        assert_eq!(v.tree.prev_sibling(v.n1_m1), Some(v.n1_l5));
        assert_eq!(v.tree.prev_sibling(v.n2_l1), Some(v.n1_l1));

        // ⛔ `if (prev == this) return nullptr` (`cpp:40`) — a first child has no predecessor.
        assert_eq!(v.tree.prev_sibling(v.n1_l5), None);
        assert_eq!(v.tree.prev_sibling(v.n1_l1), None);
        assert_eq!(v.tree.prev_sibling(v.n2_l6), None);

        // ⛔⛔ AND THE ROOT ANSWERS `None` RATHER THAN ABORTING — the C++'s
        // `DT_CHECK_MSG(getParentNode(), "expected a parent")` (`cpp:38`) is reachable only here, and
        // the root having no siblings is `getRoot`'s own invariant (`hpp:205-206`).
        assert_eq!(v.tree.prev_sibling(v.tree.root()), None);
    }

    /// 🎯 085 — THE LAST CHILD IS WHERE THE NEXT MASK NODE WILL LAND.
    ///
    /// `insertChildNode` with no position appends through this
    /// (`getLastChild()->setNextSibling(child)`, `OperationTree.cpp:251-252`), so it is also the
    /// answer to *what did `addMaskNode` add last*.
    #[test]
    fn the_last_child_is_the_end_of_the_chain() {
        let v = Vendor::build();

        assert_eq!(v.tree.last_child(v.root), Some(v.n2_l1), "the second nest");
        assert_eq!(
            v.tree.last_child(v.n1_l4),
            Some(v.n1_m2),
            "the second mask appended"
        );
        assert_eq!(v.tree.last_child(v.n2_l5), Some(v.n2_m_const));
        assert_eq!(
            v.tree.last_child(v.n2_l6),
            Some(v.n2_m_dyn),
            "one child is also the last"
        );

        // ⛔ A LEAF'S LAST CHILD IS THE SAME `None` ITS FIRST CHILD IS (`cpp:31-34`).
        assert_eq!(v.tree.last_child(v.n1_l5), None);
        assert_eq!(v.tree.last_child(v.n1_m2), None);
    }

    /// 🎯 086 — A MASK NODE CARRIES ITS START AND ITS INCREMENT, AND THEY ARE WHAT THE VENDOR EMITS.
    ///
    /// ⭐ THE CONSTANT MASK'S `start_val` IS THE VENDOR'S OWN NUMBER: `#set2`'s lower bound 48 over a
    /// `vector<64xf16>` gives `(64 - 48) / 8 = 2` (`Helper.cpp:99-105`), and the golden's
    /// `set_mask mask_value(%VAL_62)` takes `scalar_constant {value = 2 : si64}`
    /// (`dynamic_pt_masking.mlir:86-87`). ⛔ A dynamic mask starts at 0 and steps by one instead, so
    /// the two fields together are what decide which of `insertMaskOps`' two emission shapes runs.
    ///
    /// ⛔ THE EMPTY DESTRUCTOR ITSELF IS GUARDED AT BUILD TIME, not here: see the
    /// `const _: [(); 0]` beside the type. This test covers what the destructor must NOT dispose of.
    #[test]
    fn a_mask_node_carries_the_start_and_increment_the_vendor_emits() {
        let v = Vendor::build();

        assert_eq!(
            v.tree.node(v.n2_m_const),
            &LoopMaskNode::Mask(MaskNode {
                start_val: MaskedColumns(2),
                increment: MaskIncrement::Constant,
            }),
        );
        assert_eq!(
            v.tree.node(v.n1_m1),
            &LoopMaskNode::Mask(MaskNode {
                start_val: MaskedColumns(0),
                increment: MaskIncrement::PerParentLoopIteration,
            }),
        );

        // ⛔ AND A NODE'S OWN STATE IS SELF-CONTAINED: the payload copies out by value — the type is
        // `Copy` exactly because `~MaskNode() {}` releases nothing — and it carries no link with it,
        // so nothing that happens to a copy can reach the tree. That is what `clear()`'s post-order
        // walk relies on when it deletes the nodes one at a time (`OperationTree.cpp:256`).
        let copied = *v.tree.node(v.n1_m2);
        assert_eq!(
            copied,
            *v.tree.node(v.n1_m1),
            "the two dynamic masks carry the same state"
        );
        assert_eq!(v.tree.next_sibling(v.n1_m1), Some(v.n1_m2));
        assert_eq!(v.tree.parent_node(v.n1_m2), Some(v.n1_l4));
    }

    /// 🎯 087 — A LOOP NODE ADDS NO STATE, AND THE KIND IS WHAT TELLS IT FROM A MASK.
    ///
    /// `LMTLoopNode` declares no members (`LoopMaskTree.hpp:90-97`): everything it holds is the base's.
    /// ⛔ `isLoopNode()`/`isMaskNode()` are the excluded accessor pair, and the whole vendor tree is
    /// the count they answer — eleven `sentient.for`s and four masks, with the synthetic root neither.
    #[test]
    fn a_loop_node_adds_no_state_and_the_tree_is_eleven_loops_and_four_masks() {
        let v = Vendor::build();

        assert_eq!(v.tree.node(v.n1_l1), &LoopMaskNode::Loop(LMTLoopNode));
        assert_eq!(
            core::mem::size_of::<LMTLoopNode>(),
            0,
            "the class adds no members to LoopMaskNode",
        );

        // A pre-order walk over the five ported accessors alone. ⛔ NOT the ported `walk` — that is
        // entry 077, and a test may not stand in for it.
        let mut loops = 0;
        let mut masks = 0;
        let mut stack = vec![v.tree.root()];
        while let Some(n) = stack.pop() {
            match v.tree.node(n) {
                LoopMaskNode::SyntheticRoot => {
                    assert_eq!(n, v.tree.root(), "only the root is neither")
                }
                LoopMaskNode::Loop(LMTLoopNode) => loops += 1,
                LoopMaskNode::Mask(_) => masks += 1,
            }
            let mut child = v.tree.first_child(n);
            while let Some(c) = child {
                stack.push(c);
                child = v.tree.next_sibling(c);
            }
        }
        assert_eq!((loops, masks), (11, 4));
    }

    /// 🎯 079 — EVERY NODE NAMES ITS OPERATION, AND THE ROOT NAMES NONE.
    ///
    /// `OperationNode(Operation *op) : operation_op_(op)` (`OperationTree.hpp:29`) is the whole of
    /// entry 079, and `new LoopMaskNode(nullptr)` (`LoopMaskTree.cpp:175`) is the one call that
    /// passes null. This is the fact `print` branches on (`OperationTree.cpp:125-130`) and the fact
    /// that makes `findNodeFromOp` a total function over the arena.
    #[test]
    fn only_the_synthetic_root_names_no_operation() {
        let v = Vendor::build();

        assert_eq!(v.tree.operation(v.root), None, "new LoopMaskNode(nullptr)");

        let mut nodes: Vec<LoopMaskNodeId> = Vec::new();
        v.tree.walk(&mut |n| nodes.push(n));
        let nameless: Vec<LoopMaskNodeId> = nodes
            .iter()
            .copied()
            .filter(|&n| v.tree.operation(n).is_none())
            .collect();
        assert_eq!(
            nameless,
            vec![v.root],
            "every node but the root was minted over an op"
        );
    }

    /// 🎯 079 — THE NODE KEEPS THE POSITION IT WAS MINTED WITH, AND A MASK NODE'S IS ITS **COMPUTE**.
    ///
    /// `addMaskNode(loop_op, mask_related_op, …)` builds the node over `mask_related_op`
    /// (`LoopMaskTree.cpp:138`) and inserts it under `loop_op` (`:145`), so the two can disagree — and
    /// in the vendor's own case they do. `n1_m2`'s mac sits inside `for %arg5`
    /// (`dynamic_pt_masking.mlir:261`) while its tree parent is `for %arg4`, which owns the `%14` the
    /// mask is affine in. ⛔ THAT DISAGREEMENT IS THE WHOLE REASON THE OP AND THE LINKS ARE SEPARATE
    /// STATE: a reader that inferred the parent from the position would put the `set_mask` one loop
    /// too deep.
    #[test]
    fn a_mask_node_names_its_compute_while_its_parent_is_the_driving_loop() {
        let v = Vendor::build();

        let mac = v.tree.operation(v.n1_m2).expect("a mask names its compute");
        assert_eq!(mac.path(), &[2, 3, 3, 3, 14, 5]);

        // ⭐ THE MAC IS INSIDE `n1_l5`: the loop's position is a prefix of the compute's.
        let inner = v.tree.operation(v.n1_l5).expect("a loop names its for");
        assert_eq!(inner.path(), &[2, 3, 3, 3, 14]);
        assert!(mac.path().starts_with(inner.path()));

        // ⛔ YET THE TREE PARENT IS `n1_l4`, one level out.
        assert_eq!(v.tree.parent_node(v.n1_m2), Some(v.n1_l4));
        assert_eq!(
            v.tree
                .operation(v.n1_l4)
                .expect("a loop names its for")
                .path(),
            &[2, 3, 3, 3]
        );
    }

    /// 🎯 080 — DESTROYING A NODE'S PAYLOAD RUNS NO CODE, AND THE ARENA NODE IS THE EXCEPTION.
    ///
    /// `virtual ~LoopMaskNode() {}` (`LoopMaskTree.hpp:34`) says the first half; the build-time guard
    /// beside the type is what enforces it. What this test adds is the SECOND half, which no guard can
    /// state: `Node<LoopMaskNode>` *does* need dropping, because entry 079 gave it an [`OpId`] that
    /// owns its path. Pointing the guard at the wrong one of the two would silently stop guarding
    /// anything.
    #[test]
    fn a_loop_mask_node_needs_no_destructor_but_its_arena_node_does() {
        assert!(
            !core::mem::needs_drop::<LoopMaskNode>(),
            "~LoopMaskNode() {{}} — and ~MaskNode/~LMTLoopNode below it"
        );
        assert!(
            core::mem::needs_drop::<Node<LoopMaskNode>>(),
            "an arena node owns an OpId's path"
        );
    }

    /// 🎯 078 — EVERY NODE IN THE TREE IS FOUND FROM THE OPERATION IT NAMES.
    ///
    /// This is the round trip `addMaskNode` depends on: it looks the loop up by its op and asserts it
    /// is there (`DT_CHECK_MSG(found_node, "could not locate loop_op in tree")`,
    /// `LoopMaskTree.cpp:143-144`). The scan replaces `op_to_node_`, so the property to hold is that
    /// the scan and the map name the same set — every node except the root.
    #[test]
    fn every_named_node_is_found_from_its_operation() {
        let v = Vendor::build();

        let mut nodes: Vec<LoopMaskNodeId> = Vec::new();
        v.tree.walk(&mut |n| nodes.push(n));
        assert_eq!(nodes.len(), 16, "1 root + 11 loops + 4 masks");

        for n in nodes {
            match v.tree.operation(n) {
                Some(op) => assert_eq!(
                    v.tree.find_node_from_op(op),
                    Some(n),
                    "the node that names an op is the node found from it"
                ),
                None => assert_eq!(n, v.root, "only the root is nameless"),
            }
        }
    }

    /// 🎯 078 — AN OPERATION THE TREE DOES NOT NAME IS NOT FOUND, AND THAT IS AN ANSWER.
    ///
    /// `findNodeFromOp` returns `nullptr` for a miss (`:169`) and its caller tests the result
    /// (`:144`), so this is `None`, not a refusal. The ops chosen are real ones from the vendor case
    /// that the tree deliberately holds no node for: the `vector.store` at
    /// `dynamic_pt_masking.mlir:254`, the `dataflow.receive` at `:249`, and the whole program unit's
    /// body position — nothing but `sentient.for`s and masked computes gets a node
    /// (`if (!isa<sentient::ForOp>(op)) return;`, `LoopMaskTree.cpp:177`).
    #[test]
    fn an_operation_with_no_node_is_not_found() {
        let v = Vendor::build();

        for path in [
            &[2, 3, 3, 3, 13][..], // the `vector.store` after the mac
            &[2, 3, 3, 3, 8][..],  // a `dataflow.receive`
            &[0][..],              // the `arith.subi` that computes a loop bound
            &[2, 3, 3, 3, 14, 4][..], // the `create_affine_mask`, not the mac
        ] {
            assert_eq!(
                v.tree.find_node_from_op(&OpId::at(path)),
                None,
                "no node names {path:?}"
            );
        }
    }

    /// 🎯 078 — THE MAC INSIDE THE INNER LOOP FINDS ITS **MASK** NODE, NOT THE LOOP AROUND IT.
    ///
    /// Two nodes have positions in the same nest and the lookup must not confuse them: `n1_l5` names
    /// `for %arg5` at `[2, 3, 3, 3, 14]` and `n1_m2` names the mac inside it at
    /// `[2, 3, 3, 3, 14, 5]`. A prefix match would return the loop for both — the reference's
    /// `DenseMap` keys on pointer identity, and the port keys on the WHOLE path.
    #[test]
    fn a_lookup_distinguishes_a_compute_from_the_loop_containing_it() {
        let v = Vendor::build();

        assert_eq!(
            v.tree.find_node_from_op(&OpId::at(&[2, 3, 3, 3, 14, 5])),
            Some(v.n1_m2)
        );
        assert_eq!(
            v.tree.find_node_from_op(&OpId::at(&[2, 3, 3, 3, 14])),
            Some(v.n1_l5)
        );
        // ⭐ AND THE TWO NESTS ARE NOT CONFUSED EITHER, though their bodies are identical in shape:
        // `%arg4`'s two positions differ only in the outermost ordinal (`:240` vs `:282`).
        assert_eq!(
            v.tree.find_node_from_op(&OpId::at(&[5, 3, 3, 3])),
            Some(v.n2_l4)
        );
        assert_eq!(
            v.tree.find_node_from_op(&OpId::at(&[2, 3, 3, 3])),
            Some(v.n1_l4)
        );
    }

    /// 🎯 077 — THE WALK IS BREADTH-FIRST FROM THE ROOT, IN THE REFERENCE'S EXACT ORDER.
    ///
    /// `LoopMaskNode::walk<kBFS>(getRoot(), action)` (`LoopMaskTree.cpp:133`) over the queue at
    /// `OperationTree.cpp:185-197`: each node's children are pushed in sibling order after it, so the
    /// two nests interleave level by level. ⭐⭐ THE ORDER IS LOAD-BEARING, NOT INCIDENTAL — the one
    /// caller decides at `for %arg4` whether the masks below it are consistent and remembers the
    /// answer (`visited_nodes`, `LoweringPTMasks.cpp:173`), which only works if a loop is visited
    /// before everything nested in it.
    #[test]
    fn the_walk_visits_the_tree_level_by_level_from_the_root() {
        let v = Vendor::build();

        let mut order: Vec<LoopMaskNodeId> = Vec::new();
        v.tree.walk(&mut |n| order.push(n));

        assert_eq!(
            order,
            vec![
                v.root,
                // level 1 — the two top-level nests, in program order
                v.n1_l1,
                v.n2_l1,
                v.n1_l2,
                v.n2_l2,
                v.n1_l3,
                v.n2_l3,
                // level 4 — `for %arg4` in both nests
                v.n1_l4,
                v.n2_l4,
                // level 5 — the first nest's three children come before the second nest's one
                v.n1_l5,
                v.n1_m1,
                v.n1_m2,
                v.n2_l5,
                // level 6
                v.n2_l6,
                v.n2_m_const,
                // level 7
                v.n2_m_dyn,
            ]
        );

        // ⛔ AND DEPTH NEVER DECREASES ALONG THE ORDER, which is what breadth-first means and what a
        // pre-order walk of the same tree would violate at `n1_l5` → `n2_l1`.
        let depths: Vec<usize> = order.iter().map(|&n| depth(&v.tree, n)).collect();
        assert!(depths.windows(2).all(|w| w[0] <= w[1]), "{depths:?}");
        assert_eq!(depths.first(), Some(&0), "the root is depth 0");
        assert_eq!(depths.last(), Some(&7), "the deepest mask is seven down");
    }

    /// 🎯 077 — AN ACTION READS THE TREE IT IS WALKING, WHICH IS WHAT THE ONE CALLER DOES.
    ///
    /// `analyzeAndInsertMaskOps` visits each loop node and then walks that node's children looking for
    /// mask nodes, reading `getFirstChild`, `getNextSibling`, `getStartVal` and `getIncrement` as it
    /// goes (`LoweringPTMasks.cpp:164-201`). This is that shape: the closure holds a shared borrow of
    /// the tree while `walk` holds one too, and it reproduces the caller's answer — which loops carry
    /// masks directly beneath them, and how many.
    #[test]
    fn an_action_may_read_the_tree_while_walking_it() {
        let v = Vendor::build();

        let mut masked_loops: Vec<(LoopMaskNodeId, usize)> = Vec::new();
        v.tree.walk(&mut |n| {
            // `if (!n->isLoopNode() && n != pt_masking_tree->getRoot()) return n;` (`:166`).
            if !matches!(v.tree.node(n), LoopMaskNode::Loop(_)) && n != v.tree.root() {
                return;
            }
            let mut masks = 0;
            let mut child = v.tree.first_child(n);
            while let Some(c) = child {
                if matches!(v.tree.node(c), LoopMaskNode::Mask(_)) {
                    masks += 1;
                }
                child = v.tree.next_sibling(c);
            }
            if masks > 0 {
                masked_loops.push((n, masks));
            }
        });

        // ⭐ THREE LOOPS EMIT MASK OPS, and `n1_l4` is the one that emits for two masks at once — the
        // pair the vendor output collapses into a single `set_mask`/`incrmask` trio
        // (`dynamic_pt_masking.mlir:38`, `:53`, `:56`).
        assert_eq!(
            masked_loops,
            vec![(v.n1_l4, 2), (v.n2_l5, 1), (v.n2_l6, 1)]
        );
    }

    /// 🎯 077 — THE ACTION'S RETURN IS DISCARDED, SO A BFS CANNOT BE STOPPED EARLY.
    ///
    /// `(void)action(curr_node)` (`OperationTree.cpp:190`) is the line, and it is worth a test because
    /// the one caller reads as though it exits: `analyzeAndInsertMaskOps` does
    /// `signalPassFailure(); … return nullptr;` (`LoweringPTMasks.cpp:193-197`). Under `kBFS` the
    /// remaining queue is visited anyway. ⛔ SO THE PORT'S ACTION RETURNS `()` — a `-> Option<…>` here
    /// would advertise a steering the walk does not honour, and the two GUIDED orders that do honour
    /// it are unscheduled.
    #[test]
    fn the_walk_continues_after_the_action_would_have_returned_null() {
        let v = Vendor::build();

        // The caller's shape: bail out at the first dynamic mask, as `analyzeAndInsertMaskOps` does
        // when `verifyLoopNest` fails.
        let mut seen_after_bail = 0;
        let mut bailed = false;
        v.tree.walk(&mut |n| {
            if bailed {
                seen_after_bail += 1;
                return;
            }
            if let LoopMaskNode::Mask(mask) = v.tree.node(n)
                && mask.increment == MaskIncrement::PerParentLoopIteration
            {
                bailed = true;
            }
        });

        assert!(bailed, "the first dynamic mask is n1_m1");
        assert_eq!(
            seen_after_bail, 5,
            "n1_m2, n2_l5, n2_l6, n2_m_const and n2_m_dyn are all still visited"
        );
    }

    /// THE REFERENCE'S OWN WORKED EXAMPLE, NODE FOR NODE — the diagram `analyzeAndInsertMaskOps`
    /// carries above itself (`LoweringPTMasks.cpp:152-158`):
    ///
    /// ```text
    ///             A:[root]
    ///                /  \
    ///       B:[loop i]  C:[mask {2,0}]
    ///          /    \
    ///  D:[loop j]   E:[mask {0,1}]
    ///       |
    ///  F:[mask {0,1}]
    /// ```
    ///
    /// ⭐ WORTH BUILDING BESIDE [`Vendor`] BECAUSE OF WHAT IT ISOLATES: two masks that are identical on
    /// BOTH compared fields and differ only in their parent. The vendor case has no such pair — its
    /// cross-nest masks differ in start value too — so it cannot tell the parent clause apart from the
    /// other two.
    struct Diagram {
        tree: LoopMaskTree,
        b_loop_i: LoopMaskNodeId,
        c_mask: LoopMaskNodeId,
        d_loop_j: LoopMaskNodeId,
        e_mask: LoopMaskNodeId,
        f_mask: LoopMaskNodeId,
    }

    impl Diagram {
        fn build() -> Self {
            let mut base = OperationTreeBase::with_root(LoopMaskNode::SyntheticRoot);
            let root = base.root();
            let b = base.push_named_child(root, OpId::at(&[0]), loop_node());
            let c = base.push_named_child(root, OpId::at(&[1]), constant_mask(2));
            let d = base.push_named_child(b, OpId::at(&[0, 0]), loop_node());
            let e = base.push_named_child(b, OpId::at(&[0, 1]), dynamic_mask());
            let f = base.push_named_child(d, OpId::at(&[0, 0, 0]), dynamic_mask());
            Self {
                tree: LoopMaskTree { base },
                b_loop_i: LoopMaskNodeId(b),
                c_mask: LoopMaskNodeId(c),
                d_loop_j: LoopMaskNodeId(d),
                e_mask: LoopMaskNodeId(e),
                f_mask: LoopMaskNodeId(f),
            }
        }

        /// The `isMaskNode()` test and the `static_cast<MaskNode *>` that follows it, as the caller
        /// writes them (`LoweringPTMasks.cpp:174-178`).
        fn mask(&self, n: LoopMaskNodeId) -> MaskNodeId {
            self.tree
                .mask_node(n)
                .expect("the diagram's C, E and F are mask nodes")
        }
    }

    /// 🎯 171 ⛔⛔ THE DECIDING CLAUSE IS THE PARENT, PROVED ON THE REFERENCE'S OWN DIAGRAM.
    ///
    /// E and F carry `{0, 1}` each — equal on both of the fields `LoopMaskTree.cpp:125-126` compares —
    /// and the comment above the diagram still says the program is unsupported: *"We find another mask
    /// node so the program is not supported and we signal pass failure"* (`LoweringPTMasks.cpp:160-163`).
    /// Only `:124` can produce that answer.
    #[test]
    fn the_worked_examples_two_identical_masks_are_not_equivalent_across_nests() {
        let d = Diagram::build();

        // The premise: same mask, different parent.
        assert_eq!(d.tree.node(d.e_mask), d.tree.node(d.f_mask));
        assert_eq!(d.tree.parent_node(d.e_mask), Some(d.b_loop_i));
        assert_eq!(d.tree.parent_node(d.f_mask), Some(d.d_loop_j));

        assert!(
            !d.tree
                .is_mask_equivalent_to_node(d.mask(d.e_mask), d.mask(d.f_mask))
        );
        assert!(
            !d.tree
                .is_mask_equivalent_to_node(d.mask(d.f_mask), d.mask(d.e_mask))
        );

        // ⭐ AND C IS UNEQUIVALENT TO EITHER FOR ALL THREE REASONS AT ONCE — the root is its parent,
        // and `{2, 0}` matches neither field.
        assert!(
            !d.tree
                .is_mask_equivalent_to_node(d.mask(d.c_mask), d.mask(d.e_mask))
        );
    }

    /// 🎯 171 — AND THE CASE IT ALLOWS IS THE VENDOR'S OWN PAIR. `n1_m1` and `n1_m2` are both `{0, 1}`
    /// and both parented to `n1_l4`, which is *"If the mask is equivalent, it is allowed"*
    /// (`LoweringPTMasks.cpp:124`) and the reason `dynamic_pt_masking.mlir` brackets `for %arg4` ONCE
    /// (`:38`, `:53`, `:56`) with two masked macs under it.
    #[test]
    fn two_sibling_masks_with_the_same_start_and_increment_are_equivalent() {
        let v = Vendor::build();
        let m1 = v.tree.mask_node(v.n1_m1).expect("n1_m1 is a mask node");
        let m2 = v.tree.mask_node(v.n1_m2).expect("n1_m2 is a mask node");

        assert_eq!(v.tree.parent_node(v.n1_m1), v.tree.parent_node(v.n1_m2));
        assert!(v.tree.is_mask_equivalent_to_node(m1, m2));
        assert!(
            v.tree.is_mask_equivalent_to_node(m2, m1),
            "all three clauses are symmetric, so which side is the receiver cannot matter"
        );
        assert!(
            v.tree.is_mask_equivalent_to_node(m1, m1),
            "and reflexive — the BFS starts at `node` itself, so `mask_node` is compared with \
             itself first (`LoweringPTMasks.cpp:120-126`)"
        );

        // ⛔ ACROSS THE TWO NESTS IT IS FALSE EVEN FOR THE SAME MASK: `n2_m_dyn` is `{0, 1}` too, and
        // its parent is `n2_l6`.
        let dynamic = v
            .tree
            .mask_node(v.n2_m_dyn)
            .expect("n2_m_dyn is a mask node");
        assert_eq!(v.tree.node(v.n1_m1), v.tree.node(v.n2_m_dyn));
        assert!(!v.tree.is_mask_equivalent_to_node(m1, dynamic));
    }

    /// 🎯 171 — THE OTHER TWO CLAUSES, EACH ISOLATED UNDER ONE PARENT. Three masks under the same loop:
    /// `{0, incr}`, `{4, incr}` and `{0, const}`, so the second differs only in `getStartVal()` and the
    /// third only in `getIncrement()`.
    #[test]
    fn a_different_start_or_increment_is_not_equivalent_under_one_parent() {
        let mut base = OperationTreeBase::with_root(LoopMaskNode::SyntheticRoot);
        let root = base.root();
        let driving_loop = base.push_named_child(root, OpId::at(&[0]), loop_node());
        let baseline = base.push_named_child(driving_loop, OpId::at(&[0, 0]), dynamic_mask());
        let other_start = base.push_named_child(
            driving_loop,
            OpId::at(&[0, 1]),
            LoopMaskNode::Mask(MaskNode {
                start_val: MaskedColumns(4),
                increment: MaskIncrement::PerParentLoopIteration,
            }),
        );
        let other_increment =
            base.push_named_child(driving_loop, OpId::at(&[0, 2]), constant_mask(0));
        let tree = LoopMaskTree { base };
        let mask = |n| tree.mask_node(n).expect("all three are mask nodes");

        let baseline = mask(LoopMaskNodeId(baseline));
        let other_start = mask(LoopMaskNodeId(other_start));
        let other_increment = mask(LoopMaskNodeId(other_increment));

        // The premise: one parent for all three, so only the mask can differ.
        assert_eq!(
            tree.parent_node(baseline.node()),
            tree.parent_node(other_start.node())
        );
        assert_eq!(
            tree.parent_node(baseline.node()),
            tree.parent_node(other_increment.node())
        );

        assert_eq!(baseline.mask().increment, other_start.mask().increment);
        assert!(!tree.is_mask_equivalent_to_node(baseline, other_start));

        assert_eq!(baseline.mask().start_val, other_increment.mask().start_val);
        assert!(!tree.is_mask_equivalent_to_node(baseline, other_increment));
    }

    /// 🎯 171 — ONLY A MASK NODE IS CLASSIFIED AS ONE, WHICH IS THE `MaskNode *` IN THE SIGNATURE.
    /// A loop node and the synthetic root have no mask to read, so they cannot reach
    /// [`LoopMaskTree::is_mask_equivalent_to_node`] at all — there is no `static_cast` left to be
    /// wrong and no arm that has to answer for one.
    #[test]
    fn only_a_mask_node_is_classified_as_one() {
        let v = Vendor::build();

        assert_eq!(v.tree.mask_node(v.root), None);
        assert_eq!(
            v.tree.mask_node(v.n1_l4),
            None,
            "a `sentient.for` carries no mask"
        );
        assert_eq!(
            v.tree.mask_node(v.n1_m1).map(MaskNodeId::mask),
            Some(MaskNode {
                start_val: MaskedColumns(0),
                increment: MaskIncrement::PerParentLoopIteration,
            })
        );
        assert_eq!(
            v.tree.mask_node(v.n2_m_const).map(MaskNodeId::mask),
            Some(MaskNode {
                start_val: MaskedColumns(2),
                increment: MaskIncrement::Constant,
            })
        );
        assert_eq!(
            v.tree.mask_node(v.n1_m1).map(MaskNodeId::node),
            Some(v.n1_m1),
            "the witness keeps the identity it was minted from"
        );
    }
}

