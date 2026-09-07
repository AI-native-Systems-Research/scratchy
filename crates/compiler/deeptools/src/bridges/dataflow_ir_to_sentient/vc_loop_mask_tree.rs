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
/// or [`OperationTreeBase::push_child`], the field is private, and nothing removes a node (the C++
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
/// alone. Caching either would be a second source of truth for one relation, and
/// [`OperationTreeBase::push_child`] would then have two things to keep in step.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Node<N> {
    /// The three links.
    links: Links,
    /// What the derived class adds.
    payload: N,
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
    pub fn with_root(root: N) -> Self {
        Self {
            nodes: vec![Node {
                links: Links::UNLINKED,
                payload: root,
            }],
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
    /// links is [`Self::push_child`], which writes the parent of the node it just created and the
    /// `next_sibling` of an existing CHILD — never the fields of the node it was given as a parent.
    /// The root is never anyone's child, so nothing can reach either field.
    pub fn root(&self) -> OperationNodeId {
        self.root
    }

    /// A NODE'S PAYLOAD — the derived state the C++ reaches through the `static_cast`.
    pub fn payload(&self, n: OperationNodeId) -> &N {
        &self.nodes[n.0].payload
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
    /// refusal. Every node but the synthetic root has a parent — `push_child` sets it — so the only
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
    pub fn push_child(&mut self, parent: OperationNodeId, node: N) -> OperationNodeId {
        let child = OperationNodeId(self.nodes.len());
        self.nodes.push(Node {
            links: Links {
                // `child->setParentNode(this)` (`:253`).
                parent: Some(parent),
                first_child: None,
                next_sibling: None,
            },
            payload: node,
        });
        match self.last_child(parent) {
            // `if (isLeaf()) setFirstChild(child)` (`:241-243`).
            None => self.nodes[parent.0].links.first_child = Some(child),
            // `else getLastChild()->setNextSibling(child)` (`:251-252`).
            Some(last) => self.nodes[last.0].links.next_sibling = Some(child),
        }
        child
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
/// ⛔ FILLED WAVE BY WAVE, LIKE `AccessDetailsBase`. What is declared here is what entries 081-088
/// need. The private `DenseMap<Operation *, LoopMaskNode *> op_to_node_` (`hpp:137`) is deliberately
/// ABSENT: its writers and its only reader are entries 236 (`addMaskNode`), 237 (`updateNode`) and
/// 078 (`findNodeFromOp`), all of which also decide how an operation is named in this crate. A field
/// declared now would be one nothing reads and whose key type is the next batch's choice.
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
    /// ⛔ `None` IS THE SYNTHETIC ROOT AND ONLY THE ROOT. `push_child` gives every other node a
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
    /// `verifyLoopNest`'s `DT_CHECK_MSG(node->isLoopNode(), …)` (`LoweringPTMasks.cpp:112`),
    /// `print`'s three-way branch (`LoopMaskTree.cpp:102-111`) — and here the answer carries the
    /// payload with it.
    pub fn node(&self, n: LoopMaskNodeId) -> &LoopMaskNode {
        self.base.payload(n.0)
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{
        LMTLoopNode, LoopMaskNode, LoopMaskNodeId, LoopMaskTree, MaskIncrement, MaskNode,
        MaskedColumns, OperationTreeBase,
    };

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
            let n1_l1 = base.push_child(root, loop_node());
            let n1_l2 = base.push_child(n1_l1, loop_node());
            let n1_l3 = base.push_child(n1_l2, loop_node());
            let n1_l4 = base.push_child(n1_l3, loop_node());
            let n1_l5 = base.push_child(n1_l4, loop_node());
            let n2_l1 = base.push_child(root, loop_node());
            let n2_l2 = base.push_child(n2_l1, loop_node());
            let n2_l3 = base.push_child(n2_l2, loop_node());
            let n2_l4 = base.push_child(n2_l3, loop_node());
            let n2_l5 = base.push_child(n2_l4, loop_node());
            let n2_l6 = base.push_child(n2_l5, loop_node());

            // ── then `addMaskNode`, once per masked compute, in lowering order ────────────────────
            let n1_m1 = base.push_child(n1_l4, dynamic_mask());
            let n1_m2 = base.push_child(n1_l4, dynamic_mask());
            // ⭐ `#set2 = affine_set<(d0) : (d0 - 48 >= 0, -d0 + 63 >= 0)>` over `vector<64xf16>`:
            // `(64 - 48) / 8 = 2`, the `scalar_constant {value = 2}` the golden's `set_mask` takes.
            let n2_m_const = base.push_child(n2_l5, constant_mask(2));
            let n2_m_dyn = base.push_child(n2_l6, dynamic_mask());

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
}
