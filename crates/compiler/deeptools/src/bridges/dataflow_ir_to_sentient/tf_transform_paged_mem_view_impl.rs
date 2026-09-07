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

//! `TransformPagedMemViewImpl.cpp` — 39 of bridge 2's 384 functions (dependency level(s) [0, 1, 2, 3, 4, 5, 6, 7]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e118_removeValuesFromIndices` | 118/384 | 7 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:36` |
//! | `e119_replaceDimsInMapWithSyms` | 119/384 | 7 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:47` |
//! | `e120_createEqualityCondition` | 120/384 | 7 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:253` |
//! | `e121_createInequalityCondition` | 121/384 | 14 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:263` |
//! | `e122_setBuilderToInsertRef` | 122/384 | 6 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:280` |
//! | `e123_calculateStartElementsForPage` | 123/384 | 10 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:532` |
//! | `e124_createNonPagedMemView` | 124/384 | 10 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:545` |
//! | `e125_cloneMemViewIfNonPaged` | 125/384 | 9 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:633` |
//! | `e126_getUseChain` | 126/384 | 3 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:673` |
//! | `e127_cloneUseChain` | 127/384 | 3 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:678` |
//! | `e128_createNewMemOp` | 128/384 | 9 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:684` |
//! | `e129_eraseMemOpAndUseChain` | 129/384 | 3 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:698` |
//! | `e130_getStoreOp` | 130/384 | 6 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:843` |
//! | `e131_addTimeDimIndicesRanges` | 131/384 | 5 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:876` |
//! | `e132_identifyTimeDimForExplicitLoops` | 132/384 | 10 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:963` |
//! | `e133_getUseChain` | 133/384 | 2 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:328` |
//! | `e134_cloneUseChain` | 134/384 | 0 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:341` |
//! | `e135_eraseMemOpAndUseChain` | 135/384 | 0 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:364` |
//! | `e136_TPMVBase` | 136/384 | 0 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:389` |
//! | `e137_TPMVVector` | 137/384 | 0 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:397` |
//! | `e138_TPMVComposite` | 138/384 | 0 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:519` |
//! | `e197_calculateIndicesRanges` | 197/384 | 25 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:56` |
//! | `e198_createConditionsForHyperRectSubscripts` | 198/384 | 48 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:289` |
//! | `e199_createConditionsForNonHyperRectSubscripts` | 199/384 | 35 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:342` |
//! | `e200_updateTPMVInfo` | 200/384 | 17 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:382` |
//! | `e201_setLoopIteratorOrder` | 201/384 | 13 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:575` |
//! | `e202_initialize` | 202/384 | 13 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:658` |
//! | `e258_addConstraintsForIVRanges` | 258/384 | 22 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:86` |
//! | `e259_createNewSubscriptsFromStartElements` | 259/384 | 10 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:562` |
//! | `e260_gatherPageDependentDimsForPage` | 260/384 | 32 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:927` |
//! | `e294_getPageValidity` | 294/384 | 33 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:114` |
//! | `e295_createIterArgsForConditionals` | 295/384 | 121 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:401` |
//! | `e309_constructValidPage` | 309/384 | 58 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:190` |
//! | `e310_analyzeValidPages` | 310/384 | 33 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:888` |
//! | `e324_analyzeAndConstructValidPages` | 324/384 | 32 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:152` |
//! | `e325_transform_time` | 325/384 | 98 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:977` |
//! | `e326_initialize_time` | 326/384 | 23 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:1095` |
//! | `e356_transform` | 356/384 | 39 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:592` |
//! | `e373_run` | 373/384 | 6 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:647` |
//!
//! Original files homed here: `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp`, `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp`


use super::agen_access_details::{TimeBound, TimeDim};
use super::agen_helper::{AgenOpKind, store_op_from_load_store_pattern};
use crate::islands::dataflow_ir::dialects::{Op as DfirOp, Val, agen, results, uses};
use crate::islands::dataflow_ir::ty::IntegerSet;
use crate::units::DfirUnit;
use core::num::NonZeroU32;
use std::collections::BTreeSet;

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 129/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// AN `agen.vector_load`, PROVEN — the door `cast<agen::VectorLoadOp>(mem_op)` is.
///
/// ⭐⭐ EVERY OVERRIDE IN THIS FILE OPENS WITH THAT CAST, AND `cast<>` IS THE ABORTING ONE.
/// `TPMVVectorLoad`'s four overrides each begin `auto load_op = cast<agen::VectorLoadOp>(mem_op);`
/// (`TransformPagedMemViewImpl.cpp:674`, `:679`, `:688`, `:699`) — the class of `mem_ops_[0]` was
/// already settled by `TransformPagedMemViewManager` when it chose which `TPMV*` to build, so the
/// cast is a restatement of that choice and never a test. [`VectorLoadOp::of`] is where the
/// statement is inspected; past it the class is a fact, so nothing below repeats the check.
///
/// ⛔ IT CARRIES THE RESULT BECAUSE THAT IS WHAT THE OVERRIDES READ. `getUseChain` roots at it
/// (`Agen.cpp:117`) and `getStoreOp` traces its single use (`:844-846`); `getResult()` on a typed
/// `VectorLoadOp` is infallible, which is why neither asks the op its arity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorLoadOp<'a> {
    /// The statement itself.
    pub op: &'a DfirOp,
    /// `getResult()` — the vector it binds.
    pub result: Val,
}

impl<'a> VectorLoadOp<'a> {
    /// `dyn_cast<agen::VectorLoadOp>` — `None` for anything else.
    #[must_use]
    pub fn of(op: &'a DfirOp) -> Option<VectorLoadOp<'a>> {
        match op {
            DfirOp::Agen(agen::Op::VectorLoad { result, .. }) => Some(VectorLoadOp {
                op,
                result: *result,
            }),
            DfirOp::Agen(
                agen::Op::VectorStore { .. }
                | agen::Op::CompositeLoadAndStore(_)
                | agen::Op::Yield,
            )
            | DfirOp::Arith(_)
            | DfirOp::Scf(_)
            | DfirOp::Affine(_)
            | DfirOp::Dataflow(_)
            | DfirOp::VectorChain(_) => None,
        }
    }
}

/// ONE MEMORY OPERATION'S LINEAR USE CHAIN, ⛔ CARRYING WHICH WAY IT RUNS.
///
/// `TPMVBase::getUseChain` documents the contract and leaves the direction to the caller to work
/// out: *"Empty if there isn't a use chain. The use chain may be returned in order or in reverse
/// order of operations depending on the Operation type. If `mem_op` is the first element of the
/// returned vector, it is in order. If `mem_op` is the last element, it is in reverse order."*
/// (`TransformPagedMemViewImpl.hpp:321-327`).
///
/// # ⛔⛔ THAT WRITTEN CONTRACT DOES NOT MATCH EITHER IMPLEMENTATION, AND THE TYPE FIXES IT
///
/// Both dialect methods put `mem_op` FIRST, so by the letter of the comment both are "in order" —
/// and yet `VectorStoreOp::cloneUseChainToNewOp` states the opposite about its own input two files
/// away: *"The first operation in the use chain is the VectorStoreOp. The use chain is stored in
/// reverse order."* (`Agen.cpp:229-230`). Measured at both producers:
///
/// - `VectorLoadOp::getUseChain` (`Agen.cpp:115-136`) walks `curr_op = *res.getUsers().begin()`,
///   giving `[load, user, …, terminator]` — **consumer-ward**, and it is the LAST element that has
///   no results.
/// - `VectorStoreOp::getUseChain` (`Agen.cpp:207-222`) pushes the store, then
///   `getValueToStore().getDefiningOp()`, then a `vectorchain::ShuffleOp`'s own input — giving
///   `[store, producer, producer's producer]`, **producer-ward**, with the store (which has no
///   results) FIRST.
///
/// ⭐⭐ AND THE DIRECTION IS EXACTLY WHAT THE TWO ERASE LOOPS DISAGREE ON. `VectorLoadOp::eraseOpAndUseChain`
/// walks the chain BACKWARDS (`for (int idx = use_chain.size() - 1; idx >= 0; --idx)`,
/// `Agen.cpp:176-177`) while `VectorStoreOp::eraseOpAndUseChain` walks it FORWARDS
/// (`for (auto &o : use_chain)`, `:257`) — two loops that look contradictory and are the same rule:
/// **tear the chain down consumer-first**, so no erased value still has a live use. [`Self::consumer_first`]
/// is that one rule, and a chain that cannot say which way it runs cannot be handed to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UseChain<'a> {
    /// `return {}` — this operation has no linear use chain.
    ///
    /// ⭐ THE CONTRACT'S OWN "Empty if there isn't a use chain" CASE, and the only answer
    /// [`TpmvBase::use_chain`] ever gives.
    None,
    /// `[mem_op, user, …, terminator]` — `mem_op` first, running toward its consumers.
    ///
    /// What `VectorLoadOp::getUseChain` returns. Its last element binds no result.
    ConsumerWard(Vec<&'a DfirOp>),
    /// `[mem_op, producer, …]` — `mem_op` first, running back toward its producers.
    ///
    /// What `VectorStoreOp::getUseChain` returns. Its FIRST element binds no result.
    ///
    /// ⚠️ NO UNIT IN THIS FILE PRODUCES ONE YET: `TPMVVectorStore::getUseChain`
    /// (`TransformPagedMemViewImpl.cpp:721`) is not among the 384 and not among the 106 exclusions —
    /// see the note on [`erase_vector_load_and_use_chain`]. The variant is here because
    /// [`Self::consumer_first`] is only correct if it can tell the two apart.
    ProducerWard(Vec<&'a DfirOp>),
}

impl<'a> UseChain<'a> {
    /// THE CHAIN IN TEARDOWN ORDER — every consumer ahead of what it reads.
    ///
    /// This is what both `eraseOpAndUseChain` loops compute, each in the way its own chain's
    /// direction demands (`Agen.cpp:176-177` reversed, `:257` as-is).
    #[must_use]
    pub fn consumer_first(&self) -> Vec<&'a DfirOp> {
        match self {
            UseChain::None => Vec::new(),
            UseChain::ConsumerWard(chain) => chain.iter().rev().copied().collect(),
            UseChain::ProducerWard(chain) => chain.clone(),
        }
    }
}

/// `agen::VectorLoadOp::getUseChain` (`Agen.cpp:115-136`) — the DIALECT method, and a private
/// helper rather than a unit of the campaign: it lives in
/// `dataflow-scheduler/external/dataflow-scheduler-dialects/lib/Dialect/Agen/Agen.cpp`, outside
/// `dcc`, so nothing in the 384 covers it.
///
/// ```cpp
/// SmallVector<Operation*> VectorLoadOp::getUseChain() {
///   auto& op = *this;
///   Operation* curr_op = op;
///   SmallVector<Operation*> use_chain;
///   while (true) {
///     use_chain.push_back(curr_op);
///     // Only the last operation in the chain should have no results
///     // (dataflow.send, for example).
///     if (curr_op->getNumResults() == 0) break;
///     assert((curr_op->getNumResults() == 1) && "...");
///     Value res = curr_op->getResult(0);
///     assert((res.hasOneUse()) && "...");
///     curr_op = *res.getUsers().begin();
///   }
///   assert(use_chain.size() >= 2 && use_chain.back()->getNumResults() == 0);
///   return use_chain;
/// }
/// ```
///
/// ⛔⛔ ITS THREE ASSERTS ARE THE SAME QUESTION, AND [`UseChain::None`] IS THEIR ANSWER. A chain is
/// linear only while each op binds exactly one result that exactly one op reads; the moment either
/// fails there is no chain to return, which is the contract's own "Empty if there isn't a use chain"
/// (`TransformPagedMemViewImpl.hpp:322`) rather than an abort. The trailing
/// `use_chain.size() >= 2` is the same test one step later — a load nothing reads.
///
/// ⭐ AND THAT MAKES THE REFERENCE'S DEAD BRANCH LIVE. `eraseOpAndUseChain` opens
/// `if (use_chain.empty()) { op->erase(); }` (`Agen.cpp:173-175`), unreachable there because
/// `getUseChain` either asserts or returns at least two ops. Here it is the path a non-linear load
/// takes — see [`erase_vector_load_and_use_chain`].
fn vector_load_use_chain<'a>(load: VectorLoadOp<'a>, scope: &'a [DfirOp]) -> UseChain<'a> {
    let mut chain: Vec<&'a DfirOp> = Vec::new();
    let mut curr: &'a DfirOp = load.op;
    loop {
        chain.push(curr);
        let bound = results(curr);
        // "Only the last operation in the chain should have no results (dataflow.send, for example)."
        if bound.is_empty() {
            break;
        }
        // `assert(curr_op->getNumResults() == 1)`.
        let [result] = bound.as_slice() else {
            return UseChain::None;
        };
        // `assert(res.hasOneUse())`.
        let users = uses(*result, scope);
        let [user] = users.as_slice() else {
            return UseChain::None;
        };
        curr = user;
    }
    // `assert(use_chain.size() >= 2 && use_chain.back()->getNumResults() == 0)` — a load whose
    // result nothing reads never got past the census above, so what is left to reject is a load
    // that binds no result at all, which cannot be built.
    if chain.len() < 2 {
        return UseChain::None;
    }
    UseChain::ConsumerWard(chain)
}

/// Replaces: e129_eraseMemOpAndUseChain
///
/// **129/384** `TPMVVectorLoad::eraseMemOpAndUseChain` —
/// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:698` (3L).
///
/// ```cpp
/// void TPMVVectorLoad::eraseMemOpAndUseChain(Operation *mem_op) {
///   auto load_op = cast<agen::VectorLoadOp>(mem_op);
///   load_op.eraseOpAndUseChain();
/// }
/// ```
///
/// # ⭐⭐ THE OVERRIDE IS TWO LINES AND THE BEHAVIOUR IS ALL IN THE DIALECT
///
/// `TPMVBase::eraseMemOpAndUseChain` erases the one op (entry 135); this override erases the op
/// **and everything downstream of it**, because a paged load's rotate/shuffle/send tail is only
/// there to consume the load being replaced. `transform()` calls it over `mem_ops_` right after the
/// new ops are in place: `for (auto &mem_op : mem_ops_) eraseMemOpAndUseChain(mem_op);`
/// (`TransformPagedMemViewImpl.cpp:618`).
///
/// ⭐⭐ A DELETE LIST, NOT AN ERASE — AND THE ORDER IS THE REFERENCE'S. This crate does not mutate a
/// program in place, so the port returns the ops to remove in the order `eraseOpAndUseChain` would
/// remove them: consumer-first (`Agen.cpp:176-177`). Entry 038 already established the shape —
/// [`super::agen_helper::add_load_chain_to_delete_list`] returns the send before the shuffle that
/// feeds it for the same reason.
///
/// ⛔ AND THE REFERENCE'S "IMPOSSIBLE" BRANCH IS THE FALLBACK HERE. `if (use_chain.empty())
/// { op->erase(); }` (`Agen.cpp:173-175`) cannot fire in the C++ — `getUseChain` asserts first. A
/// load whose chain is not linear answers [`UseChain::None`] here instead of aborting, and then
/// that branch is exactly right: remove the load, leave what reads it alone.
///
/// ⚠️ `TPMVVectorStore::eraseMemOpAndUseChain` (`TransformPagedMemViewImpl.cpp:747`) IS THE SAME
/// TWO LINES OVER `VectorStoreOp`, and it is in neither the 384 nor the 106 exclusions — the
/// extractor deduplicates by function name within a file, so of this file's three
/// `eraseMemOpAndUseChain` definitions only the first (`:698`) and the base's (`hpp:364`) were
/// scheduled. Same for `TPMVVectorStore::{getUseChain, cloneUseChain}` (`:721`, `:726`) and
/// `TPMVVectorLoadStore::{createNewMemOp, eraseMemOpAndUseChain}` (`:779`, `:817`). They are
/// reported, not filled: filling one would put a `Replaces:` anchor on an entry nothing scheduled.
#[must_use]
pub fn erase_vector_load_and_use_chain<'a>(
    load: VectorLoadOp<'a>,
    scope: &'a [DfirOp],
) -> Vec<&'a DfirOp> {
    let to_be_erased = vector_load_use_chain(load, scope).consumer_first();
    if to_be_erased.is_empty() {
        // `if (use_chain.empty()) { auto& op = *this; op->erase(); }`.
        return vec![load.op];
    }
    to_be_erased
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 130/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// AN `agen.vector_store`, PROVEN — the door `dyn_cast<agen::VectorStoreOp>` is.
///
/// ⛔ THE RETURN TYPE OF ENTRY 130 IS `agen::VectorStoreOp`, NOT `Operation *`, and that is the
/// whole reason the reference asserts twice: the second `DT_CHECK(store_op)` exists only because a
/// typed handle cannot hold a statement of another class. Here the class is in the type and the
/// check is the constructor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorStoreOp<'a> {
    /// The statement itself.
    pub op: &'a DfirOp,
    /// `getValueToStore()` — the vector it writes.
    ///
    /// ⭐ WHAT A STORE'S USE CHAIN ROOTS AT: `VectorStoreOp::getUseChain` takes
    /// `getValueToStore().getDefiningOp()` as the next link (`Agen.cpp:216`).
    pub value: Val,
}

impl<'a> VectorStoreOp<'a> {
    /// `dyn_cast<agen::VectorStoreOp>` — `None` for anything else.
    #[must_use]
    pub fn of(op: &'a DfirOp) -> Option<VectorStoreOp<'a>> {
        match op {
            DfirOp::Agen(agen::Op::VectorStore { value, .. }) => Some(VectorStoreOp {
                op,
                value: *value,
            }),
            DfirOp::Agen(
                agen::Op::VectorLoad { .. }
                | agen::Op::CompositeLoadAndStore(_)
                | agen::Op::Yield,
            )
            | DfirOp::Arith(_)
            | DfirOp::Scf(_)
            | DfirOp::Affine(_)
            | DfirOp::Dataflow(_)
            | DfirOp::VectorChain(_) => None,
        }
    }
}

/// Replaces: e130_getStoreOp
///
/// **130/384** `TPMVVectorLoadStore::getStoreOp` —
/// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:843` (6L).
///
/// ```cpp
/// agen::VectorStoreOp TPMVVectorLoadStore::getStoreOp(
///     agen::VectorLoadOp &load_op) {
///   DT_CHECK(load_op.getResult().hasOneUse());
///   auto store_op =
///       dyn_cast<agen::VectorStoreOp>(*load_op.getResult().user_begin());
///   DT_CHECK(store_op);
///   return store_op;
/// }
/// ```
///
/// # ⭐⭐ dcc HAS THIS BODY TWICE, IN TWO FILES, AND ENTRY 036 ALREADY PORTED IT
///
/// `AgenToSentientLoweringPass::getStoreOpFromLoadStorePattern` (`Conversion/AgenToSentient/Helper.cpp:2872`,
/// entry 036) is the same three steps — one result, one use, is that use a store of this class — and
/// [`super::agen_helper::store_op_from_load_store_pattern`] is it. Calling it is the port: two
/// spellings of one rule must not become two implementations that can drift.
///
/// ⛔ THE TWO DIFFER ONLY IN WHAT THEY ASSUME ABOUT ARITY. Entry 036 takes an `Operation *` and
/// tests `getNumResults() == 1` itself; this one takes a typed `agen::VectorLoadOp &`, whose
/// `getResult()` is infallible. [`VectorLoadOp`] carries that same guarantee, so passing `load.op`
/// through the arity census re-derives a fact already held — which is exactly why the answers agree.
///
/// ⛔ AND `None` IS BOTH `DT_CHECK`s AT ONCE. `hasOneUse()` failing and the single user not being a
/// store are two aborts in the reference and one answer here: this load is not the load-and-store
/// pattern. `TransformPagedMemViewManager` only builds a `TPMVVectorLoadStore` after recognising the
/// pattern, so neither is reachable from the reference's own call sites.
#[must_use]
pub fn store_op<'a>(load: VectorLoadOp<'a>, scope: &'a [DfirOp]) -> Option<VectorStoreOp<'a>> {
    store_op_from_load_store_pattern(AgenOpKind::VectorStore, load.op, scope)
        .and_then(VectorStoreOp::of)
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 131/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ONE LOOP ITERATOR'S RANGE — `TPMVBase::IVRange`
/// (`dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:40`):
///
/// ```cpp
/// using IVRange = std::pair<int, int>;
/// ```
///
/// # ⭐⭐ BOTH ENDS ARE INCLUSIVE, AND THE `- 1` IS WHY
///
/// `calculateIndicesRanges` builds one per subscript from the owning `affine.for`:
/// `lb = lb_map.getSingleConstantResult(); ub = ub_map.getSingleConstantResult() - 1;`
/// (`TransformPagedMemViewImpl.cpp:73-75`) — an affine loop's upper bound is exclusive, so the
/// stored `ub` is the LAST value the iterator takes. [`add_time_dim_indices_ranges`] does the same
/// subtraction on a time bound.
///
/// ⛔ AND THE CONSUMER PROVES IT: `addConstraintsForIVRanges` emits `<sym> - <lb> >= 0` and
/// `-<sym> + <ub> >= 0` (`:99-104`), a closed interval, with the reference's own comments naming
/// `.first` the lower bound and `.second` the upper.
///
/// ⭐ NAMED FIELDS, NO POSITIONAL CONSTRUCTOR. `emplace_back(0, b - 1)` and `emplace_back(lb, ub)`
/// are two arguments in one order that nothing but the reader enforces; `IvRange { lb, ub }` cannot
/// be written the wrong way round.
///
/// ⚠️ `i64`, WHERE THE REFERENCE NARROWS. `calculateIndicesRanges` puts an `int64_t`
/// `getSingleConstantResult()` into an `int`, and `addTimeDimIndicesRanges` puts an `int64_t` time
/// bound into the same `int`. The island's own loop bound is `affine::Bound::Const(i64)`, and the
/// constraint expressions these become take `AffineExpr::Const(i64)`, so widening here removes a
/// truncation rather than adding one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IvRange {
    /// `.first` — the first value the iterator takes.
    pub lb: i64,
    /// `.second` — the LAST value the iterator takes, inclusive.
    pub ub: i64,
}

/// A TIME DIMENSION'S STEP COUNT, ⛔ WITH THE SENTINELS ALREADY GONE.
///
/// `addTimeDimIndicesRanges` opens with
/// `DT_CHECK_MSG(b - 1 >= 0, "no special time bound values should exist")`
/// (`TransformPagedMemViewImpl.cpp:879`) — an abort against the two flags [`TimeBound`] documents,
/// `kInvalid = -1` and `kCoalesced = -2`, plus the `0` that `computeBurstAndGroup` handles
/// separately (`AccessDetails.cpp:809`).
///
/// ⭐⭐ SO THE CHECK IS THIS TYPE. A `TimeSteps` cannot hold a flag and cannot hold zero, which
/// makes `b - 1 >= 0` true by construction and [`Self::last_index`] total. [`Self::of`] is the one
/// door, and it is the caller's job to walk through it — the reference's own abort, moved to where
/// the bound comes from.
///
/// ⭐ AND `u32` IS DELIBERATE OVER `u64`: `i64::from` a `u32` is total, so the `- 1` needs no
/// fallible conversion and no cast. The reference stores the result in an `int` (see [`IvRange`]),
/// so `u32` is already wider than the range that survives the C++.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TimeSteps(NonZeroU32);

impl TimeSteps {
    /// THE DOOR — a bound that is a real step count, or `None` for the ones the `DT_CHECK_MSG`
    /// rejects.
    #[must_use]
    pub fn of(bound: TimeBound) -> Option<TimeSteps> {
        match bound {
            // `b - 1 >= 0` holds for every `b >= 1`; `Steps(0)` is the reachable case it excludes.
            TimeBound::Steps(steps) => u32::try_from(steps).ok().and_then(NonZeroU32::new).map(TimeSteps),
            // `kCoalesced = -2` and `kInvalid = -1` — "no special time bound values should exist".
            TimeBound::Coalesced | TimeBound::Variable => None,
        }
    }

    /// `b - 1` — the last step this dimension takes, which is its range's inclusive upper bound.
    #[must_use]
    pub fn last_index(self) -> i64 {
        i64::from(self.0.get() - 1)
    }
}

/// Replaces: e131_addTimeDimIndicesRanges
///
/// **131/384** `TPMVComposite::addTimeDimIndicesRanges` —
/// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:876` (5L).
///
/// ```cpp
/// void TPMVComposite::addTimeDimIndicesRanges(
///     const SmallVectorImpl<int64_t> &time_bounds,
///     SmallVectorImpl<IVRange> &indices_ranges) {
///   for (auto &b : time_bounds) {
///     DT_CHECK_MSG(b - 1 >= 0, "no special time bound values should exist");
///     indices_ranges.emplace_back(0, b - 1);
///   }
/// }
/// ```
///
/// # ⛔⛔ IT APPENDS, AND THE POSITION IT APPENDS AT IS LOAD-BEARING
///
/// The call site runs it directly after `calculateIndicesRanges` on the SAME vector:
///
/// ```cpp
/// // Ranges of the loop iterators are used to only choose pages within the
/// // loop iteration space.
/// calculateIndicesRanges(tpmv_info_[i].indices_, tpmv_info_[i].indices_ranges_);
///
/// // Add the ranges of the time dims.
/// // Since all memory accesses for a memory op use the same time bounds,
/// // just use the first access_details_ to grab time bounds.
/// addTimeDimIndicesRanges(access_details_[0].getTimeBounds(),
///                         tpmv_info_[i].indices_ranges_);
/// DT_CHECK(subscripts_map_time[i].getNumDims() ==
///          tpmv_info_[i].indices_ranges_.size());
/// ```
/// (`:1009-1019`) — so `indices_ranges_` ends up **non-time subscripts first, then time dims**, and
/// `addConstraintsForIVRanges` reads it by symbol index: `indices_ranges[sym_idx]` (`:99-104`).
/// That layout is precisely the `i + num_non_time_dims` arithmetic of entry 132; see
/// [`NonTimeDims::sym_for`]. Clearing the vector, or prepending, would silently renumber every page
/// selection constraint.
///
/// ⭐ EVERY TIME DIM STARTS AT ZERO. Unlike a loop iterator, whose `lb` comes from its `affine.for`,
/// a time dimension's range is `[0, b - 1]` unconditionally — the time set is written over the
/// step index itself, as the vendor's `affine_set<(d0)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0)>`
/// (`dcc/test/Transform/TransformPagedMemView/paged_mem_view_loads.mlir:337`) says.
///
/// ⛔ AND THE `DT_CHECK_MSG` IS NOT HERE BECAUSE IT IS IN [`TimeSteps`]. `getTimeBounds()` returns
/// `SmallVector<int64_t>` carrying `kInvalid`/`kCoalesced` in the same slots as real counts; taking
/// `&[TimeSteps]` means a caller that has not ruled those out cannot call this at all.
pub fn add_time_dim_indices_ranges(time_bounds: &[TimeSteps], indices_ranges: &mut Vec<IvRange>) {
    for b in time_bounds {
        indices_ranges.push(IvRange {
            lb: 0,
            ub: b.last_index(),
        });
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 132/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ONE SYMBOL'S POSITION IN THE PAGE SELECTION CONSTRAINTS — the `int` that
/// `page_dependent_time_syms_` holds (`TransformPagedMemViewImpl.hpp:512`).
///
/// ⭐⭐ A SYMBOL INDEX, WHICH IS ALSO A SLOT IN `indices_ranges_`, AND NEITHER IS A DIMENSION.
/// `gatherPageDependentDimsForPage` fills the set by walking the symbol columns of a
/// `FlatLinearValueConstraints`: `for (int sym = 0; sym < num_syms; ++sym) … if (eq[sym] != 0)
/// page_dependent_time_syms_.insert(sym);` (`TransformPagedMemViewImpl.cpp:955-959`), and
/// `addConstraintsForIVRanges` indexes the SAME numbering into the ranges vector
/// (`indices_ranges[sym_idx]`, `:99-104`). A time DIM is [`TimeDim`]; the two differ by
/// [`NonTimeDims`], which is the whole content of entry 132.
///
/// ⛔ A `BTreeSet` WHERE THE REFERENCE HAS AN `unordered_set<int>`. The only operations performed on
/// it are `insert` and `find` (`:959`, `:967`), so iteration order is never observed and the ordered
/// container costs nothing while making the type's `Debug` and equality deterministic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PageSelSym(pub u32);

/// HOW MANY OF A MEMORY OP'S SUBSCRIPT DIMENSIONS ARE NOT TIME DIMENSIONS — the
/// `num_non_time_dims` parameter of entry 132.
///
/// ⛔⛔ A NEWTYPE BECAUSE THE CALL SITE HAS TWO ADJACENT DIMENSION COUNTS AND THEY ARE DIFFERENT
/// NUMBERS. `identifyTimeDimForExplicitLoops` reads `time_set_.getNumDims()` from a member and takes
/// `tpmv_info_[i].subscripts_map_.getNumDims()` as its argument
/// (`TransformPagedMemViewImpl.cpp:966`, `:1024-1025`) — both `int`, one the count of TIME dims and
/// one the count of everything else. Swapping them compiles in C++ and silently reads the wrong
/// symbol; here the time count arrives as the [`IntegerSet`] it is read off, so the two cannot be
/// exchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NonTimeDims(pub u32);

impl NonTimeDims {
    /// `i + num_non_time_dims` — WHICH SYMBOL A TIME DIM IS.
    ///
    /// ⭐ THE ARITHMETIC EXISTS ONCE, AND IT IS THE LAYOUT [`add_time_dim_indices_ranges`] CREATED:
    /// `calculateIndicesRanges` appends one range per non-time subscript, then the time dims follow
    /// in order, so time dim `i` occupies symbol slot `i + num_non_time_dims`.
    #[must_use]
    pub const fn sym_for(self, dim: TimeDim) -> PageSelSym {
        PageSelSym(dim.0 + self.0)
    }
}

/// Replaces: e132_identifyTimeDimForExplicitLoops
///
/// **132/384** `TPMVComposite::identifyTimeDimForExplicitLoops` —
/// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:963` (10L).
///
/// ```cpp
/// int TPMVComposite::identifyTimeDimForExplicitLoops(int num_non_time_dims) {
///   // Traverse innermost to outermost loops. Once a page_dependent time dim is
///   // found, break. Dims below this dim may be preserved.
///   for (int i = time_set_.getNumDims() - 1; i >= 0; --i) {
///     if (auto it = page_dependent_time_syms_.find(i + num_non_time_dims) !=
///                   page_dependent_time_syms_.end())
///       return i;
///   }
///
///   return -1;
/// }
/// ```
///
/// # ⭐⭐ THE ANSWER IS A CUT, AND THE DIRECTION OF THE SCAN IS THE ALGORITHM
///
/// The header states what it is for: *"the outermost time dim to the dim returned by this function"*
/// need explicit loops (`TransformPagedMemViewImpl.hpp:498-503`). Scanning innermost → outermost and
/// returning the FIRST hit gives the INNERMOST page-dependent dim; every dim outside it must be
/// materialised as a real `affine.for` because the page a step lands in changes with it, and
/// everything strictly inside it stays folded into the composite transfer's time set. Scanning the
/// other way would return the outermost hit and unroll dimensions that did not need it.
///
/// ⛔ `-1` IS "EVERY TIME DIM MAY BE PRESERVED", AND THE CALLER TESTS IT AS A QUESTION:
/// `// A explicit_loop_dim of -1 indicates all time dims can be preserved.` `if (explicit_loop_dim >
/// -1) { … constructExplicitTimeLoops(…) … }` (`:1028-1030`). `Option<TimeDim>` is that test, and no
/// arm can index a dimension with the sentinel — the same reason [`TimeDim`] exists.
///
/// ⚠️ THE REFERENCE'S `auto it` IS A `bool`, NOT AN ITERATOR. `=` binds looser than `!=`, so
/// `auto it = find(...) != end()` declares `it` as the comparison's result and the `if` tests it.
/// The behaviour is the intended one — the name is not. Nothing is ported around it.
///
/// ⛔ THE THREE MEMBERS ARRIVE AS PARAMETERS. `time_set_`, `page_dependent_time_syms_` and the
/// `TPMVComposite` that owns them are entry 138 (`hpp:519`), which is not scheduled; naming them in
/// the signature keeps this function honest about everything it reads instead of inventing a partial
/// struct to hold them.
#[must_use]
pub fn identify_time_dim_for_explicit_loops(
    time_set: &IntegerSet,
    page_dependent_time_syms: &BTreeSet<PageSelSym>,
    num_non_time_dims: NonTimeDims,
) -> Option<TimeDim> {
    // Traverse innermost to outermost loops. Once a page_dependent time dim is found, break.
    // Dims below this dim may be preserved.
    for i in (0..time_set.dims).rev() {
        let dim = TimeDim(i);
        if page_dependent_time_syms.contains(&num_non_time_dims.sym_for(dim)) {
            return Some(dim);
        }
    }

    None
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 133/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// THE BASE OF THE `TPMV*` HIERARCHY — `TPMVBase`
/// (`dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:28`).
///
/// One object per paged memory operation being de-paged, built by
/// `TransformPagedMemViewManager`, which picks the concrete class from the op:
/// `TPMVBase` → `TPMVVector` → {`TPMVVectorLoad`, `TPMVVectorStore`, `TPMVVectorLoadStore`} and
/// `TPMVBase` → `TPMVComposite` → {`TPMVCompositeLoad`, `TPMVCompositeStore`, …}.
///
/// # ⭐ ONLY THE TWO CONSTRUCTION-TIME MEMBERS ARE HERE, BECAUSE THAT IS WHAT THE REFERENCE SAYS
///
/// ```cpp
/// /**********************************************/
/// /*****            Class members           *****/
/// /**********************************************/
/// // Set at object construction
/// SmallVector<Operation *, 16> mem_ops_;
/// SenComponents comp_;
///
/// // Set during initialization
/// MLIRContext *context_;
/// SmallVector<TPMVInfo, 2> tpmv_info_;
/// ```
/// (`hpp:375-384`). The second group is written by `initialize()` — entry 202 for the vector
/// classes, entry 326 for the composites — and `context_` is an `MLIRContext *`, which this crate
/// has no counterpart for at all. ⚠️ `tpmv_info_` arrives with those entries; a field added now
/// would be a shape guessed ahead of its writer.
///
/// ⛔ `Vec<&'p DfirOp>` AND NOT AN OWNED CLONE. The identity of the op is the point: `mem_ops_` is
/// what `eraseMemOpAndUseChain` is called over (`:618`) and what `initialize()` casts to read the
/// paged view out of (`:659-666`). ⚠️ The reference also REPLACES entries with newly built ops
/// (`:508`, `:627`, `:1067`); in this crate a new op is a value in the emitted program rather than a
/// mutation of the input, so entries 309/325/356 decide how the replacement is represented — the
/// borrow here is exactly "the ops of the input program this object is transforming".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TpmvBase<'p> {
    /// `mem_ops_` — the memory operations being de-paged, one at construction.
    pub mem_ops: Vec<&'p DfirOp>,
    /// `comp_` — the component the operation runs on.
    ///
    /// ⭐ `SenComponents` IS [`DfirUnit`] HERE, the subset of the reference's component enumeration
    /// this crate's programs name, as [`super::agen_access_details::AccessDetailsAffineComposite`]
    /// already established.
    pub comp: DfirUnit,
}

impl<'p> TpmvBase<'p> {
    /// `TPMVBase::TPMVBase` (`TransformPagedMemViewImpl.hpp:30-32`):
    ///
    /// ```cpp
    /// TPMVBase(Operation *mem_op, SenComponents comp) : comp_(comp) {
    ///   mem_ops_ = {mem_op};
    /// }
    /// ```
    ///
    /// ⚠️ UNANCHORED ON PURPOSE — THIS CONSTRUCTOR IS ONE OF THE 106 EXCLUSIONS, binned as
    /// `comp_` at `…/TransformPagedMemViewImpl.hpp:30` under *"a one-line C++ field accessor; in Rust
    /// the field itself"*. It is not an accessor: it is a member-initialising constructor that also
    /// seeds `mem_ops_` with a one-element list. The exclusion is reported rather than reinstated —
    /// promoting it would mean adding a `Replaces:` anchor for an entry the scheduler deliberately
    /// left out — and the code it excluded still has to exist for entry 136 to delegate to, so it
    /// lives here with no anchor.
    ///
    /// ⭐ ONE ELEMENT, NOT ZERO. `initialize()` asserts `mem_ops_.size() == 1` in all six concrete
    /// classes (`:659`, `:707`, `:756`, `:1081`, `:1134`, `:1187`), so the singleton list is a real
    /// invariant of a freshly built object rather than a starting point for a loop.
    #[must_use]
    pub fn new(mem_op: &'p DfirOp, comp: DfirUnit) -> TpmvBase<'p> {
        TpmvBase {
            mem_ops: vec![mem_op],
            comp,
        }
    }

    /// Replaces: e133_getUseChain
    ///
    /// **133/384** `TPMVBase::getUseChain` —
    /// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:328` (2L).
    ///
    /// ```cpp
    /// /// @brief Collects the linear use chain from \p mem_op if it is an
    /// /// operation that has a linear use chain and returns the use chain.
    /// /// @return SmallVector<Operation *> containing the use chain. Empty if there
    /// /// isn't a use chain. ...
    /// virtual SmallVector<Operation *> getUseChain(Operation *mem_op) {
    ///   return {};
    /// };
    /// ```
    ///
    /// # ⛔⛔ "NO CHAIN" IS LIVE BEHAVIOUR FOR FOUR OF THE SIX CONCRETE CLASSES, NOT A FALLBACK
    ///
    /// Only `TPMVVectorLoad` (`:673`, entry 126) and `TPMVVectorStore` (`:721`) override it.
    /// `TPMVVectorLoadStore` and all the composites inherit this empty answer, and they are the
    /// classes whose operations genuinely have no linear chain to clone: a composite transfer's
    /// consumers live INSIDE its own region, and a load-and-store pattern's store is reached through
    /// entry 130 instead. Deleting this method as "the do-nothing case" would delete the answer for
    /// the majority of the hierarchy.
    ///
    /// ⛔ `Operation *mem_op` IS UNREAD, AND THE SIGNATURE STAYS. The overrides need it — they cast
    /// it and ask the dialect (`:674-675`) — so the parameter is the polymorphic interface, not dead
    /// weight. ⚠️ Until entries 137 and 138 land there is no trait to dispatch through; these three
    /// virtuals are inherent methods here and become a trait's provided methods then, with
    /// [`erase_vector_load_and_use_chain`] as `TPMVVectorLoad`'s override of the third.
    #[must_use]
    pub fn use_chain(&self, _mem_op: &'p DfirOp) -> UseChain<'p> {
        UseChain::None
    }

    // ══════════════════════════════════════════════════════════════════════════════════════════
    // 134/384
    // ══════════════════════════════════════════════════════════════════════════════════════════

    /// Replaces: e134_cloneUseChain
    ///
    /// **134/384** `TPMVBase::cloneUseChain` —
    /// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:341` (0L).
    ///
    /// ```cpp
    /// /// @brief Clones the appropriate Operations in the use chain of \p mem_op and
    /// /// updates the uses. It is expected the first Operation in the use chain has
    /// /// been cloned/created already and is represented by \p new_mem_op . All the
    /// /// other Operations are to be cloned and updated.
    /// virtual void cloneUseChain(OpBuilder &builder, Operation *mem_op,
    ///                            Operation *new_mem_op) {}
    /// ```
    ///
    /// # ⭐ AN EMPTY BODY THAT ANSWERS A QUESTION: THE NEW OP NEEDS NO TAIL
    ///
    /// The overriding classes clone the whole downstream tail so the replacement op feeds the same
    /// rotate/shuffle/send it did — `load_op.cloneUseChainToNewOp(builder, new_mem_op)`
    /// (`:679-681`, and `Agen.cpp:138-155` for what that does). Inheriting the empty body means the
    /// new operation stands alone, which is right for exactly the classes that inherit the empty
    /// [`Self::use_chain`] above: there is no chain, so there is nothing to clone.
    ///
    /// ⭐ IT RETURNS THE CLONES IT INSERTS — none. The reference returns `void` and communicates
    /// through the `OpBuilder`'s insertion point; this crate has no builder to position (the campaign
    /// brief allows dropping exactly that mechanism), so the ops a clone would add are the value.
    /// `Vec<DfirOp>` and not `Vec<&DfirOp>`: a clone is a new statement, not a borrow of an old one.
    ///
    /// ⛔ THE `OpBuilder &builder` PARAMETER IS THE ONE THING WITH NO COUNTERPART. Its documented
    /// contract — *"Expected to be set to the appropriate position when calling this function"* —
    /// is insertion-point state, and both overriders re-set it themselves around the call
    /// (`:692-693`, `:783-784`).
    #[must_use]
    pub fn clone_use_chain(&self, _mem_op: &'p DfirOp, _new_mem_op: &'p DfirOp) -> Vec<DfirOp> {
        Vec::new()
    }

    // ══════════════════════════════════════════════════════════════════════════════════════════
    // 135/384
    // ══════════════════════════════════════════════════════════════════════════════════════════

    /// Replaces: e135_eraseMemOpAndUseChain
    ///
    /// **135/384** `TPMVBase::eraseMemOpAndUseChain` —
    /// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:364` (0L).
    ///
    /// ```cpp
    /// /// @brief Removes mem_op and its use chain.
    /// /// @param mem_op The mem_op to erase.
    /// virtual void eraseMemOpAndUseChain(Operation *mem_op) { mem_op->erase(); }
    /// ```
    ///
    /// # ⭐ THE NAME PROMISES A CHAIN AND THE BODY ERASES ONE OP, WHICH IS CONSISTENT
    ///
    /// The classes that have a chain override this (entry 129 and `:747`); the classes that inherit
    /// it are the ones [`Self::use_chain`] answers [`UseChain::None`] for, and for them "mem_op and
    /// its use chain" is just `mem_op`. `transform()` calls it over every entry of `mem_ops_` once
    /// the replacements are in place (`TransformPagedMemViewImpl.cpp:618`).
    ///
    /// ⭐ A DELETE LIST, NOT AN ERASE — the same decision as entry 129 and entry 038: this crate
    /// does not mutate a program in place, so what a removal produces is the ops to remove, in the
    /// order they are to be removed.
    #[must_use]
    pub fn erase_mem_op_and_use_chain(&self, mem_op: &'p DfirOp) -> Vec<&'p DfirOp> {
        // `mem_op->erase();`
        vec![mem_op]
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 136/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// THE VECTOR-TRANSFER BRANCH OF THE HIERARCHY — `TPMVVector`
/// (`dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:389-395`).
///
/// ```cpp
/// class TPMVVector : public TPMVBase {
///  public:
///   TPMVVector(Operation *mem_op, SenComponents comp) : TPMVBase(mem_op, comp) {}
///
///   LogicalResult run() override final;
/// };
/// ```
///
/// ⭐ COMPOSITION WHERE THE REFERENCE HAS INHERITANCE, and the base is a named field rather than a
/// `Deref`: `TPMVVector` adds no state, so `base` holds all of it, and the one thing it does add —
/// `run() override final`, entry 373 (`:647`) — is a method that will sit on this type.
///
/// ⚠️ `TPMVVectorLoad`, `TPMVVectorStore` and `TPMVVectorLoadStore` compose over this in turn
/// (entry 137, `hpp:397`); their `initialize()`/`transform()` bodies are entries 202 and 356.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TpmvVector<'p> {
    /// The `TPMVBase` subobject.
    pub base: TpmvBase<'p>,
}

impl<'p> TpmvVector<'p> {
    /// Replaces: e136_TPMVBase
    ///
    /// **136/384** `TPMVVector::TPMVVector` —
    /// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:389` (0L).
    ///
    /// ```cpp
    /// TPMVVector(Operation *mem_op, SenComponents comp) : TPMVBase(mem_op, comp) {}
    /// ```
    ///
    /// # ⭐ THE ENTRY IS NAMED AFTER THE BASE IT DELEGATES TO, AND THAT IS THE EXTRACTOR'S HABIT
    ///
    /// A constructor whose whole body is a base-class initialiser is recorded under the base's name:
    /// entry 101 is `e101_OperationNode` for `LocalOpNode::LocalOpNode`
    /// ([`super::tf_flattening_local_regions::LocalOpNode::new`]), and the pattern repeats down this
    /// hierarchy — entry 137 is `e137_TPMVVector` for `TPMVVectorLoad`'s constructor (`hpp:397`) and
    /// entry 138 is `e138_TPMVComposite` for `TPMVCompositeLoad`'s (`hpp:519`). The anchor keeps the
    /// scheduler's name; the item is this constructor.
    ///
    /// ⛔ AND A FORWARDING CONSTRUCTOR STILL DECIDES SOMETHING: that `TPMVVector` adds no state of
    /// its own. Everything a vector transfer needs beyond the base arrives in `initialize()`, which
    /// is why the derived classes below it are the ones that declare fields.
    #[must_use]
    pub fn new(mem_op: &'p DfirOp, comp: DfirUnit) -> TpmvVector<'p> {
        TpmvVector {
            base: TpmvBase::new(mem_op, comp),
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::dialects::{Index, dataflow, vectorchain};
    use crate::islands::dataflow_ir::link::{Link, Lxlu as LxluUnit, Sfp as SfpUnit};
    use crate::islands::dataflow_ir::ty::{ElemType, MemRef, Vector};

    /// `%mem_view = dataflow.get_paged_logical_memory_view %lx, %c128 …` — the paged view being
    /// de-paged (`paged_mem_view_loads.mlir:288`).
    const VIEW: Val = Val(10);
    /// `%lxlu = dataflow.get_unit {…, type = "lxlu"}`.
    const LXLU: Val = Val(4);
    /// `%sfp0 = dataflow.get_unit {…, type = "sfp"}`.
    const SFP: Val = Val(3);
    /// `%c16` — the rotate's position operand.
    const C16: Val = Val(5);

    /// `vector<64xf16>` — the type every transfer in the vendor's vector cases carries.
    const LANES: Vector = Vector {
        len: 64,
        elem: ElemType::F16,
    };

    /// `memref<?x64x4xf16>` (`paged_mem_view_loads.mlir:296`), with the dynamic extent written as
    /// its trip count of 1. Nothing under test reads the shape.
    fn view_ty() -> MemRef {
        MemRef {
            shape: vec![1, 64, 4],
            elem: ElemType::F16,
        }
    }

    /// `%mem_view[%c0, %arg1 * 3, %arg2 * 2 + %c2]` — three subscripts, whose contents nothing under
    /// test reads.
    fn indices() -> Vec<Index> {
        vec![Index::Val(Val(1)), Index::Val(Val(20)), Index::Val(Val(21))]
    }

    /// `%load = agen.vector_load %mem_view[…] : memref<?x64x4xf16>, vector<64xf16>`
    /// (`paged_mem_view_loads.mlir:297`).
    fn vector_load(result: Val) -> DfirOp {
        DfirOp::Agen(agen::Op::VectorLoad {
            result,
            view: VIEW,
            indices: indices(),
            view_ty: view_ty(),
            ty: LANES,
        })
    }

    /// `agen.vector_store %data, %dst_mem_view[…] : memref<64x2x64x2x3xf16>, vector<64xf16>`
    /// (`paged_mem_view_load_and_store.mlir:1289`).
    fn vector_store(value: Val) -> DfirOp {
        DfirOp::Agen(agen::Op::VectorStore {
            value,
            view: Val(11),
            indices: indices(),
            view_ty: view_ty(),
            ty: LANES,
        })
    }

    /// `%rot = vectorchain.rotate %load, %c16 : vector<64xf16>, index, vector<64xf16>`
    /// (`paged_mem_view_loads.mlir:298`).
    fn rotate(result: Val, input: Val) -> DfirOp {
        DfirOp::VectorChain(vectorchain::Op::Rotate {
            result,
            input,
            position: C16,
            right_shift: true,
            input_ty: LANES,
            ty: LANES,
        })
    }

    /// `dataflow.send %sfp0, %rot {} : vector<64xf16>` (`paged_mem_view_loads.mlir:299`).
    fn send(data: Val) -> DfirOp {
        let (to, _) = Link::<LxluUnit, SfpUnit>::between(LXLU, SFP).ends();
        DfirOp::Dataflow(dataflow::Op::Send {
            to,
            data,
            ty: LANES,
        })
    }

    /// 🎯 129/384 — THE VENDOR'S OWN THREE-OP CHAIN, TORN DOWN CONSUMER-FIRST.
    ///
    /// ⭐⭐ `%load = agen.vector_load …` / `%rot = vectorchain.rotate %load, %c16` /
    /// `dataflow.send %sfp0, %rot` is `paged_mem_view_loads.mlir:297-299` verbatim — a paged vector
    /// load with a real chain. `VectorLoadOp::getUseChain` collects it load-first
    /// (`Agen.cpp:115-136`) and `eraseOpAndUseChain` walks it BACKWARDS (`:176-177`), so the send
    /// goes first and the load last.
    #[test]
    fn a_paged_loads_chain_is_erased_from_its_send_back_to_the_load() {
        let program = vec![vector_load(Val(31)), rotate(Val(41), Val(31)), send(Val(41))];
        let load = VectorLoadOp::of(&program[0]).expect("an agen.vector_load");

        assert_eq!(
            vector_load_use_chain(load, &program),
            UseChain::ConsumerWard(vec![&program[0], &program[1], &program[2]]),
            "the chain runs load, rotate, send"
        );
        assert_eq!(
            erase_vector_load_and_use_chain(load, &program),
            vec![&program[2], &program[1], &program[0]],
            "and it is erased send, rotate, load"
        );
    }

    /// 🎯 129/384 — A LOAD SENT STRAIGHT OUT IS A TWO-OP CHAIN.
    ///
    /// `%load2 = agen.vector_load %mem_view2[%c2, 3, %c4]` / `dataflow.send %sfp0, %load2`
    /// (`paged_mem_view_loads.mlir:309-310`) — the `use_chain.size() >= 2` assert's minimum case.
    #[test]
    fn a_load_sent_directly_erases_the_send_then_the_load() {
        let program = vec![vector_load(Val(31)), send(Val(31))];
        let load = VectorLoadOp::of(&program[0]).expect("an agen.vector_load");
        assert_eq!(
            erase_vector_load_and_use_chain(load, &program),
            vec![&program[1], &program[0]]
        );
    }

    /// 🎯 129/384 — ⛔ THE REFERENCE'S UNREACHABLE BRANCH IS THIS PORT'S NON-LINEAR CASE.
    ///
    /// `getUseChain` asserts on a result that is not read exactly once; here that answers
    /// [`UseChain::None`], and then `if (use_chain.empty()) { op->erase(); }`
    /// (`Agen.cpp:173-175`) — dead in the C++ — is the branch that runs: remove the load, leave what
    /// reads it alone.
    #[test]
    fn a_load_without_a_linear_chain_erases_only_itself() {
        // Nothing reads the load: `assert(res.hasOneUse())`.
        let unread = vec![vector_load(Val(31))];
        let load = VectorLoadOp::of(&unread[0]).expect("an agen.vector_load");
        assert_eq!(vector_load_use_chain(load, &unread), UseChain::None);
        assert_eq!(
            erase_vector_load_and_use_chain(load, &unread),
            vec![&unread[0]]
        );

        // Two readers: the same assert, the same answer.
        let forked = vec![
            vector_load(Val(31)),
            send(Val(31)),
            rotate(Val(41), Val(31)),
        ];
        let load = VectorLoadOp::of(&forked[0]).expect("an agen.vector_load");
        assert_eq!(vector_load_use_chain(load, &forked), UseChain::None);
        assert_eq!(
            erase_vector_load_and_use_chain(load, &forked),
            vec![&forked[0]]
        );
    }

    /// 🎯 129/384 — ⛔ THE CAST IS A DOOR: A STORE IS NOT A LOAD.
    #[test]
    fn only_a_vector_load_passes_the_load_cast() {
        assert!(VectorLoadOp::of(&vector_store(Val(31))).is_none());
        assert!(VectorLoadOp::of(&send(Val(31))).is_none());
        assert!(VectorStoreOp::of(&vector_load(Val(31))).is_none());
        let store = vector_store(Val(31));
        assert_eq!(
            VectorStoreOp::of(&store).map(|store| store.value),
            Some(Val(31)),
            "getValueToStore()"
        );
    }

    /// 🎯 129/384 — ⛔ THE TWO ERASE LOOPS ARE ONE RULE, AND THE DIRECTION IS WHAT THEY DISAGREE ON.
    ///
    /// `VectorLoadOp::eraseOpAndUseChain` reverses its chain (`Agen.cpp:176-177`) and
    /// `VectorStoreOp::eraseOpAndUseChain` does not (`:257`), because a store's chain is collected
    /// producer-ward with the store itself first (`:207-222`). Both end up consumer-first.
    #[test]
    fn the_direction_decides_whether_the_chain_is_reversed() {
        let program = vec![vector_load(Val(31)), rotate(Val(41), Val(31)), send(Val(41))];
        let ops: Vec<&DfirOp> = program.iter().collect();

        assert_eq!(
            UseChain::ConsumerWard(ops.clone()).consumer_first(),
            vec![ops[2], ops[1], ops[0]]
        );
        assert_eq!(
            UseChain::ProducerWard(ops.clone()).consumer_first(),
            vec![ops[0], ops[1], ops[2]]
        );
        assert!(UseChain::None.consumer_first().is_empty());
    }

    /// 🎯 130/384 — THE VENDOR'S LOAD-AND-STORE PATTERN, AND NOTHING ELSE.
    ///
    /// ⭐⭐ `%data = agen.vector_load %src_mem_view[…]` followed by
    /// `agen.vector_store %data, %dst_mem_view[…]` is
    /// `paged_mem_view_load_and_store.mlir:1287-1289` — the input a `TPMVVectorLoadStore` is built
    /// for, and the shape both `DT_CHECK`s hold on.
    #[test]
    fn the_stores_op_is_the_loads_single_storing_user() {
        let pattern = vec![vector_load(Val(31)), vector_store(Val(31))];
        let load = VectorLoadOp::of(&pattern[0]).expect("an agen.vector_load");
        assert_eq!(
            store_op(load, &pattern).map(|store| store.op),
            Some(&pattern[1])
        );

        // ⛔ `DT_CHECK(store_op)` — the single user is not a store.
        let sent = vec![vector_load(Val(31)), send(Val(31))];
        let load = VectorLoadOp::of(&sent[0]).expect("an agen.vector_load");
        assert!(store_op(load, &sent).is_none());

        // ⛔ `DT_CHECK(load_op.getResult().hasOneUse())` — a store AND a send read it.
        let both = vec![
            vector_load(Val(31)),
            vector_store(Val(31)),
            send(Val(31)),
        ];
        let load = VectorLoadOp::of(&both[0]).expect("an agen.vector_load");
        assert!(store_op(load, &both).is_none());
    }

    /// 🎯 131/384 — EVERY TIME BOUND BECOMES A ZERO-BASED INCLUSIVE RANGE, APPENDED.
    ///
    /// ⭐⭐ THE VENDOR'S COMPOSITE LOAD PINS THE ARITHMETIC. `time_symbols(%c3)` with
    /// `time_set = affine_set<(d0, d1, d2, d3, d4)[s0] : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 >= 0,
    /// d2 >= 0, -d2 + s0 - 1 >= 0, d3 >= 0, -d3 + 2 >= 0, d4 >= 0, -d4 + 1 >= 0)>`
    /// (`paged_mem_view_loads.mlir:332-337`) is five time dims of 2, 1, 3, 3 and 2 steps, so the
    /// ranges are `[0,1] [0,0] [0,2] [0,2] [0,1]`.
    ///
    /// ⛔ AND IT APPENDS AFTER `calculateIndicesRanges` (`TransformPagedMemViewImpl.cpp:1010-1017`),
    /// which is what makes the symbol numbering of entry 132 correct.
    #[test]
    fn each_time_bound_becomes_a_zero_based_inclusive_range_appended_to_the_vector() {
        let bounds: Vec<TimeSteps> = [2, 1, 3, 3, 2]
            .into_iter()
            .map(|steps| TimeSteps::of(TimeBound::Steps(steps)).expect("a real step count"))
            .collect();

        // The one range `calculateIndicesRanges` left behind for the single non-time subscript
        // `%arg9` of `agen.composite_load %mem_view[%arg9, 0, 0, 0, 0]` (`:331`), whose loop runs
        // `affine.for %arg9 = 0 to 56` (`:322`).
        let mut indices_ranges = vec![IvRange { lb: 0, ub: 55 }];
        add_time_dim_indices_ranges(&bounds, &mut indices_ranges);

        assert_eq!(
            indices_ranges,
            vec![
                IvRange { lb: 0, ub: 55 },
                IvRange { lb: 0, ub: 1 },
                IvRange { lb: 0, ub: 0 },
                IvRange { lb: 0, ub: 2 },
                IvRange { lb: 0, ub: 2 },
                IvRange { lb: 0, ub: 1 },
            ]
        );
    }

    /// 🎯 131/384 — ⛔ `DT_CHECK_MSG(b - 1 >= 0, "no special time bound values should exist")` IS
    /// THE TYPE.
    #[test]
    fn a_special_time_bound_never_becomes_a_step_count() {
        assert_eq!(TimeSteps::of(TimeBound::Coalesced), None, "kCoalesced = -2");
        assert_eq!(TimeSteps::of(TimeBound::Variable), None, "kInvalid = -1");
        assert_eq!(TimeSteps::of(TimeBound::Steps(0)), None, "b - 1 < 0");
        assert_eq!(
            TimeSteps::of(TimeBound::Steps(1)).map(TimeSteps::last_index),
            Some(0),
            "a single-step dim pins its index to zero"
        );
        assert_eq!(
            TimeSteps::of(TimeBound::Steps(3)).map(TimeSteps::last_index),
            Some(2)
        );
    }

    /// 🎯 132/384 — THE INNERMOST PAGE-DEPENDENT TIME DIM IS THE CUT, AND THE SCAN DIRECTION IS WHY.
    ///
    /// ⭐⭐ THE VENDOR'S CASE IS *"Composite load with hyperrectangular subscripts with some time dims
    /// preserved"* (`paged_mem_view_loads.mlir:323`): five time dims over a
    /// `memref<128x2x1x1x2xi8>`, so `subscripts_map_.getNumDims()` is 5 and time dim `i` is symbol
    /// `i + 5`. With symbols 5 and 7 page-dependent the answer is dim 2 — the INNERMOST hit, so dims
    /// 3 and 4 stay folded into the transfer and dims 0-2 become explicit loops.
    #[test]
    fn the_innermost_page_dependent_time_dim_is_the_one_returned() {
        // The vendor's `time_set`, whose only property read here is its dimension count.
        let time_set = IntegerSet::from_sizes(&[2, 1, 3, 3, 2]);
        let num_non_time_dims = NonTimeDims(5);

        let syms: BTreeSet<PageSelSym> = [PageSelSym(5), PageSelSym(7)].into_iter().collect();
        assert_eq!(
            identify_time_dim_for_explicit_loops(&time_set, &syms, num_non_time_dims),
            Some(TimeDim(2))
        );

        // ⛔ THE OUTERMOST HIT IS NOT THE ANSWER: dim 0 alone still cuts at dim 0.
        let outermost: BTreeSet<PageSelSym> = [PageSelSym(5)].into_iter().collect();
        assert_eq!(
            identify_time_dim_for_explicit_loops(&time_set, &outermost, num_non_time_dims),
            Some(TimeDim(0))
        );

        // The innermost dim being page-dependent means every time loop is explicit.
        let innermost: BTreeSet<PageSelSym> = [PageSelSym(9)].into_iter().collect();
        assert_eq!(
            identify_time_dim_for_explicit_loops(&time_set, &innermost, num_non_time_dims),
            Some(TimeDim(4))
        );
    }

    /// 🎯 132/384 — ⛔ `-1` IS "ALL TIME DIMS CAN BE PRESERVED", AND IT IS A QUESTION NOT AN INDEX.
    ///
    /// `// A explicit_loop_dim of -1 indicates all time dims can be preserved.`
    /// `if (explicit_loop_dim > -1)` (`TransformPagedMemViewImpl.cpp:1028-1030`).
    #[test]
    fn no_page_dependent_symbol_preserves_every_time_dim() {
        let time_set = IntegerSet::from_sizes(&[2, 1, 3, 3, 2]);
        assert_eq!(
            identify_time_dim_for_explicit_loops(&time_set, &BTreeSet::new(), NonTimeDims(5)),
            None
        );

        // ⛔ THE OFFSET IS NOT OPTIONAL: the same symbols read against the wrong non-time count find
        // nothing, which is exactly the confusion `NonTimeDims` exists to make unwritable.
        let syms: BTreeSet<PageSelSym> = [PageSelSym(5), PageSelSym(7)].into_iter().collect();
        assert_eq!(
            identify_time_dim_for_explicit_loops(&time_set, &syms, NonTimeDims(0)),
            None,
            "symbols 5 and 7 are not time dims 5 and 7"
        );

        // A time set with no dimensions has nothing to scan.
        let empty = IntegerSet::from_sizes(&[]);
        assert_eq!(
            identify_time_dim_for_explicit_loops(&empty, &syms, NonTimeDims(5)),
            None
        );
    }

    /// 🎯 132/384 — THE SYMBOL A TIME DIM OCCUPIES IS THE SLOT ENTRY 131 APPENDED IT TO.
    ///
    /// ⭐⭐ THIS IS THE JOIN BETWEEN THE TWO UNITS, AND IT IS CHECKABLE:
    /// `addConstraintsForIVRanges` reads `indices_ranges[sym_idx]`
    /// (`TransformPagedMemViewImpl.cpp:99-104`), and [`add_time_dim_indices_ranges`] put time dim `i`
    /// at position `num_non_time_dims + i`.
    #[test]
    fn a_time_dims_symbol_indexes_the_range_entry_131_appended() {
        let bounds: Vec<TimeSteps> = [2, 1, 3, 3, 2]
            .into_iter()
            .map(|steps| TimeSteps::of(TimeBound::Steps(steps)).expect("a real step count"))
            .collect();
        // One non-time subscript, as in the vendor's `%mem_view[%arg9, 0, 0, 0, 0]`.
        let mut indices_ranges = vec![IvRange { lb: 0, ub: 55 }];
        let num_non_time_dims = NonTimeDims(1);
        add_time_dim_indices_ranges(&bounds, &mut indices_ranges);

        for (i, bound) in bounds.iter().enumerate() {
            let dim = TimeDim(u32::try_from(i).expect("five dims fit a u32"));
            let sym = num_non_time_dims.sym_for(dim);
            assert_eq!(
                indices_ranges[sym.0 as usize],
                IvRange {
                    lb: 0,
                    ub: bound.last_index()
                },
                "time dim {i} is symbol {}",
                sym.0
            );
        }
    }

    /// 🎯 133/384, 134/384, 135/384 — THE BASE KNOWS NO CHAIN, CLONES NOTHING, AND ERASES ONE OP.
    ///
    /// ⛔⛔ AND THAT IS LIVE BEHAVIOUR FOR FOUR OF THE SIX CONCRETE CLASSES —
    /// `TPMVVectorLoadStore` and every composite inherit all three.
    #[test]
    fn the_base_class_knows_no_use_chain_and_erases_only_the_mem_op() {
        let program = vec![vector_load(Val(31)), rotate(Val(41), Val(31)), send(Val(41))];
        let base = TpmvBase::new(&program[0], DfirUnit::Lxlu);

        // Entry 133: `return {};` — even though this very op HAS a chain, which entry 129 finds.
        assert_eq!(base.use_chain(&program[0]), UseChain::None);
        assert!(base.use_chain(&program[0]).consumer_first().is_empty());

        // Entry 134: no clones.
        assert!(base.clone_use_chain(&program[0], &program[0]).is_empty());

        // Entry 135: `mem_op->erase();` — the op and nothing downstream of it.
        assert_eq!(
            base.erase_mem_op_and_use_chain(&program[0]),
            vec![&program[0]]
        );
        assert_eq!(
            erase_vector_load_and_use_chain(
                VectorLoadOp::of(&program[0]).expect("an agen.vector_load"),
                &program
            )
            .len(),
            3,
            "and the override erases three, which is why it overrides"
        );
    }

    /// 🎯 136/384 — THE CONSTRUCTOR SEEDS `mem_ops_` WITH EXACTLY ONE OP AND CARRIES THE COMPONENT.
    ///
    /// ⭐ `DT_CHECK(mem_ops_.size() == 1)` OPENS EVERY `initialize()`
    /// (`TransformPagedMemViewImpl.cpp:659`, `:707`, `:756`, `:1081`, `:1134`, `:1187`), so the
    /// singleton is the invariant this constructor establishes.
    #[test]
    fn the_vector_constructor_seeds_one_mem_op_and_the_component() {
        let program = vec![vector_load(Val(31)), send(Val(31))];
        let tpmv = TpmvVector::new(&program[0], DfirUnit::Lxlu);

        assert_eq!(tpmv.base.mem_ops, vec![&program[0]]);
        assert_eq!(tpmv.base.comp, DfirUnit::Lxlu);

        // ⭐ AND IT ADDS NO STATE OF ITS OWN: the base is all of it.
        assert_eq!(tpmv.base, TpmvBase::new(&program[0], DfirUnit::Lxlu));
    }
}
