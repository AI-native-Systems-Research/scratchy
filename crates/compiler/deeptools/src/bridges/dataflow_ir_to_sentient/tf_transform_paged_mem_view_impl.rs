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

use super::agen_access_details::{
    AccessContainer, AccessDetailsAffineComposite, ConstructedDetails, MemoryOperandIndex,
    TimeBound, TimeDim, TimeStepsInfo, construct_time_steps_info,
};
use super::agen_helper::{AgenOpKind, store_op_from_load_store_pattern};
use super::tf_utils::{
    CountedLoop, LoopBound, create_for_op_with_additional_return_value,
    get_dataflow_for_loop_info_if_iv,
};
// ⭐ THE MANAGER'S `dyn_cast` HANDLE, aliased because this file's `PagedMemView` is the island op it
// wraps — `cast<GetPagedLogicalMemoryViewOp>` at `:396` and `:663` is [`PagedMemViewHandle::of`].
use super::tf_mutable_addr_splitting::{
    clone_composite_with_new_access_info, concatenate_maps, nest_explicit_time_loops_over,
    update_subscripts_and_indices_for_explicit_time_loops, update_time_set_for_explicit_dims,
};
use super::tf_transform_paged_mem_view_manager::PagedMemView as PagedMemViewHandle;
use super::vc_vector_operands::access_map;
use crate::islands::dataflow_ir::dialects::arith::CmpIPredicate;
use crate::islands::dataflow_ir::dialects::dataflow::{Page, PageRect, PagedMemView};
use crate::islands::dataflow_ir::dialects::{
    self, Index, Op as DfirOp, Val, affine, agen, arith, dataflow, defining_op, results, scf, uses,
    vectorchain,
};
use crate::islands::dataflow_ir::ty::{
    AffineExpr, AffineMap, BoundType, Constraint, FlatConstraints, IntegerSet, MemRef, ScalarTy,
    Vector,
};
use crate::islands::dataflow_ir::{ValueMapping, Values};
use crate::units::DfirUnit;
use core::num::NonZeroU32;
use std::collections::BTreeSet;

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 121/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ONE BOUND OF ONE SUBSCRIPT, AS THE THREE OPS THAT TEST IT — an `arith.constant`, an
/// `arith.cmpi` and a one-armed `scf.if`.
///
/// ⭐ THE THREE ARE ONE THING. `arith.cmpi` takes two operands of the same type, so the bound has to
/// be minted as a constant first (see [`arith::Op::Compare`]'s `rhs`), and the `scf.if` exists only
/// to hold what the comparison guards. Naming them separately would let a caller emit two of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundGuard {
    /// The subscript under test — `lhs`, the same value in every guard of one dimension.
    pub lhs: Val,
    /// `sge` for a lower bound, `sle` for an upper one.
    pub predicate: CmpIPredicate,
    /// The literal the `arith.constant` binds — `rhs_lb` or `rhs_ub`.
    pub bound: i64,
    /// The value that constant binds.
    pub constant: Val,
    /// The `i1` the `arith.cmpi` binds, which the `scf.if` branches on.
    pub cond: Val,
}

/// THE CONDITION UNDER WHICH ONE ACCESS REACHES ONE PAGE — a nest of one-armed `scf.if`s, still
/// empty.
///
/// # ⛔⛔ THE GUARDS ARE HELD, NOT YET EMITTED, BECAUSE THE REFERENCE MOVES THE BUILDER INSTEAD
///
/// `createInequalityCondition` emits the lower bound's three ops, points the builder at the `scf.if`
/// it just made (`builder = lb_if_op.getThenBodyBuilder()`, `:270`) and emits the upper bound's three
/// INSIDE it — so the ops it creates and the statements it will guard interleave, and the caller's
/// builder is left at the innermost point. A returned nest with a hole in the middle cannot be
/// written down; a list of guards plus [`Condition::wrap`] can, and it emits the same ops in the same
/// order with the same nesting.
///
/// ⭐ WHICH ALSO MAKES THE NESTING OF SUCCESSIVE DIMENSIONS ONE OPERATION — see
/// [`Condition::and_then`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Condition {
    /// The guards, OUTERMOST FIRST: the order the reference emits them in, which is the order they
    /// nest in.
    pub guards: Vec<BoundGuard>,
}

impl Condition {
    /// THE STATEMENTS, WRAPPED IN THIS CONDITION — the whole emission of a `create*Condition` call.
    ///
    /// The list reads: `arith.constant`, `arith.cmpi`, `scf.if` — and inside that `scf.if`'s then
    /// region, the next guard's three, and inside the innermost one, `guarded`.
    ///
    /// ⭐ BUILT INSIDE OUT, so the nest needs no descent into a region it has already built and there
    /// is no position a walk could fail to find.
    #[must_use]
    pub fn wrap(&self, guarded: Vec<DfirOp>) -> Vec<DfirOp> {
        let mut ops = guarded;
        for guard in self.guards.iter().rev() {
            ops = vec![
                DfirOp::Arith(arith::Op::Constant {
                    result: guard.constant,
                    value: guard.bound,
                }),
                DfirOp::Arith(arith::Op::Compare {
                    result: guard.cond,
                    predicate: guard.predicate,
                    lhs: guard.lhs,
                    rhs: guard.constant,
                }),
                DfirOp::Scf(scf::Op::If {
                    cond: guard.cond,
                    results: Vec::new(),
                    body: ops,
                    // ⛔ ONE-ARMED, WHICH IS THE `false` LAST ARGUMENT OF EVERY `scf::IfOp::create`
                    // in this file. A page a subscript cannot reach has nothing to do, not something
                    // else to do.
                    else_body: Vec::new(),
                    dbg_name: None,
                }),
            ];
        }
        ops
    }

    /// THIS CONDITION WITH `inner`'s GUARDS INSIDE ITS OWN.
    ///
    /// ⭐⭐ THIS IS HOW SUCCESSIVE DIMENSIONS NEST. `createConditionsForHyperRectSubscripts` walks the
    /// subscript map dimension by dimension and each call emits into what the previous one created —
    /// the builder is never moved back out (`TransformPagedMemViewImpl.cpp:289-341`, entry 198) — so
    /// dimension 1's guards sit inside dimension 0's then-region. Concatenation is that nesting, and
    /// it keeps [`Condition::wrap`] the only place ops are built.
    #[must_use]
    pub fn and_then(mut self, inner: Condition) -> Condition {
        self.guards.extend(inner.guards);
        self
    }
}

/// Replaces: e121_createInequalityCondition
///
/// **121/384** `TPMVBase::createInequalityCondition` —
/// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:263` (14L).
///
/// ```cpp
/// Operation *TPMVBase::createInequalityCondition(OpBuilder &builder, Value &lhs,
///                                                int64_t rhs_lb, int64_t rhs_ub) {
///   auto lb_const = arith::ConstantIndexOp::create(builder, lhs.getLoc(), rhs_lb);
///   auto lb_cond = mlir::arith::CmpIOp::create(
///       builder, lhs.getLoc(), mlir::arith::CmpIPredicate::sge, lhs, lb_const);
///   auto lb_if_op = mlir::scf::IfOp::create(builder, lhs.getLoc(),
///                                           lb_cond.getResult(), false);
///   builder = lb_if_op.getThenBodyBuilder();
///
///   auto ub_const =
///       mlir::arith::ConstantIndexOp::create(builder, lhs.getLoc(), rhs_ub);
///   auto ub_cond = mlir::arith::CmpIOp::create(
///       builder, lhs.getLoc(), mlir::arith::CmpIPredicate::sle, lhs, ub_const);
///   return mlir::scf::IfOp::create(builder, lhs.getLoc(), ub_cond.getResult(),
///                                  false);
/// }
/// ```
///
/// # ⛔⛔ TWO CONDITIONALS, NESTED — NOT ONE `andi` OF TWO COMPARISONS
///
/// A span is `lhs >= lb` AND `lhs <= ub`, and the obvious spelling of that is one `arith.andi` over
/// two `cmpi`s under one `scf.if`. The reference emits **six** ops in **two** nested one-armed
/// `scf.if`s, and the shape is load-bearing downstream: `CFGSDataflowConditionalTree` builds its tree
/// out of `scf.if`s whose conditions are `cmpi`s, and reads the LHS and RHS straight off the
/// comparison ([`super::tf_cfgs_dataflow_conditional_tree`], entry 098). A conjunction behind an
/// `andi` is a condition it cannot decompose, so the tree that merges these conditionals across pages
/// would see one opaque predicate per access instead of two comparable bounds.
///
/// ⭐ AND THE PREDICATES ARE SIGNED — `sge` then `sle`, in that order, on `index`. A page's bounds
/// come out of [`crate::islands::dataflow_ir::ty::IntegerSet::constant_bound`], which is signed
/// arithmetic over an `affine_set`, so an unsigned comparison would be a different question.
///
/// ⛔ THE `builder` PARAMETER IS BY REFERENCE AND IS REASSIGNED (`:270`), which is the whole reason
/// this returns a [`Condition`] rather than a list of ops: the caller's insertion point ends up
/// INSIDE the inner conditional, and everything emitted afterwards is what the two comparisons guard.
/// [`Condition::wrap`] is that, and `lhs.getLoc()` — threaded through all six creations because MLIR
/// ops carry source locations — is the mechanism this island has no equivalent of.
#[must_use]
pub fn create_inequality_condition(
    vals: &mut Values,
    lhs: Val,
    rhs_lb: i64,
    rhs_ub: i64,
) -> Condition {
    // ⭐ MINTED IN THE REFERENCE'S OWN ORDER: constant, comparison, then the next bound's pair. The
    // numbering is observable — it is what the printed IR names the values — so the order is part of
    // the port.
    let lb_const = vals.mint();
    let lb_cond = vals.mint();
    let ub_const = vals.mint();
    let ub_cond = vals.mint();
    Condition {
        guards: vec![
            BoundGuard {
                lhs,
                predicate: CmpIPredicate::Sge,
                bound: rhs_lb,
                constant: lb_const,
                cond: lb_cond,
            },
            BoundGuard {
                lhs,
                predicate: CmpIPredicate::Sle,
                bound: rhs_ub,
                constant: ub_const,
                cond: ub_cond,
            },
        ],
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 122/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHAT AN INSERT REFERENCE IS — the two things `insert_refs[i]` holds, and they are not the same
/// kind of thing.
///
/// ⭐ `createConditionsFor*Subscripts` starts from `insert_refs = mem_ops_` and overwrites an entry
/// with the conditional it created for that access (`TransformPagedMemViewImpl.cpp:201` and
/// `:329-333`). So an entry is a conditional exactly when a condition was created for it, and the
/// `dyn_cast<scf::IfOp>` that [`set_builder_to_insert_ref`] does is asking which of those two
/// happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertRef<'a> {
    /// A conditional this pass created for the access — the guarded statements go in its then-region.
    Conditional(&'a Condition),
    /// The memory operation itself, still unguarded — the statements go in its own block.
    MemOp,
}

/// Replaces: e122_setBuilderToInsertRef
///
/// **122/384** `TPMVBase::setBuilderToInsertRef` —
/// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:280` (6L).
///
/// ```cpp
/// void TPMVBase::setBuilderToInsertRef(OpBuilder &builder,
///                                      Operation *insert_ref) {
///   DT_CHECK(insert_ref);
///   if (auto if_op = dyn_cast<scf::IfOp>(insert_ref))
///     builder = if_op.getThenBodyBuilder();
///   else
///     builder.setInsertionPoint(insert_ref);
/// }
/// ```
///
/// # ⛔⛔ THE TWO ARMS ARE TWO DIFFERENT PLACES, AND THE DIFFERENCE IS THE GUARD
///
/// `getThenBodyBuilder()` appends INSIDE the conditional, so the new access only happens when the
/// page is the right one. `setInsertionPoint(insert_ref)` puts it in the reference's own block,
/// immediately before the op it names — unguarded, which is correct precisely when the page needs no
/// condition (a subscript that can only reach one page). Getting these the wrong way round emits an
/// access that reads the wrong page unconditionally, and no verifier objects.
///
/// ⭐ SO THIS RETURNS THE STATEMENTS PLACED, NOT A CURSOR. The `MemOp` arm hands them back
/// unwrapped for the caller to splice at the reference's position; entry 309 then erases the original
/// access and its use chain (entry 129), so "immediately before it" ends up being "in its place".
///
/// ⛔ `DT_CHECK(insert_ref)` IS GONE BECAUSE A REFERENCE CANNOT BE NULL. The check guards the
/// `dyn_cast` below it against a null `Operation*`; [`InsertRef`] has no such state — its two
/// variants are the two things the cast distinguishes, not "present" and "absent".
#[must_use]
pub fn set_builder_to_insert_ref(insert_ref: InsertRef<'_>, ops: Vec<DfirOp>) -> Vec<DfirOp> {
    match insert_ref {
        // `builder = if_op.getThenBodyBuilder()`.
        InsertRef::Conditional(condition) => condition.wrap(ops),
        // `builder.setInsertionPoint(insert_ref)` — the reference's own block.
        InsertRef::MemOp => ops,
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 123/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ONE ELEMENT COORDINATE ALONG ONE VIEW DIMENSION — where a page begins on that axis.
///
/// ⭐ SIGNED, BECAUSE IT IS A BOUND AND NOT AN EXTENT.
/// [`crate::islands::dataflow_ir::ty::IntegerSet::constant_bound`] answers over signed arithmetic,
/// and entry 259 SUBTRACTS these from a subscript map (`<original subscript> - <start element>`,
/// `TransformPagedMemViewImpl.cpp:559-573`), where a `u64` would be the wrong thing to reach for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StartElement(pub i64);

/// Replaces: e123_calculateStartElementsForPage
///
/// **123/384** `TPMVBase::calculateStartElementsForPage` —
/// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:532` (10L).
///
/// ```cpp
/// // Calculates the elements that identify the start of the page.
/// // Example:
/// //   page idx_set: (d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0,
/// //                                 d1 - 2 >= 0, -d1 + 3 >= 0,
/// //                                 d2 >= 0, -d2 + 1 >= 0)
/// //   Start elements of this page: [0, 2, 0]
/// SmallVector<int64_t, 16> TPMVBase::calculateStartElementsForPage(
///     FlatLinearValueConstraints &page_sel_constraints) {
///   SmallVector<int64_t, 16> start_elements;
///   for (int dim = 0; dim < page_sel_constraints.getNumDimVars(); ++dim) {
///     auto lb = page_sel_constraints.getConstantBound(
///         mlir::presburger::BoundType::LB, dim);
///     DT_CHECK_MSG(lb.has_value(), "expected constant lower bound");
///     start_elements.emplace_back((int64_t)lb.value());
///   }
///
///   return start_elements;
/// }
/// ```
///
/// # ⛔⛔ THE LOWER BOUND OF EVERY DIMENSION, INCLUDING THE ONES THAT START AT ZERO
///
/// The result is positional — entry 259 subtracts `start_elements[dim]` from result `dim` of the
/// subscript map — so a dimension whose page starts at 0 contributes a 0 and not nothing. The
/// example above is the whole specification: three dimensions, `[0, 2, 0]`, and only the middle page
/// is displaced.
///
/// ⭐⭐ AND `DT_CHECK_MSG(lb.has_value(), "expected constant lower bound")` IS NOW THE TYPE.
/// [`crate::islands::dataflow_ir::dialects::dataflow::PageRect`] holds one
/// [`crate::islands::dataflow_ir::dialects::dataflow::PageSpan`] per dimension, and a span's `lo` IS
/// the constant lower bound — so there is no page whose bound is missing to check for. ⭐ THE
/// PARAMETER TYPE IS THEREFORE THE GUARD, not a comment about one: a `FlatLinearValueConstraints`
/// can hold any system at all, and the reference has to ask each dimension whether it happens to be
/// bounded. The test below pins the two against each other — the span's `lo` beside
/// [`crate::islands::dataflow_ir::ty::IntegerSet::constant_bound`]`(Lb, dim)` on the very set the
/// reference's example writes — so the type guard and the ported MLIR routine are checked to agree
/// rather than merely asserted to.
#[must_use]
pub fn calculate_start_elements_for_page(page_sel_constraints: &PageRect) -> Vec<StartElement> {
    page_sel_constraints
        .spans
        .iter()
        .map(|span| StartElement(span.lo))
        .collect()
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 124/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// A NON-PAGED VIEW OVER ONE PAGE — the ops that bind it, and the value they bind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonPagedMemView {
    /// `arith.addi` then `dataflow.get_logical_memory_view`, in emission order.
    pub ops: Vec<DfirOp>,
    /// The view the new access reads — `GetLogicalMemoryViewOp`'s result.
    pub view: Val,
    /// The sum the view starts at — `new_start_addr`, kept because it is an op's result and a caller
    /// re-reading it should not have to dig it out of `ops`.
    pub start_addr: Val,
}

/// Replaces: e124_createNonPagedMemView
///
/// **124/384** `TPMVBase::createNonPagedMemView` —
/// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:545` (10L).
///
/// ```cpp
/// dataflow::GetLogicalMemoryViewOp TPMVBase::createNonPagedMemView(
///     OpBuilder &builder, dataflow::GetPagedLogicalMemoryViewOp &paged_mem_view,
///     int page_idx) {
///   auto new_start_addr = mlir::arith::AddIOp::create(
///       builder, paged_mem_view->getLoc(), paged_mem_view.getStartAddr(),
///       paged_mem_view.getPageStartAddrs()[page_idx]);
///
///   return mlir::dataflow::GetLogicalMemoryViewOp::create(
///       builder, new_start_addr->getLoc(),
///       cast<MemRefType>(paged_mem_view.getResult().getType()),
///       paged_mem_view.getUnit(), new_start_addr->getResult(0),
///       paged_mem_view.getLayoutMap());
/// }
/// ```
///
/// # ⛔⛔ THIS IS WHAT THE WHOLE PASS EXISTS TO PRODUCE
///
/// Everything else in this file decides WHICH page and under WHAT condition; these two ops are the
/// answer. A page's `start_addr` is RELATIVE to the view's own, so the address is the SUM
/// (`arith.addi`) — emitting the page's own start would address the page as if the view began at
/// zero, which for view start 0 is right and for every other view silently reads the wrong memory.
///
/// ⭐ AND EVERYTHING ELSE IS INHERITED VERBATIM: the same `$unit`, the same `layout_map`, and the
/// paged view's own result type as the new view's (`cast<MemRefType>`). The pages differ in where
/// they start and in nothing else, which is why one `layout_map` describes them all.
///
/// ⭐ THE PAGE ARRIVES AS A [`Page`] RATHER THAN AN `int page_idx`, so
/// `getPageStartAddrs()[page_idx]` cannot index past the end and cannot be paired with another
/// page's `idx_set` — see [`Page`], which is where the reference's two parallel lists became one.
///
/// ⚠️ `paged_mem_view->getLoc()` then `new_start_addr->getLoc()` — the view is located at the sum it
/// was given, not at the paged view. Pure MLIR bookkeeping; this island carries no locations.
#[must_use]
pub fn create_non_paged_mem_view(
    vals: &mut Values,
    paged_mem_view: &PagedMemView,
    page: &Page,
) -> NonPagedMemView {
    let new_start_addr = vals.mint();
    let view = vals.mint();
    NonPagedMemView {
        ops: vec![
            DfirOp::Arith(arith::Op::AddI(arith::IntBinary {
                result: new_start_addr,
                lhs: paged_mem_view.start_addr,
                rhs: page.start_addr,
                // ⭐ AN ADDRESS IS AN `index`, which is what both operands already are
                // (`Dataflow.td:267-299` declares `Index:$start_addr` and
                // `Variadic<Index>:$page_start_addrs`).
                ty: ScalarTy::Index,
            })),
            DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                result: view,
                from: paged_mem_view.unit,
                start: new_start_addr,
                layout: paged_mem_view.layout.clone(),
                ty: paged_mem_view.ty.clone(),
            }),
        ],
        view,
        start_addr: new_start_addr,
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 125/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// A VIEW A CONDITIONAL BRANCH MAY USE — the clone, if one was needed, and the value to read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClonedMemView {
    /// The cloned `dataflow.get_logical_memory_view`, or nothing when the view was left alone.
    pub ops: Vec<DfirOp>,
    /// The value the access should read — the clone's result, or the original view unchanged.
    pub value: Val,
}

/// Replaces: e125_cloneMemViewIfNonPaged
///
/// **125/384** `TPMVBase::cloneMemViewIfNonPaged` —
/// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:633` (9L).
///
/// ```cpp
/// Value TPMVBase::cloneMemViewIfNonPaged(OpBuilder &builder, Value mem_view) {
///   if (auto non_paged_mem_view =
///           dyn_cast_or_null<dataflow::GetLogicalMemoryViewOp>(
///               mem_view.getDefiningOp())) {
///     auto new_mem_view = builder.clone(*non_paged_mem_view);
///     return new_mem_view->getResult(0);
///   }
///
///   return mem_view;
/// }
/// ```
///
/// # ⛔⛔ THE CLONE EXISTS BECAUSE THE ACCESS IS MOVING INTO A REGION
///
/// The pass copies an access into one or more conditional branches. An `scf.if` body is a new SSA
/// scope: a view DEFINED where the access used to be is still dominating, but a view whose definition
/// this pass is about to move or erase is not — so the branch gets its own copy of it. ⭐ AND THE
/// CONDITION IS PRECISELY "NON-PAGED": a *paged* view is what the pass is removing and every access
/// under it is being rewritten against a fresh [`create_non_paged_mem_view`] anyway, so cloning it
/// would duplicate the op the pass exists to delete.
///
/// ⭐ `dyn_cast_or_null` IS AN [`Option`] TWICE OVER. `mem_view.getDefiningOp()` is null for a region
/// argument — see [`dialects::defining_op`], whose `None` is exactly that — and the cast is null for
/// any other op. Both fall through to returning the value untouched, which is why one `if let` covers
/// what the C++ needs two null states for.
///
/// ⛔ AND THE CLONE BINDS ITS OWN VALUE. `builder.clone` gives the copy fresh results, which is what
/// `new_mem_view->getResult(0)` reads; see [`dialects::clone_with_fresh_results`], where minting is
/// not optional.
#[must_use]
pub fn clone_mem_view_if_non_paged(
    vals: &mut Values,
    defining_op: Option<&DfirOp>,
    mem_view: Val,
) -> ClonedMemView {
    if let Some(non_paged_mem_view @ DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView { .. })) =
        defining_op
    {
        let new_mem_view = dialects::clone_with_fresh_results(non_paged_mem_view, vals);
        // `new_mem_view->getResult(0)` — the view a `dataflow.get_logical_memory_view` binds is its
        // only result.
        let value = dialects::results(&new_mem_view)
            .first()
            .copied()
            .unwrap_or(mem_view);
        return ClonedMemView {
            ops: vec![new_mem_view],
            value,
        };
    }
    ClonedMemView {
        ops: Vec::new(),
        value: mem_view,
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 126/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e126_getUseChain
///
/// **126/384** `TPMVVectorLoad::getUseChain` —
/// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:673` (3L).
///
/// ```cpp
/// SmallVector<Operation *> TPMVVectorLoad::getUseChain(Operation *mem_op) {
///   auto load_op = cast<agen::VectorLoadOp>(mem_op);
///   return load_op.getUseChain();
/// }
/// ```
///
/// # ⭐⭐ THE OVERRIDE IS THE CAST AND A FORWARD, AND THE SUBSTANCE IS THE DIALECT'S
///
/// `agen::VectorLoadOp::getUseChain` (`Agen.cpp:115-136`) is where the walk lives, and it is
/// [`vector_load_use_chain`] — ported with entry 129, which needs the same chain to tear down. This
/// is `TPMVBase::use_chain`'s override (entry 133), so it answers in the same [`UseChain`] the base
/// does: the base says [`UseChain::None`], and a vector load says [`UseChain::ConsumerWard`].
///
/// # ⛔⛔ THE CHAIN IS THE LOAD *AND* EVERYTHING DOWNSTREAM OF IT, ENDING AT AN OP WITH NO RESULT
///
/// This is what makes the pass's clone correct: an `agen.vector_load` on its own is dead — its value
/// has to reach a `dataflow.send` or an `agen.vector_store` — so copying the load into a conditional
/// branch without copying the ops that consume it produces exactly the *"Dangling non-compute op has
/// no use"* refusal this crate has already been taught by. `use_chain[0]` is the load itself, which is
/// why entry 127 skips the first element.
///
/// ⭐ AND IT IS A CHAIN, NOT A CONE: one result, one use, all the way down. That is a real property of
/// what the scheduler emits — a vector value is consumed once — and the reference asserts it at every
/// step rather than handling a fork. A fork is therefore not a truncated chain but NO chain
/// ([`UseChain::None`], and see [`vector_load_use_chain`] for why that is the contract's own answer
/// rather than a weakening of the assert).
#[must_use]
pub fn get_use_chain<'s>(mem_op: VectorLoadOp<'s>, scope: &'s [DfirOp]) -> UseChain<'s> {
    // `return load_op.getUseChain();`
    vector_load_use_chain(mem_op, scope)
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 127/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// `agen::VectorLoadOp::cloneUseChainToNewOp` — the engine entries 127 and 128 share
/// (`Agen.cpp:138-159`).
///
/// ```cpp
/// void VectorLoadOp::cloneUseChainToNewOp(OpBuilder& builder, Operation* new_op) {
///   assert(isa<VectorLoadOp>(new_op));
///   auto use_chain = getUseChain();
///   Value prev_val = nullptr, prev_cloned_val = new_op->getResult(0);
///   for (int i = 0, last_op_idx = use_chain.size() - 1; i <= last_op_idx; ++i) {
///     // Clone all the ops except the first one. That one is covered by mem_op.
///     if (prev_val) {
///       auto cloned_op = builder.clone(*use_chain[i]);
///       cloned_op->replaceUsesOfWith(prev_val, prev_cloned_val);
///       builder.setInsertionPointAfter(cloned_op);
///       if (i != last_op_idx) {
///         prev_val = use_chain[i]->getResult(0);
///         prev_cloned_val = cloned_op->getResult(0);
///       }
///     } else {
///       prev_val = use_chain[i]->getResult(0);
///     }
///   }
/// }
/// ```
///
/// ⛔⛔ EVERY CLONE READS THE PREVIOUS **CLONE**, WHICH IS THE ONE LINE THAT MATTERS.
/// `cloned_op->replaceUsesOfWith(prev_val, prev_cloned_val)` is what stitches the copy into a chain
/// of its own; without it every clone reads the ORIGINAL chain's values, the copies compute from the
/// old load, and the guarded branch quietly reads the page the pass was rewriting away from.
///
/// ⭐ `prev_val` DOUBLES AS THE "FIRST OP" FLAG. It is null only on the first iteration, so the
/// `if (prev_val)` arm is "every op but the load"; the `else` arm records the load's result as the
/// value the next clone must stop reading. The `if (i != last_op_idx)` guard is bookkeeping: the last
/// op has no result to carry forward.
///
/// ⭐ AND THE INSERTION POINT WALKS FORWARD (`setInsertionPointAfter`), so the clones come out in
/// chain order. Here that is the order of the returned list — a `Vec` cannot hold them any other way.
fn clone_use_chain_to_new_op(
    vals: &mut Values,
    use_chain: &UseChain<'_>,
    new_result: Val,
) -> Vec<DfirOp> {
    // ⛔ NOTHING TO CLONE UNLESS THE CHAIN RUNS CONSUMER-WARD FROM THE LOAD. `getUseChain`'s own
    // "Empty if there isn't a use chain" (`TransformPagedMemViewImpl.hpp:323-324`) is the whole answer
    // for a load with no linear chain, and a producer-ward chain belongs to a STORE
    // (`VectorStoreOp::getUseChain`) — this is the load's method, so it is not one it can be handed.
    let UseChain::ConsumerWard(use_chain) = use_chain else {
        return Vec::new();
    };
    let mut cloned: Vec<DfirOp> = Vec::new();
    let mut prev: Option<(Val, Val)> = None;
    for (i, op) in use_chain.iter().enumerate() {
        match prev {
            // "Clone all the ops except the first one. That one is covered by mem_op."
            Some((prev_val, prev_cloned_val)) => {
                let mut cloned_op = dialects::clone_with_fresh_results(op, vals);
                dialects::replace_uses_of_with(&mut cloned_op, prev_val, prev_cloned_val);
                // `if (i != last_op_idx)` — the last op has no result to carry forward, and the
                // walk asserts it has none at all.
                //
                // ⭐ AND THE PAIR MOVES ONLY WHEN BOTH HALVES EXIST, so a mid-chain op with no result
                // leaves `prev` alone rather than clearing it. The reference reads `getResult(0)`
                // there and would keep its old `prev_val`; clearing it would make the next clone
                // take the first-iteration path and skip its re-pointing.
                if i != use_chain.len() - 1
                    && let Some(pair) = dialects::results(op)
                        .first()
                        .copied()
                        .zip(dialects::results(&cloned_op).first().copied())
                {
                    prev = Some(pair);
                }
                cloned.push(cloned_op);
            }
            // The load itself: `prev_val = use_chain[i]->getResult(0)`, and
            // `prev_cloned_val` is already the new load's result.
            None => {
                prev = dialects::results(op)
                    .first()
                    .copied()
                    .map(|first| (first, new_result));
            }
        }
    }
    cloned
}

/// Replaces: e127_cloneUseChain
///
/// **127/384** `TPMVVectorLoad::cloneUseChain` —
/// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:678` (3L).
///
/// ```cpp
/// void TPMVVectorLoad::cloneUseChain(OpBuilder &builder, Operation *mem_op,
///                                    Operation *new_mem_op) {
///   auto load_op = cast<agen::VectorLoadOp>(mem_op);
///   load_op.cloneUseChainToNewOp(builder, new_mem_op);
/// }
/// ```
///
/// ⭐ THE VIRTUAL SEAM: `TPMVBase` calls `cloneUseChain` on whichever manager it holds, and the
/// vector-load one forwards to the op's own method. The forwarding is the function; the substance is
/// [`clone_use_chain_to_new_op`], which entry 128 also calls.
///
/// ⛔ `assert(isa<VectorLoadOp>(new_op))` IS THE PARAMETER TYPE. `new_mem_op` arrives as a
/// [`VectorLoadOp`], so the assertion inside the callee is discharged by whoever narrowed it — and
/// `mem_op`'s own `cast<agen::VectorLoadOp>` is the same type, which is why neither is repeated here.
#[must_use]
pub fn clone_use_chain<'s>(
    vals: &mut Values,
    mem_op: VectorLoadOp<'s>,
    new_mem_op: VectorLoadOp<'_>,
    scope: &'s [DfirOp],
) -> Vec<DfirOp> {
    clone_use_chain_to_new_op(vals, &get_use_chain(mem_op, scope), new_mem_op.result)
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 128/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// THE ACCESS THIS PASS PUT IN A BRANCH — the new load, its cloned chain, and the value it binds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewMemOp {
    /// The new `agen.vector_load` followed by its cloned use chain, in emission order.
    ///
    /// ⭐ THE ORDER IS `builder.setInsertionPointAfter(new_load_op)` MADE STRUCTURAL: the chain
    /// follows the load because a `Vec` has no other way to say it.
    pub ops: Vec<DfirOp>,
    /// The new load's result — what the reference returns as `Operation *`, in the form callers use
    /// (`new_mem_ops` is walked for its result and for erasure, entry 129).
    pub result: Val,
}

/// Replaces: e128_createNewMemOp
///
/// **128/384** `TPMVVectorLoad::createNewMemOp` —
/// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:684` (9L), whose
/// first line is `agen::VectorLoadOp::cloneWithNewAccessInfo` (`Agen.cpp:161-169`).
///
/// ```cpp
/// Operation *TPMVVectorLoad::createNewMemOp(OpBuilder &builder, Operation *mem_op,
///                                           TPMVInfo &info, Value &mem_view,
///                                           AffineMap &subscripts_map,
///                                           SmallVectorImpl<Value> &indices) {
///   auto load_op = cast<agen::VectorLoadOp>(mem_op);
///   auto new_load_op = load_op.cloneWithNewAccessInfo(builder, mem_view,
///                                                     subscripts_map, indices);
///
///   builder.setInsertionPointAfter(new_load_op);
///   load_op.cloneUseChainToNewOp(builder, new_load_op);
///
///   return new_load_op;
/// }
///
/// // Agen.cpp:161-169
/// agen::VectorLoadOp VectorLoadOp::cloneWithNewAccessInfo(
///     OpBuilder& builder, const Value mem_view, const AffineMap& subscripts_map,
///     ValueRange indices) {
///   return VectorLoadOp::create(
///       builder, getLoc(), getResult().getType(), mem_view,
///       getDbgNameAttr() ? getDbgNameAttr() : builder.getStringAttr(""),
///       subscripts_map, indices, getLoadSet(), getLoadOrder(),
///       getMulticastInfo());
/// }
/// ```
///
/// # ⛔⛔ WHAT CHANGES IS THE ACCESS; WHAT THE LOAD *IS* DOES NOT
///
/// Three things are replaced — the view, the subscript map and the indices — and everything else is
/// carried over from the original load: its result type, its `load_set`, its `load_order` and its
/// multicast info. That is the point of the pass: the same vector, read out of the same shape of
/// memory, at a page-relative address.
///
/// ⭐ AND THE CHAIN COMES WITH IT, in the same call. A new load without its consumers is the dangling
/// op the emitted program is rejected for; see [`get_use_chain`].
///
/// ⚠️ THE `subscripts_map` AND `indices` PARAMETERS ARE ONE LIST HERE.
/// [`agen::Op::VectorLoad`] carries `indices: Vec<Index>` and an
/// [`Index::Strided`] holds the strided sum inline, so the reference's
/// `affine_map` attribute plus `map_indices` operands are the same information written in one place —
/// see that variant, which records why this island writes the sum inline and what `dbo-opt` says
/// about the alternative. Entry 259 produces the page-relative form.
///
/// ⛔ AND THE CLONE SUBSTITUTES AN **EMPTY** `dbgName` FOR AN ABSENT ONE (`Agen.cpp:166`), which is
/// not the same as absent: a later `getDbgNameAttr()` on the clone is non-null, so
/// `getNewDbgNameFromOp` names a loop after it rather than leaving it unnamed. `load_set`/`load_order`
/// are DERIVED at print time from the view's shape and the vector's length (`agen.rs`) rather than
/// stored, so "keep the original's" is automatic for as long as the view's shape is inherited, which
/// [`create_non_paged_mem_view`] guarantees. `getMulticastInfo()` has no island field either.
///
/// ⭐ `TPMVInfo &info` IS UNUSED IN THE BODY. It is in the signature because the composite manager's
/// override needs it (`TransformPagedMemViewImpl.hpp:519`, entry 138), so nothing is dropped by its
/// absence here.
#[must_use]
pub fn create_new_mem_op<'s>(
    vals: &mut Values,
    mem_op: VectorLoadOp<'s>,
    mem_view: Val,
    view_ty: MemRef,
    indices: Vec<Index>,
    scope: &'s [DfirOp],
) -> NewMemOp {
    let result = vals.mint();
    let new_load_op = clone_load_with_new_access_info(mem_op, result, mem_view, view_ty, indices);

    // `builder.setInsertionPointAfter(new_load_op)`, then the chain — hence the order of the list.
    let mut ops = vec![new_load_op];
    ops.extend(clone_use_chain_to_new_op(
        vals,
        &get_use_chain(mem_op, scope),
        result,
    ));
    NewMemOp { ops, result }
}

/// `agen::VectorLoadOp::cloneWithNewAccessInfo` (`Agen.cpp:161-169`) ALONE — the load twin of
/// [`clone_store_with_new_access_info`], without the use chain [`create_new_mem_op`] follows it with.
///
/// ⛔ ITS CALLERS DISAGREE ABOUT THE CHAIN AND BOTH ARE RIGHT: entry 128 clones the consumers because
/// it builds one load PER PAGE, while entry 352 re-points them because it builds exactly one.
pub(super) fn clone_load_with_new_access_info(
    mem_op: VectorLoadOp<'_>,
    result: Val,
    mem_view: Val,
    view_ty: MemRef,
    indices: Vec<Index>,
) -> DfirOp {
    DfirOp::Agen(agen::Op::VectorLoad {
        result,
        view: mem_view,
        indices,
        view_ty,
        // `getResult().getType()` — the vector the original load bound.
        ty: mem_op.ty,
        multicast_info: None,
        // `getDbgNameAttr() ? getDbgNameAttr() : builder.getStringAttr("")` (`Agen.cpp:166`).
        dbg_name: Some(mem_op.dbg_name.unwrap_or_default().to_owned()),
    })
}

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
    /// `getMemRef()` — the view read, which entry 202 casts to a [`PagedMemView`].
    pub view: Val,
    /// `getAffineMapAttr()` AND `getMapIndices()` TOGETHER — the access, as this island writes it
    /// inline. [`super::vc_vector_operands::access_map`] is the split into those two halves.
    pub indices: &'a [Index],
    /// `getResult().getType()` — the type a rebuilt load inherits.
    ///
    /// ⭐ HERE BECAUSE `cloneWithNewAccessInfo` READS IT (`Agen.cpp:161-169`, entry 128): the new
    /// load keeps the original's result type and replaces only the access. On a typed
    /// `VectorLoadOp` that read is infallible, which is the same reason `result` is here.
    pub ty: Vector,
    /// `getDbgNameAttr()` — read by the same clone, which is why it is here.
    pub dbg_name: Option<&'a str>,
}

impl<'a> VectorLoadOp<'a> {
    /// `dyn_cast<agen::VectorLoadOp>` — `None` for anything else.
    #[must_use]
    pub fn of(op: &'a DfirOp) -> Option<VectorLoadOp<'a>> {
        match op {
            DfirOp::Agen(agen::Op::VectorLoad {
                result,
                view,
                indices,
                ty,
                dbg_name,
                ..
            }) => Some(VectorLoadOp {
                op,
                result: *result,
                view: *view,
                indices,
                ty: *ty,
                dbg_name: dbg_name.as_deref(),
            }),
            DfirOp::Agen(
                agen::Op::VectorStore { .. }
                | agen::Op::SymbolicVectorLoad { .. }
                | agen::Op::SymbolicVectorStore { .. }
                | agen::Op::IndirectVectorLoad { .. }
                | agen::Op::IndirectVectorStore { .. }
                | agen::Op::CompositeLoad(_)
                | agen::Op::CompositeStore(_)
                | agen::Op::CompositeLoadAndStore(_)
                | agen::Op::CompositeIndirectLoad(_)
                | agen::Op::CompositeIndirectStore(_)
                | agen::Op::CompositeIndirectLoadAndStore(_)
                | agen::Op::CompositeMemoryInterleave { .. }
                | agen::Op::SetTransferMaskState { .. }
                | agen::Op::Yield,
            )
            | DfirOp::Arith(_)
            | DfirOp::Scf(_)
            | DfirOp::Affine(_)
            | DfirOp::Dataflow(_)
            // ⭐ `dyn_cast<agen::VectorLoadOp>` IS NULL ON AN UPSTREAM `vector.load` — a different
            // dialect's op, and the whole `TPMVBase` hierarchy is written against the `agen` ones.
            | DfirOp::Vector(_)
            | DfirOp::VectorChain(_)
            // ⭐ AND `uniform`, which loads nothing.
            | DfirOp::Uniform(_)
            | DfirOp::Symbol(_) => None,
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
/// `Agen.cpp:177-178`) while `VectorStoreOp::eraseOpAndUseChain` walks it FORWARDS
/// (`for (auto &o : use_chain)`, `:258`) — two loops that look contradictory and are the same rule:
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
    /// ⚠️ NO SCHEDULED UNIT IN THIS FILE PRODUCES ONE: `TPMVVectorStore::getUseChain`
    /// (`TransformPagedMemViewImpl.cpp:721`) is not among the 384 and not among the 106 exclusions —
    /// see the note on [`erase_vector_load_and_use_chain`]. [`vector_store_use_chain`] is the DIALECT
    /// method it would have wrapped, added unanchored for entry 307.
    ProducerWard(Vec<&'a DfirOp>),
}

impl<'a> UseChain<'a> {
    /// THE CHAIN IN TEARDOWN ORDER — every consumer ahead of what it reads.
    ///
    /// This is what both `eraseOpAndUseChain` loops compute, each in the way its own chain's
    /// direction demands (`Agen.cpp:177-178` reversed, `:258` as-is).
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
/// (`TransformPagedMemViewImpl.hpp:323-324`) rather than an abort. The trailing
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
/// remove them: consumer-first (`Agen.cpp:177-178`). Entry 038 already established the shape —
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
    /// `getMemRef()` — the view written.
    pub view: Val,
    /// `getAffineMapAttr()` AND `getMapIndices()` TOGETHER, as [`VectorLoadOp::indices`] holds them.
    pub indices: &'a [Index],
    /// The view's type, which `cloneWithNewAccessInfo` carries over unchanged.
    pub view_ty: &'a MemRef,
    /// The vector stored, likewise carried over.
    pub ty: Vector,
    /// `getDbgNameAttr()` — read by that same clone (`Agen.cpp:244`).
    pub dbg_name: Option<&'a str>,
}

impl<'a> VectorStoreOp<'a> {
    /// `dyn_cast<agen::VectorStoreOp>` — `None` for anything else.
    #[must_use]
    pub fn of(op: &'a DfirOp) -> Option<VectorStoreOp<'a>> {
        match op {
            DfirOp::Agen(agen::Op::VectorStore {
                value,
                view,
                indices,
                dbg_name,
                view_ty,
                ty,
            }) => Some(VectorStoreOp {
                op,
                value: *value,
                view: *view,
                indices,
                view_ty,
                ty: *ty,
                dbg_name: dbg_name.as_deref(),
            }),
            DfirOp::Agen(
                agen::Op::VectorLoad { .. }
                | agen::Op::SymbolicVectorLoad { .. }
                | agen::Op::SymbolicVectorStore { .. }
                | agen::Op::IndirectVectorLoad { .. }
                | agen::Op::IndirectVectorStore { .. }
                | agen::Op::CompositeLoad(_)
                | agen::Op::CompositeStore(_)
                | agen::Op::CompositeLoadAndStore(_)
                | agen::Op::CompositeIndirectLoad(_)
                | agen::Op::CompositeIndirectStore(_)
                | agen::Op::CompositeIndirectLoadAndStore(_)
                | agen::Op::CompositeMemoryInterleave { .. }
                | agen::Op::SetTransferMaskState { .. }
                | agen::Op::Yield,
            )
            | DfirOp::Arith(_)
            | DfirOp::Scf(_)
            | DfirOp::Affine(_)
            | DfirOp::Dataflow(_)
            // ⭐ `dyn_cast<agen::VectorLoadOp>` IS NULL ON AN UPSTREAM `vector.load` — a different
            // dialect's op, and the whole `TPMVBase` hierarchy is written against the `agen` ones.
            | DfirOp::Vector(_)
            | DfirOp::VectorChain(_)
            // ⭐ AND `uniform`, which stores nothing.
            | DfirOp::Uniform(_)
            | DfirOp::Symbol(_) => None,
        }
    }
}

/// `agen::VectorStoreOp::getUseChain` (`Agen.cpp:207-222`) — the DIALECT method, and the
/// producer-ward twin of [`vector_load_use_chain`]:
///
/// ```cpp
/// SmallVector<Operation*> VectorStoreOp::getUseChain() {
///   auto& op = *this;
///   SmallVector<Operation*> use_chain;
///   use_chain.push_back(op);
///   //   receive + store / constant_bitstream + shuffle + store / vector_load + store
///   auto input_op = getValueToStore().getDefiningOp();
///   use_chain.push_back(input_op);
///   if (auto shuffle_op = dyn_cast<vectorchain::ShuffleOp>(input_op))
///     use_chain.push_back(shuffle_op.getInput().getDefiningOp());
///   return use_chain;
/// }
/// ```
///
/// ⛔ NO ASSERTS AND NO WALK: the store's chain is a fixed two or three links read off the operand,
/// where the load's is a loop over uses. That is also why it comes out store-first — the order
/// [`UseChain::ProducerWard`] records and `eraseOpAndUseChain` consumes as-is.
///
/// ⛔ AND A VALUE WITH NO DEFINING OP MAKES IT [`UseChain::None`]. The reference pushes the null
/// `input_op` and every later loop dereferences it; a store fed by a block argument is that, and the
/// documented "no use chain" answer is what the load's side already returns for its own dead ends. A
/// shuffle whose own input has no producer is the same null one link further out, and it stops the
/// chain at the shuffle rather than extending it with nothing.
pub(super) fn vector_store_use_chain<'a>(
    store: VectorStoreOp<'a>,
    scope: &'a [DfirOp],
) -> UseChain<'a> {
    let Some(input_op) = defining_op(store.value, scope) else {
        return UseChain::None;
    };
    let mut chain = vec![store.op, input_op];
    // `if (auto shuffle_op = dyn_cast<vectorchain::ShuffleOp>(input_op))` — the widened bitstream's
    // producer is the third link, and only a shuffle has one.
    if let DfirOp::VectorChain(vectorchain::Op::Shuffle { input, .. }) = input_op
        && let Some(shuffle_input) = defining_op(*input, scope)
    {
        chain.push(shuffle_input);
    }
    UseChain::ProducerWard(chain)
}

/// `agen::VectorStoreOp::cloneUseChainToNewOp` (`Agen.cpp:224-238`):
///
/// ```cpp
/// void VectorStoreOp::cloneUseChainToNewOp(OpBuilder& builder, Operation* new_op) {
///   auto use_chain = getUseChain();
///   Value prev_val = nullptr, prev_cloned_val = nullptr;
///   for (int i = use_chain.size() - 1; i > 0; --i) {
///     auto cloned_op = builder.clone(*use_chain[i]);
///     if (prev_val && prev_cloned_val)
///       cloned_op->replaceUsesOfWith(prev_val, prev_cloned_val);
///     builder.setInsertionPointAfter(cloned_op);
///     prev_val = use_chain[i]->getResult(0);
///     prev_cloned_val = cloned_op->getResult(0);
///   }
///   new_op->replaceUsesOfWith(prev_val, prev_cloned_val);
/// }
/// ```
///
/// ⛔⛔ THE WALK RUNS **BACKWARDS** AND THE STORE IS RE-POINTED LAST. Cloning from the far producer
/// down to the store's immediate input puts the clones out in dependency order — which is emission
/// order, and therefore the order of the returned list — and only then does the new store stop
/// reading the original's value. The load's version (`cloneUseChainToNewOp`, entries 127/128) starts
/// from the new op's own result instead, because its chain runs the other way.
///
/// ⚠️ IT DEREFERENCES `prev_val` UNCONDITIONALLY (`:237`), so a chain of one — a store whose value has
/// no producer — is a null `replaceUsesOfWith` in the reference. [`vector_store_use_chain`] answers
/// [`UseChain::None`] for exactly that input, and this leaves the new store untouched.
pub(super) fn clone_use_chain_to_new_store(
    vals: &mut Values,
    use_chain: &UseChain<'_>,
    new_store: &mut DfirOp,
) -> Vec<DfirOp> {
    // ⛔ NOTHING TO CLONE UNLESS THE CHAIN RUNS PRODUCER-WARD FROM THE STORE — a consumer-ward chain
    // belongs to a LOAD, and this is the store's method.
    let UseChain::ProducerWard(use_chain) = use_chain else {
        return Vec::new();
    };
    let mut cloned: Vec<DfirOp> = Vec::new();
    let mut prev: Option<(Val, Val)> = None;
    // `for (int i = use_chain.size() - 1; i > 0; --i)` — every link but the store itself.
    for op in use_chain[1..].iter().rev() {
        let mut cloned_op = dialects::clone_with_fresh_results(op, vals);
        if let Some((prev_val, prev_cloned_val)) = prev {
            dialects::replace_uses_of_with(&mut cloned_op, prev_val, prev_cloned_val);
        }
        // ⭐ AND THE PAIR MOVES ONLY WHEN BOTH HALVES EXIST, as the load's does: the reference reads
        // `getResult(0)` on both and would keep its old pair for a producer that binds nothing.
        if let Some(pair) = dialects::results(op)
            .first()
            .copied()
            .zip(dialects::results(&cloned_op).first().copied())
        {
            prev = Some(pair);
        }
        cloned.push(cloned_op);
    }
    // `new_op->replaceUsesOfWith(prev_val, prev_cloned_val)`.
    if let Some((prev_val, prev_cloned_val)) = prev {
        dialects::replace_uses_of_with(new_store, prev_val, prev_cloned_val);
    }
    cloned
}

/// `agen::VectorStoreOp::cloneWithNewAccessInfo` (`Agen.cpp:240-248`) — the store twin of the clone
/// inside [`create_new_mem_op`]: the view, the subscript map and the indices are replaced and
/// everything else is carried over, `store_set` and `store_order` included.
///
/// ⛔ AND IT SUBSTITUTES AN **EMPTY** `dbgName` FOR AN ABSENT ONE (`:244`), which is why the vendor's
/// rebuilt stores print `dbgName = ""` where the input printed no such attribute
/// (`mutable_addr_splitting_one_dim_sen1p5.mlir:75`).
pub(super) fn clone_store_with_new_access_info(
    store: VectorStoreOp<'_>,
    mem_view: Val,
    indices: Vec<Index>,
) -> DfirOp {
    DfirOp::Agen(agen::Op::VectorStore {
        // `getValueToStore()` — the ORIGINAL producer's result, which is what makes
        // [`clone_use_chain_to_new_store`]'s final re-point necessary.
        value: store.value,
        view: mem_view,
        indices,
        dbg_name: Some(store.dbg_name.unwrap_or_default().to_owned()),
        view_ty: store.view_ty.clone(),
        ty: store.ty,
    })
}

/// `agen::VectorStoreOp::eraseOpAndUseChain` (`Agen.cpp:250-259`) — the delete list, and it walks the
/// chain **FORWARDS** (`for (auto& o : use_chain) o->erase();`) where the load's reverses it. Both
/// mean the same thing: consumer before producer, which is the only order that erases nothing still
/// in use. [`UseChain::consumer_first`] is where that agreement lives.
///
/// ⭐ AND ITS `use_chain.empty()` BRANCH IS LIVE HERE WHERE IT IS DEAD IN C++, exactly as the load's
/// is (see [`erase_vector_load_and_use_chain`]): `getUseChain` always pushes the store itself, so the
/// reference can only reach that line by never reaching it, and a store with no producer answers
/// [`UseChain::None`] here instead of pushing a null.
pub(super) fn erase_vector_store_and_use_chain<'a>(
    store: VectorStoreOp<'a>,
    scope: &'a [DfirOp],
) -> Vec<&'a DfirOp> {
    let to_be_erased = vector_store_use_chain(store, scope).consumer_first();
    if to_be_erased.is_empty() {
        // `if (use_chain.empty()) { auto& op = *this; op->erase(); }`.
        return vec![store.op];
    }
    to_be_erased
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
            TimeBound::Steps(steps) => u32::try_from(steps)
                .ok()
                .and_then(NonZeroU32::new)
                .map(TimeSteps),
            // `kCoalesced = -2` and `kInvalid = -1` — "no special time bound values should exist".
            TimeBound::Coalesced | TimeBound::Variable => None,
        }
    }

    /// `b - 1` — the last step this dimension takes, which is its range's inclusive upper bound.
    #[must_use]
    pub fn last_index(self) -> i64 {
        i64::from(self.0.get() - 1)
    }

    /// `b` ITSELF — an `affine.for`'s exclusive upper bound, which is what
    /// `constructExplicitTimeLoops` passes to `AffineForOp::create(builder, loc, 0, time_bounds[dim])`
    /// (`Dialect/Agen/Utils.cpp:430-431`) and what [`super::tf_mutable_addr_splitting::nest_explicit_time_loops`]
    /// takes. ⛔ THE TWO ARE ONE APART AND BOTH ARE CALLED "the time bound": mixing them up gives a
    /// nest one iteration short or a page range one element wide.
    #[must_use]
    pub fn trip_count(self) -> i64 {
        i64::from(self.0.get())
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

/// WHAT ONE PAGED ACCESS IS BEING DE-PAGED FROM — `TPMVBase::TPMVInfo`
/// (`TransformPagedMemViewImpl.hpp:42-58`), one per memory operand of the op.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TpmvInfo<'p> {
    /// `paged_mem_view_` — the view the access reads.
    ///
    /// ⛔ AN [`Option`] BECAUSE THE REFERENCE'S HANDLE CAN GO NULL: `transform()` erases the view and
    /// entry 200 tests `if (info.paged_mem_view_)` before touching it (`:390`).
    pub paged_mem_view: Option<PagedMemViewHandle<'p>>,
    /// `subscripts_map_` — the access's affine map.
    pub subscripts_map: AffineMap,
    /// `indices_` — its operands, one per dimension of that map.
    pub indices: Vec<Val>,
    /// `indices_ranges_` — each iterator's whole range, as [`calculate_indices_ranges`] computes it.
    pub indices_ranges: Vec<IvRange>,
    /// `conditional_iter_args_` — the values entry 295 computes each subscript into, and entry 199
    /// compares against a page's bounds.
    pub conditional_iter_args: Vec<Val>,
    /// `mem_index_` — which memory operand this describes.
    pub mem_index: MemoryOperandIndex,
}

impl<'p> TpmvInfo<'p> {
    /// BOTH CONSTRUCTORS (`hpp:43-50`) — the two differ only in whether `mem_index` is defaulted, and
    /// `MemoryOperandIndex::kDirSrc` is that default.
    ///
    /// ⚠️ UNANCHORED: the `TPMVInfo` members are among the 106 excluded data-member entries, and
    /// entry 202 is the first thing that needs one.
    #[must_use]
    pub fn new(
        paged_mem_view: PagedMemViewHandle<'p>,
        subscripts_map: AffineMap,
        mem_index: MemoryOperandIndex,
    ) -> TpmvInfo<'p> {
        TpmvInfo {
            paged_mem_view: Some(paged_mem_view),
            subscripts_map,
            indices: Vec::new(),
            indices_ranges: Vec::new(),
            conditional_iter_args: Vec::new(),
            mem_index,
        }
    }
}

/// THE BASE OF THE `TPMV*` HIERARCHY — `TPMVBase`
/// (`dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:28`).
///
/// One object per paged memory operation being de-paged, built by
/// `TransformPagedMemViewManager`, which picks the concrete class from the op:
/// `TPMVBase` → `TPMVVector` → {`TPMVVectorLoad`, `TPMVVectorStore`, `TPMVVectorLoadStore`} and
/// `TPMVBase` → `TPMVComposite` → {`TPMVCompositeLoad`, `TPMVCompositeStore`, …}.
///
/// # ⭐ EVERY MEMBER BUT `context_`, WHICH THIS CRATE HAS NO COUNTERPART FOR
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
/// (`hpp:375-384`). The second group is written by `initialize()`, a pure virtual (`hpp:66`) with SIX
/// `override final`s of which entry 202 is one — `TPMVVectorLoad`'s (`cpp:658`). `TPMVVectorStore`'s
/// (`:706`), `TPMVVectorLoadStore`'s (`:755`) and the three composites' (`:1080`, `:1133`, `:1186`)
/// are in neither the 384 nor the 106 exclusions. `context_` is an `MLIRContext *`, which this crate
/// has no counterpart for at all. ⭐ `tpmv_info_` IS HERE AS OF ENTRY 202, its first writer
/// ([`TpmvVectorLoad::initialize`]) — see [`TpmvInfo`].
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
    /// `tpmv_info_` — one entry per memory operand, written by `initialize()`: entry 202 is
    /// `TPMVVectorLoad`'s (`cpp:658`), and `TPMVCompositeLoadStore`'s (`:1186`) is the one that fills
    /// TWO (`:1193`, `:1200`) — it is unscheduled, as is every `initialize()` but 202.
    /// ⛔ NOT entry 326: that is `initialize_time` (`:1095`), which writes `access_details_`,
    /// `time_order_`, `time_set_` and `tpmv_comp_info_` and never touches this vector.
    pub tpmv_info: Vec<TpmvInfo<'p>>,
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
            tpmv_info: Vec::new(),
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
/// (`dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:387-392`).
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
/// ⚠️ `TPMVVectorLoad`, `TPMVVectorStore` and `TPMVVectorLoadStore` compose over this in turn —
/// and entry 137 (`hpp:397`) is only the FIRST one's constructor, cited at its mem-initialiser and so
/// named after the base. `transform()` is NOT virtual (`hpp:72`), so all three do share entry 356;
/// `initialize()` is one `override final` each and only `TPMVVectorLoad`'s is entry 202.
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

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 137/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// A PAGED VIEW READ BY ONE `agen.vector_load` — `TPMVVectorLoad`
/// (`dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:394-410`).
///
/// # ⛔ THE SIX LEAVES DIFFER ONLY IN THEIR OVERRIDES, AND THAT IS WHY THEY ARE SIX TYPES
///
/// Every one of the six leaf constructors is a bare delegation with an empty body, so a single Rust
/// struct would compile — and would make entry 139 ([`super::tf_transform_paged_mem_view_manager`]),
/// whose *entire* content is choosing WHICH leaf to build, a function that returns the same thing six
/// times. `TPMVVectorLoad::createNewMemOp` builds an `agen.vector_load` and
/// `TPMVVectorStore::createNewMemOp` an `agen.vector_store` (`hpp:400-409`, `:418-427`), and
/// `TPMVVectorLoad::getUseChain` answers a chain where [`TpmvBase::use_chain`] answers none: the type
/// IS the dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TpmvVectorLoad<'p> {
    /// The `TPMVVector` subobject.
    pub vector: TpmvVector<'p>,
}

impl<'p> TpmvVectorLoad<'p> {
    /// Replaces: e137_TPMVVector
    ///
    /// **137/384** `TPMVVectorLoad::TPMVVectorLoad` —
    /// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:397` (0L).
    ///
    /// ```cpp
    /// TPMVVectorLoad(Operation *mem_op, SenComponents comp)
    ///     : TPMVVector(mem_op, comp) {}
    /// ```
    ///
    /// # ⛔ THE UNIT IS THE **`TPMVVectorLoad`** CONSTRUCTOR, DESPITE ITS NAME
    ///
    /// `hpp:397` is the mem-initializer line `: TPMVVector(mem_op, comp) {}`, and the extractor names
    /// an entry after the token it found at the cited line — so `e137_TPMVVector` is the DERIVED
    /// constructor delegating to the base, exactly as `e136_TPMVBase` (`hpp:389`) is
    /// [`TpmvVector::new`], whose anchor records the same habit. Entry 138 (`e138_TPMVComposite`,
    /// `hpp:519`) is this pattern once more on the other branch.
    ///
    /// ⭐ ITS WHOLE BODY IS THE DELEGATION: both arguments forwarded, every other member left at its
    /// declared default. The base chain is a CALL and not a literal — `TPMVVector(mem_op, comp)` is
    /// entry 136, and `mem_ops_ = {mem_op}` happens inside [`TpmvBase::new`] two levels down.
    ///
    /// ⭐ ONLY THREE OF THE NINE TPMV CONSTRUCTORS GOT ENTRIES, because the extractor deduplicates by
    /// text: `: TPMVVector(mem_op, comp) {}` appears three times (`hpp:397`, `:415`, `:433`) and
    /// `: TPMVComposite(mem_op, comp) {}` three more (`:519`, `:534`, `:549`). The five siblings
    /// below are the deduplicated ones — identical delegations, no anchor of their own — and entry
    /// 139 builds all six.
    #[must_use]
    pub fn new(mem_op: &'p DfirOp, comp: DfirUnit) -> TpmvVectorLoad<'p> {
        TpmvVectorLoad {
            vector: TpmvVector::new(mem_op, comp),
        }
    }
}

/// A PAGED VIEW WRITTEN BY ONE `agen.vector_store` — `TPMVVectorStore` (`hpp:412-427`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TpmvVectorStore<'p> {
    /// The `TPMVVector` subobject.
    pub vector: TpmvVector<'p>,
}

impl<'p> TpmvVectorStore<'p> {
    /// `TPMVVectorStore(Operation *mem_op, SenComponents comp) : TPMVVector(mem_op, comp) {}`
    /// (`TransformPagedMemViewImpl.hpp:414-415`).
    ///
    /// ⭐ NO ANCHOR: the extractor deduplicated this against entry 137, whose mem-initializer is the
    /// same text one class up. See [`TpmvVectorLoad::new`].
    #[must_use]
    pub fn new(mem_op: &'p DfirOp, comp: DfirUnit) -> TpmvVectorStore<'p> {
        TpmvVectorStore {
            vector: TpmvVector::new(mem_op, comp),
        }
    }
}

/// A PAGED VIEW A LOAD READS AND A STORE WRITES BACK — `TPMVVectorLoadStore` (`hpp:429-445`).
///
/// The two halves of one move: `%data = agen.vector_load %src_mem_view[..]` followed by
/// `agen.vector_store %data, %dst_mem_view[..]`
/// (`dcc/test/Transform/TransformPagedMemView/paged_mem_view_load_and_store.mlir:1287-1289`). Either
/// view can be the paged one, which is why entry 139 reaches this leaf from both of its first two
/// arms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TpmvVectorLoadStore<'p> {
    /// The `TPMVVector` subobject.
    pub vector: TpmvVector<'p>,
}

impl<'p> TpmvVectorLoadStore<'p> {
    /// `TPMVVectorLoadStore(Operation *mem_op, agen::VectorStoreOp &store_op, SenComponents comp)
    /// : TPMVVector(mem_op, comp) {}` (`TransformPagedMemViewImpl.hpp:431-433`).
    ///
    /// ⭐ NO ANCHOR: deduplicated against entry 137 like [`TpmvVectorStore::new`].
    ///
    /// # ⛔⛔ THE `store_op` ARGUMENT IS ACCEPTED AND DROPPED
    ///
    /// The mem-initializer forwards `mem_op` and `comp` and nothing else, and the body is empty — so
    /// the store its caller went to the trouble of finding is not kept anywhere.
    /// `TPMVVectorLoadStore::initialize` re-derives it: it re-asserts that `mem_ops_` still holds
    /// exactly the load (`DT_CHECK(mem_ops_.size() == 1)`, `Impl.cpp:756`) and then calls
    /// [`store_op_from_load_store_pattern`] (entry 130, `Impl.cpp:843-850`) to walk the load
    /// result's single user and cast it (`Impl.cpp:766`).
    ///
    /// ⭐ AND WHAT THE STORE BECOMES IS A SECOND `TPMVInfo`, NOT A SECOND `mem_ops_` ENTRY:
    /// `tpmv_info_.emplace_back(mem_view_dst, store_op.getAffineMapAttr().getValue(),
    /// MemoryOperandIndex::kDirDst)` (`Impl.cpp:769-770`), followed by the store's `getMapIndices`
    /// (`:771-772`). `mem_ops_` is never appended to anywhere in the file — the one place it grows a
    /// new value is `createIterArgsForConditionals` REPLACING its entries through an `IRMapping`
    /// after a clone (`Impl.cpp:505-508`), which is a remap of the same count.
    ///
    /// Keeping the parameter and naming what happens to it is the faithful port; quietly dropping it
    /// from the signature would hide that entry 130 exists to undo this.
    ///
    /// ⛔ WHICH OP IS `mem_op` DEPENDS ON WHICH ARM CALLED — always the LOAD in the reference
    /// (`TransformPagedMemViewManager.cpp:33`, `:45`), even in the arm whose user search found the
    /// store. See [`super::tf_transform_paged_mem_view_manager::run`].
    #[must_use]
    pub fn new(
        mem_op: &'p DfirOp,
        _store_op: &'p DfirOp,
        comp: DfirUnit,
    ) -> TpmvVectorLoadStore<'p> {
        TpmvVectorLoadStore {
            vector: TpmvVector::new(mem_op, comp),
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 138/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// A PAGED VIEW REACHED BY A **COMPOSITE** TRANSFER — `TPMVComposite` (`hpp:447-514`).
///
/// The `agen.composite_*` family: a transfer that walks AGEN time dimensions, so the page a given
/// time step lands in is itself a function of time. That is what its extra members are for — and its
/// `run()` (`Impl.cpp:855`) is the one entry the winnow lost: the extractor deduplicated `run`
/// against entry 373 (`TPMVVector::run`, `Impl.cpp:647`), so it appears in neither the 384 nor the
/// exclusions.
///
/// ⭐ ALL FIVE MEMBERS ARE DECLARED NOW. `time_order_`, `time_set_`, `access_details_` and
/// `tpmv_comp_info_` (`hpp:509-513`) are *"Set during initialization"* and are what entry 326
/// ([`TpmvCompositeLoad::initialize_time`]) fills; `page_dependent_time_syms_` is entry 310's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TpmvComposite<'p> {
    /// The `TPMVBase` subobject.
    pub base: TpmvBase<'p>,
    /// `time_order_` — the permutation the time dimensions are walked in. `None` is MLIR's
    /// default-constructed null map, as in [`AccessDetailsAffineComposite::time_order`].
    pub time_order: Option<AffineMap>,
    /// `time_set_` — the time iteration domain, REWRITTEN IN PLACE by entry 325's
    /// `updateTimeSetForExplicitDims` once some dimensions become real loops.
    pub time_set: Option<IntegerSet>,
    /// `access_details_` — one composite access detail per memory operand, built by entry 326.
    pub access_details: AccessContainer<AccessDetailsAffineComposite<'p>>,
    /// `page_dependent_time_syms_` — which loop iterators have a bearing on page selection, as
    /// [`gather_page_dependent_dims_for_page`] adds them and [`identify_time_dim_for_explicit_loops`]
    /// reads them. Empty until [`Self::analyze_valid_pages`] runs.
    pub page_dependent_time_syms: BTreeSet<PageSelSym>,
    /// `tpmv_comp_info_` — parallel to `tpmv_info_`, one per memory operand.
    pub tpmv_comp_info: Vec<TpmvCompositeInfo>,
}

/// `TPMVComposite::TPMVCompositeInfo` (`TransformPagedMemViewImpl.hpp:455-462`) — the time address
/// map of one memory operand, plus which access detail it belongs to.
///
/// ⛔ THE C++ HOLDS AN `AccessDetailsAffineComposite&` AND THIS HOLDS ITS KEY. A reference into
/// `access_details_` would alias the container entry 326 keeps writing to and entry 325 reads
/// `access_details_[0]` off; a [`MemoryOperandIndex`] answers the same question through
/// [`AccessContainer::get`] and cannot dangle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TpmvCompositeInfo {
    /// `time_addr_map_` — this operand's address map over the time dimensions.
    pub time_addr_map: AffineMap,
    /// `access_details_` — which slot of the container the reference bound by reference.
    pub access_details: MemoryOperandIndex,
}

impl<'p> TpmvComposite<'p> {
    /// `TPMVComposite(Operation *mem_op, SenComponents comp) : TPMVBase(mem_op, comp) {}`
    /// (`TransformPagedMemViewImpl.hpp:449-450`).
    ///
    /// ⭐ NO ANCHOR: `: TPMVBase(mem_op, comp) {}` is also entry 136's cited text (`hpp:389`), so the
    /// extractor deduplicated this constructor against it. Written here because entry 138 delegates
    /// to it, the same standing [`TpmvBase::new`] itself has.
    #[must_use]
    pub fn new(mem_op: &'p DfirOp, comp: DfirUnit) -> TpmvComposite<'p> {
        TpmvComposite {
            base: TpmvBase::new(mem_op, comp),
            time_order: None,
            time_set: None,
            access_details: AccessContainer::default(),
            page_dependent_time_syms: BTreeSet::new(),
            tpmv_comp_info: Vec::new(),
        }
    }
}

/// A PAGED VIEW READ BY AN `agen.composite_load` — `TPMVCompositeLoad` (`hpp:516-529`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TpmvCompositeLoad<'p> {
    /// The `TPMVComposite` subobject.
    pub composite: TpmvComposite<'p>,
}

impl<'p> TpmvCompositeLoad<'p> {
    /// Replaces: e138_TPMVComposite
    ///
    /// **138/384** `TPMVCompositeLoad::TPMVCompositeLoad` —
    /// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:519` (0L).
    ///
    /// ```cpp
    /// TPMVCompositeLoad(Operation *mem_op, SenComponents comp)
    ///     : TPMVComposite(mem_op, comp) {}
    /// ```
    ///
    /// ⛔ THE UNIT IS THE **`TPMVCompositeLoad`** CONSTRUCTOR, DESPITE ITS NAME — `hpp:519` is the
    /// mem-initializer line. The same reading as entry 137, one branch over; see
    /// [`TpmvVectorLoad::new`].
    ///
    /// ⭐ ITS INPUT IS AN ISLAND OP NOW, AND THE EARLIER NOTE HERE SAID OTHERWISE. Entry 326
    /// `initialize_time` `dyn_cast`s `agen.composite_load`
    /// (`dcc/test/Transform/TransformPagedMemView/paged_mem_view_loads.mlir:331`) and reads six of
    /// its attributes, so it is that unit's *input* and `AGENT-BRIEF.md:87` applied:
    /// [`crate::islands::dataflow_ir::dialects::agen::Op::CompositeLoad`] declares it and
    /// [`super::tf_transform_paged_mem_view_manager::TpmvManager::run`] selects this leaf for real.
    /// `agen.composite_store` is still absent, which is why [`TpmvCompositeStore`] is not.
    ///
    /// ⭐ THE BODY IS THE DELEGATION, as in entry 137: both arguments forwarded, the four members
    /// [`TpmvComposite`] documents left unset.
    #[must_use]
    pub fn new(mem_op: &'p DfirOp, comp: DfirUnit) -> TpmvCompositeLoad<'p> {
        TpmvCompositeLoad {
            composite: TpmvComposite::new(mem_op, comp),
        }
    }
}

/// A PAGED VIEW WRITTEN BY AN `agen.composite_store` — `TPMVCompositeStore` (`hpp:531-544`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TpmvCompositeStore<'p> {
    /// The `TPMVComposite` subobject.
    pub composite: TpmvComposite<'p>,
}

impl<'p> TpmvCompositeStore<'p> {
    /// `TPMVCompositeStore(Operation *mem_op, SenComponents comp) : TPMVComposite(mem_op, comp) {}`
    /// (`TransformPagedMemViewImpl.hpp:533-534`).
    ///
    /// ⭐ NO ANCHOR: deduplicated against entry 138. Its input, `agen.composite_store`
    /// (`paged_mem_view_stores.mlir:361`), is absent from the island for the reason
    /// [`TpmvCompositeLoad::new`] records.
    #[must_use]
    pub fn new(mem_op: &'p DfirOp, comp: DfirUnit) -> TpmvCompositeStore<'p> {
        TpmvCompositeStore {
            composite: TpmvComposite::new(mem_op, comp),
        }
    }
}

/// A PAGED VIEW ONE `agen.composite_load_and_store` BOTH READS AND WRITES — `TPMVCompositeLoadStore`
/// (`hpp:546-562`).
///
/// ⭐ THE ONE COMPOSITE LEAF WHOSE INPUT THE ISLAND HAS: `agen.composite_load_and_store`
/// (`dcc/test/Transform/TransformPagedMemView/paged_mem_view_load_and_store.mlir:1094`) is
/// [`crate::islands::dataflow_ir::dialects::agen::Op::CompositeLoadAndStore`], and it is how a weight
/// leaves the HBM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TpmvCompositeLoadStore<'p> {
    /// The `TPMVComposite` subobject.
    pub composite: TpmvComposite<'p>,
}

impl<'p> TpmvCompositeLoadStore<'p> {
    /// `TPMVCompositeLoadStore(Operation *mem_op, SenComponents comp)
    /// : TPMVComposite(mem_op, comp) {}` (`TransformPagedMemViewImpl.hpp:548-549`).
    ///
    /// ⭐ NO ANCHOR: deduplicated against entry 138.
    #[must_use]
    pub fn new(mem_op: &'p DfirOp, comp: DfirUnit) -> TpmvCompositeLoadStore<'p> {
        TpmvCompositeLoadStore {
            composite: TpmvComposite::new(mem_op, comp),
        }
    }
}

/// WHETHER EVERY VALUE ASKED FOR WAS THERE — the `DT_CHECK_MSG` of [`remove_values_from_indices`].
///
/// ⛔ NOT AN ERROR TYPE. The reference aborts the compiler with *"could not find value to delete in
/// indices vector"*; this names the value it aborted on, and the caller
/// (`createConditionsForHyperRectSubscripts`) can only ever produce [`IndexRemoval::Removed`] because
/// it builds its delete list out of `indices[dim]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexRemoval {
    /// Every requested value was found and erased.
    Removed,
    /// This value was not among the indices — `std::find` returned `end()`.
    ///
    /// ⚠️ THE REQUESTS BEFORE IT ARE ALREADY GONE, which is the state the reference aborts in too.
    NotAnIndex(Val),
}

/// Replaces: e118_removeValuesFromIndices
///
/// **118/384** `TPMVBase::removeValuesFromIndices` —
/// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:36` (7L).
///
/// ```cpp
/// void TPMVBase::removeValuesFromIndices(
///     SmallVectorImpl<Value> &indices,
///     SmallVectorImpl<Value> &indices_to_delete) {
///   for (auto &index : indices_to_delete) {
///     auto it = std::find(indices.begin(), indices.end(), index);
///     DT_CHECK_MSG(it != indices.end(),
///                  "could not find value to delete in indices vector");
///     indices.erase(it);
///   }
/// }
/// ```
///
/// # ⭐⭐ WHY AN INDEX GETS DELETED AT ALL
///
/// `createConditionsForHyperRectSubscripts` (`:288-339`) walks the subscript map's dimensions and asks
/// the page-selection constraints for each one's constant bounds. Where `lb == ub` that dimension
/// takes exactly one value inside the page being selected, so the pass emits ONE equality condition
/// for it ([`create_equality_condition`]) and replaces the dimension in the subscript map with that
/// constant — *"If LB == UB, the loop iterator can be replaced by a constant in the subscripts."*
/// (`:316-317`). The loop iterator is then no longer an operand of the access, and this is what takes
/// it out of the operand list so the shortened list still matches the shortened map.
///
/// # ⛔ THE FIRST OCCURRENCE, NOT THE POSITION — AND HERE THEY COINCIDE
///
/// `std::find` + `erase(it)` removes the FIRST element equal to the value, not `indices[dim]`. The
/// two agree because the indices are the enclosing loops' induction variables, one distinct block
/// argument per dimension. The port keeps the reference's rule rather than the coincidence: a
/// duplicated index would make positional removal drop a different entry.
///
/// # ⛔ AND THE ORDER OF THE SURVIVORS IS PRESERVED
///
/// `Vec::remove`, like `SmallVector::erase`, shifts the tail down. The subscript map is rebuilt against
/// this list by position (`subscripts_map.replaceDimsAndSymbols({new_dim_exprs}, {}, num_dim_vars, 0)`
/// at `:336`), so a `swap_remove` would silently permute the access.
pub fn remove_values_from_indices(
    indices: &mut Vec<Val>,
    indices_to_delete: &[Val],
) -> IndexRemoval {
    for index in indices_to_delete {
        // `std::find(indices.begin(), indices.end(), index)`.
        let Some(at) = indices.iter().position(|held| held == index) else {
            // `DT_CHECK_MSG(it != indices.end(), ..)`.
            return IndexRemoval::NotAnIndex(*index);
        };
        indices.remove(at);
    }
    IndexRemoval::Removed
}

/// Replaces: e119_replaceDimsInMapWithSyms
///
/// **119/384** `TPMVBase::replaceDimsInMapWithSyms` —
/// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:47` (7L).
///
/// ```cpp
/// AffineMap TPMVBase::replaceDimsInMapWithSyms(AffineMap &map) {
///   SmallVector<AffineExpr, 16> sym_exprs;
///   int num_args = 0;
///   for (int dim = 0; dim < map.getNumDims(); ++dim)
///     sym_exprs.emplace_back(getAffineSymbolExpr(num_args++, context_));
///
///   return map.replaceDimsAndSymbols(sym_exprs, {}, 0, num_args);
/// }
/// ```
///
/// # ⭐⭐ `d0 -> s0`, POSITION FOR POSITION, AND THAT IS THE WHOLE FUNCTION
///
/// `num_args` counts up in lockstep with `dim`, so dimension `i` becomes symbol `i`. The result map
/// has NO dimensions and as many symbols as the input had dimensions — `(d0, d1) -> (d0 * 8 + d1)`
/// becomes `()[s0, s1] -> (s0 * 8 + s1)`.
///
/// # ⭐⭐ WHY A CONSTRAINT SYSTEM NEEDS THE SYMBOL FORM
///
/// Both callers say so in a comment: *"Create a copy of the subscripts_map that represents loop
/// iterators as symbols. This is used to form the constraints to determine which pages are valid for
/// mem_ops_."* (`:603-605`; `:1002-1004` is the same comment with `mem_op_` singular). In MLIR's
/// presburger machinery a DIMENSION is a variable the system solves for and a SYMBOL is a parameter
/// it treats as fixed-but-unknown, so a subscript
/// whose loop iterators are dimensions asks "which iterations hit this page" while the same subscript
/// with them as symbols asks "which pages can these iterators reach" — which is the question
/// `analyzeAndConstructValidPages` puts to it.
///
/// # ⛔ THE SYMBOL FORM IS NEVER PRINTED, AND THE INPUT MAP IS NOT REPLACED
///
/// Both callers bind the result to a fresh `subscripts_map_sym` and leave the original alone; the
/// access keeps its dimension form. So `syms` reaching
/// [`crate::islands::dataflow_ir::print::affine_map`] is a constraint-building map that escaped.
///
/// # ⚠️ SYMBOLS IN THE INPUT WOULD BE DROPPED, AND THE REFERENCE DROPS THEM TOO
///
/// `replaceDimsAndSymbols(sym_exprs, {}, 0, num_args)` passes an EMPTY symbol replacement list, which
/// MLIR reads as "no symbols to replace"; a map that had symbols would keep them, unrenumbered, and
/// collide with the new ones. No subscript map in this pipeline has any — every one is built by
/// `AffineMap::get(num_dims, 0, ..)` — so the case does not arise, and this port would carry an input
/// symbol through unchanged exactly as the reference does.
#[must_use]
pub fn replace_dims_in_map_with_syms(map: &AffineMap) -> AffineMap {
    AffineMap {
        // `replaceDimsAndSymbols(.., .., /*numResultDims=*/0, /*numResultSyms=*/num_args)`.
        dims: 0,
        syms: map.dims,
        results: map.results.iter().map(dims_as_syms).collect(),
    }
}

/// ONE EXPRESSION WITH EVERY `dN` REWRITTEN AS `sN` — the substitution `replaceDimsAndSymbols`
/// performs, which is structural and reaches every leaf.
fn dims_as_syms(expr: &AffineExpr) -> AffineExpr {
    match expr {
        // `getAffineSymbolExpr(num_args++, ..)` at the position `dim` counted up to.
        AffineExpr::Dim(dim) => AffineExpr::Sym(*dim),
        // ⛔ A SYMBOL IS LEFT ALONE — the replacement list for symbols is empty. See the note on
        // [`replace_dims_in_map_with_syms`].
        AffineExpr::Sym(sym) => AffineExpr::Sym(*sym),
        AffineExpr::Const(value) => AffineExpr::Const(*value),
        AffineExpr::Add(lhs, rhs) => {
            AffineExpr::Add(Box::new(dims_as_syms(lhs)), Box::new(dims_as_syms(rhs)))
        }
        AffineExpr::Mul(lhs, rhs) => {
            AffineExpr::Mul(Box::new(dims_as_syms(lhs)), Box::new(dims_as_syms(rhs)))
        }
        AffineExpr::Mod(lhs, rhs) => {
            AffineExpr::Mod(Box::new(dims_as_syms(lhs)), Box::new(dims_as_syms(rhs)))
        }
        AffineExpr::FloorDiv(lhs, rhs) => {
            AffineExpr::FloorDiv(Box::new(dims_as_syms(lhs)), Box::new(dims_as_syms(rhs)))
        }
    }
}

/// Replaces: e120_createEqualityCondition
///
/// **120/384** `TPMVBase::createEqualityCondition` —
/// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:253` (7L).
///
/// ```cpp
/// Operation *TPMVBase::createEqualityCondition(OpBuilder &builder, Value &lhs,
///                                              int64_t rhs) {
///   auto rhs_const =
///       mlir::arith::ConstantIndexOp::create(builder, lhs.getLoc(), rhs);
///   auto cond = mlir::arith::CmpIOp::create(
///       builder, lhs.getLoc(), mlir::arith::CmpIPredicate::eq, lhs, rhs_const);
///   return mlir::scf::IfOp::create(builder, lhs.getLoc(), cond.getResult(),
///                                  false);
/// }
/// ```
///
/// # ⭐⭐ WHAT THE CONDITION GUARDS
///
/// A PAGED memory view is one HBM region described as a list of pages, and a transfer reading it
/// cannot name a page with an affine subscript. The pass replaces the access with one copy per
/// candidate page, each inside a nest of conditions that hold exactly when the loop iterators are in
/// that page's range. Where a dimension's page-selection bounds collapse to a point (`lb == ub`) that
/// range is a single value, so one `arith.cmpi eq` decides it — *"If LB == UB, we only need one
/// equality condition."* (`:310`.) `createInequalityCondition` — entry 121, `:263`, not ported here — is
/// the two-sided form for a dimension that spans a sub-range: an `sge` branch with an `sle` branch
/// nested in its `then` region.
///
/// # ⛔ `withElseRegion = false`, AND NO RESULTS
///
/// The trailing `false` is `scf::IfOp::create`'s `withElseRegion`: the `then` region only, and the
/// three-argument form takes no result types, so the branch yields nothing. An `else` arm here would
/// be *"and if this dimension is not on this page"*, which is the NEXT page's copy, emitted by the
/// next turn of the caller's loop — not a second arm of this one. `else_body: Vec::new()` is that,
/// and [`scf::Op::If`]'s printer omits the `else` entirely.
///
/// # ⛔ `lhs.getLoc()` THREE TIMES IS PROVENANCE, NOT BEHAVIOUR
///
/// Every op is given the location of the index it tests, so a diagnostic points at the loop iterator
/// rather than at the pass. This island carries no locations.
///
/// # ⚠️ THE BODY IS EMPTY WHEN IT COMES BACK
///
/// The reference returns an `scf.if` with an empty `then` block, which the caller then builds into
/// ([`set_builder_to_insert_ref`], `:280-286`). Filling it is entry 122's and the caller's business —
/// [`Condition::wrap`] is where the guarded statements go.
///
/// # ⭐⭐ ONE GUARD OF THE SAME [`Condition`] THE TWO-SIDED FORM RETURNS
///
/// A [`Condition`] is a list of [`BoundGuard`]s, outermost first, because
/// [`create_inequality_condition`] leaves the caller's builder inside the nest it built. This form
/// builds ONE guard rather than two, and the difference between the two functions is exactly that
/// plus the predicate — a separate record would be two spellings of one nest, and
/// `createConditionsForHyperRectSubscripts` stores whichever it got in the same `insert_refs` slot
/// (`:313` here, `:331-332` for the two-sided form).
#[must_use]
pub fn create_equality_condition(vals: &mut Values, lhs: Val, rhs: i64) -> Condition {
    // `arith::ConstantIndexOp::create(builder, lhs.getLoc(), rhs)`.
    let rhs_const = vals.mint();
    // `arith::CmpIOp::create(builder, .., CmpIPredicate::eq, lhs, rhs_const)`, and the
    // `scf::IfOp::create(builder, .., cond.getResult(), /*withElseRegion=*/false)` it branches on —
    // emitted by [`Condition::wrap`].
    let cond = vals.mint();
    Condition {
        guards: vec![BoundGuard {
            lhs,
            predicate: CmpIPredicate::Eq,
            bound: rhs,
            constant: rhs_const,
            cond,
        }],
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 197/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// A SUBSCRIPT THAT IS A CONSTANT-BOUNDED `affine.for`'s INDUCTION VARIABLE.
///
/// # ⛔⛔ THE TWO `DT_CHECK_MSG`s AND THE `llvm_unreachable` ARE THIS TYPE
///
/// [`calculate_indices_ranges`] opens with three aborts over `indices[dim]`
/// (`TransformPagedMemViewImpl.cpp:58-78`):
///
/// ```cpp
/// auto block_arg = dyn_cast<BlockArgument>(index);
/// DT_CHECK_MSG(block_arg, "expecting only BlockArguments in indices");
/// auto *loop_op = block_arg.getOwner()->getParentOp();
/// auto affine_for = dyn_cast<affine::AffineForOp>(loop_op);
/// DT_CHECK_MSG(affine_for, "agen memory operations involving loop iterators can only have loop "
///                          "iterators from affine::AffineForOps in the subscripts");
/// ..
/// } else {
///   llvm_unreachable("only loops with constant bounds are supported");
/// }
/// ```
///
/// All three ask the same question — *is this subscript a loop iterator whose bounds are literals* —
/// and none of them is recoverable. A value that answers no has no range to contribute, so the port
/// makes it unrepresentable: [`SubscriptIv::of`] is the one door, and a `&[SubscriptIv]` is a list
/// of subscripts the reference would not have aborted on.
///
/// ⭐ ONE `None` FOR THREE ABORTS, WHICH IS NOT A LOSS. The reference's three messages differ, but
/// what a caller can do about them does not: all three stop the compiler. Keeping them apart would
/// mean an enum whose variants no code reads.
///
/// # ⭐⭐ AND THE BOUNDS COME FROM ENTRY 142, WHICH IS THE SAME TWO QUESTIONS ALREADY ANSWERED
///
/// [`super::tf_utils::get_dataflow_for_loop_info_if_iv`] walks the block arguments to find the op
/// that binds a value, then reads that op's constant bounds — the reference's
/// `getOwner()->getParentOp()` plus `hasConstantLowerBound() && hasConstantUpperBound()`, which is
/// `lb_map.isSingleConstant() && ub_map.isSingleConstant()` spelled the way `AffineForOp` spells it.
/// Reimplementing the descent here would be a second walk over the same tree.
///
/// ⚠️ IT IS STRICTER IN ONE PLACE, AND DELIBERATELY. Entry 142 ends with
/// `affine_for.getInductionVar() == val`, so a loop's CARRIED region argument answers [`None`]; the
/// reference reads `getParentOp()` and would take the enclosing loop's bounds for it. The abort
/// message names *"loop iterators"*, and a carried address is not one — an `iter_arg` in a subscript
/// would give that subscript the loop's trip range, which is a range it does not have.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubscriptIv {
    /// The subscript itself — `indices[dim]`, which is the loop's induction variable.
    pub iv: Val,
    /// `lb_map.getSingleConstantResult()` — the first value the iterator takes.
    pub lo: LoopBound,
    /// `ub_map.getSingleConstantResult()` — ⛔ EXCLUSIVE, as an `affine.for`'s upper bound is. The
    /// `- 1` that turns it into an inclusive range belongs to [`calculate_indices_ranges`], which is
    /// where the reference does it.
    pub hi: LoopBound,
}

impl SubscriptIv {
    /// THE DOOR — a subscript the reference would accept, or [`None`] for one it aborts on.
    #[must_use]
    pub fn of(index: Val, scope: &[DfirOp]) -> Option<SubscriptIv> {
        // `dyn_cast<BlockArgument>(index)`, `block_arg.getOwner()->getParentOp()`, and both
        // `isSingleConstant()` tests — see the note above on which of the three aborts each is.
        let info = get_dataflow_for_loop_info_if_iv(index, scope)?;

        // `dyn_cast<affine::AffineForOp>(loop_op)`. ⛔ AN `scf.for` IS REFUSED HERE EVEN THOUGH
        // ENTRY 142 ACCEPTS ONE: it tests that class FIRST and answers with constant bounds for it,
        // and this function's second `DT_CHECK_MSG` is precisely the one that rules it out.
        if !matches!(info.for_op, DfirOp::Affine(affine::Op::For { .. })) {
            return None;
        }

        Some(SubscriptIv {
            iv: index,
            lo: info.lo,
            hi: info.hi,
        })
    }

    /// EVERY SUBSCRIPT OR NONE — the walk over `indices` with the aborts hoisted out of it.
    ///
    /// ⛔⛔ ALL-OR-NOTHING, BECAUSE THE POSITIONS ARE READ BY NUMBER. `indices_ranges[dim]` is what
    /// entry 198 compares a page's bounds against (`:325-326`) and `indices_ranges[sym_idx]` is what
    /// `addConstraintsForIVRanges` builds its constraint rows from (`:99-104`), so dropping one
    /// subscript would shift every later range onto the wrong iterator — a silently different set of
    /// valid pages. The reference stops the compiler instead, and so does this: a caller with a
    /// subscript it cannot describe gets no list at all.
    #[must_use]
    pub fn all_of(indices: &[Val], scope: &[DfirOp]) -> Option<Vec<SubscriptIv>> {
        indices
            .iter()
            .map(|index| SubscriptIv::of(*index, scope))
            .collect()
    }
}

/// Replaces: e197_calculateIndicesRanges
///
/// **197/384** `TPMVBase::calculateIndicesRanges` —
/// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:56` (25L).
///
/// ```cpp
/// void TPMVBase::calculateIndicesRanges(
///     SmallVectorImpl<Value> &indices, SmallVectorImpl<IVRange> &indices_ranges) {
///   for (int dim = 0; dim < indices.size(); ++dim) {
///     auto index = indices[dim];
///     auto block_arg = dyn_cast<BlockArgument>(index);
///     DT_CHECK_MSG(block_arg, "expecting only BlockArguments in indices");
///
///     auto *loop_op = block_arg.getOwner()->getParentOp();
///     auto affine_for = dyn_cast<affine::AffineForOp>(loop_op);
///     DT_CHECK_MSG(
///         affine_for,
///         "agen memory operations involving loop iterators can only have loop "
///         "iterators from affine::AffineForOps in the subscripts");
///
///     auto lb_map = affine_for.getLowerBoundMap();
///     auto ub_map = affine_for.getUpperBoundMap();
///     int lb, ub;
///     if (lb_map.isSingleConstant() && ub_map.isSingleConstant()) {
///       lb = lb_map.getSingleConstantResult();
///       ub = ub_map.getSingleConstantResult() - 1;
///     } else {
///       llvm_unreachable("only loops with constant bounds are supported");
///     }
///
///     indices_ranges.emplace_back(lb, ub);
///   }
/// }
/// ```
///
/// # ⭐⭐ THE ITERATION SPACE, AS A CLOSED INTERVAL — AND THE `- 1` IS THE WHOLE ARITHMETIC
///
/// `affine.for %arg = 0 to 2` runs over `{0, 1}`, so its range is `[0, 1]`. The reference's `ub - 1`
/// converts MLIR's exclusive upper bound into the inclusive one [`IvRange`] holds, which is the form
/// `addConstraintsForIVRanges` emits (`-<sym> + <ub> >= 0`, `:99-104`) and the form entry 198
/// compares against. Carrying the exclusive bound through instead would put every page's upper
/// constraint one element too wide, and no verifier would object.
///
/// ⭐ AND ITS PURPOSE IS AT THE CALL SITE, in the reference's own comment: *"Ranges of the loop
/// iterators are used to only choose pages within the loop iteration space."* (`:609-610`, `:1008-1009`).
/// A page the iterators cannot reach is not a candidate, so a subscript whose bounds already span its
/// whole loop needs no condition at all — which is exactly the `continue` of entry 198 (`:325-327`).
///
/// # ⛔⛔ IT APPENDS, AND THE POSITION IT APPENDS AT IS LOAD-BEARING
///
/// Both call sites pass the member vector `info.indices_ranges_`, and one of them then extends the
/// SAME vector with the time dimensions (`addTimeDimIndicesRanges`, entry 131, `:1016-1017`) before
/// checking `subscripts_map_time[i].getNumDims() == indices_ranges_.size()` (`:1018-1019`). So the
/// layout is **non-time subscripts first, then time dims**, indexed by symbol number; see
/// [`add_time_dim_indices_ranges`] and [`NonTimeDims::sym_for`], which do that arithmetic. Clearing
/// the vector here, or writing over slot `dim`, would renumber every page-selection constraint.
///
/// ⚠️ `saturating_sub` FOR AN `int` SUBTRACTION THAT CANNOT UNDERFLOW IN PRACTICE. Every loop bound
/// this pipeline emits is a non-negative trip count, so `ub - 1` is ordinary arithmetic; the
/// saturating form is how the crate spells "this does not wrap" without an `assert!`, and it changes
/// nothing for any bound a program contains.
pub fn calculate_indices_ranges(indices: &[SubscriptIv], indices_ranges: &mut Vec<IvRange>) {
    for index in indices {
        // `indices_ranges.emplace_back(lb, ub)`, with `lb = lb_map.getSingleConstantResult()` and
        // `ub = ub_map.getSingleConstantResult() - 1`.
        indices_ranges.push(IvRange {
            lb: index.lo.0,
            ub: index.hi.0.saturating_sub(1),
        });
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 198/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ONE DIMENSION OF A SUBSCRIPT MAP, WITH EVERYTHING THE WALK READS AT THAT POSITION.
///
/// # ⭐⭐ THREE PARALLEL LISTS INDEXED BY ONE COUNTER BECOME ONE RECORD
///
/// `createConditionsForHyperRectSubscripts` walks `dim` from `0` to `subscripts_map.getNumDims()`
/// and reads three things at it — `page_sel_constraints.getConstantBound(.., dim)`, `indices[dim]`
/// and `indices_ranges[dim]` (`:300-326`). One record per dimension is the same information with the
/// three indexings gone, which is the shape [`Page`] already uses in this file for the reference's
/// two parallel page lists.
///
/// # ⛔⛔ AND THE TWO RANGES ARE THE SAME KIND OF THING, WHICH IS WHY THE SKIP TEST IS ONE `==`
///
/// [`Self::selected`] is which values of this iterator can reach the page being selected;
/// [`Self::whole`] is which values it takes at all. The reference compares them field by field —
/// `if (lb_val == indices_ranges[dim].first && ub_val == indices_ranges[dim].second) continue;`
/// (`:325-327`) — and a conjunction of two comparisons is a thing to get half right. One [`IvRange`]
/// equality cannot compare one end and forget the other.
///
/// ⭐ AND `getConstantBound(LB/UB, dim)` REALLY IS A BOUND ON THE ITERATOR, not on a view axis.
/// `page_sel_constraints` is built from the page's `idx_set` with its DIMENSIONS replaced by the
/// symbol-form subscript map's results (`getPageValidity`, `:114-147`, over
/// [`replace_dims_in_map_with_syms`]) and then constrained by the iterator ranges, so it has no
/// dimension variables and one symbol per iterator — position `dim` is loop iterator `dim`. The
/// vendor's own case says so: `%mem_view[%c0, %arg1 * 3, %arg2 * 2 + %c2]` against a page holding
/// `d1 >= 0, -d1 + 1 >= 0` gives `arith.cmpi eq, %arg1, 0`
/// (`dcc/test/Transform/TransformPagedMemView/paged_mem_view_loads.mlir:289-297` and `:45-46`) —
/// `arg1 * 3` inside `[0, 1]` pins `arg1`, not the axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubscriptDim {
    /// `indices[dim]` — the loop iterator this dimension stands for.
    pub index: Val,
    /// `getConstantBound(BoundType::LB, dim)` and `(BoundType::UB, dim)` — the values of the
    /// iterator that land inside the page being selected, INCLUSIVE on both ends.
    pub selected: IvRange,
    /// `indices_ranges[dim]` — the iterator's whole range, as [`calculate_indices_ranges`] computed
    /// it.
    pub whole: IvRange,
}

impl SubscriptDim {
    /// THE DOOR — ⛔ `DT_CHECK_MSG(lb.has_value() && ub.has_value(), "expected constant lower and
    /// upper bounds")` (`:304-305`).
    ///
    /// [`crate::islands::dataflow_ir::ty::IntegerSet::constant_bound`] answers an [`Option`] for the
    /// same reason `getConstantBound` answers a `std::optional`: a dimension the constraints leave
    /// open has no literal bound. Taking both optionals here means a [`SubscriptDim`] cannot hold a
    /// bound that is not constant, so the walk below has no abort left in it.
    #[must_use]
    pub fn of(
        index: Val,
        lb: Option<i64>,
        ub: Option<i64>,
        whole: IvRange,
    ) -> Option<SubscriptDim> {
        Some(SubscriptDim {
            index,
            // ⭐ ONE `DT_CHECK_MSG` OVER BOTH, so a dimension with only one constant end is refused
            // exactly as the reference refuses it.
            selected: IvRange { lb: lb?, ub: ub? },
            whole,
        })
    }
}

/// WHAT `createConditionsForHyperRectSubscripts` REWROTE — its four by-reference outputs, none of
/// them dropped.
///
/// ⭐ THE REFERENCE RETURNS `void` AND WRITES THROUGH FOUR ALIASES: `insert_refs[i]`,
/// `subscripts_map`, `indices`, and the abort inside `removeValuesFromIndices`. A caller that took
/// only the conditions would emit the access with its ORIGINAL subscripts, reading the paged view's
/// coordinates out of a view that no longer has pages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HyperRectConditions {
    /// `insert_refs` on the way out — one nest per access, in the order they came in.
    ///
    /// ⭐ AN EMPTY [`Condition`] IS THE UNTOUCHED ENTRY. Where every dimension was skipped the
    /// reference leaves `insert_refs[i]` holding the memory op itself, and
    /// [`set_builder_to_insert_ref`] then places the statements in that op's own block — which is
    /// what `Condition::wrap` over no guards does (`InsertRef::MemOp => ops`). The two spellings
    /// place the same statements in the same place, so the distinction has nothing left to carry.
    pub insert_refs: Vec<Condition>,
    /// `subscripts_map` on the way out — pinned dimensions substituted away, the rest renumbered.
    pub subscripts_map: AffineMap,
    /// `indices` on the way out — the surviving iterators, in order.
    pub indices: Vec<Val>,
    /// Whether every pinned iterator was found in `indices` — see [`IndexRemoval`], which records
    /// why this can only ever be [`IndexRemoval::Removed`] here.
    pub removal: IndexRemoval,
}

/// Replaces: e198_createConditionsForHyperRectSubscripts
///
/// **198/384** `TPMVBase::createConditionsForHyperRectSubscripts` —
/// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:289` (48L).
///
/// ```cpp
/// void TPMVBase::createConditionsForHyperRectSubscripts(
///     SmallVectorImpl<Operation *> &insert_refs,
///     FlatLinearValueConstraints &page_sel_constraints, AffineMap &subscripts_map,
///     SmallVectorImpl<Value> &indices, SmallVectorImpl<IVRange> &indices_ranges) {
///   int num_dim_vars = 0;
///   SmallVector<AffineExpr, 16> new_dim_exprs;
///   SmallVector<Value> indices_to_delete;
///
///   OpBuilder builder(context_);
///   int num_dims = subscripts_map.getNumDims();
///   for (unsigned dim = 0; dim < num_dims; ++dim) {
///     auto lb = page_sel_constraints.getConstantBound(
///         mlir::presburger::BoundType::LB, dim);
///     auto ub = page_sel_constraints.getConstantBound(
///         mlir::presburger::BoundType::UB, dim);
///     DT_CHECK_MSG(lb.has_value() && ub.has_value(),
///                  "expected constant lower and upper bounds");
///     auto lb_val = (int64_t)lb.value();
///     auto ub_val = (int64_t)ub.value();
///
///     if (lb_val == ub_val) {
///       // If LB == UB, we only need one equality condition.
///       for (int i = 0, e = insert_refs.size(); i < e; ++i) {
///         setBuilderToInsertRef(builder, insert_refs[i]);
///         insert_refs[i] = createEqualityCondition(builder, indices[dim], lb_val);
///       }
///
///       // If LB == UB, the loop iterator can be replaced by a constant in the
///       // subscripts.
///       new_dim_exprs.emplace_back(getAffineConstantExpr(lb_val, context_));
///       indices_to_delete.push_back(indices[dim]);
///     } else {
///       new_dim_exprs.emplace_back(getAffineDimExpr(num_dim_vars++, context_));
///       // If the lower and upper bounds span the whole loop iteration space,
///       // the loop iterator does not aid in identifying a unique page. It can
///       // be skipped.
///       if (lb_val == indices_ranges[dim].first &&
///           ub_val == indices_ranges[dim].second)
///         continue;
///
///       for (int i = 0, e = insert_refs.size(); i < e; ++i) {
///         setBuilderToInsertRef(builder, insert_refs[i]);
///         insert_refs[i] =
///             createInequalityCondition(builder, indices[dim], lb_val, ub_val);
///       }
///     }
///   }
///   subscripts_map = subscripts_map.replaceDimsAndSymbols({new_dim_exprs}, {},
///                                                         num_dim_vars, 0);
///   if (!indices_to_delete.empty())
///     removeValuesFromIndices(indices, indices_to_delete);
/// }
/// ```
///
/// # ⭐⭐ THIS IS HOW ONE PAGE OF A PAGED VIEW GETS ITS OWN COPY OF THE ACCESS
///
/// A `dataflow.get_paged_logical_memory_view` is one HBM region cut into pages with unrelated start
/// addresses, and an affine subscript cannot name a page. `constructValidPage` (entry 309) makes one
/// non-paged view per candidate page and calls this to say *when* the access lands on it: one
/// condition per subscript dimension whose iterator has to be inside a sub-range, nested, guarding
/// the copy. `indices` and `subscripts_map` come back rewritten for that page, because a pinned
/// iterator is a constant there and is no longer an operand at all.
///
/// # ⛔⛔ THREE CASES PER DIMENSION, AND THE MIDDLE ONE STILL COSTS A DIMENSION
///
/// | `selected` vs `whole` | condition | map |
/// |---|---|---|
/// | a single value (`lb == ub`) | one `cmpi eq` | the constant, and the iterator drops out |
/// | a strict sub-range | `cmpi sge` nested over `cmpi sle` | `d<num_dim_vars++>` |
/// | the whole range | ⭐ NONE — `continue` | `d<num_dim_vars++>` |
///
/// ⛔⛔ `num_dim_vars++` HAPPENS **BEFORE** THE `continue` (`:321` then `:325-327`), so a dimension
/// that needs no condition still takes its slot in the new map. Moving the increment after the skip
/// would renumber every later dimension down by one and silently point each surviving subscript at
/// the wrong iterator — an access that reads the right page at the wrong address, with a map that
/// verifies.
///
/// ⭐ AND THE `continue` NEEDS BOTH ENDS TO MATCH. The vendor's second page selects `arg2` over
/// `[2, 3]` out of a whole range of `[0, 3]` — the upper end agrees and the lower does not, and the
/// reference emits BOTH comparisons for it (`cmpi sge, %arg2, 2` and `cmpi sle, %arg2, 3`,
/// `paged_mem_view_loads.mlir:65-69`). See [`SubscriptDim`] for why that is one `==` here.
///
/// # ⛔ THE VALUE NUMBERING IS DIMENSION-MAJOR, ACCESS-MINOR
///
/// The `for (int i = 0, e = insert_refs.size(); i < e; ++i)` loop is INSIDE the dimension loop, and
/// each turn of it mints its own constants and comparisons — a composite load/store pair
/// (`insert_refs = mem_ops_`, `:201`) gets dimension 0's guards for both accesses before dimension
/// 1's guards for either. The printed `%N`s are what a vendored expectation is diffed against, so the
/// order is part of the port.
///
/// # ⭐ AND EACH ACCESS'S NEST GROWS OUTWARD-IN
///
/// `setBuilderToInsertRef(builder, insert_refs[i])` points the builder at what the PREVIOUS dimension
/// created and the new condition is emitted inside it, then overwrites the slot. That is
/// [`Condition::and_then`] — the guards of dimension `k + 1` sit inside those of dimension `k` — and
/// it is why an entry that arrives as [`InsertRef::Conditional`] seeds the nest with the condition it
/// already carries rather than starting empty.
///
/// # ⛔ `replaceDimsAndSymbols(.., {}, num_dim_vars, 0)` DECLARES ZERO SYMBOLS
///
/// Every subscript map on this path is built by `AffineMap::get(num_dims, 0, ..)` and has none, so
/// the new arity is exact. ⚠️ A map that did carry a symbol would keep it, unrenumbered, in a map
/// that says it has zero — the reference's own behaviour, recorded here for the same reason
/// [`replace_dims_in_map_with_syms`] records it.
///
/// # ⚠️ `num_dims` IS `dims.len()`, AND THAT REMOVES A MISMATCH RATHER THAN CHECKING FOR ONE
///
/// The reference reads the loop's extent off the map (`subscripts_map.getNumDims()`) and then indexes
/// two vectors with it; the three agree because `replaceConstOpsInSubscriptsMap` keeps `indices_` and
/// `subscripts_map_` in step (`:600-601`, `:988-989`) and the call site checks the count against the
/// ranges (`:1018-1019`). Taking the per-dimension records as the extent makes that agreement the
/// signature. ⭐ The ranges vector may be LONGER — the composite path appends the time dimensions to
/// it (entry 131) — and the reference never reaches those entries either.
#[must_use]
pub fn create_conditions_for_hyper_rect_subscripts(
    vals: &mut Values,
    insert_refs: &[InsertRef<'_>],
    subscripts_map: &AffineMap,
    dims: &[SubscriptDim],
) -> HyperRectConditions {
    // `int num_dim_vars = 0;` / `SmallVector<AffineExpr, 16> new_dim_exprs;` /
    // `SmallVector<Value> indices_to_delete;`
    let mut num_dim_vars = 0;
    let mut new_dim_exprs = Vec::new();
    let mut indices_to_delete = Vec::new();

    // `SmallVector<Operation *, 16> insert_refs = mem_ops_;` at the call site (`:201`) — every entry
    // is a bare memory op there, and this pass may have wrapped one already on an earlier page.
    let mut nests: Vec<Condition> = insert_refs
        .iter()
        .map(|insert_ref| match insert_ref {
            InsertRef::Conditional(condition) => (*condition).clone(),
            InsertRef::MemOp => Condition::default(),
        })
        .collect();

    // `for (unsigned dim = 0; dim < num_dims; ++dim)`.
    for dim in dims {
        // `if (lb_val == ub_val)` — the two `getConstantBound` calls and their `DT_CHECK_MSG` are
        // [`SubscriptDim::of`].
        if dim.selected.lb == dim.selected.ub {
            // "If LB == UB, we only need one equality condition."
            for nest in &mut nests {
                let created = create_equality_condition(vals, dim.index, dim.selected.lb);
                // `setBuilderToInsertRef(builder, insert_refs[i])` then
                // `insert_refs[i] = createEqualityCondition(..)` — the new guard nests inside what
                // this slot already holds.
                *nest = core::mem::take(nest).and_then(created);
            }

            // "If LB == UB, the loop iterator can be replaced by a constant in the subscripts."
            new_dim_exprs.push(AffineExpr::Const(dim.selected.lb));
            indices_to_delete.push(dim.index);
        } else {
            // `new_dim_exprs.emplace_back(getAffineDimExpr(num_dim_vars++, context_));` — ⛔ BEFORE
            // the skip below, so a dimension that needs no condition keeps its slot.
            new_dim_exprs.push(AffineExpr::dim(num_dim_vars));
            num_dim_vars += 1;

            // "If the lower and upper bounds span the whole loop iteration space, the loop iterator
            // does not aid in identifying a unique page. It can be skipped."
            if dim.selected == dim.whole {
                continue;
            }

            for nest in &mut nests {
                let created =
                    create_inequality_condition(vals, dim.index, dim.selected.lb, dim.selected.ub);
                *nest = core::mem::take(nest).and_then(created);
            }
        }
    }

    // `SmallVector<Value> indices = info.indices_;` at the call site (`:197`), which is the one value
    // per subscript dimension that [`SubscriptDim::index`] holds.
    let mut indices: Vec<Val> = dims.iter().map(|dim| dim.index).collect();

    // `if (!indices_to_delete.empty()) removeValuesFromIndices(indices, indices_to_delete);`
    //
    // ⭐ THE GUARD IS A NO-OP FOR THE ANSWER: an empty request erases nothing and finds nothing
    // missing, so entry 118 says [`IndexRemoval::Removed`] for it either way.
    let removal = remove_values_from_indices(&mut indices, &indices_to_delete);

    HyperRectConditions {
        insert_refs: nests,
        // `subscripts_map.replaceDimsAndSymbols({new_dim_exprs}, {}, num_dim_vars, 0)`.
        subscripts_map: AffineMap {
            dims: num_dim_vars,
            syms: 0,
            results: subscripts_map
                .results
                .iter()
                .map(|result| replace_dims(result, &new_dim_exprs))
                .collect(),
        },
        indices,
        removal,
    }
}

/// ONE EXPRESSION WITH `d<k>` REPLACED BY `with[k]` — `AffineExpr::replaceDimsAndSymbols` over the
/// dimension list alone.
///
/// # ⛔⛔ IT FOLDS, AND THE VENDOR'S OWN OUTPUT IS THE PROOF
///
/// MLIR builds every replaced node through `getAffineBinaryOpExpr`, which simplifies as it
/// constructs, so substituting a constant into `d0 * 3` yields a CONSTANT and not a multiplication.
/// The vendor's third page pins `arg1` to 1 and prints `agen.vector_load %24[0, 1, ..]`
/// (`paged_mem_view_loads.mlir:90`) — `1 * 3` folded to `3`, which entry 259 then shifts by that
/// page's start element of 2. An unfolded `1 * 3` would print as `1 * 3`, and the flattening every
/// later constraint system does would have to fold it instead.
///
/// ⭐ THE CRATE'S OWN CONSTRUCTORS DO NOT FOLD ([`AffineExpr::times`] builds the node as written),
/// which is why the folding is here rather than borrowed from them, and the rules are the ones
/// [`crate::islands::dataflow_ir::ty::AffineExpr::flatten`] states for the same operators: `mod` and
/// `floordiv` are FLOORED, so `-3 mod 4` is 1 and `-3 floordiv 4` is -1.
///
/// ⚠️ ONLY CONSTANT-OVER-CONSTANT, WHICH IS EXACTLY WHAT THIS SUBSTITUTION CAN NEWLY CREATE. MLIR
/// also folds `x * 1`, `x + 0` and the like, and a map arriving here already has those folded away
/// because MLIR built it; replicating them would change expressions this function did not touch.
///
/// ⚠️ A `d<k>` PAST THE REPLACEMENT LIST IS LEFT ALONE — MLIR's own behaviour for a short list, and
/// unreachable here because the list has one entry per dimension of the map (see the note on
/// [`create_conditions_for_hyper_rect_subscripts`]).
///
/// ⛔ A NON-POSITIVE OR NON-LITERAL DIVISOR IS LEFT UNFOLDED rather than divided by:
/// `assert(rhsConst > 0 && "RHS constant has to be positive")` is MLIR's rule for these two
/// operators, and a node the port declines to fold is still the same expression.
fn replace_dims(expr: &AffineExpr, with: &[AffineExpr]) -> AffineExpr {
    match expr {
        // `getAffineDimExpr(num_dim_vars++, ..)` or `getAffineConstantExpr(lb_val, ..)`, whichever
        // this dimension's turn of the walk pushed.
        AffineExpr::Dim(dim) => with
            .get(*dim as usize)
            .cloned()
            .unwrap_or(AffineExpr::Dim(*dim)),
        // The symbol replacement list is empty — see the note on the caller.
        AffineExpr::Sym(sym) => AffineExpr::Sym(*sym),
        AffineExpr::Const(value) => AffineExpr::Const(*value),
        AffineExpr::Add(lhs, rhs) => {
            let (lhs, rhs) = (replace_dims(lhs, with), replace_dims(rhs, with));
            match (&lhs, &rhs) {
                (AffineExpr::Const(a), AffineExpr::Const(b)) => {
                    AffineExpr::Const(a.saturating_add(*b))
                }
                _ => AffineExpr::Add(Box::new(lhs), Box::new(rhs)),
            }
        }
        AffineExpr::Mul(lhs, rhs) => {
            let (lhs, rhs) = (replace_dims(lhs, with), replace_dims(rhs, with));
            match (&lhs, &rhs) {
                (AffineExpr::Const(a), AffineExpr::Const(b)) => {
                    AffineExpr::Const(a.saturating_mul(*b))
                }
                _ => AffineExpr::Mul(Box::new(lhs), Box::new(rhs)),
            }
        }
        AffineExpr::Mod(lhs, rhs) => {
            let (lhs, rhs) = (replace_dims(lhs, with), replace_dims(rhs, with));
            match (&lhs, &rhs) {
                // "Folded at construction by `simplifyMod`, which is floored: `-3 mod 4` is 1."
                (AffineExpr::Const(a), AffineExpr::Const(b)) if *b > 0 => {
                    AffineExpr::Const(a.rem_euclid(*b))
                }
                _ => AffineExpr::Mod(Box::new(lhs), Box::new(rhs)),
            }
        }
        AffineExpr::FloorDiv(lhs, rhs) => {
            let (lhs, rhs) = (replace_dims(lhs, with), replace_dims(rhs, with));
            match (&lhs, &rhs) {
                // "Folded at construction by `simplifyFloorDiv`: `-3 floordiv 4` is -1."
                (AffineExpr::Const(a), AffineExpr::Const(b)) if *b > 0 => {
                    AffineExpr::Const(a.div_euclid(*b))
                }
                _ => AffineExpr::FloorDiv(Box::new(lhs), Box::new(rhs)),
            }
        }
    }
}
#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::dialects::dataflow::PageSpan;
    use crate::islands::dataflow_ir::dialects::{Index, dataflow, vectorchain};
    use crate::islands::dataflow_ir::link::{Link, Lxlu as LxluUnit, Sfp as SfpUnit};
    use crate::islands::dataflow_ir::print;
    use crate::islands::dataflow_ir::ty::{
        AffineExpr, AffineMap, BoundType, ElemType, MemRef, Vector,
    };

    /// The ops as MLIR text, at the top level.
    fn text(ops: &[DfirOp]) -> String {
        let mut out = String::new();
        for op in ops {
            print::emit(&mut out, op, 0);
        }
        out
    }

    /// `memref<64x4x64xf16>` — the shape both the paged view and the non-paged views cut from it
    /// carry (`dcc/test/Dialect/Dataflow/paged_mem_view.mlir:19-22`).
    fn paged_view_ty() -> MemRef {
        MemRef {
            shape: vec![64, 4, 64],
            elem: ElemType::F16,
        }
    }

    /// `affine_map<(d0, d1, d2) -> (d2 * 64 + d1 * 64 + d0)>` — IBM's own layout map for a paged
    /// view (`dcc/test/Dialect/Dataflow/paged_mem_view.mlir:28`).
    fn layout() -> AffineMap {
        AffineMap {
            dims: 3,
            syms: 0,
            results: vec![
                AffineExpr::dim(2)
                    .times(64)
                    .plus(AffineExpr::dim(1).times(64))
                    .plus(AffineExpr::dim(0)),
            ],
        }
    }

    /// `dataflow.get_paged_logical_memory_view` binding `result`, with one page.
    fn paged_view(result: Val) -> DfirOp {
        DfirOp::Dataflow(dataflow::Op::GetPagedLogicalMemoryView(Box::new(
            PagedMemView {
                result,
                unit: Val(0),
                start_addr: Val(1),
                pages: vec![page(0, 1, Val(2))],
                layout: layout(),
                ty: paged_view_ty(),
            },
        )))
    }

    /// A page of 64 lanes spanning rows `lo ..= hi`, pinned on the third axis.
    fn page(lo: i64, hi: i64, start_addr: Val) -> Page {
        Page {
            idx_set: PageRect {
                spans: vec![
                    PageSpan { lo: 0, hi: 63 },
                    PageSpan { lo, hi },
                    PageSpan { lo: 0, hi: 0 },
                ],
            },
            start_addr,
        }
    }

    /// 🎯 121/384 — A SPAN IS TWO NESTED ONE-ARMED CONDITIONALS, AND SIX OPS IN THE REFERENCE'S OWN
    /// ORDER.
    ///
    /// ⛔ THE SHAPE IS THE ASSERTION. An `arith.andi` of the two comparisons under a single `scf.if`
    /// computes the same predicate and would pass any test that only asked "is the access guarded";
    /// see [`create_inequality_condition`] for what downstream reads the nesting. The value numbers
    /// pin the emission ORDER too — constant, comparison, conditional, twice — because the printed
    /// names are what a vendored file is diffed against.
    #[test]
    fn a_span_is_two_nested_one_armed_conditionals() {
        let mut vals = Values::default();
        let lhs = vals.mint();
        let condition = create_inequality_condition(&mut vals, lhs, 2, 3);
        let guarded = vals.mint();
        let ops = condition.wrap(vec![DfirOp::Arith(arith::Op::Constant {
            result: guarded,
            value: 7,
        })]);

        assert_eq!(
            text(&ops),
            "\
%1 = arith.constant 2 : index
%2 = arith.cmpi sge, %0, %1 : index
scf.if %2 {
  %3 = arith.constant 3 : index
  %4 = arith.cmpi sle, %0, %3 : index
  scf.if %4 {
    %5 = arith.constant 7 : index
  }
}
"
        );
    }

    /// 🎯 121/384 — THE SECOND DIMENSION'S GUARDS SIT INSIDE THE FIRST'S.
    ///
    /// ⭐ WHICH IS WHAT MAKES THE CONJUNCTION ACROSS DIMENSIONS RIGHT: entry 198 walks the subscript
    /// map dimension by dimension and never moves the builder back out, so an access reaching a page
    /// is guarded by every dimension's bounds at once. Four guards, four `scf.if`s, one body.
    #[test]
    fn and_then_nests_the_next_dimension_inside_the_last() {
        let mut vals = Values::default();
        let d0 = vals.mint();
        let d1 = vals.mint();
        let outer = create_inequality_condition(&mut vals, d0, 0, 63);
        let inner = create_inequality_condition(&mut vals, d1, 2, 3);
        let guarded = vals.mint();
        let ops = outer
            .and_then(inner)
            .wrap(vec![DfirOp::Arith(arith::Op::Constant {
                result: guarded,
                value: 7,
            })]);

        assert_eq!(
            text(&ops),
            "\
%2 = arith.constant 0 : index
%3 = arith.cmpi sge, %0, %2 : index
scf.if %3 {
  %4 = arith.constant 63 : index
  %5 = arith.cmpi sle, %0, %4 : index
  scf.if %5 {
    %6 = arith.constant 2 : index
    %7 = arith.cmpi sge, %1, %6 : index
    scf.if %7 {
      %8 = arith.constant 3 : index
      %9 = arith.cmpi sle, %1, %8 : index
      scf.if %9 {
        %10 = arith.constant 7 : index
      }
    }
  }
}
"
        );
    }

    /// 🎯 122/384 — THE TWO ARMS PUT THE STATEMENTS IN TWO DIFFERENT PLACES.
    ///
    /// ⛔ AND THE UNGUARDED ARM IS NOT A NO-OP: it is the case where the subscript can only reach one
    /// page, so the access is emitted in the reference's own block with no condition at all. Reading
    /// the `dyn_cast` the wrong way round emits an unconditional access to one page of many, which
    /// nothing downstream objects to.
    #[test]
    fn the_insert_reference_decides_guarded_or_in_place() {
        let mut vals = Values::default();
        let lhs = vals.mint();
        let condition = create_inequality_condition(&mut vals, lhs, 0, 1);
        let marker = vals.mint();
        let ops = vec![DfirOp::Arith(arith::Op::Constant {
            result: marker,
            value: 7,
        })];

        // `dyn_cast<scf::IfOp>(insert_ref)` succeeds: `builder = if_op.getThenBodyBuilder()`.
        let guarded = set_builder_to_insert_ref(InsertRef::Conditional(&condition), ops.clone());
        assert_eq!(
            text(&guarded),
            "\
%1 = arith.constant 0 : index
%2 = arith.cmpi sge, %0, %1 : index
scf.if %2 {
  %3 = arith.constant 1 : index
  %4 = arith.cmpi sle, %0, %3 : index
  scf.if %4 {
    %5 = arith.constant 7 : index
  }
}
"
        );

        // `builder.setInsertionPoint(insert_ref)` — the memory operation's own block, unguarded.
        assert_eq!(
            set_builder_to_insert_ref(InsertRef::MemOp, ops.clone()),
            ops
        );
    }

    /// 🎯 123/384 — THE REFERENCE'S OWN EXAMPLE, AND THE TYPE GUARD CHECKED AGAINST THE MLIR ROUTINE.
    ///
    /// ⭐⭐ THE SET IS THE ONE `TransformPagedMemViewImpl.cpp:527-531` WRITES IN ITS COMMENT, and the
    /// answer it states is `[0, 2, 0]`. Printing the set proves
    /// [`crate::islands::dataflow_ir::dialects::dataflow::PageRect`] means what the comment's set
    /// means — six constraints, `d1 - 2 >= 0` spelled as a subtraction — rather than merely
    /// round-tripping our own construction.
    ///
    /// ⛔⛔ AND THE LAST LOOP IS WHY THE DELETED `DT_CHECK` IS SOUND. A span's `lo` is *asserted* to
    /// be `getConstantBound(LB, dim)`; here that assertion is executed, dimension by dimension,
    /// against the ported [`crate::islands::dataflow_ir::ty::IntegerSet::constant_bound`]. If the two
    /// ever disagreed, "the type makes the check unnecessary" would be false.
    #[test]
    fn the_start_elements_are_the_references_own_example() {
        let page_sel_constraints = PageRect {
            spans: vec![
                PageSpan { lo: 0, hi: 63 },
                PageSpan { lo: 2, hi: 3 },
                PageSpan { lo: 0, hi: 1 },
            ],
        };
        let set = page_sel_constraints.as_integer_set();
        assert_eq!(
            print::integer_set(&set),
            "affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 - 2 >= 0, -d1 + 3 >= 0, \
             d2 >= 0, -d2 + 1 >= 0)>"
        );

        let start_elements = calculate_start_elements_for_page(&page_sel_constraints);
        assert_eq!(
            start_elements,
            vec![StartElement(0), StartElement(2), StartElement(0)]
        );

        for (dim, element) in start_elements.iter().enumerate() {
            let dim = u32::try_from(dim).expect("three dimensions fit a u32");
            assert_eq!(set.constant_bound(BoundType::Lb, dim), Some(element.0));
        }
    }

    /// 🎯 124/384 — THE NEW VIEW STARTS AT THE **SUM**, AND INHERITS EVERYTHING ELSE.
    ///
    /// ⛔ THE VIEW'S OWN START IS `%1` AND THE PAGE'S IS `%3`; the `arith.addi` of the two is what
    /// makes the second page addressable. A view built on the page's start alone prints identically
    /// for a view that begins at zero and reads the wrong memory for every other one.
    #[test]
    fn the_non_paged_view_starts_at_the_sum_of_the_two_addresses() {
        let mut vals = Values::default();
        let unit = vals.mint();
        let view_start = vals.mint();
        let page0_start = vals.mint();
        let page1_start = vals.mint();
        let paged_result = vals.mint();
        let paged_mem_view = PagedMemView {
            result: paged_result,
            unit,
            start_addr: view_start,
            pages: vec![page(0, 1, page0_start), page(2, 3, page1_start)],
            layout: layout(),
            ty: paged_view_ty(),
        };

        let made = create_non_paged_mem_view(&mut vals, &paged_mem_view, &paged_mem_view.pages[1]);

        assert_eq!(
            text(&made.ops),
            "\
%5 = arith.addi %1, %3 : index
%6 = dataflow.get_logical_memory_view %0, %5 {layout_map = affine_map<(d0, d1, d2) -> \
             (d2 * 64 + d1 * 64 + d0)>} : index, index, memref<64x4x64xf16>
"
        );
        assert_eq!(made.start_addr, Val(5));
        assert_eq!(made.view, Val(6));
    }

    /// 🎯 125/384 — A NON-PAGED VIEW IS CLONED; A PAGED ONE AND A REGION ARGUMENT ARE NOT.
    ///
    /// ⭐ THE CLONE BINDS ITS OWN VALUE (`%3`, not `%2`) — two ops binding one value is the diagnosis
    /// [`crate::islands::dataflow_ir::Values`] exists to make impossible.
    ///
    /// ⛔ AND THE PAGED CASE MATTERS: cloning the paged view would duplicate the op the pass is
    /// deleting, in a branch that is about to read a non-paged view instead.
    #[test]
    fn only_a_non_paged_view_is_cloned() {
        let mut vals = Values::default();
        let unit = vals.mint();
        let start = vals.mint();
        let mem_view = vals.mint();
        let non_paged = DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
            result: mem_view,
            from: unit,
            start,
            layout: layout(),
            ty: paged_view_ty(),
        });

        let cloned = clone_mem_view_if_non_paged(&mut vals, Some(&non_paged), mem_view);
        assert_eq!(
            text(&cloned.ops),
            "\
%3 = dataflow.get_logical_memory_view %0, %1 {layout_map = affine_map<(d0, d1, d2) -> \
             (d2 * 64 + d1 * 64 + d0)>} : index, index, memref<64x4x64xf16>
"
        );
        assert_eq!(cloned.value, Val(3));

        // `dyn_cast_or_null<GetLogicalMemoryViewOp>` fails on the paged view: pass it through.
        let paged_result = vals.mint();
        let paged = DfirOp::Dataflow(dataflow::Op::GetPagedLogicalMemoryView(Box::new(
            PagedMemView {
                result: paged_result,
                unit,
                start_addr: start,
                pages: vec![page(0, 1, start)],
                layout: layout(),
                ty: paged_view_ty(),
            },
        )));
        let same = clone_mem_view_if_non_paged(&mut vals, Some(&paged), paged_result);
        assert_eq!(same.ops, Vec::new());
        assert_eq!(same.value, paged_result);

        // `mem_view.getDefiningOp()` is null for a region argument — `dyn_cast_or_null`'s own case.
        let arg = vals.mint();
        let untouched = clone_mem_view_if_non_paged(&mut vals, None, arg);
        assert_eq!(untouched.ops, Vec::new());
        assert_eq!(untouched.value, arg);
    }

    /// A load, an estimate over its result, and a store of that — the shape of chain the pass clones.
    ///
    /// The terminator is an `agen.vector_store`, which has no results; the reference's own example of
    /// a chain end is a `dataflow.send`, and [`a_send_terminated_chain_keeps_its_wire`] uses that one.
    fn a_chain(vals: &mut Values) -> (Val, Val, Vec<DfirOp>) {
        let view = vals.mint();
        let loaded = vals.mint();
        let estimated = vals.mint();
        let ops = vec![
            DfirOp::Agen(agen::Op::VectorLoad {
                dbg_name: None,
                result: loaded,
                view,
                indices: vec![Index::Const(0), Index::Const(0), Index::Const(0)],
                view_ty: paged_view_ty(),
                ty: LANES,
                multicast_info: None,
            }),
            DfirOp::VectorChain(vectorchain::Op::Estimate {
                result: estimated,
                input: loaded,
                kind: vectorchain::EstimateKind::Rec,
                version: None,
                input_ty: LANES,
                ty: LANES,
            }),
            DfirOp::Agen(agen::Op::VectorStore {
                dbg_name: None,
                value: estimated,
                view,
                indices: vec![Index::Const(0), Index::Const(0), Index::Const(0)],
                view_ty: paged_view_ty(),
                ty: LANES,
            }),
        ];
        (view, loaded, ops)
    }

    /// 🎯 126/384 — THE CHAIN IS THE LOAD AND EVERYTHING DOWNSTREAM, ENDING AT THE OP WITH NO RESULT.
    ///
    /// ⛔ AND A FORK IS NOT A SHORTER CHAIN, IT IS NO CHAIN. The reference asserts one use per value
    /// and aborts otherwise; the answer here is [`UseChain::None`] — `getUseChain`'s own documented
    /// "Empty if there isn't a use chain" (`TransformPagedMemViewImpl.hpp:323-324`) — so a second
    /// consumer means the load keeps the tail it already has rather than half of it being copied
    /// into a guarded branch.
    #[test]
    fn the_use_chain_runs_to_the_op_with_no_results() {
        let mut vals = Values::default();
        let (view, _, ops) = a_chain(&mut vals);
        let mem_op = VectorLoadOp::of(&ops[0]).expect("the first op is the load");

        let chain = get_use_chain(mem_op, &ops);
        assert_eq!(
            chain,
            UseChain::ConsumerWard(vec![&ops[0], &ops[1], &ops[2]])
        );
        assert!(dialects::results(&ops[2]).is_empty());

        // A second reader of the estimate: `res.hasOneUse()` is false, and the walk stops there.
        let mut forked = ops.clone();
        let estimated = dialects::results(&ops[1])[0];
        forked.push(DfirOp::Agen(agen::Op::VectorStore {
            dbg_name: None,
            value: estimated,
            view,
            indices: vec![Index::Const(1), Index::Const(0), Index::Const(0)],
            view_ty: paged_view_ty(),
            ty: LANES,
        }));
        let mem_op = VectorLoadOp::of(&forked[0]).expect("the first op is the load");
        assert_eq!(get_use_chain(mem_op, &forked), UseChain::None);
    }

    /// 🎯 127/384 — EVERY CLONE READS THE PREVIOUS **CLONE**, AND THE LOAD ITSELF IS NOT CLONED.
    ///
    /// ⛔⛔ THIS IS THE ONE DEFECT A SHAPE-ONLY TEST WOULD MISS. Drop
    /// `replaceUsesOfWith` and the printed chain still has the right ops in the right order — but
    /// `%6` reads `%1`, the ORIGINAL load, so the guarded branch computes from the page the pass was
    /// rewriting away from. The assertion is therefore on the OPERANDS: the cloned estimate reads the
    /// new load's `%4`, and the cloned store reads the cloned estimate's `%6`.
    #[test]
    fn the_cloned_chain_reads_the_new_load() {
        let mut vals = Values::default();
        let (view, _, ops) = a_chain(&mut vals);
        let mem_op = VectorLoadOp::of(&ops[0]).expect("the first op is the load");

        // The new load entry 128 creates; only its result is used here.
        let new_result = vals.mint();
        let new_load = DfirOp::Agen(agen::Op::VectorLoad {
            dbg_name: None,
            result: new_result,
            view,
            indices: vec![Index::Const(0), Index::Const(2), Index::Const(0)],
            view_ty: paged_view_ty(),
            ty: LANES,
            multicast_info: None,
        });
        let new_mem_op = VectorLoadOp::of(&new_load).expect("it is a load");

        let cloned = clone_use_chain(&mut vals, mem_op, new_mem_op, &ops);

        // Two ops: the chain without its first element.
        assert_eq!(cloned.len(), 2);
        assert_eq!(
            text(&cloned),
            "\
%4 = vectorchain.rec_estimate %3 : vector<64xf16>, vector<64xf16>
agen.vector_store %4, %0[0, 0, 0] {store_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, \
             store_set = affine_set<(d0, d1, d2) : (d0 == 0, d1 == 0, d2 >= 0, -d2 + 63 >= 0)>} \
             : memref<64x4x64xf16>, vector<64xf16>
"
        );
        assert_eq!(dialects::operands(&cloned[0]), vec![new_result]);
        assert_eq!(dialects::results(&cloned[0]), vec![Val(4)]);
    }

    /// 🎯 127/384 — A CHAIN THAT ENDS IN A `dataflow.send` HAS ITS DATA RE-POINTED AND ITS WIRE LEFT
    /// ALONE.
    ///
    /// ⭐⭐ THE REFERENCE'S OWN EXAMPLE OF A CHAIN END IS A SEND (`Agen.cpp:120`), and a send carries
    /// two operands of very different kinds: the DATA, which the clone must re-point, and the wire
    /// end, which it must not. [`dialects::operands_mut`] excludes the second on purpose; this is the
    /// test that the exclusion is the right one — the cloned send reads the cloned estimate and still
    /// drives the same unit.
    #[test]
    fn a_send_terminated_chain_keeps_its_wire() {
        let mut vals = Values::default();
        let view = vals.mint();
        let producer = vals.mint();
        let consumer = vals.mint();
        let loaded = vals.mint();
        let estimated = vals.mint();
        let (to, _) = Link::<LxluUnit, SfpUnit>::between(producer, consumer).ends();
        let ops = vec![
            DfirOp::Agen(agen::Op::VectorLoad {
                dbg_name: None,
                result: loaded,
                view,
                indices: vec![Index::Const(0), Index::Const(0), Index::Const(0)],
                view_ty: paged_view_ty(),
                ty: LANES,
                multicast_info: None,
            }),
            DfirOp::VectorChain(vectorchain::Op::Estimate {
                result: estimated,
                input: loaded,
                kind: vectorchain::EstimateKind::Rec,
                version: None,
                input_ty: LANES,
                ty: LANES,
            }),
            DfirOp::Dataflow(dataflow::Op::Send {
                to,
                data: estimated,
                ty: LANES,
            }),
        ];
        let mem_op = VectorLoadOp::of(&ops[0]).expect("the first op is the load");
        let new_result = vals.mint();
        let new_load = DfirOp::Agen(agen::Op::VectorLoad {
            dbg_name: None,
            result: new_result,
            view,
            indices: vec![Index::Const(0), Index::Const(2), Index::Const(0)],
            view_ty: paged_view_ty(),
            ty: LANES,
            multicast_info: None,
        });
        let new_mem_op = VectorLoadOp::of(&new_load).expect("it is a load");

        let cloned = clone_use_chain(&mut vals, mem_op, new_mem_op, &ops);

        assert_eq!(
            text(&cloned),
            "\
%6 = vectorchain.rec_estimate %5 : vector<64xf16>, vector<64xf16>
dataflow.send %2, %6 : vector<64xf16>
"
        );
    }

    /// 🎯 128/384 — THE NEW LOAD FIRST, THEN ITS CHAIN, ALL READING FORWARD.
    ///
    /// ⭐ THE ORDER IS `builder.setInsertionPointAfter(new_load_op)` made structural, and it is what
    /// keeps the emitted region in SSA order: the estimate cannot precede the load whose value it
    /// reads.
    ///
    /// ⛔ AND THE NEW LOAD IS THE ORIGINAL IN EVERY RESPECT BUT THE ACCESS — the same
    /// `vector<64xf16>`, the same `memref<64x4x64xf16>`, hence the same derived `load_set` and
    /// `load_order`; only the view, the page-relative indices and the `dbgName` the clone
    /// substitutes for the nameless original (`Agen.cpp:166`) differ.
    #[test]
    fn the_new_access_is_the_load_then_its_cloned_chain() {
        let mut vals = Values::default();
        let (_, _, ops) = a_chain(&mut vals);
        let mem_op = VectorLoadOp::of(&ops[0]).expect("the first op is the load");
        let mem_view = vals.mint();

        let made = create_new_mem_op(
            &mut vals,
            mem_op,
            mem_view,
            paged_view_ty(),
            // The page-relative subscripts entry 259 produces for a page starting at row 2.
            vec![Index::Const(0), Index::Const(0), Index::Const(0)],
            &ops,
        );

        assert_eq!(made.result, Val(4));
        assert_eq!(
            text(&made.ops),
            "\
%4 = agen.vector_load %3[0, 0, 0] {dbgName = \"\", \
             load_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, \
             load_set = affine_set<(d0, d1, d2) : (d0 == 0, d1 == 0, d2 >= 0, -d2 + 63 >= 0)>} \
             : memref<64x4x64xf16>, vector<64xf16>
%5 = vectorchain.rec_estimate %4 : vector<64xf16>, vector<64xf16>
agen.vector_store %5, %0[0, 0, 0] {store_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>, \
             store_set = affine_set<(d0, d1, d2) : (d0 == 0, d1 == 0, d2 >= 0, -d2 + 63 >= 0)>} \
             : memref<64x4x64xf16>, vector<64xf16>
"
        );
    }

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
            dbg_name: None,
            result,
            view: VIEW,
            indices: indices(),
            view_ty: view_ty(),
            ty: LANES,
            multicast_info: None,
        })
    }

    /// `agen.vector_store %data, %dst_mem_view[…] : memref<64x2x64x2x3xf16>, vector<64xf16>`
    /// (`paged_mem_view_load_and_store.mlir:1289`).
    fn vector_store(value: Val) -> DfirOp {
        DfirOp::Agen(agen::Op::VectorStore {
            dbg_name: None,
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
    /// (`Agen.cpp:115-136`) and `eraseOpAndUseChain` walks it BACKWARDS (`:177-178`), so the send
    /// goes first and the load last.
    #[test]
    fn a_paged_loads_chain_is_erased_from_its_send_back_to_the_load() {
        let program = vec![
            vector_load(Val(31)),
            rotate(Val(41), Val(31)),
            send(Val(41)),
        ];
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
    /// `VectorLoadOp::eraseOpAndUseChain` reverses its chain (`Agen.cpp:177-178`) and
    /// `VectorStoreOp::eraseOpAndUseChain` does not (`:258`), because a store's chain is collected
    /// producer-ward with the store itself first (`:207-222`). Both end up consumer-first.
    #[test]
    fn the_direction_decides_whether_the_chain_is_reversed() {
        let program = vec![
            vector_load(Val(31)),
            rotate(Val(41), Val(31)),
            send(Val(41)),
        ];
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
        let both = vec![vector_load(Val(31)), vector_store(Val(31)), send(Val(31))];
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
        let program = vec![
            vector_load(Val(31)),
            rotate(Val(41), Val(31)),
            send(Val(41)),
        ];
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
        let program = [vector_load(Val(31)), send(Val(31))];
        let tpmv = TpmvVector::new(&program[0], DfirUnit::Lxlu);

        assert_eq!(tpmv.base.mem_ops, vec![&program[0]]);
        assert_eq!(tpmv.base.comp, DfirUnit::Lxlu);

        // ⭐ AND IT ADDS NO STATE OF ITS OWN: the base is all of it.
        assert_eq!(tpmv.base, TpmvBase::new(&program[0], DfirUnit::Lxlu));
    }

    /// 🎯 137/384 — THE DERIVED CONSTRUCTOR SEEDS THE SAME ONE OP, TWO LEVELS DOWN.
    ///
    /// `: TPMVVector(mem_op, comp) {}` (`hpp:397`) has no body at all, so the only thing to check is
    /// that the delegation chain runs: `mem_ops_` holds the op it was handed and `comp_` the
    /// component, both written by [`TpmvBase::new`] through [`TpmvVector::new`].
    #[test]
    fn the_vector_load_constructor_carries_the_op_and_the_component() {
        let program = [vector_load(Val(31)), send(Val(31))];
        let tpmv = TpmvVectorLoad::new(&program[0], DfirUnit::Lxlu);

        assert_eq!(tpmv.vector.base.mem_ops, vec![&program[0]]);
        assert_eq!(tpmv.vector.base.comp, DfirUnit::Lxlu);

        // ⭐ AND IT ADDS NO STATE OF ITS OWN either, so the whole object is entry 136's.
        assert_eq!(tpmv.vector, TpmvVector::new(&program[0], DfirUnit::Lxlu));
    }

    /// 🎯 138/384 — THE SAME ON THE COMPOSITE BRANCH, which reaches [`TpmvBase`] through
    /// [`TpmvComposite`] rather than [`TpmvVector`].
    #[test]
    fn the_composite_load_constructor_carries_the_op_and_the_component() {
        let program = [vector_load(Val(31))];
        let tpmv = TpmvCompositeLoad::new(&program[0], DfirUnit::Sfp);

        assert_eq!(tpmv.composite.base.mem_ops, vec![&program[0]]);
        assert_eq!(tpmv.composite.base.comp, DfirUnit::Sfp);
    }

    /// ⛔ THE `store_op` ARGUMENT IS DROPPED — `mem_ops_` holds the LOAD and nothing else after
    /// construction, which `initialize` re-asserts (`DT_CHECK(mem_ops_.size() == 1)`,
    /// `Impl.cpp:756`), and entry 130's `getStoreOp` is what puts the store back — as the
    /// destination `TPMVInfo`, not as a second `mem_ops_` entry (`Impl.cpp:766-770`). A port that
    /// stashed the store here would make that entry dead.
    #[test]
    fn the_load_store_constructor_drops_the_store() {
        let program = vec![vector_load(Val(31)), vector_store(Val(31))];
        let tpmv = TpmvVectorLoadStore::new(&program[0], &program[1], DfirUnit::Lxlu);

        assert_eq!(tpmv.vector.base.mem_ops, vec![&program[0]]);
        assert_eq!(
            store_op_from_load_store_pattern(AgenOpKind::VectorStore, &program[0], &program),
            Some(&program[1])
        );
    }

    /// ⛔ THE SIX LEAVES ARE SIX TYPES, which is what makes entry 139 a choice rather than a
    /// constructor call. Nothing here can assert that at run time — the check is that the six
    /// constructors exist and yield values of six distinct types, which is a compile-time fact this
    /// test's mere existence establishes.
    #[test]
    fn all_six_leaves_construct() {
        let program = [vector_load(Val(31)), vector_store(Val(31))];
        let (load, store) = (&program[0], &program[1]);
        let comp = DfirUnit::Lxlu;

        let vector_load = TpmvVectorLoad::new(load, comp);
        let vector_store = TpmvVectorStore::new(store, comp);
        let vector_load_store = TpmvVectorLoadStore::new(load, store, comp);
        let composite_load = TpmvCompositeLoad::new(load, comp);
        let composite_store = TpmvCompositeStore::new(store, comp);
        let composite_load_store = TpmvCompositeLoadStore::new(load, comp);

        assert_eq!(vector_load.vector.base.mem_ops, vec![load]);
        assert_eq!(vector_store.vector.base.mem_ops, vec![store]);
        assert_eq!(vector_load_store.vector.base.mem_ops, vec![load]);
        assert_eq!(composite_load.composite.base.mem_ops, vec![load]);
        assert_eq!(composite_store.composite.base.mem_ops, vec![store]);
        assert_eq!(composite_load_store.composite.base.mem_ops, vec![load]);
    }
    /// 🎯 118/384 — THE COLLAPSED DIMENSIONS COME OUT AND THE SURVIVORS KEEP THEIR ORDER.
    ///
    /// The delete list is built as `indices_to_delete.push_back(indices[dim])` in ascending `dim`
    /// (`:319`), so it is a subsequence of the indices — here dimensions 1 and 3 of four.
    #[test]
    fn the_requested_indices_are_removed_in_order() {
        let mut indices = vec![Val(10), Val(11), Val(12), Val(13)];
        assert_eq!(
            remove_values_from_indices(&mut indices, &[Val(11), Val(13)]),
            IndexRemoval::Removed
        );
        assert_eq!(indices, vec![Val(10), Val(12)]);
    }

    /// 🎯 118/384 — EVERY DIMENSION COLLAPSING LEAVES NO INDICES AT ALL.
    ///
    /// A subscript whose every page-selection bound is a point is one fixed address, which is exactly
    /// the case the pass turns into a constant map with no operands.
    #[test]
    fn deleting_every_index_empties_the_list() {
        let mut indices = vec![Val(4), Val(5)];
        assert_eq!(
            remove_values_from_indices(&mut indices, &[Val(4), Val(5)]),
            IndexRemoval::Removed
        );
        assert!(indices.is_empty());

        // ⭐ AND AN EMPTY REQUEST IS THE `if (!indices_to_delete.empty())` GUARD AT `:338` — the
        // reference does not even call this, and calling it changes nothing.
        let mut untouched = vec![Val(4), Val(5)];
        assert_eq!(
            remove_values_from_indices(&mut untouched, &[]),
            IndexRemoval::Removed
        );
        assert_eq!(untouched, vec![Val(4), Val(5)]);
    }

    /// 🎯 118/384 — A VALUE THAT IS NOT AN INDEX IS THE `DT_CHECK_MSG`, NAMED.
    ///
    /// *"could not find value to delete in indices vector"*, and the requests before it are already
    /// gone — the state the reference aborts in.
    #[test]
    fn a_value_that_is_not_an_index_is_reported() {
        let mut indices = vec![Val(1), Val(2), Val(3)];
        assert_eq!(
            remove_values_from_indices(&mut indices, &[Val(2), Val(99), Val(3)]),
            IndexRemoval::NotAnIndex(Val(99))
        );
        assert_eq!(indices, vec![Val(1), Val(3)]);
    }

    /// 🎯 118/384 — A DUPLICATED INDEX LOSES ITS FIRST COPY, WHICH IS `std::find` + `erase(it)`.
    #[test]
    fn a_duplicated_index_loses_its_first_occurrence() {
        let mut indices = vec![Val(7), Val(8), Val(7)];
        assert_eq!(
            remove_values_from_indices(&mut indices, &[Val(7)]),
            IndexRemoval::Removed
        );
        assert_eq!(indices, vec![Val(8), Val(7)]);
    }

    /// 🎯 119/384 — `(d0, d1) -> (d0 * 8 + d1)` BECOMES `()[s0, s1] -> (s0 * 8 + s1)`.
    ///
    /// The shape of a real subscript map: one linearised result over two loop iterators.
    #[test]
    fn every_dimension_becomes_the_symbol_at_its_position() {
        let map = AffineMap {
            dims: 2,
            syms: 0,
            results: vec![AffineExpr::dim(0).times(8).plus(AffineExpr::dim(1))],
        };
        let syms = replace_dims_in_map_with_syms(&map);

        assert_eq!(syms.dims, 0);
        assert_eq!(syms.syms, 2);
        assert_eq!(
            print::affine_map(&syms),
            "affine_map<()[s0, s1] -> (s0 * 8 + s1)>"
        );
        // ⛔ AND THE INPUT IS UNTOUCHED — both callers keep the dimension form for the access itself.
        assert_eq!(
            print::affine_map(&map),
            "affine_map<(d0, d1) -> (d0 * 8 + d1)>"
        );
    }

    /// 🎯 119/384 — THE SUBSTITUTION REACHES EVERY LEAF, INCLUDING UNDER `mod` AND `floordiv`.
    ///
    /// `replaceDimsAndSymbols` walks the expression tree; a shallow rewrite would leave the dimensions
    /// nested inside a `(d0 mod 128) floordiv 2` — the int8 reduction map's shape — behind, and the
    /// constraint system would then solve for variables the map no longer mentions.
    #[test]
    fn the_substitution_is_structural() {
        let map = AffineMap {
            dims: 3,
            syms: 0,
            results: vec![
                AffineExpr::FloorDiv(
                    Box::new(AffineExpr::Mod(
                        Box::new(AffineExpr::dim(0)),
                        Box::new(AffineExpr::Const(128)),
                    )),
                    Box::new(AffineExpr::Const(2)),
                ),
                AffineExpr::dim(2).times(4).plus(AffineExpr::dim(1)),
                AffineExpr::Const(7),
            ],
        };
        assert_eq!(
            print::affine_map(&replace_dims_in_map_with_syms(&map)),
            "affine_map<()[s0, s1, s2] -> ((s0 mod 128) floordiv 2, s2 * 4 + s1, 7)>"
        );
    }

    /// 🎯 119/384 — A MAP WITH NO DIMENSIONS IS ALREADY ITS OWN SYMBOL FORM.
    ///
    /// The `for` runs zero times, `num_args` stays 0, and `replaceDimsAndSymbols({}, {}, 0, 0)` is the
    /// identity on a constant map. This is what a fully collapsed subscript looks like after
    /// [`remove_values_from_indices`] has taken all its operands away.
    #[test]
    fn a_constant_map_is_unchanged() {
        let map = AffineMap::constants(0, &[0, 64]);
        let syms = replace_dims_in_map_with_syms(&map);
        assert_eq!(syms, map);
        assert_eq!(print::affine_map(&syms), "affine_map<() -> (0, 64)>");
    }

    /// 🎯 120/384 — THE VENDOR'S OWN THREE OPS, IN ORDER.
    ///
    /// `dcc-opt --dcc-transform-paged-mem-view paged_mem_view_loads.mlir` produces
    ///
    /// ```text
    /// %[[VAL_17]] = arith.constant 0 : index
    /// %[[VAL_18]] = arith.cmpi eq, %[[VAL_15]], %[[VAL_17]] : index
    /// scf.if %[[VAL_18]] {
    /// ```
    ///
    /// (`dcc/test/Transform/TransformPagedMemView/paged_mem_view_loads.mlir:45-47`, `CHECK-SENT-IR`
    /// lines), where `%[[VAL_15]]` is the outer `affine.for`'s induction variable and the page-selection
    /// bounds for that dimension collapsed to 0. ⭐ NO `else`, NO RESULT LIST, AND AN EMPTY `then` — the
    /// caller is what fills it.
    #[test]
    fn the_vendors_equality_condition_is_three_ops() {
        let mut vals = Values::default();
        let iv = vals.mint();
        let condition = create_equality_condition(&mut vals, iv, 0);

        let mut out = String::new();
        for op in &condition.wrap(Vec::new()) {
            print::emit(&mut out, op, 0);
        }
        assert_eq!(
            out,
            "%1 = arith.constant 0 : index\n%2 = arith.cmpi eq, %0, %1 : index\nscf.if %2 {\n}\n"
        );

        // ⛔ ONE GUARD, `eq`, AND WHAT IT GUARDS IS WHERE THE NEXT CONDITION GOES.
        assert_eq!(
            condition.guards,
            vec![BoundGuard {
                lhs: iv,
                predicate: CmpIPredicate::Eq,
                bound: 0,
                constant: Val(1),
                cond: Val(2),
            }]
        );
    }

    /// 🎯 120/384 — THE VENDOR'S SECOND AND THIRD PAGES, WITH THEIR OWN CONSTANTS.
    ///
    /// `:79-81` tests the same induction variable against 1, and `:131-132` tests a different one
    /// against 0 — one condition per page candidate, each minting its own constant rather than sharing
    /// one. The reference builds `ConstantIndexOp` unconditionally, and the vendor's output shows the
    /// duplicates (`%17`, `%27`, `%37`, `%47` are all `arith.constant 0`/`1`).
    #[test]
    fn each_condition_mints_its_own_constant() {
        let mut vals = Values::default();
        let iv = vals.mint();
        let first = create_equality_condition(&mut vals, iv, 0);
        let second = create_equality_condition(&mut vals, iv, 1);

        assert_eq!(first.guards[0].constant, Val(1));
        assert_eq!(first.guards[0].bound, 0);
        assert_eq!(second.guards[0].constant, Val(3));
        assert_eq!(second.guards[0].bound, 1);
        assert_ne!(first.guards[0].cond, second.guards[0].cond);
        assert_ne!(first.wrap(Vec::new()), second.wrap(Vec::new()));
    }

    /// The vendor's own nest — `affine.for %arg1 = 0 to 2 { affine.for %arg2 = 0 to 4 { .. } }`
    /// (`dcc/test/Transform/TransformPagedMemView/paged_mem_view_loads.mlir:286-287`), whose body
    /// holds the paged access `%mem_view[%c0, %arg1 * 3, %arg2 * 2 + %c2]`.
    fn vendor_loop_nest() -> Vec<DfirOp> {
        vec![DfirOp::Affine(affine::Op::For {
            iv: Val(101),
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(2),
            carried: Vec::new(),
            body: vec![DfirOp::Affine(affine::Op::For {
                iv: Val(102),
                lo: affine::Bound::Const(0),
                hi: affine::Bound::Const(4),
                carried: Vec::new(),
                body: Vec::new(),
                dbg_name: None,
            })],
            dbg_name: None,
        })]
    }

    /// 🎯 197/384 — TWO LOOP ITERATORS, TWO CLOSED INTERVALS, AND THE UPPER BOUND IS ONE LESS THAN
    /// THE LOOP'S.
    ///
    /// ⛔ THE `- 1` IS THE ASSERTION. `affine.for %arg1 = 0 to 2` runs over `{0, 1}` and
    /// `%arg2 = 0 to 4` over `{0, 1, 2, 3}`, so the vendor's nest gives `[0, 1]` and `[0, 3]`. Taking
    /// the exclusive bound through would give `[0, 2]` and `[0, 4]` — one element too wide on every
    /// axis, which makes a page the iterators cannot reach look reachable.
    #[test]
    fn the_iteration_space_is_the_closed_interval_the_loop_covers() {
        let scope = vendor_loop_nest();
        let indices = SubscriptIv::all_of(&[Val(101), Val(102)], &scope)
            .expect("both subscripts are constant-bounded affine.for induction variables");

        let mut ranges = Vec::new();
        calculate_indices_ranges(&indices, &mut ranges);

        assert_eq!(
            vec![IvRange { lb: 0, ub: 1 }, IvRange { lb: 0, ub: 3 }],
            ranges
        );
    }

    /// 🎯 197/384 — IT APPENDS, WHICH IS WHAT LETS THE TIME DIMENSIONS FOLLOW.
    ///
    /// ⛔⛔ THE ORDER IS THE ASSERTION, not the contents. `:1010-1019` runs this and then
    /// [`add_time_dim_indices_ranges`] over one vector and checks the total against the map's
    /// dimension count, and every constraint row is built by indexing that vector by symbol number
    /// ([`NonTimeDims::sym_for`]). A port that cleared the vector, or prepended, would pass a test
    /// that only counted entries.
    #[test]
    fn the_ranges_append_before_the_time_dimensions() {
        let scope = vendor_loop_nest();
        let indices =
            SubscriptIv::all_of(&[Val(101), Val(102)], &scope).expect("both are iterators");

        let mut ranges = Vec::new();
        calculate_indices_ranges(&indices, &mut ranges);
        let steps = TimeSteps::of(TimeBound::Steps(8)).expect("eight steps is a real bound");
        add_time_dim_indices_ranges(&[steps], &mut ranges);

        assert_eq!(
            vec![
                IvRange { lb: 0, ub: 1 },
                IvRange { lb: 0, ub: 3 },
                IvRange { lb: 0, ub: 7 },
            ],
            ranges
        );
        // And the slot entry 132's arithmetic names for time dim 0 is where that range landed.
        let sym = NonTimeDims(2).sym_for(TimeDim(0));
        assert_eq!(PageSelSym(2), sym);
        assert_eq!(IvRange { lb: 0, ub: 7 }, ranges[sym.0 as usize]);
    }

    /// 🎯 197/384 — ⛔ `DT_CHECK_MSG(block_arg, "expecting only BlockArguments in indices")`.
    ///
    /// A subscript that no block binds — an `arith.constant` folded into the map would be one — is
    /// the reference's first abort.
    #[test]
    fn a_subscript_no_block_binds_is_not_a_loop_iterator() {
        let scope = vendor_loop_nest();

        assert_eq!(None, SubscriptIv::of(Val(7), &scope));
    }

    /// 🎯 197/384 — ⛔⛔ THE SECOND ABORT, AND IT IS THE ONE ENTRY 142 WOULD HAVE ANSWERED.
    ///
    /// *"agen memory operations involving loop iterators can only have loop iterators from
    /// affine::AffineForOps in the subscripts"* — [`super::tf_utils::get_dataflow_for_loop_info_if_iv`]
    /// tests `scf::ForOp` FIRST and answers constant bounds for it, so a port that simply forwarded
    /// its result would accept this scope. The `dyn_cast<affine::AffineForOp>` is what refuses it.
    #[test]
    fn an_scf_for_iterator_is_refused_even_though_its_bounds_are_constant() {
        let lo = Val(200);
        let hi = Val(201);
        let step = Val(202);
        let scope = vec![
            DfirOp::Arith(arith::Op::Constant {
                result: lo,
                value: 0,
            }),
            DfirOp::Arith(arith::Op::Constant {
                result: hi,
                value: 4,
            }),
            DfirOp::Arith(arith::Op::Constant {
                result: step,
                value: 1,
            }),
            DfirOp::Scf(scf::Op::For {
                iv: Val(203),
                lo,
                hi,
                step,
                carried: Vec::new(),
                body: Vec::new(),
                dbg_name: None,
            }),
        ];

        // Entry 142 does answer for it — which is why the class test above it is not redundant.
        assert!(get_dataflow_for_loop_info_if_iv(Val(203), &scope).is_some());
        assert_eq!(None, SubscriptIv::of(Val(203), &scope));
    }

    /// 🎯 197/384 — ⛔ `llvm_unreachable("only loops with constant bounds are supported")`.
    ///
    /// `affine.for %i = 0 to %extent` — `ub_map.isSingleConstant()` is false, and the reference has
    /// no path out of the `else`.
    #[test]
    fn a_symbolic_loop_bound_has_no_range() {
        let scope = vec![DfirOp::Affine(affine::Op::For {
            iv: Val(104),
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Val(Val(9)),
            carried: Vec::new(),
            body: Vec::new(),
            dbg_name: None,
        })];

        assert_eq!(None, SubscriptIv::of(Val(104), &scope));
    }

    /// 🎯 197/384 — ⛔⛔ ONE BAD SUBSCRIPT COSTS THE WHOLE LIST, BECAUSE THE POSITIONS ARE READ BY
    /// NUMBER.
    ///
    /// `indices_ranges[dim]` (`:325-326`) and `indices_ranges[sym_idx]` (`:99-104`) index this list
    /// by the subscript's dimension, so a list with a hole compacted out of it describes a different
    /// program. `all_of` declines rather than skipping.
    #[test]
    fn a_list_with_one_unusable_subscript_yields_no_list_at_all() {
        let scope = vendor_loop_nest();

        assert_eq!(None, SubscriptIv::all_of(&[Val(101), Val(7)], &scope));
        assert_eq!(None, SubscriptIv::all_of(&[Val(7), Val(102)], &scope));
    }

    /// `affine_map<(d0, d1) -> (0, d0 * 3, d1 * 2 + 2)>` — the vendor's own subscripts for the
    /// hyper-rectangular load, after `replaceConstOpsInSubscriptsMap` has folded `%c0` and `%c2` in:
    /// `%mem_view[%c0, %arg1 * 3, %arg2 * 2 + %c2]`
    /// (`dcc/test/Transform/TransformPagedMemView/paged_mem_view_loads.mlir:297`).
    fn vendor_subscripts() -> AffineMap {
        AffineMap {
            dims: 2,
            syms: 0,
            results: vec![
                AffineExpr::Const(0),
                AffineExpr::dim(0).times(3),
                AffineExpr::dim(1).times(2).plus(AffineExpr::Const(2)),
            ],
        }
    }

    /// 🎯 198/384 — THE VENDOR'S FIRST PAGE: ONE PINNED ITERATOR, ONE SUB-RANGE, THREE NESTED
    /// CONDITIONALS.
    ///
    /// `page0` holds `d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 4 >= 0` (`:291`), so `arg1 * 3` inside
    /// `[0, 1]` pins `arg1` to 0 and `arg2 * 2 + 2` inside `[0, 4]` selects `arg2` over `[0, 1]` out
    /// of its whole `[0, 3]`. ⛔ THE EMISSION IS THE ASSERTION — constant, `cmpi eq`, `scf.if`, then
    /// the two-sided form inside it, which is `:45-53` of the expectation value for value.
    #[test]
    fn the_vendors_first_page_pins_one_iterator_and_bounds_the_other() {
        let mut vals = Values::default();
        let arg1 = vals.mint();
        let arg2 = vals.mint();
        let dims = [
            SubscriptDim {
                index: arg1,
                selected: IvRange { lb: 0, ub: 0 },
                whole: IvRange { lb: 0, ub: 1 },
            },
            SubscriptDim {
                index: arg2,
                selected: IvRange { lb: 0, ub: 1 },
                whole: IvRange { lb: 0, ub: 3 },
            },
        ];

        let rewritten = create_conditions_for_hyper_rect_subscripts(
            &mut vals,
            &[InsertRef::MemOp],
            &vendor_subscripts(),
            &dims,
        );

        let load = vals.mint();
        assert_eq!(
            text(
                &rewritten.insert_refs[0].wrap(vec![DfirOp::Arith(arith::Op::Constant {
                    result: load,
                    value: 7,
                })])
            ),
            "\
%2 = arith.constant 0 : index
%3 = arith.cmpi eq, %0, %2 : index
scf.if %3 {
  %4 = arith.constant 0 : index
  %5 = arith.cmpi sge, %1, %4 : index
  scf.if %5 {
    %6 = arith.constant 1 : index
    %7 = arith.cmpi sle, %1, %6 : index
    scf.if %7 {
      %8 = arith.constant 7 : index
    }
  }
}
"
        );

        // `[0, 0, %arg2 * 2 + 2]` (`:56`): the pinned iterator became a constant, `0 * 3` FOLDED to
        // `0`, and the surviving one is renumbered to `d0`.
        assert_eq!(
            AffineMap {
                dims: 1,
                syms: 0,
                results: vec![
                    AffineExpr::Const(0),
                    AffineExpr::Const(0),
                    AffineExpr::dim(0).times(2).plus(AffineExpr::Const(2)),
                ],
            },
            rewritten.subscripts_map
        );
        // And the pinned iterator is no longer an operand of the access.
        assert_eq!(vec![arg2], rewritten.indices);
        assert_eq!(IndexRemoval::Removed, rewritten.removal);
    }

    /// 🎯 198/384 — ⛔⛔ THE SKIP NEEDS **BOTH** ENDS TO MATCH, AND THE VENDOR'S SECOND PAGE IS THE
    /// CASE THAT PROVES IT.
    ///
    /// `page1` puts `arg2 * 2 + 2` inside `[5, 9]`, which selects `arg2` over `[2, 3]` — the UPPER
    /// end is its whole range's upper end and the lower is not. The reference emits both comparisons
    /// (`cmpi sge, %arg2, 2` / `cmpi sle, %arg2, 3`, `:65-69`); a skip test that compared only the
    /// upper bound, or only one end of either, would emit none and let the access read `page0`'s
    /// address for iterations that belong to `page1`.
    #[test]
    fn a_sub_range_sharing_one_end_with_the_whole_still_needs_both_comparisons() {
        let mut vals = Values::default();
        let arg1 = vals.mint();
        let arg2 = vals.mint();
        let dims = [
            SubscriptDim {
                index: arg1,
                selected: IvRange { lb: 0, ub: 0 },
                whole: IvRange { lb: 0, ub: 1 },
            },
            SubscriptDim {
                index: arg2,
                selected: IvRange { lb: 2, ub: 3 },
                whole: IvRange { lb: 0, ub: 3 },
            },
        ];

        let rewritten = create_conditions_for_hyper_rect_subscripts(
            &mut vals,
            &[InsertRef::MemOp],
            &vendor_subscripts(),
            &dims,
        );

        let guards = &rewritten.insert_refs[0].guards;
        assert_eq!(3, guards.len());
        assert_eq!(
            (CmpIPredicate::Eq, 0, arg1),
            (guards[0].predicate, guards[0].bound, guards[0].lhs)
        );
        assert_eq!(
            (CmpIPredicate::Sge, 2, arg2),
            (guards[1].predicate, guards[1].bound, guards[1].lhs)
        );
        assert_eq!(
            (CmpIPredicate::Sle, 3, arg2),
            (guards[2].predicate, guards[2].bound, guards[2].lhs)
        );
    }

    /// 🎯 198/384 — ⛔⛔ A SKIPPED DIMENSION STILL TAKES ITS SLOT IN THE NEW MAP.
    ///
    /// `num_dim_vars++` runs at `:321`, BEFORE the `continue` at `:327`. Here the first iterator
    /// spans its whole range and needs no condition, and the second needs one — so the new map must
    /// still read `d0 * 3` for the first and `d1 * 2 + 2` for the second. Moving the increment after
    /// the skip would produce `(d0) -> (0, d0 * 3, d0 * 2 + 2)`: one iterator standing for two, with
    /// an arity that verifies.
    #[test]
    fn a_dimension_that_needs_no_condition_still_claims_its_dimension_variable() {
        let mut vals = Values::default();
        let arg1 = vals.mint();
        let arg2 = vals.mint();
        let dims = [
            SubscriptDim {
                index: arg1,
                selected: IvRange { lb: 0, ub: 1 },
                whole: IvRange { lb: 0, ub: 1 },
            },
            SubscriptDim {
                index: arg2,
                selected: IvRange { lb: 0, ub: 1 },
                whole: IvRange { lb: 0, ub: 3 },
            },
        ];

        let rewritten = create_conditions_for_hyper_rect_subscripts(
            &mut vals,
            &[InsertRef::MemOp],
            &vendor_subscripts(),
            &dims,
        );

        assert_eq!(vendor_subscripts(), rewritten.subscripts_map);
        // Only the second iterator is guarded, and nothing was removed from the operands.
        assert_eq!(2, rewritten.insert_refs[0].guards.len());
        assert_eq!(vec![arg1, arg2], rewritten.indices);
        assert_eq!(IndexRemoval::Removed, rewritten.removal);
    }

    /// 🎯 198/384 — A PINNED ITERATOR IN A PRODUCT FOLDS TO ONE CONSTANT, WHICH IS THE VENDOR'S
    /// THIRD PAGE.
    ///
    /// `page2` holds `d1 >= 2, -d1 + 3 >= 0` (`:293`), so `arg1 * 3` inside `[2, 3]` pins `arg1` to
    /// 1 and the subscript becomes `1 * 3` = **3**. The printed expectation is `[0, 1, ..]` (`:90`)
    /// because entry 259 then subtracts that page's start element of 2 — so `3` is what this function
    /// is responsible for, and an unfolded `1 * 3` is what it must not leave behind.
    #[test]
    fn a_pinned_iterator_folds_through_its_coefficient() {
        let mut vals = Values::default();
        let arg1 = vals.mint();
        let arg2 = vals.mint();
        let dims = [
            SubscriptDim {
                index: arg1,
                selected: IvRange { lb: 1, ub: 1 },
                whole: IvRange { lb: 0, ub: 1 },
            },
            SubscriptDim {
                index: arg2,
                selected: IvRange { lb: 0, ub: 1 },
                whole: IvRange { lb: 0, ub: 3 },
            },
        ];

        let rewritten = create_conditions_for_hyper_rect_subscripts(
            &mut vals,
            &[InsertRef::MemOp],
            &vendor_subscripts(),
            &dims,
        );

        assert_eq!(
            AffineMap {
                dims: 1,
                syms: 0,
                results: vec![
                    AffineExpr::Const(0),
                    AffineExpr::Const(3),
                    AffineExpr::dim(0).times(2).plus(AffineExpr::Const(2)),
                ],
            },
            rewritten.subscripts_map
        );
        assert_eq!(1, rewritten.insert_refs[0].guards[0].bound);
    }

    /// 🎯 198/384 — ⛔ THE VALUE NUMBERING IS DIMENSION-MAJOR, ACCESS-MINOR.
    ///
    /// `insert_refs = mem_ops_` is a whole composite's memory ops (`:201`), and the inner loop over
    /// them sits INSIDE the dimension loop — so the second access's outer guard is minted before the
    /// first access's inner one. An access-major port would give each nest a contiguous run of
    /// numbers and diff differently against every vendored expectation with two memory ops.
    #[test]
    fn two_accesses_interleave_their_value_numbers_dimension_by_dimension() {
        let mut vals = Values::default();
        let arg1 = vals.mint();
        let arg2 = vals.mint();
        let dims = [
            SubscriptDim {
                index: arg1,
                selected: IvRange { lb: 0, ub: 0 },
                whole: IvRange { lb: 0, ub: 1 },
            },
            SubscriptDim {
                index: arg2,
                selected: IvRange { lb: 0, ub: 1 },
                whole: IvRange { lb: 0, ub: 3 },
            },
        ];

        let rewritten = create_conditions_for_hyper_rect_subscripts(
            &mut vals,
            &[InsertRef::MemOp, InsertRef::MemOp],
            &vendor_subscripts(),
            &dims,
        );

        // Dimension 0 mints for the load (%2, %3) then for the store (%4, %5); dimension 1 then
        // mints four each, load first.
        let first: Vec<Val> = rewritten.insert_refs[0]
            .guards
            .iter()
            .map(|guard| guard.constant)
            .collect();
        let second: Vec<Val> = rewritten.insert_refs[1]
            .guards
            .iter()
            .map(|guard| guard.constant)
            .collect();
        assert_eq!(vec![Val(2), Val(6), Val(8)], first);
        assert_eq!(vec![Val(4), Val(10), Val(12)], second);
    }

    /// 🎯 198/384 — AN ENTRY THAT ARRIVES ALREADY CONDITIONAL KEEPS ITS GUARDS OUTSIDE THE NEW ONES.
    ///
    /// ⭐ `setBuilderToInsertRef` points the builder INTO whatever `insert_refs[i]` holds before the
    /// new condition is created (`:312-313`, `:330-332`), so a nest that is already there is the
    /// outer one. An empty [`Condition`] is what a bare memory op contributes, which is why
    /// [`InsertRef::MemOp`] and a condition with no guards place statements identically.
    #[test]
    fn an_existing_conditional_becomes_the_outer_nest() {
        let mut vals = Values::default();
        let arg1 = vals.mint();
        let outer = create_equality_condition(&mut vals, arg1, 9);
        let dims = [SubscriptDim {
            index: arg1,
            selected: IvRange { lb: 0, ub: 0 },
            whole: IvRange { lb: 0, ub: 1 },
        }];

        let rewritten = create_conditions_for_hyper_rect_subscripts(
            &mut vals,
            &[InsertRef::Conditional(&outer)],
            &vendor_subscripts(),
            &dims,
        );

        let guards = &rewritten.insert_refs[0].guards;
        assert_eq!(2, guards.len());
        assert_eq!(9, guards[0].bound);
        assert_eq!(0, guards[1].bound);
    }

    /// 🎯 198/384 — ⛔ `DT_CHECK_MSG(lb.has_value() && ub.has_value(), ..)`.
    ///
    /// `getConstantBound` answers `std::nullopt` for a variable the constraints leave open, and one
    /// open end is enough: *"expected constant lower and upper bounds"*.
    #[test]
    fn a_dimension_without_two_constant_bounds_is_no_dimension() {
        let whole = IvRange { lb: 0, ub: 3 };

        assert_eq!(None, SubscriptDim::of(Val(1), None, Some(3), whole));
        assert_eq!(None, SubscriptDim::of(Val(1), Some(0), None, whole));
        assert_eq!(None, SubscriptDim::of(Val(1), None, None, whole));
        assert_eq!(
            Some(SubscriptDim {
                index: Val(1),
                selected: IvRange { lb: 0, ub: 3 },
                whole,
            }),
            SubscriptDim::of(Val(1), Some(0), Some(3), whole)
        );
    }

    /// 🎯 198/384 — THE PAGE'S OWN SET ANSWERS THE BOUNDS THE DOOR TAKES.
    ///
    /// ⭐ END TO END OVER THE ISLAND'S OWN TYPES: a page written as an `affine_set` is what
    /// [`crate::islands::dataflow_ir::ty::IntegerSet::constant_bound`] reads, and its two answers are
    /// what [`SubscriptDim::of`] takes — `d1 >= 0, -d1 + 1 >= 0` giving `[0, 1]` is the vendor's
    /// `page0` on its second axis (`:291`).
    #[test]
    fn a_pages_span_is_the_constant_bound_pair_the_door_takes() {
        let set = PageRect {
            spans: vec![
                PageSpan { lo: 0, hi: 63 },
                PageSpan { lo: 0, hi: 1 },
                PageSpan { lo: 0, hi: 4 },
            ],
        }
        .as_integer_set();

        let dim = SubscriptDim::of(
            Val(1),
            set.constant_bound(BoundType::Lb, 1),
            set.constant_bound(BoundType::Ub, 1),
            IvRange { lb: 0, ub: 3 },
        )
        .expect("a hyper-rectangle has constant bounds on every axis");

        assert_eq!(IvRange { lb: 0, ub: 1 }, dim.selected);
    }

    /// 🎯 198/384 — THE SUBSTITUTION FOLDS CONSTANT OVER CONSTANT, AND FLOORS ITS DIVISIONS.
    ///
    /// ⭐ THE RULES ARE [`crate::islands::dataflow_ir::ty::AffineExpr::flatten`]'s: `-3 mod 4` is 1
    /// and `-3 floordiv 4` is -1, because MLIR's `simplifyMod` and `simplifyFloorDiv` are floored.
    ///
    /// ⛔ AND A DIVISOR THAT IS NOT A POSITIVE LITERAL IS LEFT ALONE rather than divided by —
    /// `assert(rhsConst > 0 && "RHS constant has to be positive")`.
    #[test]
    fn the_substitution_folds_the_way_mlirs_constructors_fold() {
        let pinned = [AffineExpr::Const(-3)];

        assert_eq!(
            AffineExpr::Const(1),
            replace_dims(&AffineExpr::dim(0).modulo(4), &pinned)
        );
        assert_eq!(
            AffineExpr::Const(-1),
            replace_dims(&AffineExpr::dim(0).floordiv(4), &pinned)
        );
        assert_eq!(
            AffineExpr::Const(-1),
            replace_dims(
                &AffineExpr::dim(0).times(2).plus(AffineExpr::Const(5)),
                &pinned
            )
        );

        // A zero divisor is not folded, and neither is a symbolic one.
        assert_eq!(
            AffineExpr::Const(-3).floordiv(0),
            replace_dims(&AffineExpr::dim(0).floordiv(0), &pinned)
        );
        assert_eq!(
            AffineExpr::Mod(
                Box::new(AffineExpr::Const(-3)),
                Box::new(AffineExpr::sym(0))
            ),
            replace_dims(
                &AffineExpr::Mod(Box::new(AffineExpr::dim(0)), Box::new(AffineExpr::sym(0))),
                &pinned
            )
        );
    }

    /// 🎯 199/384 — A CONSTANT RESULT CONSUMES NO ITERATOR ARGUMENT.
    ///
    /// ⛔ `arg_idx` IS THE ASSERTION. It advances only past a result that got a condition, so the
    /// pinned middle result must be compared against `iter_args[0]` and the ranged one against
    /// `iter_args[1]`. A port that indexed `iter_args[res]` would guard the wrong subscript with the
    /// right bound, and every op it emits verifies.
    #[test]
    fn a_constant_result_consumes_no_iterator_argument() {
        let mut vals = Values::default();
        let subscripts_map = AffineMap {
            dims: 2,
            syms: 0,
            results: vec![AffineExpr::Const(0), AffineExpr::dim(0), AffineExpr::dim(1)],
        };
        // One `getConstantBound` pair per RESULT — the first belongs to the constant and is skipped.
        let page_set_bounds = [
            IvRange { lb: 7, ub: 7 },
            IvRange { lb: 2, ub: 2 },
            IvRange { lb: 0, ub: 3 },
        ];

        let nests = create_conditions_for_non_hyper_rect_subscripts(
            &mut vals,
            &[InsertRef::MemOp],
            &subscripts_map,
            &[Val(50), Val(51)],
            &page_set_bounds,
        );

        let nest = nests.first().expect("one insert reference in, one out");
        assert_eq!(
            vec![
                (Val(50), CmpIPredicate::Eq, 2),
                (Val(51), CmpIPredicate::Sge, 0),
                (Val(51), CmpIPredicate::Sle, 3),
            ],
            nest.guards
                .iter()
                .map(|guard| (guard.lhs, guard.predicate, guard.bound))
                .collect::<Vec<_>>()
        );
    }

    /// 🎯 200/384 — ONLY THE REMAPPED INDEX MOVES, AND THE VIEW IS RE-CAST.
    ///
    /// ⛔ THE UNTOUCHED INDEX IS THE ASSERTION: `lookupOrDefault` would answer `Val(21)` for it too,
    /// so the two lookups only differ where a caller can see it — an iterator that was not re-cloned.
    #[test]
    fn only_the_remapped_index_moves_and_the_view_follows_the_clone() {
        let scope = vec![paged_view(VIEW), paged_view(Val(40))];
        let mut info = TpmvInfo::new(
            PagedMemViewHandle::of(&scope[0]).expect("a paged view"),
            AffineMap {
                dims: 2,
                syms: 0,
                results: vec![AffineExpr::dim(0), AffineExpr::dim(1)],
            },
            MemoryOperandIndex::DirSrc,
        );
        info.indices = vec![Val(20), Val(21)];

        let mut ir_map = ValueMapping::new();
        ir_map.map(Val(20), Val(120));
        ir_map.map(VIEW, Val(40));

        update_tpmv_info(&mut info, &ir_map, &scope);

        assert_eq!(vec![Val(120), Val(21)], info.indices);
        assert_eq!(Some(Val(40)), info.paged_mem_view.map(|view| view.result));
    }

    /// 🎯 201/384 — OUTERMOST FIRST, WHATEVER ORDER THE INDICES ARRIVE IN.
    ///
    /// ⛔ THE INPUT IS THE VENDOR'S NEST WITH ITS ITERATORS REVERSED, so an implementation that
    /// returned `0..n` unsorted — or sorted innermost first, which `isProperAncestor` reads like if
    /// its arguments are swapped — answers differently. The third index is bound by no block.
    #[test]
    fn the_iterators_sort_outermost_first() {
        let scope = vendor_loop_nest();

        assert_eq!(
            vec![1, 0, 2],
            set_loop_iterator_order(&[Val(102), Val(101), Val(9)], &scope)
        );
    }

    /// 🎯 202/384 — ONE `TPMVInfo`, CARRYING THE VIEW BEHIND `getMemRef()` AND THE ACCESS ITSELF.
    ///
    /// ⛔ THE MAP AND THE OPERANDS ARE ONE THING IN THIS ISLAND AND TWO IN THE REFERENCE
    /// (`getAffineMapAttr()`, `getMapIndices()`), so the split is what is checked: three subscripts
    /// give a three-dimensional identity map and three operands, in order.
    #[test]
    fn initialize_records_the_view_and_the_split_access() {
        let scope = vec![paged_view(VIEW), vector_load(Val(30))];
        let mut tpmv = TpmvVectorLoad::new(&scope[1], DfirUnit::Lxlu);

        tpmv.initialize(&scope);

        let info = tpmv
            .vector
            .base
            .tpmv_info
            .first()
            .expect("one memory operand, one entry");
        assert_eq!(Some(VIEW), info.paged_mem_view.map(|view| view.result));
        assert_eq!(
            AffineMap {
                dims: 3,
                syms: 0,
                results: vec![AffineExpr::dim(0), AffineExpr::dim(1), AffineExpr::dim(2)],
            },
            info.subscripts_map
        );
        assert_eq!(vec![Val(1), Val(20), Val(21)], info.indices);
        assert_eq!(MemoryOperandIndex::DirSrc, info.mem_index);
    }

    /// A `>= 0` inequality for each end of each iterator's range, and nothing at all when the page's
    /// constraints carry no symbol to bound.
    #[test]
    fn each_iterator_symbol_gains_its_two_bounds() {
        let mut page_sel = IntegerSet {
            dims: 0,
            symbols: 2,
            constraints: vec![],
        };
        add_constraints_for_iv_ranges(
            &mut page_sel,
            &[IvRange { lb: 0, ub: 7 }, IvRange { lb: 2, ub: 3 }],
        );
        assert_eq!(
            vec![
                // `s0 - 0` is `s0`, the `+ 0` folding away as it does in MLIR.
                AffineExpr::sym(0),
                AffineExpr::sym(0).times(-1).plus(AffineExpr::Const(7)),
                AffineExpr::sym(1).plus(AffineExpr::Const(-2)),
                AffineExpr::sym(1).times(-1).plus(AffineExpr::Const(3)),
            ],
            page_sel
                .constraints
                .iter()
                .map(|constraint| constraint.expr.clone())
                .collect::<Vec<_>>()
        );
        assert!(page_sel.constraints.iter().all(|c| !c.is_equality));

        // `if (num_syms == 0) return;` — a constant subscript has no iterator to bound.
        let mut no_syms = IntegerSet {
            dims: 0,
            symbols: 0,
            constraints: vec![],
        };
        add_constraints_for_iv_ranges(&mut no_syms, &[IvRange { lb: 0, ub: 7 }]);
        assert!(no_syms.constraints.is_empty());
    }

    /// The subscripts, rebased on the page's first element — and a start element of zero leaves its
    /// subscript exactly as it was.
    #[test]
    fn the_subscripts_are_rebased_on_the_pages_start_elements() {
        assert_eq!(
            AffineMap {
                dims: 2,
                syms: 0,
                results: vec![
                    AffineExpr::dim(0).plus(AffineExpr::Const(3)),
                    AffineExpr::dim(1),
                ],
            },
            create_new_subscripts_from_start_elements(
                &AffineMap {
                    dims: 2,
                    syms: 0,
                    results: vec![
                        AffineExpr::dim(0).plus(AffineExpr::Const(5)),
                        AffineExpr::dim(1),
                    ],
                },
                &[StartElement(2), StartElement(0)]
            )
        );
    }

    /// Two pages whose rows differ in one inequality and one equality, the two sets listing their
    /// constraints in DIFFERENT orders — which is what the reference's kind-by-kind walk pairs.
    #[test]
    fn only_the_symbols_of_a_moved_constraint_select_the_page() {
        let ineq = |expr: AffineExpr| Constraint {
            expr,
            is_equality: false,
        };
        let eq = |expr: AffineExpr| Constraint {
            expr,
            is_equality: true,
        };
        let page_sel = IntegerSet {
            dims: 0,
            symbols: 3,
            constraints: vec![
                eq(AffineExpr::sym(1).plus(AffineExpr::Const(-2))),
                ineq(AffineExpr::sym(0).plus(AffineExpr::Const(-4))),
                ineq(AffineExpr::sym(2)),
            ],
        };
        let compare = IntegerSet {
            dims: 0,
            symbols: 3,
            constraints: vec![
                ineq(AffineExpr::sym(0).plus(AffineExpr::Const(-8))),
                ineq(AffineExpr::sym(2)),
                eq(AffineExpr::sym(1).plus(AffineExpr::Const(-3))),
            ],
        };

        let mut page_dependent_time_syms = BTreeSet::new();
        gather_page_dependent_dims_for_page(&page_sel, &compare, &mut page_dependent_time_syms);
        // `s0`'s inequality and `s1`'s equality moved; `s2`'s inequality did not.
        assert_eq!(
            BTreeSet::from([PageSelSym(0), PageSelSym(1)]),
            page_dependent_time_syms
        );
    }

    /// 🎯 294/384 — A PAGE THE SUBSCRIPTS REACH, AND THE THREE WAYS ONE IS REFUSED.
    #[test]
    fn a_page_is_valid_where_both_the_subscripts_and_their_ranges_reach_it() {
        let ineq = |expr: AffineExpr| Constraint {
            expr,
            is_equality: false,
        };
        // The page rectangle — `d0` in [0, 63] and `d1` in [0, 127].
        let page_rect = |extra: Vec<Constraint>| IntegerSet {
            dims: 2,
            symbols: 0,
            constraints: [
                vec![
                    ineq(AffineExpr::dim(0)),
                    ineq(AffineExpr::dim(0).scaled(-1).plus(AffineExpr::Const(63))),
                    ineq(AffineExpr::dim(1)),
                    ineq(AffineExpr::dim(1).scaled(-1).plus(AffineExpr::Const(127))),
                ],
                extra,
            ]
            .concat(),
        };
        let info = |lb, ub| TpmvInfo {
            paged_mem_view: None,
            subscripts_map: AffineMap::identity(1),
            indices: Vec::new(),
            indices_ranges: vec![IvRange { lb, ub }],
            conditional_iter_args: Vec::new(),
            mem_index: MemoryOperandIndex::DirSrc,
        };
        // `(s0) -> (s0, 64)` — the access walks the first span and sits inside the second.
        let reaching = AffineMap {
            dims: 0,
            syms: 1,
            results: vec![AffineExpr::sym(0), AffineExpr::Const(64)],
        };
        // Only a valid page has rows at all; the refusals are asserted on the variant below.
        let rows = |validity: PageValidity| match validity {
            PageValidity::Valid(set) => FlatConstraints::from_integer_set(&set).inequalities,
            _ => Vec::new(),
        };

        // Both page bounds, in order, then the iterator's own two — one symbol column and a constant.
        assert_eq!(
            vec![
                vec![1, 0],
                vec![-1, 63],
                vec![0, 64],
                vec![0, 63],
                vec![1, 0],
                vec![-1, 31],
            ],
            rows(get_page_validity(
                &info(0, 31),
                &reaching,
                &page_rect(Vec::new())
            ))
        );

        // `(s0) -> (s0, s0 * 2 + 200)` leaves the second span behind whatever `s0` is.
        let past_the_page = AffineMap {
            dims: 0,
            syms: 1,
            results: vec![
                AffineExpr::sym(0),
                AffineExpr::sym(0).scaled(2).plus(AffineExpr::Const(200)),
            ],
        };
        assert_eq!(
            PageValidity::SubscriptsDoNotIntersectThePage,
            get_page_validity(&info(0, 31), &past_the_page, &page_rect(Vec::new()))
        );

        // The subscript reaches the page, but never while the iterator is in [100, 200].
        assert_eq!(
            PageValidity::IvBoundsDoNotIntersectThePage,
            get_page_validity(&info(100, 200), &reaching, &page_rect(Vec::new()))
        );

        // `-d0 - d1 + 63 >= 0` names two dimensions in one row, so the page is no rectangle.
        let diagonal = vec![ineq(
            AffineExpr::dim(0)
                .scaled(-1)
                .plus(AffineExpr::dim(1).scaled(-1))
                .plus(AffineExpr::Const(63)),
        )];
        assert_eq!(
            PageValidity::PageIsNotHyperRectangular,
            get_page_validity(&info(0, 31), &reaching, &page_rect(diagonal))
        );
    }

    /// 🎯 295/384 — THREE LOOPS, ONE ACCUMULATOR, AND THE COEFFICIENTS READ THE WAY `:481` READS THEM.
    /// ⚠️ `%0`, `%9` and `%16` ARE ENTRY 264'S FILL CONSTANTS, left dead by the `setOperand` that
    /// re-initialises each added iter_arg (`:461-462`, `:468-469`) — the reference leaves them too.
    #[test]
    fn every_loop_of_the_nest_adds_its_own_coefficient_to_the_carried_subscript() {
        let counted = |iv: Val, hi: i64, body: Vec<DfirOp>| {
            DfirOp::Affine(affine::Op::For {
                iv,
                lo: affine::Bound::Const(0),
                hi: affine::Bound::Const(hi),
                carried: Vec::new(),
                body,
                dbg_name: None,
            })
        };
        let terminator = || {
            DfirOp::Affine(affine::Op::Yield {
                operands: Vec::new(),
            })
        };
        let mut scope = vec![counted(
            Val(100),
            4,
            vec![
                counted(
                    Val(101),
                    8,
                    vec![counted(Val(102), 16, vec![terminator()]), terminator()],
                ),
                terminator(),
            ],
        )];
        // `(d0, d1, d2) -> (d0 * 10 + d1 * 100 + d2 * 1000 + 5, 7)`, whose dims are `indices`' own
        // order — mid, inner, outer. ⭐ THE LOOP ORDER IS THEREFORE [2, 0, 1], WHICH IS NOT ITS OWN
        // INVERSE: the outermost loop must add 1000, and the reference's scatter would add 100 there.
        let subscripts_map = AffineMap {
            dims: 3,
            syms: 0,
            results: vec![
                AffineExpr::dim(0)
                    .scaled(10)
                    .plus(AffineExpr::dim(1).scaled(100))
                    .plus(AffineExpr::dim(2).scaled(1000))
                    .plus(AffineExpr::Const(5)),
                AffineExpr::Const(7),
            ],
        };
        let mut vals = Values::default();

        let created = create_iter_args_for_conditionals(
            &mut vals,
            &mut scope,
            &subscripts_map,
            &[Val(101), Val(102), Val(100)],
        );

        // The INNERMOST arg of the one non-constant result, and every index its new induction variable.
        assert_eq!(vec![Val(19)], created.conditional_iter_args);
        assert_eq!(vec![Val(11), Val(18), Val(2)], created.indices);
        assert_eq!(3, created.ir_maps.len());
        assert_eq!(
            "\
%0 = arith.constant 0 : index
%6 = arith.constant 5 : index
%1 = affine.for %2 = 0 to 4 iter_args(%3 = %6) -> (index) {
  %9 = arith.constant 0 : index
  %10 = affine.for %11 = 0 to 8 iter_args(%12 = %3) -> (index) {
    %16 = arith.constant 0 : index
    %17 = affine.for %18 = 0 to 16 iter_args(%19 = %12) -> (index) {
      %20 = arith.constant 100 : index
      %21 = arith.addi %19, %20 : index
      affine.yield %21 : index
    }
    %14 = arith.constant 10 : index
    %15 = arith.addi %12, %14 : index
    affine.yield %15 : index
  }
  %7 = arith.constant 1000 : index
  %8 = arith.addi %3, %7 : index
  affine.yield %8 : index
}
",
            text(&scope)
        );
    }

    /// 🎯 309/384 — THE VENDOR'S FIRST PAGE, END TO END: THREE GUARDS, A PLAIN VIEW OVER THE PAGE'S
    /// OWN START, AND THE ACCESS AND ITS CHAIN REBUILT INSIDE THEM.
    ///
    /// `paged_mem_view_loads.mlir:288-296` in, `:44-61` out, value for value.
    ///
    /// ⛔ THE BOUND DOOR IS THE DEFECT THIS CATCHES: entry 294 hands over a system with ZERO dims and
    /// `[s0, s1]`, so reading column 0 as `d0` finds no bound and the page comes back
    /// [`ValidPage::SubscriptBoundIsNotConstant`] with three conditions never emitted.
    ///
    /// ⚠️ THREE PLACES THE PRINTED TEXT IS THE ISLAND'S AND NOT THE VENDOR'S, none of them this
    /// entry's: `memref<?x64x4xf16>` is written `8x64x4` ([`MemRef`] has no dynamic extent);
    /// `right_shift = true` prints although the `.td` defaults it (`vectorchain.rs:1255`); and
    /// `load_set` is DERIVED with the contiguous axis LAST, where the vendor CARRIES one whose
    /// contiguous axis is `d0` (`#ATTR_7`, `:24`). The guards, the `addi`, the view and the
    /// subscripts — everything `constructValidPage` decides — are the vendor's value for value.
    #[test]
    fn the_vendors_first_page_is_guarded_and_rebuilt_over_its_own_start() {
        let mut vals = Values::default();
        let lx = vals.mint();
        let c0 = vals.mint();
        let c16 = vals.mint();
        let arg1 = vals.mint();
        let arg2 = vals.mint();
        let lxlu0 = vals.mint();
        let sfp0 = vals.mint();
        let mem_view = vals.mint();
        let loaded = vals.mint();
        let rotated = vals.mint();
        let (to, _) = Link::<LxluUnit, SfpUnit>::between(lxlu0, sfp0).ends();

        // `affine_map<(d0, d1, d2) -> (d2 * 256 + d1 * 64 + d0)>` (`:289`) over `memref<?x64x4xf16>`.
        let view_ty = MemRef {
            shape: vec![8, 64, 4],
            elem: ElemType::F16,
        };
        let vendor_layout = AffineMap {
            dims: 3,
            syms: 0,
            results: vec![
                AffineExpr::dim(2)
                    .times(256)
                    .plus(AffineExpr::dim(1).times(64))
                    .plus(AffineExpr::dim(0)),
            ],
        };
        // `page0` and `page1` (`:291-292`); only the first is constructed here.
        let pages = vec![
            Page {
                idx_set: PageRect {
                    spans: vec![
                        PageSpan { lo: 0, hi: 63 },
                        PageSpan { lo: 0, hi: 1 },
                        PageSpan { lo: 0, hi: 4 },
                    ],
                },
                start_addr: c0,
            },
            Page {
                idx_set: PageRect {
                    spans: vec![
                        PageSpan { lo: 0, hi: 63 },
                        PageSpan { lo: 0, hi: 1 },
                        PageSpan { lo: 5, hi: 9 },
                    ],
                },
                start_addr: c16,
            },
        ];
        let view = PagedMemView {
            result: mem_view,
            unit: lx,
            start_addr: c0,
            pages,
            layout: vendor_layout,
            ty: view_ty.clone(),
        };
        // `%load = agen.vector_load %mem_view[%c0, %arg1 * 3, %arg2 * 2 + %c2]`, its rotate and its
        // send (`:297-299`) — the whole chain the page's guards have to end up around.
        let scope = vec![
            DfirOp::Dataflow(dataflow::Op::GetPagedLogicalMemoryView(Box::new(view))),
            DfirOp::Agen(agen::Op::VectorLoad {
                dbg_name: None,
                result: loaded,
                view: mem_view,
                indices: vec![
                    Index::Const(0),
                    Index::Strided(vec![(arg1, 3)], 0),
                    Index::Strided(vec![(arg2, 2)], 2),
                ],
                view_ty: view_ty.clone(),
                ty: LANES,
                multicast_info: None,
            }),
            DfirOp::VectorChain(vectorchain::Op::Rotate {
                result: rotated,
                input: loaded,
                position: c16,
                right_shift: true,
                input_ty: LANES,
                ty: LANES,
            }),
            DfirOp::Dataflow(dataflow::Op::Send {
                to,
                data: rotated,
                ty: LANES,
            }),
        ];

        // `affine.for %arg1 = 0 to 2` and `%arg2 = 0 to 4` (`:285-286`) are the two iterator ranges.
        let access = PagedAccess {
            mem_ops: vec![loaded],
            paged_mem_view: mem_view,
            subscripts_map: vendor_subscripts(),
            indices: vec![arg1, arg2],
            indices_ranges: vec![IvRange { lb: 0, ub: 1 }, IvRange { lb: 0, ub: 3 }],
            conditional_iter_args: Vec::new(),
        };
        let page = PagedMemViewHandle::of(&scope[0])
            .expect("the first op is the paged view")
            .view
            .pages[0]
            .clone();
        let subscripts_map_sym = replace_dims_in_map_with_syms(&access.subscripts_map);
        let info = TpmvInfo {
            paged_mem_view: None,
            subscripts_map: access.subscripts_map.clone(),
            indices: access.indices.clone(),
            indices_ranges: access.indices_ranges.clone(),
            conditional_iter_args: Vec::new(),
            mem_index: MemoryOperandIndex::DirSrc,
        };
        let PageValidity::Valid(page_sel_constraints) =
            get_page_validity(&info, &subscripts_map_sym, &page.idx_set.as_integer_set())
        else {
            panic!("the vendor's `arg1 * 3` and `arg2 * 2 + 2` both reach page0");
        };

        let mut emitted = scope.clone();
        let ValidPage::Constructed(built) = construct_valid_page(
            &mut vals,
            &mut emitted,
            &access,
            &page,
            &page_sel_constraints,
        ) else {
            panic!("page0 bounds both iterators with constants");
        };

        assert_eq!(
            text(&built.placed[0]),
            "\
%10 = arith.constant 0 : index
%11 = arith.cmpi eq, %3, %10 : index
scf.if %11 {
  %12 = arith.constant 0 : index
  %13 = arith.cmpi sge, %4, %12 : index
  scf.if %13 {
    %14 = arith.constant 1 : index
    %15 = arith.cmpi sle, %4, %14 : index
    scf.if %15 {
      %16 = arith.addi %1, %1 : index
      %17 = dataflow.get_logical_memory_view %0, %16 {layout_map = affine_map<(d0, d1, d2) -> \
             (d2 * 256 + d1 * 64 + d0)>} : index, index, memref<8x64x4xf16>
      %18 = agen.vector_load %17[0, 0, %4 * 2 + 2] {dbgName = \"\", load_order = \
             affine_map<(d0, d1, d2) -> (d0, d1, d2)>, load_set = affine_set<(d0, d1, d2) : \
             (d0 == 0, d1 == 0, d2 >= 0, -d2 + 63 >= 0)>} : memref<8x64x4xf16>, vector<64xf16>
      %19 = vectorchain.rotate %18, %2 {right_shift = true} : vector<64xf16>, index, vector<64xf16>
      dataflow.send %6, %19 : vector<64xf16>
    }
  }
}
"
        );
        // The pinned iterator is gone from the access, and nothing took the iter_args arm.
        assert_eq!(vec![loaded], built.access.mem_ops);
        assert_eq!(Val(18), built.new_mem_ops[0].result);
        assert!(built.access.conditional_iter_args.is_empty());
        assert!(built.ir_maps.is_empty());
    }

    /// 🎯 310/384 — THE VENDOR'S SIX-PAGE VIEW IS REACHED ON FOUR PAGES, AND BOTH ITERATORS SELECT.
    ///
    /// `paged_mem_view_loads.mlir:288-296` in, and the expectation pins the COUNT: the first load is
    /// rebuilt exactly four times, over start addends `%c0`, `%c1000`, `%c2000` and `%c3000`
    /// (`:44-180`) — `%arg1 * 3` is 0 or 3 and never lands in page 4's or page 5's `d1` span `[4, 5]`.
    ///
    /// ⚠️ NOT THE COMPOSITE CASE (`:316-341`) THOUGH THIS IS THE COMPOSITE CLASS: its four pages are
    /// selected by a TIME iterator that entry 326 substitutes into the subscripts, and
    /// `agen.composite_load` has no island op. The analysis being run is the base's either way.
    #[test]
    fn the_vendors_six_pages_are_reached_four_times_and_both_iterators_bear_on_which() {
        let mut vals = Values::default();
        let lx = vals.mint();
        let mem_view = vals.mint();
        let arg1 = vals.mint();
        let arg2 = vals.mint();
        // `page0`..`page5` (`:291-296`) — `d1` over three spans of two, `d2` over two spans of five.
        let pages: Vec<Page> = [
            ([0, 1], [0, 4]),
            ([0, 1], [5, 9]),
            ([2, 3], [0, 4]),
            ([2, 3], [5, 9]),
            ([4, 5], [0, 4]),
            ([4, 5], [5, 9]),
        ]
        .into_iter()
        .map(|([lo1, hi1], [lo2, hi2])| Page {
            idx_set: PageRect {
                spans: vec![
                    PageSpan { lo: 0, hi: 63 },
                    PageSpan { lo: lo1, hi: hi1 },
                    PageSpan { lo: lo2, hi: hi2 },
                ],
            },
            start_addr: vals.mint(),
        })
        .collect();
        let scope = vec![DfirOp::Dataflow(dataflow::Op::GetPagedLogicalMemoryView(
            Box::new(PagedMemView {
                result: mem_view,
                unit: lx,
                start_addr: lx,
                pages,
                layout: AffineMap::identity(3),
                ty: MemRef {
                    shape: vec![8, 64, 4],
                    elem: ElemType::F16,
                },
            }),
        ))];
        let info = TpmvInfo {
            paged_mem_view: PagedMemViewHandle::of(&scope[0]),
            subscripts_map: vendor_subscripts(),
            indices: vec![arg1, arg2],
            indices_ranges: vec![IvRange { lb: 0, ub: 1 }, IvRange { lb: 0, ub: 3 }],
            conditional_iter_args: Vec::new(),
            mem_index: MemoryOperandIndex::DirSrc,
        };
        // `analyzeValidPages` never reads `mem_op_`, so the view op stands in for the absent
        // `agen.composite_load`.
        let mut composite = TpmvComposite::new(&scope[0], DfirUnit::Lxlu);

        let reached = composite
            .analyze_valid_pages(&info, &replace_dims_in_map_with_syms(&info.subscripts_map));

        assert_eq!(
            ValidPages::Found(NonZeroU32::new(4).expect("the vendor rebuilds the load four times")),
            reached
        );
        // `d1`'s span moved for pages 2 and 3 and `d2`'s for pages 1 and 3, so both select.
        assert_eq!(
            BTreeSet::from([PageSelSym(0), PageSelSym(1)]),
            composite.page_dependent_time_syms
        );
    }

    // ── 324/384, 325/384 and 326/384 ─────────────────────────────────────────────────────────────

    /// `paged_mem_view_loads.mlir:291-296` — the six pages, `d1` over three spans of two and `d2`
    /// over two spans of five, each with its own start address.
    fn vendor_six_pages(vals: &mut Values) -> Vec<Page> {
        [
            ([0, 1], [0, 4]),
            ([0, 1], [5, 9]),
            ([2, 3], [0, 4]),
            ([2, 3], [5, 9]),
            ([4, 5], [0, 4]),
            ([4, 5], [5, 9]),
        ]
        .into_iter()
        .map(|([lo1, hi1], [lo2, hi2])| Page {
            idx_set: PageRect {
                spans: vec![
                    PageSpan { lo: 0, hi: 63 },
                    PageSpan { lo: lo1, hi: hi1 },
                    PageSpan { lo: lo2, hi: hi2 },
                ],
            },
            start_addr: vals.mint(),
        })
        .collect()
    }

    /// 🎯 324/384 — ⭐⭐ IBM'S SIX-PAGE VIEW WALKED END TO END: THE FOUR PAGES THE SUBSCRIPTS REACH
    /// ARE EACH BUILT OVER THEIR OWN START ADDRESS, AND PAGES 4 AND 5 EMIT NOTHING.
    ///
    /// `paged_mem_view_loads.mlir:288-299` in, `:44-180` out — four copies of the load chain over the
    /// addends `%c0`, `%c1000`, `%c2000` and `%c3000`, in `getIdxSets()` order.
    ///
    /// ⛔ AND EVERY PAGE READS THE PREVIOUS PAGE'S `info`: `access` is threaded page to page, which is
    /// what makes `constructValidPage`'s re-read of `mem_ops_` correct. Handing each page the ORIGINAL
    /// `access` would still emit four chains here — the difference only shows once some page takes the
    /// iter_args arm — so the count is asserted alongside the addends, not instead of them.
    #[test]
    fn the_vendors_four_reached_pages_are_each_built_over_their_own_start() {
        let mut vals = Values::default();
        let lx = vals.mint();
        let c0 = vals.mint();
        let c16 = vals.mint();
        let arg1 = vals.mint();
        let arg2 = vals.mint();
        let lxlu0 = vals.mint();
        let sfp0 = vals.mint();
        let mem_view = vals.mint();
        let loaded = vals.mint();
        let rotated = vals.mint();
        let pages = vendor_six_pages(&mut vals);
        let starts: Vec<Val> = pages.iter().map(|page| page.start_addr).collect();
        let (to, _) = Link::<LxluUnit, SfpUnit>::between(lxlu0, sfp0).ends();

        let view_ty = MemRef {
            shape: vec![8, 64, 4],
            elem: ElemType::F16,
        };
        // `affine_map<(d0, d1, d2) -> (d2 * 256 + d1 * 64 + d0)>` (`:289`).
        let vendor_layout = AffineMap {
            dims: 3,
            syms: 0,
            results: vec![
                AffineExpr::dim(2)
                    .times(256)
                    .plus(AffineExpr::dim(1).times(64))
                    .plus(AffineExpr::dim(0)),
            ],
        };
        let scope = vec![
            DfirOp::Dataflow(dataflow::Op::GetPagedLogicalMemoryView(Box::new(
                PagedMemView {
                    result: mem_view,
                    unit: lx,
                    start_addr: c0,
                    pages,
                    layout: vendor_layout,
                    ty: view_ty.clone(),
                },
            ))),
            DfirOp::Agen(agen::Op::VectorLoad {
                dbg_name: None,
                result: loaded,
                view: mem_view,
                indices: vec![
                    Index::Const(0),
                    Index::Strided(vec![(arg1, 3)], 0),
                    Index::Strided(vec![(arg2, 2)], 2),
                ],
                view_ty,
                ty: LANES,
                multicast_info: None,
            }),
            DfirOp::VectorChain(vectorchain::Op::Rotate {
                result: rotated,
                input: loaded,
                position: c16,
                right_shift: true,
                input_ty: LANES,
                ty: LANES,
            }),
            DfirOp::Dataflow(dataflow::Op::Send {
                to,
                data: rotated,
                ty: LANES,
            }),
        ];

        let access = PagedAccess {
            mem_ops: vec![loaded],
            paged_mem_view: mem_view,
            subscripts_map: vendor_subscripts(),
            indices: vec![arg1, arg2],
            indices_ranges: vec![IvRange { lb: 0, ub: 1 }, IvRange { lb: 0, ub: 3 }],
            conditional_iter_args: Vec::new(),
        };
        let info = TpmvInfo {
            paged_mem_view: PagedMemViewHandle::of(&scope[0]),
            subscripts_map: access.subscripts_map.clone(),
            indices: access.indices.clone(),
            indices_ranges: access.indices_ranges.clone(),
            conditional_iter_args: Vec::new(),
            mem_index: MemoryOperandIndex::DirSrc,
        };
        // `info` borrows the view op, so the emission goes into a clone — as entry 309's test does.
        let mut emitted = scope.clone();
        let constructed = analyze_and_construct_valid_pages(
            &mut vals,
            &mut emitted,
            &access,
            &info,
            &replace_dims_in_map_with_syms(&access.subscripts_map),
        );

        let ConstructedPages::Constructed { pages, access } = constructed else {
            panic!("`%arg1 * 3` and `%arg2 * 2 + 2` reach four of the six pages: {constructed:?}")
        };
        assert_eq!(pages.len(), 4, "pages 4 and 5 are skipped, not refused");

        /// Every `arith.addi` a page emitted, guards and all — the page's start addend.
        fn addends(ops: &[DfirOp]) -> Vec<Val> {
            ops.iter()
                .flat_map(|op| match op {
                    DfirOp::Arith(arith::Op::AddI(add)) => vec![add.rhs],
                    DfirOp::Scf(scf::Op::If {
                        body, else_body, ..
                    }) => {
                        let mut found = addends(body);
                        found.extend(addends(else_body));
                        found
                    }
                    _ => Vec::new(),
                })
                .collect()
        }
        for (page, &start) in pages.iter().zip(&starts) {
            assert_eq!(addends(&page.placed[0]), [start]);
            assert_eq!(
                page.new_mem_ops.len(),
                1,
                "one replacement per `mem_ops_` entry"
            );
        }
        // Every page's bounds are constants here, so nothing took the non-hyper-rectangular arm.
        assert!(access.conditional_iter_args.is_empty());
        assert_eq!(access.mem_ops, vec![loaded]);
    }

    /// `vector<128xi8>` — the composite key's own vector, twice the width of [`LANES`].
    const STICK: Vector = Vector {
        len: 128,
        elem: ElemType::Int(8),
    };

    /// `paged_mem_view_loads.mlir:316-341` — the composite load on its four-page view, flat, with the
    /// `%c3` its `time_set`'s `s0` reads and the `%arg9` its subscripts do.
    ///
    /// ⭐ THE FOUR OUTER LOOPS ARE LEFT OUT: only `%arg9` appears in the subscripts, and entry 197
    /// resolves an iterator through the `affine.for` that BINDS it, which needs no enclosing nest.
    fn vendor_composite_load(vals: &mut Values) -> (Vec<DfirOp>, Val, Val) {
        let lx = vals.mint();
        let sfp1 = vals.mint();
        let lxlu1 = vals.mint();
        let base = vals.mint();
        let c3 = vals.mint();
        let mem_view = vals.mint();
        let arg9 = vals.mint();
        let load_iv = vals.mint();
        let starts: Vec<Val> = (0..4).map(|_| vals.mint()).collect();
        let (to, _) = Link::<LxluUnit, SfpUnit>::between(lxlu1, sfp1).ends();

        let ty = MemRef {
            shape: vec![128, 2, 1, 1, 2],
            elem: ElemType::Int(8),
        };
        // `(d0, d1, d2, d3, d4) -> (d4 * 256 + d3 * 256 + d2 * 256 + d1 * 128 + d0)` (`:317`).
        let layout = AffineMap {
            dims: 5,
            syms: 0,
            results: vec![
                AffineExpr::dim(4)
                    .times(256)
                    .plus(AffineExpr::dim(3).times(256))
                    .plus(AffineExpr::dim(2).times(256))
                    .plus(AffineExpr::dim(1).times(128))
                    .plus(AffineExpr::dim(0)),
            ],
        };
        // `page0`..`page3` (`:319-322`) — one per value of `d1`, which is what makes the time
        // dimension that reaches `d1` page-dependent.
        let pages: Vec<Page> = starts
            .iter()
            .enumerate()
            .map(|(n, &start_addr)| Page {
                idx_set: PageRect {
                    spans: vec![
                        PageSpan { lo: 0, hi: 63 },
                        PageSpan {
                            lo: i64::try_from(n).expect("four pages"),
                            hi: i64::try_from(n).expect("four pages"),
                        },
                        PageSpan { lo: 0, hi: 63 },
                        PageSpan { lo: 0, hi: 1 },
                        PageSpan { lo: 0, hi: 7 },
                    ],
                },
                start_addr,
            })
            .collect();
        let spanning = |dim: u32, upper: AffineExpr| {
            vec![
                Constraint {
                    expr: AffineExpr::dim(dim),
                    is_equality: false,
                },
                Constraint {
                    expr: AffineExpr::dim(dim).times(-1).plus(upper),
                    is_equality: false,
                },
            ]
        };
        let pinned = |dim: u32| Constraint {
            expr: AffineExpr::dim(dim),
            is_equality: true,
        };
        let scope = vec![
            DfirOp::Arith(arith::Op::Constant {
                result: c3,
                value: 3,
            }),
            DfirOp::Affine(affine::Op::For {
                iv: arg9,
                lo: affine::Bound::Const(0),
                hi: affine::Bound::Const(56),
                carried: Vec::new(),
                body: Vec::new(),
                dbg_name: None,
            }),
            DfirOp::Dataflow(dataflow::Op::GetPagedLogicalMemoryView(Box::new(
                PagedMemView {
                    result: mem_view,
                    unit: lx,
                    start_addr: base,
                    pages,
                    layout,
                    ty: ty.clone(),
                },
            ))),
            DfirOp::Agen(agen::Op::CompositeLoad(Box::new(agen::CompositeAccess {
                view: mem_view,
                // `%mem_view[%arg9, 0, 0, 0, 0]` (`:323`).
                indices: vec![
                    Index::Val(arg9),
                    Index::Const(0),
                    Index::Const(0),
                    Index::Const(0),
                    Index::Const(0),
                ],
                view_ty: ty,
                load_iv,
                load_iv_ty: STICK,
                // `(d0 >= 0, -d0 + 127 >= 0, d1 == 0, d2 == 0, d3 == 0, d4 == 0)`.
                load_set: IntegerSet {
                    dims: 5,
                    symbols: 0,
                    constraints: spanning(0, AffineExpr::Const(127))
                        .into_iter()
                        .chain((1..5).map(pinned))
                        .collect(),
                },
                load_order: AffineMap::identity(5),
                // `time_symbols(%c3)` — `s0` below is 3, which is `d2`'s width.
                time_symbols: vec![c3],
                time_set: IntegerSet {
                    dims: 5,
                    symbols: 1,
                    constraints: spanning(0, AffineExpr::Const(1))
                        .into_iter()
                        .chain(spanning(1, AffineExpr::Const(0)))
                        .chain(spanning(2, AffineExpr::sym(0).plus(AffineExpr::Const(-1))))
                        .chain(spanning(3, AffineExpr::Const(2)))
                        .chain(spanning(4, AffineExpr::Const(1)))
                        .collect(),
                },
                // `(d0, d4, d2, d3, d1)` (`:336`).
                time_order: AffineMap {
                    dims: 5,
                    syms: 0,
                    results: vec![
                        AffineExpr::dim(0),
                        AffineExpr::dim(4),
                        AffineExpr::dim(2),
                        AffineExpr::dim(3),
                        AffineExpr::dim(1),
                    ],
                },
                // `(0, d2, 0, 0, d4 * 2)` (`:335`).
                time_addr_map: AffineMap {
                    dims: 5,
                    syms: 0,
                    results: vec![
                        AffineExpr::Const(0),
                        AffineExpr::dim(2),
                        AffineExpr::Const(0),
                        AffineExpr::Const(0),
                        AffineExpr::dim(4).times(2),
                    ],
                },
                dbg_name: None,
                body: vec![
                    DfirOp::Dataflow(dataflow::Op::Send {
                        to,
                        data: load_iv,
                        ty: STICK,
                    }),
                    DfirOp::Agen(agen::Op::Yield),
                ],
            }))),
        ];
        (scope, arg9, mem_view)
    }

    /// 🎯 326/384 — ⭐ THE VENDOR'S COMPOSITE LOAD GIVES UP ITS FIVE TIME BOUNDS IN `time_order`'S
    /// ORDER, WITH ITS SYMBOLIC DIMENSION RESOLVED THROUGH `time_symbols(%c3)`.
    ///
    /// `paged_mem_view_loads.mlir:331-337`. The set's own widths are `d0:2 d1:1 d2:s0 d3:3 d4:2`, and
    /// `time_order = (d0, d4, d2, d3, d1)` reorders them to `[2, 2, 3, 3, 1]` — which is where entry
    /// 325's three loops of 2, 2 and 3 come from.
    ///
    /// ⛔ WITHOUT `time_symbols` THIS IS `NonConstantTimeSymbol(d2)` AND THE PASS STOPS: the island's
    /// composite transfers carried no such list until this entry needed one.
    #[test]
    fn the_vendors_composite_load_resolves_its_symbolic_time_bound_through_its_symbols() {
        let mut vals = Values::default();
        let (scope, _, _) = vendor_composite_load(&mut vals);
        let mut load = TpmvCompositeLoad::new(&scope[3], DfirUnit::Lxlu);

        assert_eq!(load.initialize_time(&scope), CompositeTimeInit::Initialized);

        let details = load.composite.access_details.entries();
        assert_eq!(details.len(), 1, "a composite load has one memory operand");
        assert_eq!(
            details[0].time_bounds,
            [
                TimeBound::Steps(2),
                TimeBound::Steps(2),
                TimeBound::Steps(3),
                TimeBound::Steps(3),
                TimeBound::Steps(1)
            ]
        );
        // `time_order_`, `time_set_` and `tpmv_comp_info_` are the op's, unnarrowed until entry 325.
        let DfirOp::Agen(agen::Op::CompositeLoad(access)) = &scope[3] else {
            unreachable!("the fixture's last op is the composite load")
        };
        assert_eq!(load.composite.time_order, Some(access.time_order.clone()));
        assert_eq!(load.composite.time_set, Some(access.time_set.clone()));
        assert_eq!(
            load.composite.tpmv_comp_info,
            vec![TpmvCompositeInfo {
                time_addr_map: access.time_addr_map.clone(),
                access_details: MemoryOperandIndex::DirSrc,
            }]
        );
    }

    /// 🎯 325/384 — ⭐⭐ IBM'S COMPOSITE KEY END TO END: THREE OF ITS FIVE TIME DIMENSIONS BECOME REAL
    /// LOOPS OF 2, 2 AND 3, AND THE LOAD IS REBUILT INSIDE THEM WITH THE NARROWED `time_set`.
    ///
    /// `paged_mem_view_loads.mlir:316-341` in, `:128-131` and `:136-142` out — the three `affine.for`s
    /// and the rebuilt `agen.composite_load` still on the PAGED view, with `#[[$ATTR_9]]` pinning
    /// `d0`, `d4` and `d2` and subscripts `[%arg9, ivs[2], 0, 0, ivs[1] * 2]`.
    ///
    /// ⛔ THE PAGE GUARDS AND THE `0` AT POSITION 1 IN THE CHECK ARE **ENTRY 356's**, not this one's:
    /// `constructValidPage` rebases each page's selecting axis onto its own start, which is why all
    /// three of the vendor's arms print `0` there while this emission still names `ivs[2]`.
    #[test]
    fn the_vendors_composite_key_turns_three_time_dimensions_into_loops() {
        let mut vals = Values::default();
        let (scope, arg9, mem_view) = vendor_composite_load(&mut vals);
        let mut load = TpmvCompositeLoad::new(&scope[3], DfirUnit::Lxlu);
        assert_eq!(load.initialize_time(&scope), CompositeTimeInit::Initialized);

        // `TPMVCompositeLoad::initialize` (`Impl.cpp:1078`) is not in the 384 — it is what fills
        // `tpmv_info_` from the op's own `[%arg9, 0, 0, 0, 0]`, and entry 325 reads it here.
        load.composite.base.tpmv_info.push(TpmvInfo {
            paged_mem_view: PagedMemViewHandle::of(&scope[2]),
            subscripts_map: AffineMap {
                dims: 1,
                syms: 0,
                results: vec![
                    AffineExpr::dim(0),
                    AffineExpr::Const(0),
                    AffineExpr::Const(0),
                    AffineExpr::Const(0),
                    AffineExpr::Const(0),
                ],
            },
            indices: vec![arg9],
            indices_ranges: Vec::new(),
            conditional_iter_args: Vec::new(),
            mem_index: MemoryOperandIndex::DirSrc,
        });

        let transformed = load.composite.transform_time(&mut vals, &scope);
        let TransformedTime::ExplicitLoops(ExplicitTimeNest { nest, ivs }) = transformed else {
            panic!(
                "`d1` is reached by time dimension 2, so three dims become loops: {transformed:?}"
            )
        };
        assert_eq!(ivs.len(), 3);

        let mut level = &nest;
        let mut trips: Vec<affine::Bound> = Vec::new();
        for iv in &ivs {
            let [
                DfirOp::Affine(affine::Op::For {
                    iv: bound_iv,
                    lo,
                    hi,
                    body,
                    ..
                }),
            ] = &level[..]
            else {
                panic!("one `affine.for` per explicit time dimension: {level:?}")
            };
            assert_eq!((bound_iv, lo), (iv, &affine::Bound::Const(0)));
            trips.push(*hi);
            level = body;
        }
        assert_eq!(
            trips,
            [
                affine::Bound::Const(2),
                affine::Bound::Const(2),
                affine::Bound::Const(3)
            ],
            "the first three of `[2, 2, 3, 3, 1]`"
        );

        let [DfirOp::Agen(agen::Op::CompositeLoad(rebuilt))] = &level[..] else {
            panic!("the innermost body holds the rebuilt load: {level:?}")
        };
        assert_eq!(
            rebuilt.view, mem_view,
            "still the PAGED view — entry 356 replaces it"
        );
        assert_eq!(
            rebuilt.indices,
            vec![
                Index::Val(arg9),
                Index::Val(ivs[2]),
                Index::Const(0),
                Index::Const(0),
                Index::Strided(vec![(ivs[1], 2)], 0)
            ],
            "`(0, d2, 0, 0, d4 * 2)` reordered to `(0, d2, 0, 0, d1 * 2)` and concatenated"
        );
        // `#[[$ATTR_9]]` (`:25`) — `d0`, `d4` and `d2` pinned in `time_order` order, their six
        // constraints dropped, `d1`'s and `d3`'s kept.
        let pinned = |dim: u32| Constraint {
            expr: AffineExpr::dim(dim),
            is_equality: true,
        };
        let DfirOp::Agen(agen::Op::CompositeLoad(original)) = &scope[3] else {
            unreachable!("the fixture's last op is the composite load")
        };
        assert_eq!(
            rebuilt.time_set,
            IntegerSet {
                dims: 5,
                symbols: 1,
                constraints: vec![pinned(0), pinned(4), pinned(2)]
                    .into_iter()
                    .chain(original.time_set.constraints[2..4].iter().cloned())
                    .chain(original.time_set.constraints[6..8].iter().cloned())
                    .collect(),
            }
        );
        assert_eq!(load.composite.time_set.as_ref(), Some(&rebuilt.time_set));
        assert_eq!(rebuilt.time_symbols, original.time_symbols);
        assert_eq!(rebuilt.dbg_name, Some(String::new()), "`dbgName = \"\"`");
    }

    // ── 356/384 ──────────────────────────────────────────────────────────────────────────────────

    /// 🎯 356/384 — ⭐⭐ IBM'S PAGED LOAD, DE-PAGED END TO END: the four pages its subscripts reach
    /// each get their own load, the original load and its whole rotate/send tail go on the erase
    /// list, the paged view goes with them, and `mem_ops_` comes back as the FOUR replacements.
    ///
    /// `paged_mem_view_loads.mlir:286-299` in, `:44-180` out.
    ///
    /// ⛔ THE ERASE LIST IS CONSUMER-FIRST AND IT IS NOT JUST THE LOAD: `TPMVVectorLoad` overrides
    /// `eraseMemOpAndUseChain` (entry 129), so a port that called the base's one-op answer here would
    /// leave a `vectorchain.rotate` reading a value nothing defines.
    #[test]
    fn the_vendors_paged_load_is_replaced_by_one_load_per_reached_page() {
        let mut vals = Values::default();
        let lx = vals.mint();
        let c0 = vals.mint();
        let c16 = vals.mint();
        let arg1 = vals.mint();
        let arg2 = vals.mint();
        let lxlu0 = vals.mint();
        let sfp0 = vals.mint();
        let mem_view = vals.mint();
        let loaded = vals.mint();
        let rotated = vals.mint();
        let pages = vendor_six_pages(&mut vals);
        let (to, _) = Link::<LxluUnit, SfpUnit>::between(lxlu0, sfp0).ends();

        let view_ty = MemRef {
            shape: vec![8, 64, 4],
            elem: ElemType::F16,
        };
        // `affine_map<(d0, d1, d2) -> (d2 * 256 + d1 * 64 + d0)>` (`:289`).
        let vendor_layout = AffineMap {
            dims: 3,
            syms: 0,
            results: vec![
                AffineExpr::dim(2)
                    .times(256)
                    .plus(AffineExpr::dim(1).times(64))
                    .plus(AffineExpr::dim(0)),
            ],
        };
        // `%mem_view[%c0, %arg1 * 3, %arg2 * 2 + %c2]` (`:297`) — the view and the access both sit
        // INSIDE the nest, which is what makes `%arg1` and `%arg2` iterators entry 197 accepts.
        let innermost = vec![
            DfirOp::Dataflow(dataflow::Op::GetPagedLogicalMemoryView(Box::new(
                PagedMemView {
                    result: mem_view,
                    unit: lx,
                    start_addr: c0,
                    pages,
                    layout: vendor_layout,
                    ty: view_ty.clone(),
                },
            ))),
            DfirOp::Agen(agen::Op::VectorLoad {
                dbg_name: None,
                result: loaded,
                view: mem_view,
                indices: vec![
                    Index::Const(0),
                    Index::Strided(vec![(arg1, 3)], 0),
                    Index::Strided(vec![(arg2, 2)], 2),
                ],
                view_ty,
                ty: LANES,
                multicast_info: None,
            }),
            DfirOp::VectorChain(vectorchain::Op::Rotate {
                result: rotated,
                input: loaded,
                position: c16,
                right_shift: true,
                input_ty: LANES,
                ty: LANES,
            }),
            DfirOp::Dataflow(dataflow::Op::Send {
                to,
                data: rotated,
                ty: LANES,
            }),
        ];
        // `affine.for %arg1 = 0 to 2 { affine.for %arg2 = 0 to 4 { .. } }` (`:286-287`).
        let scope = vec![DfirOp::Affine(affine::Op::For {
            iv: arg1,
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(2),
            carried: Vec::new(),
            body: vec![DfirOp::Affine(affine::Op::For {
                iv: arg2,
                lo: affine::Bound::Const(0),
                hi: affine::Bound::Const(4),
                carried: Vec::new(),
                body: innermost,
                dbg_name: None,
            })],
            dbg_name: None,
        })];

        let load_op = defining_op(loaded, &scope).expect("the load is in the innermost body");
        let mut load = TpmvVectorLoad::new(load_op, DfirUnit::Lxlu);
        load.initialize(&scope);

        // `info` borrows the nest, so the emission goes into a clone — as entry 324's test does.
        let mut emitted = scope.clone();
        let transformed = load.vector.base.transform(&mut vals, &scope, &mut emitted);

        let TransformedPagedViews::Transformed { operands, mem_ops } = transformed else {
            panic!("every subscript is a constant-bounded iterator: {transformed:?}")
        };
        let [operand] = operands.as_slice() else {
            panic!("one memory operand, and it is paged: {operands:?}")
        };
        assert_eq!(operand.mem_index, MemoryOperandIndex::DirSrc);
        assert_eq!(
            operand.pages.len(),
            4,
            "`%arg1 * 3` and `%arg2 * 2 + 2` reach four of the six pages"
        );
        // `mem_ops_ = new_mem_ops` — one replacement per page, in `getIdxSets()` order.
        let replacements: Vec<Val> = operand
            .pages
            .iter()
            .flat_map(|page| page.new_mem_ops.iter().map(|new| new.result))
            .collect();
        assert_eq!(mem_ops, replacements);
        assert_eq!(mem_ops.len(), 4);

        // `eraseMemOpAndUseChain` over the one original load: the send, the rotate, then the load.
        let kinds: Vec<&str> = operand
            .erased
            .iter()
            .map(|op| match op {
                DfirOp::Dataflow(dataflow::Op::Send { .. }) => "send",
                DfirOp::VectorChain(vectorchain::Op::Rotate { .. }) => "rotate",
                DfirOp::Agen(agen::Op::VectorLoad { .. }) => "load",
                other => panic!("nothing else is downstream of the load: {other:?}"),
            })
            .collect();
        assert_eq!(kinds, ["send", "rotate", "load"]);

        // `info.paged_mem_view_->erase(); info.paged_mem_view_ = nullptr;`
        assert_eq!(operand.erased_mem_view, mem_view);
        let [info] = load.vector.base.tpmv_info.as_slice() else {
            unreachable!("entry 202 filled exactly one")
        };
        assert_eq!(info.paged_mem_view, None);
        // `calculateIndicesRanges` ran here and nowhere else — `0 to 2` and `0 to 4`, closed.
        assert_eq!(
            info.indices_ranges,
            [IvRange { lb: 0, ub: 1 }, IvRange { lb: 0, ub: 3 }]
        );
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 199/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e199_createConditionsForNonHyperRectSubscripts
///
/// **199/384** `TPMVBase::createConditionsForNonHyperRectSubscripts` —
/// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:342` (35L).
///
/// ⛔ THE BOUNDS ARE THE PAGE SET'S, NOT THE ITERATORS': `page_set_bounds` is one
/// `getConstantBound(LB/UB, res)` pair per map RESULT (the opening `DT_CHECK`) while `iter_args` holds
/// one value per NON-CONSTANT result — `arg_idx`, which entry 295 computes. Hence, unlike entry 198,
/// no map or index rewrite and no whole-range skip; the `DT_CHECK_MSG` on the pair is [`IvRange`].
#[must_use]
pub fn create_conditions_for_non_hyper_rect_subscripts(
    vals: &mut Values,
    insert_refs: &[InsertRef<'_>],
    subscripts_map: &AffineMap,
    iter_args: &[Val],
    page_set_bounds: &[IvRange],
) -> Vec<Condition> {
    // `SmallVector<Operation *, 16> insert_refs = mem_ops_;` at the call site (`:201`), an entry of
    // which is already a conditional once an earlier page created one for that access.
    let mut nests: Vec<Condition> = insert_refs
        .iter()
        .map(|insert_ref| match insert_ref {
            InsertRef::Conditional(condition) => (*condition).clone(),
            InsertRef::MemOp => Condition::default(),
        })
        .collect();

    // `for (int res = 0, e = subscripts_map.getNumResults(); res < e; ++res)` with
    // `if (isa<AffineConstantExpr>(..)) continue;` — and the second zip IS `iter_args[arg_idx]`
    // together with the `++arg_idx` that only a non-constant result reaches.
    let bounded = subscripts_map
        .results
        .iter()
        .zip(page_set_bounds)
        .filter(|(result, _)| !matches!(result, AffineExpr::Const(_)))
        .map(|(_, bound)| bound)
        .zip(iter_args);

    for (bound, &iter_arg) in bounded {
        for nest in &mut nests {
            let created = if bound.lb == bound.ub {
                // "Equality"
                create_equality_condition(vals, iter_arg, bound.lb)
            } else {
                // "Inequality - one condition for each bound"
                create_inequality_condition(vals, iter_arg, bound.lb, bound.ub)
            };
            // `setBuilderToInsertRef(builder, insert_refs[i])` then `insert_refs[i] = ..`: the new
            // guard nests inside whatever that slot already holds.
            *nest = core::mem::take(nest).and_then(created);
        }
    }

    nests
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 200/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e200_updateTPMVInfo
///
/// **200/384** `TPMVBase::updateTPMVInfo` —
/// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:382` (17L).
///
/// ⛔ `lookupOrNull` FOR THE INDICES AND `lookupOrDefault` FOR THE VIEW: "indices may contain
/// iterators that weren't re-cloned", so a miss leaves the index alone, while the view defaults to
/// itself and is then re-`cast`. ⛔ Where that `cast` would abort — the mapped value defines no paged
/// view — the old handle stands, which is the one thing this cannot do faithfully.
pub fn update_tpmv_info<'p>(info: &mut TpmvInfo<'p>, ir_map: &ValueMapping, scope: &'p [DfirOp]) {
    // "Update the indices to keep them in sync as we clone the loops."
    for index in &mut info.indices {
        // `auto new_index = ir_map.lookupOrNull(info.indices_[i]); if (new_index) ..`
        if let Some(new_index) = ir_map.lookup(*index) {
            *index = new_index;
        }
    }

    // "Update paged_mem_view, if it exists, since it may have changed. The mem_view may not be in the
    // innermost loop."
    if let Some(paged_mem_view) = info.paged_mem_view {
        // `ir_map.lookupOrDefault(info.paged_mem_view_)`, then
        // `cast<GetPagedLogicalMemoryViewOp>(new_mem_view.getDefiningOp())`.
        let new_mem_view = ir_map.lookup_or_default(paged_mem_view.result);
        if let Some(found) = defining_op(new_mem_view, scope).and_then(PagedMemViewHandle::of) {
            info.paged_mem_view = Some(found);
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 201/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e201_setLoopIteratorOrder
///
/// **201/384** `TPMVBase::setLoopIteratorOrder` —
/// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:575` (13L).
///
/// ⛔ A KEY, NOT A COMPARATOR: `isProperAncestor` answers false BOTH ways for two sibling loops, which
/// is not a strict weak ordering and is unspecified input to Rust's sort. A loop's ancestry IS its
/// region path and an ancestor's path is a PREFIX, so lexicographic order on that path is the
/// reference's own answer — outermost first (`:443`, "Loops are updated from outermost to innermost").
#[must_use]
pub fn set_loop_iterator_order(indices: &[Val], scope: &[DfirOp]) -> Vec<usize> {
    // "Initialize the ordered_indices_idxs vector to prepare for sorting."
    let mut ordered_indices_idxs: Vec<usize> = (0..indices.len()).collect();

    // `cast<BlockArgument>(indices[a]).getOwner()->getParentOp()`, as a position rather than a handle.
    // ⛔ AN INDEX NO REGION BINDS SORTS LAST, where the reference's `cast` aborts: `false` is the only
    // other answer available, and a stable sort then leaves it where it was.
    ordered_indices_idxs.sort_by_key(|&idx| {
        let path = loop_path_of(indices[idx], scope, &[]);
        (path.is_none(), path.unwrap_or_default())
    });

    ordered_indices_idxs
}

/// WHERE THE OP WHOSE REGION BINDS `val` SITS — `getOwner()->getParentOp()`, as the path of ordinals
/// that answers `isProperAncestor` by being a prefix. [`None`] when no region binds it.
fn loop_path_of(val: Val, scope: &[DfirOp], prefix: &[usize]) -> Option<Vec<usize>> {
    for (ordinal, op) in scope.iter().enumerate() {
        let mut path = prefix.to_vec();
        path.push(ordinal);
        if dialects::block_args(op).contains(&val) {
            return Some(path);
        }
        for region in dialects::regions(op) {
            if let Some(found) = loop_path_of(val, region, &path) {
                return Some(found);
            }
        }
    }

    None
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 202/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

impl<'p> TpmvVectorLoad<'p> {
    /// Replaces: e202_initialize
    ///
    /// **202/384** `TPMVVectorLoad::initialize` —
    /// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:658` (13L).
    ///
    /// ⛔ TWO `DT_CHECK`s AND A `cast` ARE THE THREE `let … else`: exactly one mem op, an
    /// `agen.vector_load`, and a paged view behind its `getMemRef()`. ⭐ `getAffineMapAttr()` +
    /// `getMapIndices()` are this island's inline indices split by [`access_map`]; `context_` has no
    /// counterpart and the `LogicalResult` is unconditionally success, so nothing is returned.
    pub fn initialize(&mut self, scope: &'p [DfirOp]) {
        // `DT_CHECK(mem_ops_.size() == 1);` then `dyn_cast<agen::VectorLoadOp>(mem_ops_[0])` and
        // `DT_CHECK(op)`.
        let [mem_op] = self.vector.base.mem_ops.as_slice() else {
            return;
        };
        let mem_op: &'p DfirOp = mem_op;
        let Some(op) = VectorLoadOp::of(mem_op) else {
            return;
        };

        // `cast<dataflow::GetPagedLogicalMemoryViewOp>(op.getMemRef().getDefiningOp())`.
        let Some(paged_mem_view) = defining_op(op.view, scope).and_then(PagedMemViewHandle::of)
        else {
            return;
        };

        // `tpmv_info_.emplace_back(paged_mem_view, op.getAffineMapAttr().getValue());` — the
        // two-argument constructor, so `mem_index_` stays at its default.
        let (subscripts_map, indices) = access_map(op.indices);
        let mut info = TpmvInfo::new(paged_mem_view, subscripts_map, MemoryOperandIndex::DirSrc);

        // `for (auto index : op.getMapIndices()) tpmv_info_[0].indices_.push_back(index);`
        info.indices = indices;

        self.vector.base.tpmv_info.push(info);
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 258/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e258_addConstraintsForIVRanges
///
/// **258/384** `TPMVBase::addConstraintsForIVRanges` — one pair of inequalities per symbol,
/// `s<i> - lb >= 0` and `-s<i> + ub >= 0`, appended to the page's own constraints.
///
/// ⭐ THE APPENDED SET'S SPACE MATCHES BECAUSE THE PAGE'S DIMS ARE ALREADY GONE: `getPageValidity`
/// substitutes the subscripts into `page_set` with `replaceDimsAndSymbols(.., 0, getNumSymbols())`
/// (`:121-126`), leaving ZERO dims and one symbol per loop iterator — which is also what makes
/// `indices_ranges[sym_idx]` the range OF symbol `sym_idx`, in [`calculate_indices_ranges`]' order.
pub fn add_constraints_for_iv_ranges(
    page_sel_constraints: &mut IntegerSet,
    indices_ranges: &[IvRange],
) {
    // `int num_syms = page_sel_constraints.getNumSymbolVars(); if (num_syms == 0) return;`
    let ranges = indices_ranges
        .iter()
        .take(page_sel_constraints.symbols as usize);

    for (sym_idx, range) in (0u32..).zip(ranges) {
        let sym_expr = AffineExpr::sym(sym_idx);
        // `<sym> - <lb> >= 0`, the pair's first element being the lower bound.
        page_sel_constraints.constraints.push(Constraint {
            expr: sym_expr.clone().added(AffineExpr::Const(-range.lb)),
            is_equality: false,
        });
        // `-<sym> + <ub> >= 0`, the second being the inclusive upper bound.
        page_sel_constraints.constraints.push(Constraint {
            expr: sym_expr.scaled(-1).added(AffineExpr::Const(range.ub)),
            is_equality: false,
        });
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 259/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e259_createNewSubscriptsFromStartElements
///
/// **259/384** `TPMVBase::createNewSubscriptsFromStartElements` — every subscript less the page's
/// start element in that dimension, over the input map's dims and ⛔ NO symbols.
///
/// ⚠️ `DT_CHECK(start_elements.size() >= subscripts_map.getNumResults())` IS THE PAIRING: the
/// elements are [`calculate_start_elements_for_page`]'s, one per span of the page rectangle, and the
/// map has one result per subscript of the access that reads it.
#[must_use]
pub fn create_new_subscripts_from_start_elements(
    subscripts_map: &AffineMap,
    start_elements: &[StartElement],
) -> AffineMap {
    AffineMap {
        dims: subscripts_map.dims,
        // `AffineMap::get(subscripts_map.getNumDims(), 0, ..)`.
        syms: 0,
        results: subscripts_map
            .results
            .iter()
            .zip(start_elements)
            // `subscripts_map.getResult(dim) - start_elements[dim]`.
            .map(|(result, start)| result.clone().added(AffineExpr::Const(-start.0)))
            .collect(),
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 260/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e260_gatherPageDependentDimsForPage
///
/// **260/384** `TPMVComposite::gatherPageDependentDimsForPage` — a symbol carried by a constraint
/// whose CONSTANT column moved from one page to the next has a bearing on page selection.
///
/// ⛔ BOTH `DT_CHECK`s BECOME THE PAIRING: the row counts by zipping each kind's rows in the
/// reference's own two passes, and `compare_ineq[sym] == ineq[sym]` by requiring the two coefficients
/// to agree before the symbol counts. Under the premise the reference asserts, the same insertions.
pub fn gather_page_dependent_dims_for_page(
    page_sel_constraints: &IntegerSet,
    compare_constraints: &IntegerSet,
    page_dependent_time_syms: &mut BTreeSet<PageSelSym>,
) {
    // `getInequality(i)` then `getEquality(i)` — two loops, so a row is only ever compared with a row
    // of its own kind, whatever order the set lists them in.
    for is_equality in [false, true] {
        let rows = |set: &IntegerSet| {
            set.constraints
                .iter()
                .filter(|constraint| constraint.is_equality == is_equality)
                .map(|constraint| constraint.expr.flatten(set.dims, set.symbols))
                .collect::<Vec<_>>()
        };

        for (row, compare_row) in rows(page_sel_constraints)
            .into_iter()
            .zip(rows(compare_constraints))
        {
            // `if (compare_ineq[const_col] != ineq[const_col])` — the coefficients are the same
            // across pages, so a constraint that moved moved in its constant column.
            if row.constant == compare_row.constant {
                continue;
            }
            // `for (int sym = 0; sym < num_syms; ++sym)` reads `ineq[sym]` from COLUMN 0, and that is
            // still symbol 0 here: `replaceDimsAndSymbols(.., 0, getNumSymbols())` (`:123-125`) leaves
            // the set ZERO-DIM, so the leading columns are the symbols.
            for (sym, (coeff, compare_coeff)) in
                (0u32..).zip(row.syms.iter().zip(&compare_row.syms))
            {
                // `if (ineq[sym] != 0) page_dependent_time_syms_.insert(sym);`
                if *coeff != 0 && coeff == compare_coeff {
                    page_dependent_time_syms.insert(PageSelSym(sym));
                }
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 294/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHETHER ONE PAGE CAN BE THE ONE AN ACCESS READS — [`get_page_validity`]'s answer.
///
/// ⛔ THE REFERENCE HANDS BACK AN EMPTY CONSTRAINT SYSTEM FOR BOTH REFUSALS and its caller reads
/// only `isEmpty()` (`:168`); the two variants below are that emptiness with its reason kept, and the
/// third is the `DT_CHECK_MSG` at `:117-119`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PageValidity {
    /// The page selection constraints, non-empty — the system entries 199/260/309 read.
    Valid(IntegerSet),
    /// `"Invalid page - subscripts and page do not intersect"`.
    SubscriptsDoNotIntersectThePage,
    /// `"Invalid page - IV subscripts have bounds that do not intersect the page"`.
    IvBoundsDoNotIntersectThePage,
    /// `DT_CHECK_MSG(.., "idx_set should be hyper rectangular")`.
    PageIsNotHyperRectangular,
}

/// Replaces: e294_getPageValidity
///
/// **294/384** `TPMVBase::getPageValidity` — the page's own rectangle with the access's SUBSCRIPTS
/// substituted for its dimensions, then the iterators' ranges, emptiness tested after each.
///
/// ⭐ TWO SYSTEMS, AND ONLY THE SECOND IS RETURNED: `page_set_constraints` exists solely to be
/// checked hyper-rectangular, while `page_sel_constraints` is built from `subscripts_in_page`.
pub fn get_page_validity(
    info: &TpmvInfo<'_>,
    subscripts_map_sym: &AffineMap,
    page_set: &IntegerSet,
) -> PageValidity {
    let page_set_constraints = FlatConstraints::from_integer_set(page_set);
    if !page_set_constraints.is_hyper_rectangular(0, page_set_constraints.num_cols() - 1) {
        return PageValidity::PageIsNotHyperRectangular;
    }

    // `replaceDimsAndSymbols(subscripts_map_sym.getResults(), {}, 0, getNumSymbols())` — ZERO dims
    // out, so what was "page dimension `d`" is now the subscript that indexes it.
    let mut page_sel_constraints =
        page_set.replace_dims(&subscripts_map_sym.results, 0, subscripts_map_sym.syms);
    if FlatConstraints::from_integer_set(&page_sel_constraints).is_empty() {
        return PageValidity::SubscriptsDoNotIntersectThePage;
    }

    add_constraints_for_iv_ranges(&mut page_sel_constraints, &info.indices_ranges);
    if FlatConstraints::from_integer_set(&page_sel_constraints).is_empty() {
        return PageValidity::IvBoundsDoNotIntersectThePage;
    }

    PageValidity::Valid(page_sel_constraints)
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 295/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHAT REBUILDING THE NEST HANDS BACK — [`create_iter_args_for_conditionals`]'s three outputs.
#[derive(Debug, Clone)]
pub struct IterArgsForConditionals {
    /// `info.conditional_iter_args_` — the INNERMOST arg of each non-constant map result, in result
    /// order, which is the order entries 198/199 index with their `arg_idx`.
    pub conditional_iter_args: Vec<Val>,
    /// `indices`, each now the induction variable of the loop that binds it after the rebuild.
    pub indices: Vec<Val>,
    /// One `IRMapping` per rebuilt loop, outermost first.
    ///
    /// ⛔ THE `mem_ops_` / `paged_mem_view_` / SIBLING-`TPMVInfo` RE-SYNCS AT `:497-514` ARE THE
    /// CALLER'S, and that is a type-level fact rather than a shortcut: they are `&'p` borrows of the
    /// very program this function rewrites, and Rust cannot hold one across the rewrite. Replaying
    /// them in this order over the new scope — [`update_tpmv_info`] per mapping — is those lines.
    pub ir_maps: Vec<ValueMapping>,
}

/// Replaces: e295_createIterArgsForConditionals
///
/// **295/384** `TPMVBase::createIterArgsForConditionals` — one iter_arg per non-constant subscript on
/// every loop of the nest: initialised to that subscript's constant term at the outermost loop and to
/// the enclosing loop's arg below it, each loop yielding `arg + its own coefficient`.
///
/// ⛔ ONE DELIBERATE DIVERGENCE: `:435` SCATTERS the coefficients through the loop-order permutation
/// (`ordered[perm[i]] = coeffs[i]`) while `:481` reads them GATHERED, `ordered_coeffs[c][i]` being
/// "the coefficient of the loop at POSITION `i`". The two agree only where the permutation is its own
/// inverse — the identity, or any two-loop nest. ⭐ This gathers. `DT_CHECK(..empty())` is structural.
#[must_use]
pub fn create_iter_args_for_conditionals(
    vals: &mut Values,
    scope: &mut Vec<DfirOp>,
    subscripts_map: &AffineMap,
    indices: &[Val],
) -> IterArgsForConditionals {
    let mut indices = indices.to_vec();
    let mut conditional_iter_args: Vec<Val> = Vec::new();
    let mut ir_maps: Vec<ValueMapping> = Vec::new();

    // `setLoopIteratorOrder(indices)` — position → which entry of `indices`, outermost first.
    let ordered_indices_idxs = set_loop_iterator_order(&indices, scope);

    // `for (auto &res : subscripts_map.getResults()) { if (isa<AffineConstantExpr>(res)) continue; ..`
    // — `getFlattenedAffineExpr` per remaining result, its DIM coefficients read by loop position and
    // its constant (`coeffs.back()`) kept aside. ⛔ The reference's permutation covers `coeffs.size()`
    // slots, so a symbol or a local would index `ordered_indices_idxs` out of range: dims only.
    let ordered_coeffs: Vec<(Vec<i64>, i64)> = subscripts_map
        .results
        .iter()
        .filter(|result| !matches!(result, AffineExpr::Const(_)))
        .map(|result| {
            let flat = result.flatten(subscripts_map.dims, subscripts_map.syms);
            let per_loop = ordered_indices_idxs
                .iter()
                .map(|&idx| flat.dims.get(idx).copied().unwrap_or_default())
                .collect();
            (per_loop, flat.constant)
        })
        .collect();
    let num_args = ordered_coeffs.len();

    // "Each loop will get one iter_arg per map result involving loop iterators", outermost first.
    for (position, &indices_idx) in ordered_indices_idxs.iter().enumerate() {
        // `cast<BlockArgument>(indices[indices_idx]).getOwner()->getParentOp()`, reached mutably —
        // ⛔ where that `cast` aborts, and where the op it lands on is no counted loop, nothing moves.
        let Some(path) = binding_path(indices[indices_idx], scope) else {
            continue;
        };
        let Some((ops, ordinal)) = ops_at_mut(scope, &path) else {
            continue;
        };
        let Some(counted) = ops.get(ordinal).and_then(CountedLoop::of) else {
            continue;
        };

        // `createForOpWithAdditionalReturnValue(curr_loop, ordered_coeffs.size(), ir_map, false)`.
        let grown = create_for_op_with_additional_return_value(vals, &counted, num_args, false);
        let mut new_loop = grown.op;
        let mut init_consts: Vec<DfirOp> = Vec::new();
        let (DfirOp::Affine(affine::Op::For {
            iv, carried, body, ..
        })
        | DfirOp::Scf(scf::Op::For {
            iv, carried, body, ..
        })) = &mut new_loop
        else {
            continue;
        };
        let new_iv = *iv;
        // `region_iter_args[region_iter_args.size() - num_args + c]` — the added args are the last.
        let first_added = carried.len().saturating_sub(num_args);

        for (c, (per_loop, constant)) in ordered_coeffs.iter().enumerate() {
            let added = &mut carried[first_added + c];
            if conditional_iter_args.len() < num_args {
                // Outermost: an `arith::ConstantIndexOp(ordered_coeffs[c].back())` before the loop, and
                // `setOperand` over the added init. ⚠️ Entry 264's own fill constant is left DEAD by
                // that overwrite, exactly as it is in the reference.
                let result = vals.mint();
                init_consts.push(DfirOp::Arith(arith::Op::Constant {
                    result,
                    value: *constant,
                }));
                added.init = result;
                conditional_iter_args.push(added.arg);
            } else {
                // An inner loop initialises to the arg the enclosing loop just contributed.
                added.init = conditional_iter_args[c];
                conditional_iter_args[c] = added.arg;
            }

            // "For each subscript involving the iterator of the current loop, create an addOp of the
            // new loop iterator and its coefficient" — ⭐ THE LHS IS THIS LOOP'S OWN NEW ARG, which
            // both branches above have just put in `conditional_iter_args[c]`.
            let coeff = per_loop.get(position).copied().unwrap_or_default();
            if coeff == 0 {
                continue;
            }
            let coeff_val = vals.mint();
            let sum = vals.mint();
            // `builder.setInsertionPoint(terminator)` — the two ops go before the yield.
            let at = body.len().saturating_sub(1);
            body.insert(
                at,
                DfirOp::Arith(arith::Op::Constant {
                    result: coeff_val,
                    value: coeff,
                }),
            );
            body.insert(
                at + 1,
                DfirOp::Arith(arith::Op::AddI(arith::IntBinary {
                    result: sum,
                    lhs: conditional_iter_args[c],
                    rhs: coeff_val,
                    ty: ScalarTy::Index,
                })),
            );
            // `terminator->setOperand(terminator->getNumOperands() - num_args + c, add_op)`.
            if let Some(
                DfirOp::Affine(affine::Op::Yield { operands })
                | DfirOp::Scf(scf::Op::Yield { operands }),
            ) = body.last_mut()
            {
                let yielded = operands.len().saturating_sub(num_args) + c;
                if let Some(place) = operands.get_mut(yielded) {
                    *place = sum;
                }
            }
        }

        // `curr_loop->erase()` — the new loop takes its place, both sets of constants before it.
        let mut replacement = grown.consts;
        replacement.extend(init_consts);
        replacement.push(new_loop);
        ops.splice(ordinal..ordinal + 1, replacement);

        // The `replaceAllUsesWith` entry 264 hands back for its caller to apply (`Utils.cpp:82-83`).
        for (old, new) in &grown.replacements {
            dialects::replace_all_uses(scope, *old, *new);
        }

        // "Update the indices to keep them in sync as we clone the loops."
        indices[indices_idx] = new_iv;
        let ir_map = grown.ir_map.unwrap_or_default();
        for &later in &ordered_indices_idxs[position + 1..] {
            // ⭐ `cast<BlockArgument>(new_index).getOwner()->getParentOp().getInductionVar()` IS the
            // mapped value: an induction variable maps to the clone's own induction variable, so that
            // walk back out to the owning loop is the identity. `DT_CHECK(new_index)` is the `if`.
            if let Some(new_index) = ir_map.lookup(indices[later]) {
                indices[later] = new_index;
            }
        }
        ir_maps.push(ir_map);
    }

    IterArgsForConditionals {
        conditional_iter_args,
        indices,
        ir_maps,
    }
}

/// THE REGION PATH TO THE OP WHOSE REGION BINDS `val` — `getOwner()->getParentOp()` as a route
/// through this island's `Vec<DfirOp>` tree, one (op ordinal, region ordinal) pair per level.
///
/// ⭐ THE REGION ORDINAL IS WHAT [`loop_path_of`] DROPS, and a mutable descent needs it: an
/// `scf.if` has two regions and only one of them holds the loop.
pub(super) fn binding_path(val: Val, scope: &[DfirOp]) -> Option<Vec<(usize, usize)>> {
    for (ordinal, op) in scope.iter().enumerate() {
        if dialects::block_args(op).contains(&val) {
            return Some(vec![(ordinal, 0)]);
        }
        for (region_ordinal, region) in dialects::regions(op).into_iter().enumerate() {
            if let Some(rest) = binding_path(val, region) {
                let mut path = vec![(ordinal, region_ordinal)];
                path.extend(rest);
                return Some(path);
            }
        }
    }

    None
}

/// The list an op sits in and where in it — [`binding_path`] followed mutably.
pub(super) fn ops_at_mut<'s>(
    scope: &'s mut Vec<DfirOp>,
    path: &[(usize, usize)],
) -> Option<(&'s mut Vec<DfirOp>, usize)> {
    let (&(ordinal, region_ordinal), rest) = path.split_first()?;
    if rest.is_empty() {
        return Some((scope, ordinal));
    }
    let region = dialects::regions_mut(scope.get_mut(ordinal)?)
        .into_iter()
        .nth(region_ordinal)?;

    ops_at_mut(region, rest)
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 309/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// AN ACCESS'S INDEX LIST FROM THE `affine_map` AND OPERANDS MLIR KEEPS IT AS —
/// [`super::vc_vector_operands::access_map`]'s inverse, which entry 128 needs because
/// `cloneWithNewAccessInfo` is handed the two halves (`:246`) while this island writes the strided
/// sum inline ([`Index::Strided`]).
pub(super) fn indices_from_map(subscripts_map: &AffineMap, indices: &[Val]) -> Vec<Index> {
    subscripts_map
        .results
        .iter()
        .map(|result| {
            let flat = result.flatten(subscripts_map.dims, subscripts_map.syms);
            let terms: Vec<(Val, i64)> = flat
                .dims
                .iter()
                .enumerate()
                .filter(|&(_, &coeff)| coeff != 0)
                .filter_map(|(dim, &coeff)| indices.get(dim).map(|&index| (index, coeff)))
                .collect();
            match terms.as_slice() {
                // A pinned dimension — `%mem_view[4, ..]`, which is what entry 198 substituted in.
                [] => Index::Const(flat.constant),
                // ⭐ A UNIT STRIDE WITH NO ADDEND PRINTS BARE, per [`Index::Strided`].
                [(index, 1)] if flat.constant == 0 => Index::Val(*index),
                _ => Index::Strided(terms, flat.constant),
            }
        })
        .collect()
}

/// THE `TPMVInfo` AND `mem_ops_` STATE ONE PAGE'S CONSTRUCTION READS AND REWRITES.
///
/// ⛔ EVERY HANDLE IS THE [`Val`] IT BINDS, NOT A `&'p` BORROW: `createIterArgsForConditionals`
/// CLONES the whole loop nest and `:503-516` then re-points `mem_ops_` and `paged_mem_view_` into the
/// clone. No borrow of the program being rewritten survives that, and re-resolving a value against
/// the new scope is exactly what the reference's `ir_map` lookups do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedAccess {
    /// `mem_ops_`, each by the value it binds — `:503-506`.
    pub mem_ops: Vec<Val>,
    /// `info.paged_mem_view_`, by its result — `:511-514`.
    pub paged_mem_view: Val,
    /// `info.subscripts_map_`.
    pub subscripts_map: AffineMap,
    /// `info.indices_`.
    pub indices: Vec<Val>,
    /// `info.indices_ranges_`, as [`calculate_indices_ranges`] computed them.
    pub indices_ranges: Vec<IvRange>,
    /// `info.conditional_iter_args_` — non-empty once some page took the non-hyper-rectangular arm.
    pub conditional_iter_args: Vec<Val>,
}

/// WHAT ONE PAGE'S CONSTRUCTION PRODUCED.
#[derive(Debug, Clone)]
pub struct ConstructedPage {
    /// One entry per `mem_ops_`, in that order: this page's non-paged view, the access rebuilt
    /// against it and its cloned use chain, wrapped in whatever guard the page needed.
    pub placed: Vec<Vec<DfirOp>>,
    /// `new_mem_ops` — appended in the same order, which is what entry 356 re-populates `mem_ops_`
    /// with after erasing the originals.
    pub new_mem_ops: Vec<NewMemOp>,
    /// `mem_ops_`, `paged_mem_view_`, `indices_` and `conditional_iter_args_` on the way out.
    pub access: PagedAccess,
    /// One `IRMapping` per loop the nest rebuild cloned. ⛔ THE SIBLING-`TPMVInfo` RE-SYNC AT
    /// `:518-521` IS THE CALLER'S, over [`update_tpmv_info`]: `tpmv_info_` is not this function's to
    /// hold, for the reason [`PagedAccess`] gives.
    pub ir_maps: Vec<ValueMapping>,
}

/// A PAGE, CONSTRUCTED OR REFUSED.
#[derive(Debug, Clone)]
pub enum ValidPage {
    /// The page's accesses, emitted.
    Constructed(ConstructedPage),
    /// `DT_CHECK_MSG(lb.has_value() && ub.has_value(), "expected constant lower and upper bounds")`
    /// (`:304-305`) — see [`SubscriptDim::of`], which is where that pair is taken.
    SubscriptBoundIsNotConstant,
}

/// Replaces: e309_constructValidPage
///
/// **309/384** `TPMVBase::constructValidPage` —
/// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:190` (58L).
///
/// ⛔ `insert_refs = mem_ops_` IS RE-READ EVERY PAGE (`:201`, and again at `:219`): a previous page's
/// conditions guard THAT page's copy, so no access ever arrives already wrapped. ⛔ AND THE TWO ARMS
/// READ DIFFERENT SYSTEMS — hyper-rectangular bounds come from `page_sel_constraints` (one symbol per
/// ITERATOR), non-hyper-rectangular ones from `page_set` itself (one dim per VIEW AXIS), and only the
/// rebuild writes back to `info`. ⚠️ `subscripts_map_sym` is in the signature and unread in the body.
#[must_use]
pub fn construct_valid_page(
    vals: &mut Values,
    scope: &mut Vec<DfirOp>,
    access: &PagedAccess,
    page: &Page,
    page_sel_constraints: &IntegerSet,
) -> ValidPage {
    // "If conditionals are created, every page may require different subscripts/indices."
    // `AffineMap subscripts_map = info.subscripts_map_;` / `SmallVector<Value> indices = info.indices_;`
    let mut subscripts_map = access.subscripts_map.clone();
    let mut indices = access.indices.clone();
    let mut out = access.clone();
    let mut ir_maps: Vec<ValueMapping> = Vec::new();

    // `FlatLinearValueConstraints page_set_constraints(page_set);` — the page's own rectangle, whose
    // `getConstantBound(LB/UB, res)` per dimension IS its span (entry 123 records why).
    let page_set_bounds: Vec<IvRange> = page
        .idx_set
        .spans
        .iter()
        .map(|span| IvRange {
            lb: span.lo,
            ub: span.hi,
        })
        .collect();

    // `SmallVector<Operation *, 16> insert_refs = mem_ops_;` — bare ops, on every page.
    let mut nests: Vec<Condition> = vec![Condition::default(); access.mem_ops.len()];

    // `if (!indices.empty())` — a constant access reaches one page and needs no guard at all.
    if !indices.is_empty() {
        let flat = FlatConstraints::from_integer_set(page_sel_constraints);
        // `page_sel_constraints.isHyperRectangular(0, page_sel_constraints.getNumCols() - 1)`.
        if flat.is_hyper_rectangular(0, flat.num_cols() - 1) {
            // "Hyperrectangular subscripts - creating conditionals"
            //
            // ⛔ `constant_bound_ON_VAR`, NOT `constant_bound`: entry 294 hands this system over with
            // ZERO dims and one SYMBOL per iterator, so `getConstantBound(LB, dim)`'s `dim` is a
            // COLUMN and column 0 is `s0`. Reading it as `d0` answers `None` for every iterator.
            let dims: Option<Vec<SubscriptDim>> = (0..subscripts_map.dims)
                .map(|dim| {
                    SubscriptDim::of(
                        *indices.get(dim as usize)?,
                        page_sel_constraints.constant_bound_on_var(BoundType::Lb, dim),
                        page_sel_constraints.constant_bound_on_var(BoundType::Ub, dim),
                        *out.indices_ranges.get(dim as usize)?,
                    )
                })
                .collect();
            let Some(dims) = dims else {
                return ValidPage::SubscriptBoundIsNotConstant;
            };

            let created = create_conditions_for_hyper_rect_subscripts(
                vals,
                &vec![InsertRef::MemOp; nests.len()],
                &subscripts_map,
                &dims,
            );
            // ⭐ `removal` CANNOT REFUSE HERE: entry 198 deletes from the very list it built out of
            // [`SubscriptDim::index`], so every request is present.
            nests = created.insert_refs;
            subscripts_map = created.subscripts_map;
            indices = created.indices;
        } else {
            // `if (info.conditional_iter_args_.empty())` — at most once across the whole page loop.
            if out.conditional_iter_args.is_empty() {
                // "Non-hyperrectangular subscripts - creating iter_args"
                let created =
                    create_iter_args_for_conditionals(vals, scope, &subscripts_map, &indices);
                out.conditional_iter_args = created.conditional_iter_args;
                // `info.indices_ = indices;`
                indices = created.indices;
                out.indices = indices.clone();

                // `:503-516` — `mem_ops_` and `paged_mem_view_` re-pointed into the clone, per loop.
                for ir_map in &created.ir_maps {
                    for mem_op in &mut out.mem_ops {
                        if let Some(moved) = ir_map.lookup(*mem_op) {
                            *mem_op = moved;
                        }
                    }
                    out.paged_mem_view = ir_map.lookup_or_default(out.paged_mem_view);
                }
                ir_maps = created.ir_maps;

                // `insert_refs = mem_ops_;  // Loop bodies may have been cloned.`
                nests = vec![Condition::default(); out.mem_ops.len()];
            }

            // "Non-hyperrectangular subscripts - creating conditionals"
            nests = create_conditions_for_non_hyper_rect_subscripts(
                vals,
                &vec![InsertRef::MemOp; nests.len()],
                &subscripts_map,
                &out.conditional_iter_args,
                &page_set_bounds,
            );
        }
    }

    // `calculateStartElementsForPage(page_set_constraints)` then
    // `createNewSubscriptsFromStartElements(subscripts_map, start_elements)`.
    let start_elements = calculate_start_elements_for_page(&page.idx_set);
    let new_subscripts_map =
        create_new_subscripts_from_start_elements(&subscripts_map, &start_elements);
    let new_indices = indices_from_map(&new_subscripts_map, &indices);

    // The rewrite is finished, so the program is readable again — `OpBuilder builder(mem_ops_[0])`
    // and the zip below need nothing but a look at it.
    let scope: &[DfirOp] = scope;
    let mut placed: Vec<Vec<DfirOp>> = Vec::new();
    let mut new_mem_ops: Vec<NewMemOp> = Vec::new();

    // `for (auto [insert_ref, mem_op] : zip(insert_refs, mem_ops_))`.
    for (nest, &mem_op) in nests.iter().zip(&out.mem_ops) {
        // `cast<agen::VectorLoadOp>(mem_op)` inside entry 128 — the class the manager already chose,
        // so `None` belongs to one of the five unscheduled `createNewMemOp` overrides and to nothing
        // a `TPMVVectorLoad` can hold.
        let Some(load) = defining_op(mem_op, scope).and_then(VectorLoadOp::of) else {
            continue;
        };
        let Some(paged) = defining_op(out.paged_mem_view, scope).and_then(PagedMemViewHandle::of)
        else {
            continue;
        };

        // `createNonPagedMemView(builder, info.paged_mem_view_, page_idx)`, then
        // `createNewMemOp(builder, mem_op, info, mem_view_val, new_subscripts_map, indices)`.
        let view = create_non_paged_mem_view(vals, paged.view, page);
        let created = create_new_mem_op(
            vals,
            load,
            view.view,
            paged.view.ty.clone(),
            new_indices.clone(),
            scope,
        );

        // `setBuilderToInsertRef(builder, insert_ref)` — both ops go wherever it points.
        let mut ops = view.ops;
        ops.extend(created.ops.iter().cloned());
        placed.push(set_builder_to_insert_ref(InsertRef::Conditional(nest), ops));
        new_mem_ops.push(created);
    }

    ValidPage::Constructed(ConstructedPage {
        placed,
        new_mem_ops,
        access: out,
        ir_maps,
    })
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 310/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// HOW MANY PAGES AN ACCESS CAN REACH — [`TpmvComposite::analyze_valid_pages`]'s answer.
///
/// ⛔ THE TWO REFUSALS ARE `emitOpError`s, WHICH FAIL THE PASS (`:915-921`) — not diagnostics the
/// caller may carry on past. The third is the `DT_CHECK_MSG` [`get_page_validity`] holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidPages {
    /// `valid_pages` — how many of the view's pages the subscripts reach, at least one.
    Found(NonZeroU32),
    /// `"expecting only one active page when subscripts are constant"`.
    ConstantSubscriptsReachSeveralPages,
    /// `"no valid pages found"`.
    NoValidPages,
    /// `DT_CHECK_MSG(.., "idx_set should be hyper rectangular")`, naming the page that is not one.
    PageIsNotHyperRectangular(PageIdx),
}

/// WHICH PAGE OF THE VIEW — `page_idx`, the loop variable of every `getIdxSets()` walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PageIdx(pub u32);

impl TpmvComposite<'_> {
    /// Replaces: e310_analyzeValidPages
    ///
    /// **310/384** `TPMVComposite::analyzeValidPages` —
    /// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:888` (33L).
    ///
    /// ⛔ `compare_constraints` IS SEEDED BY THE FIRST VALID PAGE AND NEVER REPLACED (`:906-909`), so
    /// every later page is compared against page one and not against its predecessor — which is what
    /// makes a symbol "differs across the pages" rather than "differs from the page before it".
    /// ⚠️ AND `DT_CHECK(page_set.getNumDims() == subscripts_map_.getNumResults())` needs no test here:
    /// a [`PageRect`] has one span per view axis and the subscript map has one result per view axis,
    /// so the two counts are the same fact.
    pub fn analyze_valid_pages(
        &mut self,
        info: &TpmvInfo<'_>,
        subscripts_map_sym: &AffineMap,
    ) -> ValidPages {
        // `info.paged_mem_view_.getIdxSets()`, dereferenced unqualified: the view is erased only by
        // `transform()`, which runs after this. A view already gone presents no pages.
        let Some(handle) = info.paged_mem_view else {
            return ValidPages::NoValidPages;
        };

        // `FlatLinearValueConstraints compare_constraints; bool do_compare = false;` — the `Option`
        // IS `do_compare`, which the reference needs because its default-constructed system is
        // indistinguishable from a real one.
        let mut compare_constraints: Option<IntegerSet> = None;
        let mut valid_pages = 0u32;

        for (page_idx, page) in handle.view.pages.iter().enumerate() {
            let page_set = page.idx_set.as_integer_set();
            let page_sel_constraints = match get_page_validity(info, subscripts_map_sym, &page_set)
            {
                PageValidity::Valid(set) => set,
                // `if (page_sel_constraints.isEmpty()) continue;` — both of entry 294's empty
                // answers are that one test.
                PageValidity::SubscriptsDoNotIntersectThePage
                | PageValidity::IvBoundsDoNotIntersectThePage => continue,
                PageValidity::PageIsNotHyperRectangular => {
                    return ValidPages::PageIsNotHyperRectangular(PageIdx(
                        u32::try_from(page_idx).expect("a page count fits a u32"),
                    ));
                }
            };

            valid_pages += 1;

            match &compare_constraints {
                Some(compare) => gather_page_dependent_dims_for_page(
                    &page_sel_constraints,
                    compare,
                    &mut self.page_dependent_time_syms,
                ),
                None => compare_constraints = Some(page_sel_constraints),
            }
        }

        // `if (subscripts_map_sym.getNumSymbols() == 0 && valid_pages > 1)` — constant subscripts
        // name one element, so two pages holding it means the pages overlap.
        if subscripts_map_sym.syms == 0 && valid_pages > 1 {
            return ValidPages::ConstantSubscriptsReachSeveralPages;
        }

        match NonZeroU32::new(valid_pages) {
            Some(found) => ValidPages::Found(found),
            None => ValidPages::NoValidPages,
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 324/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHAT THE PAGE WALK PRODUCED — every valid page's construction, or the refusal that ended it.
///
/// ⛔ THE LAST THREE FAIL THE PASS: two are `emitOpError`s (`:179`, `:183`) and one is
/// `constructValidPage(..).failed()` (`:172-175`), and `run()` returns all of them as `failure()`.
#[derive(Debug, Clone)]
pub enum ConstructedPages {
    /// One entry per valid page, in `getIdxSets()` order, plus `info` as the last page left it.
    Constructed {
        /// `new_mem_ops` and the guarded accesses, per page.
        pages: Vec<ConstructedPage>,
        /// `info.indices_` and `info.conditional_iter_args_` after the walk — see [`PagedAccess`].
        access: PagedAccess,
    },
    /// `"expecting only one active page when subscripts are constant"`.
    ConstantSubscriptsReachSeveralPages,
    /// `"no valid pages found"`, and the absent `paged_mem_view_` that presents no pages at all.
    NoValidPages,
    /// [`PageValidity::PageIsNotHyperRectangular`], naming the page that is not one.
    PageIsNotHyperRectangular(PageIdx),
    /// [`ValidPage::SubscriptBoundIsNotConstant`], naming the page whose construction hit it.
    SubscriptBoundIsNotConstant(PageIdx),
}

/// Replaces: e324_analyzeAndConstructValidPages
///
/// **324/384** `TPMVBase::analyzeAndConstructValidPages` —
/// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:152` (32L).
///
/// ⛔ `info` IS MUTATED PAGE TO PAGE AND THAT IS WHAT `access` THREADS: page two's conditionals see
/// page one's `conditional_iter_args_` and its re-pointed `mem_ops_`, which is the whole reason
/// `constructValidPage` re-reads `mem_ops_` every page. ⚠️ BUT [`get_page_validity`] IS STILL HANDED
/// THE ORIGINAL `info`: it reads only `indices_ranges_`, which nothing in the loop writes.
/// ⚠️ `FlatLinearValueConstraints compare_constraints;` is declared here and never read — entry 310
/// is where that system does its work.
#[must_use]
pub fn analyze_and_construct_valid_pages(
    vals: &mut Values,
    scope: &mut Vec<DfirOp>,
    access: &PagedAccess,
    info: &TpmvInfo<'_>,
    subscripts_map_sym: &AffineMap,
) -> ConstructedPages {
    // `DT_CHECK(new_mem_ops.empty())` — the list is this function's own here, so it starts empty.
    let mut pages: Vec<ConstructedPage> = Vec::new();
    let mut current = access.clone();

    // `auto idx_sets = info.paged_mem_view_.getIdxSets();`, dereferenced unqualified as entry 310.
    let Some(handle) = info.paged_mem_view else {
        return ConstructedPages::NoValidPages;
    };
    let mut valid_pages = 0u32;

    for (page_idx, page) in handle.view.pages.iter().enumerate() {
        let at = PageIdx(u32::try_from(page_idx).expect("a page count fits a u32"));
        // ⚠️ `DT_CHECK(page_set.getNumDims() == info.subscripts_map_.getNumResults())` is one fact in
        // two spellings — see entry 310, which asserts the same pair.
        let page_set = page.idx_set.as_integer_set();
        let page_sel_constraints = match get_page_validity(info, subscripts_map_sym, &page_set) {
            PageValidity::Valid(set) => set,
            // `if (page_sel_constraints.isEmpty()) continue;`
            PageValidity::SubscriptsDoNotIntersectThePage
            | PageValidity::IvBoundsDoNotIntersectThePage => continue,
            PageValidity::PageIsNotHyperRectangular => {
                return ConstructedPages::PageIsNotHyperRectangular(at);
            }
        };

        // `++valid_pages;` — BEFORE the construction, so a page that then refuses still counted.
        valid_pages += 1;

        match construct_valid_page(vals, scope, &current, page, &page_sel_constraints) {
            ValidPage::Constructed(built) => {
                current = built.access.clone();
                pages.push(built);
            }
            ValidPage::SubscriptBoundIsNotConstant => {
                return ConstructedPages::SubscriptBoundIsNotConstant(at);
            }
        }
    }

    // `if (subscripts_map_sym.getNumSymbols() == 0 && valid_pages > 1)` — constant subscripts name
    // one element, so two pages holding it means the pages overlap.
    if subscripts_map_sym.syms == 0 && valid_pages > 1 {
        return ConstructedPages::ConstantSubscriptsReachSeveralPages;
    }
    if valid_pages == 0 {
        return ConstructedPages::NoValidPages;
    }

    ConstructedPages::Constructed {
        pages,
        access: current,
    }
}

/// `agen::utils::orderMap` — `dcc/src/Dialect/Agen/Utils.cpp:87` (14L). Rewrites `map`'s dimensions
/// into the order `map_order` names, which is how a time address map is read in walk order.
///
/// ⛔ `map_order.getDimPosition(i)` IS A `cast`, so a `map_order` that is not a permutation aborts —
/// [`AffineMap::dim_position`]'s [`None`] leaves that dimension where it was, which is the identity
/// this crate's default-constructed orders already mean. ⚠️ THE TWO `DT_CHECK`s (`:88-89`) are
/// arities, and the callers hand over a `time_order` with no symbols by construction.
#[must_use]
pub(super) fn order_map(map: &AffineMap, map_order: &AffineMap) -> AffineMap {
    let ordered: Vec<AffineExpr> = (0..map_order.results.len())
        .map(|i| {
            AffineExpr::Dim(
                map_order
                    .dim_position(i)
                    .unwrap_or(u32::try_from(i).unwrap_or(0)),
            )
        })
        .collect();
    map.replace_dims_and_symbols(&ordered, &[], map_order.dims, 0)
}

/// `agen::utils::replaceConstOpsInSubscriptsMap` — `dcc/src/Dialect/Agen/Utils.cpp:46` (20L), with
/// `removeConstantOpsFromIndices` (`:33-44`) inlined as its closing statement.
///
/// ⛔ TWO RENUMBERINGS IN ONE PASS: a constant operand becomes an `AffineConstantExpr` and is
/// DROPPED from `indices`, and every surviving operand is renumbered densely — so `indices[dim]` and
/// `d<dim>` still line up afterwards. Folding without dropping, or dropping without renumbering,
/// leaves the map indexing the wrong operand.
pub(super) fn replace_const_ops_in_subscripts_map(
    subscripts_map: &mut AffineMap,
    indices: &mut Vec<Val>,
    scope: &[DfirOp],
) {
    let constant_at = |index: Val| match defining_op(index, scope) {
        // `dyn_cast_or_null<arith::ConstantOp>` then `cast<IntegerAttr>(getValue()).getInt()`.
        Some(DfirOp::Arith(arith::Op::Constant { value, .. })) => Some(*value),
        _ => None,
    };

    let mut operand_exprs: Vec<AffineExpr> = Vec::with_capacity(indices.len());
    let mut num_dim_vars = 0u32;
    for &index in indices.iter() {
        match constant_at(index) {
            Some(value) => operand_exprs.push(AffineExpr::Const(value)),
            None => {
                operand_exprs.push(AffineExpr::Dim(num_dim_vars));
                num_dim_vars += 1;
            }
        }
    }
    *subscripts_map = subscripts_map.replace_dims_and_symbols(&operand_exprs, &[], num_dim_vars, 0);
    indices.retain(|&index| constant_at(index).is_none());
}

/// `agen::CompositeLoadOp::cloneWithNewAccessInfo` — `Agen.cpp:653` (33L).
///
/// ⛔ ONLY THE VIEW, ITS INDICES AND THE TIME SET ARE THE CALLER'S; the load side, `time_order` and
/// `time_addr_map` come from the original, and an absent `dbgName` becomes the EMPTY STRING (`:658`).
/// ⛔ AND THE BODY IS RE-BOUND onto a fresh `load_iv` (`:678-682`), as
/// [`super::tf_mutable_addr_splitting::clone_composite_with_new_access_info`] does for the transfer.
#[must_use]
pub(super) fn clone_composite_load_with_new_access_info(
    vals: &mut Values,
    access: &agen::CompositeAccess,
    view: Val,
    indices: Vec<Index>,
    time_set: IntegerSet,
) -> DfirOp {
    let load_iv = vals.mint();
    let mut mapping = ValueMapping::new();
    mapping.map(access.load_iv, load_iv);
    let body = vals.clone_ops(&access.body, &mut mapping);
    DfirOp::Agen(agen::Op::CompositeLoad(Box::new(agen::CompositeAccess {
        view,
        indices,
        view_ty: access.view_ty.clone(),
        load_iv,
        load_iv_ty: access.load_iv_ty,
        load_set: access.load_set.clone(),
        load_order: access.load_order.clone(),
        time_symbols: access.time_symbols.clone(),
        time_set,
        time_order: access.time_order.clone(),
        time_addr_map: access.time_addr_map.clone(),
        dbg_name: Some(access.dbg_name.clone().unwrap_or_default()),
        body,
    })))
}

/// THE COMPOSITE `createNewMemOp` VIRTUAL, DISPATCHED ON THE OP CLASS — the leaf overrides at
/// `Impl.cpp:1120`, `:1173` and `:1242`, each of which is a `cast<>` to its own class and nothing
/// else. ⚠️ UNANCHORED: all three are outside the 384, and entry 325 cannot call the virtual without
/// them.
#[derive(Debug, Clone)]
pub enum NewCompositeMemOp {
    /// The replacement, preceded by any non-paged view it had to clone for the other side.
    Created(Vec<DfirOp>),
    /// `agen.composite_store` — `TPMVCompositeStore` (`:1173`) has no island op to clone, for the
    /// reason [`TpmvCompositeStore::new`] gives.
    MemOpHasNoCompositeClass,
}

/// `TPMVCompositeLoad::createNewMemOp` (`:1120`) and `TPMVCompositeLoadStore::createNewMemOp`
/// (`:1242`) — see [`NewCompositeMemOp`].
///
/// ⛔ THE LOAD-AND-STORE ARM REPLACES ONLY `info.mem_index_`'S SIDE and clones the OTHER side's view
/// through [`clone_mem_view_if_non_paged`] (`:1252`, `:1259`), keeping that side's own map and
/// indices — so a `kDirDst` info leaves the src exactly as it was.
fn create_new_composite_mem_op(
    vals: &mut Values,
    mem_op: &DfirOp,
    info: &TpmvInfo<'_>,
    mem_view: Val,
    subscripts_map: &AffineMap,
    indices: &[Val],
    time_set: IntegerSet,
    scope: &[DfirOp],
) -> NewCompositeMemOp {
    let new_indices = indices_from_map(subscripts_map, indices);
    match mem_op {
        DfirOp::Agen(agen::Op::CompositeLoad(access)) => {
            NewCompositeMemOp::Created(vec![clone_composite_load_with_new_access_info(
                vals,
                access,
                mem_view,
                new_indices,
                time_set,
            )])
        }
        DfirOp::Agen(agen::Op::CompositeLoadAndStore(transfer)) => {
            let (other, side) = match info.mem_index {
                MemoryOperandIndex::DirSrc => (transfer.dst, false),
                _ => (transfer.src, true),
            };
            let cloned = clone_mem_view_if_non_paged(vals, defining_op(other, scope), other);
            let mut ops = cloned.ops;
            ops.push(if side {
                clone_composite_with_new_access_info(
                    vals,
                    transfer,
                    cloned.value,
                    transfer.src_indices.clone(),
                    mem_view,
                    new_indices,
                    time_set,
                )
            } else {
                clone_composite_with_new_access_info(
                    vals,
                    transfer,
                    mem_view,
                    new_indices,
                    cloned.value,
                    transfer.dst_indices.clone(),
                    time_set,
                )
            });
            NewCompositeMemOp::Created(ops)
        }
        _ => NewCompositeMemOp::MemOpHasNoCompositeClass,
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 325/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// THE EXPLICIT TIME NEST ENTRY 325 EMITTED, when some time dimension crosses a page boundary.
#[derive(Debug, Clone)]
pub struct ExplicitTimeNest {
    /// `for_ops` as one nested `affine.for`, outermost first, with `mem_ops_[0]`'s replacement in the
    /// innermost body — where `builder.setInsertionPointToStart(for_ops.back().getBody())` put it.
    pub nest: Vec<DfirOp>,
    /// The nest's induction variables, outermost first, as `indices_` now names them.
    pub ivs: Vec<Val>,
}

/// WHAT `transform_time` DID — see [`TransformedTime::TimeDimsPreserved`] for the common answer.
#[derive(Debug, Clone)]
pub enum TransformedTime {
    /// `explicit_loop_dim == -1`: *"all time dims can be preserved"* (`:1028`) and the `else` arm's
    /// only statement is a debug print. ⭐ THE ORDINARY CASE — a transfer whose pages do not move
    /// with time needs no loops at all.
    TimeDimsPreserved,
    /// `explicit_loop_dim > -1`: the loops, with `mem_ops_[0]` erased and re-pointed at the new op.
    ExplicitLoops(ExplicitTimeNest),
    /// `analyzeValidPages(..).failed()` (`:1021`).
    PagesNotAnalyzed(ValidPages),
    /// `DT_CHECK(mem_ops_.size() == 1)` (`:980`, `:1032`) and
    /// `DT_CHECK(tpmv_info_.size() == tpmv_comp_info_.size())` (`:981`).
    OperandListsDisagree {
        /// `mem_ops_.size()`.
        mem_ops: usize,
        /// `tpmv_info_.size()`.
        infos: usize,
        /// `tpmv_comp_info_.size()`.
        comp_infos: usize,
    },
    /// A subscript that is not an `affine.for` induction variable with constant bounds — entry 197's
    /// own aborts, through [`SubscriptIv::of`], naming the operand.
    SubscriptIsNotALoopIterator(usize),
    /// `access_details_[0]` is empty, so there are no time bounds to add.
    AccessDetailsAreEmpty,
    /// A time bound that is [`TimeBound::Coalesced`] or [`TimeBound::Variable`] where entry 131's
    /// `DT_CHECK_MSG` and `AffineForOp::create` both need a step count. See [`TimeSteps`].
    TimeBoundIsNotAStepCount(TimeBound),
    /// `DT_CHECK(subscripts_map_time[i].getNumDims() == indices_ranges_.size())` (`:1018-1019`).
    SubscriptsAndRangesDisagree {
        /// `subscripts_map_time[i].getNumDims()`.
        dims: u32,
        /// `indices_ranges_.size()`.
        ranges: usize,
    },
    /// `time_order_` or `time_set_` is the default-constructed null — entry 326 has not run.
    TimeInfoIsAbsent,
    /// `time_order.getResult(dim)` off the end (`Agen/Utils.cpp:489`).
    TimeOrderHasNoDimension(usize),
    /// `time_order.getDimPosition(dim)` on a result that is not a bare dimension (`:497`).
    TimeOrderIsNotAPermutation(usize),
    /// `DT_CHECK((i - num_orig_dims) < for_ops.size())` (`Agen/Utils.cpp:470`).
    TimeDimensionHasNoLoop(usize),
    /// `DT_CHECK(paged_idx >= 0)` (`:1060`) — no operand reads a paged view.
    NoOperandHasAPagedView,
    /// [`NewCompositeMemOp::MemOpHasNoCompositeClass`].
    MemOpHasNoCompositeClass,
}

impl<'p> TpmvComposite<'p> {
    /// Replaces: e325_transform_time
    ///
    /// **325/384** `TPMVComposite::transform_time` —
    /// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:977` (98L).
    ///
    /// ⛔ `explicit_loop_dim` IS ASSIGNED EVERY PAGED OPERAND AND ONLY THE LAST SURVIVES (`:1024`),
    /// while `page_dependent_time_syms_` accumulates across all of them — so a two-operand transfer's
    /// nest depth is the DST's answer over the UNION of both operands' page-dependent symbols.
    /// ⛔ AND `subscripts_map_time` IS BUILT FOR EVERY OPERAND, paged or not (`:994-1000`): the
    /// non-paged one needs it in phase B to have its own subscripts updated.
    /// ⚠️ `mem_ops_[0] = new_mem_op` is [`ExplicitTimeNest`], not a write: see [`TpmvBase`].
    pub fn transform_time(&mut self, vals: &mut Values, scope: &'p [DfirOp]) -> TransformedTime {
        // `DT_CHECK(mem_ops_.size() == 1); DT_CHECK(tpmv_info_.size() == tpmv_comp_info_.size());`
        if self.base.mem_ops.len() != 1 || self.base.tpmv_info.len() != self.tpmv_comp_info.len() {
            return TransformedTime::OperandListsDisagree {
                mem_ops: self.base.mem_ops.len(),
                infos: self.base.tpmv_info.len(),
                comp_infos: self.tpmv_comp_info.len(),
            };
        }
        let (Some(time_order), Some(time_set)) = (self.time_order.clone(), self.time_set.clone())
        else {
            return TransformedTime::TimeInfoIsAbsent;
        };

        // ── Phase A: per memory operand ───────────────────────────────────────────────────────────
        let mut explicit_loop_dim: Option<TimeDim> = None;
        let mut subscripts_map_time: Vec<AffineMap> = Vec::new();
        for i in 0..self.base.tpmv_info.len() {
            // `replaceConstOpsInSubscriptsMap(tpmv_info_[i].subscripts_map_, tpmv_info_[i].indices_)`.
            let info = &mut self.base.tpmv_info[i];
            replace_const_ops_in_subscripts_map(&mut info.subscripts_map, &mut info.indices, scope);

            // "Create a new set of subscripts considering the time steps."
            let time_map = order_map(&self.tpmv_comp_info[i].time_addr_map, &time_order);
            subscripts_map_time.push(concatenate_maps(&info.subscripts_map, &time_map));

            // `if (!tpmv_info_[i].paged_mem_view_) continue;` — the non-paged operand keeps the map
            // just built and skips the analysis entirely.
            if info.paged_mem_view.is_none() {
                continue;
            }

            // "Create a copy of the subscripts_map that represents loop iterators as symbols."
            let subscripts_map_sym = replace_dims_in_map_with_syms(&subscripts_map_time[i]);

            // "Ranges of the loop iterators are used to only choose pages within the loop iteration
            // space." — entry 197 over `indices_`, which no longer holds any constant.
            let mut ivs: Vec<SubscriptIv> = Vec::with_capacity(info.indices.len());
            for &index in &info.indices {
                let Some(iv) = SubscriptIv::of(index, scope) else {
                    return TransformedTime::SubscriptIsNotALoopIterator(i);
                };
                ivs.push(iv);
            }
            calculate_indices_ranges(&ivs, &mut info.indices_ranges);

            // "Since all memory accesses for a memory op use the same time bounds, just use the first
            // access_details_ to grab time bounds." — entry 131 appends after entry 197.
            let Some(first) = self.access_details.entries().first() else {
                return TransformedTime::AccessDetailsAreEmpty;
            };
            let mut time_bounds: Vec<TimeSteps> = Vec::with_capacity(first.time_bounds.len());
            for &bound in &first.time_bounds {
                let Some(steps) = TimeSteps::of(bound) else {
                    return TransformedTime::TimeBoundIsNotAStepCount(bound);
                };
                time_bounds.push(steps);
            }
            let info = &mut self.base.tpmv_info[i];
            add_time_dim_indices_ranges(&time_bounds, &mut info.indices_ranges);
            if subscripts_map_time[i].dims as usize != info.indices_ranges.len() {
                return TransformedTime::SubscriptsAndRangesDisagree {
                    dims: subscripts_map_time[i].dims,
                    ranges: info.indices_ranges.len(),
                };
            }

            let info = self.base.tpmv_info[i].clone();
            match self.analyze_valid_pages(&info, &subscripts_map_sym) {
                ValidPages::Found(_) => {}
                refusal => return TransformedTime::PagesNotAnalyzed(refusal),
            }

            // `identifyTimeDimForExplicitLoops(tpmv_info_[i].subscripts_map_.getNumDims())`.
            explicit_loop_dim = identify_time_dim_for_explicit_loops(
                &time_set,
                &self.page_dependent_time_syms,
                NonTimeDims(info.subscripts_map.dims),
            );
        }

        // "A explicit_loop_dim of -1 indicates all time dims can be preserved."
        let Some(explicit_loop_dim) = explicit_loop_dim else {
            return TransformedTime::TimeDimsPreserved;
        };
        let time_dim = explicit_loop_dim.0 as usize;

        // ── Phase B: the dims from the outermost down to `explicit_loop_dim` become real loops ────
        //
        // `constructExplicitTimeLoops(mem_ops_[0], access_details_[0].getTimeBounds(),
        //  explicit_loop_dim)` — the bounds first, since the nest is assembled around the new op.
        let Some(first) = self.access_details.entries().first() else {
            return TransformedTime::AccessDetailsAreEmpty;
        };
        let mut trip_counts: Vec<i64> = Vec::with_capacity(time_dim + 1);
        for &bound in first.time_bounds.iter().take(time_dim + 1) {
            let Some(steps) = TimeSteps::of(bound) else {
                return TransformedTime::TimeBoundIsNotAStepCount(bound);
            };
            trip_counts.push(steps.trip_count());
        }
        if trip_counts.len() != time_dim + 1 {
            return TransformedTime::TimeDimensionHasNoLoop(time_dim);
        }
        let ivs: Vec<Val> = trip_counts.iter().map(|_| vals.mint()).collect();

        // `updateTimeSetForExplicitDims(explicit_loop_dim, time_order_, time_set_)` — IN PLACE, so
        // the new mem op below is cloned with the narrowed set.
        let mut explicit_dims: Vec<(AffineExpr, u32)> = Vec::with_capacity(time_dim + 1);
        for dim in 0..=time_dim {
            let Some(result) = time_order.results.get(dim) else {
                return TransformedTime::TimeOrderHasNoDimension(dim);
            };
            let Some(position) = time_order.dim_position(dim) else {
                return TransformedTime::TimeOrderIsNotAPermutation(dim);
            };
            explicit_dims.push((result.clone(), position));
        }
        let time_set = update_time_set_for_explicit_dims(&explicit_dims, &time_set, &time_order);
        self.time_set = Some(time_set.clone());

        for (i, map_time) in subscripts_map_time.iter().enumerate() {
            let info = &mut self.base.tpmv_info[i];
            let num_orig_dims = info.subscripts_map.dims as usize;
            let Some((subscripts_map, indices)) =
                update_subscripts_and_indices_for_explicit_time_loops(
                    &ivs,
                    time_dim,
                    num_orig_dims,
                    map_time,
                    &info.indices,
                )
            else {
                return TransformedTime::TimeDimensionHasNoLoop(time_dim);
            };
            info.subscripts_map = subscripts_map;
            info.indices = indices;
        }

        // "Replace mem_ops_ with an updated one that reflects the time dims being removed." — the
        // FIRST operand with a paged view, whatever its `mem_index_`.
        let Some(paged_idx) = self
            .base
            .tpmv_info
            .iter()
            .position(|info| info.paged_mem_view.is_some())
        else {
            return TransformedTime::NoOperandHasAPagedView;
        };
        let info = &self.base.tpmv_info[paged_idx];
        let Some(handle) = info.paged_mem_view else {
            return TransformedTime::NoOperandHasAPagedView;
        };
        let created = create_new_composite_mem_op(
            vals,
            self.base.mem_ops[0],
            info,
            handle.view.result,
            &info.subscripts_map,
            &info.indices,
            time_set,
            scope,
        );
        let NewCompositeMemOp::Created(ops) = created else {
            return TransformedTime::MemOpHasNoCompositeClass;
        };

        // `mem_ops_[0]->erase(); mem_ops_[0] = new_mem_op;` — the erase is the caller's, as
        // [`TpmvBase`] records; the new op goes in the innermost body the builder was pointing at.
        TransformedTime::ExplicitLoops(ExplicitTimeNest {
            nest: nest_explicit_time_loops_over(ops, &ivs, &trip_counts),
            ivs,
        })
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 326/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHAT `initialize_time` DID — the two `emitError`s (`:1103`, `:1110`) plus the `DT_CHECK`s.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompositeTimeInit {
    /// `access_details_`, `time_order_`, `time_set_` and `tpmv_comp_info_` are all set.
    Initialized,
    /// `DT_CHECK(mem_ops_.size() == 1)` and `DT_CHECK(op)` on the `dyn_cast<agen::CompositeLoadOp>`.
    MemOpIsNotACompositeLoad,
    /// `emplace_insert` on a slot already taken — see [`VacantSlot`].
    AccessSlotUnavailable(MemoryOperandIndex),
    /// `"Unable to construct details"`.
    DetailsNotConstructed(ConstructedDetails),
    /// `"Unable to construct time step info"`.
    TimeStepsNotConstructed(TimeStepsInfo),
}

impl<'p> TpmvCompositeLoad<'p> {
    /// Replaces: e326_initialize_time
    ///
    /// **326/384** `TPMVCompositeLoad::initialize_time` —
    /// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:1095` (23L).
    ///
    /// ⛔ THE TIME STEPS ARE BUILT WITHOUT COALESCING AND WITHOUT BURST/IL CALCS (`:1107-1108`): entry
    /// 325 needs `time_bounds_` as the op wrote them, and a coalesced dimension carries
    /// [`TimeBound::Coalesced`] where [`TimeSteps`] needs a count. ⛔ AND IT STOPS AT THE FIRST
    /// REFUSAL, unlike `MutableAddrSplitting`'s composite prologue, which runs both and reports the
    /// first (`MutableAddrSplitting.cpp:459-465`).
    /// ⚠️ `time_set_` is the op's, and entry 325 narrows it in place once loops become explicit.
    pub fn initialize_time(&mut self, scope: &'p [DfirOp]) -> CompositeTimeInit {
        // `DT_CHECK(mem_ops_.size() == 1); auto op = dyn_cast<agen::CompositeLoadOp>(mem_ops_[0]);`
        let comp = &mut self.composite;
        let [DfirOp::Agen(agen_op)] = comp.base.mem_ops[..] else {
            return CompositeTimeInit::MemOpIsNotACompositeLoad;
        };
        let agen::Op::CompositeLoad(access) = agen_op else {
            return CompositeTimeInit::MemOpIsNotACompositeLoad;
        };

        // `access_details_.emplace_insert(kDirSrc, op, comp_)` — a composite load reads, so its one
        // operand is the source however the view is paged.
        let moi = MemoryOperandIndex::DirSrc;
        let Some(slot) = comp.access_details.vacancy(moi) else {
            return CompositeTimeInit::AccessSlotUnavailable(moi);
        };
        let ad = slot.emplace_insert(AccessDetailsAffineComposite::new(agen_op, comp.base.comp));
        let details = ad.construct_details(moi, scope);
        if details != ConstructedDetails::Complete {
            return CompositeTimeInit::DetailsNotConstructed(details);
        }

        // "Construct the time steps info without coalescing and without burst/IL calcs."
        let steps = construct_time_steps_info(&mut comp.access_details, false, false, scope);
        if !steps.constructed() {
            return CompositeTimeInit::TimeStepsNotConstructed(steps);
        }

        comp.time_order = Some(access.time_order.clone());
        comp.time_set = Some(access.time_set.clone());
        // `tpmv_comp_info_.emplace_back(op.getTimeAddrMap(), access_details_[0])`.
        comp.tpmv_comp_info.push(TpmvCompositeInfo {
            time_addr_map: access.time_addr_map.clone(),
            access_details: moi,
        });

        CompositeTimeInit::Initialized
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 356/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHAT ONE PAGED MEMORY OPERAND'S DE-PAGING PRODUCED.
#[derive(Debug, Clone)]
pub struct TransformedOperand<'p> {
    /// `info.mem_index_` — which memory operand this was.
    pub mem_index: MemoryOperandIndex,
    /// Every valid page, in `getIdxSets()` order, as [`analyze_and_construct_valid_pages`] built it.
    pub pages: Vec<ConstructedPage>,
    /// `for (auto &mem_op : mem_ops_) eraseMemOpAndUseChain(mem_op);` (`:618`), consumer-first —
    /// a delete list and not an erase, for [`TpmvBase::erase_mem_op_and_use_chain`]'s reason.
    pub erased: Vec<&'p DfirOp>,
    /// `info.paged_mem_view_->erase()` (`:621`), by the value it bound: a view a cloned nest
    /// re-pointed lives in the emission and not in the input program — see [`PagedAccess`].
    pub erased_mem_view: Val,
}

/// WHAT `transform` DID.
#[derive(Debug, Clone)]
pub enum TransformedPagedViews<'p> {
    /// `LogicalResult::success()`.
    Transformed {
        /// In `tpmv_info_` order, skipping every operand whose view was not paged.
        operands: Vec<TransformedOperand<'p>>,
        /// `mem_ops_` as the last paged operand left it — every page's replacement, in page order.
        mem_ops: Vec<Val>,
    },
    /// `if (analyzeAndConstructValidPages(..).failed()) return LogicalResult::failure();` (`:614`).
    PagesNotConstructed(ConstructedPages),
    /// A subscript entry 197 aborts on, naming the operand — see [`SubscriptIv`].
    SubscriptIsNotALoopIterator(usize),
}

impl<'p> TpmvBase<'p> {
    /// Replaces: e356_transform
    ///
    /// **356/384** `TPMVBase::transform` —
    /// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:592` (39L).
    ///
    /// ⛔ `mem_ops_ = new_mem_ops` IS LOOP-CARRIED: a second paged operand builds its pages over the
    /// FIRST one's replacements, which is why `new_mem_ops` is a local per operand and `mem_ops_` is
    /// not. The write itself is the returned `mem_ops`, as entry 325's `mem_ops_[0] = new_mem_op` is
    /// [`ExplicitTimeNest`]. ⚠️ AND THE SIBLING RE-SYNC AT `:518-521` LANDS HERE —
    /// [`ConstructedPage::ir_maps`] holds it for the caller, and this is that caller.
    #[must_use]
    pub fn transform(
        &mut self,
        vals: &mut Values,
        scope: &'p [DfirOp],
        emitted: &mut Vec<DfirOp>,
    ) -> TransformedPagedViews<'p> {
        // `mem_ops_`, each by the value it binds — see [`PagedAccess`]. ⛔ A STORE BINDS NOTHING, so
        // `TPMVVectorStore`'s op cannot be named here; that is [`PagedAccess`]'s standing seam.
        let mut mem_ops: Vec<Val> = self.mem_ops.iter().flat_map(|&op| results(op)).collect();
        let mut operands: Vec<TransformedOperand<'p>> = Vec::new();

        // `for (auto &info : tpmv_info_)`, by position: the sibling re-sync below reaches the whole
        // vector while this entry is being rewritten.
        for i in 0..self.tpmv_info.len() {
            // `if (!info.paged_mem_view_) continue;`
            let Some(handle) = self.tpmv_info[i].paged_mem_view else {
                continue;
            };
            let mem_index = self.tpmv_info[i].mem_index;

            // `dcc::agen::utils::replaceConstOpsInSubscriptsMap(info.subscripts_map_, info.indices_)`.
            let info = &mut self.tpmv_info[i];
            replace_const_ops_in_subscripts_map(&mut info.subscripts_map, &mut info.indices, scope);

            // "Create a copy of the subscripts_map that represents loop iterators as symbols. This is
            // used to form the constraints to determine which pages are valid for mem_ops_."
            let subscripts_map_sym = replace_dims_in_map_with_syms(&info.subscripts_map);

            // "Ranges of the loop iterators are used to only choose pages within the loop iteration
            // space." — `calculateIndicesRanges(info.indices_, info.indices_ranges_)`, entry 197,
            // whose three aborts are [`SubscriptIv::of`]'s one door.
            let Some(ivs) = SubscriptIv::all_of(&info.indices, scope) else {
                return TransformedPagedViews::SubscriptIsNotALoopIterator(i);
            };
            calculate_indices_ranges(&ivs, &mut info.indices_ranges);

            // `SmallVector<Operation *, 16> new_mem_ops;` — this operand's own, so it starts empty,
            // and `analyzeAndConstructValidPages(new_mem_ops, info, subscripts_map_sym)`.
            let seed = PagedAccess {
                mem_ops: mem_ops.clone(),
                paged_mem_view: handle.result,
                subscripts_map: info.subscripts_map.clone(),
                indices: info.indices.clone(),
                indices_ranges: info.indices_ranges.clone(),
                conditional_iter_args: info.conditional_iter_args.clone(),
            };
            let info = self.tpmv_info[i].clone();
            let constructed =
                analyze_and_construct_valid_pages(vals, emitted, &seed, &info, &subscripts_map_sym);
            let ConstructedPages::Constructed { pages, access } = constructed else {
                return TransformedPagedViews::PagesNotConstructed(constructed);
            };

            // `info.indices_ = indices;` (`:497`) and `info.conditional_iter_args_` (`:481`) — what
            // the page walk wrote back into `info`, threaded out through [`PagedAccess`].
            let info = &mut self.tpmv_info[i];
            info.indices = access.indices.clone();
            info.conditional_iter_args = access.conditional_iter_args.clone();

            // "The other TPMVInfo objects need to be synced as the values they use for paged_mem_view
            // or indices may have changed as a result of the clones." (`:518-521`) — per cloned loop,
            // in the order `createIterArgsForConditionals` cloned them.
            for page in &pages {
                for ir_map in &page.ir_maps {
                    for sibling in &mut self.tpmv_info {
                        if sibling.mem_index != mem_index {
                            update_tpmv_info(sibling, ir_map, scope);
                        }
                    }
                }
            }

            // `for (auto &mem_op : mem_ops_) eraseMemOpAndUseChain(mem_op);` — over the re-pointed
            // list, dispatched on the class the manager chose: entry 129's override for an
            // `agen.vector_load`, the base's one-op answer (entry 135) for anything else.
            // ⛔ A MEM OP THIS TRANSFORM CREATED IS NOT IN `scope`, and its removal is its absence
            // from the emission — the seam [`PagedAccess`] already carries.
            let mut erased: Vec<&'p DfirOp> = Vec::new();
            for &mem_op in &access.mem_ops {
                let Some(op) = defining_op(mem_op, scope) else {
                    continue;
                };
                match VectorLoadOp::of(op) {
                    Some(load) => erased.extend(erase_vector_load_and_use_chain(load, scope)),
                    None => erased.extend(self.erase_mem_op_and_use_chain(op)),
                }
            }

            // "There will only be one paged_mem_view per TPMVInfo object at any given time because we
            // don't clone paged_mem_views during the lowering process." —
            // `info.paged_mem_view_->erase(); info.paged_mem_view_ = nullptr;`
            self.tpmv_info[i].paged_mem_view = None;

            // "Re-populate mem_ops_ with the new memory ops we created. Because we could be
            // introducing conditionals for multiple memory operands, the mem_op may get cloned into
            // multiple conditional branches." — `mem_ops_ = new_mem_ops;`
            mem_ops = pages
                .iter()
                .flat_map(|page| page.new_mem_ops.iter().map(|new| new.result))
                .collect();

            operands.push(TransformedOperand {
                mem_index,
                pages,
                erased,
                erased_mem_view: access.paged_mem_view,
            });
        }

        // `return LogicalResult::success();`
        TransformedPagedViews::Transformed { operands, mem_ops }
    }
}

// ⛔ RE-CREATED ANCHORS. These units' `crustify:todo:` markers were deleted without a
// `/// Replaces:` ever appearing, which removed them from every later schedule and let the
// driver report the campaign DONE. Outstanding work is now computed from UNITS.tsv.
// crustify:todo: e373_run
