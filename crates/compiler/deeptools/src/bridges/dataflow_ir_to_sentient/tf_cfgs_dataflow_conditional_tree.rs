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

//! `CFGSDataflowConditionalTree.cpp` — 18 of bridge 2's 384 functions (dependency level(s) [0, 1, 2, 3, 6, 7, 8]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e095_isOperationSelected` | 095/384 | 2 | `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:34` |
//! | `e096_createDummyYieldInElseReg` | 096/384 | 13 | `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:383` |
//! | `e097_getNewDbgNameFromList` | 097/384 | 2 | `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:456` |
//! | `e098_getLhsRhsOfEQPredicate` | 098/384 | 11 | `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:519` |
//! | `e099_ConditionalSimplificationManager` | 099/384 | 2 | `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.hpp:78` |
//! | `e100_TransformationConditionalTree` | 100/384 | 5 | `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.hpp:111` |
//! | `e176_opHasSideEffect` | 176/384 | 13 | `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:38` |
//! | `e177_mergeShallow` | 177/384 | 60 | `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:398` |
//! | `e244_isHoistable` | 244/384 | 9 | `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:82` |
//! | `e284_hoistCommonConditionals` | 284/384 | 96 | `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:94` |
//! | `e285_replaceIfOpByIterArg` | 285/384 | 56 | `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:619` |
//! | `e347_topLevelConditionsMatch` | 347/384 | 31 | `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:279` |
//! | `e348_singleOpBranchToYieldVal` | 348/384 | 18 | `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:498` |
//! | `e349_isLoopInvariant` | 349/384 | 66 | `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:681` |
//! | `e370_areShallowlyMergeable` | 370/384 | 34 | `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:343` |
//! | `e371_hoistLoopInvariantConditionals` | 371/384 | 42 | `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:750` |
//! | `e380_shallowlyMergeConditionals` | 380/384 | 76 | `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:198` |
//! | `e381_simplifyValueBasedConditionals` | 381/384 | 30 | `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:466` |
//!
//! Original files homed here: `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp`, `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.hpp`

use super::tf_program_units_reduction::HighPreference;
use super::tf_utils::{CountedLoop, Iterations, create_for_op_with_additional_return_value};
use super::vc_vector_chain_helper::ops_are_equivalent;
use crate::arch::Arch;
use crate::islands::dataflow_ir::ProgramUnit;
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::dialects::{
    self as dfir_op, Op as DfirOp, Val, affine, arith, dataflow, scf, symbol,
};
use crate::islands::dataflow_ir::ty::ScalarTy;

/// WHICH TRANSFORM MERGED THE OPERATIONS — the prefix a merged `dbgName` opens with.
///
/// # ⛔ A CLOSED SET, NOT THE REFERENCE'S `std::string`
///
/// `getNewDbgNameFromList`'s first parameter is a `std::string`, and all ten of its call sites pass
/// one of nine literals — every one of them ending in an opening bracket that the callee never
/// checks for and always closes. That asymmetry is the whole reason a debug name can come out
/// unbalanced, so here the caller names the transform and [`new_dbg_name_from_list`] writes both
/// brackets itself. Same string, one fewer way to get it wrong, and the crate's rule that a closed
/// set is an `enum` rather than a string.
///
/// ⚠️ Only [`Self::Cfgsm`] is reachable from bridge 2's 384 (entry 177, `mergeShallow`). The other
/// eight are cited so that the family is legible and so the next porter of one of those passes
/// finds the prefix already named rather than typing a fresh literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DbgNamePrefix {
    /// `"CFGSM("` — CFG simplification's shallow merge of two conditionals
    /// (`CFGSDataflowConditionalTree.cpp:456`, entry 177).
    Cfgsm,
    /// `"CFGDM("` — the deep merge of two conditionals
    /// (`Sentient/Analyses/CFGDeepMergingConditionalTree.cpp:103` and `:295`).
    Cfgdm,
    /// `"RRC("` — a redundant conditional folded into its parent
    /// (`Sentient/RemoveRedundantConditionals.cpp:209`).
    Rrc,
    /// `"LM("` — two loops merged into one (`Sentient/LoopMerging.cpp:266`).
    Lm,
    /// `"LC("` — a loop nest coalesced into its outermost loop
    /// (`Sentient/LoopCoalescing.cpp:281`). ⭐ THE ONE SITE THAT PASSES MORE THAN TWO OPERATIONS: it
    /// builds `loop_op_list` from the whole nest.
    Lc,
    /// `"SAFF("` — a store fused with the forward that feeds it
    /// (`Sentient/StoreAndForwardFusion.cpp:540`).
    Saff,
    /// `"SSRF("` — two syncs fused into one (`Sentient/SyncSendRecvFusion.cpp:93`).
    Ssrf,
    /// `"MergeOp("` — two dataflow values merged (`PCFGToDataflowIR.cpp:1773`).
    MergeOp,
    /// `"PackOp("` — two vectors packed into one (`PCFGToDataflowIR.cpp:3953`).
    PackOp,
}

impl DbgNamePrefix {
    /// THE TRANSFORM'S TAG, WITHOUT THE BRACKET — see the type's own note on why.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            DbgNamePrefix::Cfgsm => "CFGSM",
            DbgNamePrefix::Cfgdm => "CFGDM",
            DbgNamePrefix::Rrc => "RRC",
            DbgNamePrefix::Lm => "LM",
            DbgNamePrefix::Lc => "LC",
            DbgNamePrefix::Saff => "SAFF",
            DbgNamePrefix::Ssrf => "SSRF",
            DbgNamePrefix::MergeOp => "MergeOp",
            DbgNamePrefix::PackOp => "PackOp",
        }
    }
}

/// Replaces: e097_getNewDbgNameFromList
///
/// **097/384** `dataflow::utils::getNewDbgNameFromList` — `dialect_utils/Dataflow/Utils.cpp:167`
/// (19L), called from `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:455`.
///
/// ```cpp
/// mlir::StringAttr getNewDbgNameFromList(
///     std::string prefix, llvm::SmallVector<mlir::Operation *> list_of_ops) {
///   DT_CHECK_MSG(!list_of_ops.empty(), "Expect non-empty list of operations");
///   std::string new_dbg_name = prefix;
///   bool is_first_op = true;
///   for (mlir::Operation *op : list_of_ops) {
///     DT_CHECK_MSG(op, "Expect valid op");
///     const auto dbg_name_attr = dataflow::getDbgNameAttr(op);
///     if (dbg_name_attr == nullptr) return nullptr;
///     if (is_first_op) {
///       is_first_op = false;
///       new_dbg_name += dbg_name_attr.getValue().str();
///     } else {
///       new_dbg_name += ", " + dbg_name_attr.getValue().str();
///     }
///   }
///   new_dbg_name += ")";
///   return mlir::StringAttr::get(list_of_ops.front()->getContext(), new_dbg_name);
/// }
/// ```
///
/// # ⛔⛔ THE LEDGER'S CITATION IS THE CALL SITE, AND THE CALL SITE IS TWO LINES OF ANOTHER FUNCTION
///
/// `crustify-bridge2/UNITS.tsv` gives entry 097 as `CFGSDataflowConditionalTree.cpp:456` and the
/// extract's body for it is
///
/// ```cpp
/// dataflow::utils::getNewDbgNameFromList("CFGSM(", {src, dst})) {
///   dataflow::setDbgNameAttr(dst, new_dbg_name_attr);
/// ```
///
/// — the tail of `mergeShallow`'s `if`, which is entry 177's last three lines and not a function at
/// all. ⭐ THE FUNCTION OF THAT NAME LIVES IN `dialect_utils/Dataflow/Utils.cpp:167`, and since
/// `dialect_utils/` contributes nothing else to the 384, entry 097 is this campaign's only claim on
/// it. It is ported here whole, from the authority, and entry 177 will call it.
///
/// # ⛔ THE NON-EMPTY LIST IS THE HEAD AND THE REST, WHICH IS WHY THERE IS NO CHECK LEFT
///
/// `DT_CHECK_MSG(!list_of_ops.empty(), ...)` guards the `list_of_ops.front()` on the last line. A
/// head parameter beside the rest says the same thing at compile time — the crate's own precedent is
/// `Units`/`ProgramUnits`, whose heads exist for exactly this reason — and it also retires
/// `is_first_op`: the separator belongs to the tail, so the flag is the loop's shape rather than a
/// variable it carries. `DT_CHECK_MSG(op, "Expect valid op")` goes the same way: a `&str` is not
/// null.
///
/// # ⛔ AN UNNAMED OPERATION ANYWHERE IN THE LIST ABANDONS THE WHOLE NAME
///
/// `dbgName` is a *discardable* attribute, so `getDbgNameAttr` returns null for any op that has none
/// (`DataflowOpInterfaces.cpp:24-36`) — hence `Option<&str>` per operation rather than `&str`. The
/// early `return nullptr` throws away the partially built string, so the result is all-or-nothing and
/// `?` is exactly it. ⭐ AND SEVEN OF THE TEN CALLERS TREAT IT THAT WAY: `if (StringAttr n =
/// getNewDbgNameFromList(...)) setDbgNameAttr(dst, n);`, which leaves the destination's existing name
/// alone rather than clearing it — `setDbgNameAttr(dst, nullptr)` would REMOVE the attribute
/// (`DataflowOpInterfaces.cpp:49`), so the guard is load-bearing.
///
/// ⚠️ THE OTHER THREE HAND THE NULL STRAIGHT TO A `create()`. `SyncSendRecvFusion.cpp:93`,
/// `PCFGToDataflowIR.cpp:1773` and `:3953` pass the result to `SyncOp::create` / `MergeOp::create` /
/// `PackOp::create` as the new op's dbg-name attribute with no guard at all, so there `None` means
/// the op being BUILT is unnamed. Either reading is `Option<String>`; neither is a refusal.
///
/// ⚠️ THE `MLIRContext` IS DROPPED, and it is the one thing here that is pure mechanism:
/// `StringAttr::get(list_of_ops.front()->getContext(), s)` interns the string in the context that
/// already owns the ops. A `String` needs no interner. This is why the head is still worth naming
/// even though the string no longer needs `front()`.
///
/// ⭐ WHY A `String` AND NOT A NEWTYPE: a debug name is free text the scheduler wrote — `"in-7/6"`,
/// `"SCF-If #1"`, `"condition__3"` — and the SentientIR island already carries it as
/// `dbg_name: Option<String>`, so a newtype here would only be converted away at the one call site.
#[must_use]
pub fn new_dbg_name_from_list(
    prefix: DbgNamePrefix,
    first: Option<&str>,
    rest: &[Option<&str>],
) -> Option<String> {
    let mut new_dbg_name = String::from(prefix.spelling());
    new_dbg_name.push('(');
    new_dbg_name.push_str(first?);
    for dbg_name_attr in rest {
        let dbg_name_attr = (*dbg_name_attr)?;
        new_dbg_name.push_str(", ");
        new_dbg_name.push_str(dbg_name_attr);
    }
    new_dbg_name.push(')');
    Some(new_dbg_name)
}

/// THE TWO SIDES OF THE EQUALITY A CONDITIONAL BRANCHES ON.
///
/// ⭐ ONE VALUE WHERE THE REFERENCE HAS A `bool` AND TWO OUT-PARAMETERS. `getLhsRhsOfEQPredicate`
/// leaves `lhs` and `rhs` untouched when it returns false, so every caller has to remember that its
/// two `Value`s mean nothing on that path; `Option` says it once, in the type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EqPredicate {
    /// `cmpi_op.getLhs()` — the value the conditional's subtree must have in common
    /// (`CFGSDataflowConditionalTree.hpp:38-40`), which the manager then expects to be a loop's
    /// induction variable.
    pub lhs: Val,
    /// `cmpi_op.getRhs()` — the constant the iteration is tested against.
    pub rhs: Val,
}

/// Replaces: e098_getLhsRhsOfEQPredicate
///
/// **098/384** `ConditionalSimplificationManager::getLhsRhsOfEQPredicate` —
/// `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:519` (11L).
///
/// ```cpp
/// bool ConditionalSimplificationManager::getLhsRhsOfEQPredicate(Operation *op,
///                                                               Value &lhs,
///                                                               Value &rhs) {
///   DT_CHECK_MSG(op, "Expect valid Operation.");
///   auto if_op = llvm::dyn_cast<mlir::scf::IfOp>(op);
///   if (!if_op) return false;
///   auto cond = if_op.getCondition();
///   auto cmpi_op = cond.getDefiningOp<mlir::arith::CmpIOp>();
///   if (!cmpi_op || cmpi_op.getPredicate() != mlir::arith::CmpIPredicate::eq)
///     return false;
///   lhs = cmpi_op.getLhs();
///   rhs = cmpi_op.getRhs();
///   return true;
/// }
/// ```
///
/// ⛔ THE EXTRACT STOPS AT `lhs = cmpi_op.getLhs();` — it dropped `rhs = cmpi_op.getRhs();` and the
/// `return true`, which is half the answer and all of the success path
/// (`crustify-bridge2/source/bridge2.cpp:1469-1481`). Ported from the authority.
///
/// # ⭐ WHAT THIS ANSWERS, AND FOR WHOM
///
/// It is the entry test of the value-based simplification: a chain of `scf.if (iv == k)` each
/// yielding a constant is an arithmetic sequence in disguise, and can become one iteration argument
/// instead of a tree of branches (entry 285, `replaceIfOpByIterArg`). The manager's constructor calls
/// it to learn the common left-hand side (`CFGSDataflowConditionalTree.hpp:57-61`), and
/// `parseConditional` calls it again per level. `simplify-conditional.mlir:115` and `:133` are one
/// such input — `%30 = arith.cmpi eq, %26, %0 : index` feeding `%39 = scf.if %30 -> (index)`, so this
/// returns `lhs = %26` (the loop's induction variable) and `rhs = %0`.
///
/// # ⛔ THE PREDICATE TEST IS REAL AGAIN
///
/// `cmpi_op.getPredicate() != eq` could not fail against this island a commit ago, because
/// `arith::Op::Compare` printed a hard-coded `eq` and had no predicate to disagree with. That is the
/// input the island could not express, so the island grew one
/// ([`arith::CmpIPredicate`]) — the campaign brief's rule, not a liberty. Both of this function's
/// exits are now reachable from a value an emitter can build.
///
/// # ⚠️ `scope` IS THE DEF-USE WALK, WHICH IS MECHANISM
///
/// `cond.getDefiningOp<CmpIOp>()` is MLIR asking a value which operation bound it, and returns null
/// both when nothing did (`cond` is a block argument) and when what did is not an `arith.cmpi`. This
/// island has no use lists, so the caller passes the ops in scope and the search is a scan. An SSA
/// value is bound exactly once, so "find the binder and check what it is" and "find a binder that is
/// an `arith.cmpi`" agree on every well-formed program; the scan stops at the first hit either way.
#[must_use]
pub fn lhs_rhs_of_eq_predicate(op: &DfirOp, scope: &[DfirOp]) -> Option<EqPredicate> {
    // `llvm::dyn_cast<mlir::scf::IfOp>(op)`: anything that is not a conditional declines. The
    // `DT_CHECK_MSG(op, ...)` above it is discharged by the reference being a `&`.
    let DfirOp::Scf(scf::Op::If { cond, .. }) = op else {
        return None;
    };
    // `cond.getDefiningOp<mlir::arith::CmpIOp>()`, then the predicate.
    let arith::Op::Compare {
        predicate: arith::CmpIPredicate::Eq,
        lhs,
        rhs,
        ..
    } = defining_cmpi(*cond, scope)?
    else {
        return None;
    };
    Some(EqPredicate {
        lhs: *lhs,
        rhs: *rhs,
    })
}

/// THE `arith.cmpi` IN `scope` THAT BINDS `cond`, IF THE OP THAT BINDS IT IS ONE.
///
/// ⚠️ This is `getDefiningOp<CmpIOp>()` and nothing more — see [`lhs_rhs_of_eq_predicate`]'s note on
/// why a scan stands in for a use list.
fn defining_cmpi(cond: Val, scope: &[DfirOp]) -> Option<&arith::Op> {
    scope.iter().find_map(|op| match op {
        DfirOp::Arith(cmpi_op @ arith::Op::Compare { result, .. }) if *result == cond => {
            Some(cmpi_op)
        }
        _ => None,
    })
}

/// ONE VALUE A BRANCH OF A VALUE-BASED CONDITIONAL YIELDS.
///
/// ⭐ AN `index` CONSTANT, WHICH IS WHY IT IS NOT A BARE `i64`:
/// `singleOpBranchToYieldVal` (entry 348) reads it with `arith::ConstantIndexOp::value()` and refuses
/// a branch whose single yielded operand is bound by anything else
/// (`CFGSDataflowConditionalTree.cpp:498-516`). The sequence these form is described by a lower bound
/// and a stride, and becomes a loop iteration argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct YieldedIndex(pub i64);

/// Replaces: e099_ConditionalSimplificationManager
///
/// **099/384** `ConditionalSimplificationManager::~ConditionalSimplificationManager` —
/// `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.hpp:78` (2L).
///
/// ```cpp
/// // Maps iteration space to values yielded by the conditional.
/// // All entries are initialized to nullopt.
/// std::optional<int64_t> *val_array_ = nullptr;
/// // ...
/// ~ConditionalSimplificationManager() {
///   if (val_array_) delete[] val_array_;
/// }
/// ```
///
/// # ⭐ WHAT A DESTRUCTOR PORTS TO IS OWNERSHIP, AND THE `if` IS THE `Option`
///
/// The scheduled unit is two lines, and between them they say three things about `val_array_`:
/// this object OWNS it, it may be ABSENT, and it is an ARRAY (`delete[]`, not `delete`). All three
/// are in the field's type here, and the deallocation itself is then drop glue with nothing left to
/// write — no `impl Drop`, which would be an empty stand-in and would also stop the value being
/// moved out of a field.
///
/// - **`Box<[T]>`, not `Vec<T>`** — `new std::optional<int64_t>[num_iterations]` allocates the
///   iteration space once and never grows it (`hpp:73-75`, from `std::get<4>(for_op_tuple_)`, the
///   loop's trip count). A `Box<[T]>` cannot `push`.
/// - **`Option<Box<..>>`, not an empty slice** — null is a state the reference distinguishes: it is
///   the constructor's two early returns, where `is_candidate_ = false` is set BEFORE any array
///   exists (`hpp:58-70`). "Not a candidate" and "a candidate whose iteration space is empty" are
///   different, and the destructor's own `if` is the proof that null is reachable.
/// - **`Option<YieldedIndex>` per slot** — `std::optional<int64_t>`, and *"All entries are
///   initialized to nullopt"* is what `new` does to a `std::optional` and what `Default` does here.
///
/// # ⛔⛔ AND THE REFERENCE HAS A LATENT DOUBLE FREE THAT THIS TYPE CANNOT REPRODUCE
///
/// `ConditionalSimplificationManager` owns a raw pointer and declares a destructor, but no copy
/// constructor and no copy assignment — the rule of three, unfollowed. A copy would shallow-copy
/// `val_array_` and the two objects would `delete[]` the same allocation. Nothing fires it today:
/// the only instance in the tree is a local built and dropped inside one loop iteration
/// (`CFGSDataflowConditionalTree.cpp:477`, `instance` in `simplifyValueBasedConditionals`). ⭐ HERE
/// THE HAZARD IS NOT AVAILABLE: the type is not `Copy`, and its derived `Clone` deep-copies the
/// boxed slice, so a clone owns its own array. That is a divergence in the reference's favour, and it
/// is the reason `Clone` is derived rather than suppressed.
///
/// ⚠️ THE OTHER SEVEN FIELDS ARE NOT HERE, AND NEITHER IS THE CONSTRUCTOR. `is_candidate_`,
/// `top_node_`, `if_op_`, `common_lhs_`, `for_op_tuple_`, `seq_lb_` and `seq_step_` are read only by
/// `parseConditional`, `replaceIfOpByIterArg` (entry 285) and `singleOpBranchToYieldVal` (entry 348),
/// and `top_node_` is a `CondNode *` — a base-class type from `dcc/src/Analysis/`, which contributes
/// nothing to the 384. The constructor at `hpp:55-76` is in no batch and on no exclusion list;
/// entry 099 is its destructor alone. Adding the fields now would mean inventing the ctor that fills
/// them, so they arrive with the units that read them (entry 381 constructs the manager).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConditionalSimplificationManager {
    /// `std::optional<int64_t> *val_array_` — one slot per iteration of the loop whose induction
    /// variable the conditional tests, each holding the value that iteration yields, if it is known.
    ///
    /// ⭐ `None` IS THE REFERENCE'S `= nullptr`, and so is [`Default`].
    pub val_array: Option<Box<[Option<YieldedIndex>]>>,
}

/// WHICH ANALYSIS AN EQUIVALENCE COMPARISON TRACES UNDER — `OperationEquivalence`'s `debug_`.
///
/// # ⭐ A TAG, NOT A MESSAGE
///
/// `debug_` is a `const char *` that only ever reaches `DEBUG_WITH_TYPE(debug_, ...)`
/// (`dcc/src/Analysis/OperationEquivalence.cpp`, 28 sites and no other use), which is LLVM's
/// per-type debug-stream filter: the string selects which `-debug-only=` name the trace appears
/// under. So it is drawn from a closed set — one name per analysis that owns a comparison — and the
/// crate's rule that a closed set is an `enum` applies exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EquivalenceTag {
    /// `"cfg-merging-and-hoisting-cond-tree"` — `CFGSDataflowConditionalTree`, this file
    /// (`CFGSDataflowConditionalTree.hpp:112`).
    CfgMergingAndHoistingCondTree,
    /// `"cfg-simplification-cond-tree"` — `CFGSSentientLevelConditionalTree`, the same transform one
    /// IR down (`Sentient/Analyses/CFGSSentientLevelConditionalTree.hpp:316`). ⭐ THE ONE SITE THAT
    /// TURNS THE CACHE OFF.
    CfgSimplificationCondTree,
    /// `"cfg-deep-merging-cond-tree"` — `CFGDeepMergingConditionalTree`
    /// (`Sentient/Analyses/CFGDeepMergingConditionalTree.hpp:29`).
    CfgDeepMergingCondTree,
    /// `"loop-merging"` — `LoopMerging`'s own `oe_`, tagged with its `PASS_NAME`
    /// (`Sentient/LoopMerging.cpp:24-25`, `:58`).
    LoopMerging,
    /// `"loop-absorption"` — likewise (`Sentient/LoopAbsorption.cpp:18-19`, `:43`).
    LoopAbsorption,
    /// `"program-units-reduction"` — `ProgramUnitsReduction`'s own comparison, the one site that
    /// passes a [`HighPreference`] (`Dataflow/ProgramUnitsReduction.cpp:136`).
    ///
    /// ⛔ THE TAG IS A LITERAL HERE, NOT `DEBUG_TYPE`. `ProgramUnitsReduction.cpp` defines no
    /// `DEBUG_TYPE` at all, so this string is the only name that comparison traces under.
    ProgramUnitsReduction,
}

impl EquivalenceTag {
    /// THE `-debug-only=` NAME.
    #[must_use]
    pub const fn spelling(self) -> &'static str {
        match self {
            EquivalenceTag::CfgMergingAndHoistingCondTree => "cfg-merging-and-hoisting-cond-tree",
            EquivalenceTag::CfgSimplificationCondTree => "cfg-simplification-cond-tree",
            EquivalenceTag::CfgDeepMergingCondTree => "cfg-deep-merging-cond-tree",
            EquivalenceTag::LoopMerging => "loop-merging",
            EquivalenceTag::LoopAbsorption => "loop-absorption",
            EquivalenceTag::ProgramUnitsReduction => "program-units-reduction",
        }
    }
}

/// HOW DEEP AN EQUIVALENCE COMPARISON GOES — `do_recursive_compare`.
///
/// ⛔ NOT A `bool`. The reference's own call sites write `/*do_recursive_compare*/ true` beside
/// `/*all_block_args_are_equiv*/ false` because two adjacent unnamed booleans are unreadable, and a
/// comment is not checked — transposing them must be an E0308.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubregionCompare {
    /// `true` — after the operands match, compare the two ops' regions pairwise as well
    /// (`OperationEquivalence.cpp:317-330`). Every site in `dcc` asks for this.
    Recursive,
    /// `false` — the operations' own operands and attributes only.
    TopLevelOnly,
}

/// WHEN TWO BLOCK ARGUMENTS COUNT AS THE SAME OPERAND — `all_block_args_are_equiv`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockArgEquivalence {
    /// `true`, the constructor's default — any block argument matches any other
    /// (`OperationEquivalence.cpp:292-293`).
    AllEquivalent,
    /// `false` — they must be the same argument of the same block
    /// (`a_owner != b_owner || a_idx != b_idx`, `OperationEquivalence.cpp:294-300`). ⭐ WHAT EVERY
    /// CONDITIONAL TREE ASKS FOR: two `scf.if`s in different loops branch on different induction
    /// variables even though both conditions are block arguments, and merging them would be wrong.
    SameOwnerAndIndex,
}

/// WHETHER A COMPARISON REMEMBERS ITS ANSWERS — `use_equiv_classes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EquivalenceCache {
    /// `true`, the constructor's default — consult and extend the equivalence classes
    /// (`OperationEquivalence.cpp:107-113`, `:333`). ⚠️ Only POSITIVE answers are reliable, as the
    /// field's own comment says: two equivalent ops can still be in two classes that have not been
    /// unioned yet.
    Reuse,
    /// `false` — compare afresh every time. Asked for once, by the sentient-level tree, whose
    /// transform rewrites the ops it has just compared.
    Recompute,
}

/// HOW ONE ANALYSIS COMPARES TWO OPERATIONS — `dcc::OperationEquivalence`'s configuration.
///
/// # ⛔⛔ ONE SITE IN THE REFERENCE DOES PASS A FUNCTOR, AND IT IS ENTRY 193
///
/// The six-argument constructor takes `function_ref<bool(Operation&, Operation&, void*)> functor`
/// and a `void *extent` for it to read (`dcc/src/Analysis/OperationEquivalence.hpp:36-45`), and
/// `functor_` is consulted at `OperationEquivalence.cpp:116` under the comment *"Functor carries
/// higher precedence as it is user-controlled."* Six of the seven sites that use that constructor
/// pass `nullptr, nullptr` — this file's `:112`, `CFGSSentientLevelConditionalTree.hpp:316`,
/// `CFGDeepMergingConditionalTree.hpp:29`, `LoopMerging.cpp:58`, `LoopAbsorption.cpp:43` and
/// `LoopRolling.cpp:987` — and the three-argument constructor leaves it default-constructed, which
/// for a `function_ref` is null. **The seventh is `ProgramUnitsReduction.cpp:103-136`**, whose lambda
/// is what makes a program unit's core and corelet ids parametric. So the field is
/// [`HighPreference`], not an omission. (The per-call `operands_equiv_checker` on
/// `operationsAreEquivalent` is a different parameter and is also used; it belongs to the call, not
/// here.)
///
/// ⚠️ `eq_classes_` IS NOT A FIELD HERE: it is the memo the comparison fills as it runs, which is the
/// mechanism the brief lets a port drop, and it arrives with whichever unit ports
/// `operationsAreEquivalent` — a function of `dcc/src/Analysis/`, outside the 384.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperationEquivalence {
    /// `debug_` — which `-debug-only=` name this comparison traces under.
    pub debug: EquivalenceTag,
    /// `functor_` together with the `context_` it reads — the positive-only override.
    pub preference: HighPreference,
    /// `do_recursive_compare_`.
    pub subregions: SubregionCompare,
    /// `all_block_args_are_equiv_`.
    pub block_args: BlockArgEquivalence,
    /// `use_equiv_classes_`.
    pub cache: EquivalenceCache,
}

impl OperationEquivalence {
    /// THE THREE ARGUMENTS A CONDITIONAL TREE NAMES, WITH THE FOURTH LEFT AT ITS DEFAULT.
    ///
    /// ⭐ `use_equiv_classes` DEFAULTS TO `true` in both of the reference's constructors
    /// (`dcc/src/Analysis/OperationEquivalence.hpp:27-45`), and this is the seam where that default
    /// is spent: `OperationEquivalence`'s own constructor is in `dcc/src/Analysis/` and so is not one
    /// of the 384, but entry 100 calls it and its arguments are entry 100's content.
    #[must_use]
    pub const fn tagged(
        debug: EquivalenceTag,
        subregions: SubregionCompare,
        block_args: BlockArgEquivalence,
    ) -> Self {
        Self {
            debug,
            preference: HighPreference::None,
            subregions,
            block_args,
            cache: EquivalenceCache::Reuse,
        }
    }

    /// THE FUNCTOR FORM WITH ALL THREE BOOLS LEFT AT THEIR DEFAULTS — what
    /// `ProgramUnitsReduction.cpp:103-136` constructs.
    ///
    /// ⭐ THE DEFAULTS ARE THE WHOLE OF WHAT THAT SITE SAYS ABOUT THEM. It names `functor`, `extent`
    /// and `debug` and stops, so `do_recursive_compare`, `all_block_args_are_equiv` and
    /// `use_equiv_classes` all take `true` (`OperationEquivalence.hpp:36-45`) — the same three values
    /// a bare `dcc::OperationEquivalence oe;` gets.
    #[must_use]
    pub const fn preferring(preference: HighPreference, debug: EquivalenceTag) -> Self {
        Self {
            debug,
            preference,
            subregions: SubregionCompare::Recursive,
            block_args: BlockArgEquivalence::AllEquivalent,
            cache: EquivalenceCache::Reuse,
        }
    }
}

/// THE CONDITIONAL TREE OF ONE `dataflow.program_unit`, AS CFG SIMPLIFICATION CONFIGURES IT.
///
/// A conditional tree holds the unit's `scf.if`s in the nesting they have, under one synthetic root,
/// so that a transform can ask about a conditional's parent, its children and its siblings
/// (`dcc/src/Analysis/ConditionalTree.hpp`). ⚠️ THE NODES ARRIVE LATER: `ConditionalTree::compute`
/// and `CondNode` are in `dcc/src/Analysis/`, which contributes nothing to bridge 2's 384, and the
/// units that walk this tree (entries 284, 371, 380, 381) are later batches in this same file.
/// Entry 100 is the constructor, and what a constructor ports to is the initial state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CfgsDataflowConditionalTree<'u, A: Arch> {
    /// `unit_` — the operation whose conditionals this tree indexes, held by reference because the
    /// reference takes `Operation &unit` and stores its address (`ConditionalTree.hpp`).
    pub unit: &'u ProgramUnit<A>,
    /// `oe_` — how this transform compares two operations, which is the whole of what entry 100 sets.
    pub oe: OperationEquivalence,
}

impl<'u, A: Arch> CfgsDataflowConditionalTree<'u, A> {
    /// Replaces: e100_TransformationConditionalTree
    ///
    /// **100/384** `CFGSDataflowConditionalTree::CFGSDataflowConditionalTree` —
    /// `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.hpp:111` (5L).
    ///
    /// ```cpp
    /// CFGSDataflowConditionalTree(Operation &unit)
    ///     : TransformationConditionalTree(unit) {
    ///   setOE(dcc::OperationEquivalence(nullptr, nullptr,
    ///                                   "cfg-merging-and-hoisting-cond-tree",
    ///                                   /*do_recursive_compare*/ true,
    ///                                   /*all_block_args_are_equiv*/ false));
    /// }
    /// ```
    ///
    /// # ⛔ THE ENTRY IS NAMED AFTER THE BASE-CLASS INITIALISER, NOT THE CONSTRUCTOR
    ///
    /// `UNITS.tsv` calls entry 100 `TransformationConditionalTree` because the extract sliced from
    /// the member-initialiser line. `TransformationConditionalTree`'s own constructor is
    /// `ConditionalTree(unit)` and nothing else (`dcc/src/Analysis/TransformationConditionalTree.hpp`)
    /// — a file outside the 384. The function at the cited line is
    /// `CFGSDataflowConditionalTree`'s constructor, and that is what is ported.
    ///
    /// # ⭐ THE CHAIN IS THREE CONSTRUCTORS AND ONE OF THEM SAYS SOMETHING
    ///
    /// `CFGSDataflowConditionalTree(unit)` → `TransformationConditionalTree(unit)` →
    /// `ConditionalTree(unit) : OperationTreeBase(), unit_(unit), dom_info_(&unit)`. The unit is
    /// stored; the tree starts empty (`root_ = nullptr`); `dom_info_` is a `DominanceInfo` built over
    /// the unit — ⚠️ A CACHE DERIVED FROM THE UNIT, recomputed by the hoisting units that ask it
    /// whether one op dominates another, and not a field this constructor decides anything about.
    /// So the only decision entry 100 makes is the `setOE`.
    ///
    /// # ⛔ AND THE ONE FLAG IT OVERRIDES IS THE ONE THAT WOULD MERGE TWO DIFFERENT LOOPS
    ///
    /// `all_block_args_are_equiv` defaults to `true`; this passes `false`. Two `scf.if`s whose
    /// conditions are block arguments of different loops would otherwise compare equal and be merged
    /// — see [`BlockArgEquivalence::SameOwnerAndIndex`]. `do_recursive_compare` is passed `true`,
    /// which is also its default, and `use_equiv_classes` is not named at all.
    ///
    /// ⚠️ `setOE` ITSELF IS A PROTECTED BASE-CLASS SETTER (`dcc/src/Analysis/ConditionalTree.hpp`)
    /// and exists so a derived tree can configure a field it cannot initialise. There is no such
    /// two-step here: the field is set where it is declared, and the setter is not ported.
    #[must_use]
    pub const fn new(unit: &'u ProgramUnit<A>) -> Self {
        Self {
            unit,
            oe: OperationEquivalence::tagged(
                EquivalenceTag::CfgMergingAndHoistingCondTree,
                SubregionCompare::Recursive,
                BlockArgEquivalence::SameOwnerAndIndex,
            ),
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::super::tf_utils::{CountedBounds, LoopBound, LoopStep};
    use super::*;
    use crate::arch::Dd2;
    use crate::islands::dataflow_ir::Units;
    use crate::islands::dataflow_ir::dialects::dataflow;
    use crate::islands::dataflow_ir::print;
    use crate::islands::dataflow_ir::ty::{ElemType, IntegerSet, ScalarTy, Vector};
    use crate::units::{DfirUnit, Residency};

    /// 🎯 097/384 — TWO MERGES OF THREE CONDITIONALS PRODUCE THE REFERENCE'S OWN NESTED NAME.
    ///
    /// `merging.mlir` feeds `--dcc-cfg-simplification-dataflow-level` three `scf.if`s on one
    /// condition, named `"SCF-If #2"`, `"SCF-If #3"` and `"SCF-If #4"` (`:128`, `:133`, `:146`), and
    /// checks that the survivor carries
    ///
    /// ```text
    /// } {dbgName = "CFGSM(SCF-If #4, CFGSM(SCF-If #2, SCF-If #3))"}
    /// ```
    ///
    /// (`dcc/test/Transform/CFGSimplificationDataflowLevel/merging.mlir:59`, a `CHECK-SENT-IR` line.)
    ///
    /// ⭐ THE ORDER IS `{src, dst}` AND THE NESTING PROVES IT: `#2` merged into `#3` first, then `#4`
    /// merged into that result — the source is named first and the accumulated destination second.
    #[test]
    fn a_merged_name_nests_the_names_it_merged() {
        let first_merge = new_dbg_name_from_list(
            DbgNamePrefix::Cfgsm,
            Some("SCF-If #2"),
            &[Some("SCF-If #3")],
        );
        assert_eq!(first_merge.as_deref(), Some("CFGSM(SCF-If #2, SCF-If #3)"));
        let second_merge = new_dbg_name_from_list(
            DbgNamePrefix::Cfgsm,
            Some("SCF-If #4"),
            &[first_merge.as_deref()],
        );
        assert_eq!(
            second_merge.as_deref(),
            Some("CFGSM(SCF-If #4, CFGSM(SCF-If #2, SCF-If #3))"),
            "the reference's own expectation for merging.mlir"
        );
    }

    /// 🎯 097/384 — AND A DEEPER NEST STAYS BALANCED, WHICH IS THE CASE THE PREFIX ENUM PROTECTS.
    ///
    /// `xrf_increments.mlir:165` carries a name from three merges of the same condition:
    /// `"CFGSM(CFGSM(CFGSM(condition__3, condition__3), condition__3), condition__3)"`. ⭐ NOTHING
    /// HERE CAN OPEN A BRACKET IT DOES NOT CLOSE — the caller names `CFGSM` and this function writes
    /// both.
    #[test]
    fn three_merges_stay_balanced() {
        let mut name = Some("condition__3".to_owned());
        for _ in 0..3 {
            name = new_dbg_name_from_list(
                DbgNamePrefix::Cfgsm,
                name.as_deref(),
                &[Some("condition__3")],
            );
        }
        assert_eq!(
            name.as_deref(),
            Some("CFGSM(CFGSM(CFGSM(condition__3, condition__3), condition__3), condition__3)"),
            "dcc/test/Conversion/VectorChainToSentientPT/xrf_increments.mlir:165"
        );
        let opened = name.as_deref().unwrap_or_default().matches('(').count();
        let closed = name.as_deref().unwrap_or_default().matches(')').count();
        assert_eq!(opened, closed, "one bracket closed per bracket opened");
    }

    /// 🎯 097/384 — ONE OPERATION WITHOUT A `dbgName` ABANDONS THE WHOLE NAME, WHEREVER IT SITS.
    ///
    /// `if (dbg_name_attr == nullptr) return nullptr;` discards the string built so far, so the
    /// caller's `if (StringAttr n = ...)` leaves the destination's existing name alone.
    #[test]
    fn an_unnamed_operation_abandons_the_name() {
        assert_eq!(
            new_dbg_name_from_list(DbgNamePrefix::Cfgsm, None, &[Some("SCF-If #3")]),
            None,
            "the head has no dbgName"
        );
        assert_eq!(
            new_dbg_name_from_list(DbgNamePrefix::Cfgsm, Some("SCF-If #2"), &[None]),
            None,
            "the tail has no dbgName — and the partial string is not returned"
        );
        assert_eq!(
            new_dbg_name_from_list(
                DbgNamePrefix::Lc,
                Some("in-7/6"),
                &[Some("in-7/5"), None, Some("in-7/4")],
            ),
            None,
            "an unnamed op in the middle of LoopCoalescing's list"
        );
    }

    /// 🎯 097/384 — A LIST LONGER THAN TWO SEPARATES WITH `", "`, WHICH IS THE COALESCING FORM.
    ///
    /// `LoopCoalescing.cpp:279-282` is the one call site that passes a whole loop nest, so it is the
    /// one that exercises the `else` branch of `is_first_op` more than once.
    #[test]
    fn a_coalesced_nest_lists_every_loop() {
        assert_eq!(
            new_dbg_name_from_list(
                DbgNamePrefix::Lc,
                Some("in-7/6"),
                &[Some("in-7/5"), Some("in-7/4")],
            )
            .as_deref(),
            Some("LC(in-7/6, in-7/5, in-7/4)")
        );
    }

    /// THE `scf.if (%iv == %k)` FROM `simplify-conditional.mlir:115` AND `:133`, AS TYPED VALUES.
    ///
    /// `%30 = arith.cmpi eq, %26, %0 : index` then `%39 = scf.if %30 -> (index)`.
    fn a_value_based_conditional() -> (DfirOp, Vec<DfirOp>) {
        let if_op = DfirOp::Scf(scf::Op::If {
            cond: Val(30),
            results: Vec::new(),
            body: Vec::new(),
            else_body: Vec::new(),
            dbg_name: None,
        });
        let scope = vec![
            DfirOp::Arith(arith::Op::Constant {
                result: Val(0),
                value: 0,
            }),
            DfirOp::Arith(arith::Op::Compare {
                result: Val(30),
                predicate: arith::CmpIPredicate::Eq,
                lhs: Val(26),
                rhs: Val(0),
            }),
        ];
        (if_op, scope)
    }

    /// 🎯 098/384 — AN `scf.if` ON AN `arith.cmpi eq` YIELDS ITS TWO SIDES, IN ORDER.
    #[test]
    fn an_eq_conditional_yields_its_two_sides() {
        let (if_op, scope) = a_value_based_conditional();
        assert_eq!(
            lhs_rhs_of_eq_predicate(&if_op, &scope),
            Some(EqPredicate {
                lhs: Val(26),
                rhs: Val(0),
            }),
            "the induction variable is the lhs and the constant is the rhs — never swapped"
        );
    }

    /// 🎯 098/384 — A PREDICATE THAT IS NOT `eq` DECLINES, AND THAT TEST COULD NOT FAIL BEFORE.
    ///
    /// ⛔ This is the case the island could not express: `arith::Op::Compare` printed a hard-coded
    /// `eq`, so `cmpi_op.getPredicate() != eq` was unreachable and half of this function was dead.
    /// One of the five other predicates is enough to show the exit is real.
    #[test]
    fn a_non_eq_conditional_declines() {
        let (if_op, mut scope) = a_value_based_conditional();
        scope[1] = DfirOp::Arith(arith::Op::Compare {
            result: Val(30),
            predicate: arith::CmpIPredicate::Ne,
            lhs: Val(26),
            rhs: Val(0),
        });
        assert_eq!(
            lhs_rhs_of_eq_predicate(&if_op, &scope),
            None,
            "an `ne` is not an equality, so this conditional is not a candidate"
        );
    }

    /// 🎯 098/384 — AND SO DO THE TWO EARLIER EXITS: NOT AN `scf.if`, AND NO DEFINING `arith.cmpi`.
    #[test]
    fn a_conditional_without_a_cmpi_declines() {
        let (if_op, scope) = a_value_based_conditional();
        // `llvm::dyn_cast<mlir::scf::IfOp>(op)` on something that is not one.
        assert_eq!(
            lhs_rhs_of_eq_predicate(&scope[1].clone(), &scope),
            None,
            "an arith.cmpi is not a conditional"
        );
        // `cond.getDefiningOp<CmpIOp>()` where the condition is bound by something else — the
        // reference's null both for a block argument and for a non-cmpi binder.
        let bound_elsewhere = vec![DfirOp::Arith(arith::Op::ConstantInt {
            result: Val(30),
            value: arith::IntConst::Bool(true),
        })];
        assert_eq!(lhs_rhs_of_eq_predicate(&if_op, &bound_elsewhere), None);
        assert_eq!(
            lhs_rhs_of_eq_predicate(&if_op, &[]),
            None,
            "nothing in scope binds the condition at all"
        );
    }

    /// 🎯 099/384 — THE MANAGER STARTS WITH NO ARRAY, WHICH IS THE NON-CANDIDATE STATE.
    ///
    /// `std::optional<int64_t> *val_array_ = nullptr;` (`hpp:73`) is the member initialiser, and the
    /// constructor's two early returns leave it there — `if_op_->getNumResults() == 0` or no `eq`
    /// predicate, and a loop with symbolic bounds (`hpp:58-70`). The destructor's `if (val_array_)`
    /// exists for exactly those paths.
    #[test]
    fn a_manager_starts_with_no_value_array() {
        let instance = ConditionalSimplificationManager::default();
        assert_eq!(instance.val_array, None, "the reference's `= nullptr`");
    }

    /// 🎯 099/384 — AN ARRAY IS ONE SLOT PER ITERATION, ALL EMPTY, AND A CLONE OWNS ITS OWN.
    ///
    /// *"Maps iteration space to values yielded by the conditional. All entries are initialized to
    /// nullopt."* — `new std::optional<int64_t>[num_iterations]` with `num_iterations` from
    /// `std::get<4>(for_op_tuple_)` (`hpp:71-75`).
    ///
    /// ⛔ THE CLONE IS THE POINT: the reference's implicit copy constructor would shallow-copy the
    /// owning pointer and both objects would `delete[]` it. Here the boxed slice is deep-copied, so
    /// writing through one does not touch the other, and dropping both is dropping two allocations.
    #[test]
    fn a_value_array_is_owned_per_iteration() {
        let mut instance = ConditionalSimplificationManager {
            val_array: Some(vec![None; 4].into_boxed_slice()),
        };
        assert!(
            instance
                .val_array
                .as_deref()
                .is_some_and(|slots| slots.len() == 4 && slots.iter().all(Option::is_none)),
            "four iterations, every entry nullopt"
        );
        let copy = instance.clone();
        if let Some(slots) = instance.val_array.as_deref_mut() {
            slots[2] = Some(YieldedIndex(3));
        }
        assert_eq!(
            copy.val_array.as_deref().and_then(|slots| slots[2]),
            None,
            "a clone owns its own array — the reference's shallow copy is not available here"
        );
        drop(instance);
        drop(copy);
    }

    /// 🎯 100/384 — THE TREE CONFIGURES ITS COMPARISON WITH THE TAG AND THE ONE OVERRIDDEN FLAG.
    #[test]
    fn the_tree_configures_its_operation_equivalence() {
        let unit: ProgramUnit<Dd2> = ProgramUnit {
            on: Units::one(DfirUnit::Pe, Val(8)),
            precision: None,
            body: Vec::new(),
            arch: core::marker::PhantomData,
        };
        let tree = CfgsDataflowConditionalTree::new(&unit);
        assert_eq!(
            tree.oe,
            OperationEquivalence {
                debug: EquivalenceTag::CfgMergingAndHoistingCondTree,
                preference: HighPreference::None,
                subregions: SubregionCompare::Recursive,
                block_args: BlockArgEquivalence::SameOwnerAndIndex,
                cache: EquivalenceCache::Reuse,
            }
        );
        assert_eq!(
            tree.oe.debug.spelling(),
            "cfg-merging-and-hoisting-cond-tree",
            "the string the reference passes as `debug`"
        );
        assert!(
            core::ptr::eq(tree.unit, &unit),
            "`unit_(unit)` stores the unit's address, not a copy of it"
        );
    }

    /// 🎯 100/384 — AND IT DIFFERS FROM THE CONSTRUCTOR'S DEFAULTS IN EXACTLY ONE FLAG.
    ///
    /// `OperationEquivalence(bool do_recursive_compare = true, bool all_block_args_are_equiv = true,
    /// bool use_equiv_classes = true)` (`dcc/src/Analysis/OperationEquivalence.hpp:27-34`). Of the
    /// three, this tree overrides `all_block_args_are_equiv` and nothing else — so a port that
    /// dropped the `setOE` would compare two loops' block arguments as equal and merge conditionals
    /// that are not the same.
    #[test]
    fn only_the_block_argument_rule_is_overridden() {
        let unit: ProgramUnit<Dd2> = ProgramUnit {
            on: Units::one(DfirUnit::Pe, Val(8)),
            precision: None,
            body: Vec::new(),
            arch: core::marker::PhantomData,
        };
        let oe = CfgsDataflowConditionalTree::new(&unit).oe;
        let defaults = OperationEquivalence::tagged(
            oe.debug,
            SubregionCompare::Recursive,
            BlockArgEquivalence::AllEquivalent,
        );
        assert_ne!(oe, defaults);
        assert_eq!(oe.subregions, defaults.subregions);
        assert_eq!(oe.cache, defaults.cache);
        assert_eq!(oe.block_args, BlockArgEquivalence::SameOwnerAndIndex);
    }

    /// 🎯 095/384 — THE TWO CONDITIONALS ARE SELECTED AND NOTHING ELSE IS.
    ///
    /// `return isa<mlir::affine::AffineIfOp, mlir::scf::IfOp>(op);`
    /// (`Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:34-36`) — a loop is not a
    /// conditional even though the pass walks both.
    #[test]
    fn only_an_affine_if_and_an_scf_if_are_selected() {
        let scf_if = DfirOp::Scf(scf::Op::If {
            cond: Val(0),
            results: Vec::new(),
            body: Vec::new(),
            else_body: Vec::new(),
            dbg_name: None,
        });
        let affine_if = DfirOp::Affine(affine::Op::If {
            set: IntegerSet::from_sizes(&[4]),
            args: vec![Val(1)],
            symbol_args: Vec::new(),
            results: Vec::new(),
            body: Vec::new(),
            else_body: Vec::new(),
            dbg_name: None,
        });
        let parallel = DfirOp::Scf(scf::Op::Parallel {
            ivs: vec![Val(2)],
            body: Vec::new(),
        });

        assert!(is_operation_selected(&scf_if));
        assert!(is_operation_selected(&affine_if));
        assert!(!is_operation_selected(&parallel));
        assert!(!is_operation_selected(&DfirOp::Scf(scf::Op::Yield {
            operands: Vec::new(),
        })));
    }

    /// 🎯 096/384 — THE DUMMY YIELD IS WHAT MAKES AN `else` PRINT AT ALL.
    ///
    /// An `scf.if` whose second region has no block prints no `else`; one whose block holds only the
    /// elided terminator prints the reference's own text —
    ///
    /// ```text
    /// } else {
    /// }
    /// ```
    /// (`dcc/test/PT/issue-236.mlir:65-71`). So the observable of this function is a pair of braces,
    /// and it is checked here through the island's printer rather than by counting ops.
    #[test]
    fn the_dummy_yield_is_what_makes_an_else_print() {
        let mut op = DfirOp::Scf(scf::Op::If {
            cond: Val(0),
            results: Vec::new(),
            body: vec![DfirOp::Scf(scf::Op::Yield {
                operands: Vec::new(),
            })],
            else_body: Vec::new(),
            dbg_name: None,
        });

        let mut before = String::new();
        crate::islands::dataflow_ir::print::emit(&mut before, &op, 0);
        assert!(
            !before.contains("else"),
            "an empty region is NO BLOCK, and MLIR prints no else for one: {before}"
        );

        let region = EmptyElseRegion::of(&mut op).expect("an scf.if with an empty else region");
        assert_eq!(region.kind(), ConditionalKind::Scf);
        create_dummy_yield_in_else_reg(region);

        let mut after = String::new();
        crate::islands::dataflow_ir::print::emit(&mut after, &op, 0);
        assert!(after.contains("} else {"), "{after}");
        assert!(
            !after.contains("scf.yield"),
            "the terminator is ELIDED, not dropped: {after}"
        );

        let DfirOp::Scf(scf::Op::If { else_body, .. }) = &op else {
            unreachable!("built above")
        };
        assert_eq!(
            else_body.as_slice(),
            [DfirOp::Scf(scf::Op::Yield {
                operands: Vec::new(),
            })],
            "and the op IS in the region"
        );
    }

    /// 🎯 096/384 — AN `else` THAT ALREADY HAS A BLOCK IS NOT A CANDIDATE, AND NEITHER IS A
    /// CONDITIONAL OF EITHER KIND THAT YIELDS A VALUE.
    ///
    /// `if (if_op->getRegions()[1].empty() && if_op->getNumResults() == 0)`
    /// (`CFGSDataflowConditionalTree.cpp:383-386`) — the conjunction, both halves.
    #[test]
    fn a_populated_else_and_a_value_yielding_if_are_not_candidates() {
        let mut populated = DfirOp::Scf(scf::Op::If {
            cond: Val(0),
            results: Vec::new(),
            body: Vec::new(),
            else_body: vec![DfirOp::Scf(scf::Op::Yield {
                operands: Vec::new(),
            })],
            dbg_name: None,
        });
        assert!(EmptyElseRegion::of(&mut populated).is_none());

        let mut yields_a_value = DfirOp::Affine(affine::Op::If {
            set: IntegerSet::from_sizes(&[4]),
            args: vec![Val(1)],
            symbol_args: Vec::new(),
            results: vec![Val(2)],
            body: Vec::new(),
            else_body: Vec::new(),
            dbg_name: None,
        });
        assert!(EmptyElseRegion::of(&mut yields_a_value).is_none());

        // ⭐ AND THE `scf` ARM ANSWERS THE SAME WAY NOW THAT THE OP CARRIES A RESULT LIST — the
        // vendor's `%13 = scf.if %12 -> (index)` (`merging.mlir:129`) is the state the reference's
        // `getNumResults() == 0` half of the conjunction stops on.
        let mut scf_yields_a_value = DfirOp::Scf(scf::Op::If {
            cond: Val(12),
            results: vec![Val(13)],
            body: Vec::new(),
            else_body: Vec::new(),
            dbg_name: None,
        });
        assert!(EmptyElseRegion::of(&mut scf_yields_a_value).is_none());
    }

    // ══════════════════════════════════════════════════════════════════════════════════════════
    // 176/384
    // ══════════════════════════════════════════════════════════════════════════════════════════

    /// `%9 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "pe-CL0", type = "pe"}`
    /// (`merging-shallow-skip.mlir:22`) — an op of a dialect the side-effect list never mentions.
    ///
    /// ⚠️ The ops actually sitting BETWEEN that fixture's two conditionals are `dataflow.receive`,
    /// `dataflow.send` and the two `vectorchain` ops (`:55-59`), and the island's send/receive pair
    /// spends a [`crate::islands::dataflow_ir::link::Link`] end that a unit test cannot mint out of
    /// nothing. They land on the SAME classification arm as this one — `DfirOp::Dataflow(_)` — which is
    /// the fact under test.
    fn a_dataflow_op() -> DfirOp {
        DfirOp::Dataflow(dataflow::Op::GetUnit {
            result: Val(9),
            residency: Residency::Global,
            unit: DfirUnit::Pe,
            num_folds: None,
        })
    }

    /// 🎯 176/384 — THE ELEVEN NAMED OPS PASS, AND NOTHING ELSE DOES.
    ///
    /// The list is `isa<arith::ConstantOp, arith::ConstantIndexOp, arith::ConstantIntOp,
    /// arith::CmpIOp, scf::YieldOp, scf::ForOp, affine::AffineForOp, affine::AffineYieldOp,
    /// mlir::symbol::CreateSymbolOp>` plus the two conditionals `isOperationSelected` answers for
    /// (`CFGSDataflowConditionalTree.cpp:41-44`).
    #[test]
    fn only_the_named_ops_are_free_of_side_effects() {
        let harmless = [
            // `%0 = arith.constant 1 : index` (`merging-shallow-skip.mlir:13`).
            DfirOp::Arith(arith::Op::Constant {
                result: Val(0),
                value: 1,
            }),
            // `%1 = arith.constant true` (`:14`).
            DfirOp::Arith(arith::Op::ConstantInt {
                result: Val(1),
                value: arith::IntConst::Bool(true),
            }),
            // `%3 = arith.constant dense<1.000000e+00> : vector<64xf16>` (`:16`) — still an
            // `arith::ConstantOp`.
            DfirOp::Arith(arith::Op::DenseConstant {
                result: Val(3),
                splat: 1,
                ty: Vector {
                    len: 64,
                    elem: ElemType::F16,
                },
            }),
            // `%22 = arith.cmpi eq, %21, %8 : index` (`:35`).
            DfirOp::Arith(arith::Op::Compare {
                result: Val(22),
                predicate: arith::CmpIPredicate::Eq,
                lhs: Val(21),
                rhs: Val(8),
            }),
            // `scf.yield %1 : i1` (`:37`).
            DfirOp::Scf(scf::Op::Yield {
                operands: vec![Val(1)],
            }),
            // `scf.for` — a counted loop is not a movement barrier by itself.
            DfirOp::Scf(scf::Op::For {
                iv: Val(50),
                lo: Val(51),
                hi: Val(52),
                step: Val(53),
                carried: Vec::new(),
                body: Vec::new(),
                dbg_name: None,
            }),
            // `affine.for %21 = 0 to 28 {` (`:34`).
            DfirOp::Affine(affine::Op::For {
                iv: Val(21),
                lo: affine::Bound::Const(0),
                hi: affine::Bound::Const(28),
                carried: Vec::new(),
                body: Vec::new(),
                dbg_name: None,
            }),
            // `affine.yield`.
            DfirOp::Affine(affine::Op::Yield {
                operands: Vec::new(),
            }),
            // `symbol.create_symbol {SymbolId = 0 : i32} : index`.
            DfirOp::Symbol(symbol::Op::CreateSymbol {
                result: Val(38),
                symbol_id: 0,
                max_value: None,
            }),
            // `!isOperationSelected(*op) &&` — the two conditionals, empty here.
            DfirOp::Scf(scf::Op::If {
                cond: Val(23),
                results: Vec::new(),
                body: Vec::new(),
                else_body: Vec::new(),
                dbg_name: None,
            }),
            DfirOp::Affine(affine::Op::If {
                set: IntegerSet::from_sizes(&[4]),
                args: vec![Val(1)],
                symbol_args: Vec::new(),
                results: Vec::new(),
                body: Vec::new(),
                else_body: Vec::new(),
                dbg_name: None,
            }),
        ];
        for op in &harmless {
            assert!(!op_has_side_effect(op), "on the list: {op:?}");
        }

        assert!(
            op_has_side_effect(&a_dataflow_op()),
            "a dialect the list never mentions"
        );
    }

    /// 🎯 176/384 — ⛔ IT IS AN ALLOW-LIST, NOT A PURITY TEST: `arith.addi` and `arith.muli` are
    /// `Pure` in MLIR and this answers *"has a side effect"* for both, because neither is named.
    #[test]
    fn pure_arithmetic_is_not_on_the_list() {
        for op in [
            arith::Op::AddI(arith::IntBinary {
                result: Val(60),
                lhs: Val(61),
                rhs: Val(62),
                ty: ScalarTy::Index,
            }),
            arith::Op::MulI(arith::IntBinary {
                result: Val(63),
                lhs: Val(64),
                rhs: Val(65),
                ty: ScalarTy::Index,
            }),
        ] {
            assert!(op_has_side_effect(&DfirOp::Arith(op)));
        }
    }

    /// 🎯 176/384 — THE WALK IS PRE-ORDER OVER THE WHOLE SUBTREE, so a listed op holding an unlisted
    /// one answers for its body: the fixture's `scf.if %23 { %24 = dataflow.receive .. }` (`:41-47`)
    /// is a conditional — on the list — whose contents are not.
    #[test]
    fn the_walk_descends_into_every_region() {
        let with_a_dataflow_op_inside = DfirOp::Scf(scf::Op::If {
            cond: Val(23),
            results: Vec::new(),
            body: vec![a_dataflow_op()],
            else_body: Vec::new(),
            dbg_name: None,
        });
        assert!(op_has_side_effect(&with_a_dataflow_op_inside));

        // ⭐ AND THE `else` REGION IS WALKED TOO — half a conditional scanned is a merge across an op
        // that must not move.
        let only_in_the_else = DfirOp::Scf(scf::Op::If {
            cond: Val(23),
            results: Vec::new(),
            body: Vec::new(),
            else_body: vec![a_dataflow_op()],
            dbg_name: None,
        });
        assert!(op_has_side_effect(&only_in_the_else));

        // A nest of nothing but listed ops, three levels deep.
        let clean = DfirOp::Affine(affine::Op::For {
            iv: Val(21),
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(28),
            carried: Vec::new(),
            body: vec![DfirOp::Scf(scf::Op::If {
                cond: Val(23),
                results: Vec::new(),
                body: vec![DfirOp::Scf(scf::Op::Yield {
                    operands: vec![Val(1)],
                })],
                else_body: vec![DfirOp::Scf(scf::Op::Yield {
                    operands: vec![Val(2)],
                })],
                dbg_name: None,
            })],
            dbg_name: None,
        });
        assert!(!op_has_side_effect(&clean));
    }

    // ══════════════════════════════════════════════════════════════════════════════════════════
    // 177/384
    // ══════════════════════════════════════════════════════════════════════════════════════════

    /// ONE STATEMENT OF A CONDITIONAL'S ARM, IDENTIFIABLE BY WHAT IT BINDS.
    ///
    /// ⚠️ `merging.mlir`'s arms hold `dataflow.receive`/`dataflow.send` pairs, which spend a
    /// [`crate::islands::dataflow_ir::link::Link`] end a unit test cannot mint out of nothing — the
    /// same limit [`a_dataflow_op`] records. What a splice is observed by is ORDER, so a
    /// distinguishable statement suffices and an `arith.constant` is the cheapest one.
    fn statement(n: u32) -> DfirOp {
        DfirOp::Arith(arith::Op::Constant {
            result: Val(n),
            value: i64::from(n),
        })
    }

    /// AN `scf.if %12` WITH THE GIVEN ARMS AND NAME, ITS TERMINATORS EXPLICIT.
    ///
    /// ⭐ A `then` ARM ALWAYS HAS A BLOCK AND A BLOCK ALWAYS HAS ITS TERMINATOR — elided when
    /// printed, present in the region (see [`scf::Op::If::else_body`]). An `else` arm given `None`
    /// has NO BLOCK, which is the state entry 096 fills.
    fn conditional(
        results: Vec<Val>,
        body: Vec<u32>,
        else_body: Option<Vec<u32>>,
        yields: (Vec<Val>, Vec<Val>),
        dbg_name: &str,
    ) -> DfirOp {
        let arm = |ops: Vec<u32>, operands: Vec<Val>| {
            let mut arm: Vec<DfirOp> = ops.into_iter().map(statement).collect();
            arm.push(DfirOp::Scf(scf::Op::Yield { operands }));
            arm
        };
        DfirOp::Scf(scf::Op::If {
            cond: Val(12),
            results,
            body: arm(body, yields.0),
            else_body: else_body.map_or_else(Vec::new, |ops| arm(ops, yields.1)),
            dbg_name: Some(dbg_name.to_owned()),
        })
    }

    /// THE TWO ARMS AND THE NAME OF A CONDITIONAL, FOR COMPARISON.
    fn arms_and_name(op: &DfirOp) -> (&[DfirOp], &[DfirOp], Option<&str>) {
        let DfirOp::Scf(scf::Op::If {
            body, else_body, ..
        }) = op
        else {
            unreachable!("every conditional in these tests is an scf.if")
        };
        (body, else_body, dfir_op::dbg_name(op))
    }

    /// 🎯 177/384 — `merging.mlir`'S TWO MERGES END WITH ITS OWN REGIONS AND ITS OWN NESTED NAME.
    ///
    /// The fixture's three mergeable conditionals on `%12` are, in block order
    /// (`dcc/test/Transform/CFGSimplificationDataflowLevel/merging.mlir:122-146`):
    ///
    /// | name | binds | `then` | `else` |
    /// |---|---|---|---|
    /// | `"SCF-If #2"` (`:122-128`) | nothing | a receive/mac/send chain | ⛔ NO BLOCK |
    /// | `"SCF-If #3"` (`:129-133`) | `%13 : index` | `scf.yield %c1` | `scf.yield %c2` |
    /// | `"SCF-If #4"` (`:134-146`) | nothing | a chain on `%cst` | a chain on `%cst_1` |
    ///
    /// and `shallowlyMergeConditionals` performs exactly two merges over them, in this order and with
    /// these arguments (`:237-247`):
    ///
    /// 1. `n` is `#2`, which binds nothing ⇒ `mergeShallow(src: #2, dst: #3, dst_before_src: false)`.
    /// 2. `n` is now `#3`, which binds `%13` ⇒ `mergeShallow(src: #4, dst: #3, dst_before_src: true)`.
    ///
    /// ⭐⭐ AND THE EXPECTATION PINS THE RESULT ARM BY ARM (`:40-59`): one `scf.if -> (index)` whose
    /// `then` holds `#2`'s chain, then `#4`'s chain, then `scf.yield %c1`, and whose `else` holds
    /// `#4`'s `else` chain and then `scf.yield %c2` — so the earlier op's statements come first in
    /// BOTH merges even though the two took opposite orders, and the only surviving terminators are
    /// `#3`'s, the pair whose operands are the merged op's results.
    #[test]
    fn the_fixtures_two_merges_end_with_its_own_regions_and_name() {
        let c1 = Val(1);
        let c2 = Val(2);
        let mut second = conditional(
            Vec::new(),
            vec![20],
            None,
            (Vec::new(), Vec::new()),
            "SCF-If #2",
        );
        let mut third = conditional(
            vec![Val(13)],
            Vec::new(),
            Some(Vec::new()),
            (vec![c1], vec![c2]),
            "SCF-If #3",
        );
        let mut fourth = conditional(
            Vec::new(),
            vec![40],
            Some(vec![41]),
            (Vec::new(), Vec::new()),
            "SCF-If #4",
        );

        merge_shallow(&mut second, &mut third, MergeOrder::SrcFirst);
        assert_eq!(
            arms_and_name(&third),
            (
                [
                    statement(20),
                    DfirOp::Scf(scf::Op::Yield { operands: vec![c1] })
                ]
                .as_slice(),
                [DfirOp::Scf(scf::Op::Yield { operands: vec![c2] })].as_slice(),
                Some("CFGSM(SCF-If #2, SCF-If #3)"),
            ),
            "the source stands FIRST in the block, so its statements are prepended, and the \
             destination's yields survive because the destination is what binds %13"
        );

        merge_shallow(&mut fourth, &mut third, MergeOrder::DstFirst);
        assert_eq!(
            arms_and_name(&third),
            (
                [
                    statement(20),
                    statement(40),
                    DfirOp::Scf(scf::Op::Yield { operands: vec![c1] })
                ]
                .as_slice(),
                [
                    statement(41),
                    DfirOp::Scf(scf::Op::Yield { operands: vec![c2] })
                ]
                .as_slice(),
                Some("CFGSM(SCF-If #4, CFGSM(SCF-If #2, SCF-If #3))"),
            ),
            "merging.mlir:40-59 — and the name NESTS, because the second merge reads what the first \
             one wrote"
        );

        // ⭐ AND THE SOURCES ARE LEFT SPLICED OUT, NOT DELETED: `shallowlyMergeConditionals` erases
        // them afterwards through `deleteAncestorsIfPossible` (`:242`, `:257`, `:269`).
        for spliced in [&second, &fourth] {
            let (body, else_body, _) = arms_and_name(spliced);
            assert!(body.is_empty() && else_body.is_empty(), "{spliced:?}");
        }
    }

    /// 🎯 177/384 — AN ABSENT `else` ARM IS GIVEN ITS TERMINATOR BEFORE THE SPLICE, AND IS THEN A
    /// DESTINATION LIKE ANY OTHER.
    ///
    /// `if (src_region_empty) createDummyYieldInElseReg(src); else if (dst_region_empty)
    /// createDummyYieldInElseReg(dst);` (`CFGSDataflowConditionalTree.cpp:408-411`) — one call, on
    /// whichever side is missing a block, and never on both because the arm was skipped when both
    /// were empty (`:404`).
    ///
    /// ⭐ THE OBSERVABLE IS A PAIR OF BRACES THAT WAS NOT PRINTED BEFORE — a conditional whose `else`
    /// region has no block prints no `else` at all (`dcc/test/PT/issue-236.mlir:65-71`, entry 096).
    #[test]
    fn an_absent_else_arm_is_filled_before_it_is_spliced_into() {
        let mut src = conditional(
            Vec::new(),
            vec![70],
            Some(vec![71]),
            (Vec::new(), Vec::new()),
            "src",
        );
        let mut dst = conditional(Vec::new(), vec![80], None, (Vec::new(), Vec::new()), "dst");

        let mut before = String::new();
        crate::islands::dataflow_ir::print::emit(&mut before, &dst, 0);
        assert!(!before.contains("else"), "{before}");

        merge_shallow(&mut src, &mut dst, MergeOrder::DstFirst);

        let (body, else_body, name) = arms_and_name(&dst);
        assert_eq!(
            body,
            [
                statement(80),
                statement(70),
                DfirOp::Scf(scf::Op::Yield {
                    operands: Vec::new()
                })
            ],
            "the destination stands first, so the source's statements are appended"
        );
        assert_eq!(
            else_body,
            [
                statement(71),
                DfirOp::Scf(scf::Op::Yield {
                    operands: Vec::new()
                })
            ],
            "the dummy terminator was popped and the source's whole else arm took its place"
        );
        assert_eq!(name, Some("CFGSM(src, dst)"));

        let mut after = String::new();
        crate::islands::dataflow_ir::print::emit(&mut after, &dst, 0);
        assert!(after.contains("} else {"), "{after}");
    }

    /// 🎯 177/384 — THE TERMINATOR THAT SURVIVES IS THE ONE BELONGING TO WHICHEVER CONDITIONAL BINDS
    /// A VALUE, AND IT ALWAYS ENDS UP LAST.
    ///
    /// All four rows of [`merge_shallow`]'s own table, over one `then` arm holding a single statement
    /// on each side. `areShallowlyMergeable` guarantees at most one side binds anything (`:348`), so
    /// these four are the whole space.
    #[test]
    fn the_yield_of_the_side_that_binds_a_value_is_the_one_kept() {
        let kept = Val(1);
        let dropped = Val(2);

        for (order, src_results, dst_results, want) in [
            (MergeOrder::DstFirst, Vec::new(), Vec::new(), vec![80, 70]),
            (MergeOrder::DstFirst, Vec::new(), vec![kept], vec![80, 70]),
            (MergeOrder::SrcFirst, Vec::new(), Vec::new(), vec![70, 80]),
            (MergeOrder::SrcFirst, vec![kept], Vec::new(), vec![70, 80]),
        ] {
            // ⭐ THE SIDE THAT BINDS NOTHING YIELDS NOTHING, which is what makes the surviving
            // terminator identifiable: `dropped` may only ever appear on the side with no results.
            let src_yield = if src_results.is_empty() {
                vec![dropped]
            } else {
                vec![kept]
            };
            let dst_yield = if dst_results.is_empty() {
                vec![dropped]
            } else {
                vec![kept]
            };
            let mut src = conditional(
                src_results.clone(),
                vec![70],
                None,
                (src_yield, Vec::new()),
                "src",
            );
            let mut dst = conditional(
                dst_results.clone(),
                vec![80],
                None,
                (dst_yield, Vec::new()),
                "dst",
            );

            merge_shallow(&mut src, &mut dst, order);

            let (body, _, _) = arms_and_name(&dst);
            let mut expected: Vec<DfirOp> = want.into_iter().map(statement).collect();
            expected.push(DfirOp::Scf(scf::Op::Yield {
                operands: if src_results.is_empty() && dst_results.is_empty() {
                    vec![dropped]
                } else {
                    vec![kept]
                },
            }));
            assert_eq!(
                body, expected,
                "{order:?} with src binding {src_results:?} and dst binding {dst_results:?}: \
                 program order is preserved and exactly one terminator is left, last"
            );
        }
    }

    /// 🎯 177/384 — A MERGE WHOSE HALVES ARE NOT BOTH NAMED LEAVES THE DESTINATION'S NAME ALONE.
    ///
    /// `if (StringAttr new_dbg_name_attr = getNewDbgNameFromList("CFGSM(", {src, dst}))`
    /// (`CFGSDataflowConditionalTree.cpp:455-458`) — the assignment is the condition, and entry 097
    /// returns null as soon as one operation has no name.
    ///
    /// ⭐ AND THE REGIONS STILL MERGE. The name is the last three lines of the function; nothing
    /// about the splice depends on it.
    #[test]
    fn an_unnamed_half_leaves_the_destinations_name_untouched() {
        for (src_name, dst_name, want) in [
            (None, Some("dst"), Some("dst")),
            (Some("src"), None, None),
            (Some("src"), Some("dst"), Some("CFGSM(src, dst)")),
        ] {
            let mut src = conditional(Vec::new(), vec![70], None, (Vec::new(), Vec::new()), "src");
            let mut dst = conditional(Vec::new(), vec![80], None, (Vec::new(), Vec::new()), "dst");
            *dfir_op::dbg_name_mut(&mut src).expect("an scf.if carries a name") =
                src_name.map(str::to_owned);
            *dfir_op::dbg_name_mut(&mut dst).expect("an scf.if carries a name") =
                dst_name.map(str::to_owned);

            merge_shallow(&mut src, &mut dst, MergeOrder::DstFirst);

            let (body, _, name) = arms_and_name(&dst);
            assert_eq!(name, want, "src {src_name:?}, dst {dst_name:?}");
            assert_eq!(
                body,
                [
                    statement(80),
                    statement(70),
                    DfirOp::Scf(scf::Op::Yield {
                        operands: Vec::new()
                    })
                ],
                "the splice does not depend on the names"
            );
        }
    }

    /// One unit holding `body` — the operation the conditional tree indexes.
    fn unit_holding(body: Vec<DfirOp>) -> ProgramUnit<Dd2> {
        ProgramUnit {
            on: Units::one(DfirUnit::Pe, Val(8)),
            precision: None,
            body,
            arch: core::marker::PhantomData,
        }
    }

    /// An `scf.if` on `cond` with an empty `then` and no `else`.
    fn if_on(cond: Val) -> DfirOp {
        DfirOp::Scf(scf::Op::If {
            cond,
            results: Vec::new(),
            body: Vec::new(),
            else_body: Vec::new(),
            dbg_name: None,
        })
    }

    /// 🎯 380/384 — A CONDITIONAL WITH A LATER SIBLING IS OFFERED TO THE MERGE PREDICATE, AND THE
    /// SIBLING NESTED IN THE `affine.for` IS ONE — `findClosestParent` skips the unselected loop
    /// (`ConditionalTree.cpp:200`), so a pair a same-list walk would never offer reaches entry 370,
    /// which then refuses it across blocks (`:346-349`) and leaves the walk with nothing merged.
    #[test]
    fn a_conditional_and_its_next_sibling_reach_the_merge_predicate() {
        let unit = unit_holding(vec![
            if_on(Val(0)),
            DfirOp::Affine(affine::Op::For {
                iv: Val(1),
                lo: affine::Bound::Const(0),
                hi: affine::Bound::Const(4),
                carried: Vec::new(),
                body: vec![if_on(Val(0))],
                dbg_name: None,
            }),
        ]);
        assert_eq!(
            shallowly_merge_conditionals(&CfgsDataflowConditionalTree::new(&unit), None),
            None
        );
    }

    /// 🎯⛔ AND A SINGLE CONDITIONAL HAS NO SIBLING TO MERGE WITH, so the walk completes and the
    /// answer is "no more merge opportunities" (`:209-210`) rather than a merge of one.
    #[test]
    fn a_lone_conditional_yields_no_merge() {
        let unit = unit_holding(vec![if_on(Val(0))]);
        assert_eq!(
            shallowly_merge_conditionals(&CfgsDataflowConditionalTree::new(&unit), None),
            None
        );
    }

    /// 🎯 381/384 — A ROOT-CHILD CONDITIONAL REACHES THE CANDIDATE QUESTION. One `scf.if` at the top
    /// of the unit is a child of the root (`:473-474`), so it is offered to `parseConditional` (`:478`).
    #[test]
    #[should_panic(expected = "parseConditional")]
    fn a_top_level_conditional_reaches_the_candidate_question() {
        let unit = unit_holding(vec![if_on(Val(0))]);
        simplify_value_based_conditionals(&CfgsDataflowConditionalTree::new(&unit));
    }

    /// 🎯⛔ AND A CONDITIONAL UNDER A LOOP IS A ROOT CHILD TOO — `findClosestParent` skips the
    /// unselected `affine.for` (`dcc/src/Analysis/ConditionalTree.hpp:165-169`), exactly as entry
    /// 380's own sibling walk has it, so this one is offered to the question as well.
    #[test]
    #[should_panic(expected = "parseConditional")]
    fn a_conditional_under_a_loop_is_a_root_child() {
        let unit = unit_holding(vec![DfirOp::Affine(affine::Op::For {
            iv: Val(1),
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(4),
            carried: Vec::new(),
            body: vec![if_on(Val(0))],
            dbg_name: None,
        })]);
        simplify_value_based_conditionals(&CfgsDataflowConditionalTree::new(&unit));
    }

    // ══════════════════════════════════════════════════════════════════════════════════════════
    // 284/384, 285/384
    // ══════════════════════════════════════════════════════════════════════════════════════════

    /// 🎯 284/384 — THE PATTERN THE REFERENCE DRAWS AT `:98-104` REACHES THE HOIST: a conditional in the
    /// then-arm and an equivalent one in the else-arm of the same parent, both side-effect-free and both
    /// first in their block, so the legality triple at `:159-161` holds.
    #[test]
    #[should_panic(expected = "moveAncestorsToMaintainDominance")]
    fn a_conditional_common_to_both_arms_reaches_the_hoist() {
        let unit = unit_holding(vec![DfirOp::Scf(scf::Op::If {
            cond: Val(0),
            results: Vec::new(),
            body: vec![if_on(Val(1))],
            else_body: vec![if_on(Val(1))],
            dbg_name: None,
        })]);
        let _ = hoist_common_conditionals(&CfgsDataflowConditionalTree::new(&unit), None);
    }

    /// 🎯 284/384 — AND TWO CONDITIONS THAT ARE DIFFERENT REGION ARGUMENTS ARE NOT COMMON. This is the
    /// `/*all_block_args_are_equiv*/ false` entry 100 constructs the comparison with
    /// (`.hpp:112-115`): with the default `true` these two `scf.if`s would compare EQUIVALENT and the
    /// pass would hoist one conditional over another's condition.
    #[test]
    fn two_conditionals_on_different_conditions_are_not_common() {
        let unit = unit_holding(vec![DfirOp::Scf(scf::Op::If {
            cond: Val(0),
            results: Vec::new(),
            body: vec![if_on(Val(1))],
            else_body: vec![if_on(Val(2))],
            dbg_name: None,
        })]);
        assert!(
            hoist_common_conditionals(&CfgsDataflowConditionalTree::new(&unit), None).is_none(),
            "no pair is equivalent, so there is nothing to resume from"
        );
    }

    /// ONE `simplify-conditional.mlir` CASE, AS THE LOOP AND THE CONDITIONAL ENTRY 285 IS HANDED.
    ///
    /// The conditional is `%16 = scf.if %12 -> (index)` (`:295-314`) with ONE level instead of four,
    /// because `replaceIfOpByIterArg` never looks inside it — `parseConditional` did, and the sequence
    /// it read is the [`YieldedSequence`] argument. ⚠️ The consumer `%17 =
    /// dataflow.get_logical_memory_view %11, %16` (`:315`) spends a
    /// [`Link`](crate::islands::dataflow_ir::link::Link) a unit test cannot mint (the limit
    /// [`statement`] records), so what must be re-pointed at the iteration argument is an `arith` op
    /// reading the same value.
    fn a_sequence_yielding_loop(bounds: CountedBounds) -> (DfirOp, DfirOp) {
        let conditional = DfirOp::Scf(scf::Op::If {
            cond: Val(12),
            results: vec![Val(16)],
            body: vec![DfirOp::Scf(scf::Op::Yield {
                operands: vec![Val(20)],
            })],
            else_body: vec![DfirOp::Scf(scf::Op::Yield {
                operands: vec![Val(21)],
            })],
            dbg_name: None,
        });
        let consumer = DfirOp::Arith(arith::Op::MulI(arith::IntBinary {
            result: Val(17),
            lhs: Val(16),
            rhs: Val(16),
            ty: ScalarTy::Index,
        }));
        let body = |terminator| vec![conditional.clone(), consumer.clone(), terminator];
        let loop_op = match bounds {
            CountedBounds::Affine { lo, hi } => DfirOp::Affine(affine::Op::For {
                iv: Val(29),
                lo,
                hi,
                carried: Vec::new(),
                body: body(DfirOp::Affine(affine::Op::Yield {
                    operands: Vec::new(),
                })),
                dbg_name: None,
            }),
            CountedBounds::Scf { lo, hi, step } => DfirOp::Scf(scf::Op::For {
                iv: Val(29),
                lo,
                hi,
                step,
                carried: Vec::new(),
                body: body(DfirOp::Scf(scf::Op::Yield {
                    operands: Vec::new(),
                })),
                dbg_name: None,
            }),
        };
        (conditional, loop_op)
    }

    /// The three constants and the new loop, printed — everything entry 285 emits.
    fn emitted(replacement: &IterArgReplacement) -> String {
        let mut got = String::new();
        if let IterArgReplacement::IterArg { before, op, .. } = replacement {
            for one in before.iter().chain(core::iter::once(op)) {
                print::emit(&mut got, one, 0);
            }
        }
        got
    }

    /// 🎯 285/384 — THE VENDOR'S OWN OUTPUT FOR CASE 2, VERBATIM. `simplify-conditional.mlir:47-63`
    /// freezes `%34 = arith.constant 0`, `%35 = arith.constant 1024`, `%36 = arith.constant 0` (entry
    /// 264's now-dead affine fill), `%37 = affine.for %38 = 0 to 4 iter_args(%39 = %34) -> (index)`,
    /// `%46 = arith.addi %39, %35` and `affine.yield %46` — and the values are minted from 34 so the
    /// names line up with it. ⭐ THE GAP AT `%40` IS THE ERASED CONDITIONAL's clone: MLIR renumbers at
    /// print time over live ops, this island prints the [`Val`] it was minted.
    #[test]
    fn the_vendor_affine_case_carries_the_sequence_in_a_new_iteration_argument() {
        let (conditional, loop_op) = a_sequence_yielding_loop(CountedBounds::Affine {
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(4),
        });
        let mut vals = Values::default();
        while vals.issued() < 34 {
            vals.mint();
        }

        let replacement = replace_if_op_by_iter_arg(
            ValueYieldingConditional::of(&conditional).expect("an scf.if binding one value"),
            &CountedLoop::of(&loop_op).expect("an affine.for"),
            Iterations::between(LoopBound(0), LoopBound(4), LoopStep::ONE),
            YieldedSequence {
                lb: YieldedIndex(0),
                step: 1024,
            },
            &mut vals,
        );

        assert!(
            matches!(&replacement, IterArgReplacement::IterArg { replacements, .. } if replacements.is_empty()),
            "four iterations is not one, and the old loop bound no results"
        );
        assert_eq!(
            emitted(&replacement),
            concat!(
                "%34 = arith.constant 0 : index\n",
                "%35 = arith.constant 1024 : index\n",
                "%36 = arith.constant 0 : index\n",
                "%37 = affine.for %38 = 0 to 4 iter_args(%39 = %34) -> (index) {\n",
                "  %41 = arith.muli %39, %39 : index\n",
                "  %42 = arith.addi %39, %35 : index\n",
                "  affine.yield %42 : index\n",
                "}\n",
            )
        );
    }

    /// 🎯 285/384 — THE CORRECTED `scf.for` ARM, AND THE REGRESSION THE REFERENCE'S OWN EXPECTATION IS.
    /// `simplify-conditional.mlir:73` freezes `scf.for %51 = %47 to %11 step %6 iter_args(%52 = %49)`:
    /// `start_val` landed in the LOWER BOUND and the iteration argument kept entry 264's fill of `1`,
    /// which makes the sequence `1, 1025, 2049, 3073` where `seq_lb_` is 0. Here the bound stays `%4`
    /// and the argument starts at `%47`; the dead fill `%49` is still emitted because entry 264 creates
    /// it, so everything else about the print is the vendor's.
    #[test]
    fn the_scf_arm_initialises_the_argument_and_not_the_lower_bound() {
        let (conditional, loop_op) = a_sequence_yielding_loop(CountedBounds::Scf {
            lo: Val(4),
            hi: Val(11),
            step: Val(6),
        });
        let mut vals = Values::default();
        while vals.issued() < 47 {
            vals.mint();
        }

        let replacement = replace_if_op_by_iter_arg(
            ValueYieldingConditional::of(&conditional).expect("an scf.if binding one value"),
            &CountedLoop::of(&loop_op).expect("an scf.for"),
            Iterations::between(LoopBound(0), LoopBound(4), LoopStep::ONE),
            YieldedSequence {
                lb: YieldedIndex(0),
                step: 1024,
            },
            &mut vals,
        );

        assert_eq!(
            emitted(&replacement),
            concat!(
                "%47 = arith.constant 0 : index\n",
                "%48 = arith.constant 1024 : index\n",
                "%49 = arith.constant 1 : index\n",
                "%50 = scf.for %51 = %4 to %11 step %6 iter_args(%52 = %47) -> (index) {\n",
                "  %54 = arith.muli %52, %52 : index\n",
                "  %55 = arith.addi %52, %48 : index\n",
                "  scf.yield %55 : index\n",
                "}\n",
            )
        );
    }

    /// 🎯 285/384 — AND A ONE-TRIP LOOP TAKES A CONSTANT INSTEAD (`:622-630`), which is the vendor's
    /// case 1: the `affine.for %29 = 0 to 1`'s conditional becomes `%31 = arith.constant 0 : index`
    /// (`:42`) and the loop is left alone.
    #[test]
    fn a_one_trip_loop_takes_a_constant_instead() {
        let (conditional, loop_op) = a_sequence_yielding_loop(CountedBounds::Affine {
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(1),
        });
        let mut vals = Values::default();
        while vals.issued() < 31 {
            vals.mint();
        }

        let replacement = replace_if_op_by_iter_arg(
            ValueYieldingConditional::of(&conditional).expect("an scf.if binding one value"),
            &CountedLoop::of(&loop_op).expect("an affine.for"),
            Iterations::between(LoopBound(0), LoopBound(1), LoopStep::ONE),
            YieldedSequence {
                lb: YieldedIndex(0),
                step: 1024,
            },
            &mut vals,
        );

        let IterArgReplacement::Constant { op, replacement } = replacement else {
            panic!("one iteration is not four");
        };
        let mut got = String::new();
        print::emit(&mut got, &op, 0);
        assert_eq!(got, "%31 = arith.constant 0 : index\n");
        assert_eq!(
            replacement,
            (Val(16), Val(31)),
            "the conditional's own result, forwarded to the constant"
        );
    }

    /// THE COMPARISON THE CONDITIONAL TREE CONFIGURES — entry 100's own three arguments.
    fn tree_oe() -> OperationEquivalence {
        OperationEquivalence::tagged(
            EquivalenceTag::CfgMergingAndHoistingCondTree,
            SubregionCompare::Recursive,
            BlockArgEquivalence::SameOwnerAndIndex,
        )
    }

    /// 🎯 347/384 — THE SET AND THE OPERANDS BOTH DECIDE AN AFFINE PAIR, THE CONDITION'S DEFINITION
    /// DECIDES AN SCF PAIR, AND A MIXED PAIR NEVER MATCHES.
    #[test]
    fn two_conditions_match_only_within_one_dialect() {
        let affine_if = |dims: u32, args: Vec<Val>| {
            DfirOp::Affine(affine::Op::If {
                set: IntegerSet {
                    dims,
                    symbols: 0,
                    constraints: Vec::new(),
                },
                args,
                symbol_args: Vec::new(),
                results: Vec::new(),
                body: Vec::new(),
                else_body: Vec::new(),
                dbg_name: None,
            })
        };
        let cmpi = |result: Val, lhs: Val, rhs: Val| {
            DfirOp::Arith(arith::Op::Compare {
                result,
                predicate: arith::CmpIPredicate::Eq,
                lhs,
                rhs,
            })
        };
        let scope = vec![
            DfirOp::Arith(arith::Op::Constant {
                result: Val(0),
                value: 0,
            }),
            DfirOp::Arith(arith::Op::Constant {
                result: Val(1),
                value: 0,
            }),
            cmpi(Val(30), Val(26), Val(0)),
            cmpi(Val(31), Val(26), Val(1)),
            cmpi(Val(32), Val(27), Val(0)),
        ];
        let oe = tree_oe();

        // The same set read with the same region argument.
        let one = affine_if(1, vec![Val(9)]);
        assert!(top_level_conditions_match(&oe, &one, &one, &scope));
        // ⛔ A DIFFERENT ARGUMENT OF THE SAME SHAPE IS A DIFFERENT CONDITION (`:295`).
        let other_arg = affine_if(1, vec![Val(10)]);
        assert!(!top_level_conditions_match(&oe, &one, &other_arg, &scope));
        // ⛔ AND SO IS THE SAME ARGUMENT UNDER A DIFFERENT SET (`:285-286`).
        let other_set = affine_if(2, vec![Val(9)]);
        assert!(!top_level_conditions_match(&oe, &one, &other_set, &scope));
        // ⛔ ONE SIDE A REGION ARGUMENT AND THE OTHER A RESULT (`:296-297`).
        let a_result = affine_if(1, vec![Val(30)]);
        assert!(!top_level_conditions_match(&oe, &one, &a_result, &scope));

        // Two `scf.if`s on two `arith.cmpi eq` that compute the same thing.
        assert!(top_level_conditions_match(
            &oe,
            &if_on(Val(30)),
            &if_on(Val(31)),
            &scope
        ));
        // ⛔ A DIFFERENT LEFT-HAND SIDE IS A DIFFERENT CONDITION.
        assert!(!top_level_conditions_match(
            &oe,
            &if_on(Val(30)),
            &if_on(Val(32)),
            &scope
        ));
        // ⛔ AND A MIXED PAIR IS THE REFERENCE'S OWN `else` (`:308-309`), whatever they test.
        assert!(!top_level_conditions_match(
            &oe,
            &one,
            &if_on(Val(30)),
            &scope
        ));
    }

    /// 🎯 348/384 — `merging.mlir:129-133`'S TWO ARMS EACH YIELD THEIR CONSTANT, AND EVERY OTHER
    /// SHAPE DECLINES.
    ///
    /// `%13 = scf.if %12 -> (index) { scf.yield %c1 } else { scf.yield %c2 }`.
    #[test]
    fn each_single_statement_arm_yields_its_constant() {
        let scope = vec![
            DfirOp::Arith(arith::Op::Constant {
                result: Val(1),
                value: 1,
            }),
            DfirOp::Arith(arith::Op::Constant {
                result: Val(2),
                value: 2,
            }),
        ];
        let yielding = |operands: Vec<Val>| DfirOp::Scf(scf::Op::Yield { operands });
        let if_op = DfirOp::Scf(scf::Op::If {
            cond: Val(12),
            results: vec![Val(13)],
            body: vec![yielding(vec![Val(1)])],
            else_body: vec![yielding(vec![Val(2)])],
            dbg_name: None,
        });
        assert_eq!(
            single_op_branch_to_yield_val(Some(&if_op), Arm::Then, &scope),
            Some(YieldedIndex(1))
        );
        assert_eq!(
            single_op_branch_to_yield_val(Some(&if_op), Arm::Else, &scope),
            Some(YieldedIndex(2))
        );

        // ⛔ `if (!op) return false;` (`:500`).
        assert_eq!(single_op_branch_to_yield_val(None, Arm::Then, &scope), None);
        // ⛔ AN ARM THAT COMPUTES SOMETHING IS TWO OPERATIONS (`:505`).
        let mut computing = if_op.clone();
        if let DfirOp::Scf(scf::Op::If { body, .. }) = &mut computing {
            body.insert(0, statement(4));
        }
        assert_eq!(
            single_op_branch_to_yield_val(Some(&computing), Arm::Then, &scope),
            None
        );
        // ⛔ A YIELDED REGION ARGUMENT HAS NO DEFINING OP (`:509-510`), and an `else` with no block
        // never reaches `.front()` here at all.
        let un_named = DfirOp::Scf(scf::Op::If {
            cond: Val(12),
            results: vec![Val(13)],
            body: vec![yielding(vec![Val(9)])],
            else_body: Vec::new(),
            dbg_name: None,
        });
        assert_eq!(
            single_op_branch_to_yield_val(Some(&un_named), Arm::Then, &scope),
            None
        );
        assert_eq!(
            single_op_branch_to_yield_val(Some(&un_named), Arm::Else, &scope),
            None
        );
    }

    /// 🎯 349/384 — A CONDITION COMPUTED OUTSIDE THE LOOP IS INVARIANT; THE SAME CONDITIONAL ON THE
    /// LOOP'S OWN INDUCTION VARIABLE IS NOT.
    #[test]
    fn only_a_condition_free_of_the_loops_argument_is_invariant() {
        // `scf.for %20 { %40 = scf.if %cond -> (index) { scf.yield %5 } else { scf.yield %7 } }`.
        let scope_on = |cond: Val| {
            vec![
                DfirOp::Arith(arith::Op::Constant {
                    result: Val(5),
                    value: 5,
                }),
                DfirOp::Arith(arith::Op::Constant {
                    result: Val(7),
                    value: 7,
                }),
                // `%30` reads a value from outside the loop; `%31` reads its induction variable.
                DfirOp::Arith(arith::Op::Compare {
                    result: Val(30),
                    predicate: arith::CmpIPredicate::Eq,
                    lhs: Val(9),
                    rhs: Val(5),
                }),
                DfirOp::Arith(arith::Op::Compare {
                    result: Val(31),
                    predicate: arith::CmpIPredicate::Eq,
                    lhs: Val(20),
                    rhs: Val(5),
                }),
                DfirOp::Scf(scf::Op::For {
                    iv: Val(20),
                    lo: Val(5),
                    hi: Val(7),
                    step: Val(5),
                    carried: Vec::new(),
                    body: vec![DfirOp::Scf(scf::Op::If {
                        cond,
                        results: vec![Val(40)],
                        body: vec![DfirOp::Scf(scf::Op::Yield {
                            operands: vec![Val(5)],
                        })],
                        else_body: vec![DfirOp::Scf(scf::Op::Yield {
                            operands: vec![Val(7)],
                        })],
                        dbg_name: None,
                    })],
                    dbg_name: None,
                }),
            ]
        };
        let answer = |cond: Val| {
            let scope = scope_on(cond);
            let DfirOp::Scf(scf::Op::For { body, .. }) = &scope[4] else {
                unreachable!("the fixture's last statement is the loop")
            };
            let n = ScfConditional::of(&body[0]).expect("the fixture's conditional is an scf.if");
            is_loop_invariant(n, &scope[4], &scope)
        };
        assert!(answer(Val(30)));
        assert!(!answer(Val(31)));
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 095/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHICH OF THE TWO CONDITIONAL OPS ONE STATEMENT IS.
///
/// # ⛔⛔ A TYPE, BECAUSE TWO FUNCTIONS ASK AND ONE OF THEM ABORTS ON THE ANSWER
///
/// `isOperationSelected` (entry 095) needs only *whether*; `createDummyYieldInElseReg` (entry 096)
/// needs *which*, and closes with `llvm_unreachable("unexpected IfOp type")` for anything else
/// (`CFGSDataflowConditionalTree.cpp:393`). Naming the two kinds once makes the second question
/// answerable without a second `isa<>` chain, and makes that `llvm_unreachable` UNWRITABLE: there is
/// no third variant to fall past.
///
/// ⭐ AND IT IS THE WHOLE OF THE REFERENCE'S LIST. `isa<mlir::affine::AffineIfOp, scf::IfOp>` names
/// exactly these two (`:34`) — no `sentient.if`, which is one rung down, and no loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionalKind {
    /// `affine.if` — `mlir::affine::AffineIfOp`, the DataflowIR rung's own conditional, branching on
    /// whether its operands satisfy an integer set.
    Affine,
    /// `scf.if` — `scf::IfOp`, branching on an `i1` the program already computed.
    Scf,
}

impl ConditionalKind {
    /// WHICH KIND ONE STATEMENT IS, or `None` for a statement that is not a conditional.
    ///
    /// ⛔ TOTAL OVER THE ISLAND'S OPS, NO WILDCARD ANYWHERE. This is the tree's entire node
    /// predicate, so an op added to the island must say whether it is a conditional rather than
    /// inheriting `false` — a wildcard here would silently shrink every conditional tree the moment
    /// a third `if` form arrived.
    #[must_use]
    pub fn of(op: &DfirOp) -> Option<ConditionalKind> {
        match op {
            // ── the two the reference names ──────────────────────────────────────────────────────
            DfirOp::Affine(affine::Op::If { .. }) => Some(ConditionalKind::Affine),
            DfirOp::Scf(scf::Op::If { .. }) => Some(ConditionalKind::Scf),

            // ── everything else, spelled out per dialect the predicate mentions ──────────────────
            DfirOp::Affine(
                affine::Op::For { .. }
                | affine::Op::Apply { .. }
                | affine::Op::Yield { .. }
                | affine::Op::VectorLoad { .. }
                | affine::Op::VectorStore { .. },
            ) => None,
            DfirOp::Scf(scf::Op::Yield { .. } | scf::Op::Parallel { .. } | scf::Op::For { .. }) => {
                None
            }

            // ── dialects `isa<AffineIfOp, IfOp>` does not mention at all ─────────────────────────
            DfirOp::Arith(_)
            | DfirOp::Dataflow(_)
            | DfirOp::Agen(_)
            | DfirOp::Vector(_)
            | DfirOp::VectorChain(_)
        // ⭐ `uniform` IS NOT MENTIONED EITHER. A `uniformize_regions` carries regions, but they are
        // one per unit class rather than the two arms of a branch — see [`arms_of`].
        | DfirOp::Uniform(_)
            | DfirOp::Symbol(_) => None,
        }
    }
}

/// Replaces: e095_isOperationSelected
///
/// # WHETHER THE CONDITIONAL TREE TAKES THIS OP AS A NODE
///
/// ```cpp
/// bool CFGSDataflowConditionalTree::isOperationSelected(const Operation &op) {
///   return isa<mlir::affine::AffineIfOp, scf::IfOp>(&op);
/// }
/// ```
/// (`dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:33-35`)
///
/// ⭐⭐ TWO LINES THAT DECIDE WHAT SEVEN PASSES SEE. `OperationTree::compute()` builds its tree out
/// of the ops this accepts, so "the tree is empty" is a question about UNDECIDED BRANCHES and not
/// about program size — every one of `CFGSimplificationDataflowLevel`'s seven steps is a no-op on a
/// unit for which this answers `false` everywhere. `opHasSideEffect` (entry 176) also calls it, to
/// exclude a nested conditional from its own side-effect scan (`:40`).
///
/// ⛔ THE `const Operation &` IS TAKEN BY SHARED REFERENCE HERE TOO — this asks a question and
/// changes nothing, which is what separates it from entry 096 next door.
#[must_use]
pub fn is_operation_selected(op: &DfirOp) -> bool {
    ConditionalKind::of(op).is_some()
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 096/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// A CONDITIONAL WHOSE `else` REGION HAS NO BLOCK AND WHICH BINDS NOTHING — entry 096's two
/// `DT_CHECK`s, as a type.
///
/// # ⛔⛔ THE PRECONDITIONS ARE THE MINTING RULE, NOT A RUNTIME TEST
///
/// ```cpp
/// DT_CHECK_MSG(if_op, "Expect valid op.");
/// DT_CHECK_MSG(
///     if_op->getNumResults() == 0 && if_op->getRegions()[1].empty(),
///     "Expect conditionals with empty else regions to yield no values.");
/// ```
/// (`CFGSDataflowConditionalTree.cpp:384-387`)
///
/// Three facts, each discharged where it is stated:
///
/// * `if_op` non-null — ⛔ **UNREPRESENTABLE HERE.** [`Self::of`] takes `&mut DfirOp`, and a Rust
///   reference is never null.
/// * `getNumResults() == 0` — a conditional of either kind whose `results` list is non-empty is
///   declined by [`Self::of`]. ⭐ THE `scf.if` ARM CHECKS IT TOO, since entry 177 needed that op to
///   carry the result list the vendor's own input binds
///   (`dcc/test/Transform/CFGSimplificationDataflowLevel/merging.mlir:129`).
/// * `getRegions()[1].empty()` — declined by [`Self::of`] when the `else` region already holds a
///   block. ⭐ This is the one that makes the whole function observable: see the note on
///   [`super::super::islands::dataflow_ir::dialects::scf::Op::If::else_body`] for why "no block" and
///   "a block holding only a terminator" are two different ops and print differently.
///
/// ⭐ AND THE KIND TRAVELS WITH IT, so [`create_dummy_yield_in_else_reg`] cannot reach the
/// reference's `llvm_unreachable`.
#[derive(Debug)]
pub struct EmptyElseRegion<'a> {
    /// Which conditional it is — decides which dialect's terminator goes in.
    kind: ConditionalKind,
    /// The `else` region's statement list, empty at minting time.
    else_body: &'a mut Vec<DfirOp>,
}

impl<'a> EmptyElseRegion<'a> {
    /// THE ONLY WAY TO MINT ONE — `None` for any op the reference's two checks would stop on.
    #[must_use]
    pub fn of(op: &'a mut DfirOp) -> Option<EmptyElseRegion<'a>> {
        match op {
            // ⛔ THE RESULT LIST IS CHECKED FIRST, as the reference's conjunction reads it. An
            // `affine.if` that yields a value cannot take a bare `affine.yield` in one arm and a
            // value-carrying one in the other.
            DfirOp::Affine(affine::Op::If {
                results, else_body, ..
            }) if results.is_empty() && else_body.is_empty() => Some(EmptyElseRegion {
                kind: ConditionalKind::Affine,
                else_body,
            }),
            DfirOp::Scf(scf::Op::If {
                results, else_body, ..
            }) if results.is_empty() && else_body.is_empty() => Some(EmptyElseRegion {
                kind: ConditionalKind::Scf,
                else_body,
            }),
            _ => None,
        }
    }

    /// Which conditional this is.
    #[must_use]
    pub fn kind(&self) -> ConditionalKind {
        self.kind
    }
}

/// Replaces: e096_createDummyYieldInElseReg
///
/// # ADD A DUMMY YIELD TO THE `else` REGION
///
/// ```cpp
/// /// Add a dummy yield op to the else region of \p if_op.
/// /// Expect \p if_op's else region to be empty, and \p if_op to yield no values.
/// static void createDummyYieldInElseReg(Operation *if_op) {
///   DT_CHECK_MSG(if_op, "Expect valid op.");
///   DT_CHECK_MSG(
///       if_op->getNumResults() == 0 && if_op->getRegions()[1].empty(),
///       "Expect conditionals with empty else regions to yield no values.");
///   if_op->getRegions()[1].push_back(new Block);
///   OpBuilder builder(if_op->getRegions()[1]);
///   if (isa<scf::IfOp>(if_op))
///     scf::YieldOp::create(builder, if_op->getLoc());
///   else if (isa<affine::AffineIfOp>(if_op))
///     affine::AffineYieldOp::create(builder, if_op->getLoc());
///   else
///     llvm_unreachable("unexpected IfOp type");
/// }
/// ```
/// (`dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:383-396`)
///
/// # ⭐⭐ WHY A ONE-OP FUNCTION MATTERS: `mergeShallow` MERGES REGION *i* AGAINST REGION *i*
///
/// `mergeShallow` (entry 177) walks `i` over both regions and takes `src->getRegions()[i].front()`
/// and `dst->getRegions()[i].front()` — the FRONT BLOCK of each. A region with no block has no
/// front, so when exactly one side's region is empty this runs on that side first
/// (`:409-412`) and the merge then has two blocks to splice. Skipping it does not lose a terminator;
/// it dereferences a block that is not there.
///
/// ⛔ THE OP IS THE `else` REGION'S ONLY STATEMENT AND CARRIES NO OPERANDS. `scf::YieldOp::create`
/// and `AffineYieldOp::create` are both called with a location and nothing else, and the op yields
/// nothing because the conditional binds nothing — which is exactly the state
/// [`EmptyElseRegion`] certifies.
///
/// ⛔ WHICH DIALECT'S TERMINATOR IS NOT A FREE CHOICE. An `scf.if` verifies that its regions end in
/// an `scf.yield` and an `affine.if` in an `affine.yield`; putting the other one in is an invalid op,
/// not a stylistic difference. The reference distinguishes them with two `isa<>` tests and stops on
/// a third possibility — here the possibility does not exist ([`ConditionalKind`]).
pub fn create_dummy_yield_in_else_reg(if_op: EmptyElseRegion<'_>) {
    // `if_op->getRegions()[1].push_back(new Block)` and the builder positioned in it are the
    // MECHANISM for reaching an insertion point; the island's `else_body` IS the block.
    let terminator = match if_op.kind {
        ConditionalKind::Scf => DfirOp::Scf(scf::Op::Yield {
            operands: Vec::new(),
        }),
        ConditionalKind::Affine => DfirOp::Affine(affine::Op::Yield {
            operands: Vec::new(),
        }),
    };
    if_op.else_body.push(terminator);
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 176/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// THE NINE OPS THE SIDE-EFFECT SCAN NAMES AS HARMLESS — `opHasSideEffect`'s `isa<>` list.
///
/// ```cpp
/// !isa<arith::ConstantOp, arith::ConstantIndexOp, arith::ConstantIntOp,
///      arith::CmpIOp, scf::YieldOp, scf::ForOp, affine::AffineForOp,
///      affine::AffineYieldOp, mlir::symbol::CreateSymbolOp>(op)
/// ```
/// (`CFGSDataflowConditionalTree.cpp:42-44`)
///
/// ⛔ TOTAL OVER THE ISLAND'S OPS, NO WILDCARD. Falling past this list means *"has a side effect"*, so
/// a wildcard would make every op added to the island harmless by default — and the visible effect of
/// that is a conditional merged across an op that must not move.
fn is_named_harmless(op: &DfirOp) -> bool {
    match op {
        // `arith::ConstantOp, arith::ConstantIndexOp, arith::ConstantIntOp` — ⭐ ONE OP CLASS IN THE
        // ISLAND'S THREE SHAPES. The two `Constant*Op`s of the list are casters over `arith.constant`,
        // and `isa<arith::ConstantOp>` matches ANY of them — the dense vector form included, which is
        // what `%3 = arith.constant dense<1.000000e+00> : vector<64xf16>` is
        // (`dcc/test/Transform/CFGSimplificationDataflowLevel/merging-shallow-skip.mlir:16`).
        DfirOp::Arith(
            arith::Op::Constant { .. }
            | arith::Op::ConstantInt { .. }
            | arith::Op::DenseConstant { .. },
        ) => true,

        // `arith::CmpIOp` — the condition of an `scf.if` is one, so the scan has to let it through.
        DfirOp::Arith(arith::Op::Compare { .. }) => true,

        // ⛔ ARITHMETIC IS NOT ON THE LIST. `arith.addi` is `Pure` in MLIR and this still answers
        // *"has a side effect"* for it — see [`op_has_side_effect`] on why that is the point.
        DfirOp::Arith(
            arith::Op::AddI(_)
            | arith::Op::SubI(_)
            | arith::Op::MulI(_)
            | arith::Op::DivSI(_)
            | arith::Op::RemSI(_)
            // ⛔ AND NEITHER CONVERSION IS ON THE LIST, so both answer *"has a side effect"* — the
            // conservative direction, exactly as the arithmetic above does.
            | arith::Op::SiToFp(_)
            | arith::Op::FpToSi(_)
            | arith::Op::Logic { .. },
        ) => false,

        // ⛔ `arith::SelectOp` IS NOT ON THE LIST EITHER, and the omission is the reference's, not a
        // gap here: the `isa<>` above names three constant forms and `CmpIOp`, and a `select` is
        // neither. It is `Pure` in MLIR (`Arith.td`, `SelectOp`) exactly as the arithmetic above is,
        // and this still answers *"has a side effect"* for it — the conservative direction.
        DfirOp::Arith(arith::Op::Select { .. }) => false,

        // `scf::YieldOp, scf::ForOp`.
        DfirOp::Scf(scf::Op::Yield { .. } | scf::Op::For { .. }) => true,

        // ⭐ AN `scf.if` IS ON NO LIST EITHER — [`is_operation_selected`] answers for it first, which
        // is the `!isOperationSelected(*op) &&` half of the same condition. `scf.parallel` is on
        // neither, so a conditional does not hoist or merge across one.
        DfirOp::Scf(scf::Op::If { .. } | scf::Op::Parallel { .. }) => false,

        // `affine::AffineForOp, affine::AffineYieldOp`.
        DfirOp::Affine(affine::Op::For { .. } | affine::Op::Yield { .. }) => true,

        // `affine.if` is [`is_operation_selected`]'s again; `affine.apply` and the vector transfers
        // are named nowhere.
        DfirOp::Affine(
            affine::Op::If { .. }
            | affine::Op::Apply { .. }
            | affine::Op::VectorLoad { .. }
            | affine::Op::VectorStore { .. },
        ) => false,

        // `mlir::symbol::CreateSymbolOp` — the only member of that dialect the list names.
        DfirOp::Symbol(symbol::Op::CreateSymbol { .. }) => true,
        // ⛔ AND THE OTHER TWO ARE NOT ON IT. `symbol.query_map` and its mapping fall past the nine,
        // so a conditional does not move across one — the same answer `uniform.query_map` gets below.
        DfirOp::Symbol(symbol::Op::ImmutableMapping { .. } | symbol::Op::QueryMap { .. }) => false,

        // ── dialects the list does not mention at all ────────────────────────────────────────────
        // ⭐ AND THIS IS THE ARM THAT DOES THE WORK: `dataflow.receive`, `dataflow.send`,
        // `vectorchain.multiply_and_accumulate` and the composite transfers are exactly the ops
        // between the two conditionals of `merging-shallow-skip.mlir:55-59` that the pass refuses to
        // merge across.
        // ⭐ `uniform` IS ON NO LIST EITHER, and the answer is the one that blocks movement: the
        // nine-op `isa<>` above names no `uniform.` op, so a conditional does not hoist or merge
        // across a `uniformize_regions`, a `query_map` or a mapping definition.
        // ⭐ AND `vector.store` IS THE PLAINEST MEMBER OF THAT ARM: the nine-op list names neither it
        // nor `vector.load`, and a write to a view is exactly the effect a conditional must not be
        // moved across.
        DfirOp::Dataflow(_)
        | DfirOp::Agen(_)
        | DfirOp::VectorChain(_)
        | DfirOp::Vector(_)
        | DfirOp::Uniform(_) => false,
    }
}

/// Replaces: e176_opHasSideEffect
///
/// # WHETHER ANYTHING IN THIS SUBTREE IS NOT ON THE LIST
///
/// ```cpp
/// bool CFGSDataflowConditionalTree::opHasSideEffect(Operation &op) {
///   bool result = false;
///   op.walk<WalkOrder::PreOrder>([&](Operation *op) {
///     if (!isOperationSelected(*op) &&
///         !isa<arith::ConstantOp, arith::ConstantIndexOp, arith::ConstantIntOp,
///              arith::CmpIOp, scf::YieldOp, scf::ForOp, affine::AffineForOp,
///              affine::AffineYieldOp, mlir::symbol::CreateSymbolOp>(op)) {
///       result = true;
///       return WalkResult::interrupt();
///     }
///     return WalkResult::advance();
///   });
///   return result;
/// }
/// ```
/// (`dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:38-50`)
///
/// # ⛔⛔ THE NAME SAYS "SIDE EFFECT"; THE TEST IS MEMBERSHIP IN A LIST OF ELEVEN OPS
///
/// This is not `mlir::isMemoryEffectFree` and not the `Pure` trait. `arith.addi` is pure and this
/// answers `true` for it; `scf.for` may contain a `dataflow.send` and this answers `false` for the
/// loop itself — the walk finds the `send` inside it. What the function really means is *"this subtree
/// is nothing but conditionals, counted loops, integer constants, one comparison and a symbol"*, and
/// its three callers use it as a **movement permit**:
///
/// * `isHoistable` — refuses to hoist the op, and refuses if ANY op before it in the block is
///   unlisted (`:82-92`);
/// * `areShallowlyMergeable` — refuses to merge two conditionals when an op strictly between them is
///   unlisted (`:359`);
/// * the deep-merge tree's own copy asks it of both conditionals (`Sentient/Analyses/CFGDeepMergingConditionalTree.cpp:207`).
///
/// ⭐ SO THE OBSERVABLE IS A MERGE THAT DOES NOT HAPPEN, and the authority tree has a fixture named
/// for it: in `merging-shallow-skip.mlir` the two `scf.if %23` conditionals at `:41` and `:60` are left
/// alone, because between them sit `dataflow.receive`, `arith.sitofp`,
/// `vectorchain.create_affine_mask`, `vectorchain.multiply_and_accumulate` and `dataflow.send`
/// (`:55-59`). In `merging.mlir` the ops between the conditionals that DO merge are `arith.constant`s
/// and an `arith.cmpi`.
///
/// # THE WALK INCLUDES THE OP ITSELF
///
/// `Operation::walk` visits the operation it is called on before descending, which is what makes
/// `opHasSideEffect(*to_hoist)` a question about `to_hoist` and not only about its body. The descent
/// is [`dfir_op::regions`] rather than a second match of its own, and `WalkResult::interrupt()` is the
/// short circuit [`Iterator::any`] already is.
#[must_use]
pub fn op_has_side_effect(op: &DfirOp) -> bool {
    // `if (!isOperationSelected(*op) && !isa<…>(op)) { result = true; return interrupt(); }`
    if !is_operation_selected(op) && !is_named_harmless(op) {
        return true;
    }

    // `return WalkResult::advance();` — into every region of an op that passed.
    dfir_op::regions(op)
        .into_iter()
        .flatten()
        .any(op_has_side_effect)
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 177/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHICH OF THE TWO CONDITIONALS COMES FIRST IN THE BLOCK — the reference's `bool dst_before_src`.
///
/// ⭐ THE ORDER IS NOT A PREFERENCE, IT IS WHERE THE OPS LAND. Program order is what a merge has to
/// preserve: the source's statements go to the END of the destination's region when the destination
/// stands first, and to the START of it when the source does (`CFGSDataflowConditionalTree.cpp:429-431`,
/// `:449-451`). Get it wrong and a `dataflow.send` moves ahead of the `dataflow.receive` feeding it.
///
/// ⛔ AND THE CALLER DOES NOT CHOOSE FREELY EITHER. `shallowlyMergeConditionals` merges the SIBLING
/// into `n` when `n` binds a result — `mergeShallow(src: sibling, dst: n, dst_before_src: true)`,
/// because the merged op has to keep the result types of the one that yields (`:237-241`) — and
/// otherwise merges `n` into its sibling with `dst_before_src: false` (`:244-247`). So the flag is
/// determined by which candidate binds a value, and only ONE of the two ever does (`:348`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeOrder {
    /// `dst_before_src = true` — the destination stands first, so the source's ops are appended.
    DstFirst,
    /// `dst_before_src = false` — the source stands first, so its ops are prepended.
    SrcFirst,
}

/// WHICH REGION OF A CONDITIONAL A SPLICE IS WORKING ON — `getRegions()[0]` and `getRegions()[1]`.
///
/// ⛔ AN INDEX WITH TWO NAMED VALUES, because `mergeShallow`'s `for (unsigned i = 0; i < 2; ++i)`
/// (`CFGSDataflowConditionalTree.cpp:400`) is not a walk over however many regions an op has: it is
/// the two arms of a conditional, and it is the reason entry 096 exists — `then` against `then`,
/// `else` against `else`, never one against the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arm {
    /// Region 0, the `then` arm.
    Then,
    /// Region 1, the `else` arm — the one that may have no block at all.
    Else,
}

impl Arm {
    /// The reference's `i`.
    pub const fn index(self) -> usize {
        match self {
            Arm::Then => 0,
            Arm::Else => 1,
        }
    }
}

/// WHETHER THAT ARM OF THE CONDITIONAL HAS NO BLOCK — `op->getRegions()[i].empty()`.
///
/// ⭐ EMPTY MEANS NO BLOCK, WHICH IS WHY A `Vec` CAN ANSWER IT. A region MLIR would print as a block
/// holding only an elided terminator still holds that terminator here — see
/// [`crate::islands::dataflow_ir::dialects::scf::Op::If::else_body`] — so the only empty statement
/// list is the region entry 096 fills.
fn arm_is_empty(op: &DfirOp, arm: Arm) -> bool {
    dfir_op::regions(op)
        .get(arm.index())
        .is_none_or(|ops| ops.is_empty())
}

/// THE TWO ARMS OF ONE CONDITIONAL, `then` FIRST — `getRegions()[0]` and `getRegions()[1]`, mutably.
///
/// ⛔ TOTAL OVER THE ISLAND'S OPS, NO WILDCARD, for the reason [`ConditionalKind::of`] gives: a third
/// `if` form arriving in the island must say what its arms are rather than inheriting "not a
/// conditional" and being skipped by every merge.
fn arms_of(op: &mut DfirOp) -> Option<(&mut Vec<DfirOp>, &mut Vec<DfirOp>)> {
    match op {
        DfirOp::Scf(scf::Op::If {
            body, else_body, ..
        })
        | DfirOp::Affine(affine::Op::If {
            body, else_body, ..
        }) => Some((body, else_body)),

        DfirOp::Affine(
            affine::Op::For { .. }
            | affine::Op::Apply { .. }
            | affine::Op::Yield { .. }
            | affine::Op::VectorLoad { .. }
            | affine::Op::VectorStore { .. },
        )
        | DfirOp::Scf(
            scf::Op::For { .. } | scf::Op::Yield { .. } | scf::Op::Parallel { .. },
        )
        | DfirOp::Arith(_)
        | DfirOp::Dataflow(_)
        | DfirOp::Agen(_)
        | DfirOp::VectorChain(_)
        | DfirOp::Vector(_)
        // ⭐ A `uniform.uniformize_regions` HAS REGIONS AND IS STILL NOT A CONDITIONAL: it has one per
        // unit class, not a `then` and an `else`, and [`ConditionalKind::of`] declines it for the same
        // reason.
        | DfirOp::Uniform(_)
        | DfirOp::Symbol(_) => None,
    }
}

/// Replaces: e177_mergeShallow
///
/// # SPLICE ONE CONDITIONAL'S ARMS INTO ANOTHER'S, ARM AGAINST ARM
///
/// ```cpp
/// void CFGSDataflowConditionalTree::mergeShallow(Operation *src, Operation *dst,
///                                                bool dst_before_src) {
///   for (unsigned i = 0; i < 2; ++i) {
///     bool src_region_empty = src->getRegions()[i].empty();
///     bool dst_region_empty = dst->getRegions()[i].empty();
///     // No need to merge regions if both are empty.
///     if (src_region_empty && dst_region_empty) continue;
///
///     // If the src/dst region is empty, create a dummy yield to help with the
///     // merging.
///     if (src_region_empty)
///       createDummyYieldInElseReg(src);
///     else if (dst_region_empty)
///       createDummyYieldInElseReg(dst);
///
///     Block &src_bb = src->getRegions()[i].front();
///     Block &dst_bb = dst->getRegions()[i].front();
///     if (dst_before_src) {
///       // Move all of src's ops to the end of dst_bb.
///       // If dst yields results, its new terminator will simply be its old one.
///       // Otherwise the new terminator will be src's terminator.
///       auto *dst_terminator = dst_bb.getTerminator();
///       bool dst_has_no_results = (dst->getNumResults() == 0);
///       if (dst_has_no_results)
///         dst_terminator->erase();
///       else {
///         auto *src_terminator = src_bb.getTerminator();
///         DT_CHECK_MSG(src_terminator->getNumResults() == 0,
///                      "Expect src to not yield any results if dst does.");
///         src_terminator->erase();
///       }
///       while (!src_bb.getOperations().empty()) {
///         Operation &op = src_bb.getOperations().front();
///         op.moveBefore(&dst_bb, dst_bb.end());
///       }
///       if (!dst_has_no_results)
///         dst_terminator->moveBefore(&dst_bb, dst_bb.end());
///     } else {
///       // Move all of src's ops except its terminator to the start of dst_bb.
///       // If src does not yield results, dst's new terminator will simply be its
///       // old one. Otherwise the new terminator will be src's terminator.
///       auto *src_terminator = src_bb.getTerminator();
///       if (src->getNumResults() == 0)
///         src_terminator->erase();
///       else {
///         auto *dst_terminator = dst_bb.getTerminator();
///         DT_CHECK_MSG(dst_terminator->getNumResults() == 0,
///                      "Expect dst to not yield any results if src does.");
///         src_terminator->moveBefore(dst_terminator);
///         dst_terminator->erase();
///       }
///       while (!src_bb.getOperations().empty()) {
///         Operation &op = src_bb.getOperations().back();
///         op.moveBefore(&dst_bb, dst_bb.begin());
///       }
///     }
///   }
///   if (StringAttr new_dbg_name_attr =
///           dataflow::utils::getNewDbgNameFromList("CFGSM(", {src, dst})) {
///     dataflow::setDbgNameAttr(dst, new_dbg_name_attr);
///   }
/// }
/// ```
/// (`dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:398-459`)
///
/// # ⭐⭐ EXACTLY ONE TERMINATOR SURVIVES EACH ARM, AND WHICH ONE IS THE WHOLE FUNCTION
///
/// Two conditionals on the same condition become one, so each arm ends up with two statement lists
/// and two `yield`s where a block may have only one. The rule is that **the yield of the conditional
/// that binds values survives**, because its operands are the merged op's results — and
/// `areShallowlyMergeable` has already guaranteed that at most one of the two binds anything
/// (`:348`). The other three combinations follow from program order:
///
/// | order | who binds | erased | the surviving terminator ends up |
/// |---|---|---|---|
/// | [`MergeOrder::DstFirst`] | neither | `dst`'s | last, and it is `src`'s (`:421-422`, `:429-431`) |
/// | [`MergeOrder::DstFirst`] | `dst` | `src`'s | last, moved back there after the splice (`:433-434`) |
/// | [`MergeOrder::SrcFirst`] | neither | `src`'s | last, and it is `dst`'s (`:440-441`) |
/// | [`MergeOrder::SrcFirst`] | `src` | `dst`'s | last, where `dst`'s used to be (`:446-447`) |
///
/// ⛔ THE TWO `DT_CHECK`s ARE NOT CHECKS ON THE TERMINATORS THEY NAME (`:425-426`, `:444-445`).
/// `src_terminator->getNumResults()`
/// is the yield OP's own result count, which is zero for every `scf.yield` and `affine.yield` ever
/// built — a yield binds nothing, it only carries operands. The condition the messages describe
/// ("Expect src to not yield any results if dst does") is the one `areShallowlyMergeable` enforces on
/// the CONDITIONALS, so nothing is lost by their absence here.
///
/// # ⛔ THE `while` LOOPS ARE A DIRECTION, NOT A COUNT
///
/// `moveBefore(&dst_bb, dst_bb.end())` taken over `front()` repeatedly appends in order (`:429-431`);
/// `moveBefore(&dst_bb, dst_bb.begin())` taken over `back()` repeatedly prepends in order
/// (`:449-451`). Both are
/// "splice src's statements in, keeping their order" — one at the end, one at the start — which is
/// [`Vec::append`] and a prepend of the taken list.
///
/// # ⭐ THE EMPTY-ARM CASE IS WHY ENTRY 096 EXISTS
///
/// A conditional with no `else` block has nothing to splice against, so the side that is missing one
/// gets a bare terminator first ([`create_dummy_yield_in_else_reg`]) and the arm then has two lists.
/// ⚠️ THE REFERENCE FILLS REGION **1** WHATEVER `i` IS — `createDummyYieldInElseReg` indexes
/// `getRegions()[1]` unconditionally (`:388`) — so on `i == 0` it would fill the `else` arm and then
/// dereference an absent `then` block. That is unreachable for a verified conditional, whose `then`
/// region always has a block; here the same call is made and an arm that is still empty splices
/// nothing into nothing.
///
/// # ⛔ THE NAME IS PART OF THE MERGE
///
/// The last three lines are not logging: `dbgName` is how the reference's own `CHECK` lines identify a
/// merged conditional, and the expectation for this file's fixture after two merges is
/// `} {dbgName = "CFGSM(SCF-If #4, CFGSM(SCF-If #2, SCF-If #3))"}`
/// (`dcc/test/Transform/CFGSimplificationDataflowLevel/merging.mlir:59`) — nested, because the second
/// merge reads the name the first one wrote. A conditional whose halves are not both named keeps its
/// own name, which is [`new_dbg_name_from_list`] returning `None`.
///
/// ⭐ THE SOURCE IS LEFT AS AN EMPTY CONDITIONAL, NOT DELETED. `shallowlyMergeConditionals` collects
/// it into `ops_to_delete` and erases it, with its dead ancestors, after the walk (`:242`, `:257`,
/// `:269`) — so this function's postcondition is "src's arms are spliced out", not "src is gone".
pub fn merge_shallow(src: &mut DfirOp, dst: &mut DfirOp, order: MergeOrder) {
    // `for (unsigned i = 0; i < 2; ++i)`
    for arm in [Arm::Then, Arm::Else] {
        // `bool src_region_empty = src->getRegions()[i].empty();`
        let src_region_empty = arm_is_empty(src, arm);
        let dst_region_empty = arm_is_empty(dst, arm);

        // `if (src_region_empty && dst_region_empty) continue;`
        if src_region_empty && dst_region_empty {
            continue;
        }

        // `if (src_region_empty) createDummyYieldInElseReg(src);`
        // `else if (dst_region_empty) createDummyYieldInElseReg(dst);`
        if src_region_empty {
            if let Some(region) = EmptyElseRegion::of(src) {
                create_dummy_yield_in_else_reg(region);
            }
        } else if dst_region_empty {
            if let Some(region) = EmptyElseRegion::of(dst) {
                create_dummy_yield_in_else_reg(region);
            }
        }

        // `dst->getNumResults() == 0` and `src->getNumResults() == 0` — read before the arms are
        // borrowed, since the result list is the op's and not the region's.
        let dst_has_no_results = dfir_op::results(dst).is_empty();
        let src_has_no_results = dfir_op::results(src).is_empty();

        // `Block &src_bb = src->getRegions()[i].front();`
        let (Some(src_arms), Some(dst_arms)) = (arms_of(src), arms_of(dst)) else {
            continue;
        };
        let src_bb = match arm {
            Arm::Then => src_arms.0,
            Arm::Else => src_arms.1,
        };
        let dst_bb = match arm {
            Arm::Then => dst_arms.0,
            Arm::Else => dst_arms.1,
        };

        match order {
            MergeOrder::DstFirst => {
                if dst_has_no_results {
                    // `dst_terminator->erase();` then src's whole list — terminator included —
                    // appended, so src's terminator becomes the arm's.
                    dst_bb.pop();
                    dst_bb.append(src_bb);
                } else {
                    // `src_terminator->erase();` — dst's terminator is the one that must survive, so
                    // it steps aside and goes back last (`:433-434`).
                    src_bb.pop();
                    let dst_terminator = dst_bb.pop();
                    dst_bb.append(src_bb);
                    dst_bb.extend(dst_terminator);
                }
            }
            MergeOrder::SrcFirst => {
                if src_has_no_results {
                    // `src_terminator->erase();` — dst keeps its own.
                    src_bb.pop();
                } else {
                    // `src_terminator->moveBefore(dst_terminator); dst_terminator->erase();` — src's
                    // terminator takes the place of dst's, which is the end of the arm.
                    let src_terminator = src_bb.pop();
                    dst_bb.pop();
                    dst_bb.extend(src_terminator);
                }
                // `op.moveBefore(&dst_bb, dst_bb.begin())` over src's ops from the BACK: they land at
                // the FRONT, in their original order.
                let mut spliced = std::mem::take(src_bb);
                spliced.append(dst_bb);
                *dst_bb = spliced;
            }
        }
    }

    // `getNewDbgNameFromList("CFGSM(", {src, dst})` — src first, dst second, and `None` unless BOTH
    // halves are named.
    let new_dbg_name = new_dbg_name_from_list(
        DbgNamePrefix::Cfgsm,
        dfir_op::dbg_name(src),
        &[dfir_op::dbg_name(dst)],
    );
    // `dataflow::setDbgNameAttr(dst, new_dbg_name_attr);`
    if let Some(new_dbg_name) = new_dbg_name {
        if let Some(slot) = dfir_op::dbg_name_mut(dst) {
            *slot = Some(new_dbg_name);
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 380/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHICH CONDITIONAL A MERGE PASS RESUMED FROM, BY ITS POSITION IN THE TREE'S PREORDER.
///
/// ⛔ THE REFERENCE PASSES AN `Operation *` BACK IN AND COMPARES IT BY ADDRESS (`:222`), which is a
/// name for a node that no `usize` gives. Nothing else about the previous merge is read, so what the
/// parameter carries is *"where in this traversal to start"* — and that is an index into the same
/// preorder the walk produces.
///
/// ⚠️ IT NAMES THE TREE AS IT STANDS AT RETURN. The reference's pointer survives its `recompute()`
/// (`:271`) because the merged conditional is the op that STAYS; an index does not survive ops being
/// deleted before it. No caller can observe that yet — the merge below is unported — and this becomes
/// a path or a handle in the changeset that applies one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CondNodeIndex(usize);

/// ONE TREE SIBLING GROUP — every conditional whose closest SELECTED ancestor is the one owning
/// `ops`, in program order.
///
/// ⛔ SIBLINGHOOD IS NOT "THE SAME STATEMENT LIST". `ConditionalTree::compute` parents each selected
/// op at `findClosestParent(op)` and skips everything unselected on the way up
/// (`dcc/src/Analysis/ConditionalTree.cpp:200-221`), so a conditional directly in a `then` region and
/// one nested inside an `affine.for` in that same region are SIBLINGS. They are in different blocks,
/// which is what `areShallowlyMergeable` rejects them for (`:346-349`) — the tree offers the pair and
/// the predicate refuses it, and a same-list walk would never have offered it.
fn sibling_group<'a>(ops: &'a [DfirOp], out: &mut Vec<&'a DfirOp>) {
    for op in ops {
        if ConditionalKind::of(op).is_some() {
            // A selected op ends the group here — its regions open groups of their own.
            out.push(op);
        } else {
            for region in dfir_op::regions(op) {
                sibling_group(region, out);
            }
        }
    }
}

/// THE TREE'S PREORDER OVER CANDIDATE NODES, each with the sibling group it sits in and its position
/// in it — `CondNode::walk<kPreOrder>(getRoot(), …)` (`:265`).
fn walk_candidates<'a>(ops: &'a [DfirOp], visit: &mut impl FnMut(&[&'a DfirOp], usize)) {
    let mut group: Vec<&DfirOp> = Vec::new();
    sibling_group(ops, &mut group);
    for at in 0..group.len() {
        visit(&group, at);
        // The node's own subtrees, `then` before `else`, before its next sibling.
        for region in dfir_op::regions(group[at]) {
            walk_candidates(region, visit);
        }
    }
}

/// Replaces: e380_shallowlyMergeConditionals
///
/// `CFGSDataflowConditionalTree::shallowlyMergeConditionals` (`:198`) — find the first conditional
/// that can be merged with a later sibling, merge them, delete what that left dead, recompute the
/// tree and return the surviving conditional so the next call resumes from it.
///
/// ⛔ ONE MERGE PER CALL, AND `cur_merged_if_op` IS BOTH THE ANSWER AND THE GUARD (`:217-218`):
/// *"once a merge has occurred the tree below is clobbered"*, so the walk runs to completion but
/// every later candidate is skipped. A loop that merged every candidate would rewrite through stale
/// nodes.
/// ⛔ `n->isLeaf()` IN THE FILTER IS DEAD (`:217`). `compute` gives every selected conditional a
/// `CondThenNode` and a `CondElseNode` child the moment it makes the node
/// (`dcc/src/Analysis/ConditionalTree.cpp:195-198`), so `getFirstChild()` is never null for an
/// if-node, and the root — the only childless node — is excluded by the test before it.
/// ⛔ `isThenNode() || isElseNode()` HAS NO COUNTERPART HERE: those two nodes carry no operation of
/// their own (they name a region of their parent), and only conditionals are nodes in this
/// representation. Dropping them from the walk is not dropping a filter.
/// ⛔ THE MERGE DIRECTION IS DECIDED BY WHICH ONE BINDS A RESULT (`:237`), never by which is first —
/// see [`MergeOrder`] — and `areShallowlyMergeable` guarantees at most one of the pair does
/// (`:347-348`). The sibling is then MOVED right after `n` (`:249`) and, for an `scf.if`, made to
/// share `n`'s condition with the old one erased if unused (`:252-256`).
/// ⚠️ `deleteAncestorsIfPossible` (`:269`), `recompute` (`:271`) and `OperationEquivalence::clearCache`
/// (`:272`) are all outside bridge 2's 384 — all three live in `dcc/src/Analysis/`
/// (`TransformationConditionalTree.cpp:155`, `ConditionalTree.hpp:151`, `OperationEquivalence.hpp:71`).
pub fn shallowly_merge_conditionals<A: Arch>(
    tree: &CfgsDataflowConditionalTree<'_, A>,
    prev_merged_if_op: Option<CondNodeIndex>,
) -> Option<CondNodeIndex> {
    // `:211` — the merged conditional, and the guard that stops at one merge.
    let cur_merged_if_op: Option<CondNodeIndex> = None;
    // `:215` — `bool start_analysis = (prev_merged_if_op == nullptr);`
    let mut start_analysis = prev_merged_if_op.is_none();
    let mut next = CondNodeIndex(0);

    walk_candidates(&tree.unit.body, &mut |siblings, at| {
        let n = next;
        next = CondNodeIndex(n.0 + 1);

        // `:217-219` — a candidate is a conditional node, and only until one merge has happened.
        if cur_merged_if_op.is_some() {
            return;
        }

        // `:222-225` — resume at the node the previous call returned; before it, look at nothing.
        if Some(n) == prev_merged_if_op {
            start_analysis = true;
        }
        if !start_analysis {
            return;
        }

        // `:346` — `if_op0->getBlock()`, which the tree's sibling groups cross (see [`sibling_group`]).
        let Some(block) = block_of(&tree.unit.body, siblings[at]) else {
            return;
        };
        // `:229-230` — `for (sibling = n->getNextSibling(); sibling; sibling = ...->getNextSibling())`.
        for sibling in &siblings[at + 1..] {
            // `:233` — the predicate that decides the pair.
            if !are_shallowly_mergeable(&tree.oe, siblings[at], sibling, block, &tree.unit.body) {
                continue;
            }
            // `:237-259` — the merge itself, then `:260` `break`.
            todo!(
                "e177_mergeShallow rewrites the unit in place and deleteAncestorsIfPossible, \
                 recompute and clearCache are unported, so the merge of candidate {:?} with \
                 sibling {:?} on {:?} cannot be applied",
                siblings[at],
                sibling,
                tree.unit.on.kind()
            );
        }
    });

    // `:270-273` — a merge invalidates the tree and the equivalence cache; `:274` returns the survivor.
    cur_merged_if_op
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 381/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e381_simplifyValueBasedConditionals
///
/// `CFGSDataflowConditionalTree::simplifyValueBasedConditionals` (`:466`) — for each top-level
/// conditional not yet rejected, ask whether it is a candidate; if it is, replace it by an iteration
/// argument and restart, and if it is not, mark it so it is not asked twice.
///
/// ⛔ ONE SIMPLIFICATION PER PASS, THEN `break` AND RESTART (`:481`): *"we have cloned a loop so the
/// IfOp pointers are clobbered"*. `recompute()` runs at the end of EVERY pass (`:489`), including the
/// one that changed nothing, which is why it sits outside the `for` rather than beside the `break`.
/// ⛔ THE MARKER IS CALL-LOCAL, AND THE REFERENCE'S OWN TAIL PROVES IT: `:491-495` removes
/// `PROCESSED_SIMPLIFICATIONS` from every child before returning. It is set on `:486` and read on
/// `:476` only, so it is a set of conditionals rejected DURING THIS CALL — a local set here, not a
/// new island attribute, and the removal pass is discharged by that set dying with the call.
/// ⛔ ONLY ROOT'S CHILDREN ARE CONSIDERED (`:473-474`) — a conditional nested inside ANOTHER
/// CONDITIONAL is not this step's. ⛔ BUT THAT IS NOT THE TOP-LEVEL STATEMENT LIST:
/// `findClosestParent` skips unselected ancestors (`dcc/src/Analysis/ConditionalTree.hpp:165-169`),
/// so an `scf.if` inside a top-level `affine.for` IS a root child — the group [`sibling_group`]
/// already walks, and entry 380 reads the same tree.
/// ⚠️ `parseConditional`, `replaceIfOpByIterArg` (entry 285), `recompute` and the manager's
/// CONSTRUCTOR (`.hpp:55-76`, which is what sets `is_candidate_`) are unported; only 285 is a unit.
pub fn simplify_value_based_conditionals<A: Arch>(tree: &CfgsDataflowConditionalTree<'_, A>) {
    // `:486` — the `PROCESSED_SIMPLIFICATIONS` attribute, by position among root's children.
    let processed: Vec<usize> = Vec::new();

    // `:471` — `do { … } while (tree_updated);`
    loop {
        // `:472`
        let tree_updated = false;

        // `:473-474` — root's children: every conditional whose closest SELECTED ancestor is none,
        // which is NOT every top-level statement that is one.
        let mut root_children: Vec<&DfirOp> = Vec::new();
        sibling_group(&tree.unit.body, &mut root_children);

        for (at, if_op) in root_children.iter().copied().enumerate() {
            // `:476` — `if (if_op->hasAttr(PROCESSED_SIMPLIFICATIONS)) continue;`
            if processed.contains(&at) {
                continue;
            }

            // `:477` — the manager, whose `val_array_` starts absent.
            let instance = ConditionalSimplificationManager::default();

            // `:478` — the question. Its `false` arm is `processed.push(at)` (`:485-486`) and its
            // `true` arm is entry 285 followed by `tree_updated = true; break;` (`:479-481`).
            todo!(
                "ConditionalSimplificationManager::parseConditional is unported, so \
                 e285_replaceIfOpByIterArg cannot run on the {:?} at index {} of {:?} ({:?})",
                ConditionalKind::of(if_op),
                at,
                tree.unit.on.kind(),
                instance
            );
        }

        // `:489` — recompute, unported: the tree here is derived from the unit on every question.

        // `:490`
        if !tree_updated {
            break;
        }
    }

    // `:491-495` — the marker's removal, which is this set going out of scope.
    drop(processed);
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 370/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHETHER `needle` IS `root` OR SITS ANYWHERE INSIDE IT — the reference's parent walk
/// `for (cur_op = use.getOwner(); cur_op != parent_op; cur_op = cur_op->getParentOp())`, read from the
/// other end. Two siblings share a parent, so "if_op1 is on the chain" is exactly "the use is in
/// if_op1's subtree, if_op1 itself included" — and `use.getOwner() == if_op1` is the case that makes
/// the `CmpIOp` exemption below necessary.
fn in_subtree(root: &DfirOp, needle: &DfirOp) -> bool {
    core::ptr::eq(root, needle)
        || dfir_op::regions(root)
            .into_iter()
            .flatten()
            .any(|op| in_subtree(op, needle))
}

/// Replaces: e370_areShallowlyMergeable
///
/// `CFGSDataflowConditionalTree::areShallowlyMergeable` (`:343`) — whether two conditionals of one
/// block share a top-level condition and nothing strictly between them blocks the splice.
///
/// ⛔⛔ `is_between_if_ops` IS SET AFTER THE `break`, SO A BACKWARDS PAIR IS VACUOUSLY MERGEABLE
/// (`:357-359`, `:376`): if `if_op1` stands BEFORE `if_op0` the walk breaks at `if_op1` while the flag
/// is still false, and the answer is `true` with nothing checked. Entry 380 only ever offers a LATER
/// sibling, so no caller in bridge 2 reaches it.
/// ⛔ THE EXEMPTION READS OPERAND 0 BEFORE ASKING WHETHER `if_op1` IS AN `scf.if` (`:369-370`), so an
/// operandless `affine.if` indexes out of range there; `operands(..).first()` is `None`, which makes
/// the left disjunct true — the same answer the `!isa<scf::IfOp>` on its right already gives.
/// ⚠️ `DT_CHECK_MSG(if_op0 != if_op1, ..)` is `false`: a conditional cannot be merged with itself.
#[must_use]
pub fn are_shallowly_mergeable(
    oe: &OperationEquivalence,
    if_op0: &DfirOp,
    if_op1: &DfirOp,
    block: &[DfirOp],
    scope: &[DfirOp],
) -> bool {
    // `:345` — `DT_CHECK_MSG(if_op0 != if_op1, "Cannot merge a conditional with itself.")`.
    if core::ptr::eq(if_op0, if_op1) {
        return false;
    }
    // `:346-349` — `bb != if_op1->getBlock()`, and at most one of the pair may bind a result. A block
    // is a statement list here, so "the same block" is "both are elements of `block`".
    let in_block = |needle: &DfirOp| block.iter().any(|op| core::ptr::eq(op, needle));
    if !in_block(if_op0)
        || !in_block(if_op1)
        || (!dfir_op::results(if_op0).is_empty() && !dfir_op::results(if_op1).is_empty())
    {
        return false;
    }

    // `:351` — `if (!topLevelConditionsMatch(oe, if_op0, if_op1)) return false;`
    if !top_level_conditions_match(oe, if_op0, if_op1, scope) {
        return false;
    }

    // `:355-356` — *"Check that the ops strictly between if_op0, if_op1 have no side-effects and no
    // uses inside of if_op1."*
    let mut is_between_if_ops = false;
    // `:369` — `if_op1->getOperand(0).getDefiningOp()`, hoisted out of the loop because it does not
    // depend on `op`. See this function's second trap for the out-of-range read.
    let cond_def = dfir_op::operands(if_op1)
        .first()
        .and_then(|cond| dfir_op::defining_op(*cond, block));
    for op in block {
        // `:358` — ⛔ BEFORE the flag is consulted, and before it is ever set.
        if core::ptr::eq(op, if_op1) {
            break;
        }
        if is_between_if_ops {
            // `:360`
            if op_has_side_effect(op) {
                return false;
            }
            // `:368-370` — the condition `if_op1` branches on is allowed to be one of these ops,
            // because `if_op0` has an equivalent condition standing before it.
            let is_exempt_condition = matches!(cond_def, Some(def) if core::ptr::eq(def, op))
                && matches!(ConditionalKind::of(if_op1), Some(ConditionalKind::Scf));
            if !is_exempt_condition {
                // `:371-375` — every use of `op`, up its owner's parent chain to `op`'s own parent.
                for result in dfir_op::results(op) {
                    if dfir_op::uses(result, block)
                        .into_iter()
                        .any(|user| in_subtree(if_op1, user))
                    {
                        return false;
                    }
                }
            }
        }
        // `:376`
        if core::ptr::eq(op, if_op0) {
            is_between_if_ops = true;
        }
    }
    // `:378`
    true
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 371/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// THE STATEMENT LIST THAT DIRECTLY HOLDS `needle` — `Operation::getBlock()`.
fn block_of<'a>(ops: &'a [DfirOp], needle: &DfirOp) -> Option<&'a [DfirOp]> {
    if ops.iter().any(|op| core::ptr::eq(op, needle)) {
        return Some(ops);
    }
    ops.iter()
        .flat_map(dfir_op::regions)
        .find_map(|region| block_of(region, needle))
}

/// THE OP WHOSE REGION DIRECTLY HOLDS `needle` — `Operation::getParentOp()`, and [`None`] for a
/// statement of the unit's own body.
fn enclosing_op<'a>(ops: &'a [DfirOp], needle: &DfirOp) -> Option<&'a DfirOp> {
    for op in ops {
        for region in dfir_op::regions(op) {
            if region.iter().any(|inner| core::ptr::eq(inner, needle)) {
                return Some(op);
            }
            if let Some(found) = enclosing_op(region, needle) {
                return Some(found);
            }
        }
    }
    None
}

/// Replaces: e371_hoistLoopInvariantConditionals
///
/// `CFGSDataflowConditionalTree::hoistLoopInvariantConditionals` (`:750`) — hoist every simple,
/// result-yielding, loop-invariant conditional whose one result only feeds a
/// `dataflow.get_logical_memory_view` or a `dataflow.send`/`receive` out of its enclosing loop.
///
/// ⛔⛔ THE LAMBDA CAN NEVER ANSWER NON-NULL: every exit from `while (1)` (`:777-789`) is
/// `return nullptr`, so the reverse-BFS walk's early-stop is unreachable and the hoist loop ends only
/// when the parent stops being a loop or invariance fails. That is why this is a `for` over every
/// candidate rather than a walk that stops at the first hit.
/// ⛔ `n->isLeaf()` (`:762`) IS DEAD, for the reason entry 380 records; `isThenNode()`/`isElseNode()`
/// have no counterpart in a representation where only conditionals are nodes.
/// ⚠️ AN `affine.if` NODE IS NOT HOISTED: `isLoopInvariant`'s `DT_CHECK_MSG(scf_if, ..)` (`:686`)
/// aborts on one, and entry 349 answers "not invariant" there.
/// ⚠️ `moveAncestorsToMaintainDominance` (`:787`) is `dcc/src/Analysis/`
/// (`TransformationConditionalTree.cpp:141`) — outside bridge 2's 384, as at entries 284 and 380.
pub fn hoist_loop_invariant_conditionals<A: Arch>(tree: &CfgsDataflowConditionalTree<'_, A>) {
    let body = &tree.unit.body;
    for n in reverse_bfs_candidates(body) {
        // `:766-768` — `DT_CHECK_MSG(n_if_op, ..)` is discharged by the node BEING the op here, and
        // `getNumResults() != 1` is the "result-yielding" half of the pattern.
        let results = dfir_op::results(n);
        let [result] = results.as_slice() else {
            continue;
        };
        let result = *result;

        // `:770-775` — *"Restrict hoisting to IfOps only used in get_logical_mem_view or
        // dataflow.send/receive."* ⭐ A CONDITIONAL WITH NO USERS PASSES, which is the `for`'s own
        // answer over an empty list.
        if !dfir_op::uses(result, body).into_iter().all(|user| {
            matches!(
                user,
                DfirOp::Dataflow(
                    dataflow::Op::GetLogicalMemoryView { .. }
                        | dataflow::Op::Send { .. }
                        | dataflow::Op::Receive { .. }
                )
            )
        }) {
            continue;
        }

        // `:779-789` — `while (1)`, whose first two exits are the loop's own termination conditions.
        let Some(parent_for_op) = enclosing_op(body, n) else {
            continue;
        };
        if !matches!(
            parent_for_op,
            DfirOp::Scf(scf::Op::For { .. }) | DfirOp::Affine(affine::Op::For { .. })
        ) {
            continue;
        }
        // `:784` — and the `affine.if` case, which entry 349 answers `false` for.
        let Some(scf_if) = ScfConditional::of(n) else {
            continue;
        };
        if !is_loop_invariant(scf_if, parent_for_op, body) {
            continue;
        }

        // `:787` — *"Move n_if_op and its ancestors which do not dominate parent_for_op to right
        // before parent_for_op."*
        todo!(
            "moveAncestorsToMaintainDominance is unported, so e371_hoistLoopInvariantConditionals \
             cannot hoist the loop-invariant conditional {:?} out of {:?} on {:?}",
            n,
            parent_for_op,
            tree.unit.on.kind()
        );
    }
}

// ⛔ RE-CREATED ANCHORS. These units' `crustify:todo:` markers were deleted without a
// `/// Replaces:` ever appearing, which removed them from every later schedule and let the
// driver report the campaign DONE. Outstanding work is now computed from UNITS.tsv.

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 244/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e244_isHoistable
///
/// **244/384** `CFGSDataflowConditionalTree::isHoistable` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:82` (9L).
///
/// ⛔ IT IS THE WHOLE PREFIX, NOT JUST THE OP: an op is hoistable only when nothing before it in the
/// block would have to move past — see [`op_has_side_effect`]'s three callers.
/// ⛔ THE FIRST CLAUSE IS A DIFFERENT QUESTION FROM THE SECOND. `to_hoist->getBlock() !=
/// then_or_else_block` asks whether the op is in THIS block *"as opposed to a for-loop's block"*
/// (`:151`); here that is the position search, and an op from elsewhere is not hoistable.
#[must_use]
pub fn is_hoistable(to_hoist: &DfirOp, then_or_else_block: &[DfirOp]) -> bool {
    // `to_hoist->getBlock() != then_or_else_block || opHasSideEffect(*to_hoist)`, in that order.
    let Some(at) = then_or_else_block
        .iter()
        .position(|op| core::ptr::eq(op, to_hoist))
    else {
        return false;
    };
    if op_has_side_effect(to_hoist) {
        return false;
    }

    // `for (auto &op : ...) { if (&op == to_hoist) break; if (opHasSideEffect(op)) return false; }`
    then_or_else_block[..at]
        .iter()
        .all(|op| !op_has_side_effect(op))
}

#[cfg(test)]
mod is_hoistable_tests {
    use super::*;
    use crate::generated::SyncSignal;
    use crate::islands::dataflow_ir::dialects::{Val, dataflow};

    /// The vendor's own contrast, from `merging-shallow-skip.mlir`: a `dataflow.receive` standing
    /// between the block's start and the candidate is what stops the movement (`:55-59`), while a
    /// block of `arith.constant`s does not (`merging.mlir`).
    #[test]
    fn a_listed_prefix_hoists_and_an_unlisted_op_before_it_does_not() {
        let candidate = DfirOp::Arith(arith::Op::Constant {
            result: Val(2),
            value: 7,
        });
        let harmless = DfirOp::Arith(arith::Op::Constant {
            result: Val(0),
            value: 1,
        });
        let effectful = DfirOp::Dataflow(dataflow::Op::SyncRecv {
            from: Val(1),
            signal: SyncSignal::InputToLxsuToLxluToSync,
        });

        let clean = vec![harmless.clone(), candidate.clone()];
        assert!(is_hoistable(&clean[1], &clean));

        let blocked = vec![effectful, candidate.clone()];
        assert!(!is_hoistable(&blocked[1], &blocked));

        // `to_hoist->getBlock() != then_or_else_block` — the same op value, another block.
        assert!(!is_hoistable(&candidate, &clean));
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 284/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// THE CONDITIONAL TREE'S REVERSE-BFS OVER CANDIDATE NODES — `CondNode::walk<kReverseBFS>(getRoot(),
/// hoistCommon)` (`:184`).
///
/// ⛔ DEEPEST LEVEL FIRST, RIGHT-TO-LEFT WITHIN A LEVEL. `reverseBreadthFirstWalk` with
/// `keep_order = false` pushes a BFS onto a stack and runs the action over that stack in reverse
/// (`dcc/src/Analysis/OperationTree.cpp:200-237`), which is what lets one hoist expose a hoist out of
/// the parent's parent (`:105-107`).
/// ⛔ THE THEN- AND ELSE-NODES ARE LEVELS OF THEIR OWN, so a conditional's children sit TWO levels
/// below it and one level holds `n1`'s then-group, `n1`'s else-group, `n2`'s two, in that order
/// (`dcc/src/Analysis/ConditionalTree.cpp:195-198`). The walk rejects both node kinds and the root
/// (`:120`), so dropping them from the level list keeps exactly the order they impose.
fn reverse_bfs_candidates(body: &[DfirOp]) -> Vec<&DfirOp> {
    let mut levels: Vec<Vec<&DfirOp>> = Vec::new();
    // The root's own children — [`sibling_group`] is `findClosestParent`, so a conditional under an
    // unselected loop is one of them.
    let mut level: Vec<&DfirOp> = Vec::new();
    sibling_group(body, &mut level);

    while !level.is_empty() {
        let mut next: Vec<&DfirOp> = Vec::new();
        for n in &level {
            // `then` before `else`, the order `getRegions()` indexes them in.
            for region in dfir_op::regions(n) {
                sibling_group(region, &mut next);
            }
        }
        levels.push(level);
        level = next;
    }

    levels
        .iter()
        .rev()
        .flat_map(|level| level.iter().rev().copied())
        .collect()
}

/// Replaces: e284_hoistCommonConditionals
///
/// `CFGSDataflowConditionalTree::hoistCommonConditionals` (`:94`) — find one conditional that occurs
/// equivalently in BOTH arms of a parent conditional, hoist the then-copy out in front of the parent,
/// point the else-copy's users at it, and return the parent so the next call resumes there.
///
/// ⛔ ONE HOIST PER CALL, AND `cur_parent_if_op_of_hoist` IS BOTH THE ANSWER AND THE GUARD — but it is
/// tested only on NODE ENTRY (`:120`), so the `child_a` x `child_b` search that set it runs to
/// completion and may hoist again out of the SAME node (`:107-109`, *"finish analyzing the current
/// node"*).
/// ⛔ `nodes_to_skip` IS DECLARED INSIDE THE `child_a` LOOP (`:139`): an else-child skipped for one
/// then-child is offered again to the next, so the same op can be hoisted and queued for deletion
/// twice. ⛔ `n->isLeaf()`, `isThenNode()` and `isElseNode()` are dead or absent here for the reasons
/// entry 380 records; BOTH ARMS HOLDING A CONDITIONAL (`:125`) is the live part of that filter.
/// ⭐ THE NODE IS NAMED BY THE BORROW, compared with `core::ptr::eq` — the reference compares
/// `Operation *` by address (`:129`), and entry 244 already names a position that way.
/// ⚠️ `moveAncestorsToMaintainDominance` (`:163`), `deleteAncestorsIfPossible` (`:187`), `recompute`
/// and `clearCache` (`:188-190`) all live in `dcc/src/Analysis/` — outside bridge 2's 384 — so this
/// stops at the first, as entries 380 and 381 stop at theirs.
#[must_use]
pub fn hoist_common_conditionals<'u, A: Arch>(
    tree: &CfgsDataflowConditionalTree<'u, A>,
    if_op_where_last_hoist_occurred: Option<&DfirOp>,
) -> Option<&'u DfirOp> {
    // `:96` — what a hoist leaves for the deletion pass, and `:115` the guard that is also the answer.
    let ops_to_delete: Vec<&'u DfirOp> = Vec::new();
    let cur_parent_if_op_of_hoist: Option<&'u DfirOp> = None;
    // `:119` — `bool start_analysis = (if_op_where_last_hoist_occurred == nullptr);`
    let mut start_analysis = if_op_where_last_hoist_occurred.is_none();

    for n in reverse_bfs_candidates(&tree.unit.body) {
        // `:120-122` — every LATER node is skipped once a hoist has happened.
        if cur_parent_if_op_of_hoist.is_some() {
            continue;
        }

        // `:123-125` — the then- and else-nodes, each of which must hold a conditional of its own.
        let regions = dfir_op::regions(n);
        let [then_block, else_block] = regions.as_slice() else {
            continue;
        };
        let mut then_children: Vec<&DfirOp> = Vec::new();
        sibling_group(then_block, &mut then_children);
        let mut else_children: Vec<&DfirOp> = Vec::new();
        sibling_group(else_block, &mut else_children);
        if then_children.is_empty() || else_children.is_empty() {
            continue;
        }

        // `:127-134` — `DT_CHECK_MSG(n_if_op, ..)` is discharged by the node BEING the op here; then
        // resume at the node the previous call returned, and look at nothing before it.
        if if_op_where_last_hoist_occurred.is_some_and(|prev| core::ptr::eq(prev, n)) {
            start_analysis = true;
        }
        if !start_analysis {
            continue;
        }

        // `:138-142` — every (then-child, else-child) pair, in order.
        for child_a in &then_children {
            // `:139` — ⛔ HERE, so it is empty again for every `child_a`.
            let nodes_to_skip: Vec<&DfirOp> = Vec::new();
            for child_b in &else_children {
                // `:143`
                if nodes_to_skip
                    .iter()
                    .any(|skip| core::ptr::eq(*skip, *child_b))
                {
                    continue;
                }

                // `:159-161` — the legality triple. `op_a && op_b` is vacuous in this representation,
                // and the two blocks are `n`'s own then/else regions, not a nested loop's.
                if ops_are_equivalent(
                    child_a,
                    child_b,
                    &tree.unit.body,
                    tree.oe.preference,
                    tree.oe.block_args,
                ) && is_hoistable(child_a, then_block)
                    && is_hoistable(child_b, else_block)
                {
                    // `:163` — where the hoist begins, and with it `:166-176`.
                    todo!(
                        "moveAncestorsToMaintainDominance is unported \
                         (dcc/src/Analysis/TransformationConditionalTree.cpp:141, outside bridge 2), \
                         so the {:?} common to both arms of the {:?} on {:?} cannot be moved out; \
                         {} op(s) already queued for deletion under {:?}",
                        ConditionalKind::of(child_a),
                        ConditionalKind::of(n),
                        tree.unit.on.kind(),
                        ops_to_delete.len(),
                        tree.oe
                    );
                }
            }
        }
    }

    // `:187` — `for (auto *op : ops_to_delete) deleteAncestorsIfPossible(op);`, unported with the hoist
    // that fills the list; `:188-191` recomputes the tree and clears the cache and returns the parent.
    drop(ops_to_delete);
    cur_parent_if_op_of_hoist
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 285/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// THE ARITHMETIC SEQUENCE A VALUE-BASED CONDITIONAL YIELDS — `seq_lb_` and `seq_step_`
/// (`CFGSDataflowConditionalTree.hpp:49-52`), which `parseConditional` derives from `val_array_`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct YieldedSequence {
    /// `seq_lb_` — what the loop's first iteration yields.
    pub lb: YieldedIndex,
    /// `seq_step_` — the difference between what consecutive iterations yield.
    pub step: i64,
}

/// A CONDITIONAL THAT BINDS A VALUE.
///
/// ⛔ THE MANAGER'S CONSTRUCTOR REJECTS ONE THAT BINDS NONE (`.hpp:59-62`), which is what makes
/// `if_op_->getResult(0)` (`:628`, `:668`) a read rather than a dereference of nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueYieldingConditional<'a> {
    /// `if_op_` — `top_node_->getOperation()`.
    pub op: &'a DfirOp,
    /// `if_op_->getResult(0)`, the value every use of the conditional reads.
    pub yielded: Val,
}

impl<'a> ValueYieldingConditional<'a> {
    /// The conditional, or [`None`] where it is not one or binds nothing.
    #[must_use]
    pub fn of(op: &'a DfirOp) -> Option<ValueYieldingConditional<'a>> {
        ConditionalKind::of(op)?;
        let yielded = *dfir_op::results(op).first()?;
        Some(ValueYieldingConditional { op, yielded })
    }
}

/// WHAT ENTRY 285 PUTS IN PLACE OF THE CONDITIONAL — its two arms are two different rewrites.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IterArgReplacement {
    /// `num_iterations == 1` (`:622-630`): one `arith.constant seq_lb` stands where the conditional
    /// stood.
    Constant {
        /// The constant, built at `OpBuilder(if_op_)` — immediately before the conditional it replaces.
        op: DfirOp,
        /// `if_op_->getResult(0).replaceAllUsesWith(replacement)` (`:628`), old then new.
        replacement: (Val, Val),
    },
    /// Otherwise (`:631-673`): the loop one iteration argument wider, the conditional gone from its
    /// body and its uses reading that argument.
    IterArg {
        /// `start_val`, `step_val` and entry 264's own fill, in creation order — they go where the
        /// old loop stood (`:635-639`, `Utils.cpp:41-45`).
        before: Vec<DfirOp>,
        /// The new loop, which replaces the old one.
        op: DfirOp,
        /// `loop_op->getResult(i).replaceAllUsesWith(ret_op->getResult(i))` (`Utils.cpp:82-83`) — the
        /// OLD loop's results, whose users are outside the body this rewrote.
        replacements: Vec<(Val, Val)>,
    },
}

/// `Value::replaceAllUsesWith` FOLLOWED BY `Operation::erase`, over a region tree — what `:667-669`
/// does to the marked conditional once the walk has found it.
fn erase_and_forward(scope: &mut Vec<DfirOp>, from: Val, to: Val) {
    scope.retain(|op| !dfir_op::results(op).contains(&from));
    for op in scope.iter_mut() {
        dfir_op::replace_uses_of_with(op, from, to);
        for region in dfir_op::regions_mut(op) {
            erase_and_forward(region, from, to);
        }
    }
}

/// Replaces: e285_replaceIfOpByIterArg
///
/// `ConditionalSimplificationManager::replaceIfOpByIterArg` (`:619`) — replace a conditional that
/// yields a fixed-stride sequence of its loop's induction variable by a new iteration argument of that
/// loop carrying the same sequence.
///
/// ⛔ A ONE-TRIP LOOP TAKES NEITHER (`:622-630`): the sequence is one value, so an `arith.constant
/// seq_lb_` replaces the conditional and the loop is left alone.
/// ⛔⛔ THE REFERENCE MISWIRES THE `scf.for` ARM, AND ITS OWN FROZEN EXPECTATION SHOWS IT.
/// `new_for_op->setOperand(num_iter_args - 1, start_val)` (`:659`) indexes the RAW operand list: an
/// `affine.for` with constant bounds carries its inits first, so operand 0 IS the added init — but
/// `scf.for`'s operands are `lb, ub, step, inits..`, so operand 0 is its LOWER BOUND.
/// `simplify-conditional.mlir:73` prints `scf.for %51 = %47 to %11 step %6 iter_args(%52 = %49)`:
/// `%47` is `start_val` landed in the lower bound while the iteration argument still reads `%49`,
/// entry 264's scf fill of **1**, so the sequence comes out `1, 1025, 2049, 3073` where `seq_lb_ = 0`.
/// This port initialises the added argument with `seq_lb_` in both dialects and leaves the bounds
/// alone; the affine arm stays byte-identical to `:52-53`.
/// ⭐ THE MARKER ATTRIBUTE IS THE `ir_map`. `IF_OP_TO_BE_REPLACED_BY_ITER_ARG` (`:26`, `:633`, `:667`)
/// exists only to find `if_op_`'s CLONE inside the new loop; entry 264's mapping names that
/// correspondence directly, so `delete_op` is passed `false` to read it and no island attribute is
/// invented. ⭐ `for_op` AND `iterations` ARE `std::get<0>` AND `std::get<4>` OF `for_op_tuple_`
/// (`:620-621`), split because entry 264 takes a [`CountedLoop`] — which is also the `dyn_cast` pair
/// at `:645-655` and makes `DT_ERROR("no matching for operation")` (`:656`) unwritable.
#[must_use]
pub fn replace_if_op_by_iter_arg(
    if_op: ValueYieldingConditional<'_>,
    for_op: &CountedLoop<'_>,
    iterations: Iterations,
    sequence: YieldedSequence,
    vals: &mut Values,
) -> IterArgReplacement {
    // `:622` — `if (num_iterations == 1)`.
    if iterations.get() == 1 {
        let result = vals.mint();
        return IterArgReplacement::Constant {
            // `:625-626` — `arith::ConstantIndexOp::create(builder, if_op_->getLoc(), seq_lb_)`.
            op: DfirOp::Arith(arith::Op::Constant {
                result,
                value: sequence.lb.0,
            }),
            // `:628-629` — the uses, then `if_op_->erase()`.
            replacement: (if_op.yielded, result),
        };
    }

    // `:635-639` — both constants at `OpBuilder(for_op)`, i.e. immediately before the loop. `:632-634`
    // marks the conditional, which is the `ir_map` below.
    let start_val = vals.mint();
    let step_val = vals.mint();
    let mut before = vec![
        DfirOp::Arith(arith::Op::Constant {
            result: start_val,
            value: sequence.lb.0,
        }),
        DfirOp::Arith(arith::Op::Constant {
            result: step_val,
            value: sequence.step,
        }),
    ];

    // `:641-643` — one more return value, and `delete_op` false so the mapping comes back.
    let widened = create_for_op_with_additional_return_value(vals, for_op, 1, false);
    before.extend(widened.consts);
    let mut op = widened.op;
    // `IRMapping ir_map;` — [`Some`] because `delete_op` was false; the empty default is unreachable.
    let ir_map = widened.ir_map.unwrap_or_default();

    match &mut op {
        DfirOp::Affine(affine::Op::For { carried, body, .. })
        | DfirOp::Scf(scf::Op::For { carried, body, .. }) => {
            // `:659` — the added init. ⛔ THE REFERENCE WRITES OPERAND `num_iter_args - 1`; see the
            // anchor for why that is `scf.for`'s lower bound and this is the argument it meant.
            if let Some(added) = carried.last_mut() {
                added.init = start_val;
            }
            // `:648`, `:653` — `getRegionIterArgs()[num_iter_args - 1]`; entry 264 appended exactly one.
            let replacement = carried.last().map_or(start_val, |added| added.arg);

            // `:666-673` — the walk that finds the marked conditional in the new body, re-points its
            // uses at the iteration argument and erases it.
            if let Some(cloned) = ir_map.lookup(if_op.yielded) {
                erase_and_forward(body, cloned, replacement);
            }

            // `:661-663` — `AddIOp(replacement, step_val)` immediately before the terminator, whose
            // last operand — `num_iter_args - 1` of a yield, which carries exactly those — becomes it.
            let sum = vals.mint();
            let at = body.len().saturating_sub(1);
            body.insert(
                at,
                DfirOp::Arith(arith::Op::AddI(arith::IntBinary {
                    result: sum,
                    lhs: replacement,
                    rhs: step_val,
                    ty: ScalarTy::Index,
                })),
            );
            if let Some(
                DfirOp::Affine(affine::Op::Yield { operands })
                | DfirOp::Scf(scf::Op::Yield { operands }),
            ) = body.last_mut()
                && let Some(last) = operands.last_mut()
            {
                *last = sum;
            }
        }
        // ⭐ UNREACHABLE BY CONSTRUCTION: entry 264 emits one of those two `for`s and nothing else.
        DfirOp::Affine(_)
        | DfirOp::Scf(_)
        | DfirOp::Arith(_)
        | DfirOp::Dataflow(_)
        | DfirOp::Agen(_)
        | DfirOp::Vector(_)
        | DfirOp::VectorChain(_)
        | DfirOp::Symbol(_)
        | DfirOp::Uniform(_) => {}
    }

    IterArgReplacement::IterArg {
        before,
        op,
        replacements: widened.replacements,
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 347/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e347_topLevelConditionsMatch
///
/// `topLevelConditionsMatch` (`:279`) — whether two conditionals branch on the same test: an
/// `affine.if` pair must share an integer set AND have equivalent operands, an `scf.if` pair
/// equivalent CONDITIONS, and a mixed pair never matches (`:308-309`, the reference's own `else`).
///
/// ⛔ THE AFFINE ARM COMPARES EVERY OPERAND, THE SCF ARM ONLY OPERAND 0 (`:287`, `:304-305`) —
/// `affine.if`'s operands are the set's dims and its symbols, and an `scf.if` has none but the `i1`.
/// ⛔ A BLOCK-ARG PAIR IS `owner0 == owner1 && idx0 == idx1` (`:295`), WHICH HERE IS `Val` EQUALITY:
/// two distinct [`Val`]s are never one argument of one region — see
/// [`BlockArgEquivalence::SameOwnerAndIndex`].
/// ⛔ ONE SIDE A BLOCK ARG AND THE OTHER A RESULT IS A MISMATCH BEFORE ANY COMPARISON (`:296-297`),
/// which is also what keeps the `getDefiningOp()` on the line after it off a null.
#[must_use]
pub fn top_level_conditions_match(
    oe: &OperationEquivalence,
    if_op0: &DfirOp,
    if_op1: &DfirOp,
    scope: &[DfirOp],
) -> bool {
    match (if_op0, if_op1) {
        // `:281-301` — both `affine.if`.
        (
            DfirOp::Affine(affine::Op::If { set: set0, .. }),
            DfirOp::Affine(affine::Op::If { set: set1, .. }),
        ) => {
            // `:285-286` — `getIntegerSet() != getIntegerSet()`.
            if set0 != set1 {
                return false;
            }
            // `:287-300` — `llvm::zip(if_op0->getOperands(), if_op1->getOperands())`.
            for (read0, read1) in dfir_op::operands(if_op0)
                .into_iter()
                .zip(dfir_op::operands(if_op1))
            {
                match (
                    dfir_op::defining_op(read0, scope),
                    dfir_op::defining_op(read1, scope),
                ) {
                    // `:293-299` — both block arguments.
                    (None, None) => {
                        if read0 != read1 {
                            return false;
                        }
                    }
                    // `:300-302` — neither is, so both have a definition to compare.
                    (Some(def0), Some(def1)) => {
                        if !ops_are_equivalent(def0, def1, scope, oe.preference, oe.block_args) {
                            return false;
                        }
                    }
                    // `:300-301` — `is_if0_operand_block_arg || is_if1_operand_block_arg`.
                    (Some(_), None) | (None, Some(_)) => return false,
                }
            }
            true
        }
        // `:303-307` — both `scf.if`: the definitions of the two conditions, and nothing else.
        (
            DfirOp::Scf(scf::Op::If { cond: cond0, .. }),
            DfirOp::Scf(scf::Op::If { cond: cond1, .. }),
        ) => match (
            dfir_op::defining_op(*cond0, scope),
            dfir_op::defining_op(*cond1, scope),
        ) {
            (Some(def0), Some(def1)) => {
                ops_are_equivalent(def0, def1, scope, oe.preference, oe.block_args)
            }
            // ⭐ THE REFERENCE DEREFERENCES A NULL HERE: a condition that is a region argument has no
            // defining op, and `*if_op0->getOperand(0).getDefiningOp()` (`:304`) is undefined for it.
            _ => false,
        },
        // `:308-309` — a mixed pair, or an operation that is not a conditional at all.
        (_, _) => false,
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 348/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e348_singleOpBranchToYieldVal
///
/// `ConditionalSimplificationManager::singleOpBranchToYieldVal` (`:498`) — the constant index one arm
/// of an `scf.if` yields, where that arm holds nothing but its terminator.
///
/// ⛔ `scf.if` ONLY (`:501-502`): an `affine.if` is declined, so [`ConditionalKind`] is not the test.
/// ⛔ `size() != 1` COUNTS THE TERMINATOR (`:505`) — the one operation IS the `scf.yield`, so the arm
/// computes nothing and the value it yields is defined OUTSIDE it.
/// ⛔ AN `else` REGION WITH NO BLOCK ANSWERS `None`, where the reference's `.front()` on it is
/// undefined — see [`crate::islands::dataflow_ir::dialects::scf::Op::If::else_body`].
/// ⛔ A YIELDED REGION ARGUMENT IS DECLINED, NOT UNKNOWN (`:509-510`, `if (!yielded_val_op)`).
#[must_use]
pub fn single_op_branch_to_yield_val(
    op: Option<&DfirOp>,
    branch: Arm,
    scope: &[DfirOp],
) -> Option<YieldedIndex> {
    // `:500-502` — `if (!op) return false;` and then the `dyn_cast<scf::IfOp>`.
    let DfirOp::Scf(scf::Op::If {
        body, else_body, ..
    }) = op?
    else {
        return None;
    };
    // `:503-505` — the chosen arm, and the single-operation test that `getTerminator()` then relies on.
    let arm = match branch {
        Arm::Then => body,
        Arm::Else => else_body,
    };
    let [terminator] = arm.as_slice() else {
        return None;
    };
    // `:506-507` — `terminator->getOperand(0).getDefiningOp()`.
    let yielded = *dfir_op::operands(terminator).first()?;
    // `:509-515` — an `arith.constant` of index type, or nothing.
    let DfirOp::Arith(arith::Op::Constant { value, .. }) = dfir_op::defining_op(yielded, scope)?
    else {
        return None;
    };
    Some(YieldedIndex(*value))
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 349/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// AN `scf.if`, THE ONLY CONDITIONAL LOOP-INVARIANT HOISTING CONSIDERS — `DT_CHECK_MSG(scf_if,
/// "Expect valid scf::IfOp")` (`:685`) as a minting rule, so an `affine.if` is not an argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScfConditional<'a> {
    /// The operation, whose two arms are `getRegions()[0]` and `getRegions()[1]`.
    pub op: &'a DfirOp,
    /// `scf_if.getCondition()` — the `i1` it branches on, which is also its only operand.
    pub condition: Val,
}

impl<'a> ScfConditional<'a> {
    /// The conditional, or [`None`] for anything that is not an `scf.if`.
    #[must_use]
    pub fn of(op: &'a DfirOp) -> Option<ScfConditional<'a>> {
        let DfirOp::Scf(scf::Op::If { cond, .. }) = op else {
            return None;
        };
        Some(ScfConditional {
            op,
            condition: *cond,
        })
    }
}

/// Replaces: e349_isLoopInvariant
///
/// `CFGSDataflowConditionalTree::isLoopInvariant` (`:681`) — whether a conditional's condition and
/// everything its arms yield can be computed outside `parent_for_op`.
///
/// ⛔ AN ARM IS EITHER A LEAF OR EXACTLY ONE NESTED CONDITIONAL (`:706-708`, `:724-726`): a leaf must
/// hold its terminator alone and contributes the value it yields to the worklist, and a non-leaf
/// recurses AND must hold only `arith.cmpi`, `scf.if`, an index `arith.constant` and `scf.yield`.
/// ⛔ THE DEPENDENCE IS ON THE **OWNER**, NOT THE VALUE (`:736-738`): a region argument fails only
/// when the region belongs to `parent_for_op` — an argument of an inner loop is invariant here.
/// ⛔ THE WORKLIST WALKS OPERANDS TRANSITIVELY (`:741-743`), so a constant folded through three
/// `arith.addi`s outside the loop still answers `true`.
/// ⚠️ `parent_for_op` MUST POINT INTO `scope`: the owner comparison is the reference's pointer
/// equality, which is what tells one loop's argument from another's.
#[must_use]
pub fn is_loop_invariant(n: ScfConditional<'_>, parent_for_op: &DfirOp, scope: &[DfirOp]) -> bool {
    // `:690` — "Worklist of values needed to be checked for dependencies on loop iterator arguments."
    let mut worklist: Vec<Val> = Vec::new();

    // `:692-729` — the `then` node then the `else` node, the same seven lines twice.
    for arm in [Arm::Then, Arm::Else] {
        let body = dfir_op::regions(n.op)[arm.index()];
        // `n->getThenNode()->isLeaf()` — a tree node's children are the conditionals whose closest
        // SELECTED ancestor it is, which is what [`sibling_group`] walks.
        let mut children: Vec<&DfirOp> = Vec::new();
        sibling_group(body, &mut children);

        if children.is_empty() {
            // `:694-700` — `if (bb.getOperations().size() != 1) return false;` and then the value the
            // terminator yields.
            let [terminator] = body else {
                return false;
            };
            let Some(yielded) = dfir_op::operands(terminator).first().copied() else {
                return false;
            };
            worklist.push(yielded);
        } else {
            // `:703-708` — one child, and it must itself be loop invariant.
            if children.len() != 1 {
                return false;
            }
            // ⭐ AN `affine.if` CHILD ABORTS THE REFERENCE at the recursion's own `DT_CHECK`; the
            // simplicity scan below would reject it two lines later, so `false` is that answer.
            let Some(child) = ScfConditional::of(children[0]) else {
                return false;
            };
            if !is_loop_invariant(child, parent_for_op, scope) {
                return false;
            }
            // `:710-714` — "The then-branch body must be simple (only contain certain op types)."
            for op in body {
                if !matches!(
                    op,
                    DfirOp::Arith(arith::Op::Compare { .. } | arith::Op::Constant { .. })
                        | DfirOp::Scf(scf::Op::If { .. } | scf::Op::Yield { .. })
                ) {
                    return false;
                }
            }
        }
    }

    // `:731-732` — "Check the condition for dependencies."
    worklist.push(n.condition);

    // `:734-745`
    while let Some(val) = worklist.pop() {
        match dfir_op::defining_op(val, scope) {
            // `:736-739` — a region argument: whose region is it?
            None => {
                if dfir_op::region_owner(val, scope)
                    .is_some_and(|owner| core::ptr::eq(owner, parent_for_op))
                {
                    return false;
                }
            }
            // `:740-744` — otherwise every operand of its definition joins the worklist.
            Some(def) => worklist.extend(dfir_op::operands(def)),
        }
    }

    // `:746`
    true
}

#[cfg(test)]
mod merge_and_hoist_tests {
    use super::*;
    use crate::arch::Dd2;
    use crate::islands::dataflow_ir::Units;
    use crate::islands::dataflow_ir::link::{Link, Lxlu, Lxsu};
    use crate::islands::dataflow_ir::ty::{ElemType, Vector};
    use crate::units::DfirUnit;

    /// 🎯 370/384 — THE VENDOR'S OWN CONTRAST: A SIDE-EFFECTING OP BETWEEN THE PAIR REFUSES THE MERGE.
    ///
    /// `merging-shallow-skip.mlir:80-83` states the rule — *"do not shallow-merge them if … the
    /// operations between the conditionals in their block have side effects"* — and its input puts a
    /// `dataflow.receive`/`send` sequence between two `scf.if %12` on one `arith.cmpi` (`:117-131`),
    /// which the expectation leaves unmerged (`:38-77`).
    #[test]
    fn a_side_effecting_op_between_the_pair_refuses_the_merge() {
        let oe = OperationEquivalence::tagged(
            EquivalenceTag::CfgMergingAndHoistingCondTree,
            SubregionCompare::Recursive,
            BlockArgEquivalence::SameOwnerAndIndex,
        );
        let vec_ty = Vector {
            len: 64,
            elem: ElemType::F16,
        };
        let cmp = DfirOp::Arith(arith::Op::Compare {
            result: Val(12),
            predicate: arith::CmpIPredicate::Eq,
            lhs: Val(1),
            rhs: Val(2),
        });
        let an_if = || {
            DfirOp::Scf(scf::Op::If {
                cond: Val(12),
                results: Vec::new(),
                body: Vec::new(),
                else_body: Vec::new(),
                dbg_name: None,
            })
        };

        let clean = vec![cmp.clone(), an_if(), an_if()];
        assert!(are_shallowly_mergeable(
            &oe, &clean[1], &clean[2], &clean, &clean
        ));

        let (to, from) = Link::<Lxlu, Lxsu>::between(Val(6), Val(10)).ends();
        let blocked = vec![
            cmp,
            an_if(),
            DfirOp::Dataflow(dataflow::Op::Receive {
                result: Val(30),
                from,
                ty: vec_ty,
            }),
            DfirOp::Dataflow(dataflow::Op::Send {
                to,
                data: Val(30),
                ty: vec_ty,
            }),
            an_if(),
        ];
        assert!(!are_shallowly_mergeable(
            &oe,
            &blocked[1],
            &blocked[4],
            &blocked,
            &blocked
        ));
    }

    /// 🎯 371/384 — A CONDITIONAL WHOSE RESULT FEEDS ANYTHING BUT A VIEW OR A LINK IS NOT A CANDIDATE.
    ///
    /// `:770-775` restricts hoisting to results used only by `dataflow.get_logical_memory_view`,
    /// `dataflow.send` or `dataflow.receive`; an `arith.addi` reader stops the walk before it reaches
    /// the unported `moveAncestorsToMaintainDominance`, so this returns rather than panicking.
    #[test]
    fn a_conditional_read_by_arithmetic_is_not_hoisted() {
        let body = vec![DfirOp::Affine(affine::Op::For {
            iv: Val(1),
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(28),
            carried: Vec::new(),
            dbg_name: None,
            body: vec![
                DfirOp::Scf(scf::Op::If {
                    cond: Val(12),
                    results: vec![Val(20)],
                    body: Vec::new(),
                    else_body: Vec::new(),
                    dbg_name: None,
                }),
                DfirOp::Arith(arith::Op::AddI(arith::IntBinary {
                    result: Val(21),
                    lhs: Val(20),
                    rhs: Val(1),
                    ty: ScalarTy::Index,
                })),
            ],
        })];
        let unit = ProgramUnit::<Dd2> {
            on: Units::one(DfirUnit::Lxlu, Val(0)),
            precision: None,
            body,
            arch: core::marker::PhantomData,
        };
        let tree = CfgsDataflowConditionalTree::new(&unit);
        hoist_loop_invariant_conditionals(&tree);
    }
}
