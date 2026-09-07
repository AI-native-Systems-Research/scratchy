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

use crate::arch::Arch;
use crate::islands::dataflow_ir::ProgramUnit;
use crate::islands::dataflow_ir::dialects::{Op as DfirOp, Val, arith, scf};

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
/// (`DataflowOpInterfaces.cpp:24-31`) — hence `Option<&str>` per operation rather than `&str`. The
/// early `return nullptr` throws away the partially built string, so the result is all-or-nothing and
/// `?` is exactly it. ⭐ AND THE CALLERS ALL TREAT IT THAT WAY: every site is
/// `if (StringAttr n = getNewDbgNameFromList(...)) setDbgNameAttr(dst, n);`, which leaves the
/// destination's existing name alone rather than clearing it.
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
/// (`CFGSDataflowConditionalTree.cpp:479`, `instance` in `simplifyValueBasedConditionals`). ⭐ HERE
/// THE HAZARD IS NOT AVAILABLE: the type is not `Copy`, and its derived `Clone` deep-copies the
/// boxed slice, so a clone owns its own array. That is a divergence in the reference's favour, and it
/// is the reason `Clone` is derived rather than suppressed.
///
/// ⚠️ THE OTHER SIX FIELDS ARE NOT HERE, AND NEITHER IS THE CONSTRUCTOR. `is_candidate_`,
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
/// # ⛔⛔ NO FUNCTOR, BECAUSE NOTHING IN THE REFERENCE EVER PASSES ONE
///
/// The six-argument constructor takes `function_ref<bool(Operation&, Operation&, void*)> functor`
/// and a `void *extent` for it to read (`dcc/src/Analysis/OperationEquivalence.hpp:35-45`), and
/// `functor_` is consulted at `OperationEquivalence.cpp:116`. All five construction sites in `dcc`
/// pass `nullptr, nullptr` — this file's `:112`, `CFGSSentientLevelConditionalTree.hpp:316`,
/// `CFGDeepMergingConditionalTree.hpp:29`, `LoopMerging.cpp:58` and `LoopAbsorption.cpp:43` — and the
/// three-argument constructor leaves it default-constructed, which for a `function_ref` is null. ⭐ SO
/// THE OVERRIDE HOOK IS DEAD IN THE WHOLE REFERENCE, and modelling it would be modelling a
/// possibility no input can reach. (The per-call `operands_equiv_checker` on
/// `operationsAreEquivalent` is a different parameter and is used; it belongs to the call, not here.)
///
/// ⚠️ `eq_classes_` IS NOT A FIELD HERE EITHER: it is the memo the comparison fills as it runs, which
/// is the mechanism the brief lets a port drop, and it arrives with whichever unit ports
/// `operationsAreEquivalent` — a function of `dcc/src/Analysis/`, outside the 384.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperationEquivalence {
    /// `debug_` — which `-debug-only=` name this comparison traces under.
    pub debug: EquivalenceTag,
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
            subregions,
            block_args,
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
    use super::*;
    use crate::arch::Dd2;
    use crate::islands::dataflow_ir::Units;
    use crate::units::DfirUnit;

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
            body: Vec::new(),
            else_body: Vec::new(),
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
}
