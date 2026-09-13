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

//! `StandardToSentient.cpp` — 12 of bridge 2's 384 functions (dependency level(s) [0, 6, 7, 8]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e048_getSentientCmpIPredicate` | 048/384 | 18 | `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:36` |
//! | `e049_LowerAddIOpToSentient` | 049/384 | 9 | `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:79` |
//! | `e050_LowerSubIOpToSentient` | 050/384 | 10 | `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:90` |
//! | `e051_LowerMulIOpToSentient` | 051/384 | 9 | `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:102` |
//! | `e052_If` | 052/384 | 0 | `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:159` |
//! | `e053_LowerConstantIndexToSentient` | 053/384 | 8 | `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:347` |
//! | `e054_LowerConstantIntToSentient` | 054/384 | 15 | `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:358` |
//! | `e338_ConstructIFRecursively` | 338/384 | 125 | `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:113` |
//! | `e339_SimplifyOrIOp` | 339/384 | 43 | `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:389` |
//! | `e362_LowerSelectOpToSentient` | 362/384 | 19 | `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:243` |
//! | `e363_LowerLogicalOpToSentient` | 363/384 | 19 | `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:264` |
//! | `e376_runOnOperation` | 376/384 | 34 | `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:439` |

use super::SenOp;
use crate::arch::Arch;
use crate::islands::dataflow_ir::dialects::arith::{CmpIPredicate, IntBinary, IntConst, LogicKind};
use crate::islands::dataflow_ir::dialects::{
    Op as DfirOp, Val, arith, dbg_name, defining_op, region_owner, regions, regions_mut,
    replace_all_uses, results, uses,
};
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::dataflow_ir::{self as dfir};
use crate::islands::sentient::dialects::sentient as sen;

/// Replaces: e049_LowerAddIOpToSentient
///
/// **049/384** `StandardToSentientLoweringPass::LowerAddIOpToSentient` —
/// `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:79` (9L).
///
/// ```cpp
/// void StandardToSentientLoweringPass::LowerAddIOpToSentient(Operation *op) {
///   auto addi_op = llvm::dyn_cast<mlir::arith::AddIOp>(op);
///   OpBuilder builder(addi_op);
///   auto sentient_add_op = sentient::AddOp::create(
///       builder, addi_op->getLoc(), addi_op.getLhs().getType(), addi_op.getLhs(),
///       addi_op.getRhs());
///   // sentient_add_op->setAttrs(addi_op->getAttrDictionary());
///   addi_op->replaceAllUsesWith(sentient_add_op);
///   addi_op->erase();
/// }
/// ```
///
/// # ⭐ `replaceAllUsesWith` + `erase` IS "THE RESULT IS THE SAME VALUE"
///
/// The pair means every use of the `arith.addi`'s result now reads the `sentient.scalar_add`'s, and
/// the old op is gone. In a typed IR with no rewriter there is nothing to erase and nothing to
/// re-point: the returned op **binds the same [`Val`]**, so every op that already named it reads the
/// new one. That is why these lowerings take an op and return an op rather than mutating a module,
/// and it is the whole of what those two lines say.
///
/// ⛔ THE RESULT TYPE IS THE LEFT OPERAND'S. `AddOp::create(..., addi_op.getLhs().getType(), ...)`
/// passes it in explicitly, and `SameOperandsAndResultType` on the op means the emitted form prints
/// it twice: `: index, index` (`SentientOps.td:700-710`). It is not always `index` — see
/// [`ScalarTy`].
///
/// ⛔ NO REGISTER, AND NOT `unknown` EITHER. The five-argument `create` leaves `regLocale` and
/// `regIndex` at their `DefaultValuedAttr` defaults, so the op carries neither and prints no
/// attribute dictionary at all — `%29 = sentient.scalar_add %18, %1 : index, index`
/// (`dcc/test/Conversion/StandardToSentient/cmpi_select_different_BB.mlir:58`). See
/// [`sen::Op::ScalarAdd`] for why that is `None` and not [`sen::RegType::Unknown`].
///
/// ⛔ THE COMMENTED-OUT `setAttrs` IS THE REFERENCE'S OWN AND IT STAYS COMMENTED OUT. `scalar_sub`
/// says why in words — *"We do not use Arith Attributes"* (`:96`) — and only `scalar_mul` still does
/// it (see [`lower_muli_op_to_sentient`]).
#[must_use]
pub fn lower_addi_op_to_sentient(op: &IntBinary) -> sen::Op {
    sen::Op::ScalarAdd {
        lhs: op.lhs,
        rhs: op.rhs,
        result: op.result,
        reg: None,
        element_size: None,
        ty: op.ty,
    }
}

/// Replaces: e050_LowerSubIOpToSentient
///
/// **050/384** `StandardToSentientLoweringPass::LowerSubIOpToSentient` —
/// `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:90` (10L).
///
/// ```cpp
/// void StandardToSentientLoweringPass::LowerSubIOpToSentient(Operation *op) {
///   auto subi_op = llvm::dyn_cast<mlir::arith::SubIOp>(op);
///   OpBuilder builder(subi_op);
///   auto sentient_subi_op = sentient::SubOp::create(
///       builder, subi_op->getLoc(), subi_op.getLhs().getType(), subi_op.getLhs(),
///       subi_op.getRhs());
///   // We do not use Arith Attributes
///   // sentient_subi_op->setAttrs(subi_op->getAttrDictionary());
///   subi_op->replaceAllUsesWith(sentient_subi_op);
///   subi_op->erase();
/// }
/// ```
///
/// ⭐ THE ONE EXTRA LINE OVER [`lower_addi_op_to_sentient`] IS A COMMENT, AND IT IS THE EVIDENCE for
/// the whole family: *"We do not use Arith Attributes"*. An `arith` op's attribute dictionary is
/// deliberately dropped, not overlooked.
///
/// ⛔ ⭐ AND THE ORDER OF THE OPERANDS IS NOT NEGOTIABLE — subtraction does not commute, and the
/// `.td` does not mark `scalar_sub` `Commutative` while it marks `scalar_add` and `scalar_mul` so
/// (`SentientOps.td:700`, `:801`, `:816`). `%12 = sentient.scalar_sub %5, %11 : index, index` from
/// `%3 = arith.subi %c3, %arg0` (`cmpi_select_different_BB.mlir:19`) reads the loop bound minus the
/// induction variable; swapped, it counts up from a negative.
#[must_use]
pub fn lower_subi_op_to_sentient(op: &IntBinary) -> sen::Op {
    sen::Op::ScalarSub {
        lhs: op.lhs,
        rhs: op.rhs,
        result: op.result,
        reg: None,
        element_size: None,
        ty: op.ty,
    }
}

/// Replaces: e051_LowerMulIOpToSentient
///
/// **051/384** `StandardToSentientLoweringPass::LowerMulIOpToSentient` —
/// `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:102` (9L).
///
/// ```cpp
/// void StandardToSentientLoweringPass::LowerMulIOpToSentient(Operation *op) {
///   auto muli_op = llvm::dyn_cast<mlir::arith::MulIOp>(op);
///   OpBuilder builder(muli_op);
///   auto sentient_muli_op = sentient::MulOp::create(
///       builder, muli_op->getLoc(), muli_op.getLhs().getType(), muli_op.getLhs(),
///       muli_op.getRhs());
///   sentient_muli_op->setAttrs(muli_op->getAttrDictionary());
///   muli_op->replaceAllUsesWith(sentient_muli_op);
///   muli_op->erase();
/// }
/// ```
///
/// # ⛔⛔ THE ONE LINE THAT DIFFERS FROM ADD AND SUB IS THE `setAttrs`, AND IT COPIES AN EMPTY DICTIONARY
///
/// `sentient_muli_op->setAttrs(muli_op->getAttrDictionary())` replaces the emitted op's whole
/// attribute dictionary with the `arith.muli`'s. Three facts make that dictionary empty everywhere
/// this pipeline can reach it, and they are checkable rather than assumed:
///
/// 1. ⭐ NOTHING IN `dcc/src` EVER BUILDS AN `arith.muli`. The only two mentions of the class are
///    this `dyn_cast` and the `isa<>` that dispatches to it (`:103`, `:451`), so every one of them
///    arrives in the input module.
/// 2. ⭐ NO `arith.*` OP IN THE 825-FILE TEST TREE CARRIES AN ATTRIBUTE DICTIONARY, and `arith.muli`
///    appears in none of them at all.
/// 3. ⭐ AND THE `arith` OPS `dcc` DOES BUILD GET NO ATTRIBUTES EITHER. Its 12 `AddIOp::create`
///    sites pass operands only (`AgenToSentient/Helper.cpp:820`, `:996`, `:1102` and on), and of
///    the 12 `setAttrs` calls in the whole tree not one targets an `arith` op — they copy onto
///    `scf`, `vector` and `sentient` ops (`AffineToStandard.cpp:66`, `:168`,
///    `LightweightSimplification.cpp:138`, `LiveRangeReduction.cpp:653`).
///
/// ⛔ SO THE COPY IS THE IDENTITY, AND HERE IT IS THE IDENTITY **BY CONSTRUCTION**: [`IntBinary`]
/// declares no attribute at all, so there is no dictionary to copy and no way to write one. That is
/// the form of this fact a type can hold — and if the day comes that an input carries one, the type
/// has to grow before this line can lie.
///
/// ⛔ AND WHAT `setAttrs` WOULD ALSO DO IS NOTHING HERE: it overwrites the dictionary of the op just
/// created, which — see [`lower_addi_op_to_sentient`] — has no attributes of its own to lose.
///
/// ⛔ `scalar_mul` HAS NO `regIndex` TO LOSE EITHER. It declares only `regLocale`
/// (`SentientOps.td:816-829`), and the asymmetry with `scalar_add` is the reference's.
#[must_use]
pub fn lower_muli_op_to_sentient(op: &IntBinary) -> sen::Op {
    sen::Op::ScalarMul {
        lhs: op.lhs,
        rhs: op.rhs,
        result: op.result,
        reg_locale: None,
        ty: op.ty,
    }
}

/// ONE CONJUNCT OF A CONDITION, AND THE `sentient.if` IT BECOMES.
///
/// ⛔ THE COMPARISON IS ALREADY TRANSLATED. `ConstructIFRecursively`'s base case reads the
/// `arith.cmpi`'s predicate through `getSentientCmpIPredicate` (entry 048) and its two operands
/// straight through (`StandardToSentient.cpp:126-132`); by the time the nesting rule below applies,
/// the conjunct is a predicate and two values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conjunct {
    /// `$predicate`, already mapped from the `arith.cmpi`'s.
    pub predicate: sen::CmpPredicate,
    /// `$lhs` — the comparison's left operand.
    pub lhs: Val,
    /// `$rhs` — its right operand.
    pub rhs: Val,
    /// The value the `sentient.if` built for this conjunct binds.
    pub result: Val,
    /// `$dbgName` — ⛔ THE `arith.andi`'S NAME LANDS ON **BOTH** ITS CONJUNCTS' ifs, not on one
    /// (`:154-157`: `setDbgNameAttr(lhs_if_op, ...)` and `setDbgNameAttr(rhs_if_op, ...)`).
    pub dbg_name: Option<String>,
}

/// Replaces: e052_If
///
/// **052/384** `If` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:159` (0L).
///
/// ```cpp
///     // return If(lhs) {If(rhs) true_val; else false_val} else false_val;
/// ```
///
/// # ⛔⛔ ENTRY 052 IS A COMMENT, NOT A DEFINITION — AND IT STILL STATES A LAW
///
/// The cited line is the comment above the `and` branch of `ConstructIFRecursively`, inside the body
/// of entry 338. The extractor read `If(lhs) {` as a function called `If` with an empty body, which
/// is why the unit measures 0 lines and why `crustify-bridge2/source/bridge2.cpp:648` holds one
/// comment where every other entry holds a function. There is no `If` function in `dcc/src` to port.
///
/// ⭐ WHAT THE LINE DOES SAY IS THE **SHAPE** A CONJUNCTION LOWERS TO, and that shape is a fact
/// about the emitted IR rather than about the recursion that walks to it: one `sentient.if` per
/// conjunct, nested through the `then` arm, the innermost `then` yielding the true value and *every*
/// `else` yielding the false one. That is what this type and [`NestedIf::into_op`] hold. Entry 338
/// keeps what is genuinely its own — the recursion, the `arith.cmpi` base case, the `or` branch, the
/// one-result fallback and the erase bookkeeping — and reaches this for its `and` branch.
///
/// ⛔⛔ THE COMMENT NAMES THE WRONG SIDE AS THE OUTER ONE. It writes `If(lhs) {If(rhs) ...}`, but the
/// code below it returns **`rhs_if_op`** and clones `lhs_if_op` into every `yield` of the true value
/// inside it (`:161-171`), then erases `lhs_if_op`. So the right-hand conjunct is the outer `if` and
/// the left-hand one is nested in its `then` arm — the mirror of what the comment draws. A
/// conjunction commutes, so the emitted program is right either way; the note is here because a port
/// that follows the comment and a port that follows the code print different text, and only one of
/// them matches the reference's output.
///
/// ⛔ `conjuncts` IS ORDERED OUTERMOST FIRST, therefore, and a caller that has an `and_op` in hand
/// pushes `getRhs()`'s conjunct before `getLhs()`'s.
///
/// ⚠️ ONE PRINTED DIVERGENCE REMAINS, AND IT IS THE ISLAND'S, NOT THIS RULE'S: the reference prints
/// a freshly lowered `if`'s registers as `regIndices = [], regLocales = [#sentient<reg_type
/// unknown>]` — an *empty* index array beside a one-entry locale array
/// (`cmpi_select_different_BB.mlir:22`) — while [`sen::Yielded`] deliberately locks the two arrays
/// to one length and spells a locale as a quoted string. Correcting that is a change to how every
/// `sentient.if` and `sentient.for` in the island prints, which is a surface other units own, so the
/// tests below assert the **structure** this rule produces and not its text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NestedIf {
    /// The conjuncts, ⛔ OUTERMOST FIRST — see the note above on which side that is.
    pub conjuncts: Vec<Conjunct>,
    /// What the innermost `then` arm yields.
    pub true_value: Val,
    /// What every `else` arm yields.
    pub false_value: Val,
    /// The type the whole nest yields — `TypeRange{true_value.getType()}` (`:127`).
    pub ty: ScalarTy,
}

impl NestedIf {
    /// THE NEST ITSELF — one `sentient.if` per conjunct, the next one inside the last one's `then`.
    ///
    /// ⛔ THE INNER `if`'S RESULT IS WHAT THE OUTER `then` YIELDS. The reference reaches that by
    /// replacing the operand of the `yield` it clones into (`:168`:
    /// `yield_op.setOperand(0, cloned_op->getResult(0))`), so the arm holds the nested `if` **and**
    /// a `yield` of its result, in that order — which is exactly how the reference's output reads:
    ///
    /// ```text
    /// %22 = sentient.if eq, %12, %8 : index -> (index) {...}{
    ///   %23 = sentient.if eq, %12, %7 : index -> (index) {...}{
    ///     sentient.yield %20 : index
    ///   } else{
    ///     sentient.yield %19 : index
    ///   }
    ///   sentient.yield %23 : index
    /// } else{
    /// ```
    ///
    /// ⛔ AND EVERY `else` YIELDS THE FALSE VALUE, at every depth: the conjunction is false as soon
    /// as one conjunct is.
    ///
    /// ⛔ AN EMPTY `conjuncts` CANNOT ARISE AND IS NOT REFUSED. The reference only reaches the `and`
    /// branch holding an `arith.andi`, which has two operands, so a nest has at least two levels; a
    /// caller that passes one conjunct gets the single `if` the base case builds, and one that
    /// passes none gets a bare `sentient.yield` of the true value — the conjunction of nothing.
    ///
    /// ⛔ IT TAKES `self` BY VALUE BECAUSE `into_` PROMISES TO. The nest owns every conjunct's
    /// `dbgName`, and a `&self` form would have to clone each of them to hand the same names to the
    /// ops it builds — the only `into_*(&self)` in the crate, and the one shape
    /// `clippy::wrong_self_convention` names.
    #[must_use]
    pub fn into_op(self) -> Vec<super::SenOp> {
        // ⭐ BUILT INSIDE OUT, so each level already has the result its parent must yield.
        let mut nest: Vec<super::SenOp> = vec![super::SenOp::Sentient(sen::Op::Yield {
            results: vec![self.true_value],
        })];
        for conjunct in self.conjuncts.into_iter().rev() {
            nest = vec![
                super::SenOp::Sentient(sen::Op::If {
                    predicate: conjunct.predicate,
                    lhs: conjunct.lhs,
                    rhs: conjunct.rhs,
                    // ⛔ `locale_attr` IS ONE `unknown` (`:124-125`) AND `regIndices` IS AN EMPTY
                    // `ArrayRef<int32_t>()` (`:131`) — see [`NestedIf`] on how this island prints
                    // that.
                    yielded: vec![sen::Yielded {
                        result: conjunct.result,
                        reg: sen::Reg {
                            locale: sen::RegType::Unknown,
                            index: None,
                        },
                        element_size: None,
                    }],
                    dbg_name: conjunct.dbg_name,
                    then_body: nest,
                    else_body: vec![super::SenOp::Sentient(sen::Op::Yield {
                        results: vec![self.false_value],
                    })],
                }),
                super::SenOp::Sentient(sen::Op::Yield {
                    results: vec![conjunct.result],
                }),
            ];
        }
        // ⭐ THE OUTERMOST `yield` IS THE ENCLOSING REGION'S, not this rule's: the reference returns
        // the `if` op itself and its caller decides what names the result.
        nest.truncate(1);
        nest
    }
}

/// Replaces: e053_LowerConstantIndexToSentient
///
/// **053/384** `StandardToSentientLoweringPass::LowerConstantIndexToSentient` —
/// `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:347` (8L).
///
/// ```cpp
/// void StandardToSentientLoweringPass::LowerConstantIndexToSentient(
///     Operation *op) {
///   auto const_index_op = llvm::dyn_cast<mlir::arith::ConstantIndexOp>(op);
///   OpBuilder builder(const_index_op);
///   auto sentient_const_op = sentient::ConstantOp::create(
///       builder, const_index_op.getLoc(), const_index_op.getResult().getType(),
///       const_index_op.value());
///   const_index_op->replaceAllUsesWith(sentient_const_op);
///   const_index_op->erase();
/// }
/// ```
///
/// ⛔ THE LAST LINE IS NOT IN THE EXTRACT. `crustify-bridge2/source/bridge2.cpp:650-659` ends at
/// `replaceAllUsesWith`, so the extract's copy of this function leaks the op it replaces. The
/// authority has `const_index_op->erase();`; here both lines together are the returned op reusing
/// the input's [`Val`] (see [`lower_addi_op_to_sentient`]).
///
/// ⛔ THE RESULT TYPE IS THE INPUT'S, WHICH FOR THIS OP CLASS IS ALWAYS `index` —
/// `arith::ConstantIndexOp` is the index-typed constant, which is why it has a lowering of its own
/// beside [`lower_constant_int_to_sentient`].
///
/// ⛔ AND THE LOCALE IS `imm`, NOT `unknown`. The four-argument `ConstantOp::create` leaves
/// `regLocale` at its declared default, and for this op alone that default is
/// [`sen::RegType::Imm`] (`SentientOps.td:848-852`) — a constant is an instruction field until
/// something spills it. ⭐ It never prints either way ([`sen::Op::ScalarConstant`]).
///
/// ⚠️ `inherit_constants` AND `mint_constants` IN THIS BRIDGE'S `mod.rs` ALREADY LOWER
/// `arith.constant`s BY HAND, and write `Unknown` where this writes `Imm`. They are scaffolding this
/// port supersedes; folding them onto this function is entry 376's job (`runOnOperation`, the walk
/// that dispatches every op in the module), not this one's.
#[must_use]
pub fn lower_constant_index_to_sentient(result: Val, value: i64) -> sen::Op {
    sen::Op::ScalarConstant {
        is_symbol: false,
        value,
        result,
        reg_locale: sen::RegType::Imm,
        ty: ScalarTy::Index,
    }
}

/// Replaces: e054_LowerConstantIntToSentient
///
/// **054/384** `StandardToSentientLoweringPass::LowerConstantIntToSentient` —
/// `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:358` (15L).
///
/// ```cpp
/// void StandardToSentientLoweringPass::LowerConstantIntToSentient(Operation *op) {
///   auto const_int_op = llvm::dyn_cast<mlir::arith::ConstantIntOp>(op);
///   OpBuilder builder(const_int_op);
///
///   int val = -1;
///   auto bool_attr = mlir::cast<mlir::BoolAttr>(const_int_op.getValue());
///   if (bool_attr) {
///     val = bool_attr.getValue();
///   } else {
///     val = const_int_op.value();
///   }
///   auto sentient_const_op = sentient::ConstantOp::create(
///       builder, const_int_op.getLoc(), const_int_op.getResult().getType(), val);
///   const_int_op->replaceAllUsesWith(sentient_const_op);
///   const_int_op->erase();
/// }
/// ```
///
/// # ⛔⛔ THE TWO BRANCHES EXIST BECAUSE `value()` SIGN-EXTENDS AND AN `i1` TRUE WOULD ARRIVE AS -1
///
/// `ConstantIntOp::value()` reads the `IntegerAttr`'s `APInt` as a signed 64-bit integer. For a
/// signless one-bit integer that makes `true` into **-1**, and `sentient.scalar_constant` would
/// carry `{value = -1 : si64} : i1` — a value the reference's own output never shows. Reading the
/// same attribute as a `BoolAttr` gives a `bool`, so `val` becomes 0 or 1. Both spellings appear in
/// reference SentientIR and both are non-negative:
///
/// ```text
/// %2 = sentient.scalar_constant {value = 0 : si64} : i1
/// %2 = sentient.scalar_constant {value = 1 : si64} : i1
/// ```
///
/// (`dcc/test/Conversion/SentientToProgIR/simplify_or_op.mlir:8` from an `arith.constant false`, and
/// `dcc/test/PE/conditional-2and3-nested-if.mlir:38`.)
///
/// # ⛔⛔ AND THE `if (bool_attr)` IS DEAD AS WRITTEN — A DEFECT THIS PORT DOES NOT COPY
///
/// `mlir::cast` is the asserting cast: it does not return null on a type mismatch, it aborts. So on
/// an `i1` the guard is a pointer that is always non-null and the `else` branch is unreachable,
/// while on any wider integer the function never gets as far as the guard. The intent is legible —
/// `mlir::dyn_cast` and the two branches are one letter apart, `int val = -1` is initialised for a
/// path that then cannot be taken, and `else { val = const_int_op.value(); }` is written for exactly
/// the wider integers `cast` rejects. ⭐ SO THE PORT ENCODES THE DISTINCTION IN THE INPUT TYPE
/// ([`IntConst`]) INSTEAD: both arms are reachable, neither aborts, an `i1` reads as a boolean and a
/// wider integer keeps its sign-extended value. This is a deliberate divergence from the reference's
/// behaviour on a non-`i1` constant, where the reference has no behaviour to match.
///
/// ⛔ THE RESULT TYPE IS THE INPUT CONSTANT'S OWN WIDTH — `getResult().getType()`, so `i1` for a
/// boolean and `i<bits>` otherwise, never `index`. That is the whole reason this lowering is
/// separate from [`lower_constant_index_to_sentient`], and `{value = 0 : si64} : i1` beside
/// `{value = 0 : si64} : index` in one function is what it looks like when both run
/// (`simplify_or_op.mlir:6-8`).
#[must_use]
pub fn lower_constant_int_to_sentient(result: Val, value: IntConst) -> sen::Op {
    let literal = match value {
        // ⭐ `bool_attr.getValue()` — A `bool`, so 0 or 1 and never -1.
        IntConst::Bool(flag) => i64::from(flag),
        // ⭐ `const_int_op.value()` — the `APInt` read signed, sign extension and all.
        IntConst::Int { value, .. } => value,
    };
    sen::Op::ScalarConstant {
        is_symbol: false,
        value: literal,
        result,
        reg_locale: sen::RegType::Imm,
        ty: value.ty(),
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 048/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e048_getSentientCmpIPredicate
///
/// **048/384** `getSentientCmpIPredicate` —
/// `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:36` (18L).
///
/// ```cpp
/// static CmpIPredicate getSentientCmpIPredicate(
///     mlir::arith::CmpIPredicate condop) {
///   if (condop == mlir::arith::CmpIPredicate::eq) {
///     return CmpIPredicate::eq;
///   } else if (condop == mlir::arith::CmpIPredicate::ne) {
///     return CmpIPredicate::ne;
///   } else if (condop == mlir::arith::CmpIPredicate::slt) {
///     return CmpIPredicate::slt;
///   } else if (condop == mlir::arith::CmpIPredicate::sle) {
///     return CmpIPredicate::sle;
///   } else if (condop == mlir::arith::CmpIPredicate::sgt) {
///     return CmpIPredicate::sgt;
///   } else if (condop == mlir::arith::CmpIPredicate::sge) {
///     return CmpIPredicate::sge;
///   } else {
///     DT_CHECK(0);
///   }
///   // to silence to warning
///   return CmpIPredicate::eq;
/// }
/// ```
///
/// # ⭐⭐ THE SIX SIGNED PREDICATES, NARROWED ACROSS A RUNG BOUNDARY
///
/// `mlir::arith::CmpIPredicate` declares ten enumerators; `SentientTypes.td:474-489` declares six.
/// So this is a narrowing and not a cast, which is why it is a function. Its one caller is
/// `ConstructIFRecursively`'s base case (`:126-133`), which reads the `arith.cmpi`'s predicate through
/// here and wraps the answer in a `CmpIPredicateAttr` for the `sentient.if` it builds — the value that
/// ends up in [`Conjunct::predicate`].
///
/// # ⛔⛔ `DT_CHECK(0)` HAS NO INPUT HERE, BY CONSTRUCTION
///
/// `ult`/`ule`/`ugt`/`uge` are the four values that reach it. This island's [`CmpIPredicate`] declares
/// the six signed forms and nothing else, *because* of this function — see that type's own note, which
/// cites this entry by number. An unsigned comparison reaching this pipeline is a value the IR cannot
/// hold rather than a run-time stop, so the abort is unreachable by construction and needs no arm.
///
/// # ⛔ AND THE FALL-THROUGH `return ...::eq;` IS NOT A DEFAULT
///
/// The comment says what it is: *"to silence to warning"*. `DT_CHECK(0)` does not return, so the line
/// exists to give the compiler a terminating path and is dead in every build where the check is armed.
/// Porting it as `_ => Eq` would turn an abort into the answer `eq` — a comparison silently lowered to
/// the wrong branch. The `match` below is total over the six, so there is no arm to give it.
///
/// # ⚠️ AND THERE ARE TWO COPIES OF THIS FUNCTION IN THE REFERENCE
///
/// Entry 045 is the same if-chain, file-static in `SCFToSentient.cpp:33`, whose `else` is
/// `llvm_unreachable("invalid predicate")` and which therefore has no fall-through return. Both are
/// scheduled and each is ported in its own pass's home
/// ([`super::std_scf_to_sentient::get_sentient_cmp_i_predicate`]) rather than shared: one file's copy
/// diverging from the other's is a fact about the reference that a single helper would hide, and their
/// `else` arms already differ.
#[must_use]
pub const fn get_sentient_cmp_i_predicate(condop: CmpIPredicate) -> sen::CmpPredicate {
    match condop {
        CmpIPredicate::Eq => sen::CmpPredicate::Eq,
        CmpIPredicate::Ne => sen::CmpPredicate::Ne,
        CmpIPredicate::Slt => sen::CmpPredicate::Slt,
        CmpIPredicate::Sle => sen::CmpPredicate::Sle,
        CmpIPredicate::Sgt => sen::CmpPredicate::Sgt,
        CmpIPredicate::Sge => sen::CmpPredicate::Sge,
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 338/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ONE NEST OF `sentient.if`s AND THE VALUE IT BINDS — what `ConstructIFRecursively` returns.
///
/// ⛔ THE `i1` CONSTANTS ARE NOT INSIDE THE NEST: `builder.clone` copies the `if` alone, so a grafted
/// copy still reads the original constant — which is why the reference prints four constants above two
/// `if`s (`dcc/test/Conversion/SentientToProgIR/simplify_or_op.mlir:52-55` above `:56` and `:59`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nest {
    /// The `sentient.scalar_constant`s the recursion created, emitted ahead of [`Nest::op`].
    pub constants: Vec<SenOp>,
    /// The outermost `sentient.if`.
    pub op: SenOp,
    /// The value [`Nest::op`] binds — `transformed_op->getResult(0)`.
    pub result: Val,
}

/// Replaces: e338_ConstructIFRecursively
///
/// **338/384** `StandardToSentient.cpp:113` (125L): an `arith.select`'s condition, read back through
/// its `andi`/`ori` tree, becomes one nested `sentient.if` per comparison.
///
/// ⛔ THE RIGHT-HAND NEST IS THE OUTER `if`; the left is CLONED into every `yield` of the seam value —
/// `true_value` for `and`, `false_value` for `or` — and the original left nest erased (`:146-203`).
/// ⛔ A CONDITION THAT IS NEITHER is compared `eq` against a fresh `i1` true (`:205-238`); any other
/// result count is the reference's `emitError` + `signalPassFailure`, so [`None`].
#[must_use]
pub fn construct_if_recursively(
    original: &DfirOp,
    current: &DfirOp,
    true_value: Val,
    false_value: Val,
    scope: &[DfirOp],
    values: &mut dfir::Values,
    ops_to_be_erased: &mut Vec<Val>,
) -> Option<Nest> {
    match current {
        // `if (auto cmpi_op = llvm::dyn_cast<mlir::arith::CmpIOp>(current_op))`  `:117-145`
        DfirOp::Arith(arith::Op::Compare {
            result,
            predicate,
            lhs,
            rhs,
        }) => {
            let nest = one_if(
                get_sentient_cmp_i_predicate(*predicate),
                (*lhs, *rhs),
                dbg_name(current),
                (true_value, false_value),
                values,
            );
            // `ops_to_be_erased.push_back(cmpi_op);`
            ops_to_be_erased.push(*result);
            Some(nest)
        }
        // The `andi` arm (`:146-175`) and the `ori` arm (`:176-204`), which differ only in WHICH
        // yielded value the left nest is grafted onto.
        DfirOp::Arith(arith::Op::Logic {
            result,
            kind: kind @ (LogicKind::And | LogicKind::Or),
            operands,
        }) => {
            let [lhs, rhs] = operands[..] else {
                return None;
            };
            let arms = (true_value, false_value);
            let mut left = recurse_into(original, lhs, arms, scope, values, ops_to_be_erased)?;
            let mut nest = recurse_into(original, rhs, arms, scope, values, ops_to_be_erased)?;
            // ⛔ THE NAME LANDS ON BOTH SIDES (`:155-156`, `:185-186`), not on one.
            if let Some(name) = dbg_name(current) {
                name_if(&mut left.op, name);
                name_if(&mut nest.op, name);
            }
            // A conjunction is false as soon as one conjunct is, a disjunction true as soon as one is.
            let seam = if matches!(kind, LogicKind::And) {
                true_value
            } else {
                false_value
            };
            // `rhs_if_op->walk(..)` — `:161-171`, `:190-200`.
            graft(&mut nest.op, seam, &left, values);
            // `ops_to_be_erased.push_back(and_op); lhs_if_op->erase(); return rhs_if_op;`
            ops_to_be_erased.push(*result);
            left.constants.append(&mut nest.constants);
            nest.constants = left.constants;
            Some(nest)
        }
        // `} else if (current_op->getNumResults() == 1) {`  `:205-238`
        current => {
            let [only] = results(current)[..] else {
                // `emitError("The input condition to std.select should come from CMPI/AND/OR
                //  operations or an op with one return value"); signalPassFailure();`
                return None;
            };
            // ⛔ AND THIS ARM ERASES NOTHING: the op supplying the condition is still read.
            // ⛔ THE NAME IS THE `arith.select`'S HERE, not this op's (`:221-222`).
            Some(eq_true_nest(
                dbg_name(original),
                only,
                (true_value, false_value),
                values,
            ))
        }
    }
}

/// ONE OPERAND RECURSED INTO — `ConstructIFRecursively(original, v.getDefiningOp(), ..)`.
///
/// ⛔⛔ THE DEFINING OP IS OFTEN NO LONGER AN `arith` OP, and that is not an edge case: the walk
/// lowers each `cmpi` before the `andi`/`ori` over it and `replaceAllUsesWith` re-points this
/// operand at the `sentient.if` it produced, so the reference's `dyn_cast`s all miss and it takes the
/// one-result arm. The vendor's own output shows it — `sentient.if eq, %[[VAL_20]], %[[VAL_23]] : i1`
/// over an `if`, not over a comparison
/// (`dcc/test/Conversion/SentientToProgIR/simplify_or_op.mlir:59`, `%[[VAL_20]]` being the `if` at `:47`).
/// ⛔ [`None`] IS THE REGION ARGUMENT the reference dereferences as a null defining op.
fn recurse_into(
    original: &DfirOp,
    operand: Val,
    arms: (Val, Val),
    scope: &[DfirOp],
    values: &mut dfir::Values,
    ops_to_be_erased: &mut Vec<Val>,
) -> Option<Nest> {
    if let Some(def) = defining_op(operand, scope) {
        construct_if_recursively(
            original,
            def,
            arms.0,
            arms.1,
            scope,
            values,
            ops_to_be_erased,
        )
    } else if region_owner(operand, scope).is_some() {
        None
    } else {
        Some(eq_true_nest(dbg_name(original), operand, arms, values))
    }
}

/// THE `getNumResults() == 1` ARM OF ENTRY 338 — `cond == true` against a fresh `i1` 1
/// (`StandardToSentient.cpp:205-238`). ⭐ ENTRY 362 REACHES IT TOO, because a condition entry 363
/// already lowered is no longer an `arith` op at all; see [`lower_select_op_to_sentient`].
fn eq_true_nest(
    dbg_name: Option<&str>,
    only: Val,
    arms: (Val, Val),
    values: &mut dfir::Values,
) -> Nest {
    // `auto val_true = sentient::ConstantOp::create(.., builder.getI1Type(), 1);`  `:214-215`
    let val_true = values.mint();
    let mut nest = one_if(
        sen::CmpPredicate::Eq,
        (only, val_true),
        dbg_name,
        arms,
        values,
    );
    nest.constants.push(i1_constant(1, val_true));
    nest
}

/// ONE `sentient.scalar_constant` OF `i1` TYPE — `ConstantOp::create(.., getI1Type(), v)`.
fn i1_constant(value: i64, result: Val) -> SenOp {
    SenOp::Sentient(sen::Op::ScalarConstant {
        value,
        result,
        reg_locale: sen::RegType::Imm,
        ty: ScalarTy::Int(1),
        is_symbol: false,
    })
}

/// ONE `sentient.if` WITH ITS TWO YIELDING ARMS — `:123-142`, built again at `:210-231`.
/// ⛔ `regLocales` IS ONE `unknown` AND `regIndices` IS EMPTY.
fn one_if(
    predicate: sen::CmpPredicate,
    compared: (Val, Val),
    dbg_name: Option<&str>,
    arms: (Val, Val),
    values: &mut dfir::Values,
) -> Nest {
    let result = values.mint();
    Nest {
        constants: Vec::new(),
        op: SenOp::Sentient(sen::Op::If {
            predicate,
            lhs: compared.0,
            rhs: compared.1,
            yielded: vec![sen::Yielded {
                result,
                reg: sen::Reg {
                    locale: sen::RegType::Unknown,
                    index: None,
                },
                element_size: None,
            }],
            dbg_name: dbg_name.map(str::to_owned),
            then_body: vec![SenOp::Sentient(sen::Op::Yield {
                results: vec![arms.0],
            })],
            else_body: vec![SenOp::Sentient(sen::Op::Yield {
                results: vec![arms.1],
            })],
        }),
        result,
    }
}

/// `setDbgNameAttr(if_op, orig_dbg_name)` — the name goes on the nest's outermost `if`.
fn name_if(op: &mut SenOp, name: &str) {
    if let SenOp::Sentient(sen::Op::If { dbg_name, .. }) = op {
        *dbg_name = Some(name.to_owned());
    }
}

/// THE WALK THAT REPLACES ONE NEST'S SEAM WITH A COPY OF ANOTHER — every `sentient.yield` of `seam`
/// gets a FRESH clone of `copied` ahead of it and yields that clone's result instead.
fn graft(into: &mut SenOp, seam: Val, copied: &Nest, values: &mut dfir::Values) {
    if let SenOp::Sentient(sen::Op::If {
        then_body,
        else_body,
        ..
    }) = into
    {
        graft_body(then_body, seam, copied, values);
        graft_body(else_body, seam, copied, values);
    }
}

/// One region of that walk. ⛔ THE CLONE IS NOT WALKED INTO: it is inserted ahead of the `yield` the
/// walk is visiting, which the walk has already passed.
fn graft_body(body: &mut Vec<SenOp>, seam: Val, copied: &Nest, values: &mut dfir::Values) {
    let mut at = 0;
    while at < body.len() {
        let yields_seam = matches!(
            &body[at],
            SenOp::Sentient(sen::Op::Yield { results }) if results.first() == Some(&seam)
        );
        if yields_seam {
            let mut mapping = dfir::ValueMapping::new();
            // `auto cloned_op = builder.clone(*lhs_if_op);`
            let clone = clone_if(&copied.op, values, &mut mapping);
            // `yield_op.setOperand(0, cloned_op->getResult(0));`
            if let SenOp::Sentient(sen::Op::Yield { results }) = &mut body[at] {
                results[0] = mapping.lookup_or_default(copied.result);
            }
            body.insert(at, clone);
            at += 1;
        } else {
            graft(&mut body[at], seam, copied, values);
        }
        at += 1;
    }
}

/// A FRESH COPY OF ONE `sentient.if` — `builder.clone`, which mints a new value for everything the
/// copy defines and leaves everything defined OUTSIDE it alone (see [`Nest`]).
fn clone_if(op: &SenOp, values: &mut dfir::Values, mapping: &mut dfir::ValueMapping) -> SenOp {
    match op {
        SenOp::Sentient(sen::Op::If {
            predicate,
            lhs,
            rhs,
            yielded,
            dbg_name,
            then_body,
            else_body,
        }) => {
            // ⭐ THE RESULTS COME BEFORE THE REGIONS, as `Operation::clone` numbers them.
            let yielded: Vec<sen::Yielded> = yielded
                .iter()
                .map(|one| {
                    let fresh = values.mint();
                    mapping.map(one.result, fresh);
                    sen::Yielded {
                        result: fresh,
                        reg: one.reg.clone(),
                        // ⭐ `Operation::clone` COPIES EVERY ATTRIBUTE, `element_sizes` INCLUDED.
                        element_size: one.element_size,
                    }
                })
                .collect();
            SenOp::Sentient(sen::Op::If {
                predicate: *predicate,
                lhs: mapping.lookup_or_default(*lhs),
                rhs: mapping.lookup_or_default(*rhs),
                yielded,
                dbg_name: dbg_name.clone(),
                then_body: clone_body(then_body, values, mapping),
                else_body: clone_body(else_body, values, mapping),
            })
        }
        SenOp::Sentient(sen::Op::Yield { results }) => SenOp::Sentient(sen::Op::Yield {
            results: results
                .iter()
                .map(|val| mapping.lookup_or_default(*val))
                .collect(),
        }),
        // ⛔ A NEST HOLDS NOTHING ELSE, and the constants it reads sit outside it (see [`Nest`]).
        other => other.clone(),
    }
}

/// One region of that clone.
fn clone_body(
    body: &[SenOp],
    values: &mut dfir::Values,
    mapping: &mut dfir::ValueMapping,
) -> Vec<SenOp> {
    body.iter()
        .map(|op| clone_if(op, values, mapping))
        .collect()
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 339/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e339_SimplifyOrIOp
///
/// **339/384** `StandardToSentient.cpp:389` (43L): `or(cmpi P, and(cmpi ¬P, x))` becomes
/// `or(cmpi P, x)` when both comparisons read the same pair — the shape SCCP leaves behind.
///
/// ⛔ THE ROLES DEFAULT TO (and, cmpi) and swap only for a `cmpi` at operand 0 beside an `andi` at 1.
/// ⛔ THE `cmpi` INSIDE THE `andi` IS DEREFERENCED WITH NO NULL TEST (`:415`) — an `andi` holding none
/// is nothing to simplify, and that is the answer taken here.
/// ⛔ THE OUTER `cmpi` SURVIVES: only the `andi` and the negated `cmpi` go, and only if left unread.
pub fn simplify_or_i_op(or_result: Val, scope: &mut Vec<DfirOp>, values: &mut dfir::Values) {
    // `auto or_op = llvm::dyn_cast<mlir::arith::OrIOp>(op);`
    let Some((first, second)) = logic_operands(or_result, LogicKind::Or, scope) else {
        return;
    };
    // `int andi_operand_num = 0; int cmpi_operand_num = 1;` and the one test that swaps them.
    let (cmpi_val, andi_val) = if cmpi_at(first, scope).is_some()
        && logic_operands(second, LogicKind::And, scope).is_some()
    {
        (first, second)
    } else {
        (second, first)
    };
    // `if (!cmpi_op || !andi_op) return;`
    let Some((predicate, lhs, rhs)) = cmpi_at(cmpi_val, scope) else {
        return;
    };
    let Some((and_first, and_second)) = logic_operands(andi_val, LogicKind::And, scope) else {
        return;
    };
    // `cmpi_op_neg = dyn_cast<CmpIOp>(andi_op.getOperand(0)..); int other_operand_num = 1;` and the
    // retry on operand 1 (`:409-413`).
    let (neg_val, other) = if cmpi_at(and_first, scope).is_some() {
        (and_first, and_second)
    } else {
        (and_second, and_first)
    };
    let Some((neg_predicate, neg_lhs, neg_rhs)) = cmpi_at(neg_val, scope) else {
        return;
    };
    // `if (!((eq && ne) || (ne && eq))) return;`
    if !matches!(
        (predicate, neg_predicate),
        (CmpIPredicate::Eq, CmpIPredicate::Ne) | (CmpIPredicate::Ne, CmpIPredicate::Eq)
    ) {
        return;
    }
    // `if (!(same pair, in either order)) return;`
    if !((lhs == neg_lhs && rhs == neg_rhs) || (lhs == neg_rhs && rhs == neg_lhs)) {
        return;
    }
    // `new_or_op = OrIOp::create(builder, .., cmpi_op, andi_op.getOperand(other_operand_num));
    //  op->replaceAllUsesWith(new_or_op); or_op.erase();` — ⭐ the new op stands where the erased one
    // did, so it is written over it and every reader is repointed at the value it binds.
    let simplified = values.mint();
    rewrite_ori(or_result, simplified, (cmpi_val, other), scope);
    replace_all_uses(scope, or_result, simplified);
    // `if (andi_op.use_empty()) andi_op.erase();`
    if uses(andi_val, scope).is_empty() {
        erase_defining_op(andi_val, scope);
    }
    // `if (cmpi_op_neg.use_empty()) cmpi_op_neg.erase();`
    if uses(neg_val, scope).is_empty() {
        erase_defining_op(neg_val, scope);
    }
}

/// THE TWO OPERANDS OF THE `arith.andi` OR `arith.ori` AT A VALUE — [`None`] for any other op, which
/// is the null the reference tests for.
fn logic_operands(val: Val, want: LogicKind, scope: &[DfirOp]) -> Option<(Val, Val)> {
    match defining_op(val, scope) {
        Some(DfirOp::Arith(arith::Op::Logic { kind, operands, .. })) if *kind == want => {
            match operands[..] {
                [first, second] => Some((first, second)),
                _ => None,
            }
        }
        _ => None,
    }
}

/// THE COMPARISON AT A VALUE — `dyn_cast<mlir::arith::CmpIOp>(val.getDefiningOp())`.
fn cmpi_at(val: Val, scope: &[DfirOp]) -> Option<(CmpIPredicate, Val, Val)> {
    match defining_op(val, scope) {
        Some(DfirOp::Arith(arith::Op::Compare {
            predicate,
            lhs,
            rhs,
            ..
        })) => Some((*predicate, *lhs, *rhs)),
        _ => None,
    }
}

/// THE `arith.ori` BINDING `old`, REWRITTEN TO BIND `new` AND READ `operands` — one op created at the
/// erased one's position, at whatever depth that is.
fn rewrite_ori(old: Val, new: Val, operands: (Val, Val), scope: &mut [DfirOp]) {
    for op in scope.iter_mut() {
        if let DfirOp::Arith(arith::Op::Logic {
            kind: LogicKind::Or,
            result,
            operands: reads,
        }) = op
            && *result == old
        {
            *result = new;
            *reads = vec![operands.0, operands.1];
            return;
        }
        for region in regions_mut(op) {
            rewrite_ori(old, new, operands, region);
        }
    }
}

/// ONE OP GONE FROM THE BLOCK THAT HELD IT — `Operation::erase()`, at any depth.
fn erase_defining_op(val: Val, scope: &mut Vec<DfirOp>) {
    if let Some(at) = scope.iter().position(|op| results(op).contains(&val)) {
        scope.remove(at);
        return;
    }
    for op in scope.iter_mut() {
        for region in regions_mut(op) {
            erase_defining_op(val, region);
        }
    }
}

/// EVERY `arith.ori` IN ONE ROOT BLOCK, SIMPLIFIED — the first walk of `runOnOperation` (`:441-445`).
/// ⛔ COLLECTED BEFORE ANY REWRITE, because the walk cannot survive the erasures it causes.
fn simplify_every_ori(root: &mut Vec<DfirOp>, values: &mut dfir::Values) {
    let mut ors: Vec<Val> = Vec::new();
    collect_ors(root, &mut ors);
    for or in ors {
        simplify_or_i_op(or, root, values);
    }
}

/// The values every `arith.ori` in `ops` binds, preorder.
fn collect_ors(ops: &[DfirOp], into: &mut Vec<Val>) {
    for op in ops {
        if let DfirOp::Arith(arith::Op::Logic {
            kind: LogicKind::Or,
            result,
            ..
        }) = op
        {
            into.push(*result);
        }
        for region in regions(op) {
            collect_ors(region, into);
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 362/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e362_LowerSelectOpToSentient
///
/// **362/384** `StandardToSentient.cpp:243` (19L): an `arith.select` becomes the `sentient.if` nest
/// entry 338 builds from its condition; its uses move onto the nest and the condition tree goes.
///
/// ⛔ THE CONDITION IS OFTEN NO LONGER AN `arith` OP. Entry 363 reaches the `andi` first in the same
/// walk and re-points this operand at a `sentient.if` result, so `cond_def_op` is that `if` and the
/// recursion takes its one-result `eq true` arm. [`None`] is the region argument the reference
/// dereferences as a null defining op.
/// ⛔ ERASURE IS REVERSE-ORDER AND ONLY WHEN USE-EMPTY (`:255-259`); the select itself goes flatly.
#[must_use]
pub fn lower_select_op_to_sentient(
    select: &DfirOp,
    scope: &mut Vec<DfirOp>,
    values: &mut dfir::Values,
) -> Option<Nest> {
    let DfirOp::Arith(arith::Op::Select {
        result,
        condition,
        true_value,
        false_value,
        ..
    }) = select
    else {
        return None;
    };
    let (result, condition) = (*result, *condition);
    let arms = (*true_value, *false_value);
    let mut ops_to_be_erased: Vec<Val> = Vec::new();
    // `ConstructIFRecursively(select_op, cond_def_op, getTrueValue(), getFalseValue(), ..)`  `:249`
    let nest = recurse_into(
        select,
        condition,
        arms,
        scope,
        values,
        &mut ops_to_be_erased,
    )?;
    // `select_op->replaceAllUsesWith(transformed_op); select_op->erase();`  `:253-254`
    replace_all_uses(scope, result, nest.result);
    erase_defining_op(result, scope);
    erase_when_use_empty(&ops_to_be_erased, scope);
    Some(nest)
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 363/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e363_LowerLogicalOpToSentient
///
/// **363/384** `StandardToSentient.cpp:264` (19L): an `arith.andi` becomes entry 338's nest over
/// ITSELF, yielding a fresh `i1` 1 or 0 where entry 362 yields a select's two arms.
///
/// ⭐ THE `OrIOp` OVERLOAD AT `:286` IS THIS BODY BYTE FOR BYTE, so one port serves both kinds; the
/// `CmpIOp` one at `:308` is not, and is not among the 384 — see [`lower_cmpi_op_to_sentient`].
/// ⛔ AND THERE IS NO UNCONDITIONAL ERASE HERE (unlike entry 362): the `andi` goes only through the
/// reverse use-empty loop, which reaches it because `replaceAllUsesWith` ran first.
#[must_use]
pub fn lower_logical_op_to_sentient(
    logical: &DfirOp,
    scope: &mut Vec<DfirOp>,
    values: &mut dfir::Values,
) -> Option<Nest> {
    let DfirOp::Arith(arith::Op::Logic {
        kind: LogicKind::And | LogicKind::Or,
        ..
    }) = logical
    else {
        return None;
    };
    logical_nest(logical, scope, values)
}

/// `LowerLogicalOpToSentient(mlir::arith::CmpIOp)` — `StandardToSentient.cpp:308`, the third
/// overload. ⛔ NOT ONE OF THE 384, AND ENTRY 362 IS UNREACHABLE WITHOUT IT: every `arith.cmpi`
/// precedes the select it conditions in the walk, and this early return is what leaves that cmpi in
/// place for [`lower_select_op_to_sentient`] to recurse into (`:319-324`).
/// ⛔ THE `DT_CHECK` THAT ALL USERS AGREE IS AN ABORT, so the first user decides and a mixed set is
/// a program the reference does not accept.
fn lower_cmpi_op_to_sentient(
    cmpi: &DfirOp,
    scope: &mut Vec<DfirOp>,
    values: &mut dfir::Values,
) -> Option<Nest> {
    let DfirOp::Arith(arith::Op::Compare { result, .. }) = cmpi else {
        return None;
    };
    // `isa<SelectOp>(*op.getOperation()->getUsers().begin())` — unguarded on an empty use list.
    if let Some(DfirOp::Arith(arith::Op::Select { condition, .. })) =
        uses(*result, scope).first().copied()
        && condition == result
    {
        return None;
    }
    logical_nest(cmpi, scope, values)
}

/// THE BODY ALL THREE `LowerLogicalOpToSentient` OVERLOADS SHARE — `:266-282`, `:288-304`,
/// `:326-342`. ⛔ THE TWO `i1` CONSTANTS ARE MINTED BEFORE THE RECURSION and lead the nest.
fn logical_nest(op: &DfirOp, scope: &mut Vec<DfirOp>, values: &mut dfir::Values) -> Option<Nest> {
    let [result] = results(op)[..] else {
        return None;
    };
    // `val_true = ConstantOp(.., getI1Type(), 1); val_false = ConstantOp(.., getI1Type(), 0);`
    let (val_true, val_false) = (values.mint(), values.mint());
    let mut ops_to_be_erased: Vec<Val> = Vec::new();
    // `ConstructIFRecursively(op, op, val_true, val_false, insertion_point, ops_to_be_erased)`
    let mut nest = construct_if_recursively(
        op,
        op,
        val_true,
        val_false,
        scope,
        values,
        &mut ops_to_be_erased,
    )?;
    nest.constants
        .splice(0..0, [i1_constant(1, val_true), i1_constant(0, val_false)]);
    // `op->replaceAllUsesWith(transformed_op);`
    replace_all_uses(scope, result, nest.result);
    erase_when_use_empty(&ops_to_be_erased, scope);
    Some(nest)
}

/// THE REVERSE USE-EMPTY ERASE LOOP the three overloads and entry 362 all end with (`:255-259`).
fn erase_when_use_empty(ops_to_be_erased: &[Val], scope: &mut Vec<DfirOp>) {
    for val in ops_to_be_erased.iter().rev() {
        if uses(*val, scope).is_empty() {
            erase_defining_op(*val, scope);
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 376/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// EVERY `arith` OP IN ONE ROOT BLOCK, PREORDER — the second walk of `runOnOperation` (`:447-471`),
/// which descends into every region.
///
/// ⛔ SNAPSHOTTED BEFORE ANY LOWERING, because entries 362 and 363 erase the condition tree behind
/// them and no walk survives that. ⭐ THE SNAPSHOT IS STILL THE REFERENCE'S OWN ORDER: every op they
/// erase DOMINATES the one that erases it, so the walk had already visited it.
fn collect_arith(ops: &[DfirOp], into: &mut Vec<Val>) {
    for op in ops {
        if matches!(op, DfirOp::Arith(_)) {
            into.extend(results(op));
        }
        for region in regions(op) {
            collect_arith(region, into);
        }
    }
}

/// The second walk over one root block — see [`run_on_operation`] for the branch order.
fn lower_every_arith(root: &mut Vec<DfirOp>, values: &mut dfir::Values, out: &mut Vec<SenOp>) {
    let mut sites: Vec<Val> = Vec::new();
    collect_arith(root, &mut sites);
    for bound in sites {
        // ⛔ RE-READ, NEVER SNAPSHOTTED: entries 362 and 363 re-point the operands of the ops still
        // ahead of the walk, and a `cmpi` one of them already erased is simply no longer there.
        let Some(DfirOp::Arith(live)) = defining_op(bound, root) else {
            continue;
        };
        let arith_op = live.clone();
        let site = &DfirOp::Arith(arith_op.clone());
        match &arith_op {
            // `:447-448`
            arith::Op::AddI(binary) => out.push(SenOp::Sentient(lower_addi_op_to_sentient(binary))),
            // `:449-450`
            arith::Op::SubI(binary) => out.push(SenOp::Sentient(lower_subi_op_to_sentient(binary))),
            // `:451-454` — the reference's own refusal.
            arith::Op::MulI(binary) => todo!(
                "Cannot lower mul expressions to sentient: {:?} = arith.muli {:?}, {:?} \
                 (LowerMulIOpToSentient is commented out at StandardToSentient.cpp:454)",
                binary.result,
                binary.lhs,
                binary.rhs
            ),
            // `:455-456`
            arith::Op::Select { result, .. } => {
                let Some(nest) = lower_select_op_to_sentient(site, root, values) else {
                    todo!(
                        "The input condition to {:?} = arith.select comes from neither CMPI/AND/OR \
                         nor an op with one return value, which the reference reports with \
                         emitError + signalPassFailure (StandardToSentient.cpp:234-238)",
                        result
                    )
                };
                out.extend(nest.constants);
                out.push(nest.op);
            }
            // `:457-458` — `isa<ConstantIndexOp>`.
            arith::Op::Constant { result, value } => {
                out.push(SenOp::Sentient(lower_constant_index_to_sentient(
                    *result, *value,
                )));
            }
            // `:459-464` — `AndIOp`, `OrIOp` and `CmpIOp` all take entry 363's body. The `ori` arm
            // is all but unreachable behind the first walk; it is spelled out because the reference
            // spells it, and because entry 339 only folds the chains it recognises.
            arith::Op::Logic {
                result,
                kind: LogicKind::And | LogicKind::Or,
                ..
            } => {
                let Some(nest) = lower_logical_op_to_sentient(site, root, values) else {
                    todo!(
                        "An operand of {:?} = arith.andi/ori is a region argument or a multi-result \
                         op, which the reference reports with emitError + signalPassFailure \
                         (StandardToSentient.cpp:234-238)",
                        result
                    )
                };
                out.extend(nest.constants);
                out.push(nest.op);
            }
            arith::Op::Compare { .. } => {
                // ⛔ `None` IS THE EARLY RETURN, NOT A REFUSAL: a cmpi that conditions a select is
                // left for entry 362 to recurse into. See [`lower_cmpi_op_to_sentient`].
                if let Some(nest) = lower_cmpi_op_to_sentient(site, root, values) {
                    out.extend(nest.constants);
                    out.push(nest.op);
                }
            }
            // `:465-466` — `isa<ConstantIntOp>`.
            arith::Op::ConstantInt { result, value } => {
                out.push(SenOp::Sentient(lower_constant_int_to_sentient(
                    *result, *value,
                )));
            }
            // `:467-470` — the catch-all `arith::ConstantOp`, which is a pass failure.
            arith::Op::DenseConstant { result, ty, .. } => todo!(
                "StandardToSentient cannot lower {:?} = arith.constant : {:?} — the reference dumps \
                 it and signals pass failure (StandardToSentient.cpp:468-469)",
                result,
                ty
            ),
            // No arm in the reference: left exactly as they are.
            arith::Op::DivSI(_)
            | arith::Op::RemSI(_)
            // ⭐ AND THE TWO CONVERSIONS WITH THEM: `arith.sitofp` and `arith.fptosi` are named
            // nowhere in the reference's walk either.
            | arith::Op::SiToFp(_)
            | arith::Op::FpToSi(_)
            | arith::Op::Logic {
                kind: LogicKind::Not,
                ..
            } => {}
        }
    }
}

/// Replaces: e376_runOnOperation
///
/// `StandardToSentientLoweringPass::runOnOperation` (`StandardToSentient.cpp:439`) — TWO module
/// walks: the first simplifies every `arith.ori`, the second lowers the classes it recognises in the
/// reference's own `dyn_cast` order (`:447-471`); see [`lower_every_arith`].
///
/// ⛔ `arith.muli` IS A REFUSAL, NOT A GAP — `LowerMulIOpToSentient(op)` is COMMENTED OUT at `:454`
/// under `emitError("Cannot lower mul expressions to sentient")`, so the ported entry 051
/// [`lower_muli_op_to_sentient`] must NOT be called from here.
/// ⛔ AND SO IS THE TRAILING `arith.constant` (`:467-470`, `op->dump(); signalPassFailure();`): an
/// index constant went to `:457` and an integer one to `:465`, so what reaches it is a constant of
/// some OTHER type. `arith.divsi`, `arith.remsi` and `arith.not` have no arm at all.
/// ⛔ THE `arith.ori` WALK RUNS TO COMPLETION FIRST (`:441-445`) — entry 339 folds an `ori` chain
/// into one op, so lowering an `ori` before that lowers a shape entry 363 never sees.
pub fn run_on_operation<A: Arch>(
    program: &mut dfir::Program<A>,
    values: &mut dfir::Values,
) -> Vec<SenOp> {
    // `:441-445` — the first walk, `SimplifyOrIOp` on every `arith.ori`.
    simplify_every_ori(&mut program.preamble, values);
    for unit in program.units.iter_mut() {
        simplify_every_ori(&mut unit.body, values);
    }

    // `:447-471` — the second walk, over the preamble and then every unit body.
    let mut out: Vec<SenOp> = Vec::new();
    lower_every_arith(&mut program.preamble, values, &mut out);
    for unit in program.units.iter_mut() {
        lower_every_arith(&mut unit.body, values, &mut out);
    }
    out
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::sentient::dialects::Op as SenOp;
    use crate::islands::sentient::print;

    /// THE TEXT ONE SENTIENT OP PRINTS AS, trimmed.
    fn printed(op: &SenOp) -> String {
        let mut out = String::new();
        print::emit(&mut out, op, 0);
        out.trim().to_owned()
    }

    /// 🎯 049/384 — AN `arith.addi` BECOMES A `sentient.scalar_add` OVER THE SAME TWO OPERANDS,
    /// BINDING THE SAME VALUE.
    ///
    /// The input is `%15 = arith.addi %arg4, %c2048 : index` and the reference's output for it is
    /// `%29 = sentient.scalar_add %18, %1 : index, index`
    /// (`dcc/test/Conversion/StandardToSentient/cmpi_select_different_BB.mlir:58`, whose RUN line is
    /// `--dcc-standard-to-sentient` alone).
    #[test]
    fn an_addi_becomes_a_scalar_add() {
        let addi = IntBinary {
            result: Val(29),
            lhs: Val(18),
            rhs: Val(1),
            ty: ScalarTy::Index,
        };
        let lowered = lower_addi_op_to_sentient(&addi);
        assert_eq!(
            lowered,
            sen::Op::ScalarAdd {
                lhs: Val(18),
                rhs: Val(1),
                result: Val(29),
                reg: None,
                ty: ScalarTy::Index,
                element_size: None,
            },
            "the operands and the bound value carry through unchanged"
        );
        assert_eq!(
            printed(&SenOp::Sentient(lowered)),
            "%29 = sentient.scalar_add %18, %1 : index, index"
        );
    }

    /// 🎯 049/384 — AND A FRESHLY LOWERED ADD PRINTS **NO ATTRIBUTE DICTIONARY**.
    ///
    /// ⛔ NOT `{regLocale = #sentient<reg_type unknown>}`. The five-argument `AddOp::create` sets
    /// neither register attribute, and the reference's output for this pass shows the bare form. An
    /// allocator later writes both, and only then does the dictionary appear:
    ///
    /// ```text
    /// %[[VAL_20]] = sentient.scalar_add %[[VAL_17]], %[[VAL_1]] {element_size = 8 : i32, regIndex = 1 : i32, regLocale = #sentient<reg_type lrf>} : index, index
    /// ```
    ///
    /// (`dcc/test/LXLU/rotate-composite.mlir:24`, a `CHECK-SENT-IR` line — one of only three places
    /// in the authority's test tree where a `scalar_add` carries a register at all.)
    ///
    /// ⚠️ `element_size` THERE IS A DISCARDABLE ATTRIBUTE, not one of the four the `.td` declares
    /// (`SentientOps.td:700-713`), and this island does not model it. Whichever unit sets it owns
    /// that; no lowering in this file does.
    #[test]
    fn a_lowered_add_carries_no_register() {
        let lowered = lower_addi_op_to_sentient(&IntBinary {
            result: Val(2),
            lhs: Val(0),
            rhs: Val(1),
            ty: ScalarTy::Index,
        });
        assert!(
            !printed(&SenOp::Sentient(lowered)).contains('{'),
            "a lowering leaves the register unsaid, and unsaid prints as nothing"
        );
        // ⭐ AND THE SAME OP WITH A REGISTER PRINTS IT THE WAY THE REFERENCE DOES.
        assert_eq!(
            printed(&SenOp::Sentient(sen::Op::ScalarAdd {
                lhs: Val(17),
                rhs: Val(1),
                result: Val(20),
                reg: Some(sen::Reg {
                    locale: sen::RegType::Lrf,
                    index: Some(sen::RegIndex::at::<1>()),
                }),
                ty: ScalarTy::Index,
                element_size: None,
            })),
            "%20 = sentient.scalar_add %17, %1 {regIndex = 1 : i32, regLocale = #sentient<reg_type lrf>} : index, index"
        );
    }

    /// 🎯 050/384 — AN `arith.subi` BECOMES A `sentient.scalar_sub`, LEFT OPERAND STILL LEFT.
    ///
    /// `%3 = arith.subi %c3, %arg0 : index` lowers to
    /// `%12 = sentient.scalar_sub %5, %11 : index, index` (`cmpi_select_different_BB.mlir:19`).
    #[test]
    fn a_subi_becomes_a_scalar_sub() {
        let lowered = lower_subi_op_to_sentient(&IntBinary {
            result: Val(12),
            lhs: Val(5),
            rhs: Val(11),
            ty: ScalarTy::Index,
        });
        assert_eq!(
            printed(&SenOp::Sentient(lowered)),
            "%12 = sentient.scalar_sub %5, %11 : index, index"
        );
    }

    /// 🎯 051/384 — AN `arith.muli` BECOMES A `sentient.scalar_mul`, AND THE ATTRIBUTE COPY
    /// TRANSFERS NOTHING.
    ///
    /// ⛔ NO GOLDEN EXISTS FOR THIS ONE: `sentient.scalar_mul` appears in none of the authority's 825
    /// test cases and `arith.muli` in none of their inputs. What is asserted is what the three lines
    /// of the function say — the mnemonic, the operand order, the type twice, and an empty dictionary
    /// after `setAttrs(getAttrDictionary())`.
    #[test]
    fn a_muli_becomes_a_scalar_mul_with_no_attributes() {
        let lowered = lower_muli_op_to_sentient(&IntBinary {
            result: Val(7),
            lhs: Val(3),
            rhs: Val(4),
            ty: ScalarTy::Index,
        });
        assert_eq!(
            lowered,
            sen::Op::ScalarMul {
                lhs: Val(3),
                rhs: Val(4),
                result: Val(7),
                reg_locale: None,
                ty: ScalarTy::Index,
            }
        );
        assert_eq!(
            printed(&SenOp::Sentient(lowered)),
            "%7 = sentient.scalar_mul %3, %4 : index, index"
        );
    }

    /// 🎯 049,050,051/384 — AND THE TYPE IS THE OPERANDS', NOT `index` BY ASSUMPTION.
    ///
    /// ⛔ `SameOperandsAndResultType` MEANS ONE TYPE PRINTED TWICE. An `i32` add prints `: i32, i32`;
    /// hardcoding `index` here would emit a program whose scalar arithmetic disagrees with its own
    /// operands.
    #[test]
    fn the_scalar_ops_carry_their_operands_type() {
        let wide = IntBinary {
            result: Val(3),
            lhs: Val(1),
            rhs: Val(2),
            ty: ScalarTy::Int(32),
        };
        assert_eq!(
            printed(&SenOp::Sentient(lower_addi_op_to_sentient(&wide))),
            "%3 = sentient.scalar_add %1, %2 : i32, i32"
        );
        assert_eq!(
            printed(&SenOp::Sentient(lower_subi_op_to_sentient(&wide))),
            "%3 = sentient.scalar_sub %1, %2 : i32, i32"
        );
        assert_eq!(
            printed(&SenOp::Sentient(lower_muli_op_to_sentient(&wide))),
            "%3 = sentient.scalar_mul %1, %2 : i32, i32"
        );
    }

    /// 🎯 052/384 — A CONJUNCTION NESTS THROUGH THE `then` ARM, AND THE `else` ARMS ALL YIELD FALSE.
    ///
    /// ⛔ THE ASSERTION IS STRUCTURAL, not textual — see [`NestedIf`]'s note on the island's printed
    /// register arrays. What it pins is the shape the line states: the outer `if` is the FIRST
    /// conjunct, its `then` arm holds the inner `if` **and** a `yield` of the inner `if`'s result,
    /// the innermost `then` yields the true value, and both `else` arms yield the false one.
    #[test]
    fn a_conjunction_nests_through_the_then_arm() {
        let outer = Conjunct {
            predicate: sen::CmpPredicate::Eq,
            lhs: Val(12),
            rhs: Val(8),
            result: Val(22),
            dbg_name: Some("IfOp #1".to_owned()),
        };
        let inner = Conjunct {
            predicate: sen::CmpPredicate::Eq,
            lhs: Val(12),
            rhs: Val(7),
            result: Val(23),
            dbg_name: Some("IfOp #1".to_owned()),
        };
        let nest = NestedIf {
            conjuncts: vec![outer.clone(), inner.clone()],
            true_value: Val(20),
            false_value: Val(19),
            ty: ScalarTy::Index,
        };
        let unknown = sen::Reg {
            locale: sen::RegType::Unknown,
            index: None,
        };
        let expected = SenOp::Sentient(sen::Op::If {
            predicate: sen::CmpPredicate::Eq,
            lhs: Val(12),
            rhs: Val(8),
            yielded: vec![sen::Yielded {
                result: Val(22),
                reg: unknown,
                element_size: None,
            }],
            dbg_name: Some("IfOp #1".to_owned()),
            then_body: vec![
                SenOp::Sentient(sen::Op::If {
                    predicate: sen::CmpPredicate::Eq,
                    lhs: Val(12),
                    rhs: Val(7),
                    yielded: vec![sen::Yielded {
                        result: Val(23),
                        reg: unknown,
                        element_size: None,
                    }],
                    dbg_name: Some("IfOp #1".to_owned()),
                    then_body: vec![SenOp::Sentient(sen::Op::Yield {
                        results: vec![Val(20)],
                    })],
                    else_body: vec![SenOp::Sentient(sen::Op::Yield {
                        results: vec![Val(19)],
                    })],
                }),
                SenOp::Sentient(sen::Op::Yield {
                    results: vec![Val(23)],
                }),
            ],
            else_body: vec![SenOp::Sentient(sen::Op::Yield {
                results: vec![Val(19)],
            })],
        });
        assert_eq!(nest.into_op(), vec![expected]);
    }

    /// 🎯 052/384 — AND ONE CONJUNCT IS THE BASE CASE'S OWN `if`.
    ///
    /// ⛔ NOT A DEGENERATE ARM TO REFUSE. `ConstructIFRecursively` builds exactly this for a bare
    /// `arith.cmpi` (`:118-145`), so the nesting rule at depth one must agree with it.
    #[test]
    fn one_conjunct_is_a_single_if() {
        let nest = NestedIf {
            conjuncts: vec![Conjunct {
                predicate: sen::CmpPredicate::Ne,
                lhs: Val(1),
                rhs: Val(2),
                result: Val(3),
                dbg_name: None,
            }],
            true_value: Val(4),
            false_value: Val(5),
            ty: ScalarTy::Index,
        };
        assert_eq!(
            nest.into_op(),
            vec![SenOp::Sentient(sen::Op::If {
                predicate: sen::CmpPredicate::Ne,
                lhs: Val(1),
                rhs: Val(2),
                yielded: vec![sen::Yielded {
                    result: Val(3),
                    reg: sen::Reg {
                        locale: sen::RegType::Unknown,
                        index: None,
                    },
                    element_size: None,
                }],
                dbg_name: None,
                then_body: vec![SenOp::Sentient(sen::Op::Yield {
                    results: vec![Val(4)],
                })],
                else_body: vec![SenOp::Sentient(sen::Op::Yield {
                    results: vec![Val(5)],
                })],
            })]
        );
    }

    /// 🎯 053/384 — AN `arith.constant N : index` BECOMES A `sentient.scalar_constant` OF THE SAME
    /// VALUE, TYPED `index`.
    ///
    /// Every one of the nine `arith.constant`s in `cmpi_select_different_BB.mlir` lowers this way;
    /// `%c32768 = arith.constant 32768 : index` becomes
    /// `%0 = sentient.scalar_constant {value = 32768 : si64} : index` (`:6`).
    #[test]
    fn an_index_constant_becomes_a_scalar_constant() {
        assert_eq!(
            printed(&SenOp::Sentient(lower_constant_index_to_sentient(
                Val(0),
                32768
            ))),
            "%0 = sentient.scalar_constant {value = 32768 : si64} : index"
        );
    }

    /// 🎯 053/384 — AND THE LOCALE IT CARRIES IS `imm`, THE `.td`'S DEFAULT FOR THIS OP.
    ///
    /// ⛔ IT IS NOT PRINTED AND IT IS STILL NOT `unknown`. `ConstantOp::print` writes the value and
    /// the type only (`SentientOps.cpp:1698-1715`), so this is asserted on the op rather than on its
    /// text — the passes that spill a constant into a register read the attribute, not the output.
    #[test]
    fn a_lowered_constant_is_an_immediate() {
        assert_eq!(
            lower_constant_index_to_sentient(Val(0), 7),
            sen::Op::ScalarConstant {
                is_symbol: false,
                value: 7,
                result: Val(0),
                reg_locale: sen::RegType::Imm,
                ty: ScalarTy::Index,
            }
        );
    }

    /// 🎯 054/384 — AN `arith.constant false` BECOMES `{value = 0 : si64} : i1`.
    ///
    /// `dcc/test/Conversion/SentientToProgIR/simplify_or_op.mlir` feeds `%false = arith.constant
    /// false` to `--dcc-standard-to-sentient` and the output is
    /// `%2 = sentient.scalar_constant {value = 0 : si64} : i1` (`:8`).
    #[test]
    fn a_false_constant_becomes_an_i1_zero() {
        assert_eq!(
            printed(&SenOp::Sentient(lower_constant_int_to_sentient(
                Val(2),
                IntConst::Bool(false)
            ))),
            "%2 = sentient.scalar_constant {value = 0 : si64} : i1"
        );
    }

    /// 🎯 054/384 — ⛔⛔ AND AN `arith.constant true` BECOMES **ONE**, NOT MINUS ONE.
    ///
    /// This is the whole reason the reference reads an `i1` through `BoolAttr::getValue()` instead of
    /// `ConstantIntOp::value()`: the latter sign-extends a signless one-bit integer, so `true` would
    /// arrive as -1 and the emitted constant would read `{value = -1 : si64} : i1`. Reference
    /// SentientIR spells it `{value = 1 : si64} : i1`
    /// (`dcc/test/PE/conditional-2and3-nested-if.mlir:38`).
    #[test]
    fn a_true_constant_becomes_an_i1_one_and_not_a_minus_one() {
        let lowered = lower_constant_int_to_sentient(Val(2), IntConst::Bool(true));
        assert_eq!(
            printed(&SenOp::Sentient(lowered.clone())),
            "%2 = sentient.scalar_constant {value = 1 : si64} : i1"
        );
        assert_ne!(
            lowered,
            sen::Op::ScalarConstant {
                is_symbol: false,
                value: -1,
                result: Val(2),
                reg_locale: sen::RegType::Imm,
                ty: ScalarTy::Int(1),
            },
            "an i1 true read as a sign-extended integer is -1, and that is the value this lowering \
             exists to avoid"
        );
    }

    /// 🎯 054/384 — AND A WIDER INTEGER KEEPS ITS VALUE, ITS SIGN AND ITS WIDTH.
    ///
    /// ⛔ THE ARM THE REFERENCE CANNOT REACH. `mlir::cast<BoolAttr>` aborts on an `i32` attribute, so
    /// the reference has no behaviour here to match; what it has is the `else` branch it wrote for
    /// this case — `val = const_int_op.value()`, the `APInt` read signed.
    #[test]
    fn a_wide_integer_constant_keeps_its_width() {
        assert_eq!(
            printed(&SenOp::Sentient(lower_constant_int_to_sentient(
                Val(4),
                IntConst::Int {
                    value: -3,
                    bits: 32,
                }
            ))),
            "%4 = sentient.scalar_constant {value = -3 : si64} : i32"
        );
    }

    /// 🎯 053,054/384 — THE TWO CONSTANT LOWERINGS ARE TWO BECAUSE THE TYPES DIFFER, and one
    /// function's output sits beside the other's in one program.
    ///
    /// `simplify_or_op.mlir:6-8` shows `{value = 0 : si64} : index` and `{value = 0 : si64} : i1`
    /// three lines apart — same value, different type, different op class on the way in.
    #[test]
    fn the_two_constant_lowerings_differ_only_in_the_type() {
        let index = lower_constant_index_to_sentient(Val(0), 0);
        let boolean = lower_constant_int_to_sentient(Val(2), IntConst::Bool(false));
        assert_eq!(
            printed(&SenOp::Sentient(index)),
            "%0 = sentient.scalar_constant {value = 0 : si64} : index"
        );
        assert_eq!(
            printed(&SenOp::Sentient(boolean)),
            "%2 = sentient.scalar_constant {value = 0 : si64} : i1"
        );
    }

    /// 🎯 049-054/384 — AND THE INPUT SIDE PRINTS AS THE REFERENCE'S INPUTS ARE WRITTEN.
    ///
    /// ⛔ THE ISLAND HAD NO `arith.addi`, `arith.subi`, `arith.muli` OR NON-INDEX CONSTANT before
    /// these six lowerings; a lowering whose input cannot be spelled is not a lowering. These are
    /// the forms `cmpi_select_different_BB.mlir` and `simplify_or_op.mlir` feed the pass.
    #[test]
    fn the_arith_inputs_print() {
        use crate::islands::dataflow_ir::dialects::arith;
        use crate::islands::dataflow_ir::dialects::{Op as DfirOp, arith::Op as ArithOp};
        use crate::islands::dataflow_ir::print as dfir_print;

        let mut out = String::new();
        let binary = arith::IntBinary {
            result: Val(15),
            lhs: Val(14),
            rhs: Val(1),
            ty: ScalarTy::Index,
        };
        dfir_print::emit(&mut out, &DfirOp::Arith(ArithOp::AddI(binary)), 0);
        assert_eq!(out.trim(), "%15 = arith.addi %14, %1 : index");
        out.clear();
        dfir_print::emit(&mut out, &DfirOp::Arith(ArithOp::SubI(binary)), 0);
        assert_eq!(out.trim(), "%15 = arith.subi %14, %1 : index");
        out.clear();
        dfir_print::emit(&mut out, &DfirOp::Arith(ArithOp::MulI(binary)), 0);
        assert_eq!(out.trim(), "%15 = arith.muli %14, %1 : index");
        out.clear();
        // ⭐ `arith.constant false`, WITH NO TYPE — the pretty form MLIR prints for an `i1`.
        dfir_print::emit(
            &mut out,
            &DfirOp::Arith(ArithOp::ConstantInt {
                result: Val(2),
                value: IntConst::Bool(false),
            }),
            0,
        );
        assert_eq!(out.trim(), "%2 = arith.constant false");
        out.clear();
        dfir_print::emit(
            &mut out,
            &DfirOp::Arith(ArithOp::ConstantInt {
                result: Val(3),
                value: IntConst::Int {
                    value: -3,
                    bits: 32,
                },
            }),
            0,
        );
        assert_eq!(out.trim(), "%3 = arith.constant -3 : i32");
    }

    /// 🎯 048/384 — THE SIX SIGNED PREDICATES CROSS THE RUNG UNCHANGED, AND KEEP THEIR SPELLING.
    ///
    /// The input `%16 = arith.cmpi slt, %15, %c4096 : index` lowers to a `sentient.if` printing
    /// `predicate = slt` (`dcc/test/Conversion/StandardToSentient/cmpi_select_different_BB.mlir`), so
    /// a map that permuted two predicates would still be total, still exhaustive, and would invert a
    /// branch.
    #[test]
    fn the_six_signed_predicates_cross_unchanged() {
        for (arith_pred, sen_pred) in [
            (CmpIPredicate::Eq, sen::CmpPredicate::Eq),
            (CmpIPredicate::Ne, sen::CmpPredicate::Ne),
            (CmpIPredicate::Slt, sen::CmpPredicate::Slt),
            (CmpIPredicate::Sle, sen::CmpPredicate::Sle),
            (CmpIPredicate::Sgt, sen::CmpPredicate::Sgt),
            (CmpIPredicate::Sge, sen::CmpPredicate::Sge),
        ] {
            assert_eq!(get_sentient_cmp_i_predicate(arith_pred), sen_pred);
            assert_eq!(
                get_sentient_cmp_i_predicate(arith_pred).spelling(),
                arith_pred.spelling(),
                "the two rungs spell {arith_pred:?} the same way"
            );
        }
    }

    /// 🎯 048/384 — AND `eq` IS THE ANSWER TO `eq` ONLY.
    ///
    /// ⛔⛔ THE FALL-THROUGH `return ...::eq;` IS NOT A DEFAULT, and this is what it would look like
    /// if it had been ported as one: five of the six predicates answering `eq`. The line exists to
    /// silence a warning after an unreachable `DT_CHECK(0)`; see [`get_sentient_cmp_i_predicate`].
    #[test]
    fn eq_is_the_answer_to_eq_alone() {
        let eq_answers: Vec<CmpIPredicate> = [
            CmpIPredicate::Eq,
            CmpIPredicate::Ne,
            CmpIPredicate::Slt,
            CmpIPredicate::Sle,
            CmpIPredicate::Sgt,
            CmpIPredicate::Sge,
        ]
        .into_iter()
        .filter(|pred| get_sentient_cmp_i_predicate(*pred) == sen::CmpPredicate::Eq)
        .collect();
        assert_eq!(eq_answers, vec![CmpIPredicate::Eq]);
    }

    /// 🎯 048/384 + 052/384 — AND THE MAPPED PREDICATE IS THE ONE THE EMITTED `sentient.if` CARRIES.
    ///
    /// ⭐ THIS IS THE SEAM THE FUNCTION EXISTS FOR. `ConstructIFRecursively`'s base case
    /// (`StandardToSentient.cpp:126-132`) reads the `arith.cmpi`'s predicate through here and wraps
    /// the answer in a `CmpIPredicateAttr` on the `sentient.if` it builds, so the mapped value is
    /// observable in the emitted op and not just in a local. `sge` is chosen because it is the one
    /// predicate a `select` lowering in the reference's own test actually carries
    /// (`cmpi_select_different_BB.mlir`) and the one an `eq`-only island could never have produced.
    #[test]
    fn the_mapped_predicate_reaches_the_emitted_if() {
        let nest = NestedIf {
            conjuncts: vec![Conjunct {
                predicate: get_sentient_cmp_i_predicate(CmpIPredicate::Sge),
                lhs: Val(18),
                rhs: Val(1),
                result: Val(30),
                dbg_name: None,
            }],
            true_value: Val(20),
            false_value: Val(19),
            ty: ScalarTy::Index,
        };
        assert_eq!(
            nest.into_op(),
            vec![SenOp::Sentient(sen::Op::If {
                predicate: sen::CmpPredicate::Sge,
                lhs: Val(18),
                rhs: Val(1),
                yielded: vec![sen::Yielded {
                    result: Val(30),
                    reg: sen::Reg {
                        locale: sen::RegType::Unknown,
                        index: None,
                    },
                    element_size: None,
                }],
                dbg_name: None,
                then_body: vec![SenOp::Sentient(sen::Op::Yield {
                    results: vec![Val(20)],
                })],
                else_body: vec![SenOp::Sentient(sen::Op::Yield {
                    results: vec![Val(19)],
                })],
            })]
        );
    }

    /// 🎯 338/384 — THE VENDOR'S OWN CASE: A `cmpi` CONDITION IS ONE `sentient.if`.
    ///
    /// `dcc/test/Conversion/StandardToSentient/cmpi_select_different_BB.mlir` selects between two
    /// `index` values on `%9 = arith.cmpi eq, %3, %c1` and expects
    /// `sentient.if eq, %[[VAL_12]], %[[VAL_7]] : index -> (index) {regIndices = [], regLocales =
    /// [#sentient<reg_type unknown>]}` yielding the two arms — one `if`, no constants, and the `cmpi`
    /// queued for erasure.
    #[test]
    fn a_cmpi_condition_becomes_one_sentient_if() {
        let cmpi = DfirOp::Arith(arith::Op::Compare {
            result: Val(9),
            predicate: CmpIPredicate::Eq,
            lhs: Val(3),
            rhs: Val(1),
        });
        let select = DfirOp::Arith(arith::Op::Select {
            result: Val(16),
            condition: Val(9),
            true_value: Val(8),
            false_value: Val(7),
            ty: ScalarTy::Index,
            dbg_name: None,
        });
        let scope = vec![cmpi.clone(), select.clone()];
        let mut values = dfir::Values::default();
        let mut ops_to_be_erased: Vec<Val> = Vec::new();
        let nest = construct_if_recursively(
            &select,
            &cmpi,
            Val(8),
            Val(7),
            &scope,
            &mut values,
            &mut ops_to_be_erased,
        );
        assert_eq!(
            nest,
            Some(Nest {
                constants: Vec::new(),
                op: SenOp::Sentient(sen::Op::If {
                    predicate: sen::CmpPredicate::Eq,
                    lhs: Val(3),
                    rhs: Val(1),
                    yielded: vec![sen::Yielded {
                        result: Val(0),
                        reg: sen::Reg {
                            locale: sen::RegType::Unknown,
                            index: None,
                        },
                        element_size: None,
                    }],
                    dbg_name: None,
                    then_body: vec![SenOp::Sentient(sen::Op::Yield {
                        results: vec![Val(8)],
                    })],
                    else_body: vec![SenOp::Sentient(sen::Op::Yield {
                        results: vec![Val(7)],
                    })],
                }),
                result: Val(0),
            })
        );
        assert_eq!(ops_to_be_erased, vec![Val(9)]);
    }

    /// 🎯 339/384 — THE VENDOR'S OWN CASE: `or(eq, and(ne, %5))` COLLAPSES TO `or(eq, %5)`.
    ///
    /// `dcc/test/Conversion/SentientToProgIR/simplify_or_op.mlir` feeds `%13 = cmpi eq, %3, %c1`,
    /// `%14 = cmpi ne, %3, %c1`, `%15 = andi %14, %5`, `%16 = ori %13, %15` and its expected output has
    /// the `andi` and the `ne` GONE with the `ori` reading `%13` and `%5` — the `eq` survives.
    #[test]
    fn an_or_over_a_negated_and_loses_the_and_and_the_negation() {
        let mut scope = vec![
            DfirOp::Arith(arith::Op::Compare {
                result: Val(13),
                predicate: CmpIPredicate::Eq,
                lhs: Val(3),
                rhs: Val(1),
            }),
            DfirOp::Arith(arith::Op::Compare {
                result: Val(14),
                predicate: CmpIPredicate::Ne,
                lhs: Val(3),
                rhs: Val(1),
            }),
            DfirOp::Arith(arith::Op::Logic {
                result: Val(15),
                kind: LogicKind::And,
                operands: vec![Val(14), Val(5)],
            }),
            DfirOp::Arith(arith::Op::Logic {
                result: Val(16),
                kind: LogicKind::Or,
                operands: vec![Val(13), Val(15)],
            }),
        ];
        let mut values = dfir::Values::default();
        simplify_or_i_op(Val(16), &mut scope, &mut values);
        assert_eq!(
            scope,
            vec![
                DfirOp::Arith(arith::Op::Compare {
                    result: Val(13),
                    predicate: CmpIPredicate::Eq,
                    lhs: Val(3),
                    rhs: Val(1),
                }),
                DfirOp::Arith(arith::Op::Logic {
                    result: Val(0),
                    kind: LogicKind::Or,
                    operands: vec![Val(13), Val(5)],
                }),
            ]
        );
    }

    /// A program whose one unit holds `body` — this pass reads no machine fact.
    fn program_of(body: Vec<DfirOp>) -> dfir::Program<crate::arch::Target> {
        use crate::generated::OpFunc;
        use crate::islands::dataflow_ir::{
            Grid, GroupId, OpIndex, ProgramName, ProgramUnit, ProgramUnits, Units,
        };
        use crate::units::DfirUnit;
        dfir::Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            grid: Grid::single(),
            preamble: Vec::new(),
            units: ProgramUnits::of(
                ProgramUnit {
                    on: Units::one(DfirUnit::Lxlu, Val(0)),
                    precision: None,
                    body,
                    arch: core::marker::PhantomData,
                },
                Vec::new(),
            ),
            arch: core::marker::PhantomData,
        }
    }

    /// 🎯 376/384 — THE FOUR CLASSES THE WALK CAN LOWER, IN THE ORDER IT MEETS THEM.
    ///
    /// The dispatch is `:447-471`; the expectations are the three ported lowerings' own emissions
    /// (`sentient.scalar_add`, `sentient.scalar_sub`, two `sentient.scalar_constant`s), written out
    /// rather than recomputed so a change in either side shows up here.
    #[test]
    fn the_walk_lowers_the_classes_the_reference_has_an_arm_for() {
        let mut program = program_of(vec![
            DfirOp::Arith(arith::Op::AddI(IntBinary {
                result: Val(29),
                lhs: Val(18),
                rhs: Val(1),
                ty: ScalarTy::Index,
            })),
            DfirOp::Arith(arith::Op::Constant {
                result: Val(1),
                value: 2048,
            }),
            DfirOp::Arith(arith::Op::ConstantInt {
                result: Val(2),
                value: IntConst::Bool(true),
            }),
            // ⭐ NO ARM: left exactly as it is, and it must not add an op to the output.
            DfirOp::Arith(arith::Op::DivSI(IntBinary {
                result: Val(3),
                lhs: Val(1),
                rhs: Val(2),
                ty: ScalarTy::Int(32),
            })),
        ]);
        assert_eq!(
            run_on_operation(&mut program, &mut dfir::Values::default()),
            vec![
                SenOp::Sentient(sen::Op::ScalarAdd {
                    lhs: Val(18),
                    rhs: Val(1),
                    result: Val(29),
                    reg: None,
                    ty: ScalarTy::Index,
                    element_size: None,
                }),
                SenOp::Sentient(sen::Op::ScalarConstant {
                    is_symbol: false,
                    value: 2048,
                    result: Val(1),
                    reg_locale: sen::RegType::Imm,
                    ty: ScalarTy::Index,
                }),
                SenOp::Sentient(sen::Op::ScalarConstant {
                    is_symbol: false,
                    value: 1,
                    result: Val(2),
                    reg_locale: sen::RegType::Imm,
                    ty: ScalarTy::Int(1),
                }),
            ]
        );
    }

    /// 🎯⛔ 376/384 — AND AN `arith.muli` STOPS THE BUILD, BECAUSE THE REFERENCE REFUSES IT.
    ///
    /// `:451-454` emits an error and signals pass failure with `LowerMulIOpToSentient(op)` commented
    /// out beneath it. Entry 051 IS ported, so the trap this test guards is a future edit "completing"
    /// the dispatch by calling it: that would emit a `sentient.scalar_mul` the reference never emits.
    #[test]
    #[should_panic(expected = "Cannot lower mul expressions to sentient")]
    fn a_muli_is_refused_rather_than_lowered() {
        run_on_operation(
            &mut program_of(vec![DfirOp::Arith(arith::Op::MulI(IntBinary {
                result: Val(4),
                lhs: Val(1),
                rhs: Val(2),
                ty: ScalarTy::Index,
            }))]),
            &mut dfir::Values::default(),
        );
    }
    /// 🎯 362/384 — THE VENDOR'S OWN CASE: A SELECT ON A `cmpi` BECOMES ONE `sentient.if` AND BOTH
    /// `arith` OPS LEAVE THE BLOCK.
    ///
    /// `cmpi_select_different_BB.mlir:104` feeds `%12 = arith.cmpi eq, %3, %c2` and
    /// `%17 = arith.select %12, %c0, %c131072`; the expectation is
    /// `sentient.if eq, %[[VAL_12]], %[[VAL_6]] : index -> (index)` yielding `%[[VAL_8]]` then
    /// `%[[VAL_4]]` (`:46-51`), with neither the compare nor the select left anywhere in the output.
    #[test]
    fn a_select_over_a_compare_becomes_one_if_and_empties_the_block() {
        let mut scope = vec![
            DfirOp::Arith(arith::Op::Compare {
                result: Val(12),
                predicate: CmpIPredicate::Eq,
                lhs: Val(3),
                rhs: Val(2),
            }),
            DfirOp::Arith(arith::Op::Select {
                result: Val(17),
                condition: Val(12),
                true_value: Val(30),
                false_value: Val(31),
                ty: ScalarTy::Index,
                dbg_name: None,
            }),
        ];
        let select = scope[1].clone();
        let mut values = dfir::Values::default();
        let nest = lower_select_op_to_sentient(&select, &mut scope, &mut values);
        assert_eq!(
            nest,
            Some(Nest {
                constants: Vec::new(),
                op: SenOp::Sentient(sen::Op::If {
                    predicate: sen::CmpPredicate::Eq,
                    lhs: Val(3),
                    rhs: Val(2),
                    yielded: vec![unknown_reg(Val(0))],
                    dbg_name: None,
                    then_body: vec![yield_of(Val(30))],
                    else_body: vec![yield_of(Val(31))],
                }),
                result: Val(0),
            })
        );
        assert_eq!(scope, Vec::new());
    }

    /// 🎯 363/384 — THE VENDOR'S OWN CASE: A CONNECTIVE OVER TWO ALREADY-LOWERED CONDITIONS IS TWO
    /// FRESH `i1` CONSTANTS, THE TWO THE ONE-RESULT ARM MINTS, AND ONE GRAFTED NEST.
    ///
    /// `simplify_or_op.mlir:243` reaches this pass as `%16 = arith.ori %13, %5` with `%13` already a
    /// `sentient.if` result. Its expectation is four `i1` constants — `1`, `0`, `1`, `1` — then
    /// `%[[VAL_25]] = sentient.if eq, %[[VAL_13]], %[[VAL_24]] : i1` yielding `%[[VAL_21]]` and, in
    /// its else arm, `%[[VAL_26]] = sentient.if eq, %[[VAL_20]], %[[VAL_23]] : i1` (`:52-70`).
    /// ⛔ THE DISJUNCTION GRAFTS ONTO THE **FALSE** ARM; a conjunction would graft onto the true one.
    #[test]
    fn a_connective_over_lowered_conditions_grafts_onto_the_false_arm() {
        let mut scope = vec![DfirOp::Arith(arith::Op::Logic {
            result: Val(16),
            kind: LogicKind::Or,
            operands: vec![Val(20), Val(13)],
        })];
        let ori = scope[0].clone();
        let mut values = dfir::Values::default();
        let nest = lower_logical_op_to_sentient(&ori, &mut scope, &mut values);
        assert_eq!(
            nest,
            Some(Nest {
                constants: vec![
                    i1_constant(1, Val(0)),
                    i1_constant(0, Val(1)),
                    i1_constant(1, Val(2)),
                    i1_constant(1, Val(4)),
                ],
                op: SenOp::Sentient(sen::Op::If {
                    predicate: sen::CmpPredicate::Eq,
                    lhs: Val(13),
                    rhs: Val(4),
                    yielded: vec![unknown_reg(Val(5))],
                    dbg_name: None,
                    then_body: vec![yield_of(Val(0))],
                    else_body: vec![
                        SenOp::Sentient(sen::Op::If {
                            predicate: sen::CmpPredicate::Eq,
                            lhs: Val(20),
                            rhs: Val(2),
                            yielded: vec![unknown_reg(Val(6))],
                            dbg_name: None,
                            then_body: vec![yield_of(Val(0))],
                            else_body: vec![yield_of(Val(1))],
                        }),
                        yield_of(Val(6)),
                    ],
                }),
                result: Val(5),
            })
        );
        assert_eq!(scope, Vec::new());
    }

    /// One `sentient.if` result carried in an unknown register — `regLocales = [unknown]`.
    fn unknown_reg(result: Val) -> sen::Yielded {
        sen::Yielded {
            result,
            reg: sen::Reg {
                locale: sen::RegType::Unknown,
                index: None,
            },
            element_size: None,
        }
    }

    /// One `sentient.yield` of a single value.
    fn yield_of(val: Val) -> SenOp {
        SenOp::Sentient(sen::Op::Yield { results: vec![val] })
    }
}
