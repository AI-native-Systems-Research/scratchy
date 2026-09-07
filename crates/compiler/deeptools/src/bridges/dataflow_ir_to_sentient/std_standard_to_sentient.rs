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

use crate::islands::dataflow_ir::dialects::Val;
use crate::islands::dataflow_ir::dialects::arith::{IntBinary, IntConst};
use crate::islands::dataflow_ir::ty::ScalarTy;
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
/// straight through (`StandardToSentient.cpp:126-133`); by the time the nesting rule below applies,
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
    #[must_use]
    pub fn into_op(&self) -> Vec<super::SenOp> {
        // ⭐ BUILT INSIDE OUT, so each level already has the result its parent must yield.
        let mut nest: Vec<super::SenOp> = vec![super::SenOp::Sentient(sen::Op::Yield {
            results: vec![self.true_value],
        })];
        for conjunct in self.conjuncts.iter().rev() {
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
                    }],
                    dbg_name: conjunct.dbg_name.clone(),
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
        value: literal,
        result,
        reg_locale: sen::RegType::Imm,
        ty: value.ty(),
    }
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
}

