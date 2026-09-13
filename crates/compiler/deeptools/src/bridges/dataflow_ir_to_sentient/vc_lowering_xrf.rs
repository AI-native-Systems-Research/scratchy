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

//! `LoweringXRF.cpp` — 14 of bridge 2's 384 functions (dependency level(s) [0, 1, 2, 6, 7]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e090_getXrfValue` | 090/384 | 11 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:248` |
//! | `e091_getForOpBound` | 091/384 | 31 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:262` |
//! | `e092_setSentientMacXrfRegIncrAttr` | 092/384 | 11 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:665` |
//! | `e093_replaceAndEraseDummyMacOps` | 093/384 | 7 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:681` |
//! | `e172_areXrfAccessesLegal` | 172/384 | 28 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:98` |
//! | `e173_insertConstAndAddOps` | 173/384 | 10 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:296` |
//! | `e174_isXrfRelated` | 174/384 | 31 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:531` |
//! | `e175_updateYieldArgs` | 175/384 | 8 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:690` |
//! | `e240_getLayoutExpr` | 240/384 | 67 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:28` |
//! | `e241_createForOpWithReturnValue` | 241/384 | 56 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:129` |
//! | `e242_createIfOpWithReturnValue` | 242/384 | 56 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:189` |
//! | `e243_insertDummyMacOp` | 243/384 | 15 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:312` |
//! | `e345_processXrfPtrPerUnit` | 345/384 | 190 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:337` |
//! | `e367_createXrfIndexModifOps` | 367/384 | 97 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:564` |

use core::num::NonZeroU32;
use std::collections::{BTreeMap, BTreeSet};

use super::tf_mutable_start_addr_shifting::ElementsPerStick;
use super::vc_vector_operands::{OpId, layout_map_and_indices, op_at};
use crate::arch::{Arch, IsaGen};
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::dialects::dataflow::LocalUnit;
use crate::islands::dataflow_ir::dialects::{Op as DfirOp, agen, dataflow, defining_op, vector};
use crate::islands::dataflow_ir::ty::{ScalarTy, Vector};
use crate::islands::sentient::dialects::{self as sen, Definitions, Val, arith, sentient, symbol};
use crate::units::DfirUnit;

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 090/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHICH OF THE TWO XRF POINTERS — the `idx` every function in this file threads.
///
/// # ⛔⛔ ONE `int` NAMES A POINTER ROLE, A POSITION IN A PAIR AND A SLOT IN A LOOP AT ONCE
///
/// The map this file is built around carries the meaning of both its dimensions in a COMMENT, because
/// the type cannot:
///
/// ```cpp
/// // a data structure to map vector_load/store to  its corresponding xrf reg SSA.
/// // Dim0: 0, argument, 1, results. Dim1: 0,write, 1,read
/// using XrfPtrMap =
///     std::unordered_map<Operation *, std::array<std::array<Value, 2>, 2>>;
/// ```
/// (`VectorChainToSentientPT.hpp:44-47`)
///
/// `processXrfPtrPerUnit` then loops over that second dimension —
/// `for (int i = 0; i < 2; i++)` under the comment *"write ptr: i=0; read ptr: i=1"*
/// (`LoweringXRF.cpp:351-354`) — and hands the same `i` to [`xrf_value`] and to `updateYieldArgs`
/// (entry 175). So the index into the pair, the pointer's role and the position in the carried list of
/// an enclosing loop are one value, and this is that value.
///
/// ⛔ THE ORDER IS LOAD-BEARING, NOT A CONVENTION. [`sentient::Op::VectorMac`] records that
/// `$pointers` is write then read and that swapping them reads the block being written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum XrfPtr {
    /// `i = 0` — the write pointer.
    Write,
    /// `i = 1` — the read pointer.
    Read,
}

impl XrfPtr {
    /// THE POSITION IT NAMES — the `idx` the reference passes.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            XrfPtr::Write => 0,
            XrfPtr::Read => 1,
        }
    }
}

/// Replaces: e090_getXrfValue
///
/// # THE VALUE AN XRF POINTER IS *READ AS*, WHICH DEPENDS ON WHAT HOLDS IT
///
/// ```cpp
/// // utility function to get xrf_ptr value
/// Value LoweringXRF::getXrfValue(Operation *xrf_ptr, int idx = 0) {
///   Value xrf_ptr_val;
///   if (isa<sentient::YieldOp>(xrf_ptr)) {
///     xrf_ptr_val = xrf_ptr->getParentOp()->getResult(0 + idx);
///   } else if (isa<sentient::ForOp>(xrf_ptr)) {
///     xrf_ptr_val =
///         cast<sentient::ForOp>(xrf_ptr).getBody()->getArgument(1 + idx);
///   } else {
///     xrf_ptr_val = xrf_ptr->getResult(0);
///   }
///   return xrf_ptr_val;
/// }
/// ```
/// (`dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:247-259`)
///
/// # ⭐⭐ THREE ANSWERS FOR ONE POINTER, AND THE MIDDLE ONE IS THE WHOLE POINT
///
/// An xrf pointer travels as an SSA value through a loop nest, and *"who defines it"* changes what a
/// reader must name:
///
/// * an op that PRODUCES it (an `sentient.add`, a dummy `vector_mac`) → its own first result;
/// * a `sentient.for` that CARRIES it → the loop's own body argument, so ops inside the loop read the
///   pointer this iteration advanced to. ⛔ NOT the init value: `getRegionIterArgs()` is
///   `getBody()->getArguments().drop_front(1)` (`SentientOps.td:100-102`), the `1 +` here being the
///   induction variable's slot, and [`sentient::Carried::arg`] records what substituting `init`
///   instead would cost — every iteration would read the pointer the loop STARTED with;
/// * a `sentient.yield` that hands it back → the ENCLOSING op's result, because after the loop the
///   pointer is what the loop returned.
///
/// # ⛔ `getParentOp()` IS THE ONE THING THIS ISLAND CANNOT ANSWER FOR ITSELF
///
/// A region here is a `Vec` of ops with no parent pointer, so the `yield` arm takes its enclosing op
/// as a parameter. That is the *mechanism for reaching an operand* the campaign brief allows a port to
/// drop and be given instead — and it is only read on that one arm, which is why it is an
/// [`Option`]: the two other arms are answerable without it, and the reference itself only
/// dereferences the parent for a terminator.
///
/// ⭐ `None` IS AN INDEX PAST THE END, which is `getResult`/`getArgument`'s own assertion — a loop
/// carrying one pointer cannot answer for the second. In this pipeline both are carried together
/// (`processXrfPtrPerUnit` runs the same walk for `i = 0` and `i = 1`), so it is unreachable there
/// rather than tolerated.
#[must_use]
pub fn xrf_value(xrf_ptr: &sen::Op, enclosing: Option<&sen::Op>, ptr: XrfPtr) -> Option<Val> {
    match xrf_ptr {
        // `isa<sentient::YieldOp>(xrf_ptr)` — `xrf_ptr->getParentOp()->getResult(0 + idx)`.
        sen::Op::Sentient(sentient::Op::Yield { .. }) => {
            sen::results(enclosing?).get(ptr.index()).copied()
        }

        // `isa<sentient::ForOp>(xrf_ptr)` — `getBody()->getArgument(1 + idx)`.
        sen::Op::Sentient(sentient::Op::For { carried, .. }) => {
            carried.get(ptr.index()).map(|carried| carried.arg)
        }

        // `else` — `xrf_ptr->getResult(0)`. ⭐ THE REFERENCE'S OWN FALL-THROUGH, and it ignores `idx`:
        // a producing op holds one pointer, whichever of the two it is.
        other => sen::results(other).first().copied(),
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 091/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHAT A `sentient.for`'S BOUND OPERAND READS BACK AS.
///
/// # ⛔⛔ IT IS THE LOOP'S UPPER BOUND, NOT ITS TRIP COUNT — AND THE REFERENCE MEANS THAT
///
/// A lowered loop's bound is `(upper - lower) / step` written as two ops
/// ([`arith::Op::DivSI`]), and this walk returns the constant behind the SUBTRACTION'S LEFT-HAND
/// SIDE — the upper bound — not the quotient. The two coincide in every fixture the authority tree
/// has (`lower` is `%c0` and `step` is `%c1` at all of
/// `dcc/test/Conversion/VectorChainToSentientPT/dynamic_pt_masking.mlir:225-227`, `:229-231`,
/// `:233-235`, `:238-239`) and they would differ the moment either changed. ⭐ Recorded, not
/// corrected: the caller computes a pointer's travel as *"its loop iter_arg init + bound * stride"*
/// (`LoweringXRF.cpp:283-284`), and what that expression wants is the reference's own answer.
///
/// ⛔ AND A NEWTYPE BECAUSE THE ANSWER IS A COUNT OF ITERATIONS, not the pointer offset it is
/// multiplied into — `createXrfIndexModifOps` (entry 367) mixes both in one expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub struct ForOpBound(pub i64);

/// Replaces: e091_getForOpBound
///
/// # THE TRIP COUNT OF A LOWERED LOOP, READ BACK OUT OF ITS OPERAND
///
/// ```cpp
/// // utility function to get forOp bound value
/// int64_t LoweringXRF::getForOpBound(Operation *op) {
///   auto for_op = dyn_cast<sentient::ForOp>(op);
///   if (for_op) {
///     auto bound_op = for_op.getBound().getDefiningOp();
///     if (isa<mlir::arith::ConstantIndexOp>(bound_op)) {
///       auto const_op = cast<mlir::arith::ConstantIndexOp>(bound_op);
///       return const_op.value();
///     } else if (isa<mlir::arith::DivSIOp>(bound_op)) {
///       auto div_op = cast<mlir::arith::DivSIOp>(bound_op);
///       auto sub_op = div_op.getLhs().getDefiningOp<mlir::arith::SubIOp>();
///       if (auto const_op =
///               sub_op.getLhs().getDefiningOp<mlir::arith::ConstantIndexOp>()) {
///         return const_op.value();
///       } else if (auto symbol_op =
///                      sub_op.getLhs()
///                          .getDefiningOp<mlir::symbol::CreateSymbolOp>()) {
///         // We know that XRF read/write accesses don't involve loops with
///         // symbolic bounds. So, the caller of this function which is computing
///         // the movement, it can be safe to treat as zero.
///         // The movement within a loop = its loop iter_arg init + bound * stride
///         // Since symbolic loops are not involved in array subscripts, the stride
///         // is zero, and hence movement is simply same as loop iter_arg init.
///         // So, bound doesn't play role and it safe to consider as zero.
///         return 0;
///       }
///     } else {
///       op->emitError("unsupported op for getForOpBound()");
///       DT_ERROR("Could not get valid ForOp bound");
///     }
///   }
///   return 0;
/// }
/// ```
/// (`dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:261-293`)
///
/// # ⭐⭐ FOUR `getDefiningOp`s IN A ROW, AGAINST THE REFERENCE'S OWN FIXTURE
///
/// ```text
/// %c2 = arith.constant 2 : index
/// %1 = arith.subi %c2, %c0 : index
/// %2 = arith.divsi %1, %c1 : index
/// sentient.for %arg1 = %2 { .. }
/// ```
/// (`dcc/test/Conversion/VectorChainToSentientPT/dynamic_pt_masking.mlir:216`, `:225-227`) — bound
/// `%2` → `divsi` → its lhs `%1` → `subi` → its lhs `%c2` → **2**. The constants sit two regions
/// above the loop, which is why the scope is a [`Definitions`] rather than one op list; that type's
/// own note has the same listing.
///
/// # THE SYMBOLIC ARM RETURNS ZERO, AND THE REFERENCE ARGUES FOR IT
///
/// A `symbol.create_symbol` behind the subtraction means an unresolved extent, and the reference
/// answers **0** with its reasoning quoted above: *"Since symbolic loops are not involved in array
/// subscripts, the stride is zero, and hence movement is simply same as loop iter_arg init."* So this
/// is not a missing case but a stated one — and it is the arm that made [`symbol`] a dialect of the
/// Sentient island (see [`sen::Op::Symbol`]).
///
/// # THE TWO STOPS
///
/// * `else { emitError(...); DT_ERROR(...) }` — a bound that is neither an `arith.constant` nor an
///   `arith.divsi`. ⛔ A `todo!`, because the reference does not continue: `DT_ERROR` is a stop, and
///   a made-up 0 here would silently misplace every xrf access in the loop.
/// * `div_op.getLhs().getDefiningOp<arith::SubIOp>()` NOT being a `subi`. ⛔ The reference then calls
///   `sub_op.getLhs()` on a null op — undefined behaviour, not a branch — so a `todo!` names the
///   shape rather than inventing the answer the crash withheld.
///
/// ⭐ AND ONE ARM THAT IS *NOT* A STOP: a `subi` whose lhs is neither a constant nor a symbol falls
/// through both `if`s to the function's final `return 0`, which is a real answer the reference gives.
#[must_use]
pub fn for_op_bound(op: &sen::Op, definitions: Definitions<'_>) -> ForOpBound {
    // `dyn_cast<sentient::ForOp>(op)` failing skips the whole body and reaches `return 0`.
    let sen::Op::Sentient(sentient::Op::For { bound, .. }) = op else {
        return ForOpBound(0);
    };

    match definitions.of(*bound) {
        // `isa<mlir::arith::ConstantIndexOp>` — ⭐ THE INDEX CONSTANT, which is why the island keeps
        // it apart from `arith::Op::ConstantInt` ([`arith::Op::Constant`]).
        Some(sen::Op::Arith(arith::Op::Constant { value, .. })) => ForOpBound(*value),

        // `isa<mlir::arith::DivSIOp>` — the `(upper - lower) / step` a scheduler writes.
        Some(sen::Op::Arith(arith::Op::DivSI(div))) => {
            let Some(sen::Op::Arith(arith::Op::SubI(sub))) = definitions.of(div.lhs) else {
                todo!(
                    "getForOpBound: an `arith.divsi` loop bound whose lhs is not an `arith.subi` — \
                     the reference reads `sub_op.getLhs()` through a null op (LoweringXRF.cpp:276)"
                )
            };
            match definitions.of(sub.lhs) {
                Some(sen::Op::Arith(arith::Op::Constant { value, .. })) => ForOpBound(*value),
                // ⭐ THE STATED ZERO, with the reference's own reasoning quoted above.
                Some(sen::Op::Symbol(symbol::Op::CreateSymbol { .. })) => ForOpBound(0),
                // Neither `if` taken — the function's final `return 0`.
                _ => ForOpBound(0),
            }
        }

        // `else { emitError(..); DT_ERROR(..) }` — including the null `bound_op` of a block argument,
        // which the reference hands to `isa<>` unchecked.
        other => todo!(
            "getForOpBound: unsupported op for a sentient.for bound — \
             DT_ERROR(\"Could not get valid ForOp bound\") (LoweringXRF.cpp:288-291): {other:?}"
        ),
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 092/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// A `sentient.vector_mac`'S TWO XRF INCREMENT FIELDS, WITH THE TWO PREDICATES THAT DECIDE THEM.
///
/// # ⛔⛔ A WITNESS, BECAUSE THE REFERENCE'S PARAMETER IS A `sentient::MacOp *`
///
/// `setSentientMacXrfRegIncrAttr` cannot be called with anything but a mac, and `isXrfRdRelated` /
/// `isXrfWtRelated` are declared on `MacOp` alone (`SentientOps.td:274-275`) — they are not questions
/// the other twenty-eight ops of the dialect have answers to. Taking this instead of a
/// [`sentient::Op`] moves that restriction into the type: there is no "not a mac" branch inside the
/// port, and [`MacXrfIncrements::of`] is the single place the op kind is decided.
///
/// ⭐ AND IT IS CONSUMED BY THE PORT, so the two attributes cannot be written one at a time. The
/// reference sets both unconditionally — a mac that received a read increment and kept a stale write
/// increment is not a state it can produce.
#[derive(Debug)]
pub struct MacXrfIncrements<'a> {
    /// `MacOp::isXrfRdRelated()` — already answered; see [`MacXrfIncrements::of`].
    rd_related: bool,
    /// `MacOp::isXrfWtRelated()`.
    wt_related: bool,
    /// `$xrfReadIncr`.
    read: &'a mut u32,
    /// `$xrfWriteIncr`.
    write: &'a mut u32,
}

impl<'a> MacXrfIncrements<'a> {
    /// THE TWO FIELDS OF A `sentient.vector_mac`, WITH ITS XRF PREDICATES ANSWERED.
    ///
    /// ```cpp
    /// bool MacOp::isXrfRdRelated() {
    ///   if ((stringifySentientComputePort(getOpA()).contains("xrf") ||
    ///        stringifySentientComputePort(getOpB()).contains("xrf") ||
    ///        stringifySentientComputePort(getOpC()).contains("xrf")) &&
    ///       getResults().size() > 0)
    ///     return true;
    ///   return false;
    /// }
    ///
    /// bool MacOp::isXrfWtRelated() {
    ///   for (auto dest : getResultForwarding()) {
    ///     if (stringifySentientComputePort(
    ///             mlir::cast<SentientComputePortAttr>(dest).getValue())
    ///             .contains("xrf") &&
    ///         getResults().size() > 0)
    ///       return true;
    ///   }
    ///   return false;
    /// }
    /// ```
    /// (`dcc/src/Dialect/Sentient/SentientOps.cpp:1656-1674`)
    ///
    /// # ⛔⛔ `.contains("xrf")` IS A SET MEMBERSHIP TEST, AND THERE IS EXACTLY ONE MEMBER
    ///
    /// The substring is over `stringifySentientComputePort`'s output, and `xrf` is the only spelling
    /// in `SentientTypes.td`'s port list that contains those three letters — so the test is
    /// `port == Port::Xrf` and nothing else. ⭐ Which is why this reads [`sentient::Port`] and not a
    /// string: `xrfsomething` is not a port a build can name, and a substring test that could match
    /// two members would be a different function.
    ///
    /// # ⭐ AND `getResults().size() > 0` GUARDS BOTH
    ///
    /// A mac binding nothing is xrf-related in neither direction, however its ports read — the
    /// pointers it would advance are results it does not have. Both fixtures that exercise this bind
    /// two (`%[[VAL_15]]:2`, `dummy_mac_ops.mlir:33`).
    #[must_use]
    pub fn of(op: &'a mut sentient::Op) -> Option<MacXrfIncrements<'a>> {
        match op {
            sentient::Op::VectorMac {
                results,
                op_a,
                op_b,
                op_c,
                result,
                xrf_read_incr,
                xrf_write_incr,
                ..
            } => {
                // `getResults().size() > 0`, which both predicates conjoin.
                let binds = !results.is_empty();
                Some(MacXrfIncrements {
                    rd_related: binds
                        && (op_a.port == sentient::Port::Xrf
                            || op_b.port == sentient::Port::Xrf
                            || op_c.port == sentient::Port::Xrf),
                    wt_related: binds && result.forwarding.contains(&sentient::Port::Xrf),
                    read: xrf_read_incr,
                    write: xrf_write_incr,
                })
            }
            // ⭐ NOT A TOTALITY HOLE: the reference's parameter type admits no other op, so this arm
            // exists to say so rather than to answer for one. See this type's note.
            _ => None,
        }
    }

    /// `MacOp::isXrfRdRelated()` — already answered by [`MacXrfIncrements::of`].
    ///
    /// ⭐ READABLE BECAUSE A SECOND CALLER ASKS THE QUESTION WITHOUT SETTING THE ATTRIBUTES.
    /// `replaceXRFImplicitIncrWithAdd` (senpass e037) branches on both predicates and then writes
    /// the increments itself, so the two bools have to leave this type; recomputing them at that call
    /// site would be a second copy of a ported predicate.
    #[must_use]
    pub const fn rd_related(&self) -> bool {
        self.rd_related
    }

    /// `MacOp::isXrfWtRelated()` — see [`MacXrfIncrements::rd_related`].
    #[must_use]
    pub const fn wt_related(&self) -> bool {
        self.wt_related
    }
}

/// HOW FAR THE XRF READ POINTER MOVES PER MAC — `DccExtContext::getXrfRdPtrIncrValAfterMAC`.
///
/// ```cpp
/// unsigned DccExtContext::getXrfRdPtrIncrValAfterMAC(std::string prec) const {
///   bool is_int = prec.find("int") != prec.npos;
///   switch (getArch()) {
///     case SEN1P5_ISA:
///       return 8;
///       break;
///     default:  // DD1/DD2
///       if (is_int)
///         return 2;
///       else
///         return 1;
///       break;
///   }
/// }
/// ```
/// (`dcc/src/Utils/DccExtContext.cpp:354-367`)
///
/// # ⭐⭐ ALL FOUR ANSWERS ARE PINNED BY A `CHECK-SENT-IR` LINE
///
/// | fixture | arch | operand precision | `xrfReadIncr` |
/// |---|---|---|---|
/// | `uniform_pt_xrf.mlir:35` | default | `fp16` | 1 |
/// | `xrf_increments.mlir` (16 macs) | default | `mxfp4` | 1 |
/// | `loweringXRF_with_if_branch.mlir` (4 macs) | default | `int8` | 2 |
/// | `mx_precisions.mlir` (2 macs) | `SENARCH=sen1p5` | `fp16` | 8 |
///
/// The last row is the one that shows the arch outranks the precision: `fp16` reads 1 on DD2 and 8 on
/// SEN1P5, from the same lowering.
///
/// # ⛔ `prec.find("int")` ALSO MATCHES `mxint4`, AND THE ENUM MAKES THAT VISIBLE
///
/// A substring test over the spelling puts `mxint4` on the integer side along with `int1`..`int64` —
/// which is invisible in the C++ and would be easy to lose in a hand-written list. The match below
/// spells all nineteen precisions out, and a unit test compares every arm against
/// `spelling().contains("int")` so the two cannot drift.
///
/// ⭐ THE ARCH IS A TYPE PARAMETER, not a context lookup: `switch (getArch())` reads a global the
/// pipeline was configured with, and here `A::GEN` is a constant of the program being emitted.
#[must_use]
pub fn xrf_rd_ptr_incr_val_after_mac<A: Arch>(precision: sentient::Precision) -> u32 {
    // `bool is_int = prec.find("int") != prec.npos;`
    let is_int = match precision {
        sentient::Precision::Int1
        | sentient::Precision::Int2
        | sentient::Precision::Int4
        | sentient::Precision::Int8
        | sentient::Precision::Int16
        | sentient::Precision::Int24
        | sentient::Precision::Int32
        | sentient::Precision::Int64
        // ⛔ `mxint4` CONTAINS "int".
        | sentient::Precision::Mxint4 => true,

        sentient::Precision::Mxfp4
        | sentient::Precision::Mxfp8
        | sentient::Precision::Fp4
        | sentient::Precision::Fp8
        | sentient::Precision::Fp16
        | sentient::Precision::Bf16
        | sentient::Precision::IeeeFp16
        | sentient::Precision::Fp24
        | sentient::Precision::Fp32
        | sentient::Precision::None => false,
    };

    match A::GEN {
        IsaGen::Sen1p5 => 8,
        // `default: // DD1/DD2`
        IsaGen::Rcudd1a => {
            if is_int {
                2
            } else {
                1
            }
        }
    }
}

/// Replaces: e092_setSentientMacXrfRegIncrAttr
///
/// # THE TWO INCREMENTS A MAC CARRIES, SET FROM WHAT ITS PORTS TOUCH
///
/// ```cpp
/// // set sentient.mac's xrf reg increments
/// void LoweringXRF::setSentientMacXrfRegIncrAttr(
///     sentient::MacOp *mac_op, OpBuilder *builder, std::string precision,
///     const dcc::DccExtContext &dcc_ext_ctx) {
///   int xrf_read_incr_value = 0;
///   if (mac_op->isXrfRdRelated()) {
///     xrf_read_incr_value = dcc_ext_ctx.getXrfRdPtrIncrValAfterMAC(precision);
///   }
///
///   int xrf_write_incr_value = mac_op->isXrfWtRelated() ? 1 : 0;
///   auto int_attr_rd = builder->getI32IntegerAttr(xrf_read_incr_value);
///   auto int_attr_wt = builder->getI32IntegerAttr(xrf_write_incr_value);
///   mac_op->setXrfReadIncrAttr(int_attr_rd);
///   mac_op->setXrfWriteIncrAttr(int_attr_wt);
/// }
/// ```
/// (`dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:664-676`)
///
/// # ⭐⭐ THE TWO DIRECTIONS ARE NOT SYMMETRIC, AND THAT IS THE FUNCTION
///
/// The WRITE increment is **1 or 0** — a mac writing to the XRF advances the write pointer by one
/// row, always. The READ increment is *how many rows one mac consumes*, which depends on the arch
/// and the precision ([`xrf_rd_ptr_incr_val_after_mac`]). The vendor's goldens show both halves
/// independently: `dummy_mac_ops.mlir:33` has `ResultForwarding = [#sentient<compute_port xrf>]` and
/// no xrf operand, giving `xrfReadIncr = 0 : i32, xrfWriteIncr = 1 : i32`; `uniform_pt_xrf.mlir:35`
/// has `opB = #sentient<compute_port xrf>` forwarding its result to `south`, giving the exact
/// mirror, `xrfReadIncr = 1 : i32, xrfWriteIncr = 0 : i32`.
///
/// # THE PRECISION IS THE OPERAND'S ORIGINAL ONE, NOT THE UNIT'S
///
/// All three call sites pass `to_operands[0].value().orig_precision_`
/// (`VectorChainToSentientPT.cpp:217-219`, `:418-420`, `:483-485`) — the precision operand A ARRIVED
/// at, before any promotion — not `unit.getPrecision()`. ⭐ Typed [`sentient::Precision`] here,
/// which is what [`super::vc_vector_chain_to_sentient_pt::compute_unit_precision`] (entry 094)
/// produces from a unit's spelling, so the two ports meet in one enum instead of in a `std::string`.
///
/// # `OpBuilder *builder` — ⛔ **UNREPRESENTABLE HERE**
///
/// Its only use is `builder->getI32IntegerAttr(..)`, which boxes an `int` into an MLIR attribute. The
/// island's fields are already `u32` ([`sentient::Op::VectorMac`]), so there is no attribute to build
/// and no context to build it in.
pub fn set_sentient_mac_xrf_reg_incr_attr<A: Arch>(
    mac: MacXrfIncrements<'_>,
    precision: sentient::Precision,
) {
    // `int xrf_read_incr_value = 0; if (isXrfRdRelated()) …`
    *mac.read = if mac.rd_related {
        xrf_rd_ptr_incr_val_after_mac::<A>(precision)
    } else {
        0
    };

    // `int xrf_write_incr_value = mac_op->isXrfWtRelated() ? 1 : 0;`
    *mac.write = u32::from(mac.wt_related);
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 093/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// AN XRF POINTER PAIR — write then read, which is the `std::array<Value, 2>`'s own order.
///
/// ⛔ THE INDICES ARE NOT INTERCHANGEABLE and the C++ can only say so in a comment (see [`XrfPtr`]).
/// Naming the two positions is what stops `at(0)`/`at(1)` being read the wrong way round at one of
/// the four sites entry 093 has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XrfPtrPair {
    /// Index 0 — the write pointer.
    pub write: Val,
    /// Index 1 — the read pointer.
    pub read: Val,
}

impl XrfPtrPair {
    /// The pointer at one position — `at(idx)`.
    #[must_use]
    pub const fn at(&self, ptr: XrfPtr) -> Val {
        match ptr {
            XrfPtr::Write => self.write,
            XrfPtr::Read => self.read,
        }
    }
}

/// ONE `XrfPtrMap` VALUE — `std::array<std::array<Value, 2>, 2>`, both dimensions named.
///
/// ⭐ `Dim0: 0, argument, 1, results` (`VectorChainToSentientPT.hpp:45`). The ARGUMENT half is the
/// pointer a vector load/store consumes; the RESULTS half is the pointer the placeholder mac binds,
/// and it is the half entry 093 rewires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XrfPtrs {
    /// `at(0)` — the pointers taken as arguments.
    pub argument: XrfPtrPair,
    /// `at(1)` — the pointers bound as results.
    pub results: XrfPtrPair,
}

/// ONE ENTRY OF THE MAC MAP — a real `sentient.vector_mac`'s pointers beside its placeholder's.
///
/// # ⛔⛔ `item.first->getResult(0)` AND `getResult(1)` ARE WRITE AND READ, IN THAT ORDER
///
/// A mac binds its two pointers as results in the same order `$pointers` declares them
/// ([`sentient::Op::VectorMac`]: write first), which is what makes the reference's pairing correct —
/// `at(1).at(0)` (placeholder write) onto `getResult(0)` and `at(1).at(1)` (placeholder read) onto
/// `getResult(1)`. Reading the results as a bare list is how those two get crossed, so
/// [`DummyMacPtrs::of`] is the one place the positions are named.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DummyMacPtrs {
    /// The real mac's own two pointer results.
    pub mac: XrfPtrPair,
    /// `mac_op_to_xrfptr_map`'s value for it.
    pub ptrs: XrfPtrs,
}

impl DummyMacPtrs {
    /// THE MAP ENTRY FOR ONE REAL MAC — `None` when it does not bind the two pointers.
    ///
    /// ⭐ `getResult(0)`/`getResult(1)` READ ONCE, HERE. MLIR asserts on an out-of-range result; a mac
    /// reaching this map has both, since `processXrfPtrPerUnit` only records one that carries
    /// pointers.
    #[must_use]
    pub fn of(mac: &sen::Op, ptrs: XrfPtrs) -> Option<DummyMacPtrs> {
        let results = sen::results(mac);
        Some(DummyMacPtrs {
            mac: XrfPtrPair {
                write: *results.first()?,
                read: *results.get(1)?,
            },
            ptrs,
        })
    }
}

/// Replaces: e093_replaceAndEraseDummyMacOps
///
/// # THE PLACEHOLDER MACS COME OUT, AND EVERY READER MOVES TO THE REAL ONE
///
/// ```cpp
/// // replace and remove dummy mac ops with real ones
/// void LoweringXRF::replaceAndEraseDummyMacOps(XrfPtrMap &mac_op_to_xrfptr_map) {
///   for (auto item : mac_op_to_xrfptr_map) {
///     item.second.at(1).at(0).replaceAllUsesWith(item.first->getResult(0));
///     item.second.at(1).at(1).replaceAllUsesWith(item.first->getResult(1));
///     item.second.at(1).at(0).getDefiningOp()->erase();
///     item.second.at(1).at(1).getDefiningOp()->erase();
///   }
/// }
/// ```
/// (`dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:679-687`)
///
/// # ⭐⭐ WHAT THE PLACEHOLDER WAS FOR
///
/// `processXrfPtrPerUnit` runs BEFORE the computes are lowered, so the pointer values it threads
/// through the loop nest have no mac to come from yet. `insertDummyMacOp` (entry 243) mints one whose
/// only purpose is to bind them — *"function to insert dummy mac to temporily hold xrf ptrs and will
/// be erased during lowering of vector_load/store step"* (`LoweringXRF.cpp:310-311`) — with the
/// `dbgName` *"LoweringXRF dummy Mac"*. Once the real `vector_mac` exists this moves every reader
/// onto it and takes the placeholder out. The observable is an ABSENCE: `dummy_mac_ops.mlir`'s
/// `CHECK-SENT-IR` contains exactly one `sentient.vector_mac` (`:33`) and its own comment says
/// *"Should be no dangling dummy MacOps."*
///
/// # ⛔⛔ THE ORDER OF THE FOUR STATEMENTS IS LOAD-BEARING
///
/// Both rewires come before both erases. `Operation::erase()` on an op that still has uses is MLIR's
/// *"operation destroyed but still has uses"* abort — the same crash [`scf::Op::If`] records from a
/// negated-sibling `scf.if` — and the two placeholders are separate ops, so erasing the first before
/// rewiring the second is a shape this loop could take and does not.
/// ([`sen::erase_defining_op`] repeats the warning at the island.)
///
/// # ⭐ THE MAP'S ITERATION ORDER DOES NOT MATTER, AND THAT IS WORTH STATING
///
/// The reference iterates a `std::unordered_map`, whose order varies between runs. Each entry touches
/// only its own four values, so the result does not depend on it — a `&[DummyMacPtrs]` here is the
/// same function with one fewer thing unstated.
pub fn replace_and_erase_dummy_mac_ops(body: &mut Vec<sen::Op>, map: &[DummyMacPtrs]) {
    for item in map {
        // `item.second.at(1).at(0/1).replaceAllUsesWith(item.first->getResult(0/1))`
        sen::replace_all_uses_with(body, item.ptrs.results.write, item.mac.write);
        sen::replace_all_uses_with(body, item.ptrs.results.read, item.mac.read);

        // `item.second.at(1).at(0/1).getDefiningOp()->erase()` — ⛔ AFTER both rewires.
        sen::erase_defining_op(body, item.ptrs.results.write);
        sen::erase_defining_op(body, item.ptrs.results.read);
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 172/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// A DISPLACEMENT IN STICKS — the unit every number in a [`LayoutExpr`] has been divided down to.
///
/// # ⛔⛔ SIGNED, WHICH IS WHY IT IS NOT [`crate::arch::Sticks`]
///
/// `Sticks` is a `u64` capacity ("how much HBM"); this is a displacement along one, and the
/// arithmetic the reference does on it is subtraction: `curr_const - prev_const`
/// (`LoweringXRF.cpp:447`), `prev_const - curr_const + stride` (`:480-483`) and
/// `-(stride) * getForOpBound(..)` (`:495-498`) all produce negatives, and the last one is negative
/// by construction. A `u64` cannot hold the offset this file's own callers compute.
///
/// # ⭐ AND IT IS ONE UNIT, NOT TWO, IN THE REFERENCE'S OWN ARITHMETIC
///
/// `getLayoutExpr` divides both the loop coefficients and the constant by `stick_elem_num` with the
/// comment *"converted to stick number"* (`LoweringXRF.cpp:60-61`, `:75`), and
/// `processXrfPtrPerUnit` then adds `xrf_incr_after_prev_mac` — a count of XRF rows a MAC consumes,
/// from [`xrf_rd_ptr_incr_val_after_mac`] — straight into the same total
/// (`:435-436`). So the XRF pointer's step and the layout's stick number are the same quantity there,
/// and one type says so.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub struct StickOffset(pub i64);

/// WHERE ONE XRF ACCESS SITS — `LayoutExpr` (`VectorChainToSentientPT.hpp:118-121`).
///
/// ```cpp
/// struct LayoutExpr {
///   std::unordered_map<Operation *, int64_t> layout_map;
///   int64_t constant_val = 0;
/// };
/// ```
///
/// # ⭐⭐ THE MAP IS KEYED BY *ENCLOSING LOOP*, AND EVERY ENCLOSING LOOP IS IN IT
///
/// [`Self::layout_map`]'s key is a `sentient.for`, and its value is that loop's coefficient in the
/// access's flattened address, in sticks. `getLayoutExpr` (entry 240) fills it twice over: once from
/// the flattened affine expression, which only names the loops the subscripts actually read
/// (`LoweringXRF.cpp:52-63`), and again by walking the op's enclosing regions and entering a **zero**
/// for every loop the first pass missed (`:79-92`), with its own comment — *"loop iterator var that is
/// not in layout expression has zero for coefficient"*. That second pass is what makes
/// [`are_xrf_accesses_legal`] a total comparison: two accesses under the same nest have the same key
/// set, so a coefficient present in one and absent from the other is a difference in the NEST, not in
/// the subscripts.
///
/// ⛔ `constant_val` IS DELIBERATELY EXCLUDED FROM THAT COMPARISON. The reference says why:
/// *"all xrf accesses have to have the same expr coeffients except constant expr"*
/// (`LoweringXRF.cpp:115-116`) — a constant difference is the offset
/// [`insert_const_and_add_ops`] emits, so it is legal by construction.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LayoutExpr {
    /// `layout_map` — each enclosing `sentient.for`'s coefficient, in sticks.
    ///
    /// ⭐ ORDERED, NOT HASHED. The reference's `std::unordered_map` is iterated in
    /// [`are_xrf_accesses_legal`], and the answer does not depend on the order (see there); a
    /// `BTreeMap` removes the question instead of restating it.
    pub layout_map: BTreeMap<OpId, StickOffset>,
    /// `constant_val` — the access's own constant term, in sticks. ⛔ NOT COMPARED; see this type's
    /// note.
    pub constant_val: StickOffset,
}

/// `LayoutExprMap` (`VectorChainToSentientPT.hpp:123`) — one [`LayoutExpr`] per xrf access.
///
/// ```cpp
/// using LayoutExprMap = std::unordered_map<Operation *, LayoutExpr>;
/// ```
///
/// ⭐ THE KEY IS THE ACCESS, NOT THE LOOP. `createXrfIndexModifOps` fills it as
/// `xrf_layout_expr_maps[0][op] = getLayoutExpr(op, stick_elem_num)` for each `agen.vector_store` /
/// `vector.store` and `[1][op]` for each load (`LoweringXRF.cpp:641-646`), so the outer key is one of
/// the four memory accesses [`is_xrf_related`] classifies and the inner key
/// ([`LayoutExpr::layout_map`]) is a `sentient.for`. ⛔ TWO DIFFERENT OP POPULATIONS IN ONE NESTED
/// MAP, and the C++ spells both `Operation *`.
pub type LayoutExprMap = BTreeMap<OpId, LayoutExpr>;

/// `LayoutExprMap expr_maps[2]` — the write pointer's accesses beside the read pointer's.
///
/// ```cpp
/// LayoutExprMap xrf_layout_expr_maps[2] = {
///     LayoutExprMap(),   // xrf write pointer
///     LayoutExprMap()};  // xrf read pointer
/// ```
/// (`LoweringXRF.cpp:568-570`)
///
/// ⛔ AN ARRAY OF TWO WHOSE POSITIONS MEAN SOMETHING is the same shape [`XrfPtrPair`] names, and for
/// the same reason: `expr_maps[0]` holds the STORES and `expr_maps[1]` the LOADS
/// (`LoweringXRF.cpp:641-646`), so reading them the wrong way round compares a store's layout against
/// a load's and calls a legal program illegal.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct XrfLayoutExprs {
    /// `expr_maps[0]` — *"xrf write pointer"*: the stores.
    pub write: LayoutExprMap,
    /// `expr_maps[1]` — *"xrf read pointer"*: the loads.
    pub read: LayoutExprMap,
}

impl XrfLayoutExprs {
    /// One pointer's accesses — `expr_maps[i]`.
    #[must_use]
    pub const fn at(&self, ptr: XrfPtr) -> &LayoutExprMap {
        match ptr {
            XrfPtr::Write => &self.write,
            XrfPtr::Read => &self.read,
        }
    }
}

/// Replaces: e172_areXrfAccessesLegal
///
/// # ONE REGISTER PER DIRECTION MEANS EVERY ACCESS MUST TRAVEL AT THE SAME RATE
///
/// ```cpp
/// // utility function to check legality of xrf accesses
/// bool LoweringXRF::areXrfAccessesLegal(LayoutExprMap expr_maps[]) {
///   for (int i = 0; i < 2; i++) {
///     auto expr_map = expr_maps[i];
///     if (expr_map.size() <= 1) continue;
///     std::vector<Operation *> keys;
///
///     // create a key list
///     for (auto pair : expr_map) {
///       keys.push_back(pair.first);
///     }
///
///     for (int i = 0; i < keys.size() - 1; i++) {
///       for (int j = i + 1; j < keys.size(); j++) {
///         auto expr_a = expr_map[keys[i]];
///         auto expr_b = expr_map[keys[j]];
///         // only need to check union of expr_a and expr_b
///         for (auto item : expr_a.layout_map) {
///           // because there is only one reg for xrf read or write, all xrf
///           // accesses have to have the same expr coeffients except constant expr
///           if (item.first != nullptr && expr_b.layout_map.count(item.first) &&
///               expr_b.layout_map[item.first] != item.second)
///             return false;
///         }
///       }
///     }
///   }
///
///   return true;
/// }
/// ```
/// (`dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:97-126`)
///
/// # ⭐⭐ WHAT THE ANSWER DECIDES
///
/// It is the gate in front of the whole XRF pointer lowering, and the alternative is a hard error, not
/// a fallback:
///
/// ```cpp
/// if (xrf_layout_expr_maps[0].size() > 0 ||
///     xrf_layout_expr_maps[1].size() > 0) {
///   if (areXrfAccessesLegal(xrf_layout_expr_maps)) {
///     vector_op_to_xrfptr_map =
///         processXrfPtrPerUnit(unit, xrf_layout_expr_maps);
///   } else {
///     unit->emitError("XRF accesses are illegal");
///   }
/// }
/// ```
/// (`:650-659`) — an illegal unit gets a diagnostic and an EMPTY map, so nothing downstream threads
/// any pointer at all. ⛔ THE REASON IS PHYSICAL AND THE REFERENCE STATES IT: *"because there is only
/// one reg for xrf read or write"*. The write pointer is a single register advanced by one increment
/// per loop iteration, so two stores under the same nest that want different per-iteration strides
/// cannot both be served — there is no second register to advance differently.
///
/// # ⭐ THE COMMENT SAYS UNION, THE CODE DOES INTERSECTION, AND THE CODE IS RIGHT
///
/// *"only need to check union of expr_a and expr_b"* sits above a loop over `expr_a.layout_map`
/// guarded by `expr_b.layout_map.count(item.first)` — the INTERSECTION. And that is sufficient
/// **and** symmetric: a disagreement needs the loop to be in both maps, so iterating `a`'s keys finds
/// every one that iterating `b`'s would. A key in `a` alone is a loop `b` does not sit under, which is
/// a nesting difference [`LayoutExpr`]'s zero-filling pass has already removed for any two accesses
/// that share a nest.
///
/// # ⛔⛔ THE `size() <= 1` GUARD IS LOAD-BEARING, AND NOT FOR THE REASON IT LOOKS LIKE
///
/// `keys.size()` is a `size_t`. With an EMPTY map, `keys.size() - 1` is `SIZE_MAX`, so `for (int i = 0;
/// i < keys.size() - 1; i++)` becomes a loop over the whole index space with an inner loop that never
/// runs — a hang, not a wrong answer. ⭐ AND AN EMPTY MAP REACHES HERE: the caller enters the block on
/// `size() > 0` of EITHER map (`:651-652`), so a program with only loads leaves `expr_maps[0]` empty
/// and this function is called with it. The guard is what turns that into `continue`.
///
/// That underflow has no counterpart below — `keys[i + 1..]` on a one- or zero-element slice is an
/// empty slice — so the guard is documented here instead of written twice. Both answers are the same:
/// a map with fewer than two accesses has no pair to disagree.
///
/// # ⛔ `item.first != nullptr` IS UNREPRESENTABLE, WHICH IS THE POINT
///
/// A null key would be a `layout_map` entry for *no loop*. `getLayoutExpr` cannot make one: the first
/// pass runs `DT_CHECK_MSG(isa<sentient::ForOp>(for_op), ..)` before inserting
/// (`LoweringXRF.cpp:57-61`) and the second tests `parent_op &&` (`:82`). An [`OpId`] key is not
/// nullable, so the clause has nothing to test.
///
/// # ⭐ AND THE `keys` VECTOR IS MECHANISM
///
/// `std::unordered_map` has no random access, so the reference materialises a key list to index pairs
/// out of. Iterating the values in place is the same traversal — and the reference's own
/// `expr_map[keys[i]]` is `operator[]` on a by-value COPY of the map, which would default-insert on a
/// missing key and cannot, because every key came out of it.
#[must_use]
pub fn are_xrf_accesses_legal(expr_maps: &XrfLayoutExprs) -> bool {
    // `for (int i = 0; i < 2; i++) { auto expr_map = expr_maps[i]; .. }` — ⛔ AND `return false`
    // LEAVES BOTH LOOPS, so one illegal direction condemns the unit.
    [XrfPtr::Write, XrfPtr::Read]
        .into_iter()
        .all(|ptr| accesses_agree(expr_maps.at(ptr)))
}

/// ONE POINTER'S ACCESSES, PAIRWISE — the body of [`are_xrf_accesses_legal`]'s outer loop.
///
/// ⛔ THE INNER `int i` SHADOWS THE OUTER ONE in the reference (`LoweringXRF.cpp:99` and `:109` are
/// both `int i`), which is invisible until you look for it and is why the two loops are separate
/// functions here. Nothing reads the outer `i` after the shadow opens, so the shadowing is harmless —
/// but a reader cannot know that without checking, and a later edit inside the pair loop could not
/// reach the pointer index if it needed it.
fn accesses_agree(expr_map: &LayoutExprMap) -> bool {
    // `for (auto pair : expr_map) keys.push_back(pair.first);` — see the anchor on why no key list.
    let exprs: Vec<&LayoutExpr> = expr_map.values().collect();

    // `for (int i = 0; i < keys.size() - 1; i++) for (int j = i + 1; j < keys.size(); j++)` — every
    // unordered pair once.
    for (i, expr_a) in exprs.iter().enumerate() {
        for expr_b in &exprs[i + 1..] {
            // `for (auto item : expr_a.layout_map)` — a's loops, looked up in b.
            for (for_op, coeff) in &expr_a.layout_map {
                // `expr_b.layout_map.count(item.first) && expr_b.layout_map[item.first] != item.second`
                if expr_b
                    .layout_map
                    .get(for_op)
                    .is_some_and(|theirs| theirs != coeff)
                {
                    return false;
                }
            }
        }
    }

    true
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 173/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHAT AN XRF POINTER ADVANCE ADDS TO A PROGRAM — the ops, and what the pointer reads as after.
///
/// ⭐⭐ THE OPS ARE THE VALUE BECAUSE THERE IS NO BUILDER TO POSITION. The reference communicates
/// through an `OpBuilder *` its three call sites have each aimed somewhere different — *before* the
/// memory access (`LoweringXRF.cpp:448`), *before* the `sentient.yield` (`:472-473`), and *after* the
/// enclosing `sentient.for` (`:494`) — and the campaign brief allows dropping exactly that mechanism.
/// So this returns the statements and the caller splices them at the position it chose, which is the
/// shape
/// [`NewMemOp`](super::tf_transform_paged_mem_view_impl::NewMemOp) already uses for the same problem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XrfPtrAdvance {
    /// The `sentient.scalar_constant` then the `sentient.scalar_add`, in emission order —
    /// ⛔ **EMPTY** for a zero offset, which is the whole of the reference's early return.
    pub ops: Vec<sen::Op>,
    /// `getXrfValue(add_op)` — the sum, or `xrf_ptr_val` itself when nothing was emitted.
    pub value: Val,
}

/// Replaces: e173_insertConstAndAddOps
///
/// # MOVE THE XRF POINTER BY A CONSTANT, OR DO NOT MOVE IT AT ALL
///
/// ```cpp
/// // utility function to insert constantOp and addOp for xrf ptr manipulation
/// Value LoweringXRF::insertConstAndAddOps(OpBuilder *builder, Location &loc,
///                                         Type &xrf_reg_type, Value xrf_ptr_val,
///                                         int64_t val, std::string name) {
///   if (val == 0) return xrf_ptr_val;
///
///   auto const_offset =
///       sentient::ConstantOp::create(*builder, loc, xrf_reg_type, val);
///
///   Operation *add_op = sentient::AddOp::create(
///       *builder, loc, xrf_reg_type, const_offset.getOut(), xrf_ptr_val);
///
///   return getXrfValue(add_op);
/// }
/// ```
/// (`dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:295-308`)
///
/// # ⭐⭐ THE ZERO CASE EMITS NOTHING, AND THAT IS AN OPTIMISATION WITH TEETH
///
/// All three callers compute a DIFFERENCE and hand it straight in: `curr_const - prev_const` for an
/// access (`:447`), `prev_const - curr_const + stride` before a yield (`:480-483`), and
/// `-stride * bound` after a loop (`:495-498`). Zero is the common case — consecutive accesses at the
/// same offset, a loop whose stride is zero — and emitting `+ 0` for each would put a dead
/// `scalar_constant`/`scalar_add` pair in front of every one of them. ⛔ AND THE RETURNED VALUE IS
/// THEN THE **UNCHANGED** POINTER, so the caller's `xrf_ptr_val = insertConstAndAddOps(..)`
/// assignment is a no-op rather than a rebinding — which is what keeps the chain of adds tied to the
/// last op that really moved the pointer.
///
/// # ⛔ THE CONSTANT IS THE FIRST OPERAND OF THE ADD, AND THE GOLDEN PRINTS IT SECOND
///
/// `AddOp::create(.., const_offset.getOut(), xrf_ptr_val)` is const then pointer. The vendor's own
/// output for this exact op is the other way round:
///
/// ```text
/// %[[VAL_3:.*]] = sentient.scalar_constant {value = 63 : si64} : index
/// ...
/// %[[VAL_10:.*]] = sentient.scalar_add %[[VAL_9]]#1, %[[VAL_3]] {regIndex = 0 : i32, regLocale = #sentient<reg_type xrfrdptr>} : index, index
/// ```
/// (`dcc/test/PT/issue-212.mlir:16`, `:20` — pointer then constant). ⭐ THAT IS NOT THIS FUNCTION
/// CHANGING ITS MIND: `Sentient_AddOp` carries `Commutative` (`SentientOps.td:700-701`), so
/// canonicalization is free to order the operands, and the same golden shows the constant HOISTED
/// clean out of the `dataflow.program_unit` it was created inside. The emission order below is the
/// reference's; the print order is a later pass's.
///
/// # ⭐ WHAT THE GOLDEN DOES PIN
///
/// Both ops are `index`-typed, matching the only `xrf_reg_type` the three call sites pass
/// (`Type xrf_reg_type = IndexType::get(unit.getContext())`, `LoweringXRF.cpp:371`) — kept as a
/// parameter here because the reference takes one, and stated as [`ScalarTy`] so an `i32` pointer
/// register is expressible rather than assumed away. And the add carries **no** register attributes
/// when it is created: `regLocale`/`regIndex` in the golden are the register passes' work, which is
/// exactly the state [`sentient::Op::ScalarAdd`]'s `Option<Reg>` exists to spell.
///
/// # ⛔ `std::string name` IS UNUSED BY THE REFERENCE, AND ITS VALUE SAYS WHAT IT WANTED
///
/// The parameter is never read. Its one caller passes `xrf_ptr_name`, which is
/// `"imm"` under the comment *"regTypeAssignmentPass will assign reg type"*, over a commented-out
/// `(i == 0) ? "xrfwrptr" : "xrfrdptr"` (`LoweringXRF.cpp:372-374`) — so it is a `regLocale` spelling
/// that was meant for the constant and never wired. ⭐ `"imm"` IS ALREADY WHAT THE CONSTANT GETS:
/// `Sentient_ConstantOp`'s `regLocale` defaults to `SentientRegType::imm` (`SentientOps.td:852`) and
/// `ConstantOp::create(builder, loc, type, value)` passes no attribute. Dropping the parameter loses
/// nothing; wiring it would have been a no-op.
#[must_use]
pub fn insert_const_and_add_ops(
    vals: &mut Values,
    xrf_reg_type: ScalarTy,
    xrf_ptr_val: Val,
    val: StickOffset,
) -> XrfPtrAdvance {
    // `if (val == 0) return xrf_ptr_val;`
    if val == StickOffset(0) {
        return XrfPtrAdvance {
            ops: Vec::new(),
            value: xrf_ptr_val,
        };
    }

    let const_offset = vals.mint();
    let sum = vals.mint();

    XrfPtrAdvance {
        ops: vec![
            // `sentient::ConstantOp::create(*builder, loc, xrf_reg_type, val)`
            sen::Op::Sentient(sentient::Op::ScalarConstant {
                is_symbol: false,
                value: val.0,
                result: const_offset,
                // ⛔ THE DEFAULT, NOT A CHOICE — see this function's note on `name`.
                reg_locale: sentient::RegType::Imm,
                ty: xrf_reg_type,
            }),
            // `sentient::AddOp::create(*builder, loc, xrf_reg_type, const_offset.getOut(),
            //  xrf_ptr_val)` — ⛔ CONST FIRST.
            sen::Op::Sentient(sentient::Op::ScalarAdd {
                lhs: const_offset,
                rhs: xrf_ptr_val,
                result: sum,
                // `AddOp::create` passes neither `regLocale` nor `regIndex`.
                reg: None,
                element_size: None,
                ty: xrf_reg_type,
            }),
        ],
        // `return getXrfValue(add_op);` — ⭐ [`xrf_value`]'S `else` ARM, `xrf_ptr->getResult(0)`: a
        // `sentient.scalar_add` is neither a `yield` nor a `for`, so the answer is the sum and the
        // `idx` the reference does not pass is ignored.
        value: sum,
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 174/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e174_isXrfRelated
///
/// # DOES THIS ACCESS TOUCH THE PT'S TRANSPOSED REGISTER FILE?
///
/// ```cpp
/// // utility function to check whether an op is xrf related.
/// bool LoweringXRF::isXrfRelated(Operation *op) {
///   dataflow::GetLogicalMemoryViewOp memory_view_op;
///   if (auto load_op = dyn_cast<vector::LoadOp>(op)) {
///     memory_view_op =
///         load_op.getBase().getDefiningOp<dataflow::GetLogicalMemoryViewOp>();
///   } else if (auto store_op = dyn_cast<vector::StoreOp>(op)) {
///     memory_view_op =
///         store_op.getBase().getDefiningOp<dataflow::GetLogicalMemoryViewOp>();
///   } else if (auto load_op = dyn_cast<agen::VectorLoadOp>(op)) {
///     memory_view_op =
///         load_op.getMemRef().getDefiningOp<dataflow::GetLogicalMemoryViewOp>();
///   } else if (auto store_op = dyn_cast<agen::VectorStoreOp>(op)) {
///     memory_view_op =
///         store_op.getMemRef().getDefiningOp<dataflow::GetLogicalMemoryViewOp>();
///   } else if (auto tmp_op = dyn_cast<dataflow::GetLogicalMemoryViewOp>(op)) {
///     memory_view_op = tmp_op;
///   } else {
///     op->emitError("unsupported in isXrfRelated()!");
///     DT_ERROR("Could not determine if op is XRF-related");
///   }
///   std::string unit_name;
///   std::optional<std::string> unit_str_optional =
///       dcc::uniform::utils::findUnitType(memory_view_op.getFromUnit());
///   if (unit_str_optional.has_value()) {
///     unit_name = unit_str_optional.value();
///   } else {
///     memory_view_op.emitOpError("Unit type is inconsistent in memory view.");
///     DT_ERROR("Could not determine if op is XRF-related");
///   }
///   if (unit_name.find("xrf") != std::string::npos) return true;
///   return false;
/// }
/// ```
/// (`dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:530-562`)
///
/// # ⭐⭐ FIVE OP CLASSES, ONE QUESTION: WHICH UNIT IS BEHIND THE VIEW
///
/// The four memory accesses reach their view through an operand — `getBase()` for the `vector` pair
/// and `getMemRef()` for the `agen` pair, the same two spellings for one thing that
/// [`layout_map_and_indices`](super::vc_vector_operands::layout_map_and_indices) (entry 170)
/// documents — and a `dataflow.get_logical_memory_view` IS its own answer. Then the view's `from_unit`
/// operand is resolved and its name tested for `xrf`.
///
/// # ⛔⛔ `.find("xrf")` HAS EXACTLY ONE MEMBER, AND IT IS A LOCAL UNIT
///
/// `findUnitType` returns one of three spellings (`dcc/src/Dialect/Uniform/Utils.cpp:286-301`): a
/// `dataflow.get_unit`'s `type` attribute, a `dataflow.get_local_unit`'s `name` attribute, or a
/// `uniform.query_map`'s resolved unit type. Only
/// [`LocalUnit::PtXrf`](crate::islands::dataflow_ir::dialects::dataflow::LocalUnit::PtXrf) spells
/// something containing those three letters — `"ptxrf"` — and no [`DfirUnit`] spelling does. So the
/// substring test is an equality against one enum member, the same collapse
/// [`MacXrfIncrements::of`] records for `stringifySentientComputePort(..).contains("xrf")`. Both
/// matches below are exhaustive so that a new register file or unit type has to say which side it
/// falls on, and a unit test compares every arm against `spelling().contains("xrf")` so the two
/// cannot drift.
///
/// ⭐ AND THAT IS WHY *"is xrf related"* IS REALLY *"is this the PT's transposed register file"*: the
/// ARF (`"ptarf"`), the three LRFs and the L0 scale region all answer no.
///
/// # ⛔ THE `uniform.query_map` ARM HAS NO COUNTERPART, AND CANNOT REACH THIS QUESTION
///
/// `findUnitType`'s third arm reads `uniform::QueryMapOp`, which this island does not carry — for the
/// reason [`symbol::Op`](crate::islands::dataflow_ir::dialects::symbol::Op) gives about its own three
/// missing siblings. It would not help here if it did: a `query_map` resolves to a *unit type*
/// (`getUnitTypeFromUniformMappingAsString`), which is the [`DfirUnit`] vocabulary, and no member of
/// it contains `xrf`. The vendor's own uniformized fixture shows both ops side by side and the view
/// taking the LOCAL one: `%13 = uniform.query_map(map:%12, key:%arg0)` beside
/// `%15 = dataflow.get_local_unit %arg0 {name = "ptxrf"}`, and it is `%15` that
/// `dataflow.get_logical_memory_view` reads
/// (`dcc/test/Conversion/VectorChainToSentientPT/xrf_increments.mlir:386-388`, `:414`).
///
/// # THE TWO STOPS
///
/// * `else { emitError("unsupported in isXrfRelated()!"); DT_ERROR(..) }` — ⛔ A `todo!`, because
///   `DT_ERROR` does not continue. ⭐ AND IT IS UNREACHABLE FROM EVERY CALLER: all SIX call sites
///   guard with `isa<>` over the same op classes first (`LoweringXRF.cpp:575-576`, `:586-588`,
///   `:596-598`, `:632-634`, `:426-428`, `:458-460`).
/// * `getDefiningOp<dataflow::GetLogicalMemoryViewOp>()` returning null — the reference then calls
///   `getFromUnit()` through it, which is a crash rather than a branch, so a `todo!` names the shape
///   instead of inventing the answer the crash withheld. ⭐ AND THE PAGED VIEW IS THE SHAPE IT
///   WOULD BE: none of the eight `dcc/test/Conversion/VectorChainToSentientPT/*.mlir` fixtures
///   carries a `dataflow.get_paged_logical_memory_view` — every one appears under
///   `dcc/test/Transform/TransformPagedMemView/` instead, which is the pass that rewrites them
///   (entry 124).
#[must_use]
pub fn is_xrf_related(op: &DfirOp, scope: &[DfirOp]) -> bool {
    // `dataflow::GetLogicalMemoryViewOp memory_view_op;` and the chain that fills it.
    let base = match op {
        // `dyn_cast<vector::LoadOp>` / `dyn_cast<vector::StoreOp>` — `getBase()`.
        DfirOp::Vector(vector::Op::Load { base, .. })
        | DfirOp::Vector(vector::Op::Store { base, .. }) => *base,

        // `dyn_cast<agen::VectorLoadOp>` / `dyn_cast<agen::VectorStoreOp>` — `getMemRef()`.
        DfirOp::Agen(agen::Op::VectorLoad { view, .. })
        | DfirOp::Agen(agen::Op::VectorStore { view, .. }) => *view,

        // `dyn_cast<dataflow::GetLogicalMemoryViewOp>` — `memory_view_op = tmp_op`, no operand to
        // follow.
        DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView { from, .. }) => {
            return viewed_unit_is_xrf(*from, scope);
        }

        other => todo!(
            "isXrfRelated: unsupported op — DT_ERROR(\"Could not determine if op is XRF-related\") \
             (LoweringXRF.cpp:547-550): {other:?}"
        ),
    };

    // `.getDefiningOp<dataflow::GetLogicalMemoryViewOp>()`.
    match defining_op(base, scope) {
        Some(DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView { from, .. })) => {
            viewed_unit_is_xrf(*from, scope)
        }
        other => todo!(
            "isXrfRelated: a memory access whose base is not a dataflow.get_logical_memory_view — \
             the reference reads `memory_view_op.getFromUnit()` through a null op \
             (LoweringXRF.cpp:552-553): {other:?}"
        ),
    }
}

/// `findUnitType(memory_view_op.getFromUnit())` FOLLOWED BY `unit_name.find("xrf")` — see
/// [`is_xrf_related`] on why that is one enum comparison.
///
/// ```cpp
/// std::optional<std::string> findUnitType(
///     mlir::TypedValue<mlir::IndexType> unit_index_type) {
///   if (auto unit = unit_index_type.getDefiningOp<mlir::dataflow::GetUnitOp>())
///     return unit.getType().str();
///
///   if (auto memory_unit =
///           unit_index_type.getDefiningOp<mlir::dataflow::GetLocalUnitOp>())
///     return memory_unit.getName().str();
///
///   if (auto mapping_unit_op =
///           unit_index_type.getDefiningOp<mlir::uniform::QueryMapOp>())
///     return dcc::uniform::utils::getUnitTypeFromUniformMappingAsString(
///         mapping_unit_op);
///
///   return std::nullopt;
/// }
/// ```
/// (`dcc/src/Dialect/Uniform/Utils.cpp:286-301`)
fn viewed_unit_is_xrf(from_unit: Val, scope: &[DfirOp]) -> bool {
    match defining_op(from_unit, scope) {
        // `getDefiningOp<dataflow::GetUnitOp>()` → `unit.getType().str()`, the `type` attribute.
        // ⛔ EXHAUSTIVE AND ALL FALSE: no unit TYPE names a register file; see [`is_xrf_related`].
        Some(DfirOp::Dataflow(dataflow::Op::GetUnit { unit, .. })) => match unit {
            DfirUnit::Sfp
            | DfirUnit::Pe
            | DfirUnit::PtRow(_)
            | DfirUnit::Lxlu
            | DfirUnit::Lxsu
            | DfirUnit::Lx
            | DfirUnit::Hbm
            | DfirUnit::L0lu
            | DfirUnit::L0su
            | DfirUnit::L0
            | DfirUnit::L3lu
            | DfirUnit::L3su
            | DfirUnit::Constant
            | DfirUnit::SfpState
            | DfirUnit::PeState
            | DfirUnit::SfpRing
            | DfirUnit::LxVirtualIbr
            | DfirUnit::L3Ibr
            | DfirUnit::CrossPtnLink
            | DfirUnit::LxluScaleReg => false,
        },

        // `getDefiningOp<dataflow::GetLocalUnitOp>()` → `memory_unit.getName().str()`, and `"ptxrf"`
        // is the one name in that set containing `xrf`.
        Some(DfirOp::Dataflow(dataflow::Op::GetLocalUnit { which, .. })) => match which {
            LocalUnit::PtXrf => true,
            LocalUnit::PeLrf
            | LocalUnit::SfpLrf
            | LocalUnit::PtLrf
            | LocalUnit::PtArf
            | LocalUnit::PtIrf
            | LocalUnit::L0Scale => false,
        },

        // `return std::nullopt;` — `emitOpError("Unit type is inconsistent in memory view.")` and
        // `DT_ERROR`. ⛔ A `todo!`: the reference stops, and a view whose unit operand is defined by
        // neither op is one this pass cannot place.
        other => todo!(
            "isXrfRelated: Unit type is inconsistent in memory view — \
             DT_ERROR(\"Could not determine if op is XRF-related\") (LoweringXRF.cpp:556-558): \
             {other:?}"
        ),
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 175/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHICH OF THE ENCLOSING OP'S REGIONS HOLDS THE `sentient.yield` BEING EXTENDED.
///
/// # ⛔ THE REFERENCE HOLDS THE YIELD; THIS ISLAND HAS TO SAY WHERE IT IS
///
/// `updateYieldArgs` takes a `sentient::YieldOp &` and then asks it for `getParentOp()`. A region here
/// is a `Vec` of ops with no parent pointer, so the pair *(enclosing op, which of its regions)*
/// replaces the yield handle — the "mechanism for reaching operands" the campaign brief allows a port
/// to be given instead of walking. It is an `enum` and not an index because the two positions are the
/// only two that exist: [`sentient::regions`] gives one region for a `sentient.for` and exactly two,
/// `then` first, for a `sentient.if`.
///
/// ⭐ AND IT IS THE DISTINCTION THE CALLER ALREADY MAKES. The one call site branches on
/// `if_op.getThenRegion().isAncestor(yield_op->getParentRegion())`
/// (`dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:510`) to decide
/// whether the pointer chain continues or is reset to the `then` region's initial pointer, so the
/// walker that will call this in entry 345 knows which region it descended into before it gets here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum YieldRegion {
    /// Region 0 — a `sentient.for`'s body, or a `sentient.if`'s `then` region.
    BodyOrThen,
    /// Region 1 — a `sentient.if`'s `else` region. ⛔ A `sentient.for` has no second region, so this
    /// names nothing on one and [`update_yield_args`] answers [`None`].
    Else,
}

impl YieldRegion {
    /// THE POSITION `getRegions()` INDEXES IT AT.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            YieldRegion::BodyOrThen => 0,
            YieldRegion::Else => 1,
        }
    }
}

/// Replaces: e175_updateYieldArgs
///
/// # A REGION HANDS ONE MORE VALUE BACK, AND THE POINTER IS THEN READ AS THE ENCLOSING OP'S RESULT
///
/// ```cpp
/// Value LoweringXRF::updateYieldArgs(sentient::YieldOp &yield_op,
///                                    Value &xrf_ptr_val, int idx) {
///   SmallVector<Value, 2> yield_args;
///   for (auto it : yield_op.getOperands()) {
///     yield_args.push_back(it);
///   }
///   yield_args.push_back(xrf_ptr_val);
///   yield_op->setOperands(yield_args);
///   return getXrfValue(yield_op, idx);
/// }
/// ```
/// (`dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:690-699`)
///
/// ⛔ THE EXTRACT DROPPED THE `return getXrfValue(yield_op, idx);` — `crustify-bridge2/source/bridge2.cpp`
/// ends this body at `setOperands`, and `crustify-bridge2/UNITS.tsv` therefore records NO callee for
/// entry 175. The tail is the half that makes the function a *function* rather than a mutation: the
/// copy-and-append is a `push`, and what the caller assigns is [`xrf_value`]'s answer.
///
/// # ⭐⭐ THE APPEND AND THE ANSWER ARE THE SAME POSITION, WHICH IS WHY `idx` SERVES BOTH
///
/// The pointers are appended in the order `processXrfPtrPerUnit` loops them — write at `i = 0`, read at
/// `i = 1` (`LoweringXRF.cpp:351-354`) — and the enclosing op's results were created in that same order
/// by `createForOpWithReturnValue` (entry 241) and `createIfOpWithReturnValue` (entry 242), which append
/// the two xrf pointers as the last iter args and give the `if` two `IndexType` results. So the `idx`-th
/// operand of the yield and the `idx`-th result of its parent are one pointer, and the reference's own
/// expectation shows both ends of it:
///
/// ```mlir
/// %51:2 = sentient.for %52 = %28 iter_args(%53 = %50#0, %54 = %50#1) -> (index, index) {…}{
///   …
///   %63:2 = sentient.vector_mac pointers(%53, %59) {…}
///   %64 = sentient.scalar_constant {value = -4 : si64} : index
///   %65 = sentient.scalar_add %64, %63#1 : index, index
///   sentient.yield %63#0, %65 : index, index
/// }
/// %66 = sentient.scalar_constant {value = -32 : si64} : index
/// %67 = sentient.scalar_add %66, %51#1 : index, index
/// ```
/// (`dcc/test/Conversion/VectorChainToSentientPT/loweringXRF_with_if_branch.mlir:59-74`, `CHECK-SENT-IR`)
///
/// Two calls built that `sentient.yield` — one appending the write pointer `%63#0`, one the read pointer
/// `%65` — and the value the second call returned is `%51#1`, which is what the next offset is added to.
///
/// # ⚠️ IT DOES NOT CREATE THE RESULT IT NAMES
///
/// Appending a yield operand does not widen the parent's result list; the parent was built with both
/// results already — `createForOpWithReturnValue` pushes the `xrfwrptr` and `xrfrdptr` constants on as
/// the last two iter args (entry 241, `LoweringXRF.cpp:145-146`) and `createIfOpWithReturnValue` asks
/// for two `IndexType` results and puts a BARE `sentient.yield` in each region before cloning the body
/// in FRONT of it (entry 242, `LoweringXRF.cpp:192-195`, `:207`, `:212`, `:216`) — so the terminator is
/// the region's last op — and this closes the region over them. On an op whose result
/// list is shorter than `idx` the answer is [`None`] — `getResult`'s own assertion — and the pointer
/// simply has nowhere to be read from, so nothing is emitted rather than a wrong slot being read.
#[must_use]
pub fn update_yield_args(
    enclosing: &mut sen::Op,
    region: YieldRegion,
    xrf_ptr_val: Val,
    ptr: XrfPtr,
) -> Option<Val> {
    // `yield_args` is `getOperands()` copied and then `push_back(xrf_ptr_val)`, and `setOperands`
    // installs it — one `push` onto the terminator's own operand list.
    let host = match enclosing {
        sen::Op::Sentient(host) => host,
        _ => return None,
    };
    match sentient::regions_mut(host)
        .into_iter()
        .nth(region.index())
        .and_then(|ops| ops.last_mut())
    {
        Some(sen::Op::Sentient(sentient::Op::Yield { results })) => results.push(xrf_ptr_val),
        // ⭐ A REGION WHOSE LAST OP IS NOT A `sentient.yield` HAS NO YIELD TO EXTEND. The reference is
        // handed one and cannot be here; an `else` region asked of a `sentient.for` lands here too.
        Some(_) | None => return None,
    }

    // `return getXrfValue(yield_op, idx);` — entry 090's `sentient::YieldOp` arm, which is
    // `xrf_ptr->getParentOp()->getResult(0 + idx)`. Both borrows below are shared reborrows of the op
    // just written; the second lookup exists so this returns entry 090's answer rather than a copy of
    // its yield arm.
    let host = match &*enclosing {
        sen::Op::Sentient(host) => host,
        _ => return None,
    };
    let regions = sentient::regions(host);
    let yield_op = (*regions.get(region.index())?).last()?;
    xrf_value(yield_op, Some(&*enclosing), ptr)
}

#[cfg(test)]
mod unit_tests {
    use super::{
        DfirOp, DummyMacPtrs, ForOpBound, LayoutExpr, LayoutExprMap, LocalUnit, MacXrfIncrements,
        OpId, StickOffset, Values, XrfLayoutExprs, XrfPtr, XrfPtrAdvance, XrfPtrPair, XrfPtrs,
        YieldRegion, agen, are_xrf_accesses_legal, dataflow, for_op_bound,
        insert_const_and_add_ops, is_xrf_related, replace_and_erase_dummy_mac_ops,
        set_sentient_mac_xrf_reg_incr_attr, update_yield_args, vector,
        xrf_rd_ptr_incr_val_after_mac, xrf_value,
    };
    use crate::arch::{Dd2, Sen1p5};
    use crate::islands::dataflow_ir::dialects::Index;
    use crate::islands::dataflow_ir::ty::{
        AffineExpr, AffineMap, ElemType, MemRef, ScalarTy, Vector,
    };
    use crate::islands::sentient::dialects::{
        self as sen, Definitions, Val, arith, sentient, symbol,
    };
    use crate::units::{Core, Corelet, DfirUnit, Residency, Row};

    /// A `sentient.for` carrying the two xrf pointers, write then read.
    fn loop_carrying(bound: Val, iv: Val, ptrs: [(Val, Val, Val); 2]) -> sen::Op {
        sen::Op::Sentient(sentient::Op::For {
            iv,
            bound,
            bound_reg: None,
            carried: ptrs
                .into_iter()
                .map(|(init, arg, result)| sentient::Carried {
                    init,
                    arg,
                    result,
                    reg: sentient::Reg {
                        locale: sentient::RegType::Unknown,
                        index: None,
                    },
                    program_header: false,
                    element_size: None,
                })
                .collect(),
            dbg_name: None,
            body: Vec::new(),
        })
    }

    /// A `sentient.vector_mac` with the ports and results one golden shows.
    fn mac(
        op_a: sentient::Port,
        forwarding: Vec<sentient::Port>,
        results: Vec<Val>,
        precision: sentient::Precision,
    ) -> sentient::Op {
        sentient::Op::VectorMac {
            mask: None,
            xrf_write_ptr: None,
            xrf_read_ptr: None,
            results,
            op_a: sentient::Operand::from(op_a),
            op_b: sentient::Operand::from(sentient::Port::South),
            op_c: sentient::Operand::from(sentient::Port::Latch),
            result: sentient::ResultPorts {
                forwarding,
                precision,
                unroll_incr: false,
            },
            mode: sentient::FmaMode::FusedMulAdd,
            compute_precision: precision,
            fold_mode: None,
            unroll_factor: sentient::UnrollFactor::X1,
            xrf_read_incr: 0,
            xrf_write_incr: 0,
            data_transfer_only: false,
            is_data_weight: None,
            dbg_name: None,
        }
    }

    /// 🎯 090/384 — A `sentient.yield` READS THE ENCLOSING OP'S RESULT AT THE POINTER'S POSITION.
    ///
    /// `xrf_ptr->getParentOp()->getResult(0 + idx)` (`LoweringXRF.cpp:250-251`).
    #[test]
    fn a_yield_reads_the_enclosing_ops_result() {
        let enclosing = loop_carrying(
            Val(0),
            Val(1),
            [(Val(2), Val(3), Val(4)), (Val(5), Val(6), Val(7))],
        );
        let yield_op = sen::Op::Sentient(sentient::Op::Yield {
            results: vec![Val(8), Val(9)],
        });

        assert_eq!(
            xrf_value(&yield_op, Some(&enclosing), XrfPtr::Write),
            Some(Val(4)),
            "the loop's first RESULT, not the yield's own operand"
        );
        assert_eq!(
            xrf_value(&yield_op, Some(&enclosing), XrfPtr::Read),
            Some(Val(7))
        );
    }

    /// 🎯 090/384 — A `sentient.for` READS ITS OWN BODY ARGUMENT, NOT ITS INIT AND NOT ITS RESULT.
    ///
    /// `getBody()->getArgument(1 + idx)` (`LoweringXRF.cpp:252-254`) — the `1 +` skips the induction
    /// variable, which is why the init value would be the wrong answer (see [`sentient::Carried`]).
    #[test]
    fn a_loop_reads_its_own_body_argument() {
        let for_op = loop_carrying(
            Val(0),
            Val(1),
            [(Val(2), Val(3), Val(4)), (Val(5), Val(6), Val(7))],
        );

        assert_eq!(xrf_value(&for_op, None, XrfPtr::Write), Some(Val(3)));
        assert_eq!(xrf_value(&for_op, None, XrfPtr::Read), Some(Val(6)));
    }

    /// 🎯 090/384 — ANY OTHER OP ANSWERS WITH ITS FIRST RESULT, AND IGNORES WHICH POINTER WAS ASKED.
    ///
    /// `else { xrf_ptr_val = xrf_ptr->getResult(0); }` (`LoweringXRF.cpp:255-256`) — no `idx`.
    #[test]
    fn any_other_op_reads_its_first_result_for_either_pointer() {
        let producer = sen::Op::Sentient(sentient::Op::ScalarConstant {
            is_symbol: false,
            value: 7,
            result: Val(11),
            reg_locale: sentient::RegType::Imm,
            ty: ScalarTy::Index,
        });

        assert_eq!(xrf_value(&producer, None, XrfPtr::Write), Some(Val(11)));
        assert_eq!(xrf_value(&producer, None, XrfPtr::Read), Some(Val(11)));
    }

    /// 🎯 091/384 — A CONSTANT-INDEX BOUND IS ITS OWN VALUE.
    #[test]
    fn a_constant_bound_is_read_straight_off_the_op() {
        let scope = vec![sen::Op::Arith(arith::Op::Constant {
            result: Val(0),
            value: 400,
        })];
        let for_op = loop_carrying(
            Val(0),
            Val(1),
            [(Val(2), Val(3), Val(4)), (Val(5), Val(6), Val(7))],
        );

        assert_eq!(
            for_op_bound(&for_op, Definitions::from_innermost(&[&scope])),
            ForOpBound(400)
        );
    }

    /// 🎯 091/384 — THE `divsi`/`subi` CHAIN, WALKED ACROSS TWO REGIONS.
    ///
    /// The reference's own fixture, with the constants where it puts them:
    ///
    /// ```text
    /// %c2 = arith.constant 2 : index                    <- func body
    /// %c0 = arith.constant 0 : index
    /// %c1 = arith.constant 1 : index
    /// dataflow.program_unit … {
    ///   %1 = arith.subi %c2, %c0 : index                <- the unit's region
    ///   %2 = arith.divsi %1, %c1 : index
    ///   sentient.for %arg1 = %2 { .. }
    /// ```
    /// (`dcc/test/Conversion/VectorChainToSentientPT/dynamic_pt_masking.mlir:216-227`) — the answer
    /// is **2**, the loop's upper bound.
    #[test]
    fn the_divsi_subi_chain_is_walked_across_two_regions() {
        let outer = vec![
            sen::Op::Arith(arith::Op::Constant {
                result: Val(0),
                value: 2,
            }),
            sen::Op::Arith(arith::Op::Constant {
                result: Val(1),
                value: 0,
            }),
            sen::Op::Arith(arith::Op::Constant {
                result: Val(2),
                value: 1,
            }),
        ];
        let inner = vec![
            sen::Op::Arith(arith::Op::SubI(arith::IntBinary {
                result: Val(3),
                lhs: Val(0),
                rhs: Val(1),
                ty: ScalarTy::Index,
            })),
            sen::Op::Arith(arith::Op::DivSI(arith::IntBinary {
                result: Val(4),
                lhs: Val(3),
                rhs: Val(2),
                ty: ScalarTy::Index,
            })),
        ];
        let for_op = loop_carrying(
            Val(4),
            Val(5),
            [(Val(6), Val(7), Val(8)), (Val(9), Val(10), Val(11))],
        );

        assert_eq!(
            for_op_bound(&for_op, Definitions::from_innermost(&[&inner, &outer])),
            ForOpBound(2),
            "the subtraction's LHS, which is the loop's upper bound"
        );
    }

    /// 🎯 091/384 — A SYMBOLIC BOUND IS **ZERO**, WHICH IS THE REFERENCE'S OWN ANSWER.
    ///
    /// *"We know that XRF read/write accesses don't involve loops with symbolic bounds. So, the
    /// caller of this function which is computing the movement, it can be safe to treat as zero."*
    /// (`LoweringXRF.cpp:279-284`)
    #[test]
    fn a_symbolic_bound_is_zero() {
        let scope = vec![
            sen::Op::Symbol(symbol::Op::CreateSymbol {
                result: Val(0),
                symbol_id: 0,
                max_value: None,
            }),
            sen::Op::Arith(arith::Op::Constant {
                result: Val(1),
                value: 0,
            }),
            sen::Op::Arith(arith::Op::Constant {
                result: Val(2),
                value: 1,
            }),
            sen::Op::Arith(arith::Op::SubI(arith::IntBinary {
                result: Val(3),
                lhs: Val(0),
                rhs: Val(1),
                ty: ScalarTy::Index,
            })),
            sen::Op::Arith(arith::Op::DivSI(arith::IntBinary {
                result: Val(4),
                lhs: Val(3),
                rhs: Val(2),
                ty: ScalarTy::Index,
            })),
        ];
        let for_op = loop_carrying(
            Val(4),
            Val(5),
            [(Val(6), Val(7), Val(8)), (Val(9), Val(10), Val(11))],
        );

        assert_eq!(
            for_op_bound(&for_op, Definitions::from_innermost(&[&scope])),
            ForOpBound(0)
        );
    }

    /// 🎯 092/384 — THE FOUR `xrfReadIncr` VALUES THE VENDOR'S GOLDENS STATE.
    ///
    /// | fixture | arch | precision | expected |
    /// |---|---|---|---|
    /// | `uniform_pt_xrf.mlir:35` | DD2 | `fp16` | 1 |
    /// | `xrf_increments.mlir` | DD2 | `mxfp4` | 1 |
    /// | `loweringXRF_with_if_branch.mlir` | DD2 | `int8` | 2 |
    /// | `mx_precisions.mlir` (`SENARCH=sen1p5`) | SEN1P5 | `fp16` | 8 |
    #[test]
    fn the_read_increment_matches_every_golden() {
        assert_eq!(
            xrf_rd_ptr_incr_val_after_mac::<Dd2>(sentient::Precision::Fp16),
            1
        );
        assert_eq!(
            xrf_rd_ptr_incr_val_after_mac::<Dd2>(sentient::Precision::Mxfp4),
            1
        );
        assert_eq!(
            xrf_rd_ptr_incr_val_after_mac::<Dd2>(sentient::Precision::Int8),
            2
        );
        assert_eq!(
            xrf_rd_ptr_incr_val_after_mac::<Sen1p5>(sentient::Precision::Fp16),
            8,
            "the arch outranks the precision — the same fp16 that reads 1 on DD2"
        );
    }

    /// 🎯 092/384 — THE INTEGER SIDE IS EXACTLY THE SPELLINGS CONTAINING `int`, `mxint4` INCLUDED.
    ///
    /// `bool is_int = prec.find("int") != prec.npos;` (`DccExtContext.cpp:355`) — a substring test
    /// this port spells out as nineteen arms, and this is what stops the two drifting.
    #[test]
    fn the_integer_side_is_the_spelling_that_contains_int() {
        for precision in [
            sentient::Precision::Int1,
            sentient::Precision::Int2,
            sentient::Precision::Int4,
            sentient::Precision::Int8,
            sentient::Precision::Int16,
            sentient::Precision::Int24,
            sentient::Precision::Int32,
            sentient::Precision::Int64,
            sentient::Precision::Mxint4,
            sentient::Precision::Mxfp4,
            sentient::Precision::Mxfp8,
            sentient::Precision::Fp4,
            sentient::Precision::Fp8,
            sentient::Precision::Fp16,
            sentient::Precision::Bf16,
            sentient::Precision::IeeeFp16,
            sentient::Precision::Fp24,
            sentient::Precision::Fp32,
            sentient::Precision::None,
        ] {
            let expected = if precision.spelling().contains("int") {
                2
            } else {
                1
            };
            assert_eq!(
                xrf_rd_ptr_incr_val_after_mac::<Dd2>(precision),
                expected,
                "{}",
                precision.spelling()
            );
        }
    }

    /// 🎯 092/384 — A MAC THAT FORWARDS ITS RESULT TO THE XRF INCREMENTS THE **WRITE** POINTER ONLY.
    ///
    /// `dummy_mac_ops.mlir:33`: `opA = #sentient<compute_port zero>`,
    /// `ResultForwarding = [#sentient<compute_port xrf>]`, fp16, two results —
    /// `xrfReadIncr = 0 : i32, xrfWriteIncr = 1 : i32`.
    #[test]
    fn a_mac_forwarding_to_the_xrf_increments_its_write_pointer_only() {
        let mut op = mac(
            sentient::Port::Zero,
            vec![sentient::Port::Xrf],
            vec![Val(0), Val(1)],
            sentient::Precision::Fp16,
        );

        let witness = MacXrfIncrements::of(&mut op).expect("a mac");
        set_sentient_mac_xrf_reg_incr_attr::<Dd2>(witness, sentient::Precision::Fp16);

        let sentient::Op::VectorMac {
            xrf_read_incr,
            xrf_write_incr,
            ..
        } = op
        else {
            unreachable!("built above")
        };
        assert_eq!((xrf_read_incr, xrf_write_incr), (0, 1));
    }

    /// 🎯 092/384 — A MAC READING THE XRF INCREMENTS THE **READ** POINTER ONLY, THE EXACT MIRROR.
    ///
    /// `uniform_pt_xrf.mlir:35`: `opB = #sentient<compute_port xrf>` forwarding to `south`, fp16 —
    /// `xrfReadIncr = 1 : i32, xrfWriteIncr = 0 : i32`.
    #[test]
    fn a_mac_reading_the_xrf_increments_its_read_pointer_only() {
        let mut op = mac(
            sentient::Port::Xrf,
            vec![sentient::Port::South],
            vec![Val(0), Val(1)],
            sentient::Precision::Fp16,
        );

        let witness = MacXrfIncrements::of(&mut op).expect("a mac");
        set_sentient_mac_xrf_reg_incr_attr::<Dd2>(witness, sentient::Precision::Fp16);

        let sentient::Op::VectorMac {
            xrf_read_incr,
            xrf_write_incr,
            ..
        } = op
        else {
            unreachable!("built above")
        };
        assert_eq!((xrf_read_incr, xrf_write_incr), (1, 0));
    }

    /// 🎯 092/384 — A MAC THAT BINDS NOTHING IS XRF-RELATED IN NEITHER DIRECTION.
    ///
    /// `getResults().size() > 0` conjoins both predicates (`SentientOps.cpp:1657-1673`): the pointers
    /// it would advance are results it does not have.
    #[test]
    fn a_mac_binding_nothing_increments_neither_pointer() {
        let mut op = mac(
            sentient::Port::Xrf,
            vec![sentient::Port::Xrf],
            Vec::new(),
            sentient::Precision::Int8,
        );

        let witness = MacXrfIncrements::of(&mut op).expect("a mac");
        set_sentient_mac_xrf_reg_incr_attr::<Dd2>(witness, sentient::Precision::Int8);

        let sentient::Op::VectorMac {
            xrf_read_incr,
            xrf_write_incr,
            ..
        } = op
        else {
            unreachable!("built above")
        };
        assert_eq!((xrf_read_incr, xrf_write_incr), (0, 0));
    }

    /// 🎯 092/384 — THE WITNESS ADMITS ONLY A MAC.
    #[test]
    fn no_other_op_has_xrf_increments() {
        let mut op = sentient::Op::Yield {
            results: vec![Val(0)],
        };
        assert!(MacXrfIncrements::of(&mut op).is_none());
    }

    /// 🎯 093/384 — THE PLACEHOLDERS COME OUT AND THEIR READERS MOVE ONTO THE REAL MAC.
    ///
    /// `dummy_mac_ops.mlir`'s `CHECK-SENT-IR` holds ONE `sentient.vector_mac` and its own comment
    /// says *"Should be no dangling dummy MacOps."* Two placeholder constants stand for the write and
    /// read pointers; a `sentient.for` carries both; after the rewrite the loop carries the real mac's
    /// two results and nothing defines the placeholders.
    #[test]
    fn the_placeholders_come_out_and_their_readers_move_onto_the_real_mac() {
        let real = mac(
            sentient::Port::Xrf,
            vec![sentient::Port::Xrf],
            vec![Val(10), Val(11)],
            sentient::Precision::Fp16,
        );
        let mut body = vec![
            // The placeholder pair — `insertDummyMacOp`'s two results, standing in until now.
            sen::Op::Sentient(sentient::Op::ScalarConstant {
                is_symbol: false,
                value: 0,
                result: Val(0),
                reg_locale: sentient::RegType::Imm,
                ty: ScalarTy::Index,
            }),
            sen::Op::Sentient(sentient::Op::ScalarConstant {
                is_symbol: false,
                value: 0,
                result: Val(1),
                reg_locale: sentient::RegType::Imm,
                ty: ScalarTy::Index,
            }),
            sen::Op::Sentient(real.clone()),
            // The reader: a loop carrying both placeholders in.
            loop_carrying(
                Val(2),
                Val(3),
                [(Val(0), Val(4), Val(5)), (Val(1), Val(6), Val(7))],
            ),
        ];

        let entry = DummyMacPtrs::of(
            &sen::Op::Sentient(real),
            XrfPtrs {
                argument: XrfPtrPair {
                    write: Val(20),
                    read: Val(21),
                },
                results: XrfPtrPair {
                    write: Val(0),
                    read: Val(1),
                },
            },
        )
        .expect("a mac binding both pointers");

        replace_and_erase_dummy_mac_ops(&mut body, &[entry]);

        assert_eq!(body.len(), 2, "both placeholders erased");
        let sen::Op::Sentient(sentient::Op::For { carried, .. }) = &body[1] else {
            unreachable!("the loop is the last op")
        };
        assert_eq!(
            (carried[0].init, carried[1].init),
            (Val(10), Val(11)),
            "the real mac's results, write pointer first"
        );
    }

    /// 🎯 093/384 — A MAC THAT DOES NOT BIND TWO POINTERS IS NOT A MAP ENTRY.
    #[test]
    fn a_mac_binding_one_result_has_no_map_entry() {
        let one = sen::Op::Sentient(mac(
            sentient::Port::Xrf,
            vec![sentient::Port::Xrf],
            vec![Val(10)],
            sentient::Precision::Fp16,
        ));
        assert!(
            DummyMacPtrs::of(
                &one,
                XrfPtrs {
                    argument: XrfPtrPair {
                        write: Val(20),
                        read: Val(21),
                    },
                    results: XrfPtrPair {
                        write: Val(0),
                        read: Val(1),
                    },
                },
            )
            .is_none()
        );
    }

    // ══════════════════════════════════════════════════════════════════════════════════════════
    // 172/384 — `areXrfAccessesLegal`
    // ══════════════════════════════════════════════════════════════════════════════════════════

    /// One access at `path`, with the coefficients it gives its enclosing loops and no constant.
    fn access(coeffs: &[(&[u32], i64)]) -> LayoutExpr {
        LayoutExpr {
            layout_map: coeffs
                .iter()
                .map(|(loop_path, coeff)| (OpId::at(loop_path), StickOffset(*coeff)))
                .collect(),
            constant_val: StickOffset(0),
        }
    }

    /// Two accesses under one nest, keyed by position.
    fn one_pointer(a: LayoutExpr, b: LayoutExpr) -> XrfLayoutExprs {
        XrfLayoutExprs {
            write: [(OpId::at(&[0, 0]), a), (OpId::at(&[0, 1]), b)]
                .into_iter()
                .collect(),
            read: LayoutExprMap::new(),
        }
    }

    /// Two stores under the same `sentient.for` that want different per-iteration strides cannot both
    /// be served by the one write-pointer register (`LoweringXRF.cpp:115-119`).
    #[test]
    fn accesses_disagreeing_on_a_shared_loop_are_illegal() {
        let maps = one_pointer(access(&[(&[0], 4)]), access(&[(&[0], 8)]));

        assert!(!are_xrf_accesses_legal(&maps));
    }

    /// The same stride is what a single register can walk.
    #[test]
    fn accesses_agreeing_on_every_shared_loop_are_legal() {
        let maps = one_pointer(
            access(&[(&[0], 4), (&[1], 16)]),
            access(&[(&[0], 4), (&[1], 16)]),
        );

        assert!(are_xrf_accesses_legal(&maps));
    }

    /// ⛔ ONLY THE INTERSECTION IS TESTED. The reference's `expr_b.layout_map.count(item.first)`
    /// guard means a loop only one access sits under says nothing about legality — and
    /// [`LayoutExpr`]'s zero-filling pass (`LoweringXRF.cpp:79-92`) is what makes that safe for two
    /// accesses that DO share a nest.
    #[test]
    fn loops_only_one_access_names_are_not_compared() {
        let maps = one_pointer(access(&[(&[0], 4)]), access(&[(&[1], 8)]));

        assert!(are_xrf_accesses_legal(&maps));
    }

    /// ⭐ THE CONSTANT TERM IS EXCLUDED BY DESIGN — *"all xrf accesses have to have the same expr
    /// coeffients except constant expr"* (`LoweringXRF.cpp:115-116`): a constant difference is
    /// exactly what [`insert_const_and_add_ops`] emits.
    #[test]
    fn a_differing_constant_term_is_legal() {
        let mut a = access(&[(&[0], 4)]);
        let mut b = access(&[(&[0], 4)]);
        a.constant_val = StickOffset(0);
        b.constant_val = StickOffset(63);

        assert!(are_xrf_accesses_legal(&one_pointer(a, b)));
    }

    /// ⛔⛔ THE `size() <= 1` GUARD. A pointer with no accesses at all reaches this function — the
    /// caller enters on `size() > 0` of EITHER map (`LoweringXRF.cpp:651-652`) — and the reference's
    /// `keys.size() - 1` would underflow a `size_t` on it.
    #[test]
    fn a_pointer_with_fewer_than_two_accesses_is_legal() {
        let empty = XrfLayoutExprs::default();
        assert!(are_xrf_accesses_legal(&empty));

        let single = XrfLayoutExprs {
            write: [(OpId::at(&[0, 0]), access(&[(&[0], 4)]))]
                .into_iter()
                .collect(),
            read: LayoutExprMap::new(),
        };
        assert!(are_xrf_accesses_legal(&single));
    }

    /// ⛔ EITHER DIRECTION CONDEMNS THE UNIT: `return false` leaves both of the reference's loops, and
    /// the caller's only alternative is `emitError("XRF accesses are illegal")` (`:654-658`).
    #[test]
    fn an_illegal_read_pointer_condemns_a_legal_write_pointer() {
        let maps = XrfLayoutExprs {
            write: [(OpId::at(&[0, 0]), access(&[(&[0], 4)]))]
                .into_iter()
                .collect(),
            read: [
                (OpId::at(&[0, 1]), access(&[(&[0], 4)])),
                (OpId::at(&[0, 2]), access(&[(&[0], 5)])),
            ]
            .into_iter()
            .collect(),
        };

        assert!(!are_xrf_accesses_legal(&maps));
        assert!(are_xrf_accesses_legal(&XrfLayoutExprs {
            write: maps.write.clone(),
            read: LayoutExprMap::new(),
        }));
    }

    /// Which map is which — `expr_maps[0]` is the write pointer's (`LoweringXRF.cpp:568-570`).
    #[test]
    fn the_two_maps_are_reachable_by_pointer() {
        let maps = one_pointer(access(&[(&[0], 4)]), access(&[(&[0], 4)]));

        assert_eq!(maps.at(XrfPtr::Write).len(), 2);
        assert!(maps.at(XrfPtr::Read).is_empty());
    }

    // ══════════════════════════════════════════════════════════════════════════════════════════
    // 173/384 — `insertConstAndAddOps`
    // ══════════════════════════════════════════════════════════════════════════════════════════

    /// ⛔ A ZERO OFFSET EMITS NOTHING and hands back the pointer it was given, so the caller's
    /// `xrf_ptr_val = insertConstAndAddOps(..)` is a no-op (`LoweringXRF.cpp:299`).
    #[test]
    fn a_zero_offset_emits_nothing() {
        let mut vals = Values::default();
        let ptr = vals.mint();

        let advance = insert_const_and_add_ops(&mut vals, ScalarTy::Index, ptr, StickOffset(0));

        assert_eq!(
            advance,
            XrfPtrAdvance {
                ops: Vec::new(),
                value: ptr,
            }
        );
        // ⭐ AND NOTHING WAS MINTED, so no value name is burnt on an op that was not emitted.
        assert_eq!(vals.issued(), 1);
    }

    /// The vendor's own pair for this function:
    ///
    /// ```text
    /// %[[VAL_3:.*]] = sentient.scalar_constant {value = 63 : si64} : index
    /// ...
    /// %[[VAL_10:.*]] = sentient.scalar_add %[[VAL_9]]#1, %[[VAL_3]] {regIndex = 0 : i32, regLocale = #sentient<reg_type xrfrdptr>} : index, index
    /// ```
    /// (`dcc/test/PT/issue-212.mlir:16`, `:20`) — ⛔ THE GOLDEN'S OPERAND ORDER IS THE OTHER WAY
    /// ROUND and its register attributes come from a later pass; see the anchor.
    #[test]
    fn a_non_zero_offset_emits_the_constant_then_the_add() {
        let mut vals = Values::default();
        let ptr = vals.mint();

        let advance = insert_const_and_add_ops(&mut vals, ScalarTy::Index, ptr, StickOffset(63));

        let offset = Val(1);
        let sum = Val(2);
        assert_eq!(
            advance,
            XrfPtrAdvance {
                ops: vec![
                    sen::Op::Sentient(sentient::Op::ScalarConstant {
                        is_symbol: false,
                        value: 63,
                        result: offset,
                        reg_locale: sentient::RegType::Imm,
                        ty: ScalarTy::Index,
                    }),
                    sen::Op::Sentient(sentient::Op::ScalarAdd {
                        // ⛔ CONST FIRST — `AddOp::create(.., const_offset.getOut(), xrf_ptr_val)`.
                        lhs: offset,
                        rhs: ptr,
                        result: sum,
                        // ⭐ NO REGISTER YET: `regTypeAssignmentPass` assigns it (`:373-374`).
                        reg: None,
                        ty: ScalarTy::Index,
                        element_size: None,
                    }),
                ],
                value: sum,
            }
        );
    }

    /// ⛔ THE OFFSET IS SIGNED, AND ONE CALLER'S IS NEGATIVE BY CONSTRUCTION:
    /// `-(loop_stride_step) * getForOpBound(forop)` unwinds a loop's whole travel after it
    /// (`LoweringXRF.cpp:495-498`), which is why [`StickOffset`] is not [`crate::arch::Sticks`].
    #[test]
    fn a_negative_offset_is_expressible() {
        let mut vals = Values::default();
        let ptr = vals.mint();
        let stride = StickOffset(4);
        let bound = ForOpBound(8);

        let advance = insert_const_and_add_ops(
            &mut vals,
            ScalarTy::Index,
            ptr,
            StickOffset(-stride.0 * bound.0),
        );

        assert_eq!(
            advance.ops.first(),
            Some(&sen::Op::Sentient(sentient::Op::ScalarConstant {
                is_symbol: false,
                value: -32,
                result: Val(1),
                reg_locale: sentient::RegType::Imm,
                ty: ScalarTy::Index,
            }))
        );
        assert_eq!(advance.value, Val(2));
    }

    /// The pointer register's type is a parameter, and an `i32` one is expressible even though the
    /// three call sites all pass an `index` one (`LoweringXRF.cpp:371`, `:450`, `:485`, `:499`).
    #[test]
    fn the_register_type_is_carried_through() {
        let mut vals = Values::default();
        let ptr = vals.mint();

        let advance = insert_const_and_add_ops(&mut vals, ScalarTy::Int(32), ptr, StickOffset(1));

        assert!(advance.ops.iter().all(|op| matches!(
            op,
            sen::Op::Sentient(
                sentient::Op::ScalarConstant {
                    ty: ScalarTy::Int(32),
                    ..
                } | sentient::Op::ScalarAdd {
                    ty: ScalarTy::Int(32),
                    ..
                }
            )
        )));
    }

    // ══════════════════════════════════════════════════════════════════════════════════════════
    // 174/384 — `isXrfRelated`
    // ══════════════════════════════════════════════════════════════════════════════════════════

    /// `%15 = dataflow.get_local_unit %arg0 {name = "ptxrf"} : index`
    /// (`xrf_increments.mlir:388`), and its `pt_lrfreg` neighbour at `:387`.
    fn local_unit(result: Val, which: LocalUnit) -> DfirOp {
        DfirOp::Dataflow(dataflow::Op::GetLocalUnit {
            result,
            of: Val(0),
            which,
        })
    }

    /// `%45 = dataflow.get_logical_memory_view %15, %c0 {layout_map = #map}
    ///  : index, index, memref<4x64x64x1xf4E2M1FN>` (`xrf_increments.mlir:414`), whose `#map` is
    /// `affine_map<(d0, d1, d2, d3) -> (d3 * 16384 + d2 * 256 + d1 * 4 + d0)>` (`:350`).
    fn xrf_view(result: Val, from: Val) -> DfirOp {
        DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
            result,
            from,
            start: Val(1),
            layout: AffineMap {
                dims: 4,
                syms: 0,
                results: vec![
                    AffineExpr::dim(3)
                        .times(16384)
                        .plus(AffineExpr::dim(2).times(256))
                        .plus(AffineExpr::dim(1).times(4))
                        .plus(AffineExpr::dim(0)),
                ],
            },
            ty: MemRef {
                shape: vec![4, 64, 64, 1],
                elem: ElemType::F4E2M1Fn,
            },
        })
    }

    /// `agen.vector_store %44, %45[0, 0, %43 + %34 * 4 + %31 * 8 + %28 * 16, 0]
    ///  {dbgName = "transfer_lds2_src:lxlu_dst:ptrow0", store_order = #map1, store_set = #set}
    ///  : memref<4x64x64x1xf4E2M1FN>, vector<256xf4E2M1FN>` (`xrf_increments.mlir:415`).
    fn xrf_store(view: Val) -> DfirOp {
        DfirOp::Agen(agen::Op::VectorStore {
            dbg_name: None,
            value: Val(44),
            view,
            indices: vec![
                Index::Const(0),
                Index::Const(0),
                Index::Strided(
                    vec![(Val(43), 1), (Val(34), 4), (Val(31), 8), (Val(28), 16)],
                    0,
                ),
                Index::Const(0),
            ],
            view_ty: MemRef {
                shape: vec![4, 64, 64, 1],
                elem: ElemType::F4E2M1Fn,
            },
            ty: Vector {
                len: 256,
                elem: ElemType::F4E2M1Fn,
            },
        })
    }

    /// The vendor's own xrf store — a `ptxrf` view under an `agen.vector_store`
    /// (`xrf_increments.mlir:388`, `:414-415`).
    #[test]
    fn an_agen_store_through_a_ptxrf_view_is_xrf_related() {
        let scope = vec![
            local_unit(Val(15), LocalUnit::PtXrf),
            xrf_view(Val(45), Val(15)),
            xrf_store(Val(45)),
        ];

        assert!(is_xrf_related(&scope[2], &scope));
    }

    /// ⛔ THE SAME STORE OVER THE LRF ANSWERS NO. `%14 = dataflow.get_local_unit %arg0
    /// {name = "pt_lrfreg"}` (`xrf_increments.mlir:387`) sits one line above the `ptxrf` one in the
    /// same unit, so the only thing separating an xrf access from an LRF access is which handle the
    /// view was taken from.
    #[test]
    fn the_same_store_through_a_pt_lrfreg_view_is_not() {
        let scope = vec![
            local_unit(Val(14), LocalUnit::PtLrf),
            xrf_view(Val(45), Val(14)),
            xrf_store(Val(45)),
        ];

        assert!(!is_xrf_related(&scope[2], &scope));
    }

    /// ⭐ A MEMORY VIEW IS ITS OWN ANSWER — the fifth arm, which is the one the caller's `unit.walk`
    /// uses to decide whether the unit touches the XRF at all
    /// (`LoweringXRF.cpp:574-578`, `:545-546`).
    #[test]
    fn a_memory_view_is_classified_by_its_own_unit() {
        let xrf = vec![
            local_unit(Val(15), LocalUnit::PtXrf),
            xrf_view(Val(45), Val(15)),
        ];
        let lrf = vec![
            local_unit(Val(14), LocalUnit::PtLrf),
            xrf_view(Val(45), Val(14)),
        ];

        assert!(is_xrf_related(&xrf[1], &xrf));
        assert!(!is_xrf_related(&lrf[1], &lrf));
    }

    /// The plain-`vector` arm, over the SFP's register file:
    ///
    /// ```text
    /// %lrf_memory_unit = dataflow.get_local_unit %sfp_c0 {name="sfp_lrfreg"} : index
    /// %lrf_memory_fp16 = dataflow.get_logical_memory_view %lrf_memory_unit, %c0
    ///                     {layout_map = affine_map<(i, j) -> (64 * i + j) >}
    ///                     : index, index, memref<8x64xf16>
    /// %data1 = vector.load %lrf_memory_fp16[%c4, %c0] : memref<8x64xf16>, vector<64xf16>
    /// ```
    /// (`dcc/test/Conversion/VectorChainToSentientPESFP/sfp-to-sfp-ring.mlir:167-171`)
    ///
    /// ⛔ NOT A PT FIXTURE, BECAUSE THE PT TREE HAS NO PLAIN `vector.load` OVER A LOCAL UNIT — every
    /// PT access is an `agen` one. The two spellings differ only in the accessor name
    /// (`getBase()` against `getMemRef()`), so the arm is real and this is what exercises it.
    #[test]
    fn a_plain_vector_load_over_an_sfp_lrf_view_is_not_xrf_related() {
        let view = DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
            result: Val(80),
            from: Val(81),
            start: Val(82),
            layout: AffineMap {
                dims: 2,
                syms: 0,
                results: vec![AffineExpr::dim(0).times(64).plus(AffineExpr::dim(1))],
            },
            ty: MemRef {
                shape: vec![8, 64],
                elem: ElemType::F16,
            },
        });
        let scope = vec![
            local_unit(Val(81), LocalUnit::SfpLrf),
            view,
            DfirOp::Vector(vector::Op::Load {
                result: Val(83),
                base: Val(80),
                indices: vec![Index::Const(4), Index::Const(0)],
                base_ty: MemRef {
                    shape: vec![8, 64],
                    elem: ElemType::F16,
                },
                ty: Vector {
                    len: 64,
                    elem: ElemType::F16,
                },
            }),
        ];

        assert!(!is_xrf_related(&scope[2], &scope));
    }

    /// A view over a whole unit rather than a register file, from the vendor's own LX fixture:
    /// `%lx_memory_unit = dataflow.get_unit {core = 0, corelet = 0, name = "C0-CL0-LX", type="lx"}`
    /// feeding `dataflow.get_logical_memory_view` (`dcc/test/LXLU/rotate.mlir:129-132`).
    #[test]
    fn a_view_over_a_unit_type_is_not_xrf_related() {
        let scope = vec![
            DfirOp::Dataflow(dataflow::Op::GetUnit {
                result: Val(81),
                residency: Residency::Corelet {
                    core: Core::checked(0).expect("the arch has core 0"),
                    corelet: Corelet::checked(0).expect("the arch has corelet 0"),
                },
                unit: DfirUnit::Lx,
                num_folds: None,
            }),
            xrf_view(Val(45), Val(81)),
        ];

        assert!(!is_xrf_related(&scope[1], &scope));
    }

    /// ⛔⛔ THE SUBSTRING TEST HAS EXACTLY ONE MEMBER, and this is what keeps the two exhaustive
    /// matches in [`is_xrf_related`] honest: `.find("xrf")` over
    /// [`LocalUnit::spelling`](crate::islands::dataflow_ir::dialects::dataflow::LocalUnit::spelling)
    /// picks `"ptxrf"` and nothing else.
    #[test]
    fn ptxrf_is_the_only_local_unit_whose_name_contains_xrf() {
        for which in [
            LocalUnit::PeLrf,
            LocalUnit::SfpLrf,
            LocalUnit::PtLrf,
            LocalUnit::PtXrf,
            LocalUnit::PtArf,
            LocalUnit::PtIrf,
            LocalUnit::L0Scale,
        ] {
            let scope = vec![local_unit(Val(15), which), xrf_view(Val(45), Val(15))];

            assert_eq!(
                is_xrf_related(&scope[1], &scope),
                which.spelling().contains("xrf"),
                "{which:?} spells {}",
                which.spelling()
            );
        }
    }

    /// ⛔ AND NO UNIT *TYPE* CONTAINS IT — which is why `findUnitType`'s `get_unit` arm
    /// (`dcc/src/Dialect/Uniform/Utils.cpp:288-289`) is uniformly false, and why its third,
    /// `uniform.query_map` arm would be too.
    #[test]
    fn no_unit_type_spelling_contains_xrf() {
        let rows = (0..8).filter_map(Row::checked).map(DfirUnit::PtRow);
        let units = [
            DfirUnit::Sfp,
            DfirUnit::Pe,
            DfirUnit::Lxlu,
            DfirUnit::Lxsu,
            DfirUnit::Lx,
            DfirUnit::Hbm,
            DfirUnit::L0lu,
            DfirUnit::L0su,
            DfirUnit::L0,
            DfirUnit::L3lu,
            DfirUnit::L3su,
            DfirUnit::Constant,
            DfirUnit::SfpState,
            DfirUnit::PeState,
            DfirUnit::SfpRing,
            DfirUnit::LxVirtualIbr,
            DfirUnit::CrossPtnLink,
        ];

        for unit in units.into_iter().chain(rows) {
            assert!(
                !unit.spelling().contains("xrf"),
                "{unit:?} spells {}",
                unit.spelling()
            );
        }
    }

    // ══════════════════════════════════════════════════════════════════════════════════════════
    // 175/384
    // ══════════════════════════════════════════════════════════════════════════════════════════

    /// A `sentient.yield` handing back the values a region closes over.
    fn a_yield(results: Vec<Val>) -> sen::Op {
        sen::Op::Sentient(sentient::Op::Yield { results })
    }

    /// A `sentient.if` with two results and a `sentient.yield` at the end of each region — the shape
    /// `%42:2` has in `loweringXRF_with_if_branch.mlir`.
    fn if_yielding(results: [Val; 2], then_body: Vec<sen::Op>, else_body: Vec<sen::Op>) -> sen::Op {
        sen::Op::Sentient(sentient::Op::If {
            predicate: sentient::CmpPredicate::Slt,
            lhs: Val(26),
            rhs: Val(6),
            yielded: results
                .into_iter()
                .map(|result| sentient::Yielded {
                    result,
                    reg: sentient::Reg {
                        locale: sentient::RegType::Unknown,
                        index: None,
                    },
                    element_size: None,
                })
                .collect(),
            dbg_name: None,
            then_body,
            else_body,
        })
    }

    /// What the terminator of one region hands back, or nothing when that region has no terminator.
    fn yielded_operands(op: &sen::Op, region: YieldRegion) -> Vec<Val> {
        let sen::Op::Sentient(host) = op else {
            unreachable!("built above")
        };
        let regions = sentient::regions(host);
        match regions.get(region.index()).and_then(|ops| ops.last()) {
            Some(sen::Op::Sentient(sentient::Op::Yield { results })) => results.clone(),
            Some(_) | None => Vec::new(),
        }
    }

    /// 🎯 175/384 — THE VENDOR'S OWN INNERMOST LOOP: two appends build `sentient.yield %63#0, %65`,
    /// and what they hand back are the loop's two results `%51#0` and `%51#1`.
    ///
    /// `dcc/test/Conversion/VectorChainToSentientPT/loweringXRF_with_if_branch.mlir:59-74`.
    #[test]
    fn both_pointers_appended_in_order_read_back_as_the_loops_two_results() {
        // `%51:2 = sentient.for %52 = %28 iter_args(%53 = %50#0, %54 = %50#1) -> (index, index)`.
        let mut for_op = loop_carrying(
            Val(28),
            Val(52),
            [(Val(500), Val(53), Val(510)), (Val(501), Val(54), Val(511))],
        );
        if let sen::Op::Sentient(sentient::Op::For { body, .. }) = &mut for_op {
            body.push(a_yield(Vec::new()));
        }

        // `i = 0`, the write pointer: `%63#0`.
        let write = update_yield_args(
            &mut for_op,
            YieldRegion::BodyOrThen,
            Val(630),
            XrfPtr::Write,
        );
        // `i = 1`, the read pointer: `%65 = sentient.scalar_add %64, %63#1`.
        let read = update_yield_args(&mut for_op, YieldRegion::BodyOrThen, Val(65), XrfPtr::Read);

        assert_eq!(
            yielded_operands(&for_op, YieldRegion::BodyOrThen),
            vec![Val(630), Val(65)],
            "sentient.yield %63#0, %65 — write pointer first, in the order the caller loops i"
        );
        assert_eq!(
            (write, read),
            (Some(Val(510)), Some(Val(511))),
            "getXrfValue(yield_op, idx) is the parent's result at that same position — %51#0, %51#1"
        );
    }

    /// 🎯 175/384 — THE TWO REGIONS OF A `sentient.if` ARE EXTENDED SEPARATELY, and both are read back
    /// as the *same* result of the `if`: the golden's `then` ends `sentient.yield %51#0, %69` and its
    /// `else` ends `sentient.yield %77#0, %77#1`, after which `%79 = sentient.scalar_add %78, %42#1`
    /// reads one pointer for both paths
    /// (`loweringXRF_with_if_branch.mlir:77`, `:87`, `:90`).
    #[test]
    fn each_region_of_an_if_is_extended_on_its_own_and_both_answer_the_ifs_result() {
        let mut if_op = if_yielding(
            [Val(420), Val(421)],
            vec![a_yield(vec![Val(510)])],
            vec![a_yield(vec![Val(770)])],
        );

        let then_read =
            update_yield_args(&mut if_op, YieldRegion::BodyOrThen, Val(69), XrfPtr::Read);
        assert_eq!(
            yielded_operands(&if_op, YieldRegion::BodyOrThen),
            vec![Val(510), Val(69)],
            "the then region grew"
        );
        assert_eq!(
            yielded_operands(&if_op, YieldRegion::Else),
            vec![Val(770)],
            "and the else region did not"
        );

        let else_read = update_yield_args(&mut if_op, YieldRegion::Else, Val(771), XrfPtr::Read);
        assert_eq!(
            yielded_operands(&if_op, YieldRegion::Else),
            vec![Val(770), Val(771)]
        );
        assert_eq!(
            (then_read, else_read),
            (Some(Val(421)), Some(Val(421))),
            "one result of the if, whichever region yielded it"
        );
    }

    /// 🎯 175/384 — A `sentient.for` HAS NO SECOND REGION, so there is nothing to extend and nothing
    /// to read: the position names no region at all.
    #[test]
    fn a_loop_has_no_else_region_to_extend() {
        let mut for_op = loop_carrying(
            Val(28),
            Val(52),
            [(Val(500), Val(53), Val(510)), (Val(501), Val(54), Val(511))],
        );
        if let sen::Op::Sentient(sentient::Op::For { body, .. }) = &mut for_op {
            body.push(a_yield(Vec::new()));
        }

        assert_eq!(
            update_yield_args(&mut for_op, YieldRegion::Else, Val(65), XrfPtr::Read),
            None
        );
        assert!(
            yielded_operands(&for_op, YieldRegion::BodyOrThen).is_empty(),
            "and the body's own yield was left alone"
        );
    }

    /// 🎯 175/384 — A REGION WHOSE LAST OP IS NOT A `sentient.yield` IS NOT A REGION THIS CLOSES.
    /// The reference is handed a `sentient::YieldOp` and cannot be asked this; here the answer is an
    /// absence rather than an operand appended to whatever op happened to be last.
    #[test]
    fn a_region_not_terminated_by_a_yield_is_left_untouched() {
        let mut if_op = if_yielding(
            [Val(420), Val(421)],
            vec![sen::Op::Sentient(mac(
                sentient::Port::Xrf,
                Vec::new(),
                vec![Val(500), Val(501)],
                sentient::Precision::Int8,
            ))],
            Vec::new(),
        );
        assert_eq!(
            update_yield_args(&mut if_op, YieldRegion::BodyOrThen, Val(69), XrfPtr::Read),
            None
        );
        assert!(yielded_operands(&if_op, YieldRegion::BodyOrThen).is_empty());
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 240/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ONE ENCLOSING `sentient.for`, AS THE TWO THINGS `getLayoutExpr` ASKS OF IT.
///
/// ⛔ THE PARENT LINK IS THE CALLER'S, AS EVERYWHERE IN THIS FILE — see [`Definitions`]. The
/// reference reaches the loop twice, by `sub_op.getRhs().getParentRegion()->getParentOp()` and by
/// walking `op->getParentRegion()` outwards (`LoweringXRF.cpp:54`, `:79-92`), and both answers are
/// in this list: the [`OpId`] is the `layout_map` key, the [`Val`] is `getInductionVar()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnclosingLoop {
    /// The loop's position — the `Operation *` `layout_map` is keyed by.
    pub op: OpId,
    /// `getInductionVar()` — what an `arith.subi`'s rhs is matched against.
    pub iv: Val,
}

/// `layout_coeffs[i] / stick_elem_num` — *"converted to stick number"* (`LoweringXRF.cpp:61`).
///
/// ⭐ THE DIVISOR CANNOT BE ZERO, which is what [`ElementsPerStick`] is for, and `cast_signed` on a
/// stick's element count is exact: a stick is 1024 bits and the widest format is 32.
fn in_sticks(elements: i64, stick_elem_num: ElementsPerStick) -> StickOffset {
    StickOffset(elements / stick_elem_num.elements().0.cast_signed())
}

/// Replaces: e240_getLayoutExpr
///
/// **240/384** `LoweringXRF::getLayoutExpr` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:28` (67L).
///
/// ⛔ TWO PASSES, AND THE SECOND IS WHAT MAKES THE COMPARISON TOTAL: the flattened map names only
/// the loops the subscripts read, then EVERY enclosing loop gets a zero — see [`LayoutExpr`].
/// ⛔ A SUBSCRIPT IS A LOOP ITERATOR ONLY THROUGH AN `arith.subi`, whose RHS is the induction
/// variable; the reference's own TODO says the `sub_op` is a pass-ordering artefact (`:53`).
/// ⚠️ `fullyComposeAffineMapAndOperands`/`simplifyAffineMap`/`canonicalizeMapAndOperands` are
/// dropped as canonicalisations, with `agen_access_details.rs:2192-2205`' precedent.
#[must_use]
pub fn layout_expr(
    op: &OpId,
    scope: &[DfirOp],
    enclosing: &[EnclosingLoop],
    stick_elem_num: ElementsPerStick,
) -> LayoutExpr {
    let mut layout_expr = LayoutExpr::default();

    // `auto result = getLayoutMapAndIndices(op, logical_view_map, operands, logical_view_op);` —
    // ⛔ THE REFERENCE DISCARDS `result`, so a map it could not read leaves the default behind.
    let Some(view) = layout_map_and_indices(op, scope) else {
        return layout_expr;
    };

    // `getFlattenedAffineExpr(expr, operands.size(), map.getNumSymbols(), &layout_coeffs, ..)` with
    // `DT_CHECK(layout_coeffs.size() == operands.size() + 1)` — [`AffineMap::coefficients`] is that
    // row, dims then symbols then the constant.
    let layout_coeffs = view.layout_map.coefficients(0);

    for (i, operand) in view.operands.iter().enumerate() {
        // `operands[i].getDefiningOp<mlir::arith::SubIOp>()`
        let Some(DfirOp::Arith(arith::Op::SubI(sub_op))) = defining_op(*operand, scope) else {
            continue;
        };
        // `auto for_op = sub_op.getRhs().getParentRegion()->getParentOp();` and the
        // `DT_CHECK_MSG(isa<sentient::ForOp>(for_op), "indices must be constant or for loop
        // iterator")` beside it.
        let Some(driving) = enclosing.iter().find(|loop_| loop_.iv == sub_op.rhs) else {
            todo!("getLayoutExpr: subscript {operand:?} is neither a constant nor a loop iterator");
        };
        let Some(coeff) = layout_coeffs.get(i) else {
            continue;
        };
        layout_expr
            .layout_map
            .insert(driving.op.clone(), in_sticks(*coeff, stick_elem_num));
    }

    // `int const_val = layout_coeffs.back();`
    let mut const_val = layout_coeffs.last().copied().unwrap_or(0);

    // `logical_view_op.getStartAddress().getDefiningOp()` as an `arith::ConstantIndexOp`, with the
    // `DT_CHECK_MSG(false, "the base address .. should be constants for lowering")` `else`.
    let start = match op_at(&view.logical_view_op, scope) {
        Some(DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView { start, .. })) => *start,
        other => todo!("getLayoutExpr: {other:?} is not the logical memory view of an xrf access"),
    };
    match defining_op(start, scope) {
        Some(DfirOp::Arith(arith::Op::Constant { value, .. })) => const_val += *value,
        other => todo!("getLayoutExpr: an xrf view's base address must be constant, not {other:?}"),
    }

    // `layout_expr.constant_val += const_val / stick_elem_num;`
    layout_expr.constant_val.0 += in_sticks(const_val, stick_elem_num).0;

    // The outward region walk: *"loop iterator var that is not in layout expression has zero for
    // coefficient"* (`:86-87`).
    for loop_ in enclosing {
        layout_expr.layout_map.entry(loop_.op.clone()).or_default();
    }

    layout_expr
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 241/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// A LOOP OR CONDITIONAL REBUILT TO CARRY THE TWO XRF POINTERS — what entries 241 and 242 return.
///
/// ⭐ THE `OpBuilder` IS THE CALLER'S, so the ops the reference creates *around* the replacement
/// come back beside it: `OpBuilder builder(for_op)` inserts before the loop
/// (`LoweringXRF.cpp:131`), and [`Self::before`] is what goes there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XrfCarryingOp {
    /// The ops to splice in FRONT of the replacement — the two `sentient.scalar_constant`s of
    /// entry 241, and nothing at all for entry 242.
    pub before: Vec<sen::Op>,
    /// The replacement itself, which the caller puts where the original stood.
    pub op: sen::Op,
    /// The two results it now binds, write pointer first (`Dataflow.td`'s pointer order).
    pub pointers: XrfPtrPair,
}

/// Replaces: e241_createForOpWithReturnValue
///
/// **241/384** `LoweringXRF::createForOpWithReturnValue` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:129` (56L).
///
/// ⛔ THE TWO POINTERS ARE APPENDED, SO THEY ARE THE **LAST** TWO CARRIED VALUES — which is what
/// makes entry 175's `idx` name both a yield operand and a result (`:145-146`).
/// ⛔ THE BODY LOSES ITS `sentient.yield` AND GAINS A BARE ONE: the reference inserts an empty
/// terminator and clones every non-yield op in front of it (`:157-176`).
/// ⛔ THE LOCALE ARRAY IS THE OLD OP'S BUT `regIndices`/`programHeader` ARE **EMPTY** (`:151-153`),
/// so every carried value comes back unassigned and off the program header, and the two new
/// positions — past the end of the old array — are [`RegType::Unknown`].
#[must_use]
pub fn create_for_op_with_return_value(
    for_op: &sen::Op,
    vals: &mut Values,
) -> Option<XrfCarryingOp> {
    let sen::Op::Sentient(sentient::Op::For {
        iv,
        bound,
        bound_reg,
        carried,
        dbg_name,
        body,
    }) = for_op
    else {
        return None;
    };

    // `sentient::ConstantOp::create(builder, loc, xrf_reg_type, 0, xrf_wr_ptr_name)` then the same
    // for `xrfrdptr` — ⛔ THE LOCALE IS THE ONE THING THAT DISTINGUISHES THEM.
    let mut before = Vec::new();
    let mut inits = Vec::new();
    for locale in [sentient::RegType::XrfWrPtr, sentient::RegType::XrfRdPtr] {
        let result = vals.mint();
        before.push(sen::Op::Sentient(sentient::Op::ScalarConstant {
            value: 0,
            result,
            reg_locale: locale,
            ty: ScalarTy::Index,
            is_symbol: false,
        }));
        inits.push(result);
    }

    // `iter_args` is the old operand list with the two constants pushed on, and `ArrayRef<int32_t>()`
    // / `ArrayRef<bool>()` are what drop every index and header flag.
    let mut new_carried: Vec<sentient::Carried> = carried
        .iter()
        .map(|carried| sentient::Carried {
            reg: sentient::Reg {
                locale: carried.reg.locale,
                index: None,
            },
            program_header: false,
            ..*carried
        })
        .collect();
    let mut pointers = Vec::new();
    for init in inits {
        let result = vals.mint();
        new_carried.push(sentient::Carried {
            init,
            arg: vals.mint(),
            result,
            reg: sentient::Reg {
                locale: sentient::RegType::Unknown,
                index: None,
            },
            element_size: None,
            program_header: false,
        });
        pointers.push(result);
    }

    // `for (auto &it : for_op.getBody()->getOperations()) if (!isa<sentient::YieldOp>(&it)) clone`,
    // in front of the bare `sentient::YieldOp` the builder put there first.
    let mut new_body: Vec<sen::Op> = body
        .iter()
        .filter(|op| !matches!(op, sen::Op::Sentient(sentient::Op::Yield { .. })))
        .cloned()
        .collect();
    new_body.push(sen::Op::Sentient(sentient::Op::Yield {
        results: Vec::new(),
    }));

    Some(XrfCarryingOp {
        before,
        // ⭐ THE `IRMapping` IS THE ONE MECHANISM DROPPED HERE. The clone exists to give the new
        // loop fresh block arguments; reusing the originals keeps every
        // `for_op.getResult(i).replaceAllUsesWith(new_for_op.getResult(i))` (`:180-182`) a no-op,
        // and the old loop is erased on the next line either way.
        op: sen::Op::Sentient(sentient::Op::For {
            iv: *iv,
            bound: *bound,
            // ⭐ ENTRY 0 IS CARRIED OVER LOCALE-ONLY, exactly as every other entry is: `getRegLocales()`
            // is passed through whole and `ArrayRef<int32_t>()` drops the whole index array.
            bound_reg: bound_reg.map(|reg| sentient::Reg {
                locale: reg.locale,
                index: None,
            }),
            carried: new_carried,
            dbg_name: dbg_name.clone(),
            body: new_body,
        }),
        pointers: XrfPtrPair {
            write: pointers[0],
            read: pointers[1],
        },
    })
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 242/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e242_createIfOpWithReturnValue
///
/// **242/384** `LoweringXRF::createIfOpWithReturnValue` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:189` (56L).
///
/// ⛔ TWO RESULTS, NOT OLD PLUS TWO: `TypeRange({index, index})` REPLACES the result list (`:192`),
/// and the old `sentient.if` this runs on binds nothing.
/// ⛔ THE ELSE REGION IS CREATED UNCONDITIONALLY (`withElseRegion = true`, `:197`) and only CLONED
/// INTO when the original had one (`:226`) — so a one-armed `if` comes back with a bare
/// `sentient.yield` in its else, which in this island is what makes the region exist.
#[must_use]
pub fn create_if_op_with_return_value(if_op: &sen::Op, vals: &mut Values) -> Option<XrfCarryingOp> {
    let sen::Op::Sentient(sentient::Op::If {
        predicate,
        lhs,
        rhs,
        yielded,
        dbg_name,
        then_body,
        else_body,
    }) = if_op
    else {
        return None;
    };

    // `TypeRange({IndexType, IndexType})` with `if_op.getRegLocales()` and an EMPTY `regIndices` —
    // the OLD locale array read against the NEW two positions, so a locale the old op stated at that
    // position survives and anything past its end is [`RegType::Unknown`].
    let write = vals.mint();
    let read = vals.mint();
    let yielded = [write, read]
        .into_iter()
        .enumerate()
        .map(|(at, result)| sentient::Yielded {
            result,
            reg: sentient::Reg {
                locale: yielded
                    .get(at)
                    .map_or(sentient::RegType::Unknown, |old: &sentient::Yielded| {
                        old.reg.locale
                    }),
                index: None,
            },
            element_size: None,
        })
        .collect();

    // Both regions: the non-yield ops cloned in front of a bare terminator.
    let rebuild = |region: &[sen::Op]| {
        let mut ops: Vec<sen::Op> = region
            .iter()
            .filter(|op| !matches!(op, sen::Op::Sentient(sentient::Op::Yield { .. })))
            .cloned()
            .collect();
        ops.push(sen::Op::Sentient(sentient::Op::Yield {
            results: Vec::new(),
        }));
        ops
    };

    Some(XrfCarryingOp {
        // `OpBuilder builder(if_op)` creates nothing but the op itself.
        before: Vec::new(),
        op: sen::Op::Sentient(sentient::Op::If {
            predicate: *predicate,
            lhs: *lhs,
            rhs: *rhs,
            yielded,
            // `if (auto orig_dbg_name = dataflow::getDbgNameAttr(if_op)) setDbgNameAttr(..)`.
            dbg_name: dbg_name.clone(),
            then_body: rebuild(then_body),
            else_body: rebuild(else_body),
        }),
        pointers: XrfPtrPair { write, read },
    })
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 243/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// Replaces: e243_insertDummyMacOp
///
/// **243/384** `LoweringXRF::insertDummyMacOp` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:312` (15L).
///
/// ⛔ A PLACEHOLDER THAT MUST BE ERASED — [`replace_and_erase_dummy_mac_ops`] is the other half; a
/// surviving dummy would lower as a real MAC with all three operands on `lrf0`.
/// ⛔ IT BINDS ONE INDEX RESULT AND CARRIES NO POINTERS: `TypeRange(IndexType)` with an empty
/// `ValueRange()` (`:317-318`), which is what the xrf pointer is later threaded THROUGH.
/// ⭐ THE FOUR EMPTY `ArrayAttr`s ARE THE FORWARDING LISTS, and the defaulted attributes below them
/// are the `.td`'s own (`SentientOps.td:244-267`).
#[must_use]
pub fn insert_dummy_mac_op(vals: &mut Values) -> XrfPtrAdvance {
    let mask = vals.mint();
    let result = vals.mint();
    let lrf0 = sentient::Port::Lrf(sentient::LrfIndex::L0);

    XrfPtrAdvance {
        ops: vec![
            // `sentient::ConstantOp::create(*builder, loc, builder->getIndexType(), 0)` — no
            // locale, so the `.td`'s `imm` default.
            sen::Op::Sentient(sentient::Op::ScalarConstant {
                value: 0,
                result: mask,
                reg_locale: sentient::RegType::Imm,
                ty: ScalarTy::Index,
                is_symbol: false,
            }),
            sen::Op::Sentient(sentient::Op::VectorMac {
                mask: Some(mask),
                xrf_write_ptr: None,
                xrf_read_ptr: None,
                results: vec![result],
                op_a: sentient::Operand::from(lrf0),
                op_b: sentient::Operand::from(lrf0),
                op_c: sentient::Operand::from(lrf0),
                result: sentient::ResultPorts::default(),
                mode: sentient::FmaMode::FusedMulAdd,
                compute_precision: sentient::Precision::Fp16,
                fold_mode: None,
                unroll_factor: sentient::UnrollFactor::X1,
                xrf_read_incr: 0,
                xrf_write_incr: 0,
                data_transfer_only: false,
                is_data_weight: None,
                dbg_name: Some("LoweringXRF dummy Mac".to_owned()),
            }),
        ],
        // `return ret.getResult(0);`
        value: result,
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 345/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// `XrfPtrMap` (`VectorChainToSentientPT.hpp:44-47`) — one [`XrfPtrs`] per xrf access.
pub type XrfPtrMap = BTreeMap<OpId, XrfPtrs>;

/// One [`XrfPtrMap`] entry while the two passes fill it — the write half on `i = 0`, the read on `i = 1`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct PartialXrfPtrs {
    argument_write: Option<Val>,
    argument_read: Option<Val>,
    results_write: Option<Val>,
    results_read: Option<Val>,
}

impl PartialXrfPtrs {
    /// `map[op].at(0).at(i) = …` and `map[op].at(1).at(i) = …` — the two bookkeeping lines, which
    /// both access arms write together (`LoweringXRF.cpp:452`, `:456`, `:462`, `:465`).
    fn set(&mut self, ptr: XrfPtr, argument: Val, results: Val) {
        match ptr {
            XrfPtr::Write => {
                self.argument_write = Some(argument);
                self.results_write = Some(results);
            }
            XrfPtr::Read => {
                self.argument_read = Some(argument);
                self.results_read = Some(results);
            }
        }
    }

    /// Both passes recorded it, which is every entry once `for (int i = 0; i < 2; i++)` is done.
    fn complete(self) -> Option<XrfPtrs> {
        Some(XrfPtrs {
            argument: XrfPtrPair {
                write: self.argument_write?,
                read: self.argument_read?,
            },
            results: XrfPtrPair {
                write: self.results_write?,
                read: self.results_read?,
            },
        })
    }
}

/// `use_empty()` ON THE UNION RUNG — a rename to a fresh value that changes nothing had no reader.
///
/// ⭐ IT BORROWS [`sen::replace_all_uses_with`]'s TOTALITY rather than restating an operand census
/// that could miss one of the shared dialects — which is the failure that function's own note names.
fn use_empty(val: Val, scope: &[sen::Op], vals: &mut Values) -> bool {
    let mut probe = scope.to_vec();
    sen::replace_all_uses_with(&mut probe, val, vals.mint());
    probe.as_slice() == scope
}

/// The regions of one op on this rung, as [`XrfPass::walk`] descends them.
fn regions_of(op: &mut sen::Op) -> Vec<&mut Vec<sen::Op>> {
    match op {
        sen::Op::Sentient(inner) => sentient::regions_mut(inner),
        sen::Op::AffineFor(loop_op) => vec![&mut loop_op.body],
        _ => Vec::new(),
    }
}

/// ONE POINTER'S PASS — the reference's four stacks and its two running values.
struct XrfPass<'a> {
    /// Which pointer this pass moves — the `i` of `for (int i = 0; i < 2; i++)`.
    ptr: XrfPtr,
    /// `expr_map` = `expr_maps[i]`: the accesses this pointer is the ACTIVE one for.
    active: &'a LayoutExprMap,
    /// `expr_maps[1 - i]`: the other pointer's accesses, which this one still records and walks past.
    inactive: &'a LayoutExprMap,
    /// `region_ops_with_xrf_access` — filled by entry 367 and only READ here.
    xrf_regions: &'a BTreeSet<OpId>,
    /// The scope [`for_op_bound`] reads a loop's trip count through.
    definitions: Definitions<'a>,
    /// `xrf_incr_after_prev_mac`.
    incr_after_prev_mac: i64,
    /// `xrf_ptr_val`.
    value: Val,
    /// `xrf_access_offsets`.
    offsets: Vec<Vec<StickOffset>>,
    /// `loop_stride_step_stack`, whose `INT64_MIN` is [`None`].
    strides: Vec<Option<StickOffset>>,
    /// `if_then_region_init_ptr_stack`.
    then_init: Vec<Val>,
    /// `ops_to_delete`, named by the value each defines.
    to_delete: Vec<Val>,
    /// ⛔⛔ EVERY OP EITHER PASS HAS SPLICED IN, so the SECOND pass can walk past them. The reference
    /// keys its three maps by `Operation *` and is immune; an [`OpId`] is a POSITION, and entry 367
    /// measured all of them over the ORIGINAL body — so the read pass, which runs over the body the
    /// write pass already grew, would otherwise read a spliced-in constant at the position of the
    /// access it is looking for. ⭐ STRUCTURAL EQUALITY IDENTIFIES THEM EXACTLY: every inserted op
    /// binds freshly minted values, so none of them can equal an op that was already there.
    inserted: &'a mut Vec<sen::Op>,
    map: BTreeMap<OpId, PartialXrfPtrs>,
    vals: &'a mut Values,
}

impl XrfPass<'_> {
    /// `getOffsetFromOuterForOp` — the innermost ENCLOSING level's last offset, past empty levels.
    fn offset_from_outer_for_op(&self) -> StickOffset {
        self.offsets
            .iter()
            .rev()
            .skip(1)
            .find_map(|level| level.last().copied())
            .unwrap_or_default()
    }

    /// The type-1 scenario for the ACTIVE pointer (`LoweringXRF.cpp:426-456`) — the ops to insert in
    /// front of the access.
    fn active_access(&mut self, at: &OpId) -> Vec<sen::Op> {
        // `expr_map[op].constant_val`, `operator[]` defaulting a missing key to zero.
        let constant = self
            .active
            .get(at)
            .map_or(StickOffset(0), |e| e.constant_val);
        // ⭐ *"the default xrf ptr increment value of sentient.mac is included in the total"*.
        // ⛔⛔ THE STACK GETS `curr_const + incr` AND THE OFFSET OP GETS THE RAW `curr_const`
        // (`LoweringXRF.cpp:433-436` against `:447`): the increment is what the MAC itself will do, so
        // it counts towards the NEXT access's distance and not towards this one's.
        if let Some(level) = self.offsets.last_mut() {
            level.push(StickOffset(constant.0 + self.incr_after_prev_mac));
        }
        let previous = self.offsets.last().and_then(|level| {
            level
                .len()
                .checked_sub(2)
                .and_then(|prev| level.get(prev).copied())
        });
        let prev = previous.unwrap_or_else(|| self.offset_from_outer_for_op());

        let mut ops = Vec::new();
        let advance = insert_const_and_add_ops(
            self.vals,
            ScalarTy::Index,
            self.value,
            StickOffset(constant.0 - prev.0),
        );
        ops.extend(advance.ops);
        self.value = advance.value;
        let argument = self.value;
        let dummy = insert_dummy_mac_op(self.vals);
        ops.extend(dummy.ops);
        self.value = dummy.value;
        self.map
            .entry(at.clone())
            .or_default()
            .set(self.ptr, argument, dummy.value);
        ops
    }

    /// *"type 1 scenario for inactive xrf ptr"* (`LoweringXRF.cpp:458-465`) — the placeholder alone,
    /// with NO offset op: this pointer does not move at the other pointer's access.
    fn inactive_access(&mut self, at: &OpId) -> Vec<sen::Op> {
        let argument = self.value;
        let dummy = insert_dummy_mac_op(self.vals);
        self.value = dummy.value;
        self.map
            .entry(at.clone())
            .or_default()
            .set(self.ptr, argument, dummy.value);
        dummy.ops
    }

    /// A `sentient.for`/`sentient.if` WITH xrf accesses under it (`LoweringXRF.cpp:398-425`).
    fn enter(&mut self, at: &OpId, op: &mut sen::Op) {
        // `forop_stack.push(op)` is this recursion; the other two stacks are these.
        self.offsets.push(Vec::new());
        // ⭐ ANY entry whose `layout_map` names this loop answers — `are_xrf_accesses_legal` has
        // already agreed every access's coefficients, which is what makes the reference's `break`
        // on the first hit of an unordered map deterministic.
        self.strides.push(
            self.active
                .values()
                .find_map(|expr| expr.layout_map.get(at).copied()),
        );

        if let sen::Op::Sentient(sentient::Op::For { carried, .. }) = op {
            // `op->setOperand(1 + i, xrf_ptr_val)` — the slot entry 090's `For` arm reads back.
            if let Some(slot) = carried.get_mut(self.ptr.index()) {
                self.to_delete.push(slot.init);
                slot.init = self.value;
            }
            if let Some(carried) = xrf_value(op, None, self.ptr) {
                self.value = carried;
            }
        } else {
            // *"stack the init xrf ptr for else-region"*.
            self.then_init.push(self.value);
        }
    }

    /// The type-2 and type-3 scenarios (`LoweringXRF.cpp:466-521`) — close one region over the
    /// pointer, and for a loop return the ops that undo its travel AFTER it.
    fn close(&mut self, op: &mut sen::Op, region: YieldRegion) -> Vec<sen::Op> {
        // `dyn_cast<sentient::YieldOp>(op)` — the walk only reaches this on a terminator.
        if !matches!(
            regions_of(op)
                .into_iter()
                .nth(region.index())
                .and_then(|r| r.last()),
            Some(sen::Op::Sentient(sentient::Op::Yield { .. }))
        ) {
            return Vec::new();
        }

        let prev = self.offset_from_outer_for_op();
        // *"if there is no xrf access in current loop, no address offset is needed"*.
        let curr = self
            .offsets
            .last()
            .and_then(|level| level.last().copied())
            .unwrap_or(prev);
        let stride = self.strides.last().copied().flatten().unwrap_or_default();
        let advance = insert_const_and_add_ops(
            self.vals,
            ScalarTy::Index,
            self.value,
            StickOffset(prev.0 - curr.0 + stride.0),
        );
        self.value = advance.value;
        // `forop_builder.setInsertionPoint(op)` — in front of the terminator.
        self.inserted.extend(advance.ops.iter().cloned());
        if let Some(body) = regions_of(op).into_iter().nth(region.index()) {
            let before = body.len().saturating_sub(1);
            body.splice(before..before, advance.ops);
        }
        if let Some(returned) = update_yield_args(op, region, self.value, self.ptr) {
            self.value = returned;
        }

        if matches!(op, sen::Op::Sentient(sentient::Op::For { .. })) {
            // Type 3, loops only: `-stride * bound` undoes the travel the loop made.
            let bound = for_op_bound(op, self.definitions);
            let advance = insert_const_and_add_ops(
                self.vals,
                ScalarTy::Index,
                self.value,
                StickOffset(-stride.0 * bound.0),
            );
            self.value = advance.value;
            self.offsets.pop();
            self.strides.pop();
            return advance.ops;
        }

        match region {
            // The `then` region's accesses do not reach the `else` region, and neither does its
            // pointer: both start from the value stacked at the `sentient.if`.
            YieldRegion::BodyOrThen => {
                if let Some(level) = self.offsets.last_mut() {
                    level.clear();
                }
                if let Some(init) = self.then_init.pop() {
                    self.value = init;
                }
            }
            YieldRegion::Else => {
                self.offsets.pop();
                self.strides.pop();
            }
        }
        Vec::new()
    }

    /// `unit.walk<WalkOrder::PreOrder>` over one region.
    ///
    /// ⛔ THE ORDINAL IS THE **ORIGINAL** ONE, and the running index is not: every key in
    /// [`XrfPass::active`], [`XrfPass::inactive`] and [`XrfPass::xrf_regions`] was measured by
    /// entry 367 before a single op was inserted, so the ops this walk splices in must not shift the
    /// positions it is still looking up. Inserted ops are stepped over, never re-examined.
    fn walk(&mut self, region: &mut Vec<sen::Op>, prefix: &[u32], base: u32) {
        let mut ordinal = 0u32;
        let mut index = 0usize;
        while index < region.len() {
            // An op the other pass spliced in is not one of the positions the maps were measured
            // over — step past it without spending an ordinal on it.
            if self.inserted.contains(&region[index]) {
                index += 1;
                continue;
            }
            let mut path = prefix.to_vec();
            path.push(base + ordinal);
            let at = OpId::at(&path);

            let control = matches!(
                &region[index],
                sen::Op::Sentient(sentient::Op::For { .. } | sentient::Op::If { .. })
            );
            let member = control && self.xrf_regions.contains(&at);
            if !control {
                let before = if self.active.contains_key(&at) {
                    self.active_access(&at)
                } else if self.inactive.contains_key(&at) {
                    self.inactive_access(&at)
                } else {
                    Vec::new()
                };
                let inserted = before.len();
                self.inserted.extend(before.iter().cloned());
                region.splice(index..index, before);
                index += inserted;
            } else if member {
                self.enter(&at, &mut region[index]);
            }

            // The regions BELOW this op, at their original child positions.
            let lengths: Vec<u32> = regions_of(&mut region[index])
                .iter()
                .map(|ops| ops.len() as u32)
                .collect();
            let mut after: Vec<sen::Op> = Vec::new();
            for (child, kind) in [(0usize, YieldRegion::BodyOrThen), (1, YieldRegion::Else)] {
                if child >= lengths.len() {
                    break;
                }
                let child_base: u32 = lengths[..child].iter().sum();
                if let Some(inner) = regions_of(&mut region[index]).into_iter().nth(child) {
                    self.walk(inner, &path, child_base);
                }
                if member {
                    after.extend(self.close(&mut region[index], kind));
                }
            }
            for (child, _) in lengths.iter().enumerate().skip(2) {
                if let Some(inner) = regions_of(&mut region[index]).into_iter().nth(child) {
                    let child_base: u32 = lengths[..child].iter().sum();
                    self.walk(inner, &path, child_base);
                }
            }

            index += 1;
            let inserted = after.len();
            self.inserted.extend(after.iter().cloned());
            region.splice(index..index, after);
            index += inserted;
            ordinal += 1;
        }
    }
}

/// Replaces: e345_processXrfPtrPerUnit
///
/// **345/384** `LoweringXRF::processXrfPtrPerUnit` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:337` (190L).
///
/// ⛔ `isXrfRelated(op)` IS READ OFF THE MAPS, NOT RE-ASKED: entry 367 keys `expr_maps[0]` with
/// exactly the xrf-related STORES and `expr_maps[1]` with the LOADS (`:631-648`), which is the same
/// population the four `isa<>` tests here select — so membership in `expr_maps[i]` IS the active arm
/// and membership in `expr_maps[1 - i]` IS the inactive one.
/// ⛔ THE `use_empty` CHECK ON A LOOP'S OLD INIT MOVES TO THE END OF **BOTH** PASSES. The reference
/// erases inside `for (int i = 0; i < 2; i++)` (`:524`) and keys its map by `Operation *`, so a
/// deletion costs it nothing; entry 367 keyed these maps by POSITION, and erasing the write pass's
/// spent init renumbers every op after it — which is the read pass's whole lookup. Nothing after the
/// rewrite can add a reader, so deferring leaves the answer unchanged.
/// ⭐ THE TWO INIT CONSTANTS LAND IN REVERSE PASS ORDER, `setInsertionPointToStart` putting the read
/// pointer's in front of the write pointer's (`:376-380`).
pub fn process_xrf_ptr_per_unit<A: Arch>(
    body: &mut Vec<sen::Op>,
    precision: sentient::Precision,
    expr_maps: &XrfLayoutExprs,
    xrf_regions: &BTreeSet<OpId>,
    definitions: Definitions<'_>,
    vals: &mut Values,
) -> XrfPtrMap {
    let mut map: BTreeMap<OpId, PartialXrfPtrs> = BTreeMap::new();
    let mut prologue: Vec<sen::Op> = Vec::new();
    let mut inserted: Vec<sen::Op> = Vec::new();
    // `SmallVector<Operation *, 16> ops_to_delete;` — see this function's note on why one list
    // spans both passes here where the reference has one per pass.
    let mut to_delete: Vec<Val> = Vec::new();

    // *"work on write_ptr and read_ptr sequentially"*.
    for ptr in [XrfPtr::Write, XrfPtr::Read] {
        let other = match ptr {
            XrfPtr::Write => XrfPtr::Read,
            XrfPtr::Read => XrfPtr::Write,
        };
        // `unsigned xrf_incr_after_prev_mac = 1;` and the read pointer's own MAC stride.
        let incr_after_prev_mac = match ptr {
            XrfPtr::Write => 1,
            XrfPtr::Read => i64::from(xrf_rd_ptr_incr_val_after_mac::<A>(precision)),
        };

        // *"initialize xrf reg to zero and update xrf_access_offsets"* — no locale, so the `.td`'s
        // `imm`, which is what the commented-out `xrfwrptr`/`xrfrdptr` naming would have set anyway.
        let init = vals.mint();
        let mut pass = XrfPass {
            ptr,
            active: expr_maps.at(ptr),
            inactive: expr_maps.at(other),
            xrf_regions,
            definitions,
            incr_after_prev_mac,
            value: init,
            offsets: vec![vec![StickOffset(0)]],
            strides: Vec::new(),
            then_init: Vec::new(),
            to_delete: Vec::new(),
            inserted: &mut inserted,
            map,
            vals,
        };
        pass.walk(body, &[], 0);
        let XrfPass {
            map: filled,
            to_delete: consumed,
            ..
        } = pass;
        map = filled;
        to_delete.extend(consumed);

        prologue.splice(
            0..0,
            [sen::Op::Sentient(sentient::Op::ScalarConstant {
                value: 0,
                result: init,
                reg_locale: sentient::RegType::Imm,
                ty: ScalarTy::Index,
                is_symbol: false,
            })],
        );
    }
    // `for (Operation *op : ops_to_delete) op->erase();` (`:524`), once both passes have walked.
    for old in to_delete {
        if use_empty(old, body, vals) {
            sen::erase_defining_op(body, old);
        }
    }
    body.splice(0..0, prologue);

    map.into_iter()
        .filter_map(|(at, partial)| partial.complete().map(|ptrs| (at, ptrs)))
        .collect()
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 367/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// WHICH POINTER ONE ACCESS MOVES, AND THE VECTOR TYPE ITS STICK COUNT IS READ OFF.
///
/// ⛔ THE STORES ARE `expr_maps[0]` AND THE LOADS `expr_maps[1]` (`LoweringXRF.cpp:641-646`) —
/// *"assume that agen::vector_store is only used to store data to XRF in PT"*.
const fn xrf_access(op: &DfirOp) -> Option<(XrfPtr, Vector)> {
    match op {
        DfirOp::Agen(agen::Op::VectorStore { ty, .. })
        | DfirOp::Vector(vector::Op::Store { ty, .. }) => Some((XrfPtr::Write, *ty)),
        DfirOp::Agen(agen::Op::VectorLoad { ty, .. })
        | DfirOp::Vector(vector::Op::Load { ty, .. }) => Some((XrfPtr::Read, *ty)),
        _ => None,
    }
}

/// `dcc::utils::getStickElemNumForOp` (`dcc/src/Utils/Utils.cpp:522-536`) — a switch on the width of
/// `getElementType(op)`, which for all four accesses is their vector type's element.
///
/// ⛔ `default: return -1` IS THE `DT_CHECK_MSG` ABOVE THE CALL, so a 32-bit element is `None` here
/// even though `getBytesPerStick() * 8 / 32` is a perfectly good 32: the three widths the reference
/// names are the only ones it will lower, and on both arches those three ARE that quotient
/// (`case 16: 64`, `case 8: 128`, `case 4: 256` against 128 bytes per stick).
fn stick_elem_num<A: Arch>(ty: Vector) -> Option<ElementsPerStick> {
    match ty.elem.bits() {
        width @ (16 | 8 | 4) => NonZeroU32::new(width).and_then(ElementsPerStick::of::<A>),
        _ => None,
    }
}

/// ONE XRF ACCESS AS EVERYTHING `getLayoutExpr` AND `processXrfPtrPerUnit` ASK OF IT.
struct XrfAccess {
    /// Its position in the SEN body — the [`LayoutExprMap`] key entry 345 looks it up by.
    at: OpId,
    /// Its index in [`XrfCensus::dfir`], which is the scope every [`Val`]-keyed query reads.
    index: usize,
    /// Write for a store, read for a load — see [`xrf_access`].
    ptr: XrfPtr,
    /// `type($result)` / `type($valueToStore)`, for [`stick_elem_num`].
    ty: Vector,
    /// The `sentient.for`s around it, outermost first.
    enclosing: Vec<EnclosingLoop>,
}

/// ⛔⛔ ONE MIXED BODY READ AT TWO RUNGS AT ONCE, WHICH IS THE WHOLE SHAPE OF THIS ENTRY.
///
/// `unit.walk` runs over a region holding `sentient.for`/`sentient.if` wrapped around still-unlowered
/// `agen`/`vector`/`dataflow` accesses — a [`sen::Op`] body — while every question asked OF those
/// accesses ([`is_xrf_related`], [`layout_expr`]) is a DataflowIR one over a `&[DfirOp]`. So one walk
/// records both: the sen-body [`OpId`] that keys the maps entry 345 reads, and a FLATTENED
/// DataflowIR projection (shared-dialect ops in preorder, region contents spliced inline) that
/// resolves every [`Val`].
///
/// ⭐ AND THE PROJECTION IS SELF-CONSISTENT: `op_at` is the ONLY positional use either query makes of
/// it ([`layout_map_and_indices`] resolves the view by [`Val`] and hands back a position measured in
/// the same scope), so a flat index answers for the access and for the view it names.
#[derive(Default)]
struct XrfCensus {
    /// The projection — one entry per shared-dialect op, in walk order.
    dfir: Vec<DfirOp>,
    /// The four memory accesses, before `isXrfRelated` narrows them.
    accesses: Vec<XrfAccess>,
    /// Every `dataflow.get_logical_memory_view`'s index, for the *"is there any xrf view"* walk.
    views: Vec<usize>,
}

/// The regions below one op, in the order [`OpId`]'s flat ordinals run — the immutable twin of
/// [`regions_of`].
fn regions_at(op: &sen::Op) -> Vec<&[sen::Op]> {
    match op {
        sen::Op::Sentient(inner) => sentient::regions(inner),
        sen::Op::AffineFor(loop_op) => vec![loop_op.body.as_slice()],
        _ => Vec::new(),
    }
}

/// `unit.walk` over a mixed body, filling both halves of an [`XrfCensus`].
fn take_census(
    body: &[sen::Op],
    prefix: &[u32],
    base: u32,
    enclosing: &[EnclosingLoop],
    into: &mut XrfCensus,
) {
    for (ordinal, op) in body.iter().enumerate() {
        let mut path = prefix.to_vec();
        path.push(base + ordinal as u32);
        match op {
            // `getInductionVar()` is what a subscript's `arith.subi` names its loop by; the position
            // is the `layout_map` key. See [`EnclosingLoop`].
            sen::Op::Sentient(sentient::Op::For { iv, body, .. }) => {
                let mut inner = enclosing.to_vec();
                inner.push(EnclosingLoop {
                    op: OpId::at(&path),
                    iv: *iv,
                });
                take_census(body, &path, 0, &inner, into);
            }
            // ⛔ ONE FLAT ORDINAL SPACE ACROSS THE TWO REGIONS — [`OpId::block`]'s convention, and
            // the one entry 345's walk measures its own positions in.
            sen::Op::Sentient(sentient::Op::If {
                then_body,
                else_body,
                ..
            }) => {
                take_census(then_body, &path, 0, enclosing, into);
                take_census(else_body, &path, then_body.len() as u32, enclosing, into);
            }
            // ⛔ NOT A `sentient.for`, so it drives no layout coefficient — see [`layout_expr`]'s
            // `DT_CHECK_MSG(isa<sentient::ForOp>(for_op), ..)`.
            sen::Op::AffineFor(loop_op) => take_census(&loop_op.body, &path, 0, enclosing, into),
            other => {
                // This rung's own ops are not DataflowIR ops and take no projected index.
                let Some(op) = sen::lowered(other) else {
                    continue;
                };
                let index = into.dfir.len();
                if matches!(
                    op,
                    DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView { .. })
                ) {
                    into.views.push(index);
                }
                if let Some((ptr, ty)) = xrf_access(&op) {
                    into.accesses.push(XrfAccess {
                        at: OpId::at(&path),
                        index,
                        ptr,
                        ty,
                        enclosing: enclosing.to_vec(),
                    });
                }
                into.dfir.push(op);
            }
        }
    }
}

/// `isa<agen::VectorLoadOp, agen::VectorStoreOp, vector::LoadOp, vector::StoreOp>(each_op) &&
/// isXrfRelated(&each_op)` — the immediate-body test of `checkAndAddToRegionOpMap`.
fn is_xrf_access(op: &sen::Op, scope: &[DfirOp]) -> bool {
    sen::lowered(op).is_some_and(|op| xrf_access(&op).is_some() && is_xrf_related(&op, scope))
}

/// `checkAndAddToRegionOpMap` AND THE REBUILD IT DRIVES (`LoweringXRF.cpp:583-624`), as one
/// postorder pass over one region, answering whether that region marks the op that owns it.
///
/// ⛔ POSITIONLESS, AND IT HAS TO BE: each rebuilt loop brings two `sentient.scalar_constant`s in
/// front of it (entry 241), which moves every sibling after it — so the *"already marked"* half of
/// the test is this function's own answer for the child, and the *"is an xrf access"* half is asked
/// of the op itself.
/// ⛔ ONLY REGION 0 MARKS THE OWNER: `for_op.getBody()` and `if_op.getBody(0)` (`:585`, `:595`), so an
/// access in an `else` region is rebuilt but does not make its `sentient.if` carry the pointers.
/// ⛔ THE ERASE AT `:616`/`:621` IS NOT A RETRACTION — step 4 refills the mark over a rebuilt body
/// that still holds the same accesses, which is why a rebuilt op marks its owner here.
fn mark_and_rebuild(region: &mut Vec<sen::Op>, scope: &[DfirOp], vals: &mut Values) -> bool {
    let mut marks_owner = false;
    let mut index = 0usize;
    while index < region.len() {
        // `unit.walk<WalkOrder::PostOrder>` — everything below this op first.
        let mut member = false;
        for (child, inner) in regions_of(&mut region[index]).into_iter().enumerate() {
            let holds = mark_and_rebuild(inner, scope, vals);
            if child == 0 {
                member = holds;
            }
        }

        let control = matches!(
            &region[index],
            sen::Op::Sentient(sentient::Op::For { .. } | sentient::Op::If { .. })
        );
        if control && member {
            // `createForOpWithReturnValue(for_op)` / `createIfOpWithReturnValue(if_op)`.
            let rebuilt = match &region[index] {
                op @ sen::Op::Sentient(sentient::Op::For { .. }) => {
                    create_for_op_with_return_value(op, vals)
                }
                op => create_if_op_with_return_value(op, vals),
            };
            if let Some(XrfCarryingOp { before, op, .. }) = rebuilt {
                // `OpBuilder builder(for_op)` — the two init constants go in front of it.
                let inserted = before.len();
                region.splice(index..index, before);
                index += inserted;
                region[index] = op;
            }
            marks_owner = true;
        } else if !control && is_xrf_access(&region[index], scope) {
            marks_owner = true;
        }
        index += 1;
    }
    marks_owner
}

/// `checkAndAddToRegionOpMap` OVER A STABLE BODY — step 4's *"reflll after forOp clone"* (`:625`),
/// whose answers are the POSITIONS entry 345 looks `region_ops_with_xrf_access` up by.
fn mark_regions(
    body: &[sen::Op],
    prefix: &[u32],
    base: u32,
    accesses: &BTreeSet<OpId>,
    marked: &mut BTreeSet<OpId>,
) -> bool {
    let mut marks_owner = false;
    for (ordinal, op) in body.iter().enumerate() {
        let mut path = prefix.to_vec();
        path.push(base + ordinal as u32);
        let mut member = false;
        let mut child_base = 0u32;
        for (child, inner) in regions_at(op).iter().enumerate() {
            let holds = mark_regions(inner, &path, child_base, accesses, marked);
            if child == 0 {
                member = holds;
            }
            child_base += inner.len() as u32;
        }
        let at = OpId::at(&path);
        if matches!(
            op,
            sen::Op::Sentient(sentient::Op::For { .. } | sentient::Op::If { .. })
        ) {
            if member {
                marked.insert(at);
                marks_owner = true;
            }
        } else if accesses.contains(&at) {
            marks_owner = true;
        }
    }
    marks_owner
}

/// Replaces: e367_createXrfIndexModifOps
///
/// **367/384** `LoweringXRF::createXrfIndexModifOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:564` (97L).
///
/// ⛔ FIVE WALKS, AND THE FOURTH REPEATS THE SECOND — *"reflll after forOp clone"* (`:625`): the
/// rebuild erases every mark it consumed, so the regions entry 345 reads are the ones measured over
/// the FINISHED body. ⛔ `emitError("XRF accesses are illegal")` (`:657`) LEAVES THE MAP
/// DEFAULT-CONSTRUCTED — an empty map is a FAILED pass here, not an absent one — and
/// `DT_CHECK(comp == PT)` is unrepresentable for [`compute_unit_precision`](super::vc_vector_chain_to_sentient_pt::compute_unit_precision)'s reason.
/// ⛔ See [`XrfCensus`] for the two rungs one walk has to answer on.
#[must_use]
pub fn create_xrf_index_modif_ops<A: Arch>(
    body: &mut Vec<sen::Op>,
    precision: sentient::Precision,
    definitions: Definitions<'_>,
    vals: &mut Values,
) -> XrfPtrMap {
    // *"if there is no xrf memory view, skip this pass"* (`:572-579`).
    let census = {
        let mut census = XrfCensus::default();
        take_census(body, &[], 0, &[], &mut census);
        census
    };
    if !census
        .views
        .iter()
        .any(|&view| is_xrf_related(&census.dfir[view], &census.dfir))
    {
        return XrfPtrMap::new();
    }

    // `:607-623` — *"find out which forOp needs to be recreated to add xrf iter_args"*, then
    // *"add iter_args and return values for forop loops"*.
    mark_and_rebuild(body, &census.dfir, vals);

    // `:625-627` and `:629-648` — the refill and the access census, over the rebuilt body.
    let census = {
        let mut census = XrfCensus::default();
        take_census(body, &[], 0, &[], &mut census);
        census
    };
    let accesses: Vec<&XrfAccess> = census
        .accesses
        .iter()
        .filter(|access| is_xrf_related(&census.dfir[access.index], &census.dfir))
        .collect();
    let mut xrf_regions = BTreeSet::new();
    mark_regions(
        body,
        &[],
        0,
        &accesses.iter().map(|access| access.at.clone()).collect(),
        &mut xrf_regions,
    );

    let mut expr_maps = XrfLayoutExprs::default();
    for access in accesses {
        let Some(stick_elem_num) = stick_elem_num::<A>(access.ty) else {
            todo!(
                "createXrfIndexModifOps: DT_CHECK_MSG(stick_elem_num != -1, \"Could not compute \
                 stick element number\") on a {:?} access (LoweringXRF.cpp:638-639)",
                access.ty
            );
        };
        // `getLayoutExpr(op, stick_elem_num)` — entry 240, over the projection.
        let expr = layout_expr(
            &OpId::at(&[access.index as u32]),
            &census.dfir,
            &access.enclosing,
            stick_elem_num,
        );
        match access.ptr {
            XrfPtr::Write => expr_maps.write.insert(access.at.clone(), expr),
            XrfPtr::Read => expr_maps.read.insert(access.at.clone(), expr),
        };
    }

    // *"insert xrf ptr manipulation operations"* (`:650-659`).
    if expr_maps.write.is_empty() && expr_maps.read.is_empty() {
        return XrfPtrMap::new();
    }
    if !are_xrf_accesses_legal(&expr_maps) {
        return XrfPtrMap::new();
    }
    process_xrf_ptr_per_unit::<A>(body, precision, &expr_maps, &xrf_regions, definitions, vals)
}

#[cfg(test)]
mod xrf_lowering_unit_tests {
    use super::*;
    use crate::arch::Dd2;
    use crate::islands::dataflow_ir::dialects::Index;
    use crate::islands::dataflow_ir::ty::{AffineExpr, AffineMap, ElemType, MemRef, Vector};
    use core::num::NonZeroU32;

    /// 🎯 240/384 — ONE SUBSCRIPT DRIVEN BY A LOOP, AND A BASE ADDRESS THAT IS NOT ZERO.
    ///
    /// The `arith.subi`-on-an-enclosing-loop's-induction-variable subscript of
    /// `dynamic_pt_masking.mlir:226-256`; the vendor's case is a `vector.store` over a `%c0` base, so
    /// the load and the non-zero base here are this test's own variation. fp16 puts 64 elements in a
    /// stick, so a stride of 64 elements is ONE stick and a base of 128 is TWO.
    ///
    /// ⭐ AND THE SECOND ENCLOSING LOOP, WHICH NO SUBSCRIPT READS, COMES BACK ZERO — the second pass.
    #[test]
    fn a_loop_driven_subscript_is_one_stick_and_an_unread_loop_is_zero() {
        let per_stick = ElementsPerStick::of::<Dd2>(NonZeroU32::new(16).unwrap()).unwrap();
        let flat = MemRef {
            shape: vec![256],
            elem: ElemType::F16,
        };
        let scope = vec![
            DfirOp::Arith(arith::Op::Constant {
                result: Val(1),
                value: 128,
            }),
            DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                result: Val(2),
                from: Val(0),
                start: Val(1),
                layout: AffineMap {
                    dims: 1,
                    syms: 0,
                    results: vec![AffineExpr::dim(0).times(64)],
                },
                ty: flat.clone(),
            }),
            DfirOp::Arith(arith::Op::SubI(arith::IntBinary {
                result: Val(3),
                lhs: Val(4),
                rhs: Val(5),
                ty: ScalarTy::Index,
            })),
            DfirOp::Vector(vector::Op::Load {
                result: Val(6),
                base: Val(2),
                indices: vec![Index::Val(Val(3))],
                base_ty: flat,
                ty: Vector {
                    len: 64,
                    elem: ElemType::F16,
                },
            }),
        ];
        let enclosing = [
            EnclosingLoop {
                op: OpId::at(&[0]),
                iv: Val(5),
            },
            EnclosingLoop {
                op: OpId::at(&[1]),
                iv: Val(9),
            },
        ];

        assert_eq!(
            layout_expr(&OpId::at(&[3]), &scope, &enclosing, per_stick),
            LayoutExpr {
                layout_map: BTreeMap::from([
                    (OpId::at(&[0]), StickOffset(1)),
                    (OpId::at(&[1]), StickOffset(0)),
                ]),
                constant_val: StickOffset(2),
            }
        );
    }

    /// 🎯 241/384 — THE TWO POINTERS ARE APPENDED, THE INITS ARE THE TWO NEW CONSTANTS, THE BODY'S
    /// OWN `sentient.yield` IS GONE, AND THE SURVIVING CARRIED VALUE KEEPS ONLY ITS LOCALE — the
    /// empty `regIndices`/`programHeader` arrays.
    #[test]
    fn a_for_carries_two_more_values_initialised_by_the_two_locale_constants() {
        let mut vals = Values::default();
        let carried = sentient::Carried {
            init: Val(1),
            arg: Val(2),
            result: Val(3),
            reg: sentient::Reg {
                locale: sentient::RegType::Lrf,
                index: Some(sentient::RegIndex::at::<3>()),
            },
            program_header: true,
            element_size: None,
        };
        let for_op = sen::Op::Sentient(sentient::Op::For {
            iv: Val(4),
            bound: Val(5),
            bound_reg: None,
            carried: vec![carried],
            dbg_name: Some("SCF-For #1".to_owned()),
            body: vec![sen::Op::Sentient(sentient::Op::Yield {
                results: vec![Val(3)],
            })],
        });

        let built = create_for_op_with_return_value(&for_op, &mut vals).expect("a `sentient.for`");
        let locales: Vec<sentient::RegType> = built
            .before
            .iter()
            .map(|op| match op {
                sen::Op::Sentient(sentient::Op::ScalarConstant { reg_locale, .. }) => *reg_locale,
                other => panic!("the two pointer constants, not {other:?}"),
            })
            .collect();
        assert_eq!(
            locales,
            vec![sentient::RegType::XrfWrPtr, sentient::RegType::XrfRdPtr]
        );

        let sen::Op::Sentient(sentient::Op::For {
            carried: new_carried,
            dbg_name,
            body,
            ..
        }) = &built.op
        else {
            panic!("a `sentient.for` comes back")
        };
        assert_eq!(
            new_carried[0],
            sentient::Carried {
                reg: sentient::Reg {
                    locale: sentient::RegType::Lrf,
                    index: None,
                },
                program_header: false,
                ..carried
            }
        );
        assert_eq!(
            (new_carried[1].result, new_carried[2].result),
            (built.pointers.write, built.pointers.read)
        );
        assert_eq!(
            (new_carried[1].init, new_carried[2].init),
            (
                sen::results(&built.before[0])[0],
                sen::results(&built.before[1])[0]
            )
        );
        assert_eq!(new_carried[2].reg.locale, sentient::RegType::Unknown);
        assert_eq!(dbg_name.as_deref(), Some("SCF-For #1"));
        assert_eq!(
            body,
            &vec![sen::Op::Sentient(sentient::Op::Yield {
                results: Vec::new()
            })]
        );
    }

    /// 🎯 242/384 — THE RESULT LIST IS **REPLACED** BY THE TWO POINTERS, THE OLD LOCALE ARRAY IS READ
    /// AGAINST THE NEW POSITIONS, AND THE ELSE REGION EXISTS EVEN THOUGH THE ORIGINAL HAD NONE.
    #[test]
    fn an_if_binds_exactly_the_two_pointers_and_always_gains_an_else() {
        let mut vals = Values::default();
        let if_op = sen::Op::Sentient(sentient::Op::If {
            predicate: sentient::CmpPredicate::Eq,
            lhs: Val(1),
            rhs: Val(2),
            yielded: vec![sentient::Yielded {
                result: Val(3),
                reg: sentient::Reg {
                    locale: sentient::RegType::Lrf,
                    index: None,
                },
                element_size: None,
            }],
            dbg_name: Some("SCF-If #2".to_owned()),
            then_body: vec![sen::Op::Sentient(sentient::Op::Yield {
                results: vec![Val(3)],
            })],
            else_body: Vec::new(),
        });

        let built = create_if_op_with_return_value(&if_op, &mut vals).expect("a `sentient.if`");
        assert_eq!(built.before, Vec::new());
        let sen::Op::Sentient(sentient::Op::If {
            yielded,
            dbg_name,
            then_body,
            else_body,
            ..
        }) = &built.op
        else {
            panic!("a `sentient.if` comes back")
        };
        let bare = vec![sen::Op::Sentient(sentient::Op::Yield {
            results: Vec::new(),
        })];
        assert_eq!(
            yielded.iter().map(|y| y.result).collect::<Vec<_>>(),
            vec![built.pointers.write, built.pointers.read]
        );
        assert_eq!(
            (yielded[0].reg.locale, yielded[1].reg.locale),
            (sentient::RegType::Lrf, sentient::RegType::Unknown)
        );
        assert_eq!(dbg_name.as_deref(), Some("SCF-If #2"));
        assert_eq!((then_body, else_body), (&bare, &bare));
    }

    /// 🎯 243/384 — A MAC ON `lrf0` THREE TIMES OVER, MASKED BY A ZERO IMMEDIATE, AND THE VALUE
    /// HANDED BACK IS ITS RESULT.
    #[test]
    fn the_dummy_mac_reads_lrf0_three_times_and_yields_its_own_result() {
        let mut vals = Values::default();
        let advance = insert_dummy_mac_op(&mut vals);
        let [
            sen::Op::Sentient(sentient::Op::ScalarConstant {
                value,
                result: mask,
                reg_locale,
                ty,
                is_symbol,
            }),
            sen::Op::Sentient(sentient::Op::VectorMac {
                mask: mac_mask,
                results,
                op_a,
                op_b,
                op_c,
                dbg_name,
                ..
            }),
        ] = advance.ops.as_slice()
        else {
            panic!("a constant then a mac, not {:?}", advance.ops)
        };
        assert_eq!(
            (*value, *reg_locale, *ty, *is_symbol),
            (0, sentient::RegType::Imm, ScalarTy::Index, false)
        );
        assert_eq!(*mac_mask, Some(*mask));
        let lrf0 = sentient::Operand::from(sentient::Port::Lrf(sentient::LrfIndex::L0));
        assert_eq!((op_a, op_b, op_c), (&lrf0, &lrf0, &lrf0));
        assert_eq!(dbg_name.as_deref(), Some("LoweringXRF dummy Mac"));
        assert_eq!(results, &vec![advance.value]);
    }
    // ══════════════════════════════════════════════════════════════════════════════════════════
    // 345/384 — `processXrfPtrPerUnit`
    // ══════════════════════════════════════════════════════════════════════════════════════════

    /// 🎯 345/384 — ONE XRF STORE AND ONE XRF LOAD, AND BOTH PASSES OVER THE SAME BODY.
    ///
    /// ⛔ THE STORE'S ADVANCE IS ITS RAW `constant_val` — 2 sticks, not `2 + 1`: the MAC's own
    /// increment goes on the offset stack for the NEXT access (`LoweringXRF.cpp:429-446`).
    /// ⛔ AND THE READ PASS STILL FINDS THE LOAD AT `[1]` after the write pass grew the body.
    #[test]
    fn the_write_pass_advances_to_its_store_and_the_read_pass_still_finds_the_load() {
        let mut vals = Values::default();
        let (view, stored) = (vals.mint(), vals.mint());
        let mut body = vec![
            sen::Op::Agen(agen::Op::VectorStore {
                dbg_name: None,
                value: stored,
                view,
                indices: vec![Index::Const(0)],
                view_ty: MemRef {
                    shape: vec![256],
                    elem: ElemType::F16,
                },
                ty: Vector {
                    len: 64,
                    elem: ElemType::F16,
                },
            }),
            sen::Op::Agen(agen::Op::VectorLoad {
                dbg_name: None,
                result: vals.mint(),
                view,
                indices: vec![Index::Const(0)],
                view_ty: MemRef {
                    shape: vec![256],
                    elem: ElemType::F16,
                },
                ty: Vector {
                    len: 64,
                    elem: ElemType::F16,
                },
                multicast_info: None,
            }),
        ];
        let at = |path: &[u32], constant| {
            (
                OpId::at(path),
                LayoutExpr {
                    layout_map: BTreeMap::new(),
                    constant_val: StickOffset(constant),
                },
            )
        };
        let expr_maps = XrfLayoutExprs {
            write: LayoutExprMap::from([at(&[0], 2)]),
            read: LayoutExprMap::from([at(&[1], 0)]),
        };

        let map = process_xrf_ptr_per_unit::<Dd2>(
            &mut body,
            sentient::Precision::Fp16,
            &expr_maps,
            &BTreeSet::new(),
            Definitions::from_innermost(&[]),
            &mut vals,
        );

        let shape: Vec<String> = body
            .iter()
            .map(|op| match op {
                sen::Op::Sentient(sentient::Op::ScalarConstant { value, .. }) => {
                    format!("const {value}")
                }
                sen::Op::Sentient(sentient::Op::ScalarAdd { .. }) => "add".to_owned(),
                sen::Op::Sentient(sentient::Op::VectorMac { .. }) => "dummy mac".to_owned(),
                sen::Op::Agen(agen::Op::VectorStore { .. }) => "store".to_owned(),
                sen::Op::Agen(agen::Op::VectorLoad { .. }) => "load".to_owned(),
                other => format!("{other:?}"),
            })
            .collect();
        assert_eq!(
            shape,
            [
                // ⭐ THE READ POINTER'S INIT IN FRONT OF THE WRITE POINTER'S — `:376-380` inserts
                // each at the start of the region, so the second pass's lands first.
                "const 0",
                "const 0",
                // The write pointer travels its 2 sticks and leaves a placeholder at its store …
                "const 2",
                "add",
                "const 0",
                "dummy mac",
                // … where the read pointer places a placeholder and does NOT move.
                "const 0",
                "dummy mac",
                "store",
                // And at the load the two swap roles, the read pointer's advance being 0 - 0.
                "const 0",
                "dummy mac",
                "const 0",
                "dummy mac",
                "load",
            ]
        );
        // `map[op].at(0).at(0)` is the value the store's own advance produced.
        let sen::Op::Sentient(sentient::Op::ScalarAdd { result, .. }) = &body[3] else {
            panic!("the write pointer's advance, not {:?}", body[3])
        };
        assert_eq!(map.len(), 2);
        assert_eq!(map[&OpId::at(&[0])].argument.write, *result);
    }
    // ══════════════════════════════════════════════════════════════════════════════════════════
    // 367/384 — `createXrfIndexModifOps`
    // ══════════════════════════════════════════════════════════════════════════════════════════

    /// 🎯 367/384 — ONE `agen.vector_store` THROUGH A `ptxrf` VIEW INSIDE A LOOP, WHICH IS EVERY STEP.
    ///
    /// The view makes `use_xrf` true, the store marks the loop, the loop comes back carrying the two
    /// pointers — and the key entry 345 fills is the store's position AFTER the two init constants
    /// entry 241 puts in front of that loop moved it from `[3, 0]` to `[5, 0]`.
    #[test]
    fn a_store_through_a_ptxrf_view_rebuilds_its_loop_and_keys_the_map_by_the_moved_position() {
        let mut vals = Values::default();
        let (unit, start, view, stored) = (vals.mint(), vals.mint(), vals.mint(), vals.mint());
        let flat = MemRef {
            shape: vec![256],
            elem: ElemType::F16,
        };
        // ⛔ THE BOUND IS ABOVE THE UNIT, as entry 091 reads it — `Definitions` is the enclosing
        // regions, and the body itself is the one region it cannot be handed.
        let bound = vec![sen::Op::Arith(arith::Op::Constant {
            result: Val(0),
            value: 4,
        })];
        let mut body = vec![
            sen::Op::Dataflow(dataflow::Op::GetLocalUnit {
                result: unit,
                of: Val(0),
                which: LocalUnit::PtXrf,
            }),
            sen::Op::Arith(arith::Op::Constant {
                result: start,
                value: 0,
            }),
            sen::Op::Dataflow(dataflow::Op::GetLogicalMemoryView {
                result: view,
                from: unit,
                start,
                layout: AffineMap {
                    dims: 1,
                    syms: 0,
                    results: vec![AffineExpr::dim(0).times(64)],
                },
                ty: flat.clone(),
            }),
            sen::Op::Sentient(sentient::Op::For {
                iv: vals.mint(),
                bound: Val(0),
                bound_reg: None,
                carried: Vec::new(),
                dbg_name: None,
                body: vec![
                    sen::Op::Agen(agen::Op::VectorStore {
                        dbg_name: None,
                        value: stored,
                        view,
                        indices: vec![Index::Const(0)],
                        view_ty: flat,
                        ty: Vector {
                            len: 64,
                            elem: ElemType::F16,
                        },
                    }),
                    sen::Op::Sentient(sentient::Op::Yield {
                        results: Vec::new(),
                    }),
                ],
            }),
        ];

        let map = create_xrf_index_modif_ops::<Dd2>(
            &mut body,
            sentient::Precision::Fp16,
            Definitions::from_innermost(&[&bound]),
            &mut vals,
        );

        assert_eq!(
            map.keys().cloned().collect::<Vec<OpId>>(),
            vec![OpId::at(&[5, 0])]
        );
        assert_eq!(
            body.iter().find_map(|op| match op {
                sen::Op::Sentient(sentient::Op::For { carried, .. }) => Some(carried.len()),
                _ => None,
            }),
            Some(2)
        );
    }
}
