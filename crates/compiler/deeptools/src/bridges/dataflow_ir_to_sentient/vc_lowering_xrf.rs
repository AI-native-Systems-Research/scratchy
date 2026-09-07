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

use crate::arch::{Arch, IsaGen};
use crate::islands::sentient::dialects::{self as sen, Definitions, Val, arith, sentient, symbol};

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


#[cfg(test)]
mod unit_tests {
    use super::{
        DummyMacPtrs, ForOpBound, MacXrfIncrements, XrfPtr, XrfPtrPair, XrfPtrs, for_op_bound,
        replace_and_erase_dummy_mac_ops, set_sentient_mac_xrf_reg_incr_attr, xrf_value,
        xrf_rd_ptr_incr_val_after_mac,
    };
    use crate::arch::{Dd2, Sen1p5};
    use crate::islands::dataflow_ir::ty::ScalarTy;
    use crate::islands::sentient::dialects::{
        self as sen, Definitions, Val, arith, sentient, symbol,
    };

    /// A `sentient.for` carrying the two xrf pointers, write then read.
    fn loop_carrying(bound: Val, iv: Val, ptrs: [(Val, Val, Val); 2]) -> sen::Op {
        sen::Op::Sentient(sentient::Op::For {
            iv,
            bound,
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
            [
                (Val(2), Val(3), Val(4)),
                (Val(5), Val(6), Val(7)),
            ],
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
            [
                (Val(2), Val(3), Val(4)),
                (Val(5), Val(6), Val(7)),
            ],
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
        let for_op = loop_carrying(Val(0), Val(1), [
            (Val(2), Val(3), Val(4)),
            (Val(5), Val(6), Val(7)),
        ]);

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
        let for_op = loop_carrying(Val(4), Val(5), [
            (Val(6), Val(7), Val(8)),
            (Val(9), Val(10), Val(11)),
        ]);

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
        let for_op = loop_carrying(Val(4), Val(5), [
            (Val(6), Val(7), Val(8)),
            (Val(9), Val(10), Val(11)),
        ]);

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
                value: 0,
                result: Val(0),
                reg_locale: sentient::RegType::Imm,
                ty: ScalarTy::Index,
            }),
            sen::Op::Sentient(sentient::Op::ScalarConstant {
                value: 0,
                result: Val(1),
                reg_locale: sentient::RegType::Imm,
                ty: ScalarTy::Index,
            }),
            sen::Op::Sentient(real.clone()),
            // The reader: a loop carrying both placeholders in.
            loop_carrying(Val(2), Val(3), [
                (Val(0), Val(4), Val(5)),
                (Val(1), Val(6), Val(7)),
            ]),
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
}
