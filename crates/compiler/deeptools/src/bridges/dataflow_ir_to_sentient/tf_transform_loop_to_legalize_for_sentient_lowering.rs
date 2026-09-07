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

//! `TransformLoopToLegalizeForSentientLowering.cpp` — 6 of bridge 2's 384 functions (dependency level(s) [0, 1, 2, 3]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e117_transformSCFToAffineLoop` | 117/384 | 44 | `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:102` |
//! | `e194_analyzeLoop` | 194/384 | 127 | `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:268` |
//! | `e195_transformLoop` | 195/384 | 38 | `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:399` |
//! | `e257_analyzeAndTransform` | 257/384 | 3 | `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:441` |
//! | `e292_transformSCFLoopWithNonConstantUpperBound` | 292/384 | 94 | `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:169` |
//! | `e293_runOn` | 293/384 | 5 | `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:447` |

use crate::islands::dataflow_ir::dialects::{Op as DfirOp, Val, affine, scf};
use crate::islands::dataflow_ir::{ValueMapping, Values};

/// AN `scf.for` AS THIS REWRITE READS IT — the `dyn_cast`, and only the fields it touches.
///
/// ⛔ NO BOUNDS. `transformSCFToAffineLoop` never asks the loop what its bounds are: it takes
/// `lbound`, `ubound` and `step` as parameters ([`StaticBounds`]) because the whole point of the pass
/// is that the `scf.for`'s upper bound is NOT constant — an `arith.select` between two chunk widths
/// (`scf_loop_with_result.mlir:128`) — and the caller has already resolved it per branch. Reading
/// the loop's own `hi` here would read the `select`, which is exactly the value being eliminated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScfForOp<'a> {
    /// `scf_for.getInductionVar()`.
    pub iv: Val,
    /// `scf_for.getInits()` / `getRegionIterArgs()` / the loop's results — one record per carried
    /// value, which is the same [`affine::Carried`] the `affine.for` will hold.
    pub carried: &'a [affine::Carried],
    /// The region's single block, terminator included.
    pub body: &'a [DfirOp],
    /// `getDbgNameAttr(scf_for)`.
    pub dbg_name: Option<&'a str>,
}

impl<'a> ScfForOp<'a> {
    /// `dyn_cast<scf::ForOp>` — and nothing more.
    #[must_use]
    pub fn of(op: &'a DfirOp) -> Option<ScfForOp<'a>> {
        let DfirOp::Scf(scf::Op::For {
            iv,
            carried,
            body,
            dbg_name,
            ..
        }) = op
        else {
            return None;
        };
        Some(ScfForOp {
            iv: *iv,
            carried,
            body,
            dbg_name: dbg_name.as_deref(),
        })
    }

    /// THE BODY WITHOUT ITS TERMINATOR, AND THE TERMINATOR'S OPERANDS — `Block::without_terminator()`
    /// and `Block::getTerminator()`, which this rewrite uses in that order.
    ///
    /// ⭐ A TRAILING [`scf::Op::Yield`] IS THE TERMINATOR AND THERE IS NO OTHER KIND HERE. An
    /// `scf.for` carrying nothing has an IMPLICIT `scf.yield` in MLIR and no `Op::Yield` in this
    /// island, so a body that ends in something else is a loop that carries nothing — `None`
    /// operands, and every op is cloned.
    #[must_use]
    fn split_terminator(self) -> (&'a [DfirOp], Option<&'a [Val]>) {
        match self.body.split_last() {
            Some((DfirOp::Scf(scf::Op::Yield { operands }), rest)) => (rest, Some(operands)),
            _ => (self.body, None),
        }
    }
}

/// THE BOUNDS AND STEP THE CALLER RESOLVED — `int lbound, int ubound, int step`.
///
/// ⛔ NAMED FIELDS BECAUSE THREE POSITIONAL `int`s TRANSPOSE SILENTLY. `transformSCFToAffineLoop(builder,
/// scf_for, 0, then_ub, 1, affine_for)` (`:222`) is three integers in a row and getting them out of
/// order builds a loop that counts to the step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StaticBounds {
    /// `lbound` — must be 0 for the rewrite to fire.
    pub lbound: i64,
    /// `ubound` — the resolved trip bound of this branch.
    pub ubound: i64,
    /// `step` — must be 1 for the rewrite to fire.
    pub step: i64,
}

/// `LogicalResult`, WITH THE REASON THE FAILURE CARRIES IMPLICITLY.
///
/// ⛔ NOT A `Result`, AND NOT A REFUSAL. `LogicalResult::failure()` here is *"we currently support
/// lbound being 0, step being 1"* — a statement about the loop, which the caller answers by leaving
/// the `scf.for` alone and reporting *"Unable to transform SCF loop into Affine loop"* (`:255`). The
/// bounds that failed are named because a caller that has to explain itself should not have to
/// re-derive them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoopLegalization {
    /// `LogicalResult::success()`, with the `affine.for` the reference returned through
    /// `affine::AffineForOp& affine_for`.
    Transformed(DfirOp),
    /// `LogicalResult::failure()` — the lower bound is not 0, or the step is not 1.
    UnsupportedBounds {
        /// The lower bound as passed.
        lbound: i64,
        /// The step as passed.
        step: i64,
    },
}

/// Replaces: e117_transformSCFToAffineLoop
///
/// **117/384** `TransformLoopToLegalizeForSentientLowering::transformSCFToAffineLoop` —
/// `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:102` (44L).
///
/// ```cpp
/// // This method aims at creating an Affine For loop with bounds and step sizes
/// // being the parameters valued passed as an input, and the body of affine
/// // for-op is constructed from the scf_for.
/// LogicalResult
/// TransformLoopToLegalizeForSentientLowering::transformSCFToAffineLoop(
///     OpBuilder builder, scf::ForOp scf_for, int lbound, int ubound, int step,
///     affine::AffineForOp& affine_for) {
///   // we currently support lbound being 0, step being 1, non-constant ubound.
///   if (lbound == 0 && step == 1) {
///     affine_for = affine::AffineForOp::create(builder, builder.getUnknownLoc(),
///                                              0, ubound, 1, scf_for.getInits());
///
///     if (auto dbg_name_attr = getDbgNameAttr(scf_for))
///       setDbgNameAttr(affine_for, dbg_name_attr);
///
///     // Map induction var, region iterator arguments to affine loop variables.
///     IRMapping bv_map;
///     bv_map.map(scf_for.getInductionVar(), affine_for.getInductionVar());
///     for (unsigned i = 0; i < scf_for.getNumRegionIterArgs(); i++) {
///       bv_map.map(scf_for.getRegionIterArgs()[i],
///                  affine_for.getRegionIterArgs()[i]);
///     }
///
///     // Set insertion point to the beginning of the loop.
///     builder.setInsertionPointToStart(&affine_for.getRegion().front());
///
///     // Clone each operation within the body except the terminator.
///     for (auto& op : scf_for.getRegion().front().without_terminator()) {
///       builder.clone(op, bv_map);
///     }
///
///     // affine_for already has an implicit affine.yield. We need to update it if
///     // there are results.
///     if (scf_for.getNumResults() > 0) {
///       // Has results, need to update the yield with mapped operands
///       auto* scf_yield = scf_for.getRegion().front().getTerminator();
///       SmallVector<Value> mapped_operands;
///       for (auto operand : scf_yield->getOperands()) {
///         mapped_operands.push_back(bv_map.lookup(operand));
///       }
///
///       builder.setInsertionPointToEnd(&affine_for.getRegion().front());
///       builder.create<affine::AffineYieldOp>(affine_for->getLoc(),
///                                             mapped_operands);
///     }
///
///     return LogicalResult::success();
///   }
///
///   return LogicalResult::failure();
/// }
/// ```
///
/// # ⭐⭐ WHY AN `scf.for` MUST BECOME AN `affine.for` AT ALL
///
/// Everything downstream of this pass is affine: `AccessDetailsAffine` walks `affine.for`s to derive a
/// transfer's coefficients, and the Sentient loop lowering counts `affine.for` nesting against
/// [`crate::arch::Arch::MAX_NESTED_LOOPS`]. An `scf.for` reaching the lowering is not a legal input,
/// which is what this pass's name says. The reason one is there in the first place is a bound that is
/// not a constant — `%10 = arith.select %9, %c16, %c32` (`scf_loop_with_result.mlir:128`) — and
/// [`crate::islands::dataflow_ir::dialects::affine::Bound`] has no variant for a computed value
/// because `affine.for` has no way to take one.
///
/// # ⭐ `0` AND `1` ARE PASSED LITERALLY, NOT `lbound` AND `step`
///
/// `AffineForOp::create(builder, loc, 0, ubound, 1, inits)` — the guarded values are the only ones the
/// build could use, and passing them again would suggest otherwise. This island's `affine.for` has no
/// step field at all (a step of 1 is what `affine.for %i = 0 to N` means), so the `1` is discharged by
/// the type.
///
/// # ⛔ THE INITS PASS THROUGH UNCHANGED, AND THAT IS LOAD-BEARING
///
/// `scf_for.getInits()` is evaluated OUTSIDE the loop, so those values are not in the mapping and must
/// not be renumbered — the vendor's expectation shows the same `%20` initialising both branches'
/// copies (`scf_loop_with_result.mlir:41` and `:64`). Only the induction variable, the region
/// arguments and whatever the body itself defines get new names.
///
/// # ⛔ THE MAPPING IS SEEDED BEFORE THE CLONE, WHICH IS WHY THE ORDER HERE IS FIXED
///
/// Mint the results, then the induction variable, then the region arguments — MLIR's own definition
/// order, and therefore its print order — seed the mapping with `(source iv, new iv)` and each
/// `(source arg, new arg)`, and only then clone. A body op reads the induction variable, so a clone
/// that ran first would carry the old name.
///
/// # ⚠️ `bv_map.lookup` ASSERTS; THIS PORT PASSES THE VALUE THROUGH
///
/// `IRMapping::lookup` on an unmapped value returns null and MLIR then fails a verifier;
/// [`ValueMapping::lookup_or_default`] returns the value itself. The two agree on every yield operand
/// the body defines, and where the yield forwards a value from OUTSIDE the loop — a legal `scf.for`
/// (`%c2048` yielded straight through) — passing it through is the only answer that is not a stop.
/// This crate never asserts, so it is the one taken.
#[must_use]
pub fn transform_scf_to_affine_loop(
    vals: &mut Values,
    scf_for: &ScfForOp<'_>,
    bounds: StaticBounds,
) -> LoopLegalization {
    // `if (lbound == 0 && step == 1)` — *"we currently support lbound being 0, step being 1,
    // non-constant ubound."*
    if bounds.lbound != 0 || bounds.step != 1 {
        // `return LogicalResult::failure();`
        return LoopLegalization::UnsupportedBounds {
            lbound: bounds.lbound,
            step: bounds.step,
        };
    }

    // `affine::AffineForOp::create(builder, loc, 0, ubound, 1, scf_for.getInits())` — the results
    // first, then the region's arguments, which is the order MLIR defines and prints them in.
    let results: Vec<Val> = scf_for.carried.iter().map(|_| vals.mint()).collect();
    let iv = vals.mint();
    let carried: Vec<affine::Carried> = scf_for
        .carried
        .iter()
        .zip(results)
        .map(|(source, result)| affine::Carried {
            // ⛔ THE INIT IS THE SOURCE LOOP'S OWN, UNTOUCHED.
            init: source.init,
            arg: vals.mint(),
            result,
        })
        .collect();

    // `bv_map.map(getInductionVar(), ..)` then each region iter arg, positionally.
    let mut bv_map = ValueMapping::new();
    bv_map.map(scf_for.iv, iv);
    for (source, new) in scf_for.carried.iter().zip(&carried) {
        bv_map.map(source.arg, new.arg);
    }

    // `for (auto& op : ..front().without_terminator()) builder.clone(op, bv_map);`
    let (to_clone, yielded) = scf_for.split_terminator();
    let mut body = vals.clone_ops(to_clone, &mut bv_map);

    // `if (scf_for.getNumResults() > 0)` — the results are the carried values, so this is the same
    // test. *"affine_for already has an implicit affine.yield. We need to update it if there are
    // results."*
    if !carried.is_empty() {
        body.push(DfirOp::Affine(affine::Op::Yield {
            // ⚠️ `None` HERE IS A BODY WITH NO TERMINATOR, which `scf.for`'s verifier does not admit
            // and this island cannot check. It yields nothing, and the loop it produces says so
            // rather than stopping — see [`ScfForOp::split_terminator`].
            operands: yielded
                .unwrap_or_default()
                .iter()
                .map(|operand| bv_map.lookup_or_default(*operand))
                .collect(),
        }));
    }

    // `return LogicalResult::success();`, with `affine_for` assigned.
    LoopLegalization::Transformed(DfirOp::Affine(affine::Op::For {
        iv,
        lo: affine::Bound::Const(0),
        hi: affine::Bound::Const(bounds.ubound),
        carried,
        body,
        // `if (auto dbg_name_attr = getDbgNameAttr(scf_for)) setDbgNameAttr(affine_for, ..)` — the
        // name survives the rewrite, and `scf_loop_with_result.mlir:61` checks that it does.
        dbg_name: scf_for.dbg_name.map(str::to_owned),
    }))
}


#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::dialects::{Index, agen, arith, dataflow};
    use crate::islands::dataflow_ir::print;
    use crate::islands::dataflow_ir::ty::{
        AffineExpr, AffineMap, Constraint, ElemType, IntegerSet, MemRef, ScalarTy, Vector,
    };

    /// `memref<64xf16>` — the transfer's source and destination type in the vendor's case.
    fn stick() -> MemRef {
        MemRef {
            shape: vec![64],
            elem: ElemType::F16,
        }
    }

    /// `#set = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>`.
    fn load_set() -> IntegerSet {
        IntegerSet {
            dims: 1,
            symbols: 0,
            constraints: vec![
                Constraint {
                    expr: AffineExpr::dim(0),
                    is_equality: false,
                },
                Constraint {
                    expr: AffineExpr::dim(0).times(-1).plus(AffineExpr::Const(63)),
                    is_equality: false,
                },
            ],
        }
    }

    /// `#set1` — the transfer's `time_set`, four dimensions pinned to one step.
    fn time_set() -> IntegerSet {
        let bounds = [1i64, 15, 0, 0];
        IntegerSet {
            dims: 4,
            symbols: 0,
            constraints: (0..4)
                .rev()
                .flat_map(|dim| {
                    let upper = bounds[3 - usize::try_from(dim).expect("four dims")];
                    [
                        Constraint {
                            expr: AffineExpr::dim(dim),
                            is_equality: false,
                        },
                        Constraint {
                            expr: if upper == 0 {
                                AffineExpr::dim(dim).times(-1)
                            } else {
                                AffineExpr::dim(dim).times(-1).plus(AffineExpr::Const(upper))
                            },
                            is_equality: false,
                        },
                    ]
                })
                .collect(),
        }
    }

    /// THE VENDOR'S `scf.for` AND ITS CONTEXT, BUILT IN TYPED FORM.
    ///
    /// `dcc/test/Transform/TransformLoopToLegalizeForSentientLowering/scf_loop_with_result.mlir:129-148`
    /// — the input of the pass, from `%11 = scf.for %arg4 = %c0 to %10 step %c1 iter_args(%arg5 =
    /// %arg3)` down to its `scf.yield %12`, with the four values its body reads from outside minted
    /// first so the printed numbering lines up with the vendor's.
    fn vendor_scf_for(vals: &mut Values) -> DfirOp {
        let c0 = vals.mint(); // %c0
        let c2048 = vals.mint(); // %c2048
        let c64000 = vals.mint(); // %c64000
        let hbm = vals.mint(); // %5
        let lx = vals.mint(); // %6
        let outer_iv = vals.mint(); // %arg2, the enclosing affine.for's induction variable
        let outer_arg = vals.mint(); // %arg3, the value the scf.for starts from
        let c1 = vals.mint(); // %c1, the step

        // The scf.for's own values, in MLIR's numbering order.
        let scf_result = vals.mint(); // %11
        let scf_iv = vals.mint(); // %arg4
        let scf_arg = vals.mint(); // %arg5

        // `%12 = affine.for %arg6 = 0 to 8`, `%13 = .. to 4`, `%14 = .. to 1`.
        let mb_result = vals.mint();
        let mb_iv = vals.mint();
        let mb_arg = vals.mint();
        let x_result = vals.mint();
        let x_iv = vals.mint();
        let x_arg = vals.mint();
        let out_result = vals.mint();
        let out_iv = vals.mint();
        let out_arg = vals.mint();

        // The innermost body.
        let addr = vals.mint(); // %15 = arith.subi %c2048, %arg11
        let src_view = vals.mint(); // %16
        let dst_view = vals.mint(); // %17
        let load_iv = vals.mint(); // %arg12

        let transfer = agen::Op::CompositeLoadAndStore(Box::new(agen::CompositeTransfer {
            src: src_view,
            src_indices: vec![Index::Strided(
                vec![
                    (c0, 1),
                    (out_iv, 128),
                    (x_iv, 2048),
                    (mb_iv, 8192),
                    (scf_iv, 65536),
                    (outer_iv, 2_097_152),
                ],
                0,
            )],
            src_ty: stick(),
            dst: dst_view,
            dst_indices: vec![Index::Val(c0)],
            dst_ty: stick(),
            load_iv,
            load_iv_ty: Vector {
                len: 64,
                elem: ElemType::F16,
            },
            load_set: load_set(),
            load_order: AffineMap::identity(1),
            store_set: load_set(),
            store_order: AffineMap::identity(1),
            time_set: time_set(),
            time_order: AffineMap::identity(4),
            load_time_addr_map: AffineMap {
                dims: 4,
                syms: 0,
                results: vec![
                    AffineExpr::dim(3)
                        .times(64)
                        .plus(AffineExpr::dim(2).times(128))
                        .plus(AffineExpr::dim(1).times(8192))
                        .plus(AffineExpr::dim(0).times(65536)),
                ],
            },
            store_time_addr_map: AffineMap {
                dims: 4,
                syms: 0,
                results: vec![
                    AffineExpr::dim(3)
                        .times(64)
                        .plus(AffineExpr::dim(2).times(128))
                        .plus(AffineExpr::dim(1).times(2048))
                        .plus(AffineExpr::dim(0).times(2048)),
                ],
            },
            body: vec![DfirOp::Agen(agen::Op::Yield)],
        }));

        let innermost = DfirOp::Affine(affine::Op::For {
            iv: out_iv,
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(1),
            carried: vec![affine::Carried {
                init: x_arg,
                arg: out_arg,
                result: out_result,
            }],
            body: vec![
                DfirOp::Arith(arith::Op::SubI(arith::IntBinary {
                    result: addr,
                    lhs: c2048,
                    rhs: out_arg,
                    ty: ScalarTy::Index,
                })),
                DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                    result: src_view,
                    from: hbm,
                    start: c64000,
                    layout: AffineMap::identity(1),
                    ty: stick(),
                }),
                DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                    result: dst_view,
                    from: lx,
                    start: addr,
                    layout: AffineMap::identity(1),
                    ty: stick(),
                }),
                DfirOp::Agen(transfer),
                DfirOp::Affine(affine::Op::Yield {
                    operands: vec![addr],
                }),
            ],
            dbg_name: Some("c0-l3lu-loop-ds0-ds1-out".to_owned()),
        });

        let x = DfirOp::Affine(affine::Op::For {
            iv: x_iv,
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(4),
            carried: vec![affine::Carried {
                init: mb_arg,
                arg: x_arg,
                result: x_result,
            }],
            body: vec![
                innermost,
                DfirOp::Affine(affine::Op::Yield {
                    operands: vec![out_result],
                }),
            ],
            dbg_name: Some("c0-l3lu-loop-ds0-ds1-x".to_owned()),
        });

        let mb = DfirOp::Affine(affine::Op::For {
            iv: mb_iv,
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(8),
            carried: vec![affine::Carried {
                init: scf_arg,
                arg: mb_arg,
                result: mb_result,
            }],
            body: vec![
                x,
                DfirOp::Affine(affine::Op::Yield {
                    operands: vec![x_result],
                }),
            ],
            dbg_name: Some("c0-l3lu-loop-ds0-ds1-mb".to_owned()),
        });

        DfirOp::Scf(scf::Op::For {
            iv: scf_iv,
            lo: c0,
            // `%10 = arith.select %9, %c16, %c32` — THE BOUND THIS PASS EXISTS TO REMOVE. It is not
            // read by the rewrite, which is why [`ScfForOp`] does not carry it.
            hi: c1,
            step: c1,
            carried: vec![affine::Carried {
                init: outer_arg,
                arg: scf_arg,
                result: scf_result,
            }],
            body: vec![
                mb,
                DfirOp::Scf(scf::Op::Yield {
                    operands: vec![mb_result],
                }),
            ],
            dbg_name: Some("c0-l3lu-loop-ibr-chunk-y".to_owned()),
        })
    }

    fn printed(op: &DfirOp) -> String {
        let mut out = String::new();
        print::emit(&mut out, op, 0);
        out
    }

    /// THE VENDOR'S OWN EXPECTATION FOR THE `then` ARM, RENUMBERED.
    ///
    /// `scf_loop_with_result.mlir:41-61`, the `CHECK-SENT-IR-NEXT` block from
    /// `%[[VAL_23]] = affine.for %[[VAL_24]] = 0 to 16` down to its
    /// `} {dbgName = "c0-l3lu-loop-ibr-chunk-y"}`.
    ///
    /// ⭐ EVERY LOOP VALUE IS THE VENDOR'S PLUS ONE, and that is the whole content of the check: the
    /// vendor's `func.func` declares eight constants and five units before the program unit, this
    /// fixture mints only the eight values the `scf.for` actually reads, so the two preambles differ
    /// by one and every value the REWRITE creates lines up — result, induction variable, carried
    /// argument, then the body, four levels deep.
    ///
    /// ⛔ TWO ATTRIBUTES OF THE TRANSFER ARE ABSENT, AND NEITHER IS THIS REWRITE'S. The vendor's
    /// line also carries `dbgName = "c0-l3lu-transfer-lds0-src:hbm-dst:lx"` and
    /// `dir = #agen<direction PseudoRandom>`;
    /// [`crate::islands::dataflow_ir::dialects::agen::CompositeTransfer`] has no field for either, so
    /// they cannot appear here. `transformSCFToAffineLoop` clones that op opaquely — it never reads
    /// an attribute of it — so what this proves about the rewrite is unaffected, and adding the two
    /// fields is the transfer lowering's work, not this unit's.
    const VENDOR_THEN_ARM: &str = r#"%24 = affine.for %25 = 0 to 16 iter_args(%26 = %6) -> (index) {
  %27 = affine.for %28 = 0 to 8 iter_args(%29 = %26) -> (index) {
    %30 = affine.for %31 = 0 to 4 iter_args(%32 = %29) -> (index) {
      %33 = affine.for %34 = 0 to 1 iter_args(%35 = %32) -> (index) {
        %36 = arith.subi %1, %35 : index
        %37 = dataflow.get_logical_memory_view %3, %2 {layout_map = affine_map<(d0) -> (d0)>} : index, index, memref<64xf16>
        %38 = dataflow.get_logical_memory_view %4, %36 {layout_map = affine_map<(d0) -> (d0)>} : index, index, memref<64xf16>
        agen.composite_load_and_store src:%37[%0 + %34 * 128 + %31 * 2048 + %28 * 8192 + %25 * 65536 + %5 * 2097152] dst:%38[%0]
         time_symbols(), load_iv(%39:vector<64xf16>)
         {load_order = affine_map<(d0) -> (d0)>, load_set = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>, load_time_addr_map = affine_map<(d0, d1, d2, d3) -> (d3 * 64 + d2 * 128 + d1 * 8192 + d0 * 65536)>, store_order = affine_map<(d0) -> (d0)>, store_set = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>, store_time_addr_map = affine_map<(d0, d1, d2, d3) -> (d3 * 64 + d2 * 128 + d1 * 2048 + d0 * 2048)>, time_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, time_set = affine_set<(d0, d1, d2, d3) : (d3 >= 0, -d3 + 1 >= 0, d2 >= 0, -d2 + 15 >= 0, d1 >= 0, -d1 >= 0, d0 >= 0, -d0 >= 0)>}
        {
          agen.yield
        } : memref<64xf16>, memref<64xf16>
        affine.yield %36 : index
      } {dbgName = "c0-l3lu-loop-ds0-ds1-out"}
      affine.yield %33 : index
    } {dbgName = "c0-l3lu-loop-ds0-ds1-x"}
    affine.yield %30 : index
  } {dbgName = "c0-l3lu-loop-ds0-ds1-mb"}
  affine.yield %27 : index
} {dbgName = "c0-l3lu-loop-ibr-chunk-y"}
"#;

    fn transform(vals: &mut Values, scf_for: &DfirOp, ubound: i64) -> LoopLegalization {
        transform_scf_to_affine_loop(
            vals,
            &ScfForOp::of(scf_for).expect("the fixture is an scf.for"),
            StaticBounds {
                lbound: 0,
                ubound,
                step: 1,
            },
        )
    }

    /// 🎯 117/384 — THE VENDOR'S CASE, WHOLE: `scf.for` WITH A RESULT BECOMES `affine.for 0 to 16`.
    ///
    /// `dcc-opt --dcc-transform-loop-to-legalize-for-sentient-lowering scf_loop_with_result.mlir`.
    #[test]
    fn the_vendors_scf_loop_with_result_becomes_the_vendors_affine_loop() {
        let mut vals = Values::default();
        let scf_for = vendor_scf_for(&mut vals);
        let LoopLegalization::Transformed(then_arm) = transform(&mut vals, &scf_for, 16) else {
            unreachable!("lbound 0 and step 1 are the supported case");
        };
        assert_eq!(printed(&then_arm), VENDOR_THEN_ARM);
    }

    /// 🎯 117/384 — THE SAME BODY TRANSFORMED TWICE BINDS TWO DISJOINT SETS OF NAMES.
    ///
    /// ⛔⛔ THE CALLER DOES THIS ON EVERY LOOP IT LEGALISES.
    /// `transformSCFLoopWithNonConstantUpperBound` (entry 292, `:207` and `:227`) calls this once with
    /// the `then` builder and `then_ub` and once with the `else` builder and `else_ub`, on the SAME
    /// `scf.for` — the vendor's expectation is two copies of one body, at `to 16` and at `to 32`
    /// (`scf_loop_with_result.mlir:41` and `:64`). Two copies sharing SSA names is not a program, and
    /// nothing in a typed island would say so; this is what says so.
    #[test]
    fn transforming_one_loop_twice_binds_disjoint_values() {
        let mut vals = Values::default();
        let scf_for = vendor_scf_for(&mut vals);
        let LoopLegalization::Transformed(then_arm) = transform(&mut vals, &scf_for, 16) else {
            unreachable!("the then arm")
        };
        let LoopLegalization::Transformed(else_arm) = transform(&mut vals, &scf_for, 32) else {
            unreachable!("the else arm")
        };

        let (then_bound, else_bound) = (bound_of(&then_arm), bound_of(&else_arm));
        assert_eq!((then_bound, else_bound), (16, 32));

        let then_defined = defined_values(&then_arm);
        let else_defined = defined_values(&else_arm);
        assert_eq!(then_defined.len(), else_defined.len());
        for val in &then_defined {
            assert!(
                !else_defined.contains(val),
                "{val:?} is bound by both copies of the body"
            );
        }

        // ⭐ AND THE TWO ARMS START FROM THE SAME INIT — the enclosing loop's carried argument, which
        // is outside both bodies and therefore not renumbered (`:56` and `:74` both read `%20`).
        assert_eq!(init_of(&then_arm), init_of(&else_arm));
    }

    /// 🎯 117/384 — A LOWER BOUND THAT IS NOT 0, OR A STEP THAT IS NOT 1, IS `LogicalResult::failure`.
    ///
    /// *"we currently support lbound being 0, step being 1, non-constant ubound."* (`:105`.) The
    /// caller answers a failure with `if_op->emitError("Unable to transform SCF loop into Affine
    /// loop")` (`:255`) and leaves the `scf.for` where it was.
    #[test]
    fn only_a_zero_lower_bound_and_a_unit_step_are_supported() {
        let mut vals = Values::default();
        let scf_for = vendor_scf_for(&mut vals);
        let of = ScfForOp::of(&scf_for).expect("the fixture is an scf.for");

        for bounds in [
            StaticBounds {
                lbound: 1,
                ubound: 16,
                step: 1,
            },
            StaticBounds {
                lbound: 0,
                ubound: 16,
                step: 2,
            },
            StaticBounds {
                lbound: -4,
                ubound: 16,
                step: 4,
            },
        ] {
            assert_eq!(
                transform_scf_to_affine_loop(&mut vals, &of, bounds),
                LoopLegalization::UnsupportedBounds {
                    lbound: bounds.lbound,
                    step: bounds.step,
                }
            );
        }

        // ⛔ AND A DECLINE MINTS NOTHING. The reference returns before `AffineForOp::create`.
        let before = vals.issued();
        let _ = transform_scf_to_affine_loop(
            &mut vals,
            &of,
            StaticBounds {
                lbound: 8,
                ubound: 16,
                step: 1,
            },
        );
        assert_eq!(vals.issued(), before);
    }

    /// 🎯 117/384 — A LOOP THAT CARRIES NOTHING GETS NO `affine.yield`, AND EVERY BODY OP IS CLONED.
    ///
    /// *"affine_for already has an implicit affine.yield. We need to update it if there are
    /// results."* (`:129-130`) — `getNumResults() > 0` is false, so the body is the clone and nothing
    /// else. The `scf.for` this comes from has no explicit terminator in this island either, which is
    /// what [`ScfForOp::split_terminator`] answers `None` for.
    #[test]
    fn a_loop_with_no_results_gets_no_yield() {
        let mut vals = Values::default();
        let iv = vals.mint();
        let bound = vals.mint();
        let doubled = vals.mint();
        let scf_for = DfirOp::Scf(scf::Op::For {
            iv,
            lo: bound,
            hi: bound,
            step: bound,
            carried: Vec::new(),
            body: vec![DfirOp::Arith(arith::Op::AddI(arith::IntBinary {
                result: doubled,
                lhs: iv,
                rhs: iv,
                ty: ScalarTy::Index,
            }))],
            dbg_name: None,
        });

        let LoopLegalization::Transformed(affine_for) = transform(&mut vals, &scf_for, 12) else {
            unreachable!("lbound 0 and step 1")
        };
        assert_eq!(
            printed(&affine_for),
            "affine.for %3 = 0 to 12 {\n  %4 = arith.addi %3, %3 : index\n}\n"
        );
    }

    /// 🎯 117/384 — A YIELD OPERAND DEFINED OUTSIDE THE BODY COMES THROUGH UNCHANGED.
    ///
    /// ⚠️ THE ONE PLACE THIS PORT DIVERGES, DELIBERATELY. `bv_map.lookup(operand)` returns null for a
    /// value the mapping never saw and MLIR then fails a verifier;
    /// [`ValueMapping::lookup_or_default`] returns the value itself. A legal `scf.for` can forward its
    /// own init straight out — `scf.yield %arg5` — and passing it through is the only answer that is
    /// not a stop, which this crate does not have.
    #[test]
    fn a_yield_operand_from_outside_the_body_passes_through() {
        let mut vals = Values::default();
        let init = vals.mint();
        let bound = vals.mint();
        let iv = vals.mint();
        let arg = vals.mint();
        let result = vals.mint();
        let scf_for = DfirOp::Scf(scf::Op::For {
            iv,
            lo: bound,
            hi: bound,
            step: bound,
            carried: vec![affine::Carried { init, arg, result }],
            body: vec![DfirOp::Scf(scf::Op::Yield {
                // ⭐ NOT THE REGION ARGUMENT: the value the loop STARTED from, hoisted above it.
                operands: vec![init],
            })],
            dbg_name: None,
        });

        let LoopLegalization::Transformed(affine_for) = transform(&mut vals, &scf_for, 3) else {
            unreachable!("lbound 0 and step 1")
        };
        assert_eq!(
            printed(&affine_for),
            "%5 = affine.for %6 = 0 to 3 iter_args(%7 = %0) -> (index) {\n  affine.yield %0 : \
             index\n}\n"
        );
    }

    /// 🎯 117/384 — AND AN OP THAT IS NOT AN `scf.for` IS THE `dyn_cast`'s `None`.
    #[test]
    fn only_an_scf_for_is_transformable() {
        let plain = DfirOp::Arith(arith::Op::Constant {
            result: Val(0),
            value: 8,
        });
        assert!(ScfForOp::of(&plain).is_none());
        assert!(
            ScfForOp::of(&DfirOp::Affine(affine::Op::For {
                iv: Val(0),
                lo: affine::Bound::Const(0),
                hi: affine::Bound::Const(4),
                carried: Vec::new(),
                body: Vec::new(),
                dbg_name: None,
            }))
            .is_none()
        );
    }

    /// The loop's static upper bound, for a test that only cares about it.
    fn bound_of(op: &DfirOp) -> i64 {
        let DfirOp::Affine(affine::Op::For {
            hi: affine::Bound::Const(hi),
            ..
        }) = op
        else {
            unreachable!("the rewrite builds a constant-bounded affine.for");
        };
        *hi
    }

    /// The loop's single carried init.
    fn init_of(op: &DfirOp) -> Val {
        let DfirOp::Affine(affine::Op::For { carried, .. }) = op else {
            unreachable!("an affine.for");
        };
        carried[0].init
    }

    /// EVERY VALUE AN OP AND ITS REGIONS DEFINE — results and block arguments, recursively.
    fn defined_values(op: &DfirOp) -> Vec<Val> {
        let mut copy = op.clone();
        let parts = crate::islands::dataflow_ir::dialects::parts_mut(&mut copy);
        let mut found: Vec<Val> = parts
            .results
            .into_iter()
            .chain(parts.block_args)
            .map(|val| *val)
            .collect();
        let nested: Vec<Val> = parts
            .regions
            .into_iter()
            .flat_map(|region| region.iter().flat_map(defined_values).collect::<Vec<_>>())
            .collect();
        found.extend(nested);
        found
    }
}
