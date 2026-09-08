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

//! `LoopUnrollingForPTLRFRegs.cpp` — 2 of bridge 2's 384 functions (dependency level(s) [2, 3]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e249_processComputeUnit` | 249/384 | 91 | `dcc/src/Transform/Dataflow/LoopUnrollingForPTLRFRegs.cpp:37` |
//! | `e288_runOnOperation` | 288/384 | 22 | `dcc/src/Transform/Dataflow/LoopUnrollingForPTLRFRegs.cpp:131` |


use crate::islands::dataflow_ir::dialects::dataflow::LocalUnit;
use crate::islands::dataflow_ir::dialects::{
    Index, Op as DfirOp, Val, affine, agen, arith, dataflow, defining_op, region_owner, regions, scf,
    uses,
};

use super::tf_loop_unroll_for_shuffle_op::{Loop, Unroll, perform_full_unroll};

/// `dcc::utils::isLRFReg` (`dcc/src/Utils/Utils.cpp:716-718`) — the three unit-prefixed LRFs, and
/// notably NOT `ptxrf`, `ptarf` or `l0scale`.
fn is_lrf_reg(which: LocalUnit) -> bool {
    matches!(
        which,
        LocalUnit::PeLrf | LocalUnit::SfpLrf | LocalUnit::PtLrf
    )
}

/// THE INDEX LIST ONE USE OF A MEMORY VIEW READS — the four `dyn_cast`s of `:46-56`.
///
/// ⭐ ONE ARM FOR ALL FOUR, because `getIndices()` and `getMapIndices()` are the same field in this
/// island. The reference needs four casts only to reach four differently named accessors.
fn view_use_indices(op: &DfirOp) -> Option<&[Index]> {
    match op {
        DfirOp::Affine(
            affine::Op::VectorLoad { indices, .. } | affine::Op::VectorStore { indices, .. },
        )
        | DfirOp::Agen(agen::Op::VectorLoad { indices, .. } | agen::Op::VectorStore { indices, .. }) => {
            Some(indices)
        }
        _ => None,
    }
}

/// EVERY `dataflow.get_logical_memory_view` IN A UNIT, INNERMOST INCLUDED — `unit_op.walk(..)`.
fn memory_views(scope: &[DfirOp], into: &mut Vec<(Val, Val)>) {
    for op in scope {
        if let DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView { result, from, .. }) = op {
            into.push((*result, *from));
        }
        for region in regions(op) {
            memory_views(region, into);
        }
    }
}

/// EVERY LOOP-LIKE OP IN A UNIT, INNERMOST FIRST — `walk<WalkOrder::PostOrder>`, whose whole point
/// here is that unrolling an outer loop first would duplicate the inner ones it has yet to visit.
fn loops_post_order<'o>(scope: &'o [DfirOp], into: &mut Vec<(Val, &'o DfirOp)>) {
    for op in scope {
        for region in regions(op) {
            loops_post_order(region, into);
        }
        match op {
            // `LoopLikeOpInterface` in this island is exactly these three: the two for-loops and
            // `scf.parallel`, which is the op the *"Unknown for-loop"* arm at `:118` is about.
            DfirOp::Affine(affine::Op::For { iv, .. }) | DfirOp::Scf(scf::Op::For { iv, .. }) => {
                into.push((*iv, op));
            }
            DfirOp::Scf(scf::Op::Parallel { ivs, .. }) => {
                into.extend(ivs.iter().map(|iv| (*iv, op)));
            }
            _ => {}
        }
    }
}

/// ONE MARKED LOOP AND THE UNROLL IT ASKS FOR — `loop_op->removeAttr("unroll")` and the utility call
/// that follows it (`:88-121`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PtLrfUnroll {
    /// The loop, named by the induction variable an LRF subscript read.
    pub iv: Val,
    /// What `loopUnrollFull` / `loopUnrollByFactor` was asked for — see [`Unroll`].
    pub unroll: Unroll,
}

/// WHAT `processComputeUnit` LEFT MARKED, OR THE `emitError` THAT STOPPED IT.
///
/// ⛔ EACH `signalPassFailure(); return;` RETURNS FROM THE WALK'S LAMBDA, so the reference keeps
/// marking and unrolling after one — and then the pass manager discards the module. Stopping at the
/// first is the same observable, since nothing downstream reads a failed pass's IR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PtLrfUnrolling {
    /// Step 2 ran: one entry per marked loop, innermost first.
    Marked {
        /// The unroll requests, in the post-order the reference issues them.
        unrolls: Vec<PtLrfUnroll>,
    },
    /// `view_op.getFromUnit().getDefiningOp<GetLocalUnitOp>()` is null and `:41` dereferences it.
    ViewIsNotOfALocalUnit(Val),
    /// `"Encountered an unknown use of memory view"` (`:57`) — the view named by this value.
    UnknownUseOfMemoryView(Val),
    /// `"Support for unrolling loop-like op interfaces only"` (`:71-73`).
    IndexIsNotInALoopLikeOp(Val),
    /// `"Support for unrolling of loop iterators only"` (`:78-80`) — an index that is neither a
    /// region argument nor an `arith` constant.
    IndexIsNotALoopIterator(Val),
    /// `"Unknown for-loop for unrolling"` (`:118`) — a marked `scf.parallel`.
    UnknownForLoopForUnrolling(Val),
}

impl PtLrfUnrolling {
    /// DID THE PASS `signalPassFailure()` — the `emitError`s above, plus a decline from either
    /// unroll utility (`:94-97`, `:110-113`).
    #[must_use]
    pub fn failed(&self) -> bool {
        match self {
            PtLrfUnrolling::Marked { unrolls } => unrolls.iter().any(|ask| ask.unroll.failed()),
            PtLrfUnrolling::ViewIsNotOfALocalUnit(_)
            | PtLrfUnrolling::UnknownUseOfMemoryView(_)
            | PtLrfUnrolling::IndexIsNotInALoopLikeOp(_)
            | PtLrfUnrolling::IndexIsNotALoopIterator(_)
            | PtLrfUnrolling::UnknownForLoopForUnrolling(_) => true,
        }
    }
}

/// Replaces: e249_processComputeUnit
///
/// **249/384** `LoopUnrollingForPTLRFRegsPass::processComputeUnit` —
/// `dcc/src/Transform/Dataflow/LoopUnrollingForPTLRFRegs.cpp:37` (91L): an LRF has no addressing
/// hardware, so every loop whose induction variable subscripts one must be gone by lowering. Step 1
/// marks those loops; step 2 unrolls them innermost-first.
///
/// ⛔⛔ THE `scf.for` ARM'S NULL TEST IS INVERTED (`:107-108`): `if (!lb_const || !ub_const ||
/// !step_const)` guards the branch that reads `lb_const.value()`, so a constant-bounded loop takes
/// the `else` and reports *"Non-Constant trip bound"* while a non-constant one dereferences null.
/// ⭐ The polarity the body requires is the sibling pass's, verbatim, at
/// `LoopUnrollForShuffleOp.cpp:169-173` — so this delegates to [`perform_full_unroll`], which is that
/// same reconstruction already ported as entries 109/110.
#[must_use]
pub fn process_compute_unit(unit: &[DfirOp]) -> PtLrfUnrolling {
    // ── Step 1: `// Collect loops meant for unrolling` (`:39-86`) ────────────────────────────────
    let mut views: Vec<(Val, Val)> = Vec::new();
    memory_views(unit, &mut views);
    let mut marked: Vec<Val> = Vec::new();

    for (view, from) in views {
        let Some(DfirOp::Dataflow(dataflow::Op::GetLocalUnit { which, .. })) =
            defining_op(from, unit)
        else {
            return PtLrfUnrolling::ViewIsNotOfALocalUnit(from);
        };
        if !is_lrf_reg(*which) {
            continue;
        }

        for use_op in uses(view, unit) {
            let Some(indices) = view_use_indices(use_op) else {
                return PtLrfUnrolling::UnknownUseOfMemoryView(view);
            };
            for index in indices {
                // `if (mlir::isa<BlockArgument>(index))` (`:62`) — and a strided sum is one use per
                // variable in it, each of which is a subscript of this LRF.
                let vals: Vec<Val> = match index {
                    // `isa<arith::ConstantIntOp, arith::ConstantIndexOp>` (`:76-78`) — a literal
                    // subscript needs no loop, and this island writes it as the literal itself.
                    Index::Const(_) => continue,
                    Index::Val(val) => vec![*val],
                    Index::Strided(terms, _) => terms.iter().map(|(val, _)| *val).collect(),
                };
                for val in vals {
                    // `auto for_op = index.getParentRegion()->getParentOp();`
                    let Some(owner) = region_owner(val, unit) else {
                        // `else if (!isa<arith::ConstantIntOp, arith::ConstantIndexOp>(
                        //      index.getDefiningOp()))` (`:76-78`) — a subscript that is not a block
                        // argument is still fine when it is a constant, whichever of the two it is.
                        if matches!(
                            defining_op(val, unit),
                            Some(DfirOp::Arith(
                                arith::Op::Constant { .. } | arith::Op::ConstantInt { .. }
                            ))
                        ) {
                            continue;
                        }
                        return PtLrfUnrolling::IndexIsNotALoopIterator(val);
                    };
                    // `if (isa<LoopLikeOpInterface>(for_op))`
                    let is_loop_like = matches!(
                        owner,
                        DfirOp::Affine(affine::Op::For { .. })
                            | DfirOp::Scf(scf::Op::For { .. } | scf::Op::Parallel { .. })
                    );
                    if !is_loop_like {
                        return PtLrfUnrolling::IndexIsNotInALoopLikeOp(val);
                    }
                    // `if (!for_op->hasAttr("unroll")) for_op->setAttr("unroll", ..)` — the mark is
                    // the induction variable, which names the loop uniquely and needs no attribute.
                    if !marked.contains(&val) {
                        marked.push(val);
                    }
                }
            }
        }
    }

    // ── Step 2: `// Start unrolling the marked loops in reverse order` (`:88-121`) ───────────────
    let mut ordered: Vec<(Val, &DfirOp)> = Vec::new();
    loops_post_order(unit, &mut ordered);
    let mut unrolls = Vec::new();
    for (iv, loop_op) in ordered {
        if !marked.contains(&iv) {
            continue;
        }
        let Some(candidate) = Loop::of(loop_op) else {
            return PtLrfUnrolling::UnknownForLoopForUnrolling(iv);
        };
        unrolls.push(PtLrfUnroll {
            iv,
            unroll: perform_full_unroll(candidate, unit),
        });
    }

    PtLrfUnrolling::Marked { unrolls }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::islands::dataflow_ir::Values;
    use crate::islands::dataflow_ir::ty::{AffineMap, ElemType, MemRef, Vector};

    /// 🎯 249/384 — A LOOP SUBSCRIPTING A `pt_lrfreg` IS MARKED AND FULLY UNROLLED; ONE OVER `ptxrf`
    /// IS NOT TOUCHED.
    ///
    /// ⛔ THIS PASS IS NOT REGISTERED ANYWHERE IN THE AUTHORITY TREE — `LoopUnrollingForPTLRFRegs`
    /// appears only in its own file and `Transform/Dataflow/CMakeLists.txt`, with no `Passes.h`
    /// declaration, no pipeline entry and no `dcc/test` case. So there is no vendor expectation to
    /// port, and its inverted null test at `:107-108` has never run.
    #[test]
    fn a_loop_subscripting_an_lrf_is_marked_and_a_loop_over_the_xrf_is_not() {
        let load_view = |view: Val, iv: Val, result: Val| {
            DfirOp::Agen(agen::Op::VectorLoad {
                result,
                view,
                indices: vec![Index::Val(iv)],
                view_ty: MemRef {
                    shape: vec![4],
                    elem: ElemType::F16,
                },
                ty: Vector {
                    len: 128,
                    elem: ElemType::F16,
                },
            })
        };
        let program = |which: LocalUnit| {
            let mut vals = Values::default();
            let (unit, lrf, start, view) = (vals.mint(), vals.mint(), vals.mint(), vals.mint());
            let (iv, loaded) = (vals.mint(), vals.mint());
            let ops = vec![
                DfirOp::Dataflow(dataflow::Op::GetLocalUnit {
                    result: lrf,
                    of: unit,
                    which,
                }),
                DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
                    result: view,
                    from: lrf,
                    start,
                    layout: AffineMap::identity(1),
                    ty: MemRef {
                        shape: vec![4],
                        elem: ElemType::F16,
                    },
                }),
                DfirOp::Affine(affine::Op::For {
                    iv,
                    lo: affine::Bound::Const(0),
                    hi: affine::Bound::Const(4),
                    carried: Vec::new(),
                    body: vec![load_view(view, iv, loaded)],
                    dbg_name: None,
                }),
            ];
            (iv, ops)
        };

        let (iv, ops) = program(LocalUnit::PtLrf);
        let marked = process_compute_unit(&ops);
        assert_eq!(
            marked,
            PtLrfUnrolling::Marked {
                unrolls: vec![PtLrfUnroll {
                    iv,
                    unroll: Unroll::Fully,
                }],
            },
            "`loopUnrollFull` on the affine loop whose iv indexes the LRF"
        );
        assert!(!marked.failed());

        // ⛔ `isLRFReg` IS THREE NAMES, AND `ptxrf` IS NOT ONE OF THEM.
        let (_, ops) = program(LocalUnit::PtXrf);
        assert_eq!(
            process_compute_unit(&ops),
            PtLrfUnrolling::Marked {
                unrolls: Vec::new()
            },
            "a view of the transposed register file marks nothing"
        );
    }
}

// ⛔ RE-CREATED ANCHORS. These units' `crustify:todo:` markers were deleted without a
// `/// Replaces:` ever appearing, which removed them from every later schedule and let the
// driver report the campaign DONE. Outstanding work is now computed from UNITS.tsv.
// crustify:todo: e288_runOnOperation
