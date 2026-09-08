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

//! `TransformPagedMemView.cpp` — 1 of bridge 2's 384 functions (dependency level(s) [1]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e196_runOnOperation` | 196/384 | 23 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemView.cpp:41` |

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 196/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

use crate::arch::Arch;
use crate::islands::dataflow_ir::Program;
use crate::islands::dataflow_ir::dialects::{Op as DfirOp, regions};
use crate::islands::dataflow_ir::ty::GenericComp;
use crate::units::DfirUnit;

use super::tf_transform_paged_mem_view_manager::{PagedMemView, Selection, TpmvManager};

/// `is_any_of(comp, SenComponents::LXLU, SenComponents::LXSU, SenComponents::L3LU,
/// SenComponents::L3SU)` (`TransformPagedMemView.cpp:46-47`) — the units a STATIC PAGED TENSOR may be
/// accessed from.
///
/// ⛔⛔ FOUR OF THE SIX LOAD AND STORE HALVES, AND THE TWO MISSING ONES ARE THE L0'S.
/// [`super::tf_transform_loop_to_legalize_for_sentient_lowering`]'s `LOAD_AND_STORE_UNITS` is the
/// same list plus `L0LU` and `L0SU` (`:633`, `:647`), so writing this as "the load/store units" would
/// have de-paged a view on an L0 half that the reference leaves alone. A paged view is a window onto
/// the HBM or the LX (`paged_mem_view_loads.mlir:289` views `%lx`,
/// `paged_mem_view_load_and_store.mlir:1088` views `%hbm`), and the L0 is neither.
///
/// ⭐ ONE-TO-ONE HERE, WHICH IS WHY [`GenericComp`] IS AS PRECISE AS `SenComponents`. The reference
/// compares `comp` — a `SenComponents` — for equality against these four names, and
/// [`DfirUnit::generic`] maps each of `Lxlu`/`Lxsu`/`L3lu`/`L3su` to its own image; the many-to-one
/// collapse in that map is over the PT's rows and the L0LU's spellings, none of which are in this
/// list.
const PAGED_TENSOR_UNITS: [GenericComp; 4] = [
    GenericComp::Lxlu,
    GenericComp::Lxsu,
    GenericComp::L3lu,
    GenericComp::L3su,
];

/// ONE TURN OF THE PASS'S `while (true)` — the view it picked up and what the manager chose for it.
///
/// ⭐ THE VIEW IS `candidate` (`TransformPagedMemView.cpp:49-54`) and the selection is what
/// `TPMVManager::run()` answered for it (`:61-62`), which is entry 139
/// ([`TpmvManager::run`]). Keeping the pair together is what makes a round reportable: a
/// [`Selection::NotOneUser`] on its own does not say WHICH view had the wrong number of users, and
/// that is the whole content of the reference's abort message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DePagingRound<'p> {
    /// `candidate` — the paged view this round lowers.
    pub view: PagedMemView<'p>,
    /// `TPMVManager(candidate, comp).run()`.
    pub selection: Selection<'p>,
}

/// THE ROUNDS ONE `dataflow.program_unit` GETS — the body of the `if` at
/// `TransformPagedMemView.cpp:46-63`.
///
/// ⛔ PRESENT WITH NO ROUNDS IS NOT THE SAME AS ABSENT. A unit whose component is one of
/// [`PAGED_TENSOR_UNITS`] but that holds no paged view enters the `while (true)`, finds nothing and
/// breaks (`:54-55`); a unit whose component is not is never entered at all. The reference cannot
/// tell those apart because both leave the IR alone — here they are an empty [`Self::rounds`] and no
/// entry, so a test can state which of the two rules did the work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitDePaging<'p> {
    /// `comp` — `dcc::getUnitType(unit_op.getUnits()[0].getDefiningOp<dataflow::GetUnitOp>())`.
    pub comp: DfirUnit,
    /// The rounds, in the order the pass ran them.
    pub rounds: Vec<DePagingRound<'p>>,
}

/// EVERY `dataflow.get_paged_logical_memory_view` IN A REGION, in the order a walk reaches them —
/// `unit_op.walk([&](GetPagedLogicalMemoryViewOp op) { .. })`.
///
/// ⭐ THE ORDER IS THE SAME UNDER MLIR'S POST-ORDER WALK AND A PRE-ORDER ONE, because a paged view
/// carries no region. `walk`'s default is post-order, which visits an op's regions before the op
/// itself — the two orders differ only in where a REGION-CARRYING op sits relative to its children,
/// and this filter never yields one. So "depth first, in block order" is the reference's sequence.
///
/// ⚠️ THE MECHANISM FOR REACHING OPERANDS, which the brief lets a port build its own way: MLIR walks
/// an op's regions through `Operation::getRegions()`, and this island's equivalent is
/// [`regions`] — exhaustive and with no wildcard, so a newly declared region-carrying op cannot
/// hide a view from this walk.
fn paged_views<'p>(ops: &'p [DfirOp], found: &mut Vec<PagedMemView<'p>>) {
    for op in ops {
        if let Some(view) = PagedMemView::of(op) {
            found.push(view);
        }
        for region in regions(op) {
            paged_views(region, found);
        }
    }
}

/// Replaces: e196_runOnOperation
///
/// **196/384** `TransformPagedMemViewPass::runOnOperation` —
/// `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemView.cpp:41` (23L).
///
/// ```cpp
/// void TransformPagedMemViewPass::runOnOperation() {
///   ModuleOp module_op = getOperation();
///   module_op.walk([&](dataflow::ProgramUnitOp unit_op) {
///     auto comp = dcc::getUnitType(
///         unit_op.getUnits()[0].getDefiningOp<dataflow::GetUnitOp>());
///     if (is_any_of(comp, SenComponents::LXLU, SenComponents::LXSU,
///                   SenComponents::L3LU, SenComponents::L3SU)) {
///       while (true) {
///         dataflow::GetPagedLogicalMemoryViewOp candidate = nullptr;
///         unit_op.walk(
///             [&](dataflow::GetPagedLogicalMemoryViewOp paged_mem_view_op) {
///               candidate = paged_mem_view_op;
///               return WalkResult::interrupt();
///             });
///         if (!candidate) break;
///         TPMVManager tpmv_manager(candidate, comp);
///         if (tpmv_manager.run().failed()) {
///           signalPassFailure();
///           return;
///         }
///       }
///     }
///   });
/// }
/// ```
///
/// The pass that removes `dataflow.get_paged_logical_memory_view` from the rung: every access to a
/// STATIC PAGED TENSOR is rewritten into accesses to plain views, one per page that the access's
/// subscripts can actually reach, guarded by the conditionals that select between them.
///
/// # ⛔⛔ THE LOOP'S STEP IS AN ERASE, AND THE PORT HAS TO STATE IT AS A RULE
///
/// `while (true)` re-walks from scratch every round and always takes the FIRST view it finds
/// (`:49-54`). It terminates because `TPMVManager::run()` DELETES the view it lowered —
/// `info.paged_mem_view_->erase()` (`TransformPagedMemViewImpl.cpp:621`) — so the next walk finds a
/// different one. Nothing in this crate erases an op out of the caller's program, so a port that
/// simply looped would never stop; the loop has to be re-expressed over the views a walk finds ONCE,
/// with an explicit rule for which of them an earlier round already consumed.
///
/// ⛔ AND "ONE ROUND, ONE VIEW" IS THE WRONG RULE — IT DOUBLE-COUNTS EVERY LOAD-AND-STORE PAIR.
/// `TPMVVectorLoadStore::initialize` registers TWO `TPMVInfo`s, one for the load's memref and one for
/// the store's (`Impl.cpp:762`, `:769-770`), and `TPMVCompositeLoadStore::initialize` does the same
/// for `src` and `dst` (`:1191-1202`); `TPMVBase::transform` then erases the view of EVERY non-null
/// one (`:596-621`, skipping the nulls at `:599`). So a pair's second view is already gone when the
/// next round walks, and a port that gave it a round of its own would report twice the work the
/// reference does — on the vendor's own composite input, two rounds where dcc runs one
/// (`paged_mem_view_load_and_store.mlir:1084-1094`).
///
/// # ⭐ THE RULE, AND WHY IT IS EXACTLY THE ERASE SET
///
/// A view is skipped when the manager's answer for it REPEATS an answer already given in this unit.
/// That is not a heuristic — it is the erase set restated:
///
/// * A [`Selection`] names the leaf the reference builds, and every leaf holds `mem_ops_` — the ONE
///   memory op `TPMVBase::new` seeded it with
///   ([`super::tf_transform_paged_mem_view_impl::TpmvBase::new`]).
/// * Two DISTINCT views answer with the same leaf only when they have the same single user, since
///   the manager dispatches on `*paged_mem_view_->getUsers().begin()` and nothing else
///   (`TransformPagedMemViewManager.cpp:22-25`) — and an op that reads two paged views is exactly a
///   load-and-store pattern or a composite transfer, the two cases whose `initialize` registers both.
/// * The vector pair is the one case where the two views have DIFFERENT users, the load and its
///   store — and the manager normalises it: from either view it builds
///   `TPMVVectorLoadStore(load_op, store_op, comp)` (`:33`, `:45`), and that constructor keeps only
///   the load (`Impl.cpp:756-757` asserts the one-element `mem_ops_`). The two rounds would be the
///   same value, so the second is the one the erase removed.
///
/// ⛔ AND `==` IS SOUND HERE EVEN THOUGH THE OPS COMPARE BY VALUE AND NOT BY ADDRESS, which is the
/// one thing this rule could have got wrong. A leaf names its memory op, and that op names the view
/// it reads (`view` on an access, `src`/`dst` on a transfer). So two rounds whose leaves compare
/// equal are rounds whose memory ops read the SAME view — and a view read twice has two users, which
/// is [`Selection::NotOneUser`] and not a `Selected` round at all. Two structurally identical
/// accesses on DIFFERENT views cannot collide, because the view is part of what is compared.
///
/// ⭐ WHICH ALSO MAKES THE PORT TERMINATE BY CONSTRUCTION: the rounds run over a finite list of views
/// collected once, rather than over a re-walk whose progress depends on an unported erase.
///
/// # ⛔ A NON-`Selected` ANSWER ENDS THE UNIT AND ONLY THE UNIT
///
/// `if (tpmv_manager.run().failed()) { signalPassFailure(); return; }` (`:58-60`) returns from the
/// WALK LAMBDA, which is `void` — so it abandons this unit's `while (true)` and the walk carries on
/// to the next `program_unit`. It does not abandon the module.
///
/// ⭐ AND [`Selection::NotOneUser`] / [`Selection::Unsupported`] STOP IT FOR A SECOND, INDEPENDENT
/// REASON. They are the manager's `DT_CHECK` and `llvm_unreachable`, which in the reference stop the
/// process outright; entry 139 turns them into values because this crate never runtime-refuses. A
/// round that selected nothing also consumed nothing, so the reference's own loop could not have made
/// progress past one — which is why it is a hard stop there and the last round for this unit here.
///
/// ⭐ `signalPassFailure()` (`:58`) IS A FLAG ON THE PASS OBJECT, and the port's flag is the round
/// itself: a unit whose last round is not [`Selection::Selected`] is a unit the reference would have
/// failed the build over. Keeping it as data rather than as a returned status is what lets a caller
/// say which unit and which view, which is more than the reference's own diagnostic carries.
///
/// ⚠️ `run()` ITSELF CANNOT FAIL YET. Its `LogicalResult` comes from `initialize()` and `transform()`
/// (entries 202, 356 and 373, unported), so what a round reports today is which leaf the reference
/// would have run — the delegated-emission shape entry 139 already committed this pass to, and the
/// one entry 109 records for a transform that upstream MLIR performs.
///
/// # ⭐ WHAT THE MODULE IS, AND WHY THE BORROW IS SHARED
///
/// `ModuleOp module_op = getOperation()` (`:42`) — this island's module is a
/// [`Program`]: its preamble plus its units, which is how
/// [`super::tf_program_units_reduction::module_scope`] spells the same thing. And the parameter is
/// `&`, for the reason [`super::tf_cfg_simplification_dataflow_level::run_on_operation`] (entry 383)
/// records for its own: the rewriting this pass does is `initialize()` and `transform()`, entries 202
/// and 356, so there is nothing yet to write back. It becomes `&mut` in the changeset that lands the
/// erase.
///
/// ⭐ AND THE WALK FOR `dataflow::ProgramUnitOp` (`:43`) IS THE UNITS LIST, not a search.
/// [`crate::islands::dataflow_ir::ProgramUnits`] is where a program's units live — a field rather
/// than ops among ops, and non-empty by construction — so `walk` becomes `iter()`, exactly as entry
/// 383 does it. The `dataflow.program_unit` OP variant exists for printing a unit inside a body, and
/// a program that nested one there would be one this island's own emitter cannot build.
///
/// ⚠️ `opts_` (`:34`, `:38`) IS READ BY NOTHING IN THIS FUNCTION. The pass takes a
/// `dcc::CommonPassOptions` in its constructor and stores it; `runOnOperation` never mentions it. It
/// is one of the campaign's 106 exclusions (`docs/bridge2-porting-order.md:1565`), and the two
/// `createTransformPagedMemViewPass` factories that hand it over (`:70-78`) are pass registration,
/// which no entry covers.
///
/// # ⭐ WHERE THE COMPONENT COMES FROM
///
/// The reference resolves it through the op —
/// `getUnitType(unit_op.getUnits()[0].getDefiningOp<GetUnitOp>())` — because MLIR's unit list is a
/// list of `Value`s and the `type` attribute lives on their defining op.
/// [`crate::islands::dataflow_ir::Units`] carries the kind ITSELF, and carries it as the FILTER its
/// only list constructor selected on (`Units::of`), so every value in the list including
/// `getUnits()[0]` is bound by a `dataflow.get_unit` of that type. Reading `on.kind()` is the same
/// fact without the `getDefiningOp` that returns null —
/// [`super::tf_program_units_reduction`]'s `head_get_unit` goes through the op because entry 193 also
/// needs the `core`/`corelet` attributes, which only the op carries.
///
/// ⭐ AND THE SCOPE IS THE UNIT'S OWN BODY, not the module. The reference's `getUsers()` is global,
/// but a view found by `unit_op.walk` is DEFINED INSIDE that unit's region, and MLIR requires every
/// use to be dominated by its definition — a value defined inside a region is not visible outside
/// it. So the unit's body holds every user there can be.
#[must_use]
pub fn run_on_operation<A: Arch>(program: &Program<A>) -> Vec<UnitDePaging<'_>> {
    let mut per_unit: Vec<UnitDePaging<'_>> = Vec::new();

    // `module_op.walk([&](dataflow::ProgramUnitOp unit_op) { .. })` — the units are a field of the
    // program rather than ops among ops, so the walk for them is an iteration.
    for unit in program.units.iter() {
        let comp = unit.on.kind();
        // `if (is_any_of(comp, LXLU, LXSU, L3LU, L3SU))`.
        if !PAGED_TENSOR_UNITS.contains(&comp.generic()) {
            continue;
        }

        // The `while (true)`, over the views one walk finds — see the note above for why the
        // re-walk becomes a single collection plus the repeat test below.
        let mut candidates: Vec<PagedMemView<'_>> = Vec::new();
        paged_views(&unit.body, &mut candidates);

        let mut rounds: Vec<DePagingRound<'_>> = Vec::new();
        for view in candidates {
            // `TPMVManager tpmv_manager(candidate, comp); tpmv_manager.run()`.
            let selection = TpmvManager::new(view, comp).run(&unit.body);
            // The view an earlier round's `transform()` erased: its answer is that round's answer.
            if rounds.iter().any(|round| round.selection == selection) {
                continue;
            }
            let stops_here = !matches!(selection, Selection::Selected(_));
            rounds.push(DePagingRound { view, selection });
            if stops_here {
                break;
            }
        }

        per_unit.push(UnitDePaging { comp, rounds });
    }

    per_unit
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::arch::Dd2;
    use crate::generated::OpFunc;
    // ⭐ ALIASED for the reason entry 139's tests record: this module's [`PagedMemView`] is the
    // manager's `dyn_cast` handle, the island's is the op's payload.
    use crate::islands::dataflow_ir::dialects::dataflow::{
        Page, PageRect, PageSpan, PagedMemView as PagedMemViewOp,
    };
    use crate::islands::dataflow_ir::dialects::{Index, Val, affine, agen};
    use crate::islands::dataflow_ir::ty::{AffineExpr, AffineMap, ElemType, MemRef, Vector};
    use crate::islands::dataflow_ir::{
        Grid, GroupId, OpIndex, ProgramName, ProgramUnit, ProgramUnits, Units,
    };
    use crate::units::Row;

    use super::super::tf_transform_paged_mem_view_impl::{
        TpmvCompositeLoadStore, TpmvVectorLoad, TpmvVectorLoadStore,
    };
    use super::super::tf_transform_paged_mem_view_manager::Tpmv;

    /// `memref<?x64x4xf16>` as the vendor's paged-view tests declare it, with the dynamic outer
    /// extent written as 1 — nothing in this pass reads the shape.
    fn view_ty() -> MemRef {
        MemRef {
            shape: vec![1, 64, 4],
            elem: ElemType::F16,
        }
    }

    /// `vector<64xf16>`.
    const LANES: Vector = Vector {
        len: 64,
        elem: ElemType::F16,
    };

    /// `%view = dataflow.get_paged_logical_memory_view %lx, %c0 {..} {segments = ..}` —
    /// `paged_mem_view_loads.mlir:289-296`, reduced to one page. The page count is what entries 309,
    /// 310 and 324 read; this pass only asks whether the op IS one.
    fn paged_view(result: Val) -> DfirOp {
        DfirOp::Dataflow(
            crate::islands::dataflow_ir::dialects::dataflow::Op::GetPagedLogicalMemoryView(
                Box::new(PagedMemViewOp {
                    result,
                    unit: Val(1),
                    start_addr: Val(2),
                    pages: vec![Page {
                        idx_set: PageRect {
                            spans: vec![
                                PageSpan { lo: 0, hi: 0 },
                                PageSpan { lo: 0, hi: 63 },
                                PageSpan { lo: 0, hi: 3 },
                            ],
                        },
                        start_addr: Val(2),
                    }],
                    layout: AffineMap {
                        dims: 3,
                        syms: 0,
                        results: vec![
                            AffineExpr::dim(2)
                                .times(256)
                                .plus(AffineExpr::dim(1).times(64))
                                .plus(AffineExpr::dim(0)),
                        ],
                    },
                    ty: view_ty(),
                }),
            ),
        )
    }

    /// `%load = agen.vector_load %view[..] : memref<?x64x4xf16>, vector<64xf16>`.
    fn load(result: Val, view: Val) -> DfirOp {
        DfirOp::Agen(agen::Op::VectorLoad {
            dbg_name: None,
            access: agen::Access::OfView,
            result,
            view,
            indices: vec![Index::Const(0), Index::Const(0), Index::Const(0)],
            view_ty: view_ty(),
            ty: LANES,
        })
    }

    /// `agen.vector_store %value, %view[..] : memref<?x64x4xf16>, vector<64xf16>`.
    fn store(value: Val, view: Val) -> DfirOp {
        DfirOp::Agen(agen::Op::VectorStore {
            value,
            view,
            indices: vec![Index::Const(0), Index::Const(0), Index::Const(0)],
            view_ty: view_ty(),
            ty: LANES,
        })
    }

    /// `agen.composite_load_and_store src:%src[..] dst:%dst[..] { agen.yield }` —
    /// `paged_mem_view_load_and_store.mlir:1094-1107`, the one transfer op the island declares.
    fn transfer(src: Val, dst: Val) -> DfirOp {
        use crate::bridges::subtile_to_dataflow_ir::transfer::{Lanes, plan};

        let planned = plan(&[1, 1, 64], &[1, 1, 64], 64, Lanes::F16)
            .expect("one 64-lane vector is the unsplit case");
        DfirOp::Agen(agen::Op::CompositeLoadAndStore(Box::new(
            agen::CompositeTransfer {
                src,
                src_indices: vec![Index::Const(0), Index::Const(0), Index::Const(0)],
                src_ty: view_ty(),
                dst,
                dst_indices: vec![Index::Const(0), Index::Const(0), Index::Const(0)],
                dst_ty: view_ty(),
                load_iv: Val(20),
                load_iv_ty: LANES,
                load_set: planned.load_set,
                load_order: planned.load_order,
                store_set: planned.store_set,
                store_order: planned.store_order,
                time_set: planned.time_set,
                time_order: planned.time_order,
                load_time_addr_map: planned.load_time_addr_map,
                store_time_addr_map: planned.store_time_addr_map,
                body: vec![DfirOp::Agen(agen::Op::Yield)],
            },
        )))
    }

    /// `affine.for %iv = 0 to <hi> { <body> }` — the nest the vendor's views sit inside.
    fn nest(iv: Val, hi: i64, body: Vec<DfirOp>) -> DfirOp {
        DfirOp::Affine(affine::Op::For {
            iv,
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(hi),
            carried: Vec::new(),
            body,
            dbg_name: None,
        })
    }

    /// A program whose units are exactly these `(component, body)` pairs.
    ///
    /// ⭐ NO `dataflow.get_unit` IN THE PREAMBLE, because the port reads the component off
    /// [`Units::kind`] — see the note on [`run_on_operation`] for why that is the same fact as the
    /// reference's `getUnits()[0].getDefiningOp<GetUnitOp>()`.
    fn program(units: Vec<(DfirUnit, Vec<DfirOp>)>) -> Program<Dd2> {
        let mut built = units.into_iter().map(|(on, body)| ProgramUnit {
            on: Units::one(on, Val(1)),
            precision: None,
            body,
            arch: core::marker::PhantomData,
        });
        let head = built.next().expect("the fixture names at least one unit");
        Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            grid: Grid::single(),
            preamble: Vec::new(),
            units: ProgramUnits::of(head, built.collect()),
            arch: core::marker::PhantomData,
        }
    }

    /// 🎯 196/384 — THE VENDOR'S OWN TWO-VIEW LOAD UNIT. `paged_mem_view_loads.mlir:285-313`: one
    /// `lxlu` program unit, two `affine.for`s deep, holding `%mem_view` + `%load` (`:289-297`) and
    /// then `%mem_view2` + `%load2` (`:301-309`). Two views with two different users is two rounds,
    /// and the views are found in block order despite sitting inside the nest.
    #[test]
    fn two_loads_in_one_unit_are_two_rounds() {
        let body = vec![nest(
            Val(30),
            2,
            vec![nest(
                Val(31),
                4,
                vec![
                    paged_view(Val(10)),
                    load(Val(20), Val(10)),
                    paged_view(Val(11)),
                    load(Val(21), Val(11)),
                ],
            )],
        )];
        let program = program(vec![(DfirUnit::Lxlu, body)]);

        let de_paging = run_on_operation(&program);

        let [unit] = de_paging.as_slice() else {
            panic!("one program unit, one entry");
        };
        assert_eq!(unit.comp, DfirUnit::Lxlu);
        let first_load = load(Val(20), Val(10));
        let second_load = load(Val(21), Val(11));
        assert_eq!(
            unit.rounds
                .iter()
                .map(|round| (round.view.result, round.selection.clone()))
                .collect::<Vec<_>>(),
            vec![
                (
                    Val(10),
                    Selection::Selected(Tpmv::VectorLoad(TpmvVectorLoad::new(
                        &first_load,
                        DfirUnit::Lxlu
                    )))
                ),
                (
                    Val(11),
                    Selection::Selected(Tpmv::VectorLoad(TpmvVectorLoad::new(
                        &second_load,
                        DfirUnit::Lxlu
                    )))
                ),
            ]
        );
    }

    /// 🎯 196/384 ⛔⛔ THE VENDOR'S COMPOSITE UNIT IS **ONE** ROUND FOR **TWO** VIEWS.
    /// `paged_mem_view_load_and_store.mlir:1074-1121`: an `l3su` unit whose
    /// `agen.composite_load_and_store` reads `%src_mem_view` (`:1084`) and writes `%dst_mem_view`
    /// (`:1088`). `TPMVCompositeLoadStore::initialize` registers both (`Impl.cpp:1191-1202`) and
    /// `transform` erases both (`:621`), so the second walk finds nothing — the case a
    /// one-round-per-view port would double.
    #[test]
    fn a_composite_transfers_src_and_dst_are_one_round() {
        let body = vec![
            paged_view(Val(10)),
            paged_view(Val(11)),
            transfer(Val(10), Val(11)),
        ];
        let program = program(vec![(DfirUnit::L3su, body)]);

        let de_paging = run_on_operation(&program);

        let [unit] = de_paging.as_slice() else {
            panic!("one program unit, one entry");
        };
        let expected = transfer(Val(10), Val(11));
        assert_eq!(
            unit.rounds,
            vec![DePagingRound {
                view: PagedMemView::of(&paged_view(Val(10))).expect("a paged view"),
                selection: Selection::Selected(Tpmv::CompositeLoadStore(
                    TpmvCompositeLoadStore::new(&expected, DfirUnit::L3su)
                )),
            }]
        );
    }

    /// ⛔ AND SO IS THE VECTOR PAIR, WHOSE TWO VIEWS HAVE **DIFFERENT** USERS. The load reads
    /// `%src` and the store writes `%dst`, so neither view is the other's — but
    /// `TPMVVectorLoadStore::initialize` reaches the store through `getStoreOp` and registers its
    /// view too (`Impl.cpp:766-770`). Both views answer with the same leaf because the manager
    /// normalises the pair to `(load_op, store_op)` from either end
    /// (`TransformPagedMemViewManager.cpp:33`, `:45`), which is what makes the repeat test the erase
    /// set.
    #[test]
    fn a_vector_load_and_store_pair_is_one_round() {
        let body = vec![
            paged_view(Val(10)),
            paged_view(Val(11)),
            load(Val(20), Val(10)),
            store(Val(20), Val(11)),
        ];
        let program = program(vec![(DfirUnit::Lxsu, body)]);

        let de_paging = run_on_operation(&program);

        let [unit] = de_paging.as_slice() else {
            panic!("one program unit, one entry");
        };
        let paired_load = load(Val(20), Val(10));
        let paired_store = store(Val(20), Val(11));
        assert_eq!(
            unit.rounds,
            vec![DePagingRound {
                view: PagedMemView::of(&paged_view(Val(10))).expect("a paged view"),
                selection: Selection::Selected(Tpmv::VectorLoadStore(TpmvVectorLoadStore::new(
                    &paired_load,
                    &paired_store,
                    DfirUnit::Lxsu
                ))),
            }]
        );
    }

    /// ⛔⛔ FOUR COMPONENTS, NOT SIX. `is_any_of(comp, LXLU, LXSU, L3LU, L3SU)` (`:46-47`) — the L0
    /// halves are in entry 194's `LOAD_AND_STORE_UNITS` and NOT here, so a unit on one keeps its
    /// paged view. Every other kind is listed too: the filter is what decides whether the unit is
    /// entered at all.
    #[test]
    fn only_the_four_paged_tensor_components_are_visited() {
        let visited = [
            DfirUnit::Lxlu,
            DfirUnit::Lxsu,
            DfirUnit::L3lu,
            DfirUnit::L3su,
        ];
        let left_alone = [
            DfirUnit::L0lu,
            DfirUnit::L0su,
            DfirUnit::Sfp,
            DfirUnit::Pe,
            DfirUnit::PtRow(Row::checked(0).expect("row 0 exists")),
            DfirUnit::Lx,
            DfirUnit::Hbm,
            DfirUnit::L0,
            DfirUnit::Constant,
            DfirUnit::SfpState,
            DfirUnit::PeState,
            DfirUnit::SfpRing,
            DfirUnit::LxVirtualIbr,
            DfirUnit::CrossPtnLink,
        ];

        for comp in visited {
            let body = vec![paged_view(Val(10)), load(Val(20), Val(10))];
            let program = program(vec![(comp, body)]);
            let de_paging = run_on_operation(&program);
            assert_eq!(de_paging.len(), 1, "{comp:?} is one of the four");
            assert_eq!(de_paging[0].rounds.len(), 1, "{comp:?} lowers its view");
        }

        for comp in left_alone {
            let body = vec![paged_view(Val(10)), load(Val(20), Val(10))];
            let program = program(vec![(comp, body)]);
            assert!(
                run_on_operation(&program).is_empty(),
                "{comp:?} is not entered"
            );
        }
    }

    /// ⭐ ENTERED AND IDLE IS ITS OWN ANSWER. `while (true)` runs, the walk sets no `candidate`, and
    /// `if (!candidate) break` (`:54-55`) leaves the unit untouched — which is not the same event as
    /// the component filter refusing it.
    #[test]
    fn a_selected_unit_with_no_paged_view_gets_no_rounds() {
        let program = program(vec![(DfirUnit::L3lu, vec![load(Val(20), Val(9))])]);

        let de_paging = run_on_operation(&program);

        assert_eq!(
            de_paging,
            vec![UnitDePaging {
                comp: DfirUnit::L3lu,
                rounds: Vec::new(),
            }]
        );
    }

    /// ⛔ AN ABORT ENDS THE UNIT'S ROUNDS. The first view has two users, which is
    /// `DT_CHECK(paged_mem_view_->hasOneUse())` (`TransformPagedMemViewManager.cpp:23-24`) — the
    /// reference stops the compile there, and the later view it never reaches gets no round here.
    #[test]
    fn an_abort_ends_the_unit_and_leaves_the_next_view_alone() {
        let body = vec![
            paged_view(Val(10)),
            load(Val(20), Val(10)),
            load(Val(21), Val(10)),
            paged_view(Val(11)),
            load(Val(22), Val(11)),
        ];
        let program = program(vec![(DfirUnit::Lxlu, body)]);

        let de_paging = run_on_operation(&program);

        let [unit] = de_paging.as_slice() else {
            panic!("one program unit, one entry");
        };
        assert_eq!(
            unit.rounds,
            vec![DePagingRound {
                view: PagedMemView::of(&paged_view(Val(10))).expect("a paged view"),
                selection: Selection::NotOneUser,
            }]
        );
    }

    /// ⛔ AND IT ENDS ONLY THAT UNIT. `signalPassFailure(); return;` (`:58-59`) returns from a walk
    /// lambda declared `void`, so the walk carries on to the next `dataflow.program_unit` — the
    /// second unit below is still lowered.
    #[test]
    fn an_abort_in_one_unit_does_not_stop_the_walk() {
        let aborting = vec![paged_view(Val(10))];
        let clean = vec![paged_view(Val(11)), load(Val(21), Val(11))];
        let program = program(vec![(DfirUnit::Lxlu, aborting), (DfirUnit::L3lu, clean)]);

        let de_paging = run_on_operation(&program);

        let second_load = load(Val(21), Val(11));
        assert_eq!(
            de_paging.iter().map(|unit| unit.comp).collect::<Vec<_>>(),
            vec![DfirUnit::Lxlu, DfirUnit::L3lu]
        );
        assert_eq!(de_paging[0].rounds[0].selection, Selection::NotOneUser);
        assert_eq!(
            de_paging[1].rounds[0].selection,
            Selection::Selected(Tpmv::VectorLoad(TpmvVectorLoad::new(
                &second_load,
                DfirUnit::L3lu
            )))
        );
    }
}
