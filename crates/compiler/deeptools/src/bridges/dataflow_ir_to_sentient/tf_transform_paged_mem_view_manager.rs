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

//! `TransformPagedMemViewManager.cpp` — 1 of bridge 2's 384 functions (dependency level(s) [0]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e139_run` | 139/384 | 48 | `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewManager.cpp:21` |

use super::tf_transform_paged_mem_view_impl::{
    TpmvCompositeLoad, TpmvCompositeLoadStore, TpmvCompositeStore, TpmvVectorLoad,
    TpmvVectorLoadStore, TpmvVectorStore,
};
use crate::islands::dataflow_ir::dialects::{Op as DfirOp, Val, agen, dataflow, defining_op, uses};
use crate::units::DfirUnit;

/// A `dataflow.get_paged_logical_memory_view` AND NOTHING ELSE — the reference's
/// `dataflow::GetPagedLogicalMemoryViewOp` handle.
///
/// ⭐ THE CAST IS THE DOOR. Everything downstream of it in the pass already knows the op is a paged
/// view: `TransformPagedMemView.cpp:49-51` walks for one, and the impl re-`cast<>`s the same handle
/// six more times (`Impl.cpp:396`, `:512`, `:663`, `:711`, `:1085`, `:1138`). Making that a type
/// carried by the manager retires all seven, and the only fallible step is [`Self::of`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedMemView<'a> {
    /// The op itself.
    pub op: &'a DfirOp,
    /// `$data` — the view handle every access reads.
    pub result: Val,
    /// THE OP'S OWN PAYLOAD — `getIdxSets()`, `getPageStartAddrs()`, `getStartAddr()`, `getUnit()`
    /// and `getLayoutMap()`, all of which entries 124/309/310 read off the very handle the `cast`
    /// produced. Carried here because the cast already proved which op it is, so re-matching
    /// [`Self::op`] downstream would reintroduce an [`Option`] the door has closed.
    pub view: &'a dataflow::PagedMemView,
}

impl<'a> PagedMemView<'a> {
    /// `dyn_cast<dataflow::GetPagedLogicalMemoryViewOp>(op)`.
    #[must_use]
    pub fn of(op: &'a DfirOp) -> Option<PagedMemView<'a>> {
        match op {
            DfirOp::Dataflow(dataflow::Op::GetPagedLogicalMemoryView(view)) => Some(PagedMemView {
                op,
                result: view.result,
                view,
            }),
            DfirOp::Arith(_)
            | DfirOp::Affine(_)
            | DfirOp::Scf(_)
            | DfirOp::Dataflow(_)
            | DfirOp::Agen(_)
            | DfirOp::Vector(_)
            | DfirOp::VectorChain(_)
            // ⭐ `symbol` WITH THEM: `symbol.create_symbol` binds an `index`, never a memref, so it
            // is never a paged view — one more `dyn_cast<GetPagedLogicalMemoryViewOp>` null.
            // ⭐ AND `uniform` WITH IT, for the same reason: no `uniform.` op binds a memref.
            | DfirOp::Uniform(_)
            | DfirOp::Symbol(_) => None,
        }
    }
}

/// WHICH TRANSFORMATION A PAGED VIEW GETS — one variant per leaf of the `TPMVBase` hierarchy.
///
/// ⛔ SIX VARIANTS BECAUSE THE REFERENCE CONSTRUCTS SIX DIFFERENT CLASSES
/// (`TransformPagedMemViewManager.cpp:33`, `:38`, `:45`, `:49`, `:53`, `:57`, `:62`), and which one
/// it picks decides which `initialize`/`createNewMemOp` override runs. Collapsing them would make
/// [`run`] a function that answers the same thing every time. See
/// [`super::tf_transform_paged_mem_view_impl::TpmvBase`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tpmv<'a> {
    /// `TPMVVectorLoad tpmv(op, comp_);` (`:38`).
    VectorLoad(TpmvVectorLoad<'a>),
    /// `TPMVVectorStore tpmv(op, comp_);` (`:49`).
    VectorStore(TpmvVectorStore<'a>),
    /// `TPMVVectorLoadStore tpmv(load_op, store_op, comp_);` (`:33`, `:45`) — reached from BOTH
    /// vector arms.
    VectorLoadStore(TpmvVectorLoadStore<'a>),
    /// `TPMVCompositeLoad tpmv(op, comp_);` (`:53`).
    CompositeLoad(TpmvCompositeLoad<'a>),
    /// `TPMVCompositeStore tpmv(op, comp_);` (`:57`).
    CompositeStore(TpmvCompositeStore<'a>),
    /// `TPMVCompositeLoadStore tpmv(op, comp_);` (`:62`).
    CompositeLoadStore(TpmvCompositeLoadStore<'a>),
}

/// WHAT THE MANAGER'S DISPATCH FOUND — the strategy, or the reason there is none.
///
/// ⛔ THE TWO FAILURE CASES ARE DIFFERENT ABORTS IN THE REFERENCE AND ARE KEPT APART:
/// `DT_CHECK(paged_mem_view_->hasOneUse() && "expecting paged memory view to have one user")`
/// (`TransformPagedMemViewManager.cpp:23-24`) and
/// `llvm_unreachable("memory operation is not supported for static paged tensors")` (`:65-66`).
/// Both are hard stops there; here they are values, because the crate never runtime-refuses
/// (`CLAUDE.md`) and the caller — `TransformPagedMemView.cpp:49-73`, entry 350 — is what decides
/// what to say about them.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub enum Selection<'a> {
    /// The leaf the reference would have constructed and run.
    Selected(Tpmv<'a>),
    /// `DT_CHECK(paged_mem_view_->hasOneUse() && ..)` — the view is read by zero users or by more
    /// than one.
    NotOneUser,
    /// `llvm_unreachable("memory operation is not supported for static paged tensors")` — the single
    /// user is none of the five memory operations.
    Unsupported,
}

/// THE PASS'S ENTRY POINT FOR ONE PAGED VIEW — `TPMVManager` (`TransformPagedMemViewManager.hpp:20-31`).
///
/// *"Controls running the transformation to lower `dataflow::GetPagedLogicalMemoryViewOp`"*
/// (`hpp:8-9`). It holds the view and the component, and its one method chooses the strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TpmvManager<'a> {
    /// `paged_mem_view_` — the view to lower (`hpp:29`).
    ///
    /// ⭐ THE C++ MEMBER IS A `GetPagedLogicalMemoryViewOp &`, a reference to a HANDLE, and an MLIR
    /// op handle is already a pointer-sized value — so the reference buys mutation of the caller's
    /// variable, which nothing in `run()` does. A shared borrow is the same access.
    pub paged_mem_view: PagedMemView<'a>,
    /// `comp_` — the component being lowered for (`hpp:30`).
    pub comp: DfirUnit,
}

impl<'a> TpmvManager<'a> {
    /// THE MANAGER'S CONSTRUCTOR — `TPMVManager(GetPagedLogicalMemoryViewOp &paged_mem_view,
    /// SenComponents comp) : paged_mem_view_(paged_mem_view), comp_(comp) {}`
    /// (`TransformPagedMemViewManager.hpp:22-24`).
    ///
    /// ⭐ NOT A SCHEDULED UNIT: the extractor caught it as the data member `comp_`
    /// (`TransformPagedMemViewManager.hpp:24`) and it is among the excluded *"C++ data MEMBER, not a
    /// function"* entries (`docs/bridge2-porting-order.md:2963`), so it carries no anchor. Written
    /// here because entry 139 is a method on it.
    #[must_use]
    pub fn new(paged_mem_view: PagedMemView<'a>, comp: DfirUnit) -> TpmvManager<'a> {
        TpmvManager {
            paged_mem_view,
            comp,
        }
    }

    /// Replaces: e139_run
    ///
    /// **139/384** `TPMVManager::run` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewManager.cpp:21` (48L).
    ///
    /// ```cpp
    /// LogicalResult TPMVManager::run() {
    ///   Operation *op = nullptr;
    ///   DT_CHECK(paged_mem_view_->hasOneUse() &&
    ///            "expecting paged memory view to have one user");
    ///   op = *paged_mem_view_->getUsers().begin();
    ///   DT_CHECK(op);
    ///
    ///   if (auto load_op = dyn_cast<agen::VectorLoadOp>(op)) {
    ///     //// agen::VectorLoadOp + agen::VectorStoreOp ////
    ///     auto result = load_op.getResult();
    ///     if (result.hasOneUse()) {
    ///       if (auto store_op = dyn_cast<agen::VectorStoreOp>(*result.user_begin())) {
    ///         TPMVVectorLoadStore tpmv(load_op, store_op, comp_);
    ///         return tpmv.run();
    ///       }
    ///     }
    ///     //// agen::VectorLoadOp ////
    ///     TPMVVectorLoad tpmv(op, comp_);
    ///     return tpmv.run();
    ///   } else if (auto store_op = dyn_cast<agen::VectorStoreOp>(op)) {
    ///     //// agen::VectorLoadOp + agen::VectorStoreOp ////
    ///     auto val = store_op.getValueToStore();
    ///     if (auto load_op =
    ///             dyn_cast_or_null<agen::VectorLoadOp>(val.getDefiningOp())) {
    ///       TPMVVectorLoadStore tpmv(load_op, store_op, comp_);
    ///       return tpmv.run();
    ///     }
    ///     //// agen::VectorStoreOp ////
    ///     TPMVVectorStore tpmv(op, comp_);
    ///     return tpmv.run();
    ///   } else if (auto comp_load_op = dyn_cast<agen::CompositeLoadOp>(op)) {
    ///     //// agen::CompositeLoadOp ////
    ///     TPMVCompositeLoad tpmv(op, comp_);
    ///     return tpmv.run();
    ///   } else if (auto comp_store_op = dyn_cast<agen::CompositeStoreOp>(op)) {
    ///     //// agen::CompositeStoreOp ////
    ///     TPMVCompositeStore tpmv(op, comp_);
    ///     return tpmv.run();
    ///   } else if (auto comp_load_store_op =
    ///                  dyn_cast<agen::CompositeLoadAndStoreOp>(op)) {
    ///     //// agen::CompositeLoadAndStoreOp ////
    ///     TPMVCompositeLoadStore tpmv(op, comp_);
    ///     return tpmv.run();
    ///   } else {
    ///     llvm_unreachable(
    ///         "memory operation is not supported for static paged tensors");
    ///   }
    ///   return LogicalResult::failure();
    /// }
    /// ```
    ///
    /// # ⛔⛔ THE ISLAND GREW FOR THIS ENTRY, AS THE BRIEF REQUIRES
    ///
    /// `dataflow.get_paged_logical_memory_view` is this function's ONLY input and the island did not
    /// have it, so it is now [`dataflow::Op::GetPagedLogicalMemoryView`] — the brief's *"if the target
    /// IR cannot express a function's input, add the operation to the island"* (`AGENT-BRIEF.md:87`).
    /// The op is `Dataflow.td:267-299`; its verifier's *"there should be a start address and idx_set
    /// for every page"* (`DataflowOps.cpp:277-279`) became one vector of
    /// [`dataflow::Page`] pairs rather than two lists that can disagree.
    ///
    /// # ⛔ WHAT THIS RETURNS IS THE SELECTION, AND THAT IS THE WHOLE FUNCTION MINUS ONE CALL
    ///
    /// Every arm ends `return tpmv.run()`, and there are exactly two of those: `TPMVVector::run`
    /// (entry 373, `Impl.cpp:647`) and `TPMVComposite::run` (`Impl.cpp:855` — the one definition the
    /// winnow lost, deduplicated by the name `run` against 373). Neither is ported, and the crate
    /// forbids the `LogicalResult` they answer with. So this function does its own work — the use
    /// walk, the two def walks and the six-way dispatch — and hands back WHICH leaf runs. When 373
    /// lands, its call becomes this function's tail; nothing about the dispatch changes.
    ///
    /// # ⭐ THE TWO ARMS THAT REACH `TPMVVectorLoadStore` ARE NOT SYMMETRIC
    ///
    /// From the load arm, `mem_op` is the load and the store is found FORWARD, through the load
    /// result's single user (`:29-34`). From the store arm, `mem_op` is *still the load* — found
    /// BACKWARD, through `getValueToStore().getDefiningOp()` (`:41-45`) — so the op the user search
    /// returned is the one that gets dropped. The vendor's own case has both views on one pair:
    /// `%data = agen.vector_load %src_mem_view[..]` then
    /// `agen.vector_store %data, %dst_mem_view[..]`
    /// (`paged_mem_view_load_and_store.mlir:1287-1289`) — the src view enters by the first arm, the
    /// dst view by the second, and both select this leaf.
    ///
    /// # ⛔ THE LOAD ARM'S INNER TEST IS A COUNT, AND ITS FAILURE IS NOT AN ERROR
    ///
    /// `if (result.hasOneUse())` guards the store lookup only. A load whose result is read twice, or
    /// read once by something that is not a store, falls through to plain `TPMVVectorLoad` — which is
    /// what the vendor's rotate case does (`paged_mem_view_loads.mlir:296-298`: load, `vectorchain.rotate`,
    /// `dataflow.send`). [`uses`] returns one entry per USE, so a value read twice by one op is not
    /// single-used.
    ///
    /// # ⛔ ONE OF THE FIVE MEMORY CLASSES HAS NO ISLAND OP, AND THAT IS RECORDED, NOT INVENTED
    ///
    /// `agen.composite_load` (`paged_mem_view_loads.mlir:331`) IS declared now — entry 326
    /// `TPMVCompositeLoad::initialize_time` `dyn_cast`s exactly it, so `AGENT-BRIEF.md:87` applied —
    /// and this selector reaches [`Tpmv::CompositeLoad`] for real. `agen.composite_store`
    /// (`paged_mem_view_stores.mlir:361`) is still absent, so [`Tpmv::CompositeStore`] stays
    /// reachable from a test and from nothing the crate emits.
    ///
    /// # ⭐ THE TWO THINGS WITH NO COUNTERPART
    ///
    /// `DT_CHECK(op)` (`:26`) asserts the user pointer is non-null; a use here IS an op, so there is
    /// nothing to check. And `return LogicalResult::failure()` (`:68`) is unreachable — every arm
    /// returns and the `else` is `llvm_unreachable` — so it exists only because C++ cannot see that.
    pub fn run(&self, scope: &'a [DfirOp]) -> Selection<'a> {
        // `DT_CHECK(paged_mem_view_->hasOneUse() && ..)`, then
        // `op = *paged_mem_view_->getUsers().begin()`.
        let users = uses(self.paged_mem_view.result, scope);
        let [op] = users.as_slice() else {
            return Selection::NotOneUser;
        };
        let op = *op;

        match op {
            // `if (auto load_op = dyn_cast<agen::VectorLoadOp>(op))`.
            DfirOp::Agen(agen::Op::VectorLoad { result, .. }) => {
                // `auto result = load_op.getResult(); if (result.hasOneUse())`.
                if let [user] = uses(*result, scope).as_slice() {
                    // `if (auto store_op = dyn_cast<agen::VectorStoreOp>(*result.user_begin()))`.
                    if matches!(user, DfirOp::Agen(agen::Op::VectorStore { .. })) {
                        return Selection::Selected(Tpmv::VectorLoadStore(
                            TpmvVectorLoadStore::new(op, user, self.comp),
                        ));
                    }
                }
                // `TPMVVectorLoad tpmv(op, comp_)` — `op` and `load_op` are the same op here.
                Selection::Selected(Tpmv::VectorLoad(TpmvVectorLoad::new(op, self.comp)))
            }
            // `else if (auto store_op = dyn_cast<agen::VectorStoreOp>(op))`.
            DfirOp::Agen(agen::Op::VectorStore { value, .. }) => {
                // `auto val = store_op.getValueToStore();` then
                // `dyn_cast_or_null<agen::VectorLoadOp>(val.getDefiningOp())` — the `_or_null` is
                // the case where the stored vector is a region argument, which [`defining_op`]
                // answers [`None`] for.
                if let Some(load) = defining_op(*value, scope)
                    && matches!(load, DfirOp::Agen(agen::Op::VectorLoad { .. }))
                {
                    // ⛔ `load_op`, NOT `op` — see the asymmetry note.
                    return Selection::Selected(Tpmv::VectorLoadStore(TpmvVectorLoadStore::new(
                        load, op, self.comp,
                    )));
                }
                Selection::Selected(Tpmv::VectorStore(TpmvVectorStore::new(op, self.comp)))
            }
            // `else if (auto comp_load_op = dyn_cast<agen::CompositeLoadOp>(op))` (`:51-53`).
            DfirOp::Agen(agen::Op::CompositeLoad(_)) => {
                Selection::Selected(Tpmv::CompositeLoad(TpmvCompositeLoad::new(op, self.comp)))
            }
            // `else if (auto comp_load_store_op = dyn_cast<agen::CompositeLoadAndStoreOp>(op))` —
            // the LAST of the three composite arms in the reference. `CompositeStoreOp`, between
            // them, has no island op; see the note above.
            DfirOp::Agen(agen::Op::CompositeLoadAndStore(_)) => Selection::Selected(
                Tpmv::CompositeLoadStore(TpmvCompositeLoadStore::new(op, self.comp)),
            ),
            // `else { llvm_unreachable("memory operation is not supported for static paged
            // tensors"); }` — no wildcard, so an eleventh `agen` op cannot land here silently.
            // ⛔ AND A SYMBOLIC ACCESS IS THAT ARM TOO. Its address is a runtime value, which is the
            // opposite of what a STATIC paged tensor is; none of the reference's `dyn_cast`s names it.
            DfirOp::Agen(
                agen::Op::Yield
                | agen::Op::SymbolicVectorLoad { .. }
                | agen::Op::SymbolicVectorStore { .. }
                | agen::Op::IndirectVectorLoad { .. }
                | agen::Op::IndirectVectorStore { .. }
                | agen::Op::CompositeMemoryInterleave { .. }
                | agen::Op::CompositeIndirectLoadAndStore(_)
                | agen::Op::SetTransferMaskState { .. },
            )
            | DfirOp::Arith(_)
            | DfirOp::Affine(_)
            | DfirOp::Scf(_)
            | DfirOp::Dataflow(_)
            // ⛔ AN UPSTREAM `vector.load` IS THE `llvm_unreachable` ARM, NOT A FOURTH TRANSFORM.
            // The four `dyn_cast`s the reference tries are all `agen` (`:44-67`), so a paged view
            // whose one user is a plain `vector` access falls to *"memory operation is not supported
            // for static paged tensors"*.
            | DfirOp::Vector(_)
            | DfirOp::VectorChain(_)
            // ⭐ AND `symbol` WITH THEM: `symbol.create_symbol` takes no operands at all
            // (`NoMemoryEffect`, `Symbol.td:53`), so it cannot be a use of the view's result;
            // reaching this arm through one would mean the use list lied.
            // ⭐ AND `uniform` WITH THEM: a `uniformize_regions` takes `!ddl.unit` operands and the
            // mapping ops take `index`, so none of them can be a use of a paged view's memref.
            | DfirOp::Uniform(_)
            | DfirOp::Symbol(_) => Selection::Unsupported,
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    // ⭐ ALIASED: this module's own [`PagedMemView`] is the manager's `dyn_cast` handle, the island's
    // is the op's payload — the same name for the op and for the cast of it.
    use crate::islands::dataflow_ir::dialects::dataflow::{
        Page, PageRect, PageSpan, PagedMemView as PagedMemViewOp,
    };
    use crate::islands::dataflow_ir::dialects::{Index, agen, dataflow, vectorchain};
    use crate::islands::dataflow_ir::link::{Link, Lxsu, Sfp};
    use crate::islands::dataflow_ir::ty::{AffineExpr, AffineMap, ElemType, MemRef, Vector};

    /// `memref<?x64x4xf16>` as the vendor's paged-view tests declare it, with the dynamic outer
    /// extent written as 1 — nothing in the dispatch reads the shape.
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
    /// `paged_mem_view_loads.mlir:289-296`, reduced to one page. Six pages and one page dispatch
    /// identically; the page count is what entries 309, 310 and 324 read, not this one.
    fn paged_view(result: Val) -> DfirOp {
        DfirOp::Dataflow(dataflow::Op::GetPagedLogicalMemoryView(Box::new(
            PagedMemViewOp {
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
            },
        )))
    }

    /// `%load = agen.vector_load %view[..] : memref<?x64x4xf16>, vector<64xf16>`.
    fn load(result: Val, view: Val) -> DfirOp {
        DfirOp::Agen(agen::Op::VectorLoad {
            dbg_name: None,
            result,
            view,
            indices: vec![Index::Const(0), Index::Const(0), Index::Const(0)],
            view_ty: view_ty(),
            ty: LANES,
            multicast_info: None,
        })
    }

    /// `agen.vector_store %value, %view[..] : memref<?x64x4xf16>, vector<64xf16>`.
    fn store(value: Val, view: Val) -> DfirOp {
        DfirOp::Agen(agen::Op::VectorStore {
            dbg_name: None,
            value,
            view,
            indices: vec![Index::Const(0), Index::Const(0), Index::Const(0)],
            view_ty: view_ty(),
            ty: LANES,
        })
    }

    /// `%rot = vectorchain.rotate %load, %c16 : vector<64xf16>, index, vector<64xf16>` —
    /// `paged_mem_view_loads.mlir:298`. The load consumer that is NOT a store.
    fn rotate(result: Val, input: Val) -> DfirOp {
        DfirOp::VectorChain(vectorchain::Op::Rotate {
            result,
            input,
            position: Val(3),
            right_shift: true,
            input_ty: LANES,
            ty: LANES,
        })
    }

    /// The manager for one view, over the LX load unit.
    fn manager(view: &DfirOp) -> TpmvManager<'_> {
        TpmvManager::new(
            PagedMemView::of(view).expect("the fixture is a paged view"),
            DfirUnit::Lxlu,
        )
    }

    /// 🎯 139/384 — the vendor's own rotate case. `%mem_view` → `%load` → `%rot` → `dataflow.send`
    /// (`paged_mem_view_loads.mlir:289-299`): the view has ONE user, that user is a
    /// `agen.vector_load`, and the load's single user is a rotate rather than a store — so
    /// `if (auto store_op = dyn_cast<agen::VectorStoreOp>(..))` fails and the arm falls through to
    /// plain `TPMVVectorLoad` (`TransformPagedMemViewManager.cpp:36-38`).
    #[test]
    fn vector_load_is_selected_when_the_loads_user_is_not_a_store() {
        let scope = vec![
            paged_view(Val(10)),
            load(Val(20), Val(10)),
            rotate(Val(21), Val(20)),
        ];

        assert_eq!(
            manager(&scope[0]).run(&scope),
            Selection::Selected(Tpmv::VectorLoad(TpmvVectorLoad::new(
                &scope[1],
                DfirUnit::Lxlu
            )))
        );
    }

    /// ⛔ `result.hasOneUse()` IS A COUNT AND FAILING IT IS NOT AN ERROR. A load read by two ops
    /// never even reaches the store cast, so it selects `TPMVVectorLoad` — the same leaf as the
    /// case above, by a different route through `:29-34`.
    #[test]
    fn vector_load_is_selected_when_the_load_result_has_two_users() {
        let scope = vec![
            paged_view(Val(10)),
            load(Val(20), Val(10)),
            rotate(Val(21), Val(20)),
            rotate(Val(22), Val(20)),
        ];

        assert_eq!(
            manager(&scope[0]).run(&scope),
            Selection::Selected(Tpmv::VectorLoad(TpmvVectorLoad::new(
                &scope[1],
                DfirUnit::Lxlu
            )))
        );
    }

    /// ⭐ AND WHEN THE LOAD RESULT IS READ BY NOBODY. `hasOneUse()` is false at zero as well as at
    /// two, and the reference treats both the same way.
    #[test]
    fn vector_load_is_selected_when_the_load_result_has_no_user() {
        let scope = vec![paged_view(Val(10)), load(Val(20), Val(10))];

        assert_eq!(
            manager(&scope[0]).run(&scope),
            Selection::Selected(Tpmv::VectorLoad(TpmvVectorLoad::new(
                &scope[1],
                DfirUnit::Lxlu
            )))
        );
    }

    /// 🎯 139/384 — the vendor's store case. `%data = dataflow.receive %pe0` then
    /// `agen.vector_store %data, %mem_view2[..]` (`paged_mem_view_stores.mlir:339-340`): the stored
    /// vector HAS a defining op and it is not a load, so
    /// `dyn_cast_or_null<agen::VectorLoadOp>(val.getDefiningOp())` fails and the arm selects
    /// `TPMVVectorStore` (`:48-50`).
    ///
    /// ⚠️ The vendor receives from a `pe`; this island mints a receive end from a
    /// [`Link`](crate::islands::dataflow_ir::link::Link), and `pe` is not one of its unit kinds. Who
    /// produced the vector is not what the dispatch reads — only that its producer is not a load.
    #[test]
    fn vector_store_is_selected_when_the_stored_value_comes_from_a_receive() {
        let (_send, recv) = Link::<Sfp, Lxsu>::between(Val(4), Val(5)).ends();
        let scope = vec![
            paged_view(Val(10)),
            DfirOp::Dataflow(dataflow::Op::Receive {
                result: Val(20),
                from: recv,
                ty: LANES,
            }),
            store(Val(20), Val(10)),
        ];

        assert_eq!(
            manager(&scope[0]).run(&scope),
            Selection::Selected(Tpmv::VectorStore(TpmvVectorStore::new(
                &scope[2],
                DfirUnit::Lxlu
            )))
        );
    }

    /// ⛔ THE `_or_null` IN `dyn_cast_or_null` IS THE REGION-ARGUMENT CASE. A vector that is a block
    /// argument has no defining op at all, which [`defining_op`] answers [`None`] for — and the arm
    /// must still select `TPMVVectorStore` rather than reading through a null.
    #[test]
    fn vector_store_is_selected_when_the_stored_value_has_no_defining_op() {
        let scope = vec![paged_view(Val(10)), store(Val(99), Val(10))];

        assert_eq!(
            manager(&scope[0]).run(&scope),
            Selection::Selected(Tpmv::VectorStore(TpmvVectorStore::new(
                &scope[1],
                DfirUnit::Lxlu
            )))
        );
    }

    /// 🎯 139/384 — the vendor's load-and-store pair, entered from the SOURCE view.
    /// `%data = agen.vector_load %src_mem_view[..]` then `agen.vector_store %data, %dst_mem_view[..]`
    /// (`paged_mem_view_load_and_store.mlir:1287-1289`). The src view's one user is the load, the
    /// load's one user is the store, so the load arm's forward lookup succeeds (`:29-34`).
    #[test]
    fn load_store_is_selected_from_the_load_arm() {
        let scope = vec![
            paged_view(Val(10)),
            paged_view(Val(11)),
            load(Val(20), Val(10)),
            store(Val(20), Val(11)),
        ];

        assert_eq!(
            manager(&scope[0]).run(&scope),
            Selection::Selected(Tpmv::VectorLoadStore(TpmvVectorLoadStore::new(
                &scope[2],
                &scope[3],
                DfirUnit::Lxlu
            )))
        );
    }

    /// ⛔⛔ THE SAME PAIR ENTERED FROM THE DESTINATION VIEW SELECTS THE SAME LEAF, AND `mem_ops_`
    /// HOLDS THE LOAD. `%dst_mem_view`'s one user is the STORE, and the store arm passes
    /// `load_op` — recovered backward through `getValueToStore().getDefiningOp()` — as the first
    /// argument (`:41-45`), so the op the use walk returned is the one that gets dropped. A port
    /// that passed `op` here would seed `mem_ops_` with the store and make entry 130's `getStoreOp`
    /// answer about the wrong op.
    #[test]
    fn load_store_is_selected_from_the_store_arm_and_carries_the_load() {
        let scope = vec![
            paged_view(Val(10)),
            paged_view(Val(11)),
            load(Val(20), Val(10)),
            store(Val(20), Val(11)),
        ];

        let selected = manager(&scope[1]).run(&scope);

        assert_eq!(
            selected,
            Selection::Selected(Tpmv::VectorLoadStore(TpmvVectorLoadStore::new(
                &scope[2],
                &scope[3],
                DfirUnit::Lxlu
            )))
        );
        let Selection::Selected(Tpmv::VectorLoadStore(tpmv)) = selected else {
            unreachable!("just asserted equal to that variant")
        };
        assert_eq!(
            tpmv.vector.base.mem_ops,
            vec![&scope[2]],
            "must be the LOAD"
        );
    }

    /// 🎯 139/384 — the composite arm. `agen.composite_load_and_store src:%src_mem_view[..]
    /// dst:%dst_mem_view[..]` (`paged_mem_view_load_and_store.mlir:1094`) is the third of the three
    /// composite casts and the only one the island can express; see the note on [`TpmvManager::run`]
    /// for why the other two are recorded absent rather than invented.
    ///
    /// ⛔ THE TWO VIEWS MUST BE DISTINCT VALUES. `uses` counts one entry per USE, so a transfer
    /// reading one view as both `src` and `dst` gives that view two users and the manager would
    /// answer [`Selection::NotOneUser`] — which is exactly right, and not this case.
    #[test]
    fn composite_load_store_is_selected() {
        use crate::bridges::subtile_to_dataflow_ir::transfer::{Lanes, plan};

        let planned = plan(&[1, 1, 64], &[1, 1, 64], 64, Lanes::F16)
            .expect("one 64-lane vector is the unsplit case");
        let transfer = DfirOp::Agen(agen::Op::CompositeLoadAndStore(Box::new(
            agen::CompositeTransfer {
                src: Val(10),
                src_indices: vec![Index::Const(0), Index::Const(0), Index::Const(0)],
                src_ty: view_ty(),
                dst: Val(11),
                dst_indices: vec![Index::Const(0), Index::Const(0), Index::Const(0)],
                dst_ty: view_ty(),
                load_iv: Val(20),
                load_iv_ty: LANES,
                load_set: planned.load_set,
                load_order: planned.load_order,
                store_set: planned.store_set,
                store_order: planned.store_order,
                time_symbols: Vec::new(),
                time_set: planned.time_set,
                time_order: planned.time_order,
                load_time_addr_map: planned.load_time_addr_map,
                store_time_addr_map: planned.store_time_addr_map,
                body: vec![DfirOp::Agen(agen::Op::Yield)],
                dir: None,
                multicast_info: None,
                dbg_name: None,
            },
        )));
        let scope = vec![paged_view(Val(10)), paged_view(Val(11)), transfer];

        assert_eq!(
            manager(&scope[0]).run(&scope),
            Selection::Selected(Tpmv::CompositeLoadStore(TpmvCompositeLoadStore::new(
                &scope[2],
                DfirUnit::Lxlu
            )))
        );
    }

    /// ⛔ `DT_CHECK(paged_mem_view_->hasOneUse() && "expecting paged memory view to have one user")`
    /// (`:23-24`) — a view nothing reads is not one user. The reference aborts the compile here; this
    /// crate never runtime-refuses, so the caller gets the reason as a value.
    #[test]
    fn not_one_user_when_nothing_reads_the_view() {
        let scope = vec![paged_view(Val(10))];

        assert_eq!(manager(&scope[0]).run(&scope), Selection::NotOneUser);
    }

    /// ⛔ AND NEITHER IS A VIEW READ BY TWO ACCESSES. `hasOneUse()` fails above one as well as below
    /// it, and the pass has no rule for which access would win.
    #[test]
    fn not_one_user_when_two_accesses_read_the_view() {
        let scope = vec![
            paged_view(Val(10)),
            load(Val(20), Val(10)),
            load(Val(21), Val(10)),
        ];

        assert_eq!(manager(&scope[0]).run(&scope), Selection::NotOneUser);
    }

    /// ⛔ `llvm_unreachable("memory operation is not supported for static paged tensors")` (`:65-66`)
    /// — the single user is an op that reads a view but is none of the five memory classes. The
    /// match has no wildcard, so a newly declared island op cannot fall in here by accident.
    #[test]
    fn unsupported_when_the_single_user_is_not_a_memory_operation() {
        let scope = vec![
            paged_view(Val(10)),
            DfirOp::Dataflow(dataflow::Op::ImplicitSync {
                view: Val(10),
                dst: Val(4),
                size: Val(5),
                view_ty: view_ty(),
            }),
        ];

        assert_eq!(manager(&scope[0]).run(&scope), Selection::Unsupported);
    }
}
