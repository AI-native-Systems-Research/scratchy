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

//! `CanonicalizeToggle.cpp` — 1 of bridge 2's 384 functions (dependency level(s) [1]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e178_runOnOperation` | 178/384 | 42 | `dcc/src/Transform/Dataflow/CanonicalizeToggle.cpp:49` |

use crate::arch::Arch;
use crate::islands::dataflow_ir::dialects::{
    Op as DfirOp, Val, affine, agen, arith, dataflow, defining_op, scf, uniform, uses,
};
use crate::islands::dataflow_ir::{self as dfir, ProgramUnit};

/// PREORDER, DESCENDING INTO EVERY REGION — `unit.walk([&](Operation *op) { .. })` (`:76`).
///
/// ⛔ A USE-WALK IS THE ONE MECHANISM THIS CAMPAIGN MAY SIMPLIFY: MLIR reaches nested ops through
/// `Region`/`Block`, this island nests them in `Vec`s. ⚠️ AND THE `WalkResult::interrupt()` AT `:84`
/// IS NOT SIMPLIFIED AWAY — it is the reason the caller's `while (has_changed)` exists, and it is
/// modelled by [`run_on_operation`] stopping at the first match rather than by a flag here.
fn walk_preorder(ops: &[DfirOp], visit: &mut impl FnMut(&DfirOp)) {
    for op in ops {
        visit(op);
        match op {
            DfirOp::Affine(affine::Op::For { body, .. })
            | DfirOp::Scf(scf::Op::Parallel { body, .. } | scf::Op::For { body, .. })
            | DfirOp::Dataflow(dataflow::Op::ProgramUnit { body, .. }) => {
                walk_preorder(body, visit);
            }
            DfirOp::Scf(scf::Op::If {
                body, else_body, ..
            })
            | DfirOp::Affine(affine::Op::If {
                body, else_body, ..
            }) => {
                walk_preorder(body, visit);
                walk_preorder(else_body, visit);
            }
            DfirOp::Agen(agen::Op::CompositeLoadAndStore(transfer)) => {
                walk_preorder(&transfer.body, visit);
            }
            DfirOp::Agen(agen::Op::CompositeIndirectLoadAndStore(transfer)) => {
                walk_preorder(&transfer.body, visit);
            }
            // ⭐ A `uniform.uniformize_regions` HOLDS ONE REGION PER UNIT CLASS, and `Operation::walk`
            // descends into every one of them. Before `FlatteningLocalRegions` has run those regions
            // ARE the program body, so a walk that stopped at the op would see no toggle at all.
            DfirOp::Uniform(uniform::Op::UniformizeRegions { regions, .. }) => {
                for region in regions {
                    walk_preorder(&region.body, visit);
                }
            }
            DfirOp::Affine(
                affine::Op::Apply { .. }
                | affine::Op::Yield { .. }
                | affine::Op::VectorLoad { .. }
                | affine::Op::VectorStore { .. },
            )
            | DfirOp::Scf(scf::Op::Yield { .. })
            | DfirOp::Arith(_)
            | DfirOp::Dataflow(_)
            | DfirOp::Agen(_)
            | DfirOp::VectorChain(_)
            | DfirOp::Vector(_)
            // `uniform.yield` terminates one of those regions; the two mapping ops carry none.
            | DfirOp::Uniform(
                uniform::Op::Yield { .. }
                | uniform::Op::DefImmutableMapping { .. }
                | uniform::Op::QueryMap { .. },
            )
            | DfirOp::Symbol(_) => {}
        }
    }
}

/// WHICH UNITS THIS PASS LOOKS AT — the L3 loader and the L3 store unit, and nothing else.
///
/// ```cpp
/// ModuleOp module_op = getOperation();
/// std::vector<dataflow::ProgramUnitOp> candidate_units;
/// module_op.walk([&](dataflow::ProgramUnitOp unit_op) {
///   auto comp = dcc::getUnitType(
///       unit_op.getUnits()[0].getDefiningOp<dataflow::GetUnitOp>());
///   if (is_any_of(comp, SenComponents::L3LU, SenComponents::L3SU))
///     candidate_units.push_back(unit_op);
/// });
/// ```
/// (`dcc/src/Transform/Dataflow/CanonicalizeToggle.cpp:56-63`)
///
/// ⭐⭐ THE SAME PREDICATE THE COMPOSITE-TRANSFER GATE USES, so it is asked in one place:
/// [`dfir::Units::moves_memory`] is `is_any_of(comp, L3LU, L3SU)` at `Helper.cpp:2177-2179`, and this
/// is the same two components in the same order. A toggle is a *data transfer's* address, so the only
/// units holding one are the units that move memory.
///
/// ⚠️ `getUnits()[0].getDefiningOp<GetUnitOp>()` IS ASKED OF THE TYPE HERE. The island's
/// [`dfir::Units`] carries the kind as a field precisely because `Helper.cpp:2173-2176` reads it from
/// the first unit alone; there is no `get_unit` to resolve and no `[0]` to index.
fn candidate_units<A: Arch>(program: &dfir::Program<A>) -> Vec<&ProgramUnit<A>> {
    program
        .units
        .iter()
        .filter(|unit| unit.on.moves_memory())
        .collect()
}

/// A TOGGLE A SECOND DATA TRANSFER READS TOO — what the pattern finds at one memory view.
///
/// ⛔⛔ NOT ENTRY 350'S PORT. `DuplicateReusedTogglePattern::matchAndRewrite`
/// (`dcc/src/Transform/Dataflow/DuplicateReusedToggle.cpp:33`, entry 350/384, level 6) is 170 lines
/// that clone a loop nest and add an `iter_args` chain per reuse. What is here is only its MATCH
/// condition (`:36-57`), and it is here because THE DRIVER CANNOT BE WRITTEN WITHOUT IT:
/// `CanonicalizeToggle.cpp:82` reads `succeeded(...)` to decide whether the IR changed, and that
/// answer is what ends the convergence loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReusedToggle {
    /// The `arith.subi` binding the view's start address — `toggle_op` (`:36-37`).
    pub toggle: Val,
    /// How many uses of it are neither this view nor the owning loop's yield — `candidates.size()`
    /// (`:46`, `:55`, `:59`), which is how many copies the rewrite would make.
    pub reuses: usize,
}

/// WHETHER THE PATTERN WOULD FIRE ON THIS MEMORY VIEW — `matchAndRewrite`'s two `return failure()`s.
///
/// ```cpp
/// auto toggle_op = dyn_cast_or_null<arith::SubIOp>(
///     mem_view_op.getStartAddress().getDefiningOp());
/// if (!toggle_op) return failure();
///
/// auto iter_arg = dyn_cast<BlockArgument>(toggle_op->getOperand(1));
/// DT_CHECK_MSG(iter_arg,
///              "expecting second operand of toggle op to be an iter_arg");
///
/// // Any use of the toggle op that isn't either a yield operation or the memory
/// // view needs to be duplicated.
/// SmallVector<Operation *> candidates;
/// auto toggle_parent = iter_arg.getOwner()->getParentOp();
/// DT_CHECK(toggle_parent);
/// for (auto user : toggle_op->getUsers()) {
///   if (isa<affine::AffineYieldOp, scf::YieldOp>(user))
///     DT_CHECK_MSG(user->getParentOp() == toggle_parent, ..);
///   else if (user != mem_view_op)
///     candidates.push_back(user);
/// }
/// if (candidates.empty()) return failure();
/// ```
/// (`dcc/src/Transform/Dataflow/DuplicateReusedToggle.cpp:36-57`)
///
/// # ⛔ A YIELD IS NOT A REUSE, AND NEITHER IS THIS VIEW
///
/// The toggle is `next = base - carried`, carried across the loop by an `iter_args` chain
/// (`tf_mutable_addr_splitting.rs`'s [`super::tf_mutable_addr_splitting`] documents the same shape).
/// So its value is ALWAYS read at least twice — once by the view that addresses the transfer, once by
/// the yield that carries it to the next iteration — and those two are the canonical pattern
/// `AddressPinningAndToggle` produces. A third reader is a second transfer sharing one toggle, which
/// is what this pass exists to undo (`CanonicalizeToggle.cpp:8-9`).
///
/// ⚠️ THE TWO `DT_CHECK`s ARE STATEMENTS ABOUT WELL-FORMED INPUT, NOT BRANCHES. `:41-42` says the
/// toggle's second operand is a loop-carried argument, and `:51-53` that a yield reading the toggle
/// belongs to the loop that carries it; both abort the compiler when false, and this crate does not
/// refuse at runtime (`crates/compiler/deeptools/CLAUDE.md`). Neither changes which views match: the
/// island expresses the first as `defining_op(rhs, ..) == None`, since a value no op binds is a region
/// argument ([`defining_op`]).
///
/// ⚠️ `user != mem_view_op` IS OP IDENTITY, and here that is pointer identity: `view` is a reference
/// into `scope`, which is the same tree [`uses`] walks.
#[must_use]
pub fn reused_toggle(view: &DfirOp, scope: &[DfirOp]) -> Option<ReusedToggle> {
    // `dyn_cast_or_null<GetLogicalMemoryViewOp>` is the caller's `:77`; this function is asked only
    // about a view, so what is read here is `mem_view_op.getStartAddress()`.
    let DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView { start, .. }) = view else {
        return None;
    };

    // `dyn_cast_or_null<arith::SubIOp>(..getDefiningOp()); if (!toggle_op) return failure();`
    let Some(DfirOp::Arith(arith::Op::SubI(toggle))) = defining_op(*start, scope) else {
        return None;
    };

    // `for (auto user : toggle_op->getUsers())` — one entry per use, as MLIR counts them.
    let reuses = uses(toggle.result, scope)
        .into_iter()
        .filter(|user| {
            // `if (isa<affine::AffineYieldOp, scf::YieldOp>(user)) .. else if (user != mem_view_op)`
            !matches!(
                user,
                DfirOp::Affine(affine::Op::Yield { .. }) | DfirOp::Scf(scf::Op::Yield { .. })
            ) && !core::ptr::eq(*user, view)
        })
        .count();

    // `if (candidates.empty()) return failure();`
    (reuses > 0).then_some(ReusedToggle {
        toggle: toggle.result,
        reuses,
    })
}

/// GIVE EVERY DATA TRANSFER ITS OWN TOGGLE.
///
/// ```cpp
/// void runOnOperation() {
///   if (DisableThisPass) return;
///
///   // Collect the list of candidate units to analyze. Execution of the
///   // canonicalizations will lead to operation deletion so we collect the
///   // candidate units first. Only operations in the program unit operation
///   // regions will change.
///   ModuleOp module_op = getOperation();
///   std::vector<dataflow::ProgramUnitOp> candidate_units;
///   module_op.walk([&](dataflow::ProgramUnitOp unit_op) { .. });
///
///   // For each candidate unit, execute the pattern until we hit IR convergence.
///   // The IR needs to be reanalyzed after every applied transformation as the
///   // pattern application deletes operations.
///   // TODO: Using the RewritePattern drivers leads to running all basic
///   // canonicalizations (such as DCE) and will have unintended side effects if
///   // run during the DFIR pipeline. There may be a better way than manually
///   // calling the pattern on every candidate operation.
///   for (auto unit : candidate_units) {
///     bool has_changed = true;
///     while (has_changed) {
///       has_changed = false;
///       unit.walk([&](Operation *op) {
///         if (auto candidate = dyn_cast<dataflow::GetLogicalMemoryViewOp>(op)) {
///           PatternRewriter rewriter(&getContext());
///           rewriter.setInsertionPoint(op);
///
///           dcc::dataflow::DuplicateReusedTogglePattern pattern(&getContext());
///           if (succeeded(pattern.matchAndRewrite(candidate, rewriter))) {
///             has_changed = true;
///             return WalkResult::interrupt();
///           }
///         }
///         return WalkResult::advance();
///       });
///     }
///   }
/// }
/// ```
/// (`dcc/src/Transform/Dataflow/CanonicalizeToggle.cpp:49-91`; the collection walk is quoted on
/// [`candidate_units`])
///
/// # ⭐⭐ WHY THE PASS EXISTS: A LATER PASS PATTERN-MATCHES THE TOGGLE
///
/// *"This pass canonicalizes toggle operations. For example, toggles should only be used in one data
/// transfer. Some passes in the pipeline, such as AddressPinningAndToggle, require toggles to match
/// certain patterns."* (`:8-12`). So a shared toggle is not an inefficiency — it is a shape the
/// downstream pass does not handle, and `MutableAddrSplitting.cpp:832-839` records the same
/// requirement from the other side (see [`super::tf_mutable_addr_splitting`], whose
/// `a_toggled_start_address_is_not_eligible` test is that refusal).
///
/// # ⛔⛔ THE CONVERGENCE LOOP HAS ONE REACHABLE ITERATION UNTIL ENTRY 350 LANDS
///
/// `while (has_changed)` restarts the walk after every rewrite, because `matchAndRewrite` deletes and
/// clones ops and the walk it interrupted is invalid afterwards (`:65-67`, `:84`). The rewrite is
/// entry 350/384 at level 6 and is unported, so the first match a candidate unit produces has nothing
/// to apply: the port stops there, naming the unit and the toggle, rather than looping forever on a
/// rewrite that cannot happen or — worse — reporting convergence over a program it never changed.
///
/// ⭐ AND A PROGRAM WITH NO SHARED TOGGLE IS LEFT ALONE, WHICH IS THE PASS'S ANSWER FOR EVERY PROGRAM
/// THIS CRATE EMITS TODAY. The toggled-address shape comes from `AddressPinningAndToggle`, one rung
/// below; nothing here builds an `arith.subi` over an `iter_args` chain, so [`reused_toggle`] answers
/// `None` for every view and the pass is a checked no-op. ⛔ WHICH IS EXACTLY WHY IT IS PORTED NOW: on
/// the day a toggle IS shared, the answer must be a build failure naming entry 350, not a silently
/// unhandled shape that `AddressPinningAndToggle` then mis-pins.
///
/// # ⛔ WHAT THE PORT DROPS, AND WHY
///
/// - `DisableThisPass` (`:35-38`, read at `:50`) is an `llvm::cl::opt<bool>` — a command-line flag of
///   `dcc-opt`, not a program property. This crate has no pass pipeline and no flags: which pass runs
///   is a call in [`super::program`], so the switch has nowhere to live. As a parameter it would make
///   "did the compiler canonicalize" a runtime question with two answers.
/// - `PatternRewriter rewriter(&getContext()); rewriter.setInsertionPoint(op);` (`:78-79`) and the
///   per-op construction of the pattern (`:81`) are an insertion point and an allocation — mechanism.
///   The insertion point is `op` itself, which is where a ported entry 350 clones from anyway.
/// - The reference's own `TODO` at `:68-71` (that a `RewritePattern` driver would run every basic
///   canonicalization, DCE included, with unintended effects mid-pipeline) is why the walk is written
///   by hand. Here there is no driver to avoid; the note is recorded because it says the manual walk
///   is deliberate.
///
/// # ⛔ AND IT TAKES A SHARED REFERENCE
///
/// The reference rewrites its module in place. The whole of that rewriting is entry 350, so there is
/// nothing here to write back: the observable is the choice between leaving a program alone and
/// failing the build. `&mut` would claim a capability nothing behind it has and would make every
/// caller clone a program to rewrite none of them. It becomes `&mut` in the changeset that lands
/// entry 350 — the same contract [`super::tf_cfg_simplification_dataflow_level::run_on_operation`]
/// states.
///
/// Replaces: e178_runOnOperation
pub fn run_on_operation<A: Arch>(program: &dfir::Program<A>) {
    // `for (auto unit : candidate_units)` — collected first, because the rewrite deletes ops
    // (`:52-55`).
    for unit in candidate_units(program) {
        // `bool has_changed = true; while (has_changed) { has_changed = false; unit.walk(..) }`
        let mut first_match: Option<ReusedToggle> = None;
        walk_preorder(&unit.body, &mut |op| {
            // `if (auto candidate = dyn_cast<dataflow::GetLogicalMemoryViewOp>(op))` — every other
            // op is `WalkResult::advance()` (`:87`).
            if first_match.is_none() {
                first_match = reused_toggle(op, &unit.body);
            }
        });

        // `if (succeeded(pattern.matchAndRewrite(candidate, rewriter))) { has_changed = true; ..`
        if let Some(found) = first_match {
            todo!(
                "e350_matchAndRewrite: {:?} of {:?} shares its toggle {:?} with {} other \
                 use(s) of it; duplicating it needs one new iter_arg chain per use",
                unit.on.kind(),
                unit.on.vals(),
                found.toggle,
                found.reuses
            );
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{ReusedToggle, candidate_units, reused_toggle, run_on_operation};
    use crate::arch::Target;
    use crate::generated::OpFunc;
    use crate::islands::dataflow_ir::dialects::{Op as DfirOp, Val, affine, arith, dataflow};
    use crate::islands::dataflow_ir::ty::{AffineMap, ElemType, MemRef, ScalarTy};
    use crate::islands::dataflow_ir::{
        Grid, GroupId, OpIndex, Program, ProgramName, ProgramUnit, ProgramUnits, Units,
    };
    use crate::units::DfirUnit;

    /// One unit of `kind` holding `body`.
    fn unit_of(kind: DfirUnit, body: Vec<DfirOp>) -> ProgramUnit<Target> {
        ProgramUnit {
            on: Units::one(kind, Val(0)),
            precision: None,
            body,
            arch: core::marker::PhantomData,
        }
    }

    /// A program of `units`.
    fn program_of(units: Vec<ProgramUnit<Target>>) -> Program<Target> {
        let mut units = units.into_iter();
        let head = units.next().expect("a test program names its units");
        Program {
            name: ProgramName {
                group: GroupId(0),
                index: OpIndex(0),
                func: OpFunc::Add,
            },
            grid: Grid::single(),
            preamble: Vec::new(),
            units: ProgramUnits::of(head, units.collect()),
            arch: core::marker::PhantomData,
        }
    }

    /// `%v = dataflow.get_logical_memory_view %0, %start`.
    fn view(result: u32, start: u32) -> DfirOp {
        DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
            result: Val(result),
            from: Val(0),
            start: Val(start),
            layout: AffineMap::linear(&[128, 1]),
            ty: MemRef {
                shape: vec![4, 128],
                elem: ElemType::F16,
            },
        })
    }

    /// `%t = arith.subi %base, %carried` — the toggle, whose right operand is the loop-carried
    /// argument (`DuplicateReusedToggle.cpp:40`).
    fn toggle(result: u32, base: u32, carried: u32) -> DfirOp {
        DfirOp::Arith(arith::Op::SubI(arith::IntBinary {
            result: Val(result),
            lhs: Val(base),
            rhs: Val(carried),
            ty: ScalarTy::Index,
        }))
    }

    /// AN `affine.for` CARRYING `%carried`, WITH `body` INSIDE IT.
    ///
    /// The toggle's second operand is that carried argument, which is why the loop is here at all:
    /// `defining_op` finds no binder for a region argument.
    fn carrying_loop(carried: u32, result: u32, init: u32, body: Vec<DfirOp>) -> DfirOp {
        DfirOp::Affine(affine::Op::For {
            iv: Val(100),
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(28),
            carried: vec![affine::Carried {
                init: Val(init),
                arg: Val(carried),
                result: Val(result),
            }],
            body,
            dbg_name: None,
        })
    }

    /// 🎯 178/384 — ONLY THE L3 LOADER AND STORE UNIT ARE CANDIDATES.
    ///
    /// `if (is_any_of(comp, SenComponents::L3LU, SenComponents::L3SU))`
    /// (`CanonicalizeToggle.cpp:61`). A toggle is a data transfer's address, and the transfer lives on
    /// the L3 halves — a pass that looked at every unit would ask about views on a PE.
    #[test]
    fn only_the_l3_halves_are_candidate_units() {
        let program = program_of(vec![
            unit_of(DfirUnit::Lxlu, Vec::new()),
            unit_of(DfirUnit::L3lu, Vec::new()),
            unit_of(DfirUnit::Pe, Vec::new()),
            unit_of(DfirUnit::L3su, Vec::new()),
        ]);

        assert_eq!(
            candidate_units(&program)
                .iter()
                .map(|unit| unit.on.kind())
                .collect::<Vec<_>>(),
            [DfirUnit::L3lu, DfirUnit::L3su],
            "and in module order, because the rewrite is applied unit by unit"
        );
    }

    /// 🎯 178/384 — A VIEW WHOSE START ADDRESS IS NOT AN `arith.subi` IS NOT A MATCH.
    ///
    /// `auto toggle_op = dyn_cast_or_null<arith::SubIOp>(..); if (!toggle_op) return failure();`
    /// (`DuplicateReusedToggle.cpp:36-38`) — the constant-start view every transfer this crate emits
    /// uses today, and a start address bound by no op at all (a region argument), for which
    /// `getDefiningOp()` is null and the `dyn_cast_or_null` is too.
    #[test]
    fn a_view_whose_start_is_not_a_toggle_is_not_a_match() {
        let constant_start = vec![
            DfirOp::Arith(arith::Op::Constant {
                result: Val(1),
                value: 512,
            }),
            view(2, 1),
        ];
        assert_eq!(reused_toggle(&constant_start[1], &constant_start), None);

        let region_argument_start = vec![view(2, 100)];
        assert_eq!(
            reused_toggle(&region_argument_start[0], &region_argument_start),
            None
        );

        // ⭐ AND AN OP THAT IS NOT A VIEW IS NOT ASKED ABOUT: `:77`'s `dyn_cast` is what selects the
        // candidate, so every other op in the walk is `WalkResult::advance()`.
        assert_eq!(reused_toggle(&constant_start[0], &constant_start), None);
    }

    /// 🎯 178/384 — A TOGGLE READ ONLY BY ITS OWN VIEW AND ITS OWN YIELD IS THE CANONICAL SHAPE.
    ///
    /// `if (isa<affine::AffineYieldOp, scf::YieldOp>(user)) .. else if (user != mem_view_op)
    /// candidates.push_back(user); if (candidates.empty()) return failure();`
    /// (`DuplicateReusedToggle.cpp:50-57`). Both readers are excluded, so the pattern declines —
    /// which is what makes this pass a no-op on already-canonical IR rather than an infinite loop
    /// duplicating a toggle that is not shared.
    #[test]
    fn a_toggle_read_only_by_its_view_and_its_yield_is_not_a_match() {
        let body = vec![toggle(11, 10, 101), view(12, 11), yield_of(&[11])];
        let program_body = vec![carrying_loop(101, 102, 10, body)];
        let view_op = &loop_body(&program_body[0])[1];

        assert_eq!(reused_toggle(view_op, &program_body), None);
    }

    /// 🎯 178/384 — A SECOND TRANSFER READING THE SAME TOGGLE IS A MATCH, ONE COPY PER READ.
    ///
    /// `candidates` holds every user that is neither a yield nor this view, and its SIZE is how many
    /// duplicates the rewrite makes (`DuplicateReusedToggle.cpp:46`, `:55`, `:59`). Two views sharing
    /// one toggle is exactly the shape `AddressPinningAndToggle` cannot pin
    /// (`CanonicalizeToggle.cpp:8-12`).
    #[test]
    fn a_toggle_a_second_view_reads_is_a_match() {
        let body = vec![
            toggle(11, 10, 101),
            view(12, 11),
            view(13, 11),
            yield_of(&[11]),
        ];
        let program_body = vec![carrying_loop(101, 102, 10, body)];
        let views = loop_body(&program_body[0]);

        assert_eq!(
            reused_toggle(&views[1], &program_body),
            Some(ReusedToggle {
                toggle: Val(11),
                reuses: 1,
            }),
            "the OTHER view is the candidate; this one and the yield are not"
        );
        assert_eq!(
            reused_toggle(&views[2], &program_body),
            Some(ReusedToggle {
                toggle: Val(11),
                reuses: 1,
            }),
            "and the pattern is asked at each view in turn, so the other one matches too"
        );
    }

    /// 🎯 178/384 — A PROGRAM WHOSE TOGGLES ARE NOT SHARED IS LEFT ALONE, AND THAT INCLUDES EVERY
    /// PROGRAM THIS CRATE EMITS TODAY.
    ///
    /// REACHING THE END OF THE CALL IS THE ASSERTION: were any candidate unit to hold a shared
    /// toggle, [`run_on_operation`]'s `todo!` would fire and name entry 350. ⛔ IT DOES NOT COMPARE
    /// THE PROGRAM WITH A CLONE OF ITSELF — the parameter is shared, so "unchanged" is a fact about
    /// the type and re-asserting it would test `Clone`.
    ///
    /// ⭐ THE SHARED TOGGLE IS PRESENT, ON A UNIT THAT IS NOT A CANDIDATE. `:61` filters the units
    /// BEFORE any view is looked at, so a reused toggle on a `pe` unit is not this pass's business —
    /// and a port that walked every unit would stop this build.
    #[test]
    fn a_program_with_no_shared_toggle_is_left_alone() {
        let canonical = vec![carrying_loop(
            101,
            102,
            10,
            vec![toggle(11, 10, 101), view(12, 11), yield_of(&[11])],
        )];
        let shared = vec![carrying_loop(
            101,
            102,
            10,
            vec![
                toggle(11, 10, 101),
                view(12, 11),
                view(13, 11),
                yield_of(&[11]),
            ],
        )];

        run_on_operation(&program_of(vec![
            unit_of(DfirUnit::L3lu, canonical.clone()),
            unit_of(DfirUnit::L3su, canonical),
            unit_of(DfirUnit::Pe, shared),
        ]));
    }

    /// `affine.yield %operands` — the toggle's other reader.
    fn yield_of(operands: &[u32]) -> DfirOp {
        DfirOp::Affine(affine::Op::Yield {
            operands: operands.iter().copied().map(Val).collect(),
        })
    }

    /// The statements of the test's carrying loop.
    fn loop_body(op: &DfirOp) -> &[DfirOp] {
        let DfirOp::Affine(affine::Op::For { body, .. }) = op else {
            unreachable!("built by `carrying_loop`")
        };
        body
    }
}
