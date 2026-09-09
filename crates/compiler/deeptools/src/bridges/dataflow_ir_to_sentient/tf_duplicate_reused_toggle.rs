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

//! `DuplicateReusedToggle.cpp` — 1 of bridge 2's 384 functions (dependency level(s) [6]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e350_matchAndRewrite` | 350/384 | 170 | `dcc/src/Transform/Dataflow/DuplicateReusedToggle.cpp:33` |

use super::tf_loop_unroll_for_shuffle_op::is_arith_constant;
use super::tf_unit_filtering::op_at_mut;
use super::tf_utils::CountedLoop;
use super::vc_vector_chain_to_sentient_pesfp::walk_positions;
use super::vc_vector_operands::{OpId, op_at};
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::dialects::{
    Op as DfirOp, Val, affine, arith, dataflow, operands, operands_mut, regions_mut, results, scf,
};

/// WHAT ONE ATTEMPT TO GIVE EVERY READER OF A TOGGLE ITS OWN COPY ANSWERS.
///
/// ⛔ THE SIX REFUSALS ARE `DT_CHECK`s, WHICH ABORT THE COMPILER (`util/dt_exception.hpp:110-118`,
/// no `NDEBUG` opt-out). A variant that stops the pass is the faithful reading; this crate does not
/// refuse at runtime (`crates/compiler/deeptools/CLAUDE.md`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToggleDuplication {
    /// `if (!toggle_op) return failure();` (`:38`) — the view's start address is no `arith.subi`.
    NotAToggle,
    /// `if (candidates.empty()) return failure();` (`:57`) — the only readers are this view and the
    /// yield that carries the toggle, which is the canonical shape and needs no duplication.
    NoOtherUsers,
    /// `DT_CHECK_MSG(iter_arg, "expecting second operand of toggle op to be an iter_arg")` (`:40-42`).
    SubtrahendIsNotAnIterArg,
    /// `DT_CHECK_MSG(user->getParentOp() == toggle_parent, "[DuplicateReusedToggle] Toggle user is a
    /// yield for an operation other than owning loop")` (`:51-54`), carrying the offending yield —
    /// `dcc/test/Transform/CanonicalizeToggle/invalid_toggle_use.mlir` is exactly this case, a toggle
    /// yielded out of an `scf.if`.
    YieldOutsideOwningLoop(OpId),
    /// `DT_CHECK(dcc::utils::isConstant<arith::ConstantOp>(init_val))` (`:68`) — the chain's outermost
    /// initialiser, which every duplicate's own chain starts from.
    ChainInitIsNotConstant(Val),
    /// `DT_CHECK(num_orig_args > 0)` (`:101`) with its two siblings
    /// `DT_CHECK(isa<affine::AffineYieldOp>(yield_op))` (`:96`, `:99`): a loop of the chain that
    /// carries nothing, or whose body does not end in a yield, cannot be widened.
    LoopCannotGrow(OpId),
    /// `DT_CHECK(res.getUsers().empty() || isa<affine::AffineYieldOp, scf::ForOp>(*(res.getUsers()
    /// .begin())))` (`:118-121`), carrying the user that is none of those.
    ///
    /// ⛔⛔ AND THE REFERENCE'S `isa<>` LIST OMITS `scf::YieldOp`, WHICH ITS OWN GOLDEN NEEDS. In
    /// `multiple_toggle_uses.mlir` — a SUCCESS case — the mid `affine.for`'s results are read by the
    /// enclosing `scf.for`'s `scf.yield` (`:118` of that file), so the check as written aborts on the
    /// program it is documented to rewrite. This port admits the three the comment *"These should only
    /// be yield operations"* and the golden agree on: `affine.yield`, `scf.yield`, `scf.for`.
    InnerResultUserUnknown(OpId),
    /// `DT_CHECK(loop->getResult(j).getUsers().empty())` (`:147-148`) — *"The outermost loop is
    /// expected to have no uses of the results. If there were uses, we would have to be able to
    /// determine how they were used to figure out how to use the new additional results properly."*
    OutermostResultsUsed(OpId),
    /// `return success();` (`:204`), carrying the duplicate each candidate now reads, in the order
    /// `candidates` holds them.
    Duplicated(Vec<Val>),
}

/// THE ITER ARGS ONE LOOP OF THE CHAIN GREW — `upd_iter_args` and `num_orig_args` (`:94-101`).
#[derive(Debug, Clone, PartialEq, Eq)]
struct AddedIterArgs {
    /// `num_orig_args = upd_iter_args.size() - candidates.size()` — where the added ones start, and
    /// the same index in the body's yield.
    num_orig: usize,
    /// `upd_iter_args[num_orig_args..]` — one region argument per candidate.
    args: Vec<Val>,
    /// `new_loop->getResult(num_orig_args..)` — one result per candidate.
    results: Vec<Val>,
}

/// Replaces: e350_matchAndRewrite
///
/// **350/384** `DuplicateReusedTogglePattern::matchAndRewrite` —
/// `dcc/src/Transform/Dataflow/DuplicateReusedToggle.cpp:33` (170L).
///
/// One `arith.subi` toggle read by N transfers becomes N+1 toggles, each carried by its own iter_arg
/// through every loop of the chain from the toggle to the constant that initialises it — so the
/// pattern `AddressPinningAndToggle` requires (one toggle per transfer) holds again.
///
/// ⛔ THE CLONE-AND-ERASE MECHANISM IS DROPPED, NOT THE EMISSION. The reference builds each widened
/// loop with entry 264 (`:83-84`), remaps `toggle_op`, the chain and the candidates through the
/// `IRMapping`, then erases the original (`:167-179`); the island has no parent pointers, so the loops
/// are widened IN PLACE and every remap — `rewriter.replaceAllUsesWith(res, new_loop->getResult(j))`
/// included (`:122`) — is the identity. The fill constants entry 264 would emit are dropped with it:
/// `setIterArgInit` overwrites all of them (`:127-133`, `:153-155`), so none survives the reference
/// either.
#[must_use]
pub fn match_and_rewrite(
    mem_view: &OpId,
    body: &mut Vec<DfirOp>,
    vals: &mut Values,
) -> ToggleDuplication {
    // `mem_view_op.getStartAddress()` (`:37`) — the pattern is rooted on a memory view.
    let Some(DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView { start, .. })) =
        op_at(mem_view, body)
    else {
        return ToggleDuplication::NotAToggle;
    };
    let start = *start;

    // `dyn_cast_or_null<arith::SubIOp>(..getDefiningOp()); if (!toggle_op) return failure();` (`:36-38`)
    let Some(toggle_at) = definition_of(start, body) else {
        return ToggleDuplication::NotAToggle;
    };
    let Some(DfirOp::Arith(arith::Op::SubI(toggle))) = op_at(&toggle_at, body) else {
        return ToggleDuplication::NotAToggle;
    };
    let toggle = *toggle;

    // `auto iter_arg = dyn_cast<BlockArgument>(toggle_op->getOperand(1));` and the `DT_CHECK_MSG` on
    // it (`:40-42`), then `auto toggle_parent = iter_arg.getOwner()->getParentOp();` (`:47`) — one
    // question here, since the island reaches the argument through the loop that binds it.
    let Some((owner, owner_index)) = iter_arg_owner(toggle.rhs, &toggle_at, body) else {
        return ToggleDuplication::SubtrahendIsNotAnIterArg;
    };

    // `for (auto user : toggle_op->getUsers())` (`:49-56`) — *"Any use of the toggle op that isn't
    // either a yield operation or the memory view needs to be duplicated."*
    let mut candidates: Vec<OpId> = Vec::new();
    for at in users_of(toggle.result, body) {
        let Some(user) = op_at(&at, body) else {
            continue;
        };
        if matches!(
            user,
            DfirOp::Affine(affine::Op::Yield { .. }) | DfirOp::Scf(scf::Op::Yield { .. })
        ) {
            // `DT_CHECK_MSG(user->getParentOp() == toggle_parent, ..)` (`:51-54`) — a yield's parent
            // op is its block, which is its position without the last ordinal.
            if at.block() != owner.path() {
                return ToggleDuplication::YieldOutsideOwningLoop(at);
            }
        } else if &at != mem_view {
            candidates.push(at);
        }
    }
    // `if (candidates.empty()) return failure();` (`:57`)
    if candidates.is_empty() {
        return ToggleDuplication::NoOtherUsers;
    }

    // `getIterArgChain(iter_arg, init_val)` (`:65-66`, `dialect_utils/Dataflow/Utils.cpp:147-165`) —
    // *"Collect the iter_arg chain from the iter_arg used in the toggle to its initializer. The args
    // will be in innermost to outermost order."* `DT_CHECK(!iter_arg_chain.empty())` (`:67`) is the
    // lookup above having answered.
    let mut chain: Vec<OpId> = Vec::new();
    let mut init_val = toggle.rhs;
    let mut step = Some((owner, owner_index));
    while let Some((loop_at, index)) = step {
        let Some(entry) = carried_entry(&loop_at, index, body) else {
            break;
        };
        init_val = entry.init;
        step = iter_arg_owner(entry.init, &loop_at, body);
        chain.push(loop_at);
    }
    // `DT_CHECK(dcc::utils::isConstant<arith::ConstantOp>(init_val))` (`:68`) — entry 128 over the
    // one scope this island has for it.
    if !is_arith_constant(init_val, &[], body) {
        return ToggleDuplication::ChainInitIsNotConstant(init_val);
    }

    // `for (int b = iter_arg_chain.size() - 1, i = b; i >= 0; --i)` (`:80`) — *"Iterate through the
    // args in reverse to execute from outermost to innermost loop."*
    let mut prev: Option<(OpId, AddedIterArgs)> = None;
    for loop_at in chain.iter().rev() {
        match prev.take() {
            // `if (i != b) { // INNER LOOPS` (`:111-141`) — every iteration but the first.
            Some((parent_at, parent)) => {
                // `DT_CHECK(res.getUsers().empty() || isa<..>(*(res.getUsers().begin())))` (`:117-122`)
                // — *"Replace uses of original loop's results with the new ones. These should only be
                // yield operations."* The replacement itself is the identity here: widening in place
                // keeps every original result's value.
                if let Some(bad) = unexpected_result_user(loop_at, body) {
                    return ToggleDuplication::InnerResultUserUnknown(bad);
                }
                // *"The new iter_args for this loop should be initialized to the new iter_args added
                // to the previous created loop to continue the chain."* (`:124-133`) — `j` from
                // `num_orig_args`, `k` from `prev_upd_iter_args.size() - candidates.size()`, which is
                // the parent's own `num_orig_args`.
                let Some(grown) = add_iter_args(loop_at, &parent.args, vals, body) else {
                    return ToggleDuplication::LoopCannotGrow(loop_at.clone());
                };
                // *"The yield op of the previous created parent loop needs to yield the results
                // associated to the new iter_args of the current loop."* (`:135-141`), whose
                // `operand_idx = prev_yield_op->getNumOperands() - candidates.size()` is again the
                // parent's `num_orig_args` — and `DT_CHECK(operand_idx > 0)` (`:137`) is the parent's
                // own [`ToggleDuplication::LoopCannotGrow`], already answered.
                for (offset, result) in grown.results.iter().enumerate() {
                    set_yield_operand(&parent_at, parent.num_orig + offset, *result, body);
                }
                prev = Some((loop_at.clone(), grown));
            }
            // `} else { // OUTERMOST LOOPS` (`:142-156`), which `i == b` selects on the first pass.
            None => {
                // `for (unsigned j = 0, e = loop->getNumResults(); j < e; ++j)
                //    DT_CHECK(loop->getResult(j).getUsers().empty());` (`:147-148`)
                if let Some(used) = any_result_user(loop_at, body) {
                    return ToggleDuplication::OutermostResultsUsed(used);
                }
                // *"The new iter_args in the new outer loop should be initialized to the same value as
                // the original toggle."* (`:150-155`)
                let inits = vec![init_val; candidates.len()];
                let Some(grown) = add_iter_args(loop_at, &inits, vals, body) else {
                    return ToggleDuplication::LoopCannotGrow(loop_at.clone());
                };
                prev = Some((loop_at.clone(), grown));
            }
        }
    }
    // `prev_upd_iter_args` and `prev_yield_op` after the loop are the INNERMOST loop's, which is what
    // `:193` and `:201` read.
    let Some((innermost, grown)) = prev else {
        return ToggleDuplication::LoopCannotGrow(toggle_at);
    };

    // `OpBuilder builder(toggle_op); builder.setInsertionPointAfter(toggle_op);` then the loop at
    // `:186-202` — *"Now that the new loop structures are in place, duplicate the toggle for each
    // candidate."*
    let mut clones: Vec<DfirOp> = Vec::new();
    let mut duplicated: Vec<Val> = Vec::new();
    for (i, candidate) in candidates.iter().enumerate() {
        // `auto new_toggle_op = builder.clone(*toggle_op);` (`:188`) with
        // `new_toggle_op->setOperand(1, upd_iter_args[num_orig_args + i]);` (`:192-193`) folded into
        // the clone: the subtrahend is the only operand that changes.
        let result = vals.mint();
        let Some(arg) = grown.args.get(i) else {
            break;
        };
        clones.push(DfirOp::Arith(arith::Op::SubI(arith::IntBinary {
            result,
            lhs: toggle.lhs,
            rhs: *arg,
            ty: toggle.ty,
        })));
        // `candidates[i]->replaceUsesOfWith(toggle_op.getResult(), new_toggle_op->getResult(0));`
        // (`:195-197`)
        replace_operands(candidate, toggle.result, result, body);
        // `prev_yield_op->setOperand(num_orig_args + i, new_toggle_op->getResult(0));` (`:201`) —
        // *"Update the yield operation to yield the new toggle instead of the iter_arg used by the
        // toggle."*
        set_yield_operand(&innermost, grown.num_orig + i, result, body);
        duplicated.push(result);
    }
    // ⛔ THE SPLICE IS LAST, because every position above was measured on the body as it stood: the
    // clones go where `setInsertionPointAfter(toggle_op)` puts them, after which the ordinals of
    // everything below the toggle have moved.
    insert_after(&toggle_at, clones, body);

    ToggleDuplication::Duplicated(duplicated)
}

/// THE OP BINDING A VALUE, AS A POSITION — `Value::getDefiningOp()` where the answer has to be
/// reachable for mutation, so it is a path rather than a reference.
fn definition_of(val: Val, scope: &[DfirOp]) -> Option<OpId> {
    walk_positions(scope, &[], 0, &mut |op, at| {
        results(op).contains(&val).then_some(at)
    })
}

/// EVERY USE OF A VALUE, AS POSITIONS, NEWEST FIRST — `Value::getUsers()`.
///
/// ⛔⛔ NEWEST FIRST IS NOT COSMETIC, IT DECIDES WHICH DUPLICATE EACH READER GETS. MLIR prepends to a
/// value's use list, so `getUsers()` walks the readers in reverse creation order — and
/// `multiple_toggle_uses.mlir` proves it: the FIRST clone (`%82`) goes to the LAST-written reader (the
/// `arith.addi` at `:104` of that file) and the last clone (`%84`) to the first-written view (`:87`).
/// A preorder walk would assign them the other way round.
///
/// ⭐ ONE ENTRY PER USE, NOT PER USER, which is what `getUsers()` yields — see
/// [`crate::islands::dataflow_ir::dialects::uses`].
fn users_of(val: Val, scope: &[DfirOp]) -> Vec<OpId> {
    let mut found: Vec<OpId> = Vec::new();
    walk_positions(scope, &[], 0, &mut |op, at| {
        for read in operands(op) {
            if read == val {
                found.push(at.clone());
            }
        }
        None::<()>
    });
    found.reverse();
    found
}

/// THE LOOP WHOSE `iter_args` BIND A VALUE, AND AT WHICH INDEX — `cast<BlockArgument>(v).getOwner()
/// ->getParentOp()` with the argument's position in `getRegionIterArgs()`.
///
/// ⭐ SEARCHED OVER THE ANCESTORS OF `inside`, INNERMOST FIRST, because a region argument is visible
/// exactly inside the op that binds it: those ancestors are the prefixes of its position.
fn iter_arg_owner(val: Val, inside: &OpId, scope: &[DfirOp]) -> Option<(OpId, usize)> {
    for len in (1..inside.path().len()).rev() {
        let at = OpId::at(&inside.path()[..len]);
        let Some(loop_op) = op_at(&at, scope).and_then(CountedLoop::of) else {
            continue;
        };
        if let Some(index) = loop_op.carried.iter().position(|entry| entry.arg == val) {
            return Some((at, index));
        }
    }
    None
}

/// ONE `iter_args` ENTRY OF THE LOOP AT A POSITION — `getInits()[index]` with the argument and result
/// it binds.
fn carried_entry(at: &OpId, index: usize, scope: &[DfirOp]) -> Option<affine::Carried> {
    let loop_op = op_at(at, scope).and_then(CountedLoop::of)?;
    loop_op.carried.get(index).copied()
}

/// THE FIRST USER OF ANY RESULT OF THE LOOP AT A POSITION THAT IS NOT A YIELD OR AN ENCLOSING
/// `scf.for` — the `DT_CHECK` at `:118-121`, admitting the `scf.yield` its own golden needs (see
/// [`ToggleDuplication::InnerResultUserUnknown`]).
fn unexpected_result_user(at: &OpId, scope: &[DfirOp]) -> Option<OpId> {
    let loop_op = op_at(at, scope)?;
    for result in results(loop_op) {
        // `*(res.getUsers().begin())` — the newest user, which is [`users_of`]'s first, and
        // `res.getUsers().empty()` is the `||`'s left arm.
        let Some(first) = users_of(result, scope).into_iter().next() else {
            continue;
        };
        let Some(user) = op_at(&first, scope) else {
            continue;
        };
        if !matches!(
            user,
            DfirOp::Affine(affine::Op::Yield { .. })
                | DfirOp::Scf(scf::Op::Yield { .. })
                | DfirOp::Scf(scf::Op::For { .. })
        ) {
            return Some(first);
        }
    }
    None
}

/// A READER OF ANY RESULT OF THE LOOP AT A POSITION — `loop->getResult(j).getUsers().empty()` for
/// every `j` (`:147-148`).
fn any_result_user(at: &OpId, scope: &[DfirOp]) -> Option<OpId> {
    let loop_op = op_at(at, scope)?;
    for result in results(loop_op) {
        if let Some(user) = users_of(result, scope).into_iter().next() {
            return Some(user);
        }
    }
    None
}

/// WIDEN ONE LOOP IN PLACE: one `iter_args` entry per `init`, each yielding ITSELF.
///
/// ⛔⛔ NO CLONE AND NO ERASE. This is entry 264's answer without the copy: the island's loop carries
/// its `iter_args` and its body as fields, so pushing an entry and extending the terminator produces
/// the same op the reference builds and erases the old one to reach — and every value the body already
/// reads keeps its identity, which is why the reference's three remaps and
/// `replaceAllUsesWith` (`:122`, `:167-177`) have nothing to do here.
///
/// ⛔ AND NO FILL CONSTANT. Entry 264 initialises each added arg to a fresh `arith.constant` 0 or 1
/// (`Utils.cpp:43`, `:65`) that `setIterArgInit` then overwrites in EVERY caller path here
/// (`:127-133`, `:153-155`), so the init is passed in instead of emitted and replaced.
fn add_iter_args(
    at: &OpId,
    inits: &[Val],
    vals: &mut Values,
    scope: &mut [DfirOp],
) -> Option<AddedIterArgs> {
    let op = op_at_mut(at.path(), scope)?;
    // ⭐ THE TWO COUNTED LOOPS, AND WHICH ONE IT IS DOES NOT CHANGE THE WIDENING — the read side
    // enumerates them exhaustively in [`CountedLoop::of`], which is what decided this is a loop.
    let (carried, body) = match op {
        DfirOp::Affine(affine::Op::For { carried, body, .. })
        | DfirOp::Scf(scf::Op::For { carried, body, .. }) => (carried, body),
        _ => return None,
    };
    // `num_orig_args = upd_iter_args.size() - candidates.size(); DT_CHECK(num_orig_args > 0);`
    // (`:100-101`)
    let num_orig = carried.len();
    if num_orig == 0 {
        return None;
    }
    let mut args: Vec<Val> = Vec::new();
    let mut added: Vec<Val> = Vec::new();
    for init in inits {
        let arg = vals.mint();
        let result = vals.mint();
        carried.push(affine::Carried {
            init: *init,
            arg,
            result,
        });
        args.push(arg);
        added.push(result);
    }
    // `DT_CHECK(isa<affine::AffineYieldOp>(yield_op))` (`:96`, `:99`), then the added arg yielding
    // itself — entry 264's `Utils.cpp:47-50` and `:69-72`, the identity the outer loop then rewires.
    match body.last_mut() {
        Some(DfirOp::Affine(affine::Op::Yield { operands })) => {
            operands.extend(args.iter().copied())
        }
        Some(DfirOp::Scf(scf::Op::Yield { operands })) => operands.extend(args.iter().copied()),
        _ => return None,
    }
    Some(AddedIterArgs {
        num_orig,
        args,
        results: added,
    })
}

/// `yield_op->setOperand(index, value)` ON THE TERMINATOR OF THE LOOP AT A POSITION.
fn set_yield_operand(at: &OpId, index: usize, value: Val, scope: &mut [DfirOp]) {
    let Some(op) = op_at_mut(at.path(), scope) else {
        return;
    };
    let body = match op {
        DfirOp::Affine(affine::Op::For { body, .. }) | DfirOp::Scf(scf::Op::For { body, .. }) => {
            body
        }
        _ => return,
    };
    let (Some(DfirOp::Affine(affine::Op::Yield { operands }))
    | Some(DfirOp::Scf(scf::Op::Yield { operands }))) = body.last_mut()
    else {
        return;
    };
    if let Some(slot) = operands.get_mut(index) {
        *slot = value;
    }
}

/// `op->replaceUsesOfWith(from, to)` ON THE OP AT A POSITION.
fn replace_operands(at: &OpId, from: Val, to: Val, scope: &mut [DfirOp]) {
    let Some(op) = op_at_mut(at.path(), scope) else {
        return;
    };
    for slot in operands_mut(op) {
        if *slot == from {
            *slot = to;
        }
    }
}

/// `builder.setInsertionPointAfter(at)` FOLLOWED BY ONE CLONE PER OP — the block holding `at` gains
/// them in order, immediately below it.
fn insert_after(at: &OpId, ops: Vec<DfirOp>, scope: &mut Vec<DfirOp>) {
    let Some((last, prefix)) = at.path().split_last() else {
        return;
    };
    let ordinal = *last as usize;
    let (block, index) = if prefix.is_empty() {
        (&mut *scope, ordinal)
    } else {
        let Some(owner) = op_at_mut(prefix, scope) else {
            return;
        };
        let Some(found) = block_holding(owner, ordinal) else {
            return;
        };
        found
    };
    let index = (index + 1).min(block.len());
    block.splice(index..index, ops);
}

/// THE REGION OF ONE OP THAT HOLDS A FLATTENED CHILD ORDINAL, AND THAT ORDINAL REBASED ONTO IT — the
/// same flattening [`op_at`] and [`op_at_mut`] descend through.
fn block_holding(op: &mut DfirOp, child: usize) -> Option<(&mut Vec<DfirOp>, usize)> {
    let mut wanted = child;
    for region in regions_mut(op) {
        if wanted < region.len() {
            return Some((region, wanted));
        }
        wanted -= region.len();
    }
    None
}

#[cfg(test)]
mod unit_tests {
    use super::{
        AddedIterArgs, CountedLoop, DfirOp, OpId, ToggleDuplication, Val, Values, affine, arith,
        dataflow, match_and_rewrite, op_at, scf,
    };
    use crate::islands::dataflow_ir::ty::{AffineMap, ElemType, MemRef, ScalarTy};

    /// `%c = arith.constant N : index`.
    fn constant(result: Val, value: i64) -> DfirOp {
        DfirOp::Arith(arith::Op::Constant { result, value })
    }

    /// `%t = arith.subi %lhs, %rhs : index` — the toggle, and every duplicate of it.
    fn subi(result: Val, lhs: Val, rhs: Val) -> DfirOp {
        DfirOp::Arith(arith::Op::SubI(arith::IntBinary {
            result,
            lhs,
            rhs,
            ty: ScalarTy::Index,
        }))
    }

    /// `%x = arith.addi %lhs, %rhs : index` — a reader of the toggle that is not a memory view.
    fn addi(result: Val, lhs: Val, rhs: Val) -> DfirOp {
        DfirOp::Arith(arith::Op::AddI(arith::IntBinary {
            result,
            lhs,
            rhs,
            ty: ScalarTy::Index,
        }))
    }

    /// `%v = dataflow.get_logical_memory_view %mem, %start` — one data transfer's address.
    fn view(result: Val, from: Val, start: Val) -> DfirOp {
        DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView {
            result,
            from,
            start,
            layout: AffineMap::linear(&[1]),
            ty: MemRef {
                shape: vec![128],
                elem: ElemType::Int(8),
            },
        })
    }

    /// `affine.yield ..`.
    fn affine_yield(operands: &[Val]) -> DfirOp {
        DfirOp::Affine(affine::Op::Yield {
            operands: operands.to_vec(),
        })
    }

    /// `scf.yield ..`.
    fn scf_yield(operands: &[Val]) -> DfirOp {
        DfirOp::Scf(scf::Op::Yield {
            operands: operands.to_vec(),
        })
    }

    /// `%r = affine.for %iv = 0 to 2 iter_args(%arg = %init) -> (index) { .. }`.
    fn affine_for(iv: Val, init: Val, arg: Val, result: Val, body: Vec<DfirOp>) -> DfirOp {
        DfirOp::Affine(affine::Op::For {
            iv,
            lo: affine::Bound::Const(0),
            hi: affine::Bound::Const(2),
            carried: vec![affine::Carried { init, arg, result }],
            body,
            dbg_name: None,
        })
    }

    /// The loop at a path, as its `iter_args` and its body.
    fn loop_of(path: &[u32], body: &[DfirOp]) -> (Vec<affine::Carried>, Vec<DfirOp>) {
        let op = op_at(&OpId::at(path), body).expect("the nest holds an op at every asserted path");
        let counted = CountedLoop::of(op).expect("and every one of those ops is a counted loop");
        (counted.carried.to_vec(), counted.body.to_vec())
    }

    /// What a body's terminator yields.
    fn yielded(body: &[DfirOp]) -> Vec<Val> {
        match body.last() {
            Some(DfirOp::Affine(affine::Op::Yield { operands }))
            | Some(DfirOp::Scf(scf::Op::Yield { operands })) => operands.clone(),
            _ => Vec::new(),
        }
    }

    /// The value one reader of the toggle reads it through — a view's start address or an `addi`'s
    /// left operand.
    fn reads(op: &DfirOp) -> Option<Val> {
        match op {
            DfirOp::Dataflow(dataflow::Op::GetLogicalMemoryView { start, .. }) => Some(*start),
            DfirOp::Arith(arith::Op::AddI(bin)) => Some(bin.lhs),
            _ => None,
        }
    }

    /// The `init`s of the entries added past `num_orig`, and their `arg`s and `result`s.
    fn added(carried: &[affine::Carried], num_orig: usize) -> AddedIterArgs {
        AddedIterArgs {
            num_orig,
            args: carried[num_orig..].iter().map(|e| e.arg).collect(),
            results: carried[num_orig..].iter().map(|e| e.result).collect(),
        }
    }

    /// ⭐⭐ THE VENDOR'S GOLDEN, FIELD BY FIELD —
    /// `dcc/test/Transform/CanonicalizeToggle/multiple_toggle_uses.mlir`.
    ///
    /// One toggle in a four-deep nest is read by two extra views and an `arith.addi`, so all four
    /// loops grow three iter_args (1→4, 2→5, 1→4, 1→4 as that file's `:49`, `:56`, `:64` and `:74`
    /// print them), three clones appear under the toggle reading the innermost added args (`:81-84`),
    /// and each reader takes a clone in REVERSE creation order (`:87`, `:96`, `:104`).
    ///
    /// ⛔ AND THE MID LOOP'S RESULTS ARE READ BY AN `scf.yield` (`:118`), which the reference's own
    /// `DT_CHECK` at `DuplicateReusedToggle.cpp:118-121` does not admit — see
    /// [`ToggleDuplication::InnerResultUserUnknown`]. A port that copied that list would abort here.
    #[test]
    fn one_toggle_read_by_three_ops_becomes_four_chains_through_every_loop_of_the_nest() {
        let mut vals = Values::default();
        let c_init = vals.mint();
        let c_sum = vals.mint();
        let c_one = vals.mint();
        let mem = vals.mint();
        let (i0, a0, r0) = (vals.mint(), vals.mint(), vals.mint());
        let (i1, b1, q1) = (vals.mint(), vals.mint(), vals.mint());
        let (a1, r1, sum) = (vals.mint(), vals.mint(), vals.mint());
        let (i2, a2, r2) = (vals.mint(), vals.mint(), vals.mint());
        let (i3, a3, r3) = (vals.mint(), vals.mint(), vals.mint());
        let (c_big, toggle) = (vals.mint(), vals.mint());
        let (v0, v1, v2) = (vals.mint(), vals.mint(), vals.mint());
        let (x, v3) = (vals.mint(), vals.mint());

        // `%74:4 = affine.for %75 = 0 to 2 iter_args(%76 = %66) { .. }` — the toggle, the view the
        // pattern is rooted on, two more views and an `addi`, all reading it.
        let inner = vec![
            constant(c_big, 3_573_504),
            subi(toggle, c_big, a3),
            view(v0, mem, toggle),
            view(v1, mem, toggle),
            view(v2, mem, toggle),
            addi(x, toggle, c_one),
            view(v3, mem, x),
            affine_yield(&[toggle]),
        ];
        let mid = vec![affine_for(i3, a2, a3, r3, inner), affine_yield(&[r3])];
        // `%52:5 = scf.for %53 = %33 to %48 step %47 iter_args(%54 = %48, %55 = %42)` — TWO original
        // iter_args, of which the chain is the second, so `num_orig_args` differs per loop.
        let scf_body = vec![
            addi(sum, b1, c_one),
            affine_for(i2, a1, a2, r2, mid),
            scf_yield(&[sum, r2]),
        ];
        let outer_body = vec![
            DfirOp::Scf(scf::Op::For {
                iv: i1,
                lo: c_one,
                hi: c_one,
                step: c_one,
                carried: vec![
                    affine::Carried {
                        init: c_sum,
                        arg: b1,
                        result: q1,
                    },
                    affine::Carried {
                        init: a0,
                        arg: a1,
                        result: r1,
                    },
                ],
                body: scf_body,
                dbg_name: None,
            }),
            // `affine.yield %52#0` — the outermost loop yields the scf's FIRST result, and its own
            // result is read by nothing (`:120` of the golden, and `:147-148` of the pass).
            affine_yield(&[q1]),
        ];
        let mut body = vec![
            constant(c_init, 1_786_752),
            constant(c_sum, 2),
            constant(c_one, 1),
            affine_for(i0, c_init, a0, r0, outer_body),
        ];

        let outcome = match_and_rewrite(&OpId::at(&[3, 0, 1, 0, 2]), &mut body, &mut vals);

        let dups = match &outcome {
            ToggleDuplication::Duplicated(dups) => dups.clone(),
            _ => Vec::new(),
        };
        assert_eq!(dups.len(), 3, "one duplicate per candidate: {outcome:?}");

        let (outer, outer_body) = loop_of(&[3], &body);
        let (scf, scf_body) = loop_of(&[3, 0], &body);
        let (mid, mid_body) = loop_of(&[3, 0, 1], &body);
        let (inner, inner_body) = loop_of(&[3, 0, 1, 0], &body);

        // `:49`, `:56`, `:64`, `:74` — one added iter_arg per candidate on every loop of the chain.
        assert_eq!(
            (outer.len(), scf.len(), mid.len(), inner.len()),
            (4, 5, 4, 4)
        );

        // `setIterArgInit(upd_iter_args[j], init_val)` (`:153-155`) — the outermost loop's added args
        // all start from the toggle's own initialiser, `%29` in the golden.
        let outer_added = added(&outer, 1);
        assert_eq!(
            outer.iter().map(|e| e.init).collect::<Vec<_>>(),
            vec![c_init; 4]
        );

        // `setIterArgInit(upd_iter_args[j], prev_upd_iter_args[k])` (`:127-133`) — each inner loop's
        // added args start from its PARENT's added args, `k` beginning at the parent's `num_orig`.
        let scf_added = added(&scf, 2);
        let mid_added = added(&mid, 1);
        let inner_added = added(&inner, 1);
        assert_eq!(
            scf.iter().skip(2).map(|e| e.init).collect::<Vec<_>>(),
            outer_added.args
        );
        assert_eq!(
            mid.iter().skip(1).map(|e| e.init).collect::<Vec<_>>(),
            scf_added.args
        );
        assert_eq!(
            inner.iter().skip(1).map(|e| e.init).collect::<Vec<_>>(),
            mid_added.args
        );

        // `prev_yield_op->setOperand(j, new_loop->getResult(k))` (`:138-141`) — every parent yields
        // the child's added results, its original operands untouched (`:116`, `:118`, `:120`).
        assert_eq!(yielded(&outer_body), [vec![q1], scf_added.results].concat());
        assert_eq!(
            yielded(&scf_body),
            [vec![sum, r2], mid_added.results].concat()
        );
        assert_eq!(yielded(&mid_body), [vec![r3], inner_added.results].concat());
        // `prev_yield_op->setOperand(num_orig_args + i, new_toggle_op->getResult(0))` (`:201`) — the
        // innermost yield carries the original toggle and the three duplicates (`:114`).
        assert_eq!(yielded(&inner_body), [vec![toggle], dups.clone()].concat());

        // `builder.clone(*toggle_op)` with `setOperand(1, upd_iter_args[num_orig_args + i])`
        // (`:188-193`) — three `arith.subi` immediately below the toggle, each reading one added arg
        // (`:81-84`).
        assert_eq!(
            inner_body[2..5].to_vec(),
            dups.iter()
                .zip(&inner_added.args)
                .map(|(result, arg)| subi(*result, c_big, *arg))
                .collect::<Vec<_>>()
        );

        // `candidates[i]->replaceUsesOfWith(..)` (`:195-197`) — the LAST-written reader takes the
        // FIRST clone, which is the order `getUsers()` yields (`:87`, `:96`, `:104` of the golden).
        assert_eq!(
            inner_body[5..10].iter().map(reads).collect::<Vec<_>>(),
            vec![
                Some(toggle),
                Some(dups[2]),
                Some(dups[1]),
                Some(dups[0]),
                Some(x)
            ]
        );
    }

    /// ⛔ THE VENDOR'S NEGATIVE —
    /// `dcc/test/Transform/CanonicalizeToggle/invalid_toggle_use.mlir`, whose expected output is
    /// `DtException: [DuplicateReusedToggle] Toggle user is a yield for an operation other than owning
    /// loop`: the toggle is yielded out of an `scf.if`, not out of the loop that carries it.
    #[test]
    fn a_toggle_yielded_out_of_an_scf_if_is_the_pass_error() {
        let mut vals = Values::default();
        let c_init = vals.mint();
        let c_zero = vals.mint();
        let mem = vals.mint();
        let (i0, a0, r0) = (vals.mint(), vals.mint(), vals.mint());
        let (c_big, toggle) = (vals.mint(), vals.mint());
        let (cond, picked, v0) = (vals.mint(), vals.mint(), vals.mint());

        let loop_body = vec![
            constant(c_big, 3_573_504),
            subi(toggle, c_big, a0),
            constant(c_zero, 0),
            // `%401 = scf.if %400 -> (index) { scf.yield %c0 } else { scf.yield %35 }`
            DfirOp::Scf(scf::Op::If {
                cond,
                results: vec![picked],
                body: vec![scf_yield(&[c_zero])],
                else_body: vec![scf_yield(&[toggle])],
                dbg_name: None,
            }),
            view(v0, mem, toggle),
            affine_yield(&[toggle]),
        ];
        let mut body = vec![
            constant(c_init, 1_786_752),
            affine_for(i0, c_init, a0, r0, loop_body),
        ];

        let outcome = match_and_rewrite(&OpId::at(&[1, 4]), &mut body, &mut vals);

        // The `else` region's terminator, whose parent op is the `scf.if` and not the `affine.for`.
        assert_eq!(
            outcome,
            ToggleDuplication::YieldOutsideOwningLoop(OpId::at(&[1, 3, 1]))
        );
    }
}
