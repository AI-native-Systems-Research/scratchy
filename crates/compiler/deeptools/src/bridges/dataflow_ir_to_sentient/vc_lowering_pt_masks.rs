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

//! `LoweringPTMasks.cpp` — 3 of bridge 2's 384 functions (dependency level(s) [2, 3]).
//!
//! | unit | entry | lines | authority path:line (under /Users/nickm/git/deeptools-src) |
//! |---|---|---|---|
//! | `e239_insertPTMaskOps` | 239/384 | 164 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringPTMasks.cpp:41` |
//! | `e282_updateLoopMaskTreeForConstantMask` | 282/384 | 4 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringPTMasks.cpp:20` |
//! | `e283_updateLoopMaskTreeForDynamicMask` | 283/384 | 9 | `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringPTMasks.cpp:28` |


// ⛔ RE-CREATED ANCHORS. These units' `crustify:todo:` markers were deleted without a
// `/// Replaces:` ever appearing, which removed them from every later schedule and let the
// driver report the campaign DONE. Outstanding work is now computed from UNITS.tsv.
// crustify:todo: e282_updateLoopMaskTreeForConstantMask
// crustify:todo: e283_updateLoopMaskTreeForDynamicMask

use std::collections::VecDeque;

use super::vc_loop_mask_tree::{
    LoopMaskNode, LoopMaskNodeId, LoopMaskTree, MaskIncrement, MaskNodeId, MaskedColumns,
};
use super::vc_vector_operands::OpId;
use crate::islands::dataflow_ir::Values;
use crate::islands::dataflow_ir::ty::ScalarTy;
use crate::islands::sentient::dialects::{self as sen, sentient};

// ══════════════════════════════════════════════════════════════════════════════════════════════
// 239/384
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// AN `OpBuilder` INSERTION POINT AS A POSITION — what entry 239 collects instead of inserting.
///
/// ⛔⛔ THE INSERTS CANNOT BE APPLIED AS THEY ARE FOUND. Each one shifts every later [`OpId`] in its
/// block, and the tree's nodes name their ops by position; the reference's builder holds pointers, so
/// it has no such problem. Collecting and then applying in DESCENDING position order is that
/// mechanism supplied — see [`insert_pt_mask_ops`]'s tail.
struct MaskOpInsertion {
    /// The block the ops go in — an [`OpId`] path with the position dropped.
    block: Vec<u32>,
    /// Where in that block, counting the block's regions concatenated as [`OpId`] does.
    index: u32,
    /// What goes there, in order.
    ops: Vec<sen::Op>,
}

/// `dcc::utils::getNewDbgNameFromOpOrCount` — `dcc/src/Utils/Utils.cpp:503`, which is unscheduled:
/// `"<prefix>(<dbgName>)"` when the op carries one (`:492-501`), else `"<prefix> #<count>"`
/// (`dialect_utils/Dataflow/Utils.cpp:248-253`).
fn new_dbg_name_from_op_or_count(prefix: &str, op: Option<&sen::Op>, count: u32) -> Option<String> {
    match op.and_then(dbg_name_of) {
        Some(name) => Some(format!("{prefix}({name})")),
        None => Some(format!("{prefix} #{count}")),
    }
}

/// `dataflow::getDbgNameAttr(op)` for the two ops entry 239 asks about: the masked
/// `sentient.vector_mac` and the `sentient.for` a dynamic mask hangs under.
fn dbg_name_of(op: &sen::Op) -> Option<&str> {
    match op {
        sen::Op::Sentient(
            sentient::Op::VectorMac { dbg_name, .. } | sentient::Op::For { dbg_name, .. },
        ) => dbg_name.as_deref(),
        _ => None,
    }
}

/// The op at a position, this rung's [`super::vc_vector_operands::op_at`] — one [`OpId`] numbering
/// serves both islands, regions concatenated.
fn op_at<'a>(id: &OpId, scope: &'a [sen::Op]) -> Option<&'a sen::Op> {
    let (&ordinal, rest) = id.path().split_first()?;
    let op = scope.get(ordinal as usize)?;
    let Some(&next) = rest.first() else {
        return Some(op);
    };
    let sen::Op::Sentient(inner) = op else {
        return None;
    };
    let mut base = 0usize;
    for region in sentient::regions(inner) {
        if (next as usize) < base + region.len() {
            let mut sub: Vec<u32> = rest.to_vec();
            sub[0] = (next as usize - base) as u32;
            return op_at(&OpId::at(&sub), region);
        }
        base += region.len();
    }
    None
}

/// `builder.insert(op)` AT A POSITION — the ops go into `block`'s `index`th slot, in order.
fn splice(block: &mut Vec<sen::Op>, path: &[u32], index: usize, ops: Vec<sen::Op>) {
    let Some((&ordinal, rest)) = path.split_first() else {
        for (nth, op) in ops.into_iter().enumerate() {
            block.insert((index + nth).min(block.len()), op);
        }
        return;
    };
    let Some(sen::Op::Sentient(inner)) = block.get_mut(ordinal as usize) else {
        return;
    };
    // Which region of `inner` the next step lands in, and where in it — the concatenation [`OpId`]
    // numbers by.
    let target = rest.first().map_or(index, |next| *next as usize);
    let mut chosen: Option<(usize, usize)> = None;
    let mut base = 0usize;
    for (which, region) in sentient::regions(inner).iter().enumerate() {
        let end = base + region.len();
        if target < end || (rest.is_empty() && target == end) {
            chosen = Some((which, target - base));
            break;
        }
        base = end;
    }
    let Some((which, local)) = chosen else {
        return;
    };
    let mut regions = sentient::regions_mut(inner);
    let Some(region) = regions.get_mut(which) else {
        return;
    };
    if rest.is_empty() {
        splice(region, &[], local, ops);
    } else {
        let mut sub: Vec<u32> = rest.to_vec();
        sub[0] = local as u32;
        splice(region, &sub, index, ops);
    }
}

/// One `sentient.scalar_constant` and the `sentient.set_mask` reading it — the pair the reference
/// creates at each of its three `SetMaskOp` sites.
fn set_mask(
    vals: &mut Values,
    count: &mut u32,
    block: Vec<u32>,
    index: u32,
    value: i64,
    named_after: Option<&sen::Op>,
) -> MaskOpInsertion {
    let mask_val = vals.mint();
    *count += 1;
    MaskOpInsertion {
        block,
        index,
        ops: vec![
            sen::Op::Sentient(sentient::Op::ScalarConstant {
                value,
                result: mask_val,
                reg_locale: sentient::RegType::Imm,
                ty: ScalarTy::Index,
                is_symbol: false,
            }),
            sen::Op::Sentient(sentient::Op::SetMask {
                mask_value: mask_val,
                dbg_name: new_dbg_name_from_op_or_count("PTSetMask", named_after, *count),
            }),
        ],
    }
}

/// `for_op.getRegion().back().getTerminator()` AS A POSITION — this island's `sentient.for` region
/// ends in a `sentient.yield`, and an insert *before the terminator* is one slot short of the end.
fn before_terminator(loop_op: Option<&sen::Op>) -> u32 {
    let Some(sen::Op::Sentient(inner)) = loop_op else {
        return 0;
    };
    let regions = sentient::regions(inner);
    let len: usize = regions.iter().map(|region| region.len()).sum();
    let terminated = regions
        .last()
        .and_then(|region| region.last())
        .is_some_and(|op| matches!(op, sen::Op::Sentient(sentient::Op::Yield { .. })));
    (len - usize::from(terminated)) as u32
}

/// `insertMaskOps` (`LoweringPTMasks.cpp:45-104`) — the lambda entry 239 is built around.
///
/// ⛔ THE CONSTANT ARM BRACKETS THE **OP** AND THE DYNAMIC ARM BRACKETS THE **LOOP**, and the dynamic
/// arm's three names come from the PARENT loop, not from the masked op (`:81`, `:90`, `:101`).
fn insert_mask_ops(
    tree: &LoopMaskTree,
    body: &[sen::Op],
    vals: &mut Values,
    count: &mut u32,
    out: &mut Vec<MaskOpInsertion>,
    n: MaskNodeId,
) {
    let Some(op_id) = tree.operation(n.node()) else {
        return;
    };
    let start_val = i64::from(n.mask().start_val.0);

    match n.mask().increment {
        // `if (increment == 0) { … }` — CONSTANT MASK.
        MaskIncrement::Constant => {
            let Some(&at) = op_id.path().last() else {
                return;
            };
            let block = op_id.block().to_vec();
            let op = op_at(op_id, body);
            out.push(set_mask(vals, count, block.clone(), at, start_val, op));
            // `builder.setInsertionPointAfter(op)` with a fresh `0` constant, *"to reset the mask"*.
            out.push(set_mask(vals, count, block, at + 1, 0, op));
        }
        // `else { DT_CHECK_MSG(increment == 1, …); … }` — the check is [`MaskIncrement`]'s two
        // variants, so there is nothing left to test here.
        MaskIncrement::PerParentLoopIteration => {
            // `auto parent_loop = n->getParentNode()->getOperation();`
            let Some(parent) = tree.parent_node(n.node()) else {
                return;
            };
            let Some(loop_id) = tree.operation(parent) else {
                return;
            };
            let Some(&at) = loop_id.path().last() else {
                return;
            };
            let block = loop_id.block().to_vec();
            let loop_op = op_at(loop_id, body);

            out.push(set_mask(vals, count, block.clone(), at, start_val, loop_op));

            *count += 1;
            out.push(MaskOpInsertion {
                block: loop_id.path().to_vec(),
                index: before_terminator(loop_op),
                ops: vec![sen::Op::Sentient(sentient::Op::IncrMask {
                    dbg_name: new_dbg_name_from_op_or_count("PTIncrMask", loop_op, *count),
                })],
            });

            out.push(set_mask(vals, count, block, at + 1, 0, loop_op));
        }
    }
}

/// `verifyLoopNest` (`LoweringPTMasks.cpp:114-138`) — no OTHER mask under the loop children of
/// `node`, where *other* means [`LoopMaskTree::is_mask_equivalent_to_node`] says no.
///
/// ⛔ THE BFS STARTS AT `node`, NOT AT `curr_node` — so it re-walks the whole subtree once per loop
/// child, and `visited_nodes` therefore ends up holding `node`'s entire nest. That is what makes
/// [`insert_pt_mask_ops`]' `visited_nodes` early return bracket a loop ONCE for two equivalent masks.
/// ⚠️ THE `WalkResult::interrupt()` AT `:128` IS A DISCARDED TEMPORARY: `found_mask_node` is what
/// stops the sibling loop, and the walk itself runs to the end.
fn verify_loop_nest(
    tree: &LoopMaskTree,
    visited_nodes: &mut Vec<LoopMaskNodeId>,
    node: LoopMaskNodeId,
    mask_node: MaskNodeId,
) -> bool {
    let mut found_mask_node = false;
    let mut curr_node = tree.first_child(node);
    while let Some(curr) = curr_node {
        if matches!(tree.node(curr), LoopMaskNode::Loop(_)) {
            // `LoopMaskNode::walk<OperationNode::WalkOrder::kBFS>(node, …)` — the base's own
            // breadth-first walk is private to its module and [`LoopMaskTree::walk`] starts at the
            // root, so the queue is here.
            let mut queue: VecDeque<LoopMaskNodeId> = VecDeque::new();
            queue.push_back(node);
            while let Some(n) = queue.pop_front() {
                if !visited_nodes.contains(&n) {
                    visited_nodes.push(n);
                }
                if let Some(m) = tree.mask_node(n)
                    && !tree.is_mask_equivalent_to_node(m, mask_node)
                {
                    found_mask_node = true;
                }
                let mut child = tree.first_child(n);
                while let Some(c) = child {
                    queue.push_back(c);
                    child = tree.next_sibling(c);
                }
            }
        }
        if found_mask_node {
            break;
        }
        curr_node = tree.next_sibling(curr);
    }
    !found_mask_node
}

/// Replaces: e239_insertPTMaskOps
///
/// **239/384** `VectorChainToSentientPTLoweringPass::insertPTMaskOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringPTMasks.cpp:41` (164L).
///
/// ⭐ THE MASK OPS FOR A WHOLE PROGRAM UNIT, one breadth-first pass over the loop mask tree: constant
/// masks bracket their compute, dynamic masks bracket the loop that drives them.
/// ⛔ ONLY NON-DEFAULT CONSTANT MASKS EMIT ANYTHING (`if (start_val != 0)`, `:181`) — a dynamic mask
/// emits even at `start_val == 0`, because the `incrmask` is the point.
/// ⛔ THE COUNTER IS PRE-INCREMENTED ON EVERY OP THE PASS CREATES and is the PASS's, not this walk's:
/// 2 bumps per constant mask, 3 per dynamic one, shared with every other unit it lowers.
/// ⛔ THE UNSUPPORTED CASE IS A `todo!`, which is what `signalPassFailure()` plus
/// `emitOpError("Unsupported: …")` (`:189-192`) leaves for a crate that never runtime-refuses.
pub fn insert_pt_mask_ops(
    pt_masking_tree: &LoopMaskTree,
    unit_body: &mut Vec<sen::Op>,
    vals: &mut Values,
    set_incr_mask_count: &mut u32,
) {
    let mut insertions: Vec<MaskOpInsertion> = Vec::new();
    // `std::unordered_set<LoopMaskNode *> visited_nodes;`
    let mut visited_nodes: Vec<LoopMaskNodeId> = Vec::new();

    {
        let body: &[sen::Op] = unit_body;
        // `pt_masking_tree->walk(analyzeAndInsertMaskOps);`
        pt_masking_tree.walk(&mut |n| {
            // `if (!n->isLoopNode() && n != pt_masking_tree->getRoot()) return n;`
            if !matches!(pt_masking_tree.node(n), LoopMaskNode::Loop(_))
                && n != pt_masking_tree.root()
            {
                return;
            }
            let mut curr_child = pt_masking_tree.first_child(n);
            while let Some(child) = curr_child {
                // `if (visited_nodes.find(curr_child) != visited_nodes.end()) return n;` — ⛔ IT
                // ABANDONS THE WHOLE NODE, not just this child.
                if visited_nodes.contains(&child) {
                    return;
                }
                let Some(mask_node) = pt_masking_tree.mask_node(child) else {
                    curr_child = pt_masking_tree.next_sibling(child);
                    continue;
                };
                match mask_node.mask().increment {
                    MaskIncrement::Constant => {
                        if mask_node.mask().start_val != MaskedColumns(0) {
                            insert_mask_ops(
                                pt_masking_tree,
                                body,
                                vals,
                                set_incr_mask_count,
                                &mut insertions,
                                mask_node,
                            );
                        }
                    }
                    MaskIncrement::PerParentLoopIteration => {
                        if verify_loop_nest(pt_masking_tree, &mut visited_nodes, n, mask_node) {
                            insert_mask_ops(
                                pt_masking_tree,
                                body,
                                vals,
                                set_incr_mask_count,
                                &mut insertions,
                                mask_node,
                            );
                        } else {
                            todo!(
                                "Unsupported: Multiple MACs generate different PT masks for the same loop nest"
                            );
                        }
                    }
                }
                curr_child = pt_masking_tree.next_sibling(child);
            }
        });
    }

    // ⛔ DESCENDING POSITION ORDER, so an insert never moves a position not yet applied: an insert at
    // `[1]` shifts both `[2]` and `[1, k]`, and both sort above it. Ties keep collection order by
    // being applied back to front.
    insertions.sort_by_key(|insertion| {
        let mut key = insertion.block.clone();
        key.push(insertion.index);
        key
    });
    for insertion in insertions.into_iter().rev() {
        splice(
            unit_body,
            &insertion.block,
            insertion.index as usize,
            insertion.ops,
        );
    }
}

#[cfg(test)]
mod insert_pt_mask_ops_tests {
    use super::*;
    use crate::bridges::dataflow_ir_to_sentient::vc_loop_mask_tree::{MaskNode, compute_loops};

    /// The vendor's dynamic case — `dynamic_pt_masking.mlir` brackets the loop, not the mac:
    /// `sentient.set_mask` before it, `sentient.incrmask` inside it before the yield, `set_mask 0`
    /// after it (`:38`, `:53`, `:56`).
    #[test]
    fn a_dynamic_mask_brackets_its_loop_and_increments_before_the_yield() {
        let mut vals = Values::default();
        let mac = sen::Op::Sentient(sentient::Op::VectorMac {
            mask: None,
            xrf_write_ptr: None,
            xrf_read_ptr: None,
            results: Vec::new(),
            op_a: sentient::Operand::from(sentient::Port::Lx),
            op_b: sentient::Operand::from(sentient::Port::Lx),
            op_c: sentient::Operand::from(sentient::Port::Lx),
            result: sentient::ResultPorts::default(),
            mode: sentient::FmaMode::FusedMulAdd,
            compute_precision: sentient::Precision::Fp16,
            fold_mode: None,
            unroll_factor: sentient::UnrollFactor::X1,
            xrf_read_incr: 0,
            xrf_write_incr: 0,
            dbg_name: None,
        });
        let mut unit_body = vec![sen::Op::Sentient(sentient::Op::For {
            iv: vals.mint(),
            bound: vals.mint(),
            carried: Vec::new(),
            dbg_name: None,
            body: vec![
                mac,
                sen::Op::Sentient(sentient::Op::Yield {
                    results: Vec::new(),
                }),
            ],
        })];

        // The tree exactly as the pass gets it: entry 238 builds the loop nodes, entry 236 hangs the
        // mask off the enclosing one.
        let mut tree = compute_loops(&unit_body);
        tree.add_mask_node(
            Some(&OpId::at(&[0])),
            OpId::at(&[0, 0]),
            MaskNode {
                start_val: MaskedColumns(0),
                increment: MaskIncrement::PerParentLoopIteration,
            },
        );

        let mut count = 0u32;
        insert_pt_mask_ops(&tree, &mut unit_body, &mut vals, &mut count);

        // Three ops created, three names taken from the counter.
        assert_eq!(count, 3);
        assert_eq!(unit_body.len(), 5);
        let names: Vec<Option<&str>> = unit_body
            .iter()
            .map(|op| match op {
                sen::Op::Sentient(sentient::Op::SetMask { dbg_name, .. }) => dbg_name.as_deref(),
                _ => None,
            })
            .collect();
        assert_eq!(
            names,
            vec![None, Some("PTSetMask #1"), None, None, Some("PTSetMask #3")]
        );

        let sen::Op::Sentient(sentient::Op::For { body, .. }) = &unit_body[2] else {
            panic!("the loop stays in the middle");
        };
        assert!(matches!(
            &body[1],
            sen::Op::Sentient(sentient::Op::IncrMask { dbg_name })
                if dbg_name.as_deref() == Some("PTIncrMask #2")
        ));
        assert!(matches!(
            &body[2],
            sen::Op::Sentient(sentient::Op::Yield { .. })
        ));
    }
}
