// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! Ported from `DistributeWork.cpp` (SP-E2-04). TTIR's implicit-grid execution
//! model becomes an explicit per-compute-tile work loop.
//!
//! ```text
//!   %px = tt.get_program_id x     %id = ktdp.get_compute_tile_id : index
//!   ... body ...              ->  scf.for %pid = %id to %nb step %nc {
//!   tt.return                       %px = arith.index_cast %pid : index to i32
//!                                   ... body (hoisted, unchanged) ...
//!                                 }
//!                                 tt.return
//! ```
//!
//! THE LANDMARK CONTRACT: the loop's lower bound IS the
//! `ktdp.get_compute_tile_id` result. Everything downstream -- `PlanCorelets`,
//! the emitter -- identifies this loop (against K-loops and stick-loops) by
//! tracing that bound ONE definition hop. A loop whose bound is anything else is
//! not the work loop.
//!
//! MULTI-AXIS grids walk the LINEARIZED grid and recover each axis inside the body
//! with div/mod, following Triton's convention that axis 0 varies fastest:
//!
//! ```text
//!   linear = x + y*gx + z*gx*gy
//!   x = linear % gx
//!   y = (linear / gx) % gy
//!   z =  linear / (gx*gy)
//! ```
//!
//! The single-axis-0 path is deliberately kept byte-for-byte as it was (a bare
//! `index_cast` of the induction variable, NO div/mod), because every banked
//! emission is pinned to those bytes. A `% gx` there would be arithmetically
//! redundant and would still move them.

use crate::ir::*;
use crate::passes::walk::{self, OpPath};
use crate::{Refusal, Result};

const PASS: &str = "DistributeWork";

/// Default grid extent / compute-tile count for the 1-D scope.
const DEFAULT_NUM_BLOCKS: i64 = 64;
const DEFAULT_NUM_CORES: i64 = 32;

pub fn run(module: &mut Module, grid: &[i64]) -> Result<()> {
    // One function per module in this backend; the C++ walks every
    // FunctionOpInterface, which is the same thing here.
    let func_indices: Vec<usize> = module
        .ops
        .iter()
        .enumerate()
        .filter(|(_, o)| o.kind == OpKind::TtFunc)
        .map(|(i, _)| i)
        .collect();
    for fi in func_indices {
        distribute_in_function(module, fi, grid)?;
    }
    Ok(())
}

fn distribute_in_function(module: &mut Module, fi: usize, grid: &[i64]) -> Result<()> {
    // --- Step 0: collect the program-id ops, grouped by axis.
    //
    // Several reads of the SAME axis are fine (and common after inlining); what
    // matters is the set of distinct axes, because that decides the loop shape.
    let entry_ops = &module.ops[fi].regions[0].ops;
    let mut pid_positions: Vec<usize> = Vec::new();
    let mut by_axis: std::collections::BTreeMap<i64, Vec<Ssa>> = Default::default();
    let mut max_axis = 0i64;
    for (i, op) in entry_ops.iter().enumerate() {
        if op.kind != OpKind::TtGetProgramId {
            continue;
        }
        pid_positions.push(i);
        let axis = op.attr(&AttrKey::Axis).and_then(|a| a.as_int()).unwrap_or(0);
        by_axis.entry(axis).or_default().extend(op.results.iter().copied());
        max_axis = max_axis.max(axis);
    }

    // Every program-id read must sit in the ENTRY BLOCK. One nested in an
    // scf.if/scf.for cannot be hoisted from: the hoist would walk off the nested
    // block and pull that region's terminator into the new loop, leaving the
    // nested region terminator-less -- invalid IR.
    let nested = module.ops[fi]
        .ops_deep()
        .into_iter()
        .filter(|o| o.kind == OpKind::TtGetProgramId)
        .count()
        > pid_positions.len();
    if nested {
        return Err(Refusal::new(
            PASS,
            "every tt.get_program_id must sit in the function entry block; this one is \
             nested in a control-flow region (e.g. scf.if/scf.for) -- hoisting it would \
             corrupt that region's terminator",
        ));
    }
    if pid_positions.is_empty() {
        return Ok(()); // single-program kernel: nothing to distribute
    }

    let mut extents: Vec<i64> = grid.to_vec();
    let multi_axis = by_axis.len() > 1 || max_axis != 0;

    // A multi-axis kernel needs real extents; there is nothing in the IR to derive
    // them from, and the fallback placeholder would silently cover the wrong number
    // of blocks. FAIL CLOSED.
    if multi_axis && extents.is_empty() {
        return Err(Refusal::new(
            PASS,
            format!(
                "this kernel reads {} distinct program-id axes (highest axis {max_axis}), \
                 so the launch grid cannot be inferred from the IR; pass the per-axis \
                 extents via the `grid` option (fastest-varying axis first, e.g. \
                 grid=4,32). Refusing to guess: a wrong block count produces a silently \
                 wrong result rather than an error",
                by_axis.len()
            ),
        ));
    }
    if !extents.is_empty() {
        if (extents.len() as i64) <= max_axis {
            return Err(Refusal::new(
                PASS,
                format!(
                    "`grid` has {} extent(s) but the kernel reads program-id axis \
                     {max_axis}; supply an extent for every axis used",
                    extents.len()
                ),
            ));
        }
        if extents.iter().any(|g| *g <= 0) {
            return Err(Refusal::new(
                PASS,
                "`grid` extents must all be positive; got a non-positive extent",
            ));
        }
        // Only the axes the kernel actually reads take part in the walk. Trailing
        // declared-but-unread axes would inflate the block count.
        extents.truncate((max_axis + 1) as usize);
    }

    // The anchor is the FIRST program-id in program order: it fixes the insertion
    // point and the span of body ops to hoist.
    let anchor = pid_positions[0];

    // --- Step 2: loop bounds. lb = core_id, ub = num_blocks, step = num_cores.
    //
    // When `grid` is supplied it is AUTHORITATIVE: the loop walks the linearized
    // grid, so the bound is the product of the extents of the axes actually read.
    // Otherwise fall back to the pre-existing single-axis heuristic, unchanged, so
    // banked emissions stay byte-identical.
    let num_blocks_val = if !extents.is_empty() {
        extents.iter().product()
    } else {
        heuristic_num_blocks(module, fi, anchor)
    };

    let hint = module.ops[fi].regions[0].ops[anchor]
        .result()
        .map(|r| module.hint(r))
        .unwrap_or_else(|| "pid".to_string());

    let core_id = module.fresh_named(&hint);
    let num_blocks = module.fresh_named(&hint);
    let num_cores = module.fresh_named(&hint);
    let iv = module.fresh_named(&hint);

    // --- Step 1: plant the work-distribution landmark, and the two bounds.
    //
    // The C++ builder emits, at the anchor: get_compute_tile_id, then the numCores
    // constant, then the numBlocks constant (`arith::ConstantIndexOp::create` order
    // in Step 2), then the loop. The two constants are loop-invariant and placed
    // BEFORE the loop so they dominate it.
    let landmark = Op::new(OpKind::KtdpGetComputeTileId).with_result(core_id, IrType::Index);
    let nb = Op::new(OpKind::ArithConstant)
        .with_result(num_blocks, IrType::Index)
        .with_attr(AttrKey::Value, Attr::Int(num_blocks_val));
    let nc = Op::new(OpKind::ArithConstant)
        .with_result(num_cores, IrType::Index)
        .with_attr(AttrKey::Value, Attr::Int(DEFAULT_NUM_CORES));

    // --- Step 4: the index->i32 bridge(s), built into the loop body.
    let mut bridges: Vec<Op> = Vec::new();
    if !multi_axis {
        // SINGLE AXIS 0, the banked path: exactly one bare index_cast of the IV. No
        // div/mod.
        let pid_i32 = module.fresh_named(&hint);
        bridges.push(
            Op::new(OpKind::ArithIndexCast)
                .with_result(pid_i32, IrType::Scalar(DType::I32))
                .with_operands([iv]),
        );
        for old in by_axis.get(&0).cloned().unwrap_or_default() {
            walk::replace_all_uses(module, old, pid_i32);
        }
    } else {
        let mut stride = 1i64;
        for axis in 0..=max_axis {
            if let Some(olds) = by_axis.get(&axis).cloned() {
                let mut cur = iv;
                if stride != 1 {
                    let s = module.fresh_named(&hint);
                    bridges.push(
                        Op::new(OpKind::ArithConstant)
                            .with_result(s, IrType::Index)
                            .with_attr(AttrKey::Value, Attr::Int(stride)),
                    );
                    let q = module.fresh_named(&hint);
                    bridges.push(
                        Op::new(OpKind::ArithDivui)
                            .with_result(q, IrType::Index)
                            .with_operands([cur, s]),
                    );
                    cur = q;
                }
                if axis != max_axis {
                    // The `%` on the highest axis is provably redundant (the IV never
                    // reaches the bound), so it is dropped -- exactly as the C++ does.
                    let e = module.fresh_named(&hint);
                    bridges.push(
                        Op::new(OpKind::ArithConstant)
                            .with_result(e, IrType::Index)
                            .with_attr(AttrKey::Value, Attr::Int(extents[axis as usize])),
                    );
                    let r = module.fresh_named(&hint);
                    bridges.push(
                        Op::new(OpKind::ArithRemui)
                            .with_result(r, IrType::Index)
                            .with_operands([cur, e]),
                    );
                    cur = r;
                }
                let as_i32 = module.fresh_named(&hint);
                bridges.push(
                    Op::new(OpKind::ArithIndexCast)
                        .with_result(as_i32, IrType::Scalar(DType::I32))
                        .with_operands([cur]),
                );
                for old in olds {
                    walk::replace_all_uses(module, old, as_i32);
                }
            }
            stride *= extents[axis as usize];
        }
    }

    // --- Step 5: hoist the body into the loop.
    //
    // Every op strictly after the anchor, excluding the terminator, moves into the
    // loop body preserving order. Loop-invariant constants defined BEFORE the
    // anchor stay outside and continue to dominate. The other program-id ops are
    // skipped: their uses are already rewired and they are erased.
    let func = &mut module.ops[fi];
    let block = &mut func.regions[0].ops;
    let terminator = block.len().saturating_sub(1);
    let mut body: Vec<Op> = Vec::new();
    let mut keep: Vec<Op> = Vec::new();
    for (i, op) in block.drain(..).enumerate() {
        if i < anchor {
            keep.push(op);
        } else if i == terminator {
            keep.push(op); // tt.return stays outside
        } else if op.kind == OpKind::TtGetProgramId {
            // dead: all uses rewired
        } else if i == anchor {
            // the anchor itself is a program-id, handled above
        } else {
            body.push(op);
        }
    }

    let mut loop_body = bridges;
    loop_body.extend(body);

    let loopp = Op::new(OpKind::ScfFor)
        .with_operands([core_id, num_blocks, num_cores])
        .with_region(Region { args: vec![(iv, IrType::Index)], ops: loop_body });

    // The landmark and the two bounds go at the anchor, then the loop.
    keep.insert(anchor, landmark);
    keep.insert(anchor + 1, nc);
    keep.insert(anchor + 2, nb);
    keep.insert(anchor + 3, loopp);
    *block = keep;
    Ok(())
}

/// SP-E4-09c, kept because the banked single-axis emissions depend on it.
///
/// The first `construct_memory_view` carries the global rows (`sizes[0]` = GM);
/// the first `construct_access_tile` carries the per-program block rows
/// (`shape[0]` = BM). A grid of GM/BM blocks distributes over the compute tiles at
/// step = num_cores. Only override when GM > BM (a genuine multi-block grid) and BM
/// divides GM -- otherwise keep the placeholder so existing paths are byte-unchanged.
fn heuristic_num_blocks(module: &Module, fi: usize, anchor: usize) -> i64 {
    let block = &module.ops[fi].regions[0].ops;
    let mut gm = 0i64;
    let mut bm = 0i64;
    let terminator = block.len().saturating_sub(1);
    for op in block.iter().take(terminator).skip(anchor + 1) {
        if op.kind == OpKind::KtdpConstructMemoryView && gm == 0 {
            if let Some(Attr::IntList(s)) = op.attr(&AttrKey::Shape) {
                if let Some(first) = s.first() {
                    if *first > 0 {
                        gm = *first;
                    }
                }
            }
        }
        if op.kind == OpKind::KtdpConstructAccessTile && bm == 0 {
            if let Some(d) = op.result_type().and_then(|t| t.dims()) {
                if let Some(first) = d.first() {
                    if *first > 0 {
                        bm = *first;
                    }
                }
            }
        }
    }
    if gm > 0 && bm > 0 && gm > bm && gm % bm == 0 {
        gm / bm
    } else {
        DEFAULT_NUM_BLOCKS
    }
}

/// The per-core work loop: its lower bound traces one definition hop to
/// `ktdp.get_compute_tile_id`. THE LANDMARK CONTRACT, shared with `PlanCorelets`
/// so the two cannot disagree about which loop is the work loop.
pub fn is_per_core_work_loop(module: &Module, loopp: &Op) -> bool {
    if loopp.kind != OpKind::ScfFor {
        return false;
    }
    loopp
        .operands
        .first()
        .and_then(|v| module.def_of(*v))
        .map(|d| d.kind == OpKind::KtdpGetComputeTileId)
        .unwrap_or(false)
}

/// Every per-core work loop's path.
pub fn work_loops(module: &Module) -> Vec<OpPath> {
    walk::paths(module)
        .into_iter()
        .filter(|p| {
            walk::at(module, p).map(|o| is_per_core_work_loop(module, o)).unwrap_or(false)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::parse;

    const TWO_AXIS: &str = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %c4 = arith.constant 4 : i32
    %x = tt.get_program_id x : i32
    %y = tt.get_program_id y : i32
    %a = arith.divsi %y, %c4 : i32
    %b = arith.muli %x, %c4 : i32
    tt.return
  }
}
";

    #[test]
    fn a_two_axis_grid_linearizes_with_the_redundant_modulo_dropped() {
        let mut m = parse::parse(TWO_AXIS).unwrap();
        run(&mut m, &[2, 4]).unwrap();
        let f = m.kernel().unwrap();
        let top: Vec<&str> = f.regions[0].ops.iter().map(|o| o.kind.spelling()).collect();
        assert_eq!(
            top,
            vec![
                "arith.constant",            // %c4, defined before the anchor: stays out
                "ktdp.get_compute_tile_id",  // the landmark
                "arith.constant",            // num_cores = 32
                "arith.constant",            // num_blocks = 2*4 = 8
                "scf.for",
                "tt.return",
            ]
        );
        // ub = 8 (the product), step = 32.
        let forr = &f.regions[0].ops[4];
        assert_eq!(super::super::dot_to_linalg::const_int(&m, forr.operands[1]), Some(8));
        assert_eq!(super::super::dot_to_linalg::const_int(&m, forr.operands[2]), Some(32));

        // Axis 0 (stride 1, not the highest) -> remui then index_cast.
        // Axis 1 (stride 2, IS the highest) -> divui then index_cast, NO remui.
        let body: Vec<&str> = forr.regions[0].ops.iter().map(|o| o.kind.spelling()).collect();
        assert_eq!(
            body,
            vec![
                "arith.constant",     // extent 2 for the axis-0 modulo
                "arith.remui",
                "arith.index_cast",
                "arith.constant",     // stride 2 for the axis-1 divide
                "arith.divui",
                "arith.index_cast",   // and NO remui on the highest axis
                "arith.divsi",
                "arith.muli",
            ]
        );
    }

    #[test]
    fn the_loops_lower_bound_is_the_landmark() {
        let mut m = parse::parse(TWO_AXIS).unwrap();
        run(&mut m, &[2, 4]).unwrap();
        let forr = m
            .ops_deep()
            .into_iter()
            .find(|o| o.kind == OpKind::ScfFor)
            .unwrap()
            .clone();
        assert!(
            is_per_core_work_loop(&m, &forr),
            "PlanCorelets finds the work loop by exactly this trace"
        );
        assert_eq!(work_loops(&m).len(), 1);
    }

    #[test]
    fn a_multi_axis_kernel_without_a_grid_is_refused_by_name() {
        let mut m = parse::parse(TWO_AXIS).unwrap();
        let e = run(&mut m, &[]).unwrap_err();
        assert!(e.message.contains("cannot be inferred from the IR"), "got {e}");
        assert!(e.message.contains("Refusing to guess"), "got {e}");
    }

    #[test]
    fn a_grid_too_short_for_the_axes_read_is_refused_by_name() {
        let mut m = parse::parse(TWO_AXIS).unwrap();
        let e = run(&mut m, &[2]).unwrap_err();
        assert!(e.message.contains("supply an extent for every axis used"), "got {e}");
    }

    #[test]
    fn a_single_axis_kernel_gets_a_bare_index_cast_and_no_div_mod() {
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %c64 = arith.constant 64 : i32
    %p = tt.get_program_id x : i32
    %o = arith.muli %p, %c64 : i32
    tt.return
  }
}
";
        let mut m = parse::parse(src).unwrap();
        run(&mut m, &[16]).unwrap();
        let forr = m.ops_deep().into_iter().find(|o| o.kind == OpKind::ScfFor).unwrap();
        let body: Vec<&str> = forr.regions[0].ops.iter().map(|o| o.kind.spelling()).collect();
        assert_eq!(
            body,
            vec!["arith.index_cast", "arith.muli"],
            "the banked path is ONE index_cast -- a redundant `% gx` would move the bytes"
        );
    }
}
