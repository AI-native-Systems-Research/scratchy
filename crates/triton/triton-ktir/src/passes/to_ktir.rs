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

//! TTIR -> KTIR. Ported from `ToSchedulerKTIR.cpp` (Task #37 / gap G1). **THE
//! TRANSLATION IS NOT FINISHED UNTIL THIS PASS RUNS.**
//!
//! # WHY THIS IS `to_ktir` AND NOT `to_scheduler_ktir`
//!
//! The C++ file is called `ToSchedulerKTIR.cpp` and this was named after it, which was a
//! mistake worth undoing rather than preserving. The pass does NO scheduling: the
//! "scheduler" in the C++ name is V1's `dataflow-scheduler`, whose dialect registration is
//! what forced the conversion in the first place. Our path does not go through V1 at all
//! -- it is Triton -> ttir -> KTIR -> SuperDSC -> dxp_standalone on the V2 tooling -- so
//! naming the pass after a component we route around described the pass by its historical
//! motivation instead of by its job. Its job is to produce KTIR from ttir.
//!
//! The C++ name is kept in citations below, because that file really is called that.
//!
//! The pipeline is Triton -> ttir -> **Triton-free KTIR** -> backend, and the backend
//! knows nothing about Triton. `make_ktir` stops one pass short of that: its output
//! still contains `tt.*`, and the tools downstream do not register the `tt` dialect at
//! all. The C++ header quotes what that costs:
//!
//! ```text
//!   flash_ktir.mlir:30:3: error: Dialect `tt' not found for custom op 'tt.func'
//!   note: Available dialects: affine, agen, arith, builtin, dataflow, explan,
//!   func, init, ktdf, ktdf_arch, ktdf_lowering, ktdp, linalg, math, memref, scf,
//!   sdscbundle, sentient, symbol, tensor, trace, uniform, varexpr, vector,
//!   vectorchain
//! ```
//!
//! So the KTIR cannot be PARSED by the tools that would consume it, let alone
//! consumed. **Every rewrite here targets an op in a dialect that list names, and
//! that is the design rule** -- not a style preference and not an optimisation.
//!
//! The same boundary is enforced in scratchy's TYPE: `ktir_core::opkind::OpKind` has
//! fourteen `Linalg*` variants and zero `Tt*`. Triton cannot leak downstream even by
//! accident. See [`crate::consumer`].
//!
//! # THE BROADCAST SHAPE IS A V1 CONSTRAINT, AND V1 IS NOT OUR PATH ANY MORE
//!
//! `tt.broadcast` becomes a `linalg.generic` with a **yield-only body**, not the named
//! `linalg.broadcast`, even though `ktir_core`'s enum has a `LinalgBroadcast` variant and
//! the named form is obviously tidier. This used to be recorded here as "not a choice --
//! do not improve it", on the strength of V1's legality check
//! (`KTIRLegalityCheckPass::isLegalGenericBodyOp`), which ACCEPTS the generic and REJECTS
//! the named op as an unsupported compute op. Probe p07.
//!
//! **That reason no longer applies to this pass.** V1's scheduler is routed around, so its
//! legality check is not a gate we pass through. Two facts now point the other way: their
//! type names `LinalgBroadcast`, and their own producer uses it at exactly the place a
//! reduction result has to reach the lanes again.
//!
//! The generic is nonetheless what this pass still emits, for one reason that is about
//! evidence rather than design: the 12-of-12 golden diff is against the C++ toolchain's
//! output, which is V1's, so switching the shape here would break the only check that
//! says this port reproduces the pass it was ported from. **So the shape is HELD, not
//! endorsed.** Changing it is a decision about what the consumer wants, and when it is
//! made the golden gets a recorded exception rather than the diff quietly going red.
//! This is the first place V1 conventions and V2's diverge, and it will not be the last.

use std::collections::{HashMap, HashSet};

use crate::ir::*;
use crate::passes::walk::{self, OpPath};
use crate::{Refusal, Result};

const PASS: &str = "spyre-to-ktir";

fn refuse(why: impl std::fmt::Display) -> Refusal {
    Refusal::new(PASS, why.to_string())
}

/// The dialects the scheduler registers and this backend emits into. Anything else
/// surviving is a bug in THIS pass, not a gap downstream, so [`verify_only_scheduler_dialects`]
/// reports it by name.
fn is_scheduler_dialect(ns: &str) -> bool {
    matches!(
        ns,
        "builtin" | "func" | "arith" | "math" | "scf" | "tensor" | "memref" | "linalg"
            | "ktdp" | "ktdf_arch"
    )
}

pub fn run(module: &mut Module, grid: &[i64]) -> Result<()> {
    // THE LAUNCH GRID IS REQUIRED. `grid` on the emitted `func.func` is what the
    // scheduler reads to know how many compute tiles the kernel spans; inventing an
    // extent would silently schedule the wrong number of blocks.
    if grid.is_empty() {
        return Err(refuse(
            "the launch grid is required (pass grid=<extents>, fastest-varying axis \
             first, as for spyre-distribute-work). The scheduler reads a `grid` \
             attribute off the function and there is nothing in KTIR to derive it from, \
             so it is not guessed",
        ));
    }
    let mut grid_size = 1i64;
    for e in grid {
        if *e <= 0 {
            return Err(refuse(format!("grid extent must be positive; got {e}")));
        }
        grid_size *= e;
    }

    // Container first: once the pointer arguments are `index`, the casts that bridged
    // `!tt.ptr` to index are index-to-index and fold away, and the body rewrites then
    // run on a `func.func`.
    convert_func(module, grid_size)?;
    drop_corelet_plans(module);
    fold_pointer_casts(module)?;
    convert_body(module)?;
    // LAST body rewrite: fold the grid-stride work loop when it can run at most one
    // trip, so the grid position reaches the scheduler as a per-core uniform instead of
    // a loop induction variable.
    fold_grid_work_loop(module)?;
    // AFTER the fold, deliberately: a single-trip grid has no work loop left by this
    // point, so this step is a no-op on every configuration whose emission is pinned to
    // bytes, and only the genuinely multi-trip case is touched.
    //
    // ⭐ AND THE MULTI-TRIP CASE NOW HOISTS ITS OWN VIEWS, inside the unroll, because the copies
    // have to share ONE declaration of each buffer's width rather than state four. So this is
    // currently inert on all twelve. It is kept, not deleted: it is the guard for any work loop
    // that reaches here still standing, and its planted-defect test still exercises the
    // `handoff::lower` arm that reports a nested view as nested.
    hoist_buffer_views_out_of_work_loops(module);
    // LAST, and after both grid-loop arms: a TIME-tiled loop from the kernel's own source --
    // flash attention's KV sweep -- is not a grid work loop, so neither arm above looks at it, and
    // downstream every window reader walks the function's top level. Constant trip counts unroll;
    // anything else is left for the window guard to report by name. See the pass doc.
    unroll_constant_trip_loops(module)?;
    fold_unit_grid_tile_id(module);
    // AFTER the unit fold, deliberately: a unit grid already folded its landmark to zero and
    // has nothing for this pass to do, and the twelve card-verified kernels at multi-tile grids
    // (the embedding, the whole-node MLPs) never state a computed window corner, so the fire
    // gate below leaves them byte-identical. Only the genuinely corner-carrying multi-tile
    // body -- attention at 16 -- is unrolled.
    unroll_grid_positions(module)?;
    // SECOND TRIP-UNROLL ARM, deliberately after the grid unroll: a CAUSAL sweep's trip count is a
    // function of the grid position (`off_band = f(start_m)`), so the FIRST arm above -- run while
    // the bounds still root at the compute-tile id -- sees no constant and leaves the `scf.for`
    // standing. The grid unroll folds each copy's bounds to constants, and THIS arm unrolls them.
    // On the twelve card-verified kernels it is a no-op (their first-arm loops are gone; nothing
    // new became constant), and on the noncausal 16-tile it is a no-op by construction (the sweep
    // count is grid-invariant, so the first arm already took it).
    unroll_constant_trip_loops(module)?;
    // AND FOLD AGAIN: the trip unroll splices the loop BODIES out as straight line, replacing the
    // induction variable with a per-trip constant -- but the bodies' own index arithmetic on that
    // IV (`addi`, `index_cast` of the corner chains) is left for a folder to fold, and the last
    // fold ran before the splice. Without this the causal kernel's window corners reach the door
    // as `index_cast(constant)` and it refuses them -- the same message the grid-corner wall gave.
    fold_index_arithmetic(module);
    fold_splat_seeds(module);
    decompose_matmul_accumulators(module)?;
    strip_triton_attrs(module);
    // FAIL CLOSED. A conversion that leaves one `tt.*` op behind produces a module that
    // LOOKS converted and still cannot be parsed downstream.
    verify_only_scheduler_dialects(module)
}

//===----------------------------------------------------------------------===//
// Container
//===----------------------------------------------------------------------===//

/// `tt.func` -> `func.func` with a 1-D `grid`; `!tt.ptr` arguments -> `index`;
/// `tt.return` -> `func.return`.
///
/// A `!tt.ptr` argument IS a base address by this point -- the only thing KTIR does
/// with it is feed `ktdp.construct_memory_view`'s `index` offset -- so it is RETYPED
/// rather than routed through a runtime-argument protocol the consumer ignores.
fn convert_func(module: &mut Module, grid_size: i64) -> Result<()> {
    for i in 0..module.ops.len() {
        if module.ops[i].kind != OpKind::TtFunc {
            continue;
        }
        if !module.ops[i].result_types.is_empty() {
            return Err(refuse(format!(
                "a kernel returning a value has no scheduler-KTIR form (the scheduler's \
                 functions return void); got {} results",
                module.ops[i].result_types.len()
            )));
        }
        if module.ops[i].regions.first().map(|r| r.ops.is_empty()).unwrap_or(true) {
            return Err(refuse("kernel has no body"));
        }
        let fnop = &mut module.ops[i];
        fnop.kind = OpKind::FuncFunc;
        // THE GRID, already 1-D: only 1-D grids are supported downstream, and our KTIR
        // already delinearizes the axes itself in spyre-distribute-work, so the FLAT
        // extent is the faithful form rather than a lossy flattening.
        fnop.set_attr(AttrKey::Grid, Attr::IntList(vec![grid_size]));
        for (_, t) in fnop.regions[0].args.iter_mut() {
            if matches!(t, IrType::Ptr { .. }) {
                *t = IrType::Index;
            }
        }
    }
    // `tt.return` -> `func.return`, anywhere in the body.
    walk::for_each_mut(module, |op| {
        if op.kind == OpKind::TtReturn {
            op.kind = OpKind::FuncReturn;
        }
    });
    Ok(())
}

/// OUR CORELET PLAN IS DROPPED, NOT CONVERTED.
///
/// It records how our own physical emitter splits a core's work across its two
/// corelets; the scheduler makes that decision itself, from the `grid` attribute. It is
/// also why `ktdf` is a NAME COLLISION rather than a shared dialect: the scheduler's
/// `ktdf` is its Schedule IR (`ktdf.pipeline` / `ktdf.stage` / `ktdf.data_transfer`),
/// which is why our plan reads as `error: custom op 'ktdf.corelet_plan' is unknown` in
/// its tools (probe p09).
fn drop_corelet_plans(module: &mut Module) {
    let victims: Vec<OpPath> = walk::paths(module)
        .into_iter()
        .filter(|p| {
            walk::at(module, p).map(|o| o.kind == OpKind::KtdfCoreletPlan).unwrap_or(false)
        })
        .collect();
    walk::erase(module, &victims);
}

/// After the arguments are `index`, every surviving `unrealized_conversion_cast` must
/// be the no-op pointer-to-index bridge. Anything else has no lowering downstream.
fn fold_pointer_casts(module: &mut Module) -> Result<()> {
    loop {
        let Some(path) = walk::paths(module).into_iter().find(|p| {
            walk::at(module, p)
                .map(|o| o.kind == OpKind::UnrealizedConversionCast)
                .unwrap_or(false)
        }) else {
            return Ok(());
        };
        let op = walk::at(module, &path).expect("path").clone();
        let in_ty = op.operands.first().and_then(|v| module.type_of(*v));
        let out_ty = op.result_type().cloned();
        if op.operands.len() != 1 || op.results.len() != 1 || in_ty != out_ty {
            return Err(refuse(
                "an unrealized_conversion_cast survives that is not the no-op \
                 pointer-to-index bridge; the scheduler has no lowering for it",
            ));
        }
        let (from, to) = (op.results[0], op.operands[0]);
        walk::replace_all_uses(module, from, to);
        walk::erase(module, &[path]);
    }
}

//===----------------------------------------------------------------------===//
// Body
//===----------------------------------------------------------------------===//

/// The ORDER matters and is the C++'s: reduce before broadcast before expand_dims,
/// because a broadcast's operand may be an expand_dims this pass also rewrites.
fn convert_body(module: &mut Module) -> Result<()> {
    convert_all(module, OpKind::TtReduce, convert_reduce)?;
    convert_all(module, OpKind::TtBroadcast, convert_broadcast)?;
    convert_all(module, OpKind::TtExpandDims, convert_expand_dims)?;
    convert_all(module, OpKind::TtTrans, convert_trans)?;
    convert_all(module, OpKind::TtSplat, convert_splat_of_argument)?;
    expand_splat_of_scalar_argument(module)?;
    Ok(())
}

/// A `tt.splat` whose operand is a BLOCK ARGUMENT, as `tensor.splat`.
///
/// The only `tt.splat` that can reach here: `make_ttir`'s `fold_splat_of_constant` already
/// turned every splat of a compile-time constant into a `dense<>` `arith.constant`, so a
/// surviving one splats a value that has no compile-time value -- which on this path means
/// a function argument (the runtime scalar of the address-provenance ladder's rung 3).
/// `tt.splat` is `tt`-dialect and `verify_only_scheduler_dialects` refuses it, so leaving
/// it standing is not an option; `tensor.splat` is the same op in a dialect the scheduler
/// registers, and it is the spelling `decompose_dense_constants` already produces for the
/// constant case -- so downstream sees ONE form for both provenances.
///
/// ⛔ FAIL CLOSED ON THE OPERAND: a splat of anything but a block arg is refused by name
/// rather than rewritten, because the rewrite asserts the splatted value is a launch
/// binding, and an interior SSA value would make that a lie. The door's scale registry is
/// the consumer of the arg case; no other case has a contract.
fn convert_splat_of_argument(module: &mut Module, path: &OpPath) -> Result<()> {
    let op = walk::at(module, path).expect("path").clone();
    let Some(&operand) = op.operands.first() else {
        return Err(refuse(
            "`tt.splat` with no operand has no value to rewrite and no scheduler form",
        ));
    };
    // Is the operand a block ARGUMENT of the enclosing function? The region's `args`
    // are the function's parameters at this stage (`convert_func` has already run), so
    // membership is the whole test.
    let is_arg = module
        .kernel()
        .ok()
        .and_then(|f| f.regions.first())
        .is_some_and(|r| r.args.iter().any(|(ssa, _)| *ssa == operand));
    if !is_arg {
        return Err(refuse(format!(
            "`tt.splat` of an operand that is not a function argument ({}): \
             `make_ttir` folds every splat of a compile-time constant, so a surviving \
             splat is expected to be of a RUNTIME scalar argument. A splat of an \
             interior value has no contract on this path -- the scheduler does not \
             register `tt.splat`, and rewriting it as `tensor.splat` would silently \
             claim the value is a launch binding",
            operand.0
        )));
    }
    let mut rewritten = Op::new(OpKind::TensorSplat).with_operands([operand]);
    rewritten.results = op.results.clone();
    rewritten.result_types = op.result_types.clone();
    let block = walk::block_mut(module, path).expect("path");
    block[path.index()] = rewritten;
    Ok(())
}

/// A `tensor.splat` of a scalar ARGUMENT as a `ktdp.load` of a `[1,1]` buffer over it —
/// rung 3 of the address-provenance ladder (a runtime scalar the LAUNCHER binds).
///
/// # WHAT THE REWRITE EMITS, AND WHY EXACTLY THAT
///
/// `convert_splat_of_argument` has just turned `tt.splat %scale` into `tensor.splat %scale`,
/// which states the VALUE but still addresses the parameter NOWHERE. The consumer's contract
/// is the handoff guard's own (`every_parameter_states_its_width`, to_ktir_emit.rs): every
/// parameter must have a top-level `ktdp.construct_memory_view`, because
/// `lower_ktir_to_superdsc::regions` pairs parameters to buffers positionally and reads each
/// one's extent off that view. So the rewrite states the signature's OWN fact — one fp16
/// element, the width the launch binds — as the same three-op chain every pointer parameter
/// gets:
///
/// * the argument is RETYPED to `index`, exactly as `convert_func` retypes a `!tt.ptr`:
///   a scalar parameter IS the base address of a 2-byte buffer, and index is the only type
///   a `ktdp.construct_memory_view` address operand has.
/// * `ktdp.construct_memory_view` `[1,1]` fp16 HBM, with `convert_ttir_to_ktdp`'s exact
///   attribute set (Shape/Strides/CoordinateSet/MemorySpace) and an `IrType::MemRef` result;
/// * `ktdp.construct_access_tile` `[1,1]` at corner (0,0), per `build_direct_access_tile`'s
///   conventions (BaseMap identity, AccessTileSet, AccessTileOrder identity), its corner
///   operands `arith.constant 0 : index` — the form `index_constants` resolves;
/// * `ktdp.load`, whose result is a `[1,1]` fp16 tile.
///
/// The `tensor.splat` STAYS, re-pointed at the load. Both consumers need it: the emulator's
/// `tensor.splat` takes a Tile operand's FIRST flat element (dialects/tensor.rs), so the IR
/// is numerically exact there, and the vendor's `Elementwise(Mul)`/`ScalarMul` tiebreak
/// (`splat_scale_of`) still sees a splat on one side of the `arith.mulf`.
///
/// # ⛔ THE CHAIN IS HOISTED TO THE FUNCTION'S TOP, IMMEDIATELY AFTER ENTRY
///
/// The guard requires the view at the TOP LEVEL, `regions()` reads corners from the top
/// level, and a nested view is one of the two refusals that guard exists to distinguish. The
/// splat itself stays where it was — re-pointing its operand is all that changes.
///
/// # ⛔ FAIL CLOSED ON THE OPERAND TYPE
///
/// Only a scalar of a FLOAT dtype is expanded: the launch binds one fp16 (`const:t<id>`
/// encodes IEEE fp16, 2 bytes, and the runner's size check is exact), so an integer scalar
/// has no bind spelling and an `index` scalar is not a buffer base at all. Both are refused
/// by name rather than given a chain whose width nobody states.
///
/// ⛔ AND AN ARGUMENT NO SPLAT READS IS LEFT ALONE. A scalar argument consumed by something
/// other than a splat (nothing in the fixture set does) has no view and will be refused by
/// `every_parameter_states_its_width` — correctly, because what that parameter addresses is
/// a question only its consumer can answer, and inventing a `[1,1]` for it here would be
/// the assumed-extent defect the guard's doc names.
fn expand_splat_of_scalar_argument(module: &mut Module) -> Result<()> {
    use crate::passes::convert_ttir_to_ktdp::build_range_set_nd;

    // Which arguments feed a `tensor.splat`, and at what elem dtype? Deep walk: a splat may
    // already sit inside a region body. Retype them; the splice comes after.
    let kernel = module.kernel().map_err(|e| refuse(e.to_string()))?.clone();
    let Some(region) = kernel.regions.first().cloned() else {
        return Ok(());
    };
    // arg -> the element dtype its splat states (the splat's own result element).
    let mut scalar_args: Vec<(Ssa, DType)> = Vec::new();
    fn collect_splatted_args(
        ops: &[Op],
        args: &[(Ssa, IrType)],
        out: &mut Vec<(Ssa, DType)>,
    ) -> Result<()> {
        for op in ops.iter() {
            if op.kind == OpKind::TensorSplat {
                if let Some(operand) = op.operands.first().copied() {
                    if args.iter().any(|(ssa, _)| *ssa == operand) {
                        let elem = op
                            .result_types
                            .first()
                            .and_then(|t| t.elem())
                            .ok_or_else(|| {
                                refuse(format!(
                                    "a `tensor.splat` of argument {} states no element type, so the \
                                     width of the buffer the launch must bind is unknown",
                                    operand.0
                                ))
                            })?;
                        match out.iter_mut().find(|(s, _)| s == &operand) {
                            Some((_, d)) => {
                                if *d != elem {
                                    return Err(refuse(format!(
                                        "argument {} is splatted at two different element types ({:?} \
                                         and {:?}); one parameter is one buffer, so its width cannot be \
                                         stated twice",
                                        operand.0, *d, elem
                                    )));
                                }
                            }
                            None => out.push((operand, elem)),
                        }
                    }
                }
            }
            for r in op.regions.iter() {
                collect_splatted_args(&r.ops, args, out)?;
            }
        }
        Ok(())
    }
    collect_splatted_args(&region.ops, &region.args, &mut scalar_args)?;
    if scalar_args.is_empty() {
        return Ok(());
    }
    // THE RETYPE, exactly `convert_func`'s: a scalar parameter IS a base address here.
    // Refusals are collected first and reported AFTER the borrow, so the message can name
    // the argument (a hint lookup takes `&module`) while the retype holds `&mut` it.
    {
        let mut bad: Option<(Ssa, IrType)> = None;
        {
            let kernel = module.kernel_mut().map_err(|e| refuse(e.to_string()))?;
            for (ssa, t) in kernel.regions[0].args.iter_mut() {
                if !scalar_args.iter().any(|(s, _)| s == ssa) {
                    continue;
                }
                match t {
                    IrType::Scalar(DType::F16) | IrType::Scalar(DType::F32) => {
                        *t = IrType::Index;
                    }
                    other => {
                        if bad.is_none() {
                            bad = Some((*ssa, other.clone()));
                        }
                    }
                }
            }
        }
        if let Some((ssa, ty)) = bad {
            let name = module.hint(ssa);
            return Err(refuse(format!(
                "the scalar argument %{name} is splatted but is of type {ty:?}, and a rung-3 \
                 launch binding is ONE fp16 (`const:t<id>` encodes IEEE fp16, 2 B) — so any other \
                 scalar has no bind spelling and no width this rewrite can state. Declaring it \
                 `fp16` is the contract",
            )));
        }
    }

    // One `[1,1]` view/tile/load chain per scalar argument. Names minted BEFORE the body
    // borrow: `fresh_named` needs `&mut module` and the splice needs `&mut` the body, so
    // the two phases cannot interleave.
    let mut named: Vec<(Ssa, DType, Ssa, Ssa, Ssa, Ssa, Ssa)> = Vec::new();
    for (arg, elem) in &scalar_args {
        let hint = module.hint(*arg);
        let view = module.fresh_named(&hint);
        let r0 = module.fresh_named(&hint);
        let c0 = module.fresh_named(&hint);
        let tile = module.fresh_named(&hint);
        let loaded = module.fresh_named(&hint);
        named.push((*arg, *elem, view, r0, c0, tile, loaded));
    }

    let kernel = module.kernel_mut().map_err(|e| refuse(e.to_string()))?;
    let body = &mut kernel.regions[0].ops;
    for (j, (arg, elem, view, r0, c0, tile, loaded)) in named.into_iter().enumerate() {
        let view_op = Op::new(OpKind::KtdpConstructMemoryView)
            .with_result(view, IrType::MemRef { dims: vec![1, 1], elem })
            .with_operands([arg])
            .with_attr(AttrKey::Shape, Attr::IntList(vec![1, 1]))
            .with_attr(AttrKey::Strides, Attr::IntList(vec![1, 1]))
            .with_attr(AttrKey::CoordinateSet, Attr::AffineSet(build_range_set_nd(&[1, 1])))
            .with_attr(AttrKey::MemorySpace, Attr::Str("HBM".into()));

        let zero_r = Op::new(OpKind::ArithConstant)
            .with_result(r0, IrType::Index)
            .with_attr(AttrKey::Value, Attr::Int(0));
        let zero_c = Op::new(OpKind::ArithConstant)
            .with_result(c0, IrType::Index)
            .with_attr(AttrKey::Value, Attr::Int(0));
        let tile_op = Op::new(OpKind::KtdpConstructAccessTile)
            .with_result(tile, IrType::AccessTile { dims: vec![1, 1] })
            .with_operands([view, r0, c0])
            .with_attr(AttrKey::BaseMap, Attr::AffineMap(identity_map(2)))
            .with_attr(AttrKey::AccessTileSet, Attr::AffineSet(build_range_set_nd(&[1, 1])))
            .with_attr(AttrKey::AccessTileOrder, Attr::AffineMap(identity_map(2)));

        let load_op = Op::new(OpKind::KtdpLoad)
            .with_result(loaded, IrType::Tensor { dims: vec![1, 1], elem })
            .with_operands([tile]);

        // THE SPLAT IS RE-POINTED, not moved: the chain dominates it wherever the splat
        // stood, because the chain is spliced at the function's top. Deep walk — a splat
        // may sit inside a region body.
        let mut repointed = 0usize;
        repoint_deep(body, &arg, loaded, &mut repointed);
        if repointed == 0 {
            return Err(refuse(format!(
                "argument {} was collected as splatted but no `tensor.splat` names it — an \
                 internal inconsistency between the collection walk and this splice",
                arg.0
            )));
        }
        let chain = [zero_r, zero_c, view_op, tile_op, load_op];
        let at = j * chain.len();
        for (k, op) in chain.into_iter().enumerate() {
            body.insert(at + k, op);
        }
    }
    Ok(())
}

/// Re-point every `tensor.splat` over `from` to `to`, at any depth.
fn repoint_deep(ops: &mut [Op], from: &Ssa, to: Ssa, n: &mut usize) {
    for o in ops.iter_mut() {
        if o.kind == OpKind::TensorSplat && o.operands.first() == Some(from) {
            o.operands[0] = to;
            *n += 1;
        }
        for r in o.regions.iter_mut() {
            repoint_deep(&mut r.ops, from, to, n);
        }
    }
}

/// A `tt.trans` the access-tile fold could not absorb, as `linalg.transpose`.
///
/// # WHY ANY `tt.trans` REACHES HERE AT ALL
///
/// `ConvertTTIRToKTDP`'s `FoldTransIntoAccessTileOrder` is the good lowering: it folds the
/// permutation into the access tile's traversal ORDER, so the transpose costs nothing and no
/// second copy of the operand exists. Its own header spells out the stake -- a pre-transposed
/// K plane on the host DOUBLES the KV cache, the dominant memory consumer in inference.
///
/// But that fold requires the transposed value to come straight from a `ktdp.load`: there has
/// to be a tile whose order can absorb the permutation. In `decoder_layer_one` the operand is
/// ROPE'D K -- `arith.subf` output -- so there is no load and no tile, and the C++ pass leaves
/// the `tt.trans` in place for exactly the same reason (its stage-1 golden still contains two).
/// So this is a gap in BOTH paths rather than something the port missed.
///
/// # WHY `linalg.transpose` IS THE RIGHT ANSWER HERE AND NOT THE GENERAL ONE
///
/// It is a REAL transpose: it moves data, which the fold does not. That is acceptable on this
/// path and only on this path -- the Granite forward exists to exercise the end-to-end flow,
/// not to be fast. The performant answer for a computed operand is to produce it already
/// transposed (have the kernel write K^T, so the permutation is absorbed by the WRITE the way
/// the fold absorbs it into a read), and that is a kernel-shape change rather than a compiler
/// one. Recorded here so the cost is visible at the place that pays it.
fn convert_trans(module: &mut Module, path: &OpPath) -> Result<()> {
    let op = walk::at(module, path).expect("path").clone();
    let src_ty = module
        .type_of(op.operands[0])
        .ok_or_else(|| refuse("tt.trans operand has no type"))?;
    let res_ty = op
        .result_types
        .first()
        .cloned()
        .ok_or_else(|| refuse("tt.trans has no result type"))?;

    let order = match op.attr(&AttrKey::Order) {
        Some(Attr::IntList(o)) => o.clone(),
        _ => {
            return Err(refuse(
                "tt.trans has no `order`, so its permutation is unknown. Refused rather \
                 than assumed to be a reversal: a wrong permutation is a silently \
                 transposed tile",
            ))
        }
    };
    if order.len() != src_ty.rank() {
        return Err(refuse(format!(
            "tt.trans `order` has {} entries for a rank-{} operand",
            order.len(),
            src_ty.rank()
        )));
    }

    // `linalg.transpose` writes into an init, as every DPS op does.
    let hint = module.hint(op.results[0]);
    let init = module.fresh_named(&hint);
    let init_op = Op::new(OpKind::TensorEmpty).with_result(init, res_ty.clone());

    let transpose = Op::new(OpKind::Other("linalg.transpose".into()))
        .with_result(op.results[0], res_ty)
        .with_operands([op.operands[0], init])
        .with_attr(AttrKey::Order, Attr::IntList(order));

    let idx = path.index();
    let block = walk::block_mut(module, path).expect("path");
    block[idx] = transpose;
    block.insert(idx, init_op);
    Ok(())
}

fn convert_all(
    module: &mut Module,
    kind: OpKind,
    f: impl Fn(&mut Module, &OpPath) -> Result<()>,
) -> Result<()> {
    loop {
        let Some(path) = walk::paths(module)
            .into_iter()
            .find(|p| walk::at(module, p).map(|o| o.kind == kind).unwrap_or(false))
        else {
            return Ok(());
        };
        f(module, &path)?;
    }
}

/// The identity element of a combiner -- the value the DPS init must hold for
/// `linalg.generic`'s accumulate-into-`outs` form to compute the same reduction.
///
/// ONLY COMBINERS WHOSE IDENTITY IS KNOWN are supported. The alternative is seeding an
/// accumulator with a fabricated value, which produces a WRONG ANSWER SILENTLY rather
/// than a diagnostic.
fn combiner_identity(kind: &OpKind, elem: DType) -> Option<FloatBits> {
    let f16 = elem == DType::F16;
    Some(match kind {
        OpKind::ArithAddf => {
            if f16 { FloatBits { bits: 0x0000, width: 16 } } else { FloatBits::f32(0.0) }
        }
        OpKind::ArithMulf => {
            if f16 { FloatBits { bits: 0x3c00, width: 16 } } else { FloatBits::f32(1.0) }
        }
        // -inf for a max, +inf for a min.
        OpKind::ArithMaxnumf => {
            if f16 {
                FloatBits { bits: 0xFC00, width: 16 }
            } else {
                FloatBits::f32(f32::NEG_INFINITY)
            }
        }
        OpKind::ArithMinnumf => {
            if f16 { FloatBits { bits: 0x7C00, width: 16 } } else { FloatBits::f32(f32::INFINITY) }
        }
        _ => return None,
    })
}

fn identity_map(rank: usize) -> String {
    let d: Vec<String> = (0..rank).map(|i| format!("d{i}")).collect();
    format!("({}) -> ({})", d.join(", "), d.join(", "))
}

/// A map over `rank` dims whose results are `keep`, in order.
fn projected_map(rank: usize, keep: &[usize]) -> String {
    let d: Vec<String> = (0..rank).map(|i| format!("d{i}")).collect();
    let r: Vec<String> = keep.iter().map(|i| format!("d{i}")).collect();
    format!("({}) -> ({})", d.join(", "), r.join(", "))
}

/// `tt.reduce` over one axis -> a `linalg.generic` with a reduction iterator, the
/// combiner inlined, and the combiner's identity as the DPS init.
fn convert_reduce(module: &mut Module, path: &OpPath) -> Result<()> {
    let op = walk::at(module, path).expect("path").clone();
    if op.operands.len() != 1 || op.results.len() != 1 {
        return Err(refuse(format!(
            "only a single-operand, single-result tt.reduce is supported; got {} \
             operands and {} results",
            op.operands.len(),
            op.results.len()
        )));
    }
    let src_ty = module
        .type_of(op.operands[0])
        .ok_or_else(|| refuse("tt.reduce's operand has no type"))?;
    let res_ty = op.result_type().cloned().ok_or_else(|| refuse("tt.reduce has no result type"))?;
    let (src_rank, res_rank) = (src_ty.rank(), res_ty.rank());
    if src_ty.dims().is_none() || res_ty.dims().is_none() || res_rank + 1 != src_rank {
        return Err(refuse(
            "tt.reduce does not drop exactly one dimension of a ranked tensor",
        ));
    }
    let axis = op.attr(&AttrKey::Axis).and_then(|a| a.as_int()).unwrap_or(-1);
    if axis < 0 || axis as usize >= src_rank {
        return Err(refuse(format!("tt.reduce axis {axis} is out of range")));
    }
    let elem = src_ty
        .elem()
        .filter(|e| e.is_float())
        .ok_or_else(|| refuse(
            "only a float tt.reduce is supported (an integer reduction needs its own \
             identity table)",
        ))?;

    // The combiner must be ONE recognized associative op plus the terminator.
    let region = op.regions.first().ok_or_else(|| refuse("tt.reduce has no combiner region"))?;
    if region.args.len() != 2 {
        return Err(refuse(
            "tt.reduce combiner does not take exactly two arguments",
        ));
    }
    let body: Vec<&Op> = region.ops.iter().filter(|o| o.kind != OpKind::TtReduceReturn).collect();
    let ret = region
        .ops
        .iter()
        .find(|o| o.kind == OpKind::TtReduceReturn)
        .ok_or_else(|| refuse("tt.reduce combiner does not end in a single-value tt.reduce.return"))?;
    if body.len() != 1 {
        return Err(refuse(format!(
            "tt.reduce combiner must be exactly one associative op with a known \
             identity; got {} ops",
            body.len()
        )));
    }
    let combine = body[0].clone();
    if ret.operands.first() != combine.results.first() {
        return Err(refuse(
            "tt.reduce combiner does not return its combining op's result",
        ));
    }
    let (a0, a1) = (region.args[0].0, region.args[1].0);
    if combine.operands.len() != 2
        || !combine.operands.contains(&a0)
        || !combine.operands.contains(&a1)
    {
        return Err(refuse(
            "tt.reduce combiner op does not combine both block arguments",
        ));
    }
    let ident = combiner_identity(&combine.kind, elem).ok_or_else(|| {
        refuse(format!(
            "tt.reduce combiner `{}` has no known identity element, so its \
             linalg.generic accumulator cannot be seeded; refusing rather than \
             fabricating a seed",
            combine.kind.spelling()
        ))
    })?;

    // init = splat(identity), shaped like the result.
    let hint = module.hint(op.results[0]);
    let ident_scalar = module.fresh_named(&hint);
    let init = module.fresh_named(&hint);
    let ident_op = Op::new(OpKind::ArithConstant)
        .with_result(ident_scalar, IrType::Scalar(elem))
        .with_attr(AttrKey::Value, Attr::Float(ident));
    let init_op = Op::new(OpKind::TensorSplat)
        .with_result(init, res_ty.clone())
        .with_operands([ident_scalar]);

    // ins: identity over the source rank; outs: drop the reduced axis.
    let keep: Vec<usize> = (0..src_rank).filter(|i| *i as i64 != axis).collect();
    let iters: Vec<String> = (0..src_rank)
        .map(|i| if i as i64 == axis { "reduction".to_string() } else { "parallel".to_string() })
        .collect();

    // The combiner's block arg 0 is the incoming element and arg 1 the accumulator,
    // matching linalg.generic's (in, out) body arguments.
    let generic_body = Region {
        args: vec![(a0, IrType::Scalar(elem)), (a1, IrType::Scalar(elem))],
        ops: vec![
            combine.clone(),
            Op::new(OpKind::LinalgYield).with_operands([combine.results[0]]),
        ],
    };
    let generic = Op::new(OpKind::LinalgGeneric)
        .with_result(op.results[0], res_ty)
        .with_operands([op.operands[0], init])
        .with_attr(
            AttrKey::IndexingMaps,
            Attr::AffineMapList(vec![identity_map(src_rank), projected_map(src_rank, &keep)]),
        )
        .with_attr(AttrKey::IteratorTypes, Attr::StrList(iters))
        .with_region(generic_body);

    let idx = path.index();
    let block = walk::block_mut(module, path).expect("path");
    block[idx] = generic;
    block.insert(idx, init_op);
    block.insert(idx, ident_op);
    Ok(())
}

/// `tt.broadcast` fans unit dims out to the result extent: collapse the unit dims
/// away, then a `linalg.generic` whose ins map is the projected permutation onto the
/// kept dims and whose body is a BARE YIELD. See the module docs on why the named
/// `linalg.broadcast` is wrong here.
fn convert_broadcast(module: &mut Module, path: &OpPath) -> Result<()> {
    let op = walk::at(module, path).expect("path").clone();
    let src_ty = module
        .type_of(op.operands[0])
        .ok_or_else(|| refuse("tt.broadcast's operand has no type"))?;
    let res_ty = op.result_type().cloned().ok_or_else(|| refuse("tt.broadcast has no result"))?;
    let (Some(sd), Some(rd)) = (src_ty.dims(), res_ty.dims()) else {
        return Err(refuse("tt.broadcast operands are not equal-rank ranked tensors"));
    };
    if sd.len() != rd.len() {
        return Err(refuse("tt.broadcast operands are not equal-rank ranked tensors"));
    }
    let rank = rd.len();
    let mut kept: Vec<usize> = Vec::new();
    let mut bcast = vec![false; rank];
    for i in 0..rank {
        if sd[i] == rd[i] {
            kept.push(i);
        } else if sd[i] == 1 {
            bcast[i] = true;
        } else {
            return Err(refuse(format!(
                "tt.broadcast dim {i} goes from {} to {}, which is neither a unit-dim \
                 broadcast nor a match",
                sd[i], rd[i]
            )));
        }
    }
    if kept.is_empty() {
        return Err(refuse(
            "tt.broadcast from an all-unit shape is not supported (it would need a \
             rank-0 source)",
        ));
    }
    if !bcast.iter().any(|b| *b) {
        return Err(refuse(
            "tt.broadcast is a no-op (source and result shapes are equal); it should \
             have been folded",
        ));
    }

    let hint = module.hint(op.results[0]);
    // Drop the unit dims so the generic's ins map is a projected permutation.
    let collapsed_shape: Vec<i64> = kept.iter().map(|d| sd[*d]).collect();
    let collapsed_ty = IrType::Tensor {
        dims: collapsed_shape,
        elem: src_ty.elem().unwrap_or(DType::F16),
    };
    let src = module.fresh_named(&hint);
    let collapse = Op::new(OpKind::TensorCollapseShape)
        .with_result(src, collapsed_ty)
        .with_operands([op.operands[0]])
        .with_attr(AttrKey::Reassociation, Attr::IntList(kept.iter().map(|d| *d as i64).collect()));

    let init = module.fresh_named(&hint);
    let empty = Op::new(OpKind::TensorEmpty).with_result(init, res_ty.clone());

    let elem = res_ty.elem().unwrap_or(DType::F16);
    let yin = module.fresh_named(&hint);
    let yout = module.fresh_named(&hint);
    let generic = Op::new(OpKind::LinalgGeneric)
        .with_result(op.results[0], res_ty)
        .with_operands([src, init])
        .with_attr(
            AttrKey::IndexingMaps,
            Attr::AffineMapList(vec![projected_map(rank, &kept), identity_map(rank)]),
        )
        .with_attr(
            AttrKey::IteratorTypes,
            Attr::StrList(vec!["parallel".to_string(); rank]),
        )
        .with_region(Region {
            args: vec![(yin, IrType::Scalar(elem)), (yout, IrType::Scalar(elem))],
            // A BARE YIELD. Not linalg.broadcast -- see the module docs (probe p07).
            ops: vec![Op::new(OpKind::LinalgYield).with_operands([yin])],
        });

    let idx = path.index();
    let block = walk::block_mut(module, path).expect("path");
    block[idx] = generic;
    block.insert(idx, empty);
    block.insert(idx, collapse);
    Ok(())
}

/// `tt.expand_dims` -> `tensor.expand_shape`.
fn convert_expand_dims(module: &mut Module, path: &OpPath) -> Result<()> {
    let op = walk::at(module, path).expect("path").clone();
    let src_ty = module
        .type_of(op.operands[0])
        .ok_or_else(|| refuse("tt.expand_dims' operand has no type"))?;
    let res_ty = op.result_type().cloned().ok_or_else(|| refuse("tt.expand_dims has no result"))?;
    if src_ty.dims().is_none() || res_ty.dims().is_none() || res_ty.rank() != src_ty.rank() + 1 {
        return Err(refuse(
            "tt.expand_dims is not a rank-increasing reshape of ranked tensors",
        ));
    }
    let axis = op.attr(&AttrKey::Axis).and_then(|a| a.as_int()).unwrap_or(-1);
    let rd = res_ty.dims().unwrap();
    if axis < 0 || axis as usize >= rd.len() || rd[axis as usize] != 1 {
        return Err(refuse(format!(
            "tt.expand_dims axis {axis} is not a unit dim of the result"
        )));
    }
    let expand = Op::new(OpKind::TensorExpandShape)
        .with_result(op.results[0], res_ty)
        .with_operands([op.operands[0]])
        .with_attr(AttrKey::Axis, Attr::Int(axis));
    let idx = path.index();
    walk::block_mut(module, path).expect("path")[idx] = expand;
    Ok(())
}

//===----------------------------------------------------------------------===//
// The grid work loop
//===----------------------------------------------------------------------===//

/// Fold the grid-stride work loop WHEN IT CAN RUN AT MOST ONE TRIP, replacing its
/// induction variable by its lower bound -- the `ktdp.get_compute_tile_id` landmark --
/// and inlining its body.
///
/// WHY, and it is about ADDRESSING rather than tidiness. The grid position reaches
/// every memory view's start address. Left as a loop, that position is a loop induction
/// variable, and the downstream compute-group extraction lifts the enclosing loop into
/// the CALLER by design -- so the position arrives inside each extracted schedule as a
/// FUNCTION ARGUMENT, which dcc can address in neither place one is needed: not as an
/// AGEN subscript (only an `scf.for`/`affine.for` induction variable or an
/// `arith.constant`) and not as a memory view's start. Folded, the position is a
/// `ktdp.get_compute_tile_id` -- a PER-CORE UNIFORM, which is the one non-constant form
/// that route accepts.
///
/// WHAT IS GIVEN UP IS A GUARD, AND IT BECOMES A LAUNCH CONTRACT. The loop is
/// `for pid = core_id to work_items step num_cores`: a tile whose id is >=
/// `work_items` ran ZERO trips. Inlining unconditionally makes that tile run one trip
/// at an out-of-range grid position. An `scf.if` would only move the problem -- the
/// condition is an `i1` computed per core, and an i1 cannot cross a compute-group
/// boundary on this device. So the guard becomes an obligation on whoever launches:
/// **launch exactly `work_items` compute tiles**, recorded on the function as
/// `spyre.folded_grid_loop` so the caller is told, in the artifact, what it must
/// guarantee.
///
/// AT MOST ONE TRIP is `work_items <= num_cores`, given `core_id >= 0`. A loop with
/// `work_items > num_cores` is GENUINELY MULTI-TRIP -- folding it would drop every work
/// item after the first -- so it is UNROLLED instead, one straight-line copy per trip.
/// See the multi-trip arm below for why that is the only remaining answer and why it is
/// exact.
fn fold_grid_work_loop(module: &mut Module) -> Result<()> {
    let loops: Vec<OpPath> = crate::passes::distribute_work::work_loops(module);
    for path in loops.into_iter().rev() {
        let loopp = walk::at(module, &path).expect("path").clone();
        let work_items = crate::passes::dot_to_linalg::const_int(module, loopp.operands[1]);
        let num_cores = crate::passes::dot_to_linalg::const_int(module, loopp.operands[2]);
        let (Some(work_items), Some(num_cores)) = (work_items, num_cores) else {
            return Err(refuse(
                "the grid work loop's bounds are not compile-time constants, so whether \
                 it can run more than one trip is not decidable here",
            ));
        };
        if work_items > num_cores {
            // GENUINELY MULTI-TRIP, SO THE LOOP IS UNROLLED: one straight-line copy of the
            // body per trip, trip `t` addressing `core_id + t * num_cores`.
            //
            // ⛔⛔⛔ WHY THE LOOP CANNOT SIMPLY STAY, WHICH IS WHAT THIS ARM USED TO DO.
            // MEASURED at grid 4,32: `ktir_superdsc::emit::lower_ktir_to_superdsc::regions`
            // refused with
            //
            //   rope_fwd: parameter 0 (t0) states 2 of its 2 `ktdp.construct_access_tile`
            //   window(s) INSIDE a region (an `scf.for` body), and every reader here walks the
            //   function's TOP LEVEL -- so the window would fall back to the whole `[8192, 128]`
            //   view and every body below would describe a tile the program never takes.
            //
            // Their `regions`, `param_tiles` and `r_cover` all walk `IRFunction::operations`,
            // which is the top-level op list. A window inside the loop is a window none of them
            // can see, and the fallback is `unwrap_or((0, 0, rows, cols))` -- "the window is the
            // whole buffer", the one answer a work-loop program never means. `grep -rn ScfFor` is
            // EMPTY in both that crate and scratchy's `lower_subtile_tape_to_ktir.rs`: their
            // producer emits straight-line functions with the windows at the top level, and that
            // is the shape this arm has to produce.
            //
            // ⭐ AND HOISTING THE WINDOWS -- the guard's other named answer -- IS NOT AVAILABLE,
            // read off the emission rather than argued. `hoist_buffer_views_out_of_work_loops`
            // below lifts `ktdp.construct_memory_view` because a view's only operand is a
            // function parameter and its extent is a constant ATTRIBUTE, so it is loop-invariant
            // by construction. A WINDOW is not: `distribute_work`'s axis bridges give
            // `rope_fwd`'s windows the row offset `(pid / 4) * 256 + (pid % 4) * 64`, i.e.
            // `arith.divui`/`arith.remui` of the loop's INDUCTION VARIABLE. There is no single
            // window above the loop that stands for all four trips, and the only way to state
            // every one of them at the top level is to have one copy per trip. So the guard's two
            // answers collapse to one here, and it is this one.
            //
            // ⭐ THE UNROLL IS EXACT, NOT AN APPROXIMATION, AND IT GIVES UP NO GUARD. The loop is
            // `for pid = core_id to work_items step num_cores` with `core_id` in
            // `[0, num_cores)`, so core `c` runs trips at `c, c + num_cores, ...`. Substituting
            // `core_id + t * num_cores` for the induction variable in copy `t` reproduces exactly
            // those trips, in order, and every work item is still covered exactly once. Unlike
            // the fold there is no zero-trip tile to invent a trip for, so there is no launch
            // obligation to record and `spyre.folded_grid_loop` is deliberately not set -- the
            // same reasoning that applied when this arm left the loop alone.
            //
            // ⭐ WHY THE POSITION IS STILL ADDRESSABLE. The note above records that the grid
            // position must reach the scheduler either as an `scf.for`/`affine.for` induction
            // variable or as a per-core uniform, because compute-group extraction lifts an
            // enclosing loop into the CALLER and the position then arrives as a function
            // argument, which dcc can use in neither place one is needed. After the unroll every
            // trip's position is `ktdp.get_compute_tile_id` plus a constant -- the uniform form,
            // which is the same form the folded configurations already ship.
            //
            // ⛔ IT MUST DIVIDE. With `work_items % num_cores != 0` the cores do not all run the
            // same number of trips, so no fixed number of copies is right for all of them:
            // unrolling to the larger count makes the short cores compute an out-of-range work
            // item, and unrolling to the smaller drops work. Neither is a thing to do silently.
            if work_items % num_cores != 0 {
                return Err(refuse(format!(
                    "the grid work loop spreads {work_items} work items over {num_cores} compute \
                     tiles and {num_cores} does not divide {work_items}: cores 0..{short} would \
                     run {long} trips and the rest {tripc}, so no fixed number of straight-line \
                     copies covers every work item exactly once. Unrolling to {long} would \
                     compute an out-of-range work item on the short cores and unrolling to \
                     {tripc} would drop work, so neither is done. Launch a grid whose work-item \
                     count is a multiple of {num_cores}",
                    short = work_items % num_cores,
                    long = work_items / num_cores + 1,
                    tripc = work_items / num_cores,
                )));
            }
            let trips = work_items / num_cores;
            let iv = loopp.regions[0].args[0].0;
            let core_id = loopp.operands[0];
            let hint = module.hint(core_id);

            // ⭐ A LOOP-INVARIANT VIEW IS EMITTED ONCE, NOT ONCE PER TRIP. `ktdp.construct_memory_view`
            // takes the base address as its ONLY operand and states its extent/strides/space as
            // constant ATTRIBUTES, so it computes the same value on every trip -- which is the same
            // fact `hoist_buffer_views_out_of_work_loops` below is built on. Copying it per trip
            // would leave four identical declarations of each buffer's width where their
            // `regions()` expects one, and would make `r_cover` scan four copies of every window
            // set. The test is the pass's own: an op moves only if every one of its operands is
            // already defined ABOVE the loop, so one built from the work index is left in the body
            // and duplicated with it.
            let idx = path.index();
            let outer: HashSet<Ssa> = match module.kernel() {
                Ok(f) if !f.regions.is_empty() => {
                    let mut set: HashSet<Ssa> =
                        f.regions[0].args.iter().map(|(v, _)| *v).collect();
                    // Only ops the loop is actually below. When the loop is NOT at the kernel's
                    // top level this under-counts, which costs a hoist and never correctness.
                    for op in f.regions[0].ops.iter().take(idx) {
                        set.extend(op.results.iter().copied());
                    }
                    set
                }
                _ => HashSet::new(),
            };
            let mut shared: Vec<Op> = Vec::new();
            let mut body: Vec<Op> = Vec::new();
            for op in loopp.regions[0].ops.iter() {
                if op.kind == OpKind::KtdpConstructMemoryView
                    && op.operands.iter().all(|v| outer.contains(v))
                {
                    shared.push(op.clone());
                } else {
                    body.push(op.clone());
                }
            }

            let mut flat: Vec<Op> = shared;
            for t in 0..trips {
                // Trip 0 IS the landmark, so it needs no arithmetic and keeps the body's own
                // names -- which makes the unrolled module's first copy read exactly like the
                // folded one.
                let pid = if t == 0 {
                    core_id
                } else {
                    let off = module.fresh_named(&hint);
                    let pid = module.fresh_named(&hint);
                    flat.push(
                        Op::new(OpKind::ArithConstant)
                            .with_result(off, IrType::Index)
                            .with_attr(AttrKey::Value, Attr::Int(t * num_cores)),
                    );
                    flat.push(
                        Op::new(OpKind::ArithAddi)
                            .with_result(pid, IrType::Index)
                            .with_operands([core_id, off]),
                    );
                    pid
                };
                let mut copy = body.clone();
                // EVERY VALUE THE COPY DEFINES IS RENAMED, block arguments of any nested region
                // included. Four copies sharing one set of SSA names is not four trips -- it is
                // one trip whose definitions the later copies shadow, and the reader would pair
                // each use with whichever definition it found first.
                let mut map: HashMap<Ssa, Ssa> = HashMap::new();
                map.insert(iv, pid);
                if t > 0 {
                    let mut defined: Vec<Ssa> = Vec::new();
                    defined_values(&copy, &mut defined);
                    for v in defined {
                        let h = module.hint(v);
                        let fresh = module.fresh_named(&h);
                        map.insert(v, fresh);
                    }
                }
                rename_values(&mut copy, &map);
                flat.extend(copy);
            }

            {
                let block = walk::block_mut(module, &path).expect("path");
                block.remove(idx);
                for (k, o) in flat.into_iter().enumerate() {
                    block.insert(idx + k, o);
                }
            }

            // WHAT `grid` MEANS, and it is `num_cores` rather than `work_items`. `convert_func`
            // set it to the product of the launch extents, i.e. `work_items`, which is right when
            // the loop FOLDS because one tile then runs per work item. Unrolled, `num_cores`
            // tiles launch and each covers `trips` work items, so leaving `grid` at `work_items`
            // would over-launch by the trip count -- here by 4x.
            if let Some(f) = module.ops.iter_mut().find(|o| o.kind == OpKind::FuncFunc) {
                f.set_attr(AttrKey::Grid, Attr::IntList(vec![num_cores]));
            }
            continue;
        }
        // The induction variable becomes the landmark itself.
        let iv = loopp.regions[0].args[0].0;
        let core_id = loopp.operands[0];
        walk::replace_all_uses(module, iv, core_id);

        // Inline the body where the loop was -- RE-READ, never the clone taken above.
        // That clone predates the rewire, so inlining it puts the loop's dead induction
        // variable back into every address the body computes: the fold appears to work,
        // the loop is gone, and the grid position is still a value nothing defines.
        let body = walk::at(module, &path).expect("path").regions[0].ops.clone();
        let idx = path.index();
        {
            let block = walk::block_mut(module, &path).expect("path");
            block.remove(idx);
            for (k, o) in body.into_iter().enumerate() {
                block.insert(idx + k, o);
            }
        }
        // THE LAUNCH CONTRACT, recorded in the artifact -- and it carries the two
        // NUMBERS, not a bare flag. "Launch exactly `work_items` compute tiles" is an
        // obligation, and an obligation whose quantity the caller has to rediscover is
        // most of the way to no obligation at all. The C++ records
        // `{num_cores, work_items}` for that reason.
        if let Some(f) = module.ops.iter_mut().find(|o| o.kind == OpKind::FuncFunc) {
            f.set_attr(
                AttrKey::FoldedGridLoop,
                Attr::StrList(vec![
                    format!("num_cores = {num_cores} : index"),
                    format!("work_items = {work_items} : index"),
                ]),
            );
        }
    }
    Ok(())
}

//===----------------------------------------------------------------------===//
// Time tiling
//===----------------------------------------------------------------------===//

/// Unroll every REMAINING `scf.for` whose trip count is a compile-time constant, threading its
/// `iter_args` from copy to copy by SUBSTITUTION.
///
/// # WHY, AND IT IS THE SAME REASON THE GRID LOOP UNROLLS
///
/// [`fold_grid_work_loop`]'s multi-trip arm records the refusal in full: every reader in
/// `ktir_superdsc::emit::lower_ktir_to_superdsc` -- `regions`, `param_tiles`, `r_cover` -- walks
/// `IRFunction::operations`, the function's TOP-LEVEL op list, and a `ktdp.construct_access_tile`
/// inside an `scf.for` body is a window none of them can see. Their fallback is "the window is the
/// whole view", which is the one answer a tiled program never means. MEASURED on
/// `attention_flash_noncausal`:
///
/// ```text
///   attn_fwd_noncausal: parameter 1 (t1) states 1 of its 1 `ktdp.construct_access_tile`
///   window(s) INSIDE a region (an `scf.for` body) ... A time-tiled program has to be split into
///   one node per trip, or its windows hoisted, before a descriptor can span it.
/// ```
///
/// That loop is the flash-attention KV sweep. It is NOT a grid work loop, so `fold_grid_work_loop`
/// never looks at it: it comes from the kernel's own `for start_n in tl.range(lo, hi, BLOCK_N)`.
/// Hoisting its windows is unavailable for the same reason it was there -- K's window offset is an
/// `iter_arg` the body advances by `BLOCK_N` each trip, so no single window above the loop stands
/// for all four. So one copy per trip is again the only answer.
///
/// # WHAT MAKES IT DECIDABLE, AND WHY A LOOP THAT IS NOT IS LEFT ALONE
///
/// The non-causal sweep is `lo, hi = 0, N_CTX` over `tl.range(0, 256, 64)` with every bound a
/// `tl.constexpr`, so the trip count is 4 AT COMPILE TIME and the unroll is exact. The CAUSAL
/// off-band sweep is `lo, hi = 0, start_m * BLOCK_M`: its trip count is the query-block index, a
/// function of grid position, and there is no fixed number of copies that is right for every core.
///
/// **A loop whose bounds are not constants is therefore LEFT STANDING, not refused here.** That is
/// deliberate and it is not a hole: the guard quoted above is what catches it, one crate
/// downstream, by name and with the parameter and the view extent in the message. Refusing here
/// would replace a specific diagnostic with a vaguer one, and -- since this pass runs on every
/// configuration -- would turn any future loop-carrying program into a refusal at a stage that has
/// nothing to say about it.
///
/// # THE SUBSTITUTION IS THE WHOLE OF IT
///
/// Copy `t` is the body with three sets of names rewritten: the induction variable becomes the
/// constant `lb + t * step`; each `iter_arg` becomes the value copy `t - 1` yielded for it (copy 0
/// takes the loop's own init operands); and every value the copy DEFINES is renamed afresh, since
/// four copies sharing one set of names is one trip that the later copies shadow. The `scf.yield`
/// is dropped from each copy -- its operands ARE the next copy's carries -- and uses of the loop's
/// own results are rewired to the LAST copy's carries. Nothing is folded, reassociated or dropped,
/// so the unrolled straight line computes the recurrence the loop did, trip for trip.
fn unroll_constant_trip_loops(module: &mut Module) -> Result<()> {
    // DEEPEST-FIRST, so unrolling an outer loop copies bodies whose inner loops are already
    // straight-line, and no path is invalidated under us: `walk::paths` is pre-order, so reversing
    // it visits children before parents and later siblings before earlier ones.
    let paths: Vec<OpPath> = walk::paths(module)
        .into_iter()
        .filter(|p| walk::at(module, p).map(|o| o.kind == OpKind::ScfFor).unwrap_or(false))
        .collect();
    for path in paths.into_iter().rev() {
        let loopp = walk::at(module, &path).expect("path").clone();
        let lb = crate::passes::dot_to_linalg::const_int(module, loopp.operands[0]);
        let ub = crate::passes::dot_to_linalg::const_int(module, loopp.operands[1]);
        let step = crate::passes::dot_to_linalg::const_int(module, loopp.operands[2]);
        // NOT DECIDABLE HERE IS NOT AN ERROR HERE -- see the doc comment. The downstream window
        // guard is the one that reports it, and it reports it better.
        let (Some(lb), Some(ub), Some(step)) = (lb, ub, step) else {
            continue;
        };
        if step <= 0 {
            // A non-positive step is not a trip count this pass can reason about, and unrolling it
            // would loop forever. The downstream window guard reports it.
            continue;
        }
        if ub <= lb {
            // ZERO TRIPS, AND THE LOOP'S RESULTS ARE ITS INITS. Causal attention's off-band sweep
            // is exactly this at grid position 0: `scf.for(0, 0, 64, ...)` states "no KV blocks in
            // band", and leaving it standing puts its windows inside a region the door cannot read.
            // The body never runs, so every iter_arg's final value is the init it started with --
            // which is what the multi-trip arm's `carries = inits` threading already states at
            // zero iterations.
            let block = walk::block_mut(module, &path).expect("path");
            block.remove(path.index());
            for (res, init) in loopp.results.iter().zip(loopp.operands[3..].iter()) {
                walk::replace_all_uses(module, *res, *init);
            }
            continue;
        }
        let trips = (ub - lb + step - 1) / step;

        let (iv, iv_ty) = loopp.regions[0].args[0].clone();
        let carried: Vec<(Ssa, IrType)> = loopp.regions[0].args[1..].to_vec();
        let inits: Vec<Ssa> = loopp.operands[3..].to_vec();
        if carried.len() != inits.len() || loopp.results.len() != carried.len() {
            return Err(refuse(format!(
                "an `scf.for` states {} iter_arg block argument(s), {} init operand(s) and {} \
                 result(s); unrolling threads one carry per iter_arg, so a disagreement between \
                 the three is a malformed loop rather than something to guess at",
                carried.len(),
                inits.len(),
                loopp.results.len()
            )));
        }

        // The body, minus its terminator. `scf.yield`'s operands are the carries, so it is a
        // JOIN between copies rather than an op any copy contains.
        let mut body = loopp.regions[0].ops.clone();
        let yielded: Vec<Ssa> = match body.last() {
            Some(o) if o.kind == OpKind::ScfYield => {
                let ops = o.operands.clone();
                body.pop();
                ops
            }
            _ if carried.is_empty() => Vec::new(),
            _ => {
                return Err(refuse(
                    "an `scf.for` with iter_args does not end in `scf.yield`, so what each trip \
                     carries to the next is not stated and cannot be inferred",
                ))
            }
        };
        if yielded.len() != carried.len() {
            return Err(refuse(format!(
                "an `scf.for` carries {} iter_arg(s) but yields {} value(s)",
                carried.len(),
                yielded.len()
            )));
        }

        // A LOOP-INVARIANT BUFFER VIEW IS STATED ONCE, NOT ONCE PER TRIP -- the same rule, and the
        // same reason, as the grid unroll's `shared` split. `ktdp.construct_memory_view` declares a
        // buffer's WIDTH, and `regions()` expects one declaration per parameter; four copies of it
        // would have that reader scan four identical statements of the same fact. A view built from
        // the induction variable or an iter_arg is not invariant, so it stays in the body and is
        // duplicated with it.
        let mut inside: Vec<Ssa> = vec![iv];
        inside.extend(carried.iter().map(|(v, _)| *v));
        defined_values(&body, &mut inside);
        let inside: HashSet<Ssa> = inside.into_iter().collect();
        let mut flat: Vec<Op> = Vec::new();
        let mut per_trip: Vec<Op> = Vec::new();
        for op in body.into_iter() {
            if op.kind == OpKind::KtdpConstructMemoryView
                && op.operands.iter().all(|v| !inside.contains(v))
            {
                flat.push(op);
            } else {
                per_trip.push(op);
            }
        }

        // THE IV IS MATERIALISED ONLY WHERE IT IS READ. Attention's sweep advances its own
        // `iter_args` and never mentions `start_n`, so emitting a constant per trip would leave
        // four dead constants at the top level of every such program.
        let iv_used = {
            let mut used = false;
            fn scan(ops: &[Op], v: Ssa, used: &mut bool) {
                for o in ops {
                    if o.operands.contains(&v) {
                        *used = true;
                    }
                    for r in &o.regions {
                        scan(&r.ops, v, used);
                    }
                }
            }
            scan(&per_trip, iv, &mut used);
            used
        };

        let mut carries: Vec<Ssa> = inits;
        for t in 0..trips {
            let mut map: HashMap<Ssa, Ssa> = HashMap::new();
            if iv_used {
                let c = module.fresh_named(&module.hint(iv));
                flat.push(
                    Op::new(OpKind::ArithConstant)
                        .with_result(c, iv_ty.clone())
                        .with_attr(AttrKey::Value, Attr::Int(lb + t * step)),
                );
                map.insert(iv, c);
            }
            for ((a, _), carry) in carried.iter().zip(carries.iter()) {
                map.insert(*a, *carry);
            }
            let mut copy = per_trip.clone();
            // Trip 0 keeps the body's own names, so the unrolled first copy reads exactly like the
            // body did -- the same convention the grid unroll follows.
            if t > 0 {
                let mut defined: Vec<Ssa> = Vec::new();
                defined_values(&copy, &mut defined);
                for v in defined {
                    let h = module.hint(v);
                    let fresh = module.fresh_named(&h);
                    map.insert(v, fresh);
                }
            }
            rename_values(&mut copy, &map);
            flat.extend(copy);
            // The next trip's carries are THIS copy's yields, under this copy's renaming. A carry
            // the body passes straight through is an iter_arg, which `map` already resolves to the
            // incoming carry -- so the chain is closed for both shapes.
            carries = yielded.iter().map(|v| *map.get(v).unwrap_or(v)).collect();
        }

        let idx = path.index();
        {
            let block = walk::block_mut(module, &path).expect("path");
            block.remove(idx);
            for (k, o) in flat.into_iter().enumerate() {
                block.insert(idx + k, o);
            }
        }
        // The loop's results are the LAST trip's carries. Rewired after the splice, so the
        // epilogue below the loop reads the final accumulator rather than a value nothing defines.
        for (res, carry) in loopp.results.iter().zip(carries.iter()) {
            walk::replace_all_uses(module, *res, *carry);
        }
    }
    Ok(())
}

/// A GRID OF ONE HAS EXACTLY ONE COMPUTE TILE, so `ktdp.get_compute_tile_id` is statically 0 and
/// every index expression rooted at it folds to a constant. The ladder's fixtures launch at grid
/// (1, 1, 1) yet address their activation windows at `ctid * BLOCK`, and the window readers
/// downstream (`regions()`'s first-tile lookup) only resolve `arith.constant` corners -- a
/// `ctid * 64` corner is invisible to them and the region silently falls back to the whole view.
/// Folding the landmark at the source is the one place the fact "there is one tile" is known.
fn fold_unit_grid_tile_id(module: &mut Module) {
    let Some(f) = module.ops.iter().find(|o| o.kind == OpKind::FuncFunc) else {
        return;
    };
    let grid: Vec<i64> = match f.attr(&AttrKey::Grid) {
        Some(Attr::IntList(g)) => g.clone(),
        _ => return,
    };
    if grid.iter().product::<i64>() != 1 {
        return;
    }
    // THE LANDMARK BECOMES ZERO. Each use is rewritten to a fresh `arith.constant` of the same
    // type, so the fold works anywhere in the body (regions included) without a numbering pass.
    let mut replacements: Vec<(OpPath, Ssa, i64)> = Vec::new();
    for path in walk::paths(module) {
        let Some(o) = walk::at(module, &path) else { continue };
        if o.kind != OpKind::KtdpGetComputeTileId {
            continue;
        }
        for r in &o.results {
            replacements.push((path.clone(), *r, 0));
        }
    }
    for (path, res, val) in replacements {
        let ty = walk::at(module, &path)
            .and_then(|o| o.result_types.first().cloned())
            .expect("landmark type");
        let c = module.fresh_named(&module.hint(res));
        let idx = path.index();
        let block = walk::block_mut(module, &path).expect("path");
        block.remove(idx);
        block.insert(
            idx,
            Op::new(OpKind::ArithConstant)
                .with_result(c, ty)
                .with_attr(AttrKey::Value, Attr::Int(val)),
        );
        walk::replace_all_uses(module, res, c);
    }
    // CONSTANT INDEX ARITHMETIC FOLDS, so `0 * 64` and friends do not survive as a dead shell the
    // window readers still cannot read. `arith.index_cast` of a constant folds to the constant too.
    //
    // ⭐ `remui`/`divui`/`remsi` ARE HERE because the grid-id chain is made of them, measured on
    // `attention_flash_noncausal` at grid [1,1]: `start_m` is `remui(ctid, 4)` and `start_n`'s
    // head offset `divui(ctid, 4)`, so without these three the landmark fold above replaces
    // `ctid` with 0 and then STOPS — the corner chain stays `remui(0, 4)`-shaped, not a constant,
    // and the window reader refuses a corner that has been a constant since the landmark folded.
    // The grid-delinearisation is Triton's own spelling (`tl.program_id` arithmetic), so a chain
    // of exactly these ops is what any multi-dim-grid kernel's corners look like.
    fold_index_arithmetic(module);
}

/// Constant index arithmetic folds to a constant: `muli`/`addi`/`divsi`/`remui`/`divui`/`remsi`
/// of two `arith.constant`s, and `arith.index_cast` of one. Shared by the unit-grid fold (which
/// runs it over a landmark folded to 0) and the grid-position unroll (which runs it over N
/// landmarks folded to 0..N-1) — the SAME folder, so a corner materialised by either reads
/// identically downstream.
fn fold_index_arithmetic(module: &mut Module) {
    loop {
        let mut folded = false;
        for path in walk::paths(module) {
            let Some(o) = walk::at(module, &path) else { continue };
            let (a, b) = match o.kind {
                OpKind::ArithMuli | OpKind::ArithAddi | OpKind::ArithDivsi | OpKind::ArithRemui
                | OpKind::ArithDivui | OpKind::ArithRemsi => {
                    (o.operands[0], o.operands[1])
                }
                OpKind::ArithIndexCast => (o.operands[0], o.operands[0]),
                _ => continue,
            };
            let val = match o.kind {
                OpKind::ArithMuli => crate::passes::dot_to_linalg::const_int(module, a).zip(crate::passes::dot_to_linalg::const_int(module, b)).map(|(x, y)| x * y),
                OpKind::ArithAddi => crate::passes::dot_to_linalg::const_int(module, a).zip(crate::passes::dot_to_linalg::const_int(module, b)).map(|(x, y)| x + y),
                // Truncating division, the C semantics the op has; division by zero is left standing
                // rather than turned into a build error in a fold.
                OpKind::ArithDivsi => crate::passes::dot_to_linalg::const_int(module, a).zip(crate::passes::dot_to_linalg::const_int(module, b)).and_then(|(x, y)| {
                    if y == 0 { None } else { Some(x / y) }
                }),
                // Unsigned remainder/division: the semantics `remui`/`divui` name. `remsi` is
                // signed, but on the non-negative constants a grid chain produces the two agree;
                // a negative operand is left standing rather than guessed at.
                OpKind::ArithRemui => crate::passes::dot_to_linalg::const_int(module, a).zip(crate::passes::dot_to_linalg::const_int(module, b)).and_then(|(x, y)| {
                    if y == 0 || x < 0 { None } else { Some(x % y) }
                }),
                OpKind::ArithDivui => crate::passes::dot_to_linalg::const_int(module, a).zip(crate::passes::dot_to_linalg::const_int(module, b)).and_then(|(x, y)| {
                    if y == 0 || x < 0 { None } else { Some(x / y) }
                }),
                OpKind::ArithRemsi => crate::passes::dot_to_linalg::const_int(module, a).zip(crate::passes::dot_to_linalg::const_int(module, b)).and_then(|(x, y)| {
                    if y == 0 { None } else { Some(x.wrapping_rem(y)) }
                }),
                OpKind::ArithIndexCast => crate::passes::dot_to_linalg::const_int(module, a),
                _ => continue,
            };
            let Some(val) = val else { continue };
            let Some(res) = o.results.first().copied() else { continue };
            let ty = o.result_types.first().cloned().expect("folded op type");
            let c = module.fresh_named(&module.hint(res));
            let idx = path.index();
            let block = walk::block_mut(module, &path).expect("path");
            block.remove(idx);
            block.insert(
                idx,
                Op::new(OpKind::ArithConstant)
                    .with_result(c, ty)
                    .with_attr(AttrKey::Value, Attr::Int(val)),
            );
            walk::replace_all_uses(module, res, c);
            folded = true;
        }
        if !folded {
            break;
        }
    }
}

/// A GRID OF N TILES STATES N WINDOW POSITIONS, AND EVERY READER DOWNSTREAM READS CONSTANTS.
///
/// `fold_unit_grid_tile_id` handles N == 1 (landmark -> `arith.constant 0`, then the constant
/// folder materialises every corner). N > 1 leaves the landmark live, every window corner
/// downstream of it computed, and the whole-function door refuses by name: "the access tile
/// feeding this load states a window whose corner is not a constant ... A loop-carried or
/// computed corner needs hoisting or unrolling" (MEASURED on `attention_flash_noncausal` at
/// grid [4,4], 2026-09-28). Hoisting is unavailable for the reason `fold_grid_work_loop`'s
/// multi-trip arm records: no single window above the copies stands for all positions. This
/// pass is the UNROLLING that message names -- one straight-line copy of the whole body per
/// position, the landmark replaced by a constant naming that position -- after which the
/// SAME constant folder materialises every corner, exactly as it does for N == 1.
///
/// ⛔ FIRE GATE. Only a body whose `ktdp.load` tiles state NON-CONSTANT corners is touched.
/// That is precisely -- and only -- the condition the door refuses; the twelve card-verified
/// kernels at multi-tile grids (embedding_granite at 4, the whole-node MLPs) carry their grid
/// position in whole-node store regions and per-core work slices instead, never in a load
/// tile corner, so on them this pass is inert and their pinned emission is not re-rolled.
///
/// ⭐ THE GRID BECOMES 1. After the unroll ONE program covers all N positions on ONE compute
/// tile; leaving N would launch N tiles each computing all N positions (N-times the work and
/// an N-fold collision on every stored window). The door's per-core work-slice addressing is
/// what places the single tile's work.
///
/// ⛔⛔ GRID-PARTITIONED BUFFERS ARE MATERIALIZED PER COPY; SWEPT BUFFERS ARE SHARED.
/// The discriminator is read from the program after per-copy fold: a parameter whose copies
/// state EXACTLY ONE access tile over its view, at a row corner that VARIES with the
/// position, is grid-partitioned -- each position owns a disjoint window (attention's Q and
/// O: one [64, 128] window at `h*256 + m*64`). Such a corner is refused by `base_addressed`
/// on every `rb`-addressed operand downstream, so each copy whose corner is nonzero gets a
/// CLONE of the parameter: a fresh function argument and a fresh
/// `ktdp.construct_memory_view` of the ORIGINAL extent (the emission's windowed form is
/// unchanged -- the window at corner 0 over the full view, the exact shape the card-verified
/// 1-tile run exercised), and the copy's tile rebased to row corner 0. The HOST then stages
/// the position's window file at that clone -- the 16-run card protocol, measured correct on
/// all 16 positions (corr >= 0.99991 each, 2026-09-28).
///
/// A parameter with SEVERAL tiles per copy (attention's K and V: four trip windows at
/// `kvh*256 + t*64`) is SWEPT, not partitioned: no single clone could hold all four windows
/// at corner 0, and the kernel-side windowing arm already carries those corners as offsets
/// -- the mechanism the 1-tile card run exercised for every K-trip. It stays shared: one
/// staging file, the natural full layout.
fn unroll_grid_positions(module: &mut Module) -> Result<()> {
    let Some(f) = module.ops.iter().find(|o| o.kind == OpKind::FuncFunc) else {
        return Ok(());
    };
    let grid: Vec<i64> = match f.attr(&AttrKey::Grid) {
        Some(Attr::IntList(g)) => g.clone(),
        _ => return Ok(()),
    };
    let n = grid.iter().product::<i64>();
    if n <= 1 {
        return Ok(());
    }
    let kernel = module.kernel().map_err(|e| refuse(e.to_string()))?.clone();
    let body: Vec<Op> = kernel.regions[0].ops.clone();
    let Some(landmark_op) = body
        .iter()
        .find(|o| o.kind == OpKind::KtdpGetComputeTileId)
    else {
        return Ok(());
    };
    let Some(landmark) = landmark_op.result() else {
        return Ok(());
    };
    let landmark_ty = landmark_op
        .result_type()
        .cloned()
        .unwrap_or(IrType::Index);

    // ── FIRE GATE ──
    // ── FIRE GATE ──
    // A load whose tile corner is not a constant, at a grid of more than one tile, whose
    // tile feeds a MATMUL INPUT. The corner chain is the grid delinearisation, so a
    // constant here means the grid position never reached the window. And the MATMUL
    // INPUT is what separates the refused case from the card-verified whole-node ones:
    // a computed corner on a pointwise/store operand is carried by the door's
    // whole-node store regions and per-core work-slice addressing (MEASURED: rope_kv8
    // and rope_q32 at multi-tile grids, embedding_granite at 4 -- all card-exact with
    // live landmarks), while `matmul_oriented`'s `base_addressed` REFUSES a nonzero row
    // corner on the A operand (MEASURED: attention at [4,4], the refusal this pass
    // exists to resolve). The gate fires on exactly the latter, so the whole-node
    // bodies keep their pinned emission.
    let fires = body.iter().any(|o| {
        if o.kind != OpKind::KtdpLoad {
            return false;
        }
        let tile = o
            .operands
            .first()
            .and_then(|t| {
                body.iter()
                    .find(|d| d.kind == OpKind::KtdpConstructAccessTile && d.result() == Some(*t))
            })
            .cloned();
        let Some(tile) = tile else { return false };
        let computed = tile
            .operands
            .get(1..3)
            .map(|cs| {
                cs.iter()
                    .any(|c| crate::passes::dot_to_linalg::const_int(module, *c).is_none())
            })
            .unwrap_or(false);
        if !computed {
            return false;
        }
        // The tile feeds a matmul input: some linalg.matmul's operand, transitively
        // through the cast/broadcast chain the conversions build.
        feeds_matmul_input(module, &body, o.result().expect("load result"))
    });
    if !fires {
        return Ok(());
    }

    // ── SHARED vs PER-COPY, BY DATA DEPENDENCE ON THE LANDMARK ──
    //
    // An op whose operands (transitively) avoid the landmark computes the same value on
    // every position -- constants, splat seeds, and the buffer views (whose only operand is
    // a function argument). Everything else is per-copy. `func.return` is a terminator: one
    // program, one return, taken once at the end (the copies' returns are dropped with the
    // per-copy clones; a body with a return value was already refused in `convert_func`).
    let mut depends: HashSet<Ssa> = HashSet::new();
    depends.insert(landmark);
    let mut shared: Vec<Op> = Vec::new();
    let mut per_copy: Vec<Op> = Vec::new();
    let mut terminator: Option<Op> = None;
    // A region's body can read the landmark even when the op's own operands do not (a
    // loop body using it inside), so the dependence test reads operands AND regions,
    // transitively -- otherwise the op lands in `shared` and every copy shares ONE
    // landmark-reading computation, which is position-dependent and wrong on N-1 of them.
    fn reads(ops: &[Op], depends: &HashSet<Ssa>) -> bool {
        for o in ops {
            if o.operands.iter().any(|v| depends.contains(v)) {
                return true;
            }
            for r in &o.regions {
                if reads(&r.ops, depends) {
                    return true;
                }
            }
        }
        false
    }
    for op in body.into_iter() {
        if op.kind == OpKind::KtdpGetComputeTileId {
            continue;
        }
        if op.kind == OpKind::FuncReturn {
            terminator = Some(op);
            continue;
        }
        if reads(std::slice::from_ref(&op), &depends) {
            depends.extend(op.results.iter().copied());
            per_copy.push(op);
        } else {
            shared.push(op);
        }
    }

    // ── THE UNROLL ──
    //
    // Copy 0 KEEPS the original per-copy names for its corner-0 parameters (see the
    // materialization loop: only a nonzero corner gets a clone), so a position whose every
    // partitioned corner is 0 reads exactly like the folded 1-tile body. Copies 1..N-1 are
    // renamed fresh throughout, landmark included.
    let mut flat: Vec<Op> = shared.clone();
    for c in 0..n {
        let mut map: HashMap<Ssa, Ssa> = HashMap::new();
        let mut copy = per_copy.clone();
        {
            let mut defined: Vec<Ssa> = Vec::new();
            defined_values(&copy, &mut defined);
            for v in defined {
                let fresh = module.fresh_named(&module.hint(v));
                map.insert(v, fresh);
            }
        }
        rename_values(&mut copy, &map);
        // THE LANDMARK BECOMES THE POSITION. The landmark op itself never entered
        // `per_copy`, so `map` holds no rename for it and the copy's uses still name the
        // ORIGINAL landmark SSA -- rewrite them (regions included) to the position
        // constant, inserted before the copy it names.
        let cst = module.fresh_named(&module.hint(landmark));
        flat.push(
            Op::new(OpKind::ArithConstant)
                .with_result(cst, landmark_ty.clone())
                .with_attr(AttrKey::Value, Attr::Int(c)),
        );
        replace_operand_everywhere(&mut copy, &[landmark], cst);
        flat.extend(copy);
    }
    if let Some(t) = terminator {
        flat.push(t);
    }
    {
        let kernel = module.kernel_mut().map_err(|e| refuse(e.to_string()))?;
        kernel.regions[0].ops = flat;
        // ONE TILE covers the positions now.
        kernel.set_attr(AttrKey::Grid, Attr::IntList(vec![1]));
    }
    // THE SAME FOLD THE UNIT CASE RUNS, over the copies' constants, so every corner chain
    // materialises. After it, the partitioned-vs-swept question is answered by reading the
    // folded tiles -- and the materialization below runs on a body whose corners are
    // already constants, the form `region_for_operand` reads.
    fold_index_arithmetic(module);

    // ── GRID-PARTITIONED PARAMETERS, FROM THE FOLDED BODY ──
    //
    // Per parameter: its view (one per pointer in every fixture here; several views of one
    // pointer is left shared rather than guessed at), the tiles over that view across the
    // WHOLE unrolled body, and their constant row corners. Exactly N tiles at row corners
    // that vary = partitioned. More than N tiles = swept (K/V: 4 per copy -> 4N total).
    // Fewer than N = a position that does not read the buffer, left shared.
    let kernel = module.kernel().map_err(|e| refuse(e.to_string()))?.clone();
    let body: Vec<Op> = kernel.regions[0].ops.clone();
    let mut view_of: HashMap<Ssa, Ssa> = HashMap::new();
    for o in &body {
        if o.kind == OpKind::KtdpConstructMemoryView {
            if let (Some(v), Some(ptr)) = (o.result(), o.operands.first()) {
                view_of.insert(v, *ptr);
            }
        }
    }
    let mut tiles_of: HashMap<Ssa, Vec<usize>> = HashMap::new();
    for (i, o) in body.iter().enumerate() {
        if o.kind == OpKind::KtdpConstructAccessTile {
            if let Some(v) = o.operands.first() {
                if view_of.contains_key(v) {
                    tiles_of.entry(*v).or_default().push(i);
                }
            }
        }
    }
    let mut materialize: HashSet<Ssa> = HashSet::new();
    for (view, ptr) in &view_of {
        let Some(idxs) = tiles_of.get(view) else { continue };
        if idxs.len() != n as usize {
            continue;
        }
        let corners: Vec<Option<i64>> = idxs
            .iter()
            .map(|&i| {
                body[i]
                    .operands
                    .get(1)
                    .and_then(|s| crate::passes::dot_to_linalg::const_int(module, *s))
            })
            .collect();
        if corners.iter().any(|c| c.is_none()) {
            // A corner that did not fold is left for the door's named guard rather than
            // guessed at here.
            continue;
        }
        let corners: Vec<i64> = corners.into_iter().map(|c| c.unwrap()).collect();
        if corners.iter().any(|c| *c != 0) {
            materialize.insert(*ptr);
        }
    }

    // ── MATERIALIZATION ──
    //
    // One pass over the folded body. A tile over a partitioned view is rebased in place:
    // a ZERO corner keeps the shared view and the original parameter (position 0's Q/O
    // read the original argument -- the 1-tile shape, byte for byte); a nonzero corner
    // gets a fresh argument (the host stages the position's window file there), a clone
    // of the view over that argument at the ORIGINAL extent, and the tile rewritten onto
    // the clone at row corner 0. The load/store over the tile is untouched -- it reads
    // the TILE, whose SSA result the rebased op keeps.
    //
    // The shared declaration of a partitioned view STAYS even when every corner is
    // nonzero (dead but harmless), because the door pairs parameter -> view by scanning
    // for the view over that parameter, and the clones carry fresh arguments, not the
    // original -- deleting the original would leave the parameter's extent unread.
    if !materialize.is_empty() {
        let fn_args: BlockArgs = kernel.regions[0].args.clone();
        let mut new_args: BlockArgs = Vec::new();
        let mut flat: Vec<Op> = Vec::new();
        // ptr -> its ONE view's body index. A pointer with several views (none in the
        // fixtures here) has no single extent to clone from, so it is refused by name
        // rather than guessed at.
        let mut view_idx_of: HashMap<Ssa, usize> = HashMap::new();
        let mut views_of: HashMap<Ssa, Vec<usize>> = HashMap::new();
        for (i, o) in body.iter().enumerate() {
            if o.kind == OpKind::KtdpConstructMemoryView {
                if let Some(ptr) = o.operands.first() {
                    views_of.entry(*ptr).or_default().push(i);
                }
            }
        }
        for (ptr, idxs) in &views_of {
            if idxs.len() == 1 {
                view_idx_of.insert(*ptr, idxs[0]);
            }
        }
        for o in body.iter() {
            let is_part_tile = o.kind == OpKind::KtdpConstructAccessTile
                && o.operands
                    .first()
                    .and_then(|v| view_of.get(v))
                    .is_some_and(|ptr| materialize.contains(ptr))
                && o.operands.len() >= 2;
            let corner = if is_part_tile {
                o.operands
                    .get(1)
                    .and_then(|s| crate::passes::dot_to_linalg::const_int(module, *s))
            } else {
                None
            };
            if !is_part_tile || corner == Some(0) || corner.is_none() {
                // Not partitioned, or position 0's window, or a corner the fold left
                // standing (the door reports that by name; guessing here would be the
                // silent-wrong-answer defect).
                flat.push(o.clone());
                continue;
            }
            let view = *o.operands.first().expect("tile over a view");
            let ptr = *view_of.get(&view).expect("partitioned");
            let view_idx = *view_idx_of.get(&ptr).expect("view of the parameter");
            let arg_ty = fn_args
                .iter()
                .find(|(v, _)| v == &ptr)
                .map(|(_, t)| t.clone())
                .unwrap_or(IrType::Index);
            // THE CLONE: fresh argument, view of the ORIGINAL extent over it.
            let arg = module.fresh_named(&module.hint(ptr));
            new_args.push((arg, arg_ty));
            let mut view_op = body[view_idx].clone();
            let clone_view = module.fresh_named(&module.hint(view));
            view_op.results = vec![clone_view];
            view_op.result_types = vec![body[view_idx]
                .result_types
                .first()
                .cloned()
                .unwrap_or(IrType::Index)];
            view_op.operands = vec![arg];
            flat.push(view_op);
            // THE REBASED TILE: same SSA result (its loads/stores stand), corner 0.
            let zero = module.fresh_named("corner0");
            flat.push(
                Op::new(OpKind::ArithConstant)
                    .with_result(zero, IrType::Index)
                    .with_attr(AttrKey::Value, Attr::Int(0)),
            );
            let mut tile = o.clone();
            tile.operands[0] = clone_view;
            tile.operands[1] = zero;
            flat.push(tile);
        }
        {
            let kernel = module.kernel_mut().map_err(|e| refuse(e.to_string()))?;
            kernel.regions[0].ops = flat;
            kernel.regions[0].args.extend(new_args);
        }
    }
    Ok(())
}

/// Rewrite every operand (regions included, definitions never) that names any of `from` to
/// `to`. The copy loop's landmark special-case: `rename_values` already covered the copy's
/// own definitions, and this catches the landmark's remaining uses.
fn replace_operand_everywhere(ops: &mut [Op], from: &[Ssa], to: Ssa) {
    for o in ops.iter_mut() {
        for v in o.operands.iter_mut() {
            if from.contains(v) {
                *v = to;
            }
        }
        for r in o.regions.iter_mut() {
            replace_operand_everywhere(&mut r.ops, from, to);
        }
    }
}

/// Does `v` (transitively, through the cast/broadcast/expand chain the conversions build)
/// feed an input operand of a `linalg.matmul` in `ops`? The fire gate's discriminator:
/// `matmul_oriented`'s `base_addressed` refuses a computed row corner on a matmul input,
/// while pointwise/store operands ride the door's whole-node addressing -- so the unroll
/// must fire on the one and not the other.
fn feeds_matmul_input(module: &Module, ops: &[Op], v: Ssa) -> bool {
    fn go(module: &Module, ops: &[Op], v: Ssa, depth: usize) -> bool {
        if depth > 16 {
            return false;
        }
        for o in ops {
            // A matmul input, directly or through one intervening conversion op.
            if o.kind == OpKind::LinalgMatmul {
                let mut work: Vec<Ssa> = o.operands.to_vec();
                // Unwrap one layer of cast/broadcast/expand/collapse per step, so a
                // load two hops from the matmul still counts.
                for _ in 0..2 {
                    let mut next: Vec<Ssa> = Vec::new();
                    for w in work {
                        if w == v {
                            return true;
                        }
                        if let Some(d) = ops.iter().find(|d| d.results.contains(&w)) {
                            match d.kind {
                                OpKind::UnrealizedConversionCast
                                | OpKind::ArithIndexCast
                                | OpKind::TensorExpandShape
                                | OpKind::TensorCollapseShape
                                | OpKind::TensorSplat => {
                                    next.extend(d.operands.iter().copied());
                                }
                                _ => {}
                            }
                        }
                    }
                    work = next;
                }
                continue;
            }
            // Recurse into regions (a loop-carried feed still lands in the matmul).
            if !o.regions.is_empty()
                && o.regions
                    .iter()
                    .any(|r| go(module, &r.ops, v, depth + 1))
            {
                return true;
            }
        }
        false
    }
    go(module, ops, v, 0)
}

/// A `linalg.matmul`'s THIRD OPERAND IS AN ACCUMULATOR, AND THE WHOLE-FUNCTION DOOR READS TWO.
///
/// `tl.dot(a, b, acc)` lowers to `linalg.matmul ins(a, b) outs(acc)` -- the loop-carried `g`/`u`/`acc`
/// of a K- or N-blocked MLP arrive there. The door downstream takes `n_in = 2` and never looks at the
/// outs operand, so after the unroll each trip's matmul emits a descriptor writing its OWN fresh
/// intermediate and the trips never meet: MEASURED on the card, `swiglu_mlp_tiled_k` (2 K-trips,
/// 2 N-trips) returned the last K-trip's gate/up in the last N-trip's down-projection and nothing
/// else -- within_2pct 0.0024, the correlation grid showing trip-0 partials and dropped sums.
///
/// The decomposition makes the accumulation VISIBLE as an op the door already lowers 1:1: the matmul
/// keeps its two inputs and a zero splat for outs, and an `arith.addf` sums its result with the old
/// accumulator. `Elementwise(Add)` is an existing `Program`, so the add lowers through the same door
/// with both operands minted intermediates.
///
/// A ZERO outs is left in place: the door ignores operand 3 entirely, so the splat is dead and the
/// rewrite would only mint a dead add. The result SSA keeps the matmul's own name, so every consumer
/// below reads the summed value without a rewiring pass.
fn decompose_matmul_accumulators(module: &mut Module) -> Result<()> {
    // deepest-first so nested-region matmuls (none survive the unroll, but the pass is stated
    // generally) splice into the right block.
    let paths: Vec<OpPath> = walk::paths(module)
        .into_iter()
        .filter(|p| {
            walk::at(module, p)
                .map(|o| o.kind == OpKind::LinalgMatmul && o.operands.len() == 3)
                .unwrap_or(false)
        })
        .collect();
    for path in paths.into_iter().rev() {
        let mm = walk::at(module, &path).expect("collected path").clone();
        let acc = mm.operands[2];
        // A zero splat (or a zero scalar constant) is the `tl.dot` default init: the matmul
        // stands alone and the door's two-input reading is already the whole computation.
        if crate::passes::dot_to_linalg::is_zero_const(module, acc)
            || module.def_of(acc).is_some_and(|d| d.kind == OpKind::TensorSplat
                && d.operands.first().is_some_and(|s|
                    crate::passes::dot_to_linalg::is_zero_const(module, *s)))
        {
            continue;
        }
        let Some(res) = mm.results.first().copied() else { continue };
        let res_ty = mm.result_types.first().cloned();
        let Some(res_ty) = res_ty else {
            return Err(refuse("a `linalg.matmul` with an accumulator states no result type, so the zero \
                 splat that replaces the accumulator cannot be shaped".to_string()));
        };
        let elem = match &res_ty {
            IrType::Tensor { elem, .. } => *elem,
            _ => {
                return Err(refuse(
                    "a `linalg.matmul` with an accumulator has a non-tensor result type",
                ))
            }
        };
        let idx = path.index();
        // 1. zero scalar + splat, shaped like the result. Names minted before the block borrow.
        let hint = module.hint(res);
        let zero_s = module.fresh_named(&hint);
        let zero_t = module.fresh_named(&hint);
        // 2. the matmul's own result moves to a fresh name...
        let dot = module.fresh_named(&hint);
        let block = walk::block_mut(module, &path).expect("path");
        {
            let op = &mut block[idx];
            op.operands[2] = zero_t;
            op.results[0] = dot;
            op.result_types[0] = res_ty.clone();
        }
        block.insert(
            idx,
            Op::new(OpKind::ArithConstant)
                .with_result(zero_s, IrType::Scalar(elem))
                .with_attr(AttrKey::Value, Attr::Float(FloatBits::f32(0.0))),
        );
        block.insert(
            idx + 1,
            Op::new(OpKind::TensorSplat)
                .with_result(zero_t, res_ty.clone())
                .with_operands([zero_s]),
        );
        // 3. ...and `arith.addf(dot, acc)` defines the name every consumer already reads.
        block.insert(
            idx + 3,
            Op::new(OpKind::ArithAddf)
                .with_result(res, res_ty)
                .with_operands([dot, acc]),
        );
    }
    Ok(())
}

/// Fold the FIRST TRIP of a flash recurrence into the straight-line form FA2 itself
/// compiles for its first block, so the seeds never reach a region reader as operands.
///
/// # WHY THIS PASS EXISTS
///
/// The flash recurrence's carries are seeded with `tensor.splat`s -- `m_i` at `-inf`,
/// `l_i` at 1.0, `acc` at 0 -- and after [`unroll_constant_trip_loops`] those seeds are
/// OPERANDS of trip 1's pointwise ops (`max(m_seed, m)`, `exp2(m_seed - m)`, `acc·alpha`).
/// The whole-function door refuses a splat operand by name (`whole_function`'s "neither a
/// load of a parameter nor an intermediate this function produced"), and it is RIGHT to:
/// a splat reaching a compute op would have to become a caller-seeded buffer, and the C++
/// path's `materializeSplatInputs` records the value in an attribute the launcher seeds
/// at runtime -- one more bind a launch can silently skip. The seeds are COMPILE-TIME
/// CONSTANTS, so the better answer is to not lower them at all.
///
/// # THE FOLD SET, AND WHY EACH RULE IS EXACT
///
/// **Identity folds** (`maxnumf`/`-inf`, `addf`/`+0`, `subf`/`-0`, `mulf`/`1`): IEEE
/// identities, exact including signed zeros (`0 + -0 = +0` is the one clause that is not
/// bitwise exact, and it cannot fire here: the seeds are the LEFT operand of their ops
/// and `x - 0` / `1·x` / `max(-inf, x)` are exact for every x). `mulf`/`1` is what trip
/// 1's `l_i · alpha` reduces to (`l_i` seeds at 1.0, the fixture's own divide-by-zero
/// guard).
///
/// **Splat × splat pointwise** (`addf`/`subf`/`mulf`/`divf`/`maxnumf`/`minnumf` over two
/// splat-of-constant operands): folded in f64 and re-rounded through
/// [`FloatBits::to_f16`], which is the SAME round-to-nearest-even the producer used when
/// it built the f16 constant, so the folded bits are what the original chain computed --
/// f16's own arithmetic is emulated exactly by f64 for +, -, · (no double rounding at
/// f16 precision).
///
/// **`math.exp` of a NON-FINITE-or-zero splat**: `exp(-inf) = 0`, `exp(+inf) = +inf`,
/// `exp(0) = 1` are exact for every implementation, including the device's polynomial.
/// A FINITE splat is NEVER folded: the device transcendental is its own approximation
/// and the reference is torch's, so folding would change the number.
///
/// **The alpha seed** -- `subf(-inf-splat, x)` -> `-inf-splat` -- is the one rule that is
/// NOT an identity, and it is guarded to the one chain where it is exact: every use of
/// the difference must be a `mulf` by a POSITIVE splat whose own result feeds only
/// `math.exp`. That is `exp2(m_old - m_new)` with the base change the frontend emits
/// (`exp(y·ln2)`), and there `-inf - m` is `-inf` for every m the fixture can produce:
/// the fixture's mask values are -1.0e4 rather than -inf (its own delta 3, so a fully
/// masked block still leaves `m_ij` FINITE), and a NaN score means every consumer of m
/// is NaN in the original too. Unguarded the rule would be unsound (`-inf - -inf` = NaN),
/// so the guard is the rule's soundness proof and not a pattern-match convenience.
///
/// **Rank plumbing of a splat** (`expand_shape`/`collapse_shape`/`linalg.broadcast` over
/// a splat): a splat is shape-polymorphic -- the value is the whole fact -- so the chain
/// folds to a splat at the chain's final shape. Trip 1's `alpha[:, None]` spray is the
/// instance (the frontend widens `m` to `[64, 64]` for the broadcast multiply).
///
/// # WHAT TRIP 1 BECOMES
///
/// `max(-inf, m) -> m`, `exp2(-inf - m) -> 0`, `1·alpha -> alpha -> 0`, `acc·0 -> 0`,
/// so `addf(matmul(p, v, 0-splat), 0-splat)` -- trip 1's `p·V + 0·acc` collapses to the
/// two-input matmul [`decompose_matmul_accumulators`] already leaves alone, `l = 0 +
/// sum(p)`, and `m = rowmax·scale`. That IS the first-block fast path every FA2 kernel
/// writes by hand (`first`-flag or `SKIP_FIRST`), reached by folding rather than by
/// asking the kernel author to spell it.
///
/// # ORDER
///
/// Runs AFTER [`unroll_constant_trip_loops`] (the seeds only become top-level operands
/// then) and BEFORE [`decompose_matmul_accumulators`] (so trip 1's `acc·alpha` folds to
/// the zero splat that pass already treats as `tl.dot`'s default init). Iterates to a
/// fixed point: `exp(0·ln2)` needs `0·ln2` folded before `exp` can see a splat operand.
fn fold_splat_seeds(module: &mut Module) {
    loop {
        let mut changed = false;
        // ONE BLOCK AT A TIME, and the paths are RECOMPUTED inside the loop over blocks
        // whenever one folds: a splice shifts indices only within its own block, but the
        // discipline of "collect against a snapshot, apply, re-walk" is the one
        // `walk`'s own header demands, and recomposing is cheaper than reasoning about
        // which shifts are safe.
        'blocks: loop {
            let paths = walk::paths(module);
            // A block is identified by its FIRST op's path prefix; a block with no
            // foldable op is skipped by the collect below returning empty.
            let mut seen: HashSet<Vec<(usize, usize)>> = HashSet::new();
            for path in &paths {
                let key = path.0[..path.0.len() - 1].to_vec();
                if !seen.insert(key.clone()) {
                    continue;
                }
                // COLLECT against the immutable module -- every `def_of`/`type_of` query
                // happens here, before any borrow of a block.
                let folds = collect_splat_folds(module, path);
                if folds.is_empty() {
                    continue;
                }
                apply_splat_folds(module, path, folds);
                changed = true;
                continue 'blocks;
            }
            break;
        }
        if !changed {
            return;
        }
    }
}

/// The float value of `v` iff it is `tensor.splat` of an `arith.constant` float.
fn splat_value(module: &Module, v: Ssa) -> Option<FloatBits> {
    let splat = module.def_of(v).filter(|o| o.kind == OpKind::TensorSplat)?;
    let scalar = splat.operands.first().copied()?;
    let konst = module
        .def_of(scalar)
        .filter(|o| o.kind == OpKind::ArithConstant)?;
    konst.attr(&AttrKey::Value).and_then(|a| a.as_float())
}

/// One fold collected against a block index.
enum Fold {
    /// Every use of the result reads `to` instead; the op is dropped.
    Forward(Ssa),
    /// The result is the source splat's value, re-splatted at the result's own type.
    ToSplat(Ssa),
    /// The result is a splat of this computed value.
    Const(f64),
}

/// The folds for the block holding `path`'s op, computed against the unmutated module.
fn collect_splat_folds(module: &Module, path: &OpPath) -> Vec<(usize, Fold)> {
    let block = match walk::block_ref(module, path) {
        Some(b) => b,
        None => return Vec::new(),
    };
    let mut folds: Vec<(usize, Fold)> = Vec::new();
    for (i, op) in block.iter().enumerate() {
        let Some(res) = op.results.first().copied() else { continue };
        let (a, b) = (op.operands.first().copied(), op.operands.get(1).copied());
        match op.kind {
            // ── the identity folds. The seed is the LEFT operand of every flash chain
            //    (`max(m_seed, m)`, `exp2(m_seed - m)`, `l_seed · alpha`), so the left
            //    position is checked first and the right position is the same rule.
            OpKind::ArithMaxnumf => {
                let (Some(a), Some(b)) = (a, b) else { continue };
                if splat_value(module, a).is_some_and(|f| f.as_f64() == f64::NEG_INFINITY) {
                    folds.push((i, Fold::Forward(b)));
                } else if splat_value(module, b).is_some_and(|f| f.as_f64() == f64::NEG_INFINITY) {
                    folds.push((i, Fold::Forward(a)));
                }
            }
            OpKind::ArithAddf => {
                let (Some(a), Some(b)) = (a, b) else { continue };
                if splat_value(module, a).is_some_and(|f| f.as_f64() == 0.0) {
                    folds.push((i, Fold::Forward(b)));
                } else if splat_value(module, b).is_some_and(|f| f.as_f64() == 0.0) {
                    folds.push((i, Fold::Forward(a)));
                }
            }
            OpKind::ArithSubf => {
                let (Some(a), Some(b)) = (a, b) else { continue };
                if splat_value(module, b).is_some_and(|f| f.as_f64() == 0.0) {
                    folds.push((i, Fold::Forward(a)));
                }
                // ⛔ THE ALPHA SEED, guarded -- see the pass doc. `-inf - m` folds to
                // `-inf` ONLY inside `exp2`, i.e. only when every use is a multiply by a
                // positive splat whose own result feeds only `math.exp`. Outside that
                // chain `-inf - -inf` = NaN and the fold would be a silent wrong answer.
                else if splat_value(module, a).is_some_and(|f| f.as_f64() == f64::NEG_INFINITY)
                    && sub_feeds_only_exp2(module, res)
                {
                    folds.push((i, Fold::ToSplat(a)));
                }
            }
            OpKind::ArithMulf => {
                let (Some(a), Some(b)) = (a, b) else { continue };
                if splat_value(module, a).is_some_and(|f| f.as_f64() == 1.0) {
                    folds.push((i, Fold::Forward(b)));
                } else if splat_value(module, b).is_some_and(|f| f.as_f64() == 1.0) {
                    folds.push((i, Fold::Forward(a)));
                }
                // ── splat × splat: fold in f64, re-round to the element type.
                else if let (Some(x), Some(y)) = (splat_value(module, a), splat_value(module, b)) {
                    folds.push((i, Fold::Const(x.as_f64() * y.as_f64())));
                }
            }
            OpKind::ArithDivf => {
                let (Some(a), Some(b)) = (a, b) else { continue };
                if let (Some(x), Some(y)) = (splat_value(module, a), splat_value(module, b)) {
                    if y.as_f64() != 0.0 {
                        folds.push((i, Fold::Const(x.as_f64() / y.as_f64())));
                    }
                }
            }
            OpKind::ArithMinnumf => {
                let (Some(a), Some(b)) = (a, b) else { continue };
                if let (Some(x), Some(y)) = (splat_value(module, a), splat_value(module, b)) {
                    folds.push((i, Fold::Const(x.as_f64().min(y.as_f64()))));
                }
            }
            // ── `exp`/`exp2` of a splat: exact only for the non-finite and zero anchors.
            OpKind::MathExp | OpKind::MathExp2 => {
                let Some(a) = a else { continue };
                if let Some(x) = splat_value(module, a) {
                    let v = x.as_f64();
                    let folded = if v == f64::NEG_INFINITY {
                        Some(0.0)
                    } else if v == f64::INFINITY {
                        Some(f64::INFINITY)
                    } else if v == 0.0 {
                        Some(1.0)
                    } else {
                        // A FINITE nonzero splat is NEVER folded: the device
                        // transcendental is its own approximation and the reference is
                        // torch's, so folding would change the number.
                        None
                    };
                    if let Some(c) = folded {
                        folds.push((i, Fold::Const(c)));
                    }
                }
            }
            // ── rank plumbing over a splat: a splat is shape-polymorphic -- the value
            //    is the whole fact -- so the chain folds to the chain's final shape.
            //    The broadcast at this stage is `convert_broadcast`'s spelling: a
            //    yield-only `linalg.generic` over the collapsed source.
            OpKind::TensorExpandShape | OpKind::TensorCollapseShape => {
                let Some(a) = a else { continue };
                if splat_value(module, a).is_some() {
                    folds.push((i, Fold::ToSplat(a)));
                }
            }
            OpKind::LinalgGeneric => {
                let Some(a) = a else { continue };
                // `convert_broadcast`'s spelling: a yield-only body over the collapsed
                // source, with a `tensor.empty` init. A splat flowing through it is the
                // value being sprayed -- fold to the result's shape.
                let is_bare_broadcast = op.operands.len() == 2
                    && op.regions.len() == 1
                    && op.regions[0].ops.len() == 1
                    && op.regions[0].ops[0].kind == OpKind::LinalgYield
                    && op.regions[0].ops[0].operands.len() == 1
                    && op.regions[0].ops[0].operands[0] == op.regions[0].args[0].0;
                if is_bare_broadcast && splat_value(module, a).is_some() {
                    folds.push((i, Fold::ToSplat(a)));
                }
            }
            _ => {}
        }
    }
    folds
}

/// Apply the collected folds to the block holding `path`'s op.
///
/// REVERSE index order, so earlier indices stay valid across the splices, and the
/// Forward targets are RESOLVED TRANSITIVELY first: `a -> b` composed with `b -> c` in
/// one batch must rewire `a`'s uses to `c`, or they would point at the op `b`'s fold
/// just removed.
fn apply_splat_folds(module: &mut Module, path: &OpPath, folds: Vec<(usize, Fold)>) {
    // Resolve Forward chains to their final target.
    let forward: HashMap<Ssa, Ssa> = folds
        .iter()
        .filter_map(|(i, f)| match f {
            Fold::Forward(to) => Some((res_at(module, path, *i), *to)),
            _ => None,
        })
        .collect();
    let resolve = |mut v: Ssa| -> Ssa {
        while let Some(next) = forward.get(&v) {
            v = *next;
        }
        v
    };
    // EVERY query and every fresh name is minted BEFORE the block borrow -- `def_of`,
    // `type_of` and `fresh_named` all take `&Module`/`&mut Module` and the block is a
    // projection of the same module.
    struct Planned {
        i: usize,
        res: Ssa,
        res_ty: IrType,
        action: Plan,
    }
    enum Plan {
        Drop { to: Ssa },
        Splat { konst: Ssa, elem: DType, val: FloatBits },
    }
    let mut planned: Vec<Planned> = Vec::new();
    for (i, fold) in folds.into_iter().rev() {
        let Some(op) = walk::at(module, path) else { continue };
        let _ = op;
        // The op's index within the block may have shifted for earlier splices in this
        // batch -- but the batch was collected in one snapshot and applied in reverse,
        // so index `i` is the index the collect saw for every op still ahead of it in
        // the block. Re-resolve the op by its RESULT rather than trusting the index.
        let target = OpPath({
            let mut full = path.0.clone();
            full.last_mut().expect("nonempty").1 = i;
            full
        });
        let Some(op) = walk::at(module, &target) else { continue };
        let Some(res) = op.results.first().copied() else { continue };
        let Some(res_ty) = op.result_types.first().cloned() else { continue };
        let elem = match &res_ty {
            IrType::Tensor { elem, .. } => *elem,
            _ => DType::F32,
        };
        let hint = module.hint(res);
        match fold {
            Fold::Forward(to) => planned.push(Planned {
                i,
                res,
                res_ty,
                action: Plan::Drop { to: resolve(to) },
            }),
            Fold::ToSplat(src) => {
                let val = splat_value(module, src).expect("collected while immutable");
                let konst = module.fresh_named(&hint);
                planned.push(Planned { i, res, res_ty, action: Plan::Splat { konst, elem, val } });
            }
            Fold::Const(v) => {
                let val = if elem == DType::F16 {
                    FloatBits::f16_from_f32(v as f32)
                } else {
                    FloatBits::f32(v as f32)
                };
                let konst = module.fresh_named(&hint);
                planned.push(Planned { i, res, res_ty, action: Plan::Splat { konst, elem, val } });
            }
        }
    }
    for p in planned {
        let Some(block) = walk::block_mut(module, path) else { continue };
        if p.i >= block.len() {
            continue;
        }
        match p.action {
            Plan::Drop { to } => {
                // The rewire runs over the WHOLE module (uses may be in other blocks),
                // so it borrows the module after the block borrow ends: collect the
                // removal, drop the block borrow, then rewire.
                block.remove(p.i);
                walk::replace_all_uses(module, p.res, to);
            }
            Plan::Splat { konst, elem, val } => {
                block[p.i] = Op::new(OpKind::ArithConstant)
                    .with_result(konst, IrType::Scalar(elem))
                    .with_attr(AttrKey::Value, Attr::Float(val));
                block.insert(
                    p.i + 1,
                    Op::new(OpKind::TensorSplat)
                        .with_result(p.res, p.res_ty)
                        .with_operands([konst]),
                );
            }
        }
    }
}

/// The first result of the op at `path`'s block index `i`, for Forward-chain resolution.
fn res_at(module: &Module, path: &OpPath, i: usize) -> Ssa {
    let mut full = path.0.clone();
    full.last_mut().expect("nonempty").1 = i;
    walk::at(module, &OpPath(full))
        .and_then(|o| o.results.first().copied())
        .expect("collected against this index")
}

/// True iff every use of `sub`'s result is a `mulf` by a positive splat whose own
/// result feeds only `math.exp` -- the `exp2(m_old - m_new)` chain, the one context in
/// which `-inf - m` is provably `-inf`.
/// True iff every use of `sub`'s result is the `exp2(m_old - m_new)` chain -- either
/// `math.exp2` directly (the spelling at THIS stage; the `exp(x·ln2)` rewrite happens
/// later) or a multiply by a positive splat whose own result feeds only `math.exp`.
/// That is the one context in which `-inf - m` is provably `-inf` (and `exp2(-inf)`=0,
/// exact for any implementation).
fn sub_feeds_only_exp2(module: &Module, sub: Ssa) -> bool {
    let uses: Vec<&Op> = module
        .ops_deep()
        .into_iter()
        .filter(|o| o.operands.contains(&sub))
        .collect();
    if uses.is_empty() {
        return false;
    }
    for u in uses {
        let exp_like = |o: &Op| o.kind == OpKind::MathExp || o.kind == OpKind::MathExp2;
        if exp_like(u) {
            continue;
        }
        if u.kind != OpKind::ArithMulf {
            return false;
        }
        let other = if u.operands[0] == sub { u.operands[1] } else { u.operands[0] };
        let Some(s) = splat_value(module, other) else { return false };
        if !(s.as_f64() > 0.0) {
            return false;
        }
        let mul_res = u.results.first().copied().expect("mulf has a result");
        let mut mul_uses = module
            .ops_deep()
            .into_iter()
            .filter(|o| o.operands.contains(&mul_res));
        let Some(exp) = mul_uses.next() else { return false };
        if mul_uses.next().is_some() {
            return false;
        }
        if !exp_like(exp) {
            return false;
        }
    }
    true
}

/// Every value `ops` DEFINES, regions included: op results and each nested region's block
/// arguments. What an unrolled copy has to rename.
///
/// `Op::results` is a LIST (see `ir::Op`'s note), so a five-result `scf.for` contributes all
/// five; and a region's block arguments are definitions too, which is why they are collected
/// here rather than only at the top level.
fn defined_values(ops: &[Op], out: &mut Vec<Ssa>) {
    for o in ops {
        out.extend(o.results.iter().copied());
        for r in &o.regions {
            out.extend(r.args.iter().map(|(v, _)| *v));
            defined_values(&r.ops, out);
        }
    }
}

/// Substitute `map` through `ops` -- operands, results and region block arguments, recursively.
///
/// USES AND DEFINITIONS BOTH, deliberately. [`walk::replace_all_uses`] rewires uses and leaves
/// definitions alone, which is what a rewire wants; a COPY needs its definitions renamed too, or
/// the copy redefines the original's names.
///
/// No attribute carries an SSA reference in this IR ([`Attr`] has no value-typed variant), so
/// operands, results and block arguments are the whole of it.
fn rename_values(ops: &mut [Op], map: &HashMap<Ssa, Ssa>) {
    for o in ops.iter_mut() {
        for v in o.operands.iter_mut() {
            if let Some(n) = map.get(v) {
                *v = *n;
            }
        }
        for v in o.results.iter_mut() {
            if let Some(n) = map.get(v) {
                *v = *n;
            }
        }
        for r in o.regions.iter_mut() {
            for (v, _) in r.args.iter_mut() {
                if let Some(n) = map.get(v) {
                    *v = *n;
                }
            }
            rename_values(&mut r.ops, map);
        }
    }
}

/// THE WORK LOOP IS NOT WHERE A BUFFER'S WIDTH IS DECLARED.
///
/// `DistributeWork` step 5 sweeps EVERY op after the `tt.get_program_id` anchor into the
/// work loop's body, and `ConvertTTIRToKTDP` builds each parameter's
/// `ktdp.construct_memory_view` where the `tt.make_tensor_descriptor` was -- which in
/// every fixture here is after that anchor. When the loop then folds away the views land
/// back at the top level; when it is genuinely multi-trip they do not, and the whole
/// declaration of how wide each buffer is ends up inside a region.
///
/// THAT IS A DEFECT IN WHAT WE HAND OVER, MEASURED. `ktir_superdsc`'s
/// `emit::lower_ktir_to_superdsc::regions` pairs parameter `i` to its extent by a
/// NON-RECURSIVE `find` over `IRFunction::operations` for the one
/// `ktdp.construct_memory_view` whose address operand is that parameter. Nested, there is
/// none, and it refuses: `rope_fwd: parameter 0 (t0) states no
/// `ktdp.construct_memory_view``. Same kernel at grid 4,8 (32 items over 32 cores, folded)
/// walked fine; at grid 4,32 (128 over 32, kept) it did not. The trip count is not a fact
/// about how wide a buffer is.
///
/// WHY HOISTING IS EXACT AND NOT A GUESS. A view's operand list is the base address alone
/// and its extent/strides/space are constant ATTRIBUTES, so it is loop-invariant by
/// construction; it is pure, so it has no ordering obligation; and its result is consumed
/// only by `ktdp.construct_access_tile`, which stays where it is and still sees a
/// dominating definition. The guard is nonetheless spelled out rather than assumed: a view
/// moves ONLY if every one of its operands is already defined above the loop (a function
/// parameter, or the result of a top-level op preceding it). One built from the work index
/// -- a runtime extent -- is not hoistable and is LEFT ALONE, where `handoff::lower`'s
/// per-parameter guard then refuses it by name instead of this pass moving it past its own
/// operands.
///
/// It does NOT invent anything: the view moved is the program's own, with the extent the
/// program stated. A synthetic view of assumed width is the failure mode this exists to
/// avoid, not a fallback.
fn hoist_buffer_views_out_of_work_loops(module: &mut Module) {
    let Ok(func) = module.kernel() else { return };
    if func.regions.is_empty() {
        return;
    }
    // Every parameter, plus every value any top-level op defines: hoisting is only
    // allowed onto names from this set, and it is recomputed per loop from the block as
    // it stands. Conservative on purpose -- a name it fails to include costs a hoist,
    // never correctness.
    let outer_at = |block: &[Op], args: &[(Ssa, IrType)], upto: usize| -> HashSet<Ssa> {
        let mut s: HashSet<Ssa> = args.iter().map(|(v, _)| *v).collect();
        // `Op::results` is a LIST here (see `ir::Op`'s note), so a five-result `scf.for`
        // contributes all five and there is no attribute side-channel to read.
        for op in block.iter().take(upto) {
            s.extend(op.results.iter().copied());
        }
        s
    };

    loop {
        let snapshot = module.kernel().expect("checked above").clone();
        let block = &snapshot.regions[0].ops;
        let args = &snapshot.regions[0].args;
        let Some((li, victims)) = block.iter().enumerate().find_map(|(li, op)| {
            if !crate::passes::distribute_work::is_per_core_work_loop(module, op) {
                return None;
            }
            let outer = outer_at(block, args, li);
            let victims: Vec<usize> = op.regions[0]
                .ops
                .iter()
                .enumerate()
                .filter(|(_, b)| {
                    b.kind == OpKind::KtdpConstructMemoryView
                        && b.operands.iter().all(|v| outer.contains(v))
                })
                .map(|(i, _)| i)
                .collect();
            (!victims.is_empty()).then_some((li, victims))
        }) else {
            return;
        };

        let fi = module
            .ops
            .iter()
            .position(|o| o.kind == OpKind::FuncFunc || o.kind == OpKind::TtFunc)
            .expect("the kernel is in this module");
        let block = &mut module.ops[fi].regions[0].ops;
        // Removed HIGHEST INDEX FIRST, so each removal leaves the lower indices valid.
        // `moved` is therefore in reverse body order, and inserting each at `li` in that
        // order re-reverses it -- so the views keep their original relative order above the
        // loop, which is what makes the multi-trip module comparable to the folded one.
        let mut moved: Vec<Op> = Vec::with_capacity(victims.len());
        for i in victims.into_iter().rev() {
            moved.push(block[li].regions[0].ops.remove(i));
        }
        for op in moved {
            block.insert(li, op);
        }
    }
}

/// Drop Triton's discardable attributes. They are alignment/divisibility HINTS the
/// scheduler reads none of.
///
/// OUR OWN `spyre.` NAMESPACE SURVIVES. It is not a dialect the scheduler registers --
/// it is the CALLER CONTRACT this backend records in the artifact
/// (`spyre.folded_grid_loop` here), and an obligation this pass silently deleted would
/// be an obligation nobody is told about.
fn strip_triton_attrs(module: &mut Module) {
    walk::for_each_mut(module, |op| {
        op.attrs.retain(|(k, _)| !k.spelling().starts_with("tt."));
    });
    module.attrs.retain(|(k, _)| !k.spelling().starts_with("tt."));
}

/// FAIL CLOSED on any op outside the scheduler's dialect list.
fn verify_only_scheduler_dialects(module: &Module) -> Result<()> {
    for op in module.ops_deep() {
        let s = op.kind.spelling();
        let ns = s.split('.').next().unwrap_or(s);
        if !is_scheduler_dialect(ns) {
            return Err(refuse(format!(
                "'{s}' is in dialect '{ns}', which the scheduler does not register. \
                 Available dialects: affine, agen, arith, builtin, dataflow, explan, \
                 func, init, ktdf, ktdf_arch, ktdf_lowering, ktdp, linalg, math, memref, \
                 scf, sdscbundle, sentient, symbol, tensor, trace, uniform, varexpr, \
                 vector, vectorchain. A conversion that leaves one op behind produces a \
                 module that LOOKS converted and still cannot be parsed"
            )));
        }
    }
    Ok(())
}


/// THE KTDP -> KTIR STAGE, WHOLE: rewrite, then state the result as `ktir_core::ir::IRFunction`.
///
/// # WHY THIS EXISTS AND `run` + a separate converter DOES NOT
///
/// The pipeline is `triton -> ttir -> KTDP -> KTIR -> SuperDSC`, five stages each of which is its
/// own thing, all named by IBM (`lib/Dialect/KTDP`, `lib/Dialect/KTDF`). This is the third arrow.
///
/// It used to be two calls: [`run`], which rewrote the KTDP module in place, and a top-level
/// `handoff` module that then mapped that module into `ktir_core`. That made the tree look like it
/// held TWO FORMS OF KTIR with a converter between them, and I described it that way for a long
/// time. It never did: the in-place form is the KTDP IR (`ktdp.*` ops mid-conversion, exactly what
/// the C++'s own `make_ktir` goldens contain at this point) and the emitted form is KTIR. One
/// stage, one output type.
///
/// `run` remains for the tests that inspect the rewrites op by op; anything downstream of this
/// stage calls THIS, and gets `ktir_core` or a refusal.
pub fn lower<'a>(
    module: &mut Module,
    grid: &[i64],
    arena: &'a ktir_core::arena::Arena,
) -> Result<ktir_core::ir::IRFunction<'a>> {
    run(module, grid)?;
    super::to_ktir_emit::lower(module, arena)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::parse;

    /// THE FLASH ALPHA CHAIN, exactly as the unrolled 1-tile kernel spells it: the m seed
    /// `-inf` reaching `subf`, whose result feeds `mulf`-by-ln2 and then `math.exp`. The
    /// whole chain must fold to `splat(0)` -- trip 1's first-block fast path.
    const ALPHA: &str = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %neg = arith.constant -inf : f32
    %mseed = tensor.splat %neg : tensor<64xf16>
    %m = arith.constant 1.0 : f32
    %msplat = tensor.splat %m : tensor<64xf16>
    %diff = arith.subf %mseed, %msplat : tensor<64xf16>
    %ln2 = arith.constant 6.93147180e-01 : f32
    %ln2s = tensor.splat %ln2 : tensor<64xf16>
    %scaled = arith.mulf %diff, %ln2s : tensor<64xf16>
    %alpha = math.exp %scaled : tensor<64xf16>
    %zero = arith.constant 0.0 : f32
    %zseed = tensor.splat %zero : tensor<64xf16>
    %one = arith.constant 1.0 : f32
    %lseed = tensor.splat %one : tensor<64xf16>
    %l = arith.mulf %lseed, %alpha : tensor<64xf16>
    %l2 = arith.addf %l, %zseed : tensor<64xf16>
    tt.return
  }
}
";

    #[test]
    fn the_flash_alpha_seed_folds_to_the_first_block_fast_path() {
        let m = run_on(ALPHA, &[1]).unwrap();
        // The whole seed chain collapses: no subf, no exp, and the mulf/addf on
        // the chain are gone too (1.0*alpha forwards to alpha, +0 forwards).
        let ops = m.ops_deep();
        for kind in [
            OpKind::ArithSubf,
            OpKind::ArithAddf,
            OpKind::ArithMulf,
            OpKind::MathExp,
        ] {
            assert!(
                !ops.iter().any(|o| o.kind == kind),
                "{:?} survived the seed fold",
                kind
            );
        }
        // What survives is a splat(0): the folded exp(-inf) value.
        let zero_splat = ops.iter().any(|o| {
            o.kind == OpKind::TensorSplat
                && o.operands.first().is_some_and(|s| {
                    m.def_of(*s)
                        .is_some_and(|d| {
                            d.attr(&AttrKey::Value)
                                .and_then(|a| a.as_float())
                                .is_some_and(|f| f.as_f64() == 0.0)
                        })
                })
        });
        assert!(zero_splat, "a splat(0) survives as trip 1's fast-path seed");
    }

    fn run_on(src: &str, grid: &[i64]) -> Result<Module> {
        let mut m = parse::parse(src).unwrap();
        run(&mut m, grid)?;
        Ok(m)
    }

    const REDUCE: &str = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %x = arith.constant dense<0.000000e+00> : tensor<64x64xf16>
    %m = \"tt.reduce\"(%x) <{axis = 1 : i32}> ({
    ^bb0(%a: f16, %b: f16):
      %c = arith.maxnumf %a, %b : f16
      tt.reduce.return %c : f16
    }) : (tensor<64x64xf16>) -> tensor<64xf16>
    tt.return
  }
}
";

    #[test]
    fn the_grid_is_required_and_lands_flat_on_the_function() {
        let e = run_on(REDUCE, &[]).unwrap_err();
        assert!(e.message.contains("the launch grid is required"), "got {e}");
        // 2 x 4 delinearized upstream -> the FLAT extent 8 is the faithful form.
        let m = run_on(REDUCE, &[2, 4]).unwrap();
        let f = m.ops.iter().find(|o| o.kind == OpKind::FuncFunc).expect("func.func");
        assert_eq!(f.attr(&AttrKey::Grid), Some(&Attr::IntList(vec![8])));
    }

    #[test]
    fn a_pointer_argument_is_retyped_to_index() {
        let m = run_on(REDUCE, &[8]).unwrap();
        let f = m.ops.iter().find(|o| o.kind == OpKind::FuncFunc).unwrap();
        assert_eq!(f.regions[0].args[0].1, IrType::Index);
    }

    #[test]
    fn a_reduce_becomes_a_generic_with_a_reduction_iterator_and_the_right_identity() {
        let m = run_on(REDUCE, &[8]).unwrap();
        let g = m
            .ops_deep()
            .into_iter()
            .find(|o| o.kind == OpKind::LinalgGeneric)
            .expect("the reduce became a linalg.generic");
        // axis 1 of a rank-2 source -> [parallel, reduction].
        assert_eq!(
            g.attr(&AttrKey::IteratorTypes),
            Some(&Attr::StrList(vec!["parallel".into(), "reduction".into()]))
        );
        assert_eq!(
            g.attr(&AttrKey::IndexingMaps),
            Some(&Attr::AffineMapList(vec![
                "(d0, d1) -> (d0, d1)".into(),
                "(d0, d1) -> (d0)".into()
            ]))
        );
        // The combiner is INLINED, and the body ends in a linalg.yield.
        let kinds: Vec<&str> = g.regions[0].ops.iter().map(|o| o.kind.spelling()).collect();
        assert_eq!(kinds, vec!["arith.maxnumf", "linalg.yield"]);
        // THE IDENTITY: a max seeds with -inf. Seeding with 0 would silently clamp
        // every negative score to zero, which is the class of bug the C++ refuses to
        // risk by fabricating a seed.
        let seed = m
            .ops_deep()
            .into_iter()
            .find(|o| o.kind == OpKind::ArithConstant && o.result_type() == Some(&IrType::Scalar(DType::F16)))
            .expect("the identity constant");
        assert_eq!(
            seed.attr(&AttrKey::Value).and_then(|a| a.as_float()).map(|f| f.bits),
            Some(0xFC00),
            "-inf is the identity of max"
        );
    }

    #[test]
    fn a_combiner_with_no_known_identity_is_refused_rather_than_seeded() {
        let src = REDUCE.replace("arith.maxnumf", "arith.divf");
        let e = run_on(&src, &[8]).unwrap_err();
        assert!(e.message.contains("has no known identity element"), "got {e}");
        assert!(e.message.contains("fabricating a seed"), "got {e}");
    }

    #[test]
    fn a_broadcast_becomes_a_yield_only_generic_and_never_the_named_op() {
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %x = arith.constant dense<0.000000e+00> : tensor<64x1xf16>
    %b = tt.broadcast %x : tensor<64x1xf16> -> tensor<64x128xf16>
    tt.return
  }
}
";
        let m = run_on(src, &[8]).unwrap();
        let g = m
            .ops_deep()
            .into_iter()
            .find(|o| o.kind == OpKind::LinalgGeneric)
            .expect("a linalg.generic");
        // THE POINT: a BARE YIELD, and no linalg.broadcast anywhere. The downstream
        // legality check accepts the generic and rejects the named op (probe p07), so
        // "improving" this is a silent break.
        assert_eq!(
            g.regions[0].ops.iter().map(|o| o.kind.spelling()).collect::<Vec<_>>(),
            vec!["linalg.yield"]
        );
        assert!(
            !m.ops_deep().iter().any(|o| o.kind.spelling() == "linalg.broadcast"),
            "the named linalg.broadcast must never be emitted"
        );
        // ins maps the kept dim, outs is the identity.
        assert_eq!(
            g.attr(&AttrKey::IndexingMaps),
            Some(&Attr::AffineMapList(vec![
                "(d0, d1) -> (d0)".into(),
                "(d0, d1) -> (d0, d1)".into()
            ]))
        );
        // And the unit dim was collapsed away first.
        assert!(m.ops_deep().iter().any(|o| o.kind == OpKind::TensorCollapseShape));
    }

    #[test]
    fn nothing_triton_survives_and_the_guard_names_the_dialect() {
        let m = run_on(REDUCE, &[8]).unwrap();
        for op in m.ops_deep() {
            assert!(
                !op.kind.spelling().starts_with("tt."),
                "{} survived the conversion",
                op.kind.spelling()
            );
        }
        // The corelet plan is DROPPED, not converted -- `ktdf` is a name collision.
        assert!(!m.ops_deep().iter().any(|o| o.kind == OpKind::KtdfCoreletPlan));
        // And the guard fires on a planted foreign op.
        let mut bad = parse::parse(REDUCE).unwrap();
        bad.ops[0].regions[0]
            .ops
            .push(Op::new(OpKind::Other("tt.histogram".into())));
        let e = run(&mut bad, &[8]).unwrap_err();
        assert!(e.message.contains("which the scheduler does not register"), "got {e}");
    }

    /// A MULTI-TRIP GRID LOOP IS UNROLLED, AND `grid` BECOMES `num_cores`.
    ///
    /// This test has said two different things and the second was also wrong. It first
    /// asserted a REFUSAL, on the grounds that "cannot fold" is "cannot lower"; it then
    /// asserted the loop was KEPT, on the grounds that an `scf.for` induction variable is an
    /// addressable subscript. Both readings stopped at addressing. What they missed is that
    /// `ktir_superdsc::emit::lower_ktir_to_superdsc::regions` and every reader beside it walks
    /// `IRFunction::operations` -- the TOP LEVEL -- so a kept loop hides the whole body's
    /// windows from all of them, and it refused `rope_q32` by name for exactly that.
    ///
    /// So the loop goes, and it goes by UNROLLING rather than folding: `work_items / num_cores`
    /// straight-line copies, copy `t` addressing `core_id + t * num_cores`. Folding would have
    /// dropped every work item after the first.
    ///
    /// What this pins: the loop is gone, the body appears once per trip, THE COPIES READ
    /// DIFFERENT WORK ITEMS, `grid` reads `num_cores` (because only that many tiles launch and
    /// the copies cover the rest), and there is NO launch contract -- every work item is covered
    /// exactly once, so there is no zero-trip tile for a caller to guarantee anything about.
    #[test]
    fn a_genuinely_multi_trip_grid_loop_is_unrolled_and_regrids_to_num_cores() {
        // work_items 64 > num_cores 32: two trips per core.
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %tid = ktdp.get_compute_tile_id : index
    scf.for %i = %tid to %c64 step %c32 {
      %r = arith.index_cast %i : index to i32
    }
    tt.return
  }
}
";
        let m = run_on(src, &[64]).expect("a multi-trip loop lowers by unrolling");

        // THE LOOP IS GONE.
        assert!(
            !m.ops_deep().iter().any(|o| o.kind == OpKind::ScfFor),
            "the multi-trip work loop must be unrolled away: every reader downstream walks the \
             function's top level, so a surviving loop hides the body from all of them"
        );
        let f = m.ops.iter().find(|o| o.kind == OpKind::FuncFunc).expect("func.func");
        let top = &f.regions[0].ops;

        // TWO COPIES OF THE BODY, one per trip. One would be a fold in disguise.
        let casts: Vec<&Op> = top.iter().filter(|o| o.kind == OpKind::ArithIndexCast).collect();
        assert_eq!(
            casts.len(),
            2,
            "64 work items over 32 cores is two trips, so the body appears twice; {} says the \
             loop was folded and half the work items were dropped",
            casts.len()
        );

        // AND THE TWO COPIES READ DIFFERENT WORK ITEMS: `core_id`, then `core_id + 32`. Two
        // copies of the SAME pid would satisfy the count above and compute one trip twice.
        let tid = top
            .iter()
            .find(|o| o.kind == OpKind::KtdpGetComputeTileId)
            .and_then(|o| o.result())
            .expect("the work-distribution landmark survives");
        assert_eq!(casts[0].operands[0], tid, "trip 0 addresses the landmark itself");
        let addi = top
            .iter()
            .find(|o| o.kind == OpKind::ArithAddi)
            .expect("trip 1 addresses `core_id + num_cores`, which is an arith.addi");
        assert_eq!(addi.operands[0], tid, "the addend is the landmark");
        assert_eq!(
            crate::passes::dot_to_linalg::const_int(&m, addi.operands[1]),
            Some(32),
            "trip t addresses `core_id + t * num_cores`, so trip 1's offset is num_cores"
        );
        assert_eq!(casts[1].operands[0], addi.result().unwrap(), "trip 1 reads that sum");

        // AND THE COPY DEFINES ITS OWN NAME. Two copies sharing one SSA name is one trip whose
        // definition the second shadows, and every use would pair with whichever came first.
        assert_ne!(
            casts[0].result(),
            casts[1].result(),
            "an unrolled copy must rename every value it defines"
        );

        // `grid` is num_cores (32), NOT the 64 work items: only 32 tiles launch and each runs
        // both copies. Leaving it at 64 would over-launch 2x.
        assert_eq!(
            f.attr(&AttrKey::Grid),
            Some(&Attr::IntList(vec![32])),
            "grid must be num_cores when the loop is unrolled"
        );
        // And NO launch contract: the copies cover every work item exactly once, so there is
        // no zero-trip tile and nothing for a caller to guarantee.
        assert_eq!(
            f.attr(&AttrKey::FoldedGridLoop),
            None,
            "an evenly unrolled loop needs no launch contract"
        );
    }

    /// A TRIP COUNT THAT DOES NOT DIVIDE IS REFUSED BY NAME, NOT ROUNDED.
    ///
    /// The discriminating control for the unroll. With `work_items % num_cores != 0` the cores
    /// do not all run the same number of trips, so no fixed number of copies is right for all
    /// of them: unrolling to the larger count makes the short cores compute an out-of-range work
    /// item, and unrolling to the smaller drops work. Both are silent wrong answers, so neither
    /// is done -- and without this test the pass could pick either and nothing would say so.
    #[test]
    fn a_trip_count_that_does_not_divide_is_refused_naming_both_numbers() {
        // 70 work items over 32 cores: cores 0..5 run 3 trips, cores 6..31 run 2.
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %c70 = arith.constant 70 : index
    %c32 = arith.constant 32 : index
    %tid = ktdp.get_compute_tile_id : index
    scf.for %i = %tid to %c70 step %c32 {
      %r = arith.index_cast %i : index to i32
    }
    tt.return
  }
}
";
        let e = run_on(src, &[70]).expect_err("an uneven trip count has no straight-line form");
        for needle in ["70 work items", "32 compute tiles", "3 trips", "the rest 2"] {
            assert!(
                e.message.contains(needle),
                "the refusal must name {needle:?} so the launch can be fixed -- got {}",
                e.message
            );
        }
    }

    #[test]
    fn a_single_trip_grid_loop_folds_and_records_the_launch_contract() {
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %tid = ktdp.get_compute_tile_id : index
    scf.for %i = %tid to %c8 step %c32 {
      %r = arith.index_cast %i : index to i32
    }
    tt.return
  }
}
";
        let m = run_on(src, &[8]).unwrap();
        assert!(
            !m.ops_deep().iter().any(|o| o.kind == OpKind::ScfFor),
            "the grid loop is folded away"
        );
        let f = m.ops.iter().find(|o| o.kind == OpKind::FuncFunc).unwrap();
        let contract = f.attr(&AttrKey::FoldedGridLoop).expect(
            "the launch contract must be recorded, or the caller is not told what it now \
             has to guarantee",
        );
        // AND IT CARRIES THE NUMBERS: "launch exactly this many tiles" is the
        // obligation, so the quantity has to be in the artifact.
        assert_eq!(
            contract,
            &Attr::StrList(vec![
                "num_cores = 32 : index".to_string(),
                "work_items = 8 : index".to_string()
            ])
        );
        // The body's use of the IV now reads the landmark directly -- a per-core
        // uniform, which is the one non-constant form the address route accepts.
        let cast = m
            .ops_deep()
            .into_iter()
            .find(|o| o.kind == OpKind::ArithIndexCast)
            .expect("the body was inlined");
        let landmark = m
            .ops_deep()
            .into_iter()
            .find(|o| o.kind == OpKind::KtdpGetComputeTileId)
            .unwrap();
        assert_eq!(cast.operands[0], landmark.results[0]);
    }

    // RUNG 3: a `tt.splat` of a RUNTIME scalar argument. `make_ttir` folds every
    // splat of a compile-time constant, so the surviving form is exactly this one.
    // `convert_splat_of_argument` rewrites it as `tensor.splat` (one spelling for both
    // provenances, in a dialect the scheduler registers); THIS test's rewrite chain then
    // states the `[1,1]` buffer the launcher binds — so the assertions cover BOTH legs.
    const SPLAT_ARG: &str = "\
module {
  tt.func public @k(%a: !tt.ptr<f16>, %scale: f16) attributes {noinline = false} {
    %x = arith.constant dense<0.000000e+00> : tensor<64xf16>
    %s = tt.splat %scale : tensor<64xf16>
    %y = arith.mulf %x, %s : tensor<64xf16>
    tt.return
  }
}
";

    #[test]
    fn a_splat_of_an_argument_becomes_a_load_backed_tensor_splat() {
        let m = run_on(SPLAT_ARG, &[1]).unwrap();
        let ops = m.ops_deep();
        let splat = ops
            .iter()
            .find(|o| o.kind == OpKind::TensorSplat)
            .expect("the tt.splat was rewritten as tensor.splat");
        // RUNG 3: the operand is the `[1,1]` LOAD, not the argument — the value crosses
        // as a launch binding whose width the view states.
        let load = ops
            .iter()
            .find(|o| o.kind == OpKind::KtdpLoad)
            .expect("the scalar argument's [1,1] buffer chain was emitted");
        assert_eq!(splat.operands, vec![load.results[0]]);
        // The load reads a [1,1] access tile over a [1,1] view of the RETYPED argument:
        // the argument is an `index` now, exactly a pointer parameter.
        let f = m.ops.iter().find(|o| o.kind == OpKind::FuncFunc).expect("func.func");
        assert_eq!(f.regions[0].args[1].1, IrType::Index);
        let tile = ops
            .iter()
            .find(|o| o.kind == OpKind::KtdpConstructAccessTile)
            .expect("a [1,1] access tile");
        assert_eq!(tile.result_types[0], IrType::AccessTile { dims: vec![1, 1] });
        let view = ops
            .iter()
            .find(|o| o.kind == OpKind::KtdpConstructMemoryView)
            .expect("a [1,1] memory view");
        assert_eq!(
            view.result_types[0],
            IrType::MemRef { dims: vec![1, 1], elem: DType::F16 }
        );
        // The view's address operand is the argument itself (retyped), and the tile's
        // corner operands are index constants — the form `index_constants` resolves.
        assert_eq!(view.operands, vec![f.regions[0].args[1].0]);
        let consts: Vec<i64> = tile.operands[1..]
            .iter()
            .filter_map(|s| {
                ops.iter()
                    .find(|o| o.results.first() == Some(s) && o.kind == OpKind::ArithConstant)
                    .and_then(|o| match o.attr(&AttrKey::Value) {
                        Some(Attr::Int(i)) => Some(*i),
                        _ => None,
                    })
            })
            .collect();
        assert_eq!(consts, vec![0, 0]);
        // The chain is at the TOP LEVEL (the handoff guard demands it) and BEFORE the
        // splat that consumes it.
        let top = &m.kernel().unwrap().regions[0].ops;
        let view_idx = top
            .iter()
            .position(|o| o.kind == OpKind::KtdpConstructMemoryView)
            .unwrap();
        let splat_idx = top
            .iter()
            .position(|o| o.kind == OpKind::TensorSplat)
            .unwrap();
        assert!(view_idx < splat_idx, "the chain dominates the splat at the top level");
        // And no `tt.*` op survived: the dialect gate is the pass's own contract.
        assert!(
            !ops.iter().any(|o| o.kind.spelling().starts_with("tt.")),
            "a tt.* op survived to_ktir"
        );
    }

    /// ⛔ THE FAIL-CLOSED ARM: a scalar argument of a NON-fp16/f32 type. The launcher's
    /// one bind spelling is a 2-B IEEE fp16 (`const:t<id>`), so an i32 scalar has no width
    /// this rewrite can state — refused by name rather than given a `[1,1]` chain whose
    /// extent nobody binds.
    const SPLAT_ARG_I32: &str = "\
module {
  tt.func public @k(%a: !tt.ptr<f16>, %n: i32) attributes {noinline = false} {
    %x = arith.constant dense<0.000000e+00> : tensor<64xf16>
    %s = tt.splat %n : tensor<64xi32>
    %y = arith.mulf %x, %s : tensor<64xf16>
    tt.return
  }
}
";

    #[test]
    fn a_splat_of_a_non_float_scalar_argument_is_refused_by_name() {
        let e = run_on(SPLAT_ARG_I32, &[1]).unwrap_err();
        assert!(
            e.message.contains("no bind spelling"),
            "got {e:?}"
        );
    }

    // THE FAIL-CLOSED ARM: a splat of an INTERIOR value. No golden produces this and
    // no contract covers it, so the rewrite must refuse by name rather than emit a
    // `tensor.splat` whose operand the scale registry would then misread as a launch
    // binding.
    const SPLAT_INTERIOR: &str = "\
module {
  tt.func public @k(%a: !tt.ptr<f16>) attributes {noinline = false} {
    %x = arith.constant dense<0.000000e+00> : tensor<64xf16>
    %s = tt.splat %x : tensor<64xf16>
    %y = arith.mulf %x, %s : tensor<64xf16>
    tt.return
  }
}
";

    #[test]
    fn a_splat_of_an_interior_value_is_refused_by_name() {
        let mut m = parse::parse(SPLAT_INTERIOR).unwrap();
        let err = run(&mut m, &[1]).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("not a function argument"),
            "the refusal must name the unbound provenance; got: {msg}"
        );
    }
}
