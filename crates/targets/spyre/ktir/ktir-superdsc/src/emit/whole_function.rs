// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").

//! ONE FUNCTION, MANY OPS — the door for a KTIR producer whose function is a whole kernel.
//!
//! # THE PROBLEM THIS SOLVES, AND WHOSE PROBLEM IT IS
//!
//! Every entry point in [`super::lower_ktir_to_superdsc`] reads a function's PARAMETER LIST as one
//! op's operands: [`super::lower_ktir_to_superdsc::split_out`] takes the parameter written by a
//! `ktdp.store` as the output, counts the rest as inputs, and refuses when the count is not the
//! body's arity. That is exact for a function that IS one op, which is what `KtirFunc` emits — one
//! program per model-graph node.
//!
//! It is not the only legitimate KTIR. **IBM's C++ toolchain — the reference KTIR producer — puts a
//! whole kernel in ONE function**: its own checked-in output has 3 `linalg.matmul` in one
//! `func.func` for a SwiGLU MLP and 12 in one for a decoder layer. KTIR itself says nothing about
//! how many ops a function holds; a `func.func` holds what it holds. So the arity check is a
//! property of these bodies, not of the IR, and a producer matching the reference shape is refused
//! by a crate that could serve it.
//!
//! # WHY THIS IS PURELY ADDITIVE
//!
//! Nothing here changes an existing entry point, `split_out`, or any assembler. The bodies take
//! `&[Region]` and filter it by `is_out`, so handing a body the regions of ONE OP satisfies
//! `split_out` unmodified. This module's whole job is to build those per-op lists:
//!
//! ```text
//!   regions()            per PARAMETER: view <- parameter,     then the FIRST tile over the view
//!   region_for_operand() per OPERAND:   load <- tile <- view <- parameter, and THAT tile
//! ```
//!
//! The second is strictly more precise. `regions()`'s own comment flags that "a program that tiles
//! one buffer many ways is not addressed through this field"; rooting the walk at the operand reads
//! the window the op actually takes, so that case stops needing a special path.
//!
//! # WHAT IT DOES NOT DO
//!
//! It recognises NOTHING. An op becomes a `Program` only where the mapping is 1:1 by op kind
//! (`linalg.matmul` -> `Matmul`, `math.exp` -> `Elementwise(Exp)`). It does not look for a silu
//! longhand, an rmsnorm shape or an online softmax; a producer that wants `SiluMul` states it by
//! calling that entry point itself. Anything with no 1:1 mapping is REFUSED BY NAME, because a
//! guessed node kind is a well-formed descriptor computing the wrong function.

use ktir_core::ir::Ssa;
use ktir_core::opkind::OpKind;

use super::lower_ktir_to_superdsc::{Error, Region, err, regions};
use crate::ktir_node::{Elementwise, KtirNode, Program};
use crate::placement::BundleLayout;

/// The `Program` an op names, where the naming is 1:1 and needs no inference.
///
/// `None` means "not a compute op" (a view, a tile, a load, a store, a constant) — those are
/// addressing and are consumed by the operand walk, not lowered on their own.
pub fn program_of(op: OpKind) -> Option<Program> {
    Some(match op {
        OpKind::LinalgMatmul => Program::Matmul,
        OpKind::LinalgBatchMatmul => Program::Matmul,
        OpKind::MathExp => Program::Elementwise(Elementwise::Exp),
        OpKind::MathRsqrt => Program::Elementwise(Elementwise::Rsqrt),
        OpKind::MathSqrt => Program::Elementwise(Elementwise::Sqrt),
        OpKind::MathAbsf => Program::Elementwise(Elementwise::Abs),
        OpKind::MathTanh => Program::Elementwise(Elementwise::Tanh),
        OpKind::ArithAddf => Program::Elementwise(Elementwise::Add),
        OpKind::ArithSubf => Program::Elementwise(Elementwise::Sub),
        OpKind::ArithMulf => Program::Elementwise(Elementwise::Mul),
        OpKind::ArithDivf => Program::Elementwise(Elementwise::RealDiv),
        OpKind::ArithMaxnumf => Program::Elementwise(Elementwise::Maximum),
        OpKind::ArithMinnumf => Program::Elementwise(Elementwise::Minimum),
        OpKind::LinalgTranspose => Program::Transpose,
        _ => return None,
    })
}

/// Is this op addressing or bookkeeping rather than computation?
fn is_plumbing(op: OpKind) -> bool {
    matches!(
        op,
        OpKind::KtdpConstructMemoryView
            | OpKind::KtdpConstructAccessTile
            | OpKind::KtdpConstructIndirectAccessTile
            | OpKind::KtdpLoad
            | OpKind::KtdpStore
            | OpKind::KtdpGetComputeTileId
            | OpKind::ArithConstant
            | OpKind::TensorSplat
            | OpKind::TensorEmpty
            | OpKind::FuncReturn
            | OpKind::LinalgYield
            | OpKind::ScfYield
            // ⭐ INTEGER ARITHMETIC IS ADDRESSING HERE, NOT COMPUTATION, and that is decidable
            // rather than a judgement call: `Elementwise` is entirely FLOAT (`Silu`…`Minimum`, all
            // f16/f32 `OpFunc`s), so there is no node kind an integer op could become. Every one of
            // these in a Triton-derived program computes a subscript — a flattened tile id, a
            // row/head delinearisation, an `index` cast of a pointer — and is consumed by a
            // `ktdp.construct_access_tile`, never by a compute op.
            | OpKind::ArithIndexCast
            | OpKind::ArithIndexCastui
            | OpKind::ArithAddi
            | OpKind::ArithSubi
            | OpKind::ArithMuli
            | OpKind::ArithDivsi
            | OpKind::ArithDivui
            | OpKind::ArithRemsi
            | OpKind::ArithRemui
            | OpKind::ArithExtsi
            | OpKind::ArithExtui
            | OpKind::ArithTrunci
            | OpKind::ArithCeildivsi
            | OpKind::ArithCeildivui
            | OpKind::ArithFloordivsi
    )
}

/// The [`Region`] one OPERAND names, by walking back from the value to the parameter it reads.
///
/// `ktdp.load <- ktdp.construct_access_tile <- ktdp.construct_memory_view <- parameter`, and the
/// window is THAT tile's — not "the first tile over the view", which is what the parameter-rooted
/// walk in [`regions`] can only offer.
///
/// Returns `Ok(None)` when the value is not a load of a parameter at all (a splat constant, a
/// previous op's result). Those are the caller's two other cases: a constant operand, and an
/// intermediate.
pub fn region_for_operand(k: &KtirNode, v: Ssa) -> Result<Option<Region>, Error> {
    // The parameter-rooted walk already computes every field of a `Region` correctly; the only
    // thing wrong for a multi-op function is WHICH parameter and WHICH tile. So find the parameter
    // this value reads and reuse that Region, rather than rebuilding one field by field and
    // risking a different answer from the same IR.
    let f = &k.func;
    let load = f.operations.iter().find(|o| o.result == Some(v) && o.op_type == OpKind::KtdpLoad);
    let Some(load) = load else { return Ok(None) };
    let tile_v = load.operands.first().copied();
    let tile = f
        .operations
        .iter()
        .find(|o| o.result == tile_v && o.op_type == OpKind::KtdpConstructAccessTile);
    let Some(tile) = tile else { return Ok(None) };
    let view_v = tile.operands.first().copied();
    let view = f
        .operations
        .iter()
        .find(|o| o.result == view_v && o.op_type == OpKind::KtdpConstructMemoryView);
    let Some(view) = view else { return Ok(None) };
    let ptr = view.operands.first().copied();

    // Which parameter is that, and therefore which `Region` of the parameter-rooted walk.
    let idx = f.arguments.iter().position(|(a, _)| Some(*a) == ptr);
    let Some(idx) = idx else { return Ok(None) };
    let all = regions(k)?;
    let Some(r) = all.get(idx).copied() else {
        return err(format!(
            "{}: parameter {idx} has no region, though the operand walk reached it",
            f.name
        ));
    };
    Ok(Some(r))
}

/// Lower a function holding MANY ops, one `Program` per op, in program order.
///
/// Each op is handed only its own operands' regions, so the existing bodies apply unchanged.
///
/// # WHAT IS NOT IMPLEMENTED YET, AND IS REFUSED RATHER THAN GUESSED
///
/// An INTERMEDIATE — a result one op produces and the next consumes, which no `ktdp.store` writes —
/// needs a buffer of its own, declared to [`BundleLayout::synth`] so `resolve_seg_base` can place
/// it. That declaration is the caller's (it owns the layout), so this refuses and names the value
/// rather than inventing an address. A CONSTANT operand (a `tensor.splat`) is refused for the same
/// reason: `Program::ScalarMul` is the door for "times a stated constant" and choosing it here would
/// be picking a node kind on the producer's behalf.
pub fn lower_function(
    k: &KtirNode,
    layout: Option<&BundleLayout>,
    sym_id_base: &mut i64,
) -> Result<Vec<super::EmittedOp>, Error> {
    let f = &k.func;
    let mut out = Vec::new();
    let mut lowered = 0usize;

    // INTERMEDIATES, BY THE VALUE THAT PRODUCED THEM.
    //
    // A result one op produces and the next consumes is never a parameter, so the operand walk
    // cannot find it and `regions()` never had a row for it. It needs a buffer, and
    // `BundleLayout::synth` is the mechanism `resolve_seg_base` already resolves through (its case
    // 2, "the name is in the synth allocator's map → an intermediate's stable offset").
    //
    // ⭐ DECLARED HERE RATHER THAN BY THE CALLER, because `synth` takes `&self` through a `RefCell`
    // and because the caller cannot know which values are intermediate without doing this walk
    // itself. Asking it to would be asking it to duplicate the walk in order to feed the walk.
    //
    // ⛔ THE TID MUST NOT COLLIDE WITH A PARAMETER'S. `act_name(tid)` is the operand name and the
    // key into `placements`/`ids`, so a reused tid would silently alias an intermediate onto a real
    // buffer. They are minted strictly above every bound tid.
    let mut next_tid: u32 = k.bindings.iter().map(|b| b.get()).max().map_or(0, |m| m + 1);
    let mut inter: std::collections::HashMap<Ssa, Region> = std::collections::HashMap::new();

    for op in f.operations.iter() {
        if is_plumbing(op.op_type) {
            continue;
        }
        let Some(program) = program_of(op.op_type) else {
            return err(format!(
                "{}: `{:?}` has no 1:1 `Program`, so this door cannot lower it without choosing a \
                 node kind on the producer's behalf. Either it belongs to a fused kind the producer \
                 should state by calling that entry point directly, or it needs a `Program` variant",
                f.name, op.op_type
            ));
        };

        // THIS OP'S INPUTS. `linalg.matmul`'s last operand is its `outs` accumulator init, not an
        // input, and the result is the output — so the inputs are the leading operands.
        let n_in = match program {
            Program::Matmul => 2,
            Program::Elementwise(e) => match super::lower_ktir_to_superdsc::elementwise_op_func(f.name, e) {
                Ok((_, arity)) => arity,
                Err(e) => return Err(e),
            },
            Program::Transpose => 1,
            _ => op.operands.len(),
        };

        let mut per_op: Vec<Region> = Vec::with_capacity(n_in + 1);
        for (i, v) in op.operands.iter().take(n_in).enumerate() {
            match region_for_operand(k, *v)? {
                Some(mut r) => {
                    r.is_out = false;
                    per_op.push(r);
                }
                // A value the walk cannot reach is either an intermediate this function already
                // minted a buffer for, or a constant. The first is looked up; the second is
                // refused, because picking `ScalarMul` for it would be choosing a node kind on the
                // producer's behalf.
                None => match inter.get(v) {
                    Some(r) => {
                        let mut r = *r;
                        r.is_out = false;
                        per_op.push(r);
                    }
                    None => {
                        return err(format!(
                            "{}: `{:?}` input {i} is neither a load of a parameter nor an \
                             intermediate this function produced — so it is a constant (a \
                             `tensor.splat`). `Program::ScalarMul` is the door for \"times a stated \
                             constant\" and choosing it here would pick a node kind for the \
                             producer, so it is refused instead",
                            f.name, op.op_type
                        ));
                    }
                },
            }
        }

        // THIS OP'S OUTPUT: the parameter a `ktdp.store` writes from this op's result.
        let stored = f.operations.iter().find(|s| {
            s.op_type == OpKind::KtdpStore && s.operands.first().copied() == op.result
        });
        let Some(store) = stored else {
            // NOT STORED => AN INTERMEDIATE. Mint a buffer for it, declare it to the layout, and
            // record it so the consuming op finds it as an input.
            let Some(res) = op.result else {
                return err(format!("{}: `{:?}` has no result to place", f.name, op.op_type));
            };
            let dims = match op.result_type {
                Some(ktir_core::irtype::IrType::Tensor { dims, .. }) if dims.len() == 2 => {
                    [dims[0] as u32, dims[1] as u32]
                }
                _ => {
                    return err(format!(
                        "{}: `{:?}`'s result is an intermediate but states no 2-D tensor type, so \
                         its footprint is unknown and no buffer can be sized for it",
                        f.name, op.op_type
                    ));
                }
            };
            let Some(layout) = layout else {
                return err(format!(
                    "{}: `{:?}`'s result is an intermediate, which needs a `BundleLayout` to hold \
                     its buffer -- `layout: None` is the unit-test arm and cannot place one",
                    f.name, op.op_type
                ));
            };
            let tid = next_tid;
            next_tid += 1;
            layout.synth(crate::place::PlaceId::Act(tid), &dims);
            let r = Region {
                tid,
                v_rows: dims[0],
                v_cols: dims[1],
                r_start: 0,
                c_start: 0,
                r_len: dims[0],
                c_len: dims[1],
                r_cover: (0, dims[0]),
                is_out: true,
                is_fp8: false,
            };
            inter.insert(res, r);
            per_op.push(r);
            let name = f.name;
            let mut emitted = emit_one(name, program, &per_op, sym_id_base, Some(layout))?;
            out.append(&mut emitted);
            lowered += 1;
            continue;
        };
        let out_tile = store.operands.get(1).copied();
        let out_view = f
            .operations
            .iter()
            .find(|o| {
                o.op_type == OpKind::KtdpConstructAccessTile && o.result == out_tile
            })
            .and_then(|t| t.operands.first().copied());
        let out_ptr = f
            .operations
            .iter()
            .find(|o| o.op_type == OpKind::KtdpConstructMemoryView && o.result == out_view)
            .and_then(|o| o.operands.first().copied());
        let out_idx = f.arguments.iter().position(|(a, _)| Some(*a) == out_ptr);
        let Some(out_idx) = out_idx else {
            return err(format!("{}: `{:?}`'s store names no parameter", f.name, op.op_type));
        };
        let all = regions(k)?;
        let mut o = all[out_idx];
        o.is_out = true;
        per_op.push(o);

        let mut emitted = emit_one(f.name, program, &per_op, sym_id_base, layout)?;
        out.append(&mut emitted);
        lowered += 1;
    }

    if lowered == 0 {
        return err(format!("{}: no compute op in the function", f.name));
    }
    Ok(out)
}

/// Hand ONE op's regions to the body its `Program` names.
///
/// Every body here is unchanged and unaware it is being called per-op rather than per-function:
/// `split_out` filters the list by `is_out`, and this list holds exactly one output and that op's
/// inputs, so its arity check passes for the reason it was written to.
fn emit_one(
    name: &str,
    program: Program,
    per_op: &[Region],
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<super::EmittedOp>, Error> {
    Ok(match program {
        Program::Matmul => {
            // The fp8 activation-quantize dedup set is per-BUNDLE. One op at a time here, so a
            // fresh set is correct for an fp16 program; an fp8 one needs it threaded across ops and
            // that is not yet done, so it is stated rather than silently per-op.
            let mut q = std::collections::HashSet::new();
            super::lower_ktir_to_superdsc::matmul(name, per_op, sym_id_base, layout, &mut q)?
        }
        Program::Elementwise(e) => {
            super::lower_ktir_to_superdsc::elementwise(name, e, per_op, sym_id_base, layout)?
        }
        Program::Transpose => {
            super::lower_ktir_to_superdsc::transpose(name, per_op, sym_id_base, layout)?
        }
        other => {
            return err(format!(
                "{name}: `{other:?}` is not reachable from this door — it is a fused kind the \
                 producer states by calling its entry point directly"
            ))
        }
    })
}
