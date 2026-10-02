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

//! Ported from `ConvertTTIRToKTDP.cpp` (SP-E2-01 / SP-E2-02).
//!
//! ```text
//!   tt.descriptor_load/store -> ktdp.construct_memory_view
//!                             + ktdp.construct_access_tile
//!                             + ktdp.load/store
//! ```
//!
//! Then `tt.trans` of a tile load folds into that load's `access_tile_order`, so
//! flash attention's `q.K^T` costs NOTHING. The C++ records the cost of not doing
//! it: the tempting workaround -- a second, pre-transposed K plane on the host --
//! DOUBLES the KV cache, the dominant memory consumer in inference and precisely
//! what paging exists to bound.
//!
//! SCOPE: the descriptor forms only. Raw-pointer TILE access is refused BY NAME
//! here rather than passed through silently, because a silent pass-through
//! surfaces three stages later as "cannot recover the tile stick count" in
//! PlanCorelets -- a true statement whose cause is here. A raw SCALAR pointer read
//! stays legal: the attention kernel reads its per-row context length that way and
//! that value is a masking predicate, never an address.

use crate::ir::*;
use crate::passes::walk::{self, OpPath};
use crate::{Refusal, Result};

const PASS: &str = "ConvertTTIRToKTDP";

pub fn run(module: &mut Module) -> Result<()> {
    walk_1_descriptors(module)?;
    precondition_check(module)?;
    convert_access_ops(module)?;
    // AFTER the direct path, and the order is load-bearing rather than tidy: a gather's
    // `x_offsets` must already BE the lowered `ktdp.load` so its index view can be traced.
    // The C++ gets the same effect from a conversion adaptor, which hands the pattern the
    // post-conversion value.
    convert_gathers(module)?;
    fold_trans_into_access_tile_order(module);
    refuse_raw_tile_access(module)?;
    Ok(())
}

//===----------------------------------------------------------------------===//
// Walk 1: tt.make_tensor_descriptor -> ktdp.construct_memory_view
//===----------------------------------------------------------------------===//

/// Every descriptor's uses are routed to a memref view of the underlying buffer,
/// wrapped in an `unrealized_conversion_cast` so the `!tt.tensordesc`-typed uses
/// keep verifying. The access-op patterns pick the memref up through that cast.
fn walk_1_descriptors(module: &mut Module) -> Result<()> {
    while let Some(path) = walk::paths(module).into_iter().find(|p| {
        walk::at(module, p)
            .map(|o| o.kind == OpKind::TtMakeTensorDescriptor)
            .unwrap_or(false)
    }) {
        let desc = walk::at(module, &path).expect("path").clone();
        let result = desc.result().ok_or_else(|| {
            Refusal::new(PASS, "tt.make_tensor_descriptor defines no descriptor")
        })?;

        // A descriptor nothing uses is simply erased -- `attention_flash`'s
        // `desc_mask` in the non-causal configuration is exactly this.
        let used = walk::used_values(module);
        if !used.contains(&result) {
            walk::erase(module, &[path]);
            continue;
        }

        let elem = desc
            .result_type()
            .and_then(|t| t.elem())
            .ok_or_else(|| Refusal::new(PASS, "descriptor has no element type"))?;
        let built = build_base_memory_view(module, &desc, elem)?;
        let cast_result = built
            .last()
            .and_then(|o| o.result())
            .expect("the descriptor cast defines a value");

        // Splice the three ops in where the descriptor was, then rewire its uses.
        let idx = path.index();
        {
            let block = walk::block_mut(module, &path).expect("path");
            block.remove(idx);
            for (k, nop) in built.into_iter().enumerate() {
                block.insert(idx + k, nop);
            }
        }
        walk::replace_all_uses(module, result, cast_result);
    }
    Ok(())
}

/// `buildBaseMemoryView`. Shape/strides come off the descriptor's SSA values as
/// compile-time constants when available, `kDynamic` otherwise (with an
/// `index_cast` of the runtime value). Memory space is HBM.
fn build_base_memory_view(module: &mut Module, desc: &Op, elem: DType) -> Result<Vec<Op>> {
    let rank = desc.result_type().map(|t| t.rank()).unwrap_or(0);
    let base = desc
        .operands
        .first()
        .copied()
        .ok_or_else(|| Refusal::new(PASS, "descriptor has no base pointer"))?;

    let mut shape = Vec::new();
    let mut strides = Vec::new();
    // The operand layout is base, then rank shape values, then rank stride values.
    let have = desc.operands.len() > 2 * rank;
    if have {
        for i in 0..rank {
            shape.push(
                super::dot_to_linalg::const_int(module, desc.operands[1 + i]).unwrap_or(DYNAMIC),
            );
        }
        for i in 0..rank {
            strides.push(
                super::dot_to_linalg::const_int(module, desc.operands[1 + rank + i])
                    .unwrap_or(DYNAMIC),
            );
        }
    }
    // Fallback: no explicit shape -> the block shape; no explicit strides ->
    // row-major from the shape.
    if shape.is_empty() {
        shape = desc.result_type().and_then(|t| t.dims().map(|d| d.to_vec())).unwrap_or_default();
    }
    if strides.is_empty() {
        let mut s = 1i64;
        strides = vec![0; shape.len()];
        for i in (0..shape.len()).rev() {
            strides[i] = s;
            if shape[i] != DYNAMIC {
                s *= shape[i];
            }
        }
    }

    // A DYNAMIC EXTENT IS AN OPERAND, NOT A SENTINEL IN THE STATIC LIST.
    //
    // `buildBaseMemoryView` emits an `arith.index_cast` of the runtime shape value and passes
    // it to the view; the static `sizes` attribute then carries only the STATIC extents, which
    // for `vector_add` means an empty list. This port previously recorded `kDynamic` in the
    // attribute and emitted no cast at all -- and the comment above already said it should
    // ("kDynamic otherwise (with an `index_cast` of the runtime value)"), so the port had the
    // comment and not the code.
    //
    // WHY IT MATTERS MORE THAN ITS SIZE: `vector_add` and `mul` are the ONLY configurations
    // with a runtime descriptor extent, so this path is unverified by everything else.
    let base_hint = module.hint(base);
    let mut dyn_casts: Vec<Op> = Vec::new();
    let mut dyn_sizes: Vec<Ssa> = Vec::new();
    if have {
        for (i, extent) in shape.iter().enumerate() {
            if *extent != DYNAMIC {
                continue;
            }
            let src = desc.operands[1 + i];
            match module.type_of(src) {
                Some(IrType::Index) => dyn_sizes.push(src),
                _ => {
                    let v = module.fresh_named(&base_hint);
                    dyn_casts.push(
                        Op::new(OpKind::ArithIndexCast)
                            .with_result(v, IrType::Index)
                            .with_operands([src]),
                    );
                    dyn_sizes.push(v);
                }
            }
        }
    }
    // The static list keeps only the extents that ARE static.
    let static_sizes: Vec<i64> = shape.iter().copied().filter(|d| *d != DYNAMIC).collect();
    let base_idx = module.fresh_named(&base_hint);
    let ptr_cast = Op::new(OpKind::UnrealizedConversionCast)
        .with_result(base_idx, IrType::Index)
        .with_operands([base]);

    let view = module.fresh_named(&base_hint);
    let view_op = Op::new(OpKind::KtdpConstructMemoryView)
        .with_result(view, IrType::MemRef { dims: shape.clone(), elem })
        .with_operands(std::iter::once(base_idx).chain(dyn_sizes.iter().copied()))
        .with_attr(AttrKey::Shape, Attr::IntList(static_sizes))
        .with_attr(AttrKey::Strides, Attr::IntList(strides))
        .with_attr(AttrKey::CoordinateSet, Attr::AffineSet(build_range_set_nd(&shape)))
        .with_attr(AttrKey::MemorySpace, Attr::Str("HBM".into()));

    // The descriptor-typed placeholder cast, so the `!tt.tensordesc` uses keep
    // verifying until the access-op conversion reaches through it.
    let cast = Op::new(OpKind::UnrealizedConversionCast)
        .with_result(
            module.fresh_named(&base_hint),
            desc.result_type().cloned().unwrap_or(IrType::Verbatim("!tt.tensordesc".into())),
        )
        .with_operands([view]);

    // The pointer cast, then ONE `arith.index_cast` per runtime extent, then the view, then
    // the descriptor-typed cast. The caller splices this list in where the descriptor was, so
    // the order here IS the emitted order -- and it is the C++'s: the golden shows
    // `%a_desc = ...cast`, `%a_desc_4 = arith.index_cast %n`, `%a_desc_5 = ...memory_view`.
    let mut out = vec![ptr_cast];
    out.extend(dyn_casts);
    out.push(view_op);
    out.push(cast);
    Ok(out)
}

/// `buildRangeSetND`: `(d0, d1) : (d0 >= 0, -d0 + N0-1 >= 0, ...)`.
///
/// Dynamic dims become IntegerSet SYMBOLS bound positionally to the op's dynamic
/// sizes. The dim/symbol line is the mutable/immutable line, which is why a
/// dynamic extent may not be printed as a constant.
///
/// `pub(crate)` because `to_ktir`'s rung-3 rewrite states the SAME set for its `[1,1]`
/// scalar view -- one spelling of "the coordinates a tile may take", not two that
/// could drift.
pub(crate) fn build_range_set_nd(shape: &[i64]) -> String {
    let rank = shape.len();
    let dims: Vec<String> = (0..rank).map(|i| format!("d{i}")).collect();
    let mut sym = 0usize;
    let mut cons = Vec::new();
    for (i, s) in shape.iter().enumerate() {
        cons.push(format!("d{i} >= 0"));
        if *s == DYNAMIC {
            // MLIR NORMALISES THE AFFINE EXPRESSION, and the diff compares the printed body:
            // `s0 - 1 - d0` prints as `-d0 + s0 - 1`, negated dim first. Measured in
            // `test/goldens/ktir/vector_add/1_ktir.mlir`'s `#set`, which is the only
            // configuration in the tree with a runtime extent -- every Granite extent is
            // constexpr, so nothing else exercises this spelling.
            cons.push(format!("-d{i} + s{sym} - 1 >= 0"));
            sym += 1;
        } else {
            // MLIR prints `upper - d_i >= 0` with the constant folded in as
            // `-d_i + (N-1) >= 0`.
            cons.push(format!("-d{i} + {} >= 0", s - 1));
        }
    }
    let syms: Vec<String> = (0..sym).map(|i| format!("s{i}")).collect();
    let head = if syms.is_empty() {
        format!("({})", dims.join(", "))
    } else {
        format!("({})[{}]", dims.join(", "), syms.join(", "))
    };
    format!("{head} : ({})", cons.join(", "))
}

/// `AffineMap::getMultiDimIdentityMap`.
fn identity_map(rank: usize) -> String {
    let dims: Vec<String> = (0..rank).map(|i| format!("d{i}")).collect();
    format!("({}) -> ({})", dims.join(", "), dims.join(", "))
}

/// A permutation map: `order[i]` is the BLOCK dim that becomes result dim i
/// (`tt.trans`'s own convention).
fn order_map(rank: usize, order: &[i64]) -> String {
    let dims: Vec<String> = (0..rank).map(|i| format!("d{i}")).collect();
    let res: Vec<String> = order.iter().map(|d| format!("d{d}")).collect();
    format!("({}) -> ({})", dims.join(", "), res.join(", "))
}

//===----------------------------------------------------------------------===//
// Precondition + the access-op conversion
//===----------------------------------------------------------------------===//

/// Every remaining access op's `desc` operand must be a memref-backed descriptor
/// produced by walk 1. The remaining failure mode is a descriptor sourced from a
/// function argument, whose shape/stride info there is no way to recover.
fn precondition_check(module: &Module) -> Result<()> {
    for op in module.ops_deep() {
        let is_access = matches!(
            op.kind,
            OpKind::TtDescriptorLoad
                | OpKind::TtDescriptorStore
                | OpKind::TtDescriptorGather
                | OpKind::TtDescriptorScatter
        );
        if !is_access {
            continue;
        }
        let desc = op.operands.first().copied();
        let ok = desc
            .and_then(|v| module.def_of(v))
            .map(|d| {
                d.kind == OpKind::UnrealizedConversionCast
                    && d.operands
                        .first()
                        .and_then(|v| module.type_of(*v))
                        .map(|t| matches!(t, IrType::MemRef { .. }))
                        .unwrap_or(false)
            })
            .unwrap_or(false);
        if !ok {
            return Err(Refusal::new(
                PASS,
                "cannot lower descriptor op: shape and stride info is only available when \
                 the descriptor is defined by tt.make_tensor_descriptor in the same block",
            ));
        }
    }
    Ok(())
}

/// The memref behind an access op's `desc` operand.
fn descriptor_mem_view(module: &Module, access: &Op) -> Option<Ssa> {
    let desc = access.operands.first().copied()?;
    let cast = module.def_of(desc)?;
    cast.operands.first().copied()
}

fn convert_access_ops(module: &mut Module) -> Result<()> {
    while let Some(path) = walk::paths(module).into_iter().find(|p| {
        walk::at(module, p)
            .map(|o| {
                matches!(o.kind, OpKind::TtDescriptorLoad | OpKind::TtDescriptorStore)
            })
            .unwrap_or(false)
    }) {
        let op = walk::at(module, &path).expect("path").clone();
        let view = descriptor_mem_view(module, &op)
            .ok_or_else(|| Refusal::new(PASS, "descriptor operand was not lowered by walk 1"))?;

        // BLOCK SHAPE COMES FROM THE DESCRIPTOR'S TYPE, not the result tensor. A
        // rank-reduced load would otherwise build a 2-D access tile that does not
        // match the 3-D memory view -- IR that verifies here and is wrong.
        let desc_ty = op
            .operands
            .first()
            .and_then(|v| module.type_of(*v))
            .ok_or_else(|| Refusal::new(PASS, "descriptor operand has no type"))?;
        let block_shape = desc_ty.dims().map(|d| d.to_vec()).unwrap_or_default();

        let is_load = op.kind == OpKind::TtDescriptorLoad;
        // A store's operands are [desc, indices..., src]; a load's are
        // [desc, indices...].
        let indices: Vec<Ssa> = if is_load {
            op.operands[1..].to_vec()
        } else {
            op.operands[1..op.operands.len() - 1].to_vec()
        };
        let src = if is_load { None } else { op.operands.last().copied() };

        let hint = op.result().map(|r| module.hint(r)).unwrap_or_else(|| "0".into());
        let (casts, tile_op) =
            build_direct_access_tile(module, view, &block_shape, &indices, &[], &hint);
        let tile = tile_op.result().expect("the tile defines a value");

        let mut replacement: Vec<Op> = casts;
        replacement.push(tile_op);
        if is_load {
            let res = op.result().expect("a descriptor_load defines a value");
            let ty = op.result_type().cloned().ok_or_else(|| {
                Refusal::new(PASS, "tt.descriptor_load has no result type")
            })?;
            replacement
                .push(Op::new(OpKind::KtdpLoad).with_result(res, ty).with_operands([tile]));
        } else {
            replacement.push(
                Op::new(OpKind::KtdpStore).with_operands([src.expect("a store has a source"), tile]),
            );
        }

        let idx = path.index();
        let block = walk::block_mut(module, &path).expect("path");
        block.remove(idx);
        for (k, nop) in replacement.into_iter().enumerate() {
            block.insert(idx + k, nop);
        }
    }
    Ok(())
}

/// `buildDirectAccessTile`. The memory view describes the full tensor; the block
/// indices position the tile within it. `order`, when non-empty, is a PERMUTATION
/// of the block dims -- the tile is READ transposed instead of a physical
/// transpose being materialized. That is free on Spyre.
///
/// THE SET STAYS over the block's own dims in memory order; only the ORDER map --
/// and hence the tile's logical shape -- changes for a transposed read.
fn build_direct_access_tile(
    module: &mut Module,
    view: Ssa,
    block_shape: &[i64],
    indices: &[Ssa],
    order: &[i64],
    hint: &str,
) -> (Vec<Op>, Op) {
    let rank = block_shape.len();
    let mut tile_shape = block_shape.to_vec();
    let mut order_str = identity_map(rank);
    if !order.is_empty() {
        tile_shape = order.iter().map(|d| block_shape[*d as usize]).collect();
        order_str = order_map(rank, order);
    }

    // Index operands arrive as i32 from Triton and are cast to index.
    let mut casts = Vec::new();
    let mut index_operands = Vec::new();
    for idx in indices {
        let ty = module.type_of(*idx);
        if matches!(ty, Some(IrType::Index)) {
            index_operands.push(*idx);
            continue;
        }
        let v = module.fresh_named(hint);
        casts.push(
            Op::new(OpKind::ArithIndexCast)
                .with_result(v, IrType::Index)
                .with_operands([*idx]),
        );
        index_operands.push(v);
    }

    let tile = module.fresh_named(hint);
    let mut operands = vec![view];
    operands.extend(index_operands);
    let tile_op = Op::new(OpKind::KtdpConstructAccessTile)
        .with_result(tile, IrType::AccessTile { dims: tile_shape })
        .with_operands(operands)
        .with_attr(AttrKey::BaseMap, Attr::AffineMap(identity_map(rank)))
        .with_attr(AttrKey::AccessTileSet, Attr::AffineSet(build_range_set_nd(block_shape)))
        .with_attr(AttrKey::AccessTileOrder, Attr::AffineMap(order_str));
    (casts, tile_op)
}

//===----------------------------------------------------------------------===//
// FoldTransIntoAccessTileOrder
//===----------------------------------------------------------------------===//

/// Fold `tt.trans` of a tile load into that load's `access_tile_order`.
///
/// Only folds onto an as-built IDENTITY order: composing two permutations is
/// valid, but nothing in this pass produces one today, so refuse rather than emit
/// a composition no test covers.
///
/// The original load and tile are ERASED EXPLICITLY. `ktdp.load` declares no
/// memory effects, so a greedy driver would not DCE it -- leaving the untransposed
/// read in the IR beside the transposed one, which is both wasted traffic and a
/// confusing artifact. Safe because both are checked for a single use.
fn fold_trans_into_access_tile_order(module: &mut Module) {
    loop {
        let mut folded = false;
        for path in walk::paths(module) {
            let Some(op) = walk::at(module, &path) else { continue };
            if op.kind != OpKind::TtTrans {
                continue;
            }
            let trans = op.clone();
            let Some(src) = trans.operands.first().copied() else { continue };
            let Some(load) = module.def_of(src).filter(|o| o.kind == OpKind::KtdpLoad).cloned()
            else {
                continue;
            };
            // Single use of the load's result.
            if count_uses(module, src) != 1 {
                continue;
            }
            let Some(tile_v) = load.operands.first().copied() else { continue };
            let Some(tile) = module
                .def_of(tile_v)
                .filter(|o| o.kind == OpKind::KtdpConstructAccessTile)
                .cloned()
            else {
                continue;
            };
            if count_uses(module, tile_v) != 1 {
                continue;
            }
            let Some(Attr::IntList(order)) = trans.attr(&AttrKey::Order).cloned() else {
                continue;
            };
            let block_shape = match tile.result_type().and_then(|t| t.dims()) {
                Some(d) => d.to_vec(),
                None => continue,
            };
            let rank = block_shape.len();
            if order.len() != rank || order.iter().any(|d| *d < 0 || *d as usize >= rank) {
                continue;
            }
            // Only an identity as-built order.
            let cur = tile.attr(&AttrKey::AccessTileOrder).and_then(|a| match a {
                Attr::AffineMap(s) => Some(s.clone()),
                _ => None,
            });
            if cur.as_deref() != Some(identity_map(rank).as_str()) {
                continue;
            }

            let new_shape: Vec<i64> = order.iter().map(|d| block_shape[*d as usize]).collect();
            let mut new_tile = tile.clone();
            let tile_out = new_tile.results[0];
            new_tile.result_types[0] = IrType::AccessTile { dims: new_shape };
            new_tile.set_attr(AttrKey::AccessTileOrder, Attr::AffineMap(order_map(rank, &order)));

            let mut new_load = load.clone();
            new_load.results = trans.results.clone();
            new_load.result_types = trans.result_types.clone();
            new_load.operands = vec![tile_out];

            // Replace the trans in place with the new load, then delete the old
            // load and tile.
            let tile_path = find_path_of_result(module, tile_v).expect("the tile has a path");
            let load_path = find_path_of_result(module, src).expect("the load has a path");
            {
                let idx = path.index();
                let block = walk::block_mut(module, &path).expect("path");
                block[idx] = new_load;
            }
            {
                let idx = load_path.index();
                let block = walk::block_mut(module, &load_path).expect("path");
                block[idx] = new_tile;
            }
            walk::erase(module, &[tile_path]);
            folded = true;
            break;
        }
        if !folded {
            break;
        }
    }
}

fn count_uses(module: &Module, v: Ssa) -> usize {
    module
        .ops_deep()
        .iter()
        .map(|o| o.operands.iter().filter(|x| **x == v).count())
        .sum()
}

fn find_path_of_result(module: &Module, v: Ssa) -> Option<OpPath> {
    walk::paths(module)
        .into_iter()
        .find(|p| walk::at(module, p).map(|o| o.results.contains(&v)).unwrap_or(false))
}

//===----------------------------------------------------------------------===//
// Fail closed on raw tile access
//===----------------------------------------------------------------------===//

/// Scoped to TILE access -- a load/store whose value is a TENSOR. A raw SCALAR
/// pointer read is legitimate and needs no access tile.
fn refuse_raw_tile_access(module: &Module) -> Result<()> {
    for op in module.ops_deep() {
        let shaped = match op.kind {
            OpKind::TtLoad => op.result_type().map(|t| t.dims().is_some()).unwrap_or(false),
            OpKind::TtStore => op
                .operands
                .last()
                .and_then(|v| module.type_of(*v))
                .map(|t| t.dims().is_some())
                .unwrap_or(false),
            _ => false,
        };
        if shaped {
            return Err(Refusal::new(
                PASS,
                format!(
                    "'{}' is raw-pointer TILE access, which this pass cannot physicalize -- \
                     it converts only the descriptor forms \
                     (tt.descriptor_load/store/gather/scatter, from \
                     tl.make_tensor_descriptor). Leaving it untouched emits no access tile \
                     and no memory view, and the failure then surfaces three stages later \
                     as an unexplained missing-tile error in PlanCorelets. Express the \
                     access with tl.make_tensor_descriptor, or add raw-pointer support to \
                     this pass. (A raw SCALAR pointer read is fine and is not what this \
                     refuses.)",
                    op.kind.spelling()
                ),
            ));
        }
    }
    Ok(())
}


//===----------------------------------------------------------------------===//
// The gather: tt.descriptor_gather -> ktdp.construct_indirect_access_tile
//===----------------------------------------------------------------------===//

/// The subscript kinds, subscript maps, variable-space set and order for a rank-`K`
/// `x_offsets` by rank-`R` block gather.
///
/// PORTED FROM `buildGatherSubscriptMaps` (`ConvertTTIRToKTDP.cpp:581`), and the shape is
/// its, not ours. The affine map domain puts the CAPTURED SCALARS FIRST -- `c_x0..c_x{K-1}`
/// then `c_y` -- followed by one iteration variable per result dim, so a rank-1 index over a
/// rank-2 result has a four-dim domain `(d0, d1, d2, d3)` = `(c_x0, c_y, d_0, d_1)`:
///
/// ```text
///   base dim 0   INDIRECT   (d0, d1, d2, d3) -> (d0 + d2)      c_x0 + d_0
///   base dim 1   direct     (d0, d1, d2, d3) -> (d1 + d3)      c_y  + d_1
///   base dim i   direct     d_{K + i - 1}                      no offset
/// ```
///
/// The maps and the set are built as TEXT because `ir::Attr` keeps an affine map as its
/// printed body -- and the spelling has to be MLIR's exactly, because the golden diff
/// compares it against what `text::parse` read out of the C++'s own output. That is why the
/// upper bound prints `-d0 + 127` rather than `127 - d0`.
fn gather_subscripts(
    index_rank: usize,
    result_shape: &[i64],
) -> Result<(Attr, Attr, Attr, Attr)> {
    let result_rank = result_shape.len();
    if result_rank < 2 {
        return Err(Refusal::new(
            PASS,
            format!(
                "tt.descriptor_gather over a rank-{result_rank} result; an indirect access \
                 tile requires rank >= 2 (`buildGatherSubscriptMaps` asserts it)"
            ),
        ));
    }
    if index_rank < 1 || index_rank >= result_rank {
        return Err(Refusal::new(
            PASS,
            format!(
                "tt.descriptor_gather has a rank-{index_rank} index grid against a \
                 rank-{result_rank} result; K must lie in [1, R) -- K = R - blockRank + 1 \
                 with blockRank >= 2"
            ),
        ));
    }
    let k = index_rank;
    let block_rank = result_rank - k + 1;
    let dim_count = k + 1 + result_rank;
    let d0 = k + 1;

    let dims = |n: usize| -> String {
        (0..n).map(|i| format!("d{i}")).collect::<Vec<_>>().join(", ")
    };
    let domain = dims(dim_count);

    let mut kinds: Vec<String> = Vec::with_capacity(block_rank);
    let mut maps: Vec<String> = Vec::with_capacity(block_rank);

    // base dim 0: indirect, one address component per index-grid axis.
    let addr: Vec<String> = (0..k).map(|j| format!("d{j} + d{}", d0 + j)).collect();
    kinds.push("true".to_string());
    maps.push(format!("({domain}) -> ({})", addr.join(", ")));

    // base dim 1: direct, `c_y + d_K`. Block rank >= 2 makes it always present.
    kinds.push("false".to_string());
    maps.push(format!("({domain}) -> (d{k} + d{})", d0 + k));

    // base dims [2, R): direct, no offset.
    for i in 2..block_rank {
        kinds.push("false".to_string());
        maps.push(format!("({domain}) -> (d{})", d0 + k + i - 1));
    }

    // The intermediate-variable space: `0 <= d_i < result_shape[i]`.
    let space_dims = dims(result_rank);
    let mut constraints: Vec<String> = Vec::with_capacity(2 * result_rank);
    for (i, extent) in result_shape.iter().enumerate() {
        constraints.push(format!("d{i} >= 0"));
        constraints.push(format!("-d{i} + {} >= 0", extent - 1));
    }
    Ok((
        Attr::StrList(kinds),
        Attr::AffineMapList(maps),
        Attr::AffineSet(format!("({space_dims}) : ({})", constraints.join(", "))),
        Attr::AffineMap(format!("({space_dims}) -> ({space_dims})")),
    ))
}

/// Trace a gather's `x_offsets` back to the memory view its indices live in, and to the
/// anchors the index tile was built from.
///
/// `traceToSourceMemoryView` (`ConvertTTIRToKTDP.cpp:507`). The `x_offsets` operand is the
/// result of an ALREADY-LOWERED `tt.descriptor_load` -- a `ktdp.load` of a
/// `ktdp.construct_access_tile` -- so the index view is that tile's base and the anchors are
/// its indices. A trace miss is a refusal: there is no view to point the indirect tile at.
fn trace_index_view(module: &Module, x_offsets: Ssa) -> Result<(Ssa, Vec<Ssa>)> {
    let load = module
        .def_of(x_offsets)
        .filter(|o| o.kind == OpKind::KtdpLoad)
        .ok_or_else(|| {
            Refusal::new(
                PASS,
                "tt.descriptor_gather's x_offsets is not the result of a lowered descriptor \
                 load, so the memory view its indices live in cannot be recovered. The \
                 indices must be read through a descriptor in the same block; a \
                 tensor-typed function argument has no view to point at.",
            )
        })?;
    let tile_val = load
        .operands
        .first()
        .copied()
        .ok_or_else(|| Refusal::new(PASS, "ktdp.load with no access tile operand"))?;
    let tile = module
        .def_of(tile_val)
        .filter(|o| o.kind == OpKind::KtdpConstructAccessTile)
        .ok_or_else(|| {
            Refusal::new(
                PASS,
                "tt.descriptor_gather's x_offsets is loaded from something that is not a \
                 direct ktdp.construct_access_tile",
            )
        })?;
    let base = tile
        .operands
        .first()
        .copied()
        .ok_or_else(|| Refusal::new(PASS, "ktdp.construct_access_tile with no base"))?;
    Ok((base, tile.operands[1..].to_vec()))
}

/// Lower every `tt.descriptor_gather` to `ktdp.construct_indirect_access_tile` + `ktdp.load`.
///
/// `tt.descriptor_scatter` is deliberately NOT ported. The C++ has it
/// (`ConvertDescriptorScatter`, the same helper with a `ktdp.store`), and no fixture in this
/// tree emits one -- so porting it would be untested code that looks tested. An unconverted
/// descriptor op is refused by name by [`refuse_raw_tile_access`]'s sibling checks, so the
/// omission fails closed rather than passing something through.
fn convert_gathers(module: &mut Module) -> Result<()> {
    loop {
        let Some(path) = walk::paths(module).into_iter().find(|p| {
            walk::at(module, p)
                .map(|o| o.kind == OpKind::TtDescriptorGather)
                .unwrap_or(false)
        }) else {
            return Ok(());
        };
        let op = walk::at(module, &path).expect("path").clone();
        // `tt.descriptor_gather %desc[%x_offsets, %y_offset]`.
        if op.operands.len() != 3 {
            return Err(Refusal::new(
                PASS,
                format!(
                    "tt.descriptor_gather takes a descriptor, an x_offsets tensor and a \
                     y_offset; found {} operand(s)",
                    op.operands.len()
                ),
            ));
        }
        let view = descriptor_mem_view(module, &op).ok_or_else(|| {
            Refusal::new(PASS, "the gather's descriptor was not lowered by walk 1")
        })?;
        let (index_view, anchors) = trace_index_view(module, op.operands[1])?;
        let result_ty = op
            .result_type()
            .cloned()
            .ok_or_else(|| Refusal::new(PASS, "tt.descriptor_gather has no result type"))?;
        let result_shape = result_ty.dims().map(|d| d.to_vec()).unwrap_or_default();
        let (kinds, maps, space_set, space_order) =
            gather_subscripts(anchors.len(), &result_shape)?;

        let hint = op.result().map(|r| module.hint(r)).unwrap_or_else(|| "gather".into());
        // The y_offset arrives as i32 and is cast to index, exactly as the direct path casts
        // its own indices. `canonicalize` folds the cast of a constant afterwards, which is
        // why the golden shows a bare `%c0 : index`.
        let y = op.operands[2];
        let mut casts = Vec::new();
        let y_index = match module.type_of(y) {
            Some(IrType::Index) => y,
            _ => {
                let v = module.fresh_named(&hint);
                casts.push(
                    Op::new(OpKind::ArithIndexCast)
                        .with_result(v, IrType::Index)
                        .with_operands([y]),
                );
                v
            }
        };

        // OPERAND ORDER IS THE PRINTED ORDER: base, the captured anchors, `c_y`, then the
        // indirect memref. It has to be, because the golden side of the diff is built by
        // `text::parse`, which reads `%tokens` left to right off that printed form -- and
        // `triton-superdsc-lower` reads `operands.last()` as the index view on the same basis.
        let mut operands = vec![view];
        operands.extend(anchors.iter().copied());
        operands.push(y_index);
        operands.push(index_view);

        // One block argument per captured variable: the K anchors, then `c_y`.
        let n_captured = anchors.len() + 1;
        let region_args: Vec<(Ssa, IrType)> = (0..n_captured)
            .map(|_| (module.fresh(), IrType::Index))
            .collect();

        let tile_val = module.fresh_named(&hint);
        let tile = Op::new(OpKind::KtdpConstructIndirectAccessTile)
            .with_result(tile_val, IrType::AccessTile { dims: result_shape })
            .with_operands(operands)
            .with_attr(AttrKey::Other("per_dim_subscript_kinds".into()), kinds)
            .with_attr(AttrKey::Other("per_dim_subscript_maps".into()), maps)
            .with_attr(AttrKey::Other("variables_space_order".into()), space_order)
            .with_attr(AttrKey::Other("variables_space_set".into()), space_set)
            // THE REGION HAS ONE `index` BLOCK ARGUMENT PER CAPTURED VARIABLE, and no ops.
            //
            // The printed form is `{ ^bb0(%a: index, %b: index): }` -- so for K anchors plus
            // `c_y` there are K+1 of them. Emitting the region EMPTY was the first attempt and
            // the diff caught it by name: `block argument types differs, golden ["index",
            // "index"], ours []`. Worth recording, because the assumption behind that attempt
            // was that `text::parse` skips the `^bb0` line and therefore the golden side would
            // have no args either -- it does read them, and guessing rather than checking is
            // what the diff is for.
            .with_region(Region { args: region_args, ops: Vec::new() });

        let res = op.result().expect("a gather defines a value");
        let load = Op::new(OpKind::KtdpLoad)
            .with_result(res, result_ty)
            .with_operands([tile_val]);

        let mut replacement = casts;
        replacement.push(tile);
        replacement.push(load);

        let idx = path.index();
        let block = walk::block_mut(module, &path).expect("path");
        block.remove(idx);
        for (k, nop) in replacement.into_iter().enumerate() {
            block.insert(idx + k, nop);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_range_set_matches_mlir_printing() {
        // The set the golden carries for a 512x128 view.
        assert_eq!(
            build_range_set_nd(&[512, 128]),
            "(d0, d1) : (d0 >= 0, -d0 + 511 >= 0, d1 >= 0, -d1 + 127 >= 0)"
        );
        assert_eq!(
            build_range_set_nd(&[64, 128]),
            "(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)"
        );
    }

    #[test]
    fn a_dynamic_extent_becomes_a_symbol_not_a_constant() {
        // The dim/symbol line IS the mutable/immutable line: a runtime extent may
        // not be printed as a constant, or a recompile-per-shape assumption is
        // baked into a set that claims to be shape-generic.
        let s = build_range_set_nd(&[DYNAMIC, 64]);
        assert_eq!(s, "(d0, d1)[s0] : (d0 >= 0, -d0 + s0 - 1 >= 0, d1 >= 0, -d1 + 63 >= 0)");
    }

    #[test]
    fn the_maps_match_mlir_printing() {
        assert_eq!(identity_map(2), "(d0, d1) -> (d0, d1)");
        assert_eq!(order_map(2, &[1, 0]), "(d0, d1) -> (d1, d0)");
    }
}
