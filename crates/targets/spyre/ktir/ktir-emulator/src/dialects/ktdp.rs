// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! `ktdp` dialect handlers — partial port of `ktir_emulator/dialects/ktdp_ops.py`.
//!
//! This slice ports the two construct ops that build the memory-view types,
//! single-allocation path only:
//!   * `construct_memory_view`  -> `MemRef`   (logical view; does NOT allocate)
//!   * `construct_access_tile`  -> `AccessTile` over a `TileRef`
//!
//! The distributed path (`construct_distributed_memory_view`,
//! `distributed_tile_access`) and `load`/`store` follow once `memory.rs` grows
//! a real `HBMSimulator`. Symbolic access-tile sets are rejected here, matching
//! the Python handler's `NotImplementedError`.

use super::{Dispatch, LatencyCategory};
use crate::affine::AffineMap;
use crate::attrkey::AttrKey;
use crate::context::CoreContext;
use crate::dtypes::DType;
use crate::env::ExecutionEnv;
use crate::ir::{Attr, Operation, Scalar, Value};
use crate::memref::{AccessTile, DistributedMemRef, MemRef, MemorySpace, ParentRef, TileRef};
use crate::opkind::OpKind;
use crate::ops_memory::distributed_tile_access;

pub fn register(d: &mut Dispatch) {
    d.register(
        OpKind::KtdpConstructMemoryView,
        LatencyCategory::Zero,
        construct_memory_view,
    );
    d.register(
        OpKind::KtdpConstructAccessTile,
        LatencyCategory::Zero,
        construct_access_tile,
    );
}

/// HBM `construct_memory_view` ops seen, and how many were built over a pointer that is
/// NOT a tensor's own allocation base. The bounds check can only validate the first kind,
/// so this is the lock's COVERAGE; and an off-base view is itself the SuperDSC habit KTIR
/// does not have — a KTIR tensor is a buffer with one shape, not an address.
pub static VIEW_HBM_TOTAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static VIEW_HBM_OFF_BASE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Access tiles seen, and how many the bounds check could NOT compare (rank mismatch
/// between the window and its parent view). Same reason as the view counters: a check that
/// skips most of its subjects must not read as a pass.
pub static TILE_TOTAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static TILE_UNCHECKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `%v = ktdp.construct_memory_view %ptr {shape, strides, memory_space, dtype, ...}`
///
/// Builds a logical `MemRef`. Mirrors `tile_view`. Slice limitation: shape /
/// strides must be static (the Python parser also stores dynamic dims as SSA
/// names resolved at runtime — that resolution lands with grid/scope support).
fn construct_memory_view(
    op: &Operation<'static>,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    if op.operands.is_empty() {
        return Err("construct_memory_view: missing pointer operand".into());
    }
    let base_ptr = scalar_i64(ctx.get_value(op.operands[0])?, "construct_memory_view ptr")?;

    // Static `shape` (IntList) or dynamic `sizes_dyn` — the VALUES that carry the
    // sizes, read out of scope now. Mirrors the runtime SSA-size resolution in
    // `ktdp__construct_memory_view`.
    let shape: Vec<usize> = match op.attr(AttrKey::Shape) {
        Some(Attr::IntList(v)) => v.iter().map(|&n| n as usize).collect(),
        _ => match op.attr(AttrKey::SizesDyn) {
            Some(Attr::Ssas(sizes)) => sizes
                .iter()
                .map(|&v| {
                    scalar_i64(ctx.get_value(v)?, "construct_memory_view size").map(|n| n as usize)
                })
                .collect::<Result<_, String>>()?,
            _ => return Err("construct_memory_view: missing required attribute 'shape'".into()),
        },
    };
    let strides = int_list(op, AttrKey::Strides)?.to_vec();

    let space_str = str_attr(op, AttrKey::MemorySpace)?;
    let core_id = match op.attr(AttrKey::LxCoreId) {
        Some(Attr::Int(n)) => Some(*n as u32),
        _ => None,
    };
    let space = MemorySpace::parse(space_str, core_id)?;

    let dtype = dtype_attr(op, AttrKey::Dtype)?;

    let coordinate_set = match op.attr(AttrKey::CoordinateSet) {
        Some(Attr::AffineSet(s)) => Some(s.clone()),
        _ => None,
    };

    // ⭐ A VIEW MAY NOT CLAIM MORE THAN ITS ALLOCATION HOLDS.
    //
    // The emulator zero-pads a read past the end of an allocation, so an over-long memref
    // computes quietly here and only fails on real hardware, where the extent is the
    // addressable footprint. Refusing it makes the emulator reject what the card would.
    //
    // Checked for HBM only, and only at a tensor's OWN allocation base: an LX view is
    // routinely built BEFORE the bump allocator has placed its data, and an interior
    // pointer cannot be attributed to a tensor once a grown allocation overlaps the next
    // one. Both are limits on when the fact EXISTS, not exemptions from the rule.
    // ⚠️ `base_ptr` is an ELEMENT index (RFC #110), and the allocation map is keyed by BYTE
    // address — `byte_address = base_ptr * bpe`. Comparing the two directly is the
    // stick-vs-byte unit error, and it reports every view as off-base.
    let byte_addr = base_ptr * dtype.bytes_per_elem() as i64;
    if space == MemorySpace::Hbm {
        let at_base = ctx.hbm.borrow().allocation_len_at_base(byte_addr).is_some();
        VIEW_HBM_TOTAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if !at_base {
            VIEW_HBM_OFF_BASE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    if space == MemorySpace::Hbm
        && let Some(len) = ctx.hbm.borrow().allocation_len_at_base(byte_addr)
    {
        // Highest element the view can address: sum of (extent-1)*stride over the dims.
        // Strides are in ELEMENTS (the lowering states `[cols, 1]` for a row-major view).
        let last_elem: i64 = shape
            .iter()
            .zip(&strides)
            .map(|(&d, &s)| (d as i64 - 1).max(0) * s)
            .sum();
        let need = (last_elem + 1) * dtype.bytes_per_elem() as i64;
        if need > len as i64 {
            return Err(format!(
                "construct_memory_view: view {shape:?} strides {strides:?} spans {need} \
                 bytes but the tensor allocated at byte {byte_addr} holds only {len} — the \
                 hardware addresses the declared extent, so this view is out of bounds \
                 there even though HBM zero-pads it here"
            ));
        }
    }

    Ok(Some(Value::MemRef(MemRef {
        base_ptr,
        shape,
        strides,
        space,
        dtype,
        coordinate_set,
    })))
}

/// `%t = ktdp.construct_access_tile %view, %i, %j {shape, base_map, ...}`
///
/// Single-allocation path: evaluate `base_map` at the indices to get base
/// coords, fold them through the parent strides into a byte offset, and wrap
/// the resulting `TileRef` in an `AccessTile`. Mirrors `tile_access`.
fn construct_access_tile(
    op: &Operation<'static>,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    if op.operands.is_empty() {
        return Err("construct_access_tile: missing parent operand".into());
    }
    // Parent is a single-allocation MemRef or a distributed view; clone the
    // relevant one so we can drop the borrow before reading the index operands.
    enum Parent {
        Single(MemRef),
        Dist(DistributedMemRef),
    }
    let parent = match ctx.get_value(op.operands[0])? {
        Value::MemRef(m) => Parent::Single(m.clone()),
        Value::DistMemRef(d) => Parent::Dist(d.clone()),
        other => {
            return Err(format!(
                "construct_access_tile: parent is {other:?}, expected MemRef"
            ));
        }
    };

    let indices: Vec<i64> = op.operands[1..]
        .iter()
        .map(|&v| {
            ctx.get_value(v)
                .and_then(|v| scalar_i64(v, "construct_access_tile index"))
        })
        .collect::<Result<_, _>>()?;

    let access_shape = int_list(op, AttrKey::Shape)?
        .iter()
        .map(|&n| n as usize)
        .collect::<Vec<_>>();

    // base_map is always present (synthesized as identity upstream if absent).
    let base_map = match op.attr(AttrKey::BaseMap) {
        Some(Attr::AffineMap(m)) => m.clone(),
        _ => identity_map(indices.len())?,
    };

    let coordinate_set = match op.attr(AttrKey::CoordinateSet) {
        Some(Attr::AffineSet(s)) => Some(s.clone()),
        _ => None,
    };

    // Single allocation -> direct tile_access; distributed view -> resolve
    // partition routing now via distributed_tile_access (mirrors ktdp__construct_access_tile).
    let parent_ref = match parent {
        Parent::Single(m) => {
            // ⭐ THE WINDOW MAY NOT LEAVE THE VIEW. The per-op window lives in the ACCESS
            // TILE, so this is the tile-side analogue of the view bounds check: a tile
            // reaching past its parent's declared extent reads bytes the buffer does not
            // have. The emulator zero-pads that; hardware addresses the declared extent.
            //
            // Checked only at EQUAL RANK, where dim i of the window corresponds to dim i of
            // the parent with no reshape or rank-reduction to interpret. Unequal rank is
            // counted, not silently passed, so the coverage is visible.
            let base_coords = base_map.eval(&indices, &[]);
            TILE_TOTAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if access_shape.len() == m.shape.len() && base_coords.len() == m.shape.len() {
                for (i, (&ext, &parent_ext)) in access_shape.iter().zip(&m.shape).enumerate() {
                    let end = base_coords[i] + ext as i64;
                    if end > parent_ext as i64 {
                        return Err(format!(
                            "construct_access_tile: window {access_shape:?} at base \
                             {base_coords:?} runs to {end} in dim {i} of a parent view \
                             {:?} — the tile leaves its view, which the hardware addresses \
                             as declared even though the emulator zero-pads it",
                            m.shape
                        ));
                    }
                }
            } else {
                TILE_UNCHECKED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            ParentRef::Tile(tile_access(m, &indices, access_shape.clone(), &base_map))
        }
        Parent::Dist(d) => {
            let dist = distributed_tile_access(
                &d,
                &access_shape,
                &base_map,
                &indices,
                coordinate_set.as_ref(),
            )?;
            ParentRef::Dist(dist)
        }
    };

    Ok(Some(Value::AccessTile(AccessTile {
        parent_ref,
        shape: access_shape,
        base_map,
        coordinate_set,
        coordinate_order: None, // access_tile_order parsing lands with the parser slice
    })))
}

/// Port of `MemoryOps.tile_access`: indices -> base coords (via base_map) ->
/// byte offset (via parent strides) -> byte-addressed `TileRef`.
fn tile_access(
    parent: MemRef,
    indices: &[i64],
    access_shape: Vec<usize>,
    base_map: &AffineMap,
) -> TileRef {
    let base_coords = base_map.eval(indices, &[]);
    let bpe = parent.dtype.bytes_per_elem() as i64;
    let offset_elems: i64 = base_coords
        .iter()
        .zip(&parent.strides)
        .map(|(coord, stride)| coord * stride)
        .sum();
    let byte_pos = parent.byte_address() + offset_elems * bpe;

    // Take parent's fields, then MOVE it into the box — no second clone of the
    // MemRef (and its affine `coordinate_set`), which `construct_access_tile`
    // already paid once. Halves the per-access affine clone/drop churn.
    let strides = parent.strides.clone();
    let dtype = parent.dtype;
    TileRef {
        base_ptr: byte_pos,
        shape: access_shape,
        strides,
        dtype,
        memref: Box::new(parent),
        coordinate_set: None,
        partition_origin: None,
    }
}

// --- attribute helpers ---------------------------------------------------

fn int_list<'a>(op: &Operation<'a>, key: AttrKey) -> Result<&'a [i64], String> {
    match op.attr(key) {
        Some(Attr::IntList(v)) => Ok(v),
        Some(other) => Err(format!(
            "{:?}: attr {key:?} is {other:?}, expected IntList",
            op.op_type
        )),
        None => Err(format!(
            "{:?}: missing required attribute {key:?}",
            op.op_type
        )),
    }
}

fn str_attr<'a>(op: &Operation<'a>, key: AttrKey) -> Result<&'a str, String> {
    match op.attr(key) {
        Some(Attr::Str(s)) => Ok(s),
        _ => Err(format!(
            "{:?}: missing/invalid string attribute {key:?}",
            op.op_type
        )),
    }
}

fn dtype_attr(op: &Operation, key: AttrKey) -> Result<DType, String> {
    match op.attr(key) {
        Some(Attr::Dtype(d)) => Ok(*d),
        _ => Err(format!(
            "{:?}: missing/invalid dtype attribute {key:?}",
            op.op_type
        )),
    }
}

/// `Dim(0)..Dim(N)` as one `'static` table — the expressions a SYNTHESIZED identity
/// map is made of. MLIR may omit `base_map`, in which case it is the identity over
/// the access tile's indices (RFC 0682); a handler has no arena to build that in, so
/// the ranks it can synthesize are declared here.
static IDENTITY_DIMS: [crate::affine::AffineExpr<'static>; MAX_IDENTITY_RANK] = {
    let mut dims = [crate::affine::AffineExpr::Dim(0); MAX_IDENTITY_RANK];
    let mut i = 0;
    while i < MAX_IDENTITY_RANK {
        dims[i] = crate::affine::AffineExpr::Dim(i);
        i += 1;
    }
    dims
};

/// The highest rank an omitted `base_map` can be synthesized at. An access tile's
/// rank is its shape's rank, which the tile hardware bounds well below this.
const MAX_IDENTITY_RANK: usize = 16;

/// The identity map of `rank`, over the `'static` [`IDENTITY_DIMS`] table.
pub fn identity_map(rank: usize) -> Result<AffineMap<'static>, String> {
    if rank > MAX_IDENTITY_RANK {
        return Err(format!(
            "base_map omitted at rank {rank}, above the {MAX_IDENTITY_RANK} an identity \
             can be synthesized at"
        ));
    }
    Ok(AffineMap::identity_in(&IDENTITY_DIMS[..rank]))
}

fn scalar_i64(v: &Value, ctx: &str) -> Result<i64, String> {
    match v {
        Value::Index(i) => Ok(*i),
        Value::Scalar(Scalar::I32(i)) => Ok(*i as i64),
        Value::Scalar(Scalar::I64(i)) => Ok(*i),
        other => Err(format!("{ctx}: expected index/int, got {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialects::Dispatch;
    use crate::env::{ExecutionEnv, GridExecutor};
    use crate::interpreter::{execute_ops, single_core_context};

    fn run(ops: &[Operation], ctx: &mut CoreContext) -> Result<(), String> {
        let dispatch = Dispatch::new();
        let grid = GridExecutor::new((1, 1, 1));
        let env = ExecutionEnv::new(&dispatch, &grid);
        execute_ops(ops, ctx, &env)
    }

    fn build_view() -> Operation {
        // %v = construct_memory_view %p {shape=[64,32], strides=[32,1], HBM, f16}
        Operation::new(Some("%v"), "ktdp.construct_memory_view", &["%p"])
            .with_attr("shape", Attr::IntList(vec![64, 32]))
            .with_attr("strides", Attr::IntList(vec![32, 1]))
            .with_attr("memory_space", Attr::Str("HBM".into()))
            .with_attr("dtype", Attr::Str("f16".into()))
    }

    #[test]
    fn construct_view_builds_memref() {
        let mut ctx = single_core_context();
        ctx.set_value("%p", Value::Index(4)); // element index 4 (base_ptr is an element index)
        run(&[build_view()], &mut ctx).unwrap();
        match ctx.get_value("%v").unwrap() {
            Value::MemRef(m) => {
                assert_eq!(m.shape, vec![64, 32]);
                // base_ptr=4 element index at f16 (2 bytes) -> byte 8.
                assert_eq!(m.byte_address(), 4 * 2);
                assert_eq!(m.dtype, DType::F16);
            }
            other => panic!("expected MemRef, got {other:?}"),
        }
    }

    #[test]
    fn access_tile_offset_via_base_map() {
        let mut ctx = single_core_context();
        ctx.set_value("%p", Value::Index(0)); // base at byte 0 for a clean offset check
        ctx.set_value("%i", Value::Index(2));
        ctx.set_value("%j", Value::Index(3));
        // identity base_map over (i, j); offset = (2*32 + 3*1) elems * 2 bytes
        let at = Operation::new(
            Some("%t"),
            "ktdp.construct_access_tile",
            &["%v", "%i", "%j"],
        )
        .with_attr("shape", Attr::IntList(vec![1, 1]))
        .with_attr("base_map", Attr::AffineMap(AffineMap::identity(2)));
        run(&[build_view(), at], &mut ctx).unwrap();
        match ctx.get_value("%t").unwrap() {
            Value::AccessTile(a) => match &a.parent_ref {
                ParentRef::Tile(tr) => assert_eq!(tr.base_ptr, (2 * 32 + 3) * 2),
                _ => panic!("expected single-allocation TileRef parent"),
            },
            other => panic!("expected AccessTile, got {other:?}"),
        }
    }

    #[test]
    fn distributed_parent_is_flagged_unported() {
        let mut ctx = single_core_context();
        // assert the single-allocation path rejects a non-memref parent.
        ctx.set_value("%v", Value::Index(7));
        let at = Operation::new(Some("%t"), "ktdp.construct_access_tile", &["%v"])
            .with_attr("shape", Attr::IntList(vec![1]))
            .with_attr("base_map", Attr::AffineMap(AffineMap::identity(0)));
        let err = run(&[at], &mut ctx).unwrap_err();
        assert!(err.contains("expected MemRef"));
    }
}
