// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Core IR data structures — Rust port of `ktir_cpu/ir_types.py` (the
//! `Operation` / `IRFunction` / `IRModule` half; the memref/tile types live in
//! `memref.rs` and `tile.rs`).
//!
//! The keystone type here is [`Value`]: it replaces Python's `Any` as the type
//! that flows through every SSA binding, operand lookup, and handler return.

use std::collections::HashMap;

use crate::affine::{AffineMap, AffineSet};
use crate::arena::Arena;
use crate::attrkey::AttrKey;
use crate::dtypes::DType;
use crate::irtype::IrType;
use crate::memref::{
    AccessTile, DistributedMemRef, DistributedTileRef, IndirectAccessTile, MemRef, TileRef,
};
use crate::opkind::OpKind;
use crate::tile::Tile;

/// A scalar SSA value (e.g. `arith.constant`, a loop induction variable).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scalar {
    F32(f32),
    I32(i32),
    I64(i64),
    Bool(bool),
}

impl Scalar {
    pub fn as_f32(self) -> Option<f32> {
        match self {
            Scalar::F32(v) => Some(v),
            _ => None,
        }
    }
    pub fn as_i64(self) -> Option<i64> {
        match self {
            Scalar::I32(v) => Some(v as i64),
            Scalar::I64(v) => Some(v),
            _ => None,
        }
    }
}

/// Anything an SSA value can hold — the single tagged union that replaces
/// Python's `Any` in the per-core scope map. Each dialect handler `match`es to
/// extract the variant it expects; an unexpected variant is a typed error
/// rather than a runtime `AttributeError`.
///
/// Memref/tileref variants are declared but unused in the arith slice; they
/// land as `memref.rs` / `tile.rs` grow. Kept here so the enum is the one
/// authoritative list of SSA value kinds from the start.
#[derive(Clone, Debug)]
pub enum Value {
    Scalar(Scalar),
    Tile(Tile),
    Index(i64),
    Tuple(Vec<Value>),
    MemRef(MemRef),
    DistMemRef(DistributedMemRef),
    TileRef(TileRef),
    DistTileRef(DistributedTileRef),
    AccessTile(AccessTile),
    IndirectAccessTile(IndirectAccessTile),
    /// Per-core handle produced by `ktdp.inter_tile_produce`, consumed by
    /// `ktdp.inter_tile_reduce`. Carries this core's partial plus the parsed
    /// producer/groups affine sets and the resolved group index. Mirrors the
    /// Python `TileFuture` dataclass.
    TileFuture(Box<TileFuture>),
}

/// Per-core handle produced by `ktdp.inter_tile_produce`. SPMD: each core holds
/// its own instance bound to its local `%fut` SSA value; cross-core data movement
/// happens via the scheduler's ring all-reduce when the matching
/// `ktdp.inter_tile_reduce` runs, not by reading other cores' futures. 1:1 with
/// `ktir_cpu.ir_types.TileFuture`.
#[derive(Clone, Debug)]
pub struct TileFuture {
    /// This core's yielded partial — the seed for the transport. `None` when the
    /// core is in `groups_set` but outside `producer_set` (the reduce backend
    /// substitutes the identity tensor for it). The examples yield a single tile.
    pub local_partial: Option<Tile>,
    /// Parsed `producer_tiles_per_group` set, kept on the future so the consumer
    /// can build the ring plan without re-parsing.
    pub producer_set: AffineSet<'static>,
    /// Parsed `groups` set.
    pub groups_set: AffineSet<'static>,
    /// The group this core belongs to, computed once at produce time.
    pub group_idx: i64,
}

/// A parsed operation attribute. Replaces the `Any` values in Python's
/// `Operation.attributes` dict; the parser picks the variant, handlers match.
#[derive(Clone, Debug, PartialEq)]
pub enum Attr<'a> {
    Int(i64),
    IntList(&'a [i64]),
    Float(f64),
    FloatList(&'a [f64]),
    Str(&'a str),
    StrList(&'a [&'a str]),
    Bool(bool),
    Dtype(DType),
    /// An attribute that names an OPERATION — `reduce_fn`'s combiner, say. A variant, so a
    /// combiner nothing implements cannot be spelled.
    Op(OpKind),
    /// An attribute that names SSA VALUES — `scf.for`'s `iter_var`/`iter_args`, a region's block
    /// arguments, an op's `result_names`. These are values like any operand, and a pass that
    /// rewrites the op stream has to rewrite them in lockstep; carrying them as identities is what
    /// makes "renamed the uses but not the declaration" unrepresentable.
    Ssas(&'a [Ssa]),
    AffineMap(AffineMap<'a>),
    AffineMapList(&'a [AffineMap<'a>]),
    AffineSet(AffineSet<'a>),
}

/// HASHED BY VALUE, INCLUDING THE FLOATS. `Attr` cannot be `Eq` — it carries an `f64` — so its
/// hash is written rather than derived, and the float is hashed through `to_bits` so that two
/// attributes which compare equal hash equal. The discriminant leads, so `Int(0)` and `Bool(false)`
/// do not collide on payload alone.
impl std::hash::Hash for Attr<'_> {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        std::mem::discriminant(self).hash(h);
        match self {
            Attr::Int(v) => v.hash(h),
            Attr::IntList(v) => v.hash(h),
            Attr::Float(v) => v.to_bits().hash(h),
            Attr::FloatList(v) => v.iter().for_each(|f| f.to_bits().hash(h)),
            Attr::Str(v) => v.hash(h),
            Attr::StrList(v) => v.hash(h),
            Attr::Bool(v) => v.hash(h),
            Attr::Dtype(v) => v.hash(h),
            Attr::Op(v) => v.hash(h),
            Attr::Ssas(v) => v.hash(h),
            Attr::AffineMap(v) => v.hash(h),
            Attr::AffineMapList(v) => v.hash(h),
            Attr::AffineSet(v) => v.hash(h),
        }
    }
}

/// AN SSA VALUE'S NAME.
///
/// The interpreter's scope already interned every name to a `u32` and indexed its slots by that, so
/// the spelling was a lookup key thrown away on first use. The IR carries the id itself: a value is
/// named by identity, and the intern table is gone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Ssa(pub u32);

impl Ssa {
    /// The slot this value occupies.
    pub fn slot(self) -> usize {
        self.0 as usize
    }
}

/// A single IR operation. 1:1 with the Python `Operation` dataclass.
#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub struct Operation<'a> {
    /// The value this operation defines. `None` for ops with no result.
    pub result: Option<Ssa>,
    /// WHICH OPERATION THIS IS — a variant, not a spelling.
    pub op_type: OpKind,
    pub operands: &'a [Ssa],
    pub attributes: &'a [(AttrKey, Attr<'a>)],
    pub result_type: Option<IrType<'a>>,
    /// Nested regions (scf bodies). Empty for straight-line ops.
    pub regions: &'a [&'a [Operation<'a>]],
}

impl<'a> Operation<'a> {
    /// Build an operation into `a`. The arena owns the names; the operation borrows them.
    pub fn new(
        a: &'a Arena,
        result: Option<Ssa>,
        op_type: OpKind,
        operands: &[Ssa],
    ) -> Operation<'a> {
        Operation {
            result,
            op_type,
            operands: a.ssa(operands.to_vec()),
            attributes: &[],
            result_type: None,
            regions: &[],
        }
    }

    pub fn with_attr(self, a: &'a Arena, key: AttrKey, val: Attr<'a>) -> Operation<'a> {
        let mut attrs: Vec<(AttrKey, Attr<'a>)> = self
            .attributes
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        match attrs.iter_mut().find(|(k, _)| *k == key) {
            Some(slot) => slot.1 = val,
            None => attrs.push((key, val)),
        }
        Operation {
            attributes: a.attrs(attrs),
            ..self
        }
    }

    /// This operation's attribute, by name. The list is tiny, so a scan beats hashing.
    pub fn attr(&self, key: AttrKey) -> Option<&Attr<'a>> {
        self.attributes
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| v)
    }
}

/// An IR function: arguments, a flat op list, and the grid shape.
#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub struct IRFunction<'a> {
    pub name: &'a str,
    /// The values this function takes, with their types. An argument IS an SSA value.
    pub arguments: &'a [(Ssa, IrType<'a>)],
    pub operations: &'a [Operation<'a>],
    pub grid: (usize, usize, usize),
    pub return_type: Option<&'a str>,
}

impl<'a> IRFunction<'a> {
    /// The values this function takes, in order.
    pub fn arg_names(&self) -> Vec<Ssa> {
        self.arguments.iter().map(|(v, _)| *v).collect()
    }

    /// EVERY operation this function contains, regions included, in program order.
    ///
    /// ⛔ NOT `operations`, which is the TOP LEVEL only. A time-tiled op puts its whole computation
    /// inside an `scf.for` body, so anything asking "what does this program do" and reading
    /// `operations` alone sees a loop and nothing else — and reports the tiled case as empty rather
    /// than as tiled.
    pub fn ops_deep(&self) -> Vec<&Operation<'a>> {
        fn walk<'o, 'a>(ops: &'o [Operation<'a>], out: &mut Vec<&'o Operation<'a>>) {
            for o in ops {
                out.push(o);
                for r in o.regions {
                    walk(r, out);
                }
            }
        }
        let mut out = Vec::new();
        walk(self.operations, &mut out);
        out
    }

    /// Each operand's MEMORY VIEW as `(shape, strides)`, in operand order — the whole operand as this
    /// program addresses it, before the per-core window.
    ///
    /// ⭐ THE STRIDES ARE THE WALK. A declared dim ORDER plus a row-major convention says the same
    /// thing one indirection away; the strides say it in the units the addresses are actually in, so
    /// a walk that names its dims in one order and strides them in another is not expressible.
    pub fn operand_views(&self) -> Vec<(&'a [i64], &'a [i64])> {
        self.ops_deep()
            .into_iter()
            .filter(|o| o.op_type == OpKind::KtdpConstructMemoryView)
            .filter_map(
                |o| match (o.attr(AttrKey::Shape), o.attr(AttrKey::Strides)) {
                    (Some(Attr::IntList(s)), Some(Attr::IntList(d))) => Some((*s, *d)),
                    _ => None,
                },
            )
            .collect()
    }

    /// The ELEMENT OFFSET each core addresses operand `arg` at, indexed by flat core id.
    ///
    /// ⭐ RECOVERED FROM THE PROGRAM, NOT DECLARED BESIDE IT. A per-core start-address table listed
    /// these as data next to the walk that was supposed to produce them, so a table and a walk that
    /// disagreed were both representable and only the card could tell. Here the corner is the
    /// `gid * per_core_extent` arithmetic the program actually performs, folded against the operand's
    /// own strides — there is one statement of the offsets and this reads it.
    ///
    /// The trip index is 0: a time tile's `iv` term is the loop's, not a core's.
    pub fn operand_core_offsets(&self, arg: usize) -> Vec<i64> {
        let ops = self.ops_deep();
        let def = |v: Ssa| ops.iter().copied().find(|o| o.result == Some(v));
        let Some(tile) = ops
            .iter()
            .copied()
            .filter(|o| o.op_type == OpKind::KtdpConstructAccessTile)
            .nth(arg)
        else {
            return Vec::new();
        };
        let Some(view) = tile.operands.first().and_then(|v| def(*v)) else {
            return Vec::new();
        };
        let strides = match view.attr(AttrKey::Strides) {
            Some(Attr::IntList(s)) => *s,
            Some(_) | None => return Vec::new(),
        };
        // Which grid axis (if any) each dim's corner rides.
        //
        // ⛔ NOT BY `result`. A multi-axis `ktdp.get_compute_tile_id` defines ONE value in `result`
        // and the rest in `result_names`, so looking the axis up by definer finds axis 0 and is blind
        // to every other — which reads a 2-D grid as though only its first axis moved anything.
        let tile_id = ops
            .iter()
            .copied()
            .find(|o| o.op_type == OpKind::KtdpGetComputeTileId);
        let axis_of = |v: Ssa| {
            tile_id.and_then(|o| match o.attr(AttrKey::ResultNames) {
                Some(Attr::Ssas(names)) => names.iter().position(|n| *n == v),
                // A single-axis grid names its one axis by the op's own result.
                Some(_) | None => (o.result == Some(v)).then_some(0),
            })
        };
        // A `arith.constant`'s value, where the value is what this walk can fold.
        let konst = |v: Ssa| {
            def(v)
                .filter(|o| o.op_type == OpKind::ArithConstant)
                .and_then(|o| match o.attr(AttrKey::Value) {
                    Some(Attr::Int(k)) => Some(*k),
                    Some(_) | None => None,
                })
        };
        // A corner term is a constant, a `gid * per`, or their sum. The loop's `iv * per` rides no
        // grid axis, which is exactly why it contributes nothing at trip 0.
        let mut consts: Vec<i64> = Vec::new();
        let mut rides: Vec<Option<(usize, i64)>> = Vec::new();
        for v in &tile.operands[1..] {
            let mut c = 0i64;
            let mut ride = None;
            let mut stack = vec![*v];
            while let Some(s) = stack.pop() {
                let Some(o) = def(s) else { continue };
                if o.op_type == OpKind::ArithConstant {
                    c += konst(s).unwrap_or(0);
                } else if o.op_type == OpKind::ArithAddi {
                    stack.extend_from_slice(o.operands);
                } else if o.op_type == OpKind::ArithMuli
                    && let Some(ax) = o.operands.iter().find_map(|x| axis_of(*x))
                    && let Some(per) = o.operands.iter().find_map(|x| konst(*x))
                {
                    ride = Some((ax, per));
                }
            }
            consts.push(c);
            rides.push(ride);
        }
        let (gx, gy, gz) = self.grid;
        let mut out = Vec::with_capacity(gx * gy * gz);
        for core in 0..(gx * gy * gz) {
            let coord = [core % gx, (core / gx) % gy, core / (gx * gy)];
            let mut off = 0i64;
            for (i, s) in strides.iter().enumerate() {
                let base = consts.get(i).copied().unwrap_or(0)
                    + match rides.get(i).copied().flatten() {
                        Some((ax, per)) => coord.get(ax).copied().unwrap_or(0) as i64 * per,
                        None => 0,
                    };
                off += base * s;
            }
            out.push(off);
        }
        out
    }

    /// Each operand's ACCESS TILE extents, in operand order — the window this program's own core
    /// reads or writes of that operand.
    ///
    /// The tiles are constructed as the operands are walked, so position here is operand position;
    /// no name is involved. A broadcast dim is an extent of 1.
    pub fn operand_tile_shapes(&self) -> Vec<&'a [i64]> {
        self.ops_deep()
            .into_iter()
            .filter(|o| o.op_type == OpKind::KtdpConstructAccessTile)
            .filter_map(|o| match o.attr(AttrKey::Shape) {
                Some(Attr::IntList(v)) => Some(*v),
                Some(_) | None => None,
            })
            .collect()
    }
}

/// Top-level module: named functions plus module-scope attribute aliases.
#[derive(Clone, Debug, Default)]
pub struct IRModule<'a> {
    pub functions: HashMap<String, IRFunction<'a>>,
    /// `#name -> verbatim value string`, e.g. `"#X_coord_set" -> "affine_set<...>"`.
    pub aliases: HashMap<String, String>,
}

impl<'a> IRModule<'a> {
    pub fn get_function(&self, name: &str) -> Result<&IRFunction<'a>, String> {
        self.functions
            .get(name)
            .ok_or_else(|| format!("Function '{name}' not found in module"))
    }

    pub fn add_function(&mut self, func: IRFunction<'a>) {
        self.functions.insert(func.name.to_string(), func);
    }
}
