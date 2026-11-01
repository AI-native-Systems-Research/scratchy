// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").

//! A PROGRAM THAT WALKS — one constant-trip `scf.for` at the top level, its body read once per trip
//! (`page_attn_head_major_*.py`'s page walk: `walk_tiles`' one block body) — lowered to the card's FOLD: the
//! body's descriptors are built ONCE, at trip 0, and the runtime relaunches them once per trip,
//! shifting the step's sources by one pitch per relaunch.
//!
//! # WHAT A TRIP MAY CHANGE, AND WHAT IT MAY NOT
//!
//! A relaunch re-runs the SAME descriptors; the one thing that moves between relaunches is a
//! segment base. So a trip may differ from trip 0 only in ADDRESSES that advance by one fixed number
//! of bytes per trip, all in the buffers the runtime shifts. Every value the body computes from the
//! induction variable must therefore be AFFINE in it and land in a tile's addressing (an access
//! tile's corner, an indirect tile's index anchor); its trip-0 value is what the descriptors bake and
//! its per-trip advance, in bytes, is [`LoopFold::step_bytes`] — ONE number for every such tile, or
//! the program is refused here by name.
//!
//! # WHAT THE BODY CARRIES
//!
//! The loop-carried values (`iter_args`) are the walk's running state. On the card each is ONE
//! buffer that every trip reads and the trip's yielded value overwrites ([`Carried`]); the walk
//! lowering mints it before the loop, seeds it with the init value the program states, and has
//! the yielded value's producer write into it. The loop's RESULTS are that same buffer after the
//! last trip, so the ops after the loop read it directly.
//!
//! # THE FLATTENED PROGRAM
//!
//! [`flatten`] states the body inline at trip 0 — the ops before the loop, the body with every
//! induction-variable-affine integer op replaced by its trip-0 constant, then the ops after the
//! loop — so every reader of a program's windows (`regions`, `region_for_operand`, the gather
//! walk) reads trip 0's windows exactly as it reads a straight-line program's. The iter args and
//! the loop's results have no defining op in it: they are the carried buffers above.

use ktir_core::arena::Arena;
use ktir_core::attrkey::AttrKey;
use ktir_core::ir::{Attr, IRFunction, Operation, Ssa};
use ktir_core::opkind::OpKind;

use super::lower_ktir_to_superdsc::{Error, err};
use crate::ktir_node::KtirNode;

/// One loop-carried value: the body's block argument, the value it starts as, the value each trip
/// yields for it, and the loop result it ends as. All four name ONE buffer on the card.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Carried {
    pub arg: Ssa,
    pub init: Ssa,
    pub yielded: Ssa,
    pub result: Ssa,
    /// The one value the init states for every element — a `tensor.splat` of a constant, which is
    /// the only seed the card's carried buffer takes (a fill from the registry's bound const).
    pub seed: f32,
}

/// What the card does with one of a walk's body descriptors: relaunch it `trips` times, the step's
/// sources `step_bytes` further along each time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WalkTrips {
    pub trips: u32,
    pub step_bytes: u32,
}

/// The walk the card folds — see the module docs.
#[derive(Clone, Debug)]
pub struct LoopFold {
    /// How many times the body runs: the loop's constant trip count.
    pub trips: u32,
    /// The bytes every induction-variable-dependent address advances per trip — the one segment
    /// shift a relaunch applies.
    pub step_bytes: u32,
    /// The parameters whose addresses move per trip — the buffers the shift must reach, and the
    /// ONLY ones a trip may read through a moving address.
    pub stepped_params: Vec<Ssa>,
    /// ⭐ THE INDEX TABLES THE CARD LAYS OUT AT THE WALK'S PITCH — an int32 table read one entry per
    /// trip (spyre-inference's page index table, `[num_blocks, 32]`) whose own rows sit closer than
    /// the walk's step: on the card its rows are [`Self::step_bytes`] apart instead, so the one
    /// shift moves it with the data tiles. `(parameter, its own per-trip bytes)`.
    pub pitched: Vec<(Ssa, u32)>,
    /// Every value the body defines — the descriptors the card relaunches once per trip.
    pub body: std::collections::HashSet<Ssa>,
    /// The loop-carried values, in `iter_args` order.
    pub carried: Vec<Carried>,
}

impl LoopFold {
    /// The values the carried buffers are seeded with — registry constants the lowering reads.
    pub fn seeds(&self) -> impl Iterator<Item = f32> + '_ {
        self.carried.iter().map(|c| c.seed)
    }
}

/// A walking program, stated straight-line at trip 0, and the fold it lowers to.
pub struct Flattened {
    pub node: KtirNode,
    pub fold: LoopFold,
}

/// An integer value affine in the induction variable: `per_trip · i + at_trip0` at trip `i`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Affine {
    per_trip: i64,
    at_trip0: i64,
}

fn int_attr(op: &Operation<'_>) -> Option<i64> {
    op.attributes.iter().find_map(|(k, v)| match (k, v) {
        (AttrKey::Value, Attr::Int(i)) => Some(*i),
        _ => None,
    })
}

/// Which `intermediate_vars` entry an indirect tile's index anchor is: the `Sym(i)` of its indirect
/// dim's `Sym(i) + Dim(0)` subscript (the one form the gather lowering reads).
fn indirect_anchor_var(tile: &Operation<'_>) -> Option<usize> {
    use ktir_core::affine::AffineExpr;
    let (Some(Attr::StrList(kinds)), Some(Attr::AffineMapList(subs))) =
        (tile.attr(AttrKey::DimKinds), tile.attr(AttrKey::DimSubs))
    else {
        return None;
    };
    let d = kinds.iter().position(|k| *k == "indirect")?;
    match subs.get(d)?.exprs.first()? {
        AffineExpr::Add(a, b) => match (a, b) {
            (AffineExpr::Sym(i), AffineExpr::Dim(0)) | (AffineExpr::Dim(0), AffineExpr::Sym(i)) => {
                Some(*i)
            }
            _ => None,
        },
        _ => None,
    }
}

/// ⭐ A WALK'S BODY UNROLLED INTO ITS LAUNCH, AS dxp UNROLLS torch-spyre's. torch-spyre states its
/// walk as a device `scf.for` over the trips in one bundle, and dxp unrolls that loop at compile time
/// (MEASURED on the reference's compiled prefill bundle: its binary grows by one trip's code per
/// trip of the loop bound — 204 KB at 2 trips, 527 KB at 8), so every trip runs inside ONE launch.
/// Our walk relaunches its body group once a trip instead. This puts `u` trips in each launch: the
/// body's descriptors `u` times over, copy `t` reading the activation segment `t · step` further in
/// (the shift the relaunch applies to it, `fold_plan::seg_deltas`' walk arm — the step's page-table
/// row and mask tile are the only tensors a body reads there), and the walk relaunched `trips / u`
/// times at `u` steps a launch. `u` is the largest power of two that divides the trip count and keeps
/// the unrolled body within `max_descriptors` (a launch group's bound).
pub fn unroll_walk(ops: Vec<super::EmittedOp>, max_descriptors: usize) -> Vec<super::EmittedOp> {
    let (Some(first), Some(last)) = (
        ops.iter().position(|e| e.walk.is_some()),
        ops.iter().rposition(|e| e.walk.is_some()),
    ) else {
        return ops;
    };
    let Some(walk) = ops[first].walk else {
        return ops;
    };
    let body_descriptors: usize = ops[first..=last]
        .iter()
        .map(|e| e.time.max(1) as usize)
        .sum();
    let mut u = 1u32;
    while walk.trips.is_multiple_of(2 * u) && body_descriptors * 2 * u as usize <= max_descriptors {
        u *= 2;
    }
    if u == 1 {
        return ops;
    }
    let folded = WalkTrips {
        trips: walk.trips / u,
        step_bytes: walk.step_bytes * u,
    };
    let activation = crate::placement::SegRole::Activation.segment();
    let mut ops = ops;
    let tail = ops.split_off(last + 1);
    let body = ops.split_off(first);
    for t in 0..u {
        let shift = u64::from(t) * u64::from(walk.step_bytes);
        ops.extend(body.iter().map(|e| {
            let mut c = e.shifted_in_segment(activation, shift);
            c.walk = Some(folded);
            c
        }));
    }
    ops.extend(tail);
    ops
}

/// Whether `k` walks — states a top-level `scf.for` — whatever it computes. A walk's windows are
/// read at trip 0 ([`flatten`]); a straight-line program's at the top level.
pub fn walks(k: &KtirNode) -> bool {
    k.func
        .operations
        .iter()
        .any(|o| o.op_type == OpKind::ScfFor)
}

/// The walk, or `None` for a program with no top-level `scf.for`.
pub fn flatten(k: &KtirNode) -> Result<Option<Flattened>, Error> {
    let f = &k.func;
    let fors: Vec<usize> = f
        .operations
        .iter()
        .enumerate()
        .filter(|(_, o)| o.op_type == OpKind::ScfFor)
        .map(|(i, _)| i)
        .collect();
    let at = match fors[..] {
        [] => return Ok(None),
        [at] => at,
        _ => {
            return err(format!(
                "{}: {} top-level `scf.for` loops — the card folds ONE walk per program (one block \
                 body, relaunched per trip), so a second loop has no relaunch of its own",
                f.name,
                fors.len()
            ));
        }
    };
    let lp = &f.operations[at];
    // Constant integer values the bounds and the body's addressing read: top-level and body alike.
    let mut consts: std::collections::HashMap<Ssa, i64> = std::collections::HashMap::new();
    let body_ops: &[Operation<'static>] = lp.regions.first().copied().unwrap_or(&[]);
    for o in f.operations.iter().chain(body_ops.iter()) {
        if o.op_type == OpKind::ArithConstant
            && let (Some(r), Some(v)) = (o.result, int_attr(o))
        {
            consts.insert(r, v);
        }
    }
    let bound = |slot: usize, what: &str| -> Result<i64, Error> {
        lp.operands
            .get(slot)
            .and_then(|v| consts.get(v).copied())
            .ok_or_else(|| Error {
                message: format!(
                    "{}: the walk's {what} is not a constant — the card relaunches the body a \
                     fixed number of times, so the trip count has to be known when the bundle is \
                     baked",
                    f.name
                ),
            })
    };
    let (lb, ub, step) = (
        bound(0, "lower bound")?,
        bound(1, "upper bound")?,
        bound(2, "step")?,
    );
    if step <= 0 || ub <= lb {
        return err(format!(
            "{}: the walk runs [{lb}, {ub}) by {step} — an empty or non-advancing loop",
            f.name
        ));
    }
    let trips = u32::try_from((ub - lb + step - 1) / step).map_err(|_| Error {
        message: format!("{}: the walk's trip count does not fit a u32", f.name),
    })?;
    let iv = match lp.attr(AttrKey::IterVar) {
        Some(Attr::Ssas(v)) if !v.is_empty() => v[0],
        _ => {
            return err(format!(
                "{}: the walk's `scf.for` states no induction variable",
                f.name
            ));
        }
    };
    let args: Vec<Ssa> = match lp.attr(AttrKey::IterArgs) {
        Some(Attr::Ssas(v)) => v.to_vec(),
        _ => Vec::new(),
    };
    let results: Vec<Ssa> = match lp.attr(AttrKey::ResultNames) {
        Some(Attr::Ssas(v)) => v.to_vec(),
        _ => lp.result.into_iter().collect(),
    };
    let inits = &lp.operands[3.min(lp.operands.len())..];
    let Some((yield_op, body)) = body_ops.split_last() else {
        return err(format!("{}: the walk's body is empty", f.name));
    };
    if yield_op.op_type != OpKind::ScfYield {
        return err(format!(
            "{}: the walk's body does not end in `scf.yield`, so what each trip carries forward \
             is not stated",
            f.name
        ));
    }
    if args.len() != inits.len()
        || args.len() != results.len()
        || args.len() != yield_op.operands.len()
    {
        return err(format!(
            "{}: the walk carries {} iter arg(s) from {} init(s), yields {} and returns {} — one \
             of each per carried value",
            f.name,
            args.len(),
            inits.len(),
            yield_op.operands.len(),
            results.len()
        ));
    }
    let def_of = |v: Ssa| f.operations.iter().find(|o| o.result == Some(v));
    let seed_of = |init: Ssa| -> Result<f32, Error> {
        def_of(init)
            .filter(|o| o.op_type == OpKind::TensorSplat)
            .and_then(|o| o.operands.first().copied())
            .and_then(def_of)
            .filter(|o| o.op_type == OpKind::ArithConstant)
            .and_then(|o| {
                o.attributes.iter().find_map(|(k, v)| match (k, v) {
                    (AttrKey::Value, Attr::Float(x)) => Some(*x as f32),
                    _ => None,
                })
            })
            .ok_or_else(|| Error {
                message: format!(
                    "{}: the walk's carried value %{} starts as something other than a splatted \
                     constant — the card seeds a carried buffer with ONE value",
                    f.name, init.0
                ),
            })
    };
    let carried: Vec<Carried> = (0..args.len())
        .map(|i| {
            Ok(Carried {
                arg: args[i],
                init: inits[i],
                yielded: yield_op.operands[i],
                result: results[i],
                seed: seed_of(inits[i])?,
            })
        })
        .collect::<Result<_, Error>>()?;

    // THE INDUCTION VARIABLE'S AFFINE CLOSURE: `index_cast` is transparent, `muli`/`addi` against a
    // constant (or another affine value, for `addi`) compose. Anything else that reads an affine value
    // is a use of the trip number this fold cannot express, and is refused below.
    let mut affine: std::collections::HashMap<Ssa, Affine> = std::collections::HashMap::new();
    affine.insert(
        iv,
        Affine {
            per_trip: step,
            at_trip0: lb,
        },
    );
    for o in body {
        let Some(r) = o.result else { continue };
        let a = |s: usize| o.operands.get(s).and_then(|v| affine.get(v).copied());
        let c = |s: usize| o.operands.get(s).and_then(|v| consts.get(v).copied());
        let form = match o.op_type {
            OpKind::ArithIndexCast => a(0),
            OpKind::ArithMuli => match (a(0), c(1), c(0), a(1)) {
                (Some(x), Some(m), _, _) | (_, _, Some(m), Some(x)) => Some(Affine {
                    per_trip: x.per_trip * m,
                    at_trip0: x.at_trip0 * m,
                }),
                _ => None,
            },
            OpKind::ArithAddi => match (a(0), a(1), c(0), c(1)) {
                (Some(x), Some(y), _, _) => Some(Affine {
                    per_trip: x.per_trip + y.per_trip,
                    at_trip0: x.at_trip0 + y.at_trip0,
                }),
                (Some(x), None, _, Some(m)) | (None, Some(x), Some(m), _) => Some(Affine {
                    per_trip: x.per_trip,
                    at_trip0: x.at_trip0 + m,
                }),
                _ => None,
            },
            _ => None,
        };
        if let Some(form) = form {
            affine.insert(r, form);
        }
    }

    // THE PER-TRIP STEP OF EVERY ADDRESS THAT MOVES. A plain access tile moves by its corners'
    // advance times the view's strides; an indirect tile by its index anchor's advance, in index
    // words. Every other reader of an affine value is refused.
    let views: std::collections::HashMap<Ssa, &Operation<'static>> = f
        .operations
        .iter()
        .chain(body.iter())
        .filter(|o| o.op_type == OpKind::KtdpConstructMemoryView)
        .filter_map(|o| Some((o.result?, o)))
        .collect();
    let view_of = |v: Ssa| -> Result<(&Operation<'static>, Vec<i64>, usize), Error> {
        let view = views.get(&v).copied().ok_or_else(|| Error {
            message: format!(
                "{}: a tile the walk moves per trip is not over a `ktdp.construct_memory_view`",
                f.name
            ),
        })?;
        let strides = match view.attr(AttrKey::Strides) {
            Some(Attr::IntList(s)) => s.to_vec(),
            _ => {
                return err(format!(
                    "{}: a view the walk moves per trip states no `strides`, so how far one trip \
                     moves it is unknown",
                    f.name
                ));
            }
        };
        let bytes = view
            .result_type
            .and_then(|t| t.elem())
            .map(|d| d.bytes_per_elem())
            .ok_or_else(|| Error {
                message: format!(
                    "{}: a view the walk moves per trip states no element type",
                    f.name
                ),
            })?;
        Ok((view, strides, bytes))
    };
    // `(parameter, per-trip bytes, an int32 index table read one entry per trip)`.
    let mut steps: Vec<(Ssa, i64, bool)> = Vec::new();
    let is_index = |view: &Operation<'_>| {
        matches!(
            view.result_type,
            Some(ktir_core::irtype::IrType::MemRef { dims, elem: ktir_core::dtypes::DType::I32 })
                if dims.len() == 1
        )
    };
    for o in body {
        let moved: Vec<usize> = o
            .operands
            .iter()
            .enumerate()
            .filter(|(_, v)| affine.get(v).is_some_and(|a| a.per_trip != 0))
            .map(|(i, _)| i)
            .collect();
        let vars_moved = match o.attr(AttrKey::IntermediateVars) {
            Some(Attr::Ssas(vs)) => vs
                .iter()
                .any(|v| affine.get(v).is_some_and(|a| a.per_trip != 0)),
            _ => false,
        };
        if moved.is_empty() && !vars_moved {
            continue;
        }
        match o.op_type {
            // The affine closure's own ops — the addressing arithmetic itself.
            OpKind::ArithIndexCast | OpKind::ArithMuli | OpKind::ArithAddi
                if o.result.is_some_and(|r| affine.contains_key(&r)) => {}
            // ⭐ THE STEP IS THE CARD'S, NOT THE VIEW'S ROW-MAJOR STRIDES. On the card a view is its
            // stick-major flat frame `[R, C]` — element `(r, c)` at `(c/stick)·R·stick + r·stick +
            // c%stick` — so a trip that moves the ROW corner by `Δr` moves every stick column's
            // window by `Δr` whole sticks (128 bytes each), whatever the view's width; a one-row
            // (rank-1) view is flat and moves by its elements. A trip that moves a multi-row view's
            // LAST-axis corner moves its sticks apart, which no one shift states.
            OpKind::KtdpConstructAccessTile => {
                let (view, _, bytes) = view_of(o.operands[0])?;
                let dims: Vec<i64> = view
                    .result_type
                    .and_then(|t| t.dims())
                    .map(<[i64]>::to_vec)
                    .unwrap_or_default();
                let rank = dims.len();
                let mut step = 0i64;
                for &slot in &moved {
                    let Some(d) = slot.checked_sub(1).filter(|d| *d < rank) else {
                        return err(format!(
                            "{}: an access tile moves its operand {slot} per trip, which is not \
                             one of its view's corners",
                            f.name
                        ));
                    };
                    let per_trip = affine[&o.operands[slot]].per_trip;
                    step += if rank == 1 {
                        per_trip * bytes as i64
                    } else if d + 1 == rank {
                        return err(format!(
                            "{}: an access tile moves the last-axis corner of its `{dims:?}` view \
                             per trip — on the card that moves its stick columns apart, which no \
                             one shift states",
                            f.name
                        ));
                    } else {
                        let rows_per: i64 = dims[d + 1..rank - 1].iter().product();
                        per_trip * rows_per * i64::from(crate::work::STICK_BYTES)
                    };
                }
                steps.push((view.operands[0], step, is_index(view)));
            }
            OpKind::KtdpConstructIndirectAccessTile => {
                // The ANCHOR — the `Sym(i)` of the indirect dim's `Sym(i) + Dim(0)` subscript — is
                // the one capture that may move; its advance is in INDEX words, over the index
                // view's own stride. A moving DATA subscript would move the gathered rows
                // themselves, which no shift of the index can state.
                let vars = match o.attr(AttrKey::IntermediateVars) {
                    Some(Attr::Ssas(vs)) => vs.to_vec(),
                    _ => Vec::new(),
                };
                let anchor = indirect_anchor_var(o);
                let moving: Vec<usize> = (0..vars.len())
                    .filter(|&i| affine.get(&vars[i]).is_some_and(|a| a.per_trip != 0))
                    .collect();
                let (Some(anchor), [i]) = (anchor, &moving[..]) else {
                    return err(format!(
                        "{}: an indirect tile moves captures {moving:?} per trip — the fold moves \
                         ONE capture, the index anchor of its indirect dim",
                        f.name
                    ));
                };
                if *i != anchor || !moved.is_empty() {
                    return err(format!(
                        "{}: an indirect tile moves a data subscript per trip — only its index \
                         anchor may move",
                        f.name
                    ));
                }
                let index_view = o.operands.get(1).copied().ok_or_else(|| Error {
                    message: format!("{}: an indirect tile states no index view", f.name),
                })?;
                let (view, strides, bytes) = view_of(index_view)?;
                if strides[..] != [1] {
                    return err(format!(
                        "{}: the walk's index view has strides {strides:?} — the fold moves the \
                         anchor through a contiguous index",
                        f.name
                    ));
                }
                steps.push((
                    view.operands[0],
                    affine[&vars[*i]].per_trip * bytes as i64,
                    is_index(view),
                ));
            }
            other => {
                return err(format!(
                    "{}: `{other:?}` reads the walk's trip number — a relaunch re-runs the same \
                     descriptors, so the trip may move ADDRESSES only (an access tile's corner, an \
                     indirect tile's anchor)",
                    f.name
                ));
            }
        }
    }
    // THE STEP IS THE DATA TILES' — every moving tile that is not an index table advances by one
    // number of bytes, or the walk is refused. An index table read one entry per trip advances by its
    // own row, which on the card is laid out at the walk's step (`LoopFold::pitched`), so it must
    // divide the step.
    let data_step = steps
        .iter()
        .filter(|(_, _, ix)| !ix)
        .map(|(_, s, _)| *s)
        .max();
    let step = data_step
        .or_else(|| steps.iter().map(|(_, s, _)| *s).max())
        .unwrap_or(0);
    if let Some((_, other, _)) = steps.iter().find(|(_, x, ix)| !ix && *x != step) {
        return err(format!(
            "{}: the walk moves its tiles by {step} and {other} bytes per trip — a relaunch shifts \
             ONE segment base by ONE step, so every moving data tile has to advance by the same bytes",
            f.name
        ));
    }
    let mut pitched: Vec<(Ssa, u32)> = Vec::new();
    for &(param, s, ix) in &steps {
        if !ix || s == step {
            continue;
        }
        if s <= 0 || step % s != 0 {
            return err(format!(
                "{}: the walk reads an index table {s} bytes along per trip and moves its tiles by \
                 {step} — the card lays the table's rows out at the walk's step, which must be a \
                 whole number of its rows",
                f.name
            ));
        }
        if !pitched.iter().any(|(p, _)| *p == param) {
            pitched.push((param, s as u32));
        }
    }
    let step_bytes = u32::try_from(step).map_err(|_| Error {
        message: format!(
            "{}: the walk's per-trip step {step} is not a forward byte count",
            f.name
        ),
    })?;
    let mut stepped_params: Vec<Ssa> = steps.iter().map(|(p, _, _)| *p).collect();
    stepped_params.sort_by_key(|s| s.0);
    stepped_params.dedup();

    // THE FLATTENED PROGRAM: before, body at trip 0, after.
    let a = Arena::global();
    let mut ops: Vec<Operation<'static>> = Vec::with_capacity(f.operations.len() + body.len());
    ops.extend(f.operations[..at].iter().copied());
    // The body's `region.bb0_args` names the block arguments the loop binds (the induction variable
    // and the carried values) — a declaration, which the flattened program states as the IV's
    // constant and the carried buffers instead.
    for o in body.iter().filter(|o| o.op_type != OpKind::RegionBb0Args) {
        match o.result.and_then(|r| affine.get(&r).map(|x| (r, x))) {
            Some((r, x)) => ops.push(
                Operation {
                    result_type: o.result_type,
                    ..Operation::new(a, Some(r), OpKind::ArithConstant, &[])
                }
                .with_attr(a, AttrKey::Value, Attr::Int(x.at_trip0)),
            ),
            None => ops.push(*o),
        }
    }
    // The induction variable itself, should anything read it bare.
    if !body.iter().any(|o| o.result == Some(iv)) {
        ops.insert(
            at,
            Operation::new(a, Some(iv), OpKind::ArithConstant, &[]).with_attr(
                a,
                AttrKey::Value,
                Attr::Int(lb),
            ),
        );
    }
    ops.extend(f.operations[at + 1..].iter().copied());
    let body_values: std::collections::HashSet<Ssa> =
        body.iter().filter_map(|o| o.result).collect();
    let mut node = k.clone();
    node.func = IRFunction {
        operations: a.ops(ops),
        ..*f
    };
    Ok(Some(Flattened {
        node,
        fold: LoopFold {
            trips,
            step_bytes,
            stepped_params,
            pitched,
            body: body_values,
            carried,
        },
    }))
}
