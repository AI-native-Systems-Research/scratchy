// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! `ktdp` dialect handlers — grid + distributed + indirect constructors.
//!
//! Port of the remaining `ktir_emulator/dialects/ktdp_ops.py` handlers not covered by
//! `dialects/ktdp.rs` (which owns `construct_memory_view` / `construct_access_tile`),
//! together with the grid helpers from `ktir_emulator/ops/grid_ops.py`:
//!
//!   * `ktdp.get_compute_tile_id` -> grid coordinate(s) of the executing core.
//!   * `ktdp.coreid`              -> core ids matching a masked grid tuple.
//!   * `ktdp.construct_distributed_memory_view` -> `DistributedMemRef`.
//!   * `ktdp.construct_indirect_access_tile`    -> `IndirectAccessTile`.
//!
//! These build the descriptor `Value`s faithfully; the matching distributed /
//! indirect LOAD/STORE resolution lives in the memory-ops subsystem.

use crate::opkind::OpKind;
use std::collections::HashMap;

use super::{Dispatch, LatencyCategory};
use crate::affine::{AffineExpr, AffineMap};
use crate::attrkey::AttrKey;
use crate::context::CoreContext;
use crate::dtypes::DType;
use crate::env::ExecutionEnv;
use crate::ir::{Attr, Operation, Scalar, Ssa, Value};
use crate::memref::{DimSubscript, DistributedMemRef, IndirectAccessTile, MemRef, SubExpr};

pub fn register(d: &mut Dispatch) {
    d.register(
        OpKind::KtdpGetComputeTileId,
        LatencyCategory::Zero,
        get_compute_tile_id,
    );
    d.register(OpKind::KtdpCoreid, LatencyCategory::Zero, coreid);
    d.register(
        OpKind::KtdpConstructDistributedMemoryView,
        LatencyCategory::Zero,
        construct_distributed_memory_view,
    );
    d.register(
        OpKind::KtdpConstructIndirectAccessTile,
        LatencyCategory::Zero,
        construct_indirect_access_tile,
    );
}

/// `%g = ktdp.get_compute_tile_id : index`  (single-result form)
/// `%x, %y = ktdp.get_compute_tile_id : index, index`  (multi-result form)
///
/// Port of `ktdp__get_compute_tile_id`. The single-result form returns
/// `GridOps.gridid(context, 0)` — the executing core's grid x coordinate. The
/// multi-result form returns one grid coordinate per result dimension
/// (`d = 0..N`) as a tuple.
///
/// Python detects the multi-result case via `isinstance(op.result, str)`. The
/// Rust `Operation.result` is a single `Option<String>`, so the parser records
/// the result count in a `num_results` attribute; absent (or `1`) means the
/// single-result form. Mirrors `GridOps.gridid` == `context.get_grid_id(dim)`.
fn get_compute_tile_id(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    let num_dims = match op.attr(AttrKey::NumResults) {
        Some(Attr::Int(n)) if *n >= 1 => *n as usize,
        Some(Attr::Int(n)) => return Err(format!("get_compute_tile_id: invalid num_results {n}")),
        _ => 1,
    };

    if num_dims == 1 {
        return Ok(Some(Value::Index(ctx.get_grid_id(0) as i64)));
    }
    let ids = (0..num_dims)
        .map(|d| Value::Index(ctx.get_grid_id(d) as i64))
        .collect();
    Ok(Some(Value::Tuple(ids)))
}

/// `%ids = ktdp.coreid %x, %y, %z`
///
/// Port of `ktdp__coreid` -> `GridOps.coreid`. The operands resolve to grid
/// coordinates (`-1` = wildcard "all cores in that dimension"); the result is
/// the list of linear core ids matching the masked tuple, in linear order.
///
/// Mirrors `GridOps.coreid`: pad the coords to 3 dims with trailing zeros, then
/// `grid_executor.get_cores_in_group((x, y, z))`. `get_cores_in_group` is not
/// surfaced on the Rust `GridExecutor`, so the wildcard match is performed here
/// over `env.grid` using its linear<->grid transforms.
fn coreid(
    op: &Operation,
    ctx: &mut CoreContext,
    env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    let mut coords: Vec<i64> = op
        .operands
        .iter()
        .map(|&v| ctx.get_value(v).and_then(|v| scalar_i64(v, "coreid coord")))
        .collect::<Result<_, _>>()?;

    // Pad to 3 dims with trailing zeros, then read (x, y, z).
    while coords.len() < 3 {
        coords.push(0);
    }
    let mask = (coords[0], coords[1], coords[2]);

    let ids = cores_in_group(env, mask);
    Ok(Some(Value::Tuple(
        ids.into_iter().map(|id| Value::Index(id as i64)).collect(),
    )))
}

/// Linear core ids whose grid position matches `mask`. A `-1` in any axis is a
/// wildcard. Mirrors `GridExecutor.get_cores_in_group`.
fn cores_in_group(env: &ExecutionEnv, mask: (i64, i64, i64)) -> Vec<usize> {
    let mut out = Vec::new();
    for id in 0..env.grid.num_cores {
        let (x, y, z) = env.grid.linear_to_grid(id);
        let matches = (mask.0 == -1 || mask.0 == x as i64)
            && (mask.1 == -1 || mask.1 == y as i64)
            && (mask.2 == -1 || mask.2 == z as i64);
        if matches {
            out.push(id);
        }
    }
    out
}

/// `%R = ktdp.construct_distributed_memory_view (%a, %b, ... : types) : memref<...>`
///
/// Port of `ktdp__construct_distributed_memory_view`. Composes N per-partition
/// `MemRef`s (each carrying its own `coordinate_set` = B_i in global coords) into
/// one `DistributedMemRef`. Does NOT allocate or move data — partition routing
/// happens at access time in `distributed_tile_access`.
///
/// `DistributedMemRef::new` enforces the Python `__post_init__` invariants
/// (non-empty, every partition has a coordinate_set, matching dtypes).
fn construct_distributed_memory_view(
    op: &Operation,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    let partitions: Vec<MemRef> = op
        .operands
        .iter()
        .enumerate()
        .map(|(i, &name)| match ctx.get_value(name)? {
            Value::MemRef(m) => Ok(m.clone()),
            other => Err(format!(
                "construct_distributed_memory_view: operand {i} is {other:?}, expected MemRef"
            )),
        })
        .collect::<Result<_, _>>()?;

    let shape = int_list(op, AttrKey::Shape)?
        .iter()
        .map(|&n| n as usize)
        .collect::<Vec<_>>();
    let dtype = dtype_attr(op, AttrKey::Dtype)?;

    let dist = DistributedMemRef::new(partitions, shape, dtype)?;
    Ok(Some(Value::DistMemRef(dist)))
}

/// `%t = ktdp.construct_indirect_access_tile intermediate_variables(...) %X[...] {...}`
///
/// Port of `ktdp__construct_indirect_access_tile`. Builds the gather/scatter
/// descriptor: a primary memory view (`%X`), N index views (one per indirect
/// dim), and one `DimSubscript` per output dimension. The indirect LOAD/STORE
/// (`indirect_load` / `indirect_store`) that consumes this lives in memory-ops.
///
/// Attribute encoding (parser-populated):
/// - `shape`: `IntList` — output access-tile shape.
/// - `variables_space_set`: `AffineSet` — domain of the intermediate vars.
/// - `variables_space_order`: `AffineMap` (optional) — iteration order; normalized to `None` when identity, matching the Python parser.
/// - `dim_kinds`: `StrList` — per-dim kind, one of `"direct"` / `"direct_sub"` / `"direct_expr"` / `"indirect"`.
/// - `dim_data`: `IntList` — per-dim payload parallel to `dim_kinds`: variable index for `direct`, index-view index for `indirect`, ignored for `direct_expr` / `direct_sub`.
/// - `dim_map_N`: `AffineMap` for the Nth `direct_expr` dim, left-to-right.
/// - `dim_subs`: `AffineMapList` — one map per output dim, carrying the SUBSCRIPT
///   EXPRESSIONS of the `direct_sub` and `indirect` dims. Domain = the enumeration
///   point, symbols = `intermediate_vars`. See [`AttrKey::DimSubs`].
///
/// `op.operands[0]` is the primary memref; `op.operands[1..]` are the index
/// views, in indirect-dim order. Mirrors the Python handler's construction of
/// `IndirectAccessTile(parent_ref, shape, dim_subscripts, index_views, vss, vso)`.
fn construct_indirect_access_tile(
    op: &Operation<'static>,
    ctx: &mut CoreContext,
    _env: &ExecutionEnv,
) -> Result<Option<Value>, String> {
    if op.operands.is_empty() {
        return Err("construct_indirect_access_tile: missing primary memref operand".into());
    }
    let parent_ref = match ctx.get_value(op.operands[0])? {
        Value::MemRef(m) => m.clone(),
        other => {
            return Err(format!(
                "construct_indirect_access_tile: parent is {other:?}, expected MemRef"
            ));
        }
    };

    let index_views: Vec<MemRef> = op.operands[1..]
        .iter()
        .enumerate()
        .map(|(i, &name)| match ctx.get_value(name)? {
            Value::MemRef(m) => Ok(m.clone()),
            other => Err(format!(
                "construct_indirect_access_tile: index_view {i} is {other:?}, expected MemRef"
            )),
        })
        .collect::<Result<_, _>>()?;

    let shape = int_list(op, AttrKey::Shape)?
        .iter()
        .map(|&n| n as usize)
        .collect::<Vec<_>>();

    let variables_space_set =
        match op.attr(AttrKey::VariablesSpaceSet) {
            Some(Attr::AffineSet(s)) => s.clone(),
            _ => return Err(
                "construct_indirect_access_tile: missing/invalid 'variables_space_set' attribute"
                    .into(),
            ),
        };

    let variables_space_order = match op.attr(AttrKey::VariablesSpaceOrder) {
        // Python normalizes an identity order to None.
        Some(Attr::AffineMap(m)) if !m.is_identity() => Some(m.clone()),
        _ => None,
    };

    let dim_subscripts = parse_dim_subscripts(op, ctx, shape.len(), &index_views)?;

    let iat = IndirectAccessTile {
        parent_ref,
        shape,
        dim_subscripts,
        index_views,
        variables_space_set,
        variables_space_order,
        extra: HashMap::new(),
    };
    Ok(Some(Value::IndirectAccessTile(iat)))
}

/// Build the per-output-dim `DimSubscript` list from the `dim_kinds` /
/// `dim_data` / `dim_subs` attributes. Mirrors the Python `dim_subscripts`
/// resolution loop: the subscript expression of dim `d` is `dim_subs[d]`, whose
/// domain is the enumeration point and whose SYMBOLS are the `intermediate_vars`,
/// resolved to their concrete values against the value table here — the Rust
/// analogue of Python's `_resolve_node` folding `("ssa", "%name")` into
/// `("const", v)`.
///
/// ⚖️ THE KEY IS `dim_subs` AND NOT `dim_sub_<d>`. This doc comment named
/// `dim_sub_<d>` — the per-dim spelling the Python harness used — for as long as
/// the two arms below could not be built at all, and `AttrKey` never declared
/// such a key in any spelling. `dim_subs` is ONE list indexed by dimension,
/// because a per-dim key family in a closed enum has the ceiling `dim_map_0`
/// already demonstrates: only the first is declared, so the second dim of that
/// kind is unrepresentable.
///
/// A bare `direct` dim referencing an intermediate variable that is itself
/// bound in the value table (an outer SSA scalar listed in
/// `intermediate_variables`) is promoted to a constant `direct_sub` — Python's
/// "var case (a)". The legacy `direct_expr` kind (structurally-built tests) is
/// still honoured via the `dim_map_N` attributes.
fn parse_dim_subscripts(
    op: &Operation<'static>,
    ctx: &CoreContext,
    ndims: usize,
    index_views: &[MemRef],
) -> Result<Vec<DimSubscript>, String> {
    let kinds = match op.attr(AttrKey::DimKinds) {
        Some(Attr::StrList(v)) => *v,
        _ => {
            return Err(
                "construct_indirect_access_tile: missing/invalid 'dim_kinds' attribute".into(),
            );
        }
    };
    if kinds.len() != ndims {
        return Err(format!(
            "construct_indirect_access_tile: dim_kinds has {} entries but shape has {ndims} dims",
            kinds.len()
        ));
    }

    let data = match op.attr(AttrKey::DimData) {
        Some(Attr::IntList(v)) => v.to_vec(),
        // dim_data may be omitted only when no dim needs a payload.
        None => vec![0; ndims],
        Some(other) => {
            return Err(format!(
                "construct_indirect_access_tile: 'dim_data' is {other:?}, expected IntList"
            ));
        }
    };
    if data.len() != ndims {
        return Err(format!(
            "construct_indirect_access_tile: dim_data has {} entries but shape has {ndims} dims",
            data.len()
        ));
    }

    let intermediate_vars: &[Ssa] = match op.attr(AttrKey::IntermediateVars) {
        Some(Attr::Ssas(v)) => v,
        _ => &[],
    };

    // The per-dim subscript maps, and the concrete value of every intermediate
    // variable that IS an outer SSA scalar. `None` for one that is not bound in
    // the value table (a pure iteration variable); referencing such a one as a
    // symbol is refused by name in `sub_exprs`, never defaulted to 0 — a wrong
    // address computes a well-formed wrong answer.
    let dim_subs: Option<&[AffineMap<'static>]> = match op.attr(AttrKey::DimSubs) {
        Some(Attr::AffineMapList(v)) => {
            if v.len() != ndims {
                return Err(format!(
                    "construct_indirect_access_tile: dim_subs has {} map(s) but shape has \
                     {ndims} dims",
                    v.len()
                ));
            }
            Some(v)
        }
        None => None,
        Some(other) => {
            return Err(format!(
                "construct_indirect_access_tile: 'dim_subs' is {other:?}, expected AffineMapList"
            ));
        }
    };
    let syms: Vec<Option<i64>> = intermediate_vars
        .iter()
        .map(|&v| {
            ctx.get_value(v)
                .ok()
                .and_then(|x| scalar_i64(x, "dim_subs symbol").ok())
        })
        .collect();

    let mut subs = Vec::with_capacity(ndims);
    let mut expr_cursor = 0usize;
    for (d, kind) in kinds.iter().enumerate() {
        let sub = match *kind {
            "direct" => {
                // Python "var case (a)": an intermediate variable that is bound
                // in the value table is actually an outer SSA scalar — fold it
                // to a constant subscript so the SSA value (not the iterator
                // position, which would be 0 for a scalar dim) drives the coord.
                let var_index = data[d] as usize;
                match intermediate_vars
                    .get(var_index)
                    .and_then(|&v| ctx.get_value(v).ok())
                {
                    Some(v) => DimSubscript::DirectSub {
                        sub: SubExpr {
                            expr: AffineExpr::Const(scalar_i64(v, "construct_indirect")?),
                            syms: Vec::new(),
                        },
                    },
                    None => DimSubscript::Direct { var_index },
                }
            }
            // ⭐ THE SUBSCRIPT IS READ FROM `dim_subs`, NOT REBUILT HERE. An
            // `AffineExpr<'static>`'s children are BORROWS, so a tree cannot be
            // constructed while a handler runs — only read off the program. So the
            // PRODUCER states dim `d`'s subscript as `dim_subs[d]`, over the
            // enumeration point with `intermediate_vars` as symbols, and this arm
            // only resolves those symbols' values. A `direct_sub` dim with no
            // `dim_subs` is still refused: its subscript is its whole meaning.
            "direct_sub" => DimSubscript::DirectSub {
                sub: one_sub_expr(dim_subs, &syms, d, "direct_sub")?,
            },
            "indirect" => {
                let view = data[d] as usize;
                // One expression per INDEX-VIEW AXIS, dotted with that view's
                // strides by `resolve_idx_reads`. Absent `dim_subs` keeps the
                // legacy identity-subscript path — address the view by the
                // enumeration point itself — which is what the structural
                // `port_indirect_access` tests build.
                let idx_exprs = sub_exprs(dim_subs, &syms, d)?;
                // ⛔ THE ARITY IS CHECKED HERE BECAUSE `resolve_idx_reads` ZIPS.
                // `idx_exprs.iter().zip(&iv.strides)` stops at the shorter, so a
                // subscript list one short of the view's rank silently drops that
                // axis's contribution and reads a well-formed WRONG element.
                if let Some(iv) = index_views.get(view)
                    && !idx_exprs.is_empty()
                    && idx_exprs.len() != iv.strides.len()
                {
                    return Err(format!(
                        "construct_indirect_access_tile: dim {d} is indirect through \
                         index_view {view}, which is rank {}, but 'dim_subs' gives {} \
                         subscript expression(s). They are dotted with the view's strides, \
                         so a shorter list silently addresses the wrong element",
                        iv.strides.len(),
                        idx_exprs.len()
                    ));
                }
                DimSubscript::Indirect { view, idx_exprs }
            }
            "direct_expr" => {
                // Only the FIRST `direct_expr` dim is expressible: `AttrKey` declares
                // `dim_map_0` and no successors.
                if expr_cursor != 0 {
                    return Err(format!(
                        "construct_indirect_access_tile: dim {d} is the {}th direct_expr, but \
                         only `dim_map_0` is a declared attribute key",
                        expr_cursor + 1
                    ));
                }
                let map = match op.attr(AttrKey::DimMap0) {
                    Some(Attr::AffineMap(m)) => m.clone(),
                    _ => {
                        return Err(format!(
                            "construct_indirect_access_tile: dim {d} is direct_expr but \
                             attribute 'dim_map_0' is missing/invalid"
                        ));
                    }
                };
                expr_cursor += 1;
                DimSubscript::DirectExpr { map }
            }
            other => {
                return Err(format!(
                    "construct_indirect_access_tile: dim {d} has unknown kind {other:?} \
                     (expected direct/direct_sub/direct_expr/indirect)"
                ));
            }
        };
        subs.push(sub);
    }
    Ok(subs)
}

/// Dim `d`'s subscript expressions as [`SubExpr`]s, or an empty vector when the op
/// carries no `dim_subs` at all (the legacy identity-subscript path).
///
/// ⛔ A SYMBOL WITH NO RESOLVED VALUE IS A REFUSAL. `Sym(j)` names
/// `intermediate_vars[j]`; if that value is not bound in the value table there is
/// no address to compute, and substituting anything — 0 most temptingly — gathers
/// a well-formed WRONG row. So every symbol a map references is checked here,
/// before a single element is read.
fn sub_exprs(
    dim_subs: Option<&[AffineMap<'static>]>,
    syms: &[Option<i64>],
    d: usize,
) -> Result<Vec<SubExpr>, String> {
    let Some(maps) = dim_subs else {
        return Ok(Vec::new());
    };
    let map = &maps[d];
    let resolved = resolve_syms(map, syms, d)?;
    Ok(map
        .exprs
        .iter()
        .map(|e| SubExpr {
            expr: *e,
            syms: resolved.clone(),
        })
        .collect())
}

/// The single subscript expression of a one-subscript dim (`direct_sub`).
fn one_sub_expr(
    dim_subs: Option<&[AffineMap<'static>]>,
    syms: &[Option<i64>],
    d: usize,
    kind: &str,
) -> Result<SubExpr, String> {
    let mut got = sub_exprs(dim_subs, syms, d)?;
    if got.len() != 1 {
        return Err(format!(
            "construct_indirect_access_tile: dim {d} is {kind}, which is indexed by exactly ONE              subscript expression; 'dim_subs' supplies {}. State it as `dim_subs[{d}]` over the              enumeration point, with the captured scalars as symbols",
            got.len()
        ));
    }
    Ok(got.remove(0))
}

/// The symbol values one map needs, in `Sym` order, refusing an unresolved one.
fn resolve_syms(
    map: &AffineMap<'static>,
    syms: &[Option<i64>],
    d: usize,
) -> Result<Vec<i64>, String> {
    let mut highest: Option<usize> = None;
    for e in map.exprs {
        walk_syms(e, &mut |j| {
            highest = Some(highest.map_or(j, |h: usize| h.max(j)));
        });
    }
    let needed = highest.map_or(0, |h| h + 1);
    if needed > syms.len() {
        return Err(format!(
            "construct_indirect_access_tile: dim {d}'s subscript references s{} but              'intermediate_vars' names only {} value(s)",
            needed - 1,
            syms.len()
        ));
    }
    syms[..needed]
        .iter()
        .enumerate()
        .map(|(j, v)| {
            v.ok_or_else(|| {
                format!(
                    "construct_indirect_access_tile: dim {d}'s subscript references s{j} =                      intermediate_vars[{j}], which is not bound to a scalar in the value table.                      REFUSING rather than substituting a value: a wrong subscript gathers a                      well-formed wrong row"
                )
            })
        })
        .collect()
}

/// Call `f` with every `Sym` index in `e`.
fn walk_syms(e: &AffineExpr<'static>, f: &mut impl FnMut(usize)) {
    match e {
        AffineExpr::Sym(j) => f(*j),
        AffineExpr::Dim(_) | AffineExpr::Const(_) | AffineExpr::Ref(_) => {}
        AffineExpr::Neg(a) => walk_syms(a, f),
        AffineExpr::Add(a, b)
        | AffineExpr::Sub(a, b)
        | AffineExpr::Mul(a, b)
        | AffineExpr::FloorDiv(a, b)
        | AffineExpr::Mod(a, b)
        | AffineExpr::Max(a, b)
        | AffineExpr::Min(a, b) => {
            walk_syms(a, f);
            walk_syms(b, f);
        }
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

fn dtype_attr(op: &Operation, key: AttrKey) -> Result<DType, String> {
    match op.attr(key) {
        Some(Attr::Dtype(d)) => Ok(*d),
        _ => Err(format!(
            "{:?}: missing/invalid dtype attribute {key:?}",
            op.op_type
        )),
    }
}

fn scalar_i64(v: &Value, ctx: &str) -> Result<i64, String> {
    match v {
        Value::Index(i) => Ok(*i),
        Value::Scalar(Scalar::I32(i)) => Ok(*i as i64),
        Value::Scalar(Scalar::I64(i)) => Ok(*i),
        other => Err(format!("{ctx}: expected index/int, got {other:?}")),
    }
}

/// Split an indirect-access subscript into tokens.
///
/// This came from `ktir-core`'s affine TEXT PARSER, which is deleted — KTIR is constructed, never
/// parsed. `ktdp.construct_indirect_access_tile` still carries its subscript as an attribute
/// string, so the tokenizer lives with its one remaining caller until that attribute is typed too.
pub fn tokenise(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;

        // Whitespace — skip.
        if c.is_whitespace() {
            i += 1;
            continue;
        }

        // Two-char operators: `== >= <= ->`. `==` is matched before `>=` to
        // mirror the regex alternation order (group 4 in `_TOKEN_RE`).
        if i + 1 < bytes.len() {
            let pair = &text[i..i + 2];
            if matches!(pair, "==" | ">=" | "<=" | "->") {
                tokens.push(pair.to_string());
                i += 2;
                continue;
            }
        }

        // `%name` reference.
        if c == '%' {
            let start = i;
            i += 1;
            while i < bytes.len() && is_ident_char(bytes[i] as char) {
                i += 1;
            }
            tokens.push(text[start..i].to_string());
            continue;
        }

        // Bare identifier (letters / `_` then alphanumerics / `_`).
        if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            i += 1;
            while i < bytes.len() && is_ident_char(bytes[i] as char) {
                i += 1;
            }
            tokens.push(text[start..i].to_string());
            continue;
        }

        // Integer literal. A leading `-` is consumed as part of the number only
        // when it is immediately followed by a digit — otherwise it is the
        // subtraction / unary-minus operator (handled below). This matches the
        // regex group 3 `(-?\d+)` taking precedence over the `-` operator.
        if c.is_ascii_digit()
            || (c == '-' && i + 1 < bytes.len() && (bytes[i + 1] as char).is_ascii_digit())
        {
            let start = i;
            if c == '-' {
                i += 1;
            }
            while i < bytes.len() && (bytes[i] as char).is_ascii_digit() {
                i += 1;
            }
            tokens.push(text[start..i].to_string());
            continue;
        }

        // Single-char punctuation / operators.
        if matches!(c, '+' | '-' | '*' | '(' | ')' | ',' | ':' | '[' | ']') {
            tokens.push(c.to_string());
            i += 1;
            continue;
        }

        // Unknown char — skip (Python falls through to `pos += 1`).
        i += 1;
    }
    tokens
}

fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::affine::{AffineExpr, AffineSet, Constraint, ConstraintKind};
    use crate::dialects::Dispatch;
    use crate::env::{ExecutionEnv, GridExecutor};
    use crate::interpreter::{execute_ops, single_core_context};
    use crate::memref::{CoordinateSet, MemorySpace};
    use crate::test_support::Ops;

    fn run_on(
        ops: &[Operation<'static>],
        ctx: &mut CoreContext,
        grid: (usize, usize, usize),
    ) -> Result<(), String> {
        let dispatch = Dispatch::new();
        let grid = GridExecutor::new(grid);
        let env = ExecutionEnv::new(&dispatch, &grid);
        execute_ops(ops, ctx, &env)
    }

    fn hbm_part(ops: &Ops, base_stick: i64, lo: &[i64], hi: &[i64]) -> MemRef {
        MemRef {
            base_ptr: base_stick,
            shape: vec![4, 4],
            strides: vec![4, 1],
            space: MemorySpace::Hbm,
            dtype: DType::F16,
            coordinate_set: Some(ops.box_set(lo, hi)),
        }
    }

    fn lx_view(shape: Vec<usize>) -> MemRef {
        MemRef {
            base_ptr: 0,
            shape,
            strides: vec![1],
            space: MemorySpace::Lx { core_id: None },
            dtype: DType::F16,
            coordinate_set: None,
        }
    }

    /// 2-d variable space, trivially satisfiable constraint.
    fn vss_2d(ops: &Ops) -> AffineSet<'static> {
        AffineSet {
            num_dims: 2,
            num_syms: 0,
            constraints: ops.arena().constraints(vec![Constraint {
                expr: AffineExpr::Dim(0),
                kind: ConstraintKind::GreaterEq,
            }]),
        }
    }

    // --- get_compute_tile_id -------------------------------------------------

    #[test]
    fn compute_tile_id_single_returns_grid_x() {
        // core 5 in a (4,2,1) grid => x = 5 % 4 = 1.
        let g = GridExecutor::new((4, 2, 1));
        let (gx, gy, gz) = g.linear_to_grid(5);
        let mut ops = Ops::new();
        let mut ctx = single_core_context();
        ctx.grid_pos = (gx, gy, gz);

        let g_ssa = ops.ssa("%g");
        let op = ops.op(Some("%g"), OpKind::KtdpGetComputeTileId, &[]);
        run_on(&[op], &mut ctx, (4, 2, 1)).unwrap();
        match ctx.get_value(g_ssa).unwrap() {
            Value::Index(i) => assert_eq!(*i, gx as i64),
            other => panic!("expected Index, got {other:?}"),
        }
        assert_eq!(gx, 1);
    }

    #[test]
    fn compute_tile_id_multi_returns_tuple_of_coords() {
        let mut ops = Ops::new();
        let mut ctx = single_core_context();
        ctx.grid_pos = (1, 2, 3);
        let g = ops.ssa("%g");
        let op = ops.op(Some("%g"), OpKind::KtdpGetComputeTileId, &[]);
        let op = ops.attr(op, AttrKey::NumResults, Attr::Int(3));
        run_on(&[op], &mut ctx, (4, 4, 4)).unwrap();
        match ctx.get_value(g).unwrap() {
            Value::Tuple(t) => {
                let got: Vec<i64> = t
                    .iter()
                    .map(|v| match v {
                        Value::Index(i) => *i,
                        o => panic!("expected Index, got {o:?}"),
                    })
                    .collect();
                assert_eq!(got, vec![1, 2, 3]);
            }
            other => panic!("expected Tuple, got {other:?}"),
        }
    }

    // --- coreid -------------------------------------------------------------

    #[test]
    fn coreid_wildcard_x_returns_full_row() {
        // grid (4, 2, 1); mask (-1, 1, 0) => all x with y=1, z=0.
        let mut ops = Ops::new();
        let mut ctx = single_core_context();
        ctx.set_value(ops.ssa("%x"), Value::Index(-1));
        ctx.set_value(ops.ssa("%y"), Value::Index(1));
        ctx.set_value(ops.ssa("%z"), Value::Index(0));
        let ids = ops.ssa("%ids");
        let op = ops.op(Some("%ids"), OpKind::KtdpCoreid, &["%x", "%y", "%z"]);
        run_on(&[op], &mut ctx, (4, 2, 1)).unwrap();

        // y=1 => linear ids 4,5,6,7 (z*(nx*ny)+y*nx+x = 0 + 4 + x).
        let g = GridExecutor::new((4, 2, 1));
        let expect: Vec<i64> = (0..4).map(|x| g.grid_to_linear(x, 1, 0) as i64).collect();
        match ctx.get_value(ids).unwrap() {
            Value::Tuple(t) => {
                let got: Vec<i64> = t
                    .iter()
                    .map(|v| match v {
                        Value::Index(i) => *i,
                        o => panic!("expected Index, got {o:?}"),
                    })
                    .collect();
                assert_eq!(got, expect);
                assert_eq!(got, vec![4, 5, 6, 7]);
            }
            other => panic!("expected Tuple, got {other:?}"),
        }
    }

    #[test]
    fn coreid_exact_match_is_single_core() {
        let mut ops = Ops::new();
        let mut ctx = single_core_context();
        ctx.set_value(ops.ssa("%x"), Value::Index(2));
        ctx.set_value(ops.ssa("%y"), Value::Index(0));
        let ids = ops.ssa("%ids");
        let op = ops.op(Some("%ids"), OpKind::KtdpCoreid, &["%x", "%y"]);
        // only 2 operands: padded to (2, 0, 0).
        run_on(&[op], &mut ctx, (4, 2, 1)).unwrap();
        match ctx.get_value(ids).unwrap() {
            Value::Tuple(t) => {
                assert_eq!(t.len(), 1);
                // grid_to_linear(2, 0, 0) = 2.
                assert!(matches!(t[0], Value::Index(2)));
            }
            other => panic!("expected Tuple, got {other:?}"),
        }
    }

    #[test]
    fn coreid_all_wildcards_returns_every_core() {
        let mut ops = Ops::new();
        let mut ctx = single_core_context();
        ctx.set_value(ops.ssa("%x"), Value::Index(-1));
        ctx.set_value(ops.ssa("%y"), Value::Index(-1));
        ctx.set_value(ops.ssa("%z"), Value::Index(-1));
        let ids = ops.ssa("%ids");
        let op = ops.op(Some("%ids"), OpKind::KtdpCoreid, &["%x", "%y", "%z"]);
        run_on(&[op], &mut ctx, (2, 2, 1)).unwrap();
        match ctx.get_value(ids).unwrap() {
            Value::Tuple(t) => assert_eq!(t.len(), 4),
            other => panic!("expected Tuple, got {other:?}"),
        }
    }

    // --- construct_distributed_memory_view ----------------------------------

    #[test]
    fn distributed_view_composes_partitions() {
        let mut ops = Ops::new();
        let mut ctx = single_core_context();
        let a_part = hbm_part(&ops, 0, &[0, 0], &[3, 3]);
        ctx.set_value(ops.ssa("%a"), Value::MemRef(a_part));
        let b_part = hbm_part(&ops, 16, &[4, 0], &[7, 3]);
        ctx.set_value(ops.ssa("%b"), Value::MemRef(b_part));

        let r = ops.ssa("%R");
        let op = ops.op(
            Some("%R"),
            OpKind::KtdpConstructDistributedMemoryView,
            &["%a", "%b"],
        );
        let shape = ops.int_list(vec![8, 4]);
        let op = ops.attr(op, AttrKey::Shape, shape);
        let op = ops.attr(op, AttrKey::Dtype, Attr::Dtype(DType::F16));
        run_on(&[op], &mut ctx, (1, 1, 1)).unwrap();

        match ctx.get_value(r).unwrap() {
            Value::DistMemRef(d) => {
                assert_eq!(d.partitions.len(), 2);
                assert_eq!(d.shape, vec![8, 4]);
                assert_eq!(d.dtype, DType::F16);
                // partition routing: global coord [1,1] -> partition 0.
                let (i0, _) = d.find_partition(&[1, 1], &[]).unwrap();
                assert_eq!(i0, 0);
                // global coord [5,1] -> partition 1.
                let (i1, _) = d.find_partition(&[5, 1], &[]).unwrap();
                assert_eq!(i1, 1);
            }
            other => panic!("expected DistMemRef, got {other:?}"),
        }
    }

    #[test]
    fn distributed_view_rejects_non_memref_operand() {
        let mut ops = Ops::new();
        let mut ctx = single_core_context();
        let a_part = hbm_part(&ops, 0, &[0, 0], &[3, 3]);
        ctx.set_value(ops.ssa("%a"), Value::MemRef(a_part));
        ctx.set_value(ops.ssa("%b"), Value::Index(7));
        let op = ops.op(
            Some("%R"),
            OpKind::KtdpConstructDistributedMemoryView,
            &["%a", "%b"],
        );
        let shape = ops.int_list(vec![8, 4]);
        let op = ops.attr(op, AttrKey::Shape, shape);
        let op = ops.attr(op, AttrKey::Dtype, Attr::Dtype(DType::F16));
        let err = run_on(&[op], &mut ctx, (1, 1, 1)).unwrap_err();
        assert!(err.contains("expected MemRef"));
    }

    #[test]
    fn distributed_view_requires_coordinate_set() {
        // A partition without a coordinate_set is rejected by DistributedMemRef::new.
        let mut ops = Ops::new();
        let mut ctx = single_core_context();
        let mut p = hbm_part(&ops, 0, &[0, 0], &[3, 3]);
        p.coordinate_set = None;
        ctx.set_value(ops.ssa("%a"), Value::MemRef(p));
        let op = ops.op(
            Some("%R"),
            OpKind::KtdpConstructDistributedMemoryView,
            &["%a"],
        );
        let shape = ops.int_list(vec![4, 4]);
        let op = ops.attr(op, AttrKey::Shape, shape);
        let op = ops.attr(op, AttrKey::Dtype, Attr::Dtype(DType::F16));
        let err = run_on(&[op], &mut ctx, (1, 1, 1)).unwrap_err();
        assert!(err.contains("coordinate_set"));
    }

    #[test]
    fn distributed_view_dtype_mismatch_is_rejected() {
        let mut ops = Ops::new();
        let mut ctx = single_core_context();
        let a_part = hbm_part(&ops, 0, &[0, 0], &[3, 3]);
        ctx.set_value(ops.ssa("%a"), Value::MemRef(a_part));
        let op = ops.op(
            Some("%R"),
            OpKind::KtdpConstructDistributedMemoryView,
            &["%a"],
        );
        let shape = ops.int_list(vec![4, 4]);
        let op = ops.attr(op, AttrKey::Shape, shape);
        // partition is f16 but the view claims f32.
        let op = ops.attr(op, AttrKey::Dtype, Attr::Dtype(DType::F32));
        let err = run_on(&[op], &mut ctx, (1, 1, 1)).unwrap_err();
        assert!(err.contains("dtype"));
    }

    // --- construct_indirect_access_tile -------------------------------------

    #[test]
    fn indirect_tile_builds_descriptor() {
        // X[ind(IDX[%m,%k]), (%k)] over intermediate vars (%m, %k).
        let mut ops = Ops::new();
        let mut ctx = single_core_context();
        ctx.set_value(ops.ssa("%X"), Value::MemRef(lx_view(vec![16, 16])));
        ctx.set_value(ops.ssa("%IDX"), Value::MemRef(lx_view(vec![4, 4])));

        let t = ops.ssa("%t");
        let op = ops.op(
            Some("%t"),
            OpKind::KtdpConstructIndirectAccessTile,
            &["%X", "%IDX"],
        );
        let shape = ops.int_list(vec![4, 4]);
        let op = ops.attr(op, AttrKey::Shape, shape);
        let vss = vss_2d(&ops);
        let op = ops.attr(op, AttrKey::VariablesSpaceSet, Attr::AffineSet(vss));
        let kinds = ops.str_list(&["indirect", "direct"]);
        let op = ops.attr(op, AttrKey::DimKinds, kinds);
        // dim 0: indirect via index_view 0; dim 1: direct via var index 1.
        let data = ops.int_list(vec![0, 1]);
        let op = ops.attr(op, AttrKey::DimData, data);

        run_on(&[op], &mut ctx, (1, 1, 1)).unwrap();

        match ctx.get_value(t).unwrap() {
            Value::IndirectAccessTile(iat) => {
                assert_eq!(iat.shape, vec![4, 4]);
                assert_eq!(iat.index_views.len(), 1);
                assert_eq!(iat.dim_subscripts.len(), 2);
                assert!(matches!(
                    iat.dim_subscripts[0],
                    DimSubscript::Indirect { view: 0, .. }
                ));
                assert!(matches!(
                    iat.dim_subscripts[1],
                    DimSubscript::Direct { var_index: 1 }
                ));
                assert!(iat.variables_space_order.is_none());
                assert_eq!(iat.parent_ref.shape, vec![16, 16]);
            }
            other => panic!("expected IndirectAccessTile, got {other:?}"),
        }
    }

    #[test]
    fn indirect_tile_direct_expr_pulls_map() {
        let mut ops = Ops::new();
        let mut ctx = single_core_context();
        ctx.set_value(ops.ssa("%X"), Value::MemRef(lx_view(vec![16])));

        let t = ops.ssa("%t");
        let op = ops.op(Some("%t"), OpKind::KtdpConstructIndirectAccessTile, &["%X"]);
        let shape = ops.int_list(vec![4]);
        let op = ops.attr(op, AttrKey::Shape, shape);
        let vss = vss_2d(&ops);
        let op = ops.attr(op, AttrKey::VariablesSpaceSet, Attr::AffineSet(vss));
        let kinds = ops.str_list(&["direct_expr"]);
        let op = ops.attr(op, AttrKey::DimKinds, kinds);
        let data = ops.int_list(vec![0]);
        let op = ops.attr(op, AttrKey::DimData, data);
        let op = ops.attr(op, AttrKey::DimMap0, Attr::AffineMap(ops.identity_map(1)));

        run_on(&[op], &mut ctx, (1, 1, 1)).unwrap();
        match ctx.get_value(t).unwrap() {
            Value::IndirectAccessTile(iat) => {
                assert_eq!(iat.dim_subscripts.len(), 1);
                match &iat.dim_subscripts[0] {
                    DimSubscript::DirectExpr { map } => assert!(map.is_identity()),
                    other => panic!("expected DirectExpr, got {other:?}"),
                }
            }
            other => panic!("expected IndirectAccessTile, got {other:?}"),
        }
    }

    #[test]
    fn indirect_tile_nonidentity_order_is_kept() {
        let mut ops = Ops::new();
        let mut ctx = single_core_context();
        ctx.set_value(ops.ssa("%X"), Value::MemRef(lx_view(vec![16, 16])));
        ctx.set_value(ops.ssa("%IDX"), Value::MemRef(lx_view(vec![4, 4])));

        // swap order (d0,d1) -> (d1,d0) is not identity, so it must be retained.
        let swap = ops.perm_map(2, &[1, 0]);
        let t = ops.ssa("%t");
        let op = ops.op(
            Some("%t"),
            OpKind::KtdpConstructIndirectAccessTile,
            &["%X", "%IDX"],
        );
        let shape = ops.int_list(vec![4, 4]);
        let op = ops.attr(op, AttrKey::Shape, shape);
        let vss = vss_2d(&ops);
        let op = ops.attr(op, AttrKey::VariablesSpaceSet, Attr::AffineSet(vss));
        let op = ops.attr(
            op,
            AttrKey::VariablesSpaceOrder,
            Attr::AffineMap(swap.clone()),
        );
        let kinds = ops.str_list(&["indirect", "direct"]);
        let op = ops.attr(op, AttrKey::DimKinds, kinds);
        let data = ops.int_list(vec![0, 1]);
        let op = ops.attr(op, AttrKey::DimData, data);

        run_on(&[op], &mut ctx, (1, 1, 1)).unwrap();
        match ctx.get_value(t).unwrap() {
            Value::IndirectAccessTile(iat) => {
                assert_eq!(iat.variables_space_order.as_ref().unwrap(), &swap);
            }
            other => panic!("expected IndirectAccessTile, got {other:?}"),
        }
    }

    #[test]
    fn indirect_tile_identity_order_normalized_to_none() {
        let mut ops = Ops::new();
        let mut ctx = single_core_context();
        ctx.set_value(ops.ssa("%X"), Value::MemRef(lx_view(vec![16])));
        let t = ops.ssa("%t");
        let op = ops.op(Some("%t"), OpKind::KtdpConstructIndirectAccessTile, &["%X"]);
        let shape = ops.int_list(vec![4]);
        let op = ops.attr(op, AttrKey::Shape, shape);
        let vss = vss_2d(&ops);
        let op = ops.attr(op, AttrKey::VariablesSpaceSet, Attr::AffineSet(vss));
        let identity = ops.identity_map(2);
        let op = ops.attr(op, AttrKey::VariablesSpaceOrder, Attr::AffineMap(identity));
        let kinds = ops.str_list(&["direct"]);
        let op = ops.attr(op, AttrKey::DimKinds, kinds);
        let data = ops.int_list(vec![0]);
        let op = ops.attr(op, AttrKey::DimData, data);
        run_on(&[op], &mut ctx, (1, 1, 1)).unwrap();
        match ctx.get_value(t).unwrap() {
            Value::IndirectAccessTile(iat) => assert!(iat.variables_space_order.is_none()),
            other => panic!("expected IndirectAccessTile, got {other:?}"),
        }
    }

    #[test]
    fn indirect_tile_rejects_unknown_kind() {
        let mut ops = Ops::new();
        let mut ctx = single_core_context();
        ctx.set_value(ops.ssa("%X"), Value::MemRef(lx_view(vec![16])));
        let op = ops.op(Some("%t"), OpKind::KtdpConstructIndirectAccessTile, &["%X"]);
        let shape = ops.int_list(vec![4]);
        let op = ops.attr(op, AttrKey::Shape, shape);
        let vss = vss_2d(&ops);
        let op = ops.attr(op, AttrKey::VariablesSpaceSet, Attr::AffineSet(vss));
        let kinds = ops.str_list(&["bogus"]);
        let op = ops.attr(op, AttrKey::DimKinds, kinds);
        let err = run_on(&[op], &mut ctx, (1, 1, 1)).unwrap_err();
        assert!(err.contains("unknown kind"));
    }

    #[test]
    fn indirect_tile_dim_kinds_count_must_match_shape() {
        let mut ops = Ops::new();
        let mut ctx = single_core_context();
        ctx.set_value(ops.ssa("%X"), Value::MemRef(lx_view(vec![16, 16])));
        let op = ops.op(Some("%t"), OpKind::KtdpConstructIndirectAccessTile, &["%X"]);
        let shape = ops.int_list(vec![4, 4]);
        let op = ops.attr(op, AttrKey::Shape, shape);
        let vss = vss_2d(&ops);
        let op = ops.attr(op, AttrKey::VariablesSpaceSet, Attr::AffineSet(vss));
        // one kind but shape has two dims.
        let kinds = ops.str_list(&["direct"]);
        let op = ops.attr(op, AttrKey::DimKinds, kinds);
        let data = ops.int_list(vec![0]);
        let op = ops.attr(op, AttrKey::DimData, data);
        let err = run_on(&[op], &mut ctx, (1, 1, 1)).unwrap_err();
        assert!(err.contains("dim_kinds"));
    }

    #[test]
    fn indirect_tile_rejects_non_memref_parent() {
        let mut ops = Ops::new();
        let mut ctx = single_core_context();
        ctx.set_value(ops.ssa("%X"), Value::Index(3));
        let op = ops.op(Some("%t"), OpKind::KtdpConstructIndirectAccessTile, &["%X"]);
        let shape = ops.int_list(vec![4]);
        let op = ops.attr(op, AttrKey::Shape, shape);
        let vss = vss_2d(&ops);
        let op = ops.attr(op, AttrKey::VariablesSpaceSet, Attr::AffineSet(vss));
        let kinds = ops.str_list(&["direct"]);
        let op = ops.attr(op, AttrKey::DimKinds, kinds);
        let err = run_on(&[op], &mut ctx, (1, 1, 1)).unwrap_err();
        assert!(err.contains("expected MemRef"));
    }

    #[test]
    fn coordinate_set_import_surface_is_available() {
        // Smoke test that CoordinateSet is the right import surface for memref.
        let cs = CoordinateSet::Points(vec![vec![0, 0]]);
        assert!(matches!(cs, CoordinateSet::Points(_)));
    }

    // NOTE: the old `subscript_parses_floordiv_and_mod` /
    // `subscript_floordiv_mod_bind_tighter_than_add` tests exercised
    // `parse_sub_expr`, a text-expression parser that no longer exists
    // anywhere in this crate (KTIR is constructed, never parsed) — deleted
    // along with the rest of the retired textual-MLIR-parser surface, not
    // ported.
}
