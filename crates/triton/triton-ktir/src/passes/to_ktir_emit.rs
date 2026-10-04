// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Licensed under the MIT terms in the crate root.

//! THE HANDOFF: our Triton-free module -> `ktir_core::ir::IRFunction`.
//!
//! We generate KTIR; their lowering consumes KTIR. This is the one place the two meet,
//! and past it everything is their type entire -- no wrapper, no shadow copy, nothing of
//! ours riding along.
//!
//! # WHY A PER-OP MAP HERE IS NOT THE DUPLICATION WE DELETED
//!
//! An earlier attempt at this was a separate crate translating a whole parallel KTIR type
//! into theirs. That was duplication and it would have rotted: it had to track a type
//! someone else is actively changing, which is exactly how `consumer.rs` came to pin a
//! vocabulary from files that no longer existed and still report green.
//!
//! What remains is different in kind. Our interior IR is GENERIC -- one `Op` with a
//! variant per spelling, `tt.*` and `ktdp.*` coexisting mid-conversion, which is what
//! MLIR itself is and what the C++ passes this is ported from operate on. Theirs is
//! TYPED: 123 variants, no `tt.*` at all, so a Triton op is not "unsupported" but
//! unrepresentable. Crossing from a generic IR to a typed one maps every op, including
//! the ops that do not change. That map is inherent to the boundary, not a second design
//! of the same thing -- and because it is total and fails closed, an op we cannot express
//! is a REFUSAL naming the op, never a silently dropped or guessed one.
//!
//! # WHAT THIS REFUSES, AND WHY EACH REFUSAL IS THE RIGHT ANSWER
//!
//! * **`tt.*` anything.** Their `OpKind` has zero `Tt` variants. If one reaches here,
//!   [`crate::passes::to_ktir`] left it behind, and a module that LOOKS converted and is
//!   not is the failure that pass exists to prevent.
//! * **`ktdf.corelet_plan` / `ktdf.corelet`.** Dropped, not refused: the corelet plan is
//!   discarded before SuperDSC on every path, so carrying it would be noise. Dropping is
//!   recorded here rather than left implicit.
//! * **An op with more than one result.** Their `Operation.result` is `Option<Ssa>`. No
//!   pass in this crate builds a multi-result op, so this is a guard against a future one
//!   rather than a live gap.
//! * **A multi-block region.** Our own parser refuses these already; stated again because
//!   their `regions: &[&[Operation]]` cannot express a second block even in principle.
//! * **An affine form outside the subset we emit.** See [`affine_map`] / [`affine_set`].

use ktir_core::affine::{AffineExpr as KExpr, AffineMap as KMap, AffineSet as KSet, Constraint, ConstraintKind};
use ktir_core::arena::Arena;
use ktir_core::attrkey::AttrKey as KAttrKey;
use ktir_core::dtypes::DType as KDType;
use ktir_core::ir::{Attr as KAttr, IRFunction, Operation, Ssa as KSsa};
use ktir_core::irtype::IrType as KIrType;
use ktir_core::opkind::OpKind as KOpKind;

use crate::ir::{Attr, AttrKey, DType, IrType, Module, Op, OpKind, Region, Ssa};
use crate::{Refusal, Result};

const PASS: &str = "spyre-ktir-handoff";

fn refuse(why: impl std::fmt::Display) -> Refusal {
    Refusal::new(PASS, why.to_string())
}

/// `ln(2)`, for the `math.exp2` decomposition.
const LN2: f64 = std::f64::consts::LN_2;

//===----------------------------------------------------------------------===//
// Scalars, types
//===----------------------------------------------------------------------===//

fn ssa(v: Ssa) -> KSsa {
    KSsa(v.0)
}

/// Our element type as theirs.
///
/// `Index` has no `DType` counterpart -- their index-ness lives in `IrType::Index`, not
/// in an element type -- so it is refused HERE rather than mapped onto `I32`. An index
/// silently becoming a 32-bit integer is an address bug that would survive to a device
/// program.
fn dtype(d: DType) -> Result<KDType> {
    Ok(match d {
        DType::F16 => KDType::F16,
        DType::F32 => KDType::F32,
        DType::I1 => KDType::Bool,
        // Their `I32` doc says "also: si32, index", so signed 32-bit is the same variant.
        DType::I32 | DType::SI32 => KDType::I32,
        DType::I64 => KDType::I64,
        // Same variant, same spelling -- theirs is the device's own. This is the arm
        // that lights the fp8 door: a weight view whose elem maps through here stamps
        // `AttrKey::Dtype = Fp8E4m3` (via `state_shape_and_dtype`), which
        // `ktir-superdsc`'s `regions()` reads to set `is_fp8` and the arity-3
        // `matmul_fp8_descriptors` door keys on. A load's RESULT never takes this arm:
        // the kernel widens with `.to(tl.float16)` (the frontend refuses a mixed
        // `tl.dot`), so results are F16 exactly as in scratchy's own producer
        // (`KTIR_ELEM`).
        DType::Fp8E4m3 => KDType::Fp8E4m3,
        DType::Index => {
            return Err(refuse(
                "`index` is not an element type in their model: it is `IrType::Index`. \
                 Mapping it onto i32 here would turn an index into a 32-bit integer \
                 silently, so it is refused and the caller must have produced \
                 `IrType::Index` instead",
            ))
        }
    })
}

fn irtype<'a>(t: &IrType, a: &'a Arena) -> Result<KIrType<'a>> {
    Ok(match t {
        IrType::Tensor { dims, elem } => {
            KIrType::Tensor { dims: a.ints(dims.clone()), elem: dtype(*elem)? }
        }
        IrType::MemRef { dims, elem } => {
            KIrType::MemRef { dims: a.ints(dims.clone()), elem: dtype(*elem)? }
        }
        IrType::AccessTile { dims } => KIrType::AccessTile { dims: a.ints(dims.clone()) },
        IrType::Index => KIrType::Index,
        IrType::Scalar(d) => KIrType::Scalar(dtype(*d)?),
        // These three are ttir-only or an escape hatch. Reaching here means `to_ktir`
        // did not finish, and the whole point of that pass is that it must.
        IrType::Ptr { .. } => {
            return Err(refuse(
                "`!tt.ptr` survived to the handoff: to_ktir rewrites pointers to `index` \
                 and their type has no pointer at all",
            ))
        }
        IrType::TensorDesc { .. } => {
            return Err(refuse(
                "`!tt.tensordesc` survived to the handoff: the descriptor patterns consume \
                 it and their type has no counterpart",
            ))
        }
        IrType::Verbatim(s) => {
            return Err(refuse(format!(
                "type `{s}` is not modelled, so it cannot be expressed in their `IrType`. \
                 It is refused rather than passed through as text"
            )))
        }
    })
}

//===----------------------------------------------------------------------===//
// Affine
//===----------------------------------------------------------------------===//

/// The bare-`dN` result list our passes emit, as their `AffineMap`.
///
/// # THE SUBSET IS DELIBERATE AND IT IS CHECKED
///
/// Every map these passes build is an identity or a permutation of one --
/// `affine_map<(d0, d1) -> (d0, d1)>` and its reorderings. So this reads exactly that
/// form and REFUSES anything else, naming the text it could not read, rather than
/// implementing a general affine parser whose extra capability nothing exercises and no
/// test covers. Their `affine.rs` is 1,378 lines; a partial reimplementation of it here
/// that silently mishandles a `floordiv` would be the worst of the options.
pub fn affine_map<'a>(text: &str, a: &'a Arena) -> Result<KMap<'a>> {
    let body = text
        .trim()
        .strip_prefix("affine_map<")
        .and_then(|s| s.strip_suffix('>'))
        .unwrap_or(text.trim());
    let (dims, results) = body
        .split_once("->")
        .ok_or_else(|| refuse(format!("affine map `{text}` has no `->`")))?;

    let num_dims = dims.trim().trim_start_matches('(').trim_end_matches(')').split(',')
        .filter(|s| !s.trim().is_empty())
        .count();

    let mut exprs = Vec::new();
    for r in results.trim().trim_start_matches('(').trim_end_matches(')').split(',') {
        let r = r.trim();
        if r.is_empty() {
            continue;
        }
        exprs.push(affine_result(r, text, num_dims, a)?);
    }
    Ok(KMap { num_dims, num_syms: 0, exprs: a.exprs(exprs) })
}

/// One affine map result term.
///
/// The forms these passes emit, and nothing wider: a bare dimension `dN`, and the sum
/// `dN + dM` or `dN + K` that the gather's index arithmetic uses (`embedding_granite`'s
/// `(d0, d1, d2, d3) -> (d0 + d2)`). Their `AffineExpr` has `Add`, `Dim` and `Const`, so
/// each is exact rather than approximated. Anything else -- a product, a floordiv, a
/// symbol -- is refused with the text, because a partial reimplementation of their
/// 1,378-line `affine.rs` that silently mishandles one operator is the worst option
/// available.
fn affine_result<'a>(
    term: &str,
    whole: &str,
    num_dims: usize,
    a: &'a Arena,
) -> Result<KExpr<'a>> {
    let dim = |s: &str| -> Option<usize> {
        s.trim().strip_prefix('d').and_then(|n| n.parse::<usize>().ok())
    };
    let check = |i: usize| -> Result<usize> {
        if i >= num_dims {
            return Err(refuse(format!(
                "affine map `{whole}`: `d{i}` is out of range for {num_dims} dimension(s)"
            )));
        }
        Ok(i)
    };

    if let Some(i) = dim(term) {
        return Ok(KExpr::Dim(check(i)?));
    }
    if let Some((l, r)) = term.split_once('+') {
        let li = dim(l).ok_or_else(|| {
            refuse(format!("affine map `{whole}`: `{l}` is not a dimension"))
        })?;
        let lhs = KExpr::Dim(check(li)?);
        let rhs = if let Some(ri) = dim(r) {
            KExpr::Dim(check(ri)?)
        } else {
            KExpr::Const(r.trim().parse::<i64>().map_err(|_| {
                refuse(format!(
                    "affine map `{whole}`: `{r}` is neither a dimension nor an integer"
                ))
            })?)
        };
        return Ok(KExpr::Add(a.expr(lhs), a.expr(rhs)));
    }
    Err(refuse(format!(
        "affine map `{whole}`: result `{term}` is not `dN`, `dN + dM` or `dN + K`. These \
         passes emit no other form, so a wider one is refused rather than half-read"
    )))
}

/// The box constraint set our passes emit, as their `AffineSet`.
///
/// The only form emitted is a per-dimension box:
///
/// ```text
///   affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 127 >= 0)>
/// ```
///
/// which is `0 <= dN <= bound` per dimension. Their `ConstraintKind::GreaterEq` is `expr
/// >= 0`, the same convention, so each clause maps one-to-one. Anything else -- an
/// > equality, a symbol, a product -- is refused by name.
pub fn affine_set<'a>(text: &str, a: &'a Arena) -> Result<KSet<'a>> {
    let body = text
        .trim()
        .strip_prefix("affine_set<")
        .and_then(|s| s.strip_suffix('>'))
        .unwrap_or(text.trim());
    let (dims, clauses) = body
        .split_once(':')
        .ok_or_else(|| refuse(format!("affine set `{text}` has no `:`")))?;

    let num_dims = dims.trim().trim_start_matches('(').trim_end_matches(')').split(',')
        .filter(|s| !s.trim().is_empty())
        .count();

    let mut out = Vec::new();
    for c in clauses.trim().trim_start_matches('(').trim_end_matches(')').split(',') {
        let c = c.trim();
        if c.is_empty() {
            continue;
        }
        let lhs = c.strip_suffix(">= 0").map(str::trim).ok_or_else(|| {
            refuse(format!(
                "affine set `{text}`: clause `{c}` is not `<expr> >= 0`. These passes emit \
                 only box constraints in that form; an equality or a symbolic bound is \
                 refused rather than approximated"
            ))
        })?;

        let expr = if let Some(n) = lhs.strip_prefix('d') {
            // `dN` -- the lower bound.
            let idx: usize = n.trim().parse().map_err(|_| {
                refuse(format!("affine set `{text}`: `{lhs}` is not a dimension"))
            })?;
            KExpr::Dim(idx)
        } else if let Some(rest) = lhs.strip_prefix("-d") {
            // `-dN + K` -- the upper bound.
            let (n, k) = rest.split_once('+').ok_or_else(|| {
                refuse(format!(
                    "affine set `{text}`: `{lhs}` looks like an upper bound but has no `+ \
                     <bound>`"
                ))
            })?;
            let idx: usize = n.trim().parse().map_err(|_| {
                refuse(format!("affine set `{text}`: `{lhs}` has no dimension index"))
            })?;
            let bound: i64 = k.trim().parse().map_err(|_| {
                refuse(format!("affine set `{text}`: `{lhs}` has a non-integer bound"))
            })?;
            KExpr::Add(
                a.expr(KExpr::Neg(a.expr(KExpr::Dim(idx)))),
                a.expr(KExpr::Const(bound)),
            )
        } else {
            return Err(refuse(format!(
                "affine set `{text}`: clause `{c}` is neither `dN >= 0` nor `-dN + K >= 0`"
            )));
        };

        if let KExpr::Dim(i) = expr {
            if i >= num_dims {
                return Err(refuse(format!(
                    "affine set `{text}`: `d{i}` is out of range for {num_dims} dimension(s)"
                )));
            }
        }
        out.push(Constraint { expr, kind: ConstraintKind::GreaterEq });
    }
    Ok(KSet { num_dims, num_syms: 0, constraints: a.constraints(out) })
}

//===----------------------------------------------------------------------===//
// Ops and attributes
//===----------------------------------------------------------------------===//

/// What to do with one of our attribute keys at the boundary.
enum Key {
    /// Carry it, under their key.
    As(KAttrKey),
    /// Drop it: it is ours, and their consumer has no use for it.
    Drop,
    /// It belongs on the function, not on an operation.
    OnFunction,
}

/// Our attribute key, as theirs.
///
/// THE RENAMES ARE THE INTERESTING PART and each one is a real difference in their
/// vocabulary, not a spelling preference:
/// * `access_tile_order` / `access_tile_set` -> `CoordinateOrder` / `CoordinateSet`.
///   Their type has no access-tile-specific keys; the coordinate pair is what an access
///   tile's order and range are called.
/// * `order` (from `tt.trans`) -> `Permutation`.
/// * `axis` -> `Dimensions`, which is what `linalg.reduce` reads.
///
/// AND THE DROPS ARE DELIBERATE. Everything the corelet plan carries is ours: the plan is
/// discarded before SuperDSC on every path, so `pattern`, `work_division`, `index`,
/// `data_bounds`, `output_partition`, `pt_rows`, `xrf_capacity`, `role` and the ring pair
/// have no consumer. So is `spyre.canonical_verified`, which is DotToLinalg's own trust
/// tag. Dropping them is stated here so it is a decision on the record rather than an
/// omission.
fn attrkey(k: &AttrKey) -> Result<Key> {
    use AttrKey::*;
    Ok(match k {
        Value => Key::As(KAttrKey::Value),
        Dimensions | Axis => Key::As(KAttrKey::Dimensions),
        Order => Key::As(KAttrKey::Permutation),
        IndexingMaps => Key::As(KAttrKey::IndexingMaps),
        Shape => Key::As(KAttrKey::Shape),
        Strides => Key::As(KAttrKey::Strides),
        CoordinateSet | AccessTileSet => Key::As(KAttrKey::CoordinateSet),
        AccessTileOrder => Key::As(KAttrKey::CoordinateOrder),
        BaseMap => Key::As(KAttrKey::BaseMap),
        MemorySpace => Key::As(KAttrKey::MemorySpace),
        Predicate => Key::As(KAttrKey::Predicate),

        // The function's, not an operation's.
        SymName | Grid | Noinline | FoldedGridLoop => Key::OnFunction,

        // Ours. See the note above.
        Pattern | WorkDivision | Index | DataBounds | OutputPartition | PtRows
        | XrfCapacity | Role | RingSend | RingRecv | CanonicalVerified => Key::Drop,

        // `linalg.generic`'s iterator list. Their walker reads a reduction's combiner from
        // `ReduceFn` on a `linalg.reduce` instead, and there is no `IteratorTypes` in their
        // vocabulary at all, so this cannot be carried across as-is.
        IteratorTypes => {
            return Err(refuse(
                "`iterator_types` has no counterpart in their `AttrKey`: a reduction is \
                 expressed as `linalg.reduce` carrying `ReduceFn`, not as a \
                 `linalg.generic` carrying an iterator list. This is a shape decision for \
                 to_ktir, not something to rename here",
            ))
        }
        // `tensor.expand_shape`/`collapse_shape`'s reassociation. Their `AttrKey` has no
        // `Reassociation`, and MLIR's form -- a list of index LISTS, `[[0], [1, 2]]` -- has
        // no representation in their `Attr` either: there is `IntList` and no nested list.
        //
        // SO IT IS DROPPED AND `TargetShape` CARRIES THE SAME FACT. See `target_shape_of`
        // in `lower_op`: the result type's dims fully determine the reassociation for every
        // reshape these passes emit, all of which come from `tt.expand_dims` inserting a
        // size-1 axis. This is a CHOICE, not a match to their convention -- their producer
        // emits no `tensor.expand_shape` at all, so there was no convention to match, only
        // the constraint that the fact survive in a form their type can hold.
        Reassociation => Key::Drop,

        Other(s) => match s.as_str() {
            // ⛔⛔ THE INDIRECT ACCESS TILE'S SUBSCRIPTS ARE **NOT** RENAMED HERE, AND THE
            // ATTEMPT TO RENAME THEM WAS MEASURED WRONG. These two keys used to map onto
            // `dim_kinds` / `indexing_maps` on the argument that "these keys are chosen from
            // what their vocabulary offers rather than matched to a convention" -- and the
            // executor reads `dim_kinds` as the STRINGS `direct`/`direct_sub`/`direct_expr`/
            // `indirect` where ours is the C++ boolean list `["true", "false"]`, reads the
            // maps from nowhere near `indexing_maps`, and takes the subscript's domain to be
            // the ENUMERATION POINT where the C++ puts the captured scalars first. Three
            // disagreements, none of which a rename can reach.
            //
            // So they are DROPPED here and `indirect_access_tile` states the same facts in the
            // consumer's own vocabulary, from these same attributes. Dropped and not refused
            // because that arm is the one writer: a key that both arrived generically and was
            // rewritten specifically would be two answers for one question.
            "per_dim_subscript_kinds" | "per_dim_subscript_maps" => Key::Drop,

            // EVERY OTHER KEY THEIR VOCABULARY DECLARES, BY ITS OWN SPELLING.
            //
            // Adding these one refusal at a time was the wrong shape of work: an attribute
            // we carry as `Other` whose spelling their `AttrKey` already declares is not a
            // gap in either direction, it is the same key. So the whole set is listed and a
            // spelling match carries it across. Anything genuinely absent still refuses.
            "base_map" => Key::As(KAttrKey::BaseMap),
            "consumer_tiles_per_group" => Key::As(KAttrKey::ConsumerTilesPerGroup),
            "coordinate_order" => Key::As(KAttrKey::CoordinateOrder),
            "coordinate_set" => Key::As(KAttrKey::CoordinateSet),
            "dense_list" => Key::As(KAttrKey::DenseList),
            "dim" => Key::As(KAttrKey::Dim),
            "dim_data" => Key::As(KAttrKey::DimData),
            "dim_kinds" => Key::As(KAttrKey::DimKinds),
            "dim_map_0" => Key::As(KAttrKey::DimMap0),
            "dimensions" => Key::As(KAttrKey::Dimensions),
            "dtype" => Key::As(KAttrKey::Dtype),
            "groups" => Key::As(KAttrKey::Groups),
            "indexing_maps" => Key::As(KAttrKey::IndexingMaps),
            "intermediate_vars" => Key::As(KAttrKey::IntermediateVars),
            "is_tensor" => Key::As(KAttrKey::IsTensor),
            "lx_core_id" => Key::As(KAttrKey::LxCoreId),
            "memory_space" => Key::As(KAttrKey::MemorySpace),
            "n_ins" => Key::As(KAttrKey::NIns),
            "names" => Key::As(KAttrKey::Names),
            "outs_var" => Key::As(KAttrKey::OutsVar),
            "permutation" => Key::As(KAttrKey::Permutation),
            "producer_tiles_per_group" => Key::As(KAttrKey::ProducerTilesPerGroup),
            "shape" => Key::As(KAttrKey::Shape),
            "sizes_dyn" => Key::As(KAttrKey::SizesDyn),
            "slice_offsets" => Key::As(KAttrKey::SliceOffsets),
            "slice_sizes" => Key::As(KAttrKey::SliceSizes),
            "slice_strides" => Key::As(KAttrKey::SliceStrides),
            "strides" => Key::As(KAttrKey::Strides),
            "target_shape" => Key::As(KAttrKey::TargetShape),
            "variables_space_order" => Key::As(KAttrKey::VariablesSpaceOrder),
            "variables_space_set" => Key::As(KAttrKey::VariablesSpaceSet),

            _ => {
                return Err(refuse(format!(
                    "attribute `{s}` has no key in their vocabulary. Refused rather than \
                     dropped: an attribute that silently vanishes is how a tile loses a bound"
                )))
            }
        },
    })
}

fn attr<'a>(v: &Attr, a: &'a Arena) -> Result<KAttr<'a>> {
    Ok(match v {
        Attr::Int(i) => KAttr::Int(*i),
        Attr::IntList(l) => KAttr::IntList(a.ints(l.clone())),
        Attr::Float(f) => KAttr::Float(f.as_f64()),
        Attr::Str(s) => KAttr::Str(a.str(s.clone())),
        Attr::StrList(l) => {
            let refs: Vec<&str> = l.iter().map(|s| a.str(s.clone())).collect();
            KAttr::StrList(a.names(refs))
        }
        Attr::Bool(b) => KAttr::Bool(*b),
        Attr::SplatFloat(f) => KAttr::Float(f.as_f64()),
        Attr::AffineMap(t) => KAttr::AffineMap(affine_map(t, a)?),
        Attr::AffineMapList(l) => {
            let maps: Vec<KMap<'a>> =
                l.iter().map(|t| affine_map(t, a)).collect::<Result<_>>()?;
            KAttr::AffineMapList(a.maps(maps))
        }
        Attr::AffineSet(t) => KAttr::AffineSet(affine_set(t, a)?),
        Attr::Unit => {
            return Err(refuse(
                "a unit attribute has no counterpart: their `Attr` has no unit variant. \
                 Every unit attribute we attach is one of the function-level flags, which \
                 are handled as `Key::OnFunction`",
            ))
        }
        Attr::Verbatim(s) => {
            return Err(refuse(format!(
                "attribute value `{s}` was kept as text and cannot be expressed in their \
                 `Attr`. Refused rather than passed through"
            )))
        }
    })
}

/// Our op kind as theirs, or a refusal naming why not.
fn opkind(k: &OpKind) -> Result<KOpKind> {
    use OpKind as O;
    Ok(match k {
        O::FuncReturn => KOpKind::FuncReturn,
        O::ScfFor => KOpKind::ScfFor,
        O::ScfYield => KOpKind::ScfYield,
        O::ArithConstant => KOpKind::ArithConstant,
        O::ArithAddf => KOpKind::ArithAddf,
        O::ArithAddi => KOpKind::ArithAddi,
        O::ArithSubf => KOpKind::ArithSubf,
        O::ArithMulf => KOpKind::ArithMulf,
        O::ArithMuli => KOpKind::ArithMuli,
        O::ArithDivf => KOpKind::ArithDivf,
        O::ArithDivsi => KOpKind::ArithDivsi,
        O::ArithDivui => KOpKind::ArithDivui,
        O::ArithRemsi => KOpKind::ArithRemsi,
        O::ArithRemui => KOpKind::ArithRemui,
        O::ArithMaxnumf => KOpKind::ArithMaxnumf,
        O::ArithMinnumf => KOpKind::ArithMinnumf,
        O::ArithCmpi => KOpKind::ArithCmpi,
        O::ArithAndi => KOpKind::ArithAndi,
        O::ArithExtf => KOpKind::ArithExtf,
        O::ArithTruncf => KOpKind::ArithTruncf,
        O::ArithIndexCast => KOpKind::ArithIndexCast,
        O::ArithSelect => KOpKind::ArithSelect,
        O::MathExp => KOpKind::MathExp,
        O::MathLog2 => KOpKind::MathLog2,
        O::MathSqrt => KOpKind::MathSqrt,
        O::TensorSplat => KOpKind::TensorSplat,
        O::TensorEmpty => KOpKind::TensorEmpty,
        O::TensorCollapseShape => KOpKind::TensorCollapseShape,
        O::TensorExpandShape => KOpKind::TensorExpandShape,
        O::LinalgMatmul => KOpKind::LinalgMatmul,
        O::LinalgGeneric => KOpKind::LinalgGeneric,
        O::LinalgReduce => KOpKind::LinalgReduce,
        O::LinalgYield => KOpKind::LinalgYield,
        O::KtdpGetComputeTileId => KOpKind::KtdpGetComputeTileId,
        O::KtdpConstructMemoryView => KOpKind::KtdpConstructMemoryView,
        O::KtdpConstructAccessTile => KOpKind::KtdpConstructAccessTile,
        O::KtdpConstructIndirectAccessTile => KOpKind::KtdpConstructIndirectAccessTile,
        O::KtdpLoad => KOpKind::KtdpLoad,
        O::KtdpStore => KOpKind::KtdpStore,

        // `math.exp2` is decomposed, not mapped -- see `lower_op`. It should never reach
        // here, and saying so by name beats a wrong mapping.
        O::MathExp2 => {
            return Err(refuse(
                "`math.exp2` reached opkind(): their `OpKind` has MathExp and MathLog2 but \
                 no MathExp2, so it must be decomposed to exp(x * ln2) before this point. \
                 It is NOT mapped onto their catch-all `OpKind::Math`: riding a generic \
                 fallback is how a consumer receives a well-formed op it cannot act on",
            ))
        }
        O::Module | O::FuncFunc | O::TtFunc => {
            return Err(refuse(format!(
                "`{}` is a container, not an operation: it becomes their `IRFunction`. \
                 Reaching here means the walk did not treat it as the function it is",
                k.spelling()
            )))
        }
        O::KtdfCoreletPlan | O::KtdfCorelet => {
            return Err(refuse(format!(
                "`{}` should have been DROPPED, not mapped: the corelet plan is discarded \
                 before SuperDSC. Reaching here is a bug in `lower_block`",
                k.spelling()
            )))
        }
        O::UnrealizedConversionCast => {
            return Err(refuse(
                "`builtin.unrealized_conversion_cast` survived to the handoff: it is a \
                 type-conversion scaffold and their type has no counterpart",
            ))
        }
        // OUR TYPE'S ESCAPE HATCH IS NOT THEIR GAP.
        //
        // Our `OpKind` models 64 spellings; theirs models 123. So an op we carry as
        // `Other("arith.negf")` is one THEIR TYPE NAMES PROPERLY, and reporting it as
        // "no variant in their OpKind" -- which the first version of this arm did -- blames
        // the consumer for our own escape hatch. The right answer is to name their variant.
        //
        // The list is the spellings these passes actually emit as `Other`, and it stays
        // short because step 2 of the retype collapses our interior onto a generic
        // name-based IR, after which this whole arm becomes the only op lookup there is.
        O::Other(s) => match s.as_str() {
            "arith.negf" => KOpKind::ArithNegf,
            "arith.cmpf" => KOpKind::ArithCmpf,
            "arith.subi" => KOpKind::ArithSubi,
            "arith.extsi" => KOpKind::ArithExtsi,
            "arith.trunci" => KOpKind::ArithTrunci,
            "arith.ori" => KOpKind::ArithOri,
            "arith.xori" => KOpKind::ArithXori,
            "math.rsqrt" => KOpKind::MathRsqrt,
            "math.tanh" => KOpKind::MathTanh,
            "math.floor" => KOpKind::MathFloor,
            "math.absf" => KOpKind::MathAbsf,
            "math.log" => KOpKind::MathLog,
            "linalg.transpose" => KOpKind::LinalgTranspose,
            "linalg.broadcast" => KOpKind::LinalgBroadcast,
            "linalg.batch_matmul" => KOpKind::LinalgBatchMatmul,
            "linalg.fill" => KOpKind::LinalgFill,
            "tensor.extract" => KOpKind::TensorExtract,
            "tensor.extract_slice" => KOpKind::TensorExtractSlice,
            "tensor.insert_slice" => KOpKind::TensorInsertSlice,
            "tensor.from_elements" => KOpKind::TensorFromElements,
            other => {
                return Err(refuse(format!(
                    "`{other}` is carried as our `OpKind::Other` and has no arm here. If \
                     their type names it, add the arm; if it is a `tt.*` op then to_ktir \
                     left it behind, which is the one thing that pass exists to prevent"
                )))
            }
        },
        other => {
            return Err(refuse(format!(
                "`{}` has no variant in their `OpKind`",
                other.spelling()
            )))
        }
    })
}

//===----------------------------------------------------------------------===//
// The walk
//===----------------------------------------------------------------------===//

/// Mints values that do not collide with anything the module already named.
struct Fresh(u32);

impl Fresh {
    fn next(&mut self) -> KSsa {
        let v = KSsa(self.0);
        self.0 += 1;
        v
    }
}

/// One of our ops as one or more of theirs.
///
/// The list is a list because `math.exp2` becomes four ops: their vocabulary has `exp`
/// and no `exp2`, so the base change is explicit in the IR rather than assumed by the
/// consumer.
/// One of our ops as theirs, with the facts their consumer reads off an ATTRIBUTE derived from the
/// result type -- see [`state_shape_and_dtype`], which every returned operation goes through.
fn lower_op<'a>(
    op: &Op,
    a: &'a Arena,
    fresh: &mut Fresh,
) -> Result<Vec<Operation<'a>>> {
    let built = lower_op_kind(op, a, fresh)?;
    built.into_iter().map(|o| state_shape_and_dtype(o, a)).collect()
}

/// ⛔⛔⛔ THE ELEMENT DTYPE AND THE SHAPE ARE **ATTRIBUTES** IN THEIR CONVENTION, NOT ONLY RESULT
/// TYPES, AND OMITTING THEM WAS A SILENT WRONG ANSWER THAT SURVIVED SEVEN GREEN BAKES.
///
/// The C++ KTIR text carries the element dtype only in the result type --
/// `ktdp.construct_memory_view ... : memref<64x4096xf16>` -- and an access tile's window extents
/// only in ITS result type (`-> !ktdp.access_tile<64x4096xindex>`). So a port that emits result
/// types looks complete and is not. THEIR OWN PRODUCER sets both forms:
/// `scratchy:crates/targets/spyre/src/lower_subtile_tape_to_ktir.rs:2304-2310` builds a
/// `KtdpConstructMemoryView` with `AttrKey::Dtype` AND an `IrType::MemRef`, and does the same for
/// `TensorSplat` (:2585) and `TensorEmpty` (:2724). The redundancy IS the convention.
///
/// WHY NOTHING CAUGHT IT: `ktir-superdsc`'s `lower_ktir_to_superdsc` never reads `AttrKey::Dtype`
/// as a required attribute -- its one mention is inside an attribute-equality `matches!` -- so the
/// whole dxp path is blind to the omission, and `dxp_standalone` compiles and executes no
/// arithmetic. THE KTIR EXECUTOR IS THE FIRST CONSUMER THAT READS THEM, three ways:
///
///   * `ktir-emulator:src/dialects/ktdp.rs:92` -- `construct_memory_view` REFUSES without the
///     dtype, because `byte_addr = base_ptr * bpe` and the view's footprint check are both
///     denominated in it.
///   * `ktir-emulator:src/dialects/ktdp.rs:188` -- `construct_access_tile` REFUSES without `Shape`.
///   * ⚠️ `ktir-emulator:src/dialects/arith.rs:554` -- a TENSOR `arith.constant` with no dtype
///     attribute is read as `DType::F16`. THAT is the dangerous one: it does not refuse, it
///     guesses, and for an all-f16 Granite kernel it guesses right for the wrong reason. An f32
///     constant tensor would have been read as f16 and no stage would have said so.
///
/// # ⭐ WHY THIS IS A POST-PASS OVER EVERY BUILT OP AND NOT A BLOCK INSIDE THE BUILDER
///
/// It WAS a block inside the builder, and `tests/handoff_boundary.rs` caught that as wrong within a
/// minute: `lower_op_kind` returns EARLY for three rewrites -- `exp2_as_exp`, `generic_as_reduce`
/// and `generic_as_broadcast` -- each of which constructs its own `Operation`, so a `linalg.reduce`
/// producing `tensor<64xf16>` carried no dtype at all. One post-pass over everything the builder
/// returns has no such hole, and a fourth rewrite added later cannot reintroduce one.
///
/// # AND IT IS NOT A BLANKET STAMP
///
/// Derived from the RESULT TYPE, so it cannot disagree with the type the op actually produces;
/// applied only where the type HAS the fact (an `index` or an access tile has no element type, and
/// `IrType::elem` says so); and skipped where the op already states it, because two writers for one
/// key is worse than one. An attribute stamped on every op regardless is provenance that means
/// nothing -- this tree has shipped exactly that once already.
fn state_shape_and_dtype<'a>(mut o: Operation<'a>, a: &'a Arena) -> Result<Operation<'a>> {
    let Some(t) = o.result_type else { return Ok(o) };
    // A SHAPED result's element type. `IrType::elem` also answers for a bare `Scalar`, which is
    // deliberately NOT stamped: their producer sets the attribute on tiles and views, and the
    // executor reads it only for those.
    if t.dims().is_some() && o.attr(KAttrKey::Dtype).is_none() {
        if let Some(elem) = t.elem() {
            o = o.with_attr(a, KAttrKey::Dtype, KAttr::Dtype(elem));
        }
    }
    // The shape. `sizes_dyn` is the runtime-extent form of the same fact, so an op carrying that
    // states it already and must not be given a second, constant answer.
    if o.attr(KAttrKey::Shape).is_none() && o.attr(KAttrKey::SizesDyn).is_none() {
        if let Some(dims) = t.dims() {
            o = o.with_attr(a, KAttrKey::Shape, KAttr::IntList(a.ints(dims.to_vec())));
        }
    }
    Ok(o)
}

fn lower_op_kind<'a>(
    op: &Op,
    a: &'a Arena,
    fresh: &mut Fresh,
) -> Result<Vec<Operation<'a>>> {

    // MULTI-RESULT OPS. Their `Operation.result` is a single `Option<Ssa>`, but the type
    // is not therefore single-result: `AttrKey::ResultNames` and `AttrKey::NumResults`
    // exist for exactly this, and an `scf.for` has one result per loop-carried value (five
    // in attention's body). My first reading of this was wrong in a way worth recording:
    // I grepped for literal `results: vec![..]` constructions, found none with more than
    // one element, and concluded multi-result was unreachable -- the grep could not see an
    // op whose results are pushed rather than built inline.
    //
    // INFERRED FROM THE TYPE, NOT CONFIRMED BY THEIR PRODUCER: two keys named
    // `result_names` and `num_results`, in a type whose `result` field is a single option,
    // admit only this reading. If their producer spells it differently the fix is here.

    // `math.exp2 x` -> `exp(x * ln2)`.
    if op.kind == OpKind::MathExp2 {
        return exp2_as_exp(op, a, fresh);
    }

    // A reduction-shaped `linalg.generic` -> their `linalg.reduce`.
    if op.kind == OpKind::LinalgGeneric && reduced_axes(op).is_some() {
        return generic_as_reduce(op, a);
    }
    // An all-parallel `linalg.generic` with a yield-only body -> their `linalg.broadcast`.
    if op.kind == OpKind::LinalgGeneric && is_yield_only_broadcast(op) {
        return generic_as_broadcast(op, a);
    }
    // The gather. Its operand layout and its subscript vocabulary are BOTH different on the
    // two sides, so it is translated rather than renamed. See `indirect_access_tile`.
    if op.kind == OpKind::KtdpConstructIndirectAccessTile {
        return indirect_access_tile(op, a).map(|o| vec![o]);
    }

    let mut built = Operation::new(
        a,
        op.results.first().copied().map(ssa),
        opkind(&op.kind)?,
        &op.operands.iter().copied().map(ssa).collect::<Vec<_>>(),
    );
    if let Some(t) = op.result_types.first() {
        built.result_type = Some(irtype(t, a)?);
    }
    // A reshape's target shape, standing in for the reassociation their `Attr` cannot nest.
    // Emitted from the RESULT TYPE rather than from our attribute, so it cannot disagree
    // with the type the op actually produces.
    if matches!(op.kind, OpKind::TensorExpandShape | OpKind::TensorCollapseShape) {
        let dims = op
            .result_types
            .first()
            .and_then(|t| t.dims())
            .ok_or_else(|| {
                refuse(format!(
                    "`{}` has no shaped result type, so its target shape cannot be stated \
                     and their type has no other way to carry the reassociation",
                    op.kind.spelling()
                ))
            })?;
        built = built.with_attr(a, KAttrKey::TargetShape, KAttr::IntList(a.ints(dims.to_vec())));
    }

    if op.results.len() > 1 {
        let names: Vec<KSsa> = op.results.iter().copied().map(ssa).collect();
        built = built
            .with_attr(a, KAttrKey::NumResults, KAttr::Int(op.results.len() as i64))
            .with_attr(a, KAttrKey::ResultNames, KAttr::Ssas(a.ssa(names)));
    }
    for (k, v) in &op.attrs {
        match attrkey(k)? {
            Key::As(kk) => built = built.with_attr(a, kk, attr(v, a)?),
            Key::Drop | Key::OnFunction => {}
        }
    }
    built = lift_region_args(op, built, a)?;
    built.regions = lower_regions(&op.regions, a, fresh)?;
    Ok(vec![built])
}

/// `scf.for`'s induction variable and carried values are REGION ARGUMENTS in our IR and
/// ATTRIBUTES in theirs (`ir.rs:173`: a region is a bare op list, with no block
/// arguments). This lifts them, which is the whole of that structural difference.
fn lift_region_args<'a>(
    op: &Op,
    mut built: Operation<'a>,
    a: &'a Arena,
) -> Result<Operation<'a>> {
    let Some(region) = op.regions.first() else { return Ok(built) };
    if region.args.is_empty() {
        return Ok(built);
    }
    if op.kind != OpKind::ScfFor {
        // A `linalg.generic`/`linalg.reduce` body's arguments are positional and their
        // walker reads them from the op's operands, so lifting them would invent a
        // declaration their type does not have.
        return Ok(built);
    }
    let (iv, carried) = region.args.split_first().expect("non-empty");
    built = built.with_attr(a, KAttrKey::IterVar, KAttr::Ssas(a.ssa(vec![ssa(iv.0)])));
    if !carried.is_empty() {
        let args: Vec<KSsa> = carried.iter().map(|(v, _)| ssa(*v)).collect();
        built = built.with_attr(a, KAttrKey::IterArgs, KAttr::Ssas(a.ssa(args)));
    }
    Ok(built)
}

fn lower_regions<'a>(
    regions: &[Region],
    a: &'a Arena,
    fresh: &mut Fresh,
) -> Result<&'a [&'a [Operation<'a>]]> {
    let mut out: Vec<&'a [Operation<'a>]> = Vec::new();
    for r in regions {
        out.push(lower_block(&r.ops, a, fresh)?);
    }
    Ok(a.regions(out))
}

/// A flat op list, with the corelet plan dropped.
fn lower_block<'a>(
    ops: &[Op],
    a: &'a Arena,
    fresh: &mut Fresh,
) -> Result<&'a [Operation<'a>]> {
    let mut out: Vec<Operation<'a>> = Vec::new();
    for op in ops {
        if matches!(op.kind, OpKind::KtdfCoreletPlan | OpKind::KtdfCorelet) {
            continue;
        }
        out.extend(lower_op(op, a, fresh)?);
    }
    Ok(a.ops(out))
}

/// The axes a `linalg.generic`'s `iterator_types` marks `reduction`, or `None` if it marks
/// none -- which is what distinguishes a reduction from a broadcast or an elementwise map,
/// both of which are all-`parallel`.
fn reduced_axes(op: &Op) -> Option<Vec<i64>> {
    let Some(Attr::StrList(iters)) = op.attr(&AttrKey::IteratorTypes) else { return None };
    let axes: Vec<i64> = iters
        .iter()
        .enumerate()
        .filter(|(_, s)| *s == "reduction")
        .map(|(i, _)| i as i64)
        .collect();
    (!axes.is_empty()).then_some(axes)
}

/// An all-`parallel` `linalg.generic` whose body only yields its input.
///
/// This is what `to_ktir` produces for `tt.broadcast`, and attention has three of them.
fn is_yield_only_broadcast(op: &Op) -> bool {
    if reduced_axes(op).is_some() {
        return false;
    }
    op.attr(&AttrKey::IteratorTypes).is_some()
        && op
            .regions
            .first()
            .map(|r| r.ops.iter().all(|o| o.kind == OpKind::LinalgYield))
            .unwrap_or(false)
}

/// A yield-only `linalg.generic` as their `linalg.broadcast`.
///
/// # THIS IS THE V1/V2 INVERSION, AND HERE IT IS FORCED RATHER THAN CHOSEN
///
/// `to_ktir`'s header records that `tt.broadcast` must become a yield-only
/// `linalg.generic` and not the named `linalg.broadcast`, because V1's
/// `KTIRLegalityCheckPass` accepts the generic and rejects the named op (probe p07). That
/// was true of V1 and this path does not go through V1.
///
/// What settles it is not preference but expressibility: their `AttrKey` has no
/// `IteratorTypes` at all, so an all-parallel generic cannot state what it is. Their type
/// names `LinalgBroadcast` and their own producer uses it exactly where a reduced value has
/// to reach the lanes again. So the named op is the only form that survives the boundary.
///
/// `dimensions` is the set of RESULT axes the input does not index, which is MLIR's own
/// definition for `linalg.broadcast` -- read off the input's indexing map rather than
/// inferred from shapes, because two axes of equal extent would make a shape comparison
/// ambiguous and this is not.
fn generic_as_broadcast<'a>(op: &Op, a: &'a Arena) -> Result<Vec<Operation<'a>>> {
    let result_rank = op
        .result_types
        .first()
        .map(|t| t.rank())
        .filter(|r| *r > 0)
        .ok_or_else(|| refuse("a broadcast linalg.generic with no shaped result type"))?;

    let Some(Attr::AffineMapList(maps)) = op.attr(&AttrKey::IndexingMaps) else {
        return Err(refuse(
            "a broadcast linalg.generic with no `indexing_maps`: the broadcast axes are \
             read off the input's map, and there is nothing else to read them from",
        ));
    };
    let input_map = maps.first().ok_or_else(|| {
        refuse("a broadcast linalg.generic whose `indexing_maps` list is empty")
    })?;
    let parsed = affine_map(input_map, a)?;

    let mut used = vec![false; result_rank];
    for e in parsed.exprs {
        if let KExpr::Dim(i) = e {
            if *i < result_rank {
                used[*i] = true;
            }
        }
    }
    let dims: Vec<i64> = (0..result_rank).filter(|i| !used[*i]).map(|i| i as i64).collect();
    if dims.is_empty() {
        return Err(refuse(format!(
            "a yield-only linalg.generic whose input map `{input_map}` already indexes every \
             one of {result_rank} result axes broadcasts nothing. That is a copy, not a \
             broadcast, and their type has no arm for it -- refused rather than emitted as a \
             broadcast over no axes"
        )));
    }

    let mut built = Operation::new(
        a,
        op.results.first().copied().map(ssa),
        KOpKind::LinalgBroadcast,
        &op.operands.iter().copied().map(ssa).collect::<Vec<_>>(),
    );
    if let Some(t) = op.result_types.first() {
        built.result_type = Some(irtype(t, a)?);
    }
    Ok(vec![built.with_attr(a, KAttrKey::Dimensions, KAttr::IntList(a.ints(dims)))])
}

/// ⭐ THE GATHER: `ktdp.construct_indirect_access_tile`, RESTATED IN THE CONSUMER'S VOCABULARY.
///
/// This is the one op in the tree whose two sides disagree about more than a spelling, and each
/// disagreement was measured against the executor rather than inferred:
///
/// | fact | our KTIR (the C++'s form, which `pure_rust_ktir.rs` diffs) | `ktir-emulator` |
/// |---|---|---|
/// | operands | `(%table, %anchors.., %c_y, %ids_view)` -- MLIR flattens the op's two operand groups | `operands[0]` = primary, `operands[1..]` = INDEX VIEWS ONLY (`ktdp_extra.rs`) |
/// | captures | one `index` block argument each, in the op's region | `intermediate_vars`, an `Attr::Ssas` read from the value table |
/// | kinds | `per_dim_subscript_kinds`, the C++ boolean list `["true", "false"]` | `dim_kinds`, the strings `direct`/`direct_sub`/`direct_expr`/`indirect` |
/// | subscripts | `per_dim_subscript_maps`, domain `(c_x0.., c_y, d_0..d_{R-1})` -- CAPTURES FIRST | `dim_subs`, domain `(d_0..d_{R-1})` with the captures as SYMBOLS |
///
/// ⛔ SO A RENAME CANNOT REACH IT, AND THAT WAS THE STANDING DIAGNOSIS UNTIL THE DOMAINS WERE READ.
/// `per_dim_subscript_kinds -> dim_kinds` was a NAME match with the wrong values behind it, and the
/// executor's refusal -- `index_view 0 is Index(0), expected MemRef` -- named only the first of the
/// four rows above. A translation that fixed the operand layout alone would have produced the
/// SILENTLY WRONG answer the test file records: with the captures dropped and no subscript
/// attribute, `idx_exprs` is empty, the index view is addressed by the bare enumeration point, and
/// grid item 0 agrees while items 1..3 each gather from row 0 -- 192 of 256 tokens reading the wrong
/// embedding. All four rows have to be crossed at once, which is what this does.
///
/// ⚖️ WHY HERE AND NOT IN `convert_ttir_to_ktdp`. Ours is the C++'s form because the 12-of-12 golden
/// diff is what proves that pass is a faithful port, and `text::parse` reads the golden off the
/// C++'s own printed output. Changing the pass would delete the evidence. So the pass keeps emitting
/// the C++'s form and the BOUNDARY translates -- the same division `generic_as_reduce` states.
///
/// # THE REBASE, WHICH IS THE WHOLE ARITHMETIC OF THIS FUNCTION
///
/// With `C` captures and a rank-`R` result, our map domain is `C + R` wide and dim `i` means:
///
/// ```text
///   i <  C   the i-th captured scalar   ->  Sym(i)         (value read from the value table)
///   i >= C   enumeration variable i-C   ->  Dim(i - C)
/// ```
///
/// so `embedding_granite`'s `(d0, d1, d2, d3) -> (d0 + d2)` becomes `(d0, d1)[s0, s1] -> (s0 + d0)`
/// -- `ids[program_id * BLOCK_M + row]`, which is the address the fixture writes.
fn indirect_access_tile<'a>(op: &Op, a: &'a Arena) -> Result<Operation<'a>> {
    // `(base, captures.., index_view)`: the printed operand order, which `convert_gathers`
    // documents and `triton-superdsc-lower` reads `operands.last()` on.
    if op.operands.len() < 3 {
        return Err(refuse(format!(
            "ktdp.construct_indirect_access_tile has {} operand(s); the form this crate emits is \
             `(base, captures.., index_view)` with at least one capture, so fewer than three \
             cannot be re-based onto their `(primary, index_views..)` layout",
            op.operands.len()
        )));
    }
    let base = op.operands[0];
    let index_view = *op.operands.last().expect("checked");
    let captures: Vec<Ssa> = op.operands[1..op.operands.len() - 1].to_vec();
    let n_cap = captures.len();

    let rank = op
        .result_type()
        .map(|t| t.rank())
        .ok_or_else(|| refuse("ktdp.construct_indirect_access_tile has no result type"))?;

    let kinds_attr = op
        .attr(&AttrKey::Other("per_dim_subscript_kinds".into()))
        .ok_or_else(|| refuse("the indirect access tile carries no `per_dim_subscript_kinds`"))?;
    let Attr::StrList(kinds) = kinds_attr else {
        return Err(refuse("`per_dim_subscript_kinds` is not a string list"));
    };
    let maps_attr = op
        .attr(&AttrKey::Other("per_dim_subscript_maps".into()))
        .ok_or_else(|| refuse("the indirect access tile carries no `per_dim_subscript_maps`"))?;
    let Attr::AffineMapList(maps) = maps_attr else {
        return Err(refuse("`per_dim_subscript_maps` is not an affine map list"));
    };
    if kinds.len() != rank || maps.len() != rank {
        return Err(refuse(format!(
            "the indirect access tile has {} kind(s) and {} subscript map(s) for a rank-{rank} \
             result. One of each PER OUTPUT DIMENSION is what both sides index by dimension, so a \
             mismatch is refused rather than zipped over the shorter",
            kinds.len(),
            maps.len()
        )));
    }

    // ⛔ ONE INDEX VIEW OPERAND, SO EXACTLY ONE INDIRECT DIM. Their `dim_data[d]` is an index INTO
    // `index_views`, and this form carries exactly one view; a second indirect dim would have to
    // name a view that is not an operand, and defaulting it to view 0 would gather the row index of
    // one axis for another.
    let n_indirect = kinds.iter().filter(|k| is_indirect(k)).count();
    if n_indirect != 1 {
        return Err(refuse(format!(
            "the indirect access tile has {n_indirect} indirect dim(s) and exactly one index-view \
             operand. `buildGatherSubscriptMaps` emits one indirect dim (base dim 0) per gather, \
             so any other count means the op and this translation disagree about the form"
        )));
    }

    let mut k_kinds: Vec<&str> = Vec::with_capacity(rank);
    let mut k_data: Vec<i64> = Vec::with_capacity(rank);
    let mut k_subs: Vec<KMap<'a>> = Vec::with_capacity(rank);
    for (d, (kind, map)) in kinds.iter().zip(maps).enumerate() {
        // ⭐ EVERY DIM CARRIES AN EXPLICIT SUBSCRIPT, INCLUDING THE DIRECT ONES. Their `direct` arm
        // reads `dim_data[d]` BOTH as an index into `intermediate_vars` (to fold an outer scalar)
        // and as the enumeration position (`pt[var_index]`), which conflates the two whenever a
        // direct dim carries an offset -- and every direct dim this pass emits does (`c_y + d_K`).
        // `direct_sub` states the sum instead, so the offset cannot be lost.
        k_kinds.push(if is_indirect(kind) { "indirect" } else { "direct_sub" });
        // The view index for the indirect dim; ignored for `direct_sub`, and 0 rather than absent
        // because `dim_data` is read as a list parallel to `dim_kinds`.
        k_data.push(0);
        k_subs.push(rebase_subscript(map, n_cap, rank, d, a)?);
    }

    let mut built = Operation::new(
        a,
        op.results.first().copied().map(ssa),
        KOpKind::KtdpConstructIndirectAccessTile,
        &[ssa(base), ssa(index_view)],
    );
    built.result_type = Some(irtype(op.result_type().expect("checked above"), a)?);
    let kind_refs: Vec<&str> = k_kinds.iter().map(|k| a.str(k.to_string())).collect();
    built = built
        .with_attr(a, KAttrKey::DimKinds, KAttr::StrList(a.names(kind_refs)))
        .with_attr(a, KAttrKey::DimData, KAttr::IntList(a.ints(k_data)))
        .with_attr(
            a,
            KAttrKey::IntermediateVars,
            KAttr::Ssas(a.ssa(captures.into_iter().map(ssa).collect())),
        )
        .with_attr(a, KAttrKey::DimSubs, KAttr::AffineMapList(a.maps(k_subs)));

    // Everything else the op states -- `variables_space_set`, `variables_space_order`, the shape --
    // crosses unchanged. The two subscript keys are `Key::Drop` precisely so this loop cannot also
    // write them.
    for (k, v) in &op.attrs {
        match attrkey(k)? {
            Key::As(kk) => built = built.with_attr(a, kk, attr(v, a)?),
            Key::Drop | Key::OnFunction => {}
        }
    }

    // ⛔ AND THE REGION DOES NOT TRAVEL. Its block arguments ARE the captures, which now cross as
    // `intermediate_vars`; carrying an empty region beside them would be a second declaration of
    // the same values with no operand to bind it to.
    Ok(built)
}

/// The C++ boolean spelling of "this dim is indirect" (`per_dim_subscript_kinds` is a list of
/// `"true"`/`"false"`, which is what MLIR prints for an `ArrayAttr` of `BoolAttr`).
fn is_indirect(kind: &str) -> bool {
    kind.trim() == "true"
}

/// One `per_dim_subscript_maps` entry, re-based from the captures-first domain onto the enumeration
/// point with the captures as symbols.
///
/// The result COUNT is preserved and is load-bearing: an indirect dim's map has one address
/// component per index-grid axis, which is the index view's RANK, and the executor dots those
/// against the view's strides. A direct dim's map has one.
fn rebase_subscript<'a>(
    text: &str,
    n_cap: usize,
    rank: usize,
    d: usize,
    a: &'a Arena,
) -> Result<KMap<'a>> {
    let body = text
        .trim()
        .strip_prefix("affine_map<")
        .and_then(|s| s.strip_suffix('>'))
        .unwrap_or(text.trim());
    let (dims, results) = body.split_once("->").ok_or_else(|| {
        refuse(format!("the indirect tile's subscript map for dim {d}, `{text}`, has no `->`"))
    })?;
    let num_dims = dims
        .trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .count();
    // ⛔ THE DOMAIN ARITY IS THE CHECK THAT CATCHES A FORM CHANGE. `buildGatherSubscriptMaps` puts
    // the `C` captures first and the `R` iteration variables after, so the domain is exactly
    // `C + R` wide. If it ever is not, every rebase below is off by the difference and the gather
    // still produces well-formed numbers -- so this refuses instead.
    if num_dims != n_cap + rank {
        return Err(refuse(format!(
            "the indirect tile's subscript map for dim {d}, `{text}`, has a {num_dims}-dimensional \
             domain; with {n_cap} capture(s) and a rank-{rank} result it must be {}. The rebase \
             onto (enumeration dims, captured symbols) depends on that split, and getting it wrong \
             computes a well-formed WRONG address",
            n_cap + rank
        )));
    }

    let mut exprs = Vec::new();
    for term in results.trim().trim_start_matches('(').trim_end_matches(')').split(',') {
        let term = term.trim();
        if term.is_empty() {
            continue;
        }
        let mut acc: Option<KExpr<'a>> = None;
        for atom in term.split('+') {
            let atom = atom.trim();
            let e = match atom.strip_prefix('d').and_then(|n| n.parse::<usize>().ok()) {
                Some(i) if i < n_cap => KExpr::Sym(i),
                Some(i) => KExpr::Dim(i - n_cap),
                None => KExpr::Const(atom.parse::<i64>().map_err(|_| {
                    refuse(format!(
                        "the indirect tile's subscript map for dim {d}, `{text}`: `{atom}` is \
                         neither a domain dimension nor an integer. These passes emit sums of \
                         `dN` and integers only, so a wider form is refused rather than half-read"
                    ))
                })?),
            };
            acc = Some(match acc {
                None => e,
                Some(l) => KExpr::Add(a.expr(l), a.expr(e)),
            });
        }
        exprs.push(acc.ok_or_else(|| {
            refuse(format!(
                "the indirect tile's subscript map for dim {d}, `{text}`, has an empty result term"
            ))
        })?);
    }
    if exprs.is_empty() {
        return Err(refuse(format!(
            "the indirect tile's subscript map for dim {d}, `{text}`, states no subscript at all"
        )));
    }
    Ok(KMap { num_dims: rank, num_syms: n_cap, exprs: a.exprs(exprs) })
}

/// A reduction-shaped `linalg.generic` as their `linalg.reduce`.
///
/// # WHY THE SHAPE CHANGES HERE AND NOT IN `to_ktir`
///
/// Their `AttrKey` has no `IteratorTypes` at all: a reduction is a `linalg.reduce` carrying
/// `ReduceFn`, and their `Attr::Op(OpKind)` exists precisely so the combiner is a VARIANT --
/// a combiner nothing implements cannot be spelled. `to_ktir` emits the generic because
/// V1's legality check demanded it (the same constraint as the broadcast note in that
/// pass), and the 12-of-12 golden diff is against that V1 output.
///
/// So the translation belongs at the boundary rather than in the pass: `to_ktir` keeps
/// producing the form the C++ produces, the golden diff keeps proving the port, and the
/// consumer gets the form its type actually names. A target-specific shape belongs where
/// the target is.
///
/// This is NOT a recognizer in the sense that word usually carries here. It reads an
/// iterator list and a combiner that THIS CRATE ITSELF WROTE, three passes earlier, in a
/// shape it fully controls -- not a kernel's structure inferred from arithmetic. If the
/// generic is not that shape, it refuses by name.
fn generic_as_reduce<'a>(op: &Op, a: &'a Arena) -> Result<Vec<Operation<'a>>> {
    let axes = reduced_axes(op).expect("caller checked");

    let region = op.regions.first().ok_or_else(|| {
        refuse("a reduction linalg.generic with no body cannot name its combiner")
    })?;
    let body: Vec<&Op> = region.ops.iter().filter(|o| o.kind != OpKind::LinalgYield).collect();
    if body.len() != 1 {
        return Err(refuse(format!(
            "a reduction linalg.generic's body must be exactly one combining op for \
             `reduce_fn` to name it; got {} ops. Their `Attr::Op` is a single variant, so a \
             multi-op combiner has no spelling and is refused rather than flattened",
            body.len()
        )));
    }
    let combiner = opkind(&body[0].kind)?;

    let result = op.results.first().copied().map(ssa);
    let mut built = Operation::new(
        a,
        result,
        KOpKind::LinalgReduce,
        &op.operands.iter().copied().map(ssa).collect::<Vec<_>>(),
    );
    if let Some(t) = op.result_types.first() {
        built.result_type = Some(irtype(t, a)?);
    }
    built = built
        .with_attr(a, KAttrKey::ReduceFn, KAttr::Op(combiner))
        .with_attr(a, KAttrKey::Dimensions, KAttr::IntList(a.ints(axes)));

    // The combiner is the attribute now, so the body does not travel: a region that
    // repeated it would be a second, divergeable statement of the same fact.
    Ok(vec![built])
}

/// `exp2(x)` as `exp(x * ln2)`.
///
/// Four ops, because the multiplier has to exist as a value of `x`'s shape: a scalar
/// `ln2` constant, a splat of it when `x` is a tensor, the multiply, then `exp`. Their
/// `TensorSplat` is the constant-splat form their own walker mints a caller-filled buffer
/// for, so this is the shape it expects rather than a novel one.
fn exp2_as_exp<'a>(
    op: &Op,
    a: &'a Arena,
    fresh: &mut Fresh,
) -> Result<Vec<Operation<'a>>> {
    let result = op
        .results
        .first()
        .copied()
        .ok_or_else(|| refuse("`math.exp2` with no result"))?;
    let ty = op
        .result_types
        .first()
        .ok_or_else(|| refuse("`math.exp2` with no result type"))?;
    let elem = match ty {
        IrType::Tensor { elem, .. } => *elem,
        IrType::Scalar(d) => *d,
        other => {
            return Err(refuse(format!(
                "`math.exp2` on an unexpected type; only a tensor or a scalar is \
                 decomposable here, got {other:?}"
            )))
        }
    };
    let x = *op
        .operands
        .first()
        .ok_or_else(|| refuse("`math.exp2` with no operand"))?;

    let mut out = Vec::new();

    // the scalar ln2
    let c = fresh.next();
    let mut konst = Operation::new(a, Some(c), KOpKind::ArithConstant, &[]);
    konst.result_type = Some(KIrType::Scalar(dtype(elem)?));
    out.push(konst.with_attr(a, KAttrKey::Value, KAttr::Float(LN2)));

    // splat it to x's shape, when x is shaped
    let multiplier = match ty {
        IrType::Tensor { .. } => {
            let s = fresh.next();
            let mut splat = Operation::new(a, Some(s), KOpKind::TensorSplat, &[c]);
            splat.result_type = Some(irtype(ty, a)?);
            out.push(splat);
            s
        }
        _ => c,
    };

    // x * ln2
    let scaled = fresh.next();
    let mut mul = Operation::new(a, Some(scaled), KOpKind::ArithMulf, &[ssa(x), multiplier]);
    mul.result_type = Some(irtype(ty, a)?);
    out.push(mul);

    // exp of that, taking the original result name so every use is unchanged
    let mut exp = Operation::new(a, Some(ssa(result)), KOpKind::MathExp, &[scaled]);
    exp.result_type = Some(irtype(ty, a)?);
    out.push(exp);

    Ok(out)
}

//===----------------------------------------------------------------------===//
// The one precondition their first reader has
//===----------------------------------------------------------------------===//

/// Does anything, anywhere in `ops`, address `ptr` with a `ktdp.construct_memory_view`?
fn views_ptr_somewhere(ops: &[Op], ptr: Ssa) -> bool {
    ops.iter().any(|o| {
        (o.kind == OpKind::KtdpConstructMemoryView && o.operands.first() == Some(&ptr))
            || o.regions.iter().any(|r| views_ptr_somewhere(&r.ops, ptr))
    })
}

/// FAIL CLOSED: EVERY PARAMETER STATES ITS BUFFER'S WIDTH, AT THE TOP LEVEL.
///
/// This is the precondition of the FIRST reader past this boundary.
/// `ktir_superdsc::emit::lower_ktir_to_superdsc::regions` zips `IRFunction::arguments`
/// against the launch's bound buffers positionally and, for parameter `i`, takes the one
/// `ktdp.construct_memory_view` whose address operand is that parameter -- found by a
/// NON-RECURSIVE `find` over `IRFunction::operations`. Everything it goes on to compute
/// (the extent, the strides, the fp8 flag, whether the parameter is the output) comes off
/// that view.
///
/// WHY IT IS CHECKED HERE RATHER THAN LEFT TO THEM. Their refusal names `parameter 0 (t0)`
/// and nothing else, which is true and unactionable: it cannot say whether a pass buried
/// the view or whether the program never had one, and those want opposite fixes. Both were
/// live at once when this was written -- `rope_q32` at grid 4,32 had all four views inside
/// the work loop, `attention_flash_noncausal` had no view for `desc_mask` anywhere -- and
/// the single message could not tell them apart. So the two are diagnosed separately and
/// the grid travels in both, because the first of them is grid-dependent and a report that
/// omits it sends the next reader to the wrong kernel.
///
/// WHAT THIS DELIBERATELY DOES NOT DO IS SYNTHESISE ONE. A `[rows, cols]` this crate
/// invented would let the build proceed and hand the device a well-formed descriptor over a
/// buffer of assumed extent -- the exact failure this tree keeps being bitten by. A
/// parameter whose width nothing states is a REFUSAL.
fn every_parameter_states_its_width(m: &Module, region: &Region, grid: usize) -> Result<()> {
    for (i, (ptr, _)) in region.args.iter().enumerate() {
        if region.ops.iter().any(|o| {
            o.kind == OpKind::KtdpConstructMemoryView && o.operands.first() == Some(ptr)
        }) {
            continue;
        }
        // `Module::hints` is diagnostic-only and no pass may branch on it; this is a
        // message, so the parameter's own spelling is exactly what belongs here.
        let name = m.hint(*ptr);
        if views_ptr_somewhere(&region.ops, *ptr) {
            return Err(refuse(format!(
                "parameter {i} (%{name}) states its `ktdp.construct_memory_view` only inside \
                 a nested region, and the consumer pairs a parameter to its extent by a \
                 non-recursive search of the function's TOP-LEVEL ops -- so at grid [{grid}] \
                 nothing it can see says how wide the buffer this parameter addresses is. A \
                 view is loop-invariant by construction (its one operand is the base address, \
                 its extent is an attribute), so this is a pass that left it nested, not a \
                 program that needs it there"
            )));
        }
        return Err(refuse(format!(
            "parameter {i} (%{name}) is addressed NOWHERE: no `ktdp.construct_memory_view` \
             anywhere in the body takes it, so at grid [{grid}] nothing states how wide the \
             buffer it addresses is, and this crate will not invent one. The parameter still \
             occupies binding slot {i} -- the consumer numbers its buffers by parameter \
             position -- so dropping it here would silently renumber every buffer after it. \
             Either the kernel should not take it, or the launch should not bind it"
        )));
    }
    Ok(())
}

//===----------------------------------------------------------------------===//
// Entry
//===----------------------------------------------------------------------===//

/// Our Triton-free module as their `IRFunction`.
///
/// Run [`crate::passes::to_ktir`] first: this refuses every `tt.*` op by name, because
/// their type cannot hold one and a module that looks converted but is not is precisely
/// what that pass exists to prevent.
pub fn lower<'a>(m: &Module, a: &'a Arena) -> Result<IRFunction<'a>> {
    let func = m.kernel()?;
    if func.kind != OpKind::FuncFunc {
        return Err(refuse(format!(
            "the kernel is `{}`, not `func.func`: run to_ktir first",
            func.kind.spelling()
        )));
    }

    let name = match func.attr(&AttrKey::SymName) {
        Some(Attr::Str(s)) => a.str(s.clone()),
        _ => return Err(refuse("the function has no `sym_name`")),
    };

    // Their grid is a 3-tuple. Ours is the 1-D extent `to_ktir` folded the work loop
    // into, so it goes in the fastest axis and the others are 1.
    let grid = match func.attr(&AttrKey::Grid) {
        Some(Attr::IntList(g)) if g.len() == 1 => (g[0] as usize, 1, 1),
        Some(Attr::IntList(g)) => {
            return Err(refuse(format!(
                "`grid` has {} extents; to_ktir emits a 1-D grid and their `IRFunction.grid` \
                 is a 3-tuple, so a multi-axis grid needs an explicit mapping rather than a \
                 guess",
                g.len()
            )))
        }
        _ => return Err(refuse("the function has no `grid` attribute")),
    };

    let region = func
        .regions
        .first()
        .ok_or_else(|| refuse("the function has no body"))?;

    let mut arguments = Vec::new();
    for (v, t) in &region.args {
        arguments.push((ssa(*v), irtype(t, a)?));
    }
    every_parameter_states_its_width(m, region, grid.0)?;

    let mut fresh = Fresh(m.next_ssa);
    let operations = lower_block(&region.ops, a, &mut fresh)?;

    Ok(IRFunction {
        name,
        arguments: a.args(arguments),
        operations,
        grid,
        return_type: None,
    })
}
