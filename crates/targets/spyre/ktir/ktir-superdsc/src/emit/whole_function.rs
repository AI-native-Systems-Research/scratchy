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

use ktir_core::attrkey::AttrKey;
use ktir_core::ir::{Attr, IRFunction, Operation, Ssa};
use ktir_core::opkind::OpKind;

use super::lower_ktir_to_superdsc::{Error, Gather, Region, err, gathers_of, regions};
use crate::ktir_node::{Elementwise, KtirNode, Program, ReduceKind};
// The broadcast AXIS is the emission door's own vocabulary — `elementwise` selects the `In` builder
// from it — so it lives beside that door and this recogniser names the same type rather than a
// parallel one that would have to be mapped across the boundary.
use super::lower_ktir_to_superdsc::BcastAxis;
use crate::placement::BundleLayout;
use crate::work::DeviceWidth;

/// THE SILU LONGHAND, as the ONE fused op it is — `out = silu(gate) · up`.
///
/// # WHY THIS RECOGNISER EXISTS AT ALL, AND WHY IT IS NOT AN EXCEPTION TO THE MODULE HEADER
///
/// This module's header says it recognises NOTHING, and every 1:1 arm in [`program_of`] keeps that
/// promise. This is the one shape where the 1:1 rule cannot be kept, and the reason is a DEVICE fact
/// rather than a convenience: the AIU has `OpFunc::Silu` as a primitive and KTIR's `math.*` set does
/// not, so a producer has no op to write it with and writes the longhand instead
/// ([`Program::SiluMul`]'s own doc says exactly this).
///
/// ⛔⛔⛔ AND LOWERING THE LONGHAND OP-BY-OP IS NOT A SLOWER ANSWER, IT IS A DIFFERENT ONE.
/// `test/fixtures/swiglu_mlp.py` delta 5 states it at the source: "Writing the activation any other
/// way still lowers, but as four plain SuperDSCs computing a DIFFERENT algorithm, to which the
/// derived tolerance would not apply." The device primitive is a sigmoid opaque over six constants;
/// five pointwise descriptors are not that function. So the choice is not "fuse or don't" — it is
/// "recognise it, or emit something whose numbers no tolerance covers". A green `dxp_standalone`
/// exit would not catch the difference, which is what makes this a fail-closed matter.
///
/// # IT PROVES THE MATCH; IT DOES NOT PATTERN-GUESS
///
/// Built on the shape [`super::lower_ktir_to_superdsc::program_rmsnorm_eps`] and
/// `program_scalarmul_scale` already use: walk for an exact op-kind chain, and require every link to
/// be UNIQUE. Two discriminators do the real work, and neither is structural bookkeeping:
///
/// * **the divide's numerator is the value the negate consumed.** `g / (1 + exp(-g))` is
///   `g · sigmoid(g)` = `silu(g)`; `x / (1 + exp(-g))` for any other `x` is a different function
///   that happens to have the same op kinds in the same order. This is what makes it silu.
/// * **the added constant is exactly 1.0.** `sigmoid` is `1/(1+e⁻ˣ)`. Any other constant is a
///   different function, so it is refused rather than rounded into one.
///
/// A chain that almost matches is REFUSED BY NAME, never lowered as the nearest thing.
#[derive(Clone, Copy, Debug)]
pub struct SiluMulChain {
    /// The value the negate consumed AND the divide's numerator — silu's argument.
    pub gate: Ssa,
    /// The other factor of the terminal multiply.
    pub up: Ssa,
    /// The terminal `arith.mulf`'s result: the value the ONE fused op produces, and the value the
    /// rest of the program consumes.
    pub mul: Ssa,
    /// `negf`, `exp`, `addf`, `divf` results — every intermediate the fused op never materialises,
    /// so [`lower_function`] must lower NOTHING for them. The terminal multiply is deliberately not
    /// in here: it is the op that BECOMES the fused program.
    pub consumed: [Ssa; 4],
}

/// WHICH WAY a `linalg.matmul` indexes its B operand — the orientation, READ FROM THE PROGRAM, because
/// the two forms are DIFFERENT BUFFERS and the extents cannot tell them apart when `k == n`.
///
/// # ⛔⛔⛔ THE HOLE THIS CLOSES IS A SILENT WRONG ANSWER, NOT A MISSING FEATURE
///
/// Nothing else in this crate reads `IndexingMaps`. The weight's framing was instead inferred from its
/// EXTENTS, by [`super::lower_ktir_to_superdsc::matmul`]'s two guards, which cannot tell the two apart
/// WHEN `k == n`: a `[k, n]` weight then satisfies both and lowers to a descriptor that contracts the
/// other way round. Granite's attention output projection is `4096 × 4096`, so that is a shape this
/// backend really compiles. This value is what those guards now frame themselves from.
///
/// # ⭐⭐⭐⭐⭐ AND THE KERNEL SLOT ITSELF IS `[k, n]` — THE PLAIN FORM NEEDS NO RELAYOUT
///
/// A first reading of this had the plain form as the exotic one, to be lowered by transposing B into
/// the "assumed" orientation with `OpFunc::Transpose`. That was wrong twice over, and both halves were
/// measured:
///
/// * **The device slot is in-rows.** [`crate::sdsc_abstract::StickLayout::kernel`] is `[in(K),
///   out(N)]` with `rows = k_in`, and `matmul` declares exactly that (`Stk::kernel(k, n_dev, ..)`) for
///   BOTH forms. `vcache_write_offset`'s doc states the same thing for the hardware-proven attention —
///   "the V cache … the VALUE bmm reads as a `[cap, hd]` KERNEL sticked on `hd` (`out`) … the kernel is
///   `[k=cap, n=hd]`" — and `attn.rs`'s value leg contracts the V cache WHERE IT LIES, no relayout, at
///   41 tok/s. `p @ V` IS this contraction. Its twin `kcache_kt_write_offset` is the other half of the
///   argument: the score leg's kernel is `[hd, cap]`, in-rows again, which is why the K cache is
///   written already-transposed rather than transposed on the way in.
/// * **The relayout could not have computed it anyway.** MEASURED on both decoders: the transposing
///   emission produced a `[128, 64]` intermediate and then declared the consuming kernel
///   `layoutDimOrder_ ["in","out"]` with `in_ = 64, out_ = 128` over it (baked `sdsc_32` / `sdsc_33`)
///   — a k-row/n-col reading of a buffer that physically had n rows. It moved the bytes AWAY from the
///   slot's own address law. dxp then refused to schedule it on `SENARCH=MPW4` ("Implicit syncs not
///   available for architectures prior to RCUDD1A"), which is the only reason the wrong answer was
///   never baked.
///
/// So a `[k, n]` B is contracted in place, and the transposition a `[n, k]` B needs is spent OUTSIDE
/// this crate: by the single host stage (`stage_2d(&StickLayout::kernel(k, n), ..)`) for a presented
/// weight, or by the device's own already-transposed cache WRITE for a computed one.
///
/// # ⛔ WHAT THE FRONT END DOES WITH IT, SO THE TWO FORMS ARE NOT INTERCHANGEABLE HERE
///
/// `triton-ktir`'s `dot_to_linalg` REWIRES a `tt.dot`'s B past its `tt.trans` and states
/// `[[0,2],[1,2],[0,1]]` — so a transpose-B node's region is the PRE-transpose buffer, `[n, k]`. A
/// `tl.dot` with no `.T` keeps its operand and states no maps at all, which IS MLIR's plain default:
/// region `[k, n]`. Both are honest statements about real buffers; what is not honest is reading one
/// framing's extents as the other's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BOrient {
    /// `[[d0,d2],[d1,d2],[d0,d1]]` — B's map ends in the reduction dim, so B's REGION is `[n, k]` and
    /// the contraction reduces over k in place. scratchy's `KtirFunc::matmul` emits this
    /// unconditionally for a presented weight ("W BINDS VERBATIM as its on-disk `[out, in]` = `[n, k]`
    /// buffer … so there is no transpose and no strided gather"), and the host stage is where its bytes
    /// are placed into the kernel slot's `[k, n]` device order.
    TransposeB,
    /// `[[d0,d2],[d2,d1],[d0,d1]]`, or NO maps at all (MLIR's default for `linalg.matmul`) — B's REGION
    /// is `[k, n]`, which is the kernel slot's OWN device order. A computed operand (`tl.dot(p, v)`,
    /// where `v` is this kernel's V projection) arrives this way and is contracted where it lies.
    PlainB,
}

/// PROVE which of the two forms a `linalg.matmul` states, from its own `indexing_maps`.
pub fn matmul_b_orientation(
    f: &IRFunction<'static>,
    op: &ktir_core::ir::Operation<'static>,
) -> Result<BOrient, Error> {
    let maps = op.attributes.iter().find_map(|(k, v)| match (k, v) {
        (AttrKey::IndexingMaps, Attr::AffineMapList(m)) => Some(*m),
        _ => None,
    });
    // `[[d0,d2],[d1,d2],[d0,d1]]` — A row-major, B TRANSPOSED (its map ends in the reduction dim
    // d2), C row-major.
    let is_transpose_b = maps.is_some_and(|m| {
        m.len() == 3
            && m.iter()
                .zip([[0usize, 2], [1, 2], [0, 1]])
                .all(|(got, want)| {
                    got.exprs.len() == 2
                        && got.exprs.iter().zip(want).all(
                            |(e, d)| matches!(e, ktir_core::affine::AffineExpr::Dim(i) if *i == d),
                        )
                })
    });
    if is_transpose_b {
        return Ok(BOrient::TransposeB);
    }
    // `[[d0,d2],[d2,d1],[d0,d1]]` — MLIR's PLAIN `linalg.matmul`: B indexed `[k, n]`, its map ending in
    // the NON-reduction dim. A `tt.dot` with no `.T` on its second operand lowers to this, and a
    // producer that omits `indexing_maps` entirely is stating the same default.
    let is_plain_b = maps.is_none_or(|m| {
        m.len() == 3
            && m.iter()
                .zip([[0usize, 2], [2, 1], [0, 1]])
                .all(|(got, want)| {
                    got.exprs.len() == 2
                        && got.exprs.iter().zip(want).all(
                            |(e, d)| matches!(e, ktir_core::affine::AffineExpr::Dim(i) if *i == d),
                        )
                })
    });
    if is_plain_b {
        return Ok(BOrient::PlainB);
    }
    err(format!(
        "{}: this `{:?}`'s `indexing_maps` are NEITHER of the two contraction forms this door \
         lowers. The assumed one is transpose-B, `[[d0,d2],[d1,d2],[d0,d1]]` — B's map ends in the \
         reduction dim, so B is its on-disk `[out, in]` = `[n, k]` buffer and the contraction reduces \
         over k in place; that is the contract scratchy's `KtirFunc::matmul` states at its own \
         definition and emits unconditionally. The other is MLIR's plain \
         `[[d0,d2],[d2,d1],[d0,d1]]` (B as `[k, n]`), which is lowered by TRANSPOSING B into the \
         assumed orientation first. Anything else — a permuted A or C, a batch dim, a broadcast map — \
         states an iteration order no assembler here walks, and the extent guards downstream cannot \
         catch it at all when k == n (Granite's `[4096, 4096]` output projection is exactly that \
         shape), so it is named rather than lowered.",
        f.name, op.op_type,
    ))
}

/// ⛔⛔⛔ A TRANSPOSE-B `B` HAS TO BE **PLACED** BY SOMEBODY, AND ONLY A PRESENTED BUFFER HAS ANYBODY
/// TO DO IT. This is the seal on the one silent wrong answer this door used to compile.
///
/// # THE ARGUMENT, from the crate's own declarations
///
/// Stating `indexing_maps` LABELS which axis of `B` is `k`. It does not MOVE a byte. The kernel slot
/// reads exactly ONE residency and it is `[k, n]`: [`super::lower_ktir_to_superdsc::matmul`] declares
/// the operand `StickLayout::kernel(k_in, n_out)`, whose address law is `dev_off_stk`'s rank-2
/// stick-blocked form with `dims[0] = in = K` as the ROW count, and `StickLayout::addr_eq` states the
/// consequence in the crate's own words — "`RowBlocked` vs `Kernel` is the identical stick-block
/// formula". So a `[n, k]` operand is bytes in the wrong order for that law, and the transposition is
/// spent OUTSIDE the descriptor by one of exactly two agents:
///
///  * a PRESENTED weight — the HOST stage, once, at bake time (`stage_2d(&StickLayout::kernel(k, n),
///    ..)`, `sdsc_abstract.rs:2124`). Free at inference, and the whole regression bar is this case.
///  * a CACHE — the DEVICE, writing it already transposed (`kcache_kt_write_offset`), which is why the
///    shipped attention pays for a third physical Kᵀ plane (`assemble_restickify_kt_2d`) rather than
///    relabelling the one it has. **The relayout exists to MAKE a `[k, n]` buffer, never to consume
///    one.**
///
/// An IN-REGISTER COMPUTED value is NEITHER. Nothing stages it, no cache write orders it, and nothing
/// in this crate reads an access tile's `CoordinateOrder` (it occurs inside one comment and nowhere
/// else), so the transposing order a front end can set on the tile is never honoured. The descriptor
/// that comes out is well-formed and contracts `A @ B` instead of `A @ Bᵀ`.
///
/// ⛔ AND NO EXTENT GUARD CAN CATCH IT. `matmul`'s `w_k != k` / `w_n != n` checks read whichever of
/// the region's two extents this orientation names, so at `k == n` both framings pass — which
/// `decoder_block.py`'s two score matmuls (`tl.dot(q1r, k1r.T)` on `[64, 64]` RoPE results) are
/// exactly. That is why the refusal is stated on PROVENANCE, which the extents cannot express.
///
/// # WHAT THE PRODUCER DOES INSTEAD, AND WHY THIS IS NOT A DEAD END
///
/// A front end holding a `tt.trans` of a computed value must emit a REAL relayout beside the matmul —
/// a `linalg.transpose` reaching `Program::Transpose` — and then state NO maps, so the plain `[k, n]`
/// form reads a buffer that is physically `[k, n]`. `triton-ktir`'s `dot_to_linalg` does exactly that
/// as of this change: it folds a `.T` into the maps ONLY when the transposed value is a direct
/// `tt.descriptor_load`.
///
/// ⛔ THAT COSTS AN ARCH FEATURE THIS PATH DOES NOT HAVE ON MPW4, AND THE REASON IS NOW MECHANICAL
/// RATHER THAN OBSERVED. deeptools maps an opFunc to a DDL template per ISA
/// (`ddc/ddl/ddl_conversion.h`'s `opFuncToDdlTemplate`): `BATCHMATMUL_FWD` has a
/// `{"bmm_dd1.ddl", MPW4_ISA}` row and `bmm_dd1.ddl` carries NO `ddl.implicit_sync`, which is why
/// every matmul in the passing suite schedules. Both relayouts map MPW4 to a template that DOES:
/// `ReStickifyOpHBM` → `{"restickify.ddl", MPW4_ISA}` (`ddl.implicit_sync` at `restickify.ddl:80`,
/// unconditional, inside `intraslice_loop`) and `INTERSLICETRANSPOSE_FP16` →
/// `inter_slice_transpose.ddl` (`:75`, likewise). `Ddc::finalizeOps` then refuses ANY implicit sync
/// below `RCUDD1A_ISA` (`ddcv1.cpp:3416`) and `MPW4_ISA < RCUDD1A_ISA` in `IsaCoreGen`. There is no
/// `restickify_dd1.ddl`. So the refusal is a property of the TEMPLATE — not of the shape, the core
/// division, the tile geometry, or which relayout primitive is picked — and swapping primitives
/// cannot move it. A dxp refusal on a program that computes the right thing is nevertheless the
/// outcome this guard exists to force: it is a build error, not a wrong number.
fn transposed_b_is_placeable(
    name: &str,
    b: BOrient,
    b_is_parameter_backed: bool,
) -> Result<(), Error> {
    if b == BOrient::PlainB || b_is_parameter_backed {
        return Ok(());
    }
    err(format!(
        "{name}: this `linalg.matmul` states the TRANSPOSE-B form \
         (`[[d0,d2],[d1,d2],[d0,d1]]`, B's map ending in the reduction dim, so B's region is \
         `[n, k]`) over a B that is a COMPUTED value — an intermediate this function produced, \
         not a load of a parameter. The maps LABEL which axis is k; they do not PLACE it, and \
         the kernel slot reads `[k, n]` (`StickLayout::kernel(k_in, n_out)`) whatever they say. \
         A presented weight spends that transposition in the host stage \
         (`stage_2d(&StickLayout::kernel(k, n), ..)`) and a cache in the device's \
         already-transposed write (`kcache_kt_write_offset`); an in-register value has neither, \
         and nothing here reads an access tile's `CoordinateOrder`. Lowering it would emit a \
         well-formed descriptor computing `A @ B` instead of `A @ Bᵀ`, which NO extent guard can \
         catch when k == n. So the producer must emit a real relayout beside this matmul (a \
         `linalg.transpose` → `Program::Transpose`) and state the PLAIN `[k, n]` form over its \
         result. Refused rather than contracted the wrong way round."
    ))
}

/// THE RMSNORM CHAIN, as the ONE fused op it is — `out = x · rsqrt(mean(x²) + eps) · gamma`.
///
/// # WHY THIS IS THE FIX AND FILLING THE SCALE REGISTRY WAS NOT
///
/// The chain's `ms = sum(x²) · INV_D` is an `arith.mulf` by a splatted constant, so
/// [`splat_scale_of`] classifies it as a `ScalarMul` and `scalarmul_at` then asks for its multiplier's
/// registry slot. That request is the one thing that must NOT be granted:
/// `triton-ktir-superdsc/src/layout.rs`'s `scales_for` records, at its own code, that registering
/// `1/cols` WAS TRIED and is a divergence — "THE EPSILON ONLY — the mean-of-squares divisor is NOT a
/// registry scale. `1/cols` is bound at the reserved `RMS_INVCOLS_TID` as a `[1, stick]` row with its
/// own placement". The registry index IS the device tid (`SCALARMUL_SCALE_BASE - i`), so a spurious
/// slot MOVES THE ADDRESS every later constant reaches the card at.
///
/// So the divisor must stop being a standalone multiply rather than be registered around it. Fusing
/// the chain does exactly that: `INV_D` becomes chain-interior, the epsilon goes to the registry
/// (which is what `scales_for` already provides for `Program::RmsNorm`), and `1/cols` goes to the
/// reserved tid inside `assemble_rmsnorm`, which is where the shipped path reads it.
///
/// # IT PROVES THE MATCH; IT DOES NOT PATTERN-MATCH
///
/// Four discriminators, and none of them is structural bookkeeping:
///
/// * **the square's two operands are the SAME value.** `mean(x²)` is `x·x`; `x·y` with the same op
///   kinds is a different function.
/// * **the value the normaliser scales is the value that was squared.** `x · rsqrt(mean(x²)+eps)`
///   reads `x` TWICE, and that is what makes it a normalisation rather than a scale by an unrelated
///   reciprocal.
/// * **the reduction is a SUM.** A max-reduce in that slot is not a mean of squares.
/// * **the divisor is exactly `1/cols` for THIS tile's column extent.** `mean` is `sum/N`, so a
///   multiplier that is not `1/N` makes it something else — checked numerically against the operand's
///   own width, not assumed.
///
/// A chain that almost matches is REFUSED BY NAME, never lowered as the nearest thing.
#[derive(Clone, Debug)]
pub struct RmsNormChain {
    /// The normalised tensor — squared, reduced, and scaled.
    pub x: Ssa,
    /// The learned gain the chain's last multiply applies (`n1` / `gamma`).
    pub gamma: Ssa,
    /// The final `arith.mulf`'s result: what the fused program produces.
    pub out: Ssa,
    /// The epsilon this chain's own `arith.addf` adds — read per chain, because
    /// `program_rmsnorm_eps` requires ONE root in the whole function and a decoder has two.
    pub eps: f32,
    /// Every intermediate the fused op never materialises, so `lower_function` lowers NOTHING for
    /// them. The final multiply is not here: it is the op that BECOMES the fused program.
    pub consumed: Vec<Ssa>,
}

/// Read EVERY rmsnorm chain off the program, or say why what is there is not one.
///
/// One chain per `math.rsqrt`. An empty `Vec` means the program states none.
pub fn program_rmsnorm_chains(f: &IRFunction<'static>) -> Result<Vec<RmsNormChain>, Error> {
    let mut out = Vec::new();
    for root in f
        .operations
        .iter()
        .filter(|o| o.op_type == OpKind::MathRsqrt)
    {
        if let Some(c) = rmsnorm_chain_from(f, root)? {
            out.push(c);
        }
    }
    Ok(out)
}

/// The one chain rooted at `root` (a `math.rsqrt`), proven link by link.
///
/// `Ok(None)` where the shape around the root is not an rmsnorm at all AND says so unambiguously — a
/// bare `rsqrt` of a loaded tile is a legitimate `Elementwise(Rsqrt)` and must fall through to the 1:1
/// map rather than be refused. `Err` where it is RECOGNISABLY an attempt at one and a link does not
/// close, because then the 1:1 map would silently lower a mis-shaped normalisation as separate ops.
fn rmsnorm_chain_from(
    f: &IRFunction<'static>,
    root: &ktir_core::ir::Operation<'static>,
) -> Result<Option<RmsNormChain>, Error> {
    let def_of = |s: Ssa| f.operations.iter().find(|o| o.result == Some(s));
    let splat_value = |s: Ssa| -> Option<f64> {
        let sp = def_of(s)?;
        if sp.op_type != OpKind::TensorSplat {
            return None;
        }
        let c = def_of(*sp.operands.first()?)?;
        if c.op_type != OpKind::ArithConstant {
            return None;
        }
        c.attributes.iter().find_map(|(k, v)| match (k, v) {
            (AttrKey::Value, Attr::Float(x)) => Some(*x),
            _ => None,
        })
    };
    // `rsqrt(add)`. A root whose operand is not an `arith.addf` is not this shape and is not an
    // attempt at it — a plain `Elementwise(Rsqrt)`, so `Ok(None)`.
    let Some(add) = root.operands.first().copied().and_then(def_of) else {
        return Ok(None);
    };
    if add.op_type != OpKind::ArithAddf {
        return Ok(None);
    }
    // `mean + eps`: exactly one operand is a splatted constant, the other the mean.
    let (Some(eps), Some(ms)) = ({
        let mut e = None;
        let mut m = None;
        for &s in add.operands.iter() {
            match splat_value(s) {
                Some(v) => e = Some(v),
                None => m = Some(s),
            }
        }
        (e, m)
    }) else {
        return Ok(None);
    };
    // `ms = <reduce> · INV_D`.
    let Some(scale) = def_of(ms) else {
        return Ok(None);
    };
    if scale.op_type != OpKind::ArithMulf {
        return Ok(None);
    }
    let (Some(inv_d), Some(red_v)) = ({
        let mut d = None;
        let mut r = None;
        for &s in scale.operands.iter() {
            match splat_value(s) {
                Some(v) => d = Some(v),
                None => r = Some(s),
            }
        }
        (d, r)
    }) else {
        return Ok(None);
    };
    let Some(red) = def_of(red_v) else {
        return Ok(None);
    };
    if red.op_type != OpKind::LinalgReduce {
        return Ok(None);
    }
    // From here the shape IS an attempt at an rmsnorm — a reduce feeding `·c + c` feeding `rsqrt` is
    // not something else — so every remaining failure is an `Err`, not a fall-through.
    if reduce_kind_of(red) != Some(ReduceKind::Sum) {
        return err(format!(
            "{}: an rmsnorm's mean-of-squares reduces with `arith.addf`; this one reduces with \
             `{:?}`, which is a different function",
            f.name,
            red.attributes.iter().find_map(|(k, v)| match (k, v) {
                (AttrKey::ReduceFn, Attr::Op(o)) => Some(*o),
                _ => None,
            })
        ));
    }
    // ⛔ THE SQUARE'S TWO OPERANDS MUST BE THE SAME VALUE. `mean(x²)` is `x·x`; `x·y` has the same op
    // kinds and is a different function.
    let Some(sq) = red.operands.first().copied().and_then(def_of) else {
        return err(format!("{}: the rmsnorm reduce reads nothing", f.name));
    };
    if sq.op_type != OpKind::ArithMulf {
        return err(format!(
            "{}: an rmsnorm reduces the SQUARE of its input; this reduce reads a `{:?}`",
            f.name, sq.op_type
        ));
    }
    let [sa, sb] = sq.operands[..] else {
        return err(format!("{}: the rmsnorm square is not binary", f.name));
    };
    if sa != sb {
        return err(format!(
            "{}: the rmsnorm's `arith.mulf` under the reduce multiplies TWO DIFFERENT values, so it \
             is not `x·x` and the reduction is not a mean of squares. Refused rather than lowered as \
             an rmsnorm, whose descriptor squares one operand.",
            f.name
        ));
    }
    let x = sa;

    // FORWARD from the root to the two multiplies, through the rank plumbing the broadcast needs
    // (`expand_shape` -> `collapse_shape` -> `linalg.broadcast`), which the fused op absorbs.
    let mut consumed = vec![add.result, scale.result, red.result, sq.result]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let Some(root_v) = root.result else {
        return err(format!("{}: `math.rsqrt` with no result", f.name));
    };
    consumed.push(root_v);
    let deep = f.ops_deep();
    let sole_reader = |v: Ssa| -> Option<&ktir_core::ir::Operation<'static>> {
        let mut it = deep.iter().copied().filter(|o| o.operands.contains(&v));
        let first = it.next()?;
        it.next().is_none().then_some(first)
    };
    // The rank walk is OPTIONAL in shape but its ops are consumed when present: a producer that
    // broadcasts `r` to `[rows, cols]` states expand/collapse/broadcast, and one that keeps it rank-2
    // states none of them.
    let mut cursor = root_v;
    for _ in 0..3 {
        let Some(next) = sole_reader(cursor) else {
            break;
        };
        if !matches!(
            next.op_type,
            OpKind::TensorExpandShape | OpKind::TensorCollapseShape | OpKind::LinalgBroadcast
        ) {
            break;
        }
        let Some(r) = next.result else { break };
        consumed.push(r);
        cursor = r;
    }
    // `x · r` — and ⛔ IT MUST READ THE SAME `x` THAT WAS SQUARED. That is what makes this a
    // normalisation of `x` rather than a scale by an unrelated reciprocal.
    let Some(scale_mul) = sole_reader(cursor) else {
        return err(format!(
            "{}: nothing reads the rmsnorm's `rsqrt` result (through its rank plumbing), so the \
             normalising multiply is missing",
            f.name
        ));
    };
    if scale_mul.op_type != OpKind::ArithMulf || !scale_mul.operands.contains(&x) {
        return err(format!(
            "{}: the rmsnorm's normalising multiply is `{:?}` and {} the value that was squared. \
             `x · rsqrt(mean(x²)+eps)` reads `x` TWICE; a multiply of something else by the same \
             reciprocal is a different function.",
            f.name,
            scale_mul.op_type,
            if scale_mul.operands.contains(&x) {
                "reads"
            } else {
                "does NOT read"
            }
        ));
    }
    let Some(scaled) = scale_mul.result else {
        return err(format!(
            "{}: the rmsnorm's normalising multiply has no result",
            f.name
        ));
    };
    consumed.push(scaled);
    // `· gamma` — the learned gain, itself reached through its own rank plumbing.
    let Some(gain_mul) = sole_reader(scaled) else {
        return err(format!(
            "{}: nothing reads `x · rsqrt(...)`. `Program::RmsNorm` applies a learned gain; a \
             normalisation with none is not the shape this door lowers.",
            f.name
        ));
    };
    if gain_mul.op_type != OpKind::ArithMulf {
        return err(format!(
            "{}: `x · rsqrt(...)` is read by `{:?}` rather than by the gain multiply",
            f.name, gain_mul.op_type
        ));
    }
    let Some(gain_src) = gain_mul.operands.iter().copied().find(|&s| s != scaled) else {
        return err(format!(
            "{}: the rmsnorm's gain multiply squares its input",
            f.name
        ));
    };
    // Walk BACK through the gain's rank plumbing to the value a `Region` can be found for.
    let mut gamma = gain_src;
    for _ in 0..3 {
        let Some(d) = def_of(gamma) else { break };
        if !matches!(
            d.op_type,
            OpKind::TensorExpandShape | OpKind::TensorCollapseShape | OpKind::LinalgBroadcast
        ) {
            break;
        }
        consumed.push(gamma);
        let Some(src) = d.operands.first().copied() else {
            break;
        };
        gamma = src;
    }
    let Some(out) = gain_mul.result else {
        return err(format!(
            "{}: the rmsnorm's gain multiply has no result",
            f.name
        ));
    };

    // ⛔ THE DIVISOR IS `1/cols` FOR THIS TILE, CHECKED NUMERICALLY. `mean` is `sum/N`; a multiplier
    // that is not `1/N` makes the program something other than a mean of squares, and lowering it as
    // one would emit `assemble_rmsnorm`'s own `1/cols` at the reserved tid — silently substituting a
    // different constant for the one the program states. `cols` is the reduced extent, which is the
    // square's own trailing dim.
    let cols = match sq.result_type {
        Some(ktir_core::irtype::IrType::Tensor { dims, .. }) if !dims.is_empty() => {
            dims[dims.len() - 1] as f64
        }
        _ => {
            return err(format!(
                "{}: the rmsnorm square states no tensor shape, so `1/cols` cannot be checked \
                 against its divisor",
                f.name
            ));
        }
    };
    // fp32-representable reciprocals of powers of two are exact, and every `D_MODEL` here is one; the
    // comparison is done in f32 so a producer that splatted an f32 `1/N` matches bit-for-bit.
    if (inv_d as f32) != (1.0f64 / cols) as f32 {
        return err(format!(
            "{}: the rmsnorm's divisor is {inv_d}, but this tile reduces {cols} columns and a mean \
             needs 1/{cols} = {}. `assemble_rmsnorm` binds `1/cols` itself at the reserved \
             `RMS_INVCOLS_TID`, so lowering this as an rmsnorm would substitute a DIFFERENT constant \
             for the one the program states.",
            f.name,
            1.0f64 / cols
        ));
    }

    Ok(Some(RmsNormChain {
        x,
        gamma,
        out,
        eps: eps as f32,
        consumed,
    }))
}

/// Read EVERY silu longhand off the program, or say why one of them is not one.
///
/// ⭐ ONE CHAIN PER `arith.negf`, NOT ONE PER FUNCTION, and the difference is a real configuration:
/// `decoder_block.py`'s two-layer kernel has TWO MLPs in one `func.func`, so two silus. What has to
/// be unique is each LINK — every intermediate read by exactly the next op — and that is checked per
/// chain. Refusing a function for holding two was an artificial limit that turned a legitimate
/// two-layer program into a refusal.
///
/// An empty `Vec` means there is no `arith.negf` at all, i.e. nothing here claims to be a silu.
/// `Err` means one was found and the chain around it does NOT close — reported rather than left to
/// the 1:1 map, because "`ArithNegf` has no `Program`" would send a reader looking for a missing
/// `Elementwise` variant when the real answer is that a silu is mis-shaped.
pub fn program_silu_mul_chains(f: &IRFunction<'static>) -> Result<Vec<SiluMulChain>, Error> {
    let mut out = Vec::new();
    for neg in f
        .operations
        .iter()
        .filter(|o| o.op_type == OpKind::ArithNegf)
    {
        out.push(silu_mul_chain_from(f, neg)?);
    }
    Ok(out)
}

/// The one chain headed by `neg`, proven link by link.
fn silu_mul_chain_from(
    f: &IRFunction<'static>,
    neg: &ktir_core::ir::Operation<'static>,
) -> Result<SiluMulChain, Error> {
    let def_of = |s: Ssa| f.operations.iter().find(|o| o.result == Some(s));
    let splat_value = |s: Ssa| -> Option<f64> {
        let sp = def_of(s)?;
        if sp.op_type != OpKind::TensorSplat {
            return None;
        }
        let c = def_of(*sp.operands.first()?)?;
        if c.op_type != OpKind::ArithConstant {
            return None;
        }
        c.attributes.iter().find_map(|(k, v)| match (k, v) {
            (AttrKey::Value, Attr::Float(x)) => Some(*x),
            _ => None,
        })
    };
    // Every op that reads `v`, so a link's exclusivity can be PROVEN rather than assumed.
    //
    // ⛔⛔⛔ `ops_deep`, NOT `f.operations`, AND THAT IS A CORRECTNESS MATTER RATHER THAN A STYLE
    // ONE. Fusing these five ops into one descriptor DESTROYS four intermediate values: the device
    // primitive never materialises `-g`, `e^-g`, `1+e^-g` or `silu(g)`. The exclusivity check below
    // is the only thing standing between that and a program where something ELSE reads one of them.
    // Every other reader in this file walks `f.operations`, which is the TOP LEVEL only — and
    // `regions()`'s own `deep > top` guard exists in this same file precisely because that blindness
    // is real. A reader inside an `scf.for` body would be invisible to a top-level walk, the chain
    // would fuse anyway, and the in-loop consumer would read a value nothing ever wrote. Free for a
    // straight-line producer, where `ops_deep()` and `operations` are the same list.
    let deep = f.ops_deep();
    let readers = |v: Ssa| {
        deep.iter()
            .copied()
            .filter(move |o: &&ktir_core::ir::Operation<'static>| o.operands.contains(&v))
    };

    let (Some(gate), Some(neg_v)) = (neg.operands.first().copied(), neg.result) else {
        return err(format!(
            "{}: `arith.negf` with no operand or no result",
            f.name
        ));
    };

    // ONE LINK OF THE CHAIN: the single op of kind `want` that reads `v`, or a refusal saying WHICH
    // of the three ways it failed. Separated because the three are different diagnoses, and one
    // message conflating them reads as self-contradictory — "read by `ArithAddf` rather than by
    // exactly one `arith.addf`" was the real output before this, and a test caught it.
    //
    // ⛔ MORE THAN ONE READER IS A REFUSAL, NOT A DETAIL. The fusion DESTROYS `v`, so a second reader
    // is a consumer of a value the device primitive never materialises.
    let link = |v: Ssa,
                want: OpKind,
                what: &str|
     -> Result<&ktir_core::ir::Operation<'static>, Error> {
        let found: Vec<_> = readers(v).collect();
        match found.len() {
            0 => err(format!(
                "{}: nothing reads {what}, so the silu longhand does not continue there",
                f.name
            )),
            1 if found[0].op_type == want => Ok(found[0]),
            1 => err(format!(
                "{}: {what} is read by `{:?}`, but a silu longhand reads it with `{:?}`",
                f.name, found[0].op_type, want
            )),
            n => err(format!(
                "{}: {what} is read by {n} ops ({}), and this fusion DESTROYS it — the device's \
                 `OpFunc::Silu` never materialises the longhand's intermediates, so a second reader \
                 would be reading a value nothing writes. Exactly one reader, `{:?}`, is required.",
                f.name,
                found
                    .iter()
                    .map(|o| format!("{:?}", o.op_type))
                    .collect::<Vec<_>>()
                    .join(", "),
                want
            )),
        }
    };

    // negate -> exp. `math.exp` and nothing else: `exp2` is a different base and would need the
    // argument pre-scaled by log2(e), which this chain does not do.
    let exp = link(neg_v, OpKind::MathExp, "the `arith.negf`'s result")?;
    let Some(exp_v) = exp.result else {
        return err(format!("{}: `math.exp` with no result", f.name));
    };

    // exp -> 1 + exp. THE CONSTANT MUST BE 1.0: sigmoid is 1/(1+e⁻ˣ), and any other constant is a
    // different function.
    let add = link(exp_v, OpKind::ArithAddf, "`exp(-x)`")?;
    let one = add
        .operands
        .iter()
        .filter(|&&s| s != exp_v)
        .filter_map(|&s| splat_value(s))
        .next();
    match one {
        // Spelled as an inequality rather than `Some(v) if v == 1.0 => {}` plus a catch-all: a
        // float LITERAL PATTERN is a lint of its own, and this way the refusal keeps `v` to name
        // the constant the program actually states.
        Some(v) if v != 1.0 => {
            return err(format!(
                "{}: the silu longhand's `arith.addf` adds {v}, not 1.0. `sigmoid(x)` is \
                 `1/(1+e^-x)`, so any other constant is a DIFFERENT function and lowering it as \
                 `OpFunc::Silu` would compute something the program does not state.",
                f.name
            ));
        }
        Some(_) => {}
        None => {
            return err(format!(
                "{}: the silu longhand's `arith.addf` adds a value that is not a splatted \
                 constant, so nothing proves it is the 1 of `1+e^-x`",
                f.name
            ));
        }
    }
    let Some(add_v) = add.result else {
        return err(format!("{}: `arith.addf` with no result", f.name));
    };

    // (1+exp) -> divide. ⛔ THE NUMERATOR IS THE NEGATED VALUE ITSELF. This is the discriminator:
    // `g/(1+e^-g)` is `g·sigmoid(g)` = silu(g); the same five op kinds with any other numerator are
    // a different function.
    let div = link(add_v, OpKind::ArithDivf, "`1 + exp(-x)`")?;
    if div.operands.first().copied() != Some(gate) {
        return err(format!(
            "{}: the `arith.divf`'s NUMERATOR is not the value the `arith.negf` consumed. \
             `g/(1+e^-g)` is `g·sigmoid(g)` = silu(g); dividing anything else by `1+e^-g` is a \
             different function with the same five op kinds, so it is refused rather than lowered \
             as `OpFunc::Silu`.",
            f.name
        ));
    }
    let Some(div_v) = div.result else {
        return err(format!("{}: `arith.divf` with no result", f.name));
    };

    // silu -> multiply by `up`. The other operand is `up`, whatever produced it.
    // `Program::SiluMul` is `silu(gate)·up`; a bare silu with no multiply is `Elementwise` territory
    // and this door has no arm for one, so a missing multiply is named rather than half-lowered.
    let mul = link(div_v, OpKind::ArithMulf, "`silu(x)`")?;
    let Some(up) = mul.operands.iter().copied().find(|&s| s != div_v) else {
        return err(format!(
            "{}: the silu's `arith.mulf` squares it rather than multiplying by an `up` tile",
            f.name
        ));
    };
    let Some(mul_v) = mul.result else {
        return err(format!("{}: `arith.mulf` with no result", f.name));
    };

    Ok(SiluMulChain {
        gate,
        up,
        mul: mul_v,
        consumed: [neg_v, exp_v, add_v, div_v],
    })
}

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

/// WHAT THIS DOOR WILL LOWER ONE OP AS — a `Program` the crate already has, or one of the two kinds
/// that are reachable ONLY from here.
///
/// ⭐ A LOCAL ENUM RATHER THAN TWO MORE [`Program`] VARIANTS, and that is a deliberate cost trade.
/// `Program` is matched exhaustively by consumers outside this crate — a caller's layout planner has
/// one arm per kind — so every variant added there is a breaking change for them. `Silu` and `Reduce`
/// are reachable only through the whole-function walk (a per-`Program` producer states the FUSED kind
/// that contains them), so keeping them here costs those callers nothing.
#[derive(Clone, Copy, Debug)]
enum Lowering {
    /// A kind the crate's `Program` already names, lowered by its existing entry point.
    Node(Program),
    /// The five-op silu longhand, PROVEN by [`program_silu_mul_chains`] and lowered as one
    /// `silumul`.
    Silu,
    /// A bare `linalg.reduce`, whose combiner is an attribute rather than an op kind.
    Reduce(ReduceKind),
    /// `tile * <splatted constant>` — the multiplier read off THIS op, not the whole function. See
    /// [`splat_scale_of`] for the discriminator and why it cannot be per-function here.
    ScalarMul(f32),
    /// `tile * <splatted runtime scalar>` — rung 3 of the address-provenance ladder: the multiplier
    /// is a `[1,1]` PARAMETER the launch binds (`const:t<id>`), so the emission names its binding by
    /// tid rather than looking a value up in the registry. See [`bound_scale_of`] for the
    /// discriminator. The tensor operand is the splat's other side, same as [`ScalarMul`].
    BoundScalarMul(BoundScale),
    /// A fused `x · rsqrt(mean(x²)+eps) · gamma`, with the epsilon this chain's own. See
    /// [`program_rmsnorm_chains`] for why fusing it is the fix and filling the scale registry was not.
    RmsNorm(f32),
}

/// THE `Elementwise(Mul)` / `ScalarMul` TIEBREAK, read off ONE op rather than the whole function.
///
/// ⭐ THE DISCRIMINATOR IS THE CRATE'S OWN, NOT AN INVENTION. [`Program`]'s doc names it: "Elementwise
/// `Mul` and `ScalarMul` are the one pair that needs a tiebreak, and it already exists:
/// `program_scalarmul_scale` tells them apart by whether an operand is a `tensor.splat`." So an
/// `arith.mulf` with exactly one splatted-constant operand IS a scalar multiply — there is no other
/// kind it could be, because `Elementwise(Mul)` needs two tensor operands and the splat is not one.
/// Choosing it is applying a documented rule, not picking a node kind on the producer's behalf.
///
/// ⛔ AND IT MUST BE PER-OP, NOT PER-FUNCTION. `program_scalarmul_scale` reads every `arith.mulf` and
/// requires them all to agree, which is exact for a one-node function. ONE decoder layer holds 21
/// `arith.mulf`: most have no splat, and the splatted ones carry FOUR DIFFERENT constants — `INV_D`
/// (the rmsnorm reduction divisor), `QK_SCALE` (the score scale), and `RM` twice (the residual
/// multipliers). The whole-function reader returns `None` there by construction, so the scale has to
/// be read from the op that uses it.
///
/// Returns `(scale, the tensor operand)`. `None` when this is not a scalar multiply at all:
///
/// * NEITHER operand is a splat — an ordinary two-tensor `Elementwise(Mul)`, which the 1:1 map handles.
/// * BOTH are splats — a multiply of two compile-time constants. That is not a scalarmul (there is no
///   tensor to scale) and emitting one would invent an operand; it belongs to constant folding
///   upstream, so it falls through to be refused by name.
pub fn splat_scale_of(
    f: &IRFunction<'static>,
    op: &ktir_core::ir::Operation<'static>,
) -> Option<(f32, Ssa)> {
    let def_of = |s: Ssa| f.operations.iter().find(|o| o.result == Some(s));
    // A splatted compile-time FLOAT. `arith.constant` → `tensor.splat` is the only spelling this
    // crate's producers use, and it is the one `program_scalarmul_scale` already reads.
    let splat_value = |s: Ssa| -> Option<f64> {
        let sp = def_of(s)?;
        if sp.op_type != OpKind::TensorSplat {
            return None;
        }
        let c = def_of(*sp.operands.first()?)?;
        if c.op_type != OpKind::ArithConstant {
            return None;
        }
        c.attributes.iter().find_map(|(k, v)| match (k, v) {
            (AttrKey::Value, Attr::Float(x)) => Some(*x),
            _ => None,
        })
    };
    let [a, b] = op.operands[..] else { return None };
    match (splat_value(a), splat_value(b)) {
        (Some(v), None) => Some((v as f32, b)),
        (None, Some(v)) => Some((v as f32, a)),
        // Both or neither: not a scalar multiply. See the doc above.
        _ => None,
    }
}

/// A rung-3 bound scale: the `[1,1]` parameter a `tensor.splat` reads through its load, and the
/// tensor operand on the mulf's other side.
#[derive(Clone, Copy, Debug)]
pub struct BoundScale {
    /// The parameter's BINDING tid — the number the launch's `const:t<id>` names.
    pub tid: u32,
    /// The tensor operand the mulf scales.
    pub tensor: Ssa,
}

/// THE RUNG-3 HALF OF THE TIEBREAK: a `tensor.splat` over a `ktdp.load` of a `[1,1]` view of a
/// function ARGUMENT.
///
/// `to_ktir`'s `expand_splat_of_scalar_argument` emits exactly that chain for a scalar argument
/// (the argument retyped `index`, a `[1,1]` fp16 HBM view, a `[1,1]` access tile at corner (0,0),
/// a load), and re-points the splat at the load. So the discriminator is structural: a splat whose
/// operand is a `ktdp.load` over a degenerate `[1,1]` view of a parameter IS a launch-bound
/// multiplier — the ATTN_SCALE mechanism with the value bound per launch instead of baked.
///
/// ⛔ FAIL CLOSED ON THE SHAPE: only a BOTH-AXES-DEGENERATE view qualifies (`[1,1]`). A splat of a
/// wider parameter (a row vector, a `[1, cols]` gamma — the mb-broadcast defect shape) is NOT a
/// bound scalar and must fall through to the 1:1 map and be refused there, because emitting a
/// scalarmul for it would address one value where the program means a row.
///
/// ⛔ AND ON THE PROVENANCE: the view's address operand must be a function ARGUMENT. A `[1,1]` load
/// of an intermediate has no binding and no launch spelling, so it is left alone rather than
/// misread as one.
pub fn bound_scale_of(
    k: &KtirNode,
    op: &ktir_core::ir::Operation<'static>,
) -> Result<Option<BoundScale>, Error> {
    let f = &k.func;
    // The `[1,1]` VIEW a splat-of-load reads, when it reads one: splat → load → tile → view.
    // Every step is a `find` over the top level, the same walk `region_for_operand` takes.
    let splat_view = |s: Ssa| -> Option<ktir_core::ir::Ssa> {
        let sp = f.operations.iter().find(|o| o.result == Some(s))?;
        if sp.op_type != OpKind::TensorSplat {
            return None;
        }
        let loaded = *sp.operands.first()?;
        let ld = f.operations.iter().find(|o| o.result == Some(loaded))?;
        if ld.op_type != OpKind::KtdpLoad {
            return None;
        }
        let tile = *ld.operands.first()?;
        let tl = f.operations.iter().find(|o| o.result == Some(tile))?;
        if tl.op_type != OpKind::KtdpConstructAccessTile {
            return None;
        }
        let view = *tl.operands.first()?;
        let vw = f.operations.iter().find(|o| o.result == Some(view))?;
        if vw.op_type != OpKind::KtdpConstructMemoryView {
            return None;
        }
        Some(view)
    };
    let [a, b] = op.operands[..] else {
        return Ok(None);
    };
    // Which operand is the splat, which the tensor — the splat may sit on either side. The
    // TENSOR side is the mulf operand that is not the splat.
    let (view, tensor) = match (splat_view(a), splat_view(b)) {
        (Some(view), None) => (view, b),
        (None, Some(view)) => (view, a),
        // Both or neither: not a bound scalar multiply. Both-splat is constant folding's
        // business; neither is an ordinary two-tensor `Elementwise(Mul)`.
        _ => return Ok(None),
    };
    // ⛔ THE BOTH-AXES-DEGENERATE TEST: only a `[1,1]` view qualifies.
    let vw = f
        .operations
        .iter()
        .find(|o| o.result == Some(view))
        .expect("splat_view found it");
    let shape = super::lower_ktir_to_superdsc::shape_2d(vw);
    if shape != Some((1, 1)) {
        return Ok(None);
    }
    // ⛔ THE PARAMETER TEST: the view's address operand must be a function argument, and the
    // binding tid is that argument's position.
    let ptr = vw.operands.first().copied();
    let Some(idx) = f.arguments.iter().position(|(s, _)| Some(*s) == ptr) else {
        return Ok(None);
    };
    let Some(tid) = k.bindings.get(idx).map(|b| b.get()) else {
        return Ok(None);
    };
    Ok(Some(BoundScale { tid, tensor }))
}

/// The [`ReduceKind`] a `linalg.reduce` states, read off its own `ReduceFn` attribute.
///
/// ⭐ NOT A 1:1 OP-KIND MAP, WHICH IS WHY IT IS NOT IN [`program_of`]. `OpKind::LinalgReduce` says
/// "a reduction happens here" and nothing about WHICH one; the combiner is an attribute
/// (`ReduceFn = Op(ArithAddf)` for a sum, `Op(ArithMaxnumf)` for a max), so the kind has to be read
/// from the op rather than derived from its kind. `program_of` takes an `OpKind` alone by design, so
/// this sits beside it and [`lower_function`] consults both.
///
/// A combiner outside the two the device has is `None`, and the caller refuses it by name — a
/// `linalg.reduce` combining with `arith.mulf` (a product reduction) has no `sfp` reduce op-func, and
/// lowering it as the nearest one would compute a different function.
pub fn reduce_kind_of(op: &ktir_core::ir::Operation<'static>) -> Option<ReduceKind> {
    op.attributes.iter().find_map(|(k, v)| match (k, v) {
        (AttrKey::ReduceFn, Attr::Op(OpKind::ArithAddf)) => Some(ReduceKind::Sum),
        (AttrKey::ReduceFn, Attr::Op(OpKind::ArithMaxnumf)) => Some(ReduceKind::Max),
        _ => None,
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

/// ONE recognised broadcast: the value a consumer reads, resolved back to the rank-2 SOURCE the
/// chain started from, plus the axis.
#[derive(Clone, Copy)]
struct Bcast {
    src: Ssa,
    axis: BcastAxis,
}

/// ⭐⭐⭐ A BROADCAST IS ADDRESSING, NOT A NODE — so its ops are PLUMBING and the CONSUMER carries it.
///
/// A KTIR producer spells `m[:, None]` as a three-op chain over a rank-1 value:
///
/// ```text
/// %562 = LinalgReduce(..)                      -> tensor<64>
/// %461 = TensorExpandShape(%562)  TargetShape=[64, 1]  Dimensions=1
/// %761 = TensorCollapseShape(%461) TargetShape=[64]
/// %762 = TensorEmpty()
/// %462 = LinalgBroadcast(%761, %762) Dimensions=[1]    -> tensor<64x64>
/// %463 = ArithSubf(%458, %462)
/// ```
///
/// None of those four ops is a device op. The AIU has no broadcast primitive and needs none: a
/// broadcast operand is an ADDRESSING MODE of the op that consumes it, which
/// [`crate::emit::EwOperand`] states through `mb_broadcast`/`out_broadcast` and
/// [`assemble_pointwise_broadcast_off`] emits. That is not an inference from the shape of the API —
/// it is what the hardware-proven attention does: [`crate::ir::bridge::tiled_op_sdsc_op::attn`]
/// computes this identical softmax (subtract a per-row max, exp, sum, divide by a per-row sum) and
/// emits NO node for either broadcast; `attn_esub` reads `In::col(&hm(run_m))` and the reduce's
/// `[rows, 1]` accumulator is the operand. So a `Program::Broadcast` variant would be inventing a
/// node kind for something the device does as an operand mode, and the reason
/// [`pointwise_extents_agree`](super::lower_ktir_to_superdsc) refuses a degenerate operand today is
/// stated in its own refusal: the SEEDED whole-tensor builder cannot say "broadcast", and the
/// `EwOperand` builders can.
///
/// This recogniser therefore returns, for each broadcast RESULT, the source value a consumer should
/// read and the axis — and the `consumed` set of every value the chain minted, which no descriptor
/// lowers. It is the same rank walk [`program_rmsnorm_chains`] already does for its `rsqrt` and its
/// gain (through the same three op kinds, to depth 3); this generalises it to any consumer.
///
/// # ⛔ WHAT IS REFUSED RATHER THAN GUESSED
///
/// A `tensor.expand_shape` is only a broadcast when the dim it inserts is DEGENERATE — that is what
/// makes the source a row or column vector rather than a reshaped tile. A general reshape (a
/// `[64, 128]` → `[128, 64]` relayout, say) states no degenerate dim, moves real elements, and is a
/// data movement the pointwise operand modes cannot express. Those are named, not treated as
/// broadcasts. The `expand_shape`'s own `TargetShape` and the `linalg.broadcast`'s `Dimensions` state
/// the axis INDEPENDENTLY, so they are cross-checked: a program where they disagree is not a
/// broadcast this door understands.
fn program_broadcast_chains(
    f: &IRFunction<'static>,
) -> Result<
    (
        std::collections::HashMap<Ssa, Bcast>,
        std::collections::HashSet<Ssa>,
    ),
    Error,
> {
    let mut map: std::collections::HashMap<Ssa, Bcast> = std::collections::HashMap::new();
    let mut consumed: std::collections::HashSet<Ssa> = std::collections::HashSet::new();
    let deep = f.ops_deep();
    let def_of = |v: Ssa| deep.iter().copied().find(|o| o.result == Some(v));
    for bc in deep
        .iter()
        .copied()
        .filter(|o| o.op_type == OpKind::LinalgBroadcast)
    {
        let Some(res) = bc.result else { continue };
        // The AXIS AS THE BROADCAST ITSELF STATES IT — `Dimensions` names the dim(s) being ADDED, so
        // `[1]` sprays along the trailing (stick) axis and `[0]` down the rows. Exactly one dim, because
        // a two-axis broadcast from a scalar is `In::scalar`, a different operand mode, and no producer
        // in this door's programs states one; it is named rather than folded in silently.
        let dims = bc.attributes.iter().find_map(|(k, v)| match (k, v) {
            (AttrKey::Dimensions, Attr::IntList(d)) => Some(*d),
            _ => None,
        });
        let Some(dims) = dims else {
            return err(format!(
                "{}: `linalg.broadcast` states no `Dimensions`, so which axis it sprays along is \
                 unknown and the operand mode (`In::col` vs `In::mb`) cannot be chosen",
                f.name
            ));
        };
        let axis = match dims {
            [1] => BcastAxis::Col,
            [0] => BcastAxis::Mb,
            _ => {
                return err(format!(
                    "{}: `linalg.broadcast` adds dims {dims:?}. This door expresses a broadcast as an \
                     OPERAND MODE of the consuming pointwise op, and there are exactly two: `[1]` \
                     (a per-row scalar along the stick axis, `In::col`) and `[0]` (a row vector down \
                     the rows, `In::mb`). Anything else — a two-axis spray from a scalar, or a \
                     non-leading/non-trailing axis — is a different mode or a real data movement, and \
                     is named rather than emitted as one of these two",
                    f.name
                ));
            }
        };
        // WALK BACK to the rank-2 value the chain started from, through the SAME three op kinds and the
        // SAME depth `program_rmsnorm_chains` walks. The `tensor.empty` second operand is the
        // broadcast's `outs` init and carries no data, so only operand 0 is followed.
        let Some(mut cursor) = bc.operands.first().copied() else {
            return err(format!(
                "{}: `linalg.broadcast` has no source operand",
                f.name
            ));
        };
        let mut chain = vec![res];
        let mut saw_degenerate_expand = false;
        for _ in 0..3 {
            let Some(d) = def_of(cursor) else { break };
            match d.op_type {
                OpKind::TensorExpandShape => {
                    // ⛔ THE DEGENERATE DIM IS WHAT MAKES THIS A BROADCAST. `TargetShape` states the
                    // shape AFTER the insert; a `1` in it at the axis the broadcast sprays is a row or
                    // column vector. Without one this is a relayout of real elements and no operand
                    // mode expresses it.
                    let target = d.attributes.iter().find_map(|(k, v)| match (k, v) {
                        (AttrKey::TargetShape, Attr::IntList(t)) => Some(*t),
                        _ => None,
                    });
                    let Some(target) = target else {
                        return err(format!(
                            "{}: `tensor.expand_shape` states no `TargetShape`, so whether it inserts \
                             a DEGENERATE dim — the thing that makes it a broadcast rather than a \
                             relayout — cannot be decided",
                            f.name
                        ));
                    };
                    let want = match axis {
                        BcastAxis::Col => target.len().checked_sub(1),
                        BcastAxis::Mb => Some(0),
                    };
                    // CROSS-CHECKED AGAINST THE BROADCAST'S OWN `Dimensions`: two independent
                    // statements of the same axis, so a program where they disagree is refused instead
                    // of one of them being believed.
                    if want.map(|i| target.get(i).copied()) != Some(Some(1)) {
                        return err(format!(
                            "{}: `tensor.expand_shape` targets {target:?}, whose {} axis is not \
                             degenerate, but the `linalg.broadcast` reading it sprays along {axis:?}. \
                             A broadcast operand mode needs a `1` on the axis being sprayed — the \
                             source must be a row or column vector. A general reshape moves real \
                             elements and is a data movement no pointwise operand mode can express, so \
                             it is named here rather than mis-addressed as a broadcast.",
                            f.name,
                            match axis {
                                BcastAxis::Col => "trailing",
                                BcastAxis::Mb => "leading",
                            }
                        ));
                    }
                    saw_degenerate_expand = true;
                }
                OpKind::TensorCollapseShape => {}
                _ => break,
            }
            chain.push(cursor);
            let Some(next) = d.operands.first().copied() else {
                break;
            };
            cursor = next;
        }
        if !saw_degenerate_expand {
            return err(format!(
                "{}: a `linalg.broadcast` sprays along {axis:?} but its source is not reached through \
                 a `tensor.expand_shape` that inserts a DEGENERATE dim. That expand is the only thing \
                 in the program that says the source IS a row or column vector, so without it the \
                 operand's extent is unproven and addressing it as a broadcast would read past its end.",
                f.name
            ));
        }
        for v in chain {
            consumed.insert(v);
        }
        map.insert(res, Bcast { src: cursor, axis });
    }
    Ok((map, consumed))
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
///
/// ⭐⭐⭐ AN INDIRECT ACCESS TILE RESOLVES TO THE **GATHERED PARAMETER'S** REGION, and it must never
/// be reported as `None`. `Ok(None)` means "not a load of a parameter", and the caller turns it into
/// "then it is a constant (a `tensor.splat`)" — a sound deduction only while the two tile kinds are
/// the one this walk matches. A `ktdp.construct_indirect_access_tile` IS a load of a parameter, so
/// falling through named the wrong cause for every gathered operand: a Triton embedding's
/// `arith.mulf` reads a gathered `[64, 4096]` row tile and was reported as reading a splat, sending a
/// reader to look for a constant that is not there. MEASURED on `test/fixtures/embedding.py`, whose
/// `%34 = arith.mulf(%30, %33)` has the gathered load at input 0 and the splat at input 1.
///
/// ⛔ IT WAS A REFUSAL, on the grounds that "no `Program` in this crate carries an index operand" —
/// and that had stopped being true of this crate. [`Program::ScalarMul`] carries one:
/// [`super::lower_ktir_to_superdsc::scalarmul_at`] takes a [`Gather`] and routes to
/// [`crate::emit::assemble_pointwise_broadcast_gather`], which emits the cross-linked `allocate`
/// pair and the `computeOp_.indirectAccessIndexLabeledDs` the refusal said nothing emitted. MEASURED:
/// `bake_py embedding_granite` through the per-`Program` door bakes one descriptor carrying all four
/// gather fields, and `dxp_standalone --bundle` exits 0 on it. So the door was shut on a fact about
/// the crate, and the whole-function walk was the last reader still asserting it.
///
/// ⛔⛔ THE GATHERED TILE'S OWN WINDOW IS **NOT** WHAT IS RETURNED, and that is deliberate. An
/// indirect tile's `Shape` is the gathered tile (`[BLOCK_M, D_MODEL]`) and its ROWS are chosen by
/// data, so there is no window over the table for a `Region` to carry — the table is addressed at its
/// base and the index supplies every row. [`super::lower_ktir_to_superdsc::scalarmul_at`] reads this
/// region for its `tid` (the operand name, and the key into the layout) and takes the descriptor's
/// row count off the OUTPUT, which is the only place it can honestly come from. So the parameter's
/// own region is exact here, and the one thing that would be wrong on it — a row corner — is what
/// `base_addressed` refuses downstream.
pub fn region_for_operand(k: &KtirNode, v: Ssa) -> Result<Option<Region>, Error> {
    // The parameter-rooted walk already computes every field of a `Region` correctly; the only
    // thing wrong for a multi-op function is WHICH parameter and WHICH tile. So find the parameter
    // this value reads and reuse that Region, rather than rebuilding one field by field and
    // risking a different answer from the same IR.
    let f = &k.func;
    let load = f
        .operations
        .iter()
        .find(|o| o.result == Some(v) && o.op_type == OpKind::KtdpLoad);
    let Some(load) = load else { return Ok(None) };
    let tile_v = load.operands.first().copied();
    // A GATHERED LOAD. Operand 0 of the indirect tile is the DATA view and operand 1 the INDEX view —
    // [`gather_of`]'s own reading of the same op, so the two walks cannot disagree about which
    // parameter is gathered.
    if let Some(ind) = f
        .operations
        .iter()
        .find(|o| o.result == tile_v && o.op_type == OpKind::KtdpConstructIndirectAccessTile)
    {
        return region_of_view_operand(k, ind, 0);
    }
    let tile = f
        .operations
        .iter()
        .find(|o| o.result == tile_v && o.op_type == OpKind::KtdpConstructAccessTile);
    let Some(tile) = tile else { return Ok(None) };
    let Some(mut r) = region_of_view_operand(k, tile, 0)? else {
        return Ok(None);
    };
    // ⭐⭐⭐⭐⭐ THE WINDOW IS THE LOAD'S OWN TILE, NOT THE PARAMETER'S FIRST.
    //
    // `regions()` keeps the FIRST access tile over each parameter's view — a sound description of a
    // straight-line program that reads each buffer one way, and SILENTLY WRONG the moment a producer
    // states several tiles over one view. MEASURED on the unblocked (`unroll_constant_trip_loops`)
    // SwiGLU MLP, small twin (16/128/256, BLOCK_N=128, BLOCK_K=64): every gate/up matmul of every
    // k-trip and every n-trip read the SAME window (the weight's trip-0 `[128, 64]` tile at the
    // buffer base), and the card output matched `2·[silu(2·x₀Wg)·(2·x₀Wu)]Wd` to corr 0.999996 — the
    // ×2s being the duplicated no-offset trips summed by the accumulator `arith.addf`s, and the Wd
    // in it the window-0 read through TILE-extent strides (the kernel's stick-group stride derives
    // from the tile's k, not the tensor's — see `matmul_oriented`'s windowing arm).
    //
    // So the tile in hand — THIS load's own `ktdp.construct_access_tile`, whose corner constants the
    // cast-resolving `index_constants` already sees — states the window, and it is copied onto the
    // parameter's Region here. The tile's corner operands (1 = row, 2 = column) and its `Shape` are
    // read by exactly the code `regions()` reads them with (`shape_2d` and `consts.get`), so the two
    // walks cannot disagree about one tile. A tile whose corners are NOT constants names a window
    // no descriptor can carry, and is refused by name rather than silently falling back to the
    // parameter's first tile — the same discipline as `regions()`'s in-loop guard.
    if tile.operands.len() == 3 {
        let consts = super::lower_ktir_to_superdsc::index_constants(&k.func);
        let Some(shape) = super::lower_ktir_to_superdsc::shape_2d(tile) else {
            return err(format!(
                "{}: the access tile feeding this load states no 2-D `shape`, so its window is \
                 unknown",
                f.name
            ));
        };
        let (rs, cs) = match (
            tile.operands.get(1).and_then(|s| consts.get(s).copied()),
            tile.operands.get(2).and_then(|s| consts.get(s).copied()),
        ) {
            (Some(rs), Some(cs)) => (rs as u32, cs as u32),
            _ => {
                return err(format!(
                    "{}: the access tile feeding this load states a window whose corner is not a \
                     constant (`ktdp.construct_access_tile` operands 1/2), so the descriptor cannot \
                     carry the offset. A loop-carried or computed corner needs hoisting or unrolling \
                     before the whole-function door can address it.",
                    f.name
                ));
            }
        };
        r.r_start = rs;
        r.c_start = cs;
        r.r_len = shape.0;
        r.c_len = shape.1;
    }
    Ok(Some(r))
}

/// The [`Region`] of the parameter that operand `slot` of a TILE op reinterprets — the tail both the
/// plain and the indirect arm of [`region_for_operand`] share, and the walk the index operand needs a
/// door of its own for.
///
/// ⭐ ONE FUNCTION BECAUSE THE THREE CALLERS MUST AGREE. `regions()` is a positional zip of the
/// function's arguments against the node's bindings, so "which parameter" decides which row of it is
/// read; three separate `position(|(a, _)| Some(*a) == ptr)` walks is three chances to reach a
/// different row from the same IR.
fn region_of_view_operand(
    k: &KtirNode,
    tile: &Operation<'_>,
    slot: usize,
) -> Result<Option<Region>, Error> {
    let f = &k.func;
    let view_v = tile.operands.get(slot).copied();
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
    let mut next_tid: u32 = k
        .bindings
        .iter()
        .map(|b| b.get())
        .max()
        .map_or(0, |m| m + 1);
    let mut inter: std::collections::HashMap<Ssa, Region> = std::collections::HashMap::new();

    // THE ONE FUSED CHAIN THIS DOOR READS, before the 1:1 walk rather than inside it. See
    // [`program_silu_mul_chain`] for why the module header's "recognises NOTHING" survives this: the
    // device has `OpFunc::Silu` and KTIR has no op for it, so the longhand is the only spelling a
    // producer HAS — and lowering it op-by-op computes a different algorithm, not a slower one.
    let silus = program_silu_mul_chains(f)?;
    // THE RMSNORM CHAINS, read before the walk for the same reason the silus are. ⭐ AND THIS ONE IS
    // LOAD-BEARING FOR MORE THAN FUSION: its `ms = sum(x²)·INV_D` would otherwise be classified as a
    // `ScalarMul` and ask for `1/cols`'s registry slot — the exact divergence
    // `triton-ktir-superdsc`'s `scales_for` documents as refused, because the registry index IS the
    // device tid and a spurious slot moves every later constant's address. Recognising the chain makes
    // that multiply chain-interior, so the divisor never reaches the registry at all.
    let rmsnorms = program_rmsnorm_chains(f)?;
    // THE BROADCAST PLUMBING, read before the walk for the same reason: these ops lower to NOTHING and
    // the consumer carries the broadcast as an operand mode. See [`program_broadcast_chains`].
    let (bcasts, bcast_consumed) = program_broadcast_chains(f)?;
    // ⭐⭐⭐ THE GATHERS THE PROGRAM STATES, read ONCE and from [`gathers_of`] — the same joins the
    // per-`Program` door's `gather_of` makes, so the two doors cannot disagree about which parameter
    // is the table or which is the index. A program with ONE indirect tile reads identically to
    // `gather_of`; a program with SEVERAL is the unrolled-sweep shape (every tile over the SAME index
    // and value parameters, one per trip — `gathers_of` refuses different pairs by name), and the
    // walk below names the op that carries EACH tile.
    //
    // ⛔ AND THEY ARE CARRIED, NOT COUNTED: `gather_carried` below is checked after the walk, per
    // tile. A gather the program states and no descriptor declares is a silently DIRECT read of the
    // table's first rows — well formed, the right shape and dtype, and nothing else in the pipeline
    // compares an emitted descriptor against the program's indirect tile.
    let gathers = gathers_of(k)?;
    // The gathered LOAD each tile's `ktdp.load` produced — the value an op's input must BE (by
    // identity) for that tile's gather to be the one it reads. Built once here so the per-op join
    // below is a lookup, not a re-walk the walk could disagree with.
    let gathered_loads: std::collections::HashMap<Ssa, usize> = {
        let mut m = std::collections::HashMap::new();
        for (i, tile) in f
            .operations
            .iter()
            .filter(|o| o.op_type == OpKind::KtdpConstructIndirectAccessTile)
            .enumerate()
        {
            let Some(tv) = tile.result else { continue };
            for o in f.operations.iter() {
                if o.op_type == OpKind::KtdpLoad
                    && o.operands.first() == Some(&tv)
                    && let Some(r) = o.result
                {
                    m.insert(r, i);
                }
            }
        }
        m
    };
    let mut gather_carried = vec![false; gathers.len()];

    for op in f.operations.iter() {
        if is_plumbing(op.op_type) {
            continue;
        }
        // The chain's INTERIOR — negate, exp, 1+, divide — is lowered by nothing at all: the one
        // `Program::SiluMul` below stands for all five ops, and the device primitive never
        // materialises these values. Skipped BEFORE `program_of`, which would otherwise refuse the
        // `arith.negf` for having no 1:1 mapping (correctly, in isolation).
        if op.result.is_some_and(|r| {
            silus.iter().any(|c| c.consumed.contains(&r))
                || rmsnorms.iter().any(|c| c.consumed.contains(&r))
                // The expand/collapse/broadcast chain itself: no descriptor, the consumer states it.
                || bcast_consumed.contains(&r)
        }) {
            continue;
        }
        // The terminal `arith.mulf` IS the fused program, and its inputs are the chain's gate and up
        // rather than its own operands (which are `silu(gate)` and up). Everything else keeps the
        // 1:1 map exactly as before.
        let terminal = silus.iter().copied().find(|c| op.result == Some(c.mul));
        // A `linalg.reduce`'s combiner is an ATTRIBUTE, not its op kind, so the kind is read off the op
        // beside the 1:1 map rather than from it — and it is carried HERE rather than as a `Program`
        // variant, because `Program` is matched exhaustively by consumers outside this crate and a bare
        // reduce is reachable only through this door. An unrecognised combiner falls through to the
        // refusal below and is named there.
        let reduce = (op.op_type == OpKind::LinalgReduce)
            .then(|| reduce_kind_of(op))
            .flatten();
        // `arith.mulf` by a SPLATTED CONSTANT is a scalar multiply, not an `Elementwise(Mul)` — the
        // tiebreak `Program`'s own doc names. Read per-op, because one decoder layer states four
        // different multipliers. Checked BEFORE the 1:1 map, which would otherwise route it to
        // `Elementwise(Mul)` and then fail on the splat operand having no `Region`.
        let scalar = (op.op_type == OpKind::ArithMulf)
            .then(|| splat_scale_of(f, op))
            .flatten();
        // ⭐ RUNG 3: a mulf by a splat of a LOADED `[1,1]` PARAMETER is the SAME tiebreak with the
        // multiplier bound at launch rather than baked. `to_ktir`'s `expand_splat_of_scalar_argument`
        // emits the `[1,1]` view/tile/load chain for a scalar argument and re-points its splat at the
        // load; this reads WHICH PARAMETER that load reads, so the emission can name its binding.
        // Checked AFTER the constant arm (a constant-backed splat still resolves by value) and
        // BEFORE the 1:1 map, which would otherwise route the op to `Elementwise(Mul)` — the
        // mb-broadcast defect shape, a `[1,1]` region against a full tile.
        let bound_scale = if op.op_type == OpKind::ArithMulf && scalar.is_none() {
            bound_scale_of(k, op)?
        } else {
            None
        };
        // An rmsnorm chain's TERMINAL gain multiply becomes the one fused program. Checked before the
        // splat arm and before the 1:1 map, both of which would otherwise claim it.
        let rms = rmsnorms.iter().find(|c| op.result == Some(c.out));
        let Some(program) = terminal
            .map(|_| Lowering::Silu)
            .or(rms.map(|c| Lowering::RmsNorm(c.eps)))
            .or(reduce.map(Lowering::Reduce))
            .or(scalar.map(|(v, _)| Lowering::ScalarMul(v)))
            .or(bound_scale.map(Lowering::BoundScalarMul))
            .or_else(|| program_of(op.op_type).map(Lowering::Node))
        else {
            return err(format!(
                "{}: `{:?}` has no 1:1 `Program`, so this door cannot lower it without choosing a \
                 node kind on the producer's behalf. Either it belongs to a fused kind the producer \
                 should state by calling that entry point directly, or it needs a `Program` variant",
                f.name, op.op_type
            ));
        };

        // THE WEIGHT'S ORIENTATION, PROVEN FROM THE MAPS before any extent is trusted. See
        // [`BOrient`]: the extent guards downstream cannot distinguish the two framings when
        // k == n, so a square weight would otherwise lower to a descriptor contracting the other
        // way round. Threaded into `emit_one` as a value; never re-derived and never defaulted.
        let b_orient = if matches!(program, Lowering::Node(Program::Matmul)) {
            Some(matmul_b_orientation(f, op)?)
        } else {
            None
        };

        // THIS OP'S INPUTS. `linalg.matmul`'s last operand is its `outs` accumulator init, not an
        // input, and the result is the output — so the inputs are the leading operands.
        let n_in = match program {
            // The silu's two inputs are the chain's gate and up, which the recogniser proved.
            Lowering::Silu => 2,
            // `linalg.reduce`'s operands are `(data, init)`: the init is the SEED, which the reduce
            // op-func carries itself, so only the data is an input.
            Lowering::Reduce(_) => 1,
            // The splat is not an operand of the descriptor — it rides in the op as a bound `[1,1]`
            // const — so a scalar multiply reads ONE tensor.
            Lowering::ScalarMul(_) | Lowering::BoundScalarMul(_) => 1,
            // x and gamma; the epsilon and `1/cols` are the fused body's own.
            Lowering::RmsNorm(_) => 2,
            Lowering::Node(Program::Matmul) => 2,
            Lowering::Node(Program::Elementwise(e)) => {
                match super::lower_ktir_to_superdsc::elementwise_op_func(f.name, e) {
                    Ok((_, arity)) => arity,
                    Err(e) => return Err(e),
                }
            }
            Lowering::Node(Program::Transpose) => 1,
            Lowering::Node(_) => op.operands.len(),
        };

        // A fused chain's inputs are the ones the RECOGNISER proved, not the terminal op's operands.
        let in_values: Vec<Ssa> = match (terminal, scalar, bound_scale) {
            // The two tensors the rmsnorm recogniser proved, walked back through the rank plumbing to
            // values a `Region` exists for.
            _ if rms.is_some() => {
                let c = rms.expect("just matched");
                vec![c.x, c.gamma]
            }
            (Some(c), _, _) => vec![c.gate, c.up],
            // The TENSOR operand the recogniser proved, not operand 0 — the splat sits on either side
            // (`ms * INV_D` has it second, and nothing obliges a producer to put it there).
            (None, Some((_, tensor)), _) => vec![tensor],
            // RUNG 3: same rule, the splat's other side. The bound scale is a `[1,1]` parameter, not
            // an operand of the descriptor.
            (None, None, Some(b)) => vec![b.tensor],
            (None, None, None) => op.operands.iter().copied().take(n_in).collect(),
        };

        // A BROADCAST OPERAND READS ITS SOURCE. The chain minted no buffer (it emitted no op), so the
        // value the consumer names has no region of its own; the rank-2 source it sprays does. The AXIS
        // the recogniser proved is then CHECKED against that region's shape below rather than trusted:
        // two independent facts about the same operand, so a chain whose axis and extent disagree
        // cannot address a full tile as a vector.
        let bcast_axes: Vec<Option<BcastAxis>> = in_values
            .iter()
            .map(|v| bcasts.get(v).map(|b| b.axis))
            .collect();
        let in_values: Vec<Ssa> = in_values
            .iter()
            .map(|v| bcasts.get(v).map_or(*v, |b| b.src))
            .collect();

        let mut per_op: Vec<Region> = Vec::with_capacity(n_in + 1);
        // ⭐ WHICH OPERANDS ARE PARAMETER-BACKED, kept BESIDE the regions because a `Region` cannot
        // say. `region_for_operand` answers `Some` for a value the parameter-rooted walk reaches (a
        // `ktdp.load` of a `ktdp.construct_access_tile` of a `ktdp.construct_memory_view` of a
        // function argument) and `None` for one this function COMPUTED — and that is exactly the
        // distinction [`transposed_b_is_placeable`] needs, so it is recorded here rather than
        // re-derived from a footprint that is identical either way.
        let mut from_parameter: Vec<bool> = Vec::with_capacity(n_in + 1);
        for (i, v) in in_values.iter().enumerate() {
            match region_for_operand(k, *v)? {
                Some(mut r) => {
                    r.is_out = false;
                    per_op.push(r);
                    from_parameter.push(true);
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
                        from_parameter.push(false);
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

        // ⛔⛔⛔ THE STATED AXIS AND THE RESOLVED EXTENT MUST AGREE, and this is the check that makes
        // the broadcast safe rather than assumed. `program_broadcast_chains` proved the axis from the
        // program's OWN attributes (the `expand_shape`'s degenerate dim, cross-checked against the
        // `linalg.broadcast`'s `Dimensions`); the region says what the source operand actually IS. They
        // are independent facts, and the emission depends on both: the consuming descriptor addresses a
        // `Col` operand as one value per row and an `Mb` operand as one row of values. If a chain the
        // program spelled as `[m, 1]` resolved to a full `[m, n]` tile, addressing it out-broadcast
        // would read one column and spray it — a silent wrong answer with no shape error anywhere. The
        // pointwise arm downstream selects its operand mode from THIS extent, so pinning the two
        // together here is what stops the axis and the addressing from drifting apart.
        for (i, axis) in bcast_axes.iter().enumerate() {
            let Some(axis) = *axis else { continue };
            let r = per_op[i];
            let degenerate = match axis {
                BcastAxis::Col => r.c_len == 1,
                BcastAxis::Mb => r.v_rows == 1,
            };
            if !degenerate {
                let what = match axis {
                    BcastAxis::Col => "column",
                    BcastAxis::Mb => "row",
                };
                return err(format!(
                    "{}: `{:?}` input {i} is a recognised {axis:?} broadcast, but the source it \
                     resolves to (t{}) is `[{}, {}]` — not degenerate on the axis being sprayed. The \
                     program's `expand_shape`/`linalg.broadcast` say this operand is a {what} vector \
                     and its extent says it is a full tile; the descriptor would address one {what} \
                     and spray it over the whole output, silently. These are two independent \
                     statements about the same operand and they disagree, so neither is believed.",
                    f.name, op.op_type, r.tid, r.v_rows, r.c_len,
                ));
            }
        }

        // ⭐⭐⭐ THIS OP'S GATHER, IF ONE THE PROGRAM STATES IS ONE IT READS — BY IDENTITY, NOT TID.
        //
        // The single-tile reading joined by TID ("some input's region names the gathered table's
        // parameter"), which is right for one tile and WRONG for several: an unrolled sweep's every
        // trip reads the SAME two parameters, so a TID join hands EVERY gathered matmul the FIRST
        // tile's index window — trip 1 would gather trip 0's rows from a clean bake, the silent
        // wrong answer this join exists to make impossible. The identity join instead asks which
        // tile's LOAD this op's input IS: `gathered_loads` maps every gathered load result to its
        // tile, so an op reading a gathered value names ITS tile, and an op reading the table
        // directly (a plain load, no tile match) names none.
        //
        // ⛔ AND THE INDEX PARAMETER IS APPENDED TO `per_op`, because the walk has no other way to
        // reach it. The index is ADDRESSING, so it is not an operand of the `arith.mulf` at all — the
        // gathered load consumed it inside the indirect tile. Without its region `scalarmul_at` cannot
        // check the one thing that matters most ("the index buffer really holds `rows` indices", whose
        // failure mode is converting past the buffer's end), and `split_out_excluding` is the
        // mechanism that keeps the arity at 1 while the region is present: it drops the index BY TID.
        //
        // ⛔ APPENDED AFTER the inputs and BEFORE the output, which both orders below rely on.
        // `bcast_axes` is indexed by input position, so the index may not sit among them; a
        // `ScalarMul`'s gathered table is its one tensor input, so it may not come first either.
        let op_gather = 'g: {
            for (i, v) in in_values.iter().enumerate() {
                if let Some(&ti) = gathered_loads.get(v) {
                    let g = gathers[ti];
                    let idx_r = f
                        .arguments
                        .iter()
                        .zip(k.bindings.iter())
                        .position(|(_, b)| b.get() == g.index_tid)
                        .and_then(|i| regions(k).ok().and_then(|all| all.get(i).copied()));
                    let Some(mut idx_r) = idx_r else {
                        return err(format!(
                            "{}: the program gathers through t{}, which is not one of this function's \
                             bound parameters — an index buffer with no binding has no placement and no \
                             stated length, and the descriptor makes dbo's idx→address program iterate \
                             one entry per row of the node",
                            f.name, g.index_tid
                        ));
                    };
                    idx_r.is_out = false;
                    per_op.push(idx_r);
                    gather_carried[ti] = true;
                    // ⛔ AND A SECOND CONSUMER OF ONE TILE'S GATHERED LOAD IS REFUSED HERE rather
                    // than after the walk: the counts seal below checks the whole function, but a
                    // mid-walk consumer list keeps the message at the op that read it twice.
                    break 'g Some((g, ti, i));
                }
            }
            None
        };
        // ⛔⛔⛔ A GATHERED OPERAND MAY ONLY REACH A BODY THAT DECLARES THE INDEX. `ScalarMul` is the
        // one — `scalarmul_at` takes the [`Gather`] and routes to
        // `assemble_pointwise_broadcast_gather`. Every other arm of [`emit_one`] would emit an
        // ordinary descriptor over the table, i.e. read its first `rows` rows DIRECTLY and ignore the
        // ids: right shape, right dtype, right distribution, wrong rows, and nothing downstream
        // compares the two. So it is refused by name rather than lowered as the nearest thing.
        //
        // ⭐⭐⭐ THE ONE EXCEPTION — RUNG 4: A MATMUL WHOSE **B** IS THE GATHERED TABLE. The
        // materialization is synthesized HERE, before the refusal can fire, because the vendor's own
        // gathered fixtures and IBM's paged attention both spell a gathered contraction exactly one
        // way and this is it: a KERNEL-less gathered copy writing a minted intermediate, the matmul
        // reading the intermediate as plain-B. See [`gathered_matmul_materializes`], which states the
        // whole argument and does the work.
        let gathered_b = if let Some((g, ti, gi)) = op_gather
            && matches!(program, Lowering::Node(Program::Matmul))
        {
            gathered_matmul_materializes(
                k,
                layout,
                &mut GatherEmit {
                    sym_id_base,
                    out: &mut out,
                    next_tid: &mut next_tid,
                    inter: &mut inter,
                },
                g,
                ti,
                gi,
                &in_values,
            )?
        } else {
            false
        };
        if op_gather.is_some()
            && !gathered_b
            && !matches!(
                program,
                Lowering::ScalarMul(_) | Lowering::BoundScalarMul(_)
            )
        {
            return err(format!(
                "{}: `{:?}` reads a `ktdp.construct_indirect_access_tile` (t{} gathered through \
                 t{}), and this door declares an index operand only for `Program::ScalarMul` — \
                 `scalarmul_at` is the one body that carries a `Gather`. Every other assembler here \
                 would emit a descriptor reading the table DIRECTLY at its base, which is the right \
                 shape and the wrong rows, from a clean bake. Give the gathered operand its own \
                 `ScalarMul` node, or add the index to the assembler this op needs.",
                f.name,
                op.op_type,
                op_gather.map_or(0, |(g, _, _)| g.value_tid),
                op_gather.map_or(0, |(g, _, _)| g.index_tid),
            ));
        }

        // ⭐⭐⭐ RUNG 4'S SECOND HALF — THE MATMUL READS THE INTERMEDIATE, NOT THE TABLE. The copy was
        // emitted above and the region recorded under the GATHERED VALUE; this swaps `per_op`'s B slot
        // for it, so the matmul's own assembler sees a plain `[k, n]` activation. The index region
        // appended by the gather join is DROPPED from the list at the same time (truncate to `n_in`):
        // the matmul's descriptor carries no index operand (the copy declared it), and
        // `matmul_oriented`'s `ins.len() != 2` arity check would read a three-input list as the fp8
        // W8A8 form and refuse it.
        if gathered_b {
            let Some((_, _, gi)) = op_gather else {
                unreachable!("gathered_b implies op_gather")
            };
            let b_value = in_values[gi];
            let Some(r) = inter.get(&b_value) else {
                return err(format!(
                    "{}: the gathered B's materialized intermediate is missing from the walk's own \
                     map — the copy was emitted but its region was not recorded",
                    f.name
                ));
            };
            let mut r = *r;
            r.is_out = false;
            per_op[1] = r;
            per_op.truncate(n_in);
        }

        // ⭐⭐⭐⭐⭐ A PLAIN-B CONTRACTION IS CONTRACTED WHERE IT LIES — NO RELAYOUT, NO ARCH FEATURE,
        // AND NOTHING TO EMIT HERE AT ALL.
        //
        // THE KERNEL SLOT'S DEVICE ORDER IS `[in, out]` = `[k, n]`, WITH `in` AS THE STICK-BLOCK ROW
        // COUNT. That is not this emitter's choice and it is the same for both orientations:
        // `lower_ktir_to_superdsc::matmul` declares `Stk::<KernelTag>::kernel(k, n_dev, ..)`, i.e.
        // `StickLayout::kernel(k_in, n_out)` — `dev_off_stk`'s rank-2 law with `dims[0] = k`, element
        // `(k, n)` at `(n/stk)·(k·stk) + k·stk + (n%stk)`. A `[k, n]` operand is therefore ALREADY in
        // the residency the slot reads, and `StickLayout::addr_eq` says so in the crate's own words:
        // "`RowBlocked` vs `Kernel` is the identical stick-block formula", so the `[mb, out]` tile a
        // producing matmul wrote IS a `[k, n]` kernel with `in = mb`.
        //
        // ⭐ THE DEVICE EVIDENCE IS THE SHIPPED ATTENTION, AND IT IS THIS EXACT CONTRACTION.
        // `sdsc_abstract::vcache_write_offset`: "the V cache … the VALUE bmm reads as a `[cap, hd]`
        // KERNEL sticked on `hd` (`out`) … the kernel is `[k=cap, n=hd]`", and `attn.rs`'s value leg
        // declares `Stk::kernel(v_stride, hd)` — the V cache contracted in place, no relayout, granite
        // fp8 at 41 tok/s. `p @ V` is `tl.dot(p, v)`. Its twin `kcache_kt_write_offset` completes the
        // argument from the other side: the SCORE leg's kernel is `[hd, cap]`, in-rows again, which is
        // why the K cache is written ALREADY TRANSPOSED (`restickify_kt_opspec_2d`) — the relayout
        // exists to MAKE a `[k, n]` buffer, never to consume one.
        //
        // ⛔⛔⛔ WHAT USED TO BE HERE, AND WHY BOTH OF ITS CLAIMS WERE WRONG. This site transposed B
        // into a minted `[n, k]` intermediate with `OpFunc::Transpose` and handed the matmul THAT. Two
        // measurements killed it:
        //   * IT COMPUTED THE WRONG THING. Baked on both decoders, `32_transpose_o41` wrote a
        //     `[128, 64]` tile (`primaryDsInfo_` OUTPUT `stickDimOrder_ ["out","mb"]`, the 8×8
        //     inter-slice block) and `33_matmul_o42` then declared its KERNEL `layoutDimOrder_
        //     ["in","out"]` with `in_ = 64, out_ = 128` over it — a k-row/n-col reading of a buffer that
        //     physically had n rows. The relayout moved the bytes AWAY from the slot's own address law;
        //     the un-transposed `[64, 128]` value was already exactly what that descriptor reads.
        //   * IT COULD NOT RUN. dxp refuses to schedule it on `SENARCH=MPW4` — "DtException: Implicit
        //     syncs not available for architectures prior to RCUDD1A, ddcv1.cpp line 3416" — at 16
        //     cores AND at 1 core, and identically when `OpFunc::Restickify` was substituted, so the
        //     limit is the DATA-STAGE CHANGE a relayout makes and not either primitive. Across all six
        //     gated fixtures the entire exercised op set is `add`/`batchmatmul`/`mean`/`mul`/`rsqrt`/
        //     `silu`: no relayout primitive appears anywhere in the passing suite. That refusal is the
        //     only reason the wrong answer above was never baked.
        //
        // So the orientation is not lowered by an op; it is a statement about the REGION's framing, and
        // it is threaded into `matmul` (below) where the two extents are read. `assemble_transpose` and
        // its `lower_ktir_to_superdsc::transpose` door are left exactly as they were — they serve
        // `Program::Transpose`, a producer stating a `linalg.transpose` in its own right.
        //
        // ⭐ AND THE OTHER HALF IS NO LONGER A GAP — IT IS THE REFUSAL DIRECTLY BELOW. A transpose-B B
        // whose region is a COMPUTED value has nobody to place its bytes, so this door will not lower
        // it. See [`transposed_b_is_placeable`], which states the whole argument.
        if let Some(b) = b_orient {
            // ⛔ FAIL CLOSED ON THE INDEX ITSELF. `n_in` is 2 for a matmul so operand 1 always
            // exists, and defaulting a MISSING entry to "placeable" would make a malformed program
            // lower silently — the exact shape of the defect this guard closes.
            let Some(&b_from_parameter) = from_parameter.get(1) else {
                return err(format!(
                    "{}: `{:?}` states a contraction orientation but this door resolved {} \
                     input region(s), so which operand is B — and therefore whether its \
                     transposition can be placed — is unknown",
                    f.name,
                    op.op_type,
                    from_parameter.len()
                ));
            };
            transposed_b_is_placeable(f.name, b, b_from_parameter)?;
        }

        // THIS OP'S OUTPUT: the parameter a `ktdp.store` writes from this op's result.
        let stored = f
            .operations
            .iter()
            .find(|s| s.op_type == OpKind::KtdpStore && s.operands.first().copied() == op.result);
        let Some(store) = stored else {
            // NOT STORED => AN INTERMEDIATE. Mint a buffer for it, declare it to the layout, and
            // record it so the consuming op finds it as an input.
            let Some(res) = op.result else {
                return err(format!(
                    "{}: `{:?}` has no result to place",
                    f.name, op.op_type
                ));
            };
            let dims = match op.result_type {
                Some(ktir_core::irtype::IrType::Tensor { dims, .. }) if dims.len() == 2 => {
                    [dims[0] as u32, dims[1] as u32]
                }
                // A RANK-1 INTERMEDIATE IS `[rows, 1]` WHEN ITS VALUE CAME FROM A ROW REDUCTION, and
                // that is decided by PROVENANCE rather than by this op's kind.
                //
                // `linalg.reduce` over the trailing axis of `[rows, cols]` yields `[rows]` — one value
                // per row, exactly the `[rows, 1]` accumulator `assemble_reduce_seeded` writes. A
                // decoder's rmsnorm then keeps that column shape through `ms = sum(x*x) * INV_D` and
                // `rsqrt(ms + eps)`, so the SAME footprint is right for every pointwise op along that
                // chain — but for the same reason it is NOT right for a rank-1 value that is a `[cols]`
                // ROW vector (a broadcast source), where `[cols, 1]` would size the buffer correctly
                // and address it wrong.
                //
                // The discriminator is therefore the INPUT's own column extent, which is already known
                // here: a reduce writes `c_len == 1`, so anything reading a one-column tile is on the
                // reduction's side of the program. A reduce's own result qualifies by its kind because
                // it has no such input to consult.
                Some(ktir_core::irtype::IrType::Tensor { dims, .. })
                    if dims.len() == 1
                        && (matches!(program, Lowering::Reduce(_))
                            || per_op.iter().any(|r| !r.is_out && r.c_len == 1)) =>
                {
                    [dims[0] as u32, 1]
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
            // ⛔⛔⛔ A MATMUL'S INTERMEDIATE IS RESERVED AT ITS **DEVICE** WIDTH, NOT THE LOGICAL
            // SHAPE — everything else keeps the logical dims.
            //
            // The util-floor pad is only real if a SECOND party makes it real (`work.rs`'s own
            // law): the matmul emitter's `n_dev = DeviceWidth::for_output(m, n, k)` bumps a
            // sub-8-stick FLOP-heavy output (64 → 512 at m=64, k=2048, macs ≥ 2²⁰) so the gemm
            // does not strand below the util floor, and `out_width_the_weight_holds` will NOT cap
            // it away when the floor would be violated — the pad is mandatory for this shape. A
            // mint at the LOGICAL dims declares 8192 B for a buffer the emitter then addresses
            // with 65536 B — MEASURED as `resolve_seg_base`'s refusal on granite tiled_k BLOCK_N=64
            // (`synth 't5': access 0B + 65536B exceeds footprint 8192B`), and without that guard
            // it would alias the next intermediate. The mint IS the second party for a
            // whole-function intermediate (the worker's weight zero-pad is the one on the
            // per-`Program` path), so it reserves what the emitter addresses.
            //
            // ⛔ AND ONLY THE MATMUL ARM, because a POINTWISE result's bump is OPTIONAL and the
            // consumer side already yields to the layout: `pointwise_width_the_output_holds`
            // caps a pointwise emission at the width its operands' placements actually hold, so a
            // pointwise intermediate minted LOGICAL is read and written at its logical width
            // (one whole stick at 64) — the card-verified attention score-tile chain, byte for
            // byte. Minting those at `for_pointwise` width would flip the caps to no-ops and
            // re-emit every [64,64] score-tile consumer at 512 wide. The matmul's bump has no
            // such yield: the floor re-check refuses the narrowed width, so the reservation must
            // move instead.
            //
            // The REGION stays LOGICAL in every case: it carries the program's stated extents,
            // and the consuming op's own derivation reads them (the down matmul's `k = a.c_len`
            // is BLOCK_N=64, matching its weight window). A corner-0 window narrower than the
            // buffer is address-correct in the stick-blocked packing — one row's stick-groups
            // sit `rows·lanes` apart regardless of the column count — so the pad columns the
            // producer wrote (weight columns past the window; never read at corner 0) stay
            // numerically inert.
            let dev_dims = match program {
                // The producer's own rule, with `k` from the A operand's region (the same
                // `k = a.c_len` the emitter itself reads; A is `per_op`'s first input).
                Lowering::Node(Program::Matmul) => {
                    let k = per_op.first().map_or(dims[1], |a| a.c_len);
                    [dims[0], DeviceWidth::for_output(dims[0], dims[1], k).get()]
                }
                _ => dims,
            };
            layout.synth(crate::place::PlaceId::Act(tid), &dev_dims);
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
            let mut emitted = emit_one(
                name,
                program,
                &per_op,
                sym_id_base,
                Some(layout),
                WalkProof {
                    b_orient,
                    gather: op_gather.map(|(g, _, _)| g),
                    bcast: &bcast_axes,
                },
            )?;
            out.append(&mut emitted);
            lowered += 1;
            continue;
        };
        let out_tile = store.operands.get(1).copied();
        let out_view = f
            .operations
            .iter()
            .find(|o| o.op_type == OpKind::KtdpConstructAccessTile && o.result == out_tile)
            .and_then(|t| t.operands.first().copied());
        let out_ptr = f
            .operations
            .iter()
            .find(|o| o.op_type == OpKind::KtdpConstructMemoryView && o.result == out_view)
            .and_then(|o| o.operands.first().copied());
        let out_idx = f.arguments.iter().position(|(a, _)| Some(*a) == out_ptr);
        let Some(out_idx) = out_idx else {
            return err(format!(
                "{}: `{:?}`'s store names no parameter",
                f.name, op.op_type
            ));
        };
        let all = regions(k)?;
        let mut o = all[out_idx];
        o.is_out = true;
        per_op.push(o);

        let mut emitted = emit_one(
            f.name,
            program,
            &per_op,
            sym_id_base,
            layout,
            WalkProof {
                b_orient,
                gather: op_gather.map(|(g, _, _)| g),
                bcast: &bcast_axes,
            },
        )?;
        out.append(&mut emitted);
        lowered += 1;
    }

    if lowered == 0 {
        return err(format!("{}: no compute op in the function", f.name));
    }
    // ⛔⛔⛔ A GATHER THE PROGRAM STATES AND NO DESCRIPTOR CARRIES IS A SILENTLY DIRECT READ.
    //
    // `gathers_of` proved the program has indirect access tiles; if the join above matched no op's
    // input for some tile, every emitted descriptor reads the table at its base and that tile's ids
    // are never consulted — the right shape, the right dtype, and rows the program never named.
    // Nothing further down compares an emitted descriptor against the program's indirect tile, and
    // the fixture's own table rows are independent `randn` draws, so the wrong rows are
    // statistically indistinguishable from the right ones (`triton-numeric/tests/embedding_gather.rs`
    // opens with exactly this).
    //
    // ⭐ IT CANNOT BE FOLDED INTO THE JOIN. The join is per-op and cannot know whether a LATER op
    // reads the table, so "no op matched" is only knowable once the walk is done. Per TILE, so an
    // unrolled sweep's every trip is checked — not just its first.
    for (ti, g) in gathers.iter().enumerate() {
        if !gather_carried[ti] {
            return err(format!(
                "{}: the program states `ktdp.construct_indirect_access_tile` #{} over t{} indexed \
                 by t{}, and no op this walk lowered reads its gathered load — so every descriptor \
                 emitted would read the table at its BASE and the index buffer would never be \
                 consulted. That is the right shape and the wrong rows, from a clean bake, and \
                 nothing downstream compares the two. The gathered value must be an input of an op \
                 this door lowers.",
                f.name, ti, g.value_tid, g.index_tid,
            ));
        }
    }
    Ok(out)
}

/// WHAT ONLY THE WALK CAN STATE about one op. Threaded rather than re-derived, because [`emit_one`]
/// holds the REGIONS and not the ops — re-deriving any of these there would mean re-walking the
/// chain without them. They ride together as one value because they are one op's row in the walk.
///
/// ⛔ DESTRUCTURED WITHOUT `..` at the head of [`emit_one`], so a fourth proof added here and not
/// carried through is an E0027 at that line rather than a field nobody reads.
struct WalkProof<'a> {
    /// The B orientation the WALK proved from this op's own `indexing_maps` — `Some` for exactly a
    /// `Node(Program::Matmul)`, which is the only arm that reads it.
    b_orient: Option<BOrient>,
    /// The gather THIS op reads, when the one the program states is over one of its inputs — the
    /// walk's own join. `None` for every op of every program that states no
    /// `ktdp.construct_indirect_access_tile`, which is every fixture but the embedding, and then the
    /// arms below are byte-identical to what they were.
    gather: Option<Gather>,
    /// The per-input broadcast axes the WALK proved from this op's own rank-plumbing chain (see
    /// [`program_broadcast_chains`]) — positional with `per_op`'s inputs, empty when no operand is
    /// broadcast.
    bcast: &'a [Option<BcastAxis>],
}

/// Hand ONE op's regions to the body its `Program` names.
///
/// Every body here is unchanged and unaware it is being called per-op rather than per-function:
/// `split_out` filters the list by `is_out`, and this list holds exactly one output and that op's
/// inputs, so its arity check passes for the reason it was written to.
fn emit_one(
    name: &str,
    program: Lowering,
    per_op: &[Region],
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
    proof: WalkProof<'_>,
) -> Result<Vec<super::EmittedOp>, Error> {
    let WalkProof {
        b_orient,
        gather,
        bcast,
    } = proof;
    // THE FUSED SILU. Reached only through [`program_silu_mul_chains`], which PROVED the five-op
    // longhand; `per_op` is `[gate, up, out]`, which is the parameter order `silumul`'s own
    // `split_out(.., 2)` reads.
    let program = match program {
        Lowering::Silu => {
            return super::lower_ktir_to_superdsc::silumul(name, per_op, sym_id_base, layout);
        }
        // A BARE ROW REDUCTION. The entry point refuses a multi-row `max` by name — see its doc for
        // the measured device defect that makes that a correctness matter rather than a limitation.
        Lowering::Reduce(kind) => {
            return super::lower_ktir_to_superdsc::reduce(name, kind, per_op, sym_id_base, layout);
        }
        // `tile * <constant>`, with the multiplier this op's own — `scalarmul_at` is the shared body
        // the per-`Program` door reaches through `scalarmul`, so the two cannot drift about what a
        // scalar multiply emits.
        Lowering::ScalarMul(scale) => {
            // ⭐ THE PROGRAM'S OWN GATHER, THREADED. It was `None` with a note saying this door
            // "REFUSES an indirect access tile by name before any op is built" — true of
            // `region_for_operand` as it then was, and the reason the whole-function door could not
            // bake a Triton embedding while the per-`Program` door could. Both doors now reach the
            // SAME body with the SAME `gather_of` reading, so they cannot drift about what a gathered
            // scalar multiply emits.
            return super::lower_ktir_to_superdsc::scalarmul_at(
                name,
                scale,
                gather,
                per_op,
                sym_id_base,
                layout,
            );
        }
        // RUNG 3: the same body, with the multiplier's ADDRESS named by binding rather than looked
        // up by value. `scalarmul_bound` resolves the slot by tid, so a launch that binds a
        // different value than bake time saw still multiplies by what IT bound — which is the
        // whole point of the rung.
        Lowering::BoundScalarMul(b) => {
            return super::lower_ktir_to_superdsc::scalarmul_bound(
                name,
                &b,
                gather,
                per_op,
                sym_id_base,
                layout,
            );
        }
        // THE FUSED RMSNORM. `per_op` is `[x, gamma, out]`, which is `rmsnorm_at`'s own
        // `split_out(.., 2)` order. The epsilon is THIS chain's, read at its own `arith.addf`, because
        // `program_rmsnorm_eps` requires one root per FUNCTION and a decoder layer has two.
        Lowering::RmsNorm(eps) => {
            return super::lower_ktir_to_superdsc::rmsnorm_at(
                name,
                eps,
                per_op,
                sym_id_base,
                layout,
            );
        }
        Lowering::Node(p) => p,
    };
    Ok(match program {
        Program::Matmul => {
            // The fp8 activation-quantize dedup set is per-BUNDLE. One op at a time here, so a
            // fresh set is correct for an fp16 program; an fp8 one needs it threaded across ops and
            // that is not yet done, so it is stated rather than silently per-op.
            let mut q = std::collections::HashSet::new();
            // ⛔ THE ORIENTATION IS PROVEN, NEVER DEFAULTED. `b_orient` is `Some` exactly when
            // `program` is `Node(Program::Matmul)`, which is this arm — but a default here would be
            // the silent wrong contraction at `k == n`, so the impossible case refuses by name.
            let Some(b) = b_orient else {
                return err(format!(
                    "{name}: a `linalg.matmul` reached the matmul assembler with no proven B \
                     orientation. The framing of its W region (`[n, k]` for transpose-B, `[k, n]` for \
                     plain) is what the extent guards read, and at `k == n` neither framing can be \
                     recovered from the extents, so it is refused rather than assumed"
                ));
            };
            super::lower_ktir_to_superdsc::matmul_oriented(
                name,
                per_op,
                sym_id_base,
                layout,
                &mut q,
                b,
                // This door's operands are WINDOWS of the caller's parameters — the fact the
                // spurious-pad drop in `matmul_oriented` discriminates on.
                true,
            )?
        }
        Program::Elementwise(e) => {
            super::lower_ktir_to_superdsc::elementwise(name, e, per_op, bcast, sym_id_base, layout)?
        }
        Program::Transpose => {
            super::lower_ktir_to_superdsc::transpose(name, per_op, sym_id_base, layout)?
        }
        other => {
            return err(format!(
                "{name}: `{other:?}` is not reachable from this door — it is a fused kind the \
                 producer states by calling its entry point directly"
            ));
        }
    })
}

/// ⭐⭐⭐⭐⭐ RUNG 4 — THE GATHERED MATMUL'S **B** MATERIALIZES THROUGH A KERNEL-LESS COPY.
///
/// # THE PROBLEM, AS THE VENDOR'S OWN EMITTER STATES IT
///
/// A matmul's operand goes to `Stk::<KernelTag>::kernel(k, n)`, and a KERNEL-stick operand makes
/// the dxp reuse explorer run: `L3DlOpsScheduler.cpp:1550` gates it on `isReuse` and
/// `hasDimensionReuse` (`:303-321`) is `primaryDsInfo_.size() > 1 && count(KERNEL)`.
/// `calculateFlopPerByte` (`:2334`/`:2337`) then demands an LX allocate node for every
/// HBM-pinned labeledDs, which `allocAllMem` never gives an index — a BUILD-TIME `Err`
/// (this crate's `gather-on-KERNEL` guard, `emit/mod.rs`). So the gather cannot sit on the matmul
/// itself, on this arch, from any producer.
///
/// # THE SHAPE THE VENDOR'S OWN FIXTURES AND IBM'S PAGED ATTENTION BOTH SPELL
///
/// A KERNEL-less elementwise copy that carries the index, its RESULT fed to the matmul:
/// `dxp/test/test_gather_1core/sdsc_1.json` is an ordinary `identity` carrying one extra
/// `labeledDs_` and one extra `computeOp_` field, and IBM's paged attention gathers through an
/// `AddZero` — the same trick with a zero addend. The copy has one `primaryDsInfo_` entry (no
/// KERNEL among the operands), so the reuse explorer never runs for it, and [`Gather`] rides it
/// through `assemble_pointwise_broadcast_gather` — the one body that declares an index operand.
///
/// # WHAT THIS FUNCTION DOES
///
/// For a `linalg.matmul` whose B is the program's gathered table (the join `op_gather` already
/// proved — B's region `tid == g.value_tid`):
///
/// 1. checks the program's own arithmetic stays the plain gathered copy the fixtures spell — the
///    matmul's SECOND operand resolves to the table's region, and the ONLY other consumer of the
///    gathered load is the matmul itself (the counts gate in `dot_to_linalg` already refused a
///    second dot or a stray direct load at the Triton door; this is the same seal at this one);
/// 2. mints the intermediate `[entries, head]` — the gather's own row count from the indirect
///    tile's `Shape`, the head width from the table view's own columns — through `layout.synth`
///    at a tid above every bound tid, the same mint the walk uses for its own unstored results;
/// 3. emits the gathered identity copy into it: one input (`In::full().ew()` over the table),
///    `op_func: "identity"`, the index operand `attach_gather_index` declares, one leg per
///    index stick — the whole `assemble_pointwise_broadcast_gather` body, byte for byte the
///    descriptor the embedding's card-proven emission builds;
/// 4. REWRITES the matmul's B to the intermediate, as PLAIN-B `[k, n]`: the copy's output is a
///    stick-major `[entries, head]` activation and "`RowBlocked` vs `Kernel` is the identical
///    stick-blocked formula", so the contraction is done where the rows lie — the same argument
///    the walk's own plain-B note states for a producing matmul's output.
///
/// Returns `true` when it did the work, so the caller's refusal stands down for exactly this op.
/// `false` is not an error: a matmul whose B is NOT the gathered table still falls to the
/// fail-closed refusal below, with its own name in the message.
///
/// The emission's mutable scratch (`sym_id_base`, `out`, `next_tid`, `inter`) travels in
/// [`GatherEmit`] rather than as four more arguments — the walk owns one of these, threading it
/// to every materialization.
struct GatherEmit<'a> {
    sym_id_base: &'a mut i64,
    out: &'a mut Vec<super::EmittedOp>,
    next_tid: &'a mut u32,
    inter: &'a mut std::collections::HashMap<Ssa, Region>,
}

fn gathered_matmul_materializes(
    k: &KtirNode,
    layout: Option<&BundleLayout>,
    emit: &mut GatherEmit<'_>,
    g: Gather,
    tile_idx: usize,
    gathered_input: usize,
    in_values: &[Ssa],
) -> Result<bool, Error> {
    let GatherEmit {
        sym_id_base,
        out,
        next_tid,
        inter,
    } = emit;
    let f = &k.func;
    // ⛔ THE GATHERED OPERAND MUST BE THE MATMUL'S **B** (input 1) — rung 4's own rule, unchanged
    // by the multi-tile reading. A gather feeding **A** is not this exception's shape: the
    // materialization below mints a `[k, n]` intermediate for a contraction's B leg, and an
    // A-side gather (a row-gathered activation) is a different program the walk's own fail-closed
    // refusal owns. Returning `false` here hands it back to that refusal, whose message names the
    // arch reason — the control in `zz_the_gathered_matmul_materializes` pins this exact path.
    if gathered_input != 1 {
        return Ok(false);
    }
    // ⛔ AND B MUST BE THE GATHERED **LOAD**, proven by identity: the matmul's gathered input IS the
    // value this tile's load produced. A TID comparison cannot prove this — a plain load of the
    // same table resolves to the same tid, and a program whose gather feeds A while B reads the
    // table directly would otherwise materialize an intermediate for a B that is not the gathered
    // value (and then fail looking it up in `inter`, naming the wrong cause). A gather feeding A,
    // or a transposed K leg (B through a `tt.trans` — refused at the Triton door with the arch
    // reason), never matches; the walk's own refusal names them.
    let Some(b_value) = in_values.get(gathered_input).copied() else {
        return Ok(false);
    };
    // ⭐⭐⭐ THIS TILE'S LOAD — the `ti`-th `ktdp.construct_indirect_access_tile` and the
    // `ktdp.load` over it, enumerated in the SAME order `gathered_loads` numbered them, so the
    // walk's join and this materializer cannot disagree about which trip's gather this is. The
    // single-tile reading found "the" tile with `find`; a multi-trip sweep has one per trip and
    // only the identity of the load this matmul reads picks the right one.
    let tiles: Vec<&Operation<'_>> = f
        .operations
        .iter()
        .filter(|o| o.op_type == OpKind::KtdpConstructIndirectAccessTile)
        .collect();
    let Some(indirect) = tiles.get(tile_idx) else {
        return err(format!(
            "{}: the walk proved input {gathered_input} reads gather #{tile_idx}, but the program \
             states {} indirect access tile(s) — the walk and the materializer disagree about the \
             program",
            f.name,
            tiles.len(),
        ));
    };
    let Some(indirect_result) = indirect.result else {
        return Ok(false);
    };
    let gathered_result = f
        .operations
        .iter()
        .find(|o| o.op_type == OpKind::KtdpLoad && o.operands.first() == Some(&indirect_result))
        .and_then(|o| o.result);
    let Some(gathered_result) = gathered_result else {
        return Ok(false);
    };
    // ⛔ AND THE MATMUL'S GATHERED INPUT IS THAT VALUE — the identity check the tid check cannot
    // make.
    if b_value != gathered_result {
        return Ok(false);
    }
    let consumers = f
        .operations
        .iter()
        .filter(|o| o.operands.contains(&gathered_result))
        .count();
    if consumers != 1 {
        return err(format!(
            "{}: the gathered rows t{} have {} consumer(s) in this function — the materialized \
             copy this door emits for a gathered contraction stands for exactly one, and a second \
             consumer needs its own node. Split the program so each gathered read is its own op.",
            f.name, gathered_result.0, consumers,
        ));
    }
    // ⛔ A `BundleLayout` IS REQUIRED — the mint is the second party that makes the intermediate's
    // buffer real (`layout: None` is the unit-test arm and cannot place one).
    let Some(layout) = layout else {
        return err(format!(
            "{}: a gathered matmul's B needs a `BundleLayout` to hold its materialized \
             intermediate -- `layout: None` is the unit-test arm and cannot place one",
            f.name
        ));
    };
    // ── (2) THE INTERMEDIATE, minted the same way the walk mints its own unstored results.
    //
    // Rows: the indirect tile's own `Shape` — the program's statement of how many rows are
    // gathered, which `gather_of` read as `entries`. Cols: the head width the contraction itself
    // states — B's N, which is both the gather's per-row width and the matmul's output width.
    let Some((rows, head)) = super::lower_ktir_to_superdsc::shape_2d(indirect) else {
        return err(format!(
            "{}: the indirect access tile states no 2-D `shape`, so the gathered rows' footprint \
             is unknown and no intermediate can be sized for them",
            f.name
        ));
    };
    let tid = **next_tid;
    **next_tid += 1;
    layout.synth(crate::place::PlaceId::Act(tid), &[rows, head]);
    let r = Region {
        tid,
        v_rows: rows,
        v_cols: head,
        r_start: 0,
        c_start: 0,
        r_len: rows,
        c_len: head,
        r_cover: (0, rows),
        is_out: true,
        is_fp8: false,
    };
    // ── (3) THE GATHERED IDENTITY COPY — the `test_gather_1core` shape, through the one body
    // that declares an index operand. One input (the table), `identity`, the intermediate as
    // output; `assemble_pointwise_broadcast_gather` owns the one-index-stick ceiling and the
    // per-stick legs, so nothing here counts entries.
    let table_name = crate::place::act_name(g.value_tid);
    let index_name = crate::place::act_name(g.index_tid);
    let inter_name = r.name();
    let x_h = super::rbo(&table_name);
    let inputs = [super::In::full(&x_h).ew()];
    let tile_op = super::lower_ktir_to_superdsc::pointwise_tile_op(rows, head, 2);
    let copied = super::assemble_pointwise_broadcast_gather(
        super::PointwiseGather {
            op_name: &inter_name,
            tile_op: &tile_op,
            op_func: "identity",
            rows,
            cols: head,
            inputs: &inputs,
            gathered_input: 0,
            index_name: &index_name,
            first_entry: g.first_entry,
            o: &super::rbo(&inter_name),
        },
        sym_id_base,
        Some(layout),
    )
    .map_err(Error::from)?;
    out.extend(copied);
    // ── (4) THE MATMUL'S B, REWRITTEN TO THE INTERMEDIATE — recorded under the GATHERED VALUE
    // so the walk's `inter` lookup hands the matmul the intermediate's region for that operand,
    // and the caller REPLACES `per_op`'s B slot with it before `emit_one` runs.
    inter.insert(gathered_result, r);
    Ok(true)
}

/// THE SILU RECOGNISER'S FAIL-CLOSED HALF, which no fixture exercises.
///
/// `swiglu_mlp.py` produces the chain that MATCHES, so the shipped configurations only ever prove the
/// accepting path. The refusals are the part that keeps a near-miss from being lowered as
/// `OpFunc::Silu` — a well-formed descriptor computing a different function — so they are the part
/// that has to be tested deliberately. Each case below is one link of the chain broken, and the
/// MATCHING chain is the control that keeps a broken builder from passing them all vacuously.
/// THE BROADCAST RECOGNISER'S FAIL-CLOSED HALF, and every refusal has a control that isolates it.
///
/// ⭐⭐⭐ A TEST SET THAT ONLY ACCEPTS IS SATISFIED BY A PREDICATE THAT ACCEPTS EVERYTHING. This door
/// turned three refusals into support, so each thing it still refuses gets a case that FAILS without
/// that specific guard, and each accepting case is paired with the refusal one variable away from it:
/// the axis (`Col` vs `Mb`) is moved alone, the degenerate dim is removed alone, and the two
/// independent statements of the axis are made to disagree alone. Without that, "supports broadcasts"
/// and "stopped checking" are the same suite.
#[cfg(test)]
mod broadcast_chain_tests {
    use super::*;
    use ktir_core::arena::Arena;
    use ktir_core::ir::Operation;
    use ktir_core::irtype::IrType;

    /// WHAT THE PRODUCER ACTUALLY EMITS, from `KTIR_DUMP=1` on `decoder_block.py`:
    ///
    /// ```text
    /// %461 = TensorExpandShape(%0)   TargetShape=[64, 1]  Dimensions=Int(1)
    /// %761 = TensorCollapseShape(%461) TargetShape=[64]
    /// %762 = TensorEmpty()
    /// %462 = LinalgBroadcast(%761, %762) Dimensions=[1]
    /// ```
    ///
    /// `target` is the expand's `TargetShape` and `bdims` the broadcast's `Dimensions`, so a case can
    /// move either one alone. `bdims: None` drops the attribute entirely; `via_expand: false` wires the
    /// broadcast straight to the source so the degenerate-expand requirement is what fires.
    fn chain(
        target: &'static [i64],
        bdims: Option<&'static [i64]>,
        via_expand: bool,
    ) -> IRFunction<'static> {
        let a: &'static Arena = Arena::global();
        let (src, exp, col, emp, bc) = (Ssa(0), Ssa(1), Ssa(2), Ssa(3), Ssa(4));
        let mut ops = vec![];
        if via_expand {
            ops.push(
                Operation::new(a, Some(exp), OpKind::TensorExpandShape, &[src]).with_attr(
                    a,
                    AttrKey::TargetShape,
                    Attr::IntList(target),
                ),
            );
            ops.push(
                Operation::new(a, Some(col), OpKind::TensorCollapseShape, &[exp]).with_attr(
                    a,
                    AttrKey::TargetShape,
                    Attr::IntList(&[64]),
                ),
            );
        }
        ops.push(Operation::new(a, Some(emp), OpKind::TensorEmpty, &[]));
        let source = if via_expand { col } else { src };
        let mut b = Operation::new(a, Some(bc), OpKind::LinalgBroadcast, &[source, emp]);
        if let Some(d) = bdims {
            b = b.with_attr(a, AttrKey::Dimensions, Attr::IntList(d));
        }
        ops.push(b);
        IRFunction {
            name: "bcast_probe",
            arguments: a.args(vec![(src, IrType::Index)]),
            operations: a.ops(ops),
            grid: (1, 1, 1),
            return_type: None,
        }
    }

    fn refusal(target: &'static [i64], bdims: Option<&'static [i64]>, via: bool) -> String {
        program_broadcast_chains(&chain(target, bdims, via))
            .err()
            .map(|e| e.message)
            .unwrap_or_else(|| panic!("expected a refusal for target={target:?} bdims={bdims:?}"))
    }

    /// ⭐ CONTROL (Col) — THE DECODER'S `m[:, None]`. `[64] → [64, 1] → [64, 64]`, the shape
    /// `decoder_block.py`'s softmax states twice per layer. Without this the refusals below could all be
    /// a broken walk rather than working guards.
    #[test]
    fn the_col_chain_is_recognised_and_resolves_to_its_source() {
        let f = chain(&[64, 1], Some(&[1]), true);
        let (map, consumed) = program_broadcast_chains(&f).expect("the softmax's own chain");
        let b = map
            .get(&Ssa(4))
            .expect("the broadcast's result is what a consumer names");
        assert_eq!(
            b.axis,
            BcastAxis::Col,
            "`Dimensions=[1]` sprays along the stick axis"
        );
        assert_eq!(
            b.src,
            Ssa(0),
            "the consumer must read the rank-2 SOURCE, not the chain"
        );
        for v in [Ssa(1), Ssa(2), Ssa(4)] {
            assert!(
                consumed.contains(&v),
                "{v:?} is plumbing and must be lowered by nothing"
            );
        }
    }

    /// ⭐ CONTROL (Mb) — THE RMSNORM GAIN'S `n1[None, :]`, `TargetShape=[1, 128]` / `Dimensions=[0]`.
    /// This is the case that makes `BcastAxis` a variant rather than a bool: BOTH axes really occur in
    /// one decoder layer, so a bool would be a coin flip on real data.
    #[test]
    fn the_mb_chain_is_recognised_with_the_other_axis() {
        let f = chain(&[1, 128], Some(&[0]), true);
        let (map, _) = program_broadcast_chains(&f).expect("the rmsnorm gain's own chain");
        assert_eq!(
            map[&Ssa(4)].axis,
            BcastAxis::Mb,
            "`Dimensions=[0]` sprays down the rows"
        );
    }

    /// ⛔ THE CROSS-CHECK, AND IT IS THE ONE GUARD NO SINGLE ATTRIBUTE CAN PROVIDE. The expand says the
    /// LEADING dim is degenerate (`[1, 128]`, a row vector) while the broadcast says it sprays along the
    /// TRAILING axis (`Dimensions=[1]`, a per-row scalar). Each attribute is individually well-formed;
    /// they describe different operands. Believing either one alone emits an operand mode for a vector
    /// that lies the other way, so neither is believed.
    #[test]
    fn an_axis_the_two_attributes_disagree_about_is_refused() {
        let m = refusal(&[1, 128], Some(&[1]), true);
        assert!(
            m.contains("whose trailing axis is not degenerate"),
            "names WHICH axis is not degenerate — the expand's, not the broadcast's: {m}"
        );
        assert!(
            m.contains("sprays along Col"),
            "and names the axis the broadcast claimed, so the disagreement is legible: {m}"
        );
    }

    /// ⛔ A NON-DEGENERATE EXPAND IS A RELAYOUT, NOT A BROADCAST — the guard the reshape case needs.
    /// `[64, 128]` inserts nothing degenerate, so the source is a full tile; addressing it as a vector
    /// would read one column and spray it, with no shape error anywhere.
    #[test]
    fn a_non_degenerate_expand_is_refused_as_a_relayout() {
        let m = refusal(&[64, 128], Some(&[1]), true);
        assert!(
            m.contains("A general reshape moves real elements"),
            "says WHY a general reshape is different in kind, not just that it was rejected: {m}"
        );
    }

    /// ⛔ AN AXIS THAT IS NEITHER OF THE TWO OPERAND MODES. A two-axis spray from a scalar is
    /// `In::scalar`, a third mode, and this door does not silently fold it into one of the two it emits.
    #[test]
    fn a_two_axis_broadcast_is_refused_by_name() {
        let m = refusal(&[64, 1], Some(&[0, 1]), true);
        assert!(
            m.contains("In::col") && m.contains("In::mb"),
            "names both modes it does emit: {m}"
        );
    }

    /// ⛔ NO `Dimensions` AT ALL — then which axis it sprays is simply unknown, and picking one would be
    /// choosing an operand mode for the producer.
    #[test]
    fn a_broadcast_with_no_dimensions_attribute_is_refused() {
        let m = refusal(&[64, 1], None, true);
        assert!(
            m.contains("no `Dimensions`"),
            "names the missing attribute: {m}"
        );
    }

    /// ⛔ AND THE DEGENERATE EXPAND IS REQUIRED, NOT MERELY PREFERRED: with the broadcast wired straight
    /// to its source there is nothing in the program saying the source is a vector, so its extent is
    /// unproven. This is the case that fails if the walk ever accepts a chain it did not verify.
    #[test]
    fn a_broadcast_not_reached_through_a_degenerate_expand_is_refused() {
        let m = refusal(&[64, 1], Some(&[1]), false);
        assert!(
            m.contains("DEGENERATE dim"),
            "says what is missing and why it is load-bearing: {m}"
        );
    }
}

/// WHICH CONTRACTION FORM a `linalg.matmul` states, and the two this door lowers.
#[cfg(test)]
mod matmul_orientation_tests {
    use super::*;
    use ktir_core::affine::{AffineExpr, AffineMap};
    use ktir_core::arena::Arena;
    use ktir_core::ir::Operation;

    /// A `linalg.matmul` whose `indexing_maps` are the given dim triples, or NO maps when `None`.
    fn mm(
        maps: Option<[[usize; 2]; 3]>,
    ) -> (&'static IRFunction<'static>, &'static Operation<'static>) {
        let a: &'static Arena = Arena::global();
        let mut op = Operation::new(
            a,
            Some(Ssa(9)),
            OpKind::LinalgMatmul,
            &[Ssa(0), Ssa(1), Ssa(2)],
        );
        if let Some(m) = maps {
            let list: Vec<AffineMap<'static>> = m
                .iter()
                .map(|pair| AffineMap {
                    num_dims: 3,
                    num_syms: 0,
                    exprs: a.exprs(pair.iter().map(|d| AffineExpr::Dim(*d)).collect()),
                })
                .collect();
            op = op.with_attr(a, AttrKey::IndexingMaps, Attr::AffineMapList(a.maps(list)));
        }
        let f = IRFunction {
            name: "mm_probe",
            arguments: a.args(vec![]),
            operations: a.ops(vec![op]),
            grid: (1, 1, 1),
            return_type: None,
        };
        (Box::leak(Box::new(f)), Box::leak(Box::new(op)))
    }

    /// ⭐ THE PRESENTED-WEIGHT FORM: B's map ends in the reduction dim `d2`, so B's REGION is `[n, k]`
    /// and the contraction reduces over k in place. scratchy's `KtirFunc::matmul` emits this
    /// unconditionally, and the host stage is where such a weight's bytes are placed into the kernel
    /// slot's `[k, n]` device order.
    #[test]
    fn the_transpose_b_triple_is_recognised() {
        let (f, op) = mm(Some([[0, 2], [1, 2], [0, 1]]));
        assert_eq!(
            matmul_b_orientation(f, op).expect("the assumed form"),
            BOrient::TransposeB
        );
    }

    /// ⭐ MLIR'S PLAIN FORM, and the SAME answer for an op stating NO maps — which is what `tl.dot(p, v)`
    /// with no `.T` lowers to, and the reason both decoders reached this door at all. It is a real
    /// `[k, n]` buffer, which IS the kernel slot's own device order, so it is contracted where it lies.
    #[test]
    fn the_plain_triple_and_a_missing_attribute_are_both_plain_b() {
        let (f, op) = mm(Some([[0, 2], [2, 1], [0, 1]]));
        assert_eq!(
            matmul_b_orientation(f, op).expect("plain B"),
            BOrient::PlainB
        );
        let (f2, op2) = mm(None);
        assert_eq!(
            matmul_b_orientation(f2, op2).expect("no maps IS the plain default"),
            BOrient::PlainB,
            "MLIR's default for `linalg.matmul` is the plain form; treating a missing attribute as the \
             ASSUMED form would silently contract the other way round"
        );
    }

    /// ⭐ THE ACCEPTING CASE FOR [`transposed_b_is_placeable`], AND IT IS THE WHOLE REGRESSION BAR:
    /// a transpose-B weight that IS parameter-backed. `rmsnorm_granite`, both `swiglu_mlp_*` and
    /// every projection in both decoders are this case, measured byte-identical, and they are correct
    /// because the HOST stage places the bytes (`stage_2d(&StickLayout::kernel(k, n), ..)`). If this
    /// case ever refused, the guard below would be indistinguishable from "refuse every `.T`".
    #[test]
    fn a_transpose_b_over_a_presented_weight_is_placeable() {
        transposed_b_is_placeable("probe", BOrient::TransposeB, true)
            .expect("a presented weight's transposition is spent by the host stage");
    }

    /// ⛔ THE REFUSAL, **ONE VARIABLE** FROM THE CASE ABOVE — the provenance of B and nothing else.
    /// This is the decoders' two score matmuls (`tl.dot(q1r, k1r.T)` over RoPE'd `[64, 64]` tiles):
    /// the maps say `[n, k]`, the kernel slot reads `[k, n]`, and no agent exists to move the bytes,
    /// so lowering it emits a well-formed descriptor contracting `q1r @ k1r`. Delete the guard and
    /// this test fails while every other test in the crate still passes — which is the point.
    #[test]
    fn a_transpose_b_over_a_computed_value_is_refused_by_name() {
        let e = transposed_b_is_placeable("probe", BOrient::TransposeB, false)
            .expect_err("an in-register value has no host stage and no cache write");
        assert!(
            e.message.contains("COMPUTED value"),
            "names WHAT is wrong with the operand, not merely that it was rejected: {}",
            e.message
        );
        assert!(
            e.message.contains("k == n"),
            "and says why no extent guard downstream can catch it: {}",
            e.message
        );
        assert!(
            e.message.contains("Program::Transpose"),
            "and names what the producer must emit instead, so the refusal is actionable: {}",
            e.message
        );
    }

    /// ⛔ THE OTHER SINGLE-VARIABLE CONTROL: the ORIENTATION moved alone, provenance held at
    /// COMPUTED. `tl.dot(p, v)` — `v` is this kernel's own V projection, an in-register `[k, n]`
    /// tile, and `[k, n]` IS the slot's device order, so it is contracted WHERE IT LIES. That is the
    /// leg the shipped attention runs against its V cache at 41 tok/s (`vcache_write_offset`: "the
    /// kernel is `[k=cap, n=hd]`"). A guard that refused every computed B would break it, and both
    /// decoders would stop lowering for the wrong reason.
    #[test]
    fn a_plain_b_over_a_computed_value_is_contracted_where_it_lies() {
        transposed_b_is_placeable("probe", BOrient::PlainB, false)
            .expect("`p @ v`: a computed `[k, n]` tile is already in the kernel slot's own order");
    }

    /// ⛔ THE CONTROL THAT KEEPS THE TWO ABOVE HONEST: a triple that is NEITHER form — here C permuted
    /// to `[d1, d0]` — states an iteration order no assembler walks, and the extent guards downstream
    /// cannot catch it when k == n. Without this case, `matmul_b_orientation` could return `PlainB` for
    /// everything it does not recognise and every test above would still pass.
    #[test]
    fn a_triple_that_is_neither_form_is_refused_by_name() {
        let (f, op) = mm(Some([[0, 2], [1, 2], [1, 0]]));
        let e = matmul_b_orientation(f, op).expect_err("a permuted C is not either form");
        assert!(
            e.message.contains("NEITHER") && e.message.contains("k == n"),
            "names that it is neither, and why an extent guard cannot catch it: {}",
            e.message
        );
    }
}

#[cfg(test)]
mod silu_chain_tests {
    use super::*;
    use ktir_core::arena::Arena;
    use ktir_core::ir::Operation;
    use ktir_core::irtype::IrType;

    /// `g` is `%0` and `u` is `%1` — the two values the chain consumes. What varies per case is the
    /// four ops between them.
    ///
    /// `add_const` is what the `arith.addf` adds (1.0 in a real silu); `div_num` selects the divide's
    /// numerator — [`Num::Gate`] is what makes it silu, [`Num::Up`] is a value outside the chain (so
    /// the numerator check is what fires), and [`Num::Exp`] reuses a chain intermediate (so the
    /// destroyed-value check is what fires); `terminal_mul` drops the multiply entirely.
    #[derive(Clone, Copy)]
    enum Num {
        Gate,
        Up,
        Exp,
    }

    fn chain(add_const: f64, div_num: Num, terminal_mul: bool) -> IRFunction<'static> {
        let a: &'static Arena = Arena::global();
        let (g, u, c1, sp) = (Ssa(0), Ssa(1), Ssa(2), Ssa(3));
        let (neg, exp, add, div, mul) = (Ssa(4), Ssa(5), Ssa(6), Ssa(7), Ssa(8));
        let mut ops = vec![
            Operation::new(a, Some(c1), OpKind::ArithConstant, &[]).with_attr(
                a,
                AttrKey::Value,
                Attr::Float(add_const),
            ),
            Operation::new(a, Some(sp), OpKind::TensorSplat, &[c1]),
            Operation::new(a, Some(neg), OpKind::ArithNegf, &[g]),
            Operation::new(a, Some(exp), OpKind::MathExp, &[neg]),
            Operation::new(a, Some(add), OpKind::ArithAddf, &[exp, sp]),
            {
                let num = match div_num {
                    Num::Gate => g,
                    Num::Up => u,
                    Num::Exp => exp,
                };
                Operation::new(a, Some(div), OpKind::ArithDivf, &[num, add])
            },
        ];
        if terminal_mul {
            ops.push(Operation::new(a, Some(mul), OpKind::ArithMulf, &[div, u]));
        }
        IRFunction {
            name: "silu_probe",
            arguments: a.args(vec![(g, IrType::Index), (u, IrType::Index)]),
            operations: a.ops(ops),
            grid: (1, 1, 1),
            return_type: None,
        }
    }

    /// THE CONTROL. Without this passing, every refusal below could be a broken builder rather than a
    /// working guard.
    #[test]
    fn the_matching_chain_is_recognised_as_one_fused_program() {
        let f = chain(1.0, Num::Gate, true);
        let cs = program_silu_mul_chains(&f).expect("the matching chain must be recognised");
        assert_eq!(cs.len(), 1, "one `arith.negf` heads one chain");
        let c = cs[0];
        assert_eq!(c.gate, Ssa(0), "the gate is the value the negate consumed");
        assert_eq!(c.up, Ssa(1), "`up` is the multiply's other operand");
        assert_eq!(
            c.mul,
            Ssa(8),
            "the terminal multiply's result is what the program produces"
        );
        assert_eq!(
            c.consumed,
            [Ssa(4), Ssa(5), Ssa(6), Ssa(7)],
            "negate/exp/add/divide are consumed by the fusion and must be lowered by nothing"
        );
    }

    /// ⛔ `sigmoid(x)` is `1/(1+e^-x)`. Adding anything but 1 is a DIFFERENT function, and lowering it
    /// as `OpFunc::Silu` would be a descriptor computing something the program does not state.
    #[test]
    fn a_constant_that_is_not_one_is_refused_and_the_value_is_named() {
        let e = program_silu_mul_chains(&chain(2.0, Num::Gate, true))
            .expect_err("1 + e^-x with a 2 in place of the 1 is not a sigmoid");
        assert!(
            e.message.contains("adds 2") && e.message.contains("1/(1+e^-x)"),
            "the refusal must name the constant it found and the identity it broke; got: {}",
            e.message
        );
    }

    /// ⛔⛔⛔ THE DISCRIMINATOR. `g/(1+e^-g)` is `g·sigmoid(g)` = silu(g). The same five op kinds in
    /// the same order with any other numerator are a different function, and this is the ONLY thing
    /// that tells them apart.
    #[test]
    fn a_divide_whose_numerator_is_not_the_negated_value_is_refused() {
        let e = program_silu_mul_chains(&chain(1.0, Num::Up, true))
            .expect_err("dividing something other than the gate by 1+e^-g is not a silu");
        assert!(
            e.message.contains("NUMERATOR"),
            "the refusal must name the numerator as the thing that failed; got: {}",
            e.message
        );
    }

    /// A bare silu with no multiply is not `Program::SiluMul`, and this door has no arm for a lone
    /// silu — so it is named rather than half-lowered.
    #[test]
    fn a_silu_with_no_multiply_is_refused_by_name() {
        let e = program_silu_mul_chains(&chain(1.0, Num::Gate, false))
            .expect_err("`SiluMul` is silu(gate)·up; a bare silu is not it");
        assert!(
            e.message.contains("silu(x)") || e.message.contains("nothing reads"),
            "the refusal must say the multiply is what is missing; got: {}",
            e.message
        );
    }

    /// ⛔⛔⛔ THE DESTROYED-VALUE CASE, which is the whole reason the reader walk has to be complete.
    /// Fusing five ops into one descriptor means `-g`, `e^-g`, `1+e^-g` and `silu(g)` are NEVER
    /// MATERIALISED. A second reader of any of them would read a value nothing writes — and would do
    /// so silently, since the program lowers and bakes. Here the divide reuses the exp's result, so
    /// `e^-g` has two readers.
    #[test]
    fn a_chain_intermediate_with_a_second_reader_is_refused_because_the_fusion_destroys_it() {
        let e = program_silu_mul_chains(&chain(1.0, Num::Exp, true))
            .expect_err("a chain intermediate read twice cannot be fused away");
        assert!(
            e.message.contains("read by 2 ops") && e.message.contains("DESTROYS"),
            "the refusal must say how many readers there are and that the fusion destroys the \
             value; got: {}",
            e.message
        );
    }

    /// A function with NO `arith.negf` claims no silu at all, and must not be an error.
    #[test]
    fn a_function_with_no_negate_yields_no_chain_and_no_error() {
        let a: &'static Arena = Arena::global();
        let f = IRFunction {
            name: "no_silu",
            arguments: a.args(vec![(Ssa(0), IrType::Index)]),
            operations: a.ops(vec![Operation::new(
                a,
                Some(Ssa(1)),
                OpKind::MathExp,
                &[Ssa(0)],
            )]),
            grid: (1, 1, 1),
            return_type: None,
        };
        assert!(
            program_silu_mul_chains(&f)
                .expect("no negate is not an error")
                .is_empty(),
            "a function with no `arith.negf` states no silu longhand"
        );
    }

    /// TWO chains in one function is a real configuration (`decoder_two_layers_fwd` has one silu per
    /// layer), not a refusal. Built as two independent chains over four values.
    #[test]
    fn two_independent_chains_in_one_function_are_both_recognised() {
        let a: &'static Arena = Arena::global();
        let mut ops = Vec::new();
        for i in 0..2u32 {
            let b = 100 * i;
            let (g, u, c1, sp) = (Ssa(b), Ssa(b + 1), Ssa(b + 2), Ssa(b + 3));
            let (neg, exp, add, div, mul) =
                (Ssa(b + 4), Ssa(b + 5), Ssa(b + 6), Ssa(b + 7), Ssa(b + 8));
            ops.extend([
                Operation::new(a, Some(c1), OpKind::ArithConstant, &[]).with_attr(
                    a,
                    AttrKey::Value,
                    Attr::Float(1.0),
                ),
                Operation::new(a, Some(sp), OpKind::TensorSplat, &[c1]),
                Operation::new(a, Some(neg), OpKind::ArithNegf, &[g]),
                Operation::new(a, Some(exp), OpKind::MathExp, &[neg]),
                Operation::new(a, Some(add), OpKind::ArithAddf, &[exp, sp]),
                Operation::new(a, Some(div), OpKind::ArithDivf, &[g, add]),
                Operation::new(a, Some(mul), OpKind::ArithMulf, &[div, u]),
            ]);
        }
        let f = IRFunction {
            name: "two_silus",
            arguments: a.args(vec![(Ssa(0), IrType::Index), (Ssa(100), IrType::Index)]),
            operations: a.ops(ops),
            grid: (1, 1, 1),
            return_type: None,
        };
        let cs = program_silu_mul_chains(&f).expect("two layers means two silus, not a refusal");
        assert_eq!(cs.len(), 2, "one chain per `arith.negf`");
        assert_eq!(
            (cs[0].gate, cs[1].gate),
            (Ssa(0), Ssa(100)),
            "each chain's gate is its own negate's operand"
        );
    }
}
