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
//! body's arity. That is exact for a function that IS one op, which is what the splice emits — one
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
use ktir_core::dtypes::DType;
use ktir_core::ir::{Attr, IRFunction, Operation, Ssa};
use ktir_core::opkind::OpKind;

use super::lower_ktir_to_superdsc::{Error, Gather, Region, err, gathers_of, regions};
use crate::ktir_node::{Elementwise, KtirNode, Program, ReduceKind};
// The broadcast AXIS is the emission door's own vocabulary — `elementwise` selects the `In` builder
// from it — so it lives beside that door and this recogniser names the same type rather than a
// parallel one that would have to be mapped across the boundary. The CONVERT KIND is the same
// trade for a dtype-convert: the emission door's own vocabulary, not a `Program` variant.
use super::lower_ktir_to_superdsc::{BcastAxis, ConvertKind, IntBinary, RowBand};
use crate::placement::BundleLayout;
use crate::superdsc_opspec::Df;
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
#[derive(Clone, Debug)]
pub struct SiluMulChain {
    /// The value the negate consumed AND the divide's numerator — silu's argument.
    pub gate: Ssa,
    /// The other factor of the terminal multiply — `None` for the BARE form, whose longhand ends
    /// at the STORED `silu(x)` with no multiply (`elementwise.py`'s `silu_fwd`), the unary
    /// `Elementwise(Silu)` the door's own elementwise arm lowers as `("silu", 1)`.
    pub up: Option<Ssa>,
    /// The terminal's result — the `up` multiply's for the fused form, the divide's OWN for the
    /// bare one. Either way it is the value the ONE fused op produces and the rest of the program
    /// consumes.
    pub mul: Ssa,
    /// `negf`, `exp`, `addf` (and `divf`, for the fused form) results — every intermediate the
    /// fused op never materialises, so [`lower_function`] must lower NOTHING for them. The
    /// terminal is deliberately not in here: it is the op that BECOMES the fused program, which
    /// is why the bare form's divide is absent exactly as the fused form's multiply is.
    pub consumed: Vec<Ssa>,
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
    /// the contraction reduces over k in place. scratchy's matmul producer emits this
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
         over k in place; that is the contract scratchy's matmul producer states at its own \
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
    /// The OnePlusScale gain offset, when the chain's gain leg is `(1 + gamma)` rather than
    /// `gamma`: ONE `arith.addf` over the loaded row with a splatted constant, consumed by the
    /// chain exactly as the rank plumbing is. `None` is the Scale convention (no addf); the only
    /// admitted value is 1.0 — `GainConvention::from_offset`'s own law, refused by name otherwise.
    /// Read per chain the way [`super::lower_ktir_to_superdsc::rmsnorm`]'s `program_gain_offset`
    /// reads it, so the two doors cannot state different formulas for one program.
    pub ones: Option<f32>,
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
    // ⭐ THE GAIN LEG'S OWN `(1 + w)` ADDF — the OnePlusScale convention (`rmsnorm.py`'s
    // `rmsnorm_oneplus_fwd`), consumed by the chain the same way the rank plumbing is. The
    // back-walk above stops AT the addf (it is not rank plumbing), and without this arm the addf
    // would reach the walk as an `Elementwise(Add)` with a splat operand — a form the walk's own
    // operand resolution refuses, so a oneplus program could not lower at all. ONE addf over the
    // loaded row with a splatted constant: the constant must be the convention's own 1.0
    // (`GainConvention::from_offset` admits 0.0 and 1.0 only, and 0.0 is stated as NO addf), and
    // anything else is a producer bug refused by name — the same law the per-`Program` door's
    // `program_gain_offset` arm states. An addf of another shape (no splat, two tensors) is not
    // this convention and is left for the operand resolution to name, exactly as before this arm
    // existed.
    let mut ones = None;
    if let Some(d) = def_of(gamma)
        && d.op_type == OpKind::ArithAddf
    {
        let mut splat = None;
        let mut row = None;
        for &s in d.operands.iter() {
            match splat_value(s) {
                Some(v) => splat = Some(v),
                None => row = Some(s),
            }
        }
        if let (Some(v), Some(r)) = (splat, row) {
            if v != 1.0 {
                return err(format!(
                    "{}: the rmsnorm's gain leg adds {v}, not 1.0 — `GainConvention::from_offset` \
                     admits 0.0 (Scale, no addf) and 1.0 (OnePlusScale) only, so a program \
                     stating any other offset is a producer bug",
                    f.name
                ));
            }
            ones = Some(1.0);
            consumed.push(gamma);
            gamma = r;
            // The row's own rank plumbing, if it states any — the same three-op walk as the
            // gain's, run on the addf's other side.
            for _ in 0..3 {
                let Some(dd) = def_of(gamma) else { break };
                if !matches!(
                    dd.op_type,
                    OpKind::TensorExpandShape
                        | OpKind::TensorCollapseShape
                        | OpKind::LinalgBroadcast
                ) {
                    break;
                }
                consumed.push(gamma);
                let Some(src) = dd.operands.first().copied() else {
                    break;
                };
                gamma = src;
            }
        }
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
    // The multiplier is `1/cols`, stated as the constant's f16 image — the f16 grid is the only one
    // an f16-typed program constant lands on, the same law the parity test's eps check states —
    // which equals exact `1/cols` exactly when cols is a power of two (the comparison is done in
    // f32 so a producer that splatted an f32 `1/N` still matches bit-for-bit). A non-power-of-two
    // width (gemma-4's 3840) can state only f16(1/cols), so both spellings are admitted; every
    // other multiplier differs from both by far more than an f16 ulp and is still refused below.
    let inv_cols = (1.0f64 / cols) as f32;
    let inv_cols_f16 =
        ktir_core::codec::f16_units_to_f32(&ktir_core::codec::f32_to_f16_units(&[inv_cols]))[0];
    let inv_d_f32 = inv_d as f32;
    if inv_d_f32 != inv_cols && inv_d_f32 != inv_cols_f16 {
        return err(format!(
            "{}: the rmsnorm's divisor is {inv_d}, but this tile reduces {cols} columns and a mean \
             needs 1/{cols} = {} (or its f16 image {}). `assemble_rmsnorm` binds `1/cols` itself at \
             the reserved `RMS_INVCOLS_TID`, so lowering this as an rmsnorm would substitute a \
             DIFFERENT constant for the one the program states.",
            f.name,
            1.0f64 / cols,
            inv_cols_f16
        ));
    }

    Ok(Some(RmsNormChain {
        x,
        gamma,
        out,
        eps: eps as f32,
        ones,
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
/// An empty `Vec` means no `arith.negf` is a silu head, i.e. nothing here claims to be a
/// silu. `Err` means one was found and the chain around it does NOT close — reported rather than
/// left to the 1:1 map, because a half-proven silu lowered op-by-op would read like a missing
/// `Elementwise` variant when the real answer is that the silu is mis-shaped.
///
/// ⭐ THE HEAD TEST, AND WHY IT IS THE READER: a negf is a silu head ONLY IF its one
/// reader is `math.exp` — the second op of `x·sigmoid(x)`'s longhand. A negf read by
/// anything else is not a mis-shaped silu but a NEGATION the program means (the Gumbel
/// transform's `g = -log(-log(u))` negates twice around the inner log), and the card has
/// its own opFunc for it (`OpFuncs::NEG`), so the walk lowers it as
/// `Elementwise(Neg)` rather than this fusion refusing it. Before `Neg` existed there
/// WAS no such program, so every negf had to justify itself here — that is no longer
/// true, and claiming all of them would make `-log(x)` unlowerable while `0-x` lowers.
/// The mis-shaped-silu report keeps its full force INSIDE the chain: once the head
/// test passes, every link must close or the function is refused.
pub fn program_silu_mul_chains(f: &IRFunction<'static>) -> Result<Vec<SiluMulChain>, Error> {
    let mut out = Vec::new();
    for neg in f
        .operations
        .iter()
        .filter(|o| o.op_type == OpKind::ArithNegf)
    {
        // THE HEAD TEST: exactly one reader, and it is `math.exp`. Anything else —
        // no reader, several readers, or one reader of any other kind — is not this
        // fusion's negf, and `silu_mul_chain_from` below is never entered for it.
        let Some(neg_v) = neg.result else { continue };
        let deep = f.ops_deep();
        let mut readers = deep.iter().copied().filter(|o| o.operands.contains(&neg_v));
        let head = readers.next();
        if readers.next().is_some() || head.is_none_or(|o| o.op_type != OpKind::MathExp) {
            continue;
        }
        // ⭐ AND THE `exp` FEEDS A SIGMOID'S `1 + e`. An exponential of a negation that no `addf`
        // reads is not a sigmoid at all but the program's own `exp(-x)` — spyre-inference's
        // online-softmax rescale `exp(-relu(scores_max - tile_max))`, read by the running sum and
        // the running output — and lowers as the two ops it states. One that does feed an `addf`
        // is claimed, and every link of its chain must close.
        let Some(exp_v) = head.and_then(|o| o.result) else {
            continue;
        };
        if !deep
            .iter()
            .any(|o| o.op_type == OpKind::ArithAddf && o.operands.contains(&exp_v))
        {
            continue;
        }
        out.push(silu_mul_chain_from(f, neg)?);
    }
    Ok(out)
}

/// A gathered table's `[rows, cols]` — the gather's tile's data view.
fn gather_table_dims(f: &IRFunction<'static>, g: &Gather) -> Result<(u32, u32), Error> {
    let tile = f.operations.iter().find(|o| o.result == Some(g.tile));
    let view = tile
        .and_then(|t| t.operands.first().copied())
        .and_then(|v| f.operations.iter().find(|o| o.result == Some(v)));
    let dims = view.and_then(|v| v.result_type).and_then(|t| t.dims());
    match dims {
        Some(&[r, c]) => Ok((r as u32, c as u32)),
        other => err(format!(
            "{}: a gathered table states a {other:?} view — a row gather reads a 2-D `[rows, cols]` \
             table",
            f.name
        )),
    }
}

/// `r`'s buffer read in the `[rows, cols]` frame of the same card bytes.
fn reframed(r: Region, [rows, cols]: [u32; 2]) -> Region {
    Region {
        v_rows: rows,
        v_cols: cols,
        r_start: 0,
        c_start: 0,
        r_len: rows,
        c_len: cols,
        r_cover: (0, rows),
        c_cover: (0, cols),
        ..r
    }
}

/// ⭐ THE HEAD-MAJOR PAIR at `op` — a reshape/transpose pair between a stick-major `[B, A·W]` block
/// and the head-major `[A·B, W]`: `(the pair's last result, the frame it is read in, W in sticks)`.
/// At `W` one stick the pair moves no byte (below); wider, it is [`HeadMajorCopies`].
///
/// A stick-major `[B, A·S]` buffer (`S` one stick) holds stick column `a` of row `b` at
/// `(a·B + b)·S` — head-major `[A, B, S]` row-major, byte for byte. So
///
/// * `reshape [B, A·S] → [B, A, S]` read only by `transpose [1, 0, 2] → [A, B, S]` is the source in
///   the `[A·B, S]` frame (`op` the reshape), and
/// * `transpose [1, 0, 2]` of `[A, B, S]` row-major `→ [B, A, S]` read only by
///   `reshape → [B, A·S]` is the source in the stick-major `[B, A·S]` frame (`op` the transpose).
///
/// `None` for anything else.
fn head_major_pair(
    f: &IRFunction<'static>,
    op: &ktir_core::ir::Operation<'static>,
    from: &[i64],
    to: &[i64],
) -> Option<(Ssa, [u32; 2], u32)> {
    let stick = i64::from(crate::work::FP16_ELEMS_PER_STICK);
    let res = op.result?;
    let deep = f.ops_deep();
    let sole_reader = |v: Ssa| {
        let mut rs = deep
            .iter()
            .copied()
            .filter(move |o| o.operands.contains(&v));
        let first = rs.next();
        if rs.next().is_some() { None } else { first }
    };
    let perm_102 = |o: &ktir_core::ir::Operation<'static>| {
        o.op_type == OpKind::LinalgTranspose
            && o.attributes.iter().any(|(k, v)| {
                matches!((k, v), (AttrKey::Permutation, Attr::IntList(p)) if *p == [1, 0, 2])
            })
    };
    let dims_of = |o: &ktir_core::ir::Operation<'static>| -> Option<Vec<i64>> {
        o.result_type.and_then(|t| t.dims()).map(<[i64]>::to_vec)
    };
    match op.op_type {
        OpKind::TensorReshape => {
            let ([b, aw], [b2, a, s]) = (from, to) else {
                return None;
            };
            if b != b2 || s % stick != 0 || a * s != *aw {
                return None;
            }
            let t = sole_reader(res).filter(|o| perm_102(o))?;
            if dims_of(t)? != [*a, *b, *s] {
                return None;
            }
            Some((t.result?, [(a * b) as u32, *s as u32], (s / stick) as u32))
        }
        OpKind::LinalgTranspose if perm_102(op) => {
            let [a, b, s] = from else {
                return None;
            };
            if s % stick != 0 || to != [*b, *a, *s] {
                return None;
            }
            let r = sole_reader(res).filter(|o| o.op_type == OpKind::TensorReshape)?;
            if dims_of(r)? != [*b, a * s] {
                return None;
            }
            Some((r.result?, [*b as u32, (a * s) as u32], (s / stick) as u32))
        }
        _ => None,
    }
}

/// ⭐ THE HEAD-MAJOR PAIR AT A HEAD `S > 1` STICKS WIDE — the stick-major `[B, A·S·stick]` block
/// holds head `a`'s stick `s` as the run of `B` sticks at stick column `a·S + s`, and the head-major
/// `[A·B, S·stick]` holds it at `s·A·B + a·B`. So the relayout either way is `S` copies — one per
/// stick of a head — of `A` runs of `B` sticks, at pitch `S·B` on the block's side and `B` on the
/// head-major side ([`crate::ir::bridge::tiled_op_sdsc_op::StickRuns`]): spyre-inference's
/// `q_rows.unsqueeze(0).transpose(1, 2).reshape(...)` and its output's inverse, at head dim 128.
struct HeadMajorCopies<'a> {
    block: &'a Region,
    heads: &'a Region,
    a: u32,
    b: u32,
    sticks: u32,
}

impl<'a> HeadMajorCopies<'a> {
    /// The copies between `block` (`[B, A·S·stick]`) and `heads` (`[A·B, S·stick]`), `A·B` rows of
    /// `S` sticks on the head-major side.
    fn of(block: &'a Region, heads: &'a Region, rows: u32, sticks: u32) -> Result<Self, Error> {
        let stick = crate::work::FP16_ELEMS_PER_STICK;
        let b = block.v_rows;
        if b == 0 || !rows.is_multiple_of(b) || block.v_cols != (rows / b) * sticks * stick {
            return err(format!(
                "a head-major relayout between t{} `[{}, {}]` and t{} `[{rows}, {}]`: the block is \
                 not `B` rows of the `A·S` sticks the head-major side's `A·B` rows hold",
                block.tid,
                block.v_rows,
                block.v_cols,
                heads.tid,
                sticks * stick
            ));
        }
        Ok(Self {
            block,
            heads,
            a: rows / b,
            b,
            sticks,
        })
    }

    /// Stick `s` of every head, block to head-major.
    fn to_heads(
        &self,
        name: &str,
        sym_id_base: &mut i64,
        layout: &BundleLayout,
    ) -> Result<Vec<super::EmittedOp>, Error> {
        self.copies(name, true, sym_id_base, layout)
    }

    /// Stick `s` of every head, head-major to block.
    fn to_block(
        &self,
        name: &str,
        sym_id_base: &mut i64,
        layout: &BundleLayout,
    ) -> Result<Vec<super::EmittedOp>, Error> {
        self.copies(name, false, sym_id_base, layout)
    }

    fn copies(
        &self,
        name: &str,
        to_heads: bool,
        sym_id_base: &mut i64,
        layout: &BundleLayout,
    ) -> Result<Vec<super::EmittedOp>, Error> {
        use crate::ir::bridge::tiled_op_sdsc_op::{StickRuns, assemble_stick_runs_copy};
        let stick = crate::work::FP16_ELEMS_PER_STICK;
        let (a, b, n) = (self.a, self.b, self.sticks);
        (0..n)
            .map(|s| {
                // Block side: run `a` at `a·S·B + s·B` sticks; head-major side: at `s·A·B + a·B`.
                let (block_pitch, block_elems) = (n * b, s * b * stick);
                let (heads_pitch, heads_elems) = (b, s * a * b * stick);
                let (src, dst, runs) = if to_heads {
                    (
                        self.block,
                        self.heads,
                        StickRuns {
                            runs: a,
                            run_sticks: b,
                            src_pitch: block_pitch,
                            dst_pitch: heads_pitch,
                            src_elems: block_elems,
                            dst_elems: heads_elems,
                            df: crate::superdsc_opspec::Df::Fp16,
                        },
                    )
                } else {
                    (
                        self.heads,
                        self.block,
                        StickRuns {
                            runs: a,
                            run_sticks: b,
                            src_pitch: heads_pitch,
                            dst_pitch: block_pitch,
                            src_elems: heads_elems,
                            dst_elems: block_elems,
                            df: crate::superdsc_opspec::Df::Fp16,
                        },
                    )
                };
                assemble_stick_runs_copy(
                    &format!("heads{s}_o{}", dst.tid),
                    &src.name(),
                    &dst.name(),
                    runs,
                    sym_id_base,
                    Some(layout),
                )
                .map_err(|message| Error {
                    message: format!("{name}: {message}"),
                })
            })
            .collect()
    }
}

/// Whether `tile` is a `ktdp.construct_indirect_access_tile`.
fn indirect_tile_of(f: &IRFunction<'static>, tile: Option<Ssa>) -> bool {
    tile.is_some_and(|t| {
        f.operations
            .iter()
            .any(|o| o.result == Some(t) && o.op_type == OpKind::KtdpConstructIndirectAccessTile)
    })
}

/// ⭐ THE SCATTER A VALUE IS STORED THROUGH — `res`, or a shape op that moves no byte of it, written
/// by a `ktdp.store` through an indirect access tile: its rows land on the rows of the destination
/// parameter its index names. `None` when no such store reads it.
fn indirect_store_of(
    k: &KtirNode,
    res: Ssa,
) -> Result<Option<super::lower_ktir_to_superdsc::RowScatter>, Error> {
    let f = &k.func;
    let dims = |v: Ssa| -> Vec<i64> {
        f.ops_deep()
            .into_iter()
            .find(|o| o.result == Some(v))
            .and_then(|o| o.result_type)
            .and_then(|t| t.dims())
            .map(<[i64]>::to_vec)
            .unwrap_or_default()
    };
    // Through a shape op only when it moves no byte: one that does is a relayout, and the scatter
    // reads its result (the reshape's own stick permute).
    let reshaped = f
        .operations
        .iter()
        .find(|o| {
            matches!(
                o.op_type,
                OpKind::TensorReshape | OpKind::TensorCollapseShape | OpKind::TensorExpandShape
            ) && o.operands.first() == Some(&res)
        })
        .and_then(|o| o.result)
        .filter(|&v| same_card_bytes(&dims(res), &dims(v)));
    let Some(store) = f.operations.iter().find(|s| {
        s.op_type == OpKind::KtdpStore
            && (s.operands.first() == Some(&res) || s.operands.first().copied() == reshaped)
            && indirect_tile_of(f, s.operands.get(1).copied())
    }) else {
        return Ok(None);
    };
    let tile = f
        .operations
        .iter()
        .find(|o| o.result == store.operands.get(1).copied())
        .expect("indirect_tile_of found it");
    super::lower_ktir_to_superdsc::row_scatter_of(k, tile).map(Some)
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

    // silu -> multiply by `up`, OR THE STORED BARE SILU. `Program::SiluMul` is `silu(gate)·up`;
    // a bare silu — the longhand's result stored directly, no multiply — is the UNARY
    // `Elementwise(Silu)` the door's own elementwise arm lowers as `("silu", 1)`, and the door
    // ROUTES elementwise through this walk, so the chain reader recognises the store terminal as
    // the longhand's second form rather than refusing what it just proved: the same five ops,
    // ending at `o_desc.store([..], silu(x))` instead of the `up` multiply (`elementwise.py`'s
    // `silu_fwd`). Everything else keeps the named refusal — one reader of a kind that is neither
    // the multiply nor the store, no reader at all, or a second reader the fusion would starve.
    let found: Vec<_> = readers(div_v).collect();
    match found.as_slice() {
        [m] if m.op_type == OpKind::ArithMulf => {
            let Some(up) = m.operands.iter().copied().find(|&s| s != div_v) else {
                return err(format!(
                    "{}: the silu's `arith.mulf` squares it rather than multiplying by an `up` tile",
                    f.name
                ));
            };
            let Some(mul_v) = m.result else {
                return err(format!("{}: `arith.mulf` with no result", f.name));
            };
            Ok(SiluMulChain {
                gate,
                up: Some(up),
                mul: mul_v,
                consumed: vec![neg_v, exp_v, add_v, div_v],
            })
        }
        // The divide's result is the program's output: its one reader is the `ktdp.store` that
        // writes `silu(x)` out. The divide is then the TERMINAL — its result is what the store
        // stores — so it is not chain-interior and the walk lowers it as the one fused op.
        [s] if s.op_type == OpKind::KtdpStore => Ok(SiluMulChain {
            gate,
            up: None,
            mul: div_v,
            consumed: vec![neg_v, exp_v, add_v],
        }),
        [o] => err(format!(
            "{}: `silu(x)` is read by `{:?}`, but a silu longhand reads it with `{:?}` or stores it",
            f.name,
            o.op_type,
            OpKind::ArithMulf
        )),
        [] => err(format!(
            "{}: nothing reads `silu(x)`, so the silu longhand does not continue there",
            f.name
        )),
        many => err(format!(
            "{}: `silu(x)` is read by {} ops ({}), and this fusion DESTROYS it — the device's \
             `OpFunc::Silu` never materialises the longhand's intermediates, so a second reader \
             would be reading a value nothing writes. Exactly one reader is required.",
            f.name,
            many.len(),
            many.iter()
                .map(|o| format!("{:?}", o.op_type))
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
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
        OpKind::MathLog => Program::Elementwise(Elementwise::Log),
        // The Gumbel transform's negations (`g = -log(-log(u))`). `OpFuncs::NEG`
        // is the card's own opFunc, and a negf that is NOT a silu head has no
        // other owner — see [`program_silu_mul_chains`] for the head test.
        OpKind::ArithNegf => Program::Elementwise(Elementwise::Neg),
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
    /// A fused `x · rsqrt(mean(x²)+eps) · gamma`, with the epsilon this chain's own and the
    /// OnePlusScale gain offset — `Some(1.0)` when the chain's gain leg is `(1 + gamma)`, `None`
    /// for the Scale convention. See [`program_rmsnorm_chains`] for why fusing it is the fix and
    /// filling the scale registry was not.
    RmsNorm(f32, Option<f32>),
    /// A dtype-CONVERT cast — `arith.extf` f16→f32 (`dl16tofp32`) or `arith.fptosi` f32→i32
    /// (`fp32toint32`) — the vendor's own convert `OpFunc`s, 1:1 in (op kind, element pair).
    /// Local rather than a `Program` variant for the same reason `Reduce` is: a bare convert is
    /// reachable only through this walk, and `Program` is matched exhaustively by consumers
    /// outside this crate. See [`convert_kind_of`] for the discriminator and what it refuses.
    Convert(ConvertKind),
    /// A wrapping i32 BINARY op — `arith.muli`/`arith.addi` — over SENUINT32 data, the noise
    /// hash's integer class. Local for the same reason [`Convert`](Lowering::Convert) is. The
    /// discriminator is NOT the op kind alone: most integer arith in a Triton-derived program is
    /// ADDRESSING (see [`is_plumbing`]'s own law), and only the ops [`integer_compute_of`]
    /// classifies as COMPUTE reach this arm — an integer op outside that set is plumbing and
    /// never lowers, exactly as before. See [`int_binary`] for the card form (leg B1) and the
    /// shapes it refuses.
    IntBinary(IntBinary),
    /// `tile <min|max> <splatted constant>` — `ScalarMul`'s tiebreak for the binaries whose other
    /// side is a constant (the walk's `min(tile_max - scores_max, 0)` clamp): the constant is the
    /// registry's bound `[1,1]` const, sprayed over the tile exactly as a scalar multiplier is.
    ScalarBinary(Elementwise, f32),
    /// `linalg.batch_matmul` `[b, m, k] × [b, k, n] → [b, m, n]` — one 2-D contraction per batch,
    /// each over its batch's rows of the flat-frame A and output and its batch's own `[k, n]`
    /// kernel block. See [`super::lower_ktir_to_superdsc::batch_matmul`].
    BatchMatmul {
        batch: u32,
        activation: super::lower_ktir_to_superdsc::BatchFrame,
    },
    /// A rank-3 `linalg.transpose` `[b, r, c] → [b, c, r]` — one 2-D transpose per batch, whose
    /// output blocks are the `[k, n]` kernels a batched contraction reads, its input in `input`.
    /// See [`super::lower_ktir_to_superdsc::batch_transpose`].
    BatchTranspose {
        batch: u32,
        input: super::lower_ktir_to_superdsc::BatchFrame,
    },
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

/// THE CONVERT A CAST NAMES, where the naming is 1:1 in (op kind, element pair).
///
/// ⭐ NOT IN [`program_of`] for the same reason [`reduce_kind_of`] is not: `program_of` takes an
/// `OpKind` alone, and `OpKind::ArithExtf` says "a widening cast happens here" without saying
/// WHICH — `extf` f16→f32 is `dl16tofp32`, an `extf` f16→f32 over an INTEGER target is a
/// different op with no recipe. The element pair is read off the op's own types, so nothing is
/// stapled to the program a third party must restate.
///
/// The two recipes are the vendor's own convert `OpFunc`s, and they are the KSPLIT fp32 island's
/// two ends: `dl16tofp32` widens an fp16 stick to same-stick fp32 (the multiply-add's entrance —
/// ⛔ an fp16 value MUST stage through it on the way to an integer, because `Fp32ToInt32`'s DDL
/// slice law refuses a direct fp16→u32 pairing), and `fp32toint32` re-sticks an IEEE fp32 word
/// as a SENUINT32 index (the exit).
///
/// ⛔ EVERY OTHER PAIR IS A NAMED REFUSAL, NOT A FALLTHROUGH. An `extf`/`fptosi`/`sitofp`/
/// `truncf` that reaches this function with a pair outside the recipes returns `Err` here —
/// naming the op, the operand's element and the result's — rather than `Ok(None)`, because
/// `Ok(None)` would hand it to the generic "no 1:1 `Program`" refusal whose message cannot say
/// WHICH converts this door does lower.
fn convert_kind_of(
    f: &IRFunction<'static>,
    op: &ktir_core::ir::Operation<'static>,
) -> Result<Option<ConvertKind>, Error> {
    if !matches!(
        op.op_type,
        OpKind::ArithExtf | OpKind::ArithFptosi | OpKind::ArithSitofp | OpKind::ArithTruncf
    ) {
        return Ok(None);
    }
    // The operand's element type, read off its DEFINING op's result type — the same lookup
    // `splat_scale_of`'s `def_of` makes, top-level, because a cast's operand is a value this
    // function computed or loaded, never a region interior.
    let elem_of = |v: Ssa| {
        f.operations
            .iter()
            .find(|o| o.result == Some(v))
            .and_then(|o| o.result_type)
            .and_then(|t| t.elem())
    };
    let [operand] = op.operands[..] else {
        return err(format!(
            "{}: `{:?}` states {} operands — a dtype-convert reads exactly one",
            f.name,
            op.op_type,
            op.operands.len()
        ));
    };
    let in_elem = elem_of(operand);
    let out_elem = op.result_type.and_then(|t| t.elem());
    match (op.op_type, in_elem, out_elem) {
        (OpKind::ArithExtf, Some(DType::F16), Some(DType::F32)) => {
            Ok(Some(ConvertKind::Dl16ToFp32))
        }
        (OpKind::ArithFptosi, Some(DType::F32), Some(DType::I32)) => {
            Ok(Some(ConvertKind::Fp32ToInt32))
        }
        // The integer-side entrance: a SENUINT32 word → an IEEE fp32 value, the reverse recipe
        // of `fp32toint32` (card leg B1's i2f cell proved the pairing on card). The on-card
        // Gumbel noise's wrap-fold extraction rides it (front-b-noise).
        (OpKind::ArithSitofp, Some(DType::I32), Some(DType::F32)) => {
            Ok(Some(ConvertKind::Int32ToFp32))
        }
        // The narrow-back: an IEEE fp32 value → an fp16 one, the reverse recipe of
        // `dl16tofp32`. The card's `log` is FP16-ONLY (the vendor DDL binds it to
        // `[%type_fp16]` alone — no fp32 binding, unlike `tanh`), so the Gumbel draw's
        // log chain must narrow before its logs or dxp refuses the op mapping
        // (front-b-noise, measured on the card leg). The `truncf` that narrows is
        // LEGALIZE-PROOF by construction: `LegalizeTypes` removes only f16→f16
        // truncfs (widening churn after the collapse), so a genuine f32→f16 pair
        // survives the pass and reaches this arm.
        (OpKind::ArithTruncf, Some(DType::F32), Some(DType::F16)) => {
            Ok(Some(ConvertKind::Fp32ToDl16))
        }
        _ => err(format!(
            "{}: `{:?}` {:?} → {:?} is not one of the dtype-converts this door lowers. The \
             device's convert OpFuncs spell exactly four recipes reachable from a Triton kernel: \
             `arith.extf` f16 → f32 (`dl16tofp32`, the fp32 island's entrance), \
             `arith.fptosi` f32 → i32 (`fp32toint32`, its exit — an fp16 value must stage \
             through the widen first; the direct fp16→u32 cast is DDL-refused), \
             `arith.sitofp` i32 → f32 (`int32tofp32`, the entrance from the integer side), and \
             `arith.truncf` f32 → f16 (`fp32todl16`, the narrow-back — the card's `log` is \
             fp16-only, so an f32 value on its way to one must narrow first). Any other pair \
             is a producer form refused by name rather than lowered as the nearest convert",
            f.name, op.op_type, in_elem, out_elem
        )),
    }
}

/// THE BUFFER FORMAT an op's result type names.
///
/// F16 and F32 are the fp16/fp32 sticks; I32 is SENUINT32 — the 4-byte/32-stick index format
/// that is `fp32toint32`'s own output. A STATED type with none of those mappings (a bool, an
/// i64, an fp8 result) is refused BY NAME: silently defaulting to fp16 would reserve HALF the
/// bytes an f32 intermediate needs (the footprint guard refuses the first over-run, or without
/// it the next intermediate is aliased onto it) and mis-stick every descriptor over it.
///
/// ⭐ AN UNSTATED type keeps the door's historical fp16 basis. `Operation::new` defaults
/// `result_type` to `None`, and a hand-built IR (the unit tests' minimal programs) states no
/// type at all — every op the real pipeline emits carries one, so `None` here is an omitted
/// annotation, not a fact to refuse, and the only behavior-preserving reading is the fp16 basis
/// every one of those programs was lowered under before the format became readable here.
fn result_df_of(name: &str, op: &ktir_core::ir::Operation<'static>) -> Result<Df, Error> {
    match op.result_type.and_then(|t| t.elem()) {
        None => Ok(Df::Fp16),
        Some(DType::F16) => Ok(Df::Fp16),
        Some(DType::F32) => Ok(Df::Fp32),
        Some(DType::I32) => Ok(Df::SenUint32),
        Some(other) => err(format!(
            "{name}: `{:?}`'s result element type is {other:?}, which has no buffer format this \
             door mints — an intermediate of that dtype has no stick basis to reserve or address. \
             It is refused by name rather than defaulted to fp16, which would reserve the wrong \
             width and mis-stick every descriptor over it",
            op.op_type
        )),
    }
}

/// The equality-mask folds: each retype's result → the comparison's own two operands, plus the
/// i1 results the fold consumed (skipped at the walk's loop top, beside the chain interiors).
struct EqualMasks {
    by_retype: std::collections::HashMap<Ssa, (Ssa, Ssa)>,
    consumed: std::collections::HashSet<Ssa>,
}

/// ⭐⭐⭐ THE EQUALITY MASK — `arith.cmpf(oeq)` retyped by `arith.uitofp`, folded to ONE
/// `Elementwise(Equal)`.
///
/// # WHY THIS RECOGNISER EXISTS AT ALL, AND WHY IT IS NOT AN EXCEPTION TO THE MODULE HEADER
///
/// The device HAS the primitive: `OpFunc::Equal` (`broadcast_ops.ddl:41`) realizes `a == b` as
/// an fp16 0.0/1.0 TENSOR — `EQUAL` into a BOOL state reg, then `SELECT(plus1, zero)` — the
/// same arity-2 broadcast-op family as `maximum`/`minimum`. KTIR has no op for that: the
/// comparison is `arith.cmpf` (an i1) and the 0/1 tensor is a SEPARATE `arith.uitofp` retype,
/// because an i1 cannot ride the fp16 stick. So the two-op spelling is the only one a producer
/// HAS — the same argument that admits the silu longhand ([`program_silu_mul_chains`]): the
/// device's op exists and the IR cannot name it in one op, and lowering the pair op-by-op would
/// emit a form the pointwise family does not have (a bool-typed compare with no stick) rather
/// than a slower lowering of the same function. The fold is the 1:1 map: `cmpf(oeq)` +
/// `uitofp`→f16 IS `Elementwise(Equal)` over the cmpf's own two operands.
///
/// # ⛔ WHAT IS REFUSED RATHER THAN GUESSED
///
/// * the predicate must be `oeq` — `OpFunc::Equal` is EQUALITY (NaN compares FALSE on the SFP
///   unit, matching the ordered compare), and any other predicate is a DIFFERENT function this
///   fold must not silently re-mean. An unfolded `cmpf` — any other predicate, or one whose sole
///   reader is not the retype — is left alone and reaches the walk's by-name refusal, which
///   names `ArithCmpf`.
/// * the retype's target must be F16 or F32 — the two formats the elementwise door lowers. The
///   device op's wire-level result IS the 0/1 tensor (`SELECT(plus1, zero)`), stated at the
///   RETYPE'S OWN df — the walk's `op_df` reads the retype's result type — so the sampler's fp32
///   Gumbel-max mask (`(score == mm).to(tl.float32)`) folds to `Elementwise(Equal)` at Fp32
///   through the df-carrying builder, the same broadcast-op family `maximum`/`minimum` take. A
///   `uitofp` to any other width is left alone rather than re-emitted at a format the door
///   refuses by name.
/// * the `cmpf`'s i1 result must have exactly ONE consumer, the retype — the fold DESTROYS the
///   i1 value (the device op never materialises it), so a second reader would read a value
///   nothing writes. The same seal the silu recogniser states for its longhand interior.
///
/// Returns the folds ([`EqualMasks`]): each `uitofp` result → the `cmpf`'s two operands, and the
/// i1 results the fold consumed.
fn program_equal_masks(f: &IRFunction<'static>) -> Result<EqualMasks, Error> {
    let mut masks = EqualMasks {
        by_retype: std::collections::HashMap::new(),
        consumed: std::collections::HashSet::new(),
    };
    let deep = f.ops_deep();
    for op in deep
        .iter()
        .copied()
        .filter(|o| o.op_type == OpKind::ArithCmpf)
    {
        let Some(res) = op.result else { continue };
        // THE PREDICATE IS THE FUNCTION: ordered equality is `OpFunc::Equal`'s own semantics.
        let oeq = op
            .attributes
            .iter()
            .any(|(k, v)| matches!((k, v), (AttrKey::Predicate, Attr::Str(s)) if *s == "oeq"));
        if !oeq {
            continue;
        }
        let [a, b] = op.operands[..] else { continue };
        // ⛔ EXACTLY ONE READER, AND IT IS THE RETYPE — see the seal above.
        let mut consumers = deep.iter().copied().filter(|o| o.operands.contains(&res));
        let Some(u) = consumers.next() else { continue };
        if consumers.next().is_some() || u.op_type != OpKind::ArithUitofp || u.operands != [res] {
            continue;
        }
        // ⛔ THE RETYPE'S TARGET IS F16 OR F32 — the two formats the elementwise door lowers, and
        // the emitted `Elementwise(Equal)` states the RETYPE'S OWN df (`op_df` reads the result
        // type below), so the fp32 Gumbel-max mask lowers as the fp32 form of the same
        // broadcast-op family. Any other width is left alone for the by-name refusal, rather than
        // re-emitted at a format the door refuses.
        if !matches!(
            u.result_type,
            Some(ktir_core::irtype::IrType::Tensor {
                elem: DType::F16 | DType::F32,
                ..
            })
        ) {
            continue;
        }
        let Some(u_res) = u.result else { continue };
        masks.by_retype.insert(u_res, (a, b));
        masks.consumed.insert(res);
    }
    Ok(masks)
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

/// ⭐⭐⭐⭐⭐ INTEGER COMPUTE, DECIDED BY CONSUMERS — the results of the integer arith ops that are
/// DATA rather than ADDRESSING, and therefore the ONE exemption to [`is_plumbing`]'s integer rule.
///
/// That rule's claim — "every one of these in a Triton-derived program computes a subscript … and
/// is consumed by a `ktdp.construct_access_tile`, never by a compute op" — held for every kernel
/// this door had lowered until the on-card noise hash (front-b-noise): a ~20-op wrapping i32
/// mul/add chain whose result feeds an `arith.sitofp` (the wrap-fold extraction's
/// `int32tofp32`), with not one subscript in it. The device HAS the node kind — `muli32toi32` /
/// `addi32toi32` on SENUINT32 data, card leg B1 — so refusing the chain as plumbing would be
/// wrong in the other direction: the consumers decide.
///
/// The classification is a BACKWARD FIXPOINT over exactly three seeds, each a consumer that can
/// only be reading DATA:
///   * an op this walk LOWERS — not plumbing, not integer arith (the `sitofp`, an `fptosi`, a
///     float op, a reduce). A subscript never feeds one, because every body here reads tensor
///     operands, and addressing reaches a descriptor only through a tile op;
///   * a `ktdp.store`'s VALUE operand — the program's own statement that this integer is an
///     output, not an address. Before this fixpoint such a chain was skipped with its store
///     (both plumbing) and the output parameter was silently never written;
///   * an integer arith op already classified compute — the chain's own interior.
///
/// Everything else keeps the plumbing reading verbatim: an integer op whose consumers are all
/// tile constructors, splats and constants stays addressing, lowered by nothing, exactly as
/// before — so for every kernel that bakes today this set's membership among their integer ops
/// is empty and the walk below is byte-identical.
fn integer_compute_of(f: &IRFunction<'static>) -> std::collections::HashSet<Ssa> {
    let is_int_arith =
        |k: OpKind| matches!(k, OpKind::ArithMuli | OpKind::ArithAddi | OpKind::ArithSubi);
    // result → the ops naming it as an operand.
    let mut consumers: std::collections::HashMap<Ssa, Vec<&ktir_core::ir::Operation<'static>>> =
        std::collections::HashMap::new();
    for op in f.operations.iter() {
        for v in op.operands.iter() {
            consumers.entry(*v).or_default().push(op);
        }
    }
    let mut compute = std::collections::HashSet::new();
    let mut grew = true;
    while grew {
        grew = false;
        for op in f.operations.iter() {
            let Some(res) = op.result else { continue };
            if !is_int_arith(op.op_type) || compute.contains(&res) {
                continue;
            }
            let is_data = consumers.get(&res).is_some_and(|cs| {
                cs.iter().any(|c| {
                    if c.op_type == OpKind::KtdpStore {
                        // Operand 0 is the VALUE; a store never reads a subscript.
                        c.operands.first() == Some(&res)
                    } else if is_int_arith(c.op_type) {
                        // Chain interior: compute only by the fixpoint's own membership.
                        c.result.is_some_and(|r| compute.contains(&r))
                    } else {
                        // An op this walk lowers — the seed class. Plumbing consumers (tile
                        // constructors, splats, constants, casts of pointers) confer nothing.
                        !is_plumbing(c.op_type)
                    }
                })
            });
            if is_data {
                compute.insert(res);
                grew = true;
            }
        }
    }
    compute
}

/// ONE recognised broadcast: the value a consumer reads, resolved back to the rank-2 SOURCE the
/// chain started from, plus the axis.
///
/// ⭐ `flat` marks the RANK > 2 FORM — a broadcast whose result is rank 3 or more. The kernel-side
/// spelling is the same softmax chain as the rank-2 one (an `expand_shape` inserting a degenerate
/// trailing dim, a `collapse_shape`, then the `linalg.broadcast`), but the source is a full
/// `[m, n]` tile rather than a vector, and the fact the chain proves is different: the source's
/// FLAT elements are the vector — one scalar per flat row for `Col` (the `[B, G]` running max
/// sprayed over `[B, G, PS]`'s trailing axis), a whole flat row for `Mb`. The consumer therefore
/// presents the source's region FLAT — `[numel, 1]` / `[1, numel]` — which is the same bytes at
/// the same offsets the tile always had; see the presentation at the walk's extent check.
#[derive(Clone, Copy)]
struct Bcast {
    src: Ssa,
    axis: BcastAxis,
    flat: bool,
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
/// gain (through the same op kinds, to depth 3, plus the shape-preserving elementwise converts
/// below); this generalises it to any consumer.
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
        // ⭐⭐⭐ THE RESULT'S RANK DECIDES THE FRAME THE AXIS IS READ IN. Rank ≤ 2 is the law as it
        // stood: `Dimensions` names the one axis of a rank-2 tile, `[1]` the trailing (stick) axis
        // and `[0]` the rows. Rank > 2 is the batched kernel's frame — the leading axes are the
        // flat frame's ROWS (the same reading `shape_2d` states) — so the TRAILING axis
        // (`dims[0] == rank-1`) is still `Col`, but the source is a full tile whose FLAT elements
        // are the per-row scalars (a `[B, G]` running max sprayed over `[B, G, PS]`): the `flat`
        // form, which the consumer presents as `[numel, 1]`. A LEADING axis is `Mb` only when the
        // source is ONE flat row; anything else on a leading or middle axis is ROW-REPLICATION —
        // each flat row of the source repeating down the merged row axis — and is refused by name
        // below, because no operand mode addresses it (see that refusal).
        let res_dims: Option<Vec<i64>> = bc
            .attributes
            .iter()
            .find_map(|(k, v)| match (k, v) {
                (AttrKey::Shape, Attr::IntList(d)) => Some(d.to_vec()),
                _ => None,
            })
            .or_else(|| {
                bc.result_type
                    .as_ref()
                    .and_then(|t| t.dims().map(|d| d.to_vec()))
            });
        let rank = res_dims.as_ref().map(Vec::len);
        let Some(rank) = rank else {
            return err(format!(
                "{}: `linalg.broadcast` states no shaped result, so its rank — the frame the \
                 `Dimensions` axis is read in — is unknown",
                f.name
            ));
        };
        // Mutable because the Y ADMISSION below re-states a leading multi-row spray's axis and
        // flat flag after the walk-back has resolved the source's own row count.
        // ⭐ A RUN OF LEADING AXES IS ONE LEADING AXIS OF THE FLAT FRAME: the flat frame folds every
        // axis but the last into its rows, so spraying `[0..k)` (spyre-inference's mask tile over
        // `[kv_head, group]`, `mask_tile[0]` broadcast against `[kv, group, query, block]`) is the
        // leading spray `[0]` is, its count the product of the k sprayed extents (the Y admission
        // below).
        let leading = rank > 2
            && !dims.is_empty()
            && dims.len() < rank
            && dims.iter().enumerate().all(|(i, &d)| d == i as i64);
        let (mut axis, mut flat) = match (rank, dims) {
            (_, [1]) if rank <= 2 => (BcastAxis::Col, false),
            (_, [0]) if rank <= 2 => (BcastAxis::Mb, false),
            (r, [d]) if r > 2 && *d as usize == r - 1 => (BcastAxis::Col, true),
            _ if leading => (BcastAxis::Mb, true),
            // ⭐ A LEADING RUN PLUS THE STICK AXIS IS A PER-ROW SCALAR SPRAYED DOWN THE LEADING
            // AXES: `[B, KV, G] -> [J, B, KV, G, block]` (`Dimensions=[0, 4]`, a decode batch's
            // one-chunk running max against its scores) — the `Col` read and the outer row spray at
            // once. The count is the leading run's extent; the cover (the source's values ARE the
            // result's rows after the run) is proved after the walk-back.
            (r, ds)
                if r > 2
                    && ds.len() >= 2
                    && ds.last() == Some(&(r as i64 - 1))
                    && ds[..ds.len() - 1]
                        .iter()
                        .enumerate()
                        .all(|(i, &d)| d == i as i64) =>
            {
                let k = ds.len() - 1;
                let count: i64 = res_dims
                    .as_ref()
                    .map(|s| s[..k].iter().product())
                    .unwrap_or(0);
                if count <= 1 {
                    return err(format!(
                        "{}: `linalg.broadcast` adds the leading run {:?} and the stick axis of a \
                         rank-{r} tensor, to a leading extent of {count} — no spray",
                        f.name,
                        &ds[..k]
                    ));
                }
                (
                    BcastAxis::Y(super::lower_ktir_to_superdsc::Spray::outer_row_scalars(
                        count as u32,
                    )),
                    false,
                )
            }
            // ⭐ THE AXIS JUST BEFORE THE STICK IS THE INNER ROW SPRAY: each of the source's flat
            // rows repeated `y` times in a row (`[J, B, KV, 1, block] -> [J, B, KV, G, block]`, a
            // decode batch's mask over its query-head group) — the row-split mode in its
            // `[mb, y, out]` order. The count is the result's own extent on the axis; the cover
            // (the source's rows ARE the result's rows before it) is proved after the walk-back,
            // as the outer spray's is.
            (r, [d]) if r > 2 && *d as usize == r - 2 && *d > 0 => {
                match res_dims.as_ref().and_then(|s| s.get(*d as usize)) {
                    Some(&y) if y > 1 => (
                        BcastAxis::Y(super::lower_ktir_to_superdsc::Spray::inner(y as u32)),
                        false,
                    ),
                    other => {
                        return err(format!(
                            "{}: `linalg.broadcast` adds dim {d} of a rank-{r} tensor — the axis \
                             just before the stick — to an extent of {other:?}; a spray of one is \
                             no broadcast, and an unstated extent is no spray count",
                            f.name
                        ));
                    }
                }
            }
            (r, [d]) if r > 2 => {
                return err(format!(
                    "{}: `linalg.broadcast` adds dim {d} of a rank-{r} tensor — a MIDDLE axis of \
                     the flat frame with more than the stick after it. Row-major that is \
                     ROW-REPLICATION AT A STRIDE: each of the source's flat rows repeats down the \
                     merged row axis with the axes after the sprayed one interleaved between the \
                     copies. The device's pointwise operand modes state a per-row scalar \
                     (`In::col`), a row vector (`In::mb`) and a row spray of a whole source either \
                     outermost or innermost of the rows (`Y`, `[y, mb, out]` or `[mb, y, out]`); \
                     none states an address that advances every k rows with a stride between the \
                     copies, so this is named rather than half-addressed",
                    f.name
                ));
            }
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
        // WALK BACK to the rank-2 value the chain started from, through the same op kinds and the
        // SAME depth `program_rmsnorm_chains` walks, plus the shape-preserving elementwise converts
        // (a convert is one-for-one on elements, so the extent the broadcast must prove is
        // unchanged across it — the arm's own comment states the proof, and the law the binding
        // obeys: the convert EMITS and the consumer binds its result, never its operand). The
        // `tensor.empty` second operand is the broadcast's `outs` init and carries no data, so only
        // operand 0 is followed.
        let Some(mut cursor) = bc.operands.first().copied() else {
            return err(format!(
                "{}: `linalg.broadcast` has no source operand",
                f.name
            ));
        };
        let mut chain = vec![res];
        let mut saw_degenerate_expand = false;
        // ⭐⭐⭐ THE OUTERMOST CONVERT THE WALK CROSSED, if it crossed one — the value the
        // consumer's operand BINDS (see the convert arm). `None` until a convert is crossed,
        // and then never overwritten: the loop walks OUTWARD from the broadcast, so the FIRST
        // convert crossed is the outermost — the one whose result the collapse/broadcast chain
        // actually sprays.
        let mut crossed_convert: Option<Ssa> = None;
        for _ in 0..3 {
            let Some(d) = def_of(cursor) else { break };
            let mut convert_hop = false;
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
                        // A per-row scalar's spray inserts two degenerate dims, one expand each:
                        // the leading one and the stick's — this expand is either.
                        BcastAxis::Y(super::lower_ktir_to_superdsc::Spray {
                            of: super::lower_ktir_to_superdsc::SprayOf::RowScalars,
                            ..
                        }) => {
                            if target.first() == Some(&1) {
                                Some(0)
                            } else {
                                target.len().checked_sub(1)
                            }
                        }
                        // The inner spray's axis is the one just before the stick.
                        BcastAxis::Y(sp) if sp.order == crate::emit::SprayOrder::Inner => {
                            target.len().checked_sub(2)
                        }
                        // `Mb` and the (later-assigned) outer `Y` both spray a LEADING axis: the
                        // relay's insert must be the leading `1` either way — at classification
                        // time the axis is still `Mb` (the outer `Y` admission runs after the
                        // walk-back, on the resolved source's own row count), and a chain the
                        // admission flips to `Y` passed the same leading-`1` check here.
                        BcastAxis::Mb | BcastAxis::Y(_) => Some(0),
                    };
                    // CROSS-CHECKED AGAINST THE BROADCAST'S OWN `Dimensions`: two independent
                    // statements of the same axis, so a program where they disagree is refused instead
                    // of one of them being believed.
                    // ⭐ ONCE THE SPRAYED AXIS IS PROVEN by the expand nearest the broadcast, an expand
                    // further out only inserts another unit dim — a one-chunk decode batch's leading
                    // `J = 1` (`[32, 8, 4]` → `[1, 32, 8, 4]` → `[1, 32, 8, 4, 1]`) — which moves no
                    // element, so the walk crosses it.
                    if !saw_degenerate_expand
                        && want.map(|i| target.get(i).copied()) != Some(Some(1))
                    {
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
                                BcastAxis::Y(super::lower_ktir_to_superdsc::Spray {
                                    of: super::lower_ktir_to_superdsc::SprayOf::RowScalars,
                                    ..
                                }) => "leading or trailing",
                                BcastAxis::Y(sp) if sp.order == crate::emit::SprayOrder::Inner => {
                                    "second-to-last"
                                }
                                // `Mb` and the post-walk-back outer `Y` both spray the LEADING
                                // axis — the same word names it for the reader either way.
                                BcastAxis::Mb | BcastAxis::Y(_) => "leading",
                            }
                        ));
                    }
                    saw_degenerate_expand = true;
                }
                OpKind::TensorCollapseShape => {}
                // ⭐ A NUMEL-PRESERVING `TensorReshape` IS A FLAT NO-OP, SO THE WALK TRAVERSES IT.
                // A reshape states no degenerate dim and moves no element: it re-frames the SAME
                // row-major flat order (Triton's `tl.reshape`, the only producer this door sees),
                // so the value it produces and the value it consumes are the SAME BYTES and the
                // source the walk is looking for is on the far side of it. That is exactly the
                // J-slot merge's own sandwich — `merged [64, 4] -> reshape [1, 64, 4] -> collapse
                // -> broadcast` — whose source is the J-reduce's OUTPUT, a mint this function's
                // consumer looks up by SSA; stopping at the reshape would name a plumbing value
                // no mint exists for. The numel equality is the whole proof: an element-count
                // change is a real data movement (a relayout this door cannot express), named
                // below rather than traversed. This is the ONE reshape fact this walk takes;
                // forwarding a source region THROUGH a reshape for a general consumer is wall 5's
                // own door, not this one.
                OpKind::TensorReshape => {
                    let target = d.attributes.iter().find_map(|(k, v)| match (k, v) {
                        (AttrKey::TargetShape, Attr::IntList(t)) => Some(*t),
                        _ => None,
                    });
                    let Some(target) = target else {
                        return err(format!(
                            "{}: `tensor.reshape` states no `TargetShape`, so whether it \
                             preserves the element count — the thing that makes it a flat no-op \
                             the broadcast walk can traverse — cannot be decided",
                            f.name
                        ));
                    };
                    let target_numel: i64 = target.iter().product();
                    // The SOURCE's own extent, from the op that produced the reshape's operand —
                    // the next hop out, which the loop is about to walk to anyway.
                    let source_numel = d
                        .operands
                        .first()
                        .and_then(|&s| def_of(s))
                        .and_then(|sd| {
                            sd.attributes.iter().find_map(|(k, v)| match (k, v) {
                                (AttrKey::Shape, Attr::IntList(l)) => {
                                    Some(l.iter().product::<i64>())
                                }
                                _ => None,
                            })
                        })
                        .or_else(|| {
                            d.operands.first().and_then(|&s| def_of(s)).and_then(|sd| {
                                sd.result_type
                                    .as_ref()
                                    .and_then(|t| t.dims().map(|ds| ds.iter().product::<i64>()))
                            })
                        });
                    match source_numel {
                        Some(n) if n == target_numel => {}
                        Some(n) => {
                            return err(format!(
                                "{}: `tensor.reshape` targets {target:?} ({target_numel} \
                                 elements) from a {n}-element source — a reshape that changes the \
                                 element count is a real data movement, not the flat no-op a \
                                 broadcast chain traverses. The broadcast's source would be a \
                                 relayout this door cannot express, so it is named rather than \
                                 mis-addressed",
                                f.name
                            ));
                        }
                        // The source's extent is not stated here: stop AT the reshape's result
                        // and let the consumer's own region lookup name what cannot be found,
                        // rather than guessing the traversal.
                        None => break,
                    }
                }
                // ⭐⭐⭐⭐⭐ A SHAPE-PRESERVING ELEMENTWISE CONVERT IS TRAVERSED ONLY TO PROVE THE
                // EXTENT — IT IS NEVER PLUMBING, AND THE CONSUMER NEVER BINDS ITS OPERAND.
                // `arith.sitofp`/`extf`/`truncf` are elementwise — one result element per
                // operand element, converted — so the EXTENT the broadcast has to prove is
                // unchanged across the hop and the walk keeps going to the load whose own
                // window states the vector-ness (the second proof below). But a convert
                // changes the value's ENCODING, not just its shape: the bytes on the two
                // sides of the hop are NOT the same value, so the consumer binds the
                // OUTERMOST crossed convert's RESULT — a minted intermediate the main walk
                // emits through `Lowering::Convert` — and never the load the convert reads.
                //
                // ⛔ MEASURED ON THE CARD (front-b-noise, round 3, nickm2): binding the load
                // made the sampler's `noise·gate` multiply read the per-row gate's RAW
                // SENUINT32 placement as IEEE_FP32 — the staged u32 word 1 read as the
                // denormal 2^-149, an effective gate of 2^-127 — so score lost its noise
                // term entirely (score = the dl16-staggered x to denormal precision, t2 =
                // fl(g·2^-127) bit-proven) and the pick collapsed to argmax at every
                // temperature: greedy accidentally correct, sampling silently greedy.
                //
                // The proof the hop itself takes is the same class the reshape hop takes:
                // an equality stated by the IR itself, with the convert's own shape (its
                // `Shape`, which `to_ktir_emit`'s `state_shape_and_dtype` post-pass stamps
                // from the result type) on one side and its operand's (the producer's own
                // statement — the next hop out, which the loop is about to walk to anyway)
                // on the other. A convert whose two stated shapes disagree is a real data
                // movement this door cannot express, named rather than traversed; one that
                // states either side not at all stops the walk AT the convert's result,
                // where the load-window proof below honestly fails, rather than guessing
                // the traversal.
                OpKind::ArithSitofp | OpKind::ArithExtf | OpKind::ArithTruncf => {
                    let dims_of = |o: &Operation<'static>| {
                        o.attributes
                            .iter()
                            .find_map(|(k, v)| match (k, v) {
                                (AttrKey::Shape, Attr::IntList(s)) => Some(s.to_vec()),
                                _ => None,
                            })
                            .or_else(|| {
                                o.result_type
                                    .as_ref()
                                    .and_then(|t| t.dims().map(|ds| ds.to_vec()))
                            })
                    };
                    let result_dims = dims_of(d);
                    let operand_dims = d
                        .operands
                        .first()
                        .and_then(|&s| def_of(s))
                        .and_then(dims_of);
                    match (result_dims, operand_dims) {
                        (Some(r), Some(o)) if r == o => {
                            // The hop is proven shape-preserving, so the walk continues to
                            // the load for the extent proofs — and the convert EMITS: its
                            // result leaves `chain` (below), so the main walk lowers it, and
                            // the FIRST convert crossed is the outermost whose result the
                            // consumer binds.
                            if crossed_convert.is_none() {
                                crossed_convert = Some(cursor);
                            }
                            convert_hop = true;
                        }
                        (Some(r), Some(o)) => {
                            return err(format!(
                                "{}: `{:?}` converts a {o:?}-shaped operand to {r:?} — an \
                                 elementwise convert is one-for-one on elements, so a dims \
                                 change is a real data movement, not the shape-preserving hop \
                                 a broadcast chain traverses. The broadcast's source would be \
                                 a relayout this door cannot express, so it is named rather \
                                 than mis-addressed",
                                f.name, d.op_type
                            ));
                        }
                        // Either side's dims are not stated here: stop AT the convert's result
                        // and let the consumer's own region lookup name what cannot be found,
                        // rather than guessing the traversal.
                        _ => break,
                    }
                }
                _ => break,
            }
            // ⭐ A CROSSED CONVERT'S RESULT IS NOT PLUMBING — it emits through the main walk's
            // `Lowering::Convert` arm — so it never enters `chain`, whose every member is
            // "lowered by nothing". The consumer binds it through `crossed_convert` instead.
            if !convert_hop {
                chain.push(cursor);
            }
            let Some(next) = d.operands.first().copied() else {
                break;
            };
            cursor = next;
        }
        // ⭐⭐⭐ A FLAT `Mb` STILL OWES ITS ONE-ROW PROOF, AND THE EXPAND DOES NOT GIVE IT. A
        // leading degenerate expand on a `[64, 4]` source targets `[1, 64, 4]` — degenerate on
        // axis 0, exactly what the cross-check above wants — yet the source is SIXTY-FOUR flat
        // rows. The rank-2 law never needed this: a rank-2 `Mb` source IS one row by shape. The
        // flat form must say it in its own frame: the source's `shape_2d` rows — every axis but
        // the last folded — must be 1. A source that passes keeps the flat `Mb` verbatim. A
        // MULTI-row source falls to the Y ADMISSION below before any refusal is returned, so a
        // leading spray over real rows gets the mode that addresses it — or the refusal naming
        // why not — and not the generic unproven-extent one.
        let mut y_admitted = false;
        if flat && axis == BcastAxis::Mb {
            let source_rows = def_of(cursor)
                .and_then(super::lower_ktir_to_superdsc::shape_2d)
                .map(|(r, _)| r);
            if source_rows != Some(1) {
                // ⭐⭐⭐⭐⭐ THE Y ADMISSION — the ROW-SPLIT broadcast. A leading-axis spray over a
                // MULTI-row source is not row-replication when the result's own shape says the
                // split: `[y, mb, ..]` with `y > 1` and `mb` the source's OWN row count. That is
                // the J-slot merge exactly — `merged [64, 4] -> [4, 64, 4]`, `Dimensions=[0]` —
                // and the device CAN address it: the consumer op `[mb, out, y]` presents every
                // tensor in the `[y, mb, out]` frame and the source is the `[mb, out]` slice at
                // y's stride-0 (scale `y = -1`, the vendor's reduced-dim law — the same scale the
                // on-card pmask fusion uses), so flat element `(j, b, c)` reads the source's
                // `(b, c)` byte for byte. The count comes from the broadcast's OWN stated leading
                // extent and the cover from the source's own shape, so the admission is proved
                // from two independent statements — and the consumer re-proves it against the
                // resolved REGION (`elementwise_broadcast`'s want check + the frame law) before
                // any operand is built. Everything else about a leading multi-row spray really is
                // row-replication and is refused by name below.
                let res_shape = bc
                    .attributes
                    .iter()
                    .find_map(|(k, v)| match (k, v) {
                        (AttrKey::Shape, Attr::IntList(s)) => Some(s.to_vec()),
                        _ => None,
                    })
                    .or_else(|| {
                        bc.result_type
                            .as_ref()
                            .and_then(|t| t.dims().map(|d| d.to_vec()))
                    });
                let Some(s) = res_shape else {
                    return err(format!(
                        "{}: `linalg.broadcast` sprays a rank > 2 source of {} flat rows down its \
                         LEADING axis but states no shaped result, so the row-split frame — the \
                         `[y, mb, ..]` the leading dim would have to name — cannot be proven. The \
                         `mb` operand mode addresses ONE row once and the row-split mode needs the \
                         spray count, so neither can be chosen for it",
                        f.name,
                        source_rows.unwrap_or(0)
                    ));
                };
                // The spray count: the leading axes the broadcast adds, folded; the source's rows must
                // then be the rest of the row axes, folded.
                let lead = dims.len().min(s.len());
                let y: i64 = s[..lead].iter().product();
                let mb: i64 = s[lead..s.len().saturating_sub(1)].iter().product();
                if y <= 1 {
                    return err(format!(
                        "{}: `linalg.broadcast` sprays a rank > 2 source of {} flat rows down its \
                         LEADING axis to a result whose leading dim is {y} — a spray of one is no \
                         broadcast (the dim is already there), and anything else over a multi-row \
                         source is ROW-REPLICATION: each flat row repeating down the merged row \
                         axis. The device's `mb` operand mode addresses ONE row once; the \
                         row-split mode needs a real `y > 1`; neither addresses this, so it is \
                         named rather than half-addressed",
                        f.name,
                        source_rows.unwrap_or(0)
                    ));
                }
                match source_rows {
                    None => {
                        return err(format!(
                            "{}: `linalg.broadcast` sprays a rank > 2 source down its LEADING \
                             axis to {s:?}, but the source states no shape, so its row count — the \
                             split's `mb` — is unproven. The row-split frame needs `rows == y*mb` \
                             exactly, and an unproven cover either reads rows the source does not \
                             have or leaves rows unaddressed",
                            f.name
                        ));
                    }
                    Some(rows) if mb as u32 == rows => {
                        axis = BcastAxis::Y(super::lower_ktir_to_superdsc::Spray::outer(y as u32));
                        flat = false;
                        y_admitted = true;
                    }
                    Some(rows) => {
                        return err(format!(
                            "{}: `linalg.broadcast` sprays a {}-flat-row source down its LEADING \
                             axis to {s:?} — the merged axis does not line up with the source's \
                             rows, so this is ROW-REPLICATION, each flat row repeating down the \
                             merged row axis (the batched paged-decode kernel's J-slot merge is \
                             the admitted form of this shape, `[64, 4] -> [4, 64, 4]`, and it \
                             lines up: 4 sprays of the source's own 64 rows). The device's `mb` \
                             operand mode addresses ONE row once and the row-split mode needs \
                             `rows == y*mb` exactly, so neither addresses this, and no descriptor \
                             in this door's vocabulary does — it is named rather than \
                             half-addressed. The door-proven statements of the same math are a \
                             real permute (restating the merge so the merged axis is TRAILING) or \
                             unrolled per-J state variables — the kernel's own choice, not a \
                             lowering fix",
                            f.name, rows
                        ));
                    }
                }
            }
        }
        // ⭐ THE INNER SPRAY'S COVER: the source's flat rows must BE the result's rows before the
        // sprayed axis — `[J·B·KV, block]` under `[J, B, KV, G, block]` — or the copies would read
        // rows the source does not have. Proved on the resolved source's own shape, independently
        // of the count the classification read off the result.
        if let BcastAxis::Y(sp) = axis
            && sp.order == crate::emit::SprayOrder::Inner
        {
            let source_rows = def_of(cursor)
                .and_then(super::lower_ktir_to_superdsc::shape_2d)
                .map(|(r, _)| r as i64);
            let before: Option<i64> = res_dims
                .as_ref()
                .map(|s| s[..s.len().saturating_sub(2)].iter().product());
            if source_rows.is_none() || source_rows != before {
                return err(format!(
                    "{}: `linalg.broadcast` sprays its second-to-last axis {} times over a source \
                     of {source_rows:?} flat rows, but the result's rows before that axis are \
                     {before:?} — the inner row spray repeats each SOURCE row in place, so the \
                     two must be the same rows, or the copies read rows the source does not have",
                    f.name, sp.count
                ));
            }
            y_admitted = true;
        }
        // ⭐ THE PER-ROW SCALARS' COVER: the source holds one value per result row after the
        // sprayed run — `B·KV·G` values under `[J, B, KV, G, block]` — proved on the resolved
        // source's own element count.
        if let BcastAxis::Y(sp) = axis
            && sp.of == super::lower_ktir_to_superdsc::SprayOf::RowScalars
        {
            let numel = def_of(cursor)
                .and_then(super::lower_ktir_to_superdsc::shape_2d)
                .map(|(r, c)| i64::from(r) * i64::from(c));
            let rows: Option<i64> = res_dims.as_ref().map(|s| {
                s[..s.len().saturating_sub(1)].iter().product::<i64>() / i64::from(sp.count)
            });
            if numel.is_none() || numel != rows {
                return err(format!(
                    "{}: `linalg.broadcast` sprays {numel:?} per-row values down {} leading \
                     copies of {rows:?} rows each — one value per row is the cover, or the read \
                     reaches values the source does not have",
                    f.name, sp.count
                ));
            }
            y_admitted = true;
        }
        if !saw_degenerate_expand && !y_admitted {
            // ⭐ THE SOURCE'S OWN LOAD IS A SECOND PROOF OF THE SAME FACT. An fp8 weight-scale chain
            // does not spell the expand: the kernel loads the scale as a `[1, n]` WINDOW of its own
            // parameter (`matmul_fp8.py`'s `ws_desc.load([0, 0])` over a `[1, n]` descriptor),
            // collapses the degenerate dim away, and broadcasts the 1-D result down the rows. The
            // load's OWN shape states the vector-ness the expand would have — extent 1 on the
            // sprayed axis — because the window it names is the whole parameter. A source that is
            // neither an expand-proven vector nor a load of extent 1 on the spray axis keeps the
            // refusal below: its extent is unproven, and addressing it as a broadcast would read
            // past its end.
            let source_proves_vector = def_of(cursor).is_some_and(|d| {
                if flat {
                    // ⭐⭐⭐ THE FLAT FORM'S PROOFS, IN THE FLAT FRAME. `Col`: the proof is a
                    // COUNT, not an extent — a flat `Col` source is a full tile whose FLAT
                    // elements are the per-row scalars (a `[256, 4]` running max is the
                    // `[1024, 1]` column vector of the flat frame, and extent-1 on its own
                    // trailing axis is the wrong question about it), so the count is the right
                    // one: exactly one source element per result row, which is what the
                    // consumer's `[numel, 1]` presentation addresses. `Mb`: the one-row fact —
                    // the same law the check above enforces, stated for a source the walk-back
                    // reached without an expand.
                    super::lower_ktir_to_superdsc::shape_2d(d).is_some_and(|(r, c)| match axis {
                        BcastAxis::Col => {
                            let want_rows = bc.attributes.iter().find_map(|(k, v)| match (k, v) {
                                (AttrKey::Shape, Attr::IntList(s)) if s.len() > 2 => {
                                    Some(s[..s.len() - 1].iter().product::<i64>())
                                }
                                _ => None,
                            });
                            want_rows.is_some_and(|w| r as i64 * c as i64 == w)
                        }
                        BcastAxis::Mb => r == 1,
                        // Unreachable: a `Y` chain is `flat = false` by the time it gets here
                        // (the admission that mints `Y` clears the flag), and this proof block
                        // is skipped for an admitted `Y` — the row-split source is a FULL TILE
                        // by design, the opposite of a vector, so `false` is the honest arm.
                        BcastAxis::Y(_) => false,
                    })
                } else {
                    d.op_type == OpKind::KtdpLoad
                        && super::lower_ktir_to_superdsc::shape_2d(d).is_some_and(|(r, c)| {
                            match axis {
                                BcastAxis::Mb => r == 1,
                                BcastAxis::Col => c == 1,
                                // Unreachable for the same reason the flat-proof arm above
                                // states: an admitted `Y` skips this whole block.
                                BcastAxis::Y(_) => false,
                            }
                        })
                }
            });
            if !source_proves_vector {
                return err(format!(
                    "{}: a `linalg.broadcast` sprays along {axis:?} but its source is not reached \
                     through a `tensor.expand_shape` that inserts a DEGENERATE dim, nor is it a load \
                     whose own window has extent 1 on that axis{} Either is the thing that says the \
                     source IS a row or column vector, so without one the operand's extent is \
                     unproven and addressing it as a broadcast would read past its end.",
                    f.name,
                    if flat {
                        " (for the rank > 2 flat form: nor does its own proof hold — `Col` needs \
                         one source element per result row, `Mb` needs a one-flat-row source)"
                    } else {
                        ""
                    }
                ));
            }
        }
        for v in chain {
            consumed.insert(v);
        }
        map.insert(
            res,
            Bcast {
                // ⭐⭐⭐ THE CONSUMER BINDS THE OUTERMOST CROSSED CONVERT'S RESULT when the walk
                // crossed one — the minted intermediate that convert emits into — and the
                // final cursor otherwise. The extent proofs above read the CURSOR (the load)
                // either way: the convert is traversed to PROVE the extent, never to bind. A
                // consumer binding the load would read the convert's OPERAND with the
                // consumer's own format — the raw-encoding defect the round-3 card leg
                // measured (see the convert arm).
                src: crossed_convert.unwrap_or(cursor),
                axis,
                flat,
            },
        );
    }
    Ok((map, consumed))
}

/// ⭐⭐⭐ THE FLAT FRAME'S PRESENTATION OF A BROADCAST SOURCE'S [`Region`] — the same bytes at
/// the same offsets, addressed as the flat frame states them.
///
/// A rank > 2 broadcast's source is a full tile (the `[256, 4]` running max of a
/// `[256, 4, PAGE_SLOTS]` score tile), and the flat frame's law is that a rank-N tile and its
/// `[prod(leading..=second-last), last]` flattening are the SAME BYTES: element `(b, g)` at flat
/// index `b·G + g` is flat row `b·G + g`'s scalar. So the operand the consumer addresses — one
/// element per flat row (`Col`) or one flat row of the result's width (`Mb`) — is the source
/// region with every axis but the last folded into rows:
///
/// - `Col`: the window presents `[window_numel, 1]`, the view `[view_numel, 1]`, and the window's
///   flat start `r_start·v_cols + c_start` becomes the row start. The emission-side checks
///   (the walk's extent check, `elementwise_broadcast`'s want check, `In::col`) then read a
///   region whose extents are the vector's own, with no rank-3 arm anywhere downstream.
/// - `Mb`: the source is ONE flat row (proven by [`program_broadcast_chains`]'s post-walk-back
///   check before this presentation is ever reached), so it presents `[1, window_numel]` with
///   the flat start on the column axis.
///
/// ⛔ THE STARTS ARE THE FLAT ELEMENT OFFSETS, and a nonzero one keeps the emitters' own
/// base-address law: they refuse a nonzero row start by name rather than addressing row 0, so a
/// parameter-backed flat source whose window starts off-base is refused downstream — named, not
/// mis-addressed. A minted intermediate starts at 0 and is the kernel's own case.
///
/// ⭐⭐⭐ CHECKED, NOT ASSUMED — the presentation is a RE-FRAME of the source's device frame, so
/// it goes through [`crate::sdsc_abstract::RegionFrame`]: the `Col` form is the re-row
/// `[numel, 1]` (byte-faithful exactly at a one-column source, where it is the identity) and the
/// `Mb` form is the contiguous-read presentation `[1, numel]` (byte-faithful exactly at a
/// one-row or single-stick source). A multi-column source presented `[numel, 1]` would read
/// element `f = r·cols + c` at stick `f` while the buffer holds it at `r·lanes + c` — wrong for
/// every row past the first — so it is an `Err` HERE, at the reframe, rather than a clean bake
/// of wrong bytes. The `Y` axis presents nothing (its admission clears the flat flag), and its
/// arm is the identity for exactly that reason.
fn presented_flat(r: &Region, axis: BcastAxis) -> Result<Region, String> {
    // THE SOURCE VIEW'S OWN FRAME — the byte law is decided on the frame the mint reserved,
    // before any presentation field is touched.
    let frame = crate::sdsc_abstract::RegionFrame::stick_blocked(
        r.v_rows as usize,
        r.v_cols as usize,
        crate::superdsc_opspec::Df::Fp16,
    );
    let presented = match axis {
        BcastAxis::Col => frame.flat_col()?,
        // ⭐ ONE WHOLE ROW OF A MULTI-ROW, MULTI-STICK VIEW is already the row vector — the rank-2
        // `Mb` window the emitter sprays one stick column at a time when its sticks sit a
        // row-block apart (`elementwise_broadcast`'s strided row vector). No flat row presents
        // it. A ONE-stick view's row is one contiguous stick, which the flat row below presents.
        BcastAxis::Mb
            if r.r_len == 1
                && r.c_start == 0
                && r.c_len == r.v_cols
                && r.v_cols > crate::superdsc_opspec::Df::Fp16.elems_per_stick() =>
        {
            return Ok(*r);
        }
        BcastAxis::Mb => frame.flat_mb()?,
        // The row-split mode addresses the source's OWN `[mb, out]` region — the presentation
        // is the identity, and this arm exists only because the match is exhaustive.
        BcastAxis::Y(_) => return Ok(*r),
    };
    let view_numel = r.v_rows as u64 * r.v_cols as u64;
    let window_numel = r.r_len as u64 * r.c_len as u64;
    let start = r.r_start as u64 * r.v_cols as u64 + r.c_start as u64;
    debug_assert_eq!(
        (presented.rows as u64, presented.cols as u64),
        match axis {
            BcastAxis::Col => (view_numel, 1),
            BcastAxis::Mb => (1, view_numel),
            BcastAxis::Y(_) => (r.v_rows as u64, r.v_cols as u64),
        },
        "the presented frame and the presented Region state the same shape"
    );
    let mut f = *r;
    match axis {
        BcastAxis::Col => {
            f.v_rows = view_numel as u32;
            f.v_cols = 1;
            f.r_len = window_numel as u32;
            f.c_len = 1;
            f.r_cover = (0, view_numel as u32);
            f.c_cover = (0, 1);
            f.r_start = start as u32;
            f.c_start = 0;
        }
        BcastAxis::Mb => {
            f.v_rows = 1;
            f.v_cols = view_numel as u32;
            f.r_len = 1;
            f.c_len = window_numel as u32;
            f.r_cover = (0, 1);
            f.c_cover = (0, view_numel as u32);
            f.r_start = 0;
            f.c_start = start as u32;
        }
        BcastAxis::Y(_) => {}
    }
    Ok(f)
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
    let Some(r) = region_of_view_operand(k, tile, 0)? else {
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
    // parameter's Region here. The tile's corner operands (1 = row, 2 = column; a RANK-1 tile
    // states its one corner at operand 1, its single axis) and its `Shape` are read by exactly the
    // code `regions()` reads them with (`shape_2d` and `consts.get`), so the two walks cannot
    // disagree about one tile. A tile whose corners are NOT constants names a window no descriptor
    // can carry, and is refused by name rather than silently falling back to the parameter's first
    // tile — the same discipline as `regions()`'s in-loop guard.
    Ok(Some(windowed_region(k, r, tile)?))
}

/// ⭐⭐⭐⭐⭐ ONE ACCESS TILE's WINDOW, RESOLVED ONTO ITS PARAMETER's REGION — the walk the load side
/// ([`region_for_operand`]) and the store side of [`lower_function`] share, so the two cannot
/// disagree about what one `ktdp.construct_access_tile` states.
///
/// A tile with CONSTANT corners states its window directly: `r_start`/`c_start` the constants,
/// `r_len`/`c_len` the tile's own `Shape`. The COVER stays the parameter walk's in that case — a
/// constant-corner program's windows are what `regions()`'s fold already spans, and restating them
/// here would change the row basis every existing body derives from that fold.
///
/// A tile whose ROW corner is not a constant may still state a window this door can address: the
/// PER-CORE UNIFORM `stride · core_id + base` over the launch's own positions — the spelling
/// triton-ktir's `uniform_grid_corners` mints and the form the folded configurations already ship.
/// The launch's positions place core `c` at work item `c`, so when the corner provably computes
/// `base + c · stride` with `stride` the tile's own row height, the per-core tiles TILE THE BAND
/// `[base, base + positions · stride)` flush, and the band is one window a descriptor can sweep
/// with the work division carrying the per-core term — the corner is REDUNDANT with the division,
/// whatever the base. That is THE BAND CONTRACT, and it resolves the window as the BAND:
/// `r_start = base`, `r_len = positions · stride` — not the tile's height; the lift is the point,
/// `pointwise_rows`' windowed case and `matmul_oriented`'s `m = out.r_len` then sweep exactly the
/// band — and `r_cover = (base, base + positions · stride)`, the statement `regions()` cannot
/// make: its cover fold reads constants, so a uniform corner falls back to the whole view, which a
/// band trip is not. The strict single-trip form (`base == 0`, `positions · stride == v_rows`) is
/// the special case where the band IS the view and the region is the whole-view window.
///
/// ⛔ THE COLUMN CORNER STAYS CONSTANT-ONLY, deliberately. The work division carries ROW terms; a
/// computed column corner is not redundant with anything, and resolving it at the base would
/// address the wrong stick on every core past the first. (UnEQUAL constant column corners are
/// per-operand windows — the dense pointwise emission's own law — and ride this same region.)
///
/// ⛔⭐ AND A RANK-1 TILE's ONE CORNER IS A COLUMN CORNER — with one family the `out`-dividing
/// descriptors DO carry. A `[n]` view is one row of `n` columns in this door's 2-D spelling
/// ([`shape_2d`]), so the single corner a rank-1 tile states (`[64]` over `[1024]`, the
/// `mul_scale_small` row's `pid * BLOCK`) is the per-core term along the COLUMN axis. The FLUSH
/// form of it — base 0, the launch's positions times the tile's own extent exactly the view's
/// columns — is the one a descriptor division states: the scalar-weighted bodies split `out`
/// across the launch's cores (`numWkSlicesPerDim_` `out: positions`, MEASURED on the
/// `mul_scale_small` scalarmul bake: 16 cores × 64 columns of the `[1, 1024]` view), so the
/// corner is REDUNDANT with that division exactly as a row band is redundant with the row
/// division, and the window resolves as the WHOLE VIEW — the parameter fold's own answer for a
/// corner it cannot read, now PROVEN flush rather than silently fallen back to. Anything else —
/// a nonzero base, a stride that leaves gaps, a band the view does not bound — is carried by no
/// division this door states, and is refused by name below. A CONSTANT rank-1 corner states an
/// ordinary window and resolves like any other: the tile's own extent at the stated offset.
///
/// ⛔ AND THE CONTRACT IS EXACT, NOT BEST-EFFORT. A stride that is not the tile's own row height
/// (per-core tiles with gaps — the band would hold rows no core computes), a base below zero or a
/// band past the view's end, a zero position count, or a corner that is not uniform at all is
/// refused by name below. Everything refused here keeps the clone protocol.
fn windowed_region(k: &KtirNode, mut r: Region, tile: &Operation<'_>) -> Result<Region, Error> {
    let f = &k.func;
    let consts = super::lower_ktir_to_superdsc::index_constants(f);
    let Some((tr, tc)) = super::lower_ktir_to_superdsc::shape_2d(tile) else {
        return err(format!(
            "{}: the access tile states no 2-D `shape`, so its window is unknown",
            f.name
        ));
    };
    let uniform = super::lower_ktir_to_superdsc::uniform_corners(f, &consts);
    let (gx, gy, gz) = f.grid;
    let positions = (gx * gy * gz) as i64;
    // ⭐ A RANK-1 TILE STATES ONE CORNER, ITS SINGLE AXIS. A constant is an ordinary window at
    // the tile's own extent; the FLUSH per-core uniform is the whole view the `out`-dividing
    // bodies slice (see the doc); anything else computed is refused by name. A tile stating NO
    // corner addresses the whole view, which is the region already in hand.
    if tile.operands.len() < 3 {
        if tile.operands.len() == 2 {
            let corner = tile.operands[1];
            match consts.get(&corner).copied() {
                Some(cs) => {
                    r.r_start = 0;
                    r.r_len = tr;
                    r.c_start = cs as u32;
                    r.c_len = tc;
                }
                None => {
                    let flush = uniform.get(&corner).is_some_and(|u| {
                        u.base == 0
                            && u.stride > 0
                            && u.stride as u32 == tc
                            && positions > 0
                            && positions as u64 * u.stride as u64 == r.v_cols as u64
                    });
                    if !flush {
                        return err(format!(
                            "{}: the access tile {:?} is RANK-1 and its one corner is computed \
                             in a form no descriptor division carries. The single axis is a \
                             COLUMN in this door's 2-D spelling (`shape_2d` reads `[n]` as one \
                             row of `n` columns), and the one carried form is the FLUSH band: \
                             base 0, the launch's positions times the tile's own extent exactly \
                             the view's columns -- the `out`-dividing bodies then slice exactly \
                             the per-core tile the corner states (`mul_scale_small`'s \
                             `pid * BLOCK` over `[1024]` is the measured form). A nonzero base, \
                             a stride that leaves gaps, or a band the view does not bound is not \
                             carried by any division this door states, and resolving it at the \
                             base would hand every core the whole axis. Hoist or unroll the \
                             corner before the whole-function door can address it.",
                            f.name, tile.result,
                        ));
                    }
                    // The flush band IS the view: the region in hand is already the whole-view
                    // window, and the division carries the per-core slice.
                }
            }
        }
        return Ok(r);
    }
    let row = tile.operands.get(1).copied();
    let col = tile.operands.get(2).copied();
    let cs = col
        .and_then(|s| consts.get(&s).copied())
        .map(|cs| cs as u32);
    // The row corner: a CONSTANT states one window at the tile's own height; the per-core
    // uniform states the BAND the launch's own tiles tile. Anything else names a window no
    // descriptor can carry.
    let window = match row.and_then(|s| consts.get(&s).copied()) {
        Some(rs) => Some((rs as u32, tr)),
        // ⭐ A TILE AS TALL AS ITS VIEW SITS AT ROW 0 whatever its corner computes: any other corner
        // reads rows past the view. spyre-inference's prompt kernel under Triton's gather floor
        // reads its `[P, D]` query as the one tile at `query_row_index[0]` over the `[P, D]` view.
        None if tr == r.v_rows => Some((0, tr)),
        None => row.and_then(|s| {
            let u = uniform.get(&s)?;
            let band = positions.checked_mul(u.stride)?;
            (u.stride > 0
                && u.base >= 0
                && u.stride as u32 == tr
                && positions > 0
                && u.base + band <= r.v_rows as i64)
                .then_some((u.base as u32, band as u32))
        }),
    };
    let (Some((rs, r_len)), Some(cs)) = (window, cs) else {
        return err(format!(
            "{}: the access tile {:?} states a window whose corner is not a constant \
             (`ktdp.construct_access_tile` operands 1/2), so the descriptor cannot carry the \
             offset. A loop-carried or computed corner needs hoisting or unrolling before the \
             whole-function door can address it -- or, for the row corner, the per-core uniform \
             spelling whose launch tiles one band flush (stride the tile's row height, the band \
             fitting the view), which this door resolves as that band.",
            f.name, tile.result,
        ));
    };
    r.r_start = rs;
    r.c_start = cs;
    r.r_len = r_len;
    r.c_len = tc;
    // THE BAND IS THE COVER the uniform arm states (see the doc); the constant arm keeps the
    // parameter walk's fold.
    if row.is_some_and(|s| !consts.contains_key(&s)) {
        r.r_cover = (rs, rs.saturating_add(r_len));
        // ⭐⭐⭐ AND THE TYPED BAND RIDES THE REGION — the flush band the launch's own
        // tiles tile. This is the ONE construction site in the crate (`RowBand::flush`
        // is `pub(crate)` and named for this arm): the band can equal the whole view
        // (base 0, band == rows — the batched sampler's `pid`-cornered `[1, ·]` tiles
        // over `[B, ·]`), which no extent test can tell from an ordinary whole-view
        // read, so the mint reads it back as a TYPE. See [`RowBand`].
        r.band = Some(RowBand::flush(tr, positions as u32));
    }
    Ok(r)
}

/// The rows a banded op's minted intermediate must hold: a WINDOWED input's band (`r_len`), or
/// `logical` when every input is whole-view. See the mint's own doc for the full argument.
///
/// ⭐⭐⭐ A FLUSH BAND COUNTS EVEN WHEN IT EQUALS THE VIEW — that is the whole reason
/// [`RowBand`] is typed. A `pid`-cornered `[1, ·]` tile over a `[B, ·]` view resolves as
/// the band `[0, B)` — shape-identical to a whole-view read, invisible to the
/// `r_cover != (0, v_rows)` test above — so the typed field is the one witness, and the
/// mint must size to the band (`[B, ·]`) exactly as rope's mint sizes to `[1024, ·]`.
fn band_rows_of(ins: &[Region], logical: u32) -> u32 {
    ins.iter()
        .find(|x| x.r_cover != (0, x.v_rows) || x.band.is_some())
        .map_or(logical, |x| x.r_len)
}

/// The ONE band a set of inputs states, refusing disagreement: an op sweeps one band,
/// and a mint between two differently-banded inputs would size to whichever it saw
/// first. `None` when no input is banded — the ordinary case, byte-identical to before.
fn band_of(name: &str, ins: &[Region]) -> Result<Option<RowBand>, Error> {
    let mut stated: Option<RowBand> = None;
    for x in ins {
        if let Some(b) = x.band {
            match stated {
                None => stated = Some(b),
                Some(prev) if prev == b => {}
                Some(prev) => {
                    return err(format!(
                        "{name}: the op reads inputs banded differently ({prev:?} vs {b:?}) — \
                         one op sweeps one band, and a mint between them would size to whichever \
                         it saw first"
                    ));
                }
            }
        }
    }
    Ok(stated)
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

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  THE ROW-TILED FAMILY FOLD — the door's own `Region::r_cover` law, restated for the walk.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// The KIND discriminant of a row-fold candidate — one variant per foldable pointwise family,
/// carrying everything "same op" means for it. Derived exactly as [`lower_function`]'s loop
/// derives `program`, so a member the loop would classify differently can never be folded in.
#[derive(Clone, Copy, PartialEq, Debug)]
enum RowFoldKind {
    /// A recognised silu-mul chain's terminal multiply — the chain recognizer already proved every
    /// member's interior is the same four-op silu.
    SiluMul,
    /// A recognised BARE silu chain's terminal divide — the longhand stored direct, one input. A
    /// separate kind from [`RowFoldKind::SiluMul`] because the two differ in ARITY even where a
    /// terminal's op kind does not: one folded descriptor stands for one, never a mix.
    BareSilu,
    /// A bare pointwise op, discriminated by its own `OpKind` (an `arith.addf` is not an
    /// `arith.mulf`, and a mulf the splat/bound readers claim is not bare — see [`classify_row_fold`]).
    Elementwise(OpKind),
    /// `tile · const` — the multiplier is part of the kind: two blocks of one scalarmul carry the
    /// same scale, and a family whose blocks did not would compute different functions per block.
    ScalarMul(f32),
    /// `tile · loaded [1,1] parameter` — the same launch-bound multiplier.
    BoundScalarMul,
}

/// The pre-walk analyses the row fold's helpers read — the recognizers' outputs, threaded as ONE
/// value so the helpers stay under the argument budget without `#[allow]`. Built once in
/// [`lower_function`], after the recognizers run and before the op loop.
struct FoldAnalyses<'a> {
    silus: &'a [SiluMulChain],
    rmsnorms: &'a [RmsNormChain],
    gathered_loads: &'a std::collections::HashMap<Ssa, usize>,
}

/// Classify one terminal op the way [`lower_function`]'s loop does, reduced to the foldable
/// families. `None` is every kind the fold does not own — an rmsnorm chain's gain multiply (the
/// reduce inside it changes extent under a merge), a matmul (its trips are the time-tiling
/// machinery's), a transpose, anything unrouted.
fn classify_row_fold(
    k: &KtirNode,
    op: &Operation<'static>,
    analyses: &FoldAnalyses<'_>,
) -> Option<RowFoldKind> {
    let f = &k.func;
    let FoldAnalyses {
        silus, rmsnorms, ..
    } = analyses;
    // The loop's own precedence, in the loop's own order: chain terminals first (silu, then rms —
    // an rms terminal is NOT foldable), then the mulf tiebreaks, then the 1:1 map.
    if op.op_type == OpKind::ArithMulf {
        if silus.iter().any(|c| Some(c.mul) == op.result) {
            return Some(RowFoldKind::SiluMul);
        }
        if rmsnorms.iter().any(|c| op.result == Some(c.out)) {
            return None;
        }
        if let Some((v, _)) = splat_scale_of(f, op) {
            return Some(RowFoldKind::ScalarMul(v));
        }
        if let Ok(Some(_)) = bound_scale_of(k, op) {
            return Some(RowFoldKind::BoundScalarMul);
        }
        return Some(RowFoldKind::Elementwise(OpKind::ArithMulf));
    }
    // A BARE silu's terminal is the longhand's own `arith.divf` — the chain reader proved the five
    // ops and the store, so the family kind is the chain's, never the divide's. The 1:1 map would
    // otherwise call it `Elementwise(RealDiv)` and fold it as one, which would address the
    // chain-interior `1 + e^-x` operand as a second input.
    if silus
        .iter()
        .any(|c| c.up.is_none() && Some(c.mul) == op.result)
    {
        return Some(RowFoldKind::BareSilu);
    }
    match program_of(op.op_type) {
        Some(Program::Elementwise(_)) => Some(RowFoldKind::Elementwise(op.op_type)),
        _ => None,
    }
}

/// The parameter a `ktdp.store` writes — the tile→view→pointer chain the walk's own out path
/// walks, stated once so the family analysis cannot reach a different parameter than the emission
/// does.
fn store_ptr(f: &IRFunction<'static>, store: &Operation<'_>) -> Option<Ssa> {
    let tile = f.operations.iter().find(|o| {
        o.op_type == OpKind::KtdpConstructAccessTile && o.result == store.operands.get(1).copied()
    })?;
    let view = f.operations.iter().find(|o| {
        o.op_type == OpKind::KtdpConstructMemoryView && o.result == tile.operands.first().copied()
    })?;
    view.operands.first().copied()
}

/// A store's own row window — the tile's `Shape` height and its row-corner constant, read with
/// the same two readers (`shape_2d`, `index_constants`) [`region_for_operand`] reads a load's
/// tile with, so the two walks cannot disagree about one tile.
fn store_rows(
    f: &IRFunction<'static>,
    store: &Operation<'_>,
    consts: &std::collections::HashMap<Ssa, i64>,
) -> Option<(u32, u32)> {
    let tile = f.operations.iter().find(|o| {
        o.op_type == OpKind::KtdpConstructAccessTile && o.result == store.operands.get(1).copied()
    })?;
    let (r_len, _) = super::lower_ktir_to_superdsc::shape_2d(tile)?;
    let r_start = tile.operands.get(1).and_then(|s| consts.get(s).copied())? as u32;
    Some((r_start, r_len))
}

/// ⭐⭐⭐⭐⭐ THE ROW-TILED FAMILY FOLD. A row-blocked pointwise program — the kernels' own
/// `EW_LX_ELEMS` law: `silumul` always at prefill, elementwise and scalarmul above the budget —
/// states one load/chain/store family per `[BLOCK_M, N]` block, unrolled at constexpr trip counts.
/// The DOOR folds those through `Region::r_cover` into ONE descriptor spanning the view; the walk
/// lowers per-op at the LOAD'S OWN tile (the multi-trip matmul fix), so every block past the first
/// refuses at the row-corner law — and the FIRST block already emits the fold, because its windows
/// sit at row 0 and the emission's row count comes from the out region's cover.
///
/// So the fold here is the DEDUP plus a seal the door never needed: the family's members are
/// skipped (the leader's descriptor computed their rows — one op, time factor 1, no tiling of our
/// own), and the membership is proven to be ONE op over ONE window family that TILES THE VIEW
/// EXACTLY — same kind, same parameters, same column window, row windows with no gap and no
/// overlap, covering `0..view rows`. A family that almost tiles REFUSES BY NAME, never a partial
/// merge: the leader's descriptor spans the whole view, so a gap would leave rows the program
/// states unwritten and an overlap would state one row twice.
///
/// # NOT CANDIDATES, AND WHY
///
/// * GATHERED members (an unrolled embedding sweep's trips) return `Ok(None)` — their trips are
///   gather semantics, each reading its own indirect tile, and merging them would strand the
///   program's other tiles at the walk's `gather_carried` check. They keep today's per-trip
///   lowering.
/// * A family whose stores disagree about kind or parameters is not a family — the key does not
///   match, nothing merges, and the members lower (and refuse, loudly) as today.
fn row_tiled_family(
    k: &KtirNode,
    leader: &Operation<'static>,
    leader_kind: RowFoldKind,
    leader_in: &[Ssa],
    leader_n_in: usize,
    analyses: &FoldAnalyses<'_>,
) -> Result<Option<std::collections::HashSet<Ssa>>, Error> {
    let f = &k.func;
    let FoldAnalyses {
        silus,
        rmsnorms: _,
        gathered_loads,
    } = analyses;
    let consts = super::lower_ktir_to_superdsc::index_constants(f);
    // The leader's own store — a family is keyed on the parameter its result is stored to. An
    // unstored terminal is an intermediate chain, not a row-block family.
    let Some(leader_res) = leader.result else {
        return Ok(None);
    };
    let Some(leader_store) = f.operations.iter().find(|s| {
        s.op_type == OpKind::KtdpStore && s.operands.first().copied() == Some(leader_res)
    }) else {
        return Ok(None);
    };
    let Some(out_ptr) = store_ptr(f, leader_store) else {
        return Ok(None);
    };
    // EVERY store to that parameter, with its own row window — the family's full membership. ⛔
    // FAIL-CLOSED ON THE CHAIN: a `ktdp.store` whose tile→view→parameter chain does not resolve
    // cannot be classified as in or out of the family, and a membership that might be incomplete
    // is a partial merge waiting to happen — so it refuses by name rather than guessing. Every
    // producer's stores resolve (the walk's own out path reads the same chain for every op it
    // lowers), so this names a producer form, not a working program.
    let mut members: Vec<(&Operation<'static>, u32, u32)> = Vec::new();
    for s in f
        .operations
        .iter()
        .filter(|s| s.op_type == OpKind::KtdpStore)
    {
        let Some(p) = store_ptr(f, s) else {
            return err(format!(
                "{}: a `ktdp.store` states no tile→view→parameter chain, so the row-fold cannot \
                 tell whether it writes the folded output — and a membership that might be \
                 incomplete is a partial merge. The store's own op refuses the same chain when \
                 the walk reaches it",
                f.name
            ));
        };
        if p != out_ptr {
            continue;
        }
        let Some((rs, rl)) = store_rows(f, s, &consts) else {
            return err(format!(
                "{}: a `ktdp.store` to the row-folded output states no constant row window \
                 (tile corner/shape), so the fold cannot prove the family tiles the view",
                f.name
            ));
        };
        members.push((s, rs, rl));
    }
    if members.len() < 2 {
        return Ok(None);
    }
    // ⭐ ONE ROW WINDOW SHARED BY EVERY STORE IS NOT A ROW FAMILY. A column-chunked program states
    // several `[m, chunk]` windows at the SAME rows, and the walk's per-op lowering already serves
    // each chunk through its own column corner (`col_of`) — folding there would be wrong (the
    // blocks tile the COLUMN axis, which the row fold does not own) and refusing would break a
    // shape that works. So identical row windows bail to today's behavior; only a family whose
    // stores state SEVERAL row windows — the `EW_LX_ELEMS` row-blocking signature — is analyzed.
    if members
        .iter()
        .all(|(_, s, l)| (*s, *l) == (members[0].1, members[0].2))
    {
        return Ok(None);
    }

    // The leader's input regions — the tids and column windows every member must share. A value
    // the parameter walk cannot reach (an intermediate this function computed, a constant) means
    // the family is not one op over the parameters, and the loop's own resolution handles the op
    // exactly as it did before the fold existed — so this is a BAIL, not a refusal.
    let mut leader_regions: Vec<Region> = Vec::with_capacity(leader_in.len());
    for v in leader_in {
        let Some(r) = region_for_operand(k, *v)? else {
            return Ok(None);
        };
        leader_regions.push(r);
    }

    // Classify every member's terminal and collect its input values, the way the loop would.
    let mut terminals: Vec<(&Operation<'static>, RowFoldKind, Vec<Ssa>)> = Vec::new();
    for (store, _, _) in &members {
        let Some(v) = store.operands.first().copied() else {
            return err(format!(
                "{}: a `ktdp.store` to the row-blocked output states no stored value",
                f.name
            ));
        };
        let Some(term) = f.operations.iter().find(|o| o.result == Some(v)) else {
            return err(format!(
                "{}: the value a `ktdp.store` writes is produced by no top-level op",
                f.name
            ));
        };
        let Some(kind) = classify_row_fold(k, term, analyses) else {
            return err(format!(
                "{}: t{}'s row-blocked family mixes kinds — the walk folds ONE pointwise op \
                 whose blocks tile the view, and a store to the same parameter from a different \
                 kind ({:?}) is a program shape the fold refuses rather than partially merges",
                f.name, leader_regions[0].tid, term.op_type
            ));
        };
        if kind != leader_kind {
            return err(format!(
                "{}: the stores over t{} are not one op family — {leader_kind:?} at one store and \
                 {kind:?} at another. The fold merges one kind's blocks or nothing, never a mix",
                f.name, leader_regions[0].tid,
            ));
        }
        let in_values: Vec<Ssa> = match kind {
            RowFoldKind::SiluMul | RowFoldKind::BareSilu => {
                let Some(c) = silus.iter().find(|c| Some(c.mul) == term.result) else {
                    return err(format!(
                        "{}: a silu family member whose terminal is not a recognised chain",
                        f.name
                    ));
                };
                // `classify_row_fold` gave every member the leader's kind, so the chain's form
                // matches it: a silumul member states an `up`, a bare member ends at the stored
                // divide. The mismatch arm is the seal on that invariant — one folded descriptor
                // cannot stand for a family that mixes the two arities.
                match (kind, c.up) {
                    (RowFoldKind::SiluMul, Some(up)) => vec![c.gate, up],
                    (RowFoldKind::BareSilu, None) => vec![c.gate],
                    _ => {
                        return err(format!(
                            "{}: the row-block family over t{} mixes silu forms — one member's \
                             chain ends at the stored divide and another's at the `up` multiply, \
                             and the two are different ops even where their interiors are the same \
                             four-op silu",
                            f.name, leader_regions[0].tid
                        ));
                    }
                }
            }
            RowFoldKind::Elementwise(_) => {
                term.operands.iter().copied().take(leader_n_in).collect()
            }
            RowFoldKind::ScalarMul(_) => {
                let Some((_, t)) = splat_scale_of(f, term) else {
                    return err(format!(
                        "{}: a scalar-multiply family member whose multiplier is not a splat",
                        f.name
                    ));
                };
                vec![t]
            }
            RowFoldKind::BoundScalarMul => {
                let Ok(Some(b)) = bound_scale_of(k, term) else {
                    return err(format!(
                        "{}: a bound-scale family member with no bound scale",
                        f.name
                    ));
                };
                vec![b.tensor]
            }
        };
        // ⛔ THE GATHER EXCLUSION — before any tiling arithmetic, because a gathered family's
        // windows can tile exactly and the merge would still be wrong (see the doc above).
        if in_values.iter().any(|v| gathered_loads.contains_key(v)) {
            return Ok(None);
        }
        terminals.push((term, kind, in_values));
    }

    // ⛔⛔⛔ ONE OP, ONE WINDOW FAMILY, TILING THE VIEW EXACTLY. Per operand position: every
    // member reads the SAME parameter at the SAME column window, and the row windows — sorted —
    // start at 0, abut with no gap and no overlap, and end exactly at the view's rows. The stores
    // are held to the same law. A family that fails this REFUSES; a partial merge would span rows
    // the program never states.
    let check_partition = |name: &str,
                           intervals: &mut Vec<(u32, u32)>,
                           view_rows: u32,
                           what: &str|
     -> Result<(), Error> {
        intervals.sort_unstable();
        let mut cursor = 0u32;
        for &(s, len) in intervals.iter() {
            if s != cursor {
                return err(format!(
                    "{}: {} {} row windows {}..{} where row {cursor} was due — the family's \
                     windows must TILE the view exactly (start at 0, abut, end at {view_rows}); a \
                     gap or an overlap means the one folded descriptor would leave stated rows \
                     unwritten or state one row twice. The fold refuses rather than partially merge",
                    f.name,
                    name,
                    what,
                    s,
                    s + len
                ));
            }
            cursor = s + len;
        }
        if cursor != view_rows {
            return err(format!(
                "{}: {} {} row windows cover 0..{cursor} of a {view_rows}-row view — the family's \
                 windows must TILE the view exactly; the one folded descriptor spans all \
                 {view_rows} rows and a descriptor computing rows the program never states is the \
                 defect the fold exists to prevent",
                f.name, name, what
            ));
        }
        Ok(())
    };
    // Per operand position: same tid, same column window, row partition.
    for (pos, want) in leader_regions
        .iter()
        .enumerate()
        .take(leader_n_in.min(leader_regions.len()))
    {
        let mut intervals: Vec<(u32, u32)> = Vec::new();
        for (term, _, in_values) in terminals.iter() {
            let Some(v) = in_values.get(pos).copied() else {
                return err(format!(
                    "{}: a row-block family member ({:?}) resolves fewer inputs than its leader",
                    f.name, term.op_type
                ));
            };
            let Some(r) = region_for_operand(k, v)? else {
                return err(format!(
                    "{}: a row-block family member's input {:?} resolves to no parameter — the \
                     fold merges one op over the SAME parameters, never a mix of loads and \
                     computed values",
                    f.name, v
                ));
            };
            if r.tid != want.tid {
                return err(format!(
                    "{}: the row-block family over t{} reads two different tensors at input \
                     {pos} (t{} and t{}) — one op, one window family; a merge across parameters \
                     is a partial merge and refuses",
                    f.name, want.tid, want.tid, r.tid
                ));
            }
            if r.c_start != want.c_start || r.c_len != want.c_len {
                return err(format!(
                    "{}: the row-block family over t{} reads input {pos} at two different column \
                     windows ([{}, {}] and [{}, {}]) — the blocks of one op share one column \
                     window, and unequal corners would address one block's columns at another's",
                    f.name, want.tid, want.c_start, want.c_len, r.c_start, r.c_len
                ));
            }
            intervals.push((r.r_start, r.r_len));
        }
        check_partition(
            &format!("t{}", want.tid),
            &mut intervals,
            want.v_rows,
            "loads",
        )?;
    }
    // The stores, held to the same law against the output view's own rows — read the way the
    // walk's own out path reads it (the parameter's row in `regions()`), so the fold and the
    // emission cannot disagree about the view's extent.
    let out_rows = {
        let Some(out_idx) = f
            .arguments
            .iter()
            .position(|(a, _)| Some(*a) == Some(out_ptr))
        else {
            return err(format!(
                "{}: the row-folded output's store names no parameter",
                f.name
            ));
        };
        let all = regions(k)?;
        let Some(r) = all.get(out_idx) else {
            return err(format!(
                "{}: the row-folded output parameter has no region, though its store reached it",
                f.name
            ));
        };
        r.v_rows
    };
    let mut store_intervals: Vec<(u32, u32)> =
        members.iter().map(|(_, rs, rl)| (*rs, *rl)).collect();
    check_partition("the output", &mut store_intervals, out_rows, "stores")?;

    // The LEADER is the terminal this analysis was called on — the first family terminal in
    // program order, because the walk visits ops in program order and fires the gate at the first
    // one it can lower. Every other member is skipped, its rows computed by the leader's one
    // folded descriptor.
    let mut skip = std::collections::HashSet::new();
    for (term, _, _) in terminals.iter() {
        if let Some(r) = term.result
            && r != leader_res
        {
            skip.insert(r);
        }
    }
    Ok(Some(skip))
}

/// Lower a function holding MANY ops, one `Program` per op, in program order.
///
/// Each op is handed only its own operands' regions, so the existing bodies apply unchanged.
///
/// `headmajor` is the head-major o_proj handoff fact ([`HeadmajorOproj`]), threaded from the door
/// that owns the bundle's attention facts — the swap keys on the o matmul's weight tid, so of the
/// many matmuls an unrolled graph holds, exactly the one whose `w.tid` the fact names takes it.
/// `None` is the no-handoff emission every caller without a bundle fact gets.
///
/// # WHAT IS NOT IMPLEMENTED YET, AND IS REFUSED RATHER THAN GUESSED
///
/// An INTERMEDIATE — a result one op produces and the next consumes, which no `ktdp.store` writes —
/// needs a buffer of its own, declared to [`BundleLayout::synth`] so `resolve_seg_base` can place
/// it. That declaration is the caller's (it owns the layout), so this refuses and names the value
/// rather than inventing an address. A CONSTANT operand (a `tensor.splat`) is refused for the same
/// reason: `Program::ScalarMul` is the door for "times a stated constant" and choosing it here would
/// be picking a node kind on the producer's behalf.
/// Mint the tid for an intermediate of `ssa` — BUNDLE-unique, and REPLAYED when this node's
/// lowering runs twice into the same layout.
///
/// The declare pass and the real pass ([`lower_subtile_tape_to_ktir`]'s two visits of the
/// same node) must agree on every intermediate's NAME, or the real pass's mint lands beside
/// the declare pass's buffer and a consumer sees whichever the walk visited first — the
/// DMA-past-segment defect this replay closes. The node's own record
/// ([`KtirNode::minted`](crate::ktir_node::KtirNode::minted)) maps the SSA value to the tid
/// minted for it, and the replay fires only when THIS layout confirmed it (its synth map
/// holds `act_name(tid)`), so a record carried into a different layout mints fresh instead
/// of colliding with that layout's placements. `floor` is the node-derived bound from
/// [`lower_function`] — see its law doc for why the mint is bundle-seeded, not node-seeded.
pub(super) fn mint_inter(k: &KtirNode, layout: &BundleLayout, floor: u32, ssa: Ssa) -> u32 {
    mint(k, layout, floor, crate::ktir_node::MintKey::Value(ssa))
}

/// [`mint_inter`] for the `n`th scratch the lowering of `ssa` takes beside its own buffer
/// ([`crate::ktir_node::MintKey::Scratch`]).
pub(super) fn mint_scratch(
    k: &KtirNode,
    layout: &BundleLayout,
    floor: u32,
    ssa: Ssa,
    n: u8,
) -> u32 {
    mint(k, layout, floor, crate::ktir_node::MintKey::Scratch(ssa, n))
}

fn mint(k: &KtirNode, layout: &BundleLayout, floor: u32, key: crate::ktir_node::MintKey) -> u32 {
    let mut minted = k.minted.lock().unwrap_or_else(|p| p.into_inner());
    // Declared by this layout either way: an HBM synthetic, or a core-local LX one the scratchpad
    // plan kept there ([`super::lx_plan`]).
    if let Some(&tid) = minted.get(&key) {
        let name = crate::place::act_name(tid);
        if layout.synth.borrow().map.contains_key(&name) || layout.lx_core_local(&name) {
            return tid;
        }
    }
    let tid = layout.take_inter_tid(floor);
    minted.insert(key, tid);
    tid
}

/// The node-derived FLOOR [`mint_inter`] honors: above the highest non-reserved tensor the node
/// binds (see the law at [`lower_walk`]'s use).
pub(super) fn mint_floor(k: &KtirNode) -> u32 {
    k.bindings
        .iter()
        .chain(k.mask.iter())
        .map(|b| b.get())
        .filter(|t| !crate::reserved_tids::is_reserved(*t))
        .max()
        .map_or(0, |m| m + 1)
}

/// ⭐⭐⭐ THE ARRANGEMENT PROPAGATION — which of a node's PARAMETERS the bake must stage in a
/// STAGGERED arrangement, decided ENTIRELY at bake time off the TARGET-ABI CONST TABLE.
///
/// The `dl16tofp32` widen writes its output lane-permuted when both operands are standard (the
/// measured stagger — see [`crate::element_arrangement`]). That is a fact about the OPCODE, so
/// it is stated as a const table row ([`crate::element_arrangement::OP_OUTPUT_ARRANGEMENT`])
/// and THIS pass takes the table as input: a convert's output carries whatever arrangement its
/// row states, a POINTWISE op propagates the stagger to every same-width operand and its output
/// (the device reads operand buffers lane-for-lane, so a standard-staged operand of a staggered
/// op would be read permuted — the mixing refusal below), and a REDUCTION ends it (the stagger
/// permutes within 64-element blocks; a reduction over the multiset is order-free). A
/// `[m, 1]` broadcast (`Col`) is order-free the same way — one scalar per row. Everything the
/// pass decides is a BUILD-TIME fact: the marked parameters' staged bytes are emitted already
/// permuted ([`crate::wiring`]'s builders are pure in the arrangement), and the runtime only
/// indexes. The worker decides NOTHING.
///
/// ⛔ TWO REFUSALS, BOTH BY NAME, both `cargo build` failures at expansion — the typed guard the
/// arrangement cannot reach as a handle kind:
///
/// * a staggered value meeting a COMPUTED standard one where order matters (pointwise over a
///   full-width operand that is neither staggered nor a parameter the bake can restage);
/// * a staggered value reaching a `ktdp.store` — the host read-back channel is standard-order,
///   so a stored stagger is a permuted answer by construction, refused here rather than read.
///
/// Returns the marked parameters: buffer tid → the arrangement to stage. Parameters the pass
/// never marks stay [`ElementArrangement::Standard`], and the map is EMPTY for every program
/// with no staggered convert — a decoder layer changes nothing.
pub fn staggered_sources(
    k: &KtirNode,
) -> Result<std::collections::BTreeMap<u32, crate::element_arrangement::ElementArrangement>, Error>
{
    use crate::element_arrangement::ElementArrangement;
    let f = &k.func;
    let mut arr: std::collections::HashMap<Ssa, ElementArrangement> =
        std::collections::HashMap::new();
    let mut marked = std::collections::BTreeMap::new();
    // The same chain analyses the walk reads, so the propagation and the lowering cannot
    // disagree about which op is which — the classification below is the walk's own, verbatim.
    let silus = program_silu_mul_chains(f)?;
    let rmsnorms = program_rmsnorm_chains(f)?;
    let (bcasts, bcast_consumed) = program_broadcast_chains(f)?;
    let equal_masks = program_equal_masks(f)?;

    for op in f.operations.iter() {
        if is_plumbing(op.op_type) {
            continue;
        }
        if op.result.is_some_and(|r| {
            silus.iter().any(|c| c.consumed.contains(&r))
                || rmsnorms.iter().any(|c| c.consumed.contains(&r))
                || bcast_consumed.contains(&r)
                || equal_masks.consumed.contains(&r)
        }) {
            continue;
        }
        let terminal = silus.iter().find(|c| op.result == Some(c.mul));
        let reduce = (op.op_type == OpKind::LinalgReduce)
            .then(|| reduce_kind_of(op))
            .flatten();
        let scalar = (op.op_type == OpKind::ArithMulf)
            .then(|| splat_scale_of(f, op))
            .flatten();
        let bound_scale = if op.op_type == OpKind::ArithMulf && scalar.is_none() {
            bound_scale_of(k, op)?
        } else {
            None
        };
        let rms = rmsnorms.iter().find(|c| op.result == Some(c.out));
        let equal = (op.op_type == OpKind::ArithUitofp)
            .then(|| {
                op.result
                    .and_then(|r| equal_masks.by_retype.get(&r).copied())
            })
            .flatten();
        let convert = convert_kind_of(f, op)?;
        let Some(program) = terminal
            .map(|c| {
                if c.up.is_some() {
                    Lowering::Silu
                } else {
                    Lowering::Node(Program::Elementwise(Elementwise::Silu))
                }
            })
            .or(rms.map(|c| Lowering::RmsNorm(c.eps, c.ones)))
            .or(reduce.map(Lowering::Reduce))
            .or(scalar.map(|(v, _)| Lowering::ScalarMul(v)))
            .or(bound_scale.map(Lowering::BoundScalarMul))
            .or(equal.map(|_| Lowering::Node(Program::Elementwise(Elementwise::Equal))))
            .or(convert.map(Lowering::Convert))
            .or_else(|| program_of(op.op_type).map(Lowering::Node))
        else {
            // An op the walk cannot lower is the walk's own by-name refusal — the bake fails
            // there with the richer message, so the propagation has nothing to say about it.
            continue;
        };
        let n_in = match program {
            Lowering::Silu => 2,
            Lowering::Reduce(_) => 1,
            Lowering::ScalarMul(_) | Lowering::BoundScalarMul(_) => 1,
            Lowering::RmsNorm(..) => 2,
            Lowering::Convert(_) => 1,
            Lowering::Node(Program::Matmul) => 2,
            Lowering::Node(Program::Elementwise(e)) => {
                match super::lower_ktir_to_superdsc::elementwise_op_func(f.name, e) {
                    Ok((_, arity)) => arity,
                    // The walk refuses this op after us; the propagation does not need its arity.
                    Err(_) => op.operands.len(),
                }
            }
            Lowering::Node(Program::Transpose) => 1,
            Lowering::IntBinary(_) => 2,
            Lowering::ScalarBinary(..) => 1,
            Lowering::BatchMatmul { .. } => 2,
            Lowering::BatchTranspose { .. } => 1,
            Lowering::Node(_) => op.operands.len(),
        };
        let raw_in: Vec<Ssa> = match (terminal, scalar, bound_scale) {
            _ if rms.is_some() => {
                let c = rms.expect("just matched");
                vec![c.x, c.gamma]
            }
            _ if equal.is_some() => {
                let (a, b) = equal.expect("just matched");
                vec![a, b]
            }
            (Some(c), _, _) => match c.up {
                Some(up) => vec![c.gate, up],
                None => vec![c.gate],
            },
            (None, Some((_, tensor)), _) => vec![tensor],
            (None, None, Some(b)) => vec![b.tensor],
            (None, None, None) => op.operands.iter().copied().take(n_in).collect(),
        };
        // One input's (SOURCE, ORDER-FREENESS). The ruling's semantics are a SHAPE fact, not a
        // spelling fact: an operand that supplies ONE SCALAR PER OUTPUT ROW is order-free — a
        // `Col` broadcast chain, a `[·, 1]` parameter, or a computed per-row value (a reduce
        // result) whose own type states the degenerate column — while a dense operand, an
        // `Mb`/`Y` spray and a `[1, ·]` row are read LANE FOR LANE, so the consumer's
        // arrangement is the operand's. Read off the broadcast chain when the program spells
        // one, and off the operand's OWN region/type when it does not (Triton's implicit
        // broadcast needs no chain, and the emitter selects the operand mode from the same
        // degenerate extent).
        let dims_of = |v: Ssa| -> Option<&'static [i64]> {
            f.operations
                .iter()
                .find(|o| o.result == Some(v))
                .and_then(|o| o.result_type)
                .and_then(|t| t.dims())
        };
        let resolved = |v: Ssa| -> (Ssa, bool) {
            match bcasts.get(&v) {
                Some(b) => (b.src, b.axis == BcastAxis::Col),
                None => {
                    let per_row = match region_for_operand(k, v) {
                        // A parameter: the VIEW's own column extent decides.
                        Ok(Some(r)) => r.c_len == 1,
                        // A value this function computed: its own result type's shape decides —
                        // `[·, 1]` or rank-1 `[rows]` is the per-row vector a reduce produces.
                        _ => {
                            matches!(dims_of(v), Some(d) if d.len() == 1 || *d.last().unwrap_or(&0) == 1)
                        }
                    };
                    (v, per_row)
                }
            }
        };
        // `arr` is mutated inside the marking loop below, so this is a fn taking the map rather
        // than a closure borrowing it.
        fn staggered(arr: &std::collections::HashMap<Ssa, ElementArrangement>, v: Ssa) -> bool {
            arr.get(&v) == Some(&ElementArrangement::Dl16ToFp32)
        }
        let out_arr = match program {
            // A REDUCTION IS ORDER-FREE: the stagger permutes within 64-element blocks and the
            // reduction's answer is a function of the multiset, so the stagger ends here.
            Lowering::Reduce(_) => ElementArrangement::Standard,
            // THE ONLY PLACE AN ARRANGEMENT IS DECIDED: the convert's own const-table row.
            Lowering::Convert(kind) => {
                crate::element_arrangement::output_arrangement_of(kind.op_func())
                    .unwrap_or(ElementArrangement::Standard)
            }
            // POINTWISE — the stagger propagates through it, and every same-width operand that
            // is read lane-for-lane must agree with the output's arrangement.
            Lowering::Silu
            | Lowering::ScalarMul(_)
            | Lowering::BoundScalarMul(_)
            | Lowering::RmsNorm(..)
            | Lowering::Node(Program::Elementwise(_)) => {
                let mut out = ElementArrangement::Standard;
                for v in &raw_in {
                    let (src, order_free) = resolved(*v);
                    if !order_free && staggered(&arr, src) {
                        out = ElementArrangement::Dl16ToFp32;
                    }
                }
                if out == ElementArrangement::Dl16ToFp32 {
                    for v in &raw_in {
                        let (src, order_free) = resolved(*v);
                        if order_free || staggered(&arr, src) {
                            continue;
                        }
                        // A standard full-width operand of a staggered op would be READ
                        // permuted. It must be a PARAMETER the bake can restage — anything
                        // this function computed is the mixing refusal.
                        match region_for_operand(k, src)? {
                            Some(r) => {
                                marked.insert(r.tid, ElementArrangement::Dl16ToFp32);
                                // ⭐ AND THE VALUE CARRIES IT FROM NOW ON: a marked parameter is
                                // staged staggered, so every LATER consumer reads it in the
                                // arrangement — the sampler's `selG = (gramp == jG)` reads the
                                // same `gramp` the marking op did, and its output must come out
                                // staggered or the one-hot chain mixes arrangements two ops
                                // later. Without this, the second read sees a standard value and
                                // the mixing refusal fires on a program the device computes
                                // consistently.
                                arr.insert(src, ElementArrangement::Dl16ToFp32);
                            }
                            None => {
                                return err(format!(
                                    "{}: a standard value this function computed meets a \
                                     staggered one at `{:?}` — the pointwise lane order is the \
                                     stagger's, so the standard operand would be read permuted. \
                                     Stage the operand through the arrangement (a parameter the \
                                     bake can restage) or reduce before the mix",
                                    f.name, op.op_type
                                ));
                            }
                        }
                    }
                }
                out
            }
            // EVERY OTHER PROGRAM — matmul, transpose, the fused attention kinds — addresses its
            // operands in an order of its own, so a staggered input has no staging that is
            // correct. Refused by name rather than lowered as the nearest arrangement.
            _ => {
                for v in &raw_in {
                    let (src, order_free) = resolved(*v);
                    if !order_free && staggered(&arr, src) {
                        return err(format!(
                            "{}: a staggered value reaches `{:?}`, whose addressing is not \
                             lane-for-lane pointwise — there is no arrangement to stage the \
                             operand in. Reduce before the op, or keep the stagger inside the \
                             pointwise island",
                            f.name, op.op_type
                        ));
                    }
                }
                ElementArrangement::Standard
            }
        };
        // ⛔ A CONVERT'S OWN STAGGERED INPUT is a composition of two arrangements the table
        // does not state — refused before the output's row is applied.
        if matches!(program, Lowering::Convert(_))
            && raw_in
                .first()
                .is_some_and(|v| staggered(&arr, resolved(*v).0))
        {
            return err(format!(
                "{}: a staggered value enters a dtype-convert — the composition of the stagger \
                 with the convert's own output arrangement is not a fact the const table \
                 states, so it is refused rather than guessed. Reduce before the convert",
                f.name
            ));
        }
        if let Some(r) = op.result {
            arr.insert(r, out_arr);
            // ⛔ AND A STAGGERED VALUE NEVER REACHES A STORE: the host read-back channel is
            // standard-order, so a stored stagger is a permuted answer by construction.
            if out_arr == ElementArrangement::Dl16ToFp32
                && f.operations.iter().any(|s| {
                    s.op_type == OpKind::KtdpStore && s.operands.first().copied() == Some(r)
                })
            {
                return err(format!(
                    "{}: a staggered value is stored — the host read-back channel reads \
                     standard order, so the store would answer permuted. Reduce before the \
                     store (the arrangement ends at a reduction)",
                    f.name
                ));
            }
        }
    }
    Ok(marked)
}

pub fn lower_function(
    k: &KtirNode,
    layout: Option<&BundleLayout>,
    sym_id_base: &mut i64,
    headmajor: Option<super::lower_ktir_to_superdsc::HeadmajorOproj>,
) -> Result<Vec<super::EmittedOp>, Error> {
    lower_function_folded(k, layout, sym_id_base, headmajor, None)
}

/// ⭐ A PROGRAM THAT WALKS — one constant-trip `scf.for`, the paged attention's page walk — through
/// the SAME per-op walk as every straight-line program: [`super::loop_fold::flatten`] states the
/// body inline at trip 0, the walk lowers it op by op, and the ops the body emitted are marked as the
/// walk's ([`super::EmittedOp::walk`]) — the descriptors the card relaunches once per trip. The
/// loop-carried values are buffers seeded before the walk with the values the program initialises
/// them to and overwritten by each trip's yielded values.
///
/// A ONE-TRIP walk states no loop at all — the front end promotes it to its body (MLIR's
/// `SimplifyTrivialLoops`), the carried values read as their splatted inits — and lowers as the
/// straight-line program it is.
pub fn lower_walk(
    k: &KtirNode,
    layout: Option<&BundleLayout>,
    sym_id_base: &mut i64,
) -> Result<Vec<super::EmittedOp>, Error> {
    let flat = super::loop_fold::flatten(k)?;
    let (node, fold) = match &flat {
        Some(flat) => (&flat.node, Some(&flat.fold)),
        None => (k, None),
    };
    // ⭐ torch-spyre plans LX before it emits ([`super::lx_plan`]): a probe lowering learns how every
    // op slices every tensor, and the plan keeps in LX what every op hands out alike.
    let Some(layout) = layout else {
        return lower_function_folded(node, None, sym_id_base, None, fold);
    };
    super::lx_plan::plan(node, layout, |probe| {
        let mut sym = *sym_id_base;
        lower_function_folded(node, Some(probe), &mut sym, None, fold)
    })?;
    let probed = node
        .minted
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clone();
    let ops = lower_function_folded(node, Some(layout), sym_id_base, None, fold);
    // The plan's division is this program's alone: every other function of the bundle divides as
    // its own ops do.
    layout.lx_row_groups.set(None);
    let ops = ops?;
    if *node.minted.lock().unwrap_or_else(|p| p.into_inner()) != probed {
        return err(format!(
            "{}: the scratchpad plan's probe and the real lowering minted different tensors, so the \
             plan's ids name other buffers",
            node.func.name
        ));
    }
    super::lx_plan::check(layout, node.func.name)?;
    Ok(ops)
}

fn lower_function_folded(
    k: &KtirNode,
    layout: Option<&BundleLayout>,
    sym_id_base: &mut i64,
    headmajor: Option<super::lower_ktir_to_superdsc::HeadmajorOproj>,
    fold: Option<&super::loop_fold::LoopFold>,
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
    // buffer. They are minted strictly above every bound tid. ⛔ AND RESERVED TIDS ARE SKIPPED, NOT
    // STEPPED PAST: a bound reserved tid is legal (the LmLast extraction binds its destination to
    // `LAST_HIDDEN_TID`, one above the sentinel region), so the bare maximum is `u32::MAX` there and
    // `+ 1` would panic — and a mint landing INSIDE a reserved region would alias that synthetic's
    // placement. Reserved tids name producer-staged tensors, never intermediates, so the mint starts
    // above the highest NON-reserved binding and steps past any region it lands in.
    //
    // ⛔⭐ AND THAT BOUND IS A FLOOR, NOT THE SEED. MEASURED on the card (2026-10-20, granite-2b
    // fp16): seeding the mint from THIS node's bindings alone minted node B's intermediates onto
    // node A's real placements — `1310 placements collapse to 1214 distinct ids`, 96 duplicates,
    // the runtime refusing to load. The seed is the BUNDLE's
    // ([`BundleLayout::take_inter_tid`](crate::placement::BundleLayout::take_inter_tid)): lazily
    // initialized above the highest non-reserved placement in the whole bundle, monotonic from
    // there. This node-derived value survives as the FLOOR the bundle minter honors — a node
    // lowered into a layout that places its own bindings high still mints clear of its own
    // tensors — and as the replay gate below.
    let floor = mint_floor(k);
    let mut inter: std::collections::HashMap<Ssa, Region> = std::collections::HashMap::new();
    // THE fp8 SCALE MULFS THE DOT ABSORBED. The walk lowers each op once, and an absorbed epilogue
    // (`p * ws`, see the absorption at the `stored` lookup) already rode into its dot's emission —
    // so the `arith.mulf` itself must not be lowered again when the loop reaches it. Membership is
    // the mulf's RESULT value; skipped at the loop top beside the chain interiors.
    let mut fp8_epilogue: std::collections::HashSet<Ssa> = std::collections::HashSet::new();
    // THE ROW-TILED FAMILY FOLD'S SKIPPED MEMBERS. A row-blocked pointwise program states one
    // load/chain/store family per `[BLOCK_M, N]` block, and the walk lowers per-op at the LOAD'S
    // OWN tile — so every block past the first refuses at the row-corner law while the FIRST
    // block's descriptor already spans the whole view (its out region's `r_cover` is the hull of
    // every block's store, and [`node_rows`] folds that into the row count). [`row_tiled_family`]
    // proves the blocks are ONE op over windows that tile the view exactly and returns the
    // non-leader terminals; membership is the terminal's RESULT value, skipped at the loop top
    // beside the chain interiors. The leader emits the fold — one op, time factor 1.
    let mut row_fold_skip: std::collections::HashSet<Ssa> = std::collections::HashSet::new();

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
    // THE EQUALITY MASKS, read before the walk for the same reason the silus are: the device's
    // `OpFunc::Equal` is ONE op and KTIR spells it as two (`cmpf` + `uitofp`), so the comparison
    // is chain-interior and the RETYPE is the terminal this walk lowers. See
    // [`program_equal_masks`].
    let equal_masks = program_equal_masks(f)?; // ⭐⭐⭐ THE GATHERS THE PROGRAM STATES, read ONCE and from [`gathers_of`] — the same joins the
    // per-`Program` door's `gather_of` makes, so the two doors cannot disagree about which parameter
    // is the table or which is the index. A program with ONE indirect tile reads identically to
    // `gather_of`; a program with SEVERAL is either the unrolled-sweep shape (every tile over the
    // SAME index and value parameters, one per trip) or the batched paged-decode shape (K^T and V
    // gathered through one or several index buffers — DIFFERENT pairs), and in both the walk below
    // names the op that carries EACH tile, by load identity, so every descriptor still carries at
    // most one index operand.
    //
    // ⛔ AND THEY ARE CARRIED, NOT COUNTED: `gather_carried` below is checked after the walk, per
    // tile. A gather the program states and no descriptor declares is a silently DIRECT read of the
    // table's first rows — well formed, the right shape and dtype, and nothing else in the pipeline
    // compares an emitted descriptor against the program's indirect tile.
    let gathers = gathers_of(k)?;
    // The gathered LOAD each tile's `ktdp.load` produced — the value an op's input must BE (by
    // identity) for that tile's gather to be the one it reads. Read straight off [`Gather::load`]:
    // the tile walk in `gathers_of` already found each tile's load, so this is a lookup, not a
    // re-walk that could disagree with the walk about which load belongs to which tile (issue 201
    // item 3).
    let gathered_loads: std::collections::HashMap<Ssa, usize> = gathers
        .iter()
        .enumerate()
        .map(|(i, g)| (g.load, i))
        .collect();
    let mut gather_carried = vec![false; gathers.len()];
    // The second op of every head-major pair ([`head_major_pair`]), lowered with the first.
    let mut head_major_skip: std::collections::HashSet<Ssa> = std::collections::HashSet::new();
    // The row fold's own view of the same analyses — built once here so its helpers read ONE
    // value rather than threading three slices each (see [`FoldAnalyses`]).
    let analyses = FoldAnalyses {
        silus: &silus,
        rmsnorms: &rmsnorms,
        gathered_loads: &gathered_loads,
    };
    // ⭐ THE INTEGER COMPUTE SET — [`integer_compute_of`]'s own doc states the fixpoint. Read once
    // here so the loop-top exemption and the derivation arm below read ONE set and cannot
    // disagree about which integer op is data.
    let int_compute = integer_compute_of(f);

    // ⭐⭐⭐ THE WALK'S CARRIED STATE — one buffer per `iter_args` value, read by every trip and
    // overwritten by its yielded value (see [`super::loop_fold`]). Minted HERE, before any op, and
    // SEEDED with the value the program initialises it to; the loop's result names the same buffer,
    // so the ops after the walk read it directly. A value framed per row of a reduction (the running
    // max and sum: the shape of a `linalg.reduce` result) is the `[numel, 1]` column the reduction
    // writes; anything else is its flat frame.
    let mut yield_target: std::collections::HashMap<Ssa, (Ssa, Region)> =
        std::collections::HashMap::new();
    if let Some(fold) = fold {
        let Some(layout) = layout else {
            return err(format!(
                "{}: a walk's carried state needs a `BundleLayout` to hold its buffers",
                f.name
            ));
        };
        let def_of = |v: Ssa| f.operations.iter().find(|o| o.result == Some(v));
        for c in &fold.carried {
            let init = def_of(c.init);
            let dims: Vec<i64> = init
                .and_then(|o| o.result_type)
                .and_then(|t| t.dims())
                .map(<[i64]>::to_vec)
                .unwrap_or_default();
            if dims.is_empty() {
                return err(format!(
                    "{}: the walk's carried value %{} states no shaped init",
                    f.name, c.arg.0
                ));
            }
            let frame = value_frame(f, &dims);
            let tid = mint_inter(k, layout, floor, c.arg);
            layout.synth(crate::place::PlaceId::Act(tid), &frame);
            let r = Region {
                tid,
                v_rows: frame[0],
                v_cols: frame[1],
                r_start: 0,
                c_start: 0,
                r_len: frame[0],
                c_len: frame[1],
                r_cover: (0, frame[0]),
                c_cover: (0, frame[1]),
                is_out: false,
                is_fp8: false,
                band: None,
            };
            inter.insert(c.arg, r);
            inter.insert(c.result, r);
            yield_target.insert(c.yielded, (c.arg, r));
            out.extend(super::lower_ktir_to_superdsc::fill(
                f.name,
                c.seed,
                &Region { is_out: true, ..r },
                sym_id_base,
                Some(layout),
            )?);
        }
    }
    // The ops the walk's body stated — a contiguous run of the flattened program — and where the
    // descriptors they emit start and end.
    let body_at: Option<(usize, usize)> = fold.and_then(|fd| {
        let at: Vec<usize> = f
            .operations
            .iter()
            .enumerate()
            .filter(|(_, o)| o.result.is_some_and(|r| fd.body.contains(&r)))
            .map(|(i, _)| i)
            .collect();
        Some((*at.first()?, *at.last()? + 1))
    });
    let mut body_out: (usize, Option<usize>) = (0, None);
    // The int32 zero every chained index's run is copied against ([`int_zero_run`]).
    let mut int_zero: Option<u32> = None;
    // A carried buffer may take its yielded value's write only once nothing later reads the value it
    // holds: the last op reading each carried arg, by position.
    let last_read = |arg: Ssa| -> Option<usize> {
        f.operations.iter().rposition(|o| o.operands.contains(&arg))
    };
    // THE BATCH-BLOCKED BUFFERS, by their blocks' width `c` — `[r, c]` stick-major blocks one after
    // another: a batched transpose's output (the `[k, n]` kernels a batched contraction reads), and
    // a page read `[kv, block, 128]`, permuted per kv head. One stick wide a block run is the flat
    // frame's own bytes; wider it is not, and only the readers that address a block at its base
    // take one ([`wide_blocks`]).
    let mut batch_blocks: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();

    for (oi, op) in f.operations.iter().enumerate() {
        if let Some((start, end)) = body_at {
            if oi == start {
                // ⭐⭐⭐ THE BODY READS NO REGISTRY CONSTANT. A relaunch shifts the segment the step's
                // sources live in — the activation segment, where the registry's bound consts live
                // too — so a scalar read in the body would land a trip's step past its constant. Every
                // splatted constant the body reads as an operand is a BUFFER instead, filled ONCE
                // here, before the body's descriptors begin, and read dense on every trip.
                for o in &f.operations[start..end] {
                    for v in dense_splat_operands(f, o) {
                        if inter.contains_key(&v) {
                            continue;
                        }
                        if let Some((r, value)) = splat_buffer(k, layout, floor, v)? {
                            out.extend(super::lower_ktir_to_superdsc::fill(
                                f.name,
                                value,
                                &r,
                                sym_id_base,
                                layout,
                            )?);
                            inter.insert(v, Region { is_out: false, ..r });
                        }
                    }
                }
                if let Some(layout) = layout
                    && gathers.iter().any(|g| g.via.is_some())
                {
                    int_zero = Some(int_zero_run(f.name, layout)?);
                }
                body_out.0 = out.len();
            }
            if oi == end {
                body_out.1 = Some(out.len());
            }
        }
        let in_body = body_at.is_some_and(|(start, end)| (start..end).contains(&oi));
        // ⭐ THE ONE `is_plumbing` EXEMPTION: an integer arith op [`integer_compute_of`] classified
        // as COMPUTE is data, not addressing, and lowers through [`Lowering::IntBinary`] below.
        // Every other plumbing op — including every integer op of every kernel that bakes today,
        // whose consumers are all tiles and splats — keeps the skip verbatim.
        if is_plumbing(op.op_type)
            && !(matches!(
                op.op_type,
                OpKind::ArithMuli | OpKind::ArithAddi | OpKind::ArithSubi
            ) && op.result.is_some_and(|r| int_compute.contains(&r)))
        {
            continue;
        }
        // ⭐ AN INTEGER SCALAR EXTRACTED FROM A LOAD IS AN ADDRESS — a position read through a table
        // (`page = tl.load(page_index_table + i * 32)`), which a chained gather reads in place
        // ([`super::lower_ktir_to_superdsc::ChainedIndex`]); no node kind consumes it as data.
        if op.op_type == OpKind::TensorExtract
            && matches!(
                op.result_type,
                Some(ktir_core::irtype::IrType::Scalar(DType::I32 | DType::I64))
                    | Some(ktir_core::irtype::IrType::Index)
            )
        {
            continue;
        }
        // The chain's INTERIOR — negate, exp, 1+, divide — is lowered by nothing at all: the one
        // `Program::SiluMul` below stands for all five ops, and the device primitive never
        // materialises these values. Skipped BEFORE `program_of`, which would otherwise refuse the
        // `arith.negf` for having no 1:1 mapping (correctly, in isolation).
        if op.result.is_some_and(|r| {
            head_major_skip.contains(&r)
                || silus.iter().any(|c| c.consumed.contains(&r))
                || rmsnorms.iter().any(|c| c.consumed.contains(&r))
                // The expand/collapse/broadcast chain itself: no descriptor, the consumer states it.
                || bcast_consumed.contains(&r)
                // An equality mask's `arith.cmpf`: the fold destroyed the i1, and the `uitofp`
                // terminal below stands for the pair.
                || equal_masks.consumed.contains(&r)
                // An fp8 dot's absorbed scale mulf: it rode into the dot's emission above.
                || fp8_epilogue.contains(&r)
                // A row-folded family's non-leader block: the leader's one whole-view descriptor
                // already computed its rows (see [`row_tiled_family`]).
                || row_fold_skip.contains(&r)
        }) {
            continue;
        }
        // ⭐⭐⭐ A SHAPE OP THAT MOVES NO BYTE NAMES ITS SOURCE'S BUFFER — a `tensor.reshape`,
        // `collapse_shape` or `expand_shape` whose source and result frames are the same card bytes
        // ([`same_card_bytes`]) lowers to nothing, and its result is its source's region. A shape op
        // that WOULD move bytes is a relayout this door does not state, and is refused by name.
        //
        // ⭐ A GATHERED LOAD REACHING ONE IS MATERIALIZED FIRST — the page gather: the load's one
        // index entry selects a whole slab of the paged value (a page of a row-outermost cache, every kv head of it), so
        // its rows reach a consumer only through a contiguous scratch the gathered copy fills
        // ([`super::lower_ktir_to_superdsc::page_gather`]), and the shape op names that scratch.
        if matches!(
            op.op_type,
            OpKind::TensorReshape | OpKind::TensorCollapseShape | OpKind::TensorExpandShape
        ) && let (Some(res), Some(&src)) = (op.result, op.operands.first())
        {
            if let Some(r) = inter.get(&src)
                && wide_blocks(&batch_blocks, r.tid)
            {
                return err(format!(
                    "{}: `{:?}` reads t{}, whose bytes are batch blocks one after another — every \
                     frame a shape op names is the flat frame's, which those bytes are not",
                    f.name, op.op_type, r.tid
                ));
            }
            let def_dims = |v: Ssa| -> Vec<i64> {
                f.operations
                    .iter()
                    .find(|o| o.result == Some(v))
                    .and_then(|o| o.result_type)
                    .and_then(|t| t.dims())
                    .map(<[i64]>::to_vec)
                    .unwrap_or_default()
            };
            let (mut from, to) = (def_dims(src), def_dims(res));
            // A GATHER FROM A ROW-OUTERMOST BUFFER (the paged cache's allocation,
            // [`BundleLayout::allocate_row_outermost`]) is held as the page gather lands it —
            // one-stick sub-rows, its slabs' own row-major bytes — whatever shape the load states.
            // A gather from a tiled buffer lands stick-major at the load's own shape
            // ([`super::lower_ktir_to_superdsc::row_gather`]).
            let stick = i64::from(crate::work::FP16_ELEMS_PER_STICK);
            let numel: i64 = from.iter().product();
            let row_outermost =
                |gi: usize| layout.is_some_and(|l| l.is_row_outermost(gathers[gi].value_tid));
            if gathered_loads
                .get(&src)
                .is_some_and(|&gi| row_outermost(gi))
                && numel % stick == 0
            {
                from = vec![numel / stick, stick];
            }
            // ⭐ THE HEAD-MAJOR PAIR — a reshape of a stick-major `[B, A·S]` row block into
            // `[B, A, S]` read only by the `[1, 0, 2]` transpose into `[A, B, S]`: on the card the
            // two together move no byte (stick column `a` of the block IS head `a`'s rows), so the
            // transpose's result is the source's own bytes in the `[A·B, S]` frame, and both lower
            // to nothing (spyre-inference's `q_rows.unsqueeze(0).transpose(1, 2).reshape(...)`).
            if !gathered_loads.contains_key(&src)
                && let Some((t_res, frame, sticks)) = head_major_pair(f, op, &from, &to)
            {
                let source = match region_for_operand(k, src)?.or_else(|| inter.get(&src).copied())
                {
                    Some(r) => r,
                    None => {
                        return err(format!(
                            "{}: the head-major reshape reads a value this walk holds no buffer for",
                            f.name
                        ));
                    }
                };
                let r = match layout {
                    Some(layout) if sticks > 1 => {
                        let tid = mint_inter(k, layout, floor, res);
                        let dst = stick_major_region(tid, frame[0], sticks, layout);
                        out.extend(
                            HeadMajorCopies::of(&source, &dst, frame[0], sticks)?.to_heads(
                                f.name,
                                sym_id_base,
                                layout,
                            )?,
                        );
                        Region {
                            is_out: false,
                            ..dst
                        }
                    }
                    _ => reframed(source, frame),
                };
                inter.insert(res, r);
                inter.insert(t_res, r);
                head_major_skip.insert(t_res);
                continue;
            }
            // ⭐ ROW-MAJOR ROWS READ STICK-MAJOR — `[R·C, stick]` (a batched dot's per-head rows,
            // `[B, KV, G, D]`) reshaped to `[R, C·stick]` (the `[B, H·D]` output block): the stick
            // permute back, into the parameter the reshape's result is stored to when it is stored
            // whole, else into a minted buffer. Head rows several sticks wide (head dim 128) are
            // stick-major themselves, so they go row-major first ([`HeadRows`]).
            let back = stick_major_split(&to, &from)
                .map(|split| (split, None))
                .or_else(|| HeadRows::of(&to, &from).map(|h| ((h.rows, h.row_sticks), Some(h))));
            if !gathered_loads.contains_key(&src)
                && !same_card_bytes(&from, &to)
                && let Some(((rows, sticks), head_rows)) = back
            {
                let Some(layout) = layout else {
                    return err(format!(
                        "{}: a stick permute needs a `BundleLayout` to hold its result",
                        f.name
                    ));
                };
                let Some(source) = region_for_operand(k, src)?.or_else(|| inter.get(&src).copied())
                else {
                    return err(format!(
                        "{}: `{:?}` permutes a value this walk holds no buffer for",
                        f.name, op.op_type
                    ));
                };
                let stored_to = f.operations.iter().find(|s| {
                    s.op_type == OpKind::KtdpStore
                        && s.operands.first() == Some(&res)
                        && !indirect_tile_of(f, s.operands.get(1).copied())
                });
                let dst = match stored_to {
                    Some(store) => {
                        let tile = f.operations.iter().find(|o| {
                            o.op_type == OpKind::KtdpConstructAccessTile
                                && Some(o.result) == Some(store.operands.get(1).copied())
                        });
                        let o = match tile {
                            Some(t) => region_of_view_operand(k, t, 0)?
                                .map(|o| windowed_region(k, o, t))
                                .transpose()?,
                            None => None,
                        };
                        match o {
                            Some(o)
                                if (o.r_start, o.c_start, o.r_len, o.c_len)
                                    == (0, 0, rows, sticks * stick as u32) =>
                            {
                                Region { is_out: true, ..o }
                            }
                            _ => {
                                return err(format!(
                                    "{}: the permuted `[{rows}, {}]` block is stored somewhere \
                                     other than a whole `[{rows}, {}]` parameter window",
                                    f.name,
                                    sticks * stick as u32,
                                    sticks * stick as u32
                                ));
                            }
                        }
                    }
                    None => {
                        let tid = mint_inter(k, layout, floor, res);
                        layout.synth(
                            crate::place::PlaceId::Act(tid),
                            &[rows, sticks * stick as u32],
                        );
                        Region {
                            tid,
                            v_rows: rows,
                            v_cols: sticks * stick as u32,
                            r_start: 0,
                            c_start: 0,
                            r_len: rows,
                            c_len: sticks * stick as u32,
                            r_cover: (0, rows),
                            c_cover: (0, sticks * stick as u32),
                            is_out: true,
                            is_fp8: false,
                            band: None,
                        }
                    }
                };
                let source = match head_rows {
                    Some(h) => {
                        let mid = row_major_region(
                            mint_scratch(k, layout, floor, res, 0),
                            h.rows * h.row_sticks,
                            layout,
                        );
                        out.push(permute(
                            f.name,
                            &source,
                            &mid,
                            Permuted::whole((h.head_rows, h.sticks)),
                            crate::ir::bridge::tiled_op_sdsc_op::StickPermute::StickToRow,
                            sym_id_base,
                            Some(layout),
                        )?);
                        Region {
                            is_out: false,
                            ..mid
                        }
                    }
                    None => source,
                };
                out.push(permute(
                    f.name,
                    &source,
                    &dst,
                    Permuted::whole((rows, sticks)),
                    crate::ir::bridge::tiled_op_sdsc_op::StickPermute::RowToStick,
                    sym_id_base,
                    Some(layout),
                )?);
                inter.insert(
                    res,
                    Region {
                        is_out: false,
                        ..dst
                    },
                );
                continue;
            }
            // ⭐ STICK-MAJOR ROWS READ ROW-MAJOR — `[R, C·stick]` (a batched dot's `[KV, G, 128]`
            // output, heads two sticks wide) reshaped to `[R·C, stick]` or one row of `R·C` sticks
            // (spyre-inference's `attn.reshape(1, num_heads, head_size)`): the stick permute into a
            // minted row-major buffer, and the rows a scatter takes from there.
            if !gathered_loads.contains_key(&src)
                && !same_card_bytes(&from, &to)
                && let Some((rows, sticks)) = stick_major_split(&from, &to)
            {
                let Some(layout) = layout else {
                    return err(format!(
                        "{}: a stick permute needs a `BundleLayout` to hold its result",
                        f.name
                    ));
                };
                let Some(source) = region_for_operand(k, src)?.or_else(|| inter.get(&src).copied())
                else {
                    return err(format!(
                        "{}: `{:?}` permutes a value this walk holds no buffer for",
                        f.name, op.op_type
                    ));
                };
                let tid = mint_inter(k, layout, floor, res);
                let dst = row_major_region(tid, rows * sticks, layout);
                out.push(permute(
                    f.name,
                    &source,
                    &dst,
                    Permuted::whole((rows, sticks)),
                    crate::ir::bridge::tiled_op_sdsc_op::StickPermute::StickToRow,
                    sym_id_base,
                    Some(layout),
                )?);
                let r = reframed(
                    Region {
                        is_out: false,
                        ..dst
                    },
                    flat_frame(&to),
                );
                inter.insert(res, r);
                if let Some(sc) = indirect_store_of(k, res)? {
                    out.extend(super::lower_ktir_to_superdsc::row_scatter(
                        f.name,
                        &reframed(r, [sc.entries, sc.cols]),
                        sc,
                        sc.value_rows,
                        sym_id_base,
                        Some(layout),
                    )?);
                }
                continue;
            }
            // A row-outermost page lands as one-stick rows; read in a multi-stick frame, it is permuted
            // after the landing (below).
            let page_to_sticks = gathered_loads
                .get(&src)
                .is_some_and(|&gi| row_outermost(gi))
                && !same_card_bytes(&from, &to)
                && stick_major_split(&to, &from).is_some();
            if !same_card_bytes(&from, &to)
                && !page_to_sticks
                && !gathered_loads
                    .get(&src)
                    .is_some_and(|&gi| !row_outermost(gi))
            {
                return err(format!(
                    "{}: `{:?}` re-frames {from:?} as {to:?}, which moves bytes on the card (the \
                     two stick-major frames place an element at different addresses) — a \
                     relayout this door does not state",
                    f.name, op.op_type
                ));
            }
            let source = match gathered_loads.get(&src) {
                Some(&gi) => {
                    let Some(layout) = layout else {
                        return err(format!(
                            "{}: a page gather needs a `BundleLayout` to hold its scratch",
                            f.name
                        ));
                    };
                    if !row_outermost(gi) {
                        let g = gathers[gi];
                        let (table_rows, cols) = gather_table_dims(f, &g)?;
                        let tid = mint_inter(k, layout, floor, src);
                        layout.synth(crate::place::PlaceId::Act(tid), &[g.entries, cols]);
                        let r = Region {
                            tid,
                            v_rows: g.entries,
                            v_cols: cols,
                            r_start: 0,
                            c_start: 0,
                            r_len: g.entries,
                            c_len: cols,
                            r_cover: (0, g.entries),
                            c_cover: (0, cols),
                            is_out: true,
                            is_fp8: false,
                            band: None,
                        };
                        out.extend(super::lower_ktir_to_superdsc::row_gather(
                            f.name,
                            &g,
                            table_rows,
                            cols,
                            &r,
                            sym_id_base,
                            Some(layout),
                        )?);
                        gather_carried[gi] = true;
                        let r = Region { is_out: false, ..r };
                        inter.insert(src, r);
                        if let Some((t_res, frame, sticks)) = head_major_pair(f, op, &from, &to) {
                            let r2 = if sticks > 1 {
                                let tid = mint_inter(k, layout, floor, res);
                                let dst = stick_major_region(tid, frame[0], sticks, layout);
                                out.extend(
                                    HeadMajorCopies::of(&r, &dst, frame[0], sticks)?.to_heads(
                                        f.name,
                                        sym_id_base,
                                        layout,
                                    )?,
                                );
                                Region {
                                    is_out: false,
                                    ..dst
                                }
                            } else {
                                reframed(r, frame)
                            };
                            inter.insert(res, r2);
                            inter.insert(t_res, r2);
                            head_major_skip.insert(t_res);
                        } else if same_card_bytes(&from, &to) {
                            inter.insert(res, reframed(r, flat_frame(&to)));
                        } else if let Some((rows, sticks)) = stick_major_split(&from, &to) {
                            // ⭐ THE GATHERED ROWS' ROW-MAJOR SPLIT — spyre-inference's
                            // `query.index_select(0, rep_row_ids).reshape(entries, KV, G, D)`: the
                            // gathered `[R, C·stick]` block lands stick-major, and the reshape
                            // reads its `R·C` sticks row by row, so the card permutes the sticks
                            // once into a row-major `[R·C, stick]` buffer.
                            let tid = mint_inter(k, layout, floor, res);
                            let dst = row_major_region(tid, rows * sticks, layout);
                            out.push(permute(
                                f.name,
                                &r,
                                &dst,
                                Permuted::whole((rows, sticks)),
                                crate::ir::bridge::tiled_op_sdsc_op::StickPermute::StickToRow,
                                sym_id_base,
                                Some(layout),
                            )?);
                            inter.insert(
                                res,
                                reframed(
                                    Region {
                                        is_out: false,
                                        ..dst
                                    },
                                    flat_frame(&to),
                                ),
                            );
                        } else if let Some((rows, sticks)) = stick_major_split(&to, &from) {
                            // ⭐ A GATHERED ROW READ IN A MULTI-STICK FRAME — one query row of
                            // heads two sticks wide (`[NQH·128]` read as `[KV, G, 128]`): its
                            // sticks run head by head, and the frame holds each head's sticks a
                            // row block apart, so the card permutes them once.
                            let tid = mint_inter(k, layout, floor, res);
                            let dst = stick_major_region(tid, rows, sticks, layout);
                            out.push(permute(
                                f.name,
                                &r,
                                &dst,
                                Permuted::whole((rows, sticks)),
                                crate::ir::bridge::tiled_op_sdsc_op::StickPermute::RowToStick,
                                sym_id_base,
                                Some(layout),
                            )?);
                            inter.insert(
                                res,
                                Region {
                                    is_out: false,
                                    ..dst
                                },
                            );
                        } else if let Some(h) = HeadRows::of(&from, &to) {
                            // ⭐ GATHERED ROWS OF MULTI-STICK HEADS, READ AS HEAD ROWS — a batch's
                            // query rows `[R, H·S·stick]` read as `[R·KV, G, S·stick]` at head dim
                            // 128: the row-major split puts each head row's `S` sticks together,
                            // and a second permute lays the head rows stick-major, the frame the
                            // one-row arm above lands (and the batched dot reads a block of).
                            let mid = row_major_region(
                                mint_scratch(k, layout, floor, res, 0),
                                h.rows * h.row_sticks,
                                layout,
                            );
                            out.push(permute(
                                f.name,
                                &r,
                                &mid,
                                Permuted::whole((h.rows, h.row_sticks)),
                                crate::ir::bridge::tiled_op_sdsc_op::StickPermute::StickToRow,
                                sym_id_base,
                                Some(layout),
                            )?);
                            let tid = mint_inter(k, layout, floor, res);
                            let dst = stick_major_region(tid, h.head_rows, h.sticks, layout);
                            // The flat stick-major frame: a batched contraction reads its activation
                            // there, each batch's rows at their row offset in every stick plane.
                            out.push(permute(
                                f.name,
                                &Region {
                                    is_out: false,
                                    ..mid
                                },
                                &dst,
                                Permuted::whole((h.head_rows, h.sticks)),
                                crate::ir::bridge::tiled_op_sdsc_op::StickPermute::RowToStick,
                                sym_id_base,
                                Some(layout),
                            )?);
                            inter.insert(
                                res,
                                Region {
                                    is_out: false,
                                    ..dst
                                },
                            );
                        } else {
                            return err(format!(
                                "{}: `{:?}` re-frames a gathered {from:?} as {to:?}, which moves \
                                 bytes on the card — a relayout this door does not state. A \
                                 stick-major row is its heads in stick order only while a head is \
                                 ONE {stick}-element stick, and its row-major split is the rows' \
                                 sticks in order",
                                f.name, op.op_type
                            ));
                        }
                        continue;
                    }
                    let (g, page_sticks) = page_gather_of(f, gathers[gi])?;
                    // ⭐ A CHAINED INDEX runs as its two gathers: the run of index words is copied
                    // first, through the position table, into a scratch stick the gather then reads.
                    let g = match g.via {
                        Some(chain) => {
                            let zero = match int_zero {
                                Some(z) => z,
                                None => {
                                    let z = int_zero_run(f.name, layout)?;
                                    int_zero = Some(z);
                                    z
                                }
                            };
                            let run = layout.take_inter_tid(floor);
                            layout.synth_df(
                                crate::place::PlaceId::Act(run),
                                &[1, super::lower_ktir_to_superdsc::INDEX_RUN_WORDS],
                                crate::superdsc_opspec::Df::SenUint32,
                            );
                            out.extend(super::lower_ktir_to_superdsc::index_run_gather(
                                f.name,
                                &g,
                                chain,
                                (run, zero),
                                sym_id_base,
                                Some(layout),
                            )?);
                            Gather {
                                index_tid: run,
                                first_entry: crate::superdsc_opspec::EntryBase::ZERO,
                                via: None,
                                ..g
                            }
                        }
                        None => g,
                    };
                    // ⭐ AN ENTRY PAST dxp's CHUNK BOUND is copied in pieces through its index
                    // scaled by the piece count — doubled, `addi32toi32(x, x)`, into a scratch
                    // stick, so no constant is placed (the walk's body reads none).
                    let Some(pieces) = super::lower_ktir_to_superdsc::gather_pieces(page_sticks)
                    else {
                        return err(format!(
                            "{}: a gathered entry of {page_sticks} sticks splits into no power of \
                             two of whole-stick pieces under dxp's {}-byte entry bound",
                            f.name,
                            super::lower_ktir_to_superdsc::GATHER_ENTRY_BYTES_MAX
                        ));
                    };
                    let g = if pieces > 1 {
                        if g.first_entry != crate::superdsc_opspec::EntryBase::ZERO {
                            return err(format!(
                                "{}: a gathered entry split in {pieces} pieces reads its index at \
                                 entry {} — the doubling reads the index stick at its base",
                                f.name,
                                g.first_entry.entries()
                            ));
                        }
                        // The integer ops' device width, the chained index run's own
                        // ([`super::lower_ktir_to_superdsc::INDEX_RUN_WORDS`]).
                        let words = super::lower_ktir_to_superdsc::INDEX_RUN_WORDS;
                        let index_stick = |tid: u32, is_out: bool| Region {
                            tid,
                            v_rows: 1,
                            v_cols: words,
                            r_start: 0,
                            c_start: 0,
                            r_len: 1,
                            c_len: words,
                            r_cover: (0, 1),
                            c_cover: (0, words),
                            is_out,
                            is_fp8: false,
                            band: None,
                        };
                        // The doubling reads the index's first `words` words: the page id and
                        // the rest of its stick — inside the table's card pitch of a stick a
                        // score row, and refused by name where the placement is shorter.
                        let placed = layout.placements.get(&g.index_tid).map_or(0, |p| p.size);
                        if placed < u64::from(words) * 4 {
                            return err(format!(
                                "{}: a gathered entry split in {pieces} pieces scales its index \
                                 t{} by `addi32toi32` over {words} words, and t{} is placed in \
                                 {placed} bytes",
                                f.name, g.index_tid, g.index_tid
                            ));
                        }
                        let mut scaled = index_stick(g.index_tid, false);
                        for _ in 0..pieces.trailing_zeros() {
                            let tid = layout.take_inter_tid(floor);
                            layout.synth_df(
                                crate::place::PlaceId::Act(tid),
                                &[1, words],
                                crate::superdsc_opspec::Df::SenUint32,
                            );
                            out.extend(super::lower_ktir_to_superdsc::int_binary(
                                f.name,
                                super::lower_ktir_to_superdsc::IntBinary::Add,
                                &[scaled, scaled, index_stick(tid, true)],
                                &[],
                                sym_id_base,
                                Some(layout),
                            )?);
                            scaled = index_stick(tid, false);
                        }
                        Gather {
                            index_tid: scaled.tid,
                            ..g
                        }
                    } else {
                        g
                    };
                    let [rows, cols] = flat_frame(&from);
                    if cols != crate::work::FP16_ELEMS_PER_STICK {
                        return err(format!(
                            "{}: a page gather's rows are {cols} wide — the gathered copy lands \
                             a slab as one-stick sub-rows, the stick-major frame its consumers \
                             read only at one stick",
                            f.name
                        ));
                    }
                    let tid = mint_inter(k, layout, floor, src);
                    layout.synth(crate::place::PlaceId::Act(tid), &[rows, cols]);
                    let r = Region {
                        tid,
                        v_rows: rows,
                        v_cols: cols,
                        r_start: 0,
                        c_start: 0,
                        r_len: rows,
                        c_len: cols,
                        r_cover: (0, rows),
                        c_cover: (0, cols),
                        is_out: true,
                        is_fp8: false,
                        band: None,
                    };
                    // Several entries in pieces land piece by piece first, in a scratch as large.
                    let landing = (pieces > 1 && g.entries > 1).then(|| {
                        let tid = mint_scratch(k, layout, floor, src, 0);
                        layout.synth(crate::place::PlaceId::Act(tid), &[rows, cols]);
                        Region { tid, ..r }
                    });
                    out.extend(super::lower_ktir_to_superdsc::page_gather(
                        f.name,
                        &g,
                        (page_sticks, pieces),
                        &r,
                        landing.as_ref(),
                        sym_id_base,
                        Some(layout),
                    )?);
                    gather_carried[gi] = true;
                    let r = Region { is_out: false, ..r };
                    inter.insert(src, r);
                    r
                }
                None => match region_for_operand(k, src)?.or_else(|| inter.get(&src).copied()) {
                    Some(r) => Region { is_out: false, ..r },
                    None => {
                        return err(format!(
                            "{}: `{:?}` re-frames a value this walk holds no buffer for",
                            f.name, op.op_type
                        ));
                    }
                },
            };
            // ⭐ A ROW-OUTERMOST PAGE READ IN A MULTI-STICK FRAME — kv heads two sticks wide: the
            // page gather lands its slabs as one-stick rows, and the frame holds each row's sticks a
            // row block apart, so the card permutes the landing (the `restickify` torch-spyre's
            // stickification inserts where a consumer reads another layout). A rank-3
            // `[B, R, C·stick]` read is `B` blocks a batched op takes one at a time — a batched
            // transpose's input, a batched contraction's kernel — so each block is permuted on its
            // own, into blocks one after another ([`batch_blocks`]).
            if page_to_sticks && let Some((rows, sticks)) = stick_major_split(&to, &from) {
                let Some(layout) = layout else {
                    return err(format!(
                        "{}: a stick permute needs a `BundleLayout` to hold its result",
                        f.name
                    ));
                };
                let tid = mint_inter(k, layout, floor, res);
                let dst = stick_major_region(tid, rows, sticks, layout);
                let blocks = match to.as_slice() {
                    [b, _, _] => *b as u32,
                    _ => 1,
                };
                if blocks > 1 {
                    for b in 0..blocks {
                        out.push(permute(
                            f.name,
                            &source,
                            &dst,
                            Permuted {
                                rows: rows / blocks,
                                sticks,
                                block: Some(b),
                            },
                            crate::ir::bridge::tiled_op_sdsc_op::StickPermute::RowToStick,
                            sym_id_base,
                            Some(layout),
                        )?);
                    }
                    batch_blocks.insert(tid, sticks * crate::work::FP16_ELEMS_PER_STICK);
                } else {
                    out.push(permute(
                        f.name,
                        &source,
                        &dst,
                        Permuted::whole((rows, sticks)),
                        crate::ir::bridge::tiled_op_sdsc_op::StickPermute::RowToStick,
                        sym_id_base,
                        Some(layout),
                    )?);
                }
                inter.insert(
                    res,
                    Region {
                        is_out: false,
                        ..dst
                    },
                );
                continue;
            }
            // The same bytes, in the frame the result is held in — the frame its consumers read
            // (a `[1, 2048]` gathered query row is the `[8, 4, 64]` batched dot's `[32, 64]`; a
            // per-row column stays one value per stick row, [`value_frame`]).
            let [rr, rc] = value_frame(f, &to);
            let whole = source.r_start == 0
                && source.c_start == 0
                && source.r_len == source.v_rows
                && source.c_len == source.v_cols;
            let column = |[r, c]: [u32; 2]| c == 1 && r > 1;
            if whole && column([source.v_rows, source.v_cols]) != column([rr, rc]) {
                return err(format!(
                    "{}: `{:?}` reads t{} `[{}, {}]` as `[{rr}, {rc}]` — a per-row column holds one \
                     value per stick row, and a dense frame packs a row's values into its sticks: \
                     different bytes",
                    f.name, op.op_type, source.tid, source.v_rows, source.v_cols
                ));
            }
            let source = if whole && source.v_rows * source.v_cols == rr * rc {
                reframed(source, [rr, rc])
            } else {
                source
            };
            inter.insert(res, source);
            continue;
        }
        // ⭐ THE HEAD-MAJOR PAIR FROM THE TRANSPOSE's SIDE — spyre-inference's
        // `attn.reshape(1, num_heads, P, D).transpose(1, 2).reshape(P, num_heads, D)`: the source's
        // own bytes in the stick-major frame, and the rows a scatter takes from there.
        if op.op_type == OpKind::LinalgTranspose
            && let (Some(res), Some(&src)) = (op.result, op.operands.first())
        {
            let dims = |v: Ssa| -> Vec<i64> {
                f.operations
                    .iter()
                    .find(|o| o.result == Some(v))
                    .and_then(|o| o.result_type)
                    .and_then(|t| t.dims())
                    .map(<[i64]>::to_vec)
                    .unwrap_or_default()
            };
            if let Some((r_res, frame, sticks)) = head_major_pair(f, op, &dims(src), &dims(res)) {
                let Some(source) = region_for_operand(k, src)?.or_else(|| inter.get(&src).copied())
                else {
                    return err(format!(
                        "{}: the head-major transpose reads a value this walk holds no buffer for",
                        f.name
                    ));
                };
                if wide_blocks(&batch_blocks, source.tid) {
                    return err(format!(
                        "{}: the head-major transpose reads t{}, whose bytes are batch blocks one \
                         after another, as a stick-major frame",
                        f.name, source.tid
                    ));
                }
                let r =
                    match layout {
                        Some(layout) if sticks > 1 => {
                            let tid = mint_inter(k, layout, floor, r_res);
                            let dst = stick_major_region(
                                tid,
                                frame[0],
                                frame[1] / crate::work::FP16_ELEMS_PER_STICK,
                                layout,
                            );
                            out.extend(
                                HeadMajorCopies::of(&dst, &source, source.v_rows, sticks)?
                                    .to_block(f.name, sym_id_base, layout)?,
                            );
                            Region {
                                is_out: false,
                                ..dst
                            }
                        }
                        _ => reframed(source, frame),
                    };
                inter.insert(res, r);
                inter.insert(r_res, r);
                head_major_skip.insert(r_res);
                if let Some(sc) = indirect_store_of(k, r_res)? {
                    out.extend(super::lower_ktir_to_superdsc::row_scatter(
                        f.name,
                        &reframed(r, [sc.entries, sc.cols]),
                        sc,
                        sc.value_rows,
                        sym_id_base,
                        layout,
                    )?);
                }
                continue;
            }
        }
        // The terminal — the `up` multiply for the fused form, the divide itself for a bare one —
        // IS the fused program, and its inputs are the chain's gate and up rather than its own
        // operands (which are `silu(gate)` and up). Everything else keeps the 1:1 map exactly as
        // before.
        let terminal = silus.iter().find(|c| op.result == Some(c.mul));
        // A `linalg.reduce`'s combiner is an ATTRIBUTE, not its op kind, so the kind is read off the op
        // beside the 1:1 map rather than from it — and it is carried HERE rather than as a `Program`
        // variant, because `Program` is matched exhaustively by consumers outside this crate and a bare
        // reduce is reachable only through this door. An unrecognised combiner falls through to the
        // refusal below and is named there.
        let reduce = (op.op_type == OpKind::LinalgReduce)
            .then(|| reduce_kind_of(op))
            .flatten();
        // ⭐ A REDUCTION OVER THE LEADING AXIS IS A CHAIN OF COMBINES — the card's reduce folds
        // along the stick, and `Dimensions=[0]` folds the `J` blocks of rows a rank-N value's flat
        // frame holds one after another (a decode batch's `J` block slots: `torch.amax(..., dim=0)`,
        // `torch.sum(..., dim=0)`). So the `J` row windows of `R` rows each combine pairwise into
        // the `[R, ·]` result: `acc = win₀ ∘ win₁`, then `acc = acc ∘ winⱼ`. A per-row column (one
        // value per stick row, the row reductions' own form) is combined as its one-stick rows —
        // its storage is stick-wide, and the pad lanes the combine writes are never read.
        if let Some(kind) = reduce
            && op.attributes.iter().any(
                |(k, v)| matches!((k, v), (AttrKey::Dimensions, Attr::IntList(d)) if *d == [0]),
            )
            && let (Some(res), Some(&data)) = (op.result, op.operands.first())
        {
            let Some(layout) = layout else {
                return err(format!(
                    "{}: a leading-axis reduction needs a `BundleLayout` to hold its result",
                    f.name
                ));
            };
            let Some(src) = region_for_operand(k, data)?.or_else(|| inter.get(&data).copied())
            else {
                return err(format!(
                    "{}: a leading-axis reduction reads a value this walk holds no buffer for",
                    f.name
                ));
            };
            // The blocks folded: the source's elements over the result's.
            let result_numel: i64 = op
                .result_type
                .and_then(|t| t.dims())
                .map_or(0, |d| d.iter().product());
            let src_numel = i64::from(src.v_rows) * i64::from(src.c_len);
            let blocks = if result_numel > 0 && src_numel % result_numel == 0 {
                (src_numel / result_numel) as u32
            } else {
                0
            };
            // ONE block folds nothing: the result is the source's own bytes (a decode batch whose
            // chunk holds one block slot, `J = 1`), as a shape op that moves no byte names its source.
            if blocks == 1 && src.r_len == src.v_rows {
                inter.insert(
                    res,
                    Region {
                        is_out: false,
                        ..src
                    },
                );
                continue;
            }
            if blocks < 2 || !src.v_rows.is_multiple_of(blocks) || src.r_len != src.v_rows {
                return err(format!(
                    "{}: a leading-axis reduction over {blocks} block(s) of t{}'s {} rows — the \
                     combine folds whole equal row blocks of a whole view",
                    f.name, src.tid, src.v_rows
                ));
            }
            let rows = src.v_rows / blocks;
            let stick = crate::work::FP16_ELEMS_PER_STICK;
            let column = src.c_len == 1;
            let tid = mint_inter(k, layout, floor, res);
            layout.synth(crate::place::PlaceId::Act(tid), &[rows, src.v_cols]);
            let acc = Region {
                tid,
                v_rows: rows,
                v_cols: src.v_cols,
                r_start: 0,
                c_start: 0,
                r_len: rows,
                c_len: src.c_len,
                r_cover: (0, rows),
                c_cover: (0, src.v_cols),
                is_out: true,
                is_fp8: false,
                band: None,
            };
            // A column's one-stick rows, as the dense arm combines them.
            let sticks = |r: Region| {
                if column {
                    Region {
                        v_cols: stick,
                        c_len: stick,
                        c_cover: (0, stick),
                        ..r
                    }
                } else {
                    r
                }
            };
            let window = |j: u32| {
                sticks(Region {
                    r_start: j * rows,
                    r_len: rows,
                    r_cover: (j * rows, (j + 1) * rows),
                    is_out: false,
                    ..src
                })
            };
            let combine = match kind {
                ReduceKind::Max => Elementwise::Maximum,
                ReduceKind::Sum => Elementwise::Add,
            };
            // ⭐ A ROW BLOCK SEVERAL STICKS WIDE (head dim 128) combines one stick column at a time: a
            // window at a row corner is addressable only inside one column plane of its view.
            let columns = if column {
                1
            } else {
                (src.c_len / stick).max(1)
            };
            let in_column = |r: Region, c: u32| {
                if columns == 1 {
                    r
                } else {
                    let c_start = r.c_start + c * stick;
                    Region {
                        c_start,
                        c_len: stick,
                        c_cover: (c_start, c_start + stick),
                        ..r
                    }
                }
            };
            for j in 1..blocks {
                let lhs = if j == 1 {
                    window(0)
                } else {
                    sticks(Region {
                        is_out: false,
                        ..acc
                    })
                };
                for c in 0..columns {
                    out.extend(super::lower_ktir_to_superdsc::elementwise(
                        f.name,
                        combine,
                        &[
                            in_column(lhs, c),
                            in_column(window(j), c),
                            in_column(sticks(acc), c),
                        ],
                        &[],
                        sym_id_base,
                        Some(layout),
                    )?);
                }
            }
            inter.insert(
                res,
                Region {
                    is_out: false,
                    ..acc
                },
            );
            continue;
        }
        // `arith.mulf` by a SPLATTED CONSTANT is a scalar multiply, not an `Elementwise(Mul)` — the
        // tiebreak `Program`'s own doc names. Read per-op, because one decoder layer states four
        // different multipliers. Checked BEFORE the 1:1 map, which would otherwise route it to
        // `Elementwise(Mul)` and then fail on the splat operand having no `Region`.
        let scalar = (op.op_type == OpKind::ArithMulf)
            .then(|| splat_scale_of(f, op))
            .flatten()
            // A constant beside a BROADCAST operand has no scalar form: the multiply is two sprays,
            // so the splat is materialized and the op runs as its own elementwise kind.
            .filter(|(_, t)| !in_body && !bcasts.contains_key(t));
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
        // AN EQUALITY MASK'S RETYPE is the fold's terminal: the `Elementwise(Equal)` stands for the
        // `cmpf(oeq)`+`uitofp` pair, and its inputs are the comparison's own operands (resolved
        // below). Checked before the 1:1 map, which has no `ArithUitofp` arm — an unfolded retype
        // falls to the by-name refusal, correctly.
        let equal = (op.op_type == OpKind::ArithUitofp)
            .then(|| {
                op.result
                    .and_then(|r| equal_masks.by_retype.get(&r).copied())
            })
            .flatten();
        // A dtype-convert cast, 1:1 in (op kind, element pair) — see [`convert_kind_of`]. `Err` for
        // an `extf`/`fptosi` whose pair is outside the two recipes (the named refusal lives there);
        // `None` for every other op kind, which falls through to the 1:1 map.
        let convert = convert_kind_of(f, op)?;
        // An integer arith op the loop top admitted (so [`integer_compute_of`] proved it data).
        // `None` for every non-integer op; a `muli`/`addi` outside the compute set never reaches
        // here (the loop top skipped it), so the arm cannot fire for addressing.
        let int_binary = match op.op_type {
            OpKind::ArithMuli => Some(Lowering::IntBinary(IntBinary::Mul)),
            OpKind::ArithAddi => Some(Lowering::IntBinary(IntBinary::Add)),
            // ⛔ `subi` IS DATA when the fixpoint says so (the loop top admitted it), but the
            // device's integer vocabulary has no subtract — the card's own `addi32toi32` with a
            // negated operand is the spelling, and choosing it here would rewrite the producer's
            // arithmetic. Refused by name, never silently re-meaned.
            OpKind::ArithSubi => {
                return err(format!(
                    "{}: an `arith.subi` this walk classifies as integer COMPUTE — the device's \
                     integer vocabulary has no subtract (arch_enums.cpp's opFuncs carry \
                     addi32toi32/muli32toi32 only), and rewriting `a - b` as `a + (-b)` here would \
                     re-mean the producer's arithmetic rather than lower it. State the negation \
                     in the kernel and keep the add",
                    f.name
                ));
            }
            _ => None,
        };
        // `x min c` / `x max c` with `c` a splatted constant — the tiebreak `splat_scale_of` states
        // for a multiply, applied to the two binaries whose constant side the device reads the same
        // way (a bound `[1,1]` const sprayed over the tile).
        let scalar_binary = match op.op_type {
            OpKind::ArithMinnumf => Some(Elementwise::Minimum),
            OpKind::ArithMaxnumf => Some(Elementwise::Maximum),
            _ => None,
        }
        .and_then(|e| splat_scale_of(f, op).map(|(v, t)| (e, v, t)))
        .filter(|(_, _, t)| !in_body && !bcasts.contains_key(t));
        // THE BATCHED KINDS — a rank-3 `linalg.batch_matmul`, and a rank-3 `linalg.transpose` that
        // swaps the two trailing axes: the batch is the leading extent of the result type.
        let rank3_batch = || match op.result_type {
            Some(ktir_core::irtype::IrType::Tensor { dims, .. }) if dims.len() == 3 => {
                u32::try_from(dims[0]).ok()
            }
            _ => None,
        };
        let batched = match op.op_type {
            OpKind::LinalgBatchMatmul => {
                let Some(batch) = rank3_batch() else {
                    return err(format!(
                        "{}: a `linalg.batch_matmul` whose result is not a rank-3 `[b, m, n]` tensor",
                        f.name
                    ));
                };
                Some(Lowering::BatchMatmul {
                    batch,
                    activation: super::lower_ktir_to_superdsc::BatchFrame::Flat,
                })
            }
            OpKind::LinalgTranspose if rank3_batch().is_some() => {
                let perm = op.attributes.iter().find_map(|(k, v)| match (k, v) {
                    (AttrKey::Permutation, Attr::IntList(p)) => Some(*p),
                    _ => None,
                });
                if perm != Some(&[0, 2, 1][..]) {
                    return err(format!(
                        "{}: a rank-3 `linalg.transpose` with permutation {perm:?} — the batched \
                         transpose this door lowers keeps the batch axis and swaps the two trailing \
                         ones (`[0, 2, 1]`)",
                        f.name
                    ));
                }
                // Its input in blocks when a permute laid it so ([`batch_blocks`]).
                let input = match op.operands.first().and_then(|v| inter.get(v)) {
                    Some(r) if batch_blocks.contains_key(&r.tid) => {
                        super::lower_ktir_to_superdsc::BatchFrame::Blocks
                    }
                    _ => super::lower_ktir_to_superdsc::BatchFrame::Flat,
                };
                rank3_batch().map(|batch| Lowering::BatchTranspose { batch, input })
            }
            _ => None,
        };
        let Some(program) = terminal
            .map(|c| {
                if c.up.is_some() {
                    Lowering::Silu
                } else {
                    // The BARE form: the longhand's result is stored direct, so the fused op is
                    // the UNARY `Elementwise(Silu)` — `("silu", 1)` over `[gate, out]`, exactly
                    // the op the per-`Program` door's elementwise arm emitted for one before the
                    // reroute made this walk the door for elementwise too.
                    Lowering::Node(Program::Elementwise(Elementwise::Silu))
                }
            })
            .or(rms.map(|c| Lowering::RmsNorm(c.eps, c.ones)))
            .or(reduce.map(Lowering::Reduce))
            .or(scalar.map(|(v, _)| Lowering::ScalarMul(v)))
            .or(bound_scale.map(Lowering::BoundScalarMul))
            .or(equal.map(|_| Lowering::Node(Program::Elementwise(Elementwise::Equal))))
            .or(convert.map(Lowering::Convert))
            .or(int_binary)
            .or(scalar_binary.map(|(e, v, _)| Lowering::ScalarBinary(e, v)))
            .or(batched)
            .or_else(|| program_of(op.op_type).map(Lowering::Node))
        else {
            return err(format!(
                "{}: `{:?}` has no 1:1 `Program`, so this door cannot lower it without choosing a \
                 node kind on the producer's behalf. Either it belongs to a fused kind the producer \
                 should state by calling that entry point directly, or it needs a `Program` variant",
                f.name, op.op_type
            ));
        };

        // ⭐ THE OP'S OWN RESULT FORMAT, read off the result type and never assumed: every op this
        // door had lowered before the fp32 island is fp16, and the island's multiply-add states
        // F32. It is threaded to the MINT (the intermediate reserves the ACTUAL dtype's width —
        // an f32 buffer at an fp16 reservation is half the bytes, aliased onto the next tensor)
        // and to the ELEMENTWISE EMISSION (the stick basis and `dataFormat_` are the dtype's own
        // facts). Read once here so no consumer re-derives it and drifts.
        let op_df = result_df_of(f.name, op)?;

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
            // The silu's two inputs are the chain's gate and up, which the recogniser proved. (A
            // bare silu reaches the `Node(Elementwise(Silu))` arm, whose arity `elementwise_op_func`
            // states as 1.)
            Lowering::Silu => 2,
            // `linalg.reduce`'s operands are `(data, init)`: the init is the SEED, which the reduce
            // op-func carries itself, so only the data is an input.
            Lowering::Reduce(_) => 1,
            // The splat is not an operand of the descriptor — it rides in the op as a bound `[1,1]`
            // const — so a scalar multiply reads ONE tensor.
            Lowering::ScalarMul(_) | Lowering::BoundScalarMul(_) => 1,
            // x and gamma; the epsilon, `1/cols` and the oneplus offset are the fused body's own.
            Lowering::RmsNorm(..) => 2,
            // A dtype-convert reads ONE tensor; the formats ride the `OpFunc`
            // (`convert_dtypes`), never the operand list.
            Lowering::Convert(_) => 1,
            // A wrapping i32 binary reads TWO tensors; the format (SENUINT32) rides the op's
            // own result type through `op_df`.
            Lowering::IntBinary(_) => 2,
            Lowering::Node(Program::Matmul) => 2,
            Lowering::Node(Program::Elementwise(e)) => {
                match super::lower_ktir_to_superdsc::elementwise_op_func(f.name, e) {
                    Ok((_, arity)) => arity,
                    Err(e) => return Err(e),
                }
            }
            Lowering::Node(Program::Transpose) => 1,
            // The constant side rides the op as a bound `[1,1]` const, like a scalar multiplier's.
            Lowering::ScalarBinary(..) => 1,
            // A batched contraction's `outs` init is no input, exactly as the 2-D one's.
            Lowering::BatchMatmul { .. } => 2,
            Lowering::BatchTranspose { .. } => 1,
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
            // THE EQUALITY MASK's inputs are the comparison's own two operands, not the retype's
            // (the retype reads the i1 the fold destroys). The broadcast resolution below then
            // applies to them exactly as it would to a spelled two-tensor op's.
            _ if equal.is_some() => {
                let (a, b) = equal.expect("just matched");
                vec![a, b]
            }
            (Some(c), _, _) => match c.up {
                Some(up) => vec![c.gate, up],
                // The bare form's one input is the gate; the divide's other operand is the
                // chain-interior `1 + e^-x`, which the fused op never materialises.
                None => vec![c.gate],
            },
            // The TENSOR operand the recogniser proved, not operand 0 — the splat sits on either side
            // (`ms * INV_D` has it second, and nothing obliges a producer to put it there).
            (None, Some((_, tensor)), _) => vec![tensor],
            // RUNG 3: same rule, the splat's other side. The bound scale is a `[1,1]` parameter, not
            // an operand of the descriptor.
            (None, None, Some(b)) => vec![b.tensor],
            // A splatted-constant binary's tensor side, wherever the splat sits.
            (None, None, None) if scalar_binary.is_some() => {
                vec![scalar_binary.expect("just matched").2]
            }
            (None, None, None) => op.operands.iter().copied().take(n_in).collect(),
        };

        // A BROADCAST OPERAND READS ITS SOURCE. The chain minted no buffer (it emitted no op), so the
        // value the consumer names has no region of its own; the rank-2 source it sprays does. The AXIS
        // the recogniser proved is then CHECKED against that region's shape below rather than trusted:
        // two independent facts about the same operand, so a chain whose axis and extent disagree
        // cannot address a full tile as a vector.
        let mut bcast_axes: Vec<Option<BcastAxis>> = in_values
            .iter()
            .map(|v| bcasts.get(v).map(|b| b.axis))
            .collect();
        // ⭐⭐⭐ THE FLAT FORM'S FLAG RIDES BESIDE THE AXIS, because the two say different things:
        // the axis says which MODE the consumer addresses the operand in, the flag says the
        // source is a full tile whose FLAT elements are the vector — a `[256, 4]` running max
        // is the `[1024, 1]` column vector of the flat frame — so the region handed downstream
        // is PRESENTED flat (see the presentation in the region loop below) rather than being
        // the source's own `[256, 4]` window. Rank 2 keeps `false` and its verbatim behavior.
        let bcast_flat: Vec<bool> = in_values
            .iter()
            .map(|v| bcasts.get(v).is_some_and(|b| b.flat))
            .collect();
        let in_values: Vec<Ssa> = in_values
            .iter()
            .map(|v| bcasts.get(v).map_or(*v, |b| b.src))
            .collect();

        // ⭐⭐⭐⭐⭐ THE ROW-TILED FAMILY FOLD'S CANDIDACY GATE. Only the foldable pointwise kinds
        // (the same `program` this loop just derived — never re-derived here, so the gate cannot
        // disagree with the loop about what op this is), only with every operand DENSE (a
        // broadcast family's blocks do not tile the view the same way and the broadcast arms
        // address operands at their own extent). The analysis itself bails on everything that is
        // not a row-blocked family, so a single-window or column-chunked program never enters it.
        if bcast_axes.iter().all(Option::is_none)
            && matches!(
                program,
                Lowering::Silu
                    | Lowering::ScalarMul(_)
                    | Lowering::BoundScalarMul(_)
                    | Lowering::Node(Program::Elementwise(_))
            )
            && let Some(kind) = classify_row_fold(k, op, &analyses)
            && let Some(members) = row_tiled_family(k, op, kind, &in_values, n_in, &analyses)?
        {
            row_fold_skip.extend(members);
        }

        let mut per_op: Vec<Region> = Vec::with_capacity(n_in + 1);
        // ⭐ WHICH OPERANDS ARE PARAMETER-BACKED, kept BESIDE the regions because a `Region` cannot
        // say. `region_for_operand` answers `Some` for a value the parameter-rooted walk reaches (a
        // `ktdp.load` of a `ktdp.construct_access_tile` of a `ktdp.construct_memory_view` of a
        // function argument) and `None` for one this function COMPUTED — and that is exactly the
        // distinction [`transposed_b_is_placeable`] needs, so it is recorded here rather than
        // re-derived from a footprint that is identical either way.
        let mut from_parameter: Vec<bool> = Vec::with_capacity(n_in + 1);
        for (i, v) in in_values.iter().enumerate() {
            // ⭐⭐⭐ THE FLAT BROADCAST'S SOURCE IS PRESENTED IN THE FLAT FRAME — the same bytes
            // at the same offsets, addressed as `[numel, 1]` (Col) / `[1, numel]` (Mb). The
            // source's own window is a full tile (`[256, 4]`); the consumer's operand mode
            // addresses one element per flat row, and the extent check + the emission-side want
            // check below read the PRESENTED shape, so the flat form flows through them — and
            // through `In::col`/`In::mb` and the minted-buffer annotation — without a rank-3
            // arm anywhere downstream. Rank 2 presents nothing (`bcast_flat` is `false`) and
            // keeps its verbatim behavior.
            // ⛔ AND THE PRESENTATION IS CHECKED AT THE REFRAME: a multi-column source presented
            // `[numel, 1]`, or a multi-stick source presented `[1, numel]`, is a re-frame the
            // bytes do not support, and `presented_flat`'s `RegionFrame` law refuses it HERE —
            // a build failure naming the arithmetic, never a clean bake of wrong bytes.
            let present = |mut r: Region| -> Result<Region, Error> {
                if bcast_flat[i] {
                    r = presented_flat(
                        &r,
                        bcast_axes[i].expect("bcast_flat is set only where an axis is"),
                    )
                    .map_err(|e| Error {
                        message: format!("{}: {e}", f.name),
                    })?;
                }
                r.is_out = false;
                Ok(r)
            };
            match region_for_operand(k, *v)? {
                Some(r) => {
                    per_op.push(present(r)?);
                    from_parameter.push(true);
                }
                // A value the walk cannot reach is either an intermediate this function already
                // minted a buffer for, or a constant. The first is looked up; the second is
                // refused, because picking `ScalarMul` for it would be choosing a node kind on the
                // producer's behalf.
                None => match inter.get(v) {
                    Some(r) => {
                        per_op.push(present(*r)?);
                        from_parameter.push(false);
                    }
                    // ⭐ A SPLATTED CONSTANT READ AS A DENSE OPERAND IS A BUFFER OF ITS ONE VALUE —
                    // `tensor.splat`'s own lowering: minted at the splat's frame and filled from the
                    // registry's bound const ([`super::lower_ktir_to_superdsc::fill`]). The op that
                    // reads it keeps its own kind; only a multiply/min/max beside a dense tensor
                    // reads its constant side as a scalar instead (`ScalarMul`, `ScalarBinary`).
                    None => match splat_buffer(k, layout, floor, *v)? {
                        Some((r, value)) => {
                            out.extend(super::lower_ktir_to_superdsc::fill(
                                f.name,
                                value,
                                &r,
                                sym_id_base,
                                layout,
                            )?);
                            let r = Region { is_out: false, ..r };
                            inter.insert(*v, r);
                            per_op.push(present(r)?);
                            from_parameter.push(false);
                        }
                        None => {
                            return err(format!(
                                "{}: `{:?}` input {i} is neither a load of a parameter nor an \
                                 intermediate this function produced, nor a splatted float \
                                 constant (a splat of anything else names no one value to \
                                 fill)",
                                f.name, op.op_type
                            ));
                        }
                    },
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
                // One row of its view: the whole view at one row, or a one-row window of a taller
                // one (the walk's mask tile, row `i` of the stack), which the emitter addresses at
                // its own row.
                BcastAxis::Mb => r.v_rows == 1 || r.r_len == 1,
                // ⭐ THE ROW-SPLIT SOURCE IS A FULL TILE ON PURPOSE — the opposite of degenerate
                // — so there is nothing to check in THIS frame. Its proof is the consumer's own
                // want check (`elementwise_broadcast`: the source at `[rows/y, c_len]` against
                // the result's rows) plus the `RegionFrame::split_rows` law the consumer runs
                // before building the operand: `rows == y·mb`, single stick. Those two
                // independent statements own it; a third here would be a copy that can drift.
                BcastAxis::Y(_) => true,
            };
            if !degenerate {
                let what = match axis {
                    BcastAxis::Col => "column",
                    BcastAxis::Mb => "row",
                    BcastAxis::Y(_) => "row-split",
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

        // ⭐⭐⭐⭐⭐ THE BAND'S OWN `Mb` DERIVATION — a shared single-row table against a banded
        // sweep (auditor-approved with the band design, 2026-10-27). A flush band's per-row
        // operand is `[B, n]` while the tables it combines with are functions of the vocab
        // alone — whole-view `[1, n]` parameters the program never spells a
        // `linalg.broadcast` for (Triton's implicit broadcast folds into the load's own tile
        // shape), so `bcasts` states no axis and the DENSE path's extents law refuses. The
        // device expresses it exactly: an `Mb` operand is one row of values sprayed over the
        // rows, `In::mb`'s own mode. The derivation is keyed on the TYPED band, never a
        // shape guess: it fires only when an input CARRIES a band (`RowBand`, constructed
        // only by `windowed_region`'s uniform-row arm) whose height exceeds one row, and
        // only for the kinds whose door consumes the axis — the elementwise door (the
        // original admission) and, since wall 10 (auditor GO 2026-10-27, the intmb
        // scatter-harness leg), the INTEGER door: the batched sampler's hash is an
        // `IntBinary` mixing the whole-row `lane` table (`[1, PAD_V]` i32) with the
        // band-typed tile (`sample_fwd.py`'s `h = lane * hc1 + slo`), the same
        // program-invariant-table law this derivation states. The operand must be
        // whole-view single-row at row 0, un-banded, and as wide as the OUTPUT's own column
        // extent — the width check is against the op's stated result, not against a sibling
        // input, because the banded operand beside the table can be the reduce's `[B, 1]`
        // per-row scalar (mode `Col`), whose one column says nothing about the spray's width.
        // A table wider or narrower than the output is still refused by name at the extents
        // law below; every existing kernel (no banded input) is untouched.
        if matches!(
            program,
            Lowering::Node(Program::Elementwise(_)) | Lowering::IntBinary(_)
        ) && let Some(band) = band_of(f.name, &per_op)?
            && band.height() > 1
        {
            let out_cols = op
                .result_type
                .and_then(|t| t.dims())
                .and_then(|d| d.last().copied());
            for (i, x) in per_op.iter().enumerate() {
                if bcast_axes[i].is_none()
                    && x.band.is_none()
                    && x.r_start == 0
                    && x.r_cover == (0, x.v_rows)
                    && x.v_rows == 1
                    && out_cols == Some(x.c_len as i64)
                {
                    bcast_axes[i] = Some(BcastAxis::Mb);
                }
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
                    // ⭐ THE OP'S FIRST GATHERED INPUT, by identity. A second gathered input of the
                    // same op is NOT checked here: this loop takes the first match, and the whole-
                    // function consumer counting that refuses a gathered load with two consumers is
                    // the matmul materializer's own (`gathered_matmul_materializes`). An op whose
                    // SECOND input is also a gathered load of another tile reaches that refusal, or
                    // the ScalarMul-only rule below — not a mid-walk one.
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
        let gathered_b = if let Some((g, _ti, gi)) = op_gather
            && matches!(program, Lowering::Node(Program::Matmul))
        {
            gathered_matmul_materializes(
                k,
                layout,
                &mut GatherEmit {
                    sym_id_base,
                    out: &mut out,
                    floor,
                    inter: &mut inter,
                },
                g,
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

        // ⛔ A BATCHED CONTRACTION READS ITS KERNEL AS PER-BATCH `[k, n]` BLOCKS — written so by a
        // batched transpose, or a one-stick-wide flat frame (the same bytes). Any other kernel's
        // flat frame interleaves the batches stick by stick, and no base offset names one block.
        // ⛔ AND A BATCH-BLOCKED BUFFER WIDER THAN A STICK IS READ ONLY AS BLOCKS — by a batched
        // transpose (its input) or a batched contraction (its kernel). Anything else reads the flat
        // frame, whose stick columns those bytes are not.
        for (i, x) in per_op.iter().enumerate() {
            let reads_blocks = matches!(
                (&program, i),
                (Lowering::BatchTranspose { .. }, 0) | (Lowering::BatchMatmul { .. }, 1)
            );
            if wide_blocks(&batch_blocks, x.tid) && !reads_blocks {
                return err(format!(
                    "{}: input {i} t{} `[{}, {}]` holds batch blocks one after another, which only \
                     a batched transpose's input or a batched contraction's kernel reads",
                    f.name, x.tid, x.v_rows, x.v_cols
                ));
            }
        }
        if let Lowering::BatchMatmul { .. } = program
            && let Some(w) = per_op.get(1)
            && !(batch_blocks.contains_key(&w.tid) || w.c_len == crate::work::FP16_ELEMS_PER_STICK)
        {
            return err(format!(
                "{}: a batched contraction's kernel t{} is a `[{}, {}]` flat frame no batched \
                 transpose wrote — its batches' `[k, n]` blocks are not contiguous runs of it",
                f.name, w.tid, w.v_rows, w.v_cols
            ));
        }
        // THIS OP'S OUTPUT: the parameter a `ktdp.store` writes from this op's result — directly, or
        // through a shape op that moves no byte (the walk's `[NKVH, ROWS, HD]` result stored as the
        // `[NQH, MQ, HD]` the output view states).
        // ⭐ A STORE THROUGH AN INDIRECT TILE IS A SCATTER, NOT THIS OP's OUTPUT: the value is an
        // intermediate, and [`indirect_store_of`] copies it to the rows its index names after the op
        // writes it (spyre-inference's `out.index_copy_(0, query_row_index, attn)`).
        let stores_of = |v: Option<Ssa>| {
            f.operations.iter().find(|s| {
                s.op_type == OpKind::KtdpStore
                    && s.operands.first().copied() == v
                    && !indirect_tile_of(f, s.operands.get(1).copied())
            })
        };
        let mut stored = stores_of(op.result).or_else(|| {
            let res = op.result?;
            let shape = f.operations.iter().find(|o| {
                matches!(
                    o.op_type,
                    OpKind::TensorReshape | OpKind::TensorCollapseShape | OpKind::TensorExpandShape
                ) && o.operands.first() == Some(&res)
            })?;
            // Only through a shape op that moves no byte: one that does is a stick permute, which
            // the shape op's own arm lowers into the stored parameter.
            let dims = |o: &Operation<'_>| -> Vec<i64> {
                o.result_type
                    .and_then(|t| t.dims())
                    .map(<[i64]>::to_vec)
                    .unwrap_or_default()
            };
            same_card_bytes(&dims(op), &dims(shape)).then_some(())?;
            stores_of(shape.result)
        });
        // ⭐⭐⭐ THE fp8 WEIGHT-SCALE EPILOGUE, ABSORBED AT THE DOT. `matmul_fp8.py` spells the W8A8
        // contraction's scale multiply IN THE KERNEL — `p * ws` between the dot and the store — so
        // the dot's result is never stored directly: without this arm the walk would mint it an
        // intermediate and `matmul_oriented` would refuse the two-input fp8 weight ("a packed fp8
        // weight comes with its per-column `w_scale`"). The multiply is NOT a separate device op —
        // the W8A8 primitive applies the scale inside the contraction (`matmul_fp8_descriptors`
        // takes the ws as its third fact) — so the spelled `arith.mulf` is absorbed HERE, at the op
        // it belongs to, rather than emitted as a pointwise op over a minted buffer. Three seals
        // restate the kernel's contract, and each refuses by name:
        //   1. the dot's result has exactly ONE consumer, and it is the scale `arith.mulf`;
        //   2. the mulf's other operand is the broadcast row `program_broadcast_chains` proved (a
        //      `[1, n]` weight scale sprayed down the rows) — or, at `m == 1`, the `[1, n]` load
        //      ITSELF, the spray being the identity the frontend folds away — and its source
        //      resolves to a BOUND PARAMETER, the ws the launch stages;
        //   3. the mulf's result is what a `ktdp.store` writes, so the contraction's output region
        //      is the store's own.
        // ⛔ fp8 ONLY. An fp16 dot whose result feeds a multiply is a REAL two-op chain (the
        // attention score's scale, the rms mean's `· INV_D`) and lowering it as separate ops is the
        // walk's own law; absorbing those would fold a pointwise op into a contraction that has no
        // scale fact to carry it.
        if stored.is_none()
            && matches!(program, Lowering::Node(Program::Matmul))
            && per_op.len() >= 2
            && per_op[1].is_fp8
        {
            let Some(dot) = op.result else {
                return err(format!(
                    "{}: `{:?}` states no result to look for a scale epilogue on",
                    f.name, op.op_type
                ));
            };
            let deep = f.ops_deep();
            let mut consumers = deep.iter().copied().filter(|o| o.operands.contains(&dot));
            let Some(mulf) = consumers.next() else {
                return err(format!(
                    "{}: the fp8 dot's result has no consumer — `matmul_fp8.py` spells the scale \
                     multiply `p * ws` between the dot and the store, so an fp8 contraction whose \
                     result nothing reads is a new producer form this arm must be re-derived for",
                    f.name
                ));
            };
            if mulf.op_type != OpKind::ArithMulf || consumers.next().is_some() {
                return err(format!(
                    "{}: the fp8 dot's result must feed exactly one scale `arith.mulf` (the \
                     kernel's `p * ws` epilogue) — anything else is a producer form the W8A8 \
                     primitive has no fact for",
                    f.name
                ));
            }
            let Some(&scale_v) = mulf.operands.iter().find(|v| **v != dot) else {
                return err(format!(
                    "{}: the fp8 dot's scale `arith.mulf` names no operand besides the dot itself",
                    f.name
                ));
            };
            // ⭐ SEAL 2, BOTH SPRAY FORMS. The kernel sprays the `[1, n]` ws down the rows with a
            // `linalg.broadcast` — every form except `m == 1`, where the spray is the IDENTITY: a
            // `[1, n]` row over a one-row dot needs no broadcast op, and the frontend folds it
            // away, so the mulf reads the LOAD itself (`matmul_fp8.py` at the decode shape). The
            // direct form is admitted only when the load's own window is one row — the same
            // degenerate-extent proof the broadcast chain's no-expand arm turns on — because at
            // `m > 1` the frontend MATERIALSES the broadcast (the mulf's operand shapes differ),
            // so a bare `[1, n]` load here can only be the m=1 identity spray of the same
            // per-column scale.
            let ws_src = match bcasts.get(&scale_v) {
                Some(b) => b.src,
                None => {
                    let direct = f.operations.iter().any(|d| {
                        d.result == Some(scale_v)
                            && d.op_type == OpKind::KtdpLoad
                            && super::lower_ktir_to_superdsc::shape_2d(d)
                                .is_some_and(|(r, _)| r == 1)
                    });
                    if direct {
                        scale_v
                    } else {
                        return err(format!(
                            "{}: the fp8 dot's scale operand is not a recognised `linalg.broadcast` \
                             row — the W8A8 primitive's ws is a `[1, n]` per-column scale sprayed \
                             down the rows, and anything else is a shape the contraction's scale \
                             fact cannot state",
                            f.name
                        ));
                    }
                }
            };
            let Some(mut ws_r) = region_for_operand(k, ws_src)? else {
                return err(format!(
                    "{}: the fp8 dot's scale row resolves to no bound parameter — the ws is a \
                     tensor the launch stages, and a scale computed inside this function is a \
                     producer form this arm refuses",
                    f.name
                ));
            };
            let Some(mulf_res) = mulf.result else {
                return err(format!(
                    "{}: the fp8 dot's scale `arith.mulf` states no result",
                    f.name
                ));
            };
            let Some(mulf_store) = f.operations.iter().find(|s| {
                s.op_type == OpKind::KtdpStore && s.operands.first().copied() == Some(mulf_res)
            }) else {
                return err(format!(
                    "{}: the fp8 dot's scaled result is never stored — the kernel's epilogue ends \
                     at `o_desc.store` and an unstored contraction output has no region to emit",
                    f.name
                ));
            };
            ws_r.is_out = false;
            per_op.push(ws_r);
            from_parameter.push(true);
            fp8_epilogue.insert(mulf_res);
            stored = Some(mulf_store);
        }
        let Some(store) = stored else {
            // NOT STORED => AN INTERMEDIATE. Mint a buffer for it, declare it to the layout, and
            // record it so the consuming op finds it as an input.
            let Some(res) = op.result else {
                return err(format!(
                    "{}: `{:?}` has no result to place",
                    f.name, op.op_type
                ));
            };
            // ⭐⭐⭐ A PER-ROW SCALAR OF A RANK-N TILE IS A COLUMN — a reduction over a rank-N
            // tile's last axis writes one value per row of its `[rows, cols]` flat frame, i.e. the
            // `[rows, 1]` column `assemble_reduce_seeded` writes, whatever rank the program states
            // it at (`[B, G]` for a `[B, G, PS]` tile); and a pointwise op over such a column (the
            // running max's `max`, the rescale's `exp`) keeps it one. Framed `[numel, 1]`, not the
            // stated shape's own `[B, G]`, which would be a different stick per row.
            let numel: Option<i64> = op
                .result_type
                .and_then(|t| t.dims())
                .filter(|d| d.len() >= 2)
                .map(|d| d.iter().product());
            let per_row_scalar = numel.is_some_and(|n| {
                matches!(program, Lowering::Reduce(_))
                    || per_op
                        .iter()
                        .any(|r| !r.is_out && r.c_len == 1 && i64::from(r.r_len) == n)
            });
            let dims = match op.result_type {
                _ if per_row_scalar => [numel.expect("just checked") as u32, 1],
                Some(ktir_core::irtype::IrType::Tensor { dims, .. }) if dims.len() == 2 => {
                    [dims[0] as u32, dims[1] as u32]
                }
                // ⭐⭐⭐ A RANK > 2 INTERMEDIATE IS MINTED IN THE FLAT FRAME — every axis but the
                // last folded into rows — the same reading [`shape_2d`] states and every consumer
                // of the minted buffer addresses through. Row-major, the rank-N tile and its flat
                // pair are the same bytes at the same offsets, so one buffer serves both; minting
                // the STATED shape instead (`[256, 4, 256]`) would be a `[4, 256]` buffer under
                // the old reading — a quarter of the footprint the flat `[1024, 256]` consumers
                // address. The batched paged-decode kernel's score tiles and softmax states are
                // all rank-3; without this arm every one of them was refused below as "states no
                // 2-D tensor type". Rank 2 keeps the arm above verbatim.
                Some(ktir_core::irtype::IrType::Tensor { dims, .. }) if dims.len() > 2 => {
                    let rows: i64 = dims[..dims.len() - 1].iter().copied().product();
                    [rows as u32, *dims.last().expect("dims.len() > 2") as u32]
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
            // ⭐⭐⭐⭐⭐ A BANDED OP'S INTERMEDIATE IS THE BAND'S HEIGHT, NOT THE PER-CORE TILE'S.
            //
            // The program states the per-core tile (`[32, 64]` for one rope copy over 32
            // positions), but [`windowed_region`] resolves each operand as the BAND (`[1024, 64]`
            // — the launch's own tiles, flush), and the op sweeps the band: a mint at the tile's
            // height would be a buffer the emission overruns by `positions − 1` tiles on every
            // core past the first. The band is read off the inputs the same way
            // [`pointwise_rows`] reads the sweep off the output — a WINDOWED input states it as
            // its `r_len` — while a function whose inputs are all whole-view keeps the result
            // type's own rows, so every mint this door has ever made is byte-identical. (A chain
            // whose every input is a minted band intermediate states no windowed input; its mint
            // stays at the tile and the extents law refuses the sweep mismatch by name — refused
            // at the op that could have extended it, not silently mis-sized.)
            // A BROADCAST operand's window is the vector it sprays, not the op's sweep (the walk's
            // mask tile, `[ROWS, BLOCK]` of the stack, sprayed over every kv head's rows) — unless
            // its window is the launch's own flush band, which IS the sweep (the batched sampler's
            // `pid`-cornered per-row seed, `[B, 1]` sprayed along the `[B, V]` rows it banks).
            let swept: Vec<Region> = per_op
                .iter()
                .enumerate()
                .filter(|(i, r)| bcast_axes.get(*i).is_none_or(Option::is_none) || r.band.is_some())
                .map(|(_, r)| *r)
                .collect();
            let dims = [band_rows_of(&swept, dims[0]), dims[1]];
            let Some(layout) = layout else {
                return err(format!(
                    "{}: `{:?}`'s result is an intermediate, which needs a `BundleLayout` to hold \
                     its buffer -- `layout: None` is the unit-test arm and cannot place one",
                    f.name, op.op_type
                ));
            };
            // ⭐ A WALK'S YIELDED VALUE IS WRITTEN INTO ITS CARRIED BUFFER — the next trip reads it
            // there — once nothing later in the body reads the value the buffer still holds.
            let carried = yield_target.get(&res).copied();
            if let Some((arg, target)) = carried {
                if [target.v_rows, target.v_cols] != dims {
                    return err(format!(
                        "{}: the walk yields a `[{}, {}]` value into a carried `[{}, {}]` buffer",
                        f.name, dims[0], dims[1], target.v_rows, target.v_cols
                    ));
                }
                if last_read(arg).is_some_and(|last| last > oi) {
                    return err(format!(
                        "{}: the walk's carried %{} is read after the op that writes its next \
                         value — the card holds ONE buffer per carried value, so the program must \
                         read the old value before it computes the new one",
                        f.name, arg.0
                    ));
                }
            }
            let tid = match carried {
                Some((_, target)) => target.tid,
                None => mint_inter(k, layout, floor, res),
            };
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
            // ⛔ AND ONLY THE MATMUL AND CONVERT ARMS, because a POINTWISE result's bump is OPTIONAL and the
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
                    // The SAME `for_matmul` the emitter's `n_dev` takes — one decision for the
                    // buffer this mint reserves and the width the emit writes, so they cannot
                    // disagree (issue 201 item 4). `windowed = true`: this is the whole-function
                    // door, where the padded weight columns would come from the caller's windows.
                    [
                        dims[0],
                        DeviceWidth::for_matmul(dims[0], dims[1], k, true).get(),
                    ]
                }
                // ⭐⭐⭐ A CONVERT'S INTERMEDIATE IS RESERVED AT THE **DEVICE** WIDTH TOO (the β
                // ruling, front-b-noise, auditor 2026-10-27) — the SAME precedent, a different
                // producer rule. The door's β admission widens a sub-device-width column window
                // (the crossed convert's minted `[B, 1]` result, the sampler's gate chain) to
                // `DeviceWidth::for_convert` — the 64-element device width our own PlanCorelets
                // demands of a tile and `convert_opspec`'s own fp16-stick basis states — and the
                // emitter then SWEEPS that width over the buffer this mint reserves: a mint at
                // the logical `[B, 1]` (128 B/row) is half the bytes the convert addresses
                // (256 B/row), the exact `resolve_seg_base` over-run shape the matmul arm
                // records. ONE decision for the reservation and the sweep (`for_convert`), so
                // they cannot disagree. A fixed point at every 64-multiple — every convert
                // intermediate ever minted is full-row-width, so this arm is byte-identical for
                // all of them. The REGION stays `[B, 1]`: the consumer's Col window reads row
                // r's FIRST stick alone (the pad words the convert writes past it are never
                // read), the same inert-pad claim the matmul arm makes.
                Lowering::Convert(_) => [dims[0], DeviceWidth::for_convert(dims[1]).get()],
                _ => dims,
            };
            // ⭐ THE MINT RESERVES THE OP'S OWN RESULT FORMAT — `synth` is `synth_df(Fp16)`, and
            // an f32 intermediate (the widen island's multiply-add results) at an fp16
            // reservation is HALF the bytes the emission writes: the footprint guard would
            // refuse the first over-run, or without it the next intermediate would be aliased.
            // `synth_df` reserves the true width, the same fact `pointwise_broadcast_opspec_df`
            // threads into the descriptors.
            if carried.is_none() {
                layout.synth_df(crate::place::PlaceId::Act(tid), &dev_dims, op_df);
            }
            // ⭐⭐⭐ THE BAND PROPAGATES THROUGH THE MINT — a banded op's intermediate is a
            // banded operand of the NEXT op. The minted region states the band its inputs
            // stated (`band_of` refuses a disagreement), so a chain of ops over a flush band
            // mints every intermediate at the band's height: without this, the first mint
            // would drop the typed fact and the second op's inputs would look whole-view
            // again, minting `[1, ·]` while its per-row operands are `[B, ·]`.
            let band = band_of(f.name, &per_op)?;
            let r = Region {
                tid,
                v_rows: dims[0],
                v_cols: dims[1],
                r_start: 0,
                c_start: 0,
                r_len: dims[0],
                c_len: dims[1],
                r_cover: (0, dims[0]),
                c_cover: (0, dims[1]),
                is_out: true,
                is_fp8: false,
                band,
            };
            inter.insert(res, r);
            per_op.push(r);
            // ⭐⭐ A BATCHED CONTRACTION'S ACTIVATION WIDER THAN A STICK, READ AS BATCH BLOCKS. In the
            // flat frame a batch's rows are a slice whose sticks sit the TENSOR's rows apart, and the
            // card's contraction steps them by its own rows whatever the descriptor states — MEASURED
            // on granite-3.1-8b: each kv head's `[84, 128]` slice of the `[672, 128]` head-major query
            // contracted its second stick from the next kv head's first (the card's scores match that
            // read at rel-rms 0.0012, the intended one at 1.19). So the batches are copied into
            // standalone stick-major `[m, k]` blocks first — the head-major pair's block side, `[m,
            // b·k]`, one stick-run copy per stick of a batch ([`HeadMajorCopies::to_block`], the
            // attention output's own relayout) — and the contraction reads blocks.
            let program = match program {
                Lowering::BatchMatmul {
                    batch,
                    activation: super::lower_ktir_to_superdsc::BatchFrame::Flat,
                } if batch > 1
                    && per_op
                        .first()
                        .is_some_and(|a| a.v_cols > crate::work::FP16_ELEMS_PER_STICK) =>
                {
                    let a = per_op[0];
                    let stick = crate::work::FP16_ELEMS_PER_STICK;
                    if !a.v_rows.is_multiple_of(batch) || !a.v_cols.is_multiple_of(stick) {
                        return err(format!(
                            "{}: a {batch}-batch contraction's activation t{} `[{}, {}]` is not \
                             whole batches of whole sticks",
                            f.name, a.tid, a.v_rows, a.v_cols
                        ));
                    }
                    let (m, sticks) = (a.v_rows / batch, a.v_cols / stick);
                    let tid = mint_scratch(k, layout, floor, res, 1);
                    let blocks = stick_major_region(tid, m, batch * sticks, layout);
                    let copies = HeadMajorCopies::of(&blocks, &a, a.v_rows, sticks)?.to_block(
                        f.name,
                        sym_id_base,
                        layout,
                    )?;
                    // ⭐ AN ACTIVATION THE WALK'S BODY DOES NOT DEFINE IS LAID OUT ONCE, BEFORE THE
                    // WALK. The body relaunches once per page, so the decode query's copies — the
                    // same bytes on every trip — would run once per page; they go ahead of the
                    // body's first descriptor instead, into scratch no trip writes (a synthetic
                    // intermediate's footprint is its own, [`crate::placement::BundleLayout::synth`]).
                    let invariant = in_body
                        && fold.is_some_and(|fd| {
                            in_values.first().is_some_and(|v| {
                                !fd.body.contains(v) && fd.carried.iter().all(|c| c.arg != *v)
                            })
                        });
                    if invariant {
                        let n = copies.len();
                        out.splice(body_out.0..body_out.0, copies);
                        body_out.0 += n;
                    } else {
                        out.extend(copies);
                    }
                    per_op[0] = Region {
                        tid,
                        is_out: false,
                        ..a
                    };
                    batch_blocks.insert(tid, a.v_cols);
                    Lowering::BatchMatmul {
                        batch,
                        activation: super::lower_ktir_to_superdsc::BatchFrame::Blocks,
                    }
                }
                other => other,
            };
            let name = f.name;
            let mut emitted = emit_one(
                name,
                program,
                &per_op,
                sym_id_base,
                Some(layout),
                headmajor,
                WalkProof {
                    b_orient,
                    gather: op_gather.map(|(g, _, _)| g),
                    bcast: &bcast_axes,
                    df: op_df,
                },
            )?;
            out.append(&mut emitted);
            if let Lowering::BatchTranspose { .. } = program {
                batch_blocks.insert(r.tid, r.v_cols);
            }
            // ⭐ AND WHERE THE PROGRAM STORES THE VALUE THROUGH AN INDEX, THE ROWS GO THERE NOW.
            if let Some(sc) = indirect_store_of(k, res)? {
                out.extend(super::lower_ktir_to_superdsc::row_scatter(
                    f.name,
                    &reframed(r, [sc.entries, sc.cols]),
                    sc,
                    sc.value_rows,
                    sym_id_base,
                    Some(layout),
                )?);
            }
            lowered += 1;
            continue;
        };
        let out_tile = store.operands.get(1).copied();
        let tile = f
            .operations
            .iter()
            .find(|o| o.op_type == OpKind::KtdpConstructAccessTile && o.result == out_tile);
        // ⭐⭐⭐⭐⭐ THE STORE's OWN TILE IS THE OUTPUT's WINDOW — the load side's own law
        // ([`region_for_operand`]'s "THE WINDOW IS THE LOAD'S OWN TILE"), stated for the store.
        // `regions()` keeps the FIRST access tile over each parameter's view and falls back to
        // the whole view when a corner is not a constant, so a banded program's stores would
        // hand the body the whole `[v_rows, v_cols]` window: the descriptor would sweep rows the
        // program never stores and address the band's own rows at the buffer base. The store's
        // tile states the window, and `windowed_region` resolves it — constants or the per-core
        // uniform band — through the ONE walk the load side takes, so the two cannot disagree.
        let Some(tile) = tile else {
            return err(format!(
                "{}: `{:?}`'s store names no access tile",
                f.name, op.op_type
            ));
        };
        let Some(mut o) = region_of_view_operand(k, tile, 0)? else {
            return err(format!(
                "{}: `{:?}`'s store names no parameter",
                f.name, op.op_type
            ));
        };
        o = windowed_region(k, o, tile)?;
        o.is_out = true;
        per_op.push(o);
        // ⭐⭐⭐ A STORED RESULT IS STILL READABLE. The value this op writes to the store's
        // parameter lives in that parameter's buffer, and a LATER compute consumer of it
        // (the noise probe's `u`: stored as an output AND fed to `log`) must find it there
        // — without this record the consumer's lookup falls through to "a constant (a
        // `tensor.splat`)", the walk's only other reading of a value it did not mint.
        // `is_out` is cleared on the recorded copy: the semantic fact is "a readable
        // buffer holding this value", and the consumer's own `present` already forces
        // `is_out = false` at lookup, so the two statements cannot disagree.
        //
        // ⛔ BYTE-IDENTICAL FOR EVERY KERNEL THAT BAKES TODAY, BY THE WALK'S OWN REFUSAL:
        // a stored result with a compute consumer never lowered before (this exact
        // refusal fired), so the entry is dead weight unless a producer states the
        // computes-then-stores-then-reads form. The fp8 epilogue's absorbed mulf rides
        // the same insert safely — the contraction writes the SCALED result into the
        // store's region (its own out-region law), so the bytes a consumer would read
        // are the bytes the program states.
        if let Some(res) = op.result {
            let mut recorded = o;
            recorded.is_out = false;
            inter.insert(res, recorded);
        }

        let mut emitted = emit_one(
            f.name,
            program,
            &per_op,
            sym_id_base,
            layout,
            headmajor,
            WalkProof {
                b_orient,
                gather: op_gather.map(|(g, _, _)| g),
                bcast: &bcast_axes,
                df: op_df,
            },
        )?;
        out.append(&mut emitted);
        lowered += 1;
    }

    // ⭐ A PARAMETER'S ROWS SCATTERED — a `ktdp.store` of a `ktdp.load` through an indirect access
    // tile, in any program: the card's scatter copy ([`super::lower_ktir_to_superdsc::load_scatters`]).
    // The rows are a parameter's, so they move once the compute is done — which is program order
    // only while nothing the program reads is a destination they land in, so that is refused.
    let scattered = super::lower_ktir_to_superdsc::load_scatters(f.name, k, sym_id_base, layout)?;
    for sc in super::lower_ktir_to_superdsc::scatters_of(k)? {
        let read = gathers.iter().any(|g| g.value_tid == sc.value_tid)
            || f.ops_deep().iter().any(|o| {
                o.op_type == OpKind::KtdpLoad
                    && o.result.is_some_and(|v| {
                        region_for_operand(k, v)
                            .ok()
                            .flatten()
                            .is_some_and(|r| r.tid == sc.value_tid)
                    })
            });
        if read {
            return err(format!(
                "{}: the program scatters rows into t{} and also reads it — the scatter runs after \
                 the compute, so a read of its destination would see the rows before they land",
                f.name, sc.value_tid
            ));
        }
    }
    let scatters = !scattered.is_empty();
    out.extend(scattered);

    // ⭐ THE PURE-COPY ARM. A copy program states no compute op at all — its ops are loads and
    // stores, every one of them plumbing this walk skips — so `lowered == 0` is not a refusal for
    // it but the expected shape, and the one production copy is the LmLast last-row extraction
    // (`lmlast.py`: per-stick `[1, 64]` copies at a row corner, the ONE coordInfo form fp16 can
    // represent). The arm is keyed on the `Program` the node states, never on the op list: a copy
    // that grew a compute op is a NEW producer form, and lowering its compute while silently not
    // running the copy half would emit a program whose stores are never written — so that case
    // refuses here rather than half-lowering.
    if k.program == Program::LmLast {
        if lowered > 0 {
            return err(format!(
                "{}: an LmLast program with {lowered} compute op(s) — the last-row extraction is a \
                 PURE COPY (loads and stores only), so a compute op here means the producer's form \
                 changed and the copy half must be re-derived for it, not skipped",
                f.name
            ));
        }
        let r = regions(k)?;
        return super::lower_ktir_to_superdsc::lmlast(f.name, k, &r, sym_id_base, layout);
    }
    if lowered == 0 && !scatters {
        return err(format!(
            "{}: no compute op and no scatter in the function",
            f.name
        ));
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
    // ⭐⭐⭐ THE WALK'S BODY, MARKED — every descriptor the body's ops emitted is relaunched once per
    // trip ([`super::EmittedOp::walk`]), its gathers resolved through the index the trip's shift
    // points at (`kv_gathered`) and the run kept whole as one fold (`kv_page_fold`): a fold cut into
    // two groups would run every trip of the first before the second, reading the scratch the first
    // left behind.
    if let (Some(fold), Some(_)) = (fold, body_at) {
        let end = body_out.1.unwrap_or(out.len());
        let trips = super::loop_fold::WalkTrips {
            trips: fold.trips,
            step_bytes: fold.step_bytes,
        };
        for e in &mut out[body_out.0..end] {
            e.walk = Some(trips);
            e.kv_page_fold = true;
            e.kv_gathered = true;
        }
    }
    Ok(out)
}

/// THE FRAME A VALUE OF `dims` IS HELD IN: a per-row scalar of a reduction — a value shaped as a
/// `linalg.reduce` result of this function, the running max and sum — is the `[numel, 1]` column the
/// reduction writes; anything else is its flat frame.
fn value_frame(f: &IRFunction<'static>, dims: &[i64]) -> [u32; 2] {
    // The shapes a per-row column takes: a stick fold's result, a leading fold of a per-row
    // column (a decode batch's `J` block slots' running max, folded), and a shape op over one (the
    // folded max expanded back over the slots: one value per stick row in either shape) — to a
    // fixed point.
    let deep = f.ops_deep();
    let dims_of = |o: &Operation<'_>| o.result_type.and_then(|t| t.dims()).map(<[i64]>::to_vec);
    let mut columns: Vec<Vec<i64>> = deep
        .iter()
        .filter(|o| o.op_type == OpKind::LinalgReduce && folds_the_stick(o))
        .filter_map(|o| dims_of(o))
        .collect();
    loop {
        let more: Vec<Vec<i64>> = deep
            .iter()
            .filter(|o| {
                (o.op_type == OpKind::LinalgReduce && !folds_the_stick(o))
                    || matches!(
                        o.op_type,
                        OpKind::TensorReshape
                            | OpKind::TensorExpandShape
                            | OpKind::TensorCollapseShape
                    )
            })
            .filter(|o| {
                o.operands
                    .first()
                    .and_then(|&v| deep.iter().find(|d| d.result == Some(v)))
                    .and_then(|d| dims_of(d))
                    .is_some_and(|d| columns.contains(&d))
            })
            .filter_map(|o| dims_of(o))
            .filter(|d| !columns.contains(d))
            .collect();
        if more.is_empty() {
            break;
        }
        columns.extend(more);
    }
    let per_row = columns.iter().any(|d| d == dims);
    if per_row {
        [dims.iter().product::<i64>() as u32, 1]
    } else {
        flat_frame(dims)
    }
}

/// Whether a `linalg.reduce` folds its input's LAST axis — the stick, one value per row of the
/// flat frame, the per-row column a row reduction writes. A fold of any other axis (the leading
/// `J` of a decode batch's block slots) writes a tile of the input's own rows.
fn folds_the_stick(o: &Operation<'_>) -> bool {
    let rank = o.result_type.and_then(|t| t.dims()).map_or(0, <[i64]>::len);
    o.attributes.iter().any(
        |(k, v)| matches!((k, v), (AttrKey::Dimensions, Attr::IntList(d)) if *d == [rank as i64]),
    )
}

/// The values of the splatted constants the walk MATERIALIZES — a splat read as a dense operand of a
/// binary op ([`splat_buffer`]), the registry constants its fills read. A splat a multiply, `min` or
/// `max` reads beside a dense tensor is a scalar operand instead (`ScalarMul`, `ScalarBinary`), and a
/// contraction's or reduction's init is the op's own seed; neither is materialized.
pub fn splat_fill_values(f: &IRFunction<'static>) -> Result<Vec<f32>, Error> {
    let (bcasts, _) = program_broadcast_chains(f)?;
    let def_of = |s: Ssa| f.operations.iter().find(|o| o.result == Some(s));
    let splat_value = |s: Ssa| -> Option<f32> {
        let sp = def_of(s).filter(|o| o.op_type == OpKind::TensorSplat)?;
        let c = def_of(*sp.operands.first()?).filter(|o| o.op_type == OpKind::ArithConstant)?;
        c.attributes.iter().find_map(|(k, v)| match (k, v) {
            (AttrKey::Value, Attr::Float(x)) => Some(*x as f32),
            _ => None,
        })
    };
    let mut out: Vec<f32> = Vec::new();
    for op in f.operations.iter() {
        let scalar_side = matches!(
            op.op_type,
            OpKind::ArithMulf | OpKind::ArithMinnumf | OpKind::ArithMaxnumf
        );
        let binary = scalar_side
            || matches!(
                op.op_type,
                OpKind::ArithAddf | OpKind::ArithSubf | OpKind::ArithDivf
            );
        let [a, b] = op.operands[..] else { continue };
        if !binary {
            continue;
        }
        for (x, other) in [(a, b), (b, a)] {
            let Some(v) = splat_value(x) else { continue };
            let absorbed =
                scalar_side && splat_value(other).is_none() && !bcasts.contains_key(&other);
            if !absorbed && !out.iter().any(|y| y.to_bits() == v.to_bits()) {
                out.push(v);
            }
        }
    }
    Ok(out)
}

/// The splatted float constants a binary float op reads as OPERANDS — the values the walk's body
/// materializes before its descriptors begin. A contraction's `outs` and a reduction's init are not
/// among them: those are the op's own seeds, never read as data.
fn dense_splat_operands(f: &IRFunction<'static>, o: &Operation<'static>) -> Vec<Ssa> {
    if !matches!(
        o.op_type,
        OpKind::ArithAddf
            | OpKind::ArithSubf
            | OpKind::ArithMulf
            | OpKind::ArithDivf
            | OpKind::ArithMinnumf
            | OpKind::ArithMaxnumf
    ) {
        return Vec::new();
    }
    let def_of = |s: Ssa| f.operations.iter().find(|x| x.result == Some(s));
    o.operands
        .iter()
        .copied()
        .filter(|v| {
            def_of(*v).is_some_and(|sp| {
                sp.op_type == OpKind::TensorSplat
                    && sp
                        .operands
                        .first()
                        .and_then(|c| def_of(*c))
                        .is_some_and(|c| {
                            c.op_type == OpKind::ArithConstant
                                && c.attributes.iter().any(|(k, a)| {
                                    *k == AttrKey::Value && matches!(a, Attr::Float(_))
                                })
                        })
            })
        })
        .collect()
}

/// A splatted float constant's BUFFER — minted at its frame ([`value_frame`]) and declared to the
/// layout, with the value it holds — or `None` when `v` is not a `tensor.splat` of a float constant.
fn splat_buffer(
    k: &KtirNode,
    layout: Option<&BundleLayout>,
    floor: u32,
    v: Ssa,
) -> Result<Option<(Region, f32)>, Error> {
    let f = &k.func;
    let def_of = |s: Ssa| f.operations.iter().find(|o| o.result == Some(s));
    let Some(splat) = def_of(v).filter(|o| o.op_type == OpKind::TensorSplat) else {
        return Ok(None);
    };
    let value = splat
        .operands
        .first()
        .and_then(|c| def_of(*c))
        .filter(|o| o.op_type == OpKind::ArithConstant)
        .and_then(|o| {
            o.attributes.iter().find_map(|(k, a)| match (k, a) {
                (AttrKey::Value, Attr::Float(x)) => Some(*x as f32),
                _ => None,
            })
        });
    let (Some(value), Some(dims)) = (value, splat.result_type.and_then(|t| t.dims())) else {
        return Ok(None);
    };
    let Some(layout) = layout else {
        return err(format!(
            "{}: a splatted constant read as a dense operand needs a `BundleLayout` to hold its \
             buffer",
            f.name
        ));
    };
    let [rows, cols] = value_frame(f, dims);
    let tid = mint_inter(k, layout, floor, v);
    layout.synth(crate::place::PlaceId::Act(tid), &[rows, cols]);
    Ok(Some((
        Region {
            tid,
            v_rows: rows,
            v_cols: cols,
            r_start: 0,
            c_start: 0,
            r_len: rows,
            c_len: cols,
            r_cover: (0, rows),
            c_cover: (0, cols),
            is_out: true,
            is_fp8: false,
            band: None,
        },
        value,
    )))
}

/// ⭐ THE INT32 ZERO a chained index's run is copied against
/// ([`super::lower_ktir_to_superdsc::index_run_gather`], `addi32toi32(zero, run)` — the int32 copy dxp
/// binds): the decode step's host-staged zero stick (`PAGED_INT32_ZERO_TID`), [`INDEX_RUN_WORDS`] i32
/// zeros — staged by the host as every int32 the attention reads is, never made on the card. Refused
/// by name when the bundle does not place it, or places it short of a run.
///
/// [`INDEX_RUN_WORDS`]: super::lower_ktir_to_superdsc::INDEX_RUN_WORDS
fn int_zero_run(name: &str, layout: &BundleLayout) -> Result<u32, Error> {
    let tid = crate::reserved_tids::PAGED_INT32_ZERO_TID;
    let words = super::lower_ktir_to_superdsc::INDEX_RUN_WORDS;
    match layout.placements.get(&tid) {
        Some(p) if p.size >= u64::from(words) * 4 => Ok(tid),
        placed => err(format!(
            "{name}: a chained index copies its run against the host-staged int32 zero t{tid}, which \
             this bundle places in {} bytes — a run is {words} words",
            placed.map_or(0, |p| p.size)
        )),
    }
}

/// Whether `tid` holds batch blocks wider than one stick — bytes no flat frame names
/// ([`lower_function_folded`]'s `batch_blocks`).
fn wide_blocks(blocks: &std::collections::HashMap<u32, u32>, tid: u32) -> bool {
    blocks
        .get(&tid)
        .is_some_and(|&w| w > crate::work::FP16_ELEMS_PER_STICK)
}

/// A value's FLAT FRAME — every axis but the last folded into rows ([`shape_2d`]'s reading).
fn flat_frame(dims: &[i64]) -> [u32; 2] {
    match dims {
        [] => [1, 1],
        [c] => [1, *c as u32],
        [lead @ .., c] => [lead.iter().product::<i64>() as u32, *c as u32],
    }
}

/// ⭐ THE ROW-MAJOR SPLIT OF A STICK-MAJOR BLOCK — `(R, C)` when `stick_major`'s flat frame is
/// `[R, C·stick]`, `R ≥ 2` rows of `C ≥ 2` sticks, and `row_major` is the same card bytes as
/// `[R·C, stick]` (its own flat frame, or one row of `R·C` sticks, or a row-outermost page's one-stick
/// landing): the same elements, row `r`'s sticks in order, which on the card are different bytes
/// (`[C][R][stick]` vs `[R][C][stick]`) — the pair a stick permute moves between. One row of the
/// stick-major side is the same bytes in both frames, and no permute. (A head two sticks wide puts
/// granite-3.1-8b's `[kv, group, 128]` on the stick-major side.)
fn stick_major_split(stick_major: &[i64], row_major: &[i64]) -> Option<(u32, u32)> {
    let stick = crate::work::FP16_ELEMS_PER_STICK;
    let [rows, cols] = flat_frame(stick_major);
    let sticks = cols / stick;
    (cols.is_multiple_of(stick)
        && rows >= 2
        && sticks >= 2
        && same_card_bytes(row_major, &[i64::from(rows * sticks), i64::from(stick)]))
    .then_some((rows, sticks))
}

/// ⭐ STICK-MAJOR ROWS READ AS HEAD ROWS OF SEVERAL STICKS — a `[R, C·stick]` block (`R ≥ 2`) read
/// in a frame whose flat form is `[N, S·stick]`, `S ≥ 2`, `N` a multiple of `R` and `N·S = R·C`: row
/// `r`'s sticks split into `N / R` head rows of `S` sticks each. On the card that is two stick
/// permutes — the row-major split `[R·C, stick]`, then its `N` rows of `S` sticks back to stick-major
/// `[N, S·stick]` — since an element moves across all three of the head, the stick and the row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HeadRows {
    rows: u32,
    row_sticks: u32,
    head_rows: u32,
    sticks: u32,
}

impl HeadRows {
    fn of(stick_major: &[i64], heads: &[i64]) -> Option<Self> {
        let stick = crate::work::FP16_ELEMS_PER_STICK;
        let [rows, cols] = flat_frame(stick_major);
        let [head_rows, width] = flat_frame(heads);
        let (row_sticks, sticks) = (cols / stick, width / stick);
        (rows >= 2
            && cols.is_multiple_of(stick)
            && width.is_multiple_of(stick)
            && sticks >= 2
            && head_rows.is_multiple_of(rows)
            && head_rows * sticks == rows * row_sticks)
            .then_some(Self {
                rows,
                row_sticks,
                head_rows,
                sticks,
            })
    }
}

/// A minted row-major `[rows, stick]` buffer's region.
fn row_major_region(tid: u32, rows: u32, layout: &BundleLayout) -> Region {
    let stick = crate::work::FP16_ELEMS_PER_STICK;
    layout.synth(crate::place::PlaceId::Act(tid), &[rows, stick]);
    Region {
        tid,
        v_rows: rows,
        v_cols: stick,
        r_start: 0,
        c_start: 0,
        r_len: rows,
        c_len: stick,
        r_cover: (0, rows),
        c_cover: (0, stick),
        is_out: true,
        is_fp8: false,
        band: None,
    }
}

/// A minted stick-major `[rows, sticks · stick]` buffer's region.
fn stick_major_region(tid: u32, rows: u32, sticks: u32, layout: &BundleLayout) -> Region {
    let cols = sticks * crate::work::FP16_ELEMS_PER_STICK;
    layout.synth(crate::place::PlaceId::Act(tid), &[rows, cols]);
    Region {
        tid,
        v_rows: rows,
        v_cols: cols,
        r_start: 0,
        c_start: 0,
        r_len: rows,
        c_len: cols,
        r_cover: (0, rows),
        c_cover: (0, cols),
        is_out: true,
        is_fp8: false,
        band: None,
    }
}

/// What one stick permute moves: `rows` rows of `sticks` sticks — the whole buffers, or `Some(b)`:
/// block `b` of buffers holding `[rows, sticks · stick]` blocks one after another.
#[derive(Clone, Copy)]
struct Permuted {
    rows: u32,
    sticks: u32,
    block: Option<u32>,
}

impl Permuted {
    fn whole((rows, sticks): (u32, u32)) -> Self {
        Self {
            rows,
            sticks,
            block: None,
        }
    }
}

/// One stick permute between `src` and `dst` ([`crate::ir::bridge::tiled_op_sdsc_op::StickPermute`]).
fn permute(
    name: &str,
    src: &Region,
    dst: &Region,
    Permuted {
        rows,
        sticks,
        block,
    }: Permuted,
    direction: crate::ir::bridge::tiled_op_sdsc_op::StickPermute,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<super::EmittedOp, Error> {
    let base = block.map_or(0, |b| b * rows * sticks * crate::work::FP16_ELEMS_PER_STICK);
    let op_name = match block {
        None => format!("permute_o{}", dst.tid),
        Some(b) => format!("permute{b}_o{}", dst.tid),
    };
    crate::ir::bridge::tiled_op_sdsc_op::assemble_stick_permute(
        &op_name,
        &src.name(),
        &dst.name(),
        crate::ir::bridge::tiled_op_sdsc_op::StickPermuteSpec {
            rows,
            sticks,
            df: crate::superdsc_opspec::Df::Fp16,
            direction,
            src_elems: base,
            dst_elems: base,
        },
        sym_id_base,
        layout,
    )
    .map_err(|message| Error {
        message: format!("{name}: {message}"),
    })
}

/// Whether two shapes of one value are the SAME BYTES on the card: every element at the same address
/// in both stick-major flat frames (`(c / stick)·(rows·stick) + r·stick + c % stick`, a row padded to
/// whole sticks). True for equal frames, and for a re-framing that keeps every row one stick wide or
/// moves rows within one row of sticks; false — a relayout — otherwise.
fn same_card_bytes(from: &[i64], to: &[i64]) -> bool {
    if from.iter().product::<i64>() != to.iter().product::<i64>() {
        return false;
    }
    let (a, b) = (flat_frame(from), flat_frame(to));
    if a == b {
        return true;
    }
    let stick = crate::work::FP16_ELEMS_PER_STICK as u64;
    let addr = |[rows, cols]: [u32; 2], e: u64| -> u64 {
        let (r, c) = (e / cols as u64, e % cols as u64);
        (c / stick) * (rows as u64 * stick) + r * stick + c % stick
    };
    let numel: u64 = from.iter().product::<i64>() as u64;
    (0..numel).all(|e| addr(a, e) == addr(b, e))
}

/// A PAGE gather, read off its indirect tile: the tile's indirect dim enumerates `entries` index
/// words, and each selects one contiguous slab — the product of the tile's other extents —
/// `page_sticks` one-stick sub-rows long. [`gathers_of`] reads every gather as one entry per ROW
/// (the embedding's row gather); a page gather's tile states fewer entries than rows, and its own
/// shape says how many.
fn page_gather_of(f: &IRFunction<'static>, g: Gather) -> Result<(Gather, u32), Error> {
    let Some(tile) = f.operations.iter().find(|o| o.result == Some(g.tile)) else {
        return err(format!(
            "{}: the gather's tile states no defining op",
            f.name
        ));
    };
    let dims: Vec<i64> = tile
        .attributes
        .iter()
        .find_map(|(k, v)| match (k, v) {
            (AttrKey::Shape, Attr::IntList(d)) => Some(d.to_vec()),
            _ => None,
        })
        .or_else(|| tile.result_type.and_then(|t| t.dims()).map(<[i64]>::to_vec))
        .unwrap_or_default();
    let d = match tile.attr(AttrKey::DimKinds) {
        Some(Attr::StrList(kinds)) => kinds.iter().position(|k| *k == "indirect"),
        _ => None,
    };
    let (Some(d), false) = (d, dims.is_empty()) else {
        return err(format!(
            "{}: the gather's tile states no shape or no indirect dim",
            f.name
        ));
    };
    let entries = dims[d] as u32;
    let slab: i64 = dims
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != d)
        .map(|(_, x)| *x)
        .product();
    let stick = crate::work::FP16_ELEMS_PER_STICK as i64;
    if d != 0 || slab % stick != 0 {
        return err(format!(
            "{}: a page gather selects whole contiguous slabs along its LEADING dim, each a whole \
             number of sticks — this tile's indirect dim is {d} and its slab {slab} elements",
            f.name
        ));
    }
    Ok((Gather { entries, ..g }, (slab / stick) as u32))
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
    /// THE OP'S OWN RESULT FORMAT, read off the result type ([`result_df_of`]) — `Fp16` for every
    /// op this door lowered before the fp32 island, `Fp32` on the island's multiply-add. Read by
    /// the `Elementwise` arm, which states it on the tile and every `TensorArg`; unread by the
    /// arms whose formats ride other facts (the convert's ride its `OpFunc`).
    df: Df,
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
    headmajor: Option<super::lower_ktir_to_superdsc::HeadmajorOproj>,
    proof: WalkProof<'_>,
) -> Result<Vec<super::EmittedOp>, Error> {
    let WalkProof {
        b_orient,
        gather,
        bcast,
        df,
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
        // ⭐ THE OP'S OWN RESULT FORMAT RIDES IT ([`reduce_df`]): an fp16 reduce keeps the proven
        // seeded body verbatim; an fp32 one (the island's `max`/`sum` over an f32 intermediate)
        // rides the torch-spyre fp32-rmsnorm precedent's own df-carrying body, because the fp16
        // descriptor would mis-stick every read of the fp32 buffer.
        Lowering::Reduce(kind) => {
            return super::lower_ktir_to_superdsc::reduce_df(
                name,
                kind,
                per_op,
                df,
                sym_id_base,
                layout,
            );
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
        Lowering::RmsNorm(eps, ones) => {
            return super::lower_ktir_to_superdsc::rmsnorm_at(
                name,
                eps,
                ones,
                per_op,
                sym_id_base,
                layout,
            );
        }
        // A DTYPE-CONVERT — one input, one output, both the RowBlocked token stream, the formats
        // riding the `OpFunc` through `convert_dtypes`. The same `assemble_convert` body the fp8
        // quantiser's tail and the KSPLIT pair's own device recipe use, so the doors cannot
        // drift about what a convert emits.
        Lowering::Convert(kind) => {
            return super::lower_ktir_to_superdsc::convert(name, kind, per_op, sym_id_base, layout);
        }
        // A WRAPPING i32 BINARY — two inputs, one output, all SENUINT32: dense and corner-free
        // (B1's own measured form), plus the ONE-ROW `Col` broadcast operand mode through the
        // SAME body the fp doors reach (see [`int_binary`]'s doc — that admission is MEASURED
        // for BOTH kinds: the int-bcast add leg and the int-mul-bcast mul leg, nickm2,
        // bit-for-bit, 512/512 lanes each; `Y`, `Mb`, the
        // multi-row `Col` and every corner are refused there by name).
        Lowering::IntBinary(kind) => {
            return super::lower_ktir_to_superdsc::int_binary(
                name,
                kind,
                per_op,
                bcast,
                sym_id_base,
                layout,
            );
        }
        Lowering::ScalarBinary(e, value) => {
            return super::lower_ktir_to_superdsc::scalar_binary(
                name,
                e,
                value,
                per_op,
                sym_id_base,
                layout,
            );
        }
        Lowering::BatchMatmul { batch, activation } => {
            return super::lower_ktir_to_superdsc::batch_matmul(
                name,
                batch,
                activation,
                per_op,
                sym_id_base,
                layout,
            );
        }
        Lowering::BatchTranspose { batch, input } => {
            return super::lower_ktir_to_superdsc::batch_transpose(
                name,
                batch,
                input,
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
                super::lower_ktir_to_superdsc::MatmulProof {
                    b,
                    // This door's operands are WINDOWS of the caller's parameters — the fact the
                    // spurious-pad drop in `matmul_oriented` discriminates on.
                    origin: super::lower_ktir_to_superdsc::OperandOrigin::Windowed,
                    // ⭐ THE HEAD-MAJOR o_proj HANDOFF, THREADED FROM THE DOOR THAT OWNS THE BUNDLE
                    // FACT. It is minted from the attention-output row count (mq == 1 over a
                    // multi-slab head dim) and the o matmul's weight tid, neither of which a
                    // program states; `matmul_oriented` keys the swap on `w.tid == oproj_wtid`, so
                    // of the many matmuls an unrolled graph holds, exactly the one the fact names
                    // takes it. `None` — every caller without a bundle fact — is the plain staged
                    // weight, byte-identical to what this arm always emitted.
                    headmajor,
                },
            )?
        }
        Program::Elementwise(e) => super::lower_ktir_to_superdsc::elementwise_df(
            name,
            e,
            per_op,
            bcast,
            df,
            sym_id_base,
            layout,
        )?,
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
/// The emission's mutable scratch (`sym_id_base`, `out`, `inter`) travels in
/// [`GatherEmit`] rather than as three more arguments — the walk owns one of these, threading it
/// to every materialization. `floor` is [`lower_function`]'s node-derived tid bound, carried so
/// this mint honors it like the walk's own does.
struct GatherEmit<'a> {
    sym_id_base: &'a mut i64,
    out: &'a mut Vec<super::EmittedOp>,
    floor: u32,
    inter: &'a mut std::collections::HashMap<Ssa, Region>,
}

fn gathered_matmul_materializes(
    k: &KtirNode,
    layout: Option<&BundleLayout>,
    emit: &mut GatherEmit<'_>,
    g: Gather,
    gathered_input: usize,
    in_values: &[Ssa],
) -> Result<bool, Error> {
    let GatherEmit {
        sym_id_base,
        out,
        floor,
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
    // ⭐⭐⭐ THIS TILE, BY IDENTITY — `g.tile` is the tile's own result SSA, read once by the same
    // walk (`gathers_of`) that built the join, so the walk and this materializer cannot disagree
    // about which trip's gather this is (issue 201 item 3: the previous re-filter matched by
    // enumeration index, which only agrees while every walk filters identically).
    let Some(indirect) = f.operations.iter().find(|o| o.result == Some(g.tile)) else {
        return err(format!(
            "{}: the walk proved input {gathered_input} reads a gather whose tile is t{}, but no \
             operation in this function states that result — the walk and the program disagree",
            f.name, g.tile.0,
        ));
    };
    // ⛔ AND THE MATMUL'S GATHERED INPUT IS THE GATHERED LOAD — the identity check the tid check
    // cannot make. `g.load` is the `ktdp.load` over this tile, found by the walk.
    if b_value != g.load {
        return Ok(false);
    }
    let gathered_result = g.load;
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
    let tid = mint_inter(k, layout, *floor, gathered_result);
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
        c_cover: (0, head),
        is_out: true,
        is_fp8: false,
        band: None,
    };
    // ── (3) THE GATHERED IDENTITY COPY — the `test_gather_1core` shape, through the one body
    // that declares an index operand. One input (the table), `identity`, the intermediate as
    // output; `assemble_pointwise_broadcast_gather` owns the one-index-stick ceiling and the
    // per-stick legs, so nothing here counts entries.
    let table_name = crate::place::act_name(g.value_tid);
    let index_name = crate::place::act_name(g.index_tid);
    let inter_name = r.name();
    // ⭐ TRUE SHAPES, not `rbo`'s `1×64` annotation: the table and the intermediate declare the
    // swept `[rows, head]` — the scalarmul arm's convention (the index selects rows within the
    // table, so the read window is the bound; the placement door still holds the window against
    // the table's real footprint). The intermediate's `synth` above states the same pair. The
    // builder reads name + kind only, so the dims are the mint's bound and nothing else.
    let x_h = super::rb(&table_name, rows, head);
    let inputs = [super::In::full(&x_h).ew()];
    let tile_op = super::lower_ktir_to_superdsc::pointwise_tile_op(rows, head, 1, 2);
    // ⭐ THE RUN — the program's stick-checked anchor plus the whole sticks these rows span
    // ([`StickRun::covering`]). The constructor's start half re-states `gathers_of`'s anchor
    // check at the door that builds the legs, so a mid-stick anchor has no path around it.
    let run = crate::superdsc_opspec::StickRun::covering(g.first_entry.entries(), rows).map_err(
        |msg| Error {
            message: format!("{}: {msg}", f.name),
        },
    )?;
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
            run,
            o: &super::rb(&inter_name, rows, head),
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
    /// broadcast straight to the source so the degenerate-expand requirement is what fires. `res` is
    /// the broadcast result's own `Shape` — which every real frontend op states and the rank probe
    /// reads to decide the frame the axis is named in.
    fn chain(
        target: &'static [i64],
        bdims: Option<&'static [i64]>,
        via_expand: bool,
        res: &'static [i64],
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
        let mut b = Operation::new(a, Some(bc), OpKind::LinalgBroadcast, &[source, emp]).with_attr(
            a,
            AttrKey::Shape,
            Attr::IntList(res),
        );
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

    /// WHAT THE BATCHED KERNEL EMITS for its rank-3 broadcasts, from the op dump of
    /// `batched_decode.py` — the same `expand -> collapse -> empty -> broadcast` spine with a
    /// STATED source, or the `reshape -> collapse` spine the mask and the J-slot merge spell:
    ///
    /// ```text
    /// %83 = MathExp(...)                        Shape=[256, 4]        (the running max)
    /// %85 = TensorExpandShape(%83)   TargetShape=[256, 4, 1]
    /// %74 = TensorCollapseShape(%85) TargetShape=[256, 4]
    /// %75 = TensorEmpty()                       Shape=[256, 4, 256]
    /// %87 = LinalgBroadcast(%74, %75)           Shape=[256, 4, 256] Dimensions=[2]
    /// ```
    ///
    /// `src_shape` states the source op's own `Shape` (the running max `[256, 4]`, the mask
    /// `[256, 256]`, the merged max `[64, 4]`, a one-row vector `[1, 64]`), `relay` is the op
    /// between the source and the collapse — an expand (the sprays) or a reshape (the mask and
    /// the merge, which the frontend spells without any expand) — and `collapse`/`res`/`bdims`
    /// are the collapse target, the broadcast's own `Shape`, and its `Dimensions`.
    /// The op between the source and the collapse: the EXPAND the sprays spell (its own
    /// `Shape` is its `TargetShape`), or the RESHAPE the mask and the J-slot merge spell.
    enum Relay {
        Expand(&'static [i64]),
        Reshape(&'static [i64]),
    }

    fn chain3(
        src_shape: &'static [i64],
        relay: Relay,
        collapse: &'static [i64],
        res: &'static [i64],
        bdims: &'static [i64],
    ) -> IRFunction<'static> {
        let a: &'static Arena = Arena::global();
        let (src, mid, col, emp, bc) = (Ssa(0), Ssa(1), Ssa(2), Ssa(3), Ssa(4));
        let sh = |op: Operation<'static>, s: &'static [i64]| {
            op.with_attr(a, AttrKey::Shape, Attr::IntList(s))
        };
        let mut ops = vec![sh(
            Operation::new(a, Some(src), OpKind::MathExp, &[]),
            src_shape,
        )];
        let (kind, target) = match relay {
            Relay::Expand(t) => (OpKind::TensorExpandShape, t),
            Relay::Reshape(t) => (OpKind::TensorReshape, t),
        };
        ops.push(
            sh(Operation::new(a, Some(mid), kind, &[src]), target).with_attr(
                a,
                AttrKey::TargetShape,
                Attr::IntList(target),
            ),
        );
        ops.push(sh(
            Operation::new(a, Some(col), OpKind::TensorCollapseShape, &[mid]),
            collapse,
        ));
        ops.push(sh(
            Operation::new(a, Some(emp), OpKind::TensorEmpty, &[]),
            res,
        ));
        ops.push(
            sh(
                Operation::new(a, Some(bc), OpKind::LinalgBroadcast, &[col, emp]),
                res,
            )
            .with_attr(a, AttrKey::Dimensions, Attr::IntList(bdims)),
        );
        IRFunction {
            name: "bcast3_probe",
            arguments: a.args(vec![]),
            operations: a.ops(ops),
            grid: (1, 1, 1),
            return_type: None,
        }
    }

    fn refusal(
        target: &'static [i64],
        bdims: Option<&'static [i64]>,
        via: bool,
        res: &'static [i64],
    ) -> String {
        program_broadcast_chains(&chain(target, bdims, via, res))
            .err()
            .map(|e| e.message)
            .unwrap_or_else(|| panic!("expected a refusal for target={target:?} bdims={bdims:?}"))
    }

    fn refusal3(
        src_shape: &'static [i64],
        relay: Relay,
        collapse: &'static [i64],
        res: &'static [i64],
        bdims: &'static [i64],
    ) -> String {
        program_broadcast_chains(&chain3(src_shape, relay, collapse, res, bdims))
            .err()
            .map(|e| e.message)
            .unwrap_or_else(|| {
                panic!("expected a rank-3 refusal for src={src_shape:?} bdims={bdims:?}")
            })
    }

    /// ⭐ CONTROL (Col) — THE DECODER'S `m[:, None]`. `[64] → [64, 1] → [64, 64]`, the shape
    /// `decoder_block.py`'s softmax states twice per layer. Without this the refusals below could all be
    /// a broken walk rather than working guards.
    #[test]
    fn the_col_chain_is_recognised_and_resolves_to_its_source() {
        let f = chain(&[64, 1], Some(&[1]), true, &[64, 64]);
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
        let f = chain(&[1, 128], Some(&[0]), true, &[64, 128]);
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
        let m = refusal(&[1, 128], Some(&[1]), true, &[1, 64]);
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
        let m = refusal(&[64, 128], Some(&[1]), true, &[64, 64]);
        assert!(
            m.contains("A general reshape moves real elements"),
            "says WHY a general reshape is different in kind, not just that it was rejected: {m}"
        );
    }

    /// ⛔ AN AXIS THAT IS NEITHER OF THE TWO OPERAND MODES. A two-axis spray from a scalar is
    /// `In::scalar`, a third mode, and this door does not silently fold it into one of the two it emits.
    #[test]
    fn a_two_axis_broadcast_is_refused_by_name() {
        let m = refusal(&[64, 1], Some(&[0, 1]), true, &[64, 64]);
        assert!(
            m.contains("In::col") && m.contains("In::mb"),
            "names both modes it does emit: {m}"
        );
    }

    /// ⛔ NO `Dimensions` AT ALL — then which axis it sprays is simply unknown, and picking one would be
    /// choosing an operand mode for the producer.
    #[test]
    fn a_broadcast_with_no_dimensions_attribute_is_refused() {
        let m = refusal(&[64, 1], None, true, &[64, 64]);
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
        let m = refusal(&[64, 1], Some(&[1]), false, &[64, 64]);
        assert!(
            m.contains("DEGENERATE dim"),
            "says what is missing and why it is load-bearing: {m}"
        );
    }

    /// ⭐⭐⭐ THE RANK-3 TRAILING SPRAY — THE BATCHED KERNEL'S OWN `m[:, :, None]`, THE FLAT `Col`.
    /// `[256, 4] -> [256, 4, 1] -> collapse -> [256, 4, 256]` with `Dimensions=[2]`: the running
    /// max's `[256, 4]` window IS the `[1024, 1]` column vector of the flat frame (element
    /// `(b, g)` is flat row `b·4 + g`'s scalar — same bytes, same offsets), so the chain is the
    /// decoder's own Col mode with `flat` set, reading the SOURCE and not the plumbing. This is
    /// the control for every rank-3 refusal below: the accepting arm is proven on the real
    /// shape, not assumed.
    #[test]
    fn the_rank3_trailing_chain_is_flat_col() {
        let f = chain3(
            &[256, 4],
            Relay::Expand(&[256, 4, 1]),
            &[256, 4],
            &[256, 4, 256],
            &[2],
        );
        let (map, consumed) = program_broadcast_chains(&f).expect("the running max's own chain");
        let b = map
            .get(&Ssa(4))
            .expect("the broadcast's result is what a consumer names");
        assert_eq!(
            b.axis,
            BcastAxis::Col,
            "`Dimensions=[2]` is the trailing axis"
        );
        assert!(
            b.flat,
            "the rank-3 form is the FLAT Col — the consumer presents the source `[numel, 1]`"
        );
        assert_eq!(
            b.src,
            Ssa(0),
            "the consumer must read the rank-2 SOURCE (the running max), not the plumbing"
        );
        for v in [Ssa(1), Ssa(2), Ssa(4)] {
            assert!(
                consumed.contains(&v),
                "{v:?} is plumbing and must be lowered by nothing"
            );
        }
    }

    /// ⭐ THE FLAT `Mb` ADMITTED FORM — a one-flat-row source sprayed down a leading axis, the
    /// `[1, 64] -> [8, 1, 64]` vector-on-rows the rank-2 `Mb` always was, stated in the flat
    /// frame. This is the control that the leading-axis law below is a ONE-ROW law and not a
    /// blanket refusal of `Dimensions=[0]`.
    #[test]
    fn the_rank3_leading_one_row_source_is_flat_mb() {
        let f = chain3(
            &[1, 64],
            Relay::Expand(&[1, 1, 64]),
            &[1, 64],
            &[8, 1, 64],
            &[0],
        );
        let (map, _) = program_broadcast_chains(&f).expect("the one-row vector's own chain");
        let b = map
            .get(&Ssa(4))
            .expect("the broadcast's result is what a consumer names");
        assert_eq!(
            b.axis,
            BcastAxis::Mb,
            "`Dimensions=[0]` is the leading axis"
        );
        assert!(b.flat, "the rank-3 form is the FLAT Mb");
        assert_eq!(b.src, Ssa(0), "the consumer reads the rank-2 SOURCE");
    }

    /// ⭐⭐⭐ THE AXIS JUST BEFORE THE STICK — THE MASK ADD, `[256, 256] -> [256, 1, 256] ->
    /// [256, 4, 256]` with `Dimensions=[1]`, spelled `reshape -> collapse -> broadcast` with no
    /// expand at all — ADMITTED as the INNER row spray: `result[b, g, j] = source[b, j]`, flat row
    /// `b·4 + g` reading flat row `b`, which the `[mb, y, out]` row-split frame addresses byte for
    /// byte (each of the source's 256 rows repeated 4 times in a row). The cover is proved on the
    /// source's own shape: its 256 flat rows are the result's rows before the sprayed axis.
    #[test]
    fn the_axis_before_the_stick_over_a_multirow_source_is_the_inner_row_spray() {
        let f = chain3(
            &[256, 256],
            Relay::Reshape(&[256, 1, 256]),
            &[256, 256],
            &[256, 4, 256],
            &[1],
        );
        let (map, _) = program_broadcast_chains(&f).expect("the mask add's own chain");
        let b = map
            .get(&Ssa(4))
            .expect("the broadcast's result is what a consumer names");
        assert_eq!(
            b.axis,
            BcastAxis::Y(super::super::lower_ktir_to_superdsc::Spray::inner(4)),
            "each source row repeated 4 times in a row is the inner row spray"
        );
        assert!(!b.flat, "the spray's source is the `[mb, out]` tile itself");
        assert_eq!(b.src, Ssa(0), "the walk-back reaches the rank-2 SOURCE");
    }

    /// ⛔ AN INNER SPRAY WHOSE SOURCE IS NOT THE RESULT'S LEADING ROWS — a 128-row source under a
    /// result whose rows before the sprayed axis are 256: the copies would read rows the source
    /// does not have, so the cover proof refuses it by name.
    #[test]
    fn an_inner_spray_over_the_wrong_rows_is_refused() {
        let m = refusal3(
            &[128, 256],
            Relay::Reshape(&[128, 1, 256]),
            &[128, 256],
            &[256, 4, 256],
            &[1],
        );
        assert!(
            m.contains("second-to-last axis") && m.contains("rows the source does not have"),
            "names the cover the spray needs: {m}"
        );
    }

    /// ⛔⛔ A MIDDLE AXIS WITH MORE THAN THE STICK AFTER IT — `[64, 8, 64] -> [64, 4, 8, 64]` over
    /// `Dimensions=[1]`: each source row's copies are interleaved with the axes after the sprayed
    /// one, a replication at a stride no operand mode addresses, so it is refused by name.
    #[test]
    fn a_middle_axis_with_axes_after_it_is_row_replication_and_refused() {
        let m = refusal3(
            &[512, 64],
            Relay::Reshape(&[64, 1, 8, 64]),
            &[512, 64],
            &[64, 4, 8, 64],
            &[1],
        );
        assert!(
            m.contains("MIDDLE axis"),
            "names the axis class, not just the index: {m}"
        );
        assert!(
            m.contains("ROW-REPLICATION AT A STRIDE"),
            "names the data movement the device cannot address: {m}"
        );
    }

    /// ⭐⭐⭐ THE RANK-3 LEADING AXIS OVER A MULTI-ROW SOURCE — THE J-SLOT MERGE, `[64, 4] ->
    /// [1, 64, 4] -> collapse -> [4, 64, 4]` with `Dimensions=[0]`, the exact spelling the op
    /// dump shows — ADMITTED as the ROW-SPLIT (`Y`) operand mode. The source is the `[mb, out]`
    /// tile (sixty-four single-stick rows) and the broadcast sprays it down a NEW leading dim,
    /// J sprays deep: `result[j, b, c] = source[b, c]`, which the `y`-broadcast scale
    /// (`RedNonStick` on the new dim, the vendor's reduced-dim law) addresses byte for byte.
    /// The walk-back crosses the numel-preserving `tt.reshape` (a reshape is a flat no-op) to
    /// name the rank-2 SOURCE, and the frame law (`RegionFrame::split_rows`) re-proves the
    /// divisibility at the emission before any operand is built.
    #[test]
    fn the_rank3_leading_multirow_source_is_the_row_split_y_mode() {
        let f = chain3(
            &[64, 4],
            Relay::Reshape(&[1, 64, 4]),
            &[64, 4],
            &[4, 64, 4],
            &[0],
        );
        let (map, consumed) = program_broadcast_chains(&f).expect("the J-slot merge's own chain");
        let b = map
            .get(&Ssa(4))
            .expect("the broadcast's result is what a consumer names");
        assert_eq!(
            b.axis,
            BcastAxis::Y(super::super::lower_ktir_to_superdsc::Spray::outer(4)),
            "a multi-row source sprayed down a NEW leading dim is the row-split form"
        );
        assert!(
            !b.flat,
            "the row-split source is the `[mb, out]` TILE at its own extents, not a flat \
             presentation"
        );
        assert_eq!(
            b.src,
            Ssa(0),
            "the walk-back crossed the numel-preserving reshape to the rank-2 SOURCE (merged)"
        );
        for v in [Ssa(1), Ssa(2), Ssa(4)] {
            assert!(
                consumed.contains(&v),
                "{v:?} is plumbing and must be lowered by nothing"
            );
        }
    }

    /// ⛔ THE SPRAY OF ONE — the same J-slot merge spine with the result's leading dim already
    /// collapsed (`[1, 64, 4]`): the dim is already in the source, so there is no broadcast to
    /// address and no split to prove. Refused by name rather than admitted as a no-op the
    /// consumer would still try to present.
    #[test]
    fn a_rank3_leading_spray_of_one_is_refused() {
        let m = refusal3(
            &[64, 4],
            Relay::Reshape(&[1, 64, 4]),
            &[64, 4],
            &[1, 64, 4],
            &[0],
        );
        assert!(
            m.contains("a spray of one is no broadcast"),
            "names the degenerate spray: {m}"
        );
    }

    /// ⛔ THE MISALIGNED MERGE — a `[4, 32, 4]` result over the same 64-row source: the merged
    /// axis does not line up with the source's rows, so no `rows == y*mb` split exists and the
    /// shape really is row-replication. The refusal keeps naming the J-slot merge as the
    /// admitted form of the shape so it reads as a work item, not a wall.
    #[test]
    fn a_rank3_leading_spray_that_misaligns_the_rows_is_refused() {
        let m = refusal3(
            &[64, 4],
            Relay::Reshape(&[1, 64, 4]),
            &[64, 4],
            &[4, 32, 4],
            &[0],
        );
        assert!(
            m.contains("ROW-REPLICATION"),
            "names the data movement the device cannot address: {m}"
        );
        assert!(
            m.contains("J-slot merge"),
            "names the kernel site whose shape IS admitted, so the refusal reads as a work \
             item: {m}"
        );
    }

    /// ⛔ THE ELEMENT-COUNT-CHANGING RESHAPE — the merge spine with a reshape that DOUBLES the
    /// element count (`[64, 4]` → `[1, 128, 4]`): a numel change is a real data movement, not
    /// the flat no-op the walk traverses, so the broadcast's source would be a relayout this
    /// door cannot express. Refused by name at the reshape itself.
    #[test]
    fn a_reshape_that_changes_the_element_count_is_refused() {
        let m = refusal3(
            &[64, 4],
            Relay::Reshape(&[1, 128, 4]),
            &[128, 4],
            &[4, 128, 4],
            &[0],
        );
        assert!(
            m.contains("a reshape that changes the element count is a real data movement"),
            "names the relayout at the reshape itself: {m}"
        );
    }

    /// WHAT THE SAMPLER'S GATE CHAIN LOOKS LIKE, from the post-fold KTIR of `sample_fwd.py`: the
    /// kernel loads the per-row gate as a `[1, 1]` i32 window of the seed table (the hash's own
    /// law makes that table i32), Triton's implicit promotion converts it to f32
    /// (`sgate.to(tl.float32)`), the collapse drops the degenerate dim, and the broadcast sprays
    /// the per-row scalar along the stick axis:
    ///
    /// ```text
    /// %sgate_2 = KtdpLoad(...)                        Shape=[1, 1]
    /// %gate = arith.sitofp(%sgate_2)                  Shape=[1, 1]
    /// %score_5 = TensorCollapseShape(%gate)           TargetShape=[1]
    /// %score_6 = TensorEmpty()                        Shape=[1, 64]
    /// %score_7 = LinalgBroadcast(%score_5, %score_6)  Shape=[1, 64] Dimensions=[1]
    /// ```
    ///
    /// `load_shape`/`conv_shape` state the two sides of the traversal's proof independently, so a
    /// case can break the equality alone; `arg_source: true` replaces the load with a function
    /// argument, the case whose operand side is stated NOWHERE.
    fn chain_conv(
        load_shape: &'static [i64],
        conv_shape: &'static [i64],
        arg_source: bool,
        res: &'static [i64],
    ) -> IRFunction<'static> {
        let a: &'static Arena = Arena::global();
        let (src, conv, col, emp, bc) = (Ssa(0), Ssa(1), Ssa(2), Ssa(3), Ssa(4));
        let sh = |op: Operation<'static>, s: &'static [i64]| {
            op.with_attr(a, AttrKey::Shape, Attr::IntList(s))
        };
        let mut ops = vec![];
        let mut args = vec![];
        if arg_source {
            args.push((src, IrType::Index));
        } else {
            ops.push(sh(
                Operation::new(a, Some(src), OpKind::KtdpLoad, &[]),
                load_shape,
            ));
        }
        ops.push(sh(
            Operation::new(a, Some(conv), OpKind::ArithSitofp, &[src]),
            conv_shape,
        ));
        ops.push(
            sh(
                Operation::new(a, Some(col), OpKind::TensorCollapseShape, &[conv]),
                &[1],
            )
            .with_attr(a, AttrKey::TargetShape, Attr::IntList(&[1])),
        );
        ops.push(sh(
            Operation::new(a, Some(emp), OpKind::TensorEmpty, &[]),
            res,
        ));
        ops.push(
            sh(
                Operation::new(a, Some(bc), OpKind::LinalgBroadcast, &[col, emp]),
                res,
            )
            .with_attr(a, AttrKey::Dimensions, Attr::IntList(&[1])),
        );
        IRFunction {
            name: "bcast_conv_probe",
            arguments: a.args(args),
            operations: a.ops(ops),
            grid: (1, 1, 1),
            return_type: None,
        }
    }

    fn refusal_conv(
        load_shape: &'static [i64],
        conv_shape: &'static [i64],
        arg_source: bool,
        res: &'static [i64],
    ) -> String {
        program_broadcast_chains(&chain_conv(load_shape, conv_shape, arg_source, res))
            .err()
            .map(|e| e.message)
            .unwrap_or_else(|| {
                panic!(
                    "expected a convert-chain refusal for load={load_shape:?} conv={conv_shape:?}"
                )
            })
    }

    /// ⭐⭐⭐ THE SAMPLER'S GATE CHAIN — load `[1, 1]` i32 → `sitofp` `[1, 1]` f32 → collapse →
    /// broadcast `Dimensions=[1]`. The walk crosses the SHAPE-PRESERVING convert (the IR's own
    /// `[1, 1] == [1, 1]` is the proof) and the LOAD's own window supplies the vector-ness: extent
    /// 1 on the sprayed axis, the second proof.
    ///
    /// ⭐⭐⭐⭐⭐ THE PIN REVERSED (front-b-noise, round 3, measured on the card): the consumer
    /// binds the CONVERT'S RESULT — the minted `[B, 1]` intermediate the convert emits into —
    /// never the load. The earlier pin (`b.src ==` the load, the convert "plumbing and lowered
    /// by nothing") was WRONG: a convert changes the value's ENCODING, and binding the load made
    /// the card read the per-row gate's raw SENUINT32 placement as IEEE_FP32 — the staged u32
    /// word 1 as the denormal 2^-149, an effective gate of 2^-127 — so score lost its noise
    /// term and the pick collapsed to argmax at every temperature (greedy accidentally
    /// correct). The convert now EMITS through the main walk's `Lowering::Convert` arm; only
    /// the collapse and the broadcast remain plumbing.
    #[test]
    fn the_gate_chain_crosses_its_sitofp_and_binds_the_converts_result() {
        let f = chain_conv(&[1, 1], &[1, 1], false, &[1, 64]);
        let (map, consumed) = program_broadcast_chains(&f).expect("the gate's own chain");
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
            Ssa(1),
            "the consumer binds the CONVERT'S result — the minted intermediate it emits into"
        );
        for v in [Ssa(2), Ssa(4)] {
            assert!(
                consumed.contains(&v),
                "{v:?} is plumbing and must be lowered by nothing"
            );
        }
        assert!(
            !consumed.contains(&Ssa(1)),
            "the sitofp EMITS (`Lowering::Convert`) — a convert is never plumbing"
        );
    }

    /// ⛔ A CONVERT WHOSE TWO STATED SHAPES DISAGREE — load `[1, 1]`, convert `[2, 1]`: an
    /// elementwise convert is one-for-one on elements, so a dims change is a real data movement
    /// and the traversal's proof fails at the convert itself. Named there, not mis-addressed.
    #[test]
    fn a_convert_whose_stated_shapes_disagree_is_refused() {
        let m = refusal_conv(&[1, 1], &[2, 1], false, &[1, 64]);
        assert!(
            m.contains("one-for-one on elements"),
            "says WHY a dims change is different in kind, not just that it was rejected: {m}"
        );
        assert!(
            m.contains("real data movement"),
            "names the relayout at the convert itself: {m}"
        );
    }

    /// ⛔ AN OPERAND SIDE STATED NOWHERE — the convert reads a function argument (no producer, no
    /// `Shape`), so the equality has no second side and the walk stops AT the convert's result:
    /// the load-window proof below then honestly fails, because the value it would read is the
    /// convert's own result. This is the control that the traversal is PROVEN per hop and not
    /// granted to every convert the walk meets.
    #[test]
    fn a_convert_of_an_unstated_operand_stops_the_walk() {
        let m = refusal_conv(&[1, 1], &[1, 1], true, &[1, 64]);
        assert!(
            m.contains("DEGENERATE dim"),
            "the unproven-extent refusal fires with the walk stopped at the convert: {m}"
        );
    }
}

/// ⭐ THE FLAT PRESENTATION IS CHECKED AT THE REFRAME — [`presented_flat`] runs the source's own
/// device frame through `RegionFrame`'s re-row law, so a presentation the bytes do not support
/// is an `Err` at the presentation, never a clean bake of wrong bytes. The multi-column
/// `[256, 4]` mint presented as the `[1024, 1]` column vector is the defect class the law
/// exists for: the flat frame reads element `f` at `f·64`, but the mint holds `(r, c)` at
/// `r·64 + c`.
#[cfg(test)]
mod presented_flat_tests {
    use super::*;

    fn reg(rows: u32, cols: u32) -> Region {
        Region {
            tid: 0,
            v_rows: rows,
            v_cols: cols,
            r_start: 0,
            c_start: 0,
            r_len: rows,
            c_len: cols,
            r_cover: (0, rows),
            c_cover: (0, cols),
            is_out: false,
            is_fp8: false,
            band: None,
        }
    }

    /// The real kernel's own flat `Col` source — the rank-1-from-reduce mint `[1024, 1]` — is
    /// the IDENTITY: the flat frame is the mint's own frame, so the presentation changes
    /// nothing and the chain passes byte-identically.
    #[test]
    fn the_rank1_mint_presents_flat_col_as_the_identity() {
        let p = presented_flat(&reg(1024, 1), BcastAxis::Col).expect("the identity");
        assert_eq!((p.v_rows, p.v_cols), (1024, 1));
    }

    /// ⛔ THE MULTI-COLUMN MINT CANNOT BE PRESENTED `[numel, 1]` — the flat frame would read
    /// element `f` at `f·64` while the mint holds `(r, c)` at `r·64 + c`. Refused at the
    /// reframe, naming the re-row law that failed.
    #[test]
    fn a_multi_column_mint_is_refused_the_flat_col_presentation() {
        let Err(m) = presented_flat(&reg(256, 4), BcastAxis::Col) else {
            panic!("a [256, 4] mint presented [1024, 1] reads bytes nobody wrote");
        };
        assert!(
            m.contains("stride BETWEEN stick columns"),
            "names the re-row law: {m}"
        );
    }

    /// A single-stick-wide source CAN be presented `[1, numel]` — its rows are contiguous, so
    /// the flat row reads the same bytes the mint holds.
    #[test]
    fn a_single_stick_source_presents_flat_mb() {
        let p =
            presented_flat(&reg(256, 4), BcastAxis::Mb).expect("cols <= one stick is contiguous");
        assert_eq!((p.v_rows, p.v_cols), (1, 1024));
    }

    /// ⛔ A MULTI-STICK SOURCE CANNOT — presenting `[1, numel]` interleaves the stick columns
    /// (the flat row strides one stick while the mint's stick columns sit a whole row-block
    /// apart). Refused, naming the order.
    #[test]
    fn a_multi_stick_source_is_refused_the_flat_mb_presentation() {
        let Err(m) = presented_flat(&reg(256, 128), BcastAxis::Mb) else {
            panic!("a [256, 128] mint presented [1, 32768] interleaves its stick columns");
        };
        assert!(m.contains("WRONG ORDER"), "names the order: {m}");
    }

    /// The row-split mode presents NOTHING — the source keeps its own `[mb, out]` extents and
    /// the `y`-broadcast scale owns the addressing, so the presentation is the identity.
    #[test]
    fn the_row_split_mode_presents_nothing() {
        let p = presented_flat(
            &reg(64, 4),
            BcastAxis::Y(super::super::lower_ktir_to_superdsc::Spray::outer(4)),
        )
        .expect("the identity");
        assert_eq!((p.v_rows, p.v_cols), (64, 4));
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
    /// and the contraction reduces over k in place. scratchy's matmul producer emits this
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

    /// The BARE silu's builder — the same five ops with the divide STORED rather than multiplied,
    /// `elementwise.py`'s `silu_fwd` form. Only the VALUE slot of the store matters to the reader
    /// (the reader matches the op kind and the read value, never the tile), so a one-operand
    /// `KtdpStore` states the form.
    fn stored_chain() -> IRFunction<'static> {
        let a: &'static Arena = Arena::global();
        let (g, c1, sp) = (Ssa(0), Ssa(1), Ssa(2));
        let (neg, exp, add, div) = (Ssa(3), Ssa(4), Ssa(5), Ssa(6));
        let ops = vec![
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
            Operation::new(a, None, OpKind::KtdpStore, &[div]),
        ];
        IRFunction {
            name: "silu_stored_probe",
            arguments: a.args(vec![(g, IrType::Index)]),
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
        let c = &cs[0];
        assert_eq!(c.gate, Ssa(0), "the gate is the value the negate consumed");
        assert_eq!(c.up, Some(Ssa(1)), "`up` is the multiply's other operand");
        assert_eq!(
            c.mul,
            Ssa(8),
            "the terminal multiply's result is what the program produces"
        );
        assert_eq!(
            c.consumed,
            vec![Ssa(4), Ssa(5), Ssa(6), Ssa(7)],
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

    /// ⭐ THE SECOND TERMINAL FORM: the STORED BARE SILU. `elementwise.py`'s `silu_fwd` spells the
    /// same five ops and stores `silu(x)` directly, with no `up` multiply — the per-`Program`
    /// door's own elementwise arm lowered that as `("silu", 1)`, and routing elementwise through
    /// this walk makes the walk the door for it. So the chain reader recognises the store terminal
    /// as the chain's UNARY form rather than refusing what it just proved. (It WAS that refusal:
    /// "this door has no arm for a lone silu" was true of the whole-function door until the
    /// reroute; the per-`Program` elementwise arm it now feeds was always the arm.)
    #[test]
    fn a_stored_bare_silu_is_recognised_as_the_chain_s_unary_form() {
        let f = stored_chain();
        let cs = program_silu_mul_chains(&f).expect("the stored longhand is the same five ops");
        assert_eq!(cs.len(), 1, "one `arith.negf` heads one chain");
        let c = &cs[0];
        assert_eq!(c.gate, Ssa(0), "the gate is the value the negate consumed");
        assert!(
            c.up.is_none(),
            "no `up`: the longhand ends at the stored `silu(x)`"
        );
        assert_eq!(
            c.mul,
            Ssa(6),
            "the terminal is the divide's own result — the value the store writes"
        );
        assert_eq!(
            c.consumed,
            vec![Ssa(3), Ssa(4), Ssa(5)],
            "negate/exp/add are consumed by the fusion; the divide is the terminal, exactly as \
             the multiply is for the fused form"
        );
    }

    /// A longhand whose result NOTHING reads — neither an `up` multiply nor a store — is still
    /// refused: the five ops compute a value the program throws away, and no fused op stands for
    /// one of those.
    #[test]
    fn a_silu_with_no_reader_at_all_is_refused_by_name() {
        let e = program_silu_mul_chains(&chain(1.0, Num::Gate, false))
            .expect_err("a longhand nothing reads is not a program this door lowers");
        assert!(
            e.message.contains("nothing reads `silu(x)`"),
            "the refusal must say the terminal is what is missing; got: {}",
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

/// ⭐ THE EQUALITY-MASK FOLD'S REWIDTH — the sampler's Gumbel-max body spells its mask as
/// `(score == mm).to(tl.float32)` (the whole noise chain is fp32), and the fold used to accept
/// only a `uitofp` to F16, so the fp32 mask missed the fold and the bare `cmpf` reached the
/// walk's by-name refusal (`ArithCmpf has no 1:1 Program`) — the run-5 wall after the fp32
/// broadcast arm. The gate now accepts F16 OR F32 (the two formats the elementwise door lowers),
/// and the emitted `Elementwise(Equal)` states the RETYPE'S OWN df (`op_df` reads the result
/// type), so the fp32 mask lowers as the fp32 form of the same broadcast-op family
/// `maximum`/`minimum` take. This suite pins the widened gate against the fold's own laws: the
/// fp32 retype folds (new), the fp16 one still folds (control), and each refusal — a non-`oeq`
/// predicate, a retype to a format the door refuses, a second reader of the i1 — still fires.
#[cfg(test)]
mod equal_mask_tests {
    use super::*;
    use ktir_core::arena::Arena;
    use ktir_core::irtype::IrType;

    /// The two-op spelling a producer MUST emit (KTIR has no one-op equality tensor):
    /// `%i1 = arith.cmpf(pred, %x, %y)` then `%m = arith.uitofp(%i1)` retyped to `elem`.
    /// `second_reader` adds a second consumer of the i1, breaking the fold's one-reader seal.
    fn mask_fn(pred: &'static str, elem: DType, second_reader: bool) -> IRFunction<'static> {
        let a: &'static Arena = Arena::global();
        let (x, y, i1, m) = (Ssa(0), Ssa(1), Ssa(2), Ssa(3));
        let mut ops = vec![
            Operation::new(a, Some(i1), OpKind::ArithCmpf, &[x, y]).with_attr(
                a,
                AttrKey::Predicate,
                Attr::Str(pred),
            ),
            Operation {
                result_type: Some(IrType::Tensor {
                    dims: &[4, 256],
                    elem,
                }),
                ..Operation::new(a, Some(m), OpKind::ArithUitofp, &[i1])
            },
        ];
        if second_reader {
            ops.push(Operation::new(a, Some(Ssa(4)), OpKind::ArithUitofp, &[i1]));
        }
        IRFunction {
            name: "equal_mask_probe",
            arguments: a.args(vec![(x, IrType::Index), (y, IrType::Index)]),
            operations: a.ops(ops),
            grid: (1, 1, 1),
            return_type: None,
        }
    }

    #[test]
    fn the_fp32_retype_folds() {
        // The sampler's own spelling: `(score == mm).to(tl.float32)`.
        let masks =
            program_equal_masks(&mask_fn("oeq", DType::F32, false)).expect("the fold never errs");
        assert_eq!(
            masks.by_retype.get(&Ssa(3)).copied(),
            Some((Ssa(0), Ssa(1))),
            "the fp32 mask folds to the comparison's own operands"
        );
        assert!(
            masks.consumed.contains(&Ssa(2)),
            "the fold consumed the i1 the device op never materialises"
        );
    }

    #[test]
    fn the_fp16_retype_still_folds() {
        // The control: the fp16 mask the MoE ports rest on keeps its fold, unchanged.
        let masks =
            program_equal_masks(&mask_fn("oeq", DType::F16, false)).expect("the fold never errs");
        assert_eq!(
            masks.by_retype.get(&Ssa(3)).copied(),
            Some((Ssa(0), Ssa(1))),
            "the fp16 mask folds exactly as before the gate widened"
        );
    }

    #[test]
    fn a_non_oeq_predicate_is_not_folded() {
        // `OpFunc::Equal` is EQUALITY; any other predicate is a different function and must be
        // left for the by-name refusal rather than silently re-meant.
        let masks =
            program_equal_masks(&mask_fn("olt", DType::F32, false)).expect("the fold never errs");
        assert!(
            masks.by_retype.is_empty(),
            "a `cmpf(olt)` does not fold to `Elementwise(Equal)`"
        );
    }

    #[test]
    fn a_retype_to_a_format_the_door_refuses_is_not_folded() {
        // F16 and F32 are the two formats the elementwise door lowers; anything else would be
        // re-emitted at a format the door refuses by name, so the fold leaves it alone.
        let masks = program_equal_masks(&mask_fn("oeq", DType::Fp8E4m3, false))
            .expect("the fold never errs");
        assert!(
            masks.by_retype.is_empty(),
            "a `uitofp` to a non-F16/F32 width does not fold"
        );
    }

    #[test]
    fn a_second_reader_of_the_i1_blocks_the_fold() {
        // The fold DESTROYS the i1 (the device op never materialises it), so a second reader
        // would read a value nothing writes.
        let masks =
            program_equal_masks(&mask_fn("oeq", DType::F32, true)).expect("the fold never errs");
        assert!(
            masks.by_retype.is_empty(),
            "a cmpf whose i1 has two consumers does not fold"
        );
    }
}
