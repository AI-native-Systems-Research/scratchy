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
use ktir_core::ir::{Attr, IRFunction, Operation, Ssa};
use ktir_core::opkind::OpKind;

use super::lower_ktir_to_superdsc::{Error, Gather, Region, err, gathers_of, regions};
use crate::ktir_node::{Elementwise, KtirNode, Program, ReduceAxes, ReduceKind};
// The broadcast AXIS is the emission door's own vocabulary — `elementwise` selects the `In` builder
// from it — so it lives beside that door and this recogniser names the same type rather than a
// parallel one that would have to be mapped across the boundary.
use super::lower_ktir_to_superdsc::BcastAxis;
use crate::placement::BundleLayout;
use crate::sdsc_abstract::{BlockCols, Lanes, RowCount};
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

/// ⭐⭐⭐ WALL 13's ISLAND — the dtype CAST an op states, as the (from, to) pair the vendor's
/// convert binds name. `None` for every non-cast op.
///
/// The pair is read off the op's OWN operand/result types — the one statement of the transition —
/// never re-derived at the door. Only the two pairs the closed inventory vendor-binds are admitted
/// (`dl16tofp32`/`fp32todl16`, `quantization_double_pad.ddl:58/:59`); a cast stating any other
/// pair is REFUSED BY NAME rather than routed through a nearest-neighbor opfunc, because a convert
/// that changes the wrong format is a silent byte re-interpretation, not a loud failure.
///
/// ⛔ THE CAST'S DIRECTION IS ITS KIND'S OWN SEMANTICS, CHECKED: `arith.extf` widens and
/// `arith.truncf` narrows — a program stating the reverse is mistyped (its value's bytes would
/// re-interpret at the WRONG width), and naming that here is cheaper than diagnosing the garbage
/// it would bake. `arith.convertf` states no direction of its own, so both pairs are legal at it.
fn cast_of(f: &IRFunction<'_>, op: &Operation<'_>) -> Result<Option<Lowering>, Error> {
    if !matches!(
        op.op_type,
        OpKind::ArithExtf | OpKind::ArithTruncf | OpKind::ArithConvertf
    ) {
        return Ok(None);
    }
    let kind = op.op_type;
    let name = f.name;
    let refuse = |msg: String| Error {
        message: format!("{name}: `{kind:?}` {msg}"),
    };
    let elem_of = |v: Ssa, role: &str| -> Result<crate::superdsc_opspec::Df, Error> {
        let t = type_of(f, v).ok_or_else(|| Error {
            message: format!(
                "{name}: `{kind:?}`'s {role} states no type, so the format transition is \
                     unknown and no convert can be chosen for it"
            ),
        })?;
        df_of_dtype(match t {
            ktir_core::irtype::IrType::Tensor { elem, .. }
            | ktir_core::irtype::IrType::MemRef { elem, .. }
            | ktir_core::irtype::IrType::Scalar(elem) => elem,
            ktir_core::irtype::IrType::AccessTile { .. } | ktir_core::irtype::IrType::Index => {
                return Err(refuse(format!(
                    "{role} is an address type, not a value a dtype convert reads"
                )));
            }
        })
        .ok_or_else(|| Error {
            message: format!(
                "{name}: `{kind:?}`'s {role} states an element type with no stick geometry \
                     (bool/i64), which no convert bind in the vendor's DDLs addresses"
            ),
        })
    };
    let src = *op.operands.first().ok_or_else(|| Error {
        message: format!("{name}: `{kind:?}` states no operand to convert"),
    })?;
    let res = op.result.ok_or_else(|| Error {
        message: format!("{name}: `{kind:?}` states no result"),
    })?;
    let from = elem_of(src, "input")?;
    let to = elem_of(res, "result")?;
    // The direction check: extf widens, truncf narrows — in the f16/f32 lattice the vendor binds.
    let widens = matches!((from, to), (Df::Fp16, Df::Fp32));
    let narrows = matches!((from, to), (Df::Fp32, Df::Fp16));
    match kind {
        OpKind::ArithExtf if !widens => {
            return Err(refuse(format!(
                "states a WIDENING cast whose types are {from:?} → {to:?} — `arith.extf` extends, \
                 so this program is mistyped: the value's bytes would re-interpret at the wrong \
                 width no matter which convert emits them"
            )));
        }
        OpKind::ArithTruncf if !narrows => {
            return Err(refuse(format!(
                "states a NARROWING cast whose types are {from:?} → {to:?} — `arith.truncf` \
                 truncates, so this program is mistyped: the value's bytes would re-interpret at \
                 the wrong width no matter which convert emits them"
            )));
        }
        _ => {}
    }
    match (from, to) {
        (Df::Fp16, Df::Fp32) | (Df::Fp32, Df::Fp16) => Ok(Some(Lowering::Convert { from, to })),
        _ => Err(refuse(format!(
            "states the format transition {from:?} → {to:?}, and the closed convert inventory \
             vendor-binds only the fp16↔fp32 pair (`dl16tofp32`/`fp32todl16`, \
             quantization_double_pad.ddl:58/:59). Every other transition is refused BY NAME rather \
             than routed through a nearest-neighbor opfunc — a convert that changes the wrong \
             format is a silent byte re-interpretation, not a loud failure"
        ))),
    }
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
    /// A bare `linalg.reduce`, whose combiner is an attribute rather than an op kind. Carries
    /// the op's own axis facts (wall 12): the `Dimensions` it reduces and the rank it states
    /// them against, read at the same place the combiner is and threaded as a value — never
    /// re-derived, never defaulted — because the door classifies trailing-vs-leading off them
    /// and a frame that has already merged the leading dims into rows cannot.
    Reduce { kind: ReduceKind, axes: ReduceAxes },
    /// `tile * <splatted constant>` — the multiplier read off THIS op, not the whole function. See
    /// [`splat_scale_of`] for the discriminator and why it cannot be per-function here.
    ScalarMul(f32),
    /// `maximum(tile, C)` / `minimum(tile, C)` with `C` a splatted constant — the extremum's own
    /// scalar form, the same tiebreak [`ScalarMul`] states for multiply, applied to the two ops
    /// whose identity lives at the range's edge (the online softmax's running-max initial, any
    /// clamp against a stated bound). See [`splat_scale_of`] for the discriminator and
    /// `scalarextremum_at` for the body it lowers through.
    ScalarExtremum { value: f32, max: bool },
    /// `tile − C` / `C − tile` with `C` a splatted constant — the non-commutative member of the
    /// constant-operand pointwise class. The ORDER is the program's own statement and is carried
    /// here (`const_first`: the splat is the op's first input), because subtraction is the one op
    /// of the class whose two orders are DIFFERENT functions: the emission derives its operand
    /// layout from this fact (see [`SplatFact`]) rather than assuming one. The online softmax's
    /// running-max rescale at trip 0 — `m − m_new` against the `tl.full` initial — is the shape.
    ScalarSub { value: f32, const_first: bool },
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
    /// ⭐⭐⭐ WALL 13's ISLAND — a dtype CAST (`arith.extf` / `arith.truncf` / `arith.convertf`),
    /// lowered as a FIRST-CLASS CONVERT op through the vendor's own pair (`dl16tofp32` /
    /// `fp32todl16`, `quantization_double_pad.ddl:58/:59`). The cast's whole purpose is to CHANGE
    /// df, so the (from, to) pair is a property of the op's own operand/result types — read once
    /// here, never re-derived at the door.
    ///
    /// ⛔ THE MIXED-df PIN (the auditor's law, enforced here at the walk): a mixed-df TTIR op
    /// NEVER emits as ONE op. The casts are the ONLY ops whose operands may differ in df, and
    /// they surface as these convert ops — so the DD2 mixed-op refusal ("Unsupported result
    /// precision conversion", the `reduce.rs` const law's sibling) is UNREACHABLE by our
    /// emission: every elementwise/reduce op the walk emits states one df on every operand, and
    /// every df transition is one of these converts.
    Convert { from: Df, to: Df },
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
/// Returns the constant-operand pointwise's stated shape: the constant's VALUE, the TENSOR
/// operand, and WHICH INPUT the constant occupies — the ORDER fact. `None` when this op does
/// not state exactly one splatted-constant operand:
///
/// * NEITHER operand is a splat — an ordinary two-tensor `Elementwise` op, which the 1:1 map handles.
/// * BOTH are splats — a multiply of two compile-time constants. That is not a scalarmul (there is no
///   tensor to scale) and emitting one would invent an operand; it belongs to constant folding
///   upstream, so it falls through to be refused by name.
///
/// ⛔ THE ORDER IS STRUCTURAL, NOT COSMETIC. For a COMMUTATIVE op (`multiply`, `maximum`,
/// `minimum`) the two orders lower to the same function and the emission may swap operands for
/// its own placement laws. For a NON-commutative one (`subtract`) the two orders are DIFFERENT
/// functions — `C − tile` is not `tile − C` — so the emission derives its operand layout from
/// this fact and never assumes one.
#[derive(Clone, Copy, Debug)]
pub struct SplatFact {
    pub value: f32,
    pub tensor: Ssa,
    /// The splatted constant is the op's FIRST input (`C ∘ tile`); `false` means `tile ∘ C`.
    pub const_first: bool,
    /// THE SPLAT OP ITSELF — the SSA whose `result_type` states the element format the constant
    /// is splatted at. The scale-registry completion reads the slot's df from THIS op (the
    /// splat's own result, the same seat the value came from), so the registry's format column
    /// cannot drift from the value column: one fact, one matcher.
    pub splat: Ssa,
}

pub fn splat_scale_of(
    f: &IRFunction<'static>,
    op: &ktir_core::ir::Operation<'static>,
) -> Option<SplatFact> {
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
        (Some(v), None) => Some(SplatFact {
            value: v as f32,
            tensor: b,
            const_first: true,
            splat: a,
        }),
        (None, Some(v)) => Some(SplatFact {
            value: v as f32,
            tensor: a,
            const_first: false,
            splat: b,
        }),
        // Both or neither: not a scalar multiply. See the doc above.
        _ => None,
    }
}

/// ⭐⭐⭐ A `ktdp.store` WHOSE VALUE IS A SPLATTED CONSTANT — the FILL's own discriminator, the
/// store-side twin of [`splat_scale_of`]'s operand test. A store of `tensor.splat` over an
/// `arith.constant` writes one stated value across its whole window; no load feeds it and no
/// compute op consumes it, so the walk's compute loop would never reach it and the program it
/// belongs to would die at "no compute op in the function" — a pure-fill program being the
/// shape the online-softmax carry's seed states (max at the finite −65504 sentinel, sum and
/// accumulator at zero).
///
/// The constant is resolved through the SAME registry the constant-operand pointwise class uses
/// ([`program_stated_scales`] collects these stores beside the mulf/subf/extrema splats), so the
/// fill and the scale ops share one staging channel: a worker-bound `[1,1]` const, broadcast over
/// the store's own window by the identity pointwise — the same descriptor shape
/// [`super::lower_ktir_to_superdsc::scalarmul_scaled`] gives a scalar op, with the broadcast
/// const as the only input.
///
/// Returns the VALUE and the splat's own SSA (the seat the value and its element format both
/// came from — the same provenance [`SplatFact::splat`] carries). `None` when the store's value
/// is not exactly a splatted compile-time float (a loaded or computed value; those belong to the
/// compute ops that produce them).
pub fn stored_splat_of(
    f: &IRFunction<'static>,
    op: &ktir_core::ir::Operation<'static>,
) -> Option<(f32, Ssa)> {
    let def_of = |s: Ssa| f.operations.iter().find(|o| o.result == Some(s));
    // The same `arith.constant` → `tensor.splat` spelling [`splat_scale_of`]'s `splat_value`
    // reads — the only spelling this crate's producers use, read at the store instead of at a
    // constant-operand op.
    let v = *op.operands.first()?;
    let sp = def_of(v)?;
    if sp.op_type != OpKind::TensorSplat {
        return None;
    }
    let c = def_of(*sp.operands.first()?)?;
    if c.op_type != OpKind::ArithConstant {
        return None;
    }
    let value = c.attributes.iter().find_map(|(k, val)| match (k, val) {
        (AttrKey::Value, Attr::Float(x)) => Some(*x),
        _ => None,
    })?;
    Some((value as f32, v))
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

/// ⭐⭐⭐ WALL 12 — a `linalg.reduce`'s own [`ReduceAxes`], read off its `Dimensions` and result
/// `Shape` attributes the same way [`reduce_kind_of`] reads its combiner: the axes are the OP's
/// statement, not the walk's frame, and the door classifies the block from them
/// ([`crate::emit::lower_ktir_to_superdsc::reduce_axis_block`]).
///
/// ⛔ FAIL-CLOSED on a missing or oversized statement: a reduce stating no `Dimensions` cannot
/// be classified (trailing and leading are indistinguishable without it), and one stating more
/// than four reduced dims has no bind in the template's own vocabulary (`summeanmaxexx2.ddl:9`
/// declares exactly `%reduce_dim:4`). Both are named here, at the read, rather than guessed at
/// the door — the same refusal discipline [`reduce_kind_of`]'s caller applies to the combiner.
pub(crate) fn reduce_axes_of(
    f_name: &str,
    op: &ktir_core::ir::Operation<'static>,
) -> Result<ReduceAxes, Error> {
    let dims: &[i64] = op
        .attributes
        .iter()
        .find_map(|(k, v)| match (k, v) {
            (AttrKey::Dimensions, Attr::IntList(d)) => Some(*d),
            _ => None,
        })
        .ok_or_else(|| Error {
            message: format!(
                "{f_name}: a `linalg.reduce` states no `Dimensions`, so which axes it reduces is \
                 unknown and the door cannot classify it (a trailing reduce is the row-reduce form, \
                 a leading prefix is the nonstick form — they are indistinguishable without the \
                 op's own statement)"
            ),
        })?;
    if dims.is_empty() {
        return Err(Error {
            message: format!(
                "{f_name}: a `linalg.reduce` states an EMPTY `Dimensions` list — a reduce over \
                 no axes is the identity, and TTIR never states one; defensively refused rather \
                 than guessed at the door"
            ),
        });
    }
    if dims.len() > 4 {
        return Err(Error {
            message: format!(
                "{f_name}: a `linalg.reduce` stating {} reduced dims has no bind to match — the \
                 reduce template's own dimension vocabulary is four reduce symbols \
                 (`summeanmaxexx2.ddl:9`, `%reduce_dim:4`)",
                dims.len()
            ),
        });
    }
    // The rank the dims are stated against: the result `Shape`'s rank plus the reduced count
    // (a reduce removes exactly the axes it names). A reduce stating no result `Shape` cannot
    // be ranked — fail-closed, same discipline as the missing `Dimensions` above.
    let out_rank: usize = op
        .attributes
        .iter()
        .find_map(|(k, v)| match (k, v) {
            (AttrKey::Shape, Attr::IntList(s)) => Some(s.len()),
            _ => None,
        })
        .ok_or_else(|| Error {
            message: format!(
                "{f_name}: a `linalg.reduce` states no result `Shape`, so the rank its \
                 `Dimensions` are stated against is unknown and the door cannot classify the \
                 block"
            ),
        })?;
    let mut padded = [0i64; 4];
    padded[..dims.len()].copy_from_slice(dims);
    Ok(ReduceAxes {
        dims: padded,
        n_dims: dims.len() as u8,
        in_rank: (out_rank + dims.len()) as u8,
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

/// ⭐⭐⭐⭐⭐ IS THIS RESULT READ BY ANYTHING OTHER THAN ITS STORE? The store arm binds a store to
/// its producing op only when the result is the store's ALONE — that arm emits the op writing the
/// store's window IN PLACE, at the op's own program position, which is correct only while no later
/// op still reads the value the window held BEFORE. The online-softmax fold body's running max is
/// the first value this door has seen that is BOTH stored (the carry's write-back at the trip's
/// end) and consumed on the way there (`m − m_new`, `qk − m_new[.., .., None]`): binding it would
/// overwrite the old max before the subtraction reads it (`m_new − m_new = 0` — a descriptor reads
/// at ITS OWN program position, never at the load's). When this returns true the walk mints the
/// result as the intermediate it already is to those consumers, and the STORE lowers at its own
/// position as the write-back copy (`mintcopy_at`).
///
/// The walk is FORWARD over users: a STORE user is the binder (or a second store of the same value
/// — also a binder); a user that only REFRAMES the value without computing on it (the reshape and
/// broadcast/expand plumbing the store's own value chain runs through) hands its result to the
/// frontier; anything else — a compute op, a yield, a return — is a consumer. The yields and
/// returns are consumers rather than reframes on purpose: they carry real dataflow, and treating
/// one as a reframe would silently drop it from the frontier.
fn result_consumed_beyond_store(f: &IRFunction<'static>, res: Ssa) -> bool {
    let carries = |c: &ktir_core::ir::Operation<'static>| {
        (is_plumbing(c.op_type)
            && !matches!(
                c.op_type,
                OpKind::KtdpStore | OpKind::FuncReturn | OpKind::LinalgYield | OpKind::ScfYield
            ))
            || matches!(
                c.op_type,
                OpKind::TensorReshape
                    | OpKind::LinalgBroadcast
                    | OpKind::TensorExpandShape
                    | OpKind::TensorCollapseShape
            )
    };
    let mut frontier = vec![res];
    let mut seen = std::collections::HashSet::new();
    while let Some(v) = frontier.pop() {
        if !seen.insert(v) {
            continue;
        }
        for c in f.operations.iter() {
            if !c.operands.contains(&v) {
                continue;
            }
            if c.op_type == OpKind::KtdpStore {
                continue; // the binder itself
            }
            if carries(c) {
                if let Some(r) = c.result {
                    frontier.push(r);
                }
                continue;
            }
            return true;
        }
    }
    false
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
        let rank = bc
            .attributes
            .iter()
            .find_map(|(k, v)| match (k, v) {
                (AttrKey::Shape, Attr::IntList(d)) => Some(d.len()),
                _ => None,
            })
            .or_else(|| {
                bc.result_type
                    .as_ref()
                    .and_then(|t| t.dims().map(|d| d.len()))
            });
        let Some(rank) = rank else {
            return err(format!(
                "{}: `linalg.broadcast` states no shaped result, so its rank — the frame the \
                 `Dimensions` axis is read in — is unknown",
                f.name
            ));
        };
        // Mutable because the Y ADMISSION below re-states a leading multi-row spray's axis and
        // flat flag after the walk-back has resolved the source's own row count.
        let (mut axis, mut flat) = match (rank, dims) {
            (_, [1]) if rank <= 2 => (BcastAxis::Col, false),
            (_, [0]) if rank <= 2 => (BcastAxis::Mb, false),
            (r, [d]) if r > 2 && *d as usize == r - 1 => (BcastAxis::Col, true),
            (r, [d]) if r > 2 && *d == 0 => (BcastAxis::Mb, true),
            (r, [d]) if r > 2 => {
                return err(format!(
                    "{}: `linalg.broadcast` adds dim {d} of a rank-{r} tensor — a MIDDLE axis of \
                     the flat frame. Row-major that is ROW-REPLICATION: the source's extent on \
                     that axis is 1 by the broadcast's own law, so each of the source's flat rows \
                     repeats down the merged row axis (the batched paged-decode kernel's mask add \
                     is exactly this shape, `[B, 1, PS] -> [B, G, PS]`). The device's pointwise \
                     operand modes state a per-row scalar (`In::col`, one address per row) and a \
                     row vector (`In::mb`, one row addressed once); neither states an address that \
                     advances every k rows, and no descriptor in this door's vocabulary does, so \
                     this is named rather than half-addressed. The door-proven statement of the \
                     same math is `attn.py`'s own mask law: the host stages the mask at the score \
                     tile's own rows and the kernel adds it at base — no broadcast at all",
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
                        // `Mb` and the (later-assigned) `Y` both spray a LEADING axis: the
                        // relay's insert must be the leading `1` either way — at classification
                        // time the axis is still `Mb` (the `Y` admission runs after the
                        // walk-back, on the resolved source's own row count), and a chain the
                        // admission flips to `Y` passed the same leading-`1` check here.
                        BcastAxis::Mb | BcastAxis::Y(_) => Some(0),
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
                                // `Mb` and the post-walk-back `Y` both spray the LEADING axis —
                                // the same word names it for the reader either way.
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
                _ => break,
            }
            chain.push(cursor);
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
                let y = s.first().copied().unwrap_or(0);
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
                    Some(rows) if s.get(1).is_some_and(|&m| m as u32 == rows) => {
                        axis = BcastAxis::Y(y as u32);
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
                src: cursor,
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

/// One reshape's FLAT-PRESENTATION verdict — the #741 re-row law
/// ([`crate::sdsc_abstract::RegionFrame::reframe`], the door
/// [`crate::sdsc_abstract::Access::reframe`] states for a reshape over a minted access) run at
/// the forward, from the source's own type to the reshape target's flat presentation.
#[derive(Clone, Debug)]
enum ReshapeFlat {
    /// The target's flat `[rows', cols']` is byte-faithful over the source's frame (the
    /// identity, or a pad-up inside one stick): every consumer that reads the value flat or at
    /// the source's own framing reads the bytes the source holds. Every reshape of the batched
    /// paged-decode kernel's flat values is this — `[1024, 64] -> [256, 4, 64]` is flat
    /// `[1024, 64]`, the identity.
    Admitted,
    /// The re-row law REFUSES the flat presentation — the target's framing moves bytes the
    /// source's frame does not (`[32, 131072] -> [256, 64, 256]` is flat `[16384, 256]`, and the
    /// inter-stick stride is the ROW count, so the re-row moves every element past the first
    /// stick column). The value is still FORWARDED — the bytes are the source's, and the ONE
    /// consumer that states its own affine access over the source may lawfully read it (the
    /// gathered matmul's B, the `Access::gathered` mint) — but a consumer that reads it at the
    /// target's framing without stating that access is refused at the resolution site, naming
    /// the law's own clause.
    Refused(String),
}

/// The stick law's element format — the [`crate::superdsc_opspec::Df`] whose `elems_per_stick`
/// is the frame's lane count, from the value's own element type. `None` for the formats with no
/// stick geometry in this dialect (bool, i64): a reshape over those states no frame law this
/// door can check, which the verdict refuses by name.
pub(crate) fn df_of_dtype(elem: ktir_core::dtypes::DType) -> Option<crate::superdsc_opspec::Df> {
    use ktir_core::dtypes::DType;
    Some(match elem {
        DType::F16 => crate::superdsc_opspec::Df::Fp16,
        DType::F32 => crate::superdsc_opspec::Df::Fp32,
        DType::Fp8E4m3 => crate::superdsc_opspec::Df::Fp8,
        DType::I32 => crate::superdsc_opspec::Df::SenUint32,
        DType::Bool | DType::I64 => return None,
    })
}

/// A value's own type — the argument list's entry, or the defining operation's `result_type`.
/// Every SSA in a TTIR function carries one; `None` says the program did not state it.
fn type_of<'f>(f: &IRFunction<'f>, v: Ssa) -> Option<ktir_core::irtype::IrType<'f>> {
    if let Some(&(_, t)) = f.arguments.iter().find(|&&(s, _)| s == v) {
        return Some(t);
    }
    f.ops_deep()
        .into_iter()
        .find(|o| o.result == Some(v))
        .and_then(|o| o.result_type)
}

/// A shape's FLAT `[rows, cols]` presentation — the leading product over the last dim, the
/// frame a whole-value flat read addresses. `None` for an empty or negative shape.
fn flat_of(dims: &[i64]) -> Option<(usize, usize)> {
    let (&last, lead) = dims.split_last()?;
    if last < 0 || lead.iter().any(|&d| d < 0) {
        return None;
    }
    let mut rows: u64 = 1;
    for &d in lead {
        rows = rows.checked_mul(d as u64)?;
    }
    usize::try_from(rows).ok().map(|r| (r, last as usize))
}

/// The re-row law for ONE reshape — the target's flat presentation against the source's own
/// frame, both stated from the program's own types (the source's defining type, the reshape's
/// `result_type`). A reshape whose flat law cannot be stated (either side unshaped, an element
/// format with no stick law, a target that changes the element type) is the REFUSED verdict's
/// sibling: this walk does not forward a lens it cannot vouch for, so the consumer-side check
/// refuses it by name rather than passing it blind.
fn reshape_flat_law(f: &IRFunction<'_>, op: &Operation<'_>, src: Ssa) -> ReshapeFlat {
    let refuse = |what: &str| {
        ReshapeFlat::Refused(format!(
            "the reshape of t{} states {what}, so its flat presentation's byte law cannot be \
             stated — the #741 re-row law this forward checks needs both frames",
            src.0
        ))
    };
    let Some(tt) = op.result_type else {
        return refuse("no result type");
    };
    let Some(target) = tt.dims().and_then(flat_of) else {
        return refuse("a result shape the flat presentation cannot read");
    };
    let Some(st) = type_of(f, src) else {
        return refuse("no type on the value it reshapes");
    };
    let Some(source) = st.dims().and_then(flat_of) else {
        return refuse("a source shape the flat presentation cannot read");
    };
    let Some(df) = st.elem().and_then(df_of_dtype) else {
        return refuse("an element format with no stick law");
    };
    if tt.elem().is_some_and(|e| Some(e) != st.elem()) {
        return refuse("a different element type than the value it reshapes");
    }
    match crate::sdsc_abstract::RegionFrame::stick_blocked(source.0, source.1, df)
        .reframe(target.0, target.1)
    {
        Ok(_) => ReshapeFlat::Admitted,
        Err(msg) => ReshapeFlat::Refused(msg),
    }
}

/// ⭐⭐⭐ THE `tensor.reshape` FORWARDS — every reshape result maps to the value it reshapes, and
/// every consumer of a reshape reads THROUGH it to that value.
///
/// `tl.reshape` is numpy semantics: the result preserves the source's flat C-order index exactly
/// (element at flat index `i` stays at flat index `i`), never permuting. So a reshape moves no
/// byte and states no computation — it is a LENS the consumer reads the same bytes through. The
/// walk therefore treats the op as plumbing (no descriptor, nothing minted) and resolves every
/// value a consumer names through the chain to the value underneath: a load (the region is the
/// parameter's), a gathered load (the identity join finds its tile), a compute op's result (the
/// minted intermediate). Reshape-of-reshape resolves transitively — the map records the
/// ULTIMATE source, built in definition order so a later reshape reads the earlier one's entry.
///
/// ⭐⭐⭐ AND THE LENS IS LAW-CHECKED AT THE REFRAME — [`reshape_flat_law`] runs the #741 re-row
/// law at the forward itself, the same law [`crate::sdsc_abstract::Access::reframe`] states for
/// a reshape over a minted access: the target's flat presentation is admitted only when every
/// element addresses the byte it did (`rows' == rows`, the stick-column count unchanged). A
/// refused reshape does not stop the forward — the bytes are the source's either way — but it
/// marks the value CARRIED: only the consumer that states its own affine access over the source
/// (the gathered matmul's B, the `Access::gathered` mint) may read that framing, and the
/// consumer-side check refuses every other reader by name. Without that check a `[1024, 64]`
/// value reshaped `[64, 1024]` and consumed by a row-framed op would read the source's frame at
/// the target's extents — a clean bake of wrong bytes, exactly what the re-row law exists to
/// refuse.
///
/// The reshape forwards' maps: (result → ultimate source), (result → the reshape's own input,
/// for the per-hop law walk), each result's law verdict, and the set of every reshape result
/// (the plumbing the walk's op loop skips).
type ReshapeForwards = (
    std::collections::HashMap<Ssa, Ssa>,
    std::collections::HashMap<Ssa, Ssa>,
    std::collections::HashMap<Ssa, ReshapeFlat>,
    std::collections::HashSet<Ssa>,
);

/// Returns the forwards map (result → ultimate source), the immediate-source map (result → the
/// reshape's own input, for the per-hop law walk), each result's law verdict, and the set of
/// every reshape result — the values the walk's op loop skips as plumbing.
fn reshape_forwards(f: &IRFunction<'_>) -> ReshapeForwards {
    let mut forwards = std::collections::HashMap::new();
    let mut hops = std::collections::HashMap::new();
    let mut verdicts = std::collections::HashMap::new();
    let mut consumed = std::collections::HashSet::new();
    for op in f.ops_deep() {
        if op.op_type != OpKind::TensorReshape {
            continue;
        }
        let Some(res) = op.result else { continue };
        let Some(src) = op.operands.first().copied() else {
            continue;
        };
        // The ultimate source: a reshape of a reshape reads the earlier entry, so the map is
        // transitively closed by construction (definitions precede uses in straight-line SSA).
        let ultimate = forwards.get(&src).copied().unwrap_or(src);
        forwards.insert(res, ultimate);
        hops.insert(res, src);
        verdicts.insert(res, reshape_flat_law(f, op, src));
        consumed.insert(res);
    }
    (forwards, hops, verdicts, consumed)
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
    }
    Ok(r)
}

/// The rows a banded op's minted intermediate must hold: a WINDOWED input's band (`r_len`), or
/// `logical` when every input is whole-view. See the mint's own doc for the full argument.
///
/// ⛔⭐⭐⭐ A BROADCAST OPERAND's WINDOW IS NOT A BAND THE OP SWEEPS, so it states nothing here.
/// The band law reads the sweep off an input the op addresses ROW BY ROW — a dense operand of a
/// banded program (the per-core uniform rope copy). A broadcast operand is addressed at its
/// MODE's own extent, never at the output's rows: `Col` one lane per row, `Mb` one row once,
/// and `Y` one spray's worth (`rows/y`) at a per-trip window whose corner rides `In::y_at` —
/// the prefill mask law, where the window is exactly the thing that is NOT the sweep (the op
/// sweeps the output's whole `[B·mq, 64]` while the mask window holds one `[mq, 64]` slice).
/// Reading the band off a broadcast operand mints the output at the SOURCE's window height and
/// refuses the dense operand beside it as an extent mismatch. MEASURED (wall 18, build 9, the
/// granite-2b prefill re-roll at `prefill_mq=256`): the fold statement threads the ladder's
/// 8-page region into every rung's mask view (`[8192, 64]`) while the one-shot twin's own
/// 1-page walk hulls its load windows to `(0, 1024)` — the mask became the first "banded"
/// input, the add's output minted at the mask window's `[256, 64]`, and qk (`[8192, 64]`,
/// dense) was refused against it. `bcast` is indexed by input position exactly as `ins` is
/// ([`lower_function`]'s region loop pushes one region per `in_values` entry in order); the
/// gathered-index regions appended past the inputs have no flag and keep the verbatim law.
fn band_rows_of(ins: &[Region], bcast: &[Option<BcastAxis>], logical: u32) -> u32 {
    ins.iter()
        .enumerate()
        .find(|(i, x)| bcast.get(*i).is_none() && x.r_cover != (0, x.v_rows))
        .map_or(logical, |(_, x)| x.r_len)
}

/// ⭐⭐⭐⭐⭐ THE ROW SWEEP A RANK-2 RESULT RESPELLS — `None` keeps the stated shape verbatim.
///
/// A row reduce writes ONE VALUE PER DATA ROW: the accum is `Scale::RedStick` (one whole stick
/// per row, value k at stick k — `win_reduce_accum`'s own law), so the reduce's result, however
/// the program SPELLS it, is the data's row-sweep vector in the accumulator's frame. A `[256, 4]`
/// result of a reduce over `[256, 4, 64]` (flat data rows 1024, the last axis one stick) IS the
/// `[1024, 1]` accumulator respelled — while the STICK-BLOCKED `[256, 4]` reading (value (i, j)
/// at stick i, lane j) is a DIFFERENT BYTE LAYOUT, not a respelling of it, so the verbatim mint
/// and the accumulator cannot describe one buffer. MEASURED (wall 11, the spliced online
/// softmax, `attn_s1` geom=(32,8,64) mq=1): the verbatim mint placed the reduce's output at
/// `[256, 4]` — a 256-stick footprint — while the accum writes 1024 sticks, and the door refused
/// correctly ("a row reduction along the stick axis writes ONE value per row — `[1024, 1]`").
///
/// The discriminator is the NUMEL, and it is exact in both directions:
///
/// * THE REDUCE's OWN RESULT (`reduce_rows`, the data's `r_len` — the door's own `rows`
///   derivation, read off the data rather than the output): one value per row means the result
///   holds exactly `r_len` values, so the respelling fires iff the stated numel EQUALS it. A
///   `[4, 64]` result of an axis-0 reduce (numel 256 ≠ 1024) keeps its shape and the door
///   refuses it by name — the right answer, that reduce is not the row-reduce form.
/// * EVERY OTHER MEMBER of the family: an op READING a one-column tile is on the reduction's
///   side of the program (the rank-1 arm's own discriminator, one rank up), and a rank-2 result
///   whose numel equals that input's `r_len` is the same value vector respelled — elementwise
///   TTIR semantics preserve the flat index map, so the frame is byte-true. The online
///   softmax's `[256, 4]` add of two `[1024, 1]` intermediates is the measured shape.
///
/// ⛔ AND ONLY THOSE TWO: a WIDE input's row count is not a frame fact for anything but a
/// reduce (a `[32, 64]` gather copy over a `[2048, 64]` table has numel 2048 == the table's
/// rows — minting it `[2048, 1]` would reframe the copy the matmul reads as an activation), and
/// a result whose numel matches NO family input keeps its stated shape and meets its own door's
/// law — fail-closed there, never silently re-framed here.
pub(crate) fn respelled_frame_rows(
    per_op: &[Region],
    numel: u64,
    reduce_rows: Option<u32>,
) -> Option<u32> {
    if let Some(r) = reduce_rows
        && r as u64 == numel
    {
        return Some(r);
    }
    per_op
        .iter()
        .find(|r| !r.is_out && r.c_len == 1 && r.r_len as u64 == numel)
        .map(|r| r.r_len)
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
    /// `maximum(tile, C)` / `minimum(tile, C)` with `C` a splatted constant — the extremum's own
    /// scalar form. The bound AND the direction are part of the kind, for the same reason the
    /// multiplier is part of [`RowFoldKind::ScalarMul`]: a family whose blocks clamped at
    /// different bounds (or in different directions) computes different functions per block.
    ScalarExtremum { value: f32, max: bool },
    /// `tile − C` / `C − tile` with `C` a splatted constant — the subtraction's own scalar form. The
    /// bound AND the ORDER are part of the kind, for the same reason the multiplier is part of
    /// [`RowFoldKind::ScalarMul`]: subtraction is not commutative, so a family whose blocks stated
    /// different orders computes different functions per block.
    ScalarSub { value: f32, const_first: bool },
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
        if let Some(sf) = splat_scale_of(f, op) {
            return Some(RowFoldKind::ScalarMul(sf.value));
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
    // The extremum's scalar tiebreak, the same precedence the loop states — BEFORE the 1:1 map,
    // which would otherwise call a clamp-with-constant `Elementwise(Maximum/Minimum)` and address
    // the splat as a second input.
    if matches!(op.op_type, OpKind::ArithMaxnumf | OpKind::ArithMinnumf)
        && let Some(sf) = splat_scale_of(f, op)
    {
        return Some(RowFoldKind::ScalarExtremum {
            value: sf.value,
            max: op.op_type == OpKind::ArithMaxnumf,
        });
    }
    // The subtraction's scalar tiebreak, the same precedence and for the same reason — and the
    // ORDER rides the kind, because `C − tile` and `tile − C` are different functions a merge
    // must not average over.
    if op.op_type == OpKind::ArithSubf
        && let Some(sf) = splat_scale_of(f, op)
    {
        return Some(RowFoldKind::ScalarSub {
            value: sf.value,
            const_first: sf.const_first,
        });
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
                let Some(sf) = splat_scale_of(f, term) else {
                    return err(format!(
                        "{}: a scalar-multiply family member whose multiplier is not a splat",
                        f.name
                    ));
                };
                vec![sf.tensor]
            }
            RowFoldKind::ScalarExtremum { .. } => {
                let Some(sf) = splat_scale_of(f, term) else {
                    return err(format!(
                        "{}: an extremum family member whose bound is not a splat",
                        f.name
                    ));
                };
                vec![sf.tensor]
            }
            RowFoldKind::ScalarSub { .. } => {
                let Some(sf) = splat_scale_of(f, term) else {
                    return err(format!(
                        "{}: a scalar-subtract family member whose subtrahend is not a splat",
                        f.name
                    ));
                };
                vec![sf.tensor]
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
fn mint_inter(k: &KtirNode, layout: &BundleLayout, floor: u32, ssa: Ssa) -> u32 {
    let mut minted = k.minted.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(&tid) = minted.get(&ssa)
        && layout
            .synth
            .borrow()
            .map
            .contains_key(&crate::place::act_name(tid))
    {
        return tid;
    }
    let tid = layout.take_inter_tid(floor);
    minted.insert(ssa, tid);
    tid
}

pub fn lower_function(
    k: &KtirNode,
    layout: Option<&BundleLayout>,
    sym_id_base: &mut i64,
    headmajor: Option<super::lower_ktir_to_superdsc::HeadmajorOproj>,
) -> Result<Vec<super::EmittedOp>, Error> {
    let f = &k.func;
    let mut out = Vec::new();
    // ⭐ THE EMITTER CONTEXT, MINTED ONCE AT THIS DOOR — the sink this walk owns, with its own pair
    // parameters as the counter and the placement authority (the mint-once-at-door pattern). Every
    // push, sub-door call and materialization below goes through the ONE `&mut ctx`.
    let mut ctx = super::EmitCtx {
        ops: &mut out,
        sym_id_base,
        layout,
    };
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
    let floor: u32 = k
        .bindings
        .iter()
        .chain(k.mask.iter())
        .map(|b| b.get())
        .filter(|t| !crate::reserved_tids::is_reserved(*t))
        .max()
        .map_or(0, |m| m + 1);
    // ⛔ A BOUND PARAMETER THE BUNDLE DOES NOT PLACE IS REFUSED HERE, BY NAME. A placement is
    // a promise the WORKER fills: the program's `ktdp.construct_memory_view` states the
    // extent, but nothing states who writes the bytes. A per-step input (an index table, a
    // mask) synth-declared into the Intermediate segment would bake a buffer no runtime step
    // ever writes, and the bundle would read garbage on card — so the walk declares nothing
    // itself. The two honest fixes live at the producer: a graph tensor is placed by the tape
    // walk, and a per-step input needs an Activation placement plus a forward-tape fill step.
    if let Some(l) = ctx.layout {
        for r in regions(k)?.iter() {
            if l.placements.contains_key(&r.tid) || crate::reserved_tids::is_reserved(r.tid) {
                continue;
            }
            return err(format!(
                "{}: the bound parameter t{} (its view states [{}, {}]) has no placement in \
                 this bundle — no step is promised to fill it. A graph tensor is placed by the \
                 tape walk; a per-step input needs an Activation placement plus a forward-tape \
                 fill step at its producer. The walk declares nothing itself: a walk-side synth \
                 would bake a buffer nothing writes.",
                f.name, r.tid, r.v_rows, r.v_cols
            ));
        }
    }
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
    // ⭐⭐⭐ THE RESHAPE FORWARDS, read before the walk for the same reason the broadcast plumbing
    // is: a `tensor.reshape` moves no byte (numpy semantics — the flat C-order index is
    // preserved), so the op lowers to NOTHING and every consumer reads through it to the value
    // underneath. See [`reshape_forwards`] for the law: the re-row verdict at the forward and
    // the consumer-side check below together are the `Access::reframe` instance — a refused
    // lens is readable only by the consumer that states its own affine access.
    let (reshapes, reshape_hops, reshape_verdicts, reshape_consumed) = reshape_forwards(f);
    let through = |v: Ssa| reshapes.get(&v).copied().unwrap_or(v);
    // ⭐⭐⭐ THE GATHERS THE PROGRAM STATES, read ONCE and from [`gathers_of`] — the same joins the
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
    // The indexed WRITES the program states ([`scatters_of`]), lowered at their stores below.
    let scatters = super::lower_ktir_to_superdsc::scatters_of(k)?;
    // The row fold's own view of the same analyses — built once here so its helpers read ONE
    // value rather than threading three slices each (see [`FoldAnalyses`]).
    let analyses = FoldAnalyses {
        silus: &silus,
        rmsnorms: &rmsnorms,
        gathered_loads: &gathered_loads,
    };

    for op in f.operations.iter() {
        // ⭐ AN INDEXED STORE — a `ktdp.store` through a scatter tile — lowers HERE, at its own
        // program position, so a later gather of the same table reads what it wrote (dxp program
        // order is call order).
        if op.op_type == OpKind::KtdpStore
            && let Some(sc) = scatters
                .iter()
                .find(|s| op.operands.get(1) == Some(&s.tile))
        {
            ctx.ops
                .push(lower_indexed_write(k, sc, ctx.sym_id_base, ctx.layout)?);
            continue;
        }
        // ⭐⭐⭐ A STORE OF A SPLATTED CONSTANT — a FILL, the store-side member of the
        // constant-operand class the `scalar`/`extremum`/`scalar_sub` tiebreaks above state for
        // compute ops. A store of `tensor.splat` over an `arith.constant` is lowered HERE, at its
        // own program position, as the identity broadcast of the value's `[1,1]` registry const
        // over the store's own window (the window resolved through the store's tile exactly as
        // the compute store arm below resolves its output — the load-side law "the window is the
        // tile's own", stated for a store with no compute op to carry it). A program whose stores
        // are ALL fills is then not the "no compute op" refusal's shape but a legitimate program:
        // `lowered` counts these, so the online-softmax carry's seed (three constant stores, one
        // program) lowers as three descriptors instead of dying at the door.
        if op.op_type == OpKind::KtdpStore
            && let Some((value, _splat)) = stored_splat_of(f, op)
        {
            let out_tile = op.operands.get(1).copied();
            let tile = f
                .operations
                .iter()
                .find(|o| o.op_type == OpKind::KtdpConstructAccessTile && o.result == out_tile);
            let Some(tile) = tile else {
                return err(format!(
                    "{}: a constant store names no access tile — nothing states the window it fills",
                    f.name
                ));
            };
            let Some(mut o) = region_of_view_operand(k, tile, 0)? else {
                return err(format!(
                    "{}: a constant store names no parameter — a fill with no bound buffer has \
                     no window to write",
                    f.name
                ));
            };
            // ⭐⭐⭐ THE CARRY SEED'S CROSS-CHECK — the sentinel pair's own door, fired where the
            // program's stated fill and the buffer it targets are both in hand. The paged fold's
            // max carry ([`crate::reserved_tids::PagedFoldCarry::Max`]) is seeded by the kernel's
            // own `tl.full` constant, and the validity lanes the HOST stages carry
            // [`crate::sdsc_abstract::ATTN_SEED_FLOOR`] — ONE value by the seed-safety law (a
            // masked lane's score saturates at the floor, so its `p` is 0 only while `m` sits
            // above the seed the lanes name). The two spellings are independent producers, and a
            // drift on one side only changes the expunge margins in the dangerous direction —
            // fluent output, never a fault — so the walk refuses the desync, naming both values.
            // The Sum/Acc carries seed at the additive identity 0, a value only the kernel
            // spells: no second spelling exists to desync from, so no door fires there.
            if o.tid == crate::reserved_tids::PagedFoldCarry::Max.tid()
                && value != crate::sdsc_abstract::ATTN_SEED_FLOOR.to_f32_const()
            {
                return err(format!(
                    "{}: the fill of the paged-fold carry's max seeds at {value}, but the \
                     validity lanes the host stages carry the finite f16 floor {} — the seed and \
                     the lanes are ONE value (a masked lane's p is 0 only because its score \
                     saturates at the floor the seed holds). Align the kernel's seed constant \
                     with ATTN_SEED_FLOOR.",
                    f.name,
                    crate::sdsc_abstract::ATTN_SEED_FLOOR.to_f32_const()
                ));
            }
            o = windowed_region(k, o, tile)?;
            o.is_out = true;
            ctx.ops
                .append(&mut super::lower_ktir_to_superdsc::scalarfill_at(
                    f.name,
                    value,
                    &o,
                    ctx.sym_id_base,
                    ctx.layout,
                )?);
            lowered += 1;
            continue;
        }
        // ⭐⭐⭐⭐⭐ A STORE OF A VALUE THIS FUNCTION MINTED — the write-back copy. A result other
        // compute ops still read is not the store's to bind (the unbind at the `stored` find
        // states the law), so its producer minted it an intermediate above; the STORE lowers
        // HERE, at its own program position, as an identity copy from the mint to the store's
        // window — the same window resolution the compute store arm and the fill above take.
        // Order is the whole point: the copy is the ONLY write to the buffer, and it lands at
        // the position the store occupies (dxp program order is call order), so every consumer
        // before it read the bytes the program says they read. The value reads through reshape
        // hops exactly as the `stored` find does — the store's operand may be spelled through a
        // lens — and a store whose value is neither scattered, filled, minted, nor bound to its
        // producer below keeps the walk's own named refusal rather than a silent unwritten
        // window.
        if op.op_type == OpKind::KtdpStore
            && let Some(v) = op.operands.first().copied()
        {
            let mut cur = v;
            while let Some(&next) = reshape_hops.get(&cur) {
                cur = next;
            }
            if let Some(src) = inter.get(&cur).copied() {
                let out_tile = op.operands.get(1).copied();
                let tile = f
                    .operations
                    .iter()
                    .find(|o| o.op_type == OpKind::KtdpConstructAccessTile && o.result == out_tile);
                let Some(tile) = tile else {
                    return err(format!(
                        "{}: a store of a minted value names no access tile — nothing states \
                         the window it writes back to",
                        f.name
                    ));
                };
                let Some(mut o) = region_of_view_operand(k, tile, 0)? else {
                    return err(format!(
                        "{}: a store of a minted value names no parameter — a write-back with \
                         no bound buffer has no window to write",
                        f.name
                    ));
                };
                o = windowed_region(k, o, tile)?;
                o.is_out = true;
                ctx.ops
                    .append(&mut super::lower_ktir_to_superdsc::mintcopy_at(
                        f.name,
                        &src,
                        &o,
                        ctx.sym_id_base,
                        ctx.layout,
                    )?);
                lowered += 1;
                continue;
            }
        }
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
                // A `tensor.reshape` result: no descriptor, no minted buffer — the consumer
                // reads through it to the value underneath ([`reshape_forwards`]).
                || reshape_consumed.contains(&r)
                // An fp8 dot's absorbed scale mulf: it rode into the dot's emission above.
                || fp8_epilogue.contains(&r)
                // A row-folded family's non-leader block: the leader's one whole-view descriptor
                // already computed its rows (see [`row_tiled_family`]).
                || row_fold_skip.contains(&r)
        }) {
            continue;
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
        // refusal below and is named there. The AXIS FACTS ride beside the kind (wall 12): read off
        // the op's own `Dimensions`/`Shape` attrs, fail-closed on a statement the door cannot
        // classify — a missing or oversized one is a producer defect to name, not a frame fact to
        // guess.
        let reduce = match (op.op_type == OpKind::LinalgReduce, reduce_kind_of(op)) {
            (true, Some(kind)) => Some(Lowering::Reduce {
                kind,
                axes: reduce_axes_of(f.name, op)?,
            }),
            _ => None,
        };
        // `arith.mulf` by a SPLATTED CONSTANT is a scalar multiply, not an `Elementwise(Mul)` — the
        // tiebreak `Program`'s own doc names. Read per-op, because one decoder layer states four
        // different multipliers. Checked BEFORE the 1:1 map, which would otherwise route it to
        // `Elementwise(Mul)` and then fail on the splat operand having no `Region`.
        let scalar = (op.op_type == OpKind::ArithMulf)
            .then(|| splat_scale_of(f, op))
            .flatten();
        // ⭐ THE EXTREMUM'S OWN SCALAR TIEBREAK — `maximum(x, C)`/`minimum(x, C)` with `C` a splatted
        // constant. The same discriminator the mulf arm applies, on the two ops whose identity lives
        // at the range's edge: the online softmax's running-max initial (−65504, the f16 floor),
        // any clamp against a stated bound. Checked before the 1:1 map, which would otherwise
        // route it to `Elementwise(Maximum/Minimum)` — a two-TENSOR op — and refuse the splat
        // operand at the input collection, because a splat states no `Region`. The MoE kernel's
        // fp8 clamps carry both bounds as f16 TILES and keep the two-tensor arm; only the stated
        // constant takes this one.
        let extremum = matches!(op.op_type, OpKind::ArithMaxnumf | OpKind::ArithMinnumf)
            .then(|| splat_scale_of(f, op))
            .flatten();
        // ⭐ THE NON-COMMUTATIVE MEMBER of the same class: `arith.subf` against a splatted
        // constant. Same discriminator, and the fact the others could drop is the one this op
        // cannot: WHICH input the constant occupies, since `C − tile` and `tile − C` are
        // different functions. Checked before the 1:1 map for the same reason as the others —
        // `Elementwise(Sub)` would refuse the splat at the input collection.
        let scalar_sub = (op.op_type == OpKind::ArithSubf)
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
        // ⭐ WALL 13's ISLAND — a dtype CAST is a FIRST-CLASS CONVERT, read here off the op's own
        // operand/result types (the (from, to) pair the vendor's `dl16tofp32`/`fp32todl16` binds
        // state). Checked beside the reduce, before the 1:1 map — `program_of` has no cast arm
        // because a cast is not an elementwise kind; without this arm every `arith.extf`/
        // `arith.truncf` in an f32 island fell to the refusal below and named itself as needing a
        // `Program` variant, which is exactly wrong: the convert door already exists.
        let cast = cast_of(f, op)?;
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
            .or(reduce)
            .or(cast)
            .or(scalar.map(|sf| Lowering::ScalarMul(sf.value)))
            .or(extremum.map(|sf| Lowering::ScalarExtremum {
                value: sf.value,
                max: op.op_type == OpKind::ArithMaxnumf,
            }))
            .or(scalar_sub.map(|sf| Lowering::ScalarSub {
                value: sf.value,
                const_first: sf.const_first,
            }))
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
            // The silu's two inputs are the chain's gate and up, which the recogniser proved. (A
            // bare silu reaches the `Node(Elementwise(Silu))` arm, whose arity `elementwise_op_func`
            // states as 1.)
            Lowering::Silu => 2,
            // `linalg.reduce`'s operands are `(data, init)`: the init is the SEED, which the reduce
            // op-func carries itself, so only the data is an input.
            Lowering::Reduce { .. } => 1,
            // A dtype convert reads ONE tensor; the format pair rides the op (`Lowering::Convert`).
            Lowering::Convert { .. } => 1,
            // The splat is not an operand of the descriptor — it rides in the op as a bound `[1,1]`
            // const — so a scalar multiply reads ONE tensor. The extremum's scalar form is the
            // same: the clamp's bound rides the op, the tile is the one operand. And so is the
            // subtraction's: whichever SIDE the constant was stated on, the tile is the one
            // tensor the descriptor reads.
            Lowering::ScalarMul(_)
            | Lowering::ScalarExtremum { .. }
            | Lowering::ScalarSub { .. }
            | Lowering::BoundScalarMul(_) => 1,
            // x and gamma; the epsilon, `1/cols` and the oneplus offset are the fused body's own.
            Lowering::RmsNorm(..) => 2,
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
        let in_values: Vec<Ssa> = match (terminal, scalar, bound_scale, extremum, scalar_sub) {
            // The two tensors the rmsnorm recogniser proved, walked back through the rank plumbing to
            // values a `Region` exists for.
            _ if rms.is_some() => {
                let c = rms.expect("just matched");
                vec![c.x, c.gamma]
            }
            (Some(c), _, _, _, _) => match c.up {
                Some(up) => vec![c.gate, up],
                // The bare form's one input is the gate; the divide's other operand is the
                // chain-interior `1 + e^-x`, which the fused op never materialises.
                None => vec![c.gate],
            },
            // The TENSOR operand the recogniser proved, not operand 0 — the splat sits on either side
            // (`ms * INV_D` has it second, and nothing obliges a producer to put it there).
            (None, Some(sf), _, _, _) => vec![sf.tensor],
            // RUNG 3: same rule, the splat's other side. The bound scale is a `[1,1]` parameter, not
            // an operand of the descriptor.
            (None, None, Some(b), _, _) => vec![b.tensor],
            // The extremum's scalar form: the TENSOR operand, the clamp's other side — the same
            // rule the mulf arm states.
            (None, None, None, Some(sf), _) => vec![sf.tensor],
            // The subtraction's: same rule again — the constant rides the op whatever side it was
            // stated on, and the tile is the one operand.
            (None, None, None, None, Some(sf)) => vec![sf.tensor],
            (None, None, None, None, None) => op.operands.iter().copied().take(n_in).collect(),
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
        // ⭐⭐⭐ AND THROUGH THE RESHAPES — the last rewrite before any consumer-side lookup, so
        // every join downstream (the region loop, the `inter` mint lookup, the gather identity
        // join, the row fold) names the value UNDERNEATH the lens and not the lens. A reshape
        // result is never a load, never a minted intermediate, and never a gathered load's
        // identity — without this rewrite every one of those lookups fails on it; with it, the
        // region/gather/mint of the SOURCE answers, and the consumer's own extent guards are
        // what say whether the split it reads through is one its math can use.
        let pre_through: Vec<Ssa> = in_values.clone();
        let in_values: Vec<Ssa> = in_values.iter().map(|v| through(*v)).collect();
        // ⭐⭐⭐ AND THE LENS IS LAW-CHECKED AT THE CONSUMER — every hop the consumer's own
        // operand spelling reads through must be flat-ADMITTED (the #741 re-row law the forward
        // ran), unless this op is the ONE consumer that states its own affine access over the
        // source: a matmul whose B is the gathered load, whose presentation is the
        // `Access::gathered` mint over the ultimate source — a DIFFERENT set of byte laws (the
        // total bound, the gather's own), stated where that access is minted. Anything else
        // reading a refused reshape takes the target's framing on the source's bytes — the
        // clean-bake wrong-byte class the re-row law exists to refuse, so it is a build error
        // here, naming the law's own clause.
        let states_own_access =
            matches!(op.op_type, OpKind::LinalgMatmul | OpKind::LinalgBatchMatmul)
                && gathers
                    .iter()
                    .any(|g| Some(g.load) == in_values.get(1).copied());
        for (i, v) in pre_through.iter().enumerate() {
            let mut cur = *v;
            while let Some(&next) = reshape_hops.get(&cur) {
                if let Some(ReshapeFlat::Refused(msg)) = reshape_verdicts.get(&cur)
                    && !(states_own_access && i == 1)
                {
                    return err(format!(
                        "{}: operand {i} reads t{} through a reshape whose flat presentation \
                         the re-row law refuses — {msg}. A consumer may take that framing only \
                         by stating its own affine access over the source (the gathered \
                         matmul's B does); this op does not, so it would read the source's \
                         bytes at a framing that moves them",
                        f.name, cur.0
                    ));
                }
                cur = next;
            }
        }

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
                    | Lowering::ScalarExtremum { .. }
                    | Lowering::ScalarSub { .. }
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
        // ⭐⭐⭐ LAW (a) — A CONTRACTION MATERIALIZES **EVERY** GATHERED OPERAND. The join below
        // collects ALL of this op's gathered inputs, in input order. The single-gathered-input
        // admission this replaced took the FIRST match only, so a contraction whose A and B are
        // both gathered (the decode attention's dot: A gathers Q through its repetition table,
        // B gathers the pool's blocks) materialized one side and left the other reading the
        // walk's raw gathered load — right shape, right dtype, wrong rows, from a clean bake.
        // Each gathered input now takes its own materialization below; an input that cannot
        // materialize is refused by name at the door that used to admit only the first.
        let op_gathers: Vec<(Gather, usize, usize)> = in_values
            .iter()
            .enumerate()
            .filter_map(|(i, v)| gathered_loads.get(v).map(|&ti| (gathers[ti], ti, i)))
            .collect();
        for &(g, ti, _gi) in &op_gathers {
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
        }
        // ⛔⛔⛔ A GATHERED OPERAND MAY ONLY REACH A BODY THAT DECLARES THE INDEX. `ScalarMul` is the
        // one — `scalarmul_at` takes the [`Gather`] and routes to
        // `assemble_pointwise_broadcast_gather`. Every other arm of [`emit_one`] would emit an
        // ordinary descriptor over the table, i.e. read its first `rows` rows DIRECTLY and ignore the
        // ids: right shape, right dtype, right distribution, wrong rows, and nothing downstream
        // compares the two. So it is refused by name rather than lowered as the nearest thing.
        //
        // ⭐⭐⭐ THE ONE EXCEPTION — RUNG 4: A CONTRACTION WHOSE OPERAND IS THE GATHERED TABLE. The
        // materialization is synthesized HERE, before the refusal can fire, because the vendor's own
        // gathered fixtures and IBM's paged attention both spell a gathered contraction exactly one
        // way and this is it: a KERNEL-less gathered copy writing a minted intermediate, the matmul
        // reading the intermediate as a plain activation. See [`gathered_matmul_materializes`], which
        // states the whole argument and does the work.
        // ⭐⭐⭐⭐⭐ AND THE BATCHED FORM IS ADMITTED AT THE SAME DOOR — a `linalg.batch_matmul`
        // whose B is the gathered table read through its reshape states its geometry in its OWN
        // operand types, and the materializer's batch branch writes the C-order copy the
        // per-batch bmm reads (see [`emit::batched_gathered`]). A rank-2 gathered dot never
        // enters it: it keeps rung 4's card-proven `[k, n]` path, byte for byte.
        // ⭐⭐⭐⭐⭐ AND LAW (a) MAKES IT **EVERY** GATHERED OPERAND, NOT ONLY B — the loop below
        // materializes each gathered input in turn, so a dot whose A and B are both gathered
        // (the decode attention's) serves both sides through the same door instead of leaving
        // the second reading the walk's raw gathered load. The frame each side takes is stated
        // by the operand's own geometry, never by its position: see the loop's doc.
        let scalar_body = matches!(
            program,
            Lowering::ScalarMul(_)
                | Lowering::ScalarExtremum { .. }
                | Lowering::ScalarSub { .. }
                | Lowering::BoundScalarMul(_)
        );
        let mut gathered_forms: Vec<(usize, GatheredB)> = Vec::new();
        if !scalar_body {
            for &(g, _ti, gi) in &op_gathers {
                gathered_forms.push((
                    gi,
                    gathered_matmul_materializes(
                        k,
                        GatherEmit {
                            ctx: &mut ctx,
                            floor,
                            inter: &mut inter,
                        },
                        g,
                        gi,
                        &in_values,
                        GatherConsumer {
                            op,
                            pre_through: &pre_through,
                        },
                    )?,
                ));
            }
        }
        let gathered_b = gathered_forms
            .iter()
            .any(|(_, f)| !matches!(f, GatheredB::No));
        let batched_geo = gathered_forms.iter().find_map(|(_, f)| match f {
            GatheredB::Batched(bg) => Some(*bg),
            _ => None,
        });
        // The FIRST gathered input that did not materialize, by input position — the refusal
        // names it, so a program with two gathered operands is told which one failed and not
        // only that "the op reads a tile".
        let unmaterialized = op_gathers.iter().find_map(|&(g, _ti, gi)| {
            let materialized = gathered_forms
                .iter()
                .any(|&(fgi, ref f)| fgi == gi && !matches!(f, GatheredB::No));
            (!materialized).then_some((g, gi))
        });
        if !scalar_body && let Some((g, gi)) = unmaterialized {
            return err(format!(
                "{}: `{:?}` reads a `ktdp.construct_indirect_access_tile` at input {gi} (t{} \
                 gathered through t{}), and this door declares an index operand only for the \
                 constant-operand pointwise bodies (`Program::ScalarMul`, the extremum's and the \
                 subtraction's own scalar forms) — `scalarmul_scaled` is the one body that carries a \
                 `Gather`. Every other assembler here would emit a descriptor reading the table \
                 DIRECTLY at its base, which is the right shape and the wrong rows, from a clean \
                 bake. Give the gathered operand its own `ScalarMul` node, or add the index to the \
                 assembler this op needs.",
                f.name,
                op.op_type,
                g.source.tid(),
                g.index_tid,
            ));
        }

        // ⭐⭐⭐ RUNG 4'S SECOND HALF — THE MATMUL READS THE INTERMEDIATE, NOT THE TABLE. The copy was
        // emitted above and the region recorded under the GATHERED VALUE; this swaps each gathered
        // operand's `per_op` slot for it (LAW (a): every one of them, not only the first), so the
        // matmul's own assembler sees plain activations. The index regions appended by the gather
        // join are DROPPED from the list at the same time (truncate to `n_in`): the matmul's
        // descriptor carries no index operand (the copies declared them), and
        // `matmul_oriented`'s `ins.len() != 2` arity check would read a longer list as the fp8
        // W8A8 form and refuse it.
        if gathered_b {
            for &(gi, ref form) in &gathered_forms {
                if matches!(form, GatheredB::No) {
                    continue;
                }
                let b_value = in_values[gi];
                let Some(r) = inter.get(&b_value) else {
                    return err(format!(
                        "{}: the gathered operand at input {gi} has no materialized intermediate \
                         in the walk's own map — the copy was emitted but its region was not \
                         recorded",
                        f.name
                    ));
                };
                let mut r = *r;
                r.is_out = false;
                per_op[gi] = r;
            }
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

        // THIS OP'S OUTPUT: the parameter a `ktdp.store` writes from this op's result. ⭐⭐⭐ THE
        // STORE'S VALUE READS THROUGH RESHAPES — the producer's own spelling ends
        // `o_desc.store([0, 0], tl.reshape(o, [rows, cols]))`, so the store's operand is the
        // LENS, not the value: the find resolves it through [`reshape_forwards`] to the op's
        // result, exactly as the input-side rewrites above do, and a store whose value no
        // through-reshape resolution matches keeps the not-stored arm below (mint + named
        // refusal if nothing consumes it) rather than a silent unwritten output.
        let mut stored = f.operations.iter().find(|s| {
            s.op_type == OpKind::KtdpStore
                && s.operands
                    .first()
                    .copied()
                    .is_some_and(|v| Some(through(v)) == op.result)
        });
        // ⭐⭐⭐ AND THE STORE IS A FLAT CONSUMER OF THE SAME LAW — it writes the value's whole
        // flat content at its window, so the chain it reads through must be flat-ADMITTED like
        // every other consumer's: a store whose value arrives through a REFUSED reshape would
        // take the target's framing on the source's bytes, and the window's rows are the
        // producer's own statement. There is no store-side exemption — the one access-stating
        // consumer is the gathered matmul's B.
        if let Some(s) = stored
            && let Some(v) = s.operands.first().copied()
        {
            let mut cur = v;
            while let Some(&next) = reshape_hops.get(&cur) {
                if let Some(ReshapeFlat::Refused(msg)) = reshape_verdicts.get(&cur) {
                    return err(format!(
                        "{}: the `ktdp.store` of t{} reads t{} through a reshape whose flat \
                         presentation the re-row law refuses — {msg}. The store writes the \
                         value's whole flat content at its window, so a framing that moves \
                         bytes is not one it can write",
                        f.name, v.0, cur.0
                    ));
                }
                cur = next;
            }
        }
        // ⭐⭐⭐⭐⭐ AND NOT WHEN THE RESULT IS STILL READ. The arm below emits this op writing the
        // store's window IN PLACE at this op's own position — honest only while the result is the
        // store's alone. A result other compute ops still read (the fold body's running max:
        // stored at the trip's end, read by `m − m_new` and `qk − m_new[.., .., None]` on the way
        // there) would have its old value overwritten before those consumers read it — a
        // descriptor reads at ITS OWN program position, never at the load's. Such a result is
        // minted as the intermediate it already is to those consumers, and the store itself
        // lowers at its own position as the write-back copy the `mintcopy` arm at the top of the
        // walk states. This runs BEFORE the fp8 epilogue arm so that absorption (whose absorbed
        // `arith.mulf` has exactly the one dot consumer) is untouched by it.
        if let Some(res) = op.result
            && stored.is_some()
            && result_consumed_beyond_store(f, res)
        {
            stored = None;
        }
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
            // ⭐⭐⭐ WALL 13 — THE INTERMEDIATE MINTS AT THE VALUE'S OWN `df`. The result type's
            // element dtype is the one statement of the value's format, so the synth reservation
            // prices an f32 intermediate at 4 B/elem (`synth_df`, the island's footprint law) and
            // the region carries the same `df` every door this value reaches classifies at. Every
            // f16 intermediate is byte-identical (`synth` IS `synth_df(Df::Fp16)`), and a dtype
            // with no stick geometry (bool, i64) keeps the fp16 basis — inert for the DDL families
            // this value can reach, the same honest fallback `regions()` states for a parameter.
            let df = op
                .result_type
                .and_then(|t| match t {
                    ktir_core::irtype::IrType::Tensor { elem, .. }
                    | ktir_core::irtype::IrType::MemRef { elem, .. } => Some(elem),
                    ktir_core::irtype::IrType::Scalar(d) => Some(d),
                    ktir_core::irtype::IrType::AccessTile { .. }
                    | ktir_core::irtype::IrType::Index => None,
                })
                .and_then(df_of_dtype)
                .unwrap_or(Df::Fp16);
            let dims = match op.result_type {
                // ⭐⭐⭐⭐⭐ A RANK-2 RESULT THAT RESPELLS A REDUCE-FAMILY VECTOR IS MINTED IN THAT
                // FAMILY's FRAME — `[rows, 1]`, the accumulator's own — not in its stated spelling.
                // The placement and the region every consumer reads are minted HERE, before the
                // door runs, so the frame is decided at the one site that owns both (`layout.synth`
                // below, `inter.insert` below) — a door-side respell could not reach the consumers
                // and would leave the placement at the wrong footprint. [`respelled_frame_rows`]
                // is the law and carries the wall-11 measurement; `None` falls through to the
                // verbatim arm below, so every non-family shape is byte-identical to before.
                Some(ktir_core::irtype::IrType::Tensor { dims, .. })
                    if dims.len() == 2
                        && let Some(rows) = respelled_frame_rows(
                            &per_op,
                            dims[0] as u64 * dims[1] as u64,
                            (matches!(program, Lowering::Reduce { .. }))
                                .then(|| per_op.iter().find(|r| !r.is_out).map(|r| r.r_len))
                                .flatten(),
                        ) =>
                {
                    [rows, 1]
                }
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
                        && (matches!(program, Lowering::Reduce { .. })
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
            // core past the first. The band is read off the DENSE inputs the same way
            // [`pointwise_rows`] reads the sweep off the output — a WINDOWED dense input states
            // it as its `r_len` — while a function whose inputs are all whole-view keeps the
            // result type's own rows, so every mint this door has ever made is byte-identical.
            // (A chain whose every input is a minted band intermediate states no windowed input;
            // its mint stays at the tile and the extents law refuses the sweep mismatch by name —
            // refused at the op that could have extended it, not silently mis-sized.) ⛔ AND A
            // BROADCAST OPERAND DOES NOT STATE THE BAND even when its region is windowed: its
            // mode addresses it at the mode's own extent (`Col` a lane per row, `Mb` one row,
            // `Y` one spray at a per-trip window), so its `r_len` is the source's slice height,
            // never the sweep — [`band_rows_of`]'s own law, with the wall-18 measurement.
            let dims = [band_rows_of(&per_op, &bcast_axes, dims[0]), dims[1]];
            let Some(layout) = ctx.layout else {
                return err(format!(
                    "{}: `{:?}`'s result is an intermediate, which needs a `BundleLayout` to hold \
                     its buffer -- `layout: None` is the unit-test arm and cannot place one",
                    f.name, op.op_type
                ));
            };
            let tid = mint_inter(k, layout, floor, res);
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
                // ⛔ NOT THE BATCHED DOT: its per-stick cut addresses the output at the LOGICAL
                // `n` (one stick column block at a time), so the reservation IS the logical dims
                // — `for_matmul`'s util-floor bump is the ORIENTED emitter's own law and rides
                // only the arm that addresses at the bumped width.
                Lowering::Node(Program::Matmul) if batched_geo.is_none() => {
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
                _ => dims,
            };
            layout.synth_df(crate::place::PlaceId::Act(tid), &dev_dims, df);
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
                // The value's OWN element type (wall 13): an f32 intermediate is reserved and
                // classified at IEEE_FP32; every f16 one keeps the fp16 mint byte-identical.
                df,
            };
            inter.insert(res, r);
            per_op.push(r);
            let name = f.name;
            let mut emitted = emit_one(
                name,
                program,
                &per_op,
                ctx.sym_id_base,
                Some(layout),
                headmajor,
                WalkProof {
                    b_orient,
                    gather: op_gathers.first().map(|&(g, _, _)| g),
                    bcast: &bcast_axes,
                    batched: batched_geo,
                },
            )?;
            ctx.ops.append(&mut emitted);
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

        let mut emitted = emit_one(
            f.name,
            program,
            &per_op,
            ctx.sym_id_base,
            ctx.layout,
            headmajor,
            WalkProof {
                b_orient,
                gather: op_gathers.first().map(|&(g, _, _)| g),
                bcast: &bcast_axes,
                batched: batched_geo,
            },
        )?;
        ctx.ops.append(&mut emitted);
        lowered += 1;
    }

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
        return super::lower_ktir_to_superdsc::lmlast(f.name, k, &r, ctx.sym_id_base, ctx.layout);
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
                f.name,
                ti,
                g.source.tid(),
                g.index_tid,
            ));
        }
    }
    Ok(out)
}

/// ⭐ ONE INDEXED WRITE — [`super::assemble_indexed_write`] over the scatter's own facts. The
/// stored rows must be a LOAD of a parameter (a value this walk has a region for at a whole-tensor
/// corner); the table and the index are the tile's own parameters.
fn lower_indexed_write(
    k: &KtirNode,
    sc: &super::lower_ktir_to_superdsc::Scatter,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<super::EmittedOp, Error> {
    let f = &k.func;
    let Some(src) = region_for_operand(k, sc.value)? else {
        return err(format!(
            "{}: the indexed store's value is not a load of a parameter — an indexed write copies \
             rows a parameter holds",
            f.name
        ));
    };
    // ⛔ THE SOURCE IS READ FROM ITS BASE, so a row window elsewhere is refused by name rather
    // than copied from row 0.
    if src.r_start != 0 || src.c_start != 0 {
        return err(format!(
            "{}: the indexed store's rows start at ({}, {}) of t{} — this door copies a source read \
             from its base",
            f.name, src.r_start, src.c_start, src.tid
        ));
    }
    if src.r_len != sc.entries {
        return err(format!(
            "{}: the indexed store writes {} rows but its tile states {} entries",
            f.name, src.r_len, sc.entries
        ));
    }
    let all = regions(k)?;
    let region_of = |tid: u32| -> Result<Region, Error> {
        k.bindings
            .iter()
            .position(|b| b.get() == tid)
            .and_then(|i| all.get(i).copied())
            .ok_or_else(|| Error {
                message: format!("{}: t{tid} is not a bound parameter", f.name),
            })
    };
    let table = region_of(sc.value_tid)?;
    let _ = region_of(sc.index_tid)?;
    if table.v_cols != src.c_len {
        return err(format!(
            "{}: the indexed store writes {}-wide rows into a {}-wide table",
            f.name, src.c_len, table.v_cols
        ));
    }
    let run = crate::superdsc_opspec::StickRun::covering(sc.first_entry.entries(), sc.entries)
        .map_err(|m| Error {
            message: format!("{}: {m}", f.name),
        })?;
    let name = format!("{}_scatter_t{}", f.name, sc.value_tid);
    super::assemble_indexed_write(
        super::IndexedWrite {
            op_name: &name,
            rows: sc.entries,
            cols: src.c_len,
            src: &src.name(),
            table: &table.name(),
            table_rows: table.v_rows,
            index_name: &crate::place::act_name(sc.index_tid),
            run,
        },
        sym_id_base,
        layout,
    )
    .map_err(Error::from)
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
    /// ⭐ THE BATCHED GATHERED DOT'S GEOMETRY — `Some` for exactly a `linalg.batch_matmul` whose
    /// B is a gathered table (its B in `per_op` is then the C-order copy the materializer wrote,
    /// not a `[k, n]` window). `None` everywhere else, and then the matmul arm is byte-identical
    /// to what it was: a rank-2 gathered dot keeps rung 4's own frame.
    batched: Option<super::batched_gathered::BatchedGathered>,
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
        batched,
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
        Lowering::Reduce { kind, axes } => {
            return super::lower_ktir_to_superdsc::reduce(
                name,
                kind,
                axes,
                per_op,
                sym_id_base,
                layout,
            );
        }
        // ⭐ WALL 13's ISLAND — a dtype CAST as a first-class convert. `per_op` is `[data, out]`,
        // and the door checks the regions' own dfs against the (from, to) pair the op stated —
        // the mixed-df pin's convert-side seat (an elementwise op is the other).
        Lowering::Convert { from, to } => {
            return super::lower_ktir_to_superdsc::convert(
                name,
                from,
                to,
                per_op,
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
        // `maximum(tile, C)` / `minimum(tile, C)` — the extremum's scalar form, through the ONE
        // shared body the mulf arm reaches (`scalarmul_scaled` with the op's own `opFuncName`),
        // so the two tiebreaks cannot drift about what a constant-operand pointwise emits.
        // The program's own gather, threaded for the same reason the mulf arm threads it.
        Lowering::ScalarExtremum { value, max } => {
            return super::lower_ktir_to_superdsc::scalarextremum_at(
                name,
                value,
                max,
                gather,
                per_op,
                sym_id_base,
                layout,
            );
        }
        // `tile − C` / `C − tile` — the non-commutative member, through the same ONE shared body,
        // with the ORDER the program stated. The body derives its operand layout from that fact
        // (and refuses the one gathered shape no order can serve), so the two sides of a
        // subtraction are distinguishable all the way into the descriptor.
        Lowering::ScalarSub { value, const_first } => {
            return super::lower_ktir_to_superdsc::scalarsub_at(
                name,
                value,
                const_first,
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
        Lowering::Node(p) => p,
    };
    Ok(match program {
        // ⭐⭐⭐ THE BATCHED GATHERED DOT — the entry axis as the batch axis. Its B in `per_op`
        // is the C-order copy the materializer wrote, NOT a `[k, n]` window, so the oriented
        // matmul's own frame (and its B-orientation proof) does not apply: the per-batch bmm
        // states its own. `None` everywhere else — the arm below is byte-identical to what it
        // was.
        Program::Matmul if let Some(bg) = batched => {
            super::batched_gathered::emit_dot(name, &bg, per_op, sym_id_base, layout)?
        }
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
/// proved — B's region `tid == g.source.tid()`):
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
/// The emission context and the walk's own scratch travel in [`GatherEmit`] rather than as more
/// arguments: the ONE emission context (`super::EmitCtx` — sink, counter, placement authority)
/// by re-borrow from the door that minted it, beside the two facts only this walk states.
/// `floor` is [`lower_function`]'s node-derived tid bound, carried so this mint honors it like
/// the walk's own does; `inter` is the same walk's intermediates.
struct GatherEmit<'a, 'b> {
    ctx: &'b mut super::EmitCtx<'a>,
    floor: u32,
    inter: &'b mut std::collections::HashMap<Ssa, Region>,
}

/// THE CONSUMER'S OWN ROW IN THE WALK — the two facts the batched form's admission reads that
/// rung 4's 2-D path ignores: the op itself (its kind names the batched form) and its PRE-LENS
/// operand spellings (A's and B's stated types, through their reshapes). Carried as one value so
/// the materializer's parameter list stays the size its siblings' are.
struct GatherConsumer<'a> {
    op: &'a Operation<'a>,
    pre_through: &'a [Ssa],
}

/// What [`gathered_matmul_materializes`] did for one op: nothing (the walk's fail-closed refusal
/// owns the op), rung 4's 2-D `[k, n]` materialization, or the batched C-order copy together with
/// the geometry the program stated for it.
enum GatheredB {
    No,
    TwoD,
    Batched(super::batched_gathered::BatchedGathered),
}

fn gathered_matmul_materializes(
    k: &KtirNode,
    emit: GatherEmit<'_, '_>,
    g: Gather,
    gathered_input: usize,
    in_values: &[Ssa],
    consumer: GatherConsumer<'_>,
) -> Result<GatheredB, Error> {
    let GatherEmit { ctx, floor, inter } = emit;
    let f = &k.func;
    // ⛔ THE GATHERED OPERAND MUST BE THE OP'S OWN GATHERED INPUT, proven by identity — the
    // input at `gathered_input` IS the value this tile's load produced. A TID comparison cannot
    // prove this — a plain load of the same table resolves to the same tid, and a program whose
    // gather feeds input 0 while input 1 reads the table directly would otherwise materialize an
    // intermediate for an operand that is not the gathered value (and then fail looking it up in
    // `inter`, naming the wrong cause). A transposed K leg (input 1 through a `tt.trans` —
    // refused at the Triton door with the arch reason) never matches; the walk's own refusal
    // names it.
    // ⭐ LAW (a): WHICH SIDE OF THE CONTRACTION THE GATHER FEEDS IS STATED BY THE OPERAND'S OWN
    // GEOMETRY, NEVER BY AN INPUT-POSITION RULE. A rank-3 B read through its reshape lens
    // states the batched form's `[batch, k, n]` frame and takes the batch branch below; every
    // other shape — a contraction's A above all, rung 4's rank-2 B — takes the raw 2-D
    // `[entries, width]` identity copy this function's tail mints, the flat frame a dot's
    // corner-0 A sweep reads. The single-gathered-input admission this replaces (B only, first
    // match only) is gone, not branched beside: no "if input 1" special case survives here, and
    // the caller materializes EVERY gathered input through this one door.
    // ⭐ ANY CONSUMER OF A GATHERED LOAD READS THE MATERIALIZED ROWS — the copy is the gather, and
    // what reads its result (a contraction's B, a transpose, an elementwise op) reads an ordinary
    // activation. The one exception keeps its own body: `ScalarMul` carries the index itself.
    let Some(b_value) = in_values.get(gathered_input).copied() else {
        return Ok(GatheredB::No);
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
        return Ok(GatheredB::No);
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
    let Some(layout) = ctx.layout else {
        return err(format!(
            "{}: a gathered matmul's B needs a `BundleLayout` to hold its materialized \
             intermediate -- `layout: None` is the unit-test arm and cannot place one",
            f.name
        ));
    };
    // ⭐⭐⭐⭐⭐ THE BATCHED FORM — the gather's ENTRY AXIS AS THE DOT'S BATCH AXIS. The seals above
    // (B is the gathered load by identity, one consumer, a layout) are the same ones the 2-D frame
    // runs; only the intermediate's frame and the copy that writes it differ. See
    // [`super::batched_gathered`] for the frame law and its byte gate — the entry-size authority
    // is the gathered load's own row width, checked against the reshape's `[batch, k, n]` by
    // [`batched_geometry_of`] at one door.
    let GatherConsumer { op, pre_through } = consumer;
    if let Some(bg) = batched_geometry_of(f, op, pre_through, gathered_input, g)? {
        // ── THE GATHERED SOURCE'S WINDOW, ONE LAW (b) WITH THE POINTWISE FAMILY MINT
        // ([`BundleLayout::gathered_source_rows`], through [`super::pooled_reach_of`]'s
        // unit-checked reach lookup): a sweep that fits the placement keeps its window; an
        // over-sweep REQUIRES a declared reach — the typed bound on the index values — and is
        // refused, fail-closed, without one; with one, the family is read from the placement
        // (a placement holding the whole reach is a source extent, `min(swept, ceiling)`; a
        // slice is a pool, one entry's direct coverage). The values' bound is the reach's
        // ceiling, enforced where the values are born — the staging family's law, not this
        // door's.
        let table_name = crate::place::act_name(g.source.tid());
        let index_name = crate::place::act_name(g.index_tid);
        let what = format!(
            "{}: the batched gathered dot's copy of '{table_name}'",
            f.name
        );
        let src_window = |rows: u32| {
            crate::sdsc_abstract::StickLayout::row_blocked_df(
                rows as usize,
                bg.entry_elems() as usize,
                crate::superdsc_opspec::Df::Fp16,
            )
        };
        let reach = super::pooled_reach_of(&what, Some(layout), &table_name, bg.entry_elems())
            .map_err(Error::from)?;
        let rows = layout
            .gathered_source_rows(
                &table_name,
                bg.entries,
                bg.entry_elems(),
                crate::superdsc_opspec::Df::Fp16,
                reach,
            )
            .map_err(|message| Error { message })?;
        layout
            .check_window(
                &table_name,
                src_window(rows),
                0,
                rows,
                bg.entry_elems(),
                &what,
            )
            .map_err(|message| Error { message })?;
        // ── THE C-ORDER SCRATCH, minted the way the walk mints its own unstored results: one
        // stick wide, `copy_rows` sub-rows — the contiguous `[batch, k, n]` buffer the reshape
        // names (see the module doc for why the bmm's row-major walk and this buffer are one
        // fact). NOT rung 4's `[entries, head]`: that frame is stick-major at multi-stick width,
        // and reading it row-major moves every byte past the first stick column.
        let rows = bg.copy_rows();
        let stick = crate::work::FP16_ELEMS_PER_STICK;
        let tid = mint_inter(k, layout, floor, gathered_result);
        layout.synth(crate::place::PlaceId::Act(tid), &[rows, stick]);
        let r = Region {
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
            df: Df::Fp16,
        };
        let inter_name = r.name();
        let copies = super::batched_gathered::copy_opspecs(
            &bg,
            &table_name,
            &index_name,
            &inter_name,
            g.first_entry,
        )
        .map_err(|msg| Error {
            message: format!("{}: {msg}", f.name),
        })?;
        for (leg, cop) in copies.into_iter().enumerate() {
            // THE PLACEMENT HALF of the destination's two-door mint (`gather_copy_opspec` ran
            // the declared-shape half layout-free): the run's window against the synth this mint
            // declared. Both numbers read off the BUILT op — the leg's `mb` extent and the
            // output operand's own base — so the check cannot disagree with the emission about
            // which window the leg writes.
            let mut cop = cop;
            let leg_rows = cop.iter.extent("mb");
            let dest_base = cop.args.last().map_or(0, |a| a.view().offset_elems) as u64;
            super::mint_full_window(
                &what,
                &super::rb(
                    &inter_name,
                    RowCount::of_token_rows(rows),
                    BlockCols::of_one_stick(Lanes::FP16),
                ),
                dest_base,
                leg_rows,
                stick,
                Some(layout),
            )
            .map_err(|message| Error { message })?;
            super::apply_lx_residency(&mut cop, Some(layout));
            let folds = crate::superdsc_opspec::SdscFoldSet::new(cop.iter.cores_used());
            // 🅿️ PARKED — NO STAMP ON THE BATCHED DOOR'S COPY LEGS. These legs are the
            // INNER ops of a door-lowered carrier program; the addressing fact is stated at
            // the CARRIER's mint (`EmittedOp::paged_region` / `paged_fold`, the one doors)
            // and stamped by the walk onto every op it lowers from that carrier — so a door
            // stamp here would be a SECOND writer for the same fact on the same ops, which
            // the one-stamping-site law forbids. A pool-reach table in a NON-carrier program
            // (none exists in the tree's models today) is the parked general-door class; see
            // the parked note at the pointwise door's reach binding (`emit/mod.rs`).
            ctx.ops.push(
                super::emit_sdsc_tiled(
                    &format!("{inter_name}_g{leg}"),
                    &cop,
                    &folds,
                    ctx.sym_id_base,
                    Some(layout),
                )
                .map_err(Error::from)?,
            );
        }
        inter.insert(gathered_result, r);
        return Ok(GatheredB::Batched(bg));
    }
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
    // ⭐⭐⭐ LAW (a)'s CHECKED HALF — THE MINTED `[entries, width]` FRAME MUST FLAT-EQUAL THE
    // SWEEP THE CONSUMER ITSELF STATES FOR THIS OPERAND. The tile's `Shape` (the walk's
    // authority) and the consumer's own operand type (the program's) are two statements of the
    // same bytes; a disagreement means the copy writes a frame its consumer cannot address —
    // right count, wrong shape — so it is refused here, by name, with both numbers. Stated
    // generically over the consumer's OWN type, never over an input position: a batched dot's
    // A states `[batch, m, k]`, so this is exactly `batch·m·k == entries·k` with the width the
    // reduction's own; rung 4's rank-2 B states `[k, n]`; an elementwise consumer states the
    // swept `[rows, cols]` itself and the check holds trivially.
    {
        let dims = type_of(f, op.operands[gathered_input])
            .and_then(|t| t.dims().map(|d| d.to_vec()))
            .ok_or_else(|| Error {
                message: format!(
                    "{}: the gathered operand at input {gathered_input} of `{:?}` states no tensor \
                     type, so the sweep its materialized copy must serve is unknown",
                    f.name, op.op_type,
                ),
            })?;
        let flat: i64 = dims.iter().product();
        if flat != i64::from(rows) * i64::from(head)
            || dims.last().copied() != Some(i64::from(head))
        {
            return err(format!(
                "{}: the gathered operand at input {gathered_input} of `{:?}` states a sweep of \
                 {dims:?} ({flat} elements, innermost {}), but its tile gathers [{rows}, {head}] \
                 ({} elements). The materialized copy would write a frame the consumer cannot \
                 address — the two authorities must state the same bytes. Fix the tile's shape or \
                 the consumer's operand type; they disagree here.",
                f.name,
                op.op_type,
                dims.last().copied().unwrap_or(0),
                i64::from(rows) * i64::from(head),
            ));
        }
    }
    let tid = mint_inter(k, layout, floor, gathered_result);
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
        df: Df::Fp16,
    };
    // ── (3) THE GATHERED IDENTITY COPY — the `test_gather_1core` shape, through the one body
    // that declares an index operand. One input (the table), `identity`, the intermediate as
    // output; `assemble_pointwise_broadcast_gather` owns the one-index-stick ceiling and the
    // per-stick legs, so nothing here counts entries.
    let table_name = crate::place::act_name(g.source.tid());
    let index_name = crate::place::act_name(g.index_tid);
    let inter_name = r.name();
    // ⭐ TRUE SHAPES, not `rbo`'s `1×64` annotation: the table and the intermediate declare the
    // swept `[rows, head]` — the scalarmul arm's convention (the index selects rows within the
    // table, so the read window is the bound; the placement door still holds the window against
    // the table's real footprint). The intermediate's `synth` above states the same pair. The
    // builder reads name + kind only, so the dims are the mint's bound and nothing else.
    let x_h = super::rb(
        &table_name,
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(head),
    );
    let inputs = [super::In::full(&x_h).ew()];
    let tile_op = super::lower_ktir_to_superdsc::pointwise_tile_op(rows, head, 1, 2, Df::Fp16);
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
            o: &super::rb(
                &inter_name,
                RowCount::of_token_rows(rows),
                BlockCols::of_feature_cols(head),
            ),
        },
        ctx.sym_id_base,
        Some(layout),
    )
    .map_err(Error::from)?;
    ctx.ops.extend(copied);
    // ── (4) THE MATMUL'S B, REWRITTEN TO THE INTERMEDIATE — recorded under the GATHERED VALUE
    // so the walk's `inter` lookup hands the matmul the intermediate's region for that operand,
    // and the caller REPLACES `per_op`'s B slot with it before `emit_one` runs.
    inter.insert(gathered_result, r);
    Ok(GatheredB::TwoD)
}

/// ⭐ THE BATCHED FORM'S STATED GEOMETRY — read from the program's OWN statements, at one door,
/// before any copy leg or dot op is built.
///
/// The admission is: the consumer is a `linalg.batch_matmul` whose GATHERED operand is its **B**
/// (input 1 — A-side gathers are not this form, same rule as rung 4), and B's stated type THROUGH
/// its reshape lens is rank-3. Everything else is the geometry door's own arithmetic: A's type
/// `[batch, m, k]`, the reshape's `[batch, k, n]`, and the gathered load's own `[entries, row]`
/// pair — the row width being the ENTRY-SIZE AUTHORITY, cross-checked against the tile's own
/// entry count. A rank-2 B never enters (it is the 2-D frame's own form and keeps rung 4's path
/// below); a batched form whose numbers disagree refuses HERE, by name, rather than materializing
/// a copy whose bytes the dot cannot address.
fn batched_geometry_of(
    f: &IRFunction<'_>,
    op: &Operation<'_>,
    pre_through: &[Ssa],
    gathered_input: usize,
    g: Gather,
) -> Result<Option<super::batched_gathered::BatchedGathered>, Error> {
    if gathered_input != 1 || op.op_type != OpKind::LinalgBatchMatmul {
        return Ok(None);
    }
    let dims_of = |v: Option<&Ssa>, what: &str| -> Result<Vec<i64>, Error> {
        v.and_then(|v| type_of(f, *v))
            .and_then(|t| t.dims().map(|d| d.to_vec()))
            .ok_or_else(|| Error {
                message: format!(
                    "{}: the batched gathered dot's {what} states no tensor type, so its \
                     [batch, ·, ·] frame is unknown",
                    f.name
                ),
            })
    };
    // A rank-2 B is the 2-D frame's own form — the batched arm does not take it.
    let b = dims_of(pre_through.get(1), "B")?;
    if b.len() != 3 {
        return Ok(None);
    }
    let a = dims_of(pre_through.first(), "A")?;
    let load = dims_of(Some(&g.load), "gathered load")?;
    let [load_rows, entry_elems] = load[..] else {
        return err(format!(
            "{}: the gathered load t{} states {load:?}, not the [entries, row] pair a whole-row \
             gather fetches",
            f.name, g.load.0
        ));
    };
    if load_rows != g.entries as i64 {
        return err(format!(
            "{}: the gathered load states {load_rows} rows but its tile gathers {} entries",
            f.name, g.entries
        ));
    }
    let entry_elems = u32::try_from(entry_elems).map_err(|_| Error {
        message: format!(
            "{}: the gathered row width {entry_elems} is not a u32",
            f.name
        ),
    })?;
    super::batched_gathered::BatchedGathered::of(&a, &b, g.entries, entry_elems)
        .map(Some)
        .map_err(|msg| Error {
            message: format!("{}: {msg}", f.name),
        })
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

    /// ⛔⛔ THE RANK-3 MIDDLE AXIS — THE MASK ADD, `[256, 256] -> [256, 1, 256] -> [256, 4, 256]`
    /// with `Dimensions=[1]`, spelled `reshape -> collapse -> broadcast` with no expand at all.
    /// Row-major, spraying the middle axis repeats each of the source's flat rows down the
    /// merged row axis (`result[b, g, j] = source[b, 0, j]`: flat row `b·4 + g` reads flat row
    /// `b` — the row index ADVANCES every G rows), and no operand mode in this door's vocabulary
    /// addresses that. The refusal must NAME the shape and `attn.py`'s door-proven restatement —
    /// the host stages the mask at the score tile's own rows and the kernel adds it at base —
    /// because that restatement is the kernel's move, not a lowering fix.
    #[test]
    fn the_rank3_middle_axis_is_row_replication_and_refused() {
        let m = refusal3(
            &[256, 256],
            Relay::Reshape(&[256, 1, 256]),
            &[256, 256],
            &[256, 4, 256],
            &[1],
        );
        assert!(
            m.contains("MIDDLE axis"),
            "names the axis class, not just the index: {m}"
        );
        assert!(
            m.contains("ROW-REPLICATION"),
            "names the data movement the device cannot address: {m}"
        );
        assert!(
            m.contains("attn.py"),
            "names the door-proven restatement so the kernel side knows the move: {m}"
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
            BcastAxis::Y(4),
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
            df: Df::Fp16,
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
        let p = presented_flat(&reg(64, 4), BcastAxis::Y(4)).expect("the identity");
        assert_eq!((p.v_rows, p.v_cols), (64, 4));
    }
}

#[cfg(test)]
/// ⭐⭐⭐ THE RESHAPE FORWARDS' LAW, at the batched paged-decode kernel's own geometry — the
/// verdicts the consumer-side check reads, pinned so the #741 re-row law's application to
/// `tensor.reshape` (the `Access::reframe` instance) cannot drift silently.
mod reshape_forward_tests {
    use super::*;
    use ktir_core::arena::Arena;
    use ktir_core::dtypes::DType;
    use ktir_core::irtype::IrType;

    /// One straight-line function: the argument `Ssa(0)` of type `src_dims` reshaped to
    /// `target_dims`, the result `Ssa(1)`.
    fn reshaped(src_dims: &[i64], target_dims: &[i64]) -> &'static IRFunction<'static> {
        let a: &'static Arena = Arena::global();
        let f16 = |dims: &[i64]| IrType::Tensor {
            dims: a.ints(dims.to_vec()),
            elem: DType::F16,
        };
        let op = Operation {
            result: Some(Ssa(1)),
            op_type: OpKind::TensorReshape,
            operands: a.ssa(vec![Ssa(0)]),
            attributes: &[],
            result_type: Some(f16(target_dims)),
            regions: &[],
        };
        let f = IRFunction {
            name: "reshape_probe",
            arguments: a.args(vec![(Ssa(0), f16(src_dims))]),
            operations: a.ops(vec![op]),
            grid: (1, 1, 1),
            return_type: None,
        };
        Box::leak(Box::new(f))
    }

    /// ⭐ EVERY FLAT RESHAPE OF THE KERNEL IS THE IDENTITY at its flat presentation — q's
    /// `[1024, 64] -> [256, 4, 64]`, the mask's, the merge section's and the final store's —
    /// so the forward admits them and every consumer reads the bytes the source holds.
    #[test]
    fn the_kernels_flat_reshapes_are_admitted() {
        for (src, target) in [
            // q, pre-repeated in entry order, restated per [batch, gqa, hd].
            (&[1024, 64][..], &[256, 4, 64][..]),
            // the additive mask, staged at the score tile's own rows.
            (&[1024, 256][..], &[256, 4, 256][..]),
            // the merge section: the running max, l and acc restated [j, b·nkvh, gqa(, hd)].
            (&[256, 4][..], &[4, 64, 4][..]),
            (&[1024, 64][..], &[4, 64, 4, 64][..]),
            // the final store's [b·nkvh·gqa, hd].
            (&[64, 4, 64][..], &[256, 64][..]),
        ] {
            let (forwards, _hops, verdicts, consumed) = reshape_forwards(reshaped(src, target));
            assert_eq!(
                forwards.get(&Ssa(1)),
                Some(&Ssa(0)),
                "forwards to the source"
            );
            assert!(
                consumed.contains(&Ssa(1)),
                "the reshape result is plumbing the walk skips"
            );
            assert!(
                matches!(verdicts.get(&Ssa(1)), Some(ReshapeFlat::Admitted)),
                "{src:?} -> {target:?} is the identity at the flat presentation"
            );
        }
    }

    /// ⭐⭐⭐ THE GATHERED OPERANDS' RESHAPES ARE CARRIED, NOT FLAT — kt's
    /// `[32, 131072] -> [256, 64, 256]` and v's `[32, 131072] -> [256, 256, 64]` are flat
    /// `[16384, 256]` / `[65536, 64]` against a 32-row source, and the inter-stick stride is
    /// the ROW count, so the re-row law refuses them: only the gathered matmul's B (the
    /// `Access::gathered` mint) may read that framing. The refusal names the law's own clause.
    #[test]
    fn the_gathered_reshapes_are_carried_not_flat() {
        for (src, target) in [
            (&[32, 131072][..], &[256, 64, 256][..]),
            (&[32, 131072][..], &[256, 256, 64][..]),
        ] {
            let (forwards, hops, verdicts, _) = reshape_forwards(reshaped(src, target));
            assert_eq!(forwards.get(&Ssa(1)), Some(&Ssa(0)), "still forwards");
            assert_eq!(hops.get(&Ssa(1)), Some(&Ssa(0)), "one hop to the source");
            let Some(ReshapeFlat::Refused(msg)) = verdicts.get(&Ssa(1)) else {
                panic!("{src:?} -> {target:?} must be refused as a flat presentation");
            };
            assert!(
                msg.contains("stride BETWEEN stick columns"),
                "names the re-row law's own clause: {msg}"
            );
        }
    }

    /// ⛔ A FLAT FRAMING THAT MOVES BYTES — `[1024, 64] -> [64, 1024]`: same numel, but the
    /// rows (the inter-stick stride) and the stick-column count both change, so a row-framed
    /// consumer would read the source's frame at the target's extents. This is the clean-bake
    /// wrong-byte class the consumer-side check refuses.
    #[test]
    fn a_row_swapped_flat_target_is_refused() {
        let (_, _, verdicts, _) = reshape_forwards(reshaped(&[1024, 64], &[64, 1024]));
        assert!(
            matches!(verdicts.get(&Ssa(1)), Some(ReshapeFlat::Refused(_))),
            "a row swap is not a re-row"
        );
    }

    /// ⛔ AND A RESHAPE THAT STATES NO LAW IS NOT FORWARDED BLIND — no result type means the
    /// flat presentation cannot be stated, and the verdict refuses by name so the
    /// consumer-side check fails closed.
    #[test]
    fn an_untyped_reshape_is_refused_not_forwarded() {
        let a: &'static Arena = Arena::global();
        let op = Operation {
            result: Some(Ssa(1)),
            op_type: OpKind::TensorReshape,
            operands: a.ssa(vec![Ssa(0)]),
            attributes: &[],
            result_type: None,
            regions: &[],
        };
        let f = IRFunction {
            name: "reshape_probe",
            arguments: a.args(vec![(
                Ssa(0),
                IrType::Tensor {
                    dims: a.ints(vec![1024, 64]),
                    elem: DType::F16,
                },
            )]),
            operations: a.ops(vec![op]),
            grid: (1, 1, 1),
            return_type: None,
        };
        let f: &'static IRFunction<'static> = Box::leak(Box::new(f));
        let (_, _, verdicts, _) = reshape_forwards(f);
        let Some(ReshapeFlat::Refused(msg)) = verdicts.get(&Ssa(1)) else {
            panic!("an untyped reshape must be refused, not forwarded blind");
        };
        assert!(msg.contains("no result type"), "names the cause: {msg}");
    }

    /// ⭐ A CHAIN RESOLVES TRANSITIVELY — reshape of reshape reads the earlier entry, so the
    /// forwards map names the ULTIMATE source while the hops stay per-reshape for the law walk.
    #[test]
    fn a_reshape_chain_resolves_to_the_ultimate_source() {
        let a: &'static Arena = Arena::global();
        let f16 = |dims: Vec<i64>| IrType::Tensor {
            dims: a.ints(dims),
            elem: DType::F16,
        };
        let one = Operation {
            result: Some(Ssa(1)),
            op_type: OpKind::TensorReshape,
            operands: a.ssa(vec![Ssa(0)]),
            attributes: &[],
            result_type: Some(f16(vec![256, 4, 64])),
            regions: &[],
        };
        let two = Operation {
            result: Some(Ssa(2)),
            op_type: OpKind::TensorReshape,
            operands: a.ssa(vec![Ssa(1)]),
            attributes: &[],
            result_type: Some(f16(vec![1024, 64])),
            regions: &[],
        };
        let f = IRFunction {
            name: "reshape_probe",
            arguments: a.args(vec![(Ssa(0), f16(vec![1024, 64]))]),
            operations: a.ops(vec![one, two]),
            grid: (1, 1, 1),
            return_type: None,
        };
        let f: &'static IRFunction<'static> = Box::leak(Box::new(f));
        let (forwards, hops, verdicts, _) = reshape_forwards(f);
        assert_eq!(forwards.get(&Ssa(1)), Some(&Ssa(0)));
        assert_eq!(forwards.get(&Ssa(2)), Some(&Ssa(0)), "the ultimate source");
        assert_eq!(hops.get(&Ssa(2)), Some(&Ssa(1)), "hops stay per-reshape");
        assert!(matches!(verdicts.get(&Ssa(1)), Some(ReshapeFlat::Admitted)));
        assert!(matches!(verdicts.get(&Ssa(2)), Some(ReshapeFlat::Admitted)));
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
