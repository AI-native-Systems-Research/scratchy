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

use super::lower_ktir_to_superdsc::{
    Error, Gather, Region, err, gather_facts_at_tile, gathers_of_excluding, regions,
};
use crate::ktir_node::{Elementwise, GatedAct, KtirNode, Program, ReduceKind};
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

/// ⭐ THE PROGRAM'S OWN DEF/USE, AS ONE INDEX — every value's DEFINER, every value's READERS in
/// program order, and every op's POSITION in it.
///
/// # WHY THIS EXISTS, AND WHY IT IS NOT A FRAMEWORK
///
/// Until this index, every chain recogniser below rebuilt the same three walks as local
/// closures: a `def_of` (a `find` per lookup), a `sole_reader` (a filter for exactly one
/// consumer), and a `splat_value` (constant → splat → value). Three copies of one question
/// asked of the same IR — the shape this crate refuses everywhere else — and the fused MoE
/// kernel is the program that makes it untenable: its checked-in dump states ≈1,000 ops in one
/// `func.func`, with K top-k rounds and K per-pair projections, and every recogniser of those
/// chains asks the same def/reader/constant questions per link that the silu and rmsnorm
/// readers already asked.
///
/// So this is those closures' ONE implementation, extended with the two facts the fused chains
/// need and the closures never computed: a value's READERS AS A LIST (a scatter accumulator
/// has SEVERAL readers, and the question is which one is the accumulator, not whether there is
/// exactly one), and PROGRAM-ORDER POSITION (a gather must be proven to read a store that
/// PRECEDES it — no extent or attribute says that; only the order the ops appear in does).
///
/// # THE ONE SEMANTIC CHOICE, STATED
///
/// The index is built over [`IRFunction::ops_deep`] — every op, regions included, in program
/// order — which is the walk the silu reader's reader list and the rmsnorm `sole_reader`
/// already take, for the reason their own comments state: fusing a chain DESTROYS its
/// intermediates, so a reader hidden inside a region is exactly the consumer a fusion must not
/// blind itself to. The `def_of` closures the silu, rmsnorm and scale readers each rebuilt
/// searched the TOP LEVEL only; over the deep walk they answer identically for every value
/// defined there (the deep walk visits top-level ops first and in order) and additionally
/// resolve a value a REGION defines, which the top-level `find` silently could not. SSA names
/// one definer, so the first definer in program order is THE definer.
///
/// An op that names the same operand twice (`mulf x, x`) is ONE reader of `x`, not two — the
/// count every `contains`-based closure it replaced already reported, and the count the silu
/// reader's more-than-one-reader refusal names.
pub struct ProgramReads<'f> {
    ops: Vec<&'f Operation<'static>>,
    defs: std::collections::HashMap<Ssa, usize>,
    readers: std::collections::HashMap<Ssa, Vec<usize>>,
}

/// Read the program's def/use index. ONE pass over [`IRFunction::ops_deep`].
pub fn program_reads<'f>(f: &'f IRFunction<'static>) -> ProgramReads<'f> {
    let ops = f.ops_deep();
    let mut defs: std::collections::HashMap<Ssa, usize> =
        std::collections::HashMap::with_capacity(ops.len());
    let mut readers: std::collections::HashMap<Ssa, Vec<usize>> =
        std::collections::HashMap::with_capacity(ops.len());
    for (i, o) in ops.iter().enumerate() {
        // `or_insert` keeps the FIRST definer in program order — the same answer the per-chain
        // `find` closures gave. SSA makes a second definer unrepresentable, so this stays a
        // no-op rather than becoming an assertion a malformed producer could trip.
        if let Some(r) = o.result {
            defs.entry(r).or_insert(i);
        }
        for (n, &v) in o.operands.iter().enumerate() {
            // `operands[..n].contains`: an op naming `v` on both of its sides reads it ONCE,
            // so the readers list holds the op once — the count a `contains` filter gives.
            if !o.operands[..n].contains(&v) {
                readers.entry(v).or_default().push(i);
            }
        }
    }
    ProgramReads { ops, defs, readers }
}

impl<'f> ProgramReads<'f> {
    /// Every op in program order, regions included — the same list
    /// [`IRFunction::ops_deep`] returns, held so a recogniser's iteration and this index's
    /// positions name THE SAME ops.
    pub fn ops(&self) -> &[&'f Operation<'static>] {
        &self.ops
    }

    /// The op at `pos` — a position from [`readers_of`](Self::readers_of) or
    /// [`pos_of_result`](Self::pos_of_result), turned back into the op it names.
    pub fn op_at(&self, pos: usize) -> &'f Operation<'static> {
        self.ops[pos]
    }

    /// The op that DEFINES `v`, or `None` when `v` is a function argument nothing defines.
    /// First definer in program order.
    pub fn def_of(&self, v: Ssa) -> Option<&'f Operation<'static>> {
        self.defs.get(&v).copied().map(|i| self.ops[i])
    }

    /// The positions of the ops that READ `v`, in program order. An op naming `v` on both of
    /// its sides appears once.
    pub fn readers_of(&self, v: Ssa) -> &[usize] {
        self.readers.get(&v).map_or(&[], Vec::as_slice)
    }

    /// `v`'s ONE reader, when exactly one op reads it — the exclusivity a fused chain's links
    /// are proven by. `None` when none or several do.
    pub fn sole_reader(&self, v: Ssa) -> Option<&'f Operation<'static>> {
        match self.readers_of(v) {
            [only] => Some(self.ops[*only]),
            _ => None,
        }
    }

    /// The POSITION of `v`'s defining op. Precedence questions — does the store that lands the
    /// top-k ids precede the gather that reads them? — compare positions; no extent or
    /// attribute can answer them.
    pub fn pos_of_result(&self, v: Ssa) -> Option<usize> {
        self.defs.get(&v).copied()
    }

    /// The position of `op` itself, for RESULTLESS ops (a `ktdp.store`) whose position no
    /// result can name. ⛔ MATCHED BY ADDRESS: `op` must be a reference this index's own
    /// storage backs — one taken from [`ops`](Self::ops) or from walking the same function —
    /// not a copy. `Operation` is `Copy`, so a copied op is a different op even when it
    /// compares equal: two identical resultless stores are two stores, and only the address
    /// says which one a position names.
    pub fn pos_of_op(&self, op: &Operation<'static>) -> Option<usize> {
        self.ops.iter().position(|o| std::ptr::eq(*o, op))
    }

    /// A splatted compile-time FLOAT: `arith.constant` → `tensor.splat`, the one spelling this
    /// crate's producers use and the one `program_scalarmul_scale` already read. The closure
    /// three recognisers each rebuilt; one implementation now.
    pub fn splat_value(&self, s: Ssa) -> Option<f64> {
        let sp = self.def_of(s)?;
        if sp.op_type != OpKind::TensorSplat {
            return None;
        }
        let c = self.def_of(*sp.operands.first()?)?;
        if c.op_type != OpKind::ArithConstant {
            return None;
        }
        c.attributes.iter().find_map(|(k, v)| match (k, v) {
            (AttrKey::Value, Attr::Float(x)) => Some(*x),
            _ => None,
        })
    }

    /// The value `v` reads THROUGH rank plumbing — `tensor.expand_shape`,
    /// `tensor.collapse_shape`, `linalg.broadcast` — or `v` itself when it states none.
    ///
    /// This is the question a fused-chain recogniser asks of an operand: "is this mulf's
    /// operand, through the broadcast the program wraps it in, the round-r max-reduce's
    /// result?" The walk is the SAME bounded back-walk `program_broadcast_chains` and the
    /// rmsnorm reader's gain walk take (≤3 steps over the same three op kinds), because that
    /// is the deepest rank plumbing this door's producers state. A deeper chain is a shape
    /// none of them recognises either, so the walk stops at the value it reached and hands
    /// the caller "not plumbing I resolve" rather than guessing past the bound.
    ///
    /// ⭐ FROM A BROADCAST'S RESULT THIS REACHES THE SAME VALUE `Bcast::src` NAMES: the chains
    /// reader starts at the broadcast and walks its source; this starts at the result and
    /// walks through the broadcast. One law, two entry points — a recogniser that asks
    /// `through_plumbing(op.operand) == reduce_result` cannot disagree with one that looks
    /// the operand up in the chains map.
    ///
    /// ⛔ THIS IS IDENTITY, NOT CONSUMPTION. It says which value the plumbing CARRIES; it
    /// does not mark the plumbing ops consumed — `program_broadcast_chains` owns that for
    /// the chains it recognises, and a composite recogniser collects its own interior (the
    /// router's round ops, say) itself.
    pub fn through_plumbing(&self, v: Ssa) -> Ssa {
        let mut cursor = v;
        for _ in 0..3 {
            let Some(d) = self.def_of(cursor) else { break };
            if !matches!(
                d.op_type,
                OpKind::TensorExpandShape | OpKind::TensorCollapseShape | OpKind::LinalgBroadcast
            ) {
                break;
            }
            let Some(src) = d.operands.first().copied() else {
                break;
            };
            cursor = src;
        }
        cursor
    }

    /// THE `Elementwise(Mul)` / `ScalarMul` TIEBREAK, read off ONE op rather than the whole
    /// function.
    ///
    /// ⭐ THE DISCRIMINATOR IS THE CRATE'S OWN, NOT AN INVENTION. [`Program`]'s doc names it:
    /// "Elementwise `Mul` and `ScalarMul` are the one pair that needs a tiebreak, and it
    /// already exists: `program_scalarmul_scale` tells them apart by whether an operand is a
    /// `tensor.splat`." So an `arith.mulf` with exactly one splatted-constant operand IS a
    /// scalar multiply — there is no other kind it could be, because `Elementwise(Mul)` needs
    /// two tensor operands and the splat is not one. Choosing it is applying a documented
    /// rule, not picking a node kind on the producer's behalf.
    ///
    /// ⛔ AND IT MUST BE PER-OP, NOT PER-FUNCTION. `program_scalarmul_scale` reads every
    /// `arith.mulf` and requires them all to agree, which is exact for a one-node function.
    /// ONE decoder layer holds 21 `arith.mulf`: most have no splat, and the splatted ones
    /// carry FOUR DIFFERENT constants — `INV_D` (the rmsnorm reduction divisor), `QK_SCALE`
    /// (the score scale), and `RM` twice (the residual multipliers). The whole-function
    /// reader returns `None` there by construction, so the scale has to be read from the op
    /// that uses it.
    ///
    /// Returns `(scale, the tensor operand)`. `None` when this is not a scalar multiply at
    /// all:
    ///
    /// * NEITHER operand is a splat — an ordinary two-tensor `Elementwise(Mul)`, which the
    ///   1:1 map handles.
    /// * BOTH are splats — a multiply of two compile-time constants. That is not a scalarmul
    ///   (there is no tensor to scale) and emitting one would invent an operand; it belongs
    ///   to constant folding upstream, so it falls through to be refused by name.
    pub fn splat_scale_of(&self, op: &Operation<'static>) -> Option<(f32, Ssa)> {
        let [a, b] = op.operands[..] else { return None };
        match (self.splat_value(a), self.splat_value(b)) {
            (Some(v), None) => Some((v as f32, b)),
            (None, Some(v)) => Some((v as f32, a)),
            // Both or neither: not a scalar multiply. See the doc above.
            _ => None,
        }
    }

    /// THE RUNG-3 BOUND SCALE: the `[1,1]` parameter a `tensor.splat` reads through its load,
    /// and the tensor operand on the mulf's other side.
    ///
    /// `to_ktir`'s `expand_splat_of_scalar_argument` emits exactly that chain for a scalar
    /// argument (the argument retyped `index`, a `[1,1]` fp16 HBM view, a `[1,1]` access tile
    /// at corner (0,0), a load), and re-points the splat at the load. So the discriminator is
    /// structural: a splat whose operand is a `ktdp.load` over a degenerate `[1,1]` view of a
    /// parameter IS a launch-bound multiplier — the ATTN_SCALE mechanism with the value bound
    /// per launch instead of baked.
    ///
    /// ⛔ FAIL CLOSED ON THE SHAPE: only a BOTH-AXES-DEGENERATE view qualifies (`[1,1]`). A
    /// splat of a wider parameter (a row vector, a `[1, cols]` gamma — the mb-broadcast
    /// defect shape) is NOT a bound scalar and must fall through to the 1:1 map and be
    /// refused there, because emitting a scalarmul for it would address one value where the
    /// program means a row.
    ///
    /// ⛔ AND ON THE PROVENANCE: the view's address operand must be a function ARGUMENT. A
    /// `[1,1]` load of an intermediate has no binding and no launch spelling, so it is left
    /// alone rather than misread as one.
    ///
    /// ⭐ TAKES THE NODE, not just the op, because the binding tid is the LAUNCH's fact: this
    /// index answers what the program reads, and `k.bindings` says which slot that read lands
    /// in. The index must be over `k.func` — one built from another function answers about the
    /// wrong program.
    pub fn bound_scale_of(
        &self,
        k: &KtirNode,
        op: &Operation<'static>,
    ) -> Result<Option<BoundScale>, Error> {
        let f = &k.func;
        // The `[1,1]` VIEW a splat-of-load reads, when it reads one: splat → load → tile →
        // view. Every step is a `def_of` off this index — the same four steps
        // `region_for_operand` walks.
        let splat_view = |s: Ssa| -> Option<Ssa> {
            let sp = self.def_of(s)?;
            if sp.op_type != OpKind::TensorSplat {
                return None;
            }
            let loaded = *sp.operands.first()?;
            let ld = self.def_of(loaded)?;
            if ld.op_type != OpKind::KtdpLoad {
                return None;
            }
            let tile = *ld.operands.first()?;
            let tl = self.def_of(tile)?;
            if tl.op_type != OpKind::KtdpConstructAccessTile {
                return None;
            }
            let view = *tl.operands.first()?;
            let vw = self.def_of(view)?;
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
        let vw = self.def_of(view).expect("splat_view found it");
        let shape = super::lower_ktir_to_superdsc::shape_2d(vw);
        if shape != Some((1, 1)) {
            return Ok(None);
        }
        // ⛔ THE PARAMETER TEST: the view's address operand must be a function argument, and
        // the binding tid is that argument's position.
        let ptr = vw.operands.first().copied();
        let Some(idx) = f.arguments.iter().position(|(s, _)| Some(*s) == ptr) else {
            return Ok(None);
        };
        let Some(tid) = k.bindings.get(idx).map(|b| b.get()) else {
            return Ok(None);
        };
        Ok(Some(BoundScale { tid, tensor }))
    }
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
    let reads = program_reads(f);
    let mut out = Vec::new();
    for root in f
        .operations
        .iter()
        .filter(|o| o.op_type == OpKind::MathRsqrt)
    {
        if let Some(c) = rmsnorm_chain_from(&reads, f, root)? {
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
    reads: &ProgramReads<'_>,
    f: &IRFunction<'static>,
    root: &ktir_core::ir::Operation<'static>,
) -> Result<Option<RmsNormChain>, Error> {
    // `rsqrt(add)`. A root whose operand is not an `arith.addf` is not this shape and is not an
    // attempt at it — a plain `Elementwise(Rsqrt)`, so `Ok(None)`.
    let Some(add) = root.operands.first().copied().and_then(|s| reads.def_of(s)) else {
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
            match reads.splat_value(s) {
                Some(v) => e = Some(v),
                None => m = Some(s),
            }
        }
        (e, m)
    }) else {
        return Ok(None);
    };
    // `ms = <reduce> · INV_D`.
    let Some(scale) = reads.def_of(ms) else {
        return Ok(None);
    };
    if scale.op_type != OpKind::ArithMulf {
        return Ok(None);
    }
    let (Some(inv_d), Some(red_v)) = ({
        let mut d = None;
        let mut r = None;
        for &s in scale.operands.iter() {
            match reads.splat_value(s) {
                Some(v) => d = Some(v),
                None => r = Some(s),
            }
        }
        (d, r)
    }) else {
        return Ok(None);
    };
    let Some(red) = reads.def_of(red_v) else {
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
    let Some(sq) = red.operands.first().copied().and_then(|s| reads.def_of(s)) else {
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
    // The rank walk is OPTIONAL in shape but its ops are consumed when present: a producer that
    // broadcasts `r` to `[rows, cols]` states expand/collapse/broadcast, and one that keeps it rank-2
    // states none of them.
    let mut cursor = root_v;
    for _ in 0..3 {
        let Some(next) = reads.sole_reader(cursor) else {
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
    let Some(scale_mul) = reads.sole_reader(cursor) else {
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
    let Some(gain_mul) = reads.sole_reader(scaled) else {
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
        let Some(d) = reads.def_of(gamma) else { break };
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
    silu_mul_chains(f, &program_gelu_mul_chains(f)?)
}

/// The silu reader over the negfs the gelu reader has NOT claimed.
///
/// ⭐ THE SKIP IS THE COOPERATION, NOT A RELAXATION. The gelu longhand shares the silu
/// longhand's five op kinds with a different numerator, and the discriminator inside
/// [`silu_mul_chain_from`] refuses that near-miss on purpose — a gelu misread as a silu
/// is exactly the wrong function the fail-closed law exists for. Running the gelu reader
/// FIRST and skipping its claims here divides the negfs between the two readers exactly
/// once each, and every negf NEITHER claims is still refused by the discriminator: a
/// mangled gelu is a refusal either way, with the message of whichever reader can see
/// it.
fn silu_mul_chains(
    f: &IRFunction<'static>,
    gelus: &[GeluMulChain],
) -> Result<Vec<SiluMulChain>, Error> {
    let reads = program_reads(f);
    let mut out = Vec::new();
    for neg in f
        .operations
        .iter()
        .filter(|o| o.op_type == OpKind::ArithNegf)
    {
        // A claimed gelu's negf is not a silu head — the walk below lowers the one fused
        // op the gelu reader proved, so this reader has nothing to say about it.
        if neg.result.is_some_and(|r| gelus.iter().any(|c| c.neg == r)) {
            continue;
        }
        out.push(silu_mul_chain_from(&reads, f, neg)?);
    }
    Ok(out)
}

/// The one chain headed by `neg`, proven link by link.
fn silu_mul_chain_from(
    reads: &ProgramReads<'_>,
    f: &IRFunction<'static>,
    neg: &ktir_core::ir::Operation<'static>,
) -> Result<SiluMulChain, Error> {
    // Every op that reads `v`, so a link's exclusivity can be PROVEN rather than assumed.
    //
    // ⛔⛔⛔ DEEP, NOT TOP-LEVEL, AND THAT IS A CORRECTNESS MATTER RATHER THAN A STYLE ONE.
    // Fusing these five ops into one descriptor DESTROYS four intermediate values: the device
    // primitive never materialises `-g`, `e^-g`, `1+e^-g` or `silu(g)`. The exclusivity check below
    // is the only thing standing between that and a program where something ELSE reads one of them.
    // A reader inside an `scf.for` body would be invisible to a top-level walk, the chain would
    // fuse anyway, and the in-loop consumer would read a value nothing ever wrote. Free for a
    // straight-line producer, where the deep walk and `f.operations` are the same list.
    //
    // ⭐ THE INDEX IS THAT DEEP WALK: [`ProgramReads`] is built over `ops_deep`, so `readers_of`
    // answers with the counting already done, and this law lives once — in the index's own doc —
    // rather than being restated per closure.
    let readers = |v: Ssa| reads.readers_of(v).iter().map(|&p| reads.op_at(p));

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
        .filter_map(|&s| reads.splat_value(s))
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

/// THE TANH-GELU LONGHAND, as the ONE fused op it is — `out = gelu(gate) · up`.
///
/// # WHY THIS RECOGNISER EXISTS, AND WHY IT RUNS **BEFORE** THE SILU READER
///
/// `kernels/moe.py` — gemma-4's whole sparse block as one Triton program — spells its
/// experts' activation as the tanh-form gelu through the sigmoid identity,
/// `g / (1 + exp(-c₁·(g + c₂·g³)))`, the cubic in an f32 island the ladder narrows back
/// to the tile's f16. That longhand SHARES the silu longhand's five op kinds — negate,
/// exp, add, divide, multiply — with a DIFFERENT numerator (the divide divides `g`; the
/// exp's chain head is the cubic `v`), so the silu reader's fail-closed discriminator
/// refuses it, and because the pre-reads gate the whole per-op walk, NOTHING downstream
/// is reachable while it stands. This reader claims those chains first, so the silu
/// reader sees only the negfs that are genuinely its own.
///
/// # IT PROVES THE MATCH; IT DOES NOT PATTERN-GUESS — AND THE PROOF IS THE CONSTANTS
///
/// The silu reader's discriminators carry over, and three more do the real work here,
/// every one READ from the program rather than assumed:
///
/// * **the divide's numerator is the cubic's base `g`**, and the exp's chain head is
///   exactly `v = c₁·(g + c₂·g³)` — that association, link by link, each link unique;
/// * **c₁ is `2·√(2/π)` and c₂ is `0.044715`**, each checked as the value the f16 grid
///   states (the island is narrowed back, so the splats are f16 images: `1.595703125`
///   and `0.044708251953125`, measured off the live moe.py artifact). Any other
///   constants are a different function with the same op kinds;
/// * **every op of the chain states f16** — the f32 island the kernel's own comment
///   describes (an f16 `x³` overflows at |x| ≥ 41) narrowed back at both ends. A chain
///   the ladder left at f32 would have `OpFunc::Gelu` compute it in the tensor's f16
///   storage, substituting one arithmetic for the other.
///
/// A chain that almost matches is REFUSED BY NAME, never lowered as the nearest thing —
/// and a chain that fails BEFORE the numerator discriminator (a broken exp link, a
/// missing divide) is left to the silu reader, which refuses it with the message that
/// names the link: this reader claims a chain only whole.
#[derive(Clone, Copy, Debug)]
pub struct GeluMulChain {
    /// The cubic's base AND the divide's numerator — gelu's argument.
    pub gate: Ssa,
    /// The other factor of the terminal multiply.
    pub up: Ssa,
    /// The terminal `arith.mulf`'s result: the value the ONE fused op produces, and the
    /// value the rest of the program consumes.
    pub mul: Ssa,
    /// The chain's `arith.negf` result — the key the silu reader skips on, so the two
    /// readers cannot claim one negf.
    pub neg: Ssa,
    /// The cubic's three products and its sum, the e-argument, and the four spine
    /// results — every intermediate the fused op never materialises, so
    /// [`lower_function`] must lower NOTHING for them. The terminal multiply is
    /// deliberately not in here: it is the op that BECOMES the fused program.
    pub consumed: [Ssa; 9],
}

/// Read every tanh-gelu longhand off the program, or say why one of them is not one.
///
/// ⭐ ONE CHAIN PER `arith.negf`, the silu reader's own law: the fused MoE kernel spells
/// one gelu per expert pair — eight in one `func.func` at top-8 — and what has to be
/// unique is each LINK, checked per chain. A negf this reader returns no chain for is a
/// silu (claimed by the silu reader, which runs after this one over the remainder) or a
/// near-miss the silu reader refuses; `Err` means a chain that divides something OTHER
/// than the negated value by `1+e^-x` — recognisably a gated-act longhand that is
/// neither silu nor the tanh gelu — and refusing it is the fail-closed law.
pub fn program_gelu_mul_chains(f: &IRFunction<'static>) -> Result<Vec<GeluMulChain>, Error> {
    let reads = program_reads(f);
    // The tanh gelu's two constants, as the f16 grid states them. The island is narrowed
    // back (checked per chain below), so an f16 splat is the only spelling the program
    // can carry, and the grid's image of the exact constant is the one value that is
    // that constant at f16 precision — a producer's decimal within half an ulp of
    // 0.044715 rounds to the same image, and one further away is a different constant.
    // Both f64 -> f32 -> f16 double roundings land on the f16 the direct rounding
    // picks, measured: 1.595703125 and 0.044708251953125, the values the live moe.py
    // artifact states.
    let f16_image = |x: f64| {
        ktir_core::codec::f16_units_to_f32(&ktir_core::codec::f32_to_f16_units(&[x as f32]))[0]
            as f64
    };
    let (c1, c2) = (
        f16_image(2.0 * (2.0 / std::f64::consts::PI).sqrt()),
        f16_image(0.044715),
    );
    let mut out = Vec::new();
    for neg in f
        .operations
        .iter()
        .filter(|o| o.op_type == OpKind::ArithNegf)
    {
        if let Some(chain) = gelu_mul_chain_from(&reads, f, neg, c1, c2)? {
            out.push(chain);
        }
    }
    Ok(out)
}

/// The one chain headed by `neg`, proven link by link — or `Ok(None)` when the chain is
/// not this reader's to claim.
fn gelu_mul_chain_from(
    reads: &ProgramReads<'_>,
    f: &IRFunction<'static>,
    neg: &Operation<'static>,
    c1: f64,
    c2: f64,
) -> Result<Option<GeluMulChain>, Error> {
    // THE SHARED SPINE, in the silu reader's own link law but SILENT: negate -> exp ->
    // 1+ -> divide, each link exactly one reader of the right kind. A break here is a
    // near-miss of BOTH longhands — the two spellings share these four ops — and the
    // silu reader, which runs after this one over the negfs this reader did not claim,
    // refuses it with the message that names the link. Skipping rather than duplicating
    // that refusal keeps ONE message per defect.
    let sole = |v: Ssa, want: OpKind| -> Option<&Operation<'static>> {
        match reads.readers_of(v) {
            [p] if reads.op_at(*p).op_type == want => Some(reads.op_at(*p)),
            _ => None,
        }
    };
    let (Some(v), Some(neg_v)) = (neg.operands.first().copied(), neg.result) else {
        return Ok(None);
    };
    let Some(exp) = sole(neg_v, OpKind::MathExp) else {
        return Ok(None);
    };
    let Some(exp_v) = exp.result else {
        return Ok(None);
    };
    let Some(add) = sole(exp_v, OpKind::ArithAddf) else {
        return Ok(None);
    };
    let Some(add_v) = add.result else {
        return Ok(None);
    };
    let Some(div) = sole(add_v, OpKind::ArithDivf) else {
        return Ok(None);
    };
    let Some(div_v) = div.result else {
        return Ok(None);
    };

    // ⭐ THE DISCRIMINATOR, the silu reader's own: the divide's numerator. `g/(1+e^-g)`
    // is silu — this reader's `Ok(None)`, the silu reader's claim. The numerator is the
    // operand BESIDE the denominator rather than `operands.first()`, so a divide with
    // its sides spelled the other way round cannot hand this reader the denominator as
    // a gate.
    let Some(gate) = div.operands.iter().copied().find(|&s| s != add_v) else {
        return err(format!(
            "{}: the gelu longhand's `arith.divf` divides `1+exp(-x)` by itself",
            f.name
        ));
    };
    if gate == v {
        return Ok(None);
    }
    // FROM HERE THE CHAIN IS CLAIMED, and every failure is a refusal by name: a chain
    // that divides a cubic base by `1+e^-v` is recognisably a gated-act longhand, and
    // lowering a near-miss as `OpFunc::Gelu` would compute a different function.

    // The denominator's added constant must be 1.0 — `1/(1+e^-x)`, the sigmoid identity
    // the silu reader's own law states. A candidate that divides a cubic base by
    // `c + e^-v` for any other `c` is a different function, refused here rather than
    // rounded into the gelu. Spelled as an inequality rather than a float literal
    // pattern, the silu reader's own way: the refusal keeps `x` to name the constant
    // the program actually states.
    let one = add
        .operands
        .iter()
        .copied()
        .filter(|&s| s != exp_v)
        .find_map(|s| reads.splat_value(s));
    match one {
        Some(x) if x != 1.0 => {
            return err(format!(
                "{}: the gelu longhand's `arith.addf` adds {x}, not 1.0. `g/(1+e^-v)` is \
                 the sigmoid identity, so any other constant is a DIFFERENT function and \
                 lowering it as `OpFunc::Gelu` would compute something the program does \
                 not state.",
                f.name
            ));
        }
        Some(_) => {}
        None => {
            return err(format!(
                "{}: the gelu longhand's `arith.addf` adds a value that is not a splatted \
                 constant, so nothing proves it is the 1 of `1+e^-v`",
                f.name
            ));
        }
    }

    // `v = c₁·(g + c₂·g³)`, walked from the negf's operand DOWN to the cubic's base —
    // each step the op that DEFINES the value, each link checked for the association
    // the tanh form states. ⛔ THE ASSOCIATION IS FIXED, LEFT-DEEP IN THE CUBIC:
    // `c₂·g` first, then `·g` twice, then `g + ·`, then `·c₁` — the spelling the ladder
    // produces, and a different association over the same ops states a different
    // program, refused rather than re-associating on the producer's behalf.
    let def_of_kind = |x: Ssa, want: OpKind, what: &str| -> Result<&Operation<'static>, Error> {
        match reads.def_of(x) {
            Some(d) if d.op_type == want => Ok(d),
            Some(d) => err(format!(
                "{}: the gelu longhand's {what} is defined by `{:?}`, but the tanh form \
                 `c₁·(g + c₂·g³)` states `{:?}`",
                f.name, d.op_type, want
            )),
            None => err(format!(
                "{}: the gelu longhand's {what} is not computed by this function — the \
                 tanh form states the cubic inline",
                f.name
            )),
        }
    };
    // `x = <tile> · <splatted constant>`: the multiply's ONE splatted operand, and the
    // other operand. Refuses when the constant is missing (two tiles multiplied where
    // the tanh form states a constant) and when there are two (a constant-times-constant
    // the ladder folds before this door).
    let mul_by_const = |d: &Operation<'static>, what: &str| -> Result<(Ssa, f64), Error> {
        let mut splats = d
            .operands
            .iter()
            .copied()
            .filter_map(|s| reads.splat_value(s).map(|x| (s, x)));
        let Some((splat_ssa, splat_v)) = splats.next() else {
            return err(format!(
                "{}: the gelu longhand's {what} multiplies two tiles, but the tanh form \
                 `c₁·(g + c₂·g³)` states a splatted constant there",
                f.name
            ));
        };
        if let Some((_, second)) = splats.next() {
            return err(format!(
                "{}: the gelu longhand's {what} multiplies two splatted constants \
                 ({splat_v} and {second}), a constant fold the ladder performs before \
                 this door",
                f.name
            ));
        }
        let other = d
            .operands
            .iter()
            .copied()
            .find(|&s| s != splat_ssa)
            .expect("a binary multiply has an operand beside the splat");
        Ok((other, splat_v))
    };
    // The operand BESIDE the cubic's base, so the association is proven rather than
    // positional. Refuses when the op does not read `g` — `c₁·(g + c₂·g³)` reads it once
    // per factor — and when it reads it twice (`g·g` is a square, not the cubic).
    let beside_gate = |d: &Operation<'static>, what: &str| -> Result<Ssa, Error> {
        let n = d.operands.iter().filter(|&&s| s == gate).count();
        if n != 1 {
            return err(format!(
                "{}: the gelu longhand's {what} reads the divide's numerator {n} times, \
                 but the tanh cubic `c₁·(g + c₂·g³)` reads it exactly once per factor",
                f.name
            ));
        }
        Ok(d.operands
            .iter()
            .copied()
            .find(|&s| s != gate)
            .expect("one operand is the numerator, so the other is beside it"))
    };

    // v = c₁ · (g + c₂·g³) — c₁ READ off the splat, not assumed.
    let dv = def_of_kind(v, OpKind::ArithMulf, "e-argument")?;
    let (t4, found_c1) = mul_by_const(dv, "e-argument")?;
    if found_c1 != c1 {
        return err(format!(
            "{}: the gelu longhand scales `(g + c₂·g³)` by {found_c1}, not the tanh \
             gelu's c₁ = 2·√(2/π) = 1.5957691216057308 (the f16 grid's {c1}). Any other \
             constant is a different function with the same op kinds, so it is refused \
             rather than lowered as `OpFunc::Gelu`.",
            f.name
        ));
    }
    // t4 = g + c₂·g³
    let dt4 = def_of_kind(t4, OpKind::ArithAddf, "cubic's sum")?;
    let t3 = beside_gate(dt4, "cubic's sum")?;
    // t3 = t2 · g
    let dt3 = def_of_kind(t3, OpKind::ArithMulf, "cubic's third factor")?;
    let t2 = beside_gate(dt3, "cubic's third factor")?;
    // t2 = t1 · g
    let dt2 = def_of_kind(t2, OpKind::ArithMulf, "cubic's second factor")?;
    let t1 = beside_gate(dt2, "cubic's second factor")?;
    // t1 = g · c₂ — the base beside the constant must be THE divide's numerator, or the
    // fused op would gate one tile and divide another.
    let dt1 = def_of_kind(t1, OpKind::ArithMulf, "cubic's first factor")?;
    let (t1_base, found_c2) = mul_by_const(dt1, "cubic's first factor")?;
    if t1_base != gate {
        return err(format!(
            "{}: the gelu longhand's cubic starts from a value that is not the divide's \
             numerator — `c₁·(g + c₂·g³)` and `g/(1+e^-v)` must state the SAME g, or the \
             fused `OpFunc::Gelu` would gate one tile and divide another",
            f.name
        ));
    }
    if found_c2 != c2 {
        return err(format!(
            "{}: the gelu longhand's cubic multiplies g by {found_c2}, not the tanh \
             gelu's c₂ = 0.044715 (the f16 grid's {c2}). Any other constant is a \
             different function with the same op kinds, so it is refused rather than \
             lowered as `OpFunc::Gelu`.",
            f.name
        ));
    }

    // ⛔ EACH CUBIC INTERMEDIATE HAS EXACTLY ONE READER — the same destruction law the
    // spine's `sole` enforces above. The fused `OpFunc::Gelu` never materialises
    // `t1`..`v`, so a second reader would be reading a value nothing writes. The one
    // reader is the next link by construction (each definer reads its operand), so the
    // check is a count, not an identity hunt.
    for (x, what) in [
        (t1, "the cubic's first factor"),
        (t2, "the cubic's second factor"),
        (t3, "the cubic's third factor"),
        (t4, "the cubic's sum"),
        (v, "the e-argument"),
    ] {
        if reads.readers_of(x).len() != 1 {
            return err(format!(
                "{}: {what} is read by {} ops, and this fusion DESTROYS it — the device's \
                 `OpFunc::Gelu` never materialises the longhand's intermediates, so a \
                 second reader would be reading a value nothing writes. Exactly one \
                 reader is required.",
                f.name,
                reads.readers_of(x).len()
            ));
        }
    }

    // gelu -> multiply by `up`. The silu reader's terminal law, refusing in this
    // reader's own name: `OpFunc::Gelu` is `gelu(g)·up`, and a bare gelu with no
    // multiply is a different program this door has no arm for.
    let mul = match reads.readers_of(div_v) {
        [p] if reads.op_at(*p).op_type == OpKind::ArithMulf => reads.op_at(*p),
        [] => {
            return err(format!(
                "{}: nothing reads the gelu longhand's quotient, so the gated product \
                 does not continue there",
                f.name
            ));
        }
        [p] => {
            return err(format!(
                "{}: the gelu longhand's quotient is read by `{:?}`, but the gated \
                 product reads it with `ArithMulf`",
                f.name,
                reads.op_at(*p).op_type
            ));
        }
        rs => {
            return err(format!(
                "{}: the gelu longhand's quotient is read by {} ops, and this fusion \
                 DESTROYS it — exactly one reader, `ArithMulf`, is required",
                f.name,
                rs.len()
            ));
        }
    };
    let Some(up) = mul.operands.iter().copied().find(|&s| s != div_v) else {
        return err(format!(
            "{}: the gelu's `arith.mulf` squares it rather than multiplying by an `up` tile",
            f.name
        ));
    };
    let Some(mul_v) = mul.result else {
        return err(format!("{}: `arith.mulf` with no result", f.name));
    };

    // ⛔ THE f32 ISLAND NARROWED BACK, checked on EVERY op the chain claims. The
    // kernel's own comment states why the island exists (an f16 `x³` overflows at
    // |x| ≥ 41) and the ladder narrows it to the tile's f16 when both ends are f16 —
    // measured on the live artifact, every op states f16. `OpFunc::Gelu` computes in
    // the tensor's storage format, so lowering a chain the ladder left at f32 would
    // substitute f16 arithmetic for the program's f32 — a different function, silently.
    for (op, what) in [
        (dv, "the e-argument's product"),
        (dt4, "the cubic's sum"),
        (dt3, "the cubic's third factor"),
        (dt2, "the cubic's second factor"),
        (dt1, "the cubic's first factor"),
        (neg, "the negation"),
        (exp, "the exponential"),
        (add, "the denominator's sum"),
        (div, "the quotient"),
        (mul, "the gated product"),
    ] {
        if op.result_type.and_then(|t| t.elem()) != Some(ktir_core::dtypes::DType::F16) {
            return err(format!(
                "{}: {what} of the gelu longhand does not state f16 — the tanh form's f32 \
                 island must be narrowed back before this door, or `OpFunc::Gelu` would \
                 compute it in the tensor's storage format instead of the program's \
                 arithmetic",
                f.name
            ));
        }
    }

    Ok(Some(GeluMulChain {
        gate,
        up,
        mul: mul_v,
        neg: neg_v,
        consumed: [t1, t2, t3, t4, v, neg_v, exp_v, add_v, div_v],
    }))
}

// ═══════════════════════════════════════════════════════════════════════════════
//  THE W8A8 GATE/UP SLAB GEMMS — the fused MoE kernel's expert projections.
// ═══════════════════════════════════════════════════════════════════════════════

/// One claimed gate/up gemm: the links the fusion destroys, and the two tiles it
/// claims out of the gathers census.
///
/// ⭐ THE TERMINAL IS NOT CONSUMED — it is the value the REST of the program
/// reads (pair `p`'s `[1, n]` projection row, the gated act's gate input), so the
/// walk records it as a WINDOW of the projection's output intermediate and every
/// consumer resolves it there. The same law the gelu reader states for its
/// terminal multiply.
#[derive(Clone, Copy, Debug)]
pub struct W8a8Chain {
    /// The pair this chain serves — its anchor word in the index buffer, proven
    /// one of `0..k` by the projection's anchors check.
    pub pair: u32,
    /// The weight tile — this chain's claim out of [`gathers_of`]'s census
    /// (`GatherFacts::tile`'s identity, never a position).
    pub weight_tile: Ssa,
    /// The `tensor.collapse_shape` that flattens the gathered slab to `[n, inn]`.
    pub collapsed: Ssa,
    /// The `linalg.matmul` — the fp8 contraction the fused emission replaces.
    pub dot: Ssa,
    /// The dequant's first leg: `dot · broadcast(sx)`.
    pub mul_s: Ssa,
    /// The dequant's terminal: `mul_s · the pair's gathered router scale` — the
    /// chain's OUTPUT value.
    pub terminal: Ssa,
    /// The router-scale tile — the chain's second claim out of the census.
    pub scale_tile: Ssa,
}

/// One gate/up projection: every pair's chain over one `(weight, scale)` bank
/// pair, plus the shared facts the emission needs.
#[derive(Clone, Debug)]
pub struct W8a8Projection {
    /// The expert input's tensor id — the shared quantize's `Fp8Facts::a_tid`.
    pub a_tid: u32,
    /// The routing parameter's tensor id — the XIdx landing is keyed off it.
    pub idx_tid: u32,
    /// The weight bank's tensor id (the `[E, n, inn]` fp8 parameter).
    pub weight_tid: u32,
    /// The scale bank's tensor id (the `[E, n]` f16 parameter).
    pub scale_tid: u32,
    /// The top-k width — one chain per index word, proven `0..k` complete.
    pub k: u32,
    /// The contraction width (`inn`, the expert input's row width).
    pub inn: u32,
    /// The projection's output width per pair.
    pub n: u32,
    /// The experts the bank holds (`E`).
    pub e_count: u32,
    /// The spine's codes and per-token scale — the cross-projection consistency
    /// facts: gate and up are two projections over ONE expert input, so these
    /// must be the SAME values in both.
    pub qx: Ssa,
    pub sx: Ssa,
    /// The chains, sorted by pair.
    pub chains: Vec<W8a8Chain>,
    /// The FIRST terminal in program order — the op the walk meets first and
    /// lowers the whole projection at.
    pub emit_at: Ssa,
    /// Everything the fusion destroys: the spine's eleven compute ops (shared by
    /// every chain, so stated once) and each chain's three interior links. The
    /// terminals are deliberately absent — they are the projection's output
    /// windows.
    pub consumed: Vec<Ssa>,
}

/// The per-chain claim, before grouping — the recogniser's working state.
struct W8a8Claim {
    chain: W8a8Chain,
    weight_tid: u32,
    scale_tid: u32,
    a_load: Ssa,
    a_tid: u32,
    qx: Ssa,
    sx: Ssa,
    idx_tid: u32,
    anchor: i64,
    e_count: u32,
    n: u32,
    inn: u32,
    /// The spine's eleven destroyed compute ops: abs, the amax reduce, its
    /// scaling, the floor, the scale itself, the reciprocal, the codes' scaling
    /// and two clamps, the truncation, and the codes. `(a_load, qx, sx)`
    /// equality across chains proves this is ONE set of ops by SSA, so the
    /// group's first claim carries it for the projection.
    spine_consumed: [Ssa; 11],
}

/// The W8A8 spine's constants, as the f16 grid states them — computed once per
/// program, measured off the live moe.py artifact (`_fp8_scale`/`_fp8_codes`):
/// the splats are f16 images `0.002231597900390625` (1/448) and
/// `5.0067901611328125e-6` (1/448²), the ±448 clamps, and the reciprocal's 1.
struct W8a8Consts {
    inv_qmax: f64,
    floor: f64,
    cap: f64,
    one: f64,
    lo: f64,
    hi: f64,
}

/// The PARAMETER a `ktdp.store` writes — its tile's view's own pointer argument.
/// `None` for a store whose tile is not a plain view over a function argument.
///
/// ⛔ THIS IS THE DOWN DISCRIMINATOR'S RAW MATERIAL: the parameters in the set
/// this builds are the program's SCRATCH (the fused MoE kernel round-trips its
/// gelu output through `desc_act`), and a quantize spine rooting at a load of
/// one is a RELOAD, not an expert input.
fn store_param_ptr(reads: &ProgramReads<'_>, store: &Operation<'static>) -> Option<Ssa> {
    let tile = reads.def_of(*store.operands.get(1)?)?;
    if tile.op_type != OpKind::KtdpConstructAccessTile {
        return None;
    }
    let view = reads.def_of(*tile.operands.first()?)?;
    if view.op_type != OpKind::KtdpConstructMemoryView {
        return None;
    }
    view.operands.first().copied()
}

/// The PARAMETER a plain `ktdp.load` reads — load → access tile → memory view →
/// the pointer argument. `None` for a gathered load (an INDIRECT tile: the A side
/// of a gathered gemm is not this form) and for a computed value: neither states
/// one parameter to name.
fn plain_load_param_ptr(reads: &ProgramReads<'_>, load: &Operation<'static>) -> Option<Ssa> {
    let tile = reads.def_of(*load.operands.first()?)?;
    if tile.op_type != OpKind::KtdpConstructAccessTile {
        return None;
    }
    let view = reads.def_of(*tile.operands.first()?)?;
    if view.op_type != OpKind::KtdpConstructMemoryView {
        return None;
    }
    view.operands.first().copied()
}

/// THE W8A8 GATE/UP SLAB GEMMS, as the ONE fused emission they are — the card's
/// own `quantscalepertokenfp8` + gathered `matmulfp8` + two-multiply dequant.
///
/// # WHY THIS RECOGNISER EXISTS
///
/// `kernels/moe.py` — gemma-4's whole sparse block as one Triton program — spells
/// each pair's gate/up projection as the longhand the card's W8A8 primitive
/// replaces: ONE per-token quantize of the expert input
/// (`abs → amax → ·INV_QMAX → max(SCALE_FLOOR) → min(448)` = `sx`;
/// `x·(1/sx)` clamped to ±448, truncated to fp8, widened back = the codes `qx`),
/// then per pair a gathered fp8 slab gemm `qx @ wᵀ` whose product is dequantised
/// in two multiplies (`raw · broadcast(sx) · the pair's gathered router scale`).
/// Lowered op-by-op that is eleven pointwise ops and a PLAIN FP16 MATMUL per
/// pair — a different algorithm, not a slower one (the fp8 contraction and its
/// per-token scale are the card's own W8A8 numerics) — and at the whole-function
/// door the per-pair slab gathers refuse the gathers census besides. This reader
/// claims the chains whole, so [`lower_function`] emits each projection as one
/// [`super::lower_ktir_to_superdsc::expert_matmul_fused_m1`].
///
/// # THE PROOF, EVERY FACT READ FROM THE PROGRAM
///
/// * **B is a collapsed gathered slab** — `tensor.collapse_shape` of a
///   `ktdp.load` over a `ktdp.construct_indirect_access_tile`, its facts read
///   through the same per-tile walk [`gathers_of`] uses, so the two readings of
///   one tile cannot drift. The router's matmul (a plain load) fails here and is
///   not this reader's;
/// * **TransposeB** — the W8A8 form contracts over the slab's own `[n, inn]`
///   view. A plain-B gathered gemm is a different program, left to the walk;
/// * **the spine's constants** — `INV_QMAX = 1/448`, `SCALE_FLOOR = 1/448²`, the
///   `448` cap, the reciprocal's `1.0` numerator and the codes' `±448` clamps,
///   each checked as the value the f16 grid states (the island is narrowed
///   back, so the splats are f16 images — measured off the live artifact). Any
///   other constants are a different function with the same op kinds;
/// * **the amax reduce is `Max` over `abs(a)` seeded `−inf`**, its result `[1]`
///   f16 — the `quantscalepertokenfp8` primitive's own semantics (a Max seeded
///   any higher clamps every-negative rows and is a different function);
/// * **the spine is over ONE row** — the amax's `abs` reads the SAME value the
///   codes multiply reads, and the codes every dot contracts are ONE shared
///   value read only by claimed dots (checked across ALL projections after
///   grouping: gate and up share one spine);
/// * **destruction is exclusive** — every destroyed link has exactly the one
///   reader the chain states, `sx` is read only by the reciprocal and rank
///   plumbing, and no `ktdp.store` reads a terminal (the fusion writes the
///   projection's output intermediate; a stored terminal is a write the walk's
///   hook would silently drop);
/// * **the anchors are exactly `0..k`** — every pair's two tiles anchor at its
///   own index word, the whole top-k and nothing else;
/// * **the island is narrowed back** — the truncation states `Fp8E4m3` and the
///   widening `F16`, the boundaries of the one fp8 island the codes are.
///
/// # ⛔ THE DOWN DISCRIMINATOR IS PROVENANCE, NOT SHAPE
///
/// The down projection's spine is the SAME eleven ops over a RELOAD of the act
/// scratch the program round-trips (store then load, pair by pair). What
/// separates gate/up from down is where the spine ROOTS: gate/up at a load of
/// the EXPERT INPUT — a parameter nothing in the program stores to — down at a
/// load of a parameter the program stores to. That fact survives the down leg's
/// own recogniser and does not depend on `M` (a tile-shape proxy would misfire
/// the day `M > 1` is admitted: the `xe` tile `[1, H]` is its view at `M = 1`
/// only). A spine rooting at a STORED parameter returns `Ok(None)` — not this
/// reader's, not a refusal.
///
/// A chain that almost matches is REFUSED BY NAME, never lowered as the nearest
/// thing — and a chain that fails BEFORE the provenance check (a broken spine
/// link, a plain-B orientation, a non-gathered B) is left to the walk, which
/// refuses it with the message that names the link: this reader claims a chain
/// only whole.
pub fn program_w8a8_gemm_chains(k: &KtirNode) -> Result<Vec<W8a8Projection>, Error> {
    let f = &k.func;
    let reads = program_reads(f);
    // The constants, as the f16 grid states them — the gelu reader's own law:
    // the island is narrowed back, so an f16 splat is the only spelling the
    // program can carry, and the grid's image of the exact constant is the one
    // value that is that constant at f16 precision.
    let f16_image = |x: f64| {
        ktir_core::codec::f16_units_to_f32(&ktir_core::codec::f32_to_f16_units(&[x as f32]))[0]
            as f64
    };
    let consts = W8a8Consts {
        inv_qmax: f16_image(1.0 / 448.0),
        floor: f16_image(1.0 / 448.0 / 448.0),
        cap: f16_image(448.0),
        one: 1.0,
        lo: f16_image(-448.0),
        hi: f16_image(448.0),
    };
    // ⛔ THE PARAMETERS THIS PROGRAM STORES TO ARE ITS SCRATCH — the provenance
    // the down discriminator keys on.
    let stored: std::collections::HashSet<Ssa> = reads
        .ops()
        .iter()
        .copied()
        .filter(|o| o.op_type == OpKind::KtdpStore)
        .filter_map(|s| store_param_ptr(&reads, s))
        .collect();
    let mut claims: Vec<W8a8Claim> = Vec::new();
    for op in reads
        .ops()
        .iter()
        .copied()
        .filter(|o| o.op_type == OpKind::LinalgMatmul)
    {
        if let Some(claim) = w8a8_chain_from(k, &reads, &stored, op, &consts)? {
            claims.push(claim);
        }
    }
    if claims.is_empty() {
        return Ok(Vec::new());
    }
    // `qx` is read only by claimed dots — across ALL projections, because gate
    // and up share one spine. A stray reader would read a value the fusion
    // destroys (the fp8 contraction reads the codes on the card).
    let dots: std::collections::HashSet<Ssa> = claims.iter().map(|c| c.chain.dot).collect();
    for c in &claims {
        for &p in reads.readers_of(c.qx) {
            let r = reads.op_at(p);
            let claimed =
                r.op_type == OpKind::LinalgMatmul && r.result.is_some_and(|d| dots.contains(&d));
            if !claimed {
                return err(format!(
                    "{}: the W8A8 quantize spine's codes {:?} are read by `{:?}` — the fusion \
                     DESTROYS the codes (the fp8 contraction reads them on the card), so exactly \
                     the claimed gemms may read them",
                    f.name, c.qx, r.op_type
                ));
            }
        }
    }
    // Group by (weight, scale) bank pair — gate and up are two projections over
    // one expert input, and the anchors law is stated per projection.
    let mut groups: Vec<((u32, u32), Vec<&W8a8Claim>)> = Vec::new();
    for c in &claims {
        let key = (c.weight_tid, c.scale_tid);
        match groups.iter_mut().find(|(g, _)| *g == key) {
            Some((_, v)) => v.push(c),
            None => groups.push((key, vec![c])),
        }
    }
    let mut out = Vec::with_capacity(groups.len());
    for ((weight_tid, scale_tid), group) in groups {
        // ⭐ ONE SPINE PER PROJECTION: every chain's quantize is the SAME eleven
        // ops. `(a_load, qx, sx)` equality proves the whole spine is one set of
        // ops by SSA (one definer per value), so the first claim's spine list IS
        // the projection's.
        let first = group[0];
        for c in &group[1..] {
            if (c.a_load, c.qx, c.sx, c.idx_tid)
                != (first.a_load, first.qx, first.sx, first.idx_tid)
            {
                return err(format!(
                    "{}: two gemm chains over the weight bank t{weight_tid} root at DIFFERENT \
                     quantize spines — the W8A8 form states ONE quantize of the expert input, \
                     shared by every pair's gemm",
                    f.name
                ));
            }
        }
        // ⛔ THE ANCHORS ARE EXACTLY `0..k` — the whole top-k, one gemm per
        // index word. A gap is a routing shape this door cannot address (the
        // landing tiles results `0..k-1`); a repeat is two gemms on one pair.
        let mut anchors: Vec<i64> = group.iter().map(|c| c.anchor).collect();
        anchors.sort_unstable();
        let k = anchors.len() as u32;
        if anchors != (0..k as i64).collect::<Vec<_>>() {
            return err(format!(
                "{}: the gemm chains over the weight bank t{weight_tid} anchor at {anchors:?} — \
                 the pairs of one projection are the whole top-k, one gemm per index word 0..k, \
                 and a gap or a repeat is a routing shape this door cannot address",
                f.name
            ));
        }
        // The geometry agrees across the group (each chain read its own views).
        for c in &group[1..] {
            if (c.e_count, c.n, c.inn) != (first.e_count, first.n, first.inn) {
                return err(format!(
                    "{}: two gemm chains over the weight bank t{weight_tid} state DIFFERENT \
                     bank geometries — one projection is one bank",
                    f.name
                ));
            }
        }
        let mut chains: Vec<W8a8Chain> = group.iter().map(|c| c.chain).collect();
        chains.sort_by_key(|c| c.pair);
        // The emission hangs on the group's FIRST terminal in program order —
        // the op the walk meets before any of the group's other chain ops.
        let mut emit_at = chains[0].terminal;
        let mut emit_pos = usize::MAX;
        for c in &chains {
            let Some(p) = reads.pos_of_result(c.terminal) else {
                return err(format!(
                    "{}: a claimed W8A8 terminal has no defining position — the claim walked it \
                     from its own definer, so this is not reachable",
                    f.name
                ));
            };
            if p < emit_pos {
                emit_pos = p;
                emit_at = c.terminal;
            }
        }
        // The spine's eleven compute ops (one set — proven above) plus each
        // chain's three destroyed links. The terminals are deliberately absent:
        // they are the projection's output windows.
        let mut consumed = first.spine_consumed.to_vec();
        for c in &chains {
            consumed.extend([c.collapsed, c.dot, c.mul_s]);
        }
        out.push(W8a8Projection {
            a_tid: first.a_tid,
            idx_tid: first.idx_tid,
            weight_tid,
            scale_tid,
            k,
            inn: first.inn,
            n: first.n,
            e_count: first.e_count,
            qx: first.qx,
            sx: first.sx,
            chains,
            emit_at,
            consumed,
        });
    }
    // ⭐ ONE EXPERT INPUT, ONE SPINE, ONE GEOMETRY — gate and up are two
    // projections over the SAME facts. A disagreement is two different inputs
    // wearing one form, refused rather than adjudicated.
    for p in out.iter().skip(1) {
        let first = &out[0];
        if (p.a_tid, p.qx, p.sx, p.inn, p.n, p.e_count, p.k, p.idx_tid)
            != (
                first.a_tid,
                first.qx,
                first.sx,
                first.inn,
                first.n,
                first.e_count,
                first.k,
                first.idx_tid,
            )
        {
            return err(format!(
                "{}: the W8A8 projections over banks t{} and t{} disagree about the expert \
                 input or the routing — gate and up are two projections over the SAME input, \
                 and a disagreement is two functions wearing one form",
                f.name, first.weight_tid, p.weight_tid
            ));
        }
    }
    Ok(out)
}

/// The one chain headed by this `linalg.matmul`, proven link by link — or
/// `Ok(None)` when the gemm is not this reader's to claim.
fn w8a8_chain_from(
    k: &KtirNode,
    reads: &ProgramReads<'_>,
    stored: &std::collections::HashSet<Ssa>,
    op: &Operation<'static>,
    consts: &W8a8Consts,
) -> Result<Option<W8a8Claim>, Error> {
    let f = &k.func;
    // The definer of `v`, when it is a `want`-kind op — the link-walk idiom.
    let def_kind = |v: Ssa, want: OpKind| -> Option<&Operation<'static>> {
        let d = reads.def_of(v)?;
        (d.op_type == want).then_some(d)
    };
    // A two-operand clamp/multiply's `(chain link, splatted constant)` halves,
    // whichever way round the producer spelled them. `None` when an operand is
    // missing or neither side is a splat — a computed bound is a different
    // form, not this reader's.
    let halves = |ops: &[Ssa]| -> Option<(Ssa, Ssa)> {
        match ops {
            [a, b] => match (
                reads.splat_value(*a).is_some(),
                reads.splat_value(*b).is_some(),
            ) {
                (false, true) => Some((*a, *b)),
                (true, false) => Some((*b, *a)),
                _ => None,
            },
            _ => None,
        }
    };
    let Some(dot) = op.result else {
        return Ok(None);
    };
    // `linalg.matmul`'s own operand law: `[A, B, init]`.
    let [a, b, init] = op.operands[..] else {
        return Ok(None);
    };
    // The accumulator is the zero splat — a W8A8 gemm starts from scratch.
    if reads.splat_value(init) != Some(0.0) {
        return Ok(None);
    }
    // B is a COLLAPSED GATHERED SLAB: `tensor.collapse_shape` of a `ktdp.load`
    // over a `ktdp.construct_indirect_access_tile`. The router's matmul reads a
    // plain load and fails here — not this reader's, not a refusal.
    let Some(b_collapse) = def_kind(b, OpKind::TensorCollapseShape) else {
        return Ok(None);
    };
    let Some(b_load) = b_collapse
        .operands
        .first()
        .copied()
        .and_then(|v| def_kind(v, OpKind::KtdpLoad))
    else {
        return Ok(None);
    };
    let Some(b_tile) = b_load
        .operands
        .first()
        .copied()
        .and_then(|v| def_kind(v, OpKind::KtdpConstructIndirectAccessTile))
    else {
        return Ok(None);
    };
    let wf = gather_facts_at_tile(k, b_tile)?;
    // TransposeB — the W8A8 contraction is over the slab's own `[n, inn]` view.
    // A neither-form maps is every reader's refusal, propagated.
    if matmul_b_orientation(f, op)? != BOrient::TransposeB {
        return Ok(None);
    }
    // A is the spine's CODES half, kinds only — a break is a near-miss of every
    // reader (the walk refuses it, naming the link).
    let Some(ext) = def_kind(a, OpKind::ArithExtf) else {
        return Ok(None);
    };
    let Some(trunc) = ext
        .operands
        .first()
        .copied()
        .and_then(|v| def_kind(v, OpKind::ArithTruncf))
    else {
        return Ok(None);
    };
    let Some(codes_min) = trunc
        .operands
        .first()
        .copied()
        .and_then(|v| def_kind(v, OpKind::ArithMinnumf))
    else {
        return Ok(None);
    };
    let Some((codes_max_v, hi_c)) = halves(codes_min.operands) else {
        return Ok(None);
    };
    let Some(codes_max) = def_kind(codes_max_v, OpKind::ArithMaxnumf) else {
        return Ok(None);
    };
    let Some((codes_mul_v, lo_c)) = halves(codes_max.operands) else {
        return Ok(None);
    };
    let Some(codes_mul) = def_kind(codes_mul_v, OpKind::ArithMulf) else {
        return Ok(None);
    };
    // The codes multiply: one operand resolves through rank plumbing to the
    // RECIPROCAL (a divf), the other is the row being quantized.
    let inv_side = |v: Ssa| -> bool {
        reads
            .def_of(reads.through_plumbing(v))
            .is_some_and(|d| d.op_type == OpKind::ArithDivf)
    };
    let (row, codes_bcast) = match (
        inv_side(codes_mul.operands[0]),
        inv_side(codes_mul.operands[1]),
    ) {
        (true, false) => (codes_mul.operands[1], codes_mul.operands[0]),
        (false, true) => (codes_mul.operands[0], codes_mul.operands[1]),
        _ => return Ok(None),
    };
    let Some(inv_op) = reads.def_of(reads.through_plumbing(codes_bcast)) else {
        return Ok(None);
    };
    // The reciprocal `1.0 / sx`: one operand the splatted 1, the other the scale.
    let Some((sx, num)) = halves(inv_op.operands) else {
        return Ok(None);
    };
    // The scale's clamp pair: `min(max(amax · INV_QMAX, floor), cap)`.
    let Some(sx_min) = def_kind(sx, OpKind::ArithMinnumf) else {
        return Ok(None);
    };
    let Some((sx_max_v, cap_c)) = halves(sx_min.operands) else {
        return Ok(None);
    };
    let Some(sx_max) = def_kind(sx_max_v, OpKind::ArithMaxnumf) else {
        return Ok(None);
    };
    let Some((sx_mul_v, floor_c)) = halves(sx_max.operands) else {
        return Ok(None);
    };
    let Some(sx_mul) = def_kind(sx_mul_v, OpKind::ArithMulf) else {
        return Ok(None);
    };
    // The amax leg: one operand the reduce, the other the INV_QMAX splat.
    let Some((red_v, amax_c)) = halves(sx_mul.operands) else {
        return Ok(None);
    };
    let Some(reduce) = def_kind(red_v, OpKind::LinalgReduce) else {
        return Ok(None);
    };
    let [red_data, red_seed] = reduce.operands[..] else {
        return Ok(None);
    };
    let Some(abs) = def_kind(red_data, OpKind::MathAbsf) else {
        return Ok(None);
    };
    // ⭐ ONE SPINE, ONE ROW: the amax's `abs` reads the SAME value the codes
    // multiply reads — the identity that makes the scale the row's own. A
    // mismatch is two different activations wearing one chain: not this
    // reader's, not a refusal.
    if abs.operands.first() != Some(&row) {
        return Ok(None);
    }
    // ⛔⛔⛔ CLAIM POINT — PROVENANCE. The spine's root must be a load of a
    // parameter the program never stores to: the EXPERT INPUT. A spine rooting
    // at a STORED parameter is the down projection's reload — not this reader's
    // (its own recogniser is pending, and the discriminator must not depend on
    // shapes that change at M > 1); a root that is not a plain parameter load
    // at all is a computed activation this form does not state.
    let Some(row_load) = def_kind(row, OpKind::KtdpLoad) else {
        return Ok(None);
    };
    let Some(ptr) = plain_load_param_ptr(reads, row_load) else {
        return Ok(None);
    };
    if stored.contains(&ptr) {
        return Ok(None);
    }
    let Some(arg_pos) = f.arguments.iter().position(|(p, _)| *p == ptr) else {
        return Ok(None);
    };
    let a_tid = k.bindings[arg_pos].get();
    // FROM HERE THE CHAIN IS CLAIMED, and every failure is a refusal by name:
    // recognisably a W8A8 gate/up longhand, and lowering a near-miss would
    // compute a different function.

    // ⛔ THE CONSTANTS, as the f16 grid states them — any other values are a
    // different quantize law with the same op kinds.
    let want_splat = |v: Ssa, want: f64, what: &str| -> Result<(), Error> {
        match reads.splat_value(v) {
            Some(got) if got == want => Ok(()),
            Some(got) => err(format!(
                "{}: the W8A8 quantize spine's {what} is {got}, not {want} — the card's own \
                 `quantscalepertokenfp8` constant family (the f16 images of 1/448 and 1/448², \
                 the ±448 clamps), and any other value is a different quantize law, refused \
                 rather than fused",
                f.name
            )),
            None => err(format!(
                "{}: the W8A8 quantize spine's {what} is not a splatted constant — nothing \
                 proves it is the constant the primitive states",
                f.name
            )),
        }
    };
    want_splat(amax_c, consts.inv_qmax, "amax scaling (1/448)")?;
    want_splat(floor_c, consts.floor, "scale floor (1/448²)")?;
    want_splat(cap_c, consts.cap, "scale cap (448)")?;
    want_splat(num, consts.one, "reciprocal numerator (1.0)")?;
    want_splat(lo_c, consts.lo, "codes lower clamp (−448)")?;
    want_splat(hi_c, consts.hi, "codes upper clamp (+448)")?;

    // The amax: Max combiner, −inf seed, `[1]` f16 result — the primitive's own
    // semantics. A Max seeded any higher clamps every-negative rows; a Sum is a
    // different reduce entirely.
    if reduce_kind_of(reduce) != Some(ReduceKind::Max) {
        return err(format!(
            "{}: the W8A8 spine's amax reduce is not a Max — the card's \
             `quantscalepertokenfp8` computes the row's maximum magnitude, and any other \
             combiner is a different scale",
            f.name
        ));
    }
    match reads.splat_value(red_seed) {
        Some(f64::NEG_INFINITY) => {}
        got => {
            return err(format!(
                "{}: the W8A8 spine's amax reduce seeds {got:?}, not −inf — a Max seeded any \
                 higher clamps every-negative row's scale and is a different function than the \
                 primitive's",
                f.name
            ));
        }
    }
    let red_ok = match reduce.result_type {
        Some(ktir_core::irtype::IrType::Tensor { dims, elem }) => {
            dims.len() == 1 && dims[0] == 1 && elem == ktir_core::dtypes::DType::F16
        }
        _ => false,
    };
    if !red_ok {
        return err(format!(
            "{}: the W8A8 spine's amax reduce does not state a `[1]` f16 result — one scale \
             per token row is the primitive's own shape",
            f.name
        ));
    }
    // ⛔ THE ISLAND'S TWO BOUNDARIES: the truncation states `Fp8E4m3` and the
    // widening `F16`. A spine the ladder left at f32 would hand the fused
    // emission constants the card's own primitive never states.
    if trunc.result_type.and_then(|t| t.elem()) != Some(ktir_core::dtypes::DType::Fp8E4m3) {
        return err(format!(
            "{}: the W8A8 spine's codes truncation does not state Fp8E4m3 — the codes are the \
             values the fp8 contraction reads, and any other element type is a different \
             function",
            f.name
        ));
    }
    if ext.result_type.and_then(|t| t.elem()) != Some(ktir_core::dtypes::DType::F16) {
        return err(format!(
            "{}: the W8A8 spine's codes widening does not state f16 — the island is narrowed \
             back before this door, or the contraction would read a different precision than \
             the card's own W8A8",
            f.name
        ));
    }

    // ⛔ DESTRUCTION IS EXCLUSIVE — every destroyed link has exactly the one
    // reader the chain states.
    let sole = |v: Ssa, want: OpKind, what: &str| -> Result<(), Error> {
        match reads.readers_of(v) {
            [p] if reads.op_at(*p).op_type == want => Ok(()),
            [p] => err(format!(
                "{}: the W8A8 quantize spine's {what} is read by `{:?}` — the fusion destroys \
                 it, so exactly the chain's next link may read it",
                f.name,
                reads.op_at(*p).op_type
            )),
            rs => err(format!(
                "{}: the W8A8 quantize spine's {what} is read by {} ops — the fusion destroys \
                 it, so exactly the chain's next link may read it",
                f.name,
                rs.len()
            )),
        }
    };
    let (Some(abs_v), Some(trunc_v), Some(ext_v), Some(codes_min_v)) =
        (abs.result, trunc.result, ext.result, codes_min.result)
    else {
        return err(format!(
            "{}: a W8A8 spine link states no result — not reachable through the def/use index",
            f.name
        ));
    };
    let Some(inv_v) = inv_op.result else {
        return err(format!(
            "{}: a W8A8 spine link states no result — not reachable through the def/use index",
            f.name
        ));
    };
    sole(abs_v, OpKind::LinalgReduce, "absolute value")?;
    sole(red_v, OpKind::ArithMulf, "amax")?;
    sole(sx_mul_v, OpKind::ArithMaxnumf, "amax scaling")?;
    sole(sx_max_v, OpKind::ArithMinnumf, "scale floor")?;
    sole(codes_mul_v, OpKind::ArithMaxnumf, "codes scaling")?;
    sole(codes_max_v, OpKind::ArithMinnumf, "codes lower clamp")?;
    sole(codes_min_v, OpKind::ArithTruncf, "codes upper clamp")?;
    sole(trunc_v, OpKind::ArithExtf, "fp8 codes")?;
    sole(b, OpKind::LinalgMatmul, "collapsed slab")?;
    // `sx` SURVIVES as a value the chain never materialises (the emission's
    // quantize recomputes the scale as the primitive's own output), so its
    // readers are the reciprocal and rank plumbing only.
    for &p in reads.readers_of(sx) {
        let r = reads.op_at(p);
        let ok = (r.op_type == OpKind::ArithDivf && r.result == Some(inv_v))
            || matches!(
                r.op_type,
                OpKind::TensorExpandShape | OpKind::TensorCollapseShape | OpKind::LinalgBroadcast
            );
        if !ok {
            return err(format!(
                "{}: the W8A8 spine's per-token scale is read by `{:?}` — the fusion keeps it \
                 only as the dequant's broadcast and the reciprocal's denominator",
                f.name, r.op_type
            ));
        }
    }
    // The reciprocal's readers: rank plumbing only (the codes' broadcast chain).
    for &p in reads.readers_of(inv_v) {
        let r = reads.op_at(p);
        if !matches!(
            r.op_type,
            OpKind::TensorExpandShape | OpKind::TensorCollapseShape | OpKind::LinalgBroadcast
        ) {
            return err(format!(
                "{}: the W8A8 spine's reciprocal is read by `{:?}` — the fusion destroys it in \
                 the codes, and only the broadcast chain may carry it there",
                f.name, r.op_type
            ));
        }
    }

    // The dequant's first leg: the dot's ONE reader, an `arith.mulf` whose other
    // operand resolves through rank plumbing to `sx`.
    let Some(mul_s_op) = reads
        .sole_reader(dot)
        .filter(|o| o.op_type == OpKind::ArithMulf)
    else {
        return err(format!(
            "{}: the W8A8 gemm's product {dot:?} is not read by exactly one `arith.mulf` — the \
             chain's dequant leg (`raw · broadcast(sx)`) is the only form this fusion states",
            f.name
        ));
    };
    let Some(mul_s_v) = mul_s_op.result else {
        return err(format!(
            "{}: a W8A8 chain link states no result — not reachable through the def/use index",
            f.name
        ));
    };
    let Some(ms_bcast) = mul_s_op.operands.iter().copied().find(|&v| v != dot) else {
        return err(format!(
            "{}: the W8A8 gemm's dequant multiply names no operand beside its product",
            f.name
        ));
    };
    if reads.through_plumbing(ms_bcast) != sx {
        return err(format!(
            "{}: the W8A8 gemm's dequant multiplies the product by a value that is not the \
             spine's per-token scale — the W8A8 form dequantises with `sx` (the quantize's own \
             scale), and any other factor is a different function",
            f.name
        ));
    }
    // The terminal: the first leg's ONE reader, an `arith.mulf` whose other
    // operand is the pair's gathered ROUTER scale.
    let Some(term_op) = reads
        .sole_reader(mul_s_v)
        .filter(|o| o.op_type == OpKind::ArithMulf)
    else {
        return err(format!(
            "{}: the W8A8 gemm's scaled product is not read by exactly one `arith.mulf` — the \
             chain's terminal (`· the pair's router scale`) is the only form this fusion states",
            f.name
        ));
    };
    let Some(terminal) = term_op.result else {
        return err(format!(
            "{}: a W8A8 chain link states no result — not reachable through the def/use index",
            f.name
        ));
    };
    // ⛔ NO STORE READS A TERMINAL. The fusion writes the projection's output
    // intermediate and the walk hands consumers the terminal as a window of it;
    // a `ktdp.store` of a terminal is a write the walk's hook would silently
    // drop, so it is refused rather than dropped.
    if reads
        .readers_of(terminal)
        .iter()
        .any(|&p| reads.op_at(p).op_type == OpKind::KtdpStore)
    {
        return err(format!(
            "{}: the W8A8 gemm's output {terminal:?} is stored by the program — the fusion \
             writes its projection's output intermediate, and a stored terminal is a write the \
             walk would silently drop",
            f.name
        ));
    }
    let Some(scale_operand) = term_op.operands.iter().copied().find(|&v| v != mul_s_v) else {
        return err(format!(
            "{}: the W8A8 gemm's terminal multiply names no operand beside its scaled product",
            f.name
        ));
    };
    let Some(scale_load) = def_kind(scale_operand, OpKind::KtdpLoad) else {
        return err(format!(
            "{}: the W8A8 gemm's terminal multiplies by a value that is not a load — the \
             router's per-pair scale is a gathered load, and a computed factor is a different \
             function",
            f.name
        ));
    };
    let Some(scale_tile) = scale_load
        .operands
        .first()
        .copied()
        .and_then(|v| def_kind(v, OpKind::KtdpConstructIndirectAccessTile))
    else {
        return err(format!(
            "{}: the W8A8 gemm's router scale is not gathered — the pair's scale is selected by \
             the same index that selects its slab, and a plain load is a different program",
            f.name
        ));
    };
    let sf = gather_facts_at_tile(k, scale_tile)?;
    // Same index parameter, same anchor — the pair's OWN scale, not another
    // pair's.
    if sf.index_tid != wf.index_tid || sf.first_entry != wf.first_entry {
        return err(format!(
            "{}: the W8A8 gemm's dequant reads its pair's router scale through a different \
             index anchor than its weight slab — one pair's two gathers anchor at ONE index word",
            f.name
        ));
    }

    // The geometry, read off the two views the program states. The weight bank is
    // the RAW `[E, n, inn]` fp8 parameter (the emission's own refusal guards the
    // staging forms it cannot address); the scale bank is `[E, n]` f16.
    let Some(w_view) = b_tile
        .operands
        .first()
        .copied()
        .and_then(|v| reads.def_of(v))
    else {
        return err(format!(
            "{}: the W8A8 gemm's weight tile names a view nothing defines",
            f.name
        ));
    };
    let (e_count, n, inn) = match w_view.result_type {
        Some(ktir_core::irtype::IrType::MemRef { dims, elem })
            if elem == ktir_core::dtypes::DType::Fp8E4m3
                && dims.len() == 3
                && dims.iter().all(|&d| d > 0) =>
        {
            (dims[0] as u32, dims[1] as u32, dims[2] as u32)
        }
        _ => {
            return err(format!(
                "{}: the W8A8 gemm's weight bank t{} is not an `[E, n, in]` Fp8E4m3 view — the \
                 fused door reads the program's own expert-major bank, and any other layout is \
                 the staged-bank machinery it refuses by name",
                f.name, wf.value_tid
            ));
        }
    };
    let Some(s_view) = scale_tile
        .operands
        .first()
        .copied()
        .and_then(|v| reads.def_of(v))
    else {
        return err(format!(
            "{}: the W8A8 gemm's scale tile names a view nothing defines",
            f.name
        ));
    };
    let s_ok = match s_view.result_type {
        Some(ktir_core::irtype::IrType::MemRef { dims, elem }) => {
            elem == ktir_core::dtypes::DType::F16 && dims == [e_count as i64, n as i64]
        }
        _ => false,
    };
    if !s_ok {
        return err(format!(
            "{}: the W8A8 gemm's scale bank t{} is not the weight bank's own `[E, n]` f16 view \
             — the per-output-channel scales ride the same expert count and projection width",
            f.name, sf.value_tid
        ));
    }
    let dot_ok = match op.result_type {
        Some(ktir_core::irtype::IrType::Tensor { dims, elem }) => {
            elem == ktir_core::dtypes::DType::F16 && dims == [1, n as i64]
        }
        _ => false,
    };
    if !dot_ok {
        return err(format!(
            "{}: the W8A8 gemm's product does not state a `[1, n]` f16 result at the bank's own \
             n={n}",
            f.name
        ));
    }
    // ⛔ M=1 ONLY — the expert input's own row count. The fused W8A8 is the
    // decode form; M > 1 is the per-`Program` door's gathered prefill body and
    // is refused here by name rather than lowered as the nearest thing.
    let Some(a_region) = region_for_operand(k, row)? else {
        return err(format!(
            "{}: the W8A8 gemm's expert input does not resolve to a parameter region — the \
             spine rooted at a plain parameter load, so the region walk that refused it is the \
             defect to read",
            f.name
        ));
    };
    if a_region.c_len != inn {
        return err(format!(
            "{}: the W8A8 gemm's expert input t{} is {} columns wide but the weight bank t{} \
             contracts {inn} — the activation's `in` must match the bank's",
            f.name, a_region.tid, a_region.c_len, wf.value_tid
        ));
    }
    if a_region.v_rows != 1 {
        return err(format!(
            "{}: the W8A8 gemm chain's expert input t{} is {} rows wide — this door's fused \
             W8A8 is the M=1 (decode) form, and M > 1 is refused by name rather than lowered as \
             the nearest thing (the per-`Program` expert door's gathered m>1 body is the \
             prefill form)",
            f.name, a_region.tid, a_region.v_rows
        ));
    }
    Ok(Some(W8a8Claim {
        chain: W8a8Chain {
            pair: wf.first_entry as u32,
            weight_tile: wf.tile,
            collapsed: b,
            dot,
            mul_s: mul_s_v,
            terminal,
            scale_tile: sf.tile,
        },
        weight_tid: wf.value_tid,
        scale_tid: sf.value_tid,
        a_load: row,
        a_tid,
        qx: a,
        sx,
        idx_tid: wf.index_tid,
        anchor: wf.first_entry,
        e_count,
        n,
        inn,
        spine_consumed: [
            abs_v,
            red_v,
            sx_mul_v,
            sx_max_v,
            sx,
            inv_v,
            codes_mul_v,
            codes_max_v,
            codes_min_v,
            trunc_v,
            ext_v,
        ],
    }))
}

/// One claimed down gemm: the links the down fusion destroys, and the two tiles
/// it claims out of the gathers census — the gate/up chain's own fields, read at
/// the projection the program routes BACKWARD (the router's ids select the slabs;
/// the act windows select the spines).
///
/// ⭐ THE TERMINAL IS NOT CONSUMED — same law as [`W8a8Chain`]'s: it is the value
/// the combine multiply reads (pair `p`'s `[1, n]` down-projection row), resolved
/// by the walk as a WINDOW of the projection's output intermediate.
#[derive(Clone, Copy, Debug)]
pub struct W8a8DownChain {
    /// The pair this chain serves — its anchor word in the index buffer AND its
    /// window index in the act buffer, proven EQUAL by the per-pair agreement.
    pub pair: u32,
    /// The weight tile — this chain's claim out of [`gathers_of`]'s census
    /// (`GatherFacts::tile`'s identity, never a position).
    pub weight_tile: Ssa,
    /// The `tensor.collapse_shape` that flattens the gathered slab to `[n, inn]`.
    pub collapsed: Ssa,
    /// The `linalg.matmul` — the fp8 contraction the fused emission replaces.
    pub dot: Ssa,
    /// The dequant's first leg: `dot · broadcast(sa_p)` — THIS pair's own scale.
    pub mul_s: Ssa,
    /// The dequant's terminal: `mul_s · the pair's gathered weight scale` — the
    /// chain's OUTPUT value.
    pub terminal: Ssa,
    /// The weight-scale tile — the chain's second claim out of the census.
    pub scale_tile: Ssa,
}

/// One down projection: every pair's chain over one `(weight, scale)` bank pair,
/// plus the facts the per-pair emissions need.
#[derive(Clone, Debug)]
pub struct W8a8DownProjection {
    /// The act buffer's tensor id — the round-tripped gelu output whose per-pair
    /// `[1, inn]` windows the per-pair quantizes read; the chain's tail tensors
    /// are keyed off it (the `Fq*` synth family, the same law the dense chain's
    /// are keyed off its activation).
    pub a_tid: u32,
    /// The routing parameter's tensor id — the XIdx landing is keyed off it.
    pub idx_tid: u32,
    /// The weight bank's tensor id (the `[E, n, inn]` fp8 parameter).
    pub weight_tid: u32,
    /// The scale bank's tensor id (the `[E, n]` f16 parameter).
    pub scale_tid: u32,
    /// The top-k width — one chain per index word, proven `0..k` complete.
    pub k: u32,
    /// The contraction width (`inn`, the gated act's per-pair width).
    pub inn: u32,
    /// The projection's output width per pair.
    pub n: u32,
    /// The experts the bank holds (`E`).
    pub e_count: u32,
    /// The chains, sorted by pair.
    pub chains: Vec<W8a8DownChain>,
    /// The FIRST terminal in program order — the op the walk mints the output
    /// intermediate at. Every terminal (this one included) then emits its OWN
    /// pair's legs there: the per-position law, because program order is tape
    /// order is execution order and the unrolled kernel interleaves
    /// `gelu_j → down_j → combine_j` per pair.
    pub mint_at: Ssa,
    /// Everything the fusion destroys: each chain's OWN eleven spine ops (the
    /// down form states one whole quantize per pair — no spine is shared) plus
    /// its three interior links. The terminals are deliberately absent — they
    /// are the projection's output windows.
    pub consumed: Vec<Ssa>,
}

/// The per-chain claim, before grouping — the down recogniser's working state.
struct W8a8DownClaim {
    chain: W8a8DownChain,
    weight_tid: u32,
    scale_tid: u32,
    a_tid: u32,
    idx_tid: u32,
    anchor: i64,
    e_count: u32,
    n: u32,
    inn: u32,
    /// THIS chain's eleven destroyed spine ops — the down form's own law: every
    /// pair quantizes its own act window (its own amax, its own reciprocal, its
    /// own codes), so no spine is shared and every claim carries its own list.
    spine_consumed: [Ssa; 11],
}

/// THE W8A8 DOWN SLAB GEMMS, as the per-pair fused emissions they are — the
/// gate/up reader's own chain form, read at the projection whose spine ROOTS AT
/// THE ROUND-TRIPPED ACT.
///
/// # WHY THIS RECOGNISER EXISTS
///
/// `kernels/moe.py`'s down half spells each pair's projection as the SAME
/// longhand the gate/up reader claims — the eleven-op quantize spine, the
/// gathered fp8 slab gemm, the two-multiply dequant — but over a RELOAD of the
/// act scratch the program round-trips: `gelu(gate)·up` is stored to
/// `desc_act`'s window `j·I`, and pair `j`'s spine reloads exactly that window
/// and quantizes it at its OWN scale (`sa_j` — one scale per `(token, slot)`
/// row, the file's own stated law, so a prefix cached by either path computes
/// the same numbers). Lowered op-by-op that is eleven pointwise ops and a PLAIN
/// f16 matmul per pair — a different algorithm, not a slower one — and at the
/// whole-function door the per-pair slab gathers refuse the gathers census
/// besides. This reader claims the chains whole, so [`lower_function`] emits
/// each pair's legs at its own terminal through
/// [`super::lower_ktir_to_superdsc::expert_down_fused_m1`].
///
/// # THE PROOF, EVERY FACT READ FROM THE PROGRAM
///
/// Every law the gate/up reader states, verbatim — the collapsed gathered slab,
/// TransposeB, the f16-grid constants, the Max/−inf/`[1]`-f16 amax, the one-row
/// spine, exclusive destruction, the `[E, n, inn]`/`[E, n]` bank geometry, the
/// M=1 row count — PLUS the three facts that make a down chain a down chain:
///
/// * **PROVENANCE, INVERTED** — the spine roots at a load of a parameter the
///   program STORES to (the gate/up reader returns `Ok(None)` on exactly this
///   spine; the two readers partition the matmuls by where the spine roots, and
///   neither depends on a shape proxy that misfires the day `M > 1` is
///   admitted);
/// * **ROOT-THROUGH-STORE** — the load's own tile window is written by a
///   `ktdp.store` of a CLAIMED GELU TERMINAL through the SAME window (the
///   store-arm law's own tile reader, so the two readings of one window cannot
///   drift): the round trip is proven, not assumed, and the gelu emission's
///   output is what the quantize reads;
/// * **PER-PAIR AGREEMENT** — the window IS the pair: the chain that gathers
///   its expert through index word `anchor` reads the act window at
///   `anchor · inn` (moe.py's `j` is both the index word `desc_ids + m·K + j`
///   and the window index `act_desc.load([m, j·I])`). A mismatch is two pairs
///   wearing one chain — pair `p`'s scale would quantize pair `q`'s activation.
///
/// # ⛔ NO SHARED SPINE — AND WHY THAT IS THE ORDERING LAW
///
/// The gate/up form quantizes ONE expert input ONCE (every pair's gemm reads
/// the same codes), so its emission fires whole at the first terminal. The down
/// form CANNOT: each pair's codes are its own (its own `sa_j`), and the program
/// interleaves the pairs — `gelu_j`'s act store, pair `j`'s down chain,
/// `combine_j`'s read of the terminal, then pair `j+1` — because that is the
/// order the unrolled kernel states it. A single fire that lowered the whole
/// projection at the FIRST terminal would read act windows whose gelu writers
/// have not run (pairs `1..k−1` — garbage on card, not a slower answer) or
/// write out windows the combine has already read; the shared clip/convert has
/// no legal position at all (it needs every pair's scaled act before it and
/// every matmul after it, but pair 0's matmul precedes pair 1's gelu in program
/// order). So this reader claims all `k` chains at once and the walk emits each
/// pair's whole leg set AT THAT PAIR'S OWN TERMINAL — the one position between
/// its gelu store and its combine read. See `expert_down_fused_m1`'s own doc
/// for the leg census and the equivalence pin that guards the numerics.
///
/// A chain that almost matches is REFUSED BY NAME, never lowered as the nearest
/// thing — and a chain that fails BEFORE the provenance check is left to the
/// walk, which refuses it with the message that names the link: this reader
/// claims a chain only whole.
pub fn program_w8a8_down_chains(
    k: &KtirNode,
    gelus: &[GeluMulChain],
) -> Result<Vec<W8a8DownProjection>, Error> {
    let f = &k.func;
    let reads = program_reads(f);
    // The constants, as the f16 grid states them — the SAME family the gate/up
    // reader measures (the helpers are shared verbatim; only the quantized row
    // differs).
    let f16_image = |x: f64| {
        ktir_core::codec::f16_units_to_f32(&ktir_core::codec::f32_to_f16_units(&[x as f32]))[0]
            as f64
    };
    let consts = W8a8Consts {
        inv_qmax: f16_image(1.0 / 448.0),
        floor: f16_image(1.0 / 448.0 / 448.0),
        cap: f16_image(448.0),
        one: 1.0,
        lo: f16_image(-448.0),
        hi: f16_image(448.0),
    };
    // ⛔ THE PARAMETERS THIS PROGRAM STORES TO ARE ITS SCRATCH — the same set
    // the gate/up reader builds; here it is the provenance the claim REQUIRES.
    let stored: std::collections::HashSet<Ssa> = reads
        .ops()
        .iter()
        .copied()
        .filter(|o| o.op_type == OpKind::KtdpStore)
        .filter_map(|s| store_param_ptr(&reads, s))
        .collect();
    let mut claims: Vec<W8a8DownClaim> = Vec::new();
    for op in reads
        .ops()
        .iter()
        .copied()
        .filter(|o| o.op_type == OpKind::LinalgMatmul)
    {
        if let Some(claim) = w8a8_down_chain_from(k, &reads, &stored, gelus, op, &consts)? {
            claims.push(claim);
        }
    }
    if claims.is_empty() {
        return Ok(Vec::new());
    }
    // ⛔ EACH PAIR'S CODES ARE READ ONLY BY ITS OWN DOT. The down form states one
    // whole quantize per pair, so a pair's codes have exactly that pair's gemm as
    // their reader — a stray reader would read a value the fusion destroys (the
    // fp8 contraction reads the codes on the card).
    for c in &claims {
        for &p in reads.readers_of(c.spine_consumed[10]) {
            let r = reads.op_at(p);
            if !(r.op_type == OpKind::LinalgMatmul && r.result == Some(c.chain.dot)) {
                return err(format!(
                    "{}: the W8A8 DOWN quantize spine's codes are read by `{:?}` — the fusion \
                     DESTROYS them (the fp8 contraction reads them on the card), and the down \
                     form gives each pair's codes exactly that pair's own gemm as their reader",
                    f.name, r.op_type
                ));
            }
        }
    }
    // Group by (weight, scale) bank pair — one projection per bank pair, and the
    // anchors law is stated per projection.
    let mut groups: Vec<((u32, u32), Vec<&W8a8DownClaim>)> = Vec::new();
    for c in &claims {
        let key = (c.weight_tid, c.scale_tid);
        match groups.iter_mut().find(|(g, _)| *g == key) {
            Some((_, v)) => v.push(c),
            None => groups.push((key, vec![c])),
        }
    }
    let mut out = Vec::with_capacity(groups.len());
    for ((weight_tid, scale_tid), group) in groups {
        let first = group[0];
        // ⛔ THE ANCHORS ARE EXACTLY `0..k` — the whole top-k, one gemm per index
        // word. A gap is a routing shape this door cannot address; a repeat is
        // two gemms on one pair.
        let mut anchors: Vec<i64> = group.iter().map(|c| c.anchor).collect();
        anchors.sort_unstable();
        let k = anchors.len() as u32;
        if anchors != (0..k as i64).collect::<Vec<_>>() {
            return err(format!(
                "{}: the W8A8 DOWN gemm chains over the weight bank t{weight_tid} anchor at \
                 {anchors:?} — the pairs of one projection are the whole top-k, one gemm per \
                 index word 0..k, and a gap or a repeat is a routing shape this door cannot \
                 address",
                f.name
            ));
        }
        // The geometry agrees across the group (each chain read its own views).
        // ⛔ AND NO SPINE EQUALITY — the down form has none to state: every pair
        // carries its own quantize, so `(a_load, qx, sx)` DISAGREEING across
        // chains is the form working as designed, not a defect.
        for c in &group[1..] {
            if (c.e_count, c.n, c.inn) != (first.e_count, first.n, first.inn) {
                return err(format!(
                    "{}: two W8A8 DOWN gemm chains over the weight bank t{weight_tid} state \
                     DIFFERENT bank geometries — one projection is one bank",
                    f.name
                ));
            }
        }
        let mut chains: Vec<W8a8DownChain> = group.iter().map(|c| c.chain).collect();
        chains.sort_by_key(|c| c.pair);
        // The output intermediate mints at the group's FIRST terminal in program
        // order; every terminal then emits its own pair's legs there.
        let mut mint_at = chains[0].terminal;
        let mut mint_pos = usize::MAX;
        for c in &chains {
            let Some(p) = reads.pos_of_result(c.terminal) else {
                return err(format!(
                    "{}: a claimed W8A8 DOWN terminal has no defining position — the claim \
                     walked it from its own definer, so this is not reachable",
                    f.name
                ));
            };
            if p < mint_pos {
                mint_pos = p;
                mint_at = c.terminal;
            }
        }
        // Each chain's OWN spine plus its three destroyed links — the terminals
        // are deliberately absent (they are the projection's output windows).
        let mut consumed = Vec::new();
        for c in group {
            consumed.extend(c.spine_consumed);
            consumed.extend([c.chain.collapsed, c.chain.dot, c.chain.mul_s]);
        }
        out.push(W8a8DownProjection {
            a_tid: first.a_tid,
            idx_tid: first.idx_tid,
            weight_tid,
            scale_tid,
            k,
            inn: first.inn,
            n: first.n,
            e_count: first.e_count,
            chains,
            mint_at,
            consumed,
        });
    }
    // ⭐ ONE ACT BUFFER, ONE ROUTING, ONE GEOMETRY — the down projections of one
    // program round-trip ONE act buffer and read ONE routing. A disagreement is
    // two different functions wearing one form, refused rather than adjudicated.
    for p in out.iter().skip(1) {
        let first = &out[0];
        if (p.a_tid, p.inn, p.n, p.e_count, p.k, p.idx_tid)
            != (
                first.a_tid,
                first.inn,
                first.n,
                first.e_count,
                first.k,
                first.idx_tid,
            )
        {
            return err(format!(
                "{}: the W8A8 DOWN projections over banks t{} and t{} disagree about the act \
                 buffer or the routing — one program round-trips ONE act buffer, and a \
                 disagreement is two functions wearing one form",
                f.name, first.weight_tid, p.weight_tid
            ));
        }
    }
    Ok(out)
}

/// The one DOWN chain headed by this `linalg.matmul`, proven link by link — or
/// `Ok(None)` when the gemm is not this reader's to claim. The gate/up reader's
/// own walk, verbatim through the spine, with the claim point INVERTED and the
/// round trip proven at the root.
fn w8a8_down_chain_from(
    k: &KtirNode,
    reads: &ProgramReads<'_>,
    stored: &std::collections::HashSet<Ssa>,
    gelus: &[GeluMulChain],
    op: &Operation<'static>,
    consts: &W8a8Consts,
) -> Result<Option<W8a8DownClaim>, Error> {
    let f = &k.func;
    // The definer of `v`, when it is a `want`-kind op — the link-walk idiom.
    let def_kind = |v: Ssa, want: OpKind| -> Option<&Operation<'static>> {
        let d = reads.def_of(v)?;
        (d.op_type == want).then_some(d)
    };
    // A two-operand clamp/multiply's `(chain link, splatted constant)` halves,
    // whichever way round the producer spelled them.
    let halves = |ops: &[Ssa]| -> Option<(Ssa, Ssa)> {
        match ops {
            [a, b] => match (
                reads.splat_value(*a).is_some(),
                reads.splat_value(*b).is_some(),
            ) {
                (false, true) => Some((*a, *b)),
                (true, false) => Some((*b, *a)),
                _ => None,
            },
            _ => None,
        }
    };
    let Some(dot) = op.result else {
        return Ok(None);
    };
    // `linalg.matmul`'s own operand law: `[A, B, init]`.
    let [a, b, init] = op.operands[..] else {
        return Ok(None);
    };
    // The accumulator is the zero splat — a W8A8 gemm starts from scratch.
    if reads.splat_value(init) != Some(0.0) {
        return Ok(None);
    }
    // B is a COLLAPSED GATHERED SLAB, kinds only — the same three links the
    // gate/up reader walks.
    let Some(b_collapse) = def_kind(b, OpKind::TensorCollapseShape) else {
        return Ok(None);
    };
    let Some(b_load) = b_collapse
        .operands
        .first()
        .copied()
        .and_then(|v| def_kind(v, OpKind::KtdpLoad))
    else {
        return Ok(None);
    };
    let Some(b_tile) = b_load
        .operands
        .first()
        .copied()
        .and_then(|v| def_kind(v, OpKind::KtdpConstructIndirectAccessTile))
    else {
        return Ok(None);
    };
    let wf = gather_facts_at_tile(k, b_tile)?;
    // TransposeB — the W8A8 contraction is over the slab's own `[n, inn]` view.
    if matmul_b_orientation(f, op)? != BOrient::TransposeB {
        return Ok(None);
    }
    // A is the spine's CODES half, kinds only — the same chain the gate/up
    // reader walks: extf ← truncf ← the clamp pair ← the scale multiply.
    let Some(ext) = def_kind(a, OpKind::ArithExtf) else {
        return Ok(None);
    };
    let Some(trunc) = ext
        .operands
        .first()
        .copied()
        .and_then(|v| def_kind(v, OpKind::ArithTruncf))
    else {
        return Ok(None);
    };
    let Some(codes_min) = trunc
        .operands
        .first()
        .copied()
        .and_then(|v| def_kind(v, OpKind::ArithMinnumf))
    else {
        return Ok(None);
    };
    let Some((codes_max_v, hi_c)) = halves(codes_min.operands) else {
        return Ok(None);
    };
    let Some(codes_max) = def_kind(codes_max_v, OpKind::ArithMaxnumf) else {
        return Ok(None);
    };
    let Some((codes_mul_v, lo_c)) = halves(codes_max.operands) else {
        return Ok(None);
    };
    let Some(codes_mul) = def_kind(codes_mul_v, OpKind::ArithMulf) else {
        return Ok(None);
    };
    // The codes multiply: one operand resolves through rank plumbing to the
    // RECIPROCAL (a divf), the other is the row being quantized.
    let inv_side = |v: Ssa| -> bool {
        reads
            .def_of(reads.through_plumbing(v))
            .is_some_and(|d| d.op_type == OpKind::ArithDivf)
    };
    let (row, codes_bcast) = match (
        inv_side(codes_mul.operands[0]),
        inv_side(codes_mul.operands[1]),
    ) {
        (true, false) => (codes_mul.operands[1], codes_mul.operands[0]),
        (false, true) => (codes_mul.operands[0], codes_mul.operands[1]),
        _ => return Ok(None),
    };
    let Some(inv_op) = reads.def_of(reads.through_plumbing(codes_bcast)) else {
        return Ok(None);
    };
    // The reciprocal `1.0 / sa_p`: one operand the splatted 1, the other the
    // pair's own scale.
    let Some((sx, num)) = halves(inv_op.operands) else {
        return Ok(None);
    };
    // The scale's clamp pair: `min(max(amax · INV_QMAX, floor), cap)`.
    let Some(sx_min) = def_kind(sx, OpKind::ArithMinnumf) else {
        return Ok(None);
    };
    let Some((sx_max_v, cap_c)) = halves(sx_min.operands) else {
        return Ok(None);
    };
    let Some(sx_max) = def_kind(sx_max_v, OpKind::ArithMaxnumf) else {
        return Ok(None);
    };
    let Some((sx_mul_v, floor_c)) = halves(sx_max.operands) else {
        return Ok(None);
    };
    let Some(sx_mul) = def_kind(sx_mul_v, OpKind::ArithMulf) else {
        return Ok(None);
    };
    // The amax leg: one operand the reduce, the other the INV_QMAX splat.
    let Some((red_v, amax_c)) = halves(sx_mul.operands) else {
        return Ok(None);
    };
    let Some(reduce) = def_kind(red_v, OpKind::LinalgReduce) else {
        return Ok(None);
    };
    let [red_data, red_seed] = reduce.operands[..] else {
        return Ok(None);
    };
    let Some(abs) = def_kind(red_data, OpKind::MathAbsf) else {
        return Ok(None);
    };
    // ⭐ ONE SPINE, ONE ROW: the amax's `abs` reads the SAME value the codes
    // multiply reads — the identity that makes the scale the row's own.
    if abs.operands.first() != Some(&row) {
        return Ok(None);
    }
    // ⛔⛔⛔ CLAIM POINT — PROVENANCE, INVERTED. The spine's root must be a load
    // of a parameter the program STORES to: the act scratch the kernel
    // round-trips its gated act through, pair by pair. A spine rooting at a
    // parameter NOTHING stores is the gate/up reader's expert input — not this
    // reader's (that reader returns `Ok(None)` on exactly the opposite test, so
    // the two readers partition the matmuls by where the spine roots); a root
    // that is not a plain parameter load at all is a computed activation this
    // form does not state.
    let Some(row_load) = def_kind(row, OpKind::KtdpLoad) else {
        return Ok(None);
    };
    let Some(ptr) = plain_load_param_ptr(reads, row_load) else {
        return Ok(None);
    };
    if !stored.contains(&ptr) {
        return Ok(None);
    }
    let Some(arg_pos) = f.arguments.iter().position(|(p, _)| *p == ptr) else {
        return Ok(None);
    };
    let a_tid = k.bindings[arg_pos].get();
    // ⛔ ROOT-THROUGH-STORE — the round trip itself. The load's own tile window
    // must be written by a `ktdp.store` of a CLAIMED GELU TERMINAL through the
    // SAME window: the program computes `gelu(gate)·up`, stores it to the act
    // buffer's window `j·I`, and this chain reloads exactly that window. The
    // store's window is read with the SAME tile reader the walk's store arm
    // uses, so the two readings of one window cannot drift (the own-tile law).
    let Some(row_tile_v) = row_load.operands.first().copied() else {
        return Ok(None);
    };
    let Some(row_tile) = reads.def_of(row_tile_v) else {
        return Ok(None);
    };
    let Some(row_win) = tile_window(f, row_tile, "the W8A8 DOWN spine's act load")? else {
        return err(format!(
            "{}: the W8A8 DOWN spine's act load reads through a tile that states no corner \
             window — the per-pair act window is the fact this fusion addresses, and a \
             windowless tile cannot carry it",
            f.name
        ));
    };
    let mut writer: Option<&Operation<'static>> = None;
    for s in reads
        .ops()
        .iter()
        .copied()
        .filter(|s| s.op_type == OpKind::KtdpStore)
    {
        if store_param_ptr(reads, s) != Some(ptr) {
            continue;
        }
        let Some(tile_v) = s.operands.get(1).copied() else {
            continue;
        };
        let Some(tile) = reads.def_of(tile_v) else {
            continue;
        };
        let win = tile_window(f, tile, "the W8A8 DOWN spine's act store")?;
        if win == Some(row_win) {
            writer = Some(s);
            break;
        }
    }
    let Some(writer) = writer else {
        return err(format!(
            "{}: the W8A8 DOWN spine's act window {row_win:?} is written by no `ktdp.store` of \
             the parameter t{a_tid} — the down form round-trips the gated act through the act \
             buffer (the gelu chain's own output, stored at window `j·I` and reloaded by the \
             pair's quantize), and a window nothing stores is a value this door cannot prove \
             the provenance of",
            f.name
        ));
    };
    let Some(&stored_value) = writer.operands.first() else {
        return err(format!(
            "{}: the W8A8 DOWN spine's act store states no stored value — not reachable \
             through the def/use index",
            f.name
        ));
    };
    if !gelus.iter().any(|g| g.mul == stored_value) {
        return err(format!(
            "{}: the W8A8 DOWN spine's act window is stored by a value that is no claimed gelu \
             terminal — the down projection quantizes the gated act (the gelu chain's own \
             output), and any other producer is a different function",
            f.name
        ));
    }
    // FROM HERE THE CHAIN IS CLAIMED, and every failure is a refusal by name:
    // recognisably a W8A8 down longhand, and lowering a near-miss would compute
    // a different function.

    // ⛔ THE CONSTANTS, as the f16 grid states them — the same family the gate/up
    // reader checks (the helpers are shared verbatim), any other values are a
    // different quantize law with the same op kinds.
    let want_splat = |v: Ssa, want: f64, what: &str| -> Result<(), Error> {
        match reads.splat_value(v) {
            Some(got) if got == want => Ok(()),
            Some(got) => err(format!(
                "{}: the W8A8 DOWN quantize spine's {what} is {got}, not {want} — the card's own \
                 `quantscalepertokenfp8` constant family (the f16 images of 1/448 and 1/448², \
                 the ±448 clamps), and any other value is a different quantize law, refused \
                 rather than fused",
                f.name
            )),
            None => err(format!(
                "{}: the W8A8 DOWN quantize spine's {what} is not a splatted constant — nothing \
                 proves it is the constant the primitive states",
                f.name
            )),
        }
    };
    want_splat(amax_c, consts.inv_qmax, "amax scaling (1/448)")?;
    want_splat(floor_c, consts.floor, "scale floor (1/448²)")?;
    want_splat(cap_c, consts.cap, "scale cap (448)")?;
    want_splat(num, consts.one, "reciprocal numerator (1.0)")?;
    want_splat(lo_c, consts.lo, "codes lower clamp (−448)")?;
    want_splat(hi_c, consts.hi, "codes upper clamp (+448)")?;

    // The amax: Max combiner, −inf seed, `[1]` f16 result — the primitive's own
    // semantics, per pair's own row.
    if reduce_kind_of(reduce) != Some(ReduceKind::Max) {
        return err(format!(
            "{}: the W8A8 DOWN spine's amax reduce is not a Max — the card's \
             `quantscalepertokenfp8` computes the row's maximum magnitude, and any other \
             combiner is a different scale",
            f.name
        ));
    }
    match reads.splat_value(red_seed) {
        Some(f64::NEG_INFINITY) => {}
        got => {
            return err(format!(
                "{}: the W8A8 DOWN spine's amax reduce seeds {got:?}, not −inf — a Max seeded \
                 any higher clamps every-negative row's scale and is a different function than \
                 the primitive's",
                f.name
            ));
        }
    }
    let red_ok = match reduce.result_type {
        Some(ktir_core::irtype::IrType::Tensor { dims, elem }) => {
            dims.len() == 1 && dims[0] == 1 && elem == ktir_core::dtypes::DType::F16
        }
        _ => false,
    };
    if !red_ok {
        return err(format!(
            "{}: the W8A8 DOWN spine's amax reduce does not state a `[1]` f16 result — one \
             scale per token row is the primitive's own shape",
            f.name
        ));
    }
    // ⛔ THE ISLAND'S TWO BOUNDARIES: the truncation states `Fp8E4m3` and the
    // widening `F16` — per pair, the same island the gate/up spine states.
    if trunc.result_type.and_then(|t| t.elem()) != Some(ktir_core::dtypes::DType::Fp8E4m3) {
        return err(format!(
            "{}: the W8A8 DOWN spine's codes truncation does not state Fp8E4m3 — the codes are \
             the values the fp8 contraction reads, and any other element type is a different \
             function",
            f.name
        ));
    }
    if ext.result_type.and_then(|t| t.elem()) != Some(ktir_core::dtypes::DType::F16) {
        return err(format!(
            "{}: the W8A8 DOWN spine's codes widening does not state f16 — the island is \
             narrowed back before this door, or the contraction would read a different \
             precision than the card's own W8A8",
            f.name
        ));
    }

    // ⛔ DESTRUCTION IS EXCLUSIVE — every destroyed link has exactly the one
    // reader the chain states.
    let sole = |v: Ssa, want: OpKind, what: &str| -> Result<(), Error> {
        match reads.readers_of(v) {
            [p] if reads.op_at(*p).op_type == want => Ok(()),
            [p] => err(format!(
                "{}: the W8A8 DOWN quantize spine's {what} is read by `{:?}` — the fusion \
                 destroys it, so exactly the chain's next link may read it",
                f.name,
                reads.op_at(*p).op_type
            )),
            rs => err(format!(
                "{}: the W8A8 DOWN quantize spine's {what} is read by {} ops — the fusion \
                 destroys it, so exactly the chain's next link may read it",
                f.name,
                rs.len()
            )),
        }
    };
    let (Some(abs_v), Some(trunc_v), Some(ext_v), Some(codes_min_v)) =
        (abs.result, trunc.result, ext.result, codes_min.result)
    else {
        return err(format!(
            "{}: a W8A8 DOWN spine link states no result — not reachable through the def/use \
             index",
            f.name
        ));
    };
    let Some(inv_v) = inv_op.result else {
        return err(format!(
            "{}: a W8A8 DOWN spine link states no result — not reachable through the def/use \
             index",
            f.name
        ));
    };
    sole(abs_v, OpKind::LinalgReduce, "absolute value")?;
    sole(red_v, OpKind::ArithMulf, "amax")?;
    sole(sx_mul_v, OpKind::ArithMaxnumf, "amax scaling")?;
    sole(sx_max_v, OpKind::ArithMinnumf, "scale floor")?;
    sole(codes_mul_v, OpKind::ArithMaxnumf, "codes scaling")?;
    sole(codes_max_v, OpKind::ArithMinnumf, "codes lower clamp")?;
    sole(codes_min_v, OpKind::ArithTruncf, "codes upper clamp")?;
    sole(trunc_v, OpKind::ArithExtf, "fp8 codes")?;
    sole(b, OpKind::LinalgMatmul, "collapsed slab")?;
    // `sa_p` SURVIVES as a value the chain never materialises (the emission's
    // quantize recomputes the scale as the primitive's own output), so its
    // readers are the reciprocal and rank plumbing only.
    for &p in reads.readers_of(sx) {
        let r = reads.op_at(p);
        let ok = (r.op_type == OpKind::ArithDivf && r.result == Some(inv_v))
            || matches!(
                r.op_type,
                OpKind::TensorExpandShape | OpKind::TensorCollapseShape | OpKind::LinalgBroadcast
            );
        if !ok {
            return err(format!(
                "{}: the W8A8 DOWN spine's per-pair scale is read by `{:?}` — the fusion keeps \
                 it only as the dequant's broadcast and the reciprocal's denominator",
                f.name, r.op_type
            ));
        }
    }
    // The reciprocal's readers: rank plumbing only (the codes' broadcast chain).
    for &p in reads.readers_of(inv_v) {
        let r = reads.op_at(p);
        if !matches!(
            r.op_type,
            OpKind::TensorExpandShape | OpKind::TensorCollapseShape | OpKind::LinalgBroadcast
        ) {
            return err(format!(
                "{}: the W8A8 DOWN spine's reciprocal is read by `{:?}` — the fusion destroys \
                 it in the codes, and only the broadcast chain may carry it there",
                f.name, r.op_type
            ));
        }
    }

    // The dequant's first leg: the dot's ONE reader, an `arith.mulf` whose other
    // operand resolves through rank plumbing to `sa_p`.
    let Some(mul_s_op) = reads
        .sole_reader(dot)
        .filter(|o| o.op_type == OpKind::ArithMulf)
    else {
        return err(format!(
            "{}: the W8A8 DOWN gemm's product {dot:?} is not read by exactly one `arith.mulf` — \
             the chain's dequant leg (`raw · broadcast(sa_p)`) is the only form this fusion \
             states",
            f.name
        ));
    };
    let Some(mul_s_v) = mul_s_op.result else {
        return err(format!(
            "{}: a W8A8 DOWN chain link states no result — not reachable through the def/use \
             index",
            f.name
        ));
    };
    let Some(ms_bcast) = mul_s_op.operands.iter().copied().find(|&v| v != dot) else {
        return err(format!(
            "{}: the W8A8 DOWN gemm's dequant multiply names no operand beside its product",
            f.name
        ));
    };
    if reads.through_plumbing(ms_bcast) != sx {
        return err(format!(
            "{}: the W8A8 DOWN gemm's dequant multiplies the product by a value that is not \
             the spine's per-pair scale — the W8A8 form dequantises with the quantize's own \
             scale, and any other factor is a different function",
            f.name
        ));
    }
    // The terminal: the first leg's ONE reader, an `arith.mulf` whose other
    // operand is the pair's gathered WEIGHT scale.
    let Some(term_op) = reads
        .sole_reader(mul_s_v)
        .filter(|o| o.op_type == OpKind::ArithMulf)
    else {
        return err(format!(
            "{}: the W8A8 DOWN gemm's scaled product is not read by exactly one `arith.mulf` — \
             the chain's terminal (`· the pair's gathered weight scale`) is the only form this \
             fusion states",
            f.name
        ));
    };
    let Some(terminal) = term_op.result else {
        return err(format!(
            "{}: a W8A8 DOWN chain link states no result — not reachable through the def/use \
             index",
            f.name
        ));
    };
    // ⛔ NO STORE READS A TERMINAL — the same law the gate/up reader states: the
    // fusion writes the projection's output intermediate and the walk hands
    // consumers the terminal as a window of it; a `ktdp.store` of a terminal is
    // a write the walk's hook would silently drop.
    if reads
        .readers_of(terminal)
        .iter()
        .any(|&p| reads.op_at(p).op_type == OpKind::KtdpStore)
    {
        return err(format!(
            "{}: the W8A8 DOWN gemm's output {terminal:?} is stored by the program — the \
             fusion writes its projection's output intermediate, and a stored terminal is a \
             write the walk would silently drop",
            f.name
        ));
    }
    let Some(scale_operand) = term_op.operands.iter().copied().find(|&v| v != mul_s_v) else {
        return err(format!(
            "{}: the W8A8 DOWN gemm's terminal multiply names no operand beside its scaled \
             product",
            f.name
        ));
    };
    let Some(scale_load) = def_kind(scale_operand, OpKind::KtdpLoad) else {
        return err(format!(
            "{}: the W8A8 DOWN gemm's terminal multiplies by a value that is not a load — the \
             weight bank's per-channel scale is a gathered load, and a computed factor is a \
             different function",
            f.name
        ));
    };
    let Some(scale_tile) = scale_load
        .operands
        .first()
        .copied()
        .and_then(|v| def_kind(v, OpKind::KtdpConstructIndirectAccessTile))
    else {
        return err(format!(
            "{}: the W8A8 DOWN gemm's weight scale is not gathered — the pair's scale is \
             selected by the same index that selects its slab, and a plain load is a different \
             program",
            f.name
        ));
    };
    let sf = gather_facts_at_tile(k, scale_tile)?;
    // Same index parameter, same anchor — the pair's OWN scale, not another
    // pair's.
    if sf.index_tid != wf.index_tid || sf.first_entry != wf.first_entry {
        return err(format!(
            "{}: the W8A8 DOWN gemm's dequant reads its pair's weight scale through a different \
             index anchor than its weight slab — one pair's two gathers anchor at ONE index word",
            f.name
        ));
    }

    // The geometry, read off the two views the program states. The weight bank
    // is the RAW `[E, n, inn]` fp8 parameter; the scale bank is `[E, n]` f16.
    let Some(w_view) = b_tile
        .operands
        .first()
        .copied()
        .and_then(|v| reads.def_of(v))
    else {
        return err(format!(
            "{}: the W8A8 DOWN gemm's weight tile names a view nothing defines",
            f.name
        ));
    };
    let (e_count, n, inn) = match w_view.result_type {
        Some(ktir_core::irtype::IrType::MemRef { dims, elem })
            if elem == ktir_core::dtypes::DType::Fp8E4m3
                && dims.len() == 3
                && dims.iter().all(|&d| d > 0) =>
        {
            (dims[0] as u32, dims[1] as u32, dims[2] as u32)
        }
        _ => {
            return err(format!(
                "{}: the W8A8 DOWN gemm's weight bank t{} is not an `[E, n, in]` Fp8E4m3 view — \
                 the fused door reads the program's own expert-major bank, and any other layout \
                 is the staged-bank machinery it refuses by name",
                f.name, wf.value_tid
            ));
        }
    };
    let Some(s_view) = scale_tile
        .operands
        .first()
        .copied()
        .and_then(|v| reads.def_of(v))
    else {
        return err(format!(
            "{}: the W8A8 DOWN gemm's scale tile names a view nothing defines",
            f.name
        ));
    };
    let s_ok = match s_view.result_type {
        Some(ktir_core::irtype::IrType::MemRef { dims, elem }) => {
            elem == ktir_core::dtypes::DType::F16 && dims == [e_count as i64, n as i64]
        }
        _ => false,
    };
    if !s_ok {
        return err(format!(
            "{}: the W8A8 DOWN gemm's scale bank t{} is not the weight bank's own `[E, n]` f16 \
             view — the per-output-channel scales ride the same expert count and projection \
             width",
            f.name, sf.value_tid
        ));
    }
    let dot_ok = match op.result_type {
        Some(ktir_core::irtype::IrType::Tensor { dims, elem }) => {
            elem == ktir_core::dtypes::DType::F16 && dims == [1, n as i64]
        }
        _ => false,
    };
    if !dot_ok {
        return err(format!(
            "{}: the W8A8 DOWN gemm's product does not state a `[1, n]` f16 result at the \
             bank's own n={n}",
            f.name
        ));
    }
    // ⛔ M=1 ONLY — the act buffer's own row count. The fused W8A8 is the decode
    // form; M > 1 is the per-`Program` door's gathered prefill body and is
    // refused here by name rather than lowered as the nearest thing.
    let Some(a_region) = region_for_operand(k, row)? else {
        return err(format!(
            "{}: the W8A8 DOWN gemm's act reload does not resolve to a parameter region — the \
             spine rooted at a plain parameter load, so the region walk that refused it is the \
             defect to read",
            f.name
        ));
    };
    if a_region.c_len != inn {
        return err(format!(
            "{}: the W8A8 DOWN gemm's act window t{} is {} columns wide but the weight bank \
             t{} contracts {inn} — the gated act's `in` must match the bank's",
            f.name, a_region.tid, a_region.c_len, wf.value_tid
        ));
    }
    if a_region.v_rows != 1 {
        return err(format!(
            "{}: the W8A8 DOWN gemm chain's act buffer t{} is {} rows wide — this door's fused \
             W8A8 is the M=1 (decode) form, and M > 1 is refused by name rather than lowered as \
             the nearest thing (the per-`Program` expert door's gathered m>1 body is the \
             prefill form)",
            f.name, a_region.tid, a_region.v_rows
        ));
    }
    // ⛔ PER-PAIR AGREEMENT — the window IS the pair. The chain that gathers its
    // expert through index word `anchor` reads the act buffer's window at
    // `anchor · inn`: moe.py's `j` is both the index word (`desc_ids + m·K + j`)
    // and the window index (`act_desc.load([m, j·I])`), and the gelu the
    // root-through-store proved is the one stored at THAT window. A mismatch is
    // two pairs wearing one chain — pair `p`'s scale would quantize pair `q`'s
    // activation, and every pair's down projection would compute another pair's
    // numbers.
    if row_win != (0, wf.first_entry as u32 * inn, 1, inn) {
        return err(format!(
            "{}: the W8A8 DOWN gemm anchored at index word {} reads the act window {row_win:?} \
             — the pair's own window starts at that word times the contraction width \
             (word·{inn}), and a mismatch is two pairs wearing one chain",
            f.name, wf.first_entry
        ));
    }
    Ok(Some(W8a8DownClaim {
        chain: W8a8DownChain {
            pair: wf.first_entry as u32,
            weight_tile: wf.tile,
            collapsed: b,
            dot,
            mul_s: mul_s_v,
            terminal,
            scale_tile: sf.tile,
        },
        weight_tid: wf.value_tid,
        scale_tid: sf.value_tid,
        a_tid,
        idx_tid: wf.index_tid,
        anchor: wf.first_entry,
        e_count,
        n,
        inn,
        spine_consumed: [
            abs_v,
            red_v,
            sx_mul_v,
            sx_max_v,
            sx,
            inv_v,
            codes_mul_v,
            codes_max_v,
            codes_min_v,
            trunc_v,
            ext_v,
        ],
    }))
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

/// THE `Elementwise(Mul)` / `ScalarMul` TIEBREAK — [`ProgramReads::splat_scale_of`] over a
/// fresh index, for the callers that hold the function and not a walk's index
/// (`triton-ktir-superdsc`'s `scales_for`). [`lower_function`]'s own walk calls the method on
/// the ONE index it builds, so a mulf-heavy program pays for one def/use walk, not one per
/// `arith.mulf`.
pub fn splat_scale_of(
    f: &IRFunction<'static>,
    op: &ktir_core::ir::Operation<'static>,
) -> Option<(f32, Ssa)> {
    program_reads(f).splat_scale_of(op)
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

/// THE RUNG-3 BOUND-SCALE TIEBREAK — [`ProgramReads::bound_scale_of`] over a fresh index, for
/// the callers that hold the node alone (the `zz_the_bound_scale_is_a_launch_binding` suite).
/// [`lower_function`]'s walk calls the method on the ONE index it builds.
pub fn bound_scale_of(
    k: &KtirNode,
    op: &ktir_core::ir::Operation<'static>,
) -> Result<Option<BoundScale>, Error> {
    program_reads(&k.func).bound_scale_of(k, op)
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
    let reads = program_reads(f);
    for bc in reads
        .ops()
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
            let Some(d) = reads.def_of(cursor) else { break };
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
/// ⭐ The window one `ktdp.construct_access_tile` states — its corner constants
/// (operands 1/2) and its own `shape` — read with exactly the code `regions()` reads
/// them (`shape_2d` and `index_constants`), so every reader of a tile's window agrees.
///
/// `Ok(None)` when the tile is not the corner form (`operands.len() != 3`): the caller
/// keeps the region it already holds. ⛔ A tile whose corners are NOT constants names a
/// window no descriptor can carry, and is refused by name rather than silently falling
/// back to the parameter's first tile — the same discipline as `regions()`'s in-loop
/// guard. ONE function because its two callers — the load arm of
/// [`region_for_operand`] and the walk's store arm — are two readers of ONE fact about
/// one tile, and two readings of a tile is the drift the own-tile law exists to
/// prevent (issue 201 item 3's shape).
fn tile_window(
    f: &IRFunction<'static>,
    tile: &Operation<'_>,
    whose: &str,
) -> Result<Option<(u32, u32, u32, u32)>, Error> {
    if tile.operands.len() != 3 {
        return Ok(None);
    }
    let consts = super::lower_ktir_to_superdsc::index_constants(f);
    let Some(shape) = super::lower_ktir_to_superdsc::shape_2d(tile) else {
        return err(format!(
            "{}: the access tile {whose} states no 2-D `shape`, so its window is unknown",
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
                "{}: the access tile {whose} states a window whose corner is not a constant \
                 (`ktdp.construct_access_tile` operands 1/2), so the descriptor cannot carry the \
                 offset. A loop-carried or computed corner needs hoisting or unrolling before the \
                 whole-function door can address it.",
                f.name
            ));
        }
    };
    Ok(Some((rs, cs, shape.0, shape.1)))
}

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
    if let Some((rs, cs, r_len, c_len)) = tile_window(f, tile, "feeding this load")? {
        r.r_start = rs;
        r.c_start = cs;
        r.r_len = r_len;
        r.c_len = c_len;
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

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  THE ROUTER — the round-form top-k, the temperature softmax, and the per-expert scale.
//
// `kernels/moe.py`'s router section is ONE fused product — `lg = xn @ rw.T`, then K rounds of
// (max → lane-select → argmax-reduce → one-hot accumulate → mask-out), then
// `softmax(vals · H^-0.5) · psel` — spelled as ≈50·K ops the 1:1 walk cannot lower (`arith.cmpf`,
// `arith.select`, `tensor.generate`, `arith.fptosi` have no `Program`). Lowering it op-by-op is
// not a slower answer but a different one: the round form's tie-break (`reduce_max` over
// `select(cmp, offs, −1)`) and the replicated top-k pair's slot law are the SELECTION semantics,
// and the door's fused legs (`RouteTopK`, `ScalarMul`, `RouteSoftmax`, `RouteExpertScale`) are
// the only spelling of them the card has. So the section is recognised WHOLE, claimed from the
// ids store backward, and emitted at ONE fire point.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// THE ROUTER CHAIN, as the ONE fused emission it is.
///
/// # THE CLAIM, AND THE TWO VALUES IT DELIBERATELY DOES NOT CONSUME
///
/// The proof walks every link of every round and the whole softmax tail, and each value it
/// touches lands in [`RouterChain::consumed`] — the interior-skip arm's law, same as every
/// other chain this door reads. TWO values are walked but NOT consumed, because other arms own
/// them:
///
/// * `lg` — the dot's own result. The 1:1 walk's matmul arm lowers it and mints its region,
///   which the fire point reads through `inter` (the dot PRECEDES the fire point in program
///   order: the rounds read it).
/// * `emit_at` — the lane-offset identity generate. It is the FIRE POINT itself: the walk
///   reaches it unclaimed and the fire arm below claims the whole chain there.
#[derive(Clone, Debug)]
pub struct RouterChain {
    /// The logits — the router projection's `linalg.matmul` result. Not consumed; see above.
    pub lg: Ssa,
    /// The per-expert scale row's LOAD result — the one-hot select's sprayed operand. The fire
    /// point resolves it to the parameter's region with [`region_for_operand`].
    pub pes: Ssa,
    /// `desc_ids`' binding tid — the scratch synth key [`route_topk_legs`] and the expert
    /// doors' `XIdx` reservation share, so the fused router and the per-pair projections land
    /// on ONE scratch.
    pub ids_tid: u32,
    /// `desc_rs`' binding tid — the route-weights scratch. The emission never writes it; the
    /// scratch census proves every read of it is a claimed chain's.
    pub rs_tid: u32,
    /// Every DIRECT load of the route-weights scratch (the combine's `wrow` read; empty in a
    /// router-only kernel). Checked by the scratch census; resolved to the minted
    /// route-weights product at the fire point, so the combine rung's hook reads the
    /// emission's own buffer through the map every cross-chain handoff uses.
    pub rs_loads: Vec<Ssa>,
    /// The softmax temperature, read off the `·TEMP` multiply's own splat (the f16 image of
    /// the source constant — `scale_slot`'s f16-image fallback resolves it to the registry
    /// slot).
    pub temp: f32,
    /// The top-k width — the round count the accumulation walk proved, cross-checked against
    /// the `koffs` generate's width and the ids store's own view.
    pub k: u32,
    /// The fire point: the lane-offset identity generate's result. Not consumed; see above.
    pub emit_at: Ssa,
    /// Everything else the proof touched: the rounds' three accumulators and their update
    /// multiplies, the lane/one-hot/mask compares and selects, the reduces, the converts, and
    /// the tail's softmax chain. The two stores' VALUES ride in here — the emission writes
    /// minted landings, never the scratch parameters (the write-dead law).
    pub consumed: Vec<Ssa>,
}

/// ⭐⭐⭐ THE ROUTER CHAINS THIS FUNCTION HOLDS.
///
/// THE ROOT is the program's INTEGER-valued store: the slab gather reads expert indices as
/// i32, so the ids land through an `arith.fptosi` — the one store in a router whose value is
/// an integer convert. No such store ⇒ no router here (the expert-only door kernels lower
/// untouched); every such store ⇒ a full proof or a refusal naming the link.
/// `ArithFptosi` has no 1:1 `Program`, so a function that reached the walk with one already
/// refused — rooting here is purely additive.
pub fn program_router_chains(k: &KtirNode) -> Result<Vec<RouterChain>, Error> {
    let f = &k.func;
    let reads = program_reads(f);
    let mut chains = Vec::new();
    for store in f
        .operations
        .iter()
        .filter(|s| s.op_type == OpKind::KtdpStore)
    {
        let is_router_root = store
            .operands
            .first()
            .and_then(|&v| reads.def_of(v))
            .is_some_and(|d| d.op_type == OpKind::ArithFptosi);
        if is_router_root {
            chains.push(router_chain_at(k, &reads, store)?);
        }
    }
    Ok(chains)
}

/// THE ROUTER CHAIN AT ONE IDS STORE — the whole proof, from the store's value backward.
///
/// # WHY THE WALK RUNS BACKWARD, AND WHY IT IS EXACT
///
/// The only value the program PINS is the store (everything else has several plausible
/// spellings), so the proof starts there and unwinds: the store's value is the fptosi of the
/// ids accumulator; the accumulators update through one SHARED one-hot, which names the other
/// two; each round's one-hot names its round compare, its k-offset vector and its round index;
/// the three updates' value sides name the round's three reduces; the reduces' data names the
/// lane select and the one-hot select; those selects' compares name the logits, the lane
/// offsets, and each other. Every step is a def/use question this index answers, every
/// constant is read off its own splat, and every disagreement is a REFUSAL naming the link —
/// never a guess past it. The mask select is found FROM the lane-match compare (its reader),
/// so the walk never searches forward: the compare's readers are exactly the one-hot select
/// and — at every round but the last — the mask.
///
/// The round indices read off the one-hot compares self-verify the count (they must descend by
/// exactly one), and `k` is cross-checked three ways at the end: the walk, the `koffs`
/// generate's width, and the ids landing's own columns.
fn router_chain_at(
    k: &KtirNode,
    reads: &ProgramReads<'_>,
    ids_store: &Operation<'static>,
) -> Result<RouterChain, Error> {
    let f = &k.func;
    let name = f.name;
    let mut consumed: Vec<Ssa> = Vec::new();

    // ── THE IDS STORE: `desc_ids` ← `arith.fptosi`(the ids accumulator) ──
    let (ids_tid, rows, k_view) =
        store_param(k, reads, ids_store, name, "the router's", "the routed ids")?;
    let Some(&ids_i32) = ids_store.operands.first() else {
        return err(format!("{name}: the router's ids store states no value"));
    };
    let fptosi = reads.def_of(ids_i32).ok_or_else(|| Error {
        message: format!("{name}: the router's ids store value is a function argument"),
    })?;
    if fptosi.op_type != OpKind::ArithFptosi {
        return err(format!(
            "{name}: the router's ids store value is defined by `{:?}` — the slab gather reads \
             INTEGER expert indices, so the ids accumulator lands through an `arith.fptosi`",
            fptosi.op_type
        ));
    }
    let [idsf] = fptosi.operands[..] else {
        return err(format!("{name}: the router's ids convert is not unary"));
    };
    consumed.push(ids_i32);

    // ── THE FINAL ROUND'S THREE ACCUMULATORS, recovered from the ONE-HOT the updates share ──
    //
    // Every round updates THREE `[rows, k]` accumulators — the selected maxima, the selected
    // ids, the selected per-expert scales — each as `acc + x·onehot` (round 0 folded to
    // `x·onehot`). The ids accumulator is the store's own value; the other two are recovered
    // from the one-hot: EVERY reader of the one-hot broadcast is one of the three updates, so
    // the ids update's one-hot names the other two. A reader outside the three is an op this
    // proof has no account of, and the fusion would skip a value it reads — refused, not
    // skipped.
    let (_, ids_mulf) = router_acc_update(reads, name, idsf, "the ids")?;
    let (oh_b, _) = router_onehot_of(reads, name, ids_mulf, "the ids update")?;
    let mut updates: Vec<Ssa> = Vec::with_capacity(3);
    for o in reads.readers_of(oh_b).iter().map(|&p| reads.op_at(p)) {
        if o.op_type != OpKind::ArithMulf {
            return err(format!(
                "{name}: the router's one-hot has a reader `{:?}` that is not a round update — \
                 the round form updates exactly the three accumulators through one shared \
                 one-hot, and anything else reading it is an op this door does not recognise",
                o.op_type
            ));
        }
        let Some(r) = o.result else {
            return err(format!(
                "{name}: the router's one-hot update states no result"
            ));
        };
        updates.push(r);
    }
    if updates.len() != 3 || !updates.contains(&ids_mulf) {
        return err(format!(
            "{name}: the router's one-hot has {} update readers — the round form updates \
             exactly THREE accumulators (the selected maxima, the ids, the per-expert scales) \
             through one shared one-hot",
            updates.len()
        ));
    }
    // Each update's accumulator at the final round: the update's own result (a single-round
    // kernel, where round 0 IS the final round) or the RUNNING add that reads it. An update
    // read by two adds is ambiguous and refused.
    let acc_of_update = |m: Ssa| -> Result<Ssa, Error> {
        let adds: Vec<&Operation<'static>> = reads
            .readers_of(m)
            .iter()
            .map(|&p| reads.op_at(p))
            .filter(|o| o.op_type == OpKind::ArithAddf)
            .collect();
        match adds.as_slice() {
            [] => Ok(m),
            [only] => only.result.ok_or_else(|| Error {
                message: format!("{name}: the router's running accumulator add states no result"),
            }),
            _ => err(format!(
                "{name}: the router's round update is read by two running adds — the round form \
                 feeds ONE accumulator per update"
            )),
        }
    };
    // WHICH UPDATE IS WHICH — by the reduce each update's value side resolves to: the maxima
    // update to a MAX reduce, the scales update to a SUM reduce. (The loop below re-proves all
    // three spines per round; this is the assignment, not the proof.)
    let mut classify = |m: Ssa| -> Result<OpKind, Error> {
        let (_, x) = router_onehot_of(reads, name, m, "a round update")?;
        let src = plumbing_source(reads, x, &mut consumed);
        let Some(d) = reads.def_of(src) else {
            return err(format!(
                "{name}: the router's round update value is a function argument"
            ));
        };
        d.attributes
            .iter()
            .find_map(|(kk, vv)| match (kk, vv) {
                (AttrKey::ReduceFn, Attr::Op(o)) => Some(*o),
                _ => None,
            })
            .ok_or_else(|| Error {
                message: format!(
                    "{name}: the router's round update value resolves to `{:?}`, not a \
                         `linalg.reduce` — each update's value side is the round's reduce, \
                         resticked to the accumulator's width",
                    d.op_type
                ),
            })
    };
    let others: Vec<Ssa> = updates.iter().copied().filter(|&u| u != ids_mulf).collect();
    let [ma, mb] = others[..] else {
        return err(format!(
            "{name}: the router's one-hot recovery found no second update"
        ));
    };
    let (mut vals, mut psel) = match (classify(ma)?, classify(mb)?) {
        (OpKind::ArithMaxnumf, OpKind::ArithAddf) => (acc_of_update(ma)?, acc_of_update(mb)?),
        (OpKind::ArithAddf, OpKind::ArithMaxnumf) => (acc_of_update(mb)?, acc_of_update(ma)?),
        (a, b) => {
            return err(format!(
                "{name}: the router's round updates resolve to `{a:?}` and `{b:?}` reduces — the \
                 three accumulators are a MAX over the logits (the selected maxima), a MAX over \
                 the lane select (the ids), and a SUM over the one-hot select (the scales)"
            ));
        }
    };
    let vals_final = vals;
    let psel_final = psel;
    let mut idsf = idsf;

    // ── THE LOCKSTEP WALK, from the final round down to round 0 ──
    //
    // Per round j (descending from K−1): the three updates share one one-hot
    // `oh_j = select(koffs == j, 1, 0)`; the maxima spine is `mx_j = reduce_max(lg_j, −inf)`;
    // the ids spine is `idx_j = reduce_max(select(lg_j == mx_j, offs, −1), −inf)`; the scales
    // spine is `pj_j = reduce_sum(select(offs == idx_j, pes, 0), 0)`; and — at every round but
    // the last — the lane-match compare ALSO feeds the mask select
    // `lg_{j+1} = select(offs == idx_j, −inf, lg_j)` that burns the chosen lane out of the
    // logits for the round above. Every resolution goes through `plumbing_source`, so no link
    // the proof walked is left for the 1:1 walk to refuse.
    let mut expect_j: Option<u32> = None;
    let mut prev_lg: Option<Ssa> = None;
    let mut rounds: u32 = 0;
    let mut offs: Option<(Ssa, Ssa, u32)> = None; // (the sitofp, the generate, E)
    let mut pes: Option<Ssa> = None;
    let mut koffs_w: Option<u32> = None;
    loop {
        if rounds >= k_view {
            return err(format!(
                "{name}: the router's accumulation walk found {} rounds against an ids landing \
                 of {k_view} slots — the walk counts one round per top-k slot and cannot run \
                 past the landing's own width",
                rounds + 1
            ));
        }
        let (ids_prev, ids_mulf) = router_acc_update(reads, name, idsf, "the ids")?;
        let (vals_prev, vals_mulf) = router_acc_update(reads, name, vals, "the selected maxima")?;
        let (psel_prev, psel_mulf) = router_acc_update(reads, name, psel, "the selected scales")?;
        // ⛔ THE FORM AGREES: round 0's updates are FOLDED (`x·onehot`, the zero init folded
        // away) and every deeper round's are RUNNING (`acc + x·onehot`) — all three together,
        // or not a router.
        let folded = ids_prev.is_none();
        if folded != vals_prev.is_none() || folded != psel_prev.is_none() {
            return err(format!(
                "{name}: the router's round {rounds} mixes a folded round-0 update with a running \
                 one — the three accumulators fold or run TOGETHER (round 0 is `x·onehot`, the \
                 rest `acc + x·onehot`)"
            ));
        }
        consumed.push(idsf);
        consumed.push(vals);
        consumed.push(psel);
        consumed.push(ids_mulf);
        consumed.push(vals_mulf);
        consumed.push(psel_mulf);
        // THE SHARED ONE-HOT: the same widened one-hot at all three updates.
        let (oh_b, _) = router_onehot_of(reads, name, ids_mulf, "the ids update")?;
        for (m, what) in [
            (vals_mulf, "the selected maxima"),
            (psel_mulf, "the selected scales"),
        ] {
            let (their_oh, _) = router_onehot_of(reads, name, m, what)?;
            if their_oh != oh_b {
                return err(format!(
                    "{name}: the router's round {rounds} updates read two different one-hots — \
                     the three accumulators share ONE round compare"
                ));
            }
        }
        // oh = select(koffs == j, 1, 0), widened to [1, k] by a BARE expand (no
        // `linalg.broadcast` roots at it, so the broadcast-chain reader claims none of it —
        // this walk does).
        let oh = plumbing_source(reads, oh_b, &mut consumed);
        let oh_d = reads.def_of(oh).ok_or_else(|| Error {
            message: format!("{name}: the router's one-hot is a function argument"),
        })?;
        if oh_d.op_type != OpKind::ArithSelect {
            return err(format!(
                "{name}: the router's one-hot is `{:?}` — the round compare's one-hot is an \
                 `arith.select` of two splats",
                oh_d.op_type
            ));
        }
        let [oh_cond, oh_one, oh_zero] = oh_d.operands[..] else {
            return err(format!(
                "{name}: the router's one-hot select is not ternary"
            ));
        };
        if !router_splat_is(reads, oh_one, 1.0) || !router_splat_is(reads, oh_zero, 0.0) {
            return err(format!(
                "{name}: the router's one-hot select reads splats other than 1.0/0.0 — the round \
                 compare's one-hot is exactly `select(koffs == j, 1, 0)`"
            ));
        }
        consumed.push(oh);
        consumed.push(oh_cond);
        // THE ROUND INDEX, off the compare's own splat — and the k-offset vector on its other
        // side. The j values must descend by exactly one per round: the count the walk
        // cross-checks against the landing and the k-offset generate at the end.
        let (ka, kb) = router_oeq(reads, name, oh_cond, "the one-hot's round compare")?;
        let (j_splat, koffs_v) = match (reads.splat_value(ka), reads.splat_value(kb)) {
            (Some(_), None) => (ka, kb),
            (None, Some(_)) => (kb, ka),
            _ => {
                return err(format!(
                    "{name}: the router's round compare reads no integer-valued round splat — the \
                     one-hot is `koffs == j` with j the round's own index"
                ));
            }
        };
        let Some(jf) = reads.splat_value(j_splat) else {
            return err(format!(
                "{name}: the router's round splat is not a constant"
            ));
        };
        if jf < 0.0 || jf.fract() != 0.0 || jf > 255.0 {
            return err(format!(
                "{name}: the router's round compare states j = {jf} — the round index is a small \
                 non-negative integer"
            ));
        }
        let j = jf as u32;
        match expect_j {
            None => expect_j = Some(j),
            Some(e) if e == j + 1 => expect_j = Some(j),
            _ => {
                return err(format!(
                    "{name}: the router's rounds state j = {j} where {} was next — the rounds run \
                     0..K−1 and the accumulation walk reads them descending",
                    expect_j.unwrap_or(0)
                ));
            }
        }
        // KOFFS: the k-offset identity vector, the SAME at every round.
        let koffs = plumbing_source(reads, koffs_v, &mut consumed);
        let koffs_d = reads.def_of(koffs).ok_or_else(|| Error {
            message: format!("{name}: the router's k-offset vector is a function argument"),
        })?;
        if koffs_d.op_type != OpKind::ArithSitofp {
            return err(format!(
                "{name}: the router's k-offset vector is `{:?}` — `tl.arange(0, K)` lands as a \
                 `tensor.generate` widened by `arith.sitofp`",
                koffs_d.op_type
            ));
        }
        let [koffs_gen] = koffs_d.operands[..] else {
            return err(format!(
                "{name}: the router's k-offset convert is not unary"
            ));
        };
        let Some(this_koffs_w) = router_identity_generate(reads, koffs_gen) else {
            return err(format!(
                "{name}: the router's k-offset generate is not the identity `tl.arange` — the \
                 round compare reads `koffs == j` with koffs the offset vector itself"
            ));
        };
        match koffs_w {
            None => koffs_w = Some(this_koffs_w),
            Some(w) if w == this_koffs_w => {}
            Some(_) => {
                return err(format!(
                    "{name}: the router's rounds read two different k-offset vectors — one \
                     `tl.arange(0, K)` serves every round compare"
                ));
            }
        }
        consumed.push(koffs);
        consumed.push(koffs_gen);

        // ── THE THREE SPINES ──
        // THE MAXIMA: mx_j = reduce_max(lg_j, −inf), resticked to [1, k].
        let (_, vals_x) = router_onehot_of(reads, name, vals_mulf, "the selected maxima")?;
        let mx = plumbing_source(reads, vals_x, &mut consumed);
        let mx_data = router_seeded_reduce(
            reads,
            name,
            mx,
            OpKind::ArithMaxnumf,
            f64::NEG_INFINITY,
            "the round's maxima reduce",
        )?;
        // THE IDS: idx_j = reduce_max(lanes_j, −inf) over the lane select.
        let (_, ids_x) = router_onehot_of(reads, name, ids_mulf, "the ids")?;
        let idx = plumbing_source(reads, ids_x, &mut consumed);
        let lanes = router_seeded_reduce(
            reads,
            name,
            idx,
            OpKind::ArithMaxnumf,
            f64::NEG_INFINITY,
            "the round's lane reduce",
        )?;
        // THE SCALES: pj_j = reduce_sum(pj_sel, 0) over the one-hot select.
        let (_, psel_x) = router_onehot_of(reads, name, psel_mulf, "the selected scales")?;
        let pj = plumbing_source(reads, psel_x, &mut consumed);
        let pj_sel = router_seeded_reduce(
            reads,
            name,
            pj,
            OpKind::ArithAddf,
            0.0,
            "the round's scale reduce",
        )?;
        consumed.push(mx);
        consumed.push(idx);
        consumed.push(pj);

        // THE LANE SELECT: lanes_j = select(lg_j == mx_j, offs, −1). The mask-out must be
        // −1.0 — the tie-break law: `reduce_max` over `select(cmp, offs, −1)` takes the LAST
        // lane holding the maximum (scan order), so any other mask-out is a different
        // selection rule on ties.
        let lanes_d = reads.def_of(lanes).ok_or_else(|| Error {
            message: format!("{name}: the router's lane select is a function argument"),
        })?;
        if lanes_d.op_type != OpKind::ArithSelect {
            return err(format!(
                "{name}: the router's lane reduce's data is `{:?}` — the ids spine reduces a LANE \
                 SELECT (`select(lg == mx, offs, −1)`), whose max is the argmax lane",
                lanes_d.op_type
            ));
        }
        let [lanes_cmp, lanes_true, lanes_false] = lanes_d.operands[..] else {
            return err(format!("{name}: the router's lane select is not ternary"));
        };
        if !router_splat_is(reads, lanes_false, -1.0) {
            return err(format!(
                "{name}: the router's lane select masks the unselected lanes with a splat other \
                 than −1.0 — the tie-break law: `reduce_max` over `select(cmp, offs, −1)` takes \
                 the LAST lane of the maximum, and any other mask-out selects a different expert \
                 on ties"
            ));
        }
        consumed.push(lanes);
        consumed.push(lanes_cmp);
        // THE LANE OFFSETS: `tl.arange(0, E)` widened to f16. ⛔ The generate itself is NOT
        // consumed — it is the op the walk's fire arm anchors at.
        let offs_v = plumbing_source(reads, lanes_true, &mut consumed);
        let offs_d = reads.def_of(offs_v).ok_or_else(|| Error {
            message: format!("{name}: the router's lane offsets are a function argument"),
        })?;
        if offs_d.op_type != OpKind::ArithSitofp {
            return err(format!(
                "{name}: the router's lane offsets are `{:?}` — `tl.arange(0, E)` lands as a \
                 `tensor.generate` widened by `arith.sitofp`",
                offs_d.op_type
            ));
        }
        let [offs_gen] = offs_d.operands[..] else {
            return err(format!(
                "{name}: the router's lane-offset convert is not unary"
            ));
        };
        let Some(offs_w) = router_identity_generate(reads, offs_gen) else {
            return err(format!(
                "{name}: the router's lane-offset generate is not the identity `tl.arange` — the \
                 lane select compares the logits against the OFFSET VECTOR, so the offsets are \
                 the identity generate's own values"
            ));
        };
        match offs {
            None => offs = Some((offs_v, offs_gen, offs_w)),
            Some((_, _, w)) if w == offs_w => {}
            Some(_) => {
                return err(format!(
                    "{name}: the router's rounds read two different lane-offset vectors — one \
                     `tl.arange(0, E)` serves every round's lane select"
                ));
            }
        }
        consumed.push(offs_v);
        // THE LANE COMPARE: lg_j == mx_j (widened). The lg operand is the side that does not
        // resolve to this round's maxima reduce; the maxima reduce's own data must then BE
        // that lg — two independent facts about one value, checked against each other.
        let (la, lb) = router_oeq(reads, name, lanes_cmp, "the lane compare")?;
        let la_src = plumbing_source(reads, la, &mut consumed);
        let lb_src = plumbing_source(reads, lb, &mut consumed);
        let lg_j = if la_src == mx {
            lb
        } else if lb_src == mx {
            la
        } else {
            return err(format!(
                "{name}: the router's lane compare reads neither side against the round's maxima \
                 reduce — the lane select is `select(lg == mx, offs, −1)`"
            ));
        };
        if mx_data != lg_j {
            return err(format!(
                "{name}: the router's maxima reduce reads a different logits value than the lane \
                 compare — the round's max and the round's lane select read ONE logits value"
            ));
        }

        // THE SCALE SELECT: pj_sel = select(offs == idx_j, pes, 0) — the one-hot over the
        // CHOSEN lane, spraying the per-expert scale row.
        let pj_d = reads.def_of(pj_sel).ok_or_else(|| Error {
            message: format!("{name}: the router's scale select is a function argument"),
        })?;
        if pj_d.op_type != OpKind::ArithSelect {
            return err(format!(
                "{name}: the router's scale reduce's data is `{:?}` — the scales spine reduces a \
                 ONE-HOT SELECT (`select(offs == idx, pes, 0)`), whose sum is the chosen lane's \
                 scale",
                pj_d.op_type
            ));
        }
        let [sel2, pj_true, pj_false] = pj_d.operands[..] else {
            return err(format!("{name}: the router's scale select is not ternary"));
        };
        if !router_splat_is(reads, pj_false, 0.0) {
            return err(format!(
                "{name}: the router's scale select masks the unselected lanes with a splat other \
                 than 0.0 — the one-hot's sum must pick exactly the chosen lane's scale"
            ));
        }
        consumed.push(pj_sel);
        consumed.push(sel2);
        // THE PER-EXPERT SCALE ROW: the select's true side, a LOAD of the pes parameter (the
        // fire point resolves it to the parameter's region).
        let pes_row = plumbing_source(reads, pj_true, &mut consumed);
        let pes_d = reads.def_of(pes_row).ok_or_else(|| Error {
            message: format!("{name}: the router's per-expert scale row is a function argument"),
        })?;
        if pes_d.op_type != OpKind::KtdpLoad {
            return err(format!(
                "{name}: the router's per-expert scale operand is `{:?}` — the one-hot select \
                 sprays the pes parameter's own row, read once per round",
                pes_d.op_type
            ));
        }
        match pes {
            None => pes = Some(pes_row),
            Some(p) if p == pes_row => {}
            Some(_) => {
                return err(format!(
                    "{name}: the router's rounds read two different per-expert scale rows — one \
                     pes parameter serves every round's one-hot"
                ));
            }
        }
        // THE LANE-MATCH COMPARE: offs == idx_j (widened). One side resolves to the lane
        // offsets; the other to THIS round's ids reduce.
        let (sa, sb) = router_oeq(reads, name, sel2, "the lane-match compare")?;
        let sa_src = plumbing_source(reads, sa, &mut consumed);
        let sb_src = plumbing_source(reads, sb, &mut consumed);
        if !((sa_src == offs_v && sb_src == idx) || (sb_src == offs_v && sa_src == idx)) {
            return err(format!(
                "{name}: the router's lane-match compare does not read the lane offsets against \
                 this round's ids reduce — the one-hot is `offs == idx`"
            ));
        }

        // THE MASK SELECT — sel2's second reader (every round but the last):
        // `lg_{j+1} = select(sel2, −inf, lg_j)`. Found FROM the scale select's condition, so
        // the walk never searches forward: the compare's readers are exactly the one-hot
        // select and — when the next round exists — the mask.
        let mut mask: Option<Ssa> = None;
        for o in reads.readers_of(sel2).iter().map(|&p| reads.op_at(p)) {
            if o.result == Some(pj_sel) {
                continue; // the scale select, proven above
            }
            if o.op_type == OpKind::ArithSelect {
                let [m_cond, m_true, m_false] = o.operands[..] else {
                    return err(format!("{name}: the router's mask select is not ternary"));
                };
                if m_cond == sel2
                    && router_splat_is(reads, m_true, f64::NEG_INFINITY)
                    && m_false == lg_j
                {
                    let Some(mr) = o.result else {
                        return err(format!("{name}: the router's mask select states no result"));
                    };
                    consumed.push(mr);
                    mask = Some(mr);
                    continue;
                }
            }
            return err(format!(
                "{name}: the router's lane-match compare has a reader `{:?}` this door does not \
                 recognise — the compare feeds exactly the one-hot select and (at every round but \
                 the last) the mask select that burns the chosen lane out of the logits",
                o.op_type
            ));
        }
        // The final round's mask is DCE'd (no next round reads its result); every deeper round
        // MUST carry one, and its result must be the logits the round ABOVE reads — the walk's
        // own continuity check, which is what makes "the same logits value" a proven fact
        // rather than an assumption at every round.
        if rounds == 0 {
            if mask.is_some() {
                return err(format!(
                    "{name}: the router's final round carries a live mask select — the mask burns \
                     the chosen lane out of the logits for the NEXT round, and the final round has \
                     none, so a live one here is a different shape"
                ));
            }
        } else {
            let Some(p) = prev_lg else {
                return err(format!(
                    "{name}: the router's walk lost the round above's logits"
                ));
            };
            if mask != Some(p) {
                return err(format!(
                    "{name}: the router's round {j} mask select does not write the logits the \
                     round above reads — the mask burns the chosen lane out of the SAME logits \
                     the next round's lane compare reads"
                ));
            }
        }
        prev_lg = Some(lg_j);

        // Round 0 is the folded form — the walk ends here.
        if folded {
            if expect_j != Some(0) {
                return err(format!(
                    "{name}: the router's folded round states j = {} — the folded update is \
                     round 0's (the zero init folded away), so the round count and the j values \
                     disagree",
                    expect_j.unwrap_or(0)
                ));
            }
            break;
        }
        idsf = ids_prev.expect("the running form's previous accumulator");
        vals = vals_prev.expect("the running form's previous accumulator");
        psel = psel_prev.expect("the running form's previous accumulator");
        rounds += 1;
    }
    // (Named `topk` because `k` is the node, matching every other recogniser in this file.)
    let topk = rounds + 1;

    // ── K, CROSS-CHECKED THREE WAYS: the walk, the k-offset generate, the ids landing ──
    let Some((_, offs_gen, e_gen)) = offs else {
        return err(format!("{name}: the router's walk proved no rounds"));
    };
    let Some(koffs_width) = koffs_w else {
        return err(format!(
            "{name}: the router's walk proved no k-offset vector"
        ));
    };
    let Some(pes) = pes else {
        return err(format!(
            "{name}: the router's walk proved no per-expert scale row"
        ));
    };
    if topk != koffs_width || topk != k_view {
        return err(format!(
            "{name}: the router's top-k is k = {topk} by the accumulation walk, {koffs_width} by \
             the k-offset generate's width, and {k_view} by the ids landing's own columns — the \
             three must agree, or the walk proved a different selection than the program stored"
        ));
    }

    // ── ROUND 0'S LOGITS: the projection dot, `[rows, E]` ──
    let lg = prev_lg.expect("set at every round");
    let lg_d = reads.def_of(lg).ok_or_else(|| Error {
        message: format!("{name}: the router's round-0 logits are a function argument"),
    })?;
    if lg_d.op_type != OpKind::LinalgMatmul {
        return err(format!(
            "{name}: the router's round-0 logits are `{:?}` — the selection runs over the ROUTER \
             PROJECTION's own dot (`lg = xn @ rwᵀ`), which the 1:1 walk lowers and the fused \
             emission reads through its minted region",
            lg_d.op_type
        ));
    }
    let dot_dims = match lg_d.result_type {
        Some(ktir_core::irtype::IrType::Tensor { dims, .. }) if dims.len() == 2 => {
            [dims[0] as u32, dims[1] as u32]
        }
        _ => {
            return err(format!(
                "{name}: the router's logits dot states no 2-D tensor type, so its extents are \
                 unknown"
            ));
        }
    };
    if dot_dims[0] != rows {
        return err(format!(
            "{name}: the router's logits dot is [{}, {}] against an ids landing of {rows} rows — \
             the selection is row-preserving",
            dot_dims[0], dot_dims[1]
        ));
    }
    let e = dot_dims[1];
    if e_gen != e {
        return err(format!(
            "{name}: the router's lane offsets span {e_gen} experts against a logits dot of {e} \
             columns — the lane select compares the logits against the offset vector, so the two \
             must name the same expert count"
        ));
    }

    // ── THE TAIL: `·TEMP` → the row softmax → `·psel` → the rs store ──
    //
    // `sv = vals·TEMP`; `ex = exp(sv − max(sv))`; `rs = (ex / Σex)·psel` — all f16 (the
    // program's `.to(f32)` legalized away; the door's softmax legs compute the same chain).
    // The values accumulator's one reader outside the rounds is the temperature multiply, and
    // every link after it is proven by a sole-reader check.
    let sv_op = sole_reader_matching(reads, name, vals_final, "temperature multiply", |o| {
        o.op_type == OpKind::ArithMulf
    })?;
    let [sv_a, sv_b] = sv_op.operands[..] else {
        return err(format!(
            "{name}: the router's temperature multiply is not binary"
        ));
    };
    let (temp, sv_tensor) = match (reads.splat_value(sv_a), reads.splat_value(sv_b)) {
        (Some(t), None) => (t, sv_b),
        (None, Some(t)) => (t, sv_a),
        _ => {
            return err(format!(
                "{name}: the router's temperature multiply reads no splatted constant — the \
                 softmax's temperature is `vals · TEMP` with TEMP the router's own constant"
            ));
        }
    };
    if sv_tensor != vals_final {
        return err(format!(
            "{name}: the router's temperature multiply reads a value other than the selected \
             maxima accumulator"
        ));
    }
    let Some(sv) = sv_op.result else {
        return err(format!(
            "{name}: the router's temperature multiply states no result"
        ));
    };
    consumed.push(sv);
    let rmax_op = sole_reader_matching(reads, name, sv, "softmax maxima reduce", |o| {
        o.op_type == OpKind::LinalgReduce
    })?;
    let Some(rmax) = rmax_op.result else {
        return err(format!(
            "{name}: the router's softmax maxima reduce states no result"
        ));
    };
    let rmax_data = router_seeded_reduce(
        reads,
        name,
        rmax,
        OpKind::ArithMaxnumf,
        f64::NEG_INFINITY,
        "the softmax's maxima reduce",
    )?;
    if rmax_data != sv {
        return err(format!(
            "{name}: the router's softmax maxima reduce reads a value other than the temperature \
             product"
        ));
    }
    consumed.push(rmax);
    let sub_op = sole_reader_matching(reads, name, sv, "softmax subtract", |o| {
        o.op_type == OpKind::ArithSubf
    })?;
    let [sub_a, sub_b] = sub_op.operands[..] else {
        return err(format!(
            "{name}: the router's softmax subtract is not binary"
        ));
    };
    let sub_other = if sub_a == sv {
        sub_b
    } else if sub_b == sv {
        sub_a
    } else {
        return err(format!(
            "{name}: the router's softmax subtract reads the temperature product on neither side"
        ));
    };
    let rmax_b_src = plumbing_source(reads, sub_other, &mut consumed);
    if rmax_b_src != rmax {
        return err(format!(
            "{name}: the router's softmax subtract reads a value other than the maxima reduce's \
             broadcast — the softmax is `exp(sv − max(sv))`"
        ));
    }
    let Some(sub) = sub_op.result else {
        return err(format!(
            "{name}: the router's softmax subtract states no result"
        ));
    };
    consumed.push(sub);
    let ex_op = sole_reader_matching(reads, name, sub, "softmax exp", |o| {
        o.op_type == OpKind::MathExp
    })?;
    let Some(ex) = ex_op.result else {
        return err(format!("{name}: the router's softmax exp states no result"));
    };
    consumed.push(ex);
    let den_op = sole_reader_matching(reads, name, ex, "softmax sum reduce", |o| {
        o.op_type == OpKind::LinalgReduce
    })?;
    let Some(den) = den_op.result else {
        return err(format!(
            "{name}: the router's softmax sum reduce states no result"
        ));
    };
    let den_data = router_seeded_reduce(
        reads,
        name,
        den,
        OpKind::ArithAddf,
        0.0,
        "the softmax's sum reduce",
    )?;
    if den_data != ex {
        return err(format!(
            "{name}: the router's softmax sum reduce reads a value other than the exp product"
        ));
    }
    consumed.push(den);
    let div_op = sole_reader_matching(reads, name, ex, "softmax divide", |o| {
        o.op_type == OpKind::ArithDivf
    })?;
    let [div_a, div_b] = div_op.operands[..] else {
        return err(format!("{name}: the router's softmax divide is not binary"));
    };
    let div_other = if div_a == ex {
        div_b
    } else if div_b == ex {
        div_a
    } else {
        return err(format!(
            "{name}: the router's softmax divide reads the exp product on neither side"
        ));
    };
    let den_b_src = plumbing_source(reads, div_other, &mut consumed);
    if den_b_src != den {
        return err(format!(
            "{name}: the router's softmax divide reads a value other than the sum reduce's \
             broadcast — the softmax is `exp / Σexp`"
        ));
    }
    let Some(div) = div_op.result else {
        return err(format!(
            "{name}: the router's softmax divide states no result"
        ));
    };
    consumed.push(div);
    let rs_op = sole_reader_matching(reads, name, div, "route-weights multiply", |o| {
        o.op_type == OpKind::ArithMulf
    })?;
    let [rs_a, rs_b] = rs_op.operands[..] else {
        return err(format!(
            "{name}: the router's route-weights multiply is not binary"
        ));
    };
    let psel_side = if rs_a == div {
        rs_b
    } else if rs_b == div {
        rs_a
    } else {
        return err(format!(
            "{name}: the router's route-weights multiply reads the softmax's probabilities on \
             neither side"
        ));
    };
    if psel_side != psel_final {
        return err(format!(
            "{name}: the router's route-weights multiply reads a value other than the selected \
             per-expert scales — the weights are `softmax(vals·TEMP) · psel`"
        ));
    }
    let Some(rs_val) = rs_op.result else {
        return err(format!(
            "{name}: the router's route-weights multiply states no result"
        ));
    };
    consumed.push(rs_val);
    let rs_store_op = sole_reader_matching(reads, name, rs_val, "route-weights store", |o| {
        o.op_type == OpKind::KtdpStore
    })?;
    let (rs_tid, rs_rows, rs_cols) = store_param(
        k,
        reads,
        rs_store_op,
        name,
        "the router's",
        "the route weights",
    )?;
    if rs_tid == ids_tid {
        return err(format!(
            "{name}: the router's route weights and routed ids land on ONE parameter t{rs_tid} — \
             they are two scratch buffers"
        ));
    }
    if rs_rows != rows || rs_cols != topk {
        return err(format!(
            "{name}: the router's route-weights landing is [{rs_rows}, {rs_cols}] against the ids \
             landing [{rows}, {topk}] — the two landings carry the same rows and slot count"
        ));
    }

    Ok(RouterChain {
        lg,
        pes,
        ids_tid,
        rs_tid,
        rs_loads: direct_loads_of(k, reads, rs_tid),
        temp: temp as f32,
        k: topk,
        emit_at: offs_gen,
        consumed,
    })
}

/// ONE ACCUMULATOR UPDATE — the folded round-0 form (`acc = x·onehot`, the zero init folded
/// away) or the running form (`acc = acc + x·onehot`). Returns `(the previous accumulator,
/// when running; the update multiply's result)`. Order is not trusted on either side of the
/// add, and an add reading a multiply on BOTH sides is refused — that is a different function
/// than the round form.
fn router_acc_update(
    reads: &ProgramReads<'_>,
    name: &str,
    acc: Ssa,
    what: &str,
) -> Result<(Option<Ssa>, Ssa), Error> {
    let Some(d) = reads.def_of(acc) else {
        return err(format!(
            "{name}: the router's {what} accumulator is a function argument, not a computed value"
        ));
    };
    match d.op_type {
        OpKind::ArithMulf => Ok((None, acc)),
        OpKind::ArithAddf => {
            let [a, b] = d.operands[..] else {
                return err(format!(
                    "{name}: the router's {what} accumulator's running add is not binary"
                ));
            };
            match (
                reads.def_of(a).map(|x| x.op_type),
                reads.def_of(b).map(|x| x.op_type),
            ) {
                // ⭐ ROUND 1'S ADD READS THE FOLDED ROUND-0 UPDATE ON ITS PREV SIDE —
                // `acc_0 = x·onehot_0` has no add of its own, so both sides of that ONE add
                // are multiplies. The ROUND INDEX tells them apart: the update's one-hot
                // states j, the folded prev's states j−1, and the gap must be exactly one.
                // (Deeper rounds' prevs are adds, so this is the only place two multiplies
                // meet; a shape that puts them anywhere else does not survive the gap check.)
                (Some(OpKind::ArithMulf), Some(OpKind::ArithMulf)) => {
                    match (router_round_splat(reads, a), router_round_splat(reads, b)) {
                        (Some(ja), Some(jb)) if ja == jb + 1 => Ok((Some(b), a)),
                        (Some(ja), Some(jb)) if jb == ja + 1 => Ok((Some(a), b)),
                        _ => err(format!(
                            "{name}: the router's {what} accumulator's add reads two \
                             multiplies whose one-hots state round indices that are not one \
                             apart — the round form is `acc + x·onehot` with round 0 folded, \
                             so round 1's add is the only place two multiplies meet and their \
                             one-hots state j and j−1"
                        )),
                    }
                }
                (Some(OpKind::ArithMulf), _) => Ok((Some(b), a)),
                (_, Some(OpKind::ArithMulf)) => Ok((Some(a), b)),
                _ => err(format!(
                    "{name}: the router's {what} accumulator's add reads no multiply update — \
                     the round form is `acc + x·onehot`"
                )),
            }
        }
        other => err(format!(
            "{name}: the router's {what} accumulator is `{other:?}` — neither the folded \
             round-0 multiply (`x·onehot`) nor the running add (`acc + x·onehot`)"
        )),
    }
}

/// AN UPDATE'S ONE-HOT OPERAND — the operand defined DIRECTLY by a `tensor.expand_shape`,
/// against the value operand (the round's reduce, resticked through a `linalg.broadcast`).
/// Returns `(the one-hot, the value)`. The discriminator is exact at every round: the one-hot
/// is a `[k]` vector the program widens with a BARE expand — no `linalg.broadcast` roots at
/// it, so the broadcast-chain reader claims none of it and this walk does — while the value
/// side always rides a broadcast.
fn router_onehot_of(
    reads: &ProgramReads<'_>,
    name: &str,
    mulf: Ssa,
    what: &str,
) -> Result<(Ssa, Ssa), Error> {
    let Some(d) = reads.def_of(mulf) else {
        return err(format!(
            "{name}: the router's {what} is a function argument"
        ));
    };
    if d.op_type != OpKind::ArithMulf {
        return err(format!(
            "{name}: the router's {what} is `{:?}`, not a multiply",
            d.op_type
        ));
    }
    let [a, b] = d.operands[..] else {
        return err(format!("{name}: the router's {what} is not binary"));
    };
    match (
        reads.def_of(a).map(|x| x.op_type),
        reads.def_of(b).map(|x| x.op_type),
    ) {
        (Some(OpKind::TensorExpandShape), Some(OpKind::TensorExpandShape)) => {
            // ⭐ TOP-1: at k=1 the value's `[:, None]` widening is an expand
            // too — the `[1, 1]` one-hot needs no `linalg.broadcast` of its
            // own, so the multiply's own broadcasting absorbs the value's —
            // and both sides of the update are expands. The one-hot is still
            // exact ONE LINK DOWN: its expand widens the one-hot select
            // (`where(koffs == j, 1, 0)`), the value's widens the round's
            // reduce — never a select.
            let widens_the_onehot = |v: Ssa| {
                reads.def_of(v).is_some_and(|x| {
                    x.operands.first().is_some_and(|&u| {
                        reads
                            .def_of(u)
                            .is_some_and(|d| d.op_type == OpKind::ArithSelect)
                    })
                })
            };
            let input_kind = |v: Ssa| {
                reads
                    .def_of(v)
                    .and_then(|x| x.operands.first().copied())
                    .and_then(|u| reads.def_of(u))
                    .map(|d| d.op_type)
            };
            match (widens_the_onehot(a), widens_the_onehot(b)) {
                (true, false) => Ok((a, b)),
                (false, true) => Ok((b, a)),
                _ => err(format!(
                    "{name}: the router's {what} reads a widened one-hot on BOTH sides — the \
                     round form is `x · onehot`, and at the top-1 width the two widens root at \
                     {:?} and {:?}, not the one-hot select and the round's reduce",
                    input_kind(a),
                    input_kind(b)
                )),
            }
        }
        (Some(OpKind::TensorExpandShape), _) => Ok((a, b)),
        (_, Some(OpKind::TensorExpandShape)) => Ok((b, a)),
        _ => err(format!(
            "{name}: the router's {what} reads no widened one-hot — the round form multiplies \
             the round's value by the one-hot `select(koffs == j, 1, 0)` widened to `[1, k]`"
        )),
    }
}

/// A splatted constant BY BITS — `splat_value` with the exact-value reading this proof needs:
/// the round constants (−inf, −1.0, 0.0, 1.0) are exact in f16, so the f64 the splat carries
/// names them exactly — and −0.0 ≠ 0.0 (a masked-out lane of −0.0 is a different tie-break
/// input, and fails closed).
fn router_splat_is(reads: &ProgramReads<'_>, v: Ssa, want: f64) -> bool {
    reads
        .splat_value(v)
        .is_some_and(|x| x.to_bits() == want.to_bits())
}

/// THE ROUND INDEX an update multiply's one-hot states — `onehot(mulf) → select → cmpf → the
/// j splat`, read WITHOUT the rest of the round proof. The loop re-proves every link; this
/// answers ONE question, and it is the discriminator for the one place two multiplies meet
/// (round 1's add, whose prev side is round 0's FOLDED update — see
/// [`router_acc_update`]): which of the two is this round's update.
fn router_round_splat(reads: &ProgramReads<'_>, mulf: Ssa) -> Option<u32> {
    let d = reads.def_of(mulf)?;
    if d.op_type != OpKind::ArithMulf {
        return None;
    }
    let oh_b = d.operands.iter().copied().find(|&v| {
        reads
            .def_of(v)
            .is_some_and(|x| x.op_type == OpKind::TensorExpandShape)
    })?;
    let oh = reads.def_of(oh_b)?.operands.first().copied()?;
    let sel = reads.def_of(oh)?;
    if sel.op_type != OpKind::ArithSelect {
        return None;
    }
    let cond = sel.operands.first().copied()?;
    let cmpf = reads.def_of(cond)?;
    if cmpf.op_type != OpKind::ArithCmpf {
        return None;
    }
    let j = cmpf.operands.iter().find_map(|&v| reads.splat_value(v))?;
    if j < 0.0 || j.fract() != 0.0 || j > 255.0 {
        return None;
    }
    Some(j as u32)
}

/// An `arith.cmpf` with the ORDERED-EQUAL predicate — the round compares' own. Returns the
/// two operands; order is not trusted.
fn router_oeq(
    reads: &ProgramReads<'_>,
    name: &str,
    v: Ssa,
    what: &str,
) -> Result<(Ssa, Ssa), Error> {
    let Some(d) = reads.def_of(v) else {
        return err(format!(
            "{name}: the router's {what} is a function argument"
        ));
    };
    if d.op_type != OpKind::ArithCmpf {
        return err(format!(
            "{name}: the router's {what} is `{:?}`, not an `arith.cmpf`",
            d.op_type
        ));
    }
    let oeq = d
        .attributes
        .iter()
        .any(|(k, w)| matches!((k, w), (AttrKey::Predicate, Attr::Str("oeq"))));
    if !oeq {
        return err(format!(
            "{name}: the router's {what} is not an ordered-equal compare — the round form \
             compares lane values with `oeq`"
        ));
    }
    let [a, b] = d.operands[..] else {
        return err(format!("{name}: the router's {what} is not binary"));
    };
    Ok((a, b))
}

/// A `linalg.reduce` BY COMBINER AND SEED — the round ops' own law, one place. Proves the
/// kind, the `ReduceFn` combiner, the `[1]` reduction dims, and the seed splat's VALUE;
/// returns the reduce's DATA operand (the init is the seed splat, identified by value, order
/// not trusted).
fn router_seeded_reduce(
    reads: &ProgramReads<'_>,
    name: &str,
    v: Ssa,
    combiner: OpKind,
    seed: f64,
    what: &str,
) -> Result<Ssa, Error> {
    let Some(d) = reads.def_of(v) else {
        return err(format!(
            "{name}: the router's {what} is a function argument"
        ));
    };
    if d.op_type != OpKind::LinalgReduce {
        return err(format!(
            "{name}: the router's {what} is `{:?}`, not a `linalg.reduce`",
            d.op_type
        ));
    }
    let stated = d.attributes.iter().find_map(|(k, w)| match (k, w) {
        (AttrKey::ReduceFn, Attr::Op(o)) => Some(*o),
        _ => None,
    });
    if stated != Some(combiner) {
        return err(format!(
            "{name}: the router's {what} reduces with `{:?}` — the round form states `{:?}`",
            stated, combiner
        ));
    }
    let dims = d.attributes.iter().find_map(|(k, w)| match (k, w) {
        (AttrKey::Dimensions, Attr::IntList(x)) => Some(*x),
        _ => None,
    });
    if !matches!(dims, Some([1])) {
        return err(format!(
            "{name}: the router's {what} reduces over dims {dims:?} — the round form reduces \
             over the lane axis `[1]`, one value per token"
        ));
    }
    let [a, b] = d.operands[..] else {
        return err(format!("{name}: the router's {what} is not binary"));
    };
    // The seed is the splat; the data is the other side. A seed on both sides is a different
    // function (a reduce of a constant), and no seed at all is not this form.
    if router_splat_is(reads, a, seed) && !router_splat_is(reads, b, seed) {
        Ok(b)
    } else if router_splat_is(reads, b, seed) && !router_splat_is(reads, a, seed) {
        Ok(a)
    } else {
        err(format!(
            "{name}: the router's {what} reads no lone seed splat {seed} — the round form seeds \
             the reduce with a splatted constant and reduces the data on the other side"
        ))
    }
}

/// `tl.arange`'s landing: a rank-1 `tensor.generate` whose region yields its own block
/// argument — the IDENTITY function, so the result IS the offset vector. Proven, not assumed:
/// a generate is the one op in this body that could carry ANY region, and the round compares
/// mean "lane offset", not "any `[n]` vector". Returns the generate's width.
fn router_identity_generate(reads: &ProgramReads<'_>, v: Ssa) -> Option<u32> {
    let d = reads.def_of(v)?;
    if d.op_type != OpKind::TensorGenerate {
        return None;
    }
    let region = d.regions.first()?;
    if region.len() != 2 {
        return None;
    }
    let [bb0, yld] = [*region.first()?, *region.get(1)?];
    if bb0.op_type != OpKind::RegionBb0Args || yld.op_type != OpKind::TensorYield {
        return None;
    }
    let names = bb0.attributes.iter().find_map(|(k, w)| match (k, w) {
        (AttrKey::Names, Attr::Ssas(xs)) => Some(*xs),
        _ => None,
    })?;
    let [x] = names else {
        return None;
    };
    if yld.operands.len() != 1 || yld.operands[0] != *x {
        return None;
    }
    match d.result_type {
        Some(ktir_core::irtype::IrType::Tensor { dims, .. }) if dims.len() == 1 => {
            u32::try_from(dims[0]).ok()
        }
        _ => None,
    }
}

/// [`ProgramReads::through_plumbing`] THAT COLLECTS — the same bounded (≤3 steps) back-walk
/// over `tensor.expand_shape`/`tensor.collapse_shape`/`linalg.broadcast`, pushing every link
/// it passes through into `consumed`. The chain reader answers IDENTITY; a composite
/// recogniser also owes consumption for its own interior, and the bare `expand_shape` the
/// one-hot rides is claimed by no other reader. Every resolution this proof makes goes
/// through here, so no link the proof walked is left for the 1:1 walk to refuse.
fn plumbing_source(reads: &ProgramReads<'_>, v: Ssa, consumed: &mut Vec<Ssa>) -> Ssa {
    let mut cursor = v;
    for _ in 0..3 {
        let Some(d) = reads.def_of(cursor) else { break };
        if !matches!(
            d.op_type,
            OpKind::TensorExpandShape | OpKind::TensorCollapseShape | OpKind::LinalgBroadcast
        ) {
            break;
        }
        let Some(src) = d.operands.first().copied() else {
            break;
        };
        consumed.push(cursor);
        cursor = src;
    }
    cursor
}

/// The ONE reader of `v` matching `pred` — the tail's finder. Zero or several matches are
/// both refusals: the softmax tail is a single-consumer chain, and a second matching reader
/// is a different function than the one the fused emission computes.
fn sole_reader_matching<'f>(
    reads: &ProgramReads<'f>,
    name: &str,
    v: Ssa,
    what: &str,
    pred: impl Fn(&Operation<'static>) -> bool,
) -> Result<&'f Operation<'static>, Error> {
    let mut hits = reads
        .readers_of(v)
        .iter()
        .map(|&p| reads.op_at(p))
        .filter(|o| pred(o));
    let Some(only) = hits.next() else {
        return err(format!(
            "{name}: the router's {what} — no reader matches; the softmax tail is a \
             single-consumer chain"
        ));
    };
    if hits.next().is_some() {
        return err(format!(
            "{name}: the router's {what} — SEVERAL readers match; the softmax tail is a \
             single-consumer chain and a second one is a different function"
        ));
    }
    Ok(only)
}

/// The PARAMETER a store writes — `ktdp.store` ← `ktdp.construct_access_tile` ←
/// `ktdp.construct_memory_view` ← argument → binding, plus the view's own `[rows, cols]`.
/// The same walk [`region_for_operand`] takes for loads, read here for the stores the fused
/// chains replace (the router's two scratch stores, the combine's o): their parameters go
/// WRITE-DEAD or write-direct (the emission lands its own minted buffers or writes the output
/// itself), so this is the only place their bindings are ever named. `whose` spells the door
/// in the refusals, so each chain's wording stays its own.
fn store_param(
    k: &KtirNode,
    reads: &ProgramReads<'_>,
    store: &Operation<'static>,
    name: &str,
    whose: &str,
    what: &str,
) -> Result<(u32, u32, u32), Error> {
    let f = &k.func;
    let tile_v = store.operands.get(1).copied().ok_or_else(|| Error {
        message: format!("{name}: {whose} {what} store states no tile"),
    })?;
    let tile = reads.def_of(tile_v).ok_or_else(|| Error {
        message: format!("{name}: {whose} {what} store's tile is a function argument"),
    })?;
    if tile.op_type != OpKind::KtdpConstructAccessTile {
        return err(format!(
            "{name}: {whose} {what} store writes through `{:?}`, not an access tile",
            tile.op_type
        ));
    }
    let view_v = tile.operands.first().copied().ok_or_else(|| Error {
        message: format!("{name}: {whose} {what} store's tile states no view"),
    })?;
    let view = reads.def_of(view_v).ok_or_else(|| Error {
        message: format!("{name}: {whose} {what} store's view is a function argument"),
    })?;
    if view.op_type != OpKind::KtdpConstructMemoryView {
        return err(format!(
            "{name}: {whose} {what} store's tile reinterprets `{:?}`, not a memory view",
            view.op_type
        ));
    }
    let ptr = view.operands.first().copied().ok_or_else(|| Error {
        message: format!("{name}: {whose} {what} store's view states no pointer"),
    })?;
    let idx = f
        .arguments
        .iter()
        .position(|(a, _)| *a == ptr)
        .ok_or_else(|| Error {
            message: format!(
                "{name}: {whose} {what} store's view addresses no function argument — \
                     the scratch parameters are the kernel's own"
            ),
        })?;
    let tid = k.bindings.get(idx).map(|b| b.get()).ok_or_else(|| Error {
        message: format!("{name}: {whose} {what} store's parameter has no binding"),
    })?;
    let Some((rows, cols)) = super::lower_ktir_to_superdsc::shape_2d(view) else {
        return err(format!(
            "{name}: {whose} {what} store's view states no 2-D shape, so the landing's \
             extents are unknown"
        ));
    };
    Ok((tid, rows, cols))
}

/// Every DIRECT load of the parameter bound to `tid` — a `ktdp.load` over a plain
/// `ktdp.construct_access_tile` (NOT the indirect form; the gathers census owns those) whose
/// view reinterprets `tid`'s argument. The scratch census's iteration set.
fn direct_loads_of(k: &KtirNode, reads: &ProgramReads<'_>, tid: u32) -> Vec<Ssa> {
    let f = &k.func;
    f.operations
        .iter()
        .filter(|o| o.op_type == OpKind::KtdpLoad)
        .filter_map(|o| {
            let res = o.result?;
            let tile = reads.def_of(*o.operands.first()?)?;
            if tile.op_type != OpKind::KtdpConstructAccessTile {
                return None;
            }
            let view = reads.def_of(*tile.operands.first()?)?;
            if view.op_type != OpKind::KtdpConstructMemoryView {
                return None;
            }
            let ptr = view.operands.first().copied()?;
            let idx = f.arguments.iter().position(|(a, _)| *a == ptr)?;
            (k.bindings.get(idx)?.get() == tid).then_some(res)
        })
        .collect()
}

/// ⭐⭐⭐ THE SCRATCH CENSUS — the write-dead law's soundness obligation, stated at the door.
///
/// The fused router writes its own minted landings and NEVER the scratch parameters
/// `desc_ids`/`desc_rs`: the program's stores into them lower to nothing (write-dead), so a
/// reader that is not a chain this door recognises would read a buffer nothing wrote. The
/// claim therefore owes the program a proof: EVERY direct load of either parameter is
/// consumed by a recognised chain — its result, or the value it feeds through rank plumbing,
/// is in the claimed set. A reader outside the claimed set is refused BY NAME, because the
/// alternative is a well-formed bundle computing garbage. Gathered reads need no census here:
/// an indirect tile is the gathers census's own iteration set, and the w8a8 claims already
/// joined theirs.
fn router_scratch_census(
    k: &KtirNode,
    reads: &ProgramReads<'_>,
    routers: &[RouterChain],
    claimed: &std::collections::HashSet<Ssa>,
) -> Result<(), Error> {
    let f = &k.func;
    for c in routers {
        for (tid, what) in [
            (c.ids_tid, "the routed ids"),
            (c.rs_tid, "the route weights"),
        ] {
            for load in direct_loads_of(k, reads, tid) {
                // Forward through rank plumbing; every COMPUTE reader must be claimed. A load
                // nothing reads lowers to nothing (loads are plumbing) and owes no proof.
                let mut stack = vec![load];
                while let Some(v) = stack.pop() {
                    for &p in reads.readers_of(v) {
                        let r = reads.op_at(p);
                        if r.result.is_some_and(|res| claimed.contains(&res)) {
                            continue;
                        }
                        if matches!(
                            r.op_type,
                            OpKind::TensorExpandShape
                                | OpKind::TensorCollapseShape
                                | OpKind::LinalgBroadcast
                        ) {
                            if let Some(res) = r.result {
                                stack.push(res);
                            }
                            continue;
                        }
                        return err(format!(
                            "{}: {what} (t{tid}) is read by `{:?}`, which no recognised chain \
                             claims — the fused router writes its own minted landings and never \
                             the scratch parameter, so this read would address a buffer nothing \
                             wrote. The combine rung is the recogniser for the route-weights read; \
                             any other reader is a shape this door refuses",
                            f.name, r.op_type
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  THE EXPERT COMBINE — the fused kernel's last section, rooted at the route-weights read.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ⭐⭐⭐ ONE EXPERT-COMBINE CHAIN — the weighted sum that folds the pairs' down outputs back
/// into the token stream, `o = Σ_j down_j·w_j`, claimed whole from the router's
/// route-weights read.
///
/// # THE ROOT IS THE ROUTE-WEIGHTS READ
///
/// The kernel's combine RELOADS the route weights the router stored (`wrow = rw1_desc.load`) —
/// one of [`RouterChain::rs_loads`], the scratch census's own iteration entries — extracts each
/// pair's weight with a ONE-HOT select-sum (`where(koffs == j, wrow, 0)`, reduced over the k
/// lanes), and multiplies it into that pair's down output. The router's fire point already
/// registered every rs load in `inter` → the minted route-weights product (slot j compact at
/// lane j, the `re*` legs' law), so the combine's fire resolves the read to the emission's own
/// buffer through the map every cross-chain handoff uses — never the write-dead scratch
/// parameter.
///
/// # THE ACCUMULATION ORDER, AND WHY THE EMISSION KEEPS IT
///
/// moe.py seeds `acc` with `tl.zeros` and adds per pair in slot order — but the TTIR FOLDS the
/// zero seed (measured, the checked-in dump): `k−1` adds, LEFT-DEEP in slot order, pair 0's
/// product seeding the running sum. torch-spyre consumes the same `make_ttir` fold, so the
/// folded form IS the reference semantics. The emission states the same chain — slot 0's
/// product seeds the accumulator, every later slot is a product leg then an add leg, the final
/// add writes the output — through [`super::lower_ktir_to_superdsc::expert_combine_pair_legs`],
/// the ONE leg implementation the per-`Program` door shares, so the f16 op sequence is the
/// TTIR's own, operand for operand.
///
/// The one spelling difference, stated: the kernel EXTRACTS the weight as a zero-seeded
/// select-sum; the emission reads LANE j of the router's minted product. Value-identical for
/// route weights: softmax outputs are ≥ 0 (never −0.0), and the only divergence class of a
/// zero-init sum vs a lane read is the SIGN OF ZERO — unreachable here, and it washes out in
/// the add regardless (`s + ±0.0 = s` for `s ≠ 0`). The per-`Program` door already made this
/// exact choice.
#[derive(Clone, Debug)]
pub struct ExpertCombineChain {
    /// The route-weights read — ONE of [`RouterChain::rs_loads`], resolved to the minted
    /// route-weights product through `inter` at the fire point. Not consumed: a load is
    /// plumbing, and the proof claims its READERS.
    pub wrow: Ssa,
    /// The pair count — the one-hot select count, cross-checked against the router's own `k`,
    /// the read's own width, and the down projection's.
    pub k: u32,
    /// The ONE shared `koffs` widen — `[k] → [1, k]`, the combine's OWN expand (CSE does not
    /// merge it with the router's unexpanded `[k]` form — measured). Claimed by this chain.
    pub koffs: Ssa,
    /// The `desc_o` parameter's binding — the function's OUTPUT, which the final add's leg
    /// writes directly (the store folds into the producing add, the same law the act store
    /// and the rs/ids stores ride).
    pub o_tid: u32,
    /// The o tile's row count — `1`, the door's m=1 law (the read, the products, and the o
    /// view all state it; a wider m refuses by name).
    pub rows: u32,
    /// The combine's output width — the down projection's own `n`.
    pub n: u32,
    /// Per pair `j`: the product `mulf` — the FIRE ANCHOR. The walk lowers pair `j`'s legs at
    /// this op's own position (the per-position law: the unrolled kernel interleaves
    /// `gelu_j → down_j → combine_j`, so the emitted op sequence is the kernel's own
    /// interleaving).
    pub products: Vec<Ssa>,
    /// Per pair `j`: the down chain terminal the product reads — resolved to its window
    /// `Region` through `inter` (the down fire registers every terminal at the projection's
    /// mint).
    pub terminals: Vec<Ssa>,
    /// Everything else the proof touched: the `k` lane compares, `k` selects, `k` extraction
    /// reduces, `k−1` accumulating adds, and the ONE koffs widen. The read, the zero splats,
    /// the j splats, the reduce seeds, and the o store are `is_plumbing`; the weight's
    /// expand/collapse/broadcast spray is `bcast_consumed` (the broadcast chains reader's own
    /// claim); the products are the fire anchors.
    pub consumed: Vec<Ssa>,
}

/// ⭐⭐⭐ THE EXPERT COMBINE CHAINS THIS FUNCTION HOLDS — one per route-weights read that has
/// compute readers, proved WHOLE or refused BY NAME.
///
/// THE ROOT SET is [`RouterChain::rs_loads`] — the scratch census's own iteration set. A load
/// nothing reads lowers to nothing and owes no proof (the census's own law); a load whose
/// compute readers are not exactly one combine's one-hot selects is a reader outside every
/// recognised chain — the census would refuse it, and this proof refuses FIRST, naming the
/// link. A router-only kernel holds no rs reads and therefore no combine: the composed
/// router kernel lowers unchanged.
pub fn program_expert_combine_chains(
    k: &KtirNode,
    reads: &ProgramReads<'_>,
    routers: &[RouterChain],
    downs: &[W8a8DownProjection],
) -> Result<Vec<ExpertCombineChain>, Error> {
    let mut out = Vec::new();
    for c in routers {
        for &wrow in &c.rs_loads {
            // The read's compute readers, forwarding through rank plumbing — the census's own
            // walk. Each is carried WITH the value it was reached through, so the select proof
            // can state the TRUE side by identity even under forwarding. Everything else must
            // be the combine's one-hot select.
            let mut selects: Vec<(&Operation<'static>, Ssa)> = Vec::new();
            let mut stack = vec![wrow];
            while let Some(v) = stack.pop() {
                for &p in reads.readers_of(v) {
                    let r = reads.op_at(p);
                    match r.op_type {
                        OpKind::ArithSelect => selects.push((r, v)),
                        OpKind::TensorExpandShape
                        | OpKind::TensorCollapseShape
                        | OpKind::LinalgBroadcast => {
                            if let Some(res) = r.result {
                                stack.push(res);
                            }
                        }
                        _ => {
                            return err(format!(
                                "{}: the route weights (t{}) is read by `{:?}`, which no \
                                 recognised chain claims — the expert combine is the recogniser \
                                 for the route-weights read, and this reader is not its form",
                                k.func.name, c.rs_tid, r.op_type
                            ));
                        }
                    }
                }
            }
            if selects.is_empty() {
                // A load nothing reads lowers to nothing (loads are plumbing) and owes no
                // proof — the census's own law.
                continue;
            }
            out.push(expert_combine_chain_at(
                k, reads, wrow, c.k, selects, downs,
            )?);
        }
    }
    Ok(out)
}

/// Prove ONE combine chain at a route-weights read whose compute readers are the one-hot
/// selects — the whole proof, from the read forward. Every fact is read from the program;
/// every disagreement is a refusal naming the link, never a guess past it.
///
/// The walk runs FORWARD from the read (the router's runs backward from its store — the only
/// value the combine PINS is the read itself): the read's compute readers are exactly the `k`
/// one-hot selects; each select's sole consumer is the extraction reduce; the reduce's spray
/// feeds the pair's product multiply; the product's other side is that pair's OWN down
/// terminal (the per-pair agreement); the products accumulate left-deep in slot order; and
/// the final add's (at `k=1` the single product's) sole consumer is the o store, whose
/// parameter the last leg writes directly.
fn expert_combine_chain_at(
    k: &KtirNode,
    reads: &ProgramReads<'_>,
    wrow: Ssa,
    router_k: u32,
    selects: Vec<(&Operation<'static>, Ssa)>,
    downs: &[W8a8DownProjection],
) -> Result<ExpertCombineChain, Error> {
    let f = &k.func;
    let name = f.name;
    let mut consumed: Vec<Ssa> = Vec::new();

    // ── THE READ'S OWN SHAPE: `[1, k]`, the one token's row the m=1 door lowers ──
    let load = reads.def_of(wrow).ok_or_else(|| Error {
        message: format!("{name}: the combine's route-weights read is a function argument"),
    })?;
    let Some((m, lk)) = super::lower_ktir_to_superdsc::shape_2d(load) else {
        return err(format!(
            "{name}: the combine's route-weights read states no 2-D shape, so its extents are \
             unknown"
        ));
    };
    if m != 1 || lk != router_k {
        return err(format!(
            "{name}: the combine's route-weights read is [{m}, {lk}] — the router proved \
             k={router_k}, and the read is the one token's `[1, k]` row (the door lowers the \
             m=1 decode form)"
        ));
    }
    if selects.len() != router_k as usize {
        return err(format!(
            "{name}: the combine's route-weights read has {} one-hot selects — the router \
             proved k={router_k}, and one select per pair is the combine's own form",
            selects.len()
        ));
    }

    // ── THE ONE-HOT SELECTS, one per pair, each proving its own lane ──
    let mut lanes: Vec<(u32, Ssa)> = Vec::with_capacity(selects.len());
    let mut koffs: Option<Ssa> = None;
    for (s, v) in &selects {
        let Some(sel_res) = s.result else {
            return err(format!(
                "{name}: the combine's one-hot select states no result"
            ));
        };
        let [cond, hit, miss] = s.operands[..] else {
            return err(format!(
                "{name}: the combine's one-hot select is not ternary"
            ));
        };
        // The select MASKS the read: `where(lane == j, wrow, 0)`. The TRUE side is the weights
        // row THIS chain roots at (the reader walk reached the select through it, forwarding
        // included); the FALSE side is a zero splat. An inverted mask is a different function
        // and refuses here.
        if hit != *v {
            return err(format!(
                "{name}: the combine's one-hot select reads the route weights on its FALSE \
                 side — the mask is inverted, a different function than the one-hot this door \
                 lowers"
            ));
        }
        if !router_splat_is(reads, miss, 0.0) {
            return err(format!(
                "{name}: the combine's one-hot select's miss side is not a zero splat — the \
                 one-hot zeroes every lane but the pair's own"
            ));
        }
        // The lane compare: `oeq` over (the koffs widen, the j splat).
        let (a, b) = combine_lane_compare(reads, name, cond)?;
        let (j, ko) = match (combine_splat_index(reads, a), combine_splat_index(reads, b)) {
            (Some(j), None) => (j, b),
            (None, Some(j)) => (j, a),
            _ => {
                return err(format!(
                    "{name}: the combine's one-hot compare states no lone lane-index splat \
                     against the koffs vector — the pair index is read off the splat's VALUE \
                     (the compiler CSEs pair 0's zero with the zeros splat itself), so both \
                     sides carrying indices, or neither, is not this form"
                ));
            }
        };
        match koffs {
            None => koffs = Some(ko),
            Some(prev) if prev == ko => {}
            Some(prev) => {
                return err(format!(
                    "{name}: the combine's one-hot compares read two koffs vectors (s{} and \
                     s{}) — the combine widens the router's lane vector ONCE and every pair's \
                     compare reads it",
                    prev.0, ko.0
                ));
            }
        }
        lanes.push((j, sel_res));
        consumed.push(cond);
        consumed.push(sel_res);
    }
    // The lanes are the pairs, exactly once each: a repeated lane is two pairs wearing one
    // slot, a missing one a pair with no weight.
    lanes.sort_by_key(|(j, _)| *j);
    if !lanes.iter().map(|(j, _)| *j).eq(0..router_k) {
        return err(format!(
            "{name}: the combine's one-hot selects state lanes {:?} — one select per pair \
             0..{router_k} is the combine's own form",
            lanes.iter().map(|(j, _)| *j).collect::<Vec<_>>()
        ));
    }

    // ── THE ONE KOFFS WIDEN every compare shares ──
    let koffs = koffs.expect("a combine with selects named its koffs widen");
    let ko_def = reads.def_of(koffs).ok_or_else(|| Error {
        message: format!("{name}: the combine's koffs vector is a function argument"),
    })?;
    if ko_def.op_type != OpKind::TensorExpandShape {
        return err(format!(
            "{name}: the combine's koffs vector is `{:?}`, not a `tensor.expand_shape` — the \
             combine widens the router's `[k]` lane vector to `[1, k]` for the one-hot mask",
            ko_def.op_type
        ));
    }
    let Some((km, kk)) = super::lower_ktir_to_superdsc::shape_2d(ko_def) else {
        return err(format!(
            "{name}: the combine's koffs widen states no 2-D shape, so its extents are unknown"
        ));
    };
    if km != 1 || kk != router_k {
        return err(format!(
            "{name}: the combine's koffs widen is [{km}, {kk}] — the router proved \
             k={router_k}, and the one-hot mask is the `[1, k]` widen of its lane vector"
        ));
    }
    consumed.push(koffs);

    // ── PER PAIR: the extraction reduce → the spray → the product → the terminal ──
    //
    // Every down terminal, keyed to its projection and chain — the per-pair agreement's map.
    let down_terminal_of: std::collections::HashMap<Ssa, (usize, usize)> = downs
        .iter()
        .enumerate()
        .flat_map(|(pi, p)| {
            p.chains
                .iter()
                .enumerate()
                .map(move |(ci, c)| (c.terminal, (pi, ci)))
        })
        .collect();
    let mut products: Vec<Ssa> = Vec::with_capacity(router_k as usize);
    let mut terminals: Vec<Ssa> = Vec::with_capacity(router_k as usize);
    let mut down_pi: Option<usize> = None;
    for (j, sel_res) in &lanes {
        let j = *j;
        // The extraction: the select's sole consumer is the zero-seeded addf reduce.
        let Some(red) = reads.sole_reader(*sel_res) else {
            return err(format!(
                "{name}: the combine's pair-{j} one-hot select is not read by exactly one op — \
                 the extraction reduce is its only consumer"
            ));
        };
        let Some(r_res) = red.result else {
            return err(format!(
                "{name}: the combine's pair-{j} extraction states no result"
            ));
        };
        let data = combine_weight_reduce(reads, name, r_res)?;
        if data != *sel_res {
            return err(format!(
                "{name}: the combine's pair-{j} extraction reduces s{}, not the one-hot \
                 select's own row — the extraction sums the masked weights row and nothing else",
                data.0
            ));
        }
        consumed.push(r_res);
        // The spray: the extracted weight's sole consumers forward through the rank plumbing
        // (the broadcast chains reader claims the links) to the pair's product multiply — the
        // THIRD multiply of the kernel's per-pair dequant chain, the one the combine owns.
        // SSA is acyclic, so the forward walk terminates; a spray deeper than the chains
        // reader recognises is refused by ITS claim, at the walk.
        let mut cursor = r_res;
        let product = loop {
            let Some(r) = reads.sole_reader(cursor) else {
                return err(format!(
                    "{name}: the combine's pair-{j} extracted weight is not read by exactly \
                     one op — the spray to the pair's product multiply is single-consumer"
                ));
            };
            match r.op_type {
                OpKind::TensorExpandShape
                | OpKind::TensorCollapseShape
                | OpKind::LinalgBroadcast => {
                    let Some(res) = r.result else {
                        return err(format!(
                            "{name}: the combine's pair-{j} weight spray link states no result"
                        ));
                    };
                    cursor = res;
                }
                OpKind::ArithMulf => break r,
                other => {
                    return err(format!(
                        "{name}: the combine's pair-{j} extracted weight is consumed by \
                         `{other:?}`, not the pair's product multiply — the weight scales the \
                         pair's own down output and nothing else"
                    ));
                }
            }
        };
        let Some(p_res) = product.result else {
            return err(format!(
                "{name}: the combine's pair-{j} product states no result"
            ));
        };
        let [a, b] = product.operands[..] else {
            return err(format!(
                "{name}: the combine's pair-{j} product is not binary"
            ));
        };
        // The weight side is the spray this walk just walked; the expert side is the other —
        // ROLES, not positions: fp multiply is commutative to the bit, so the operand order
        // the producer chose is not a fact the proof owes.
        let e_side = if a == cursor {
            b
        } else if b == cursor {
            a
        } else {
            return err(format!(
                "{name}: the combine's pair-{j} product does not read the extracted weight — \
                 the weight scales the pair's own down output, and a product over anything \
                 else is an op this door does not recognise"
            ));
        };
        let Some(&(pi, ci)) = down_terminal_of.get(&e_side) else {
            return err(format!(
                "{name}: the combine's pair-{j} product's expert side (s{}) is not a down \
                 projection's terminal — the combine weights the pair's OWN down output, and a \
                 product over anything else is an op this door does not recognise",
                e_side.0
            ));
        };
        // The per-pair agreement: the terminal's pair IS the lane the one-hot states. A
        // mismatch is two pairs wearing one chain — pair `p`'s down output weighted by slot
        // `q`'s score.
        let pair = downs[pi].chains[ci].pair;
        if pair != j {
            return err(format!(
                "{name}: the combine's lane {j} weights pair {pair}'s down output — the \
                 one-hot's lane and the terminal's pair must agree, and a mismatch is pair \
                 {pair}'s output weighted by slot {j}'s score"
            ));
        }
        match down_pi {
            None => down_pi = Some(pi),
            Some(prev) if prev == pi => {}
            Some(prev) => {
                return err(format!(
                    "{name}: the combine's pairs read down projections {prev} and {pi} — one \
                     combine folds ONE projection's pairs"
                ));
            }
        }
        // The product's own shape: `[1, n]`, the m=1 law at the projection's own width.
        let Some((pm, pn)) = super::lower_ktir_to_superdsc::shape_2d(product) else {
            return err(format!(
                "{name}: the combine's pair-{j} product states no 2-D shape, so its extents \
                 are unknown"
            ));
        };
        let n = downs[pi].n;
        if pm != 1 || pn != n {
            return err(format!(
                "{name}: the combine's pair-{j} product is [{pm}, {pn}] — the door lowers the \
                 m=1 decode form, at the down projection's own width {n}"
            ));
        }
        products.push(p_res);
        terminals.push(e_side);
    }
    let Some(pi) = down_pi else {
        return err(format!(
            "{name}: the combine proved no pairs — unreachable: a combine with selects holds \
             at least one"
        ));
    };
    if downs[pi].k != router_k {
        return err(format!(
            "{name}: the combine holds {router_k} pairs but its down projection holds {} — one \
             combine folds ONE projection's whole top-k",
            downs[pi].k
        ));
    }
    let n = downs[pi].n;

    // ── THE LEFT-DEEP SLOT-ORDER SUM: pair 0's product seeds the running value (the TTIR
    // folds moe.py's zero seed — measured), every later pair's product is read by exactly one
    // add whose other side is the running value, and the running value's own sole reader is
    // that SAME add — so no add feeds anything but the next, and no product feeds anything
    // but its add. The INTRA-ADD operand order is not trusted: fp add is commutative to the
    // bit, so the order the producer chose is not a fact the proof owes. A right-deep or
    // permuted sum is a different ACCUMULATION ORDER, and f16 is not associative — refused
    // rather than silently re-associated. ──
    let mut running = products[0];
    for j in 1..router_k {
        let p = products[j as usize];
        let Some(add) = reads.sole_reader(p) else {
            return err(format!(
                "{name}: the combine's pair-{j} product is not read by exactly one op — each \
                 product feeds the accumulator and nothing else"
            ));
        };
        if add.op_type != OpKind::ArithAddf {
            return err(format!(
                "{name}: the combine's pair-{j} product is consumed by `{:?}`, not the \
                 accumulating add — each product feeds the accumulator and nothing else",
                add.op_type
            ));
        }
        let Some(add_pos) = reads.pos_of_op(add) else {
            return err(format!(
                "{name}: the combine's pair-{j} add is not in the program's own walk — a \
                 walk/recogniser disagreement, not a producer shape"
            ));
        };
        if reads.readers_of(running) != [add_pos] {
            return err(format!(
                "{name}: the combine's accumulator is not read by exactly pair {j}'s add — the \
                 sum is left-deep in slot order (pair 0's product seeds it), and a branching or \
                 reordered accumulation is a different function than the one this door lowers"
            ));
        }
        let [a, b] = add.operands[..] else {
            return err(format!("{name}: the combine's pair-{j} add is not binary"));
        };
        if !((a == running && b == p) || (b == running && a == p)) {
            return err(format!(
                "{name}: the combine's pair-{j} add reads s{}/s{}, not the running sum and \
                 pair {j}'s product — the sum is left-deep in slot order (pair 0's product \
                 seeds it), and any other accumulation order is a different function than the \
                 one this door lowers",
                a.0, b.0
            ));
        }
        let Some(r) = add.result else {
            return err(format!(
                "{name}: the combine's pair-{j} add states no result"
            ));
        };
        consumed.push(r);
        running = r;
    }

    // ── THE FOLDED STORE: the final add's (at k=1 the single product's) sole consumer is
    // the `desc_o` store, whose parameter the last leg writes directly. ──
    let Some(store) = reads.sole_reader(running) else {
        return err(format!(
            "{name}: the combine's result is not read by exactly one op — the o store is its \
             only consumer"
        ));
    };
    if store.op_type != OpKind::KtdpStore {
        return err(format!(
            "{name}: the combine's result is consumed by `{:?}`, not the o store — the \
             weighted sum is the function's output and nothing else's input",
            store.op_type
        ));
    }
    if store.operands.first() != Some(&running) {
        return err(format!(
            "{name}: the combine's o store does not read the weighted sum as its value — the \
             store's VALUE is the final add, and anything else in its place is not this form"
        ));
    }
    let (o_tid, o_rows, o_cols) = store_param(k, reads, store, name, "the expert combine's", "o")?;
    if o_rows != 1 {
        return err(format!(
            "{name}: the combine's o tile is [{o_rows}, {o_cols}] — the door lowers the m=1 \
             decode form, one token's row"
        ));
    }
    if o_cols != n {
        return err(format!(
            "{name}: the combine's o tile is [{o_rows}, {o_cols}] but the down projection's \
             width is {n} — the combine writes the projection's own output width"
        ));
    }

    Ok(ExpertCombineChain {
        wrow,
        k: router_k,
        koffs,
        o_tid,
        rows: o_rows,
        n,
        products,
        terminals,
        consumed,
    })
}

/// The one-hot's LANE COMPARE — `arith.cmpf oeq` over (the koffs widen, the j splat), the
/// combine's own form. Returns the two operands; order is not trusted. The router's
/// [`router_oeq`] asks the same question with the router's wording; the combine states its
/// own so each door's refusals name their own links.
fn combine_lane_compare(reads: &ProgramReads<'_>, name: &str, v: Ssa) -> Result<(Ssa, Ssa), Error> {
    let Some(d) = reads.def_of(v) else {
        return err(format!(
            "{name}: the combine's one-hot compare is a function argument"
        ));
    };
    if d.op_type != OpKind::ArithCmpf {
        return err(format!(
            "{name}: the combine's one-hot select is driven by `{:?}`, not an `arith.cmpf`",
            d.op_type
        ));
    }
    let oeq = d
        .attributes
        .iter()
        .any(|(k, w)| matches!((k, w), (AttrKey::Predicate, Attr::Str("oeq"))));
    if !oeq {
        return err(format!(
            "{name}: the combine's one-hot compare is not an ordered-equal — the one-hot \
             matches the pair's lane with `oeq`"
        ));
    }
    let [a, b] = d.operands[..] else {
        return err(format!(
            "{name}: the combine's one-hot compare is not binary"
        ));
    };
    Ok((a, b))
}

/// A splatted LANE INDEX — an integral splat value in `0..=255`, the one-hot's `j`. Pure
/// predicate; `None` when `v` is not a splat or not an index.
fn combine_splat_index(reads: &ProgramReads<'_>, v: Ssa) -> Option<u32> {
    let j = reads.splat_value(v)?;
    (j >= 0.0 && j.fract() == 0.0 && j <= 255.0).then_some(j as u32)
}

/// The one-hot's EXTRACTED WEIGHT — `linalg.reduce addf` over the lane axis `[1]`, seeded by
/// a zero splat (`tl.sum` of the masked row). Returns the reduce's DATA operand; the seed is
/// identified by value, order not trusted.
fn combine_weight_reduce(reads: &ProgramReads<'_>, name: &str, v: Ssa) -> Result<Ssa, Error> {
    let Some(d) = reads.def_of(v) else {
        return err(format!(
            "{name}: the combine's extracted weight is a function argument"
        ));
    };
    if d.op_type != OpKind::LinalgReduce {
        return err(format!(
            "{name}: the combine's one-hot select is consumed by `{:?}`, not a `linalg.reduce`",
            d.op_type
        ));
    }
    let stated = d.attributes.iter().find_map(|(k, w)| match (k, w) {
        (AttrKey::ReduceFn, Attr::Op(o)) => Some(*o),
        _ => None,
    });
    if stated != Some(OpKind::ArithAddf) {
        return err(format!(
            "{name}: the combine's extraction reduces with `{:?}` — the one-hot sum is an \
             `addf` reduce",
            stated
        ));
    }
    let dims = d.attributes.iter().find_map(|(k, w)| match (k, w) {
        (AttrKey::Dimensions, Attr::IntList(x)) => Some(*x),
        _ => None,
    });
    if !matches!(dims, Some([1])) {
        return err(format!(
            "{name}: the combine's extraction reduces over dims {dims:?} — the one-hot sums \
             over the lane axis `[1]`, one weight per pair"
        ));
    }
    let [a, b] = d.operands[..] else {
        return err(format!("{name}: the combine's extraction is not binary"));
    };
    if router_splat_is(reads, a, 0.0) && !router_splat_is(reads, b, 0.0) {
        Ok(b)
    } else if router_splat_is(reads, b, 0.0) && !router_splat_is(reads, a, 0.0) {
        Ok(a)
    } else {
        err(format!(
            "{name}: the combine's extraction reads no lone zero seed — the one-hot sum seeds \
             the reduce with a splatted zero and reduces the masked row on the other side"
        ))
    }
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

    // THE TANH-GELU LONGHANDS, read BEFORE the silu reader they front for. The gelu
    // shares the silu longhand's five op kinds with a different numerator, so the silu
    // reader's fail-closed discriminator refuses it — and the pre-reads gate the whole
    // per-op walk, so nothing downstream is reachable while a gelu stands unclaimed.
    // [`program_gelu_mul_chains`] claims each chain only after proving it exactly
    // (constants read off the splats, cubic association, island narrowed back), and the
    // silu reader below skips the claims, so a mangled gelu is still a refusal — the
    // silu reader's, naming the link.
    let gelus = program_gelu_mul_chains(f)?;
    // THE ONE FUSED CHAIN THIS DOOR READS, before the 1:1 walk rather than inside it. See
    // [`program_silu_mul_chain`] for why the module header's "recognises NOTHING" survives this: the
    // device has `OpFunc::Silu` and KTIR has no op for it, so the longhand is the only spelling a
    // producer HAS — and lowering it op-by-op computes a different algorithm, not a slower one.
    let silus = silu_mul_chains(f, &gelus)?;
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
    // ⭐⭐⭐ THE W8A8 GATE/UP SLAB GEMMS, read BEFORE the gathers census they claim out of.
    // [`program_w8a8_gemm_chains`] proves each bank's whole top-k of chains — ONE quantize
    // spine, per-pair gathered fp8 slab gemms, the two-multiply dequant — and the walk below
    // lowers each projection as ONE `expert_matmul_fused_m1`. The claims are the chains'
    // tiles: consumed by the fusion, not forgiven by the census (the law
    // [`gathers_of_excluding`] states), so the discriminator still judges every bank no
    // recogniser claimed — the down projection's, until its own reader lands.
    let w8a8s = program_w8a8_gemm_chains(k)?;
    let mut w8a8_claims: Vec<Ssa> = w8a8s
        .iter()
        .flat_map(|p| p.chains.iter().flat_map(|c| [c.weight_tile, c.scale_tile]))
        .collect();
    // Everything the fusions destroy — the spine's eleven compute ops (ONE set: gate and up
    // are two projections over one expert input, proven by `(a_load, qx, sx)` equality) and
    // each chain's three interior links. The terminals are deliberately absent: they are the
    // projections' output windows.
    let w8a8_consumed: std::collections::HashSet<Ssa> = w8a8s
        .iter()
        .flat_map(|p| p.consumed.iter().copied())
        .collect();
    // Every chain's terminal, keyed to its projection — the value the REST of the program
    // reads (the gelu chains' gate and up), resolved through the window the hook records.
    let w8a8_terminals: std::collections::HashMap<Ssa, usize> = w8a8s
        .iter()
        .enumerate()
        .flat_map(|(pi, p)| p.chains.iter().map(move |c| (c.terminal, pi)))
        .collect();
    // ⭐⭐⭐ THE W8A8 DOWN PROJECTIONS, read before the gathers census for the same reason
    // the gate/up chains are: [`program_w8a8_down_chains`] proves each bank's whole top-k of
    // chains — each pair's OWN quantize over its OWN round-tripped act window, the gathered
    // fp8 slab gemm, the two-multiply dequant — and the walk below lowers each PAIR's leg
    // set at ITS OWN terminal (the per-position law: the unrolled kernel interleaves
    // `gelu_j → down_j → combine_j`, so a single fire at the first terminal would read act
    // windows whose gelu writers have not run). The claims are the chains' tiles, joined to
    // the gate/up claims so the census judges neither reader's banks.
    let downs = program_w8a8_down_chains(k, &gelus)?;
    w8a8_claims.extend(
        downs
            .iter()
            .flat_map(|p| p.chains.iter().flat_map(|c| [c.weight_tile, c.scale_tile])),
    );
    // Everything the down fusions destroy — each pair's OWN eleven-op spine (the down form
    // states one whole quantize per pair; nothing is shared) plus its three interior links.
    // The terminals are deliberately absent: they are the projection's output windows.
    let down_consumed: std::collections::HashSet<Ssa> = downs
        .iter()
        .flat_map(|p| p.consumed.iter().copied())
        .collect();
    // Every down chain's terminal, keyed to its projection AND its chain — the walk fires
    // at EVERY terminal, each emitting its own pair's legs into the projection's ONE output
    // intermediate (minted once, at the projection's first terminal).
    let down_terminals: std::collections::HashMap<Ssa, (usize, usize)> = downs
        .iter()
        .enumerate()
        .flat_map(|(pi, p)| {
            p.chains
                .iter()
                .enumerate()
                .map(move |(ci, c)| (c.terminal, (pi, ci)))
        })
        .collect();
    // The down projections' minted output intermediates, keyed by projection index.
    let mut down_out_tids: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();
    // The fp8 activation-quantize dedup set is per-BUNDLE: gate and up are two projections
    // over ONE expert input, so the emission's quantize is emitted once and named once —
    // the same dedup law the per-`Program` door's fp8 matmul states across its ops.
    let mut quantized: std::collections::HashSet<String> = std::collections::HashSet::new();
    // ⭐⭐⭐ THE ROUTER CHAINS — the round-form top-k, the temperature softmax, and the
    // per-expert scale, claimed WHOLE from the ids store backward. [`program_router_chains`]
    // proves every link (the rounds' shared one-hot, the three spines, the mask continuity,
    // the tail), and the walk below lowers the whole chain at ONE fire point: the lane-offset
    // identity generate. The claims are the chain's interior — consumed by the fusion, so the
    // 1:1 walk skips them (the body's compares/selects/generates have no `Program` anyway).
    let routers = program_router_chains(k)?;
    let router_consumed: std::collections::HashSet<Ssa> = routers
        .iter()
        .flat_map(|c| c.consumed.iter().copied())
        .collect();
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
    //
    // ⭐ MINUS THE W8A8 CLAIMS, so a projection the recogniser proved whole is not also held to the
    // single-pair census law its per-pair tiles would break — the claimed tiles are this door's to
    // lower, not the discriminator's to judge.
    let gathers = gathers_of_excluding(k, &w8a8_claims)?;
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

    // ⭐ THE PROGRAM'S DEF/USE, built ONCE for the walk below. The two scale tiebreaks ask
    // def-of questions per `arith.mulf`, and a mulf-heavy fused kernel would pay for a fresh
    // def/use walk per op — expansion cost this door refuses to add. The chain readers above
    // each build their own index because their signatures are public and function-shaped; the
    // walk is private to here.
    let reads = program_reads(f);

    // ⭐⭐⭐ THE EXPERT COMBINE CHAINS — the weighted sum that folds the pairs' down outputs
    // back into the token stream, claimed whole from the router's route-weights reads (the
    // scratch census's own iteration set). [`program_expert_combine_chains`] proves every
    // link — the one-hot selects, the extraction reduces, the per-pair agreement with the
    // down terminals, the left-deep slot-order sum, the folded o store — and the walk below
    // lowers each PAIR's legs at that pair's own product: the per-position law, the same
    // `gelu_j → down_j → combine_j` interleaving the unrolled kernel states. A router-only
    // kernel holds no rs reads and therefore no combine; the composed kernel lowers
    // unchanged.
    let combines = program_expert_combine_chains(k, &reads, &routers, &downs)?;
    let combine_consumed: std::collections::HashSet<Ssa> = combines
        .iter()
        .flat_map(|c| c.consumed.iter().copied())
        .collect();
    // Every chain's product, keyed to its chain and pair — the walk's fire anchors.
    let combine_products: std::collections::HashMap<Ssa, (usize, u32)> = combines
        .iter()
        .enumerate()
        .flat_map(|(ci, c)| {
            c.products
                .iter()
                .enumerate()
                .map(move |(j, &p)| (p, (ci, j as u32)))
        })
        .collect();

    // ⭐⭐⭐ THE SCRATCH CENSUS — the write-dead law's soundness obligation, BEFORE the walk:
    // the fused router writes its own minted landings and never the `desc_ids`/`desc_rs`
    // parameters, so every direct load of either must be consumed by a chain this door
    // recognises. The claimed set is the union of every chain reader's — one proof, so no
    // reader can be judged against a set its own chain did not join. See
    // [`router_scratch_census`].
    if !routers.is_empty() {
        let mut claimed = bcast_consumed.clone();
        claimed.extend(silus.iter().flat_map(|c| c.consumed.iter().copied()));
        claimed.extend(gelus.iter().flat_map(|c| c.consumed.iter().copied()));
        claimed.extend(rmsnorms.iter().flat_map(|c| c.consumed.iter().copied()));
        claimed.extend(w8a8_consumed.iter().copied());
        claimed.extend(down_consumed.iter().copied());
        claimed.extend(router_consumed.iter().copied());
        claimed.extend(combine_consumed.iter().copied());
        router_scratch_census(k, &reads, &routers, &claimed)?;
    }

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
                // The gelu chain's interior — the cubic's three products and sum, the e-argument,
                // and the same four spine results — same law, one fused `Program::ExpertGatedAct`.
                || gelus.iter().any(|c| c.consumed.contains(&r))
                || rmsnorms.iter().any(|c| c.consumed.contains(&r))
                // The expand/collapse/broadcast chain itself: no descriptor, the consumer states it.
                || bcast_consumed.contains(&r)
                // The W8A8 projections' interiors — the shared quantize spine and each chain's
                // collapsed slab, dot, and scale multiply — same law, one fused
                // `expert_matmul_fused_m1` per projection.
                || w8a8_consumed.contains(&r)
                // The DOWN projections' interiors — each pair's OWN quantize spine (nothing
                // shared) plus its three links — same law, one `expert_down_fused_m1` leg set
                // per pair, at that pair's own terminal.
                || down_consumed.contains(&r)
                // The ROUTER'S interior — the rounds, the converts, and the softmax tail —
                // same law, one fused emission at the lane-offset generate (the fire arm
                // below).
                || router_consumed.contains(&r)
                // The EXPERT COMBINE'S interior — the one-hot compares, selects, and
                // extraction reduces, the accumulating adds, and the ONE koffs widen — same
                // law, one legs-per-pair emission at each pair's own product (the fire arm
                // below).
                || combine_consumed.contains(&r)
        }) {
            continue;
        }
        // ⭐⭐⭐ A W8A8 PROJECTION'S TERMINAL — the op the fused emission stands for. Every chain's
        // terminal names its pair's `[1, n]` row of the projection's output; the walk lowers the
        // WHOLE projection at the FIRST terminal in program order and every other terminal is
        // lowered by nothing at all — its value is a WINDOW of the output the emission already
        // wrote, recorded in `inter` for the consumers (the gelu chains' gate and up), the same
        // law the mint arm states for a chain's interior.
        if let Some(r) = op.result
            && let Some(&pi) = w8a8_terminals.get(&r)
        {
            let p = &w8a8s[pi];
            if r != p.emit_at {
                continue;
            }
            // The projection's output intermediate, minted at its LOGICAL `[1, k·n]` — the width
            // the pointwise dequant (`emo`) writes, so this is the POINTWISE arm's own mint law,
            // not the matmul arm's device-width reservation: the fused emission's `matmulfp8`
            // products land in the scratch the emission itself reserves, and only the dequant's
            // output reaches this buffer.
            let Some(layout) = layout else {
                return err(format!(
                    "{}: the W8A8 projection over t{} is an intermediate product, which needs \
                     a `BundleLayout` to hold its buffer -- `layout: None` is the unit-test arm \
                     and cannot place one",
                    f.name, p.weight_tid
                ));
            };
            let out_tid = next_tid;
            next_tid += 1;
            layout.synth(crate::place::PlaceId::Act(out_tid), &[1, p.k * p.n]);
            for c in &p.chains {
                // Each pair's terminal resolves to its OWN window of the output — the `[1, n]`
                // columns the emission's dequant wrote for that pair — so a consumer naming the
                // terminal addresses the rows it names, not the projection's whole width.
                inter.insert(
                    c.terminal,
                    Region {
                        tid: out_tid,
                        v_rows: 1,
                        v_cols: p.k * p.n,
                        r_start: 0,
                        r_len: 1,
                        c_start: c.pair * p.n,
                        c_len: p.n,
                        r_cover: (0, 1),
                        c_cover: (c.pair * p.n, (c.pair + 1) * p.n),
                        is_out: true,
                        is_fp8: false,
                    },
                );
            }
            let mut emitted = super::lower_ktir_to_superdsc::expert_matmul_fused_m1(
                &super::lower_ktir_to_superdsc::W8a8FusedFacts {
                    name: f.name,
                    a_tid: p.a_tid,
                    idx_tid: p.idx_tid,
                    w_tid: p.weight_tid,
                    s_tid: p.scale_tid,
                    k: p.k,
                    inn: p.inn,
                    n: p.n,
                    e_count: p.e_count,
                },
                out_tid,
                sym_id_base,
                Some(layout),
                &mut quantized,
            )?;
            out.append(&mut emitted);
            lowered += 1;
            continue;
        }
        // ⭐⭐⭐ A W8A8 DOWN PROJECTION'S TERMINAL — the PER-POSITION firing. Every chain's
        // terminal names its pair's `[1, n]` row of the projection's output, and the walk
        // lowers THAT PAIR'S whole leg set at its own terminal — never the projection's
        // whole at the first: program order is tape order is execution order, and the
        // unrolled kernel interleaves `gelu_j → down_j → combine_j` per pair, so a single
        // fire at the first terminal would read act windows whose gelu writers have not run
        // (pairs 1..k−1) or write out windows the combine has already read. The output
        // intermediate is minted ONCE per projection at its first terminal — `mint_at`, the
        // first terminal in program order, which is the first one this walk meets — and
        // every terminal, the mint one included, emits its own pair's legs into it.
        if let Some(r) = op.result
            && let Some(&(pi, ci)) = down_terminals.get(&r)
        {
            let p = &downs[pi];
            let chain = &p.chains[ci];
            let Some(layout) = layout else {
                return err(format!(
                    "{}: the W8A8 DOWN projection over t{} is an intermediate product, which \
                     needs a `BundleLayout` to hold its buffer -- `layout: None` is the unit-test \
                     arm and cannot place one",
                    f.name, p.weight_tid
                ));
            };
            let out_tid = if r == p.mint_at {
                let out_tid = next_tid;
                next_tid += 1;
                // The projection's output intermediate, minted at its LOGICAL `[1, k·n]` —
                // the same mint law the gate/up hook states: every write into it is
                // pointwise (the matmuls write the XRaw scratch the emission itself
                // reserves), so only the dequant's output reaches this buffer.
                layout.synth(crate::place::PlaceId::Act(out_tid), &[1, p.k * p.n]);
                for c in &p.chains {
                    // Each pair's terminal resolves to its OWN window of the output — the
                    // `[1, n]` columns that pair's dequant wrote — so a consumer naming the
                    // terminal addresses the columns it names.
                    inter.insert(
                        c.terminal,
                        Region {
                            tid: out_tid,
                            v_rows: 1,
                            v_cols: p.k * p.n,
                            r_start: 0,
                            r_len: 1,
                            c_start: c.pair * p.n,
                            c_len: p.n,
                            r_cover: (0, 1),
                            c_cover: (c.pair * p.n, (c.pair + 1) * p.n),
                            is_out: true,
                            is_fp8: false,
                        },
                    );
                }
                down_out_tids.insert(pi, out_tid);
                out_tid
            } else if let Some(&t) = down_out_tids.get(&pi) {
                t
            } else {
                return err(format!(
                    "{}: the W8A8 DOWN projection over t{} reached pair {}'s terminal before its \
                     mint terminal — the walk reads program order and the mint terminal is the \
                     first terminal in it, so this is not reachable",
                    f.name, p.weight_tid, chain.pair
                ));
            };
            let mut emitted = super::lower_ktir_to_superdsc::expert_down_fused_m1(
                &super::lower_ktir_to_superdsc::W8a8DownFacts {
                    name: f.name,
                    a_tid: p.a_tid,
                    idx_tid: p.idx_tid,
                    w_tid: p.weight_tid,
                    s_tid: p.scale_tid,
                    k: p.k,
                    inn: p.inn,
                    n: p.n,
                    e_count: p.e_count,
                },
                out_tid,
                chain.pair,
                sym_id_base,
                Some(layout),
            )?;
            out.append(&mut emitted);
            lowered += 1;
            continue;
        }
        // ⭐⭐⭐ A ROUTER CHAIN'S FIRE POINT — the lane-offset identity generate the rounds'
        // compares read. Everything the emission needs is already in hand at this op: the
        // dot's region (the matmul arm minted it — the dot PRECEDES this generate in program
        // order, because the rounds read it), the pes parameter, and the proofs. The FIVE
        // products are minted HERE — the two `[m_pad, 64]` landings, the `·TEMP` output, the
        // softmax output, and the route weights the combine rung reads — and every direct
        // load of the rs scratch is resolved to the route-weights product in `inter`, so the
        // combine's hook will find the emission's own buffer through the same map every
        // cross-chain handoff uses.
        if let Some(r) = op.result
            && let Some(chain) = routers.iter().find(|c| c.emit_at == r)
        {
            let Some(layout) = layout else {
                return err(format!(
                    "{}: the router over t{} is an intermediate product, which needs a \
                     `BundleLayout` to hold its buffers -- `layout: None` is the unit-test arm \
                     and cannot place one",
                    f.name, chain.ids_tid
                ));
            };
            let lg = inter.get(&chain.lg).copied().ok_or_else(|| Error {
                message: format!(
                    "{}: the router's logits dot (s{}) has no intermediate region — the \
                             matmul that computes it must precede the lane-offset generate this \
                             fire point anchors at, and the walk reads program order",
                    f.name, chain.lg.0
                ),
            })?;
            let rows = lg.v_rows;
            let m_pad = rows.next_multiple_of(crate::work::FP16_ELEMS_PER_STICK);
            // The five products, in mint order: the landings at the legs' own `[m_pad, 64]`
            // law (the pair's transposed output, one stick per token), the three softmax-chain
            // products at `[rows, 64]` (the ·TEMP output, the softmax out, the route weights —
            // the width every consumer addresses, head slots included).
            let ids_out_tid = next_tid;
            next_tid += 1;
            layout.synth(
                crate::place::PlaceId::Act(ids_out_tid),
                &[m_pad, crate::work::FP16_ELEMS_PER_STICK],
            );
            let vals_out_tid = next_tid;
            next_tid += 1;
            layout.synth(
                crate::place::PlaceId::Act(vals_out_tid),
                &[m_pad, crate::work::FP16_ELEMS_PER_STICK],
            );
            let temp_out_tid = next_tid;
            next_tid += 1;
            layout.synth(
                crate::place::PlaceId::Act(temp_out_tid),
                &[rows, crate::work::FP16_ELEMS_PER_STICK],
            );
            let smax_out_tid = next_tid;
            next_tid += 1;
            layout.synth(
                crate::place::PlaceId::Act(smax_out_tid),
                &[rows, crate::work::FP16_ELEMS_PER_STICK],
            );
            let weights_out_tid = next_tid;
            next_tid += 1;
            layout.synth(
                crate::place::PlaceId::Act(weights_out_tid),
                &[rows, crate::work::FP16_ELEMS_PER_STICK],
            );
            // The combine's handoff: every direct rs load resolves to the route-weights
            // product. The census proved each is a claimed chain's read; this records WHERE
            // the emission actually put the weights, so the combine rung's hook reads the
            // emission's own buffer and never the write-dead scratch parameter.
            for l in &chain.rs_loads {
                inter.insert(
                    *l,
                    Region {
                        tid: weights_out_tid,
                        v_rows: rows,
                        v_cols: crate::work::FP16_ELEMS_PER_STICK,
                        r_start: 0,
                        r_len: rows,
                        c_start: 0,
                        c_len: crate::work::FP16_ELEMS_PER_STICK,
                        r_cover: (0, rows),
                        c_cover: (0, crate::work::FP16_ELEMS_PER_STICK),
                        is_out: false,
                        is_fp8: false,
                    },
                );
            }
            let Some(pes) = region_for_operand(k, chain.pes)? else {
                return err(format!(
                    "{}: the router's per-expert scale row (s{}) resolved to no parameter — the \
                     recogniser proved it is a load of a parameter, so this is a walk/recogniser \
                     disagreement and not a producer shape",
                    f.name, chain.pes.0
                ));
            };
            let mut emitted = super::lower_ktir_to_superdsc::router_fused_m1(
                &super::lower_ktir_to_superdsc::RouterFusedFacts {
                    name: f.name,
                    lg,
                    pes,
                    ids_tid: chain.ids_tid,
                    ids_out_tid,
                    vals_out_tid,
                    temp_out_tid,
                    smax_out_tid,
                    weights_out_tid,
                    k: chain.k,
                    temp: chain.temp,
                },
                sym_id_base,
                Some(layout),
            )?;
            out.append(&mut emitted);
            lowered += 1;
            continue;
        }
        // ⭐⭐⭐ AN EXPERT-COMBINE PRODUCT — the PER-POSITION firing, the down arm's own law:
        // the walk lowers THAT PAIR'S legs at the pair's own product multiply, so the emitted
        // op sequence is the kernel's own interleaving `gelu_j → down_j → combine_j` — after
        // the down emission's `emdo_p{j}` wrote its window, before pair `j+1`'s gelu. The
        // legs themselves are [`super::lower_ktir_to_superdsc::expert_combine_pair_legs`] —
        // the ONE implementation the per-`Program` door shares, so the two doors cannot drift
        // about what a combine emits.
        if let Some(r) = op.result
            && let Some(&(ci, j)) = combine_products.get(&r)
        {
            let chain = &combines[ci];
            let Some(layout) = layout else {
                return err(format!(
                    "{}: the expert combine over t{} is an intermediate product, which needs \
                     a `BundleLayout` to hold its buffers -- `layout: None` is the unit-test \
                     arm and cannot place one",
                    f.name, chain.o_tid
                ));
            };
            // The two handoffs, through `inter` — the map every cross-chain handoff uses. The
            // router's fire registered the route-weights read; the down fire registered every
            // terminal at its projection's mint. Both PRECEDE this product in program order
            // (the router's generate stands at the function's head; the mint terminal
            // precedes the first product — measured), so a miss is a walk/recogniser
            // disagreement, not a producer shape.
            let weights = inter.get(&chain.wrow).copied().ok_or_else(|| Error {
                message: format!(
                    "{}: the combine's route-weights read (s{}) has no intermediate region — \
                     the router's fire point must precede the combine's products, and the walk \
                     reads program order",
                    f.name, chain.wrow.0
                ),
            })?;
            let window = inter
                .get(&chain.terminals[j as usize])
                .copied()
                .ok_or_else(|| Error {
                    message: format!(
                        "{}: the combine's pair-{j} down terminal (s{}) has no intermediate \
                         region — the down projection's mint terminal must precede the \
                         combine's products, and the walk reads program order",
                        f.name, chain.terminals[j as usize].0
                    ),
                })?;
            // The corner agreement: the legs' own block corner (`col_of(rows, full, j·n)`)
            // must be the window the down emission recorded for this terminal — the same
            // corner the down door's `dq_off` states. A disagreement is a walk/recogniser
            // drift, refused rather than silently re-windowed.
            if window.v_rows != chain.rows
                || window.v_cols != chain.k * chain.n
                || window.c_start != j * chain.n
                || window.c_len != chain.n
                || weights.v_rows != chain.rows
            {
                return err(format!(
                    "{}: the combine's pair-{j} window is [{}:{}] of a [{}, {}] intermediate \
                     and the route weights are [{}, {}] — the recogniser proved {} rows over a \
                     [{}, {}] down output, so this is a walk/recogniser disagreement and not a \
                     producer shape",
                    f.name,
                    window.c_start,
                    window.c_start + window.c_len,
                    window.v_rows,
                    window.v_cols,
                    weights.v_rows,
                    weights.v_cols,
                    chain.rows,
                    chain.k * chain.n,
                    chain.n
                ));
            }
            let pair_h = crate::emit::rb(&window.name(), chain.rows, chain.k * chain.n);
            let scores_h = crate::emit::rb(
                &weights.name(),
                chain.rows,
                crate::work::FP16_ELEMS_PER_STICK,
            );
            let o_h = crate::emit::rb(&crate::place::act_name(chain.o_tid), chain.rows, chain.n);
            let legs = super::lower_ktir_to_superdsc::ExpertCombineLegs {
                rows: chain.rows,
                w: chain.n,
                full: chain.k * chain.n,
                k: chain.k,
                out_tid: chain.o_tid,
                pair: &pair_h,
                scores: &scores_h,
                out: &o_h,
            };
            let emitted = super::lower_ktir_to_superdsc::expert_combine_pair_legs(
                &legs,
                j,
                sym_id_base,
                Some(layout),
            );
            out.extend(emitted);
            lowered += 1;
            continue;
        }
        // The terminal `arith.mulf` IS the fused program, and its inputs are the chain's gate and up
        // rather than its own operands (which are `silu(gate)` and up). Everything else keeps the
        // 1:1 map exactly as before.
        let terminal = silus.iter().copied().find(|c| op.result == Some(c.mul));
        // A gelu chain's terminal `arith.mulf`, same rule: the fused program's inputs are the
        // recognised gate and up, not the mulf's own operands (`gelu(g)` is destroyed by the
        // fusion).
        let gated = gelus.iter().find(|c| op.result == Some(c.mul));
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
            .then(|| reads.splat_scale_of(op))
            .flatten();
        // ⭐ RUNG 3: a mulf by a splat of a LOADED `[1,1]` PARAMETER is the SAME tiebreak with the
        // multiplier bound at launch rather than baked. `to_ktir`'s `expand_splat_of_scalar_argument`
        // emits the `[1,1]` view/tile/load chain for a scalar argument and re-points its splat at the
        // load; this reads WHICH PARAMETER that load reads, so the emission can name its binding.
        // Checked AFTER the constant arm (a constant-backed splat still resolves by value) and
        // BEFORE the 1:1 map, which would otherwise route the op to `Elementwise(Mul)` — the
        // mb-broadcast defect shape, a `[1,1]` region against a full tile.
        let bound_scale = if op.op_type == OpKind::ArithMulf && scalar.is_none() {
            reads.bound_scale_of(k, op)?
        } else {
            None
        };
        // An rmsnorm chain's TERMINAL gain multiply becomes the one fused program. Checked before the
        // splat arm and before the 1:1 map, both of which would otherwise claim it.
        let rms = rmsnorms.iter().find(|c| op.result == Some(c.out));
        let Some(program) = terminal
            .map(|_| Lowering::Silu)
            // The claimed gelu's fused kind — the SAME `Program` a per-op producer states for a
            // gated act, so `emit_one` dispatches to the one `gated_act` body both doors share
            // and they cannot drift about what a gelu emits.
            .or(gated.map(|_| Lowering::Node(Program::ExpertGatedAct(GatedAct::Gelu))))
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
            // The gated act's two inputs are the chain's gate and up, which the recogniser
            // proved — the terminal mulf's own operands are `gelu(g)` and up, and the first is
            // DESTROYED by the fusion.
            Lowering::Node(Program::ExpertGatedAct(_)) => 2,
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
            // The gelu chain's two tensors, same rule as the silu's: the terminal mulf's own
            // first operand is `gelu(g)` — destroyed by the fusion — so the recognised gate and
            // up are what the emission takes.
            _ if gated.is_some() => {
                let c = gated.expect("just matched");
                vec![c.gate, c.up]
            }
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
                    next_tid: &mut next_tid,
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
                c_cover: (0, dims[1]),
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
        let out_tile_op = f
            .operations
            .iter()
            .find(|o| o.op_type == OpKind::KtdpConstructAccessTile && o.result == out_tile);
        let out_view = out_tile_op.and_then(|t| t.operands.first().copied());
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
        // ⭐⭐⭐⭐⭐ THE WINDOW IS THE STORE'S OWN TILE, NOT THE PARAMETER'S FIRST — the store
        // arm of the law [`region_for_operand`]'s load arm has enforced since the unblocked
        // SwiGLU measurement (MEASURED there: every trip's matmul read the weight through
        // trip 0's tile, corr 0.999996 against the doubled-window product). `regions()`
        // keeps the FIRST access tile over each parameter's view, so a function that stores
        // SEVERAL windows of one buffer — the MoE gate/up act rows, pair j's `[1, I]` window
        // at column `j·I` of the `[1, K·I]` act buffer — would otherwise write EVERY pair's
        // activation through pair 0's tile: right shape, right dtype, every pair landed on
        // the first pair's columns. The store in hand is THIS op's own (found by result
        // identity above), so its tile is the window this emission must write.
        if let Some(tile) = out_tile_op
            && let Some((rs, cs, r_len, c_len)) = tile_window(f, tile, "this store writes through")?
        {
            o.r_start = rs;
            o.c_start = cs;
            o.r_len = r_len;
            o.c_len = c_len;
        }
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
                super::lower_ktir_to_superdsc::MatmulProof {
                    b,
                    // This door's operands are WINDOWS of the caller's parameters — the fact the
                    // spurious-pad drop in `matmul_oriented` discriminates on.
                    origin: super::lower_ktir_to_superdsc::OperandOrigin::Windowed,
                    // ⭐ NO HEAD-MAJOR o_proj HANDOFF HERE, BY CONSTRUCTION. That fact is minted
                    // from the BUNDLE's attention-output row count (mq == 1 over a multi-slab head
                    // dim) and lives on the spyre door's `BundleAttnParams`; this whole-function
                    // door lowers an UNROLLED graph whose per-layer o matmuls key on per-layer
                    // weight tids the bundle fact's one canonical tid cannot name. The unrolled
                    // bake therefore keeps the pre-restructure emission — see the codegen's
                    // headmajor mutation sites.
                    headmajor: None,
                },
            )?
        }
        Program::Elementwise(e) => {
            super::lower_ktir_to_superdsc::elementwise(name, e, per_op, bcast, sym_id_base, layout)?
        }
        Program::Transpose => {
            super::lower_ktir_to_superdsc::transpose(name, per_op, sym_id_base, layout)?
        }
        // THE GELU LONGHAND'S FUSED KIND — the SAME body the per-`Program` door reaches, so
        // the two doors cannot drift about what a gated act emits. Reached only through
        // [`program_gelu_mul_chains`], which PROVED the longhand; `per_op` is
        // `[gate, up, out]`, the parameter order `gated_act`'s own `split_out(.., 2)` reads.
        Program::ExpertGatedAct(act) => {
            super::lower_ktir_to_superdsc::gated_act(name, act, per_op, sym_id_base, layout)?
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
        c_cover: (0, head),
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
    ///
    /// ⭐ DRIVEN THROUGH THE PRIVATE READER, NOT THE PUB WRAPPER: the wrapper fronts the gelu
    /// reader, which CLAIMS every wrong-numerator chain as a gelu candidate and refuses it on the
    /// cubic verification (the companion test below pins that composed order). This discriminator
    /// is the silu reader's own backstop — reachable whenever a negf reaches it unclaimed — and is
    /// pinned here directly so the two readers' division of the negfs cannot drift silently.
    #[test]
    fn a_divide_whose_numerator_is_not_the_negated_value_is_refused() {
        let e = silu_mul_chains(&chain(1.0, Num::Up, true), &[])
            .expect_err("dividing something other than the gate by 1+e^-g is not a silu");
        assert!(
            e.message.contains("NUMERATOR"),
            "the refusal must name the numerator as the thing that failed; got: {}",
            e.message
        );
    }

    /// ⭐ THE COMPOSED ORDER: the same wrong-numerator chain through the PUB wrapper is refused
    /// by the GELU reader instead — the gelu reader runs first, the numerator is not the negated
    /// value, so the chain is a gelu candidate, and the e-argument (here a bare function
    /// argument, not a cubic) fails the tanh form's verification. A chain that is neither silu
    /// nor gelu is a refusal EITHER WAY; this pins WHICH reader owns it at the pub door, so a
    /// reorder of the pre-reads fails here rather than silently rerouting the message.
    #[test]
    fn a_wrong_numerator_chain_is_refused_by_the_gelu_reader_at_the_pub_door() {
        let e = program_silu_mul_chains(&chain(1.0, Num::Up, true))
            .expect_err("a chain that is neither silu nor gelu must not lower");
        assert!(
            e.message.contains("gelu longhand's e-argument"),
            "the pub wrapper fronts the gelu reader, so a wrong-numerator chain is refused \
             as a gelu candidate; got: {}",
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

#[cfg(test)]
mod program_reads_tests {
    use super::*;
    use ktir_core::arena::Arena;
    use ktir_core::ir::Operation;
    use ktir_core::irtype::IrType;

    /// A constant → splat → mulf chain over one argument, with an optional second reader of the
    /// splat so `readers_of` has both shapes to answer for. SSA ids are the ops' results.
    fn probe(second_reader: bool) -> IRFunction<'static> {
        let a: &'static Arena = Arena::global();
        let (arg, c0, sp1, m2, m3) = (Ssa(0), Ssa(1), Ssa(2), Ssa(3), Ssa(4));
        let mut ops = vec![
            Operation::new(a, Some(c0), OpKind::ArithConstant, &[]).with_attr(
                a,
                AttrKey::Value,
                Attr::Float(0.5),
            ),
            Operation::new(a, Some(sp1), OpKind::TensorSplat, &[c0]),
            Operation::new(a, Some(m2), OpKind::ArithMulf, &[sp1, arg]),
        ];
        if second_reader {
            ops.push(Operation::new(a, Some(m3), OpKind::ArithMulf, &[sp1, m2]));
        }
        IRFunction {
            name: "reads_probe",
            arguments: a.args(vec![(arg, IrType::Index)]),
            operations: a.ops(ops),
            grid: (1, 1, 1),
            return_type: None,
        }
    }

    /// ⭐ CONTROL — def, readers, position, splat and the scale tiebreak all round-trip through
    /// ONE index. Without this the corner laws below could all be a broken index rather than
    /// working guards.
    #[test]
    fn defs_readers_positions_and_splats_round_trip() {
        let f = probe(false);
        let reads = program_reads(&f);
        let (arg, c0, sp1, m2) = (Ssa(0), Ssa(1), Ssa(2), Ssa(3));
        // def_of: the splat's definer is the splat; an argument's is none.
        assert_eq!(
            reads.def_of(sp1).map(|o| o.op_type),
            Some(OpKind::TensorSplat)
        );
        assert_eq!(
            reads.def_of(arg),
            None,
            "an argument is defined by nothing in the body"
        );
        // readers_of / sole_reader: exactly one op reads the splat.
        let rs = reads.readers_of(sp1);
        assert_eq!(rs.len(), 1);
        assert_eq!(reads.op_at(rs[0]).op_type, OpKind::ArithMulf);
        assert_eq!(reads.sole_reader(sp1).map(|o| o.result), Some(Some(m2)));
        // Positions: the reader sits AFTER the definer — the whole point of positions.
        let (d, r) = (
            reads.pos_of_result(sp1).unwrap(),
            reads.pos_of_result(m2).unwrap(),
        );
        assert!(d < r, "the definer precedes its reader in program order");
        // splat_value and the tiebreak, off the same index.
        assert_eq!(reads.splat_value(sp1), Some(0.5));
        assert_eq!(
            reads.splat_scale_of(reads.op_at(r)),
            Some((0.5, arg)),
            "one splatted operand, the other the tensor: the ScalarMul tiebreak"
        );
        assert_eq!(
            reads.splat_value(c0),
            None,
            "the constant itself is not a splat"
        );
    }

    /// A second reader retires `sole_reader` without touching `readers_of`'s order — the
    /// exclusivity a fused chain's links are proven by is a QUESTION, not a constant.
    #[test]
    fn a_second_reader_retires_sole_reader_but_keeps_the_list() {
        let f = probe(true);
        let reads = program_reads(&f);
        let rs = reads.readers_of(Ssa(2));
        assert_eq!(rs.len(), 2, "both mulfs read the splat");
        assert!(rs[0] < rs[1], "readers are listed in program order");
        assert_eq!(reads.sole_reader(Ssa(2)), None);
    }

    /// ⛔ AN OP NAMING THE SAME OPERAND ON BOTH SIDES IS ONE READER. The silu reader's
    /// more-than-one-reader refusal counts READERS, not operand slots — `mulf x, x` is ONE op
    /// reading `x`, not two — and the index's count must agree with the `contains`-based filter
    /// it replaced, or that refusal's arithmetic changes under it.
    #[test]
    fn an_op_naming_both_sides_is_one_reader() {
        let a: &'static Arena = Arena::global();
        let arg = Ssa(0);
        let f = IRFunction {
            name: "square_probe",
            arguments: a.args(vec![(arg, IrType::Index)]),
            operations: a.ops(vec![Operation::new(
                a,
                Some(Ssa(1)),
                OpKind::ArithMulf,
                &[arg, arg],
            )]),
            grid: (1, 1, 1),
            return_type: None,
        };
        let reads = program_reads(&f);
        assert_eq!(reads.readers_of(arg).len(), 1, "`mulf x, x` reads `x` once");
        assert_eq!(
            reads.sole_reader(arg).map(|o| o.op_type),
            Some(OpKind::ArithMulf)
        );
    }

    /// ⛔ THE DEEP LAW: a value a REGION defines resolves, where the top-level `find` closures
    /// this index replaced were silently blind. That blindness is the silu reader's own stated
    /// fear — a reader inside an `scf.for` body invisible to a top-level walk — and the index
    /// is the one walk that cannot have it. Pinned on the smallest program that states it: a
    /// loop body squaring the argument, a top-level op reading the square.
    #[test]
    fn a_region_defined_value_resolves() {
        let a: &'static Arena = Arena::global();
        let (arg, r4, r5, r6) = (Ssa(0), Ssa(1), Ssa(2), Ssa(3));
        let body = vec![Operation::new(a, Some(r5), OpKind::ArithMulf, &[arg, arg])];
        let forop = Operation {
            regions: a.regions(vec![a.ops(body)]),
            ..Operation::new(a, Some(r4), OpKind::ScfFor, &[])
        };
        let f = IRFunction {
            name: "region_probe",
            arguments: a.args(vec![(arg, IrType::Index)]),
            operations: a.ops(vec![
                forop,
                Operation::new(a, Some(r6), OpKind::ArithAddf, &[r5, r5]),
            ]),
            grid: (1, 1, 1),
            return_type: None,
        };
        let reads = program_reads(&f);
        // Deep order: the `scf.for` at 0, its body's mulf at 1, the top-level addf at 2.
        assert_eq!(
            reads.def_of(r5).map(|o| o.op_type),
            Some(OpKind::ArithMulf),
            "the region's mulf is r5's definer; a top-level-only find could not see it"
        );
        assert_eq!(reads.pos_of_result(r4), Some(0));
        assert_eq!(
            reads.pos_of_result(r5),
            Some(1),
            "the body op sits after its loop's own position"
        );
        assert_eq!(reads.sole_reader(r5).map(|o| o.result), Some(Some(r6)));
        assert_eq!(reads.readers_of(arg).len(), 1);
        assert_eq!(
            reads.readers_of(arg)[0],
            1,
            "the body's `mulf arg, arg` is ONE reader"
        );
    }

    /// ⛔ `pos_of_op` MATCHES BY ADDRESS, and a COPY is a different op. `Operation` is `Copy`,
    /// so nothing stops a caller from holding a copy of an indexed op — and two identical
    /// resultless stores are two stores, which only the address distinguishes. The copy must
    /// answer `None` rather than the position of its equal.
    #[test]
    fn a_copied_op_is_not_the_indexed_op() {
        let f = probe(false);
        let reads = program_reads(&f);
        let indexed = reads.ops()[0];
        assert_eq!(reads.pos_of_op(indexed), Some(0));
        let copy = *indexed;
        assert_eq!(
            reads.pos_of_op(&copy),
            None,
            "a copy compares equal but is not the op at any position"
        );
    }

    /// `through_plumbing` reaches the value a rank chain CARRIES — the question a fused
    /// recogniser asks of an operand — and agrees with the chains reader's `Bcast::src` from
    /// the other entry point. The chain is the decoder's own (`[64] → [64, 1] → [64] →
    /// broadcast`), the shape `broadcast_chain_tests` pins from `decoder_block.py`.
    #[test]
    fn through_plumbing_reaches_the_value_the_rank_chain_carries() {
        let a: &'static Arena = Arena::global();
        let (src, exp, col, emp, bc, m) = (Ssa(0), Ssa(1), Ssa(2), Ssa(3), Ssa(4), Ssa(5));
        let f = IRFunction {
            name: "plumbing_probe",
            arguments: a.args(vec![(src, IrType::Index)]),
            operations: a.ops(vec![
                Operation::new(a, Some(exp), OpKind::TensorExpandShape, &[src]).with_attr(
                    a,
                    AttrKey::TargetShape,
                    Attr::IntList(&[64, 1]),
                ),
                Operation::new(a, Some(col), OpKind::TensorCollapseShape, &[exp]),
                Operation::new(a, Some(emp), OpKind::TensorEmpty, &[]),
                Operation::new(a, Some(bc), OpKind::LinalgBroadcast, &[col, emp]),
                Operation::new(a, Some(m), OpKind::ArithMulf, &[bc, src]),
            ]),
            grid: (1, 1, 1),
            return_type: None,
        };
        let reads = program_reads(&f);
        assert_eq!(
            reads.through_plumbing(bc),
            src,
            "from the broadcast's result: the same value `Bcast::src` names"
        );
        assert_eq!(
            reads.through_plumbing(col),
            src,
            "the expand is plumbing too — the walk does not stop at the collapse"
        );
        assert_eq!(
            reads.through_plumbing(m),
            m,
            "a computed value carries itself; the walk stops where the plumbing does"
        );
        assert_eq!(
            reads.through_plumbing(src),
            src,
            "an argument carries itself"
        );
    }

    /// ⛔ THE WALK IS BOUNDED AT THREE STEPS — the same bound the chains reader and the
    /// rmsnorm gain walk state — so a fourth plumbing op is handed back unresolved rather
    /// than guessed past. Pins the bound the method's doc states.
    #[test]
    fn through_plumbing_stops_at_the_same_three_step_bound() {
        let a: &'static Arena = Arena::global();
        // src <- e1 <- e2 <- e3 <- e4: four collapses deep.
        let (src, e1, e2, e3, e4) = (Ssa(0), Ssa(1), Ssa(2), Ssa(3), Ssa(4));
        let f = IRFunction {
            name: "deep_plumbing_probe",
            arguments: a.args(vec![(src, IrType::Index)]),
            operations: a.ops(vec![
                Operation::new(a, Some(e1), OpKind::TensorCollapseShape, &[src]),
                Operation::new(a, Some(e2), OpKind::TensorCollapseShape, &[e1]),
                Operation::new(a, Some(e3), OpKind::TensorCollapseShape, &[e2]),
                Operation::new(a, Some(e4), OpKind::TensorCollapseShape, &[e3]),
            ]),
            grid: (1, 1, 1),
            return_type: None,
        };
        let reads = program_reads(&f);
        assert_eq!(reads.through_plumbing(e4), e1, "three steps up, not four");
    }
}

/// THE OWN-TILE WINDOW READER'S FAIL-CLOSED HALF — each refusal has a control that
/// isolates it, the door suite's own law.
///
/// [`tile_window`] is the ONE reading of a `ktdp.construct_access_tile`'s window, shared by
/// [`region_for_operand`]'s load arm and the walk's store arm. The accepting path is pinned
/// end-to-end by the MoE door tests (moe.py's per-pair act stores, each with its own tile on
/// the shared act buffer); what only a unit test can isolate is the reader's own contract:
/// two constant-corner tiles over one view state two windows, a computed corner is REFUSED
/// BY NAME rather than silently falling back to the parameter's first tile, and a tile that
/// is not the corner form states no window at all so the caller keeps what it holds.
#[cfg(test)]
mod tile_window_tests {
    use super::*;
    use ktir_core::arena::Arena;
    use ktir_core::ir::Operation;
    use ktir_core::irtype::IrType;

    /// moe.py's own spelling of the shared act buffer: TWO tiles over ONE view — pair 0's at
    /// column 0, pair 1's at column `I` — each stating its own `[1, 64]` shape (the attribute
    /// `shape_2d` reads). `second_col` is `Some(v)` for the constant corner every shipped
    /// config states, or `None` to wire pair 1's corner to an ARGUMENT no constant defines: a
    /// loop-carried or computed corner, the one thing the reader must refuse rather than guess.
    fn pairs_fn(second_col: Option<i64>) -> IRFunction<'static> {
        let a: &'static Arena = Arena::global();
        let (view, rc, c0, c1, t0, t1) = (Ssa(8), Ssa(0), Ssa(1), Ssa(2), Ssa(3), Ssa(4));
        let mut ops = vec![
            Operation::new(a, Some(rc), OpKind::ArithConstant, &[]).with_attr(
                a,
                AttrKey::Value,
                Attr::Int(0),
            ),
            Operation::new(a, Some(c0), OpKind::ArithConstant, &[]).with_attr(
                a,
                AttrKey::Value,
                Attr::Int(0),
            ),
            Operation::new(
                a,
                Some(t0),
                OpKind::KtdpConstructAccessTile,
                &[view, rc, c0],
            )
            .with_attr(a, AttrKey::Shape, Attr::IntList(&[1i64, 64])),
        ];
        let second = match second_col {
            Some(v) => {
                ops.push(
                    Operation::new(a, Some(c1), OpKind::ArithConstant, &[]).with_attr(
                        a,
                        AttrKey::Value,
                        Attr::Int(v),
                    ),
                );
                c1
            }
            None => Ssa(9),
        };
        ops.push(
            Operation::new(
                a,
                Some(t1),
                OpKind::KtdpConstructAccessTile,
                &[view, rc, second],
            )
            .with_attr(a, AttrKey::Shape, Attr::IntList(&[1i64, 64])),
        );
        IRFunction {
            name: "tile_probe",
            arguments: a.args(vec![(view, IrType::Index), (Ssa(9), IrType::Index)]),
            operations: a.ops(ops),
            grid: (1, 1, 1),
            return_type: None,
        }
    }

    /// The tile defining `res` — the probe's ops are addressed by result, the way the walk
    /// itself finds a tile.
    fn tile_of(f: &IRFunction<'static>, res: Ssa) -> &'static Operation<'static> {
        f.operations
            .iter()
            .find(|o| o.result == Some(res))
            .expect("the probe states the tile")
    }

    /// ⭐ CONTROL — the accepting path, stated as the defect's own shape: TWO tiles over ONE
    /// view must state TWO windows. The parameter-rooted `regions()` join can only offer the
    /// FIRST tile's window, which is exactly the collapse the store arm used to inherit —
    /// pair 1's column must not come from pair 0's tile.
    #[test]
    fn two_tiles_over_one_view_state_two_windows() {
        let f = pairs_fn(Some(64));
        assert_eq!(
            tile_window(&f, tile_of(&f, Ssa(3)), "pair 0's tile").expect("constant corners"),
            Some((0, 0, 1, 64)),
            "pair 0's window is its own tile's"
        );
        assert_eq!(
            tile_window(&f, tile_of(&f, Ssa(4)), "pair 1's tile").expect("constant corners"),
            Some((0, 64, 1, 64)),
            "pair 1's window is ITS OWN tile's — pair 0's column 0 must not leak into it"
        );
    }

    /// ⛔ A COMPUTED CORNER IS REFUSED BY NAME — the fail-closed half. Before the store arm
    /// read its own tile, a computed corner meant a silent write through the parameter's
    /// FIRST tile (right shape, wrong window); now it is a build error naming the corner, the
    /// same discipline `regions()`'s in-loop guard holds. The refusal is per-tile: pair 0's
    /// constant tile still states its window while pair 1's is refused.
    #[test]
    fn a_computed_corner_is_refused_by_name_and_only_on_its_tile() {
        let f = pairs_fn(None);
        let e = tile_window(&f, tile_of(&f, Ssa(4)), "pair 1's tile")
            .expect_err("a computed corner names a window no descriptor can carry")
            .message;
        assert!(
            e.contains("corner is not a constant"),
            "the refusal must name the corner, not fall back to window 0: {e}"
        );
        assert_eq!(
            tile_window(&f, tile_of(&f, Ssa(3)), "pair 0's tile").expect("constant corners"),
            Some((0, 0, 1, 64)),
            "pair 0's constant tile still states its window"
        );
    }

    /// A tile that is not the corner form — no corner operands to read — states NO window,
    /// and the caller keeps the region it already holds: the reader must not invent one.
    #[test]
    fn a_tile_without_corners_states_no_window() {
        let a: &'static Arena = Arena::global();
        let (view, tile) = (Ssa(8), Ssa(1));
        let f = IRFunction {
            name: "bare_tile_probe",
            arguments: a.args(vec![(view, IrType::Index)]),
            operations: a.ops(vec![Operation::new(
                a,
                Some(tile),
                OpKind::KtdpConstructAccessTile,
                &[view],
            )]),
            grid: (1, 1, 1),
            return_type: None,
        };
        assert_eq!(
            tile_window(&f, tile_of(&f, Ssa(1)), "the bare tile")
                .expect("a tile that is not the corner form is not an error"),
            None,
            "no corners to read, so no window to state"
        );
    }
}
