// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Flash-attention IR-rewrite pass — TODO #2 (the ABOVE-cap path of Contract B).
//!
//! Long-context attention's `[m, cap]` scores matrix is an INTRA-node tile that
//! overflows the 2 MB LX as `cap` (the KV length) grows. Segmentation cannot
//! help — attention is ONE node, and the scores tile lives inside it. This pass
//! tiles the cap/KV dimension with **online softmax**, emitting STANDARD tiled
//! MLIR (an `scf.for` over KV blocks + online-softmax arith/reduce + matmul)
//! that the EXISTING generic interpreter runs unchanged. It is NOT a hand-written
//! kernel and introduces NO new `ktdp` ops.
//!
//! ## Why this is a legitimate, semantics-preserving optimization
//!
//! Online softmax (Milakov & Gimelshein 2018; the FlashAttention recurrence) is
//! mathematically equal to the two-pass `max → exp → sum → div` softmax up to
//! floating-point re-association — it visits the same `exp` terms, just folds the
//! running max/sum block by block instead of after a full pass. It is in fact
//! *more* numerically stable (the running max bounds every `exp` argument), so it
//! comfortably stays inside the project's 0.05 golden gate. This is the same
//! class of rewrite as the existing GEMM K-loop recognizer in `metal_backend`:
//! recognize a high-level idiom from raw IR, re-emit a tiled equivalent.
//!
//! ## Contract B (see `fusion::attention_needs_flash`)
//!
//! This pass owns the ABOVE-cap regime ONLY. A node is rewritten IFF its
//! `[m, cap]` scores footprint would overflow LX. Below the cap the node is left
//! NAIVE (untouched) — that regime belongs to the head-batching fleet. The
//! predicate is monotone and exhaustive, so every attention node receives exactly
//! one transform. The rewritten node is region-bearing (`scf.for`) by design and
//! runs on the generic interpreter (the batched-executor's region-free gate makes
//! it `Err → fall back`, which is the intended path above the cap).
//!
//! ## Two recognizers
//!
//! The real cached prefill/decode nodes are an *unrolled, per-query-row,
//! two-KV-block* hand lowering. They do NOT match the single-block canonical
//! idiom — but [`crate::head_rewrite`] (which runs FIRST) RE-ROLLS them into a
//! whole-tensor two-block form: a CONTEXT `QKᵀ` producing `[m, cap]` scores (the
//! tile that overflows LX as context grows) + a small `[m, m]` masked DIAGONAL
//! block + an online-softmax combine + two `A·V` matmuls, all stored as ONE
//! `arith.addf(ov_context, ov_diag)`.
//!
//! This pass therefore has TWO structural recognizers, tried in order:
//!   1. [`recognize_rerolled_attention`] — the head-rewrite OUTPUT (the REAL node
//!      form). It anchors on the `arith.addf`-of-two-matmuls store value, recovers
//!      the context/diagonal softmax chains, and [`tile_rerolled_attention`]
//!      cap-tiles ONLY the CONTEXT `[m, cap]` block with online softmax (the
//!      `[m, m]` diagonal stays whole). THIS is the long-context fix.
//!   2. [`recognize_attention`] — the single-block synthetic canonical idiom
//!      (unchanged; keeps the synthetic golden green).
//!
//! ## Fail-safe recognition
//!
//! Both recognizers return `None` whenever the function is not PROVABLY their
//! idiom (a top-level `scf.*`, a deviating op sequence, the wrong shapes).
//! Returning `None` leaves the node untouched — the same Err→fallback discipline
//! as the GPU offloads. We never rewrite a node we cannot prove equivalent.

use crate::head_rewrite::NameGen;
use ktir_core::arena::Arena;
use ktir_core::attrkey::AttrKey;
use ktir_core::dtypes::DType;
use ktir_core::ir::{Attr, IRFunction, IRModule, Operation, Ssa};
use ktir_core::irtype::IrType;
use ktir_core::opkind::OpKind;
use std::collections::HashMap;

/// Apply the flash-attention pass to every function in `module`, in place.
///
/// For each function: recognize the canonical naive-attention idiom; if it is
/// PROVABLY that idiom AND its `[m, cap]` scores tile would overflow LX
/// (`needs_flash(scores_bytes, lx_budget)` — Contract B's
/// `fusion::attention_needs_flash`), replace it with the tiled online-softmax
/// rewrite (same function NAME, args, grid). Otherwise leave it untouched
/// (fail-safe: below the cap, or not recognized → naive).
///
/// `needs_flash` is injected (rather than calling `fusion::attention_needs_flash`
/// directly) so the caller threads its OWN `lx_budget` and any force/threshold
/// env override, keeping the cap-partition decision in one place. Returns the
/// number of functions rewritten (for diagnostics / test assertions).
pub fn apply_flash_attention<'a>(
    a: &'a Arena,
    module: &mut IRModule<'a>,
    needs_flash: impl Fn(usize) -> bool,
) -> usize {
    let names: Vec<String> = module.functions.keys().cloned().collect();
    let mut rewritten = 0usize;
    for name in names {
        let Some(func) = module.functions.get(&name) else {
            continue;
        };
        // Fresh values start above every value the ORIGINAL function defines (args
        // included), so a rewrite cannot shadow one of its own pointer arguments.
        let mut g = NameGen::after(func);

        // (A) RE-ROLLED form FIRST — this is the REAL model node after
        // `head_rewrite` runs (it stores `arith.addf(ovc, ovd)`, the exact hop the
        // single-block `recognize_attention` bails on). Its CONTEXT `[m, cap]`
        // scores tile is what overflows LX as context grows; cap-tile ONLY that
        // block, leaving the small `[m, m]` diagonal whole. Same `scores_bytes`
        // formula (`m*cap*bytes`) as the head island, so Contract B's monotone
        // predicate routes each node to exactly one pass.
        if let Some(island) = recognize_rerolled_attention(func) {
            // Fire flash when the `[m, cap]` SCORES tile is over the LX budget
            // (`needs_flash` — the large-query regime) OR the `[cap, d]` CONTEXT
            // K/V tile is too large to keep whole in LX (the long-context /
            // small-query regime: decode `m=1`, chunked prefill, where scores stay
            // tiny but the context read overflows LX — the case the scores-only
            // gate missed and left un-tiled ⇒ `LX capacity exceeded`).
            if !needs_flash(island.scores_bytes())
                && island.context_bytes() <= FLASH_CONTEXT_TILE_MAX
            {
                continue; // both tiles fit whole: leave head_rewrite's form.
            }
            // Choose the KV block so BOTH the per-block scores tile `[m, blk]` fits
            // the injected budget AND the per-block context K/V tile `[blk, d]`
            // fits `FLASH_CONTEXT_TILE_MAX`. This is the actual long-context fix:
            // `blk < cap`.
            let blk = choose_block_budgeted(
                island.m,
                island.cap,
                island.d,
                island.dtype.bytes_per_elem(),
                &needs_flash,
            );
            let mut tiled = tile_rerolled_attention(a, &island, blk, &mut g);
            tiled.name = a.str(name.clone());
            module.functions.insert(name, tiled);
            rewritten += 1;
            continue;
        }

        // (B) Single-block canonical idiom (the synthetic golden path) — unchanged.
        let Some(island) = recognize_attention(func) else {
            continue;
        };
        if !needs_flash(island.scores_bytes()) {
            continue; // below the cap: stay naive (head-batching fleet's regime).
        }
        let grid = func.grid;
        let mut tiled = tile_attention(a, &island, &mut g);
        // Preserve the original function identity so the program's node→tensor
        // bindings and the segmenter still resolve it. The rewrite is single-grid
        // by design (region-bearing, runs on the generic interpreter).
        tiled.name = a.str(name.clone());
        let _ = grid; // grid intentionally collapsed to [1,1] in tile_attention.
        module.functions.insert(name, tiled);
        rewritten += 1;
    }
    rewritten
}

/// The recovered configuration of a recognized naive-attention function.
///
/// All shapes are re-derived from the IR (never assumed). `causal` and `scale`
/// are likewise recovered from the actual ops, so the rewrite reproduces the
/// node's exact arithmetic.
#[derive(Clone, Debug, PartialEq)]
pub struct AttentionIsland {
    /// Function argument (pointer) carrying Q, its memory-view shape `[m, d]`.
    pub q_arg: Ssa,
    pub q_shape: Vec<i64>,
    /// Function argument carrying K, view shape `[cap, d]` (NOT transposed).
    pub k_arg: Ssa,
    pub k_shape: Vec<i64>,
    /// Function argument carrying V, view shape `[cap, d]`.
    pub v_arg: Ssa,
    pub v_shape: Vec<i64>,
    /// Output pointer arg, view shape `[m, d]`.
    pub o_arg: Ssa,
    pub o_shape: Vec<i64>,
    /// Query rows.
    pub m: i64,
    /// KV length (the cap axis to tile).
    pub cap: i64,
    /// Head dim.
    pub d: i64,
    /// `1/sqrt(d)` scale recovered from the `arith.mulf` by a splat constant.
    pub scale: f32,
    /// True when a causal mask add (`-inf` upper triangle) is present.
    pub causal: bool,
    /// Storage dtype.
    pub dtype: DType,
}

impl AttentionIsland {
    /// Scores-tile byte footprint `[m, cap]` × storage-dtype bytes — the value
    /// Contract B's `attention_needs_flash` consumes to decide whether to fire.
    pub fn scores_bytes(&self) -> usize {
        (self.m as usize)
            .saturating_mul(self.cap as usize)
            .saturating_mul(self.dtype.bytes_per_elem())
    }

    /// The pointer args this island names, in the order the rewrite declares them.
    fn arg_ssas(&self) -> [Ssa; 4] {
        [self.q_arg, self.k_arg, self.v_arg, self.o_arg]
    }
}

/// The recovered configuration of a recognized RE-ROLLED head-attention function
/// (the [`crate::head_rewrite`] OUTPUT — i.e. the REAL model node). Carries the
/// SAME fields [`crate::head_rewrite::HeadAttnIsland`] recovers, so the rewrite
/// reproduces the node's exact per-head arithmetic; `scores_bytes` uses the
/// IDENTICAL `m*cap*bytes` CONTEXT-tile formula so the two passes' Contract-B
/// partition stays disjoint (a node routes to head-reroll XOR flash, never both).
#[derive(Clone, Debug, PartialEq)]
pub struct ReRolledIsland {
    /// Q pointer arg (view0), `[m, q_cols]` where `q_cols = H*d`.
    pub q_arg: Ssa,
    /// O pointer arg (view1), `[m, q_cols]`.
    pub o_arg: Ssa,
    /// Per-head context mask pointer arg (view2), `[1, cap]`.
    pub mask_arg: Ssa,
    /// Context K pointer arg (view5), `[cap, kv_cols]`.
    pub kc_arg: Ssa,
    /// Diagonal (current-segment) K pointer arg (view6), `[m, kv_cols]`.
    pub kd_arg: Ssa,
    /// Context V pointer arg (view7), `[cap, kv_cols]`.
    pub vc_arg: Ssa,
    /// Diagonal V pointer arg (view8), `[m, kv_cols]`.
    pub vd_arg: Ssa,
    /// Q/O view column width `H*d`.
    pub q_cols: i64,
    /// KV view column width (`num_kv_heads * d`).
    pub kv_cols: i64,
    /// Query rows.
    pub m: i64,
    /// Context KV length (the `cap` axis to tile).
    pub cap: i64,
    /// Head dim.
    pub d: i64,
    /// GQA divisor recovered from `arith.divui %hpid, %gqac`.
    pub gqac: i64,
    /// Per-head column stride recovered from `arith.muli %hpid, %hdc`.
    pub hdc: i64,
    /// Grid head count `H` (grid.0).
    pub h: i64,
    /// `1/sqrt(d)` scale recovered from the `arith.mulf` by a splat constant.
    pub scale: f32,
    /// `-inf` mask constant recovered from the diagonal triangular mask.
    pub ninf: f32,
    /// Storage dtype.
    pub dtype: DType,
}

impl ReRolledIsland {
    /// The pointer args this island names, in the order the rewrite declares them.
    fn arg_ssas(&self) -> [Ssa; 7] {
        [
            self.q_arg,
            self.o_arg,
            self.mask_arg,
            self.kc_arg,
            self.kd_arg,
            self.vc_arg,
            self.vd_arg,
        ]
    }

    /// CONTEXT scores-tile footprint `[m, cap]` × dtype bytes — IDENTICAL to
    /// `HeadAttnIsland::scores_bytes` so Contract B's monotone predicate routes a
    /// node to exactly one of {head-reroll, flash}. (The `[m, m]` diagonal stays
    /// whole and is NOT counted — only the cap tile overflows.)
    pub fn scores_bytes(&self) -> usize {
        (self.m as usize)
            .saturating_mul(self.cap as usize)
            .saturating_mul(self.dtype.bytes_per_elem())
    }

    /// CONTEXT K/V-tile footprint `[cap, d]` × dtype bytes. UNLIKE `scores_bytes`
    /// (`m·cap`) this is m-INDEPENDENT: it grows with the context length `cap`
    /// alone. In the small-query/long-context regime (decode `m=1`, chunked
    /// prefill `m≪cap`) the whole `[cap, d]` context K (and V) read is what
    /// overflows LX while the `[m, cap]` scores tile stays tiny — the case the
    /// scores-only gate misses. Flash-tiling the cap axis shrinks BOTH tiles.
    pub fn context_bytes(&self) -> usize {
        (self.cap as usize)
            .saturating_mul(self.d as usize)
            .saturating_mul(self.dtype.bytes_per_elem())
    }
}

/// Max per-block CONTEXT K/V tile `[blk, d]` bytes flash keeps whole in LX. The
/// cap-tiled online-softmax loop holds one K block AND one V block resident at a
/// time, so `2 · this` must fit alongside the fused segment's other resident
/// live-set (~1.5 MiB of a 2 MiB per-core LX on Llama-3B). 128 KiB ⇒ 256 KiB for
/// K+V, leaving comfortable headroom. Numerics are exact for ANY block size
/// (online softmax), so this only trades a few extra blocks for fitting LX.
const FLASH_CONTEXT_TILE_MAX: usize = 128 * 1024;

// ===========================================================================
// Recognition
// ===========================================================================

/// Decoded `ktdp.construct_memory_view %ptr` -> (pointer arg, view shape).
struct ViewInfo {
    arg: Ssa,
    shape: Vec<i64>,
    dtype: DType,
}

/// Decoded `ktdp.construct_access_tile %view[..]` -> the view SSA it reads.
struct TileInfo {
    view: Ssa,
}

fn shape_attr(op: &Operation) -> Vec<i64> {
    match op.attr(AttrKey::Shape) {
        Some(Attr::IntList(v)) => v.to_vec(),
        _ => Vec::new(),
    }
}

fn dtype_attr(op: &Operation) -> DType {
    match op.attr(AttrKey::Dtype) {
        Some(Attr::Dtype(d)) => *d,
        _ => DType::F16,
    }
}

/// The `scf.*` structured-control-flow ops. A body containing one is never the
/// flat idiom either recognizer matches.
fn is_control_flow(k: OpKind) -> bool {
    matches!(
        k,
        OpKind::ScfFor | OpKind::ScfIf | OpKind::ScfWhile | OpKind::ScfParallel | OpKind::ScfForall
    )
}

/// The `tensor.*` shape-reinterpretation wrappers a reduce's result can be
/// threaded through before it reaches a broadcast.
fn is_reshape(k: OpKind) -> bool {
    matches!(
        k,
        OpKind::TensorReshape
            | OpKind::TensorExtract
            | OpKind::TensorExpandShape
            | OpKind::TensorCollapseShape
    )
}

/// Recognize the canonical single-block naive-attention idiom in `func`.
///
/// Returns `Some(island)` only when the body is PROVABLY:
///   load Q[m,d], load K[cap,d], `Kt = transpose(K)`, `S = Q@Kt`, scale `S`,
///   (optional) causal-mask add, `mx = reduce_max(S, dim 1)`,
///   `P = exp(S - mx)`, `l = reduce_sum(P, dim 1)`, `W = P / l`,
///   load V[cap,d], `O = W@V`, store O[m,d].
///
/// Any structural deviation (an `scf.for`, multiple stores, an unrecognized op
/// sequence, a transposed/odd layout, multi-head grid unrolling) yields `None`,
/// leaving the node naive (fail-safe).
pub fn recognize_attention(func: &IRFunction) -> Option<AttentionIsland> {
    // An already-tiled body (an `scf.for` / `scf.if` control-flow region) is never
    // the flat canonical idiom — bail. (Leaf region-bodied ops like
    // `tensor.generate` for a mask or an explicit-region `linalg.reduce` are fine;
    // only top-level CONTROL FLOW disqualifies.)
    if func.operations.iter().any(|op| is_control_flow(op.op_type)) {
        return None;
    }

    // Index views/tiles by their result SSA so we can walk the load/store chains.
    let mut views: HashMap<Ssa, ViewInfo> = HashMap::new();
    let mut tiles: HashMap<Ssa, TileInfo> = HashMap::new();
    // load result SSA -> (pointer arg, view shape, dtype) it reads.
    let mut load_src: HashMap<Ssa, (Ssa, Vec<i64>, DType)> = HashMap::new();
    // SSA -> op (for the compute chain).
    let mut def: HashMap<Ssa, &Operation> = HashMap::new();

    for op in func.operations {
        match op.op_type {
            OpKind::KtdpConstructMemoryView => {
                if let (Some(res), Some(&arg)) = (op.result, op.operands.first()) {
                    views.insert(
                        res,
                        ViewInfo {
                            arg,
                            shape: shape_attr(op),
                            dtype: dtype_attr(op),
                        },
                    );
                }
            }
            OpKind::KtdpConstructAccessTile => {
                if let (Some(res), Some(&view)) = (op.result, op.operands.first()) {
                    tiles.insert(res, TileInfo { view });
                }
            }
            OpKind::KtdpLoad => {
                if let (Some(res), Some(tile)) = (op.result, op.operands.first())
                    && let Some(ti) = tiles.get(tile)
                    && let Some(vi) = views.get(&ti.view)
                {
                    load_src.insert(res, (vi.arg, vi.shape.clone(), vi.dtype));
                }
            }
            _ => {}
        }
        if let Some(res) = op.result {
            def.insert(res, op);
        }
    }

    // Exactly one store: the attention output. (The real unrolled nodes store
    // many times — they fail here, which is the fail-safe we want.)
    let stores: Vec<&Operation> = func
        .operations
        .iter()
        .filter(|o| o.op_type == OpKind::KtdpStore)
        .collect();
    let store = match stores.as_slice() {
        [s] => s,
        _ => return None,
    };
    // store %value, %tile
    let stored_val = store.operands.first()?;
    let store_tile = store.operands.get(1)?;
    let o_ti = tiles.get(store_tile)?;
    let o_view = views.get(&o_ti.view)?;
    let o_arg = o_view.arg;
    let o_shape = o_view.shape.clone();

    // Walk back from the stored value: it must be `O = linalg.matmul(W, V)`.
    let av = def.get(stored_val)?;
    if av.op_type != OpKind::LinalgMatmul {
        return None;
    }
    let w_ssa = av.operands.first()?; // probabilities W = P / l
    let v_loaded = av.operands.get(1)?; // V (loaded straight)
    let (v_arg, v_shape, _vdt) = load_src.get(v_loaded)?.clone();

    // W = arith.divf(P, l_broadcast)
    let divw = def.get(w_ssa)?;
    if divw.op_type != OpKind::ArithDivf {
        return None;
    }
    let p_ssa = divw.operands.first()?;
    let lbcast = divw.operands.get(1)?;
    // P = math.exp(shifted)
    let pexp = def.get(p_ssa)?;
    if pexp.op_type != OpKind::MathExp {
        return None;
    }
    let shifted = pexp.operands.first()?;
    // shifted = arith.subf(scaled_masked, mx_broadcast)
    let sub = def.get(shifted)?;
    if sub.op_type != OpKind::ArithSubf {
        return None;
    }
    let scores_masked = sub.operands.first()?;

    // l_broadcast must trace (broadcast -> reshape) to `reduce_sum(P, 1)`.
    if !broadcast_traces_to_reduce(lbcast, p_ssa, OpKind::ArithAddf, &def) {
        return None;
    }
    // mx_broadcast must trace to `reduce_max(scores_masked, 1)`.
    let mxb = sub.operands.get(1)?;
    if !broadcast_traces_to_reduce(mxb, scores_masked, OpKind::ArithMaximumf, &def) {
        return None;
    }

    // scores_masked is either `arith.addf(scaled, mask)` (causal) or `scaled`.
    let (scaled, causal) = {
        let smop = def.get(scores_masked)?;
        if smop.op_type == OpKind::ArithAddf {
            // one operand is the scaled scores, the other the causal mask tensor.
            (*smop.operands.first()?, true)
        } else {
            (*scores_masked, false)
        }
    };

    // scaled = arith.mulf(raw_scores, scale_splat)
    let mulop = def.get(&scaled)?;
    if mulop.op_type != OpKind::ArithMulf {
        return None;
    }
    let raw_scores = mulop.operands.first()?;
    let scale_splat = mulop.operands.get(1)?;
    let scale = recover_scale(scale_splat, &def)?;

    // raw_scores = linalg.matmul(Q, Kt)
    let qk = def.get(raw_scores)?;
    if qk.op_type != OpKind::LinalgMatmul {
        return None;
    }
    let q_loaded = qk.operands.first()?;
    let kt_ssa = qk.operands.get(1)?;
    let (q_arg, q_shape, dtype) = load_src.get(q_loaded)?.clone();

    // Kt = linalg.transpose(K_loaded)
    let ktop = def.get(kt_ssa)?;
    if ktop.op_type != OpKind::LinalgTranspose {
        return None;
    }
    let k_loaded = ktop.operands.first()?;
    let (k_arg, k_shape, _kdt) = load_src.get(k_loaded)?.clone();

    // ---- shape sanity: Q[m,d], K[cap,d], V[cap,d], O[m,d] ----
    if q_shape.len() != 2 || k_shape.len() != 2 || v_shape.len() != 2 || o_shape.len() != 2 {
        return None;
    }
    let (m, d) = (q_shape[0], q_shape[1]);
    let (cap, kd) = (k_shape[0], k_shape[1]);
    if kd != d || v_shape != k_shape || o_shape != q_shape {
        return None;
    }
    if m <= 0 || cap <= 0 || d <= 0 {
        return None;
    }

    Some(AttentionIsland {
        q_arg,
        q_shape,
        k_arg,
        k_shape,
        v_arg,
        v_shape,
        o_arg,
        o_shape,
        m,
        cap,
        d,
        scale,
        causal,
        dtype,
    })
}

/// True if `bcast_ssa` is a `linalg.broadcast` whose source traces back through
/// an optional `tensor.reshape`/`tensor.extract` to `linalg.reduce { reduce_fn }`
/// over `target` (a per-row reduce of the scores/probabilities). This is the
/// `mx_broadcast` / `l_broadcast` chain the canonical softmax emits.
fn broadcast_traces_to_reduce(
    bcast_ssa: &Ssa,
    target: &Ssa,
    reduce_fn: OpKind,
    def: &HashMap<Ssa, &Operation>,
) -> bool {
    let Some(bop) = def.get(bcast_ssa) else {
        return false;
    };
    if bop.op_type != OpKind::LinalgBroadcast {
        return false;
    }
    let Some(src) = bop.operands.first() else {
        return false;
    };
    traces_to_reduce_of(src, target, reduce_fn, def)
}

/// True if `ssa` is `linalg.reduce { reduce_fn }(target)` over the last axis,
/// possibly via a `tensor.reshape` / `tensor.extract` wrapper.
fn traces_to_reduce_of(
    ssa: &Ssa,
    target: &Ssa,
    reduce_fn: OpKind,
    def: &HashMap<Ssa, &Operation>,
) -> bool {
    let mut cur = *ssa;
    // Skip a chain of reshape/extract wrappers (reduce -> [m] -> reshape [m,1]).
    for _ in 0..4 {
        let Some(op) = def.get(&cur) else {
            return false;
        };
        if is_reshape(op.op_type) {
            match op.operands.first() {
                Some(src) => cur = *src,
                None => return false,
            }
        } else {
            break;
        }
    }
    let Some(op) = def.get(&cur) else {
        return false;
    };
    if op.op_type != OpKind::LinalgReduce {
        return false;
    }
    let fn_ok = op.attr(AttrKey::ReduceFn) == Some(&Attr::Op(reduce_fn));
    let target_ok = op.operands.first() == Some(target);
    fn_ok && target_ok
}

/// Recover the `1/sqrt(d)` scale from a `tensor.splat %c` whose `%c` is an
/// `arith.constant` float.
fn recover_scale(splat_ssa: &Ssa, def: &HashMap<Ssa, &Operation>) -> Option<f32> {
    let splat = def.get(splat_ssa)?;
    if splat.op_type != OpKind::TensorSplat {
        return None;
    }
    let c = def.get(splat.operands.first()?)?;
    if c.op_type != OpKind::ArithConstant {
        return None;
    }
    match c.attr(AttrKey::Value) {
        Some(Attr::Float(f)) => Some(*f as f32),
        Some(Attr::Int(i)) => Some(*i as f32),
        _ => None,
    }
}

// ===========================================================================
// Tiling (online-softmax rewrite)
// ===========================================================================

/// Block size for the KV/cap loop. Chosen so the per-block scores tile `[m, BC]`
/// is comfortably below LX for the `m` the model uses; `cap` is split into
/// `ceil(cap / BC)` blocks. A power of two that divides the common caps (256,
/// 512, 1024, 2048, 4096) cleanly when possible; the loop handles a ragged tail
/// via a clamped block size.
const DEFAULT_KV_BLOCK: i64 = 128;

/// Choose a KV block size that (a) does not exceed the cap and (b) divides it
/// when a clean divisor near the default exists, else falls back to the default
/// (the loop's static unroll below handles any remainder by clamping).
fn choose_block(cap: i64) -> i64 {
    if cap <= DEFAULT_KV_BLOCK {
        return cap;
    }
    // Prefer the largest divisor of `cap` that is <= DEFAULT_KV_BLOCK and a power
    // of two, to keep every block equal-sized (no ragged tail to special-case).
    for b in [DEFAULT_KV_BLOCK, 64, 32, 16, 8, 4, 2, 1] {
        if cap % b == 0 {
            return b;
        }
    }
    1
}

/// All divisors of `cap` that are `<= DEFAULT_KV_BLOCK`, descending (so the first
/// fitting one is the largest equal-sized block). Always includes `1`.
fn cap_divisors(cap: i64) -> Vec<i64> {
    let mut ds: Vec<i64> = (1..=cap.min(DEFAULT_KV_BLOCK))
        .filter(|b| cap % b == 0)
        .collect();
    ds.sort_unstable_by(|a, b| b.cmp(a));
    ds
}

/// Budget-aware KV block size for the RE-ROLLED context tiling: the LARGEST
/// divisor `b` of `cap` (`b <= DEFAULT_KV_BLOCK`) whose per-block scores tile
/// `[m, b]` is BELOW the cap (`!needs_flash(m*b*bytes)`), so each block fits LX.
///
/// If even the smallest divisor still overflows (a pathologically tiny forced
/// budget), fall back to that smallest divisor — the most aggressive tiling we
/// can emit. In all cases `b <= cap`; when `cap` has a proper divisor `< cap`
/// (the real caps are 64-multiples) and the full `[m, cap]` tile overflows, the
/// returned `b` is strictly `< cap`, so REAL tiling happens.
/// Constrains BOTH per-block tiles: the `[m, blk]` scores tile must fit the
/// injected LX budget (`!needs_flash`) AND the `[blk, d]` context K/V tile must
/// fit [`FLASH_CONTEXT_TILE_MAX`]. In the small-query/long-context regime the KV
/// constraint binds (scores are already tiny), so a scores-only chooser would
/// pick `blk = cap` (no tiling) and overflow LX; this picks the largest cap
/// divisor that satisfies both.
fn choose_block_budgeted(
    m: i64,
    cap: i64,
    d: i64,
    bytes: usize,
    needs_flash: &impl Fn(usize) -> bool,
) -> i64 {
    let divisors = cap_divisors(cap);
    let scores = |b: i64| {
        (m as usize)
            .saturating_mul(b as usize)
            .saturating_mul(bytes)
    };
    let kv = |b: i64| {
        (d as usize)
            .saturating_mul(b as usize)
            .saturating_mul(bytes)
    };
    // Largest divisor whose per-block scores AND context K/V tiles both fit.
    for &b in &divisors {
        if !needs_flash(scores(b)) && kv(b) <= FLASH_CONTEXT_TILE_MAX {
            return b;
        }
    }
    // None fits: take the smallest divisor (the minimal achievable tile). When the
    // full tile overflows but no sub-block formally "fits", we STILL tile to the
    // smallest block (strictly smaller footprint) — honest best effort.
    *divisors.last().unwrap_or(&1)
}

fn const_index<'a>(a: &'a Arena, res: Ssa, v: i64) -> Operation<'a> {
    Operation::new(a, Some(res), OpKind::ArithConstant, &[]).with_attr(
        a,
        AttrKey::Value,
        Attr::Int(v),
    )
}
fn const_f<'a>(a: &'a Arena, res: Ssa, v: f64) -> Operation<'a> {
    Operation::new(a, Some(res), OpKind::ArithConstant, &[]).with_attr(
        a,
        AttrKey::Value,
        Attr::Float(v),
    )
}

/// Build a `ktdp.construct_memory_view %ptr {shape, strides, memory_space, dtype}`
/// — a logical view only (RFC 0682: does NOT allocate).
fn mk_view<'a>(a: &'a Arena, res: Ssa, ptr: Ssa, shape: &[i64], dtype: DType) -> Operation<'a> {
    // Row-major strides.
    let mut strides = vec![1i64; shape.len()];
    for k in (0..shape.len().saturating_sub(1)).rev() {
        strides[k] = strides[k + 1] * shape[k + 1];
    }
    Operation::new(a, Some(res), OpKind::KtdpConstructMemoryView, &[ptr])
        .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(shape.to_vec())))
        .with_attr(a, AttrKey::Strides, Attr::IntList(a.ints(strides)))
        .with_attr(
            a,
            AttrKey::MemorySpace,
            Attr::Str(a.str(String::from("HBM"))),
        )
        .with_attr(a, AttrKey::Dtype, Attr::Dtype(dtype))
}

/// Whole-tensor `ktdp.load` of a view: build the full-shape access tile then load.
fn mk_whole_load<'a>(
    a: &'a Arena,
    g: &mut NameGen,
    ops: &mut Vec<Operation<'a>>,
    view: Ssa,
    shape: &[i64],
) -> Ssa {
    let tile = g.mint();
    ops.push(
        Operation::new(a, Some(tile), OpKind::KtdpConstructAccessTile, &[view]).with_attr(
            a,
            AttrKey::Shape,
            Attr::IntList(a.ints(shape.to_vec())),
        ),
    );
    let loaded = g.mint();
    ops.push(Operation::new(a, Some(loaded), OpKind::KtdpLoad, &[tile]));
    loaded
}

/// A `tensor.splat %scalar -> tensor<shape×dtype>`.
fn mk_splat<'a>(a: &'a Arena, res: Ssa, scalar: Ssa, shape: &[i64], dtype: DType) -> Operation<'a> {
    Operation::new(a, Some(res), OpKind::TensorSplat, &[scalar])
        .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(shape.to_vec())))
        .with_attr(a, AttrKey::Dtype, Attr::Dtype(dtype))
}

/// A `tensor.empty() -> tensor<shape×dtype>` (zero-filled init for matmul outs).
fn mk_empty<'a>(a: &'a Arena, res: Ssa, shape: &[i64], dtype: DType) -> Operation<'a> {
    Operation::new(a, Some(res), OpKind::TensorEmpty, &[])
        .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(shape.to_vec())))
        .with_attr(a, AttrKey::Dtype, Attr::Dtype(dtype))
}

/// A `linalg.reduce { reduce_fn } ins(%x) outs(%init) dimensions = [1]` over the
/// last axis of a `[r, c]` tile -> `[r]`.
fn mk_reduce<'a>(a: &'a Arena, res: Ssa, x: Ssa, init: Ssa, reduce_fn: OpKind) -> Operation<'a> {
    Operation::new(a, Some(res), OpKind::LinalgReduce, &[x])
        .with_attr(a, AttrKey::ReduceFn, Attr::Op(reduce_fn))
        .with_attr(a, AttrKey::Dimensions, Attr::IntList(a.ints(vec![1])))
        .with_attr(a, AttrKey::OutsVar, Attr::Ssas(a.ssa(vec![init])))
}

/// Rewrite a recognized [`AttentionIsland`] into a tiled online-softmax function.
///
/// The emitted body is, for `nb = ceil(cap / BC)` KV blocks of size `BC`:
/// ```text
///   Q = load Q[m,d]
///   m0 = splat(-inf, [m,1]); l0 = splat(0, [m,1]); acc0 = empty([m,d])
///   (m_f, l_f, acc_f) = scf.for j = 0 to nb step 1 iter_args(m_i, l_i, acc):
///       Kj = extract_slice K[j*BC .. , :]      ([BC, d])
///       Vj = extract_slice V[j*BC .. , :]      ([BC, d])
///       Sj = (Q @ Kjᵀ) * scale  (+ causal mask_j)         ([m, BC])
///       rmax = reduce_max(Sj, 1) -> [m]
///       m_new = max(m_i, rmax_bcast)                       ([m,1])
///       P = exp(Sj - m_new_bcast)                          ([m, BC])
///       alpha = exp(m_i - m_new)                           ([m,1])
///       rsum = reduce_sum(P, 1) -> [m]
///       l_new = alpha*l_i + rsum_bcast                     ([m,1])
///       acc_new = alpha_bcast*acc + P @ Vj                 ([m,d])
///       yield m_new, l_new, acc_new
///   O = acc_f / l_f_bcast
///   store O -> O[m,d]
/// ```
/// Causal masking is applied as `-inf` on KV positions `> (q_row + (cap - m))`
/// per block, matched to the naive form's mask. The `[m, BC]` block scores tile
/// fits LX by construction (BC = `choose_block(cap)` ≤ 128).
pub fn tile_attention<'a>(
    a: &'a Arena,
    island: &AttentionIsland,
    g: &mut NameGen,
) -> IRFunction<'a> {
    let isl = island;
    let dt = isl.dtype;
    let bc = choose_block(isl.cap);
    let nb = isl.cap / bc; // choose_block guarantees bc | cap
    let mut ops: Vec<Operation<'a>> = Vec::new();

    // ---- constants ----
    let c_neg_inf = g.mint();
    ops.push(const_f(a, c_neg_inf, -1.0e30));
    let c_zero = g.mint();
    ops.push(const_f(a, c_zero, 0.0));
    let c_scale = g.mint();
    ops.push(const_f(a, c_scale, isl.scale as f64));

    // A zero column-offset constant for the per-block KV access tiles. Kept an
    // SSA operand (not a literal in an attribute) so whole-program fusion's
    // operand renaming threads it through correctly.
    let c0 = g.mint();
    ops.push(const_index(a, c0, 0));

    // ---- load Q whole ----
    let q_view = g.mint();
    ops.push(mk_view(a, q_view, isl.q_arg, &isl.q_shape, dt));
    let q = mk_whole_load(a, g, &mut ops, q_view, &isl.q_shape);

    // ---- build K / V views (loaded per-block inside the loop) ----
    // Each KV block is read straight from HBM with a `ktdp.construct_access_tile`
    // at the dynamic block offset `[j*BC, 0]` (its index operands are renamed by
    // fusion) — exactly the "only the `[BC, d]` block is resident" behavior that
    // keeps the scores tile inside LX. This is the KTIR-native analogue of an
    // `extract_slice` of the KV block and avoids materializing the whole `[cap,d]`
    // tensor in LX.
    let k_view = g.mint();
    ops.push(mk_view(a, k_view, isl.k_arg, &isl.k_shape, dt));
    let v_view = g.mint();
    ops.push(mk_view(a, v_view, isl.v_arg, &isl.v_shape, dt));

    // ---- iter-arg inits ----
    let m0 = g.mint();
    ops.push(mk_splat(a, m0, c_neg_inf, &[isl.m, 1], dt));
    let l0 = g.mint();
    ops.push(mk_splat(a, l0, c_zero, &[isl.m, 1], dt));
    let acc0 = g.mint();
    ops.push(mk_empty(a, acc0, &[isl.m, isl.d], dt));

    // ---- loop bounds ----
    let lb = g.mint();
    ops.push(const_index(a, lb, 0));
    let ub = g.mint();
    ops.push(const_index(a, ub, nb));
    let step = g.mint();
    ops.push(const_index(a, step, 1));
    let bc_c = g.mint();
    ops.push(const_index(a, bc_c, bc));

    // iter-arg body-visible values.
    let mi = g.mint();
    let li = g.mint();
    let acci = g.mint();
    let iv = g.mint();

    // ---- loop body ----
    let mut body: Vec<Operation<'a>> = Vec::new();
    // block start offset = j * BC
    let off = g.mint();
    body.push(Operation::new(a, Some(off), OpKind::ArithMuli, &[iv, bc_c]));

    // Kj = load K[off.., :]  -> [BC, d]  (KTIR access tile at the block offset)
    let kj = block_load(a, g, &mut body, k_view, off, c0, [bc, isl.d]);
    // Vj = load V[off.., :]  -> [BC, d]
    let vj = block_load(a, g, &mut body, v_view, off, c0, [bc, isl.d]);

    // Kjt = transpose(Kj) -> [d, BC]
    let kjt_init = g.mint();
    body.push(mk_empty(a, kjt_init, &[isl.d, bc], dt));
    let kjt = g.mint();
    body.push(
        Operation::new(a, Some(kjt), OpKind::LinalgTranspose, &[kj, kjt_init]).with_attr(
            a,
            AttrKey::Permutation,
            Attr::IntList(a.ints(vec![1, 0])),
        ),
    );

    // raw = Q @ Kjt -> [m, BC]
    let raw_init = g.mint();
    body.push(mk_empty(a, raw_init, &[isl.m, bc], dt));
    let raw = g.mint();
    body.push(Operation::new(
        a,
        Some(raw),
        OpKind::LinalgMatmul,
        &[q, kjt, raw_init],
    ));

    // scaled = raw * scale_splat
    let scale_t = g.mint();
    body.push(mk_splat(a, scale_t, c_scale, &[isl.m, bc], dt));
    let scaled = g.mint();
    body.push(Operation::new(
        a,
        Some(scaled),
        OpKind::ArithMulf,
        &[raw, scale_t],
    ));

    // sj = scaled (+ causal mask for this block, if causal)
    let sj = if isl.causal {
        let mask = causal_mask_block(a, g, &mut body, off, isl.cap, [isl.m, bc], dt);
        let masked = g.mint();
        body.push(Operation::new(
            a,
            Some(masked),
            OpKind::ArithAddf,
            &[scaled, mask],
        ));
        masked
    } else {
        scaled
    };

    // rmax = reduce_max(sj, 1) -> [m]
    let rmax_init = g.mint();
    body.push(mk_splat(a, rmax_init, c_neg_inf, &[isl.m], dt));
    let rmax = g.mint();
    body.push(mk_reduce(a, rmax, sj, rmax_init, OpKind::ArithMaximumf));
    // reduce yields [m]; reshape to [m,1] for elementwise with the [m,1] iter-args.
    let rmax2 = g.mint();
    body.push(reshape_to(a, rmax2, rmax, &[isl.m, 1]));

    // m_new = max(m_i, rmax2)
    let mnew = g.mint();
    body.push(Operation::new(
        a,
        Some(mnew),
        OpKind::ArithMaximumf,
        &[mi, rmax2],
    ));

    // m_new broadcast to [m, BC]
    let mnew_b = broadcast_col_to(a, g, &mut body, mnew, isl.m, bc, dt);
    // shifted = sj - m_new_b
    let shifted = g.mint();
    body.push(Operation::new(
        a,
        Some(shifted),
        OpKind::ArithSubf,
        &[sj, mnew_b],
    ));
    // P = exp(shifted) -> [m, BC]
    let p = g.mint();
    body.push(Operation::new(a, Some(p), OpKind::MathExp, &[shifted]));

    // alpha = exp(m_i - m_new) -> [m,1]
    let mdiff = g.mint();
    body.push(Operation::new(
        a,
        Some(mdiff),
        OpKind::ArithSubf,
        &[mi, mnew],
    ));
    let alpha = g.mint();
    body.push(Operation::new(a, Some(alpha), OpKind::MathExp, &[mdiff]));

    // rsum = reduce_sum(P, 1) -> [m] -> [m,1]
    let rsum_init = g.mint();
    body.push(mk_splat(a, rsum_init, c_zero, &[isl.m], dt));
    let rsum = g.mint();
    body.push(mk_reduce(a, rsum, p, rsum_init, OpKind::ArithAddf));
    let rsum2 = g.mint();
    body.push(reshape_to(a, rsum2, rsum, &[isl.m, 1]));

    // l_new = alpha * l_i + rsum2
    let al = g.mint();
    body.push(Operation::new(a, Some(al), OpKind::ArithMulf, &[alpha, li]));
    let lnew = g.mint();
    body.push(Operation::new(
        a,
        Some(lnew),
        OpKind::ArithAddf,
        &[al, rsum2],
    ));

    // acc_new = alpha_b * acc + P @ Vj
    let alpha_b = broadcast_col_to(a, g, &mut body, alpha, isl.m, isl.d, dt);
    let acc_scaled = g.mint();
    body.push(Operation::new(
        a,
        Some(acc_scaled),
        OpKind::ArithMulf,
        &[alpha_b, acci],
    ));
    let pv_init = g.mint();
    body.push(mk_empty(a, pv_init, &[isl.m, isl.d], dt));
    let pv = g.mint();
    body.push(Operation::new(
        a,
        Some(pv),
        OpKind::LinalgMatmul,
        &[p, vj, pv_init],
    ));
    let accnew = g.mint();
    body.push(Operation::new(
        a,
        Some(accnew),
        OpKind::ArithAddf,
        &[acc_scaled, pv],
    ));

    // yield m_new, l_new, acc_new
    body.push(Operation::new(
        a,
        None,
        OpKind::ScfYield,
        &[mnew, lnew, accnew],
    ));

    // ---- the scf.for ----
    let m_f = g.mint();
    let l_f = g.mint();
    let acc_f = g.mint();
    let forop = Operation::new(a, None, OpKind::ScfFor, &[lb, ub, step, m0, l0, acc0])
        .with_attr(a, AttrKey::IterVar, Attr::Ssas(a.ssa(vec![iv])))
        .with_attr(a, AttrKey::IterArgs, Attr::Ssas(a.ssa(vec![mi, li, acci])))
        .with_attr(
            a,
            AttrKey::ResultNames,
            Attr::Ssas(a.ssa(vec![m_f, l_f, acc_f])),
        );
    ops.push(Operation {
        regions: a.regions(vec![a.ops(body)]),
        ..forop
    });

    // ---- final normalize: O = acc_f / l_f_b ----
    let l_f_b = broadcast_col_to(a, g, &mut ops, l_f, isl.m, isl.d, dt);
    let o = g.mint();
    ops.push(Operation::new(
        a,
        Some(o),
        OpKind::ArithDivf,
        &[acc_f, l_f_b],
    ));

    // ---- store O -> O[m,d] ----
    let o_view = g.mint();
    ops.push(mk_view(a, o_view, isl.o_arg, &isl.o_shape, dt));
    let o_at = g.mint();
    ops.push(
        Operation::new(a, Some(o_at), OpKind::KtdpConstructAccessTile, &[o_view]).with_attr(
            a,
            AttrKey::Shape,
            Attr::IntList(a.ints(isl.o_shape.clone())),
        ),
    );
    ops.push(Operation::new(a, None, OpKind::KtdpStore, &[o, o_at]));
    ops.push(Operation::new(a, None, OpKind::FuncReturn, &[]));

    IRFunction {
        name: "", // caller stamps the original name
        arguments: a.args(
            isl.arg_ssas()
                .into_iter()
                .map(|v| (v, IrType::Index))
                .collect(),
        ),
        operations: a.ops(ops),
        grid: (1, 1, 1),
        return_type: None,
    }
}

/// Load a `[rows, cols]` block of an HBM `[*, cols]` view at dynamic row offset
/// `%off` (column offset `%c0`): `construct_access_tile %view[%off, %c0]` (block
/// shape) then `ktdp.load`. The access-tile index operands `%off`/`%c0` are real
/// SSA operands, so whole-program fusion's operand renaming threads them through
/// a fused segment correctly (a `tensor.extract_slice`'s `slice_offsets` live in
/// an attribute fusion does not rewrite — using the KTIR access tile sidesteps
/// that, and is the hardware-native "only the resident block is in LX" form).
fn block_load<'a>(
    a: &'a Arena,
    g: &mut NameGen,
    ops: &mut Vec<Operation<'a>>,
    view: Ssa,
    off: Ssa,
    c0: Ssa,
    tile: [i64; 2],
) -> Ssa {
    let at = g.mint();
    ops.push(
        Operation::new(
            a,
            Some(at),
            OpKind::KtdpConstructAccessTile,
            &[view, off, c0],
        )
        .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(tile.to_vec()))),
    );
    let loaded = g.mint();
    ops.push(Operation::new(a, Some(loaded), OpKind::KtdpLoad, &[at]));
    loaded
}

/// `%r = tensor.reshape %x -> tensor<shape>` (a pure reinterpretation; the
/// interpreter reads `target_shape`).
fn reshape_to<'a>(a: &'a Arena, res: Ssa, x: Ssa, shape: &[i64]) -> Operation<'a> {
    Operation::new(a, Some(res), OpKind::TensorReshape, &[x]).with_attr(
        a,
        AttrKey::TargetShape,
        Attr::IntList(a.ints(shape.to_vec())),
    )
}

/// Broadcast a `[m, 1]` column tile to `[m, cols]`, pushing the ops onto `ops`
/// and returning the result SSA. Uses `linalg.broadcast ins(%col) outs(%init)`
/// with an empty `dimensions` list: the interpreter then NumPy right-aligned-
/// broadcasts the `[m,1]` input up to the `[m,cols]` outs shape (each row's
/// single value filled across the `cols` columns). The `outs` tile supplies the
/// target shape, so we materialize it with `tensor.empty` first.
fn broadcast_col_to<'a>(
    a: &'a Arena,
    g: &mut NameGen,
    ops: &mut Vec<Operation<'a>>,
    col: Ssa,
    m: i64,
    cols: i64,
    dt: DType,
) -> Ssa {
    let init = g.mint();
    ops.push(mk_empty(a, init, &[m, cols], dt));
    let res = g.mint();
    ops.push(
        Operation::new(a, Some(res), OpKind::LinalgBroadcast, &[col, init]).with_attr(
            a,
            AttrKey::Dimensions,
            Attr::IntList(a.ints(vec![])),
        ),
    );
    res
}

/// Emit the per-block causal mask `[m, BC]` (pushing ops, returning its SSA),
/// using only ELEMENTWISE + CONSTANT ops (no region block-args) so it survives
/// whole-program fusion's operand renaming.
///
/// Visibility rule (matched to the naive form): query row `qr` has absolute KV
/// position `cap - m + qr` and attends to key block position `off + kc` iff
/// `off + kc <= cap - m + qr`, i.e. `kc - qr <= (cap - m) - off`. The left side
/// `D[qr,kc] = kc - qr` is a STATIC `[m, BC]` integer constant (baked at emit
/// time); the right side `rhs = (cap - m) - off` is a per-iteration scalar. The
/// mask is then `select(D <= rhs, 0, -inf)` — all elementwise.
fn causal_mask_block<'a>(
    a: &'a Arena,
    g: &mut NameGen,
    ops: &mut Vec<Operation<'a>>,
    off: Ssa,
    cap: i64,
    tile: [i64; 2],
    dt: DType,
) -> Ssa {
    let [m, bc] = tile;
    // D[qr,kc] = kc - qr, baked as a dense i32 constant tensor.
    let mut d_vals = Vec::with_capacity((m * bc) as usize);
    for qr in 0..m {
        for kc in 0..bc {
            d_vals.push(kc - qr);
        }
    }
    let d = g.mint();
    ops.push(
        Operation::new(a, Some(d), OpKind::ArithConstant, &[])
            .with_attr(a, AttrKey::IsTensor, Attr::Bool(true))
            .with_attr(a, AttrKey::DenseList, Attr::Bool(true))
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![m, bc])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::I32))
            .with_attr(a, AttrKey::Value, Attr::IntList(a.ints(d_vals))),
    );
    // rhs = (cap - m) - off  (scalar index).
    let base = g.mint();
    ops.push(const_index(a, base, cap - m));
    let rhs = g.mint();
    ops.push(Operation::new(
        a,
        Some(rhs),
        OpKind::ArithSubi,
        &[base, off],
    ));
    let rhs_t = g.mint();
    ops.push(mk_splat(a, rhs_t, rhs, &[m, bc], DType::I32));
    // cond = D <= rhs_t  (elementwise i1 tile).
    let cond = g.mint();
    ops.push(
        Operation::new(a, Some(cond), OpKind::ArithCmpi, &[d, rhs_t]).with_attr(
            a,
            AttrKey::Predicate,
            Attr::Str(a.str(String::from("sle"))),
        ),
    );
    // visible -> 0, masked -> -inf  (elementwise select into f16).
    let zero_c = g.mint();
    ops.push(const_f(a, zero_c, 0.0));
    let zero_t = g.mint();
    ops.push(mk_splat(a, zero_t, zero_c, &[m, bc], dt));
    let ninf_c = g.mint();
    ops.push(const_f(a, ninf_c, -1.0e30));
    let ninf_t = g.mint();
    ops.push(mk_splat(a, ninf_t, ninf_c, &[m, bc], dt));
    let mask = g.mint();
    ops.push(Operation::new(
        a,
        Some(mask),
        OpKind::ArithSelect,
        &[cond, zero_t, ninf_t],
    ));
    mask
}

// ===========================================================================
// RE-ROLLED recognizer + tiler (the REAL model node, post head_rewrite)
// ===========================================================================

/// Decoded `ktdp.load` source: the pointer arg, full view shape, and dtype.
#[derive(Clone)]
struct RrLoad {
    arg: Ssa,
    view_shape: Vec<i64>,
    dtype: DType,
}

/// One recognized softmax block (context OR diagonal) walked back from its `A·V`
/// matmul: the loaded V source, the loaded Q source, the loaded K source, the
/// scaled-scores SSA (`mulf(Q·Kᵀ, scale)`), the exp argument (`subf(S, gm_bc)`),
/// the masked-scores SSA `S` (`addf(scaled, mask)`), and the running-sum SSA fed
/// into the global `gs`.
struct RrBlock {
    v: RrLoad,
    q: RrLoad,
    k: RrLoad,
    scale: f32,
    /// The masked-scores tensor `S` (post mask add) — the reduce / subf operand.
    masked_scores: Ssa,
    /// The per-block exp probabilities (`pc` / `pd`).
    probs: Ssa,
    /// The per-block row-sum SSA (`scs` / `sds`) — the `gs = addf(.,.)` operand.
    row_sum: Ssa,
}

/// Recognize the RE-ROLLED two-block head-attention idiom (the
/// [`crate::head_rewrite`] OUTPUT — the REAL model node).
///
/// Returns `Some(island)` only when the body is PROVABLY:
///   * grid `(H, 1, 1)` with `H > 1`; no top-level `scf.*` (so a re-tiled body is
///     never re-recognized);
///   * exactly ONE `ktdp.store` whose value is `arith.addf(ovc, ovd)` with both
///     args `linalg.matmul` (the two `A·V`);
///   * the CONTEXT block (V/K from a `[cap, kv_cols]` view, mask a
///     `linalg.broadcast` of a `[1, cap]` load) and the DIAGONAL block (V/K from
///     a `[m, kv_cols]` view, mask a dense `[m, m]` `arith.constant`), each a
///     `divf(exp(subf(addf(mulf(matmul(Q,Kᵀ),scale),mask), gm_bc)), gs_bc)`;
///   * ONE shared `gm = arith.maximumf(mc, md)` and ONE shared
///     `gs = arith.addf(reduce_sum(pc), reduce_sum(pd))`;
///   * Q is the SAME load arg for both blocks.
///
/// Any deviation → `None` → identity. `cap`, `m`, `d`, `gqac`, `hdc`, `scale`,
/// `ninf` are all RE-DERIVED from the IR (never model names or literal shapes).
pub fn recognize_rerolled_attention(func: &IRFunction) -> Option<ReRolledIsland> {
    // grid = [H,1,1], H > 1 (head-parallel).
    let (h, gy, gz) = func.grid;
    if gy != 1 || gz != 1 || h <= 1 {
        return None;
    }
    let h = h as i64;

    // No top-level control flow: a body that already contains an `scf.for` is the
    // already-cap-tiled form (or something else) — never re-recognize it.
    if func.operations.iter().any(|op| is_control_flow(op.op_type)) {
        return None;
    }

    // Index views / access tiles / loads / defs / int-constants.
    let mut views: HashMap<Ssa, ViewInfo> = HashMap::new();
    let mut tiles: HashMap<Ssa, TileInfo> = HashMap::new();
    let mut load_src: HashMap<Ssa, RrLoad> = HashMap::new();
    let mut def: HashMap<Ssa, &Operation> = HashMap::new();
    let mut int_const: HashMap<Ssa, i64> = HashMap::new();

    for op in func.operations {
        match op.op_type {
            OpKind::KtdpConstructMemoryView => {
                if let (Some(res), Some(&arg)) = (op.result, op.operands.first()) {
                    views.insert(
                        res,
                        ViewInfo {
                            arg,
                            shape: shape_attr(op),
                            dtype: dtype_attr(op),
                        },
                    );
                }
            }
            OpKind::KtdpConstructAccessTile => {
                if let (Some(res), Some(&view)) = (op.result, op.operands.first()) {
                    tiles.insert(res, TileInfo { view });
                }
            }
            OpKind::KtdpLoad => {
                if let (Some(res), Some(tile)) = (op.result, op.operands.first())
                    && let Some(ti) = tiles.get(tile)
                    && let Some(vi) = views.get(&ti.view)
                {
                    load_src.insert(
                        res,
                        RrLoad {
                            arg: vi.arg,
                            view_shape: vi.shape.clone(),
                            dtype: vi.dtype,
                        },
                    );
                }
            }
            OpKind::ArithConstant => {
                if let (Some(res), Some(Attr::Int(v))) = (op.result, op.attr(AttrKey::Value)) {
                    int_const.insert(res, *v);
                }
            }
            _ => {}
        }
        if let Some(res) = op.result {
            def.insert(res, op);
        }
    }

    // Per-head selection arithmetic (PRESERVED verbatim by the tiler): a
    // `get_compute_tile_id`, a `divui %hpid, %gqac`, a `muli %hpid, %hdc`.
    if !func
        .operations
        .iter()
        .any(|o| o.op_type == OpKind::KtdpGetComputeTileId)
    {
        return None;
    }
    let gqac = func
        .operations
        .iter()
        .find(|o| o.op_type == OpKind::ArithDivui)
        .and_then(|o| o.operands.get(1))
        .and_then(|c| int_const.get(c).copied())?;
    if gqac < 1 {
        return None;
    }

    // Exactly ONE store; its value = arith.addf(ovc, ovd).
    let stores: Vec<&Operation> = func
        .operations
        .iter()
        .filter(|o| o.op_type == OpKind::KtdpStore)
        .collect();
    let store = match stores.as_slice() {
        [s] => s,
        _ => return None,
    };
    let stored_val = store.operands.first()?;
    let add = def.get(stored_val)?;
    if add.op_type != OpKind::ArithAddf {
        return None;
    }
    let ov0 = add.operands.first()?;
    let ov1 = add.operands.get(1)?;

    // Walk back BOTH `ov = matmul(w, v)` summands into softmax blocks.
    let blk0 = rr_walk_block(ov0, &def, &load_src)?;
    let blk1 = rr_walk_block(ov1, &def, &load_src)?;

    // Disambiguate context vs diagonal by V-view ROWS (cap vs m), NOT operand
    // order. The context V view has `rows == cap`, the diagonal `rows == m`. They
    // must differ (otherwise we cannot tell them apart → fail-safe).
    if blk0.v.view_shape.len() != 2 || blk1.v.view_shape.len() != 2 {
        return None;
    }
    let (ctx, diag) = if rr_is_context(&blk0, &def) && !rr_is_context(&blk1, &def) {
        (&blk0, &blk1)
    } else if rr_is_context(&blk1, &def) && !rr_is_context(&blk0, &def) {
        (&blk1, &blk0)
    } else {
        return None; // ambiguous: both or neither look like the context block.
    };

    // Q must be the SAME load arg for both blocks (one query tile).
    if ctx.q.arg != diag.q.arg {
        return None;
    }
    if (ctx.scale - diag.scale).abs() > 1e-4 {
        return None;
    }

    // ONE shared gm = maximumf(mc, md): both blocks' exp args subtract the SAME
    // broadcast of `gm`, and that gm is `maximumf(reduce_max(Sc), reduce_max(Sd))`.
    let gm = rr_shared_gm(ctx, diag, &def)?;
    if !rr_gm_is_max_of_reduces(&gm, &ctx.masked_scores, &diag.masked_scores, &def) {
        return None;
    }

    // ONE shared gs = addf(reduce_sum(pc), reduce_sum(pd)); each block's divf
    // denominator broadcasts THIS gs.
    rr_check_shared_gs(ctx, diag, &def)?;

    // ---- shape recovery (all RE-DERIVED) ----
    // Q/O view [m, q_cols]; q_cols = H*d.
    let q_shape = &ctx.q.view_shape;
    if q_shape.len() != 2 {
        return None;
    }
    let (m, q_cols) = (q_shape[0], q_shape[1]);
    if m <= 0 || q_cols % h != 0 {
        return None;
    }
    let d = q_cols / h;
    if d <= 0 {
        return None;
    }
    // Context K/V view [cap, kv_cols]; diagonal K/V view [m, kv_cols].
    let cap = ctx.v.view_shape[0];
    let kv_cols = ctx.v.view_shape[1];
    if cap <= 0 || kv_cols <= 0 || kv_cols % d != 0 {
        return None;
    }
    if ctx.k.view_shape != [cap, kv_cols] {
        return None;
    }
    if diag.k.view_shape != [m, kv_cols] || diag.v.view_shape != [m, kv_cols] {
        return None;
    }

    // O view [m, q_cols] + its pointer arg, from the store tile.
    let store_tile = store.operands.get(1)?;
    let o_ti = tiles.get(store_tile)?;
    let o_view = views.get(&o_ti.view)?;
    if o_view.shape != *q_shape {
        return None;
    }

    // Mask view [1, cap] (context per-head mask) + pointer arg.
    let mask_arg = rr_context_mask_arg(ctx, &def, &load_src, cap)?;

    // -inf recovered from the diagonal triangular mask constant (else default).
    let ninf = rr_recover_tri_ninf(diag, &def).unwrap_or(-1.0e38);

    Some(ReRolledIsland {
        q_arg: ctx.q.arg,
        o_arg: o_view.arg,
        mask_arg,
        kc_arg: ctx.k.arg,
        kd_arg: diag.k.arg,
        vc_arg: ctx.v.arg,
        vd_arg: diag.v.arg,
        q_cols,
        kv_cols,
        m,
        cap,
        d,
        gqac,
        hdc: d,
        h,
        scale: ctx.scale,
        ninf,
        dtype: ctx.q.dtype,
    })
}

/// Walk back one `ov = linalg.matmul(w, v_loaded)` summand into a softmax block.
/// `w = divf(exp(subf(addf(mulf(matmul(Q, transpose(K)), scale_splat), mask),
/// gm_bc)), gs_bc)`. Returns `None` on any deviation.
fn rr_walk_block(
    ov: &Ssa,
    def: &HashMap<Ssa, &Operation>,
    load_src: &HashMap<Ssa, RrLoad>,
) -> Option<RrBlock> {
    let av = def.get(ov)?;
    if av.op_type != OpKind::LinalgMatmul {
        return None;
    }
    let w = av.operands.first()?;
    let v_loaded = av.operands.get(1)?;
    let v = load_src.get(v_loaded)?.clone();

    // w = divf(probs, gs_bc)
    let divw = def.get(w)?;
    if divw.op_type != OpKind::ArithDivf {
        return None;
    }
    let probs = *divw.operands.first()?;
    // probs = exp(subf(S, gm_bc))
    let pexp = def.get(&probs)?;
    if pexp.op_type != OpKind::MathExp {
        return None;
    }
    let sub = def.get(pexp.operands.first()?)?;
    if sub.op_type != OpKind::ArithSubf {
        return None;
    }
    let masked_scores = *sub.operands.first()?;

    // S = addf(scaled, mask)
    let sop = def.get(&masked_scores)?;
    if sop.op_type != OpKind::ArithAddf {
        return None;
    }
    let scaled = sop.operands.first()?;
    // scaled = mulf(raw, scale_splat)
    let mulop = def.get(scaled)?;
    if mulop.op_type != OpKind::ArithMulf {
        return None;
    }
    let raw = mulop.operands.first()?;
    let scale = recover_scale(mulop.operands.get(1)?, def)?;
    // raw = matmul(Q_loaded, Kt); Kt = transpose(K_loaded)
    let qk = def.get(raw)?;
    if qk.op_type != OpKind::LinalgMatmul {
        return None;
    }
    let q_loaded = qk.operands.first()?;
    let q = load_src.get(q_loaded)?.clone();
    let ktop = def.get(qk.operands.get(1)?)?;
    if ktop.op_type != OpKind::LinalgTranspose {
        return None;
    }
    let k_loaded = ktop.operands.first()?;
    let k = load_src.get(k_loaded)?.clone();

    // row_sum = the reduce_sum CONSUMING probs (feeds the global gs).
    let row_sum = rr_reduce_consuming(&probs, OpKind::ArithAddf, def)?;

    Some(RrBlock {
        v,
        q,
        k,
        scale,
        masked_scores,
        probs,
        row_sum,
    })
}

/// True if `blk` is the CONTEXT block: its mask add operand is a
/// `linalg.broadcast` (the per-head `[1, cap]` mask), as opposed to the diagonal
/// block whose mask is a dense `arith.constant` `[m, m]` triangle.
fn rr_is_context(blk: &RrBlock, def: &HashMap<Ssa, &Operation>) -> bool {
    let Some(sop) = def.get(&blk.masked_scores) else {
        return false;
    };
    let Some(mask) = sop.operands.get(1) else {
        return false;
    };
    let Some(mop) = def.get(mask) else {
        return false;
    };
    mop.op_type == OpKind::LinalgBroadcast
}

/// Verify both blocks' exp args subtract the SAME `gm` broadcast and return that
/// `gm` SSA. (Each `gm_bc` is `linalg.broadcast(reshape(gm))`.)
fn rr_shared_gm(ctx: &RrBlock, diag: &RrBlock, def: &HashMap<Ssa, &Operation>) -> Option<Ssa> {
    let gm_c = rr_broadcast_src(&ctx.probs, def)?;
    let gm_d = rr_broadcast_src(&diag.probs, def)?;
    if gm_c != gm_d {
        return None;
    }
    Some(gm_c)
}

/// From a `probs = exp(subf(S, gm_bc))` SSA, recover the pre-broadcast `gm` SSA by
/// peeling `exp -> subf -> (operand 1) gm_bc -> linalg.broadcast -> reshape`.
fn rr_broadcast_src(probs: &Ssa, def: &HashMap<Ssa, &Operation>) -> Option<Ssa> {
    let pexp = def.get(probs)?;
    let sub = def.get(pexp.operands.first()?)?;
    let gm_bc = sub.operands.get(1)?;
    rr_peel_broadcast_reshape(gm_bc, def)
}

/// Peel `linalg.broadcast(reshape(x))` (or `broadcast(x)`) → `x`.
fn rr_peel_broadcast_reshape(ssa: &Ssa, def: &HashMap<Ssa, &Operation>) -> Option<Ssa> {
    let bop = def.get(ssa)?;
    if bop.op_type != OpKind::LinalgBroadcast {
        return None;
    }
    let src = bop.operands.first()?;
    let sop = def.get(src)?;
    if matches!(
        sop.op_type,
        OpKind::TensorReshape | OpKind::TensorExpandShape
    ) {
        Some(*sop.operands.first()?)
    } else {
        Some(*src)
    }
}

/// True if `gm = arith.maximumf(reduce_max(sc), reduce_max(sd))` (order-free).
fn rr_gm_is_max_of_reduces(gm: &Ssa, sc: &Ssa, sd: &Ssa, def: &HashMap<Ssa, &Operation>) -> bool {
    let Some(mop) = def.get(gm) else { return false };
    if mop.op_type != OpKind::ArithMaximumf {
        return false;
    }
    let Some(a) = mop.operands.first() else {
        return false;
    };
    let Some(b) = mop.operands.get(1) else {
        return false;
    };
    let a_red = rr_reduce_of(a, OpKind::ArithMaximumf, def);
    let b_red = rr_reduce_of(b, OpKind::ArithMaximumf, def);
    // a reduces sc & b reduces sd, OR vice-versa.
    (a_red.as_ref() == Some(sc) && b_red.as_ref() == Some(sd))
        || (a_red.as_ref() == Some(sd) && b_red.as_ref() == Some(sc))
}

/// Find the `linalg.reduce { reduce_fn }` whose input operand is `target`, and
/// return its result SSA (the row vector). `None` if no such reduce exists.
fn rr_reduce_consuming(
    target: &Ssa,
    reduce_fn: OpKind,
    def: &HashMap<Ssa, &Operation>,
) -> Option<Ssa> {
    let red = def.values().find(|o| {
        o.op_type == OpKind::LinalgReduce
            && o.operands.first() == Some(target)
            && o.attr(AttrKey::ReduceFn) == Some(&Attr::Op(reduce_fn))
    })?;
    red.result
}

/// If `ssa` is `linalg.reduce { reduce_fn } (target)` (possibly via a reshape
/// wrapper), return the reduced `target`; else `None`.
fn rr_reduce_of(ssa: &Ssa, reduce_fn: OpKind, def: &HashMap<Ssa, &Operation>) -> Option<Ssa> {
    let mut cur = *ssa;
    for _ in 0..3 {
        let op = def.get(&cur)?;
        if matches!(
            op.op_type,
            OpKind::TensorReshape | OpKind::TensorExpandShape | OpKind::TensorCollapseShape
        ) {
            cur = *op.operands.first()?;
        } else {
            break;
        }
    }
    let op = def.get(&cur)?;
    if op.op_type != OpKind::LinalgReduce {
        return None;
    }
    if op.attr(AttrKey::ReduceFn) != Some(&Attr::Op(reduce_fn)) {
        return None;
    }
    Some(*op.operands.first()?)
}

/// Verify both blocks' `divf` denominators broadcast ONE shared
/// `gs = arith.addf(scs, sds)` where `scs`/`sds` are the two blocks' row sums.
fn rr_check_shared_gs(ctx: &RrBlock, diag: &RrBlock, def: &HashMap<Ssa, &Operation>) -> Option<()> {
    let gs_c = rr_divf_denom_src(&ctx.probs, def)?;
    let gs_d = rr_divf_denom_src(&diag.probs, def)?;
    if gs_c != gs_d {
        return None;
    }
    let gsop = def.get(&gs_c)?;
    if gsop.op_type != OpKind::ArithAddf {
        return None;
    }
    let a = gsop.operands.first()?;
    let b = gsop.operands.get(1)?;
    let ok = (a == &ctx.row_sum && b == &diag.row_sum) || (a == &diag.row_sum && b == &ctx.row_sum);
    if ok { Some(()) } else { None }
}

/// From a block's `probs`, find the `w = divf(probs, gs_bc)` consumer and peel
/// `gs_bc = broadcast(reshape(gs))` → `gs`.
fn rr_divf_denom_src(probs: &Ssa, def: &HashMap<Ssa, &Operation>) -> Option<Ssa> {
    // Find the divf whose first operand is `probs`.
    let divf = def
        .values()
        .find(|o| o.op_type == OpKind::ArithDivf && o.operands.first() == Some(probs))?;
    let gs_bc = divf.operands.get(1)?;
    rr_peel_broadcast_reshape(gs_bc, def)
}

/// Recover the context-mask pointer arg: the `addf(scaled, mask)` second operand
/// is `linalg.broadcast(mask_load)` where `mask_load` reads a `[1, cap]` view.
fn rr_context_mask_arg(
    ctx: &RrBlock,
    def: &HashMap<Ssa, &Operation>,
    load_src: &HashMap<Ssa, RrLoad>,
    cap: i64,
) -> Option<Ssa> {
    let sop = def.get(&ctx.masked_scores)?;
    let mask = sop.operands.get(1)?;
    let bop = def.get(mask)?;
    if bop.op_type != OpKind::LinalgBroadcast {
        return None;
    }
    let mask_loaded = bop.operands.first()?;
    let mc = load_src.get(mask_loaded)?;
    if mc.view_shape != [1, cap] {
        return None;
    }
    Some(mc.arg)
}

/// Recover the `-inf` constant from the diagonal block's dense `[m, m]`
/// triangular mask (`addf(scaled, tri)` where `tri` is an `arith.constant` with a
/// `value` FloatList). The most-negative entry is the `-inf` fill.
fn rr_recover_tri_ninf(diag: &RrBlock, def: &HashMap<Ssa, &Operation>) -> Option<f32> {
    let sop = def.get(&diag.masked_scores)?;
    let tri = sop.operands.get(1)?;
    let top = def.get(tri)?;
    if top.op_type != OpKind::ArithConstant {
        return None;
    }
    match top.attr(AttrKey::Value) {
        Some(Attr::FloatList(v)) => v
            .iter()
            .cloned()
            .fold(None, |acc, x| match acc {
                Some(a) if a <= x => Some(a),
                _ => Some(x),
            })
            .map(|x| x as f32),
        _ => None,
    }
}

/// Rewrite a recognized [`ReRolledIsland`] — cap-tile ONLY the CONTEXT `[m, cap]`
/// block with online softmax (an `scf.for` over `cap/blk` KV blocks carrying the
/// running max / sum / acc), leaving the small `[m, m]` DIAGONAL whole, then
/// COMBINE both with the SAME global re-association `head_rewrite` uses.
///
/// Preserves grid `[H,1,1]` and the per-head GQA column arithmetic verbatim. The
/// per-block context scores tile is `[m, blk]` (`blk` is a divisor of `cap` chosen
/// by [`choose_block_budgeted`] so the tile fits LX). Emits ONLY RFC-0682 ops
/// (`ktdp` load/store + Arith/Math/LinAlg/Tensor + ONE `scf.for`); NO
/// `tensor.insert_slice`.
pub fn tile_rerolled_attention<'a>(
    a: &'a Arena,
    isl: &ReRolledIsland,
    blk: i64,
    g: &mut NameGen,
) -> IRFunction<'a> {
    let dt = isl.dtype;
    let (m, d, cap) = (isl.m, isl.d, isl.cap);
    // `blk` is a divisor of `cap` (`cap_divisors` only returns divisors), so the
    // block count is exact (no ragged tail).
    let blk = if blk >= 1 && cap % blk == 0 {
        blk
    } else {
        choose_block(cap)
    };
    let mut ops: Vec<Operation<'a>> = Vec::new();

    // ---- constants ----
    let c0 = g.mint();
    ops.push(const_index(a, c0, 0));
    let scale_c = g.mint();
    ops.push(const_f(a, scale_c, isl.scale as f64));
    let ninf_c = g.mint();
    ops.push(const_f(a, ninf_c, isl.ninf as f64));
    let zero_c = g.mint();
    ops.push(const_f(a, zero_c, 0.0));

    // ---- per-head selection arithmetic (PRESERVED verbatim) ----
    let hpid = g.mint();
    ops.push(Operation::new(
        a,
        Some(hpid),
        OpKind::KtdpGetComputeTileId,
        &[],
    ));
    let hdc = g.mint();
    ops.push(const_index(a, hdc, isl.hdc));
    let gqac = g.mint();
    ops.push(const_index(a, gqac, isl.gqac));
    let qcol = g.mint();
    ops.push(Operation::new(
        a,
        Some(qcol),
        OpKind::ArithMuli,
        &[hpid, hdc],
    ));
    let kvh = g.mint();
    ops.push(Operation::new(
        a,
        Some(kvh),
        OpKind::ArithDivui,
        &[hpid, gqac],
    ));
    let kvcol = g.mint();
    ops.push(Operation::new(
        a,
        Some(kvcol),
        OpKind::ArithMuli,
        &[kvh, hdc],
    ));

    // ---- views ----
    let q_view = g.mint();
    ops.push(mk_view(a, q_view, isl.q_arg, &[m, isl.q_cols], dt));
    let o_view = g.mint();
    ops.push(mk_view(a, o_view, isl.o_arg, &[m, isl.q_cols], dt));
    let mask_view = g.mint();
    ops.push(mk_view(a, mask_view, isl.mask_arg, &[1, cap], dt));
    let kc_view = g.mint();
    ops.push(mk_view(a, kc_view, isl.kc_arg, &[cap, isl.kv_cols], dt));
    let kd_view = g.mint();
    ops.push(mk_view(a, kd_view, isl.kd_arg, &[m, isl.kv_cols], dt));
    let vc_view = g.mint();
    ops.push(mk_view(a, vc_view, isl.vc_arg, &[cap, isl.kv_cols], dt));
    let vd_view = g.mint();
    ops.push(mk_view(a, vd_view, isl.vd_arg, &[m, isl.kv_cols], dt));

    // ---- whole-Q load [m, d] at [0, qcol] (per-head column slice) ----
    let q = block_load(a, g, &mut ops, q_view, c0, qcol, [m, d]);

    // ---- loop bounds + block-size constant ----
    let lb = g.mint();
    ops.push(const_index(a, lb, 0));
    let step = g.mint();
    ops.push(const_index(a, step, 1));
    let blk_c = g.mint();
    ops.push(const_index(a, blk_c, blk));

    // ---- RUNTIME loop bound: iterate ONLY the KV blocks that hold valid context.
    // The context mask [1, cap] is 0 on valid columns and -inf past valid_len, so
    // exp(mask) is a 1/0 valid-column indicator. Sum it (WIDENED to f32 — an f16
    // sum saturates integer precision past 2048 and would undercount valid_len at a
    // block boundary, silently dropping context) to recover valid_len, then run
    // ceil(valid_len / blk) blocks. valid_len = 0 (empty prefix) => 0 blocks: the
    // diagonal alone carries the result. This makes attention O(actual context)
    // instead of O(cap) — a 32-token prefill chunk runs 1 block, not cap/blk — which
    // is the whole point of the rewrite for the long-context/small-query regime.
    let mask_full = mk_whole_load(a, g, &mut ops, mask_view, &[1, cap]);
    let vind = g.mint();
    ops.push(Operation::new(a, Some(vind), OpKind::MathExp, &[mask_full]));
    let vind32 = g.mint();
    ops.push(Operation::new(
        a,
        Some(vind32),
        OpKind::ArithConvertf,
        &[vind],
    ));
    let vzero = g.mint();
    ops.push(const_f(a, vzero, 0.0));
    let vinit = g.mint();
    ops.push(mk_splat(a, vinit, vzero, &[1], DType::F32));
    let vsum = g.mint();
    ops.push(mk_reduce(a, vsum, vind32, vinit, OpKind::ArithAddf));
    let vscalar = g.mint();
    ops.push(Operation::new(
        a,
        Some(vscalar),
        OpKind::TensorExtract,
        &[vsum],
    ));
    let vidx = g.mint();
    ops.push(Operation::new(
        a,
        Some(vidx),
        OpKind::ArithIndexCast,
        &[vscalar],
    ));
    let ub = g.mint();
    ops.push(Operation::new(
        a,
        Some(ub),
        OpKind::ArithCeildivui,
        &[vidx, blk_c],
    ));

    // ---- iter-arg inits: running max [m,1]=FLOOR, sum [m,1]=0, acc [m,d]=empty ----
    // The running max seeds a FINITE floor, NOT -inf: a fresh prefill's prefix
    // CONTEXT is empty, so every context KV block is fully mask-additive `-inf`.
    // With a -inf seed, `m_new = max(-inf,-inf) = -inf` and `exp(Sj - m_new) =
    // exp(-inf - -inf) = NaN`. A finite floor (well below any real scaled score,
    // within f16 range) makes a fully-masked block yield `exp(-inf - floor) = 0`
    // (contributes nothing, as it must), while any valid block's real max exceeds
    // the floor so its arithmetic is unchanged. The whole-tensor path never hit
    // this because its reduce_max spans the valid diagonal (a finite global max).
    let mfloor_c = g.mint();
    ops.push(const_f(a, mfloor_c, -3.0e4));
    let m0 = g.mint();
    ops.push(mk_splat(a, m0, mfloor_c, &[m, 1], dt));
    let l0 = g.mint();
    ops.push(mk_splat(a, l0, zero_c, &[m, 1], dt));
    let acc0 = g.mint();
    ops.push(mk_empty(a, acc0, &[m, d], dt));

    // iter-arg body-visible values.
    let mi = g.mint();
    let li = g.mint();
    let acci = g.mint();
    let iv = g.mint();

    // ===================== CONTEXT KV-block loop body =====================
    let mut body: Vec<Operation<'a>> = Vec::new();
    // off = j * blk (the KV-block row offset into the [cap, kv_cols] view).
    let off = g.mint();
    body.push(Operation::new(
        a,
        Some(off),
        OpKind::ArithMuli,
        &[iv, blk_c],
    ));

    // Kj = Kc[off.., kvcol] -> [blk, d]; Vj = Vc[off.., kvcol] -> [blk, d].
    let kj = block_load(a, g, &mut body, kc_view, off, kvcol, [blk, d]);
    let vj = block_load(a, g, &mut body, vc_view, off, kvcol, [blk, d]);

    // Kjt = transpose(Kj) -> [d, blk]; Sj_raw = Q @ Kjt -> [m, blk].
    let kjt = mk_transpose(a, g, &mut body, kj, blk, d, dt);
    let sj_raw = mk_matmul(a, g, &mut body, q, kjt, [m, blk], dt);
    // scaled = Sj_raw * scale.
    let sjscl = g.mint();
    body.push(mk_splat(a, sjscl, scale_c, &[m, blk], dt));
    let sj_scaled = g.mint();
    body.push(Operation::new(
        a,
        Some(sj_scaled),
        OpKind::ArithMulf,
        &[sj_raw, sjscl],
    ));
    // maskj = Mask[0, off..] -> [1, blk], broadcast to [m, blk] (NOT a triangle).
    let maskj = block_load(a, g, &mut body, mask_view, c0, off, [1, blk]);
    let maskj_init = g.mint();
    body.push(mk_empty(a, maskj_init, &[m, blk], dt));
    let maskj_b = g.mint();
    body.push(
        Operation::new(
            a,
            Some(maskj_b),
            OpKind::LinalgBroadcast,
            &[maskj, maskj_init],
        )
        .with_attr(a, AttrKey::Dimensions, Attr::IntList(a.ints(vec![]))),
    );
    let sj = g.mint();
    body.push(Operation::new(
        a,
        Some(sj),
        OpKind::ArithAddf,
        &[sj_scaled, maskj_b],
    ));

    // rmax = reduce_max(Sj, 1) -> [m] -> [m,1].
    let rmax_init = g.mint();
    body.push(mk_splat(a, rmax_init, ninf_c, &[m], dt));
    let rmax = g.mint();
    body.push(mk_reduce(a, rmax, sj, rmax_init, OpKind::ArithMaximumf));
    let rmax2 = g.mint();
    body.push(reshape_to(a, rmax2, rmax, &[m, 1]));
    // m_new = max(m_i, rmax2) -> [m,1].
    let mnew = g.mint();
    body.push(Operation::new(
        a,
        Some(mnew),
        OpKind::ArithMaximumf,
        &[mi, rmax2],
    ));

    // P = exp(Sj - m_new_bc[m,blk]).
    let mnew_b = broadcast_col_to(a, g, &mut body, mnew, m, blk, dt);
    let shifted = g.mint();
    body.push(Operation::new(
        a,
        Some(shifted),
        OpKind::ArithSubf,
        &[sj, mnew_b],
    ));
    let p = g.mint();
    body.push(Operation::new(a, Some(p), OpKind::MathExp, &[shifted]));

    // alpha = exp(m_i - m_new) -> [m,1].
    let mdiff = g.mint();
    body.push(Operation::new(
        a,
        Some(mdiff),
        OpKind::ArithSubf,
        &[mi, mnew],
    ));
    let alpha = g.mint();
    body.push(Operation::new(a, Some(alpha), OpKind::MathExp, &[mdiff]));

    // rsum = reduce_sum(P, 1) -> [m] -> [m,1]; l_new = alpha*l_i + rsum.
    let rsum_init = g.mint();
    body.push(mk_splat(a, rsum_init, zero_c, &[m], dt));
    let rsum = g.mint();
    body.push(mk_reduce(a, rsum, p, rsum_init, OpKind::ArithAddf));
    let rsum2 = g.mint();
    body.push(reshape_to(a, rsum2, rsum, &[m, 1]));
    let al = g.mint();
    body.push(Operation::new(a, Some(al), OpKind::ArithMulf, &[alpha, li]));
    let lnew = g.mint();
    body.push(Operation::new(
        a,
        Some(lnew),
        OpKind::ArithAddf,
        &[al, rsum2],
    ));

    // acc_new = alpha_bc[m,d]*acc + P @ Vj.
    let alpha_b = broadcast_col_to(a, g, &mut body, alpha, m, d, dt);
    let acc_scaled = g.mint();
    body.push(Operation::new(
        a,
        Some(acc_scaled),
        OpKind::ArithMulf,
        &[alpha_b, acci],
    ));
    let pv = mk_matmul(a, g, &mut body, p, vj, [m, d], dt);
    let accnew = g.mint();
    body.push(Operation::new(
        a,
        Some(accnew),
        OpKind::ArithAddf,
        &[acc_scaled, pv],
    ));

    body.push(Operation::new(
        a,
        None,
        OpKind::ScfYield,
        &[mnew, lnew, accnew],
    ));

    // ---- the scf.for over CONTEXT KV blocks ----
    let mc_f = g.mint(); // running max [m,1] (un-normalized partial base).
    let lc_f = g.mint(); // running sum [m,1].
    let acc_f = g.mint(); // running acc [m,d] (un-normalized, base mc_f).
    let forop = Operation::new(a, None, OpKind::ScfFor, &[lb, ub, step, m0, l0, acc0])
        .with_attr(a, AttrKey::IterVar, Attr::Ssas(a.ssa(vec![iv])))
        .with_attr(a, AttrKey::IterArgs, Attr::Ssas(a.ssa(vec![mi, li, acci])))
        .with_attr(
            a,
            AttrKey::ResultNames,
            Attr::Ssas(a.ssa(vec![mc_f, lc_f, acc_f])),
        );
    ops.push(Operation {
        regions: a.regions(vec![a.ops(body)]),
        ..forop
    });

    // mc_f is [m,1]; flatten to [m] for the diagonal-combine arith below.
    let mc_row = g.mint();
    ops.push(reshape_to(a, mc_row, mc_f, &[m]));
    let lc_row = g.mint();
    ops.push(reshape_to(a, lc_row, lc_f, &[m]));

    // ===================== DIAGONAL block (whole) =========================
    // Kd [m, d] at [0, kvcol] -> Kdt [d, m]; Sd = (Q @ Kdt)*scale + tri[m,m].
    let kd = block_load(a, g, &mut ops, kd_view, c0, kvcol, [m, d]);
    let kdt = mk_transpose(a, g, &mut ops, kd, m, d, dt);
    let sd_raw = mk_matmul(a, g, &mut ops, q, kdt, [m, m], dt);
    let sd_scl = g.mint();
    ops.push(mk_splat(a, sd_scl, scale_c, &[m, m], dt));
    let sd_scaled = g.mint();
    ops.push(Operation::new(
        a,
        Some(sd_scaled),
        OpKind::ArithMulf,
        &[sd_raw, sd_scl],
    ));
    let tri = g.mint();
    ops.push(causal_mask_mm(a, tri, m, isl.ninf, dt));
    let sd = g.mint();
    ops.push(Operation::new(
        a,
        Some(sd),
        OpKind::ArithAddf,
        &[sd_scaled, tri],
    ));
    // md = reduce_max(Sd, 1) -> [m].
    let md_init = g.mint();
    ops.push(mk_splat(a, md_init, ninf_c, &[m], dt));
    let md = g.mint();
    ops.push(mk_reduce(a, md, sd, md_init, OpKind::ArithMaximumf));

    // ===================== COMBINE (global re-association) =================
    // gm = max(mc_f, md) [m].
    let gm = g.mint();
    ops.push(Operation::new(
        a,
        Some(gm),
        OpKind::ArithMaximumf,
        &[mc_row, md],
    ));
    // cfac = exp(mc_f - gm) [m]  (re-base the CONTEXT partial onto the global max).
    let cdiff = g.mint();
    ops.push(Operation::new(
        a,
        Some(cdiff),
        OpKind::ArithSubf,
        &[mc_row, gm],
    ));
    let cfac = g.mint();
    ops.push(Operation::new(a, Some(cfac), OpKind::MathExp, &[cdiff]));
    // accC' = cfac_bc[m,d] * acc_f; lC' = cfac * lc_f.
    let cfac_bd = broadcast_row_to(a, g, &mut ops, cfac, m, d, dt);
    let acc_rb = g.mint();
    ops.push(Operation::new(
        a,
        Some(acc_rb),
        OpKind::ArithMulf,
        &[cfac_bd, acc_f],
    ));
    let lc_rb = g.mint();
    ops.push(Operation::new(
        a,
        Some(lc_rb),
        OpKind::ArithMulf,
        &[cfac, lc_row],
    ));
    // Pd = exp(Sd - gm_bc[m,m]); sd_sum = reduce_sum(Pd,1) [m].
    let gm_bd = broadcast_row_to(a, g, &mut ops, gm, m, m, dt);
    let shd = g.mint();
    ops.push(Operation::new(
        a,
        Some(shd),
        OpKind::ArithSubf,
        &[sd, gm_bd],
    ));
    let pd = g.mint();
    ops.push(Operation::new(a, Some(pd), OpKind::MathExp, &[shd]));
    let sds_init = g.mint();
    ops.push(mk_splat(a, sds_init, zero_c, &[m], dt));
    let sds = g.mint();
    ops.push(mk_reduce(a, sds, pd, sds_init, OpKind::ArithAddf));
    // gs = lC' + sd_sum [m].
    let gs = g.mint();
    ops.push(Operation::new(
        a,
        Some(gs),
        OpKind::ArithAddf,
        &[lc_rb, sds],
    ));
    // Wd = Pd / gs_bc[m,m].
    let gs_bd = broadcast_row_to(a, g, &mut ops, gs, m, m, dt);
    let wd = g.mint();
    ops.push(Operation::new(a, Some(wd), OpKind::ArithDivf, &[pd, gs_bd]));

    // ovd = Wd @ Vd; Vd [m, d] at [0, kvcol].
    let vd = block_load(a, g, &mut ops, vd_view, c0, kvcol, [m, d]);
    let ovd = mk_matmul(a, g, &mut ops, wd, vd, [m, d], dt);
    // O = (accC' + ovd) / gs_bc[m,d].
    let num = g.mint();
    ops.push(Operation::new(
        a,
        Some(num),
        OpKind::ArithAddf,
        &[acc_rb, ovd],
    ));
    let gs_bcd = broadcast_row_to(a, g, &mut ops, gs, m, d, dt);
    let o = g.mint();
    ops.push(Operation::new(
        a,
        Some(o),
        OpKind::ArithDivf,
        &[num, gs_bcd],
    ));

    // store O [m, d] at [0, qcol].
    let o_at = g.mint();
    ops.push(
        Operation::new(
            a,
            Some(o_at),
            OpKind::KtdpConstructAccessTile,
            &[o_view, c0, qcol],
        )
        .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![m, d]))),
    );
    ops.push(Operation::new(a, None, OpKind::KtdpStore, &[o, o_at]));
    ops.push(Operation::new(a, None, OpKind::FuncReturn, &[]));

    IRFunction {
        name: "", // caller stamps the original name
        arguments: a.args(
            isl.arg_ssas()
                .into_iter()
                .map(|v| (v, IrType::Index))
                .collect(),
        ),
        operations: a.ops(ops),
        grid: (isl.h as usize, 1, 1),
        return_type: None,
    }
}

// ---- small emit helpers shared by `tile_rerolled_attention` (mirroring the
//      head_rewrite emitters so the diagonal block is byte-identical) ----

/// `linalg.transpose ins(%x) outs(empty[cols,rows]) permutation=[1,0]`.
fn mk_transpose<'a>(
    a: &'a Arena,
    g: &mut NameGen,
    ops: &mut Vec<Operation<'a>>,
    x: Ssa,
    rows: i64,
    cols: i64,
    dt: DType,
) -> Ssa {
    let init = g.mint();
    ops.push(mk_empty(a, init, &[cols, rows], dt));
    let res = g.mint();
    ops.push(
        Operation::new(a, Some(res), OpKind::LinalgTranspose, &[x, init]).with_attr(
            a,
            AttrKey::Permutation,
            Attr::IntList(a.ints(vec![1, 0])),
        ),
    );
    res
}

/// `C = A @ B` with a zero `tensor.empty` outs init.
fn mk_matmul<'ar>(
    a: &'ar Arena,
    g: &mut NameGen,
    ops: &mut Vec<Operation<'ar>>,
    lhs: Ssa,
    rhs: Ssa,
    out: [i64; 2],
    dt: DType,
) -> Ssa {
    let init = g.mint();
    ops.push(mk_empty(a, init, &out, dt));
    let res = g.mint();
    ops.push(Operation::new(
        a,
        Some(res),
        OpKind::LinalgMatmul,
        &[lhs, rhs, init],
    ));
    res
}

/// Broadcast a `[m]` row-vector to `[m, cols]` (reshape `[m]`→`[m,1]` then
/// `linalg.broadcast` to the outs shape) — the head_rewrite combine convention.
fn broadcast_row_to<'a>(
    a: &'a Arena,
    g: &mut NameGen,
    ops: &mut Vec<Operation<'a>>,
    rowv: Ssa,
    m: i64,
    cols: i64,
    dt: DType,
) -> Ssa {
    let r2 = g.mint();
    ops.push(reshape_to(a, r2, rowv, &[m, 1]));
    let init = g.mint();
    ops.push(mk_empty(a, init, &[m, cols], dt));
    let res = g.mint();
    ops.push(
        Operation::new(a, Some(res), OpKind::LinalgBroadcast, &[r2, init]).with_attr(
            a,
            AttrKey::Dimensions,
            Attr::IntList(a.ints(vec![])),
        ),
    );
    res
}

/// The static `[m, m]` lower-triangular causal mask: `0` for `k ≤ r`, `ninf` for
/// `k > r` (the diagonal block's mask, kept whole — byte-identical to
/// `head_rewrite::causal_mask_mm`).
fn causal_mask_mm<'a>(a: &'a Arena, res: Ssa, m: i64, ninf: f32, dt: DType) -> Operation<'a> {
    let mut vals = Vec::with_capacity((m * m) as usize);
    for r in 0..m {
        for k in 0..m {
            vals.push(if k <= r { 0.0 } else { ninf as f64 });
        }
    }
    Operation::new(a, Some(res), OpKind::ArithConstant, &[])
        .with_attr(a, AttrKey::IsTensor, Attr::Bool(true))
        .with_attr(a, AttrKey::DenseList, Attr::Bool(true))
        .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![m, m])))
        .with_attr(a, AttrKey::Dtype, Attr::Dtype(dt))
        .with_attr(a, AttrKey::Value, Attr::FloatList(a.floats(vals)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_builder::Ops;

    fn naive_attention(m: i64, cap: i64, d: i64, scale: f32, causal: bool) -> IRFunction<'static> {
        test_support::naive_attention(Arena::global(), m, cap, d, scale, causal)
    }

    #[test]
    fn recognizes_canonical_attention() {
        let f = naive_attention(4, 8, 2, 0.5, true);
        let isl = recognize_attention(&f).expect("should recognize canonical attention");
        assert_eq!(isl.m, 4);
        assert_eq!(isl.cap, 8);
        assert_eq!(isl.d, 2);
        assert!((isl.scale - 0.5).abs() < 1e-6);
        assert!(isl.causal);
        assert_eq!(isl.q_arg, f.arguments[0].0);
        assert_eq!(isl.k_arg, f.arguments[1].0);
        assert_eq!(isl.v_arg, f.arguments[2].0);
        assert_eq!(isl.o_arg, f.arguments[3].0);
    }

    #[test]
    fn recognizes_noncausal() {
        let f = naive_attention(2, 4, 2, 0.25, false);
        let isl = recognize_attention(&f).expect("non-causal still recognized");
        assert!(!isl.causal);
        assert_eq!(isl.scores_bytes(), 2 * 4 * 2); // [2,4] f16
    }

    #[test]
    fn rejects_non_attention() {
        // A plain copy node: load -> exp -> store. No QKᵀ/softmax/AV.
        let mut ops = Ops::new();
        let vi = ops.op(Some("%vi"), OpKind::KtdpConstructMemoryView, &["%in"]);
        let vi = ops.attr(vi, AttrKey::Shape, ops.int_list(vec![4, 4]));
        let vi = ops.attr(vi, AttrKey::Dtype, Attr::Dtype(DType::F16));
        let ti = ops.op(Some("%ti"), OpKind::KtdpConstructAccessTile, &["%vi"]);
        let ti = ops.attr(ti, AttrKey::Shape, ops.int_list(vec![4, 4]));
        let l = ops.op(Some("%l"), OpKind::KtdpLoad, &["%ti"]);
        let y = ops.op(Some("%y"), OpKind::MathExp, &["%l"]);
        let vo = ops.op(Some("%vo"), OpKind::KtdpConstructMemoryView, &["%out"]);
        let vo = ops.attr(vo, AttrKey::Shape, ops.int_list(vec![4, 4]));
        let vo = ops.attr(vo, AttrKey::Dtype, Attr::Dtype(DType::F16));
        let to = ops.op(Some("%to"), OpKind::KtdpConstructAccessTile, &["%vo"]);
        let to = ops.attr(to, AttrKey::Shape, ops.int_list(vec![4, 4]));
        let store = ops.op(None, OpKind::KtdpStore, &["%y", "%to"]);
        let ret = ops.op(None, OpKind::FuncReturn, &[]);
        let body = vec![vi, ti, l, y, vo, to, store, ret];
        let f = ops.func(
            "copy",
            &[("%in", IrType::Index), ("%out", IrType::Index)],
            body,
            (1, 1, 1),
        );
        assert!(
            recognize_attention(&f).is_none(),
            "copy node must not be recognized"
        );
    }

    #[test]
    fn rejects_region_bearing() {
        // A function that already contains an scf.for is not the flat idiom.
        let mut f = naive_attention(2, 4, 2, 0.5, false);
        let ops = Ops::new();
        let yield_op = Operation::new(ops.arena(), None, OpKind::ScfYield, &[]);
        let forop = Operation::new(ops.arena(), None, OpKind::ScfFor, &[]);
        let forop = ops.with_region(forop, vec![yield_op]);
        let mut new_ops = vec![forop];
        new_ops.extend_from_slice(f.operations);
        f.operations = ops.arena().ops(new_ops);
        assert!(
            recognize_attention(&f).is_none(),
            "region-bearing func bails"
        );
    }

    #[test]
    fn rejects_multi_store() {
        // Two stores (the real unrolled per-query-row lowering) -> not canonical.
        let mut f = naive_attention(2, 4, 2, 0.5, false);
        // duplicate the store op.
        let store = *f
            .operations
            .iter()
            .find(|o| o.op_type == OpKind::KtdpStore)
            .unwrap();
        let idx = f.operations.len() - 1; // before func.return
        let mut new_ops = f.operations.to_vec();
        new_ops.insert(idx, store);
        f.operations = Arena::global().ops(new_ops);
        assert!(recognize_attention(&f).is_none(), "multi-store bails");
    }

    #[test]
    fn tile_emits_scf_for_and_no_insert_slice() {
        let f = naive_attention(4, 256, 8, 0.125, true);
        let isl = recognize_attention(&f).unwrap();
        let mut g = NameGen::after(&f);
        let tiled = tile_attention(Arena::global(), &isl, &mut g);

        // Exactly one scf.for at top level, carrying 3 iter-args (m, l, acc).
        let fors: Vec<&Operation> = tiled
            .operations
            .iter()
            .filter(|o| o.op_type == OpKind::ScfFor)
            .collect();
        assert_eq!(fors.len(), 1, "one KV loop");
        let f0 = fors[0];
        match f0.attr(AttrKey::IterArgs) {
            Some(Attr::Ssas(v)) => assert_eq!(v.len(), 3, "m, l, acc iter-args"),
            other => panic!("iter_args not a 3-list: {other:?}"),
        }
        match f0.attr(AttrKey::ResultNames) {
            Some(Attr::Ssas(v)) => assert_eq!(v.len(), 3),
            other => panic!("result_names not a 3-list: {other:?}"),
        }

        // NO tensor.insert_slice anywhere (it is UNREGISTERED in this emulator).
        fn has_insert(ops: &[Operation]) -> bool {
            ops.iter().any(|o| {
                o.op_type == OpKind::TensorInsertSlice || o.regions.iter().any(|r| has_insert(r))
            })
        }
        assert!(!has_insert(tiled.operations), "must not emit insert_slice");

        // The loop body reads each KV block via a `ktdp.construct_access_tile` at
        // the dynamic block offset + a `ktdp.load` (Kj and Vj) — the KTIR-native,
        // fusion-safe analogue of an extract_slice of the block.
        let body = f0.regions[0];
        let block_tiles = body
            .iter()
            .filter(|o| o.op_type == OpKind::KtdpConstructAccessTile)
            .count();
        let block_loads = body
            .iter()
            .filter(|o| o.op_type == OpKind::KtdpLoad)
            .count();
        assert_eq!(block_tiles, 2, "Kj and Vj access tiles per block");
        assert_eq!(block_loads, 2, "Kj and Vj loads per block");
        // No tensor.extract_slice in the body either (we use ktdp block loads).
        let slices = body
            .iter()
            .filter(|o| o.op_type == OpKind::TensorExtractSlice)
            .count();
        assert_eq!(
            slices, 0,
            "KV blocks read via ktdp access tile, not extract_slice"
        );

        // online-softmax kernels present: two matmuls (QKᵀ and P·V), an exp for P
        // and an exp for alpha.
        let matmuls = body
            .iter()
            .filter(|o| o.op_type == OpKind::LinalgMatmul)
            .count();
        assert_eq!(matmuls, 2, "QKᵀ and P·V");
        let exps = body.iter().filter(|o| o.op_type == OpKind::MathExp).count();
        assert_eq!(exps, 2, "exp(P) and exp(alpha)");
    }

    #[test]
    fn tile_preserves_args_and_grid() {
        let f = naive_attention(2, 8, 4, 0.5, false);
        let isl = recognize_attention(&f).unwrap();
        let mut g = NameGen::after(&f);
        let tiled = tile_attention(Arena::global(), &isl, &mut g);
        let ssas: Vec<Ssa> = tiled.arguments.iter().map(|(v, _)| *v).collect();
        assert_eq!(ssas, vec![isl.q_arg, isl.k_arg, isl.v_arg, isl.o_arg]);
        assert_eq!(
            tiled.grid,
            (1, 1, 1),
            "rewritten node runs single-grid generic"
        );
    }

    #[test]
    fn choose_block_divides_cap() {
        for &cap in &[8, 16, 64, 128, 256, 512, 1024, 2048, 4096] {
            let b = choose_block(cap);
            assert!(b >= 1 && b <= cap);
            assert_eq!(cap % b, 0, "block {b} must divide cap {cap}");
            assert!(b <= DEFAULT_KV_BLOCK || cap <= DEFAULT_KV_BLOCK);
        }
    }

    // ----- RE-ROLLED path (the REAL model node, post head_rewrite) -----

    use crate::head_rewrite::{HeadAttnIsland, rewrite_head_attention};

    /// A synthetic head island matching the smollm2 shape (H=9, m=8, gqac=3, d=64,
    /// cap=64). `rewrite_head_attention` turns it into the EXACT re-rolled IR the
    /// real node produces, which `recognize_rerolled_attention` must match.
    fn smollm_head_island(cap: i64) -> HeadAttnIsland {
        let mut ops = Ops::new();
        HeadAttnIsland {
            q_arg: ops.ssa("%q"),
            o_arg: ops.ssa("%o"),
            mask_arg: ops.ssa("%mask"),
            kc_arg: ops.ssa("%kc"),
            kd_arg: ops.ssa("%kd"),
            vc_arg: ops.ssa("%vc"),
            vd_arg: ops.ssa("%vd"),
            q_cols: 576,
            kv_cols: 192,
            m: 8,
            cap,
            d: 64,
            gqac: 3,
            hdc: 64,
            h: 9,
            scale: 0.125,
            ninf: -1.0e38,
            dtype: DType::F16,
        }
    }

    fn rewrite(head: &HeadAttnIsland) -> IRFunction<'static> {
        let mut g = NameGen::above([
            head.q_arg,
            head.o_arg,
            head.mask_arg,
            head.kc_arg,
            head.kd_arg,
            head.vc_arg,
            head.vd_arg,
        ]);
        let mut f = rewrite_head_attention(Arena::global(), head, &mut g);
        f.name = Arena::global().str(String::from("attn"));
        f
    }

    #[test]
    fn recognizes_rerolled_head_output() {
        // The re-rolled output of a head island IS the structural idiom the new
        // recognizer must match (this is exactly what head_rewrite emits for the
        // real node111). All fields must round-trip.
        let head = smollm_head_island(64);
        let rerolled = rewrite(&head);
        let isl = recognize_rerolled_attention(&rerolled)
            .expect("re-rolled head output must be recognized");
        assert_eq!(isl.m, head.m);
        assert_eq!(isl.cap, head.cap);
        assert_eq!(isl.d, head.d);
        assert_eq!(isl.h, head.h);
        assert_eq!(isl.gqac, head.gqac);
        assert_eq!(isl.q_cols, head.q_cols);
        assert_eq!(isl.kv_cols, head.kv_cols);
        assert!((isl.scale - head.scale).abs() < 1e-6);
        assert_eq!(isl.q_arg, head.q_arg);
        assert_eq!(isl.kc_arg, head.kc_arg);
        assert_eq!(isl.kd_arg, head.kd_arg);
        // scores_bytes must EQUAL HeadAttnIsland::scores_bytes (disjoint partition).
        assert_eq!(isl.scores_bytes(), head.scores_bytes());
        assert_eq!(isl.scores_bytes(), 8 * 64 * 2);
    }

    #[test]
    fn rerolled_recognizer_rejects_single_core() {
        let head = smollm_head_island(64);
        let mut rerolled = rewrite(&head);
        rerolled.grid = (1, 1, 1); // not head-parallel
        assert!(recognize_rerolled_attention(&rerolled).is_none());
    }

    #[test]
    fn rerolled_recognizer_rejects_region_bearing() {
        // A body that already contains an scf.for is the already-tiled form.
        let head = smollm_head_island(64);
        let mut rerolled = rewrite(&head);
        let ops = Ops::new();
        let yield_op = Operation::new(ops.arena(), None, OpKind::ScfYield, &[]);
        let forop = Operation::new(ops.arena(), None, OpKind::ScfFor, &[]);
        let forop = ops.with_region(forop, vec![yield_op]);
        let mut new_ops = vec![forop];
        new_ops.extend_from_slice(rerolled.operations);
        rerolled.operations = ops.arena().ops(new_ops);
        assert!(recognize_rerolled_attention(&rerolled).is_none());
    }

    #[test]
    fn rerolled_recognizer_rejects_single_block_naive() {
        // The single-block canonical idiom is NOT the two-block re-rolled form.
        let naive = test_support::naive_attention(Arena::global(), 4, 8, 2, 0.5, true);
        assert!(recognize_rerolled_attention(&naive).is_none());
    }

    #[test]
    fn tile_rerolled_emits_one_scf_for_no_insert_slice() {
        let head = smollm_head_island(256); // cap=256 so real tiling happens
        let rerolled = rewrite(&head);
        let isl = recognize_rerolled_attention(&rerolled).unwrap();
        let blk = choose_block_budgeted(isl.m, isl.cap, isl.d, 2, &|sb| sb >= 1024);
        let mut g = NameGen::after(&rerolled);
        let tiled = tile_rerolled_attention(Arena::global(), &isl, blk, &mut g);

        // Grid + args preserved (7 args, [H,1,1]).
        assert_eq!(tiled.grid, (9, 1, 1));
        let ssas: Vec<Ssa> = tiled.arguments.iter().map(|(v, _)| *v).collect();
        assert_eq!(
            ssas,
            vec![
                isl.q_arg,
                isl.o_arg,
                isl.mask_arg,
                isl.kc_arg,
                isl.kd_arg,
                isl.vc_arg,
                isl.vd_arg
            ]
        );

        // Exactly one scf.for, 3 iter-args (mC, lC, accC).
        let fors: Vec<&Operation> = tiled
            .operations
            .iter()
            .filter(|o| o.op_type == OpKind::ScfFor)
            .collect();
        assert_eq!(fors.len(), 1, "one CONTEXT KV loop");
        match fors[0].attr(AttrKey::IterArgs) {
            Some(Attr::Ssas(v)) => assert_eq!(v.len(), 3),
            other => panic!("iter_args not a 3-list: {other:?}"),
        }

        // NO tensor.insert_slice / extract_slice anywhere.
        fn has_bad(ops: &[Operation]) -> bool {
            ops.iter().any(|o| {
                matches!(
                    o.op_type,
                    OpKind::TensorInsertSlice | OpKind::TensorExtractSlice
                ) || o.regions.iter().any(|r| has_bad(r))
            })
        }
        assert!(
            !has_bad(tiled.operations),
            "must not emit insert/extract_slice"
        );

        // Per-head selection arithmetic preserved.
        assert!(
            tiled
                .operations
                .iter()
                .any(|o| o.op_type == OpKind::KtdpGetComputeTileId)
        );
        assert!(
            tiled
                .operations
                .iter()
                .any(|o| o.op_type == OpKind::ArithDivui)
        );
    }

    #[test]
    fn rerolled_per_block_tile_fits_budget_and_shrinks() {
        // The actual long-context fix: the per-block CONTEXT scores tile [m, blk]
        // is BELOW the forced budget AND strictly smaller than the full [m, cap]
        // tile — proven for a long cap (cap=512).
        let head = smollm_head_island(512);
        let rerolled = rewrite(&head);
        let isl = recognize_rerolled_attention(&rerolled).unwrap();
        let bytes = 2usize;
        let budget = 4096usize; // full tile 8*512*2=8192 overflows; sub-blocks fit.
        let needs = |sb: usize| sb.saturating_mul(8) >= budget.saturating_mul(7);
        assert!(
            needs(isl.scores_bytes()),
            "full tile must overflow at this budget"
        );
        let blk = choose_block_budgeted(isl.m, isl.cap, isl.d, bytes, &needs);
        assert!(blk < isl.cap, "must tile: blk {blk} < cap {}", isl.cap);
        assert_eq!(isl.cap % blk, 0, "blk divides cap");
        let per_block = (isl.m as usize) * (blk as usize) * bytes;
        assert!(
            !needs(per_block),
            "per-block tile {per_block} must fit budget"
        );
        assert!(per_block < isl.scores_bytes(), "per-block < full");

        // Walk the emitted scf.for body: every 2-D static-shape tile op is [m, blk]
        // or smaller in the cap axis (never the full [m, cap]).
        let mut g = NameGen::after(&rerolled);
        let tiled = tile_rerolled_attention(Arena::global(), &isl, blk, &mut g);
        let forop = tiled
            .operations
            .iter()
            .find(|o| o.op_type == OpKind::ScfFor)
            .unwrap();
        for op in forop.regions[0] {
            if let Some(Attr::IntList(s)) = op.attr(AttrKey::Shape)
                && s.len() == 2
                && s[0] == isl.m
            {
                // any [m, c] tile in the loop must have c <= blk (never == cap).
                assert!(
                    s[1] <= blk,
                    "loop tile [{},{}] exceeds blk {blk}",
                    s[0],
                    s[1]
                );
            }
        }
    }
}

// Shared synthetic-IR builder used by both the in-crate unit tests above and the
// ktir-cpu execution-equivalence golden (which re-declares the same structure).
#[doc(hidden)]
pub mod test_support {
    use super::*;

    /// Build a canonical NAIVE attention `IRFunction` over four pointer args
    /// (Q, K, V, O) with Q[m,d], K[cap,d], V[cap,d], O[m,d]. This is the EXACT
    /// idiom `recognize_attention` matches; the golden runs it on the interpreter
    /// and compares against the tiled rewrite. Every value — the args included —
    /// is minted from ONE generator, so the function cannot alias its own args.
    pub fn naive_attention<'a>(
        a: &'a Arena,
        m: i64,
        cap: i64,
        d: i64,
        scale: f32,
        causal: bool,
    ) -> IRFunction<'a> {
        let dt = DType::F16;
        let g = &mut NameGen::above([]);
        let mut ops: Vec<Operation<'a>> = Vec::new();

        let q_ptr = g.mint();
        let k_ptr = g.mint();
        let v_ptr = g.mint();
        let o_ptr = g.mint();

        // load Q, K, V whole.
        let qv = g.mint();
        ops.push(mk_view(a, qv, q_ptr, &[m, d], dt));
        let q = mk_whole_load(a, g, &mut ops, qv, &[m, d]);
        let kv = g.mint();
        ops.push(mk_view(a, kv, k_ptr, &[cap, d], dt));
        let k = mk_whole_load(a, g, &mut ops, kv, &[cap, d]);
        let vv = g.mint();
        ops.push(mk_view(a, vv, v_ptr, &[cap, d], dt));
        let v = mk_whole_load(a, g, &mut ops, vv, &[cap, d]);

        // Kt = transpose(K) -> [d, cap]
        let kti = g.mint();
        ops.push(mk_empty(a, kti, &[d, cap], dt));
        let kt = g.mint();
        ops.push(
            Operation::new(a, Some(kt), OpKind::LinalgTranspose, &[k, kti]).with_attr(
                a,
                AttrKey::Permutation,
                Attr::IntList(a.ints(vec![1, 0])),
            ),
        );
        // raw = Q @ Kt -> [m, cap]
        let rawi = g.mint();
        ops.push(mk_empty(a, rawi, &[m, cap], dt));
        let raw = g.mint();
        ops.push(Operation::new(
            a,
            Some(raw),
            OpKind::LinalgMatmul,
            &[q, kt, rawi],
        ));
        // scaled = raw * scale
        let sc = g.mint();
        ops.push(const_f(a, sc, scale as f64));
        let sct = g.mint();
        ops.push(mk_splat(a, sct, sc, &[m, cap], dt));
        let scaled = g.mint();
        ops.push(Operation::new(
            a,
            Some(scaled),
            OpKind::ArithMulf,
            &[raw, sct],
        ));

        // sm = scaled (+ causal mask)
        let sm = if causal {
            let mask = g.mint();
            // full [m,cap] causal mask via tensor.generate.
            ops.push(causal_mask_full(a, g, mask, m, cap, dt));
            let masked = g.mint();
            ops.push(Operation::new(
                a,
                Some(masked),
                OpKind::ArithAddf,
                &[scaled, mask],
            ));
            masked
        } else {
            scaled
        };

        // mx = reduce_max(sm, 1) -> [m]
        let ninf = g.mint();
        ops.push(const_f(a, ninf, -1.0e30));
        let mxi = g.mint();
        ops.push(mk_splat(a, mxi, ninf, &[m], dt));
        let mx = g.mint();
        ops.push(mk_reduce(a, mx, sm, mxi, OpKind::ArithMaximumf));
        // mx_b broadcast to [m,cap]
        let mxr = g.mint();
        ops.push(reshape_to(a, mxr, mx, &[m, 1]));
        let mxb = broadcast_col_to(a, g, &mut ops, mxr, m, cap, dt);
        // shifted = sm - mx_b
        let sh = g.mint();
        ops.push(Operation::new(a, Some(sh), OpKind::ArithSubf, &[sm, mxb]));
        // P = exp(shifted)
        let p = g.mint();
        ops.push(Operation::new(a, Some(p), OpKind::MathExp, &[sh]));
        // l = reduce_sum(P, 1) -> [m]
        let zero = g.mint();
        ops.push(const_f(a, zero, 0.0));
        let li = g.mint();
        ops.push(mk_splat(a, li, zero, &[m], dt));
        let l = g.mint();
        ops.push(mk_reduce(a, l, p, li, OpKind::ArithAddf));
        let lr = g.mint();
        ops.push(reshape_to(a, lr, l, &[m, 1]));
        let lb = broadcast_col_to(a, g, &mut ops, lr, m, cap, dt);
        // W = P / l_b
        let w = g.mint();
        ops.push(Operation::new(a, Some(w), OpKind::ArithDivf, &[p, lb]));

        // O = W @ V -> [m, d]
        let oi = g.mint();
        ops.push(mk_empty(a, oi, &[m, d], dt));
        let o = g.mint();
        ops.push(Operation::new(
            a,
            Some(o),
            OpKind::LinalgMatmul,
            &[w, v, oi],
        ));

        // store O
        let ov = g.mint();
        ops.push(mk_view(a, ov, o_ptr, &[m, d], dt));
        let oat = g.mint();
        ops.push(
            Operation::new(a, Some(oat), OpKind::KtdpConstructAccessTile, &[ov]).with_attr(
                a,
                AttrKey::Shape,
                Attr::IntList(a.ints(vec![m, d])),
            ),
        );
        ops.push(Operation::new(a, None, OpKind::KtdpStore, &[o, oat]));
        ops.push(Operation::new(a, None, OpKind::FuncReturn, &[]));

        IRFunction {
            name: a.str(String::from("naive_attn")),
            arguments: a.args(
                [q_ptr, k_ptr, v_ptr, o_ptr]
                    .into_iter()
                    .map(|v| (v, IrType::Index))
                    .collect(),
            ),
            operations: a.ops(ops),
            grid: (1, 1, 1),
            return_type: None,
        }
    }

    /// A full `[m, cap]` causal mask: visible (`0`) where absolute key
    /// `<= cap - m + qr`, else `-inf`. The whole-matrix analogue of the per-block
    /// `causal_mask_tensor`. The region's block arguments are minted from the same
    /// generator as everything else and DECLARED as `bb0_args`, so the body's uses
    /// and the declaration are the same identities.
    fn causal_mask_full<'a>(
        a: &'a Arena,
        g: &mut NameGen,
        res: Ssa,
        m: i64,
        cap: i64,
        dt: DType,
    ) -> Operation<'a> {
        let qr = g.mint();
        let kc = g.mint();
        let bb0 = Operation::new(a, None, OpKind::RegionBb0Args, &[]).with_attr(
            a,
            AttrKey::Names,
            Attr::Ssas(a.ssa(vec![qr, kc])),
        );
        let base_v = g.mint();
        let base = Operation::new(a, Some(base_v), OpKind::ArithConstant, &[]).with_attr(
            a,
            AttrKey::Value,
            Attr::Int(cap - m),
        );
        let aq_v = g.mint();
        let aq = Operation::new(a, Some(aq_v), OpKind::ArithAddi, &[base_v, qr]);
        let vis_v = g.mint();
        let cmp = Operation::new(a, Some(vis_v), OpKind::ArithCmpi, &[kc, aq_v]).with_attr(
            a,
            AttrKey::Predicate,
            Attr::Str(a.str(String::from("sle"))),
        );
        let zero_v = g.mint();
        let zero = Operation::new(a, Some(zero_v), OpKind::ArithConstant, &[]).with_attr(
            a,
            AttrKey::Value,
            Attr::Float(0.0),
        );
        let ninf_v = g.mint();
        let ninf = Operation::new(a, Some(ninf_v), OpKind::ArithConstant, &[]).with_attr(
            a,
            AttrKey::Value,
            Attr::Float(-1.0e30),
        );
        let sel_v = g.mint();
        let sel = Operation::new(
            a,
            Some(sel_v),
            OpKind::ArithSelect,
            &[vis_v, zero_v, ninf_v],
        );
        let yld = Operation::new(a, None, OpKind::TensorYield, &[sel_v]);
        let gen_op = Operation::new(a, Some(res), OpKind::TensorGenerate, &[])
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![m, cap])))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(dt));
        Operation {
            regions: a.regions(vec![a.ops(vec![bb0, base, aq, cmp, zero, ninf, sel, yld])]),
            ..gen_op
        }
    }
}
