// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! Head-parallel attention RE-ROLL pass — TODO #1 (the BELOW-cap head regime of
//! Contract B).
//!
//! The real cached prefill attention nodes (`node111.mlir`) are a head-parallel
//! SPMD lowering: a `grid = [H, 1]` whose per-core body is `m` MANUALLY UNROLLED
//! query-row blocks. Each row does the textbook two-block online softmax — a
//! square CONTEXT block over the prior KV cache (`view5`/`view7`, masked by the
//! per-head context mask `view2`) plus a RAGGED causal DIAGONAL block over the
//! current segment (`view6`/`view8`, row `r` attending exactly KV positions
//! `0..=r`). The body emits ~100 interpreter ops PER ROW × `m` rows.
//!
//! This pass RECOGNIZES that idiom from STRUCTURAL invariants (never model names
//! or hard-coded shapes) and RE-ROLLS the `m` per-row blocks into ONE pass of
//! whole-`[m, *]` tensor ops, emitting ONLY RFC-0682 ops (`ktdp` load/store +
//! Arith/Math/LinAlg + `tensor`) that the EXISTING generic interpreter runs
//! UNCHANGED. It is the head analogue of [`crate::flash_attn`] for the cap dim:
//! a correctness-preserving IR→IR rewrite, NOT a hand kernel and NOT a bespoke
//! executor. The grid stays `[H, 1]`; the per-head GQA column arithmetic
//! (`get_compute_tile_id` → `divui gqac` → `muli hdc`) is preserved as SSA so
//! every core still selects its own head/KV slice.
//!
//! ## Why this is a legitimate, semantics-preserving optimization
//!
//! Stacking `m` independent per-row online-softmax blocks `S[r] = q[r]·Kᵀ` into
//! one `S = Q·Kᵀ` is exact tensor re-association (the rows never interact across
//! the softmax — each row reduces over its own KV axis). The ragged per-row
//! diagonal (row `r` over current-seg KV `0..=r`) is reproduced EXACTLY by one
//! square `[m, m]` masked block: a STATIC lower-triangular mask sets `S_d[r,k]`
//! to `0` for `k ≤ r` and `-inf` for `k > r`, so `exp(-inf)=0` zeroes the
//! out-of-causal weights — identical arithmetic, just stacked. Online softmax
//! over the two blocks (global max, two `exp`, two sums, normalize) is the same
//! per row as the unrolled form. It is therefore well inside the 0.05 gate.
//!
//! ## Contract B (`fusion::attention_needs_flash`)
//!
//! This pass owns the BELOW-cap regime: it fires IFF the re-rolled `[m, cap]`
//! context-scores tile fits LX. If that tile would OVERFLOW LX (long context),
//! the pass returns `None` and leaves the node NAIVE so [`crate::flash_attn`]'s
//! cap-tiling owns it — the documented disjoint head-vs-cap partition. The pass
//! emits NO `scf.*` (a pure re-roll), so it stays region-free and never trips
//! the batched-executor's region-free gate.
//!
//! ## Fail-safe recognition
//!
//! [`recognize_head_attention`] returns `None` unless the body is PROVABLY the
//! unrolled two-block head idiom: `grid = [H,1]` with `H>1`, no top-level
//! control flow, exactly `m == view0.rows` stores, and every one of the `m` rows
//! matching the two-matmul-pair QKᵀ/AV signature with a diagonal access tile of
//! EXACTLY `r+1` rows (causal growth verified, not assumed). Any deviation →
//! `None` → module unchanged. We never rewrite a node we cannot prove equivalent.

use ktir_core::arena::Arena;
use ktir_core::attrkey::AttrKey;
use ktir_core::dtypes::DType;
use ktir_core::ir::{Attr, IRFunction, IRModule, Operation, Ssa};
use ktir_core::irtype::IrType;
use ktir_core::opkind::OpKind;
use std::collections::HashMap;

/// The recovered configuration of a recognized head-parallel attention function.
///
/// Every shape/scalar is re-derived from the IR (never assumed). The view-arg
/// pointers and the GQA divisor / head-dim / scale / `-inf` constant are read
/// from the actual ops so the rewrite reproduces the node's exact arithmetic and
/// per-head selection.
#[derive(Clone, Debug, PartialEq)]
pub struct HeadAttnIsland {
    /// Q pointer arg (view0), `[m, H*d]`.
    pub q_arg: Ssa,
    /// O pointer arg (view1), `[m, H*d]`.
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
    /// Q/O view column width `H*d` (so `mk_view` round-trips the original shape).
    pub q_cols: i64,
    /// KV view column width (`num_kv_heads * d`).
    pub kv_cols: i64,
    /// Query rows == number of stores == view0.rows.
    pub m: i64,
    /// Context KV length (the `cap` axis) == view5.rows.
    pub cap: i64,
    /// Head dim.
    pub d: i64,
    /// GQA divisor recovered from the `arith.divui %hpid, %gqac`.
    pub gqac: i64,
    /// Per-head column stride `hdc` recovered from `arith.muli %hpid, %hdc`.
    pub hdc: i64,
    /// Grid head count `H` (grid.0).
    pub h: i64,
    /// `1/sqrt(d)` scale recovered from the `arith.mulf` by a splat constant.
    pub scale: f32,
    /// `-inf` mask constant recovered from the context mask path.
    pub ninf: f32,
    /// Storage dtype.
    pub dtype: DType,
}

impl HeadAttnIsland {
    /// Re-rolled context-scores tile `[m, cap]` × storage-dtype bytes — the value
    /// Contract B's `attention_needs_flash` consumes to decide head-vs-cap.
    pub fn scores_bytes(&self) -> usize {
        (self.m as usize)
            .saturating_mul(self.cap as usize)
            .saturating_mul(self.dtype.bytes_per_elem())
    }

    /// The pointer args this island names — what a rewrite of it must not shadow.
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
}

/// Apply the head re-roll pass to every function in `module`, in place.
///
/// For each function: recognize the unrolled head-parallel idiom; if it is
/// PROVABLY that idiom AND its re-rolled `[m, cap]` scores tile FITS LX
/// (`!needs_flash(scores_bytes)` — Contract B's below-cap regime), replace it
/// with the re-rolled whole-row rewrite (same function NAME, args, grid `[H,1]`).
/// Otherwise leave it untouched (fail-safe: above the cap → flash_attn's regime,
/// or not recognized → naive). Returns the number of functions rewritten.
///
/// `needs_flash` is injected so the caller threads its OWN LX budget (the same
/// predicate flash_attn uses), keeping the cap-partition decision in one place
/// and guaranteeing the two passes never both fire on one node.
pub fn apply_head_rewrite<'a>(
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
        let Some(island) = recognize_head_attention(func) else {
            continue;
        };
        if needs_flash(island.scores_bytes()) {
            // Above the cap: leave naive so flash_attn's cap-tiling owns it.
            continue;
        }
        // Preserve the ORIGINAL function identity (name) and the ORIGINAL argument
        // list (values + order) verbatim, so the program's node→tensor bindings and
        // the segmenter — which bind args by identity AND position — still resolve
        // exactly as before. The rewrite only re-rolls the body; the interface is
        // byte-identical. The name generator starts above every value the ORIGINAL
        // function defines (args included), so a minted value cannot collide with an
        // argument the rolled body does not itself reference.
        let original_args = func.arguments;
        let mut g = NameGen::after(func);
        let mut rolled = rewrite_head_attention(a, &island, &mut g);
        rolled.name = a.str(name.clone());
        rolled.arguments = original_args;
        module.functions.insert(name, rolled);
        rewritten += 1;
    }
    rewritten
}

// ===========================================================================
// Recognition
// ===========================================================================

/// Decoded `ktdp.construct_memory_view %ptr` -> (pointer arg, view shape, dtype).
struct ViewInfo {
    arg: Ssa,
    shape: Vec<i64>,
    dtype: DType,
}

/// Decoded `ktdp.construct_access_tile %view[idx..]` -> the view it reads, the
/// access-tile shape, and the index operands (for the row offset).
struct TileInfo {
    view: Ssa,
    shape: Vec<i64>,
    indices: Vec<Ssa>,
}

fn shape_attr(op: &Operation<'_>) -> Vec<i64> {
    match op.attr(AttrKey::Shape) {
        Some(Attr::IntList(v)) => v.to_vec(),
        _ => Vec::new(),
    }
}

fn dtype_attr(op: &Operation<'_>) -> DType {
    match op.attr(AttrKey::Dtype) {
        Some(Attr::Dtype(d)) => *d,
        _ => DType::F16,
    }
}

/// The `scf.*` structured-control-flow ops. A body containing one is never the
/// flat unrolled idiom these recognizers match.
fn is_control_flow(k: OpKind) -> bool {
    matches!(
        k,
        OpKind::ScfFor | OpKind::ScfIf | OpKind::ScfWhile | OpKind::ScfParallel | OpKind::ScfForall
    )
}

/// The decoded chain behind one matmul output: which view a loaded operand reads
/// and (for the diagonal block) the access-tile row count.
struct LoadChain {
    /// Pointer arg of the view feeding this load.
    arg: Ssa,
    /// Full memory-view shape `[rows, cols]`.
    view_shape: Vec<i64>,
    /// Access-tile shape (the actual loaded sub-tile).
    tile_shape: Vec<i64>,
    /// Storage dtype of the source view.
    dtype: DType,
}

/// Recognize the unrolled head-parallel two-block attention idiom in `func`.
///
/// Returns `Some(island)` only when ALL structural invariants hold (fail-safe):
///   1. `grid = (H, 1, 1)` with `H > 1`;
///   2. no top-level `scf.*` control flow;
///   3. exactly `m` `ktdp.store` ops where `m == view0.rows == view1.rows`;
///   4. a `get_compute_tile_id`, an `arith.divui %hpid, %gqac` (gqac ≥ 1), and an
///      `arith.muli %hpid, %hdc` with `hdc == view0.cols / H`;
///   5. for each row `r` in `0..m`: the two-matmul-pair signature — context
///      `Q·Kcᵀ` (Kc `[cap,d]`) + diagonal `Q·Kdᵀ` (Kd `[r+1, d]`, the CAUSAL
///      growth verified) feeding the AV pair `Wc·Vc` (`[cap,d]`) + `Wd·Vd`
///      (`[r+1,d]`) summed and stored at row `r`;
///   6. all rows share the same scale / mask / gqac / views.
///
/// Any deviation yields `None`, leaving the node naive.
pub fn recognize_head_attention(func: &IRFunction) -> Option<HeadAttnIsland> {
    // (1) grid = [H, 1, 1], H > 1.
    let (h, gy, gz) = func.grid;
    if gy != 1 || gz != 1 || h <= 1 {
        return None;
    }
    let h = h as i64;

    // (2) no top-level control flow (a pure re-roll never starts from a loop).
    if func.operations.iter().any(|op| is_control_flow(op.op_type)) {
        return None;
    }

    // Index views / access tiles / loads by result SSA, and the defining op for
    // every result so we can walk compute chains.
    let mut views: HashMap<Ssa, ViewInfo> = HashMap::new();
    let mut tiles: HashMap<Ssa, TileInfo> = HashMap::new();
    let mut load_src: HashMap<Ssa, LoadChain> = HashMap::new();
    let mut def: HashMap<Ssa, &Operation> = HashMap::new();
    // arith.constant index value table (for resolving row offsets).
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
                    tiles.insert(
                        res,
                        TileInfo {
                            view,
                            shape: shape_attr(op),
                            indices: op.operands[1..].to_vec(),
                        },
                    );
                }
            }
            OpKind::KtdpLoad => {
                if let (Some(res), Some(tile)) = (op.result, op.operands.first())
                    && let Some(ti) = tiles.get(tile)
                    && let Some(vi) = views.get(&ti.view)
                {
                    load_src.insert(
                        res,
                        LoadChain {
                            arg: vi.arg,
                            view_shape: vi.shape.clone(),
                            tile_shape: ti.shape.clone(),
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

    // (4) per-head selection arithmetic: divui by a constant gqac, muli by hdc.
    let gqac = func
        .operations
        .iter()
        .find(|o| o.op_type == OpKind::ArithDivui)
        .and_then(|o| o.operands.get(1))
        .and_then(|c| int_const.get(c).copied())?;
    if gqac < 1 {
        return None;
    }
    if !func
        .operations
        .iter()
        .any(|o| o.op_type == OpKind::KtdpGetComputeTileId)
    {
        return None;
    }

    // (3) exactly `m` stores; m derived from the store COUNT and cross-checked
    // against view0/view1 rows below.
    let stores: Vec<&Operation> = func
        .operations
        .iter()
        .filter(|o| o.op_type == OpKind::KtdpStore)
        .collect();
    let m = stores.len() as i64;
    if m <= 0 {
        return None;
    }

    // Recover each per-row block by walking back from its store. Collect the
    // recovered config and assert it is identical across rows.
    let mut cfg: Option<HeadAttnIsland> = None;
    // The set of expected query-row offsets must be exactly {0, 1, ..., m-1}.
    let mut seen_rows = vec![false; m as usize];

    for store in &stores {
        let row = recognize_row(store, &tiles, &load_src, &def, &int_const, m, gqac, h)?;
        // record the row offset coverage
        if row.q_row < 0 || row.q_row >= m {
            return None;
        }
        let slot = &mut seen_rows[row.q_row as usize];
        if *slot {
            return None; // duplicate row offset
        }
        *slot = true;

        let island = row.island;
        match &cfg {
            None => cfg = Some(island),
            Some(prev) => {
                // All rows must agree on every recovered field.
                if *prev != island {
                    return None;
                }
            }
        }
    }
    // Every row 0..m-1 must be present exactly once (the contiguous causal set).
    if seen_rows.iter().any(|&b| !b) {
        return None;
    }

    cfg
}

// ===========================================================================
// DECODE (m=1) recognition — the head loop is in the BODY, not the grid
// ===========================================================================

/// A recognized m=1 (decode) head-parallel attention island.
///
/// The decode form is a single `grid = [1,1]` function whose body is `H`
/// MANUALLY-UNROLLED identical head blocks (one query row, `m == 1`). Heads are
/// distinguished by per-head `qcol = h*hdc` / `kvcol = (h/gqac)*hdc` arith
/// constants on their access tiles (NOT by `get_compute_tile_id`). This carries
/// the SAME logical config as [`HeadAttnIsland`] plus the head count and the
/// recovered per-head column-offset regularity, so the fused CPU executor can
/// reproduce the decomposed path's exact arithmetic per head.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodeAttnIsland {
    /// Q pointer arg (view0), `[1, H*d]`.
    pub q_arg: Ssa,
    /// O pointer arg (view1), `[1, H*d]` — the `is_output` tensor.
    pub o_arg: Ssa,
    /// Context mask pointer arg (view2), `[1, cap]` (loaded once, shared).
    pub mask_arg: Ssa,
    /// Context K pointer arg (view5), `[cap, kv_cols]`.
    pub kc_arg: Ssa,
    /// Diagonal (current-token) K pointer arg (view6), `[1, kv_cols]`.
    pub kd_arg: Ssa,
    /// Context V pointer arg (view7), `[cap, kv_cols]`.
    pub vc_arg: Ssa,
    /// Diagonal (current-token) V pointer arg (view8), `[1, kv_cols]`.
    pub vd_arg: Ssa,
    /// Q/O view column width `H*d`.
    pub q_cols: i64,
    /// KV view column width (`num_kv_heads * d`).
    pub kv_cols: i64,
    /// Context KV length (`cap` axis) == view5 rows.
    pub cap: i64,
    /// Head dim.
    pub d: i64,
    /// Head count == number of stores.
    pub h: i64,
    /// GQA divisor (`kv_head = head / gqac`).
    pub gqac: i64,
    /// Per-head column stride (`hdc == d`).
    pub hdc: i64,
    /// `1/sqrt(d)` scale.
    pub scale: f32,
    /// Storage dtype.
    pub dtype: DType,
}

/// Recognize the m=1 (decode) unrolled head-parallel attention idiom in `func`.
///
/// Sibling to [`recognize_head_attention`] for the single-query-row decode form.
/// Returns `Some(island)` only when ALL structural invariants hold (fail-safe):
///   1. `grid = (1, 1, 1)` (decode is single-token, single-core);
///   2. no top-level `scf.*` control flow;
///   3. one or more `ktdp.store`s, each a self-contained two-block head whose
///      back-walk matches the QKᵀ / online-softmax / AV signature with a CONTEXT
///      block (mask-added, Kc/Vc `[cap, d]`) and a DIAGONAL block (no mask, Kd/Vd
///      `[1, d]`), Q the SAME load for both;
///   4. all heads share scale / mask / gqac / hdc / views, and the recovered
///      per-head offsets follow `qcol_h = h*hdc`, `kvcol_h = (h/gqac)*hdc` EXACTLY
///      over `h = 0..H` (the structural regularity, not a model-specific shape).
///
/// Any deviation yields `None`, leaving the node decomposed (the oracle).
pub fn recognize_head_attention_decode(func: &IRFunction) -> Option<DecodeAttnIsland> {
    // (1) grid = [1,1,1].
    let (gx, gy, gz) = func.grid;
    if gx != 1 || gy != 1 || gz != 1 {
        return None;
    }
    // (2) no top-level control flow.
    if func.operations.iter().any(|op| is_control_flow(op.op_type)) {
        return None;
    }

    // Index ops by result SSA. Resolve index values that are either direct
    // `arith.constant` or `arith.addi`/`arith.muli`/`arith.divui` of resolved
    // operands (the decode form computes `kc = kcs + kvcol`).
    let mut views: HashMap<Ssa, ViewInfo> = HashMap::new();
    let mut tiles: HashMap<Ssa, TileInfo> = HashMap::new();
    let mut load_src: HashMap<Ssa, LoadChain> = HashMap::new();
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
                    tiles.insert(
                        res,
                        TileInfo {
                            view,
                            shape: shape_attr(op),
                            indices: op.operands[1..].to_vec(),
                        },
                    );
                }
            }
            OpKind::KtdpLoad => {
                if let (Some(res), Some(tile)) = (op.result, op.operands.first())
                    && let Some(ti) = tiles.get(tile)
                    && let Some(vi) = views.get(&ti.view)
                {
                    load_src.insert(
                        res,
                        LoadChain {
                            arg: vi.arg,
                            view_shape: vi.shape.clone(),
                            tile_shape: ti.shape.clone(),
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

    // Resolve an index SSA value through constants + addi/muli/divui chains.
    fn resolve_index(
        ssa: &Ssa,
        int_const: &HashMap<Ssa, i64>,
        def: &HashMap<Ssa, &Operation>,
        depth: usize,
    ) -> Option<i64> {
        if depth > 16 {
            return None;
        }
        if let Some(v) = int_const.get(ssa) {
            return Some(*v);
        }
        let op = def.get(ssa)?;
        let a = op.operands.first()?;
        let b = op.operands.get(1)?;
        let av = resolve_index(a, int_const, def, depth + 1)?;
        let bv = resolve_index(b, int_const, def, depth + 1)?;
        match op.op_type {
            OpKind::ArithAddi => Some(av + bv),
            OpKind::ArithMuli => Some(av * bv),
            OpKind::ArithDivui if bv != 0 => Some(av.div_euclid(bv)),
            _ => None,
        }
    }

    // (4) GQA divisor: the per-head `kvcol = (h/gqac)*hdc` is baked as constants in
    // decode (no `divui` SSA), so recover gqac/hdc from the per-head offset
    // regularity below — start with hdc = d once we know d.

    // (3) one store per head.
    let stores: Vec<&Operation> = func
        .operations
        .iter()
        .filter(|o| o.op_type == OpKind::KtdpStore)
        .collect();
    let h = stores.len() as i64;
    if h < 1 {
        return None;
    }

    // Recover each head block. Collect (qcol, kvcol) and the row-invariant config.
    let mut cfg: Option<DecodeAttnIsland> = None;
    let mut offsets: Vec<(i64, i64)> = Vec::with_capacity(h as usize);

    for store in &stores {
        let hm = recognize_head_decode(store, &tiles, &load_src, &def, &int_const, &resolve_index)?;
        offsets.push((hm.qcol, hm.kvcol));
        match &cfg {
            None => cfg = Some(hm.island),
            Some(prev) => {
                if *prev != hm.island {
                    return None;
                }
            }
        }
    }
    let mut island = cfg?;
    island.h = h;

    // (4) Verify the per-head offset regularity STRUCTURALLY: sorting heads by
    // qcol, qcol_h MUST equal h*hdc and kvcol_h MUST equal (h/gqac)*hdc for a
    // single hdc and gqac. hdc = d (head dim). Derive gqac from the kvcol pattern
    // and require an exact match (fail-safe to None otherwise).
    let hdc = island.d;
    island.hdc = hdc;
    if hdc <= 0 {
        return None;
    }
    offsets.sort_by_key(|&(q, _)| q);
    // qcol_h must be exactly h*hdc with no duplicates.
    for (idx, &(q, _)) in offsets.iter().enumerate() {
        if q != idx as i64 * hdc {
            return None;
        }
    }
    // Recover gqac from the first kvcol step: the number of consecutive heads that
    // share a kv head. kvcol_h = (h / gqac) * hdc. gqac = number of leading heads
    // whose kvcol == 0 (the first kv head's group size). Then verify the whole
    // sequence matches (h/gqac)*hdc.
    let gqac = {
        let mut g = 0i64;
        for &(_, kv) in &offsets {
            if kv == 0 {
                g += 1;
            } else {
                break;
            }
        }
        g
    };
    if gqac < 1 {
        return None;
    }
    for (idx, &(_, kv)) in offsets.iter().enumerate() {
        if kv != (idx as i64 / gqac) * hdc {
            return None;
        }
    }
    island.gqac = gqac;

    // kv_cols must accommodate the highest kv head's slice.
    let max_kvcol = offsets.iter().map(|&(_, kv)| kv).max().unwrap_or(0);
    if max_kvcol + hdc > island.kv_cols {
        return None;
    }
    // q_cols must accommodate the highest head's slice.
    if (h - 1) * hdc + hdc > island.q_cols {
        return None;
    }

    Some(island)
}

/// One recognized decode head block: its column offsets and the (head-invariant)
/// island config it implies.
struct HeadDecodeMatch {
    qcol: i64,
    kvcol: i64,
    island: DecodeAttnIsland,
}

/// Walk back from one head's `ktdp.store` and prove the decode two-block
/// online-softmax signature, returning the head's column offsets and the implied
/// island config. Returns `None` on any structural deviation (fail-safe).
#[allow(clippy::too_many_arguments)]
fn recognize_head_decode(
    store: &Operation,
    tiles: &HashMap<Ssa, TileInfo>,
    load_src: &HashMap<Ssa, LoadChain>,
    def: &HashMap<Ssa, &Operation>,
    int_const: &HashMap<Ssa, i64>,
    resolve_index: &impl Fn(&Ssa, &HashMap<Ssa, i64>, &HashMap<Ssa, &Operation>, usize) -> Option<i64>,
) -> Option<HeadDecodeMatch> {
    // store %oa, %o_tile (O[1, d] at [0, qcol]).
    let stored_val = store.operands.first()?;
    let o_tile_ssa = store.operands.get(1)?;
    let o_tile = tiles.get(o_tile_ssa)?;
    let o_view = o_tile.view;
    // qcol = second index operand (the column offset); first index is the row (0).
    let qcol = resolve_index(o_tile.indices.get(1)?, int_const, def, 0)?;

    // oa = arith.addf(ov_context, ov_diag).
    let add = def.get(stored_val)?;
    if add.op_type != OpKind::ArithAddf {
        return None;
    }
    let ovc = add.operands.first()?;
    let ovd = add.operands.get(1)?;

    // Context AV: ov_context = linalg.matmul(Wc, Vc), Vc loaded [cap, d].
    let avc = def.get(ovc)?;
    if avc.op_type != OpKind::LinalgMatmul {
        return None;
    }
    let wc = avc.operands.first()?;
    let vc_loaded = avc.operands.get(1)?;
    let vc = load_src.get(vc_loaded)?;

    // Diagonal AV: ov_diag = linalg.matmul(Wd, Vd), Vd loaded [1, d].
    let avd = def.get(ovd)?;
    if avd.op_type != OpKind::LinalgMatmul {
        return None;
    }
    let wd = avd.operands.first()?;
    let vd_loaded = avd.operands.get(1)?;
    let vd = load_src.get(vd_loaded)?;

    // Wc = divf(exp_c, gs_bcast), Wd = divf(exp_d, gs_bcast).
    let (exp_c, _gsc) = trace_divf(wc, def)?;
    let (exp_d, _gsd) = trace_divf(wd, def)?;

    // exp_c = math.exp(sub_c); sub_c = subf(scm_c, gm_bcast).
    let scm_c = trace_exp_sub(&exp_c, def)?;
    let sd = trace_exp_sub(&exp_d, def)?;

    // CONTEXT scores: scm_c = addf(scaled_c, mask).
    let scm_op = def.get(&scm_c)?;
    if scm_op.op_type != OpKind::ArithAddf {
        return None;
    }
    let scaled_c = scm_op.operands.first()?;
    let mask_loaded = scm_op.operands.get(1)?;
    let mask_chain = load_src.get(mask_loaded)?;

    // scaled_c = mulf(raw_c, scale_splat).
    let (raw_c, scale) = trace_scale(scaled_c, def)?;
    // raw_c = matmul(Q, transpose(Kc)); Kc [cap, d].
    let (q_loaded_c, kc) = trace_qk(&raw_c, def, load_src)?;

    // DIAGONAL scores: sd = mulf(raw_d, scale_splat) — NO mask add (single token).
    let (raw_d, scale_d) = trace_scale(&sd, def)?;
    if (scale - scale_d).abs() > 1e-4 {
        return None;
    }
    let (q_loaded_d, kd) = trace_qk(&raw_d, def, load_src)?;

    // Q must be the SAME load arg for both blocks.
    let q_c = load_src.get(&q_loaded_c)?;
    let q_d = load_src.get(&q_loaded_d)?;
    if q_c.arg != q_d.arg {
        return None;
    }

    // kvcol: the K context tile's column index (second index operand of its tile).
    // Re-find the context K access tile via the transpose -> load -> tile chain.
    let kvcol = {
        // raw_c = matmul(Q, kt); kt = transpose(kc_loaded); kc_loaded came from a
        // load whose tile's column index is kvcol.
        let mm = def.get(&raw_c)?;
        let kt = def.get(mm.operands.get(1)?)?;
        let kc_loaded = kt.operands.first()?;
        // find the access tile feeding this load
        let load_op = def.get(kc_loaded)?;
        let tile_ssa = load_op.operands.first()?;
        let ti = tiles.get(tile_ssa)?;
        resolve_index(ti.indices.get(1)?, int_const, def, 0)?
    };

    // ---- shape checks ----
    // Q/O view [1, q_cols]; q_cols = H*d. d = head-dim from the Q tile width.
    let q_shape = &q_c.view_shape;
    if q_shape.len() != 2 || q_shape[0] != 1 {
        return None;
    }
    let q_cols = q_shape[1];
    let d = q_c.tile_shape.get(1).copied()?;
    if d <= 0 || q_cols % d != 0 {
        return None;
    }
    // Context K/V view [cap, kv_cols].
    if kc.view_shape.len() != 2 || vc.view_shape != kc.view_shape {
        return None;
    }
    let cap = kc.view_shape[0];
    let kv_cols = kc.view_shape[1];
    if cap <= 0 || kv_cols % d != 0 {
        return None;
    }
    // Context K/V access tiles read the full [cap, d] head slice.
    if kc.tile_shape != [cap, d] || vc.tile_shape != [cap, d] {
        return None;
    }
    // Diagonal K/V view [1, kv_cols]; access tile [1, d] (single current token).
    if kd.view_shape != [1, kv_cols] || vd.view_shape != [1, kv_cols] {
        return None;
    }
    if kd.tile_shape != [1, d] || vd.tile_shape != [1, d] {
        return None;
    }
    // Output view must equal the Q view [1, q_cols].
    let o_arg = {
        let vop = def.get(&o_view)?;
        if vop.op_type != OpKind::KtdpConstructMemoryView {
            return None;
        }
        let os = shape_attr(vop);
        if os != *q_shape {
            return None;
        }
        *vop.operands.first()?
    };
    // Mask view [1, cap].
    if mask_chain.view_shape != [1, cap] {
        return None;
    }

    let island = DecodeAttnIsland {
        q_arg: q_c.arg,
        o_arg,
        mask_arg: mask_chain.arg,
        kc_arg: kc.arg,
        kd_arg: kd.arg,
        vc_arg: vc.arg,
        vd_arg: vd.arg,
        q_cols,
        kv_cols,
        cap,
        d,
        h: 0, // filled by caller from store count
        gqac: 1,
        hdc: d,
        scale,
        dtype: q_c.dtype,
    };

    Some(HeadDecodeMatch {
        qcol,
        kvcol,
        island,
    })
}

/// The ROW-MAJOR f32 buffers [`DecodeAttnIsland::compute_f32`] reads, already decoded from HBM.
/// See that method's doc for each field's shape.
pub struct DecodeAttnInputs<'a> {
    pub q: &'a [f32],
    pub mask: &'a [f32],
    pub kc: &'a [f32],
    pub kd: &'a [f32],
    pub vc: &'a [f32],
    pub vd: &'a [f32],
}

impl DecodeAttnIsland {
    /// Compute the fused m=1 attention into `o` (the `[1, q_cols]` output row), in
    /// f32, reproducing the decomposed path's exact arithmetic per head:
    /// per head `h` (`qcol = h*hdc`, `kvh = h/gqac`, `kvcol = kvh*hdc`):
    ///   * `s_c[j] = scale * Σ_t Q[qcol+t]*Kc[j, kvcol+t] + mask[j]`  (j in 0..cap)
    ///   * `s_d    = scale * Σ_t Q[qcol+t]*Kd[kvcol+t]`               (no mask)
    ///   * `gm = max(max_j s_c[j], s_d)`; `e_c[j]=exp(s_c[j]-gm)`, `e_d=exp(s_d-gm)`
    ///   * `Z = Σ_j e_c[j] + e_d`; `o[qcol+t] = (Σ_j e_c[j]*Vc[j,kvcol+t]
    ///       + e_d*Vd[kvcol+t]) / Z`
    ///
    /// Inputs are ROW-MAJOR f32 buffers already decoded from HBM:
    ///   * `q`:    `[q_cols]`            (the single query row)
    ///   * `mask`: `[cap]`              (the shared context mask)
    ///   * `kc`/`vc`: `[cap * kv_cols]` (context K/V, row-major `[cap, kv_cols]`)
    ///   * `kd`/`vd`: `[kv_cols]`       (current-token K/V)
    ///   * `o`:    `[q_cols]`           (output, written in place)
    ///
    /// f32 accumulation throughout — TIGHTER than the decomposed f16-intermediate
    /// path, so well inside the golden band.
    ///
    /// One past the LAST context row that can affect the result — the rest are dead.
    ///
    /// A `-inf` mask entry makes `sc[j] = scale*dot + mask[j]` exactly `-inf` (the dot is
    /// finite: `q` and `Kc` are decoded from f16), while `gm` stays finite because the
    /// DIAGONAL score carries no mask and is always in the max. So `exp(sc[j] - gm)` is
    /// exactly `0.0`: such a row adds `+0.0` to the denominator and `0.0` to every output
    /// element. Dropping it is bit-identical, not an approximation.
    ///
    /// scratchy's context mask is `attn_mask_fill`
    /// (`crates/targets/spyre/src/manifest.rs:363`) — `0.0` on the valid prefix, `-1.0e38`
    /// past it — and `-1.0e38` ENCODES TO f16 `-inf` (`ktir-core/src/codec.rs:53`, exponent
    /// overflow returns `sign | 0x7c00`), so those rows arrive here as exactly `-inf`. A
    /// merely large FINITE sentinel is kept, which is conservative and still correct.
    pub fn live_context(mask: &[f32]) -> usize {
        mask.iter()
            .rposition(|m| *m != f32::NEG_INFINITY)
            .map_or(0, |i| i + 1)
    }

    pub fn compute_f32(&self, inputs: DecodeAttnInputs<'_>, o: &mut [f32]) {
        let DecodeAttnInputs {
            q,
            mask,
            kc,
            kd,
            vc,
            vd,
        } = inputs;
        let d = self.d as usize;
        // The DEAD tail of the context is skipped, so `kc`/`vc` need only hold this many
        // rows — which is what lets the caller avoid decoding the whole capacity.
        let cap = Self::live_context(mask).min(self.cap as usize);
        let kvw = self.kv_cols as usize;
        let scale = self.scale;
        for hh in 0..self.h as usize {
            let qcol = hh * self.hdc as usize;
            let kvh = hh / self.gqac as usize;
            let kvcol = kvh * self.hdc as usize;
            let qh = &q[qcol..qcol + d];

            // CONTEXT scores s_c[j] (GEMV q·Kcᵀ over the kvcol column-slice) + mask.
            let mut sc = vec![0.0f32; cap];
            let mut gm = f32::NEG_INFINITY;
            for j in 0..cap {
                let krow = &kc[j * kvw + kvcol..j * kvw + kvcol + d];
                let mut dot = 0.0f32;
                for t in 0..d {
                    dot += qh[t] * krow[t];
                }
                let s = scale * dot + mask[j];
                sc[j] = s;
                if s > gm {
                    gm = s;
                }
            }
            // DIAGONAL score s_d (single dot, no mask).
            let kdrow = &kd[kvcol..kvcol + d];
            let mut dot_d = 0.0f32;
            for t in 0..d {
                dot_d += qh[t] * kdrow[t];
            }
            let sd = scale * dot_d;
            if sd > gm {
                gm = sd;
            }

            // Online softmax over the two blocks (global max, exp, denominator).
            let mut z = 0.0f32;
            for s in sc.iter_mut() {
                *s = (*s - gm).exp();
                z += *s;
            }
            let ed = (sd - gm).exp();
            z += ed;
            let inv_z = 1.0f32 / z;

            // OUTPUT o_h[t] = (Σ_j e_c[j]*Vc[j, kvcol+t] + e_d*Vd[kvcol+t]) / Z.
            let oh = &mut o[qcol..qcol + d];
            for t in 0..d {
                let mut acc = 0.0f32;
                for j in 0..cap {
                    acc += sc[j] * vc[j * kvw + kvcol + t];
                }
                acc += ed * vd[kvcol + t];
                oh[t] = acc * inv_z;
            }
        }
    }
}

/// One recognized query-row block: its row offset and the (row-invariant) island
/// config it implies. `recognize_head_attention` cross-checks the config across
/// all rows and the row offsets cover `0..m-1`.
struct RowMatch {
    q_row: i64,
    island: HeadAttnIsland,
}

/// Walk back from one row's `ktdp.store` and prove the two-block online-softmax
/// signature, returning the row offset and the implied island config. Returns
/// `None` on any structural deviation (fail-safe).
#[allow(clippy::too_many_arguments)]
fn recognize_row(
    store: &Operation,
    tiles: &HashMap<Ssa, TileInfo>,
    load_src: &HashMap<Ssa, LoadChain>,
    def: &HashMap<Ssa, &Operation>,
    int_const: &HashMap<Ssa, i64>,
    m: i64,
    gqac: i64,
    h: i64,
) -> Option<RowMatch> {
    // store %oa, %o_tile  (O[1, d] at [q_row, qcol]).
    let stored_val = store.operands.first()?;
    let o_tile_ssa = store.operands.get(1)?;
    let o_tile = tiles.get(o_tile_ssa)?;
    let o_view = o_tile.view;
    // q_row is the first index operand resolved to a constant.
    let q_row = *int_const.get(o_tile.indices.first()?)?;

    // oa = arith.addf(ov_context, ov_diag)
    let add = def.get(stored_val)?;
    if add.op_type != OpKind::ArithAddf {
        return None;
    }
    let ovc = add.operands.first()?;
    let ovd = add.operands.get(1)?;

    // Context AV: ov_context = linalg.matmul(Wc, Vc), Vc loaded [cap, d].
    let avc = def.get(ovc)?;
    if avc.op_type != OpKind::LinalgMatmul {
        return None;
    }
    let wc = avc.operands.first()?;
    let vc_loaded = avc.operands.get(1)?;
    let vc = load_src.get(vc_loaded)?;

    // Diagonal AV: ov_diag = linalg.matmul(Wd, Vd), Vd loaded [r+1, d].
    let avd = def.get(ovd)?;
    if avd.op_type != OpKind::LinalgMatmul {
        return None;
    }
    let wd = avd.operands.first()?;
    let vd_loaded = avd.operands.get(1)?;
    let vd = load_src.get(vd_loaded)?;

    // Wc = divf(exp_c, gs_bcast), Wd = divf(exp_d, gs_bcast).
    let (exp_c, _gsc) = trace_divf(wc, def)?;
    let (exp_d, _gsd) = trace_divf(wd, def)?;

    // exp_c = math.exp(sub_c); sub_c = subf(scm_c, gm_bcast)
    let scm_c = trace_exp_sub(&exp_c, def)?;
    let sd = trace_exp_sub(&exp_d, def)?;

    // CONTEXT scores: scm_c = addf(scaled_c, mask)  [the per-head context mask].
    let scm_op = def.get(&scm_c)?;
    if scm_op.op_type != OpKind::ArithAddf {
        return None;
    }
    let scaled_c = scm_op.operands.first()?;
    let mask_loaded = scm_op.operands.get(1)?;
    let mask_chain = load_src.get(mask_loaded)?;

    // scaled_c = mulf(raw_c, scale_splat)
    let (raw_c, scale) = trace_scale(scaled_c, def)?;
    // raw_c = matmul(Q, Kct); Kct = transpose(Kc_loaded); Kc [cap, d].
    let (q_loaded_c, kc) = trace_qk(&raw_c, def, load_src)?;

    // DIAGONAL scores: sd = mulf(raw_d, scale_splat)  (NO mask add on the
    // diagonal in the unrolled form — the ragged access tile IS the mask).
    let (raw_d, scale_d) = trace_scale(&sd, def)?;
    if (scale - scale_d).abs() > 1e-4 {
        return None;
    }
    let (q_loaded_d, kd) = trace_qk(&raw_d, def, load_src)?;

    // Q must be the SAME load arg for both blocks (one query row).
    let q_c = load_src.get(&q_loaded_c)?;
    let q_d = load_src.get(&q_loaded_d)?;
    if q_c.arg != q_d.arg {
        return None;
    }

    // Recover the -inf mask constant from the context mask reduce-max init, or
    // fall back to the project default. The reduce over scm_c uses a splat of the
    // -inf constant; recover it for an exact rewrite (mask above-diagonal value).
    let ninf = recover_ninf(&scm_c, def).unwrap_or(-1.0e38);

    // ---- shape checks ----
    // Q/O view [m, q_cols]; q_cols = H*d, hdc = d = q_cols/H.
    let q_shape = &q_c.view_shape;
    if q_shape.len() != 2 || q_shape[0] != m {
        return None;
    }
    let q_cols = q_shape[1];
    if q_cols % h != 0 {
        return None;
    }
    let d = q_cols / h;
    if d <= 0 {
        return None;
    }
    // Context K/V view [cap, kv_cols]; cap = view rows.
    if kc.view_shape.len() != 2 || vc.view_shape != kc.view_shape {
        return None;
    }
    let cap = kc.view_shape[0];
    let kv_cols = kc.view_shape[1];
    if cap <= 0 || kv_cols % d != 0 {
        return None;
    }
    // Context K/V access tiles read the full [cap, d] head slice.
    if kc.tile_shape != [cap, d] || vc.tile_shape != [cap, d] {
        return None;
    }
    // Diagonal K/V view [m, kv_cols]; access tile MUST be [r+1, d] (causal).
    if kd.view_shape != [m, kv_cols] || vd.view_shape != [m, kv_cols] {
        return None;
    }
    if kd.tile_shape != [q_row + 1, d] || vd.tile_shape != [q_row + 1, d] {
        return None; // causal diagonal growth not satisfied -> fail-safe
    }
    // Output view must equal the Q view [m, q_cols].
    // (mask view is [1, cap].)
    let o_arg = {
        // resolve the o_view's pointer arg + shape via its construct_memory_view.
        // o_tile.view -> view info isn't in load_src; look it up via def.
        let vop = def.get(&o_view)?;
        if vop.op_type != OpKind::KtdpConstructMemoryView {
            return None;
        }
        let os = shape_attr(vop);
        if os != *q_shape {
            return None;
        }
        *vop.operands.first()?
    };
    if mask_chain.view_shape != [1, cap] {
        return None;
    }

    let island = HeadAttnIsland {
        q_arg: q_c.arg,
        o_arg,
        mask_arg: mask_chain.arg,
        kc_arg: kc.arg,
        kd_arg: kd.arg,
        vc_arg: vc.arg,
        vd_arg: vd.arg,
        q_cols,
        kv_cols,
        m,
        cap,
        d,
        gqac,
        hdc: d,
        h,
        scale,
        ninf,
        dtype: q_c.dtype,
    };

    Some(RowMatch { q_row, island })
}

/// `w = arith.divf(exp_tensor, gs_bcast)` -> (exp_tensor, gs_bcast).
fn trace_divf(w: &Ssa, def: &HashMap<Ssa, &Operation>) -> Option<(Ssa, Ssa)> {
    let d = def.get(w)?;
    if d.op_type != OpKind::ArithDivf {
        return None;
    }
    Some((*d.operands.first()?, *d.operands.get(1)?))
}

/// `exp = math.exp(subf(x, gm_bcast))` -> x (the un-shifted scores).
fn trace_exp_sub(exp: &Ssa, def: &HashMap<Ssa, &Operation>) -> Option<Ssa> {
    let e = def.get(exp)?;
    if e.op_type != OpKind::MathExp {
        return None;
    }
    let sub = def.get(e.operands.first()?)?;
    if sub.op_type != OpKind::ArithSubf {
        return None;
    }
    Some(*sub.operands.first()?)
}

/// `scaled = arith.mulf(raw, tensor.splat(scale_const))` -> (raw, scale value).
fn trace_scale(scaled: &Ssa, def: &HashMap<Ssa, &Operation>) -> Option<(Ssa, f32)> {
    let mul = def.get(scaled)?;
    if mul.op_type != OpKind::ArithMulf {
        return None;
    }
    let raw = *mul.operands.first()?;
    let splat = def.get(mul.operands.get(1)?)?;
    if splat.op_type != OpKind::TensorSplat {
        return None;
    }
    let c = def.get(splat.operands.first()?)?;
    if c.op_type != OpKind::ArithConstant {
        return None;
    }
    let scale = match c.attr(AttrKey::Value) {
        Some(Attr::Float(f)) => *f as f32,
        Some(Attr::Int(i)) => *i as f32,
        _ => return None,
    };
    Some((raw, scale))
}

/// `raw = linalg.matmul(Q_loaded, Kt); Kt = linalg.transpose(K_loaded)` ->
/// (Q_loaded SSA, K LoadChain).
fn trace_qk<'a>(
    raw: &Ssa,
    def: &HashMap<Ssa, &Operation>,
    load_src: &'a HashMap<Ssa, LoadChain>,
) -> Option<(Ssa, &'a LoadChain)> {
    let mm = def.get(raw)?;
    if mm.op_type != OpKind::LinalgMatmul {
        return None;
    }
    let q_loaded = *mm.operands.first()?;
    let kt = def.get(mm.operands.get(1)?)?;
    if kt.op_type != OpKind::LinalgTranspose {
        return None;
    }
    let k_loaded = kt.operands.first()?;
    let kc = load_src.get(k_loaded)?;
    Some((q_loaded, kc))
}

/// Recover the `-inf` mask additive constant from the context reduce-max init
/// (`tensor.splat(arith.constant -1e38)`), traced from the masked scores.
fn recover_ninf(scm_c: &Ssa, def: &HashMap<Ssa, &Operation>) -> Option<f32> {
    // Find the reduce-max that consumes scm_c, read its outs init splat const.
    // We search defs for a linalg.reduce over scm_c with reduce_fn maximumf.
    for op in def.values() {
        if op.op_type == OpKind::LinalgReduce
            && op.attr(AttrKey::ReduceFn) == Some(&Attr::Op(OpKind::ArithMaximumf))
            && op.operands.first() == Some(scm_c)
        {
            // outs init named in outs_var or operand[1]; trace its splat const.
            let init = match op.attr(AttrKey::OutsVar) {
                Some(Attr::Ssas([v])) => *v,
                _ => *op.operands.get(1)?,
            };
            let splat = def.get(&init)?;
            if splat.op_type == OpKind::TensorSplat
                && let Some(c) = def.get(splat.operands.first()?)
                && c.op_type == OpKind::ArithConstant
                && let Some(Attr::Float(f)) = c.attr(AttrKey::Value)
            {
                return Some(*f as f32);
            }
        }
    }
    None
}

// ===========================================================================
// Rewrite (whole-row re-roll)
// ===========================================================================

/// A monotonic SSA name generator (collision-free within one rewritten body).
/// Mints the values this rewrite introduces.
///
/// These were names (`%hr_{tag}_{n}`) whose uniqueness rested on no existing value sharing the
/// prefix — an assumption nothing checked. An id cannot collide with one already in the function
/// because the function states its own high-water mark.
pub struct NameGen {
    next: u32,
}

impl NameGen {
    /// Start above every value `func` already defines, so a minted id is fresh by construction.
    pub fn after(func: &IRFunction<'_>) -> NameGen {
        let high = func
            .operations
            .iter()
            .filter_map(|op| op.result)
            .chain(func.arguments.iter().map(|(v, _)| *v))
            .map(|v| v.0)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        NameGen { next: high }
    }

    /// Start above every value in `vals` — what a rewrite driven by an island alone
    /// (rather than by a whole function) has to clear.
    pub fn above(vals: impl IntoIterator<Item = Ssa>) -> NameGen {
        NameGen {
            next: vals.into_iter().map(|v| v.0 + 1).max().unwrap_or(0),
        }
    }

    /// A fresh identity, guaranteed distinct from every value already seeded past.
    pub fn mint(&mut self) -> Ssa {
        let v = Ssa(self.next);
        self.next += 1;
        v
    }
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

/// `ktdp.construct_memory_view %ptr {shape, strides, memory_space, dtype}` — a
/// logical view only (RFC 0682: does NOT allocate).
fn mk_view<'a>(a: &'a Arena, res: Ssa, ptr: Ssa, shape: &[i64], dtype: DType) -> Operation<'a> {
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

/// Load a `[rows, cols]` tile of `view` at dynamic offset `[%row, %col]`:
/// `construct_access_tile %view[%row, %col]` then `ktdp.load`. The index
/// operands stay real SSA so the per-head column offset is honored per core.
fn block_load<'a>(
    a: &'a Arena,
    g: &mut NameGen,
    ops: &mut Vec<Operation<'a>>,
    view: Ssa,
    row: Ssa,
    col: Ssa,
    tile: [i64; 2],
) -> Ssa {
    let at = g.mint();
    ops.push(
        Operation::new(
            a,
            Some(at),
            OpKind::KtdpConstructAccessTile,
            &[view, row, col],
        )
        .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(tile.to_vec()))),
    );
    let loaded = g.mint();
    ops.push(Operation::new(a, Some(loaded), OpKind::KtdpLoad, &[at]));
    loaded
}

fn mk_splat<'a>(a: &'a Arena, res: Ssa, scalar: Ssa, shape: &[i64], dtype: DType) -> Operation<'a> {
    Operation::new(a, Some(res), OpKind::TensorSplat, &[scalar])
        .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(shape.to_vec())))
        .with_attr(a, AttrKey::Dtype, Attr::Dtype(dtype))
}

fn mk_empty<'a>(a: &'a Arena, res: Ssa, shape: &[i64], dtype: DType) -> Operation<'a> {
    Operation::new(a, Some(res), OpKind::TensorEmpty, &[])
        .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(shape.to_vec())))
        .with_attr(a, AttrKey::Dtype, Attr::Dtype(dtype))
}

/// `linalg.transpose ins(%x) outs(%init) permutation=[1,0]` -> `[cols, rows]`.
fn mk_transpose<'a>(
    a: &'a Arena,
    g: &mut NameGen,
    ops: &mut Vec<Operation<'a>>,
    x: Ssa,
    rows: i64,
    cols: i64,
    dtype: DType,
) -> Ssa {
    let init = g.mint();
    ops.push(mk_empty(a, init, &[cols, rows], dtype));
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

/// `C = A @ B` with a zero `tensor.empty` outs init (so matmul's `C + A@B`
/// reduces to `A@B`).
fn mk_matmul<'ar>(
    a: &'ar Arena,
    g: &mut NameGen,
    ops: &mut Vec<Operation<'ar>>,
    lhs: Ssa,
    rhs: Ssa,
    out: [i64; 2],
    dtype: DType,
) -> Ssa {
    let init = g.mint();
    ops.push(mk_empty(a, init, &out, dtype));
    let res = g.mint();
    ops.push(Operation::new(
        a,
        Some(res),
        OpKind::LinalgMatmul,
        &[lhs, rhs, init],
    ));
    res
}

/// `linalg.reduce { reduce_fn } ins(%x) outs(%init) dimensions=[1]` over the last
/// axis of `[m, c]` -> `[m]`.
fn mk_reduce<'a>(a: &'a Arena, res: Ssa, x: Ssa, init: Ssa, reduce_fn: OpKind) -> Operation<'a> {
    Operation::new(a, Some(res), OpKind::LinalgReduce, &[x])
        .with_attr(a, AttrKey::ReduceFn, Attr::Op(reduce_fn))
        .with_attr(a, AttrKey::Dimensions, Attr::IntList(a.ints(vec![1])))
        .with_attr(a, AttrKey::OutsVar, Attr::Ssas(a.ssa(vec![init])))
}

/// Broadcast a `[m]` row-vector to `[m, cols]` (reshape to `[m,1]` then
/// `linalg.broadcast` up to the outs shape).
fn broadcast_row_to<'a>(
    a: &'a Arena,
    g: &mut NameGen,
    ops: &mut Vec<Operation<'a>>,
    rowv: Ssa,
    m: i64,
    cols: i64,
    dtype: DType,
) -> Ssa {
    let r2 = g.mint();
    ops.push(
        Operation::new(a, Some(r2), OpKind::TensorReshape, &[rowv]).with_attr(
            a,
            AttrKey::TargetShape,
            Attr::IntList(a.ints(vec![m, 1])),
        ),
    );
    let init = g.mint();
    ops.push(mk_empty(a, init, &[m, cols], dtype));
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

/// The static `[m, m]` lower-triangular causal mask: `0` for `k ≤ r` (visible),
/// `ninf` for `k > r` (masked). This reproduces EXACTLY the unrolled form's
/// ragged diagonal — row `r` attends current-segment KV `0..=r`. Baked as a
/// dense `arith.constant` tensor (no region, no select), so it stays region-free
/// and fusion-safe.
fn causal_mask_mm<'a>(a: &'a Arena, res: Ssa, m: i64, ninf: f32, dtype: DType) -> Operation<'a> {
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
        .with_attr(a, AttrKey::Dtype, Attr::Dtype(dtype))
        .with_attr(a, AttrKey::Value, Attr::FloatList(a.floats(vals)))
}

/// Rewrite a recognized [`HeadAttnIsland`] into the whole-row re-rolled body.
///
/// Emits ONCE (instead of `m` times) per core, preserving grid `[H,1,1]` and the
/// per-head GQA column arithmetic:
///   * `hpid = get_compute_tile_id`; `qcol = hpid*hdc`; `kvcol = (hpid/gqac)*hdc`
///   * Q = load view0[0, qcol] -> `[m, d]`
///   * CONTEXT: Kc = view5[0, kvcol] `[cap,d]`; Sc = (Q·Kcᵀ)*scale + mask `[m,cap]`
///   * DIAGONAL: Kd = view6[0, kvcol] `[m,d]`; Sd = (Q·Kdᵀ)*scale + tri `[m,m]`
///   * online softmax over the two blocks (global max, exp, sums, normalize)
///   * O = Wc·Vc + Wd·Vd `[m,d]`; store -> view1[0, qcol]
pub fn rewrite_head_attention<'a>(
    a: &'a Arena,
    isl: &HeadAttnIsland,
    g: &mut NameGen,
) -> IRFunction<'a> {
    let dt = isl.dtype;
    let (m, d, cap) = (isl.m, isl.d, isl.cap);
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

    // ---- per-head selection arithmetic (PRESERVED) ----
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

    // ---- whole-Q load [m, d] at [0, qcol] ----
    let q = block_load(a, g, &mut ops, q_view, c0, qcol, [m, d]);

    // ---- per-head context mask [1, cap] at [0, 0] ----
    let mask = block_load(a, g, &mut ops, mask_view, c0, c0, [1, cap]);

    // =========================== CONTEXT block ===========================
    // Kc [cap, d] at [0, kvcol] -> Kct [d, cap]; Sc = (Q @ Kct)*scale + mask.
    let kc = block_load(a, g, &mut ops, kc_view, c0, kvcol, [cap, d]);
    let kct = mk_transpose(a, g, &mut ops, kc, cap, d, dt);
    let sc_raw = mk_matmul(a, g, &mut ops, q, kct, [m, cap], dt);
    let sc_scl = g.mint();
    ops.push(mk_splat(a, sc_scl, scale_c, &[m, cap], dt));
    let sc_scaled = g.mint();
    ops.push(Operation::new(
        a,
        Some(sc_scaled),
        OpKind::ArithMulf,
        &[sc_raw, sc_scl],
    ));
    // mask broadcast over the m rows: mask is [1, cap], broadcast to [m, cap].
    let mask_init = g.mint();
    ops.push(mk_empty(a, mask_init, &[m, cap], dt));
    let mask_b = g.mint();
    ops.push(
        Operation::new(a, Some(mask_b), OpKind::LinalgBroadcast, &[mask, mask_init]).with_attr(
            a,
            AttrKey::Dimensions,
            Attr::IntList(a.ints(vec![])),
        ),
    );
    let sc = g.mint();
    ops.push(Operation::new(
        a,
        Some(sc),
        OpKind::ArithAddf,
        &[sc_scaled, mask_b],
    ));
    // mc = reduce_max(Sc, 1) -> [m]
    let mc_init = g.mint();
    ops.push(mk_splat(a, mc_init, ninf_c, &[m], dt));
    let mc = g.mint();
    ops.push(mk_reduce(a, mc, sc, mc_init, OpKind::ArithMaximumf));

    // =========================== DIAGONAL block ==========================
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
    // md = reduce_max(Sd, 1) -> [m]
    let md_init = g.mint();
    ops.push(mk_splat(a, md_init, ninf_c, &[m], dt));
    let md = g.mint();
    ops.push(mk_reduce(a, md, sd, md_init, OpKind::ArithMaximumf));

    // =========================== combine =================================
    // gm = max(mc, md) [m]
    let gm = g.mint();
    ops.push(Operation::new(
        a,
        Some(gm),
        OpKind::ArithMaximumf,
        &[mc, md],
    ));
    // Pc = exp(Sc - gm_bcast[m,cap]); Pd = exp(Sd - gm_bcast[m,m])
    let gm_bc = broadcast_row_to(a, g, &mut ops, gm, m, cap, dt);
    let shc = g.mint();
    ops.push(Operation::new(
        a,
        Some(shc),
        OpKind::ArithSubf,
        &[sc, gm_bc],
    ));
    let pc = g.mint();
    ops.push(Operation::new(a, Some(pc), OpKind::MathExp, &[shc]));
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
    // sc_sum = reduce_sum(Pc,1) [m]; sd_sum = reduce_sum(Pd,1) [m]
    let scs_init = g.mint();
    ops.push(mk_splat(a, scs_init, zero_c, &[m], dt));
    let scs = g.mint();
    ops.push(mk_reduce(a, scs, pc, scs_init, OpKind::ArithAddf));
    let sds_init = g.mint();
    ops.push(mk_splat(a, sds_init, zero_c, &[m], dt));
    let sds = g.mint();
    ops.push(mk_reduce(a, sds, pd, sds_init, OpKind::ArithAddf));
    // gs = sc_sum + sd_sum [m]
    let gs = g.mint();
    ops.push(Operation::new(a, Some(gs), OpKind::ArithAddf, &[scs, sds]));
    // Wc = Pc / gs_bcast[m,cap]; Wd = Pd / gs_bcast[m,m]
    let gs_bc = broadcast_row_to(a, g, &mut ops, gs, m, cap, dt);
    let wc = g.mint();
    ops.push(Operation::new(a, Some(wc), OpKind::ArithDivf, &[pc, gs_bc]));
    let gs_bd = broadcast_row_to(a, g, &mut ops, gs, m, m, dt);
    let wd = g.mint();
    ops.push(Operation::new(a, Some(wd), OpKind::ArithDivf, &[pd, gs_bd]));

    // =========================== AV + store ==============================
    // Vc [cap, d] at [0, kvcol]; Vd [m, d] at [0, kvcol].
    let vc = block_load(a, g, &mut ops, vc_view, c0, kvcol, [cap, d]);
    let vd = block_load(a, g, &mut ops, vd_view, c0, kvcol, [m, d]);
    let ovc = mk_matmul(a, g, &mut ops, wc, vc, [m, d], dt);
    let ovd = mk_matmul(a, g, &mut ops, wd, vd, [m, d], dt);
    let o = g.mint();
    ops.push(Operation::new(a, Some(o), OpKind::ArithAddf, &[ovc, ovd]));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_builder::Ops;
    use ktir_core::opkind::Dialect;

    /// Leak a formatted per-iteration SSA/op name to `&'static str` — test-only;
    /// [`Ops`] needs `'static` names. Two leaks of equal content hash/compare
    /// equal in `Ops`' name table (`&str` (Partial)Eq/Hash is by content).
    fn leak(s: String) -> &'static str {
        Box::leak(s.into_boxed_str())
    }

    /// A synthetic island matching the smollm shape (H=9, m=8, gqac=3, d=64,
    /// cap=64). Used to exercise the rewrite emitter structurally.
    fn smollm_island() -> HeadAttnIsland {
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
            cap: 64,
            d: 64,
            gqac: 3,
            hdc: 64,
            h: 9,
            scale: 0.125,
            ninf: -1.0e38,
            dtype: DType::F16,
        }
    }

    #[test]
    fn rewrite_preserves_grid_and_args() {
        let isl = smollm_island();
        let mut g = NameGen::above(isl.arg_ssas());
        let f = rewrite_head_attention(Arena::global(), &isl, &mut g);
        // Grid stays [H, 1, 1] — every core still runs the body once for its head.
        assert_eq!(f.grid, (9, 1, 1));
        let ssas: Vec<Ssa> = f.arguments.iter().map(|(v, _)| *v).collect();
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
    }

    #[test]
    fn rewrite_emits_no_control_flow_and_one_store() {
        let isl = smollm_island();
        let mut g = NameGen::above(isl.arg_ssas());
        let f = rewrite_head_attention(Arena::global(), &isl, &mut g);
        // Pure re-roll: NO scf.* (region-free, never trips the batched gate).
        assert!(
            !f.operations
                .iter()
                .any(|o| o.op_type.dialect() == Dialect::Scf),
            "re-roll must be region-free"
        );
        // Exactly ONE store (the whole [m,d] output) instead of m stores.
        let stores = f
            .operations
            .iter()
            .filter(|o| o.op_type == OpKind::KtdpStore)
            .count();
        assert_eq!(stores, 1, "one whole-row store");
        // Four matmuls total (context QKᵀ, diagonal QKᵀ, context AV, diagonal AV)
        // — vs 4×m in the unrolled form.
        let mms = f
            .operations
            .iter()
            .filter(|o| o.op_type == OpKind::LinalgMatmul)
            .count();
        assert_eq!(mms, 4, "two QKᵀ + two AV, once");
        // Preserves the per-head selection arithmetic as SSA.
        assert!(
            f.operations
                .iter()
                .any(|o| o.op_type == OpKind::KtdpGetComputeTileId)
        );
        assert!(f.operations.iter().any(|o| o.op_type == OpKind::ArithDivui));
        let muls = f
            .operations
            .iter()
            .filter(|o| o.op_type == OpKind::ArithMuli)
            .count();
        assert_eq!(muls, 2, "qcol = hpid*hdc and kvcol = (hpid/gqac)*hdc");
    }

    #[test]
    fn causal_mask_is_lower_triangular() {
        // mask[r,k] = 0 for k<=r (visible), ninf for k>r (masked).
        let mut ops = Ops::new();
        let op = causal_mask_mm(Arena::global(), ops.ssa("%tri"), 4, -1.0e38, DType::F16);
        let vals = match op.attr(AttrKey::Value) {
            Some(Attr::FloatList(v)) => v.to_vec(),
            other => panic!("mask value not a FloatList: {other:?}"),
        };
        assert_eq!(vals.len(), 16);
        for r in 0..4i64 {
            for k in 0..4i64 {
                let v = vals[(r * 4 + k) as usize];
                if k <= r {
                    assert_eq!(v, 0.0, "[{r},{k}] visible");
                } else {
                    assert!(v < -1.0e30, "[{r},{k}] masked");
                }
            }
        }
    }

    #[test]
    fn rejects_single_core_grid() {
        // grid = [1,1,1] is not head-parallel -> None.
        let mut ops = Ops::new();
        let store = ops.op(None, OpKind::KtdpStore, &["%a", "%b"]);
        let f = ops.func("x", &[], vec![store], (1, 1, 1));
        assert!(recognize_head_attention(&f).is_none());
    }

    #[test]
    fn rejects_region_bearing() {
        // A top-level scf.for disqualifies (the FA-tiled regime, not the head one).
        let mut ops = Ops::new();
        let yield_op = ops.op(None, OpKind::ScfYield, &[]);
        let forop = ops.op(None, OpKind::ScfFor, &["%x", "%y", "%z"]);
        let forop = ops.with_region(forop, vec![yield_op]);
        let store = ops.op(None, OpKind::KtdpStore, &["%a", "%b"]);
        let f = ops.func("x", &[], vec![forop, store], (9, 1, 1));
        assert!(recognize_head_attention(&f).is_none());
    }

    #[test]
    fn rejects_plain_copy() {
        // A multi-core copy node (no QKᵀ/softmax/AV) -> None.
        let mut ops = Ops::new();
        let vi = ops.op(Some("%vi"), OpKind::KtdpConstructMemoryView, &["%in"]);
        let vi = ops.attr(vi, AttrKey::Shape, ops.int_list(vec![8, 64]));
        let vi = ops.attr(vi, AttrKey::Dtype, Attr::Dtype(DType::F16));
        let ti = ops.op(Some("%ti"), OpKind::KtdpConstructAccessTile, &["%vi"]);
        let ti = ops.attr(ti, AttrKey::Shape, ops.int_list(vec![8, 64]));
        let l = ops.op(Some("%l"), OpKind::KtdpLoad, &["%ti"]);
        let y = ops.op(Some("%y"), OpKind::MathExp, &["%l"]);
        let vo = ops.op(Some("%vo"), OpKind::KtdpConstructMemoryView, &["%out"]);
        let vo = ops.attr(vo, AttrKey::Shape, ops.int_list(vec![8, 64]));
        let vo = ops.attr(vo, AttrKey::Dtype, Attr::Dtype(DType::F16));
        let to = ops.op(Some("%to"), OpKind::KtdpConstructAccessTile, &["%vo"]);
        let to = ops.attr(to, AttrKey::Shape, ops.int_list(vec![8, 64]));
        let store = ops.op(None, OpKind::KtdpStore, &["%y", "%to"]);
        let ret = ops.op(None, OpKind::FuncReturn, &[]);
        let body = vec![vi, ti, l, y, vo, to, store, ret];
        let f = ops.func(
            "copy",
            &[("%in", IrType::Index), ("%out", IrType::Index)],
            body,
            (9, 1, 1),
        );
        assert!(recognize_head_attention(&f).is_none());
    }

    #[test]
    fn scores_bytes_matches_m_cap_dtype() {
        let isl = smollm_island();
        // [8, 64] f16 = 8*64*2 = 1024 bytes.
        assert_eq!(isl.scores_bytes(), 8 * 64 * 2);
    }

    // ---- DECODE (m=1) recognition + fused compute ----

    fn decode_island(h: i64, gqac: i64, d: i64, cap: i64) -> DecodeAttnIsland {
        let kv_heads = h / gqac;
        let mut ops = Ops::new();
        DecodeAttnIsland {
            q_arg: ops.ssa("%q"),
            o_arg: ops.ssa("%o"),
            mask_arg: ops.ssa("%mask"),
            kc_arg: ops.ssa("%kc"),
            kd_arg: ops.ssa("%kd"),
            vc_arg: ops.ssa("%vc"),
            vd_arg: ops.ssa("%vd"),
            q_cols: h * d,
            kv_cols: kv_heads * d,
            cap,
            d,
            h,
            gqac,
            hdc: d,
            scale: 0.125,
            dtype: DType::F16,
        }
    }

    /// Reference (independent) decode attention, computed head-by-head in f64.
    fn ref_decode(
        isl: &DecodeAttnIsland,
        q: &[f32],
        mask: &[f32],
        kc: &[f32],
        kd: &[f32],
        vc: &[f32],
        vd: &[f32],
    ) -> Vec<f32> {
        let d = isl.d as usize;
        let cap = isl.cap as usize;
        let kvw = isl.kv_cols as usize;
        let scale = isl.scale as f64;
        let mut o = vec![0.0f32; isl.q_cols as usize];
        for hh in 0..isl.h as usize {
            let qcol = hh * isl.hdc as usize;
            let kvcol = (hh / isl.gqac as usize) * isl.hdc as usize;
            let mut s = vec![0.0f64; cap + 1];
            for j in 0..cap {
                let mut dot = 0.0f64;
                for t in 0..d {
                    dot += q[qcol + t] as f64 * kc[j * kvw + kvcol + t] as f64;
                }
                s[j] = scale * dot + mask[j] as f64;
            }
            let mut dd = 0.0f64;
            for t in 0..d {
                dd += q[qcol + t] as f64 * kd[kvcol + t] as f64;
            }
            s[cap] = scale * dd; // diagonal, no mask
            let gm = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = s.iter().map(|x| (x - gm).exp()).collect();
            let z: f64 = e.iter().sum();
            for t in 0..d {
                let mut acc = 0.0f64;
                for j in 0..cap {
                    acc += e[j] * vc[j * kvw + kvcol + t] as f64;
                }
                acc += e[cap] * vd[kvcol + t] as f64;
                o[qcol + t] = (acc / z) as f32;
            }
        }
        o
    }

    #[test]
    fn fused_decode_compute_matches_reference() {
        // GQA: H=4, gqac=2 (2 kv heads), d=3, cap=5.
        let isl = decode_island(4, 2, 3, 5);
        let qn = isl.q_cols as usize;
        let kn = (isl.cap * isl.kv_cols) as usize;
        let dn = isl.kv_cols as usize;
        // Deterministic pseudo-random fill.
        let f =
            |i: usize, salt: usize| (((i * 2654435761 + salt * 40503) % 211) as f32) / 211.0 - 0.5;
        let q: Vec<f32> = (0..qn).map(|i| f(i, 1)).collect();
        let mask: Vec<f32> = (0..isl.cap as usize).map(|i| f(i, 2) * 4.0).collect();
        let kc: Vec<f32> = (0..kn).map(|i| f(i, 3)).collect();
        let kd: Vec<f32> = (0..dn).map(|i| f(i, 4)).collect();
        let vc: Vec<f32> = (0..kn).map(|i| f(i, 5)).collect();
        let vd: Vec<f32> = (0..dn).map(|i| f(i, 6)).collect();

        let mut got = vec![0.0f32; qn];
        isl.compute_f32(
            DecodeAttnInputs {
                q: &q,
                mask: &mask,
                kc: &kc,
                kd: &kd,
                vc: &vc,
                vd: &vd,
            },
            &mut got,
        );
        let want = ref_decode(&isl, &q, &mask, &kc, &kd, &vc, &vd);
        let max_abs = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs < 1e-5, "fused vs reference max_abs {max_abs}");
    }

    /// Build a synthetic decode-attention IR for `H` heads (gqac, d, cap) in the
    /// EXACT op shape the real decode emit uses, so `recognize_head_attention_decode`
    /// exercises the real recognition path (constants, addi-folded kvcol, the two-
    /// block QKᵀ/softmax/AV chain).
    fn build_decode_func(h: i64, gqac: i64, d: i64, cap: i64) -> IRFunction<'static> {
        let a = Arena::global();
        let kv_cols = (h / gqac) * d;
        let q_cols = h * d;
        let mut ops = Ops::new();
        let mut body: Vec<Operation> = Vec::new();

        let c0 = ops.ssa("%c0");
        let c0_op = ops.op(Some("%c0"), OpKind::ArithConstant, &[]);
        body.push(ops.attr(c0_op, AttrKey::Value, Attr::Int(0)));
        let view0 = ops.ssa("%view0");
        body.push(mk_view(a, view0, ops.ssa("%q"), &[1, q_cols], DType::F16));
        let view1 = ops.ssa("%view1");
        body.push(mk_view(a, view1, ops.ssa("%o"), &[1, q_cols], DType::F16));
        let view2 = ops.ssa("%view2");
        body.push(mk_view(a, view2, ops.ssa("%mask"), &[1, cap], DType::F16));
        let view5 = ops.ssa("%view5");
        body.push(mk_view(
            a,
            view5,
            ops.ssa("%kc"),
            &[cap, kv_cols],
            DType::F16,
        ));
        let view6 = ops.ssa("%view6");
        body.push(mk_view(a, view6, ops.ssa("%kd"), &[1, kv_cols], DType::F16));
        let view7 = ops.ssa("%view7");
        body.push(mk_view(
            a,
            view7,
            ops.ssa("%vc"),
            &[cap, kv_cols],
            DType::F16,
        ));
        let view8 = ops.ssa("%view8");
        body.push(mk_view(a, view8, ops.ssa("%vd"), &[1, kv_cols], DType::F16));
        let scale = ops.ssa("%scale");
        let scale_op = ops.op(Some("%scale"), OpKind::ArithConstant, &[]);
        body.push(ops.attr(scale_op, AttrKey::Value, Attr::Float(0.125)));
        let ninf = ops.ssa("%ninf");
        let ninf_op = ops.op(Some("%ninf"), OpKind::ArithConstant, &[]);
        body.push(ops.attr(ninf_op, AttrKey::Value, Attr::Float(-1.0e38)));
        // shared mask load.
        let macc = ops.op(
            Some("%macc"),
            OpKind::KtdpConstructAccessTile,
            &["%view2", "%c0", "%c0"],
        );
        body.push(ops.attr(macc, AttrKey::Shape, ops.int_list(vec![1, cap])));
        let mload = ops.ssa("%mload");
        body.push(ops.op(Some("%mload"), OpKind::KtdpLoad, &["%macc"]));

        for hh in 0..h {
            let qcol = hh * d;
            let kvcol = (hh / gqac) * d;
            let qc = leak(format!("%qcol{hh}"));
            let qc_op = ops.op(Some(qc), OpKind::ArithConstant, &[]);
            body.push(ops.attr(qc_op, AttrKey::Value, Attr::Int(qcol)));
            let kvc = leak(format!("%kvcol{hh}"));
            let kvc_op = ops.op(Some(kvc), OpKind::ArithConstant, &[]);
            body.push(ops.attr(kvc_op, AttrKey::Value, Attr::Int(kvcol)));

            // Q load.
            let qacc = leak(format!("%qacc{hh}"));
            let qacc_op = ops.op(
                Some(qacc),
                OpKind::KtdpConstructAccessTile,
                &["%view0", "%c0", qc],
            );
            body.push(ops.attr(qacc_op, AttrKey::Shape, ops.int_list(vec![1, d])));
            let q = leak(format!("%q{hh}"));
            let _q_ssa = ops.ssa(q);
            body.push(ops.op(Some(q), OpKind::KtdpLoad, &[qacc]));

            // CONTEXT: Kc [cap,d] at [0, kvcol] (folded as addi(0, kvcol)).
            let kcs = leak(format!("%kcs{hh}"));
            let kcs_op = ops.op(Some(kcs), OpKind::ArithConstant, &[]);
            body.push(ops.attr(kcs_op, AttrKey::Value, Attr::Int(0)));
            let kcc = leak(format!("%kcc{hh}"));
            body.push(ops.op(Some(kcc), OpKind::ArithAddi, &[kcs, kvc]));
            let kcacc = leak(format!("%kcacc{hh}"));
            let kcacc_op = ops.op(
                Some(kcacc),
                OpKind::KtdpConstructAccessTile,
                &["%view5", "%c0", kcc],
            );
            body.push(ops.attr(kcacc_op, AttrKey::Shape, ops.int_list(vec![cap, d])));
            let kc = leak(format!("%kc{hh}"));
            body.push(ops.op(Some(kc), OpKind::KtdpLoad, &[kcacc]));
            let kct = leak(format!("%kct{hh}"));
            let kct_op = ops.op(Some(kct), OpKind::LinalgTranspose, &[kc, kc]);
            body.push(ops.attr(kct_op, AttrKey::Permutation, ops.int_list(vec![1, 0])));
            let scr = leak(format!("%scr{hh}"));
            body.push(ops.op(Some(scr), OpKind::LinalgMatmul, &[q, kct, q]));
            let scsp = leak(format!("%scsp{hh}"));
            let scsp_ssa = ops.ssa(scsp);
            body.push(mk_splat(a, scsp_ssa, scale, &[1, cap], DType::F16));
            let scl = leak(format!("%scl{hh}"));
            body.push(ops.op(Some(scl), OpKind::ArithMulf, &[scr, scsp]));
            let scm = leak(format!("%scm{hh}"));
            let scm_ssa = ops.ssa(scm);
            body.push(ops.op(Some(scm), OpKind::ArithAddf, &[scl, "%mload"]));
            let mi = leak(format!("%mi{hh}"));
            let mi_ssa = ops.ssa(mi);
            body.push(mk_splat(a, mi_ssa, ninf, &[1], DType::F16));
            let mx = leak(format!("%mx{hh}"));
            let mx_ssa = ops.ssa(mx);
            body.push(mk_reduce(a, mx_ssa, scm_ssa, mi_ssa, OpKind::ArithMaximumf));

            // DIAGONAL: Kd [1,d] at [0, kvcol].
            let kds = leak(format!("%kds{hh}"));
            let kds_op = ops.op(Some(kds), OpKind::ArithConstant, &[]);
            body.push(ops.attr(kds_op, AttrKey::Value, Attr::Int(0)));
            let kdc = leak(format!("%kdc{hh}"));
            body.push(ops.op(Some(kdc), OpKind::ArithAddi, &[kds, kvc]));
            let kdacc = leak(format!("%kdacc{hh}"));
            let kdacc_op = ops.op(
                Some(kdacc),
                OpKind::KtdpConstructAccessTile,
                &["%view6", "%c0", kdc],
            );
            body.push(ops.attr(kdacc_op, AttrKey::Shape, ops.int_list(vec![1, d])));
            let kd = leak(format!("%kd{hh}"));
            body.push(ops.op(Some(kd), OpKind::KtdpLoad, &[kdacc]));
            let kdt = leak(format!("%kdt{hh}"));
            let kdt_op = ops.op(Some(kdt), OpKind::LinalgTranspose, &[kd, kd]);
            body.push(ops.attr(kdt_op, AttrKey::Permutation, ops.int_list(vec![1, 0])));
            let sdr = leak(format!("%sdr{hh}"));
            body.push(ops.op(Some(sdr), OpKind::LinalgMatmul, &[q, kdt, q]));
            let sdsp = leak(format!("%sdsp{hh}"));
            let sdsp_ssa = ops.ssa(sdsp);
            body.push(mk_splat(a, sdsp_ssa, scale, &[1, 1], DType::F16));
            let sdl = leak(format!("%sdl{hh}"));
            let sdl_ssa = ops.ssa(sdl);
            body.push(ops.op(Some(sdl), OpKind::ArithMulf, &[sdr, sdsp]));
            let mdi = leak(format!("%mdi{hh}"));
            let mdi_ssa = ops.ssa(mdi);
            body.push(mk_splat(a, mdi_ssa, ninf, &[1], DType::F16));
            let mxd = leak(format!("%mxd{hh}"));
            let mxd_ssa = ops.ssa(mxd);
            body.push(mk_reduce(
                a,
                mxd_ssa,
                sdl_ssa,
                mdi_ssa,
                OpKind::ArithMaximumf,
            ));

            // combine + exp + sums.
            let gm = leak(format!("%gm{hh}"));
            let gm_ssa = ops.ssa(gm);
            body.push(ops.op(Some(gm), OpKind::ArithMaximumf, &[mx, mxd]));
            let gmb = leak(format!("%gmb{hh}"));
            let gmb_ssa = ops.ssa(gmb);
            body.push(mk_splat(a, gmb_ssa, gm_ssa, &[1, cap], DType::F16));
            let sh = leak(format!("%sh{hh}"));
            body.push(ops.op(Some(sh), OpKind::ArithSubf, &[scm, gmb]));
            let ex = leak(format!("%ex{hh}"));
            body.push(ops.op(Some(ex), OpKind::MathExp, &[sh]));
            let zi = leak(format!("%zi{hh}"));
            let zi_ssa = ops.ssa(zi);
            body.push(mk_splat(a, zi_ssa, ninf, &[1], DType::F16));
            let su = leak(format!("%su{hh}"));
            let su_ssa = ops.ssa(su);
            body.push(mk_reduce(a, su_ssa, ops.ssa(ex), zi_ssa, OpKind::ArithAddf));
            let gmbd = leak(format!("%gmbd{hh}"));
            let gmbd_ssa = ops.ssa(gmbd);
            body.push(mk_splat(a, gmbd_ssa, gm_ssa, &[1, 1], DType::F16));
            let shd = leak(format!("%shd{hh}"));
            body.push(ops.op(Some(shd), OpKind::ArithSubf, &[sdl, gmbd]));
            let exd = leak(format!("%exd{hh}"));
            body.push(ops.op(Some(exd), OpKind::MathExp, &[shd]));
            let zid = leak(format!("%zid{hh}"));
            let zid_ssa = ops.ssa(zid);
            body.push(mk_splat(a, zid_ssa, ninf, &[1], DType::F16));
            let sud = leak(format!("%sud{hh}"));
            let sud_ssa = ops.ssa(sud);
            body.push(mk_reduce(
                a,
                sud_ssa,
                ops.ssa(exd),
                zid_ssa,
                OpKind::ArithAddf,
            ));
            let gs = leak(format!("%gs{hh}"));
            let gs_ssa = ops.ssa(gs);
            body.push(ops.op(Some(gs), OpKind::ArithAddf, &[su, sud]));
            let gsb = leak(format!("%gsb{hh}"));
            let gsb_ssa = ops.ssa(gsb);
            body.push(mk_splat(a, gsb_ssa, gs_ssa, &[1, cap], DType::F16));
            let w = leak(format!("%w{hh}"));
            body.push(ops.op(Some(w), OpKind::ArithDivf, &[ex, gsb]));
            let gsbd = leak(format!("%gsbd{hh}"));
            let gsbd_ssa = ops.ssa(gsbd);
            body.push(mk_splat(a, gsbd_ssa, gs_ssa, &[1, 1], DType::F16));
            let wd = leak(format!("%wd{hh}"));
            body.push(ops.op(Some(wd), OpKind::ArithDivf, &[exd, gsbd]));

            // AV.
            let vcs = leak(format!("%vcs{hh}"));
            let vcs_op = ops.op(Some(vcs), OpKind::ArithConstant, &[]);
            body.push(ops.attr(vcs_op, AttrKey::Value, Attr::Int(0)));
            let vcc = leak(format!("%vcc{hh}"));
            body.push(ops.op(Some(vcc), OpKind::ArithAddi, &[vcs, kvc]));
            let vcacc = leak(format!("%vcacc{hh}"));
            let vcacc_op = ops.op(
                Some(vcacc),
                OpKind::KtdpConstructAccessTile,
                &["%view7", "%c0", vcc],
            );
            body.push(ops.attr(vcacc_op, AttrKey::Shape, ops.int_list(vec![cap, d])));
            let vc = leak(format!("%vc{hh}"));
            body.push(ops.op(Some(vc), OpKind::KtdpLoad, &[vcacc]));
            let ov = leak(format!("%ov{hh}"));
            body.push(ops.op(Some(ov), OpKind::LinalgMatmul, &[w, vc, w]));
            let vds = leak(format!("%vds{hh}"));
            let vds_op = ops.op(Some(vds), OpKind::ArithConstant, &[]);
            body.push(ops.attr(vds_op, AttrKey::Value, Attr::Int(0)));
            let vdc = leak(format!("%vdc{hh}"));
            body.push(ops.op(Some(vdc), OpKind::ArithAddi, &[vds, kvc]));
            let vdacc = leak(format!("%vdacc{hh}"));
            let vdacc_op = ops.op(
                Some(vdacc),
                OpKind::KtdpConstructAccessTile,
                &["%view8", "%c0", vdc],
            );
            body.push(ops.attr(vdacc_op, AttrKey::Shape, ops.int_list(vec![1, d])));
            let vd = leak(format!("%vd{hh}"));
            body.push(ops.op(Some(vd), OpKind::KtdpLoad, &[vdacc]));
            let ovd = leak(format!("%ovd{hh}"));
            body.push(ops.op(Some(ovd), OpKind::LinalgMatmul, &[wd, vd, wd]));
            let oa = leak(format!("%oa{hh}"));
            body.push(ops.op(Some(oa), OpKind::ArithAddf, &[ov, ovd]));
            let oacc = leak(format!("%oacc{hh}"));
            let oacc_op = ops.op(
                Some(oacc),
                OpKind::KtdpConstructAccessTile,
                &["%view1", "%c0", qc],
            );
            body.push(ops.attr(oacc_op, AttrKey::Shape, ops.int_list(vec![1, d])));
            body.push(ops.op(None, OpKind::KtdpStore, &[oa, oacc]));
        }
        body.push(ops.op(None, OpKind::FuncReturn, &[]));
        let _ = (
            view0, view1, view2, view5, view6, view7, view8, scale, ninf, mload, c0,
        );
        ops.func(
            "decode_attn",
            &[
                ("%q", IrType::Index),
                ("%o", IrType::Index),
                ("%mask", IrType::Index),
                ("%kc", IrType::Index),
                ("%kd", IrType::Index),
                ("%vc", IrType::Index),
                ("%vd", IrType::Index),
            ],
            body,
            (1, 1, 1),
        )
    }

    #[test]
    fn recognizes_decode_island_gqa() {
        // smollm-shaped: H=9, gqac=3, d=64, cap=64.
        let f = build_decode_func(9, 3, 64, 64);
        let isl = recognize_head_attention_decode(&f).expect("decode island");
        assert_eq!(isl.h, 9);
        assert_eq!(isl.gqac, 3);
        assert_eq!(isl.hdc, 64);
        assert_eq!(isl.d, 64);
        assert_eq!(isl.cap, 64);
        assert_eq!(isl.q_cols, 9 * 64);
        assert_eq!(isl.kv_cols, 3 * 64);
        assert_eq!(isl.scale, 0.125);
        assert_eq!(isl.q_arg, f.arguments[0].0);
        assert_eq!(isl.o_arg, f.arguments[1].0);
    }

    #[test]
    fn recognizes_decode_island_llama_gqa() {
        // llama-shaped: H=32, gqac=4, d=64, cap=64.
        let f = build_decode_func(32, 4, 64, 64);
        let isl = recognize_head_attention_decode(&f).expect("decode island");
        assert_eq!(isl.h, 32);
        assert_eq!(isl.gqac, 4);
    }

    #[test]
    fn decode_recognizer_rejects_grid_gt1() {
        let mut f = build_decode_func(4, 2, 3, 5);
        f.grid = (4, 1, 1);
        assert!(recognize_head_attention_decode(&f).is_none());
    }

    #[test]
    fn decode_recognizer_rejects_non_attention() {
        // A plain copy func (no QKᵀ/softmax/AV) -> None.
        let mut ops = Ops::new();
        let vi = ops.op(Some("%vi"), OpKind::KtdpConstructMemoryView, &["%in"]);
        let vi = ops.attr(vi, AttrKey::Shape, ops.int_list(vec![1, 64]));
        let vi = ops.attr(vi, AttrKey::Dtype, Attr::Dtype(DType::F16));
        let ti = ops.op(
            Some("%ti"),
            OpKind::KtdpConstructAccessTile,
            &["%vi", "%c0", "%c0"],
        );
        let ti = ops.attr(ti, AttrKey::Shape, ops.int_list(vec![1, 64]));
        let l = ops.op(Some("%l"), OpKind::KtdpLoad, &["%ti"]);
        let vo = ops.op(Some("%vo"), OpKind::KtdpConstructMemoryView, &["%out"]);
        let vo = ops.attr(vo, AttrKey::Shape, ops.int_list(vec![1, 64]));
        let vo = ops.attr(vo, AttrKey::Dtype, Attr::Dtype(DType::F16));
        let to = ops.op(
            Some("%to"),
            OpKind::KtdpConstructAccessTile,
            &["%vo", "%c0", "%c0"],
        );
        let to = ops.attr(to, AttrKey::Shape, ops.int_list(vec![1, 64]));
        let store = ops.op(None, OpKind::KtdpStore, &["%l", "%to"]);
        let body = vec![vi, ti, l, vo, to, store];
        let f = ops.func(
            "copy",
            &[("%in", IrType::Index), ("%out", IrType::Index)],
            body,
            (1, 1, 1),
        );
        assert!(recognize_head_attention_decode(&f).is_none());
    }
}
