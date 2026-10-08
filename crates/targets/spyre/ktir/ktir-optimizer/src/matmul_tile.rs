// SPDX-License-Identifier: Apache-2.0
//! Matmul LX-FIT TILING pass — `M` across the grid, `N` in column blocks, `K` as an accumulating
//! loop.
//!
//! ⭐⭐⭐ THIS IS A SCHEDULING DECISION, AND IT BELONGS TO THE EMULATOR ALONE. The producer
//! (`lower_subtile_tape_to_ktir.rs`, now the Triton splice) emits ONE untiled `linalg.matmul`
//! over the whole `[m, k] × [k, n]`, because that is what the IR means. Who tiles it, and how, is a
//! property of the DEVICE that runs it:
//!
//! * The **emulator** executes the ops itself against a real 2 MB LX, so it cannot hold
//!   `W[2048, 2048]` fp16 (8 MB) as one tile. It needs this pass, and the numbers here
//!   ([`k_block`]/[`n_block`]) are the ones it was measured against.
//! * The **card** does not. `SubtileIR → SuperDSC` hands `assemble_matmul` a whole GEMM and the
//!   work division is DECLARED, not built: `WorkPlan::divide` fills `numWkSlicesPerDim_` for dxp's
//!   scheduler and `WorkPlan::time_tile_for_lx` fills `OpSpec.time_tile`, which `render_dxp_input`
//!   expands into trips. main's `lower_matmul_node` says it outright — "the Spyre tape does NOT
//!   K-chunk (each `MatmulTile` is a whole GEMM; SuperDSC owns the K-split via the cost model)".
//!
//! ⛔ SO A PRE-TILED KTIR WAS A BUG FOR ONE OF ITS TWO CONSUMERS. The nest below used to be built
//! during KTIR construction, which meant `KTIR → SuperDSC` received a 64-trip `scf.for` where the
//! proven emitter wanted one contraction — an `ScfFor` it had no mapping for, because there is no
//! SuperDSC op that means "loop". Un-tiling it in the KTIR→SuperDSC direction would have meant
//! pattern-matching a tiling back out of the IR; emitting untiled KTIR and tiling HERE, for the one
//! consumer that needs it, is the same decision made in the right place.
//!
//! ⭐ MOVED VERBATIM, and that is the correctness argument. Every extent, block size, op order and
//! attribute below is what the producer's `KtirFunc::matmul` built before (the splice reproduces it
//! byte-for-byte), so the emulator sees the same KTIR it already runs at its measured rate. This
//! pass is a relocation, not a redesign.

use crate::head_rewrite::NameGen;
use ktir_core::affine::{AffineExpr, AffineMap};
use ktir_core::arena::Arena;
use ktir_core::attrkey::AttrKey;
use ktir_core::dtypes::DType;
use ktir_core::ir::{Attr, IRFunction, IRModule, Operation, Ssa};
use ktir_core::irtype::IrType;
use ktir_core::opkind::OpKind;
use std::collections::HashMap;

/// K-block width, verbatim from the construction this replaces.
///
/// Cap the W tile small: across sequential N-blocks each block leaves its accumulator and zero-init
/// in scope, and the emulator reclaims LX only at scope exit.
fn k_block(k: i64, n: i64) -> i64 {
    const W_TILE_CAP_BYTES: i64 = 400_000;
    let max_kb = (W_TILE_CAP_BYTES / (n * 4)).max(1);
    let mut kb = 1;
    let mut cand = 2;
    while cand <= max_kb && k % cand == 0 {
        kb = cand;
        cand *= 2;
    }
    kb
}

/// Output column-block width, verbatim from the construction this replaces.
///
/// The GEMM offload holds the `[m, bw]` accumulator (plus its K-loop siblings and the cross-block
/// residue) resident in the 2 MB LX, so the constraint is on `m · bw`, NOT `n` alone.
fn n_block(n: i64, m: i64) -> i64 {
    const WHOLE_N_FITS_M1: i64 = 90_000;
    const BLOCK_N: i64 = 16_384;
    const BLOCK_MN_BUDGET: i64 = 512 * 1024;
    let m = m.max(1);
    if m.saturating_mul(n) <= WHOLE_N_FITS_M1 {
        n
    } else {
        BLOCK_N.min(BLOCK_MN_BUDGET / m).max(1)
    }
}

/// One untiled contraction, as this pass needs to see it: the whole-GEMM `linalg.matmul` plus the
/// view/tile/load chain feeding it and the `ktdp.store` draining it.
struct Untiled {
    /// Index of the `linalg.matmul` in `func.operations`.
    at: usize,
    /// Index of the `ktdp.store` that writes its result.
    store_at: usize,
    a_view: Ssa,
    w_view: Ssa,
    out_view: Ssa,
    /// The activation's row corner — an `scf`-free `index`, carried through unchanged.
    a_row: Ssa,
    /// The untiled contraction's `outs` seed — a `tensor.splat` of the zero scalar. ⛔ THE SCALAR
    /// IS REUSED, NOT THE SPLAT: the loop body's seeds must be shaped `[1, bw]` (one output row
    /// of one N-block), while the untiled splat spans the whole `[m, n]` product. Reusing the
    /// SPLAT itself — as this pass originally did, reading "reused, not rebuilt" one level too
    /// far — is only shape-correct at `m == 1` with one N-block, i.e. decode. On the CPU
    /// interpreter the per-iteration `linalg.matmul` then refuses with `outs shape [31, 512] !=
    /// product shape [1, 512]` (the Metal/NAX offload never runs the body, so macOS never saw
    /// it — the Linux CI gate did). The splat's SCALAR operand, however, is what must be kept
    /// alive: the splat zeroes an immediate scalar (`scalar(0.0)`), and minting a
    /// fresh constant in place of the operand would orphan the original splat's parameter —
    /// the `dce` below then drops its view chain and the emulator refuses with `no shape
    /// derivable for tensor t4294967275` (`u32::MAX - 20`, registry slot 0).
    init_scalar: Ssa,
    m: i64,
    n: i64,
    k: i64,
    elem: DType,
    /// ⭐ THE DEQUANT SCALE, when the store drains a trailing `arith.mulf(matmul, scale)` instead
    /// of the matmul itself. Both fp8 producers end that way — the deleted builder's
    /// `KtirFunc::matmul_fp8` and the spliced `matmul_fp8_fwd` (whose ladder lowering spells the
    /// same `p * ws` the kernel states) — and WITHOUT recognizing it the whole `[n, k]` weight
    /// tile loads UNTILED: at granite-2b's q_proj that is a `[2048, 2048]` fp8 tile (4 MB)
    /// against a 2 MB LX, and the MEASURED failure is `KtdpLoad: LX capacity exceeded
    /// 4194304 over budget 2097152 charging %21 tile [2048, 2048]` — on the BUILDER path
    /// identically (the splice falls through to it at prefill m=31), so this is the fp8 E2E
    /// wall for both producers. `None` for the dense contraction (store drains the matmul
    /// directly), which keeps every dense program byte-identical to what this pass emitted
    /// before the field existed.
    scale: Option<Ssa>,
    /// The matmul's own result SSA — the operand the surviving scaled epilogue's
    /// mulf must be re-pointed from, onto the nest's result.
    matmul_res: Ssa,
}

fn shape_of(op: &Operation<'_>) -> Option<Vec<i64>> {
    op.attributes
        .iter()
        .find(|(k, _)| *k == AttrKey::Shape)
        .and_then(|(_, v)| match v {
            Attr::IntList(l) => Some(l.to_vec()),
            _ => None,
        })
        .or_else(|| op.result_type.and_then(|t| t.dims().map(|d| d.to_vec())))
}

/// Recognize the untiled form [`Self`] tiles. Deliberately narrow: it matches exactly what the
/// producer's matmul emits and nothing else, so an unrecognized matmul is LEFT ALONE rather than
/// rewritten on a guess.
fn recognize(func: &IRFunction<'_>) -> Vec<Untiled> {
    let def: HashMap<Ssa, (usize, &Operation<'_>)> = func
        .operations
        .iter()
        .enumerate()
        .filter_map(|(i, op)| op.result.map(|r| (r, (i, op))))
        .collect();
    // `ktdp.load` → the access tile it reads → the view that tile windows.
    let through_load = |v: Ssa| -> Option<(Ssa, Ssa, Vec<i64>)> {
        let (_, ld) = def.get(&v)?;
        if ld.op_type != OpKind::KtdpLoad {
            return None;
        }
        let (_, acc) = def.get(ld.operands.first()?)?;
        if acc.op_type != OpKind::KtdpConstructAccessTile {
            return None;
        }
        let view = *acc.operands.first()?;
        let corner = *acc.operands.get(1)?;
        Some((view, corner, shape_of(acc)?))
    };
    // ⭐ THE WSCALE READ, AS THE REAL PROGRAMS SPELL IT: the loaded `[1, n]` row
    // flows through `TensorCollapseShape` (`[1, n] → [n]`, the fusion's
    // canonical 1-D form) and `LinalgBroadcast` (the mulf's own broadcast to the
    // result's shape) before the mulf sees it. Unwrap both — MEASURED on
    // granite-3.1-2b-fp8's `matmul_s*` functions, where the chain is
    // `mulf ← broadcast ← collapse ← load(access_tile(view))`.
    let through_scale = |v: Ssa| -> Option<Vec<i64>> {
        let mut v = v;
        for _ in 0..4 {
            let (_, op) = def.get(&v)?;
            match op.op_type {
                OpKind::LinalgBroadcast | OpKind::TensorCollapseShape => {
                    v = *op.operands.first()?;
                }
                OpKind::KtdpLoad => {
                    let (_, acc) = def.get(op.operands.first()?)?;
                    if acc.op_type != OpKind::KtdpConstructAccessTile {
                        return None;
                    }
                    return shape_of(acc);
                }
                _ => return None,
            }
        }
        None
    };

    let mut out = Vec::new();
    for (i, op) in func.operations.iter().enumerate() {
        if op.op_type != OpKind::LinalgMatmul {
            continue;
        }
        let (Some(&av), Some(&wv), Some(&init)) =
            (op.operands.first(), op.operands.get(1), op.operands.get(2))
        else {
            continue;
        };
        // The `outs` seed must be a `tensor.splat` over a scalar — the form the
        // producer's zero-seed emits. The SCALAR is what the rewrite reuses;
        // the splat itself is `[m, n]`-shaped and gets rebuilt per N-block.
        let Some(&init_scalar) = def.get(&init).and_then(|(_, o)| o.operands.first()) else {
            continue;
        };
        if def.get(&init).map(|(_, o)| o.op_type) != Some(OpKind::TensorSplat) {
            continue;
        }
        let (Some((a_view, a_row, a_dims)), Some((w_view, _, w_dims))) =
            (through_load(av), through_load(wv))
        else {
            continue;
        };
        let Some(res) = op.result else { continue };
        // ⭐ THE STORE DRAINS EITHER THE MATMUL OR ITS DEQUANT SCALE — the dense
        // program stores the contraction directly, while BOTH fp8 producers (the deleted
        // builder's `KtirFunc::matmul_fp8`,
        // the spliced `matmul_fp8_fwd`) end `linalg.matmul → arith.mulf(part, scale) →
        // ktdp.store`. The mulf is admitted ONLY as the exact fp8 epilogue: one operand must
        // BE the matmul result and the other a loaded `[1, n]` scale row whose width is this
        // contraction's n — anything else (a fused activation multiplier, a residual add) is a
        // different epilogue this rewrite must not touch, so it disqualifies the node. A
        // non-fp8 example exists and is exactly that: attention's `scores = matmul(q, k) ·
        // (1/sqrt(hd))` multiplies by a SPLAT, not a loaded row, so it never matches the
        // loaded-scale-row shape test and keeps the direct-store recognition.
        let (store_val, scale) = {
            let direct = func
                .operations
                .iter()
                .find(|o| o.op_type == OpKind::KtdpStore && o.operands.first() == Some(&res));
            match direct {
                Some(_) => (res, None),
                None => {
                    // Find the mulf that consumes `res`, then the store that drains it.
                    let Some(mulf) = func
                        .operations
                        .iter()
                        .find(|o| o.op_type == OpKind::ArithMulf && o.operands.contains(&res))
                    else {
                        continue;
                    };
                    let other = if mulf.operands[0] == res {
                        mulf.operands[1]
                    } else {
                        mulf.operands[0]
                    };
                    // The scale side must be a loaded `[1, n]` row — the load chain
                    // every fp8 producer's wscale read states (the builder's
                    // `tile(scale_view, 0, 0, 1, n)`; the ladder's ws descriptor `[1, N]`),
                    // possibly through collapse/broadcast (see `through_scale`).
                    let Some(s_dims) = through_scale(other) else {
                        continue;
                    };
                    // Rank 2, one row, whose width must equal the contraction's n
                    // (W's first extent — checked where n is bound, below).
                    if s_dims.len() != 2 || s_dims[0] != 1 {
                        continue;
                    }
                    let Some(mulf_res) = mulf.result else {
                        continue;
                    };
                    if !func.operations.iter().any(|o| {
                        o.op_type == OpKind::KtdpStore && o.operands.first() == Some(&mulf_res)
                    }) {
                        continue;
                    }
                    (mulf_res, Some(other))
                }
            }
        };
        // The store that drains it, and the view it writes.
        let Some((store_at, st)) = func.operations.iter().enumerate().find(|(_, o)| {
            o.op_type == OpKind::KtdpStore && o.operands.first() == Some(&store_val)
        }) else {
            continue;
        };
        let Some((_, out_acc)) = st.operands.get(1).and_then(|t| def.get(t)) else {
            continue;
        };
        let Some(&out_view) = out_acc.operands.first() else {
            continue;
        };
        let (Some(&m), Some(&k)) = (a_dims.first(), a_dims.get(1)) else {
            continue;
        };
        // ⭐ THE W TILE IS THE BUILDER'S OWN `[n, k]` FRAMING (transpose-B) — every
        // producer this pass rewrites (the deleted builder's `KtirFunc::matmul`, the spliced
        // kernels, whose `.T` `dot_to_linalg` folds into the same transpose-B maps)
        // loads the weight as its on-disk `[n, k]` region. A `[k, n]` W tile is a
        // foreign plain-B form — not ours, left alone rather than contracted the
        // wrong way round.
        let (Some(&n), Some(&wk)) = (w_dims.first(), w_dims.get(1)) else {
            continue;
        };
        debug_assert_eq!(k, wk, "the recognized W tile's k matches A's");
        // ⛔ THE SCALE ROW'S WIDTH MUST BE THIS n. The surviving mulf broadcasts the
        // `[1, n]` row against the `[m, n]` result, so a loaded `[1, x]` row with
        // x != n is a DIFFERENT epilogue — it disqualifies the node rather than
        // misbroadcasting.
        if let Some(scale_ssa) = scale {
            let s_dims = through_scale(scale_ssa);
            if s_dims
                .as_deref()
                .map(|d| d.len() != 2 || d[0] != 1 || d[1] != n)
                != Some(false)
            {
                continue;
            }
        }
        let elem = op.result_type.and_then(|t| t.elem()).unwrap_or(DType::F16);
        // ⛔ THE SCALED FORM NEEDS ONE N-BLOCK. The rewrite replaces only the matmul
        // with the K-loop nest and leaves the surviving `arith.mulf(result, scale)`
        // untouched (below) — and that mulf broadcasts the `[1, n]` scale row against
        // the matmul result, which is only well-formed when the result spans the WHOLE
        // output width. A multi-N-block nest yields `[m, bw]` partial products whose
        // mulf against the `[1, n]` row would silently misbroadcast. Dense programs
        // keep any N-blocking they had (the store tiles each block separately); the
        // scaled form takes exactly one block, which is the fp8 door's own
        // single-tile contract (`verify_canonical_fp8_matmul_kernel` refuses
        // `BLOCK_N < N`).
        if scale.is_some() && n_block(n, m) < n {
            continue;
        }
        out.push(Untiled {
            at: i,
            store_at,
            a_view,
            w_view,
            out_view,
            a_row,
            init_scalar,
            m,
            n,
            k,
            elem,
            scale,
            matmul_res: res,
        });
    }
    out
}

fn const_index<'a>(a: &'a Arena, res: Ssa, v: i64) -> Operation<'a> {
    let mut op = Operation::new(a, Some(res), OpKind::ArithConstant, &[]).with_attr(
        a,
        AttrKey::Value,
        Attr::Int(v),
    );
    op.result_type = Some(IrType::Index);
    op
}

/// Tile the recognized contractions of one function, in place.
fn tile_func<'a>(a: &'a Arena, func: &mut IRFunction<'a>) -> usize {
    let found = recognize(func);
    // ⛔ A CONTRACTION THIS PASS DOES NOT RECOGNISE IS A LATENT LX OVERFLOW, so it is reported
    // rather than skipped quietly. The emulator's GEMM offload reads a K-loop's access-tile row
    // index as an OFFSET (`matmul_a_row_offset` → `base + m_row_off · k`); an UNTILED contraction
    // takes the generic tile path instead, which bounds-checks the window against its view and
    // refuses a row-sliced activation — MEASURED as `construct_access_tile: window [1, 576] at base
    // [30, 0] runs to 31 in dim 0 of a parent view [1, 576]`, the prefill lm-head tail.
    let total = func
        .operations
        .iter()
        .filter(|o| o.op_type == OpKind::LinalgMatmul)
        .count();
    if total != found.len() && std::env::var_os("KTIR_REWRITE_VERBOSE").is_some() {
        eprintln!(
            "[ktir-optimizer] matmul tiling: {} of {total} contraction(s) in `{}` recognised — the \
             rest stay UNTILED and will take the generic (bounds-checked) tile path",
            found.len(),
            func.name,
        );
    }
    if found.is_empty() {
        return 0;
    }
    let mut g = NameGen::after(func);
    let mut done = 0usize;
    // Rewrite back-to-front so earlier indices stay valid.
    let mut plans = found;
    plans.sort_by_key(|p| std::cmp::Reverse(p.at));

    for p in plans {
        let elem = p.elem;
        let tensor = |dims: Vec<i64>| IrType::Tensor {
            dims: a.ints(dims),
            elem,
        };
        let mut pre: Vec<Operation<'a>> = Vec::new();

        // ⭐ M ACROSS THE GRID. Cores that split the CONTRACTION would hold partial products only a
        // shared PSUM could sum, and KTIR states no cross-core accumulation — so `m` is the grid and
        // `k` is a loop. At `m == 1` the grid stays `[1,1]` and the row corner is the one the
        // untiled form already carried, which is byte-identical to the decode path.
        let row_idx = if p.m > 1 {
            let pid = g.mint();
            let mut op = Operation::new(a, Some(pid), OpKind::KtdpGetComputeTileId, &[]);
            op.result_type = Some(IrType::Index);
            pre.push(op);
            func.grid = (p.m as usize, 1, 1);
            pid
        } else {
            p.a_row
        };

        let nblk = n_block(p.n, p.m);
        let mut n_off = 0i64;
        // The nest's result SSA (the loop's yield) — the scaled form has exactly
        // one N-block, so the single loop's result IS the contraction the
        // surviving mulf scales. Captured so the epilogue rewire below can point
        // the mulf's matmul operand at it.
        let mut nest_result: Option<Ssa> = None;
        while n_off < p.n {
            let bw = nblk.min(p.n - n_off);
            let kb = k_block(p.k, bw);
            let acc_dims = vec![1, bw];

            let (noff, lb, ub, step, zero) = (g.mint(), g.mint(), g.mint(), g.mint(), g.mint());
            pre.push(const_index(a, noff, n_off));
            pre.push(const_index(a, lb, 0));
            pre.push(const_index(a, ub, p.k));
            pre.push(const_index(a, step, kb));
            pre.push(const_index(a, zero, 0));

            // ⭐ THE SEEDS ARE PER-N-BLOCK, `tensor<1x{bw}>` — the shape the pre-move
            // construction emitted (`dense<0.0> : tensor<1x{bw}>` in the old textual
            // builder). Both the loop's `iter_args` init and the per-iteration matmul
            // `outs` seed are splats of the UNTILED FORM'S OWN SCALAR operand — the
            // one the zero-seed bound — so the parameter keeps its consumer
            // (and its derivable shape) while the splats themselves carry this block's
            // `[1, bw]` shape. Reusing the untiled `[m, n]` splat wholesale is only
            // shape-correct at m=1 and one block (decode); see `Untiled::init_scalar`.
            let splat = |res: Ssa, dims: Vec<i64>| {
                let mut op = Operation::new(a, Some(res), OpKind::TensorSplat, &[p.init_scalar])
                    .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(dims.clone())))
                    .with_attr(a, AttrKey::Dtype, Attr::Dtype(elem));
                op.result_type = Some(tensor(dims));
                op
            };
            let (azero, cinit) = (g.mint(), g.mint());
            pre.push(splat(azero, acc_dims.clone()));
            pre.push(splat(cinit, acc_dims.clone()));

            // ── the loop body ──
            let (accit, result, kv) = (g.mint(), g.mint(), g.mint());
            let mut body: Vec<Operation<'a>> = Vec::new();

            let (a_acc, a_val) = (g.mint(), g.mint());
            let mut ta = Operation::new(
                a,
                Some(a_acc),
                OpKind::KtdpConstructAccessTile,
                &[p.a_view, row_idx, kv],
            )
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![1, kb])));
            ta.result_type = Some(IrType::AccessTile {
                dims: a.ints(vec![1, kb]),
            });
            body.push(ta);
            let mut la = Operation::new(a, Some(a_val), OpKind::KtdpLoad, &[a_acc]).with_attr(
                a,
                AttrKey::Shape,
                Attr::IntList(a.ints(vec![1, kb])),
            );
            la.result_type = Some(tensor(vec![1, kb]));
            body.push(la);

            // B tile = the contiguous row-block of the weight, in the transpose-B
            // orientation every producer states (`[n, k]` view): `[n_off ..+bw, kv ..+kb]`,
            // tile `[bw, kb]`.
            let (w_acc, w_val) = (g.mint(), g.mint());
            let (wt_dims, wt_corner) = (vec![bw, kb], vec![noff, kv]);
            let mut tw = Operation::new(
                a,
                Some(w_acc),
                OpKind::KtdpConstructAccessTile,
                &[p.w_view, wt_corner[0], wt_corner[1]],
            )
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(wt_dims.clone())));
            tw.result_type = Some(IrType::AccessTile {
                dims: a.ints(wt_dims.clone()),
            });
            body.push(tw);
            let mut lw = Operation::new(a, Some(w_val), OpKind::KtdpLoad, &[w_acc]).with_attr(
                a,
                AttrKey::Shape,
                Attr::IntList(a.ints(wt_dims.clone())),
            );
            lw.result_type = Some(tensor(wt_dims));
            body.push(lw);

            // ⭐ THE MAPS SAY WHICH AXIS OF W IS k — transpose-B `[[0,2],[1,2],[0,1]]`,
            // the builder's own spelling (W binds verbatim as its on-disk `[n, k]`),
            // which is also what `dot_to_linalg` folds the spliced kernels' `.T` into.
            let (part, maps): (Ssa, Vec<AffineMap<'a>>) = {
                let part = g.mint();
                let table: &[&[i64]] = &[&[0, 2], &[1, 2], &[0, 1]];
                let maps = table
                    .iter()
                    .map(|mm| AffineMap {
                        num_dims: 3,
                        num_syms: 0,
                        exprs: a.exprs(mm.iter().map(|d| AffineExpr::Dim(*d as usize)).collect()),
                    })
                    .collect();
                (part, maps)
            };
            let mut mmop =
                Operation::new(a, Some(part), OpKind::LinalgMatmul, &[a_val, w_val, cinit])
                    .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(acc_dims.clone())))
                    .with_attr(a, AttrKey::IndexingMaps, Attr::AffineMapList(a.maps(maps)));
            mmop.result_type = Some(tensor(acc_dims.clone()));
            body.push(mmop);

            let accnext = g.mint();
            let mut add = Operation::new(a, Some(accnext), OpKind::ArithAddf, &[accit, part])
                .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(acc_dims.clone())));
            add.result_type = Some(tensor(acc_dims.clone()));
            body.push(add);
            body.push(Operation::new(a, None, OpKind::ScfYield, &[accnext]));

            let forop = Operation::new(a, Some(result), OpKind::ScfFor, &[lb, ub, step, azero])
                .with_attr(a, AttrKey::IterVar, Attr::Ssas(a.ssa(vec![kv])))
                .with_attr(a, AttrKey::IterArgs, Attr::Ssas(a.ssa(vec![accit])));
            let mut forop = Operation {
                regions: a.regions(vec![a.ops(body)]),
                ..forop
            };
            forop.result_type = Some(tensor(acc_dims.clone()));
            pre.push(forop);
            nest_result = Some(result);

            // Store this N block — unless the store is the scaled epilogue's (the
            // fp8 form): there the ORIGINAL `arith.mulf(result, scale) → store`
            // survives, re-pointed at the nest's result below, so the wscale load
            // chain keeps its consumer and the store's tile is the mulf's own.
            if p.scale.is_none() {
                let st_acc = g.mint();
                let mut ts = Operation::new(
                    a,
                    Some(st_acc),
                    OpKind::KtdpConstructAccessTile,
                    &[p.out_view, row_idx, noff],
                )
                .with_attr(
                    a,
                    AttrKey::Shape,
                    Attr::IntList(a.ints(acc_dims.clone())),
                );
                ts.result_type = Some(IrType::AccessTile {
                    dims: a.ints(acc_dims.clone()),
                });
                pre.push(ts);
                pre.push(Operation::new(
                    a,
                    None,
                    OpKind::KtdpStore,
                    &[result, st_acc],
                ));
            }

            n_off += bw;
        }

        // Splice: the tiled nest REPLACES the untiled matmul and its store. Everything the untiled
        // form built above it (the views, and the whole-tile load chain) is dropped with them — the
        // nest builds its own tiles off the same views.
        let mut ops: Vec<Operation<'a>> = func.operations.to_vec();
        let store_at = p.store_at;
        let at = p.at;
        // ⭐ THE SCALED FORM KEEPS ITS EPILOGUE: the store at `store_at` drains
        // `arith.mulf(matmul, wscale)`, and that mulf survives the splice — only
        // its MATMUL operand is re-pointed at the nest's result (the scaled form
        // takes exactly one N-block, so the nest's single result IS the whole
        // `[m, n]` contraction the mulf scales). The wscale load chain therefore
        // keeps its consumer through `dce`, which is what keeps the wscale
        // tensor's shape derivable downstream.
        if let Some(_scale) = p.scale {
            let matmul_res = p.matmul_res;
            let nest_result = nest_result.expect("the scaled form takes one N-block");
            let mulf_at = ops
                .iter()
                .position(|o| o.op_type == OpKind::ArithMulf && o.operands.contains(&matmul_res))
                .expect("recognize found the mulf draining the matmul");
            let pos = ops[mulf_at]
                .operands
                .iter()
                .position(|&o| o == matmul_res)
                .expect("the mulf's operand IS the matmul result");
            // `operands` is an arena slice — the rewire is a REPLACED op, same
            // everything, one operand swapped for the nest's result.
            let mut mulf = ops[mulf_at];
            mulf.operands = a.ssa({
                let mut v = ops[mulf_at].operands.to_vec();
                v[pos] = nest_result;
                v
            });
            ops[mulf_at] = mulf;
            // The store stays where it is; only the matmul is replaced.
            ops.splice(at..=at, pre);
        } else {
            // Remove the store first when it sits after the matmul, so both indices stay valid.
            if store_at > at {
                ops.remove(store_at);
                ops.splice(at..=at, pre);
            } else {
                ops.splice(at..=at, pre);
                ops.remove(store_at);
            }
        }
        func.operations = a.ops(ops);
        done += 1;
    }
    // ⛔ AND THE REPLACED FORM'S FEEDERS MUST GO, because the emulator is an INTERPRETER: it walks
    // every op, so a `construct_access_tile` left behind with no consumer is still CONSTRUCTED and
    // still bounds-checked. The untiled form's whole-tensor A/W tile+load chain is exactly that once
    // the nest replaces the matmul it fed — MEASURED as `window [1, 576] at base [30, 0] runs to 31
    // in dim 0 of a parent view [1, 576]`: the prefill lm-head tail's dead `[1, hidden]` tile, whose
    // row-sliced view only the (now-removed) GEMM offload path could address.
    dce(a, func);
    done
}

/// Drop every value-producing op nothing consumes, to a fixpoint. Side-effecting ops
/// (`ktdp.store`, the terminators, and anything carrying a region) are always kept.
fn dce<'a>(a: &'a Arena, func: &mut IRFunction<'a>) {
    loop {
        let mut used: std::collections::HashSet<Ssa> =
            func.arguments.iter().map(|(v, _)| *v).collect();
        fn mark(ops: &[Operation<'_>], used: &mut std::collections::HashSet<Ssa>) {
            for op in ops {
                used.extend(op.operands.iter().copied());
                for (_, v) in op.attributes.iter() {
                    if let Attr::Ssas(list) = v {
                        used.extend(list.iter().copied());
                    }
                }
                for rg in op.regions {
                    mark(rg, used);
                }
            }
        }
        mark(func.operations, &mut used);
        let keep: Vec<Operation<'a>> = func
            .operations
            .iter()
            .filter(|op| match op.result {
                None => true,
                Some(r) => used.contains(&r) || !op.regions.is_empty(),
            })
            .cloned()
            .collect();
        if keep.len() == func.operations.len() {
            return;
        }
        func.operations = a.ops(keep);
    }
}

/// Tile every recognized untiled contraction in `module`. Returns how many were rewritten.
pub fn apply_matmul_tiling<'a>(a: &'a Arena, module: &mut IRModule<'a>) -> usize {
    let names: Vec<String> = module.functions.keys().cloned().collect();
    let mut n = 0usize;
    for name in names {
        if let Some(func) = module.functions.get_mut(&name) {
            n += tile_func(a, func);
        }
    }
    n
}
