// SPDX-License-Identifier: Apache-2.0
//! Matmul LX-FIT TILING pass — `M` across the grid, `N` in column blocks, `K` as an accumulating
//! loop.
//!
//! ⭐⭐⭐ THIS IS A SCHEDULING DECISION, AND IT BELONGS TO THE EMULATOR ALONE. The construction in
//! `lower_subtile_tape_to_superdsc.rs`'s `KtirFunc::matmul` now emits ONE untiled `linalg.matmul`
//! over the whole `[m, k] × [k, n]`, because that is what the IR means. Who tiles it, and how, is a
//! property of the DEVICE that runs it:
//!
//! * The **emulator** executes the ops itself against a real 2 MB LX, so it cannot hold
//!   `W[2048, 2048]` fp16 (8 MB) as one tile. It needs this pass, and the numbers here
//!   ([`k_block`]/[`n_block`]) are the ones it was measured against.
//! * The **card** does not. `SubtileIR → SuperDSC` hands `assemble_matmul` a whole GEMM and the
//!   work division is DECLARED, not built: `WorkPlan::divide` fills `numWkSlicesPerDim_` for dxp's
//!   scheduler and `WorkPlan::time_tile_for_lx` fills `OpSpec.time_tile`, which `render_dxp_input`
//!   expands into trips. `lower_matmul_node` says it outright — "the Spyre tape does NOT K-chunk
//!   (each `MatmulTile` is a whole GEMM; SuperDSC owns the K-split via the cost model)".
//!
//! ⛔ SO A PRE-TILED KTIR WAS A BUG FOR ONE OF ITS TWO CONSUMERS. The nest below used to be built
//! during KTIR construction, which meant `KTIR → SuperDSC` received a 64-trip `scf.for` where the
//! proven emitter wanted one contraction — an `ScfFor` it had no mapping for, because there is no
//! SuperDSC op that means "loop". Un-tiling it in the KTIR→SuperDSC direction would have meant
//! pattern-matching a tiling back out of the IR; emitting untiled KTIR and tiling HERE, for the one
//! consumer that needs it, is the same decision made in the right place.
//!
//! ⭐ MOVED VERBATIM, and that is the correctness argument. Every extent, block size, op order and
//! attribute below is what `KtirFunc::matmul` built before, so the emulator sees the same KTIR it
//! already runs at its measured rate. This pass is a relocation, not a redesign.

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
    /// The untiled contraction's `outs` seed. ⛔ REUSED, NOT REBUILT: it is a splat of the BOUND
    /// zero constant at its reserved tid (`KtirFunc::splat_zero`), so minting a fresh immediate in
    /// its place orphans that parameter — the `dce` below then drops its view chain and the emulator
    /// refuses with `no shape derivable for tensor t4294967275` (`u32::MAX - 20`, registry slot 0).
    init: Ssa,
    m: i64,
    n: i64,
    k: i64,
    elem: DType,
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

/// Recognize the untiled form [`Self`] tiles. Deliberately narrow: it matches exactly what
/// `KtirFunc::matmul` emits and nothing else, so an unrecognized matmul is LEFT ALONE rather than
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
        let (Some((a_view, a_row, a_dims)), Some((w_view, _, w_dims))) =
            (through_load(av), through_load(wv))
        else {
            continue;
        };
        let Some(res) = op.result else { continue };
        // The store that drains it, and the view it writes.
        let Some((store_at, st)) = func
            .operations
            .iter()
            .enumerate()
            .find(|(_, o)| o.op_type == OpKind::KtdpStore && o.operands.first() == Some(&res))
        else {
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
        let Some(&n) = w_dims.first() else { continue };
        // Already tiled (a K-blocked A tile) ⇒ not ours.
        if w_dims.get(1) != Some(&k) {
            continue;
        }
        let elem = op.result_type.and_then(|t| t.elem()).unwrap_or(DType::F16);
        out.push(Untiled {
            at: i,
            store_at,
            a_view,
            w_view,
            out_view,
            a_row,
            init,
            m,
            n,
            k,
            elem,
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

            // ⭐ THE SEED IS THE ONE THE UNTILED FORM CARRIED — a splat of the bound zero at its
            // reserved tid. Reused for the loop's `iter_args` init AND the per-iteration matmul seed,
            // so the parameter keeps a consumer and its shape stays derivable.
            let azero = p.init;
            let cinit = p.init;

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

            // B tile = the contiguous row-block `[n_off ..+bw, kv ..+kb]` of `[n, k]`.
            let (w_acc, w_val) = (g.mint(), g.mint());
            let mut tw = Operation::new(
                a,
                Some(w_acc),
                OpKind::KtdpConstructAccessTile,
                &[p.w_view, noff, kv],
            )
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![bw, kb])));
            tw.result_type = Some(IrType::AccessTile {
                dims: a.ints(vec![bw, kb]),
            });
            body.push(tw);
            let mut lw = Operation::new(a, Some(w_val), OpKind::KtdpLoad, &[w_acc]).with_attr(
                a,
                AttrKey::Shape,
                Attr::IntList(a.ints(vec![bw, kb])),
            );
            lw.result_type = Some(tensor(vec![bw, kb]));
            body.push(lw);

            // ⭐ W BINDS VERBATIM as its on-disk `[out, in]` = `[n, k]` buffer: the matmul reads it
            // with transpose-B `indexing_maps` (B's map ends in the reduction dim), so there is no
            // transpose and no strided gather.
            let part = g.mint();
            let maps: Vec<AffineMap<'a>> = [[0i64, 2], [1, 2], [0, 1]]
                .iter()
                .map(|mm| AffineMap {
                    num_dims: 3,
                    num_syms: 0,
                    exprs: a.exprs(mm.iter().map(|d| AffineExpr::Dim(*d as usize)).collect()),
                })
                .collect();
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

            // Store this N block.
            let st_acc = g.mint();
            let mut ts = Operation::new(
                a,
                Some(st_acc),
                OpKind::KtdpConstructAccessTile,
                &[p.out_view, row_idx, noff],
            )
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(acc_dims.clone())));
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

            n_off += bw;
        }

        // Splice: the tiled nest REPLACES the untiled matmul and its store. Everything the untiled
        // form built above it (the views, and the whole-tile load chain) is dropped with them — the
        // nest builds its own tiles off the same views.
        let mut ops: Vec<Operation<'a>> = func.operations.to_vec();
        let store_at = p.store_at;
        let at = p.at;
        // Remove the store first when it sits after the matmul, so both indices stay valid.
        if store_at > at {
            ops.remove(store_at);
            ops.splice(at..=at, pre);
        } else {
            ops.splice(at..=at, pre);
            ops.remove(store_at);
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
