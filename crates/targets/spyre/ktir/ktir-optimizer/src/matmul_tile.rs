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
///
/// ⛔ THE BUDGET IS A BYTE BUDGET. An F32 accumulator — the fp8 pre-scale code-dot
/// `lower_subtile_tape_to_ktir`'s `matmul_fp8` keeps in f32 (the un-scaled dot overflows f16
/// long before the scale `mulf` brings it home) — is 2× the bytes of the f16 one at the same
/// `[m, bw]`, so the element budget it may spend is `BLOCK_MN_BYTES / bytes_per_elem`:
/// MEASURED at the gemma-4-12b fp8 prefill (m=31, n=15360), an f32 `[31, 15360]` GEMM out is
/// 1_904_640 B and the init splat beside it another 1_904_640 B — 3.8 MB against the 2 MB LX,
/// where the f16 pair (952_320 + 952_320) fit exactly. The f16 element budget (1 MB / 2 B =
/// 512K elements) is the historical value verbatim; the f32 one halves to 256K, which blocks
/// the f32 form at half the width so the interpreter reclaims per block.
fn n_block(n: i64, m: i64, elem: DType) -> i64 {
    const WHOLE_N_FITS_M1: i64 = 90_000;
    const BLOCK_N: i64 = 16_384;
    const BLOCK_MN_BYTES: i64 = 1024 * 1024;
    let m = m.max(1);
    if m.saturating_mul(n) <= WHOLE_N_FITS_M1 {
        n
    } else {
        let budget = BLOCK_MN_BYTES / elem.bytes_per_elem() as i64;
        BLOCK_N.min(budget / m).max(1)
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
    /// The fp8 dequant form's per-output-column scale VIEW (`[1, n]`), when the store drains an
    /// `arith.mulf` of the matmul result by that row instead of the result itself —
    /// [`KtirFunc::matmul_fp8`]'s shape. `None` for the plain bf16 form, whose recognized set and
    /// emitted IR are byte-identical to before this field existed.
    scale_view: Option<Ssa>,
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
        let (Some(&m), Some(&k)) = (a_dims.first(), a_dims.get(1)) else {
            continue;
        };
        let Some(&n) = w_dims.first() else { continue };
        // Already tiled (a K-blocked A tile) ⇒ not ours.
        if w_dims.get(1) != Some(&k) {
            continue;
        }
        // The store that drains the contraction, and the view it writes. Two forms:
        // * bf16: `ktdp.store(matmul_res, …)` — the direct drain.
        // * fp8: `arith.mulf(matmul_res, scale_row)` then a store of THAT — `KtirFunc::matmul_fp8`'s
        //   dequant, whose `[1, n]` scale view must be carried so each N-block can scale its own
        //   slice. ⛔ THE MULF MUST BE THE MATMUL RESULT'S ONLY OTHER CONSUMER: the splice below
        //   removes the matmul, so a second consumer would dangle; and the scale row it reads must
        //   be exactly `[1, n]`, the per-output-column row the checkpoint ships.
        let mut scale_view = None;
        let mut store_at = None;
        let mut st = None;
        if let Some((j, o)) = func
            .operations
            .iter()
            .enumerate()
            .find(|(_, o)| o.op_type == OpKind::KtdpStore && o.operands.first() == Some(&res))
        {
            store_at = Some(j);
            st = Some(o);
        } else if let Some((_, o)) = func.operations.iter().enumerate().find(|(_, o)| {
            o.op_type == OpKind::ArithMulf
                && o.operands.first() == Some(&res)
                && func
                    .operations
                    .iter()
                    .filter(|c| c.operands.contains(&res))
                    .count()
                    == 1
        }) {
            let Some(scaled) = o.result else { continue };
            let Some((js, sst)) = func.operations.iter().enumerate().find(|(_, so)| {
                so.op_type == OpKind::KtdpStore && so.operands.first() == Some(&scaled)
            }) else {
                continue;
            };
            let scale_ok = (|| {
                let (_, sld) = def.get(o.operands.get(1)?)?;
                if sld.op_type != OpKind::KtdpLoad {
                    return None;
                }
                let (_, sacc) = def.get(sld.operands.first()?)?;
                if sacc.op_type != OpKind::KtdpConstructAccessTile {
                    return None;
                }
                let s_view = *sacc.operands.first()?;
                let s_shape = shape_of(sacc)?;
                if s_shape.first() != Some(&1) || s_shape.get(1) != Some(&n) {
                    return None;
                }
                Some(s_view)
            })();
            if let Some(s_view) = scale_ok {
                store_at = Some(js);
                st = Some(sst);
                scale_view = Some(s_view);
            }
        }
        let (Some(store_at), Some(st)) = (store_at, st) else {
            continue;
        };
        let Some((_, out_acc)) = st.operands.get(1).and_then(|t| def.get(t)) else {
            continue;
        };
        let Some(&out_view) = out_acc.operands.first() else {
            continue;
        };
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
            scale_view,
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

        let nblk = n_block(p.n, p.m, elem);
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
            // ⭐ FP8 DEQUANT: the block's accumulator is scaled by ITS OWN slice of the checkpoint's
            // `[1, n]` per-output-column row — columns `[n_off, n_off+bw)` — before the store. This
            // mirrors what `KtirFunc::matmul_fp8` did untiled (mulf by the whole row), and is
            // arithmetic the contraction owes, not a tiling decision, so the N-blocking does not get
            // to change it. The slice load is outside the loop: it is loop-invariant, and hoisting
            // it keeps the loop body exactly the bf16 one (the GEMM offload's `recognize_matmul_loop`
            // requires exactly one matmul and an `addf`-only yield chain).
            let stored = if let Some(scale_view) = p.scale_view {
                let (s_acc, s_val) = (g.mint(), g.mint());
                let mut tsc = Operation::new(
                    a,
                    Some(s_acc),
                    OpKind::KtdpConstructAccessTile,
                    &[scale_view, zero, noff],
                )
                .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![1, bw])));
                tsc.result_type = Some(IrType::AccessTile {
                    dims: a.ints(vec![1, bw]),
                });
                pre.push(tsc);
                let mut lsc = Operation::new(a, Some(s_val), OpKind::KtdpLoad, &[s_acc]).with_attr(
                    a,
                    AttrKey::Shape,
                    Attr::IntList(a.ints(vec![1, bw])),
                );
                lsc.result_type = Some(tensor(vec![1, bw]));
                pre.push(lsc);
                let scaled = g.mint();
                let mut mul = Operation::new(a, Some(scaled), OpKind::ArithMulf, &[result, s_val])
                    .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(acc_dims.clone())));
                mul.result_type = Some(tensor(acc_dims.clone()));
                pre.push(mul);
                scaled
            } else {
                result
            };
            pre.push(Operation::new(
                a,
                None,
                OpKind::KtdpStore,
                &[stored, st_acc],
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_builder::Ops;
    use ktir_core::attrkey::AttrKey;
    use ktir_core::dtypes::DType;
    use ktir_core::ir::Attr;
    use ktir_core::irtype::IrType;

    /// One untiled contraction exactly as [`KtirFunc::matmul`] / [`KtirFunc::matmul_fp8`] emit it.
    /// `m`, `n`, `k` fix the geometry; `dequant` inserts the fp8 form's `arith.mulf` by a loaded
    /// `[1, scale_cols]` scale row between the matmul and the store (`scale_cols` lets a test ship
    /// a row of the WRONG width, which the recognizer must refuse).
    fn contraction(m: i64, n: i64, k: i64, dequant: bool, scale_cols: i64) -> IRFunction<'static> {
        let mut ops = Ops::new();
        let a = ops.arena();
        let c0 = ops.op(Some("%c0"), OpKind::ArithConstant, &[]);
        let c0 = ops.attr(c0, AttrKey::Value, Attr::Int(0));
        let mk_view =
            |ops: &mut Ops, res: &'static str, arg: &'static str, dims: Vec<i64>, elem: DType| {
                let o = ops.op(Some(res), OpKind::KtdpConstructMemoryView, &[arg]);
                let o = ops.attr(o, AttrKey::Shape, ops.int_list(dims.clone()));
                let o = ops.attr(o, AttrKey::Strides, ops.int_list(vec![dims[1], 1]));
                let o = ops.attr(o, AttrKey::MemorySpace, Attr::Str("HBM"));
                ops.attr(o, AttrKey::Dtype, Attr::Dtype(elem))
            };
        let mk_tile = |ops: &mut Ops, res: &'static str, view: &'static str, dims: Vec<i64>| {
            let o = ops.op(
                Some(res),
                OpKind::KtdpConstructAccessTile,
                &[view, "%c0", "%c0"],
            );
            ops.attr(o, AttrKey::Shape, ops.int_list(dims))
        };
        let f16_tensor = |dims: Vec<i64>| IrType::Tensor {
            dims: a.ints(dims),
            elem: DType::F16,
        };

        let mut body = vec![c0];
        body.push(mk_view(&mut ops, "%va", "%a_ptr", vec![m, k], DType::F16));
        body.push(mk_view(
            &mut ops,
            "%vw",
            "%w_ptr",
            vec![n, k],
            DType::Fp8E4m3,
        ));
        body.push(mk_view(
            &mut ops,
            "%vs",
            "%s_ptr",
            vec![1, scale_cols],
            DType::F16,
        ));
        body.push(mk_view(
            &mut ops,
            "%vout",
            "%out_ptr",
            vec![m, n],
            DType::F16,
        ));
        body.push(mk_tile(&mut ops, "%ta", "%va", vec![m, k]));
        let la = ops.op(Some("%a_val"), OpKind::KtdpLoad, &["%ta"]);
        let la = ops.attr(la, AttrKey::Shape, ops.int_list(vec![m, k]));
        body.push(ops.ty(la, f16_tensor(vec![m, k])));
        body.push(mk_tile(&mut ops, "%tw", "%vw", vec![n, k]));
        let lw = ops.op(Some("%w_val"), OpKind::KtdpLoad, &["%tw"]);
        let lw = ops.attr(lw, AttrKey::Shape, ops.int_list(vec![n, k]));
        body.push(ops.ty(lw, f16_tensor(vec![n, k])));
        let zc = ops.op(Some("%zc"), OpKind::ArithConstant, &[]);
        let zc = ops.attr(zc, AttrKey::Value, Attr::Float(0.0));
        body.push(ops.ty(zc, IrType::Scalar(DType::F16)));
        let init = ops.op(Some("%init"), OpKind::TensorSplat, &["%zc"]);
        let init = ops.attr(init, AttrKey::Shape, ops.int_list(vec![m, n]));
        let init = ops.attr(init, AttrKey::Dtype, Attr::Dtype(DType::F16));
        body.push(ops.ty(init, f16_tensor(vec![m, n])));
        let mm = ops.op(
            Some("%part"),
            OpKind::LinalgMatmul,
            &["%a_val", "%w_val", "%init"],
        );
        let mm = ops.attr(mm, AttrKey::Shape, ops.int_list(vec![m, n]));
        body.push(ops.ty(mm, f16_tensor(vec![m, n])));
        let stored: &'static str = if dequant {
            body.push(mk_tile(&mut ops, "%ts", "%vs", vec![1, scale_cols]));
            let ls = ops.op(Some("%s_val"), OpKind::KtdpLoad, &["%ts"]);
            let ls = ops.attr(ls, AttrKey::Shape, ops.int_list(vec![1, scale_cols]));
            body.push(ops.ty(ls, f16_tensor(vec![1, scale_cols])));
            let mul = ops.op(Some("%scaled"), OpKind::ArithMulf, &["%part", "%s_val"]);
            let mul = ops.attr(mul, AttrKey::Shape, ops.int_list(vec![m, n]));
            body.push(ops.ty(mul, f16_tensor(vec![m, n])));
            "%scaled"
        } else {
            "%part"
        };
        body.push(mk_tile(&mut ops, "%tout", "%vout", vec![m, n]));
        body.push(ops.op(None, OpKind::KtdpStore, &[stored, "%tout"]));
        ops.func(
            "contraction",
            &[
                ("%a_ptr", IrType::Index),
                ("%w_ptr", IrType::Index),
                ("%s_ptr", IrType::Index),
                ("%out_ptr", IrType::Index),
            ],
            body,
            (1, 1, 1),
        )
    }

    /// The fp8 dequant form IS recognized, with the scale view carried — the gemma-4-12b fp8 emu
    /// gate. Before this recognition the pass reported "0 of 1 contraction(s) recognised" for every
    /// `matmul_fp8` node, leaving a whole `[n, k]` fp8 weight tile in LX (q_proj `[4096, 3840]` =
    /// 15.7 MB against 2 MB).
    #[test]
    fn the_fp8_dequant_form_is_recognised_with_its_scale_view() {
        let f = contraction(
            1, 96, 128, /* dequant: */ true, /* scale_cols: */ 96,
        );
        let found = recognize(&f);
        assert_eq!(
            found.len(),
            1,
            "the fp8 dequant contraction must be recognized"
        );
        assert!(
            found[0].scale_view.is_some(),
            "the scale view must be carried"
        );
        assert_eq!(found[0].n, 96);
        assert_eq!(found[0].k, 128);
    }

    /// The plain bf16 form keeps its DIRECT store drain and carries no scale view.
    #[test]
    fn the_plain_form_still_recognises_without_a_scale() {
        let f = contraction(1, 96, 128, false, 96);
        let found = recognize(&f);
        assert_eq!(found.len(), 1);
        assert!(found[0].scale_view.is_none());
    }

    /// A scale row of the WRONG width is not this contraction's dequant — the recognizer must
    /// refuse rather than scale N-blocks by an off-width slice.
    #[test]
    fn an_off_width_scale_row_is_refused() {
        let f = contraction(1, 96, 128, true, /* scale_cols: */ 64);
        assert!(recognize(&f).is_empty());
    }

    /// After tiling, the fp8 form stores an `arith.mulf` of the loop result by a `[1, bw]` scale
    /// slice — one mulf and one scale load per N-block, outside the K-loop body.
    #[test]
    fn tiling_scales_each_n_block_by_its_own_scale_slice() {
        let mut f = contraction(1, 96, 128, true, 96);
        let done = tile_func(Arena::global(), &mut f);
        assert_eq!(done, 1);
        let mul_count = f
            .operations
            .iter()
            .filter(|o| o.op_type == OpKind::ArithMulf)
            .count();
        let scale_loads = f
            .operations
            .iter()
            .filter(|o| {
                o.op_type == OpKind::KtdpLoad
                    && o.operands
                        .first()
                        .and_then(|acc| f.operations.iter().find(|d| d.result == Some(*acc)))
                        .is_some_and(|acc| {
                            acc.attr(AttrKey::Shape)
                                == Some(&Attr::IntList(Arena::global().ints(vec![1, 96])))
                        })
            })
            .count();
        assert!(mul_count >= 1, "the dequant mulf must survive tiling");
        assert!(
            scale_loads >= 1,
            "the scale slice load must survive tiling (whole-n block here)"
        );
        // Every store drains an mulf, not a bare loop result.
        for o in f
            .operations
            .iter()
            .filter(|o| o.op_type == OpKind::KtdpStore)
        {
            let val = o.operands[0];
            let def = f.operations.iter().find(|d| d.result == Some(val));
            assert!(
                def.is_some_and(|d| d.op_type == OpKind::ArithMulf),
                "the fp8 store must drain the scaled value"
            );
        }
    }

    /// ⛔ The N-block budget is a BYTE budget. The gemma-4-12b fp8 prefill (m=31, n=15360)
    /// keeps the pre-scale code-dot in f32, so its `[31, 15360]` GEMM out is 1_904_640 B —
    /// beside the equally-sized f32 init splat that is 3.8 MB against the 2 MB LX, where the
    /// f16 pair (952_320 + 952_320) fit exactly. An F32 accumulator must therefore block at
    /// HALF the f16 element budget, while the f16 form keeps its exact old width.
    #[test]
    fn the_n_block_budget_is_bytes_not_elements() {
        // f16 (the historical budget): 512K elements / m=31 = 16912, capped at 16384.
        assert_eq!(n_block(200_000, 31, DType::F16), 16_384);
        // f32: half the elements — 256K / 31 = 8456 — so a wide f32 out blocks below 1 MB.
        assert_eq!(n_block(200_000, 31, DType::F32), 8_456);
        // The MEASURED failure geometry: f32 [31, 15360] must NOT stay one whole-n block.
        // m·n = 476_160 > 90_000, and with the f16 budget min(16384, 16912) = 16384 >= n
        // would keep it whole (the 3.8 MB pair); the byte budget splits it.
        assert!(
            n_block(15_360, 31, DType::F32) < 15_360,
            "the f32 12b prefill out must block, not stay whole-n"
        );
        // Whole-n small outputs are unchanged by dtype (decode's [1, n] is tiny either way).
        assert_eq!(n_block(90_000, 1, DType::F32), 90_000);
        assert_eq!(n_block(90_000, 1, DType::F16), 90_000);
    }
}
