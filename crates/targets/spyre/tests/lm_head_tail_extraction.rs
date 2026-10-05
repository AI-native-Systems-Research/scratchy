// SPDX-License-Identifier: Apache-2.0
// ⛔ RUN THIS FILE WITH `SCRATCHY_PLAN_ONLY_BAKE=1` — the door side of these gates
// runs the KTIR→SuperDSC lowering, whose fixtures must not demand a device compiler.
//
//     SCRATCHY_PLAN_ONLY_BAKE=1 cargo test -p scratchy-target-spyre \
//         --features spyre-emu --test lm_head_tail_extraction
//
// (See `superdsc_time_tile.rs`' header for why the variable is not set from
// inside the test.)
#![cfg(feature = "spyre-emu")]
//! THE PREFILL LM-HEAD TAIL, END TO END — the extraction must feed the matmul the
//! LAST REAL prompt row, at every `n <= m_cap`.
//!
//! The 26b's first sampled token was EOS on a binary whose static bundle is
//! correct (lmlast corners at row `mq-1`, m=1 matmul reading the reserved tid,
//! the fused-segment `SliceForward` slicing at exactly those offsets). The
//! defect was UPSTREAM of the bundle, in the emulator serving path's staging:
//! the emitter folds the vocab-wide lm-head tail to m=1 over row
//! `selector_lastrow_col(m_cap)` — the LAST BAKED row — but a prompt rarely
//! fills its rung (the 26b: 23 tokens on the m=31 rung), and the emulator bound
//! pad rows as ZEROS (embedding zero, rope zero, raw causal triangle at the pad
//! position) while the card path REPLICATES the last real token into them
//! (`ChunkRows::row_logical_pos`, Kani-pinned). The extraction then read a pad
//! row's forward — not any token's — and the sampler chose from it.
//!
//! Three cases, all through the real serving path
//! (`lower_graph_to_ktir` → `ktir_groups` → `SpyreSession::new_multi` →
//! `run_step`) at a small geometry of the 26b's own proportions:
//!
//!   * the FULL rung (n == m) — the extraction, the SSA forwarding, the m=1
//!     matmul and the row-0 result placement all hold on their own;
//!   * the PADDED rung (n < m), staged exactly as the (fixed) serving path
//!     stages it (pad rows replicate the last real token) — pins the CONTRACT
//!     the staging must meet for the tail to read the last real token;
//!   * m=1 (the decode-shaped tail) — the fold must not fire and the plain
//!     path must hold.
//!
//! The graph (sources first, eval_dag's contract; the weight is the FUF `[k, n]`
//! logical orientation, while the SESSION is staged with the on-disk `[n, k]`
//! transpose the real loader hands the emulator — `spyre_load.rs` keeps the
//! buffer VERBATIM as `[out, in]` and `KtirFunc::matmul` reads it with
//! transpose-B `indexing_maps`):
//!
//!   t0 = x `[m, hidden]` (source)
//!   t1 = gamma `[1, hidden]` (source)
//!   t2 = lm_head weight `[hidden, vocab]` (source)
//!   t3 = norm out `[m, hidden]`     RmsNorm
//!   t4 = logits `[m, vocab]`        MatmulTile (vocab-wide ⇒ the prefill fold)
//!   t5 = capped `[m, vocab]`        TanhSoftCap
//!
//! Golden = `eval_dag` (the host reference the whole tree treats as ground
//! truth). The fixture's rows differ elementwise by O(1) post-norm, so a
//! mis-extracted row moves every logit by O(1) — far above the 5% bar.
//! ⛔ NO SEED SEARCH — well-conditioned by construction.

use scratchy_subtile::subtile_ir::{
    GainConvention, SubOp, SubtileIR, SubtileId, SubtileNode, TensorId, TensorRegion, TensorShape,
};
use scratchy_target_spyre::lower_subtile_tape_to_superdsc as superdsc;

/// The lm-head tail as one graph: RmsNorm → vocab-wide MatmulTile → TanhSoftCap.
/// Sources are the FIRST tensors (eval_dag's contract), the weight is the FUF
/// `[k, n]` logical orientation.
fn tail_ir(m: u32, hidden: u32, vocab: u32) -> SubtileIR {
    let tensors = vec![
        TensorShape { rows: m, cols: hidden },     // t0 x (source)
        TensorShape { rows: 1, cols: hidden },     // t1 gamma (source)
        TensorShape { rows: hidden, cols: vocab }, // t2 lm_head weight [k, n] (source)
        TensorShape { rows: m, cols: hidden },     // t3 norm out
        TensorShape { rows: m, cols: vocab },      // t4 logits
        TensorShape { rows: m, cols: vocab },      // t5 capped (result)
    ];
    let num_sources = 3u32;
    let whole = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: tensors[t].whole(),
    };
    let mk = |id: usize, op: SubOp, inputs: Vec<TensorRegion>, output: TensorRegion| SubtileNode {
        id: SubtileId::from_index(id),
        op,
        inputs,
        output,
    };
    let nodes = vec![
        mk(
            0,
            SubOp::RmsNorm {
                eps: 1e-6,
                gain: GainConvention::Scale,
            },
            vec![whole(0), whole(1)],
            whole(3),
        ),
        mk(
            1,
            SubOp::MatmulTile {
                n: vocab,
                weight: scratchy_subtile::lower::GemmWeight::Dense,
            },
            vec![whole(3), whole(2)],
            whole(4),
        ),
        mk(
            2,
            SubOp::TanhSoftCap { cap: 30.0 },
            vec![whole(4)],
            whole(5),
        ),
    ];
    SubtileIR {
        tensors,
        num_sources,
        nodes,
        result: TensorId::from_index(5),
        op_output: Vec::new(),
    }
}

/// Deterministic, well-conditioned fixtures (no seed search): elements in
/// [-0.75, 0.75), so the norm's per-row RMS is O(1) and every dot over 512
/// elements is ~±10 — the tanh cap's linear regime, where a wrong row's sign
/// pattern is fully visible in the logits.
fn synth(id: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|j| synth_elem(id, j))
        .collect()
}

fn synth_elem(id: usize, j: usize) -> f32 {
    (((id * 131 + j * 7) % 197) as f32 / 197.0 - 0.5) * 1.5
}

/// The lm_head weight in BOTH orientations: `(logical [k, n] row-major for
/// eval_dag, on-disk [n, k] row-major for the session)`.
fn weight_pair(k: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut fuf = vec![0f32; k * n];
    let mut disk = vec![0f32; k * n];
    for i in 0..k {
        for j in 0..n {
            let v = synth_elem(2, i * n + j);
            fuf[i * n + j] = v;
            disk[j * k + i] = v;
        }
    }
    (fuf, disk)
}

/// Drive the graph through the real session path and return `(t5 result, t4 raw
/// logits)` for the given pre-staged sources.
fn run_tail(
    ir: &SubtileIR<scratchy_subtile::subtile_ir::NeoX>,
    srcs: &[Vec<f32>],
    extra_out: &[u64],
) -> (Vec<f32>, Vec<f32>) {
    let weight_ids = std::collections::HashSet::new();
    let (ops, _layout) = superdsc::lower_graph_to_ktir(ir, &weight_ids, superdsc::ActiveCap::FULL, false)
        .unwrap_or_else(|e| panic!("lower the lm-head tail: {e:?}"));
    let groups = superdsc::ktir_groups(&ops, superdsc::FoldGrouping::Split)
        .expect("bind the lm-head tail's programs");
    let out_tid = ir.result.index() as u64;
    let mut results: Vec<u64> = vec![out_tid];
    results.extend_from_slice(extra_out);
    let mut session = scratchy_target_spyre::runner::SpyreSession::new_multi(
        &[(&groups, &results)],
        Vec::new(),
    )
    .expect("build the lm-head tail session");
    let mut out = session
        .run_step(
            0,
            srcs
                .iter()
                .enumerate()
                .map(|(id, v)| {
                    (
                        id as u64,
                        v.clone(),
                        vec![
                            ir.tensors[id].rows as usize,
                            ir.tensors[id].cols as usize,
                        ],
                    )
                })
                .collect(),
            &results.iter().map(|&t| (t, 0)).collect::<Vec<_>>(),
        )
        .expect("run the lm-head tail");
    let result = out.remove(&out_tid).expect("the result tensor");
    let raw = out.get(&4).cloned().unwrap_or_default();
    (result, raw)
}

#[test]
fn the_prefill_lm_head_tail_extracts_the_last_row() {
    // m=7 (prefill-shaped, m>1), hidden=512 (a whole-stick multiple, 8 sticks),
    // vocab=96 (narrow — the fold fires on WIDTH vs the RESULT's width, not on
    // vocab magnitude; 96 keeps the seconds-scale bound).
    let (m, hidden, vocab) = (7u32, 512u32, 96u32);
    let ir = tail_ir(m, hidden, vocab);

    // Golden: the host reference over the SAME sources (weight in FUF [k, n]).
    let (w_fuf, w_disk) = weight_pair(hidden as usize, vocab as usize);
    let x = synth(0, (m * hidden) as usize);
    let gamma = synth(1, hidden as usize);
    let golden_srcs: Vec<Vec<f32>> = vec![x.clone(), gamma.clone(), w_fuf];
    let refs: Vec<&[f32]> = golden_srcs.iter().map(|v| v.as_slice()).collect();
    let bufs = scratchy_subtile::subtile_ir::eval_dag(&ir, &refs);
    let golden = bufs[ir.result.index()].clone();

    // The fold must have fired: lower and check for the lmlast program.
    let weight_ids = std::collections::HashSet::new();
    let (ops, _layout) = superdsc::lower_graph_to_ktir(
        &ir,
        &weight_ids,
        superdsc::ActiveCap::FULL,
        false,
    )
    .unwrap_or_else(|e| panic!("lower the lm-head tail: {e:?}"));
    let lmlast: Vec<&str> = ops
        .iter()
        .filter_map(|e| e.ktir.as_ref().map(|k| k.func.name))
        .filter(|n| n.starts_with("lmlast_s"))
        .collect();
    assert_eq!(
        lmlast.len(),
        1,
        "the vocab-wide matmul at m>1 folds to the m=1 tail: {lmlast:?}"
    );

    // The session is staged with the on-disk [n, k] weight (the loader's
    // verbatim orientation).
    let session_srcs: Vec<Vec<f32>> = vec![x, gamma, w_disk];
    let (got, _raw_logits) = run_tail(&ir, &session_srcs, &[]);

    let v = vocab as usize;
    let last = m as usize - 1;
    // ⭐ THE RESULT ROW THE TAIL WRITES IS ROW 0 — the fold re-lowers the matmul
    // at m=1 (`node_at_one_row` narrows the output to `Range::new(rows.start,
    // 1)`), so the [1, vocab] tile lands at row 0 of the [m, vocab] view, and
    // the emulator serving path reads exactly that row (`res[..vocab]` in
    // `spyre_forward`'s KTIR twin). Compare ROW 0 against the golden LAST row.
    let got_row0 = &got[..v];
    let golden_last = &golden[last * v..(last + 1) * v];
    for (j, (g, w)) in got_row0.iter().zip(golden_last).enumerate() {
        let d = (g - w).abs();
        assert!(
            d < 0.05 * w.abs().max(1.0),
            "logit row0[{j}] = {g}, golden last-row {w} — the tail consumed the wrong row \
             (the extraction is the only row-selecting op in the graph)"
        );
    }
}

#[test]
fn the_tail_extracts_the_last_real_row_not_the_pad_row() {
    // ⭐ THE 26b's OWN SHAPE: real tokens < the baked rung width (23 on the m=31
    // rung). The fold extracts row `mq-1` — the LAST BAKED row — which is a PAD
    // row here; the staging below is the (fixed) serving contract that makes
    // that row numerically the last real token's forward. A staging that zeroes
    // the pad rows instead (the pre-fix emulator) hands the extraction a row that
    // is not any token's forward, and the first sampled token comes from it.
    let (m, n_real, hidden, vocab) = (7u32, 5u32, 512u32, 96u32);
    let ir = tail_ir(m, hidden, vocab);
    let (w_fuf, w_disk) = weight_pair(hidden as usize, vocab as usize);
    // The staged activation, EXACTLY as the (fixed) emulator serving path builds
    // it: real rows their own tokens, pad rows a REPLICA of the last real token
    // (`ChunkRows::row_logical_pos`'s clamp — see `spyre_forward`'s KTIR twin).
    // This graph has no rope and no attention, so the embed row IS the row's
    // whole input; the pad row's norm output then equals the last real token's.
    let real = synth(0, (n_real * hidden) as usize);
    let mut x = vec![0f32; (m * hidden) as usize];
    for i in 0..m as usize {
        let r = i.min(n_real as usize - 1);
        x[i * hidden as usize..(i + 1) * hidden as usize]
            .copy_from_slice(&real[r * hidden as usize..(r + 1) * hidden as usize]);
    }
    let gamma = synth(1, hidden as usize);

    // Golden: the HOST reference over the real tokens only — a [n_real, hidden]
    // graph whose last row is the last REAL token.
    let real_ir = tail_ir(n_real, hidden, vocab);
    let golden_srcs: Vec<Vec<f32>> = vec![real.clone(), gamma.clone(), w_fuf];
    let refs: Vec<&[f32]> = golden_srcs.iter().map(|v| v.as_slice()).collect();
    let bufs = scratchy_subtile::subtile_ir::eval_dag(&real_ir, &refs);
    let golden = bufs[real_ir.result.index()].clone();

    let session_srcs: Vec<Vec<f32>> = vec![x, gamma, w_disk];
    let (got, _raw_logits) = run_tail(&ir, &session_srcs, &[]);
    let v = vocab as usize;
    let last_real = n_real as usize - 1;
    for (j, (g, w)) in got[..v].iter().zip(&golden[last_real * v..]).enumerate() {
        let d = (g - w).abs();
        assert!(
            d < 0.05 * w.abs().max(1.0),
            "logit row0[{j}] = {g}, golden last-REAL-row {w} — the tail extracted the PAD row \
             (row m-1) instead of the last real token's row (row n-1)"
        );
    }
}

#[test]
fn the_tail_at_m1_still_matches_the_reference() {
    // The decode-shaped tail: m=1 means no fold (rows.len == 1 fails
    // `is_prefill_lm_head_tail`) and the plain m=1 path must hold.
    let (m, hidden, vocab) = (1u32, 512u32, 96u32);
    let ir = tail_ir(m, hidden, vocab);
    let (w_fuf, w_disk) = weight_pair(hidden as usize, vocab as usize);
    let x = synth(0, (m * hidden) as usize);
    let gamma = synth(1, hidden as usize);
    let golden_srcs: Vec<Vec<f32>> = vec![x.clone(), gamma.clone(), w_fuf];
    let refs: Vec<&[f32]> = golden_srcs.iter().map(|v| v.as_slice()).collect();
    let bufs = scratchy_subtile::subtile_ir::eval_dag(&ir, &refs);
    let golden = bufs[ir.result.index()].clone();

    let weight_ids = std::collections::HashSet::new();
    let (ops, _layout) = superdsc::lower_graph_to_ktir(
        &ir,
        &weight_ids,
        superdsc::ActiveCap::FULL,
        false,
    )
    .unwrap_or_else(|e| panic!("lower the m=1 tail: {e:?}"));
    let lmlast: Vec<&str> = ops
        .iter()
        .filter_map(|e| e.ktir.as_ref().map(|k| k.func.name))
        .filter(|n| n.starts_with("lmlast_s"))
        .collect();
    assert!(
        lmlast.is_empty(),
        "m=1 must not fold to the prefill tail: {lmlast:?}"
    );

    let session_srcs: Vec<Vec<f32>> = vec![x, gamma, w_disk];
    let (got, _raw_logits) = run_tail(&ir, &session_srcs, &[]);
    for (j, (g, w)) in got.iter().zip(&golden).enumerate() {
        let d = (g - w).abs();
        assert!(
            d < 0.05 * w.abs().max(1.0),
            "m=1 logit[{j}] = {g}, golden {w}"
        );
    }
}
