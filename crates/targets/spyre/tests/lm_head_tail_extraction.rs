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
        TensorShape {
            rows: m,
            cols: hidden,
        }, // t0 x (source)
        TensorShape {
            rows: 1,
            cols: hidden,
        }, // t1 gamma (source)
        TensorShape {
            rows: hidden,
            cols: vocab,
        }, // t2 lm_head weight [k, n] (source)
        TensorShape {
            rows: m,
            cols: hidden,
        }, // t3 norm out
        TensorShape {
            rows: m,
            cols: vocab,
        }, // t4 logits
        TensorShape {
            rows: m,
            cols: vocab,
        }, // t5 capped (result)
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
    (0..n).map(|j| synth_elem(id, j)).collect()
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
    let (ops, _layout) =
        superdsc::lower_graph_to_ktir(ir, &weight_ids, superdsc::ActiveCap::FULL, false)
            .unwrap_or_else(|e| panic!("lower the lm-head tail: {e:?}"));
    let groups = superdsc::ktir_groups(&ops, superdsc::FoldGrouping::Split)
        .expect("bind the lm-head tail's programs");
    let out_tid = ir.result.index() as u64;
    let mut results: Vec<u64> = vec![out_tid];
    results.extend_from_slice(extra_out);
    let mut session = scratchy_target_spyre::runner::SpyreSession::new_multi(
        &[(&groups, &results)],
        Vec::new(),
        // No scale slot in this program's sources — the empty stage table is the
        // byte-identical incumbent (seat (i) declares nothing when nothing is f32).
        &[],
    )
    .expect("build the lm-head tail session");
    let mut out = session
        .run_step(
            0,
            srcs.iter()
                .enumerate()
                .map(|(id, v)| {
                    (
                        id as u64,
                        v.clone(),
                        vec![ir.tensors[id].rows as usize, ir.tensors[id].cols as usize],
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
    let (ops, _layout) =
        superdsc::lower_graph_to_ktir(&ir, &weight_ids, superdsc::ActiveCap::FULL, false)
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
    let (ops, _layout) =
        superdsc::lower_graph_to_ktir(&ir, &weight_ids, superdsc::ActiveCap::FULL, false)
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

/// The gemma-4 parity fixture's collision, minimized: a q-projection matmul
/// whose output width EQUALS the result's (vocab 512 == 4 heads × hd 128 there;
/// 64 == 1 head × hd 64 here), feeding the result through RoPE.
///
/// ⛔ THIS WAS A REAL CORRUPTION, not a hypothetical: the width-only predicate
/// folded every global layer's Q to m=1 — `lmlast` extracted the last prompt
/// row, the matmul ran at one row, rope rotated at position 0 (identity), and
/// attention saw `m-1` zero rows. The first sampled token was garbage while
/// every width-128 tensor stayed fp16-clean against the oracle. RoPE is
/// deliberately NOT row-local (`RopeAppend` also writes the paged KV cache, a
/// consumer the prefill graph cannot see), so the fold must refuse this chain.
#[test]
fn the_fold_refuses_a_width_collision_that_feeds_the_result_through_rope() {
    // m=7 rows, hidden=128, q width == result width == 64 (1 head × hd 64).
    let (m, hidden, qw) = (7u32, 128u32, 64u32);
    let hd = 64u32;

    // t0 x [m, hidden]; t1 Wq [hidden, qw]; t2 cos [m, qw]; t3 sin [m, qw];
    // t4 q [m, qw]; t5 roped q [m, qw] (the RESULT — same width as q).
    // Sources FIRST (eval_dag's contract: tensors[0..num_sources] are leaves).
    // The cos/sin tables are PER-POSITION `[m, qw]` rows — the rope program views
    // them as `[rows*tbl_cols]` and reads position `ri`'s row at `ri*tbl_cols`.
    let tensors = vec![
        TensorShape {
            rows: m,
            cols: hidden,
        }, // t0 x (source)
        TensorShape {
            rows: hidden,
            cols: qw,
        }, // t1 Wq (source)
        TensorShape { rows: m, cols: qw }, // t2 cos (source)
        TensorShape { rows: m, cols: qw }, // t3 sin (source)
        TensorShape { rows: m, cols: qw }, // t4 q
        TensorShape { rows: m, cols: qw }, // t5 roped (result)
    ];
    let num_sources = 4u32; // x, Wq, cos, sin
    let whole = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: tensors[t].whole(),
    };
    // Row `ri` of tensor `t` — the PER-ROW rope node form the real front end's
    // `head_tile_rope` decomposition produces (each node ONE row's [1, qw]
    // slice; the cos/sin input region is that row's table slice, so eval_dag's
    // flat `cos[d]` read and the KTIR's per-position read agree).
    let row = |t: usize, ri: u32| TensorRegion {
        tensor: TensorId::from_index(t),
        region: scratchy_subtile::subtile_ir::Region {
            rows: scratchy_subtile::subtile_ir::Range::new(ri, 1),
            cols: scratchy_subtile::subtile_ir::Range::new(0, qw),
        },
    };
    let mk = |id: usize, op: SubOp, inputs: Vec<TensorRegion>, output: TensorRegion| SubtileNode {
        id: SubtileId::from_index(id),
        op,
        inputs,
        output,
    };
    // The q matmul (whole [m, qw]), then ONE PER-ROW rope node per position.
    let mut nodes = vec![mk(
        0,
        SubOp::MatmulTile {
            n: qw,
            weight: scratchy_subtile::lower::GemmWeight::Dense,
        },
        vec![whole(0), whole(1)],
        whole(4),
    )];
    for ri in 0..m {
        nodes.push(mk(
            1 + ri as usize,
            SubOp::RopeRotate {
                head_dim: scratchy_subtile::model_geometry::HeadDim::new(hd),
                _form: std::marker::PhantomData,
            },
            vec![row(4, ri), row(2, ri), row(3, ri)],
            row(5, ri),
        ));
    }
    let ir = SubtileIR {
        tensors,
        num_sources,
        nodes,
        result: TensorId::from_index(5),
        op_output: Vec::new(),
    };

    // The fold must NOT fire: the q matmul's width collides with the result's,
    // but its chain to the result passes through RoPE.
    let weight_ids = std::collections::HashSet::new();
    let (ops, _layout) =
        superdsc::lower_graph_to_ktir(&ir, &weight_ids, superdsc::ActiveCap::FULL, false)
            .unwrap_or_else(|e| panic!("lower the collision graph: {e:?}"));
    let lmlast: Vec<&str> = ops
        .iter()
        .filter_map(|e| e.ktir.as_ref().map(|k| k.func.name))
        .filter(|n| n.starts_with("lmlast_s"))
        .collect();
    assert!(
        lmlast.is_empty(),
        "a width collision whose chain passes through RoPE must NOT fold to the m=1 tail: \
         {lmlast:?} — this is the gemma-4 global-q corruption"
    );

    // ⭐ AND THE MATMUL MUST KEEP EVERY ROW — the fold's whole damage was that
    // the q matmul's program viewed its activation as ONE row. Pin the emitted
    // programs' own view dims directly: the matmul's activation view is
    // `[m, hidden]` (7 rows), not `[1, hidden]`, and the per-row rope programs
    // exist for EVERY position (7 of them). This is the exact observable the
    // corruption had (the bundle's q matmul program viewed x as [1, hidden] and
    // only row 0 of the roped q was ever real).
    let mut matmul_rows = Vec::new();
    let mut rope_count = 0usize;
    for e in &ops {
        let Some(k) = e.ktir.as_ref() else { continue };
        for op in k.func.operations {
            if op.op_type != ktir_core::opkind::OpKind::KtdpConstructMemoryView {
                continue;
            }
            let Some(ktir_core::ir::Attr::IntList(dims)) =
                op.attr(ktir_core::attrkey::AttrKey::Shape)
            else {
                continue;
            };
            if dims.len() == 2 && dims[1] == hidden as i64 {
                matmul_rows.push(dims[0]);
            }
        }
        if k.func.name.starts_with("rope_s") {
            rope_count += 1;
        }
    }
    // The activation view is the `[rows, hidden]` one whose row count is the
    // prompt's; the weight's transposed view `[qw, hidden]` also matches
    // `dims[1] == hidden`, so assert on the FOLD SIGNATURE directly: an
    // `[1, hidden]` activation view (and no `[m, hidden]` one) is the m=1
    // fold; here BOTH must be absent/1-free.
    assert!(
        matmul_rows.contains(&(m as i64)),
        "the collision matmul must view its activation at all {m} rows, saw {matmul_rows:?}"
    );
    assert!(
        !matmul_rows.contains(&1),
        "the collision matmul must not fold to a [1, hidden] activation view, saw {matmul_rows:?}"
    );
    assert_eq!(
        rope_count, m as usize,
        "one per-row rope program per position ({m})"
    );
}

/// ⭐ THE TAIL'S SECOND HALF AS THE FRONT END PRODUCES IT — granite's 49155-wide
/// logits ScalarMul arrives at the splice as `n_blocks` COLUMN CHUNKS sharing one
/// output tensor (six of 8192 + one of 3 in production; two of 128 + one of 64
/// here, the same shape minimized to whole sticks), and the fold's width conjunct
/// reads the TENSOR's cols — not the chunk region's — so EVERY chunk folds to the
/// m=1 tail: no chunk left behind.
///
/// ⛔ THIS IS THE LEG-55 DEFECT'S REGRESSION PIN. With the width conjunct keyed on
/// the region (`node.output.region.cols.len`), every chunk's width failed the
/// equality against the result's, the chunks lowered un-folded at `[m, chunk]`,
/// and the read-before-write walk REFUSED the bake — the chunks read m−1 rows the
/// folded m=1 matmul never wrote (prefill leg 55, `scalarmul_o1410` reading
/// `t1409`; main carries the same shape, verified by token-diff of both the
/// chunker and the predicate). The test lowers through the descriptor-bearing
/// entry (`lower_graph_to_superdsc`) and runs the WALK itself over the emission,
/// so its verdict is stated here, not inherited from whichever entry happens to
/// walk: pre-fix it dies with the `read-before-write:` refusal naming the chunk
/// op and the unwritten interval.
#[test]
fn the_chunked_logits_scalarmul_folds_every_chunk_under_the_folded_tail() {
    // m=7 (prefill-shaped), hidden=512, vocab=2560 = 2×1024 + 512 — the production
    // shape (6×8192 + 3) minimized UNDER THE DEVICE-WIDTH LAW: the scalarmul family
    // pads its window and storage through `DeviceWidth::for_pointwise`, whose stick
    // bump takes any sub-8-stick width to 8 sticks — so a minimized chunk must
    // already clear the 8-stick floor (1024 = 16 sticks, the 512 tail = 8) or a
    // corner chunk's padded tile leaves its (equally padded) view and the door
    // refuses. Production's 8192-wide chunks (128 sticks) fill the machine and
    // never bump; this geometry states the same law at test scale.
    let (m, hidden, vocab, nb) = (7u32, 512u32, 2560u32, 1024u32);
    let scale = 0.5f32;
    let ir = scalarmul_chunk_tail_ir(m, hidden, vocab, nb, scale);
    // The production chunker's own count — the fixture states the front end's law,
    // not a hand-rolled twin of it.
    let chunks = scratchy_subtile::subtile_ir::n_blocks(
        vocab,
        std::num::NonZeroU32::new(nb).expect("nb nonzero"),
    )
    .len();
    assert!(chunks >= 1, "chunk count sanity");
    assert_eq!(
        chunks, 3,
        "2560 = 2×1024 + 512 is THREE chunks (full chunks and a short tail)"
    );

    // (a) THE WALK PASSES over the descriptor-bearing emission — the same verdict
    // the bake-time walk states on the tape path. Pre-fix this is the leg-55
    // refusal: the un-folded chunks read m−1 rows the folded m=1 matmul never
    // wrote, and the walk names the chunk op and the unwritten interval. (The
    // walk needs the door's stamped bindings, so this lowers through the
    // descriptor-bearing entry; the program-view assertions below lower the same
    // graph again through the ktir entry, which keeps the programs.)
    let weight_ids = std::collections::HashSet::new();
    let (sdsc_ops, _layout) =
        superdsc::lower_graph_to_superdsc(&ir, &weight_ids, superdsc::ActiveCap::FULL, false)
            .unwrap_or_else(|e| panic!("lower the chunked tail through the door: {e:?}"));
    scratchy_target_spyre::read_before_write::walk_read_before_write(&[(
        "chunked tail",
        &sdsc_ops,
    )])
    .unwrap_or_else(|e| panic!("the walk's verdict on the chunked tail: {e}"));

    // (a′) TOUCHED PIN B — the corner chunk's stamp, at the 8-STICK FLOOR. The
    // production corner is 3 cols → 1 fp16 stick → `bump_sticks_to_splittable`
    // pads it to 8 sticks = 512 lanes = 1,024 B (work.rs:218-222; the leg-56
    // covered interval [98,304, 99,328)). This fixture's corner is 512 logical
    // cols = exactly 8 sticks — the SAME floor the production corner bumps up to,
    // so the stamp pins the number both geometries ride: ONE merged block of
    // 1,024 B, no strided outer, at the base.
    let corner = sdsc_ops
        .iter()
        .rev()
        .find(|e| e.op_name.starts_with("scalarmul_o"))
        .expect("the corner chunk's door op");
    let t4 = corner
        .arg_bindings
        .iter()
        .find(|b| b.is_input && b.buffer == "t4")
        .expect("the corner chunk's logits input binding");
    assert_eq!(
        t4.touched.run_bytes, 1024,
        "the corner chunk sweeps the 8-stick floor: ONE run of 1,024 B (the number the \
         production 3-wide corner bumps to), got {}",
        t4.touched.run_bytes
    );
    assert!(
        t4.touched.outer.is_empty(),
        "the corner's 8 stick groups MERGE at m=1 (stride == run): no strided outer"
    );
    assert_eq!(
        t4.touched.offsets(),
        vec![0],
        "one block, at the folded row-0 base the walk credits"
    );

    let (ops, layout) =
        superdsc::lower_graph_to_ktir(&ir, &weight_ids, superdsc::ActiveCap::FULL, false)
            .unwrap_or_else(|e| panic!("lower the chunked tail: {e:?}"));

    // (b) The matmul half folded: ONE last-row extraction.
    let lmlast: Vec<&str> = ops
        .iter()
        .filter_map(|e| e.ktir.as_ref().map(|k| k.func.name))
        .filter(|n| n.starts_with("lmlast_s"))
        .collect();
    assert_eq!(
        lmlast.len(),
        1,
        "the vocab-wide matmul at m>1 folds: {lmlast:?}"
    );

    // (c) EVERY chunk folded — no chunk left behind. One scalarmul program per
    // chunk, and every memory view it constructs is ONE row tall (the m=1 tail's
    // prefix view `[1, n_total]`), never m rows: the un-folded chunk is exactly
    // the `[m, chunk]` view the leg-55 refusal convicted.
    let mut scalarmul_programs = 0usize;
    for e in &ops {
        let Some(k) = e.ktir.as_ref() else { continue };
        if !k.func.name.starts_with("scalarmul_s") {
            continue;
        }
        scalarmul_programs += 1;
        for op in k.func.operations {
            if op.op_type != ktir_core::opkind::OpKind::KtdpConstructMemoryView {
                continue;
            }
            let Some(ktir_core::ir::Attr::IntList(dims)) =
                op.attr(ktir_core::attrkey::AttrKey::Shape)
            else {
                continue;
            };
            if dims.len() == 2 {
                assert_eq!(
                    dims[0], 1,
                    "the folded chunk views ONE row (the m=1 tail), saw {dims:?} in {}",
                    k.func.name
                );
            }
        }
    }
    assert_eq!(
        scalarmul_programs, chunks,
        "all {chunks} chunks folded — no chunk left behind"
    );

    // (d) Numerics: row 0 of the session result equals the golden LAST row. The
    // golden is the WHOLE-region twin (one vocab-wide ScalarMul) under `eval_dag`
    // — the chunks partition the same columns, so the fold is correct iff row 0
    // matches elementwise.
    let (w_fuf, w_disk) = weight_pair(hidden as usize, vocab as usize);
    let x = synth(0, (m * hidden) as usize);
    let gamma = synth(1, hidden as usize);
    let golden_ir = whole_scalarmul_tail_ir(m, hidden, vocab, scale);
    let golden_srcs: Vec<Vec<f32>> = vec![x.clone(), gamma.clone(), w_fuf];
    let refs: Vec<&[f32]> = golden_srcs.iter().map(|v| v.as_slice()).collect();
    let bufs = scratchy_subtile::subtile_ir::eval_dag(&golden_ir, &refs);
    let golden = bufs[golden_ir.result.index()].clone();

    let groups = superdsc::ktir_groups(&ops, superdsc::FoldGrouping::Split)
        .expect("bind the chunked tail's programs");
    let out_tid = ir.result.index() as u64;
    let mut session = scratchy_target_spyre::runner::SpyreSession::new_multi(
        &[(&groups, &[out_tid])],
        Vec::new(),
        &[],
    )
    .expect("build the chunked tail session");
    // ⭐ THE COMPILE-TIME SCALARS, STAGED THE WORKER'S OWN WAY: each registry
    // scale at its reserved `scalarmul_scale_tid(i)` as a `[1,1]` dynamic source
    // (`spyre_forward`'s own law — leaving one unbound is a silent zero).
    let mut staged: Vec<(u64, Vec<f32>, Vec<usize>)> = vec![
        (0, x, vec![m as usize, hidden as usize]),
        (1, gamma, vec![1, hidden as usize]),
    ];
    for (i, v) in layout.scalarmul_scales.iter().enumerate() {
        let tid = superdsc::scalarmul_scale_tid(i);
        staged.push((u64::from(tid), vec![*v], vec![1, 1]));
    }
    // The weight in the on-disk [n, k] orientation, staged at the TENSOR's own
    // registered shape — `run_tail`'s law (`ir.tensors[id]` dims, the loader's
    // verbatim orientation).
    staged.push((2, w_disk, vec![hidden as usize, vocab as usize]));
    let mut out = session
        .run_step(0, staged, &[(out_tid, 0)])
        .expect("run the chunked tail");
    let got = out
        .remove(&out_tid)
        .expect("the chunked tail's result tensor");

    let v = vocab as usize;
    let last = m as usize - 1;
    let got_row0 = &got[..v];
    let golden_last = &golden[last * v..(last + 1) * v];
    for (j, (g, w)) in got_row0.iter().zip(golden_last).enumerate() {
        let d = (g - w).abs();
        assert!(
            d < 0.05 * w.abs().max(1.0),
            "scaled logit row0[{j}] = {g}, golden last-row {w} — a chunk folded to the wrong \
             row or corner (the c_start class, #566, is the corner's own law)"
        );
    }
}

/// ⭐ THE TAIL'S THIRD KIND, CHUNKED — the TanhSoftCap arm sits behind the SAME
/// one-site predicate, so a chunked softcap must fold too. The fixture is a
/// HAND-BUILT corner-0 chunk (the front end's chunker never tiles the softcap —
/// its chunk set is the elementwise/scalarmul/silumul family — and the splice's
/// own corner guard REFUSES a softcap at a nonzero column corner as a stated
/// kernel-family worklist item), so the corner-0 chunk is the legal form the
/// predicate must fold: the softcap sits ON the logits, and the folded tail caps
/// ONE row instead of m rows of which m−1 were never written.
#[test]
fn the_chunked_logits_softcap_folds_its_corner_zero_chunk() {
    // vocab=2560 with the chunk at cols [0, 512) — the result's columns past 512
    // are written by nothing and read by nothing — the walk judges reads. (512 =
    // 8 sticks, the same device-width floor the scalarmul fixture states.)
    let (m, hidden, vocab, w) = (7u32, 512u32, 2560u32, 512u32);
    let ir = softcap_chunk_tail_ir(m, hidden, vocab, w);

    // The walk over the descriptor-bearing emission (the door's stamped
    // bindings) — the softcap twin of the scalarmul pin's verdict. The
    // program-view assertions below lower the same graph through the ktir
    // entry, which keeps the programs.
    let weight_ids = std::collections::HashSet::new();
    let (sdsc_ops, _layout) =
        superdsc::lower_graph_to_superdsc(&ir, &weight_ids, superdsc::ActiveCap::FULL, false)
            .unwrap_or_else(|e| panic!("lower the chunked softcap tail through the door: {e:?}"));
    scratchy_target_spyre::read_before_write::walk_read_before_write(&[(
        "softcap tail",
        &sdsc_ops,
    )])
    .unwrap_or_else(|e| panic!("the walk's verdict on the chunked softcap tail: {e}"));
    let (ops, _layout) =
        superdsc::lower_graph_to_ktir(&ir, &weight_ids, superdsc::ActiveCap::FULL, false)
            .unwrap_or_else(|e| panic!("lower the chunked softcap tail: {e:?}"));

    let lmlast: Vec<&str> = ops
        .iter()
        .filter_map(|e| e.ktir.as_ref().map(|k| k.func.name))
        .filter(|n| n.starts_with("lmlast_s"))
        .collect();
    assert_eq!(
        lmlast.len(),
        1,
        "the vocab-wide matmul at m>1 folds: {lmlast:?}"
    );

    // The softcap chunk folded: its program's memory views are ONE row tall.
    let mut softcap_programs = 0usize;
    for e in &ops {
        let Some(k) = e.ktir.as_ref() else { continue };
        if !k.func.name.starts_with("tanhsoftcap_s") {
            continue;
        }
        softcap_programs += 1;
        for op in k.func.operations {
            if op.op_type != ktir_core::opkind::OpKind::KtdpConstructMemoryView {
                continue;
            }
            let Some(ktir_core::ir::Attr::IntList(dims)) =
                op.attr(ktir_core::attrkey::AttrKey::Shape)
            else {
                continue;
            };
            if dims.len() == 2 {
                assert_eq!(
                    dims[0], 1,
                    "the folded softcap chunk views ONE row, saw {dims:?} in {}",
                    k.func.name
                );
            }
        }
    }
    assert_eq!(
        softcap_programs, 1,
        "the corner-0 softcap chunk folded to the m=1 tail"
    );

    // Numerics: row 0's first `w` columns equal the golden last row's — the
    // whole-twin softcap graph under eval_dag, compared over the chunk's columns.
    let (w_fuf, w_disk) = weight_pair(hidden as usize, vocab as usize);
    let x = synth(0, (m * hidden) as usize);
    let gamma = synth(1, hidden as usize);
    let golden_ir = tail_ir(m, hidden, vocab);
    let golden_srcs: Vec<Vec<f32>> = vec![x.clone(), gamma.clone(), w_fuf];
    let refs: Vec<&[f32]> = golden_srcs.iter().map(|v| v.as_slice()).collect();
    let bufs = scratchy_subtile::subtile_ir::eval_dag(&golden_ir, &refs);
    let golden = bufs[golden_ir.result.index()].clone();

    let (got, _raw) = run_tail(&ir, &[x, gamma, w_disk], &[]);
    let last = m as usize - 1;
    for (j, (g, w_ref)) in got[..w as usize]
        .iter()
        .zip(&golden[last * vocab as usize..][..w as usize])
        .enumerate()
    {
        let d = (g - w_ref).abs();
        assert!(
            d < 0.05 * w_ref.abs().max(1.0),
            "capped logit row0[{j}] = {g}, golden last-row {w_ref}"
        );
    }
}

/// The chunked logits ScalarMul tail: RmsNorm → vocab-wide MatmulTile → the
/// production chunker's `n_blocks(vocab, nb)` ScalarMul chunks, all sharing the
/// result tensor `t5`. The chunk nodes are the front end's own region-slice form
/// (`lower_region`'s elementwise arm).
fn scalarmul_chunk_tail_ir(m: u32, hidden: u32, vocab: u32, nb: u32, scale: f32) -> SubtileIR {
    let tensors = vec![
        TensorShape {
            rows: m,
            cols: hidden,
        }, // t0 x (source)
        TensorShape {
            rows: 1,
            cols: hidden,
        }, // t1 gamma (source)
        TensorShape {
            rows: hidden,
            cols: vocab,
        }, // t2 lm_head weight [k, n] (source)
        TensorShape {
            rows: m,
            cols: hidden,
        }, // t3 norm out
        TensorShape {
            rows: m,
            cols: vocab,
        }, // t4 logits
        TensorShape {
            rows: m,
            cols: vocab,
        }, // t5 scaled (result)
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
    let mut nodes = vec![
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
    ];
    // The production chunker, verbatim: contiguous blocks, last may be shorter.
    let blocks = scratchy_subtile::subtile_ir::n_blocks(
        vocab,
        std::num::NonZeroU32::new(nb).expect("nb nonzero"),
    );
    for (i, blk) in blocks.iter().enumerate() {
        let chunk = |t: usize| TensorRegion {
            tensor: TensorId::from_index(t),
            region: scratchy_subtile::subtile_ir::Region {
                rows: scratchy_subtile::subtile_ir::Range::new(0, m),
                cols: *blk,
            },
        };
        nodes.push(mk(
            2 + i,
            SubOp::ScalarMul { scale },
            vec![chunk(4)],
            chunk(5),
        ));
    }
    SubtileIR {
        tensors,
        num_sources,
        nodes,
        result: TensorId::from_index(5),
        op_output: Vec::new(),
    }
}

/// The whole-region twin of the chunked ScalarMul tail (the golden's graph): one
/// vocab-wide ScalarMul, no chunking — `nb = vocab` collapses the production
/// chunker to a single whole block.
fn whole_scalarmul_tail_ir(m: u32, hidden: u32, vocab: u32, scale: f32) -> SubtileIR {
    scalarmul_chunk_tail_ir(m, hidden, vocab, vocab, scale)
}

/// The chunked softcap tail: ONE corner-0 TanhSoftCap chunk of the vocab-wide
/// result. The columns past the chunk are written by nothing and read by nothing.
fn softcap_chunk_tail_ir(m: u32, hidden: u32, vocab: u32, w: u32) -> SubtileIR {
    let mut ir = scalarmul_chunk_tail_ir(m, hidden, vocab, vocab, 0.5);
    // Replace the whole-region ScalarMul epilogue with a TanhSoftCap over just
    // the first `w` columns.
    let last = ir.nodes.pop().expect("the epilogue node");
    let SubOp::ScalarMul { .. } = last.op else {
        unreachable!("the chunk builder's last node is the epilogue")
    };
    let chunk = |t: usize| TensorRegion {
        tensor: TensorId::from_index(t),
        region: scratchy_subtile::subtile_ir::Region {
            rows: scratchy_subtile::subtile_ir::Range::new(0, m),
            cols: scratchy_subtile::subtile_ir::Range::new(0, w),
        },
    };
    ir.nodes.push(SubtileNode {
        id: last.id,
        op: SubOp::TanhSoftCap { cap: 30.0 },
        inputs: vec![chunk(4)],
        output: chunk(5),
    });
    ir
}
