// SPDX-License-Identifier: Apache-2.0
//! PrimIR — the SECOND distinct intermediate IR on the SDSC lowering ladder, below `MacroIR`.
//!
//! Its OWN type, OWN op vocabulary, OWN `eval`. The pass `lower_macro_to_prim` expands each `MacroOp`
//! into AT MOST 3 PrimOps — pure primitives a device backend can map 1:1 (matmul / elementwise /
//! row-reduce / broadcast-mul / the three attention sub-steps):
//!   * `InvRms      → {Square, RowSum, RsqrtMeanEps}`          (square → Σ rows → 1/√(mean+eps))
//!   * `ApplyNorm   → {MulRowBcast, MulColBcast}`              (x·inv[row] then ·γ[col])
//!   * `RopeBlend   → {MulHeadBcast, MulHeadBcast, Add}`       (x·cos + rot·sin)
//!   * `Attn        → {Score, Softmax, WeightedValue}`         (QKᵀ·scale+mask → softmax → ·V)
//!   * `Matmul/Sum/Mul/Add/Silu/RotateHalf` map 1:1.
//! `Softmax` is itself 6 sub-ops (max/sub/exp/sum/recip/mul) > 3 — it is split by the NEXT IR
//! (`SoftmaxIR`), exactly as the ≤3-fan-out rule requires.
//!
//! Every PrimOp's `eval` reproduces the matching `MacroOp` (and thus `eval_node`) arithmetic in the
//! SAME f32 order, so the decomposition is bit-faithful — the bridge guard [`verify_against_eval_dag`]
//! anchors PrimIR's forward result against `eval_dag` (the golden) per original tensor.

use crate::macro_ir::{MacroIR, MacroOp, seed_src};
use scratchy_subtile::subtile_ir::{RopeForm, SubtileIR, eval_dag};

/// Logical row-major shape (PrimIR's own).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub rows: u32,
    pub cols: u32,
}

/// A per-input region into the WHOLE tensor: `(rows.start, rows.len, cols.start, cols.len)` — the
/// `gather` geometry, carried for attention (prefix-K/V `valid_len−1` slice + head-block col slice).
pub type Region = (u32, u32, u32, u32);

/// One PrimIR op. `ins`/`out` index `PrimIR::tensors`.
#[derive(Clone, Debug)]
pub enum PrimOp {
    /// `out[m,n] = Σ_k a[m,k]·w[k,n]`, w row-major `[K,N]`. ins: [a, w].
    Matmul,
    /// `out = Σ ins` (split-K combine / residual). ins: ≥1.
    Sum,
    /// `out = a·b` (elementwise, equal shape). ins: [a, b].
    Mul,
    /// `out = a+b` (elementwise, equal shape). ins: [a, b].
    Add,
    /// `out = silu(a)`. ins: [a].
    Silu,
    /// `out = a·scale` (unary scalar multiply — granite multipliers). ins: [a].
    ScalarMul { scale: f32 },
    /// NeoX rotate-half per head. ins: [x].
    RotateHalf { head_dim: u32 },
    /// `out = a²` (elementwise). ins: [a].
    Square,
    /// Row reduce: `out[m,1] = Σ_j in[m,j]`. ins: [x(m,d)].
    RowSum,
    /// `out[m,1] = 1/√(in[m,0]/n + eps)` (n = the original row width). ins: [ss(m,1)].
    RsqrtMeanEps { eps: f32, n: u32 },
    /// Broadcast-mul over columns: `out[i,j] = a[i,j]·b[i,0]`. ins: [a(m,d), b(m,1)].
    MulRowBcast,
    /// Broadcast-mul over rows: `out[i,j] = a[i,j]·b[0,j]`. ins: [a(m,d), b(1,d)].
    MulColBcast,
    /// Broadcast-mul over rows AND heads: `out[r, h·hd+d] = a[r,h·hd+d]·b[0,d]`. ins: [a, b(1,hd)].
    MulHeadBcast { head_dim: u32 },
    /// Attention scores `S[mq·qh, seq] = (Q·Kᵀ)·scale` with causal mask (`s > seq−mq+qi ⇒ −inf`),
    /// region-gathered + GQA-mapped. ins: [Q, K0, K1, …]; `regions` parallel to ins.
    Score {
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        scale: f32,
        regions: Vec<Region>,
        out_cols_start: u32,
    },
    /// Rowwise softmax of `[rows, cols]` (max-shift, exp, normalize). ins: [S].
    Softmax,
    /// `out[qi, hl·hd+d] = Σ_s P[qi·qh+hl, s]·V[s, kv, d]`, region-gathered V + GQA. ins: [P, V0, V1, …];
    /// `v_regions` parallel to the V inputs (ins[1..]).
    WeightedValue {
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        v_regions: Vec<Region>,
        out_cols_start: u32,
    },
}

#[derive(Clone, Debug)]
pub struct PrimNode {
    pub op: PrimOp,
    pub ins: Vec<usize>,
    pub out: usize,
}

/// A distinct lowering IR: a primitive op graph. `tensors[0..num_sources]` are leaf sources
/// (index-aligned with the originating SubtileIR's sources); ids `≥num_sources` are op outputs.
#[derive(Clone, Debug)]
pub struct PrimIR {
    pub tensors: Vec<Shape>,
    pub num_sources: u32,
    pub nodes: Vec<PrimNode>,
    pub result: usize,
}

impl PrimIR {
    /// Host interpreter. `sources[s]` = row-major buffer for source `s`. Each op reproduces the
    /// matching `MacroOp`/`eval_node` arithmetic in the same f32 order (bit-faithful decomposition).
    pub fn eval(&self, sources: &[&[f32]]) -> Vec<Vec<f32>> {
        let mut bufs: Vec<Vec<f32>> = self
            .tensors
            .iter()
            .map(|t| vec![0f32; (t.rows * t.cols) as usize])
            .collect();
        for (s, src) in sources.iter().enumerate() {
            bufs[s].copy_from_slice(src);
        }
        for n in &self.nodes {
            let osh = self.tensors[n.out];
            let (or, oc) = (osh.rows as usize, osh.cols as usize);
            let out: Vec<f32> = match &n.op {
                PrimOp::Matmul => {
                    let a = &bufs[n.ins[0]];
                    let w = &bufs[n.ins[1]];
                    let ash = self.tensors[n.ins[0]];
                    let (m, k, nn) = (ash.rows as usize, ash.cols as usize, oc);
                    let mut o = vec![0f32; m * nn];
                    for i in 0..m {
                        for j in 0..nn {
                            let mut s = 0f32;
                            for l in 0..k {
                                s += a[i * k + l] * w[l * nn + j];
                            }
                            o[i * nn + j] = s;
                        }
                    }
                    o
                }
                PrimOp::Sum => {
                    let mut o = vec![0f32; or * oc];
                    for &inp in &n.ins {
                        for (d, v) in o.iter_mut().zip(&bufs[inp]) {
                            *d += *v;
                        }
                    }
                    o
                }
                PrimOp::Mul => bufs[n.ins[0]]
                    .iter()
                    .zip(&bufs[n.ins[1]])
                    .map(|(&x, &y)| x * y)
                    .collect(),
                PrimOp::Add => bufs[n.ins[0]]
                    .iter()
                    .zip(&bufs[n.ins[1]])
                    .map(|(&x, &y)| x + y)
                    .collect(),
                PrimOp::Silu => bufs[n.ins[0]]
                    .iter()
                    .map(|&x| x / (1.0 + (-x).exp()))
                    .collect(),
                PrimOp::ScalarMul { scale } => bufs[n.ins[0]].iter().map(|&x| x * *scale).collect(),
                PrimOp::RotateHalf { head_dim } => {
                    let x = &bufs[n.ins[0]];
                    let hd = *head_dim as usize;
                    let half = hd / 2;
                    let heads = oc / hd;
                    let mut o = x.clone();
                    for r in 0..or {
                        for h in 0..heads {
                            let base = r * oc + h * hd;
                            for d in 0..half {
                                o[base + d] = -x[base + half + d];
                                o[base + half + d] = x[base + d];
                            }
                        }
                    }
                    o
                }
                PrimOp::Square => bufs[n.ins[0]].iter().map(|&x| x * x).collect(),
                PrimOp::RowSum => {
                    let x = &bufs[n.ins[0]];
                    let d = self.tensors[n.ins[0]].cols as usize;
                    let mut o = vec![0f32; or];
                    for i in 0..or {
                        let row = &x[i * d..(i + 1) * d];
                        o[i] = row.iter().copied().sum();
                    }
                    o
                }
                PrimOp::RsqrtMeanEps { eps, n: nn } => {
                    let x = &bufs[n.ins[0]];
                    x.iter()
                        .map(|&ss| 1.0 / (ss / *nn as f32 + *eps).sqrt())
                        .collect()
                }
                PrimOp::MulRowBcast => {
                    let a = &bufs[n.ins[0]];
                    let b = &bufs[n.ins[1]]; // [m,1]
                    let mut o = vec![0f32; or * oc];
                    for i in 0..or {
                        for j in 0..oc {
                            o[i * oc + j] = a[i * oc + j] * b[i];
                        }
                    }
                    o
                }
                PrimOp::MulColBcast => {
                    let a = &bufs[n.ins[0]];
                    let b = &bufs[n.ins[1]]; // [1,d]
                    let mut o = vec![0f32; or * oc];
                    for i in 0..or {
                        for j in 0..oc {
                            o[i * oc + j] = a[i * oc + j] * b[j];
                        }
                    }
                    o
                }
                PrimOp::MulHeadBcast { head_dim } => {
                    let a = &bufs[n.ins[0]];
                    let b = &bufs[n.ins[1]]; // [1,hd]
                    let hd = *head_dim as usize;
                    let heads = oc / hd;
                    let mut o = vec![0f32; or * oc];
                    for r in 0..or {
                        for h in 0..heads {
                            let base = r * oc + h * hd;
                            for d in 0..hd {
                                o[base + d] = a[base + d] * b[d];
                            }
                        }
                    }
                    o
                }
                PrimOp::Score {
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    scale,
                    regions,
                    out_cols_start,
                } => {
                    let hd = *head_dim as usize;
                    let gqa = (*num_q_heads / (*num_kv_heads).max(1)) as usize;
                    let g =
                        |idx: usize| -> Vec<f32> { gather(self, &bufs, n.ins[idx], regions[idx]) };
                    let q = g(0);
                    let (_, mq, qc, _) = (
                        regions[0].0,
                        regions[0].1 as usize,
                        regions[0].3 as usize,
                        regions[0].2,
                    );
                    let qh_count = qc / hd;
                    let qh_start = *out_cols_start as usize / hd;
                    let kvh_start = regions[1].2 as usize / hd;
                    let mut k_all: Vec<f32> = Vec::new();
                    let mut kv_count = 0usize;
                    for i in 1..n.ins.len() {
                        kv_count = regions[i].3 as usize / hd;
                        k_all.extend_from_slice(&g(i));
                    }
                    let seg_w = kv_count * hd;
                    let seq_len = if seg_w > 0 { k_all.len() / seg_w } else { 0 };
                    let mut o = vec![0f32; mq * qh_count * seq_len];
                    for qi in 0..mq {
                        for hl in 0..qh_count {
                            let global_kv = (qh_start + hl) / gqa.max(1);
                            let local_kv = global_kv - kvh_start;
                            let q_off = qi * qh_count * hd + hl * hd;
                            let causal_bound = seq_len + qi - mq;
                            let row = qi * qh_count + hl;
                            for s in 0..seq_len {
                                if s > causal_bound {
                                    o[row * seq_len + s] = f32::NEG_INFINITY;
                                    continue;
                                }
                                let k_off = s * seg_w + local_kv * hd;
                                let mut dot = 0f32;
                                for d in 0..hd {
                                    dot += q[q_off + d] * k_all[k_off + d];
                                }
                                o[row * seq_len + s] = dot * scale;
                            }
                        }
                    }
                    o
                }
                PrimOp::Softmax => {
                    let x = &bufs[n.ins[0]];
                    let cols = oc;
                    let mut o = vec![0f32; or * cols];
                    for r in 0..or {
                        let row = &x[r * cols..(r + 1) * cols];
                        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mut sum = 0f32;
                        let dst = &mut o[r * cols..(r + 1) * cols];
                        for (d, &v) in dst.iter_mut().zip(row) {
                            *d = (v - mx).exp();
                            sum += *d;
                        }
                        for d in dst.iter_mut() {
                            *d /= sum;
                        }
                    }
                    o
                }
                PrimOp::WeightedValue {
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    v_regions,
                    out_cols_start,
                } => {
                    let hd = *head_dim as usize;
                    let gqa = (*num_q_heads / (*num_kv_heads).max(1)) as usize;
                    let p = &bufs[n.ins[0]]; // scores [mq*qh_count, seq_len]
                    let qh_count = oc / hd;
                    let mq = or;
                    let qh_start = *out_cols_start as usize / hd;
                    let kvh_start = v_regions[0].2 as usize / hd;
                    let mut v_all: Vec<f32> = Vec::new();
                    let mut kv_count = 0usize;
                    for (vi, i) in (1..n.ins.len()).enumerate() {
                        kv_count = v_regions[vi].3 as usize / hd;
                        v_all.extend_from_slice(&gather(self, &bufs, n.ins[i], v_regions[vi]));
                    }
                    let seg_w = kv_count * hd;
                    let seq_len = if seg_w > 0 { v_all.len() / seg_w } else { 0 };
                    let mut o = vec![0f32; mq * qh_count * hd];
                    for qi in 0..mq {
                        for hl in 0..qh_count {
                            let global_kv = (qh_start + hl) / gqa.max(1);
                            let local_kv = global_kv - kvh_start;
                            let srow = qi * qh_count + hl;
                            let q_off = qi * qh_count * hd + hl * hd;
                            for d in 0..hd {
                                let mut val = 0f32;
                                for s in 0..seq_len {
                                    val += p[srow * seq_len + s]
                                        * v_all[s * seg_w + local_kv * hd + d];
                                }
                                o[q_off + d] = val;
                            }
                        }
                    }
                    o
                }
            };
            bufs[n.out] = out;
        }
        bufs
    }
}

/// gather `tid`'s region `(r0,nr,c0,nc)` from the WHOLE tensor (stride = its full cols).
fn gather(ir: &PrimIR, bufs: &[Vec<f32>], tid: usize, region: Region) -> Vec<f32> {
    let full_cols = ir.tensors[tid].cols as usize;
    let (r0, nr, c0, nc) = (
        region.0 as usize,
        region.1 as usize,
        region.2 as usize,
        region.3 as usize,
    );
    let src = &bufs[tid];
    let mut g = Vec::with_capacity(nr * nc);
    for i in 0..nr {
        let base = (r0 + i) * full_cols + c0;
        g.extend_from_slice(&src[base..base + nc]);
    }
    g
}

/// THE PASS: `MacroIR → PrimIR`. Each `MacroOp` expands into ≤3 PrimOps (see module docs). Original
/// tensor ids carry over (so `verify_against_eval_dag` can seed identically); decomposition
/// intermediates get fresh ids `≥ tensors.len()`. Pair every call with `verify_against_eval_dag`.
pub fn lower_macro_to_prim(m: &MacroIR) -> PrimIR {
    let mut tensors: Vec<Shape> = m
        .tensors
        .iter()
        .map(|t| Shape {
            rows: t.rows,
            cols: t.cols,
        })
        .collect();
    let mut nodes: Vec<PrimNode> = Vec::new();
    let push_shape = |tensors: &mut Vec<Shape>, sh: Shape| -> usize {
        tensors.push(sh);
        tensors.len() - 1
    };
    for node in &m.nodes {
        let out = node.out;
        let ins = node.ins.clone();
        let osh = tensors[out];
        match &node.op {
            MacroOp::Matmul => nodes.push(PrimNode {
                op: PrimOp::Matmul,
                ins,
                out,
            }),
            MacroOp::Sum => nodes.push(PrimNode {
                op: PrimOp::Sum,
                ins,
                out,
            }),
            MacroOp::Mul => nodes.push(PrimNode {
                op: PrimOp::Mul,
                ins,
                out,
            }),
            MacroOp::Add => nodes.push(PrimNode {
                op: PrimOp::Add,
                ins,
                out,
            }),
            MacroOp::Silu => nodes.push(PrimNode {
                op: PrimOp::Silu,
                ins,
                out,
            }),
            MacroOp::ScalarMul { scale } => nodes.push(PrimNode {
                op: PrimOp::ScalarMul { scale: *scale },
                ins,
                out,
            }),
            MacroOp::RotateHalf { head_dim } => nodes.push(PrimNode {
                op: PrimOp::RotateHalf {
                    head_dim: *head_dim,
                },
                ins,
                out,
            }),
            MacroOp::InvRms { eps } => {
                // Square(x)→sq[m,d]; RowSum(sq)→ss[m,1]; RsqrtMeanEps(ss)→inv[m,1]  (fan-out 3)
                let xsh = tensors[ins[0]];
                let d = xsh.cols;
                let sq = push_shape(&mut tensors, xsh);
                nodes.push(PrimNode {
                    op: PrimOp::Square,
                    ins: vec![ins[0]],
                    out: sq,
                });
                let ss = push_shape(
                    &mut tensors,
                    Shape {
                        rows: osh.rows,
                        cols: 1,
                    },
                );
                nodes.push(PrimNode {
                    op: PrimOp::RowSum,
                    ins: vec![sq],
                    out: ss,
                });
                nodes.push(PrimNode {
                    op: PrimOp::RsqrtMeanEps { eps: *eps, n: d },
                    ins: vec![ss],
                    out,
                });
            }
            MacroOp::ApplyNorm => {
                // MulRowBcast(x, inv)→t[m,d]; MulColBcast(t, gamma)→out  (fan-out 2)
                let t = push_shape(&mut tensors, osh);
                nodes.push(PrimNode {
                    op: PrimOp::MulRowBcast,
                    ins: vec![ins[0], ins[1]],
                    out: t,
                });
                nodes.push(PrimNode {
                    op: PrimOp::MulColBcast,
                    ins: vec![t, ins[2]],
                    out,
                });
            }
            MacroOp::RopeBlend { head_dim } => {
                // MulHeadBcast(x,cos)→a; MulHeadBcast(rot,sin)→b; Add(a,b)→out  (fan-out 3)
                let a = push_shape(&mut tensors, osh);
                nodes.push(PrimNode {
                    op: PrimOp::MulHeadBcast {
                        head_dim: *head_dim,
                    },
                    ins: vec![ins[0], ins[2]],
                    out: a,
                });
                let b = push_shape(&mut tensors, osh);
                nodes.push(PrimNode {
                    op: PrimOp::MulHeadBcast {
                        head_dim: *head_dim,
                    },
                    ins: vec![ins[1], ins[3]],
                    out: b,
                });
                nodes.push(PrimNode {
                    op: PrimOp::Add,
                    ins: vec![a, b],
                    out,
                });
            }
            MacroOp::Attn {
                num_q_heads,
                num_kv_heads,
                head_dim,
                scale,
                in_regions,
                out_cols_start,
            } => {
                // Score(Q,Ks)→S[mq·qh, seq]; Softmax(S)→P; WeightedValue(P,Vs)→out  (fan-out 3).
                let hd = *head_dim as usize;
                let q_region = in_regions[0];
                let mq = q_region.1 as usize;
                let qh_count = q_region.3 as usize / hd;
                // ins = [Q, K0, V0, K1, V1, …]; split into K-list / V-list (regions parallel).
                let (mut k_ins, mut k_regs) = (Vec::new(), Vec::new());
                let (mut v_ins, mut v_regs) = (Vec::new(), Vec::new());
                let mut seq_len = 0usize;
                let mut i = 1;
                while i < ins.len() {
                    k_ins.push(ins[i]);
                    k_regs.push(in_regions[i]);
                    v_ins.push(ins[i + 1]);
                    v_regs.push(in_regions[i + 1]);
                    seq_len += in_regions[i].1 as usize; // K seg rows.len
                    i += 2;
                }
                // Score: ins [Q, K0, K1, …], regions [Q, K0, K1, …].
                let scores = push_shape(
                    &mut tensors,
                    Shape {
                        rows: (mq * qh_count) as u32,
                        cols: seq_len as u32,
                    },
                );
                let mut score_ins = vec![ins[0]];
                score_ins.extend_from_slice(&k_ins);
                let mut score_regs = vec![q_region];
                score_regs.extend_from_slice(&k_regs);
                nodes.push(PrimNode {
                    op: PrimOp::Score {
                        num_q_heads: *num_q_heads,
                        num_kv_heads: *num_kv_heads,
                        head_dim: *head_dim,
                        scale: *scale,
                        regions: score_regs,
                        out_cols_start: *out_cols_start,
                    },
                    ins: score_ins,
                    out: scores,
                });
                // Softmax on the scores matrix.
                let probs = push_shape(
                    &mut tensors,
                    Shape {
                        rows: (mq * qh_count) as u32,
                        cols: seq_len as u32,
                    },
                );
                nodes.push(PrimNode {
                    op: PrimOp::Softmax,
                    ins: vec![scores],
                    out: probs,
                });
                // WeightedValue: ins [P, V0, V1, …]; v_regions parallel to the V inputs.
                let mut wv_ins = vec![probs];
                wv_ins.extend_from_slice(&v_ins);
                nodes.push(PrimNode {
                    op: PrimOp::WeightedValue {
                        num_q_heads: *num_q_heads,
                        num_kv_heads: *num_kv_heads,
                        head_dim: *head_dim,
                        v_regions: v_regs,
                        out_cols_start: *out_cols_start,
                    },
                    ins: wv_ins,
                    out,
                });
            }
        }
    }
    PrimIR {
        tensors,
        num_sources: m.num_sources,
        nodes,
        result: m.result,
    }
}

/// VERIFY the `MacroIR → PrimIR` bridge against the golden `eval_dag`, per original tensor, with the
/// SAME per-tensor RELATIVE-L2 metric as bridge 1 (`‖g−o‖₂/‖g‖₂ ≤ 1e-2`): robust to f32 cancellation
/// at a lone element, trips hard on any real op/region bug (whole-tensor or per-head). PrimIR's sources
/// are index-aligned with `subtile`, so the same per-index seed feeds both. Panics (un-catchable
/// `cargo build` failure) on divergence, naming the producing PrimOp.
pub fn verify_against_eval_dag<F: RopeForm>(prim: &PrimIR, subtile: &SubtileIR<F>) {
    const RELL2_TOL: f64 = 1e-2;
    const EPS: f64 = 1e-9;
    assert_eq!(
        prim.num_sources, subtile.num_sources,
        "PrimIR bridge: source count {} != SubtileIR {} (sources must be index-aligned)",
        prim.num_sources, subtile.num_sources
    );
    let src: Vec<Vec<f32>> = (0..subtile.num_sources as usize)
        .map(|s| {
            let sh = subtile.tensors[s];
            (0..(sh.rows * sh.cols) as usize)
                .map(|j| seed_src(s, j))
                .collect()
        })
        .collect();
    let refs: Vec<&[f32]> = src.iter().map(|v| v.as_slice()).collect();
    let golden = eval_dag(subtile, &refs);
    let got = prim.eval(&refs);

    let n_orig = subtile.tensors.len();
    let mut worst_rell2 = 0f64;
    let mut worst_tid = 0usize;
    for tid in 0..n_orig {
        let g = &golden[tid];
        let o = &got[tid];
        assert_eq!(
            g.len(),
            o.len(),
            "PrimIR bridge: tensor t{tid} size {} != golden {} — the pass changed a tensor's shape",
            o.len(),
            g.len()
        );
        let mut num = 0f64;
        let mut den = 0f64;
        for i in 0..g.len() {
            let (gv, ov) = (g[i] as f64, o[i] as f64);
            num += (gv - ov) * (gv - ov);
            den += gv * gv;
        }
        let rell2 = num.sqrt() / den.sqrt().max(EPS);
        if rell2 > worst_rell2 {
            worst_rell2 = rell2;
            worst_tid = tid;
        }
    }
    let producing_op = prim
        .nodes
        .iter()
        .find(|n| n.out == worst_tid)
        .map(|n| format!("{:?}", n.op))
        .unwrap_or_else(|| "<source/leaf>".into());
    let (g, o) = (&golden[worst_tid], &got[worst_tid]);
    let (mut n_div, mut worst_rel, mut wg, mut wo) = (0usize, 0f64, 0f64, 0f64);
    for i in 0..g.len() {
        let (gv, ov) = (g[i] as f64, o[i] as f64);
        let rel = (gv - ov).abs() / gv.abs().max(EPS);
        if rel > 1e-2 {
            n_div += 1;
        }
        if rel > worst_rel {
            worst_rel = rel;
            wg = gv;
            wo = ov;
        }
    }
    if std::env::var_os("SCRATCHY_SUPERDSC_DBG").is_some() {
        eprintln!(
            "[sdsc-stage] PrimIR bridge: {n_orig} tensors, worst relative-L2 {worst_rell2:.2e} at \
             t{worst_tid} (op {producing_op}); worst element eval_dag={wg} PrimIR={wo} (relΔ \
             {worst_rel:.2e}); {n_div}/{} elements >1% [relL2 ≤ {RELL2_TOL} ⇒ faithful]",
            g.len()
        );
    }
    assert!(
        worst_rell2 <= RELL2_TOL,
        "PrimIR bridge diverges at t{worst_tid} (op {producing_op}): relative-L2 {worst_rell2:.2e} > \
         {RELL2_TOL}, {n_div}/{} elements >1% (worst element eval_dag={wg} PrimIR={wo}). The \
         MacroIR→PrimIR pass changed the math at this op — THIS bridge is the bug.",
        g.len()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::macro_ir::{MacroIR, MacroNode, MacroOp, Shape as MShape};

    /// PrimIR.eval == MacroIR.eval on a hand-built rmsnorm+MLP+residual shape — proving the
    /// `MacroIR→PrimIR` decomposition (InvRms→{Square,RowSum,RsqrtMeanEps}, ApplyNorm→{MulRowBcast,
    /// MulColBcast}, etc.) is bit-faithful. (Full-graph anchoring vs eval_dag runs at build time.)
    #[test]
    fn prim_eval_matches_macro_rmsnorm_mlp() {
        let (m, d, inter) = (2usize, 4usize, 6usize);
        let tensors = vec![
            MShape {
                rows: m as u32,
                cols: d as u32,
            }, // 0 x
            MShape {
                rows: 1,
                cols: d as u32,
            }, // 1 gamma
            MShape {
                rows: d as u32,
                cols: inter as u32,
            }, // 2 Wg
            MShape {
                rows: d as u32,
                cols: inter as u32,
            }, // 3 Wu
            MShape {
                rows: inter as u32,
                cols: d as u32,
            }, // 4 Wd
            MShape {
                rows: m as u32,
                cols: 1,
            }, // 5 inv
            MShape {
                rows: m as u32,
                cols: d as u32,
            }, // 6 normed
            MShape {
                rows: m as u32,
                cols: inter as u32,
            }, // 7 gate
            MShape {
                rows: m as u32,
                cols: inter as u32,
            }, // 8 up
            MShape {
                rows: m as u32,
                cols: inter as u32,
            }, // 9 silu
            MShape {
                rows: m as u32,
                cols: inter as u32,
            }, // 10 silu*up
            MShape {
                rows: m as u32,
                cols: d as u32,
            }, // 11 down
            MShape {
                rows: m as u32,
                cols: d as u32,
            }, // 12 out = x + down
        ];
        let nodes = vec![
            MacroNode {
                op: MacroOp::InvRms { eps: 1e-5 },
                ins: vec![0],
                out: 5,
            },
            MacroNode {
                op: MacroOp::ApplyNorm,
                ins: vec![0, 5, 1],
                out: 6,
            },
            MacroNode {
                op: MacroOp::Matmul,
                ins: vec![6, 2],
                out: 7,
            },
            MacroNode {
                op: MacroOp::Matmul,
                ins: vec![6, 3],
                out: 8,
            },
            MacroNode {
                op: MacroOp::Silu,
                ins: vec![7],
                out: 9,
            },
            MacroNode {
                op: MacroOp::Mul,
                ins: vec![9, 8],
                out: 10,
            },
            MacroNode {
                op: MacroOp::Matmul,
                ins: vec![10, 4],
                out: 11,
            },
            MacroNode {
                op: MacroOp::Add,
                ins: vec![0, 11],
                out: 12,
            },
        ];
        let mir = MacroIR {
            tensors,
            num_sources: 5,
            nodes,
            result: 12,
        };
        let prim = lower_macro_to_prim(&mir);
        let src: Vec<Vec<f32>> = (0..5)
            .map(|s| {
                (0..(mir.tensors[s].rows * mir.tensors[s].cols) as usize)
                    .map(|j| seed_src(s, j))
                    .collect()
            })
            .collect();
        let refs: Vec<&[f32]> = src.iter().map(|v| v.as_slice()).collect();
        let gm = mir.eval(&refs);
        let gp = prim.eval(&refs);
        for tid in 0..mir.tensors.len() {
            for i in 0..gm[tid].len() {
                assert!(
                    (gm[tid][i] - gp[tid][i]).abs() < 1e-6,
                    "t{tid}[{i}] macro={} prim={}",
                    gm[tid][i],
                    gp[tid][i]
                );
            }
        }
    }
}
