// SPDX-License-Identifier: Apache-2.0
//! FlatIR — the FOURTH distinct intermediate IR on the SDSC ladder, below `SmIR1`. It finishes the
//! softmax split, leaving a FULLY-FLAT math vocabulary (matmul / unary-ew / binary-ew / broadcast-ew /
//! row-reduce / gather-attn) that the layout IRs (`TiledIR`, `DeviceIR`) build on without further
//! semantic decomposition:
//!   `SubExp    → {SubRowBcast, Exp}`        (x − m[row], then eˣ)
//!   `Normalize → {RowSum, DivRowBcast}`     (Σ row, then e / s[row])
//! Every other `SmOp` maps 1:1. `DivRowBcast` (and `RsqrtMeanEps`/`Silu`) are still device-macro at the
//! SEN169 level — the `DeviceIR` bridge expands those into the on-card recip/Newton sequence. Each
//! `eval` reproduces the matching `SmOp`/`eval_node` arithmetic in the same f32 order.

use crate::macro_ir::seed_src;
use crate::prim_ir::Region;
use crate::sm_ir::{SmIR, SmOp};
use scratchy_subtile::subtile_ir::{RopeForm, SubtileIR, eval_dag};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub rows: u32,
    pub cols: u32,
}

/// One FlatIR op. Identical to `SmOp` except `SubExp`/`Normalize` are replaced by their primitives.
#[derive(Clone, Debug)]
pub enum FlatOp {
    Matmul,
    Sum,
    Mul,
    Add,
    Silu,
    ScalarMul {
        scale: f32,
    },
    RotateHalf {
        head_dim: u32,
    },
    Square,
    RowSum,
    RsqrtMeanEps {
        eps: f32,
        n: u32,
    },
    MulRowBcast,
    MulColBcast,
    MulHeadBcast {
        head_dim: u32,
    },
    Score {
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        scale: f32,
        regions: Vec<Region>,
        out_cols_start: u32,
    },
    WeightedValue {
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        v_regions: Vec<Region>,
        out_cols_start: u32,
    },
    RowMax,
    /// `out[r,j] = a[r,j] − b[r,0]` (subtract row-broadcast). ins: [a(R,C), b(R,1)].
    SubRowBcast,
    /// `out = eˣ` (elementwise). ins: [x].
    Exp,
    /// `out[r,j] = a[r,j] / b[r,0]` (divide row-broadcast). ins: [a(R,C), b(R,1)].
    DivRowBcast,
}

#[derive(Clone, Debug)]
pub struct FlatNode {
    pub op: FlatOp,
    pub ins: Vec<usize>,
    pub out: usize,
}

#[derive(Clone, Debug)]
pub struct FlatIR {
    pub tensors: Vec<Shape>,
    pub num_sources: u32,
    pub nodes: Vec<FlatNode>,
    pub result: usize,
}

impl FlatIR {
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
                FlatOp::Matmul => {
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
                FlatOp::Sum => {
                    let mut o = vec![0f32; or * oc];
                    for &inp in &n.ins {
                        for (d, v) in o.iter_mut().zip(&bufs[inp]) {
                            *d += *v;
                        }
                    }
                    o
                }
                FlatOp::Mul => bufs[n.ins[0]]
                    .iter()
                    .zip(&bufs[n.ins[1]])
                    .map(|(&x, &y)| x * y)
                    .collect(),
                FlatOp::Add => bufs[n.ins[0]]
                    .iter()
                    .zip(&bufs[n.ins[1]])
                    .map(|(&x, &y)| x + y)
                    .collect(),
                FlatOp::Silu => bufs[n.ins[0]]
                    .iter()
                    .map(|&x| x / (1.0 + (-x).exp()))
                    .collect(),
                FlatOp::ScalarMul { scale } => bufs[n.ins[0]].iter().map(|&x| x * *scale).collect(),
                FlatOp::RotateHalf { head_dim } => {
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
                FlatOp::Square => bufs[n.ins[0]].iter().map(|&x| x * x).collect(),
                FlatOp::RowSum => {
                    let x = &bufs[n.ins[0]];
                    let d = self.tensors[n.ins[0]].cols as usize;
                    let mut o = vec![0f32; or];
                    for i in 0..or {
                        o[i] = x[i * d..(i + 1) * d].iter().copied().sum();
                    }
                    o
                }
                FlatOp::RsqrtMeanEps { eps, n: nn } => bufs[n.ins[0]]
                    .iter()
                    .map(|&ss| 1.0 / (ss / *nn as f32 + *eps).sqrt())
                    .collect(),
                FlatOp::MulRowBcast => {
                    let a = &bufs[n.ins[0]];
                    let b = &bufs[n.ins[1]];
                    let mut o = vec![0f32; or * oc];
                    for i in 0..or {
                        for j in 0..oc {
                            o[i * oc + j] = a[i * oc + j] * b[i];
                        }
                    }
                    o
                }
                FlatOp::MulColBcast => {
                    let a = &bufs[n.ins[0]];
                    let b = &bufs[n.ins[1]];
                    let mut o = vec![0f32; or * oc];
                    for i in 0..or {
                        for j in 0..oc {
                            o[i * oc + j] = a[i * oc + j] * b[j];
                        }
                    }
                    o
                }
                FlatOp::MulHeadBcast { head_dim } => {
                    let a = &bufs[n.ins[0]];
                    let b = &bufs[n.ins[1]];
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
                FlatOp::Score {
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    scale,
                    regions,
                    out_cols_start,
                } => {
                    let hd = *head_dim as usize;
                    let gqa = (*num_q_heads / (*num_kv_heads).max(1)) as usize;
                    let g = |idx: usize| -> Vec<f32> {
                        gather(&self.tensors, &bufs, n.ins[idx], regions[idx])
                    };
                    let q = g(0);
                    let (mq, qc) = (regions[0].1 as usize, regions[0].3 as usize);
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
                FlatOp::WeightedValue {
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    v_regions,
                    out_cols_start,
                } => {
                    let hd = *head_dim as usize;
                    let gqa = (*num_q_heads / (*num_kv_heads).max(1)) as usize;
                    let p = &bufs[n.ins[0]];
                    let qh_count = oc / hd;
                    let mq = or;
                    let qh_start = *out_cols_start as usize / hd;
                    let kvh_start = v_regions[0].2 as usize / hd;
                    let mut v_all: Vec<f32> = Vec::new();
                    let mut kv_count = 0usize;
                    for (vi, i) in (1..n.ins.len()).enumerate() {
                        kv_count = v_regions[vi].3 as usize / hd;
                        v_all.extend_from_slice(&gather(
                            &self.tensors,
                            &bufs,
                            n.ins[i],
                            v_regions[vi],
                        ));
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
                FlatOp::RowMax => {
                    let x = &bufs[n.ins[0]];
                    let c = self.tensors[n.ins[0]].cols as usize;
                    let mut o = vec![0f32; or];
                    for i in 0..or {
                        o[i] = x[i * c..(i + 1) * c]
                            .iter()
                            .cloned()
                            .fold(f32::NEG_INFINITY, f32::max);
                    }
                    o
                }
                FlatOp::SubRowBcast => {
                    let a = &bufs[n.ins[0]];
                    let b = &bufs[n.ins[1]]; // [R,1]
                    let mut o = vec![0f32; or * oc];
                    for i in 0..or {
                        for j in 0..oc {
                            o[i * oc + j] = a[i * oc + j] - b[i];
                        }
                    }
                    o
                }
                FlatOp::Exp => bufs[n.ins[0]].iter().map(|&x| x.exp()).collect(),
                FlatOp::DivRowBcast => {
                    let a = &bufs[n.ins[0]];
                    let b = &bufs[n.ins[1]]; // [R,1]
                    let mut o = vec![0f32; or * oc];
                    for i in 0..or {
                        for j in 0..oc {
                            o[i * oc + j] = a[i * oc + j] / b[i];
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

fn gather(tensors: &[Shape], bufs: &[Vec<f32>], tid: usize, region: Region) -> Vec<f32> {
    let full_cols = tensors[tid].cols as usize;
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

/// THE PASS: `SmIR1 → FlatIR`. `SubExp → {SubRowBcast, Exp}`; `Normalize → {RowSum, DivRowBcast}`;
/// everything else 1:1. Original ids carry over; the new intermediates (`d=x−m`, `s=Σe`) get fresh ids.
pub fn lower_sm_to_flat(sm: &SmIR) -> FlatIR {
    let mut tensors: Vec<Shape> = sm
        .tensors
        .iter()
        .map(|t| Shape {
            rows: t.rows,
            cols: t.cols,
        })
        .collect();
    let mut nodes: Vec<FlatNode> = Vec::new();
    for node in &sm.nodes {
        let out = node.out;
        let ins = node.ins.clone();
        let osh = tensors[out];
        macro_rules! one {
            ($op:expr) => {
                nodes.push(FlatNode { op: $op, ins, out })
            };
        }
        match &node.op {
            SmOp::Matmul => one!(FlatOp::Matmul),
            SmOp::Sum => one!(FlatOp::Sum),
            SmOp::Mul => one!(FlatOp::Mul),
            SmOp::Add => one!(FlatOp::Add),
            SmOp::Silu => one!(FlatOp::Silu),
            SmOp::ScalarMul { scale } => one!(FlatOp::ScalarMul { scale: *scale }),
            SmOp::RotateHalf { head_dim } => one!(FlatOp::RotateHalf {
                head_dim: *head_dim
            }),
            SmOp::Square => one!(FlatOp::Square),
            SmOp::RowSum => one!(FlatOp::RowSum),
            SmOp::RsqrtMeanEps { eps, n } => one!(FlatOp::RsqrtMeanEps { eps: *eps, n: *n }),
            SmOp::MulRowBcast => one!(FlatOp::MulRowBcast),
            SmOp::MulColBcast => one!(FlatOp::MulColBcast),
            SmOp::MulHeadBcast { head_dim } => one!(FlatOp::MulHeadBcast {
                head_dim: *head_dim
            }),
            SmOp::Score {
                num_q_heads,
                num_kv_heads,
                head_dim,
                scale,
                regions,
                out_cols_start,
            } => {
                one!(FlatOp::Score {
                    num_q_heads: *num_q_heads,
                    num_kv_heads: *num_kv_heads,
                    head_dim: *head_dim,
                    scale: *scale,
                    regions: regions.clone(),
                    out_cols_start: *out_cols_start,
                })
            }
            SmOp::WeightedValue {
                num_q_heads,
                num_kv_heads,
                head_dim,
                v_regions,
                out_cols_start,
            } => {
                one!(FlatOp::WeightedValue {
                    num_q_heads: *num_q_heads,
                    num_kv_heads: *num_kv_heads,
                    head_dim: *head_dim,
                    v_regions: v_regions.clone(),
                    out_cols_start: *out_cols_start,
                })
            }
            SmOp::RowMax => one!(FlatOp::RowMax),
            SmOp::SubExp => {
                // SubRowBcast(x,m)→d[R,C]; Exp(d)→out  (fan-out 2)
                let x = ins[0];
                let m = ins[1];
                let d = {
                    tensors.push(osh);
                    tensors.len() - 1
                };
                nodes.push(FlatNode {
                    op: FlatOp::SubRowBcast,
                    ins: vec![x, m],
                    out: d,
                });
                nodes.push(FlatNode {
                    op: FlatOp::Exp,
                    ins: vec![d],
                    out,
                });
            }
            SmOp::Normalize => {
                // RowSum(e)→s[R,1]; DivRowBcast(e,s)→out  (fan-out 2)
                let e = ins[0];
                let s = {
                    tensors.push(Shape {
                        rows: osh.rows,
                        cols: 1,
                    });
                    tensors.len() - 1
                };
                nodes.push(FlatNode {
                    op: FlatOp::RowSum,
                    ins: vec![e],
                    out: s,
                });
                nodes.push(FlatNode {
                    op: FlatOp::DivRowBcast,
                    ins: vec![e, s],
                    out,
                });
            }
        }
    }
    FlatIR {
        tensors,
        num_sources: sm.num_sources,
        nodes,
        result: sm.result,
    }
}

/// VERIFY the `SmIR1 → FlatIR` bridge against the golden `eval_dag`, per original tensor, relative-L2
/// ≤ 1e-2. Panics (un-catchable `cargo build` failure) on divergence.
pub fn verify_against_eval_dag<F: RopeForm>(flat: &FlatIR, subtile: &SubtileIR<F>) {
    const RELL2_TOL: f64 = 1e-2;
    const EPS: f64 = 1e-9;
    assert_eq!(
        flat.num_sources, subtile.num_sources,
        "FlatIR bridge: source count {} != SubtileIR {}",
        flat.num_sources, subtile.num_sources
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
    let got = flat.eval(&refs);

    let n_orig = subtile.tensors.len();
    let mut worst_rell2 = 0f64;
    let mut worst_tid = 0usize;
    for tid in 0..n_orig {
        let g = &golden[tid];
        let o = &got[tid];
        assert_eq!(
            g.len(),
            o.len(),
            "FlatIR bridge: tensor t{tid} size changed"
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
    let producing_op = flat
        .nodes
        .iter()
        .find(|n| n.out == worst_tid)
        .map(|n| format!("{:?}", n.op))
        .unwrap_or_else(|| "<source/leaf>".into());
    if std::env::var_os("SCRATCHY_SUPERDSC_DBG").is_some() {
        eprintln!(
            "[sdsc-stage] FlatIR bridge: {n_orig} tensors, worst relative-L2 {worst_rell2:.2e} at \
             t{worst_tid} (op {producing_op}) [relL2 ≤ {RELL2_TOL} ⇒ faithful]"
        );
    }
    assert!(
        worst_rell2 <= RELL2_TOL,
        "FlatIR bridge diverges at t{worst_tid} (op {producing_op}): relative-L2 {worst_rell2:.2e} > \
         {RELL2_TOL}. The SmIR1→FlatIR softmax-finish split changed the math — THIS bridge is the bug."
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prim_ir::{PrimIR, PrimNode, PrimOp, Shape as PShape};
    use crate::sm_ir::lower_prim_to_sm;

    /// FlatIR.eval == golden softmax (full PrimIR→SmIR1→FlatIR chain on a softmax row block).
    #[test]
    fn flat_softmax_chain_matches() {
        let (r, c) = (4usize, 7usize);
        let tensors = vec![
            PShape {
                rows: r as u32,
                cols: c as u32,
            },
            PShape {
                rows: r as u32,
                cols: c as u32,
            },
        ];
        let nodes = vec![PrimNode {
            op: PrimOp::Softmax,
            ins: vec![0],
            out: 1,
        }];
        let pir = PrimIR {
            tensors,
            num_sources: 1,
            nodes,
            result: 1,
        };
        let src: Vec<Vec<f32>> = vec![(0..(r * c)).map(|j| seed_src(0, j)).collect()];
        let refs: Vec<&[f32]> = src.iter().map(|v| v.as_slice()).collect();
        let gp = pir.eval(&refs);
        let flat = lower_sm_to_flat(&lower_prim_to_sm(&pir));
        let gf = flat.eval(&refs);
        for i in 0..(r * c) {
            assert!(
                (gp[1][i] - gf[1][i]).abs() < 1e-6,
                "softmax[{i}] prim={} flat={}",
                gp[1][i],
                gf[1][i]
            );
        }
    }
}
