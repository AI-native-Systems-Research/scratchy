// SPDX-License-Identifier: Apache-2.0
//! SmIR1 — the THIRD distinct intermediate IR on the SDSC ladder, below `PrimIR`. It splits the one
//! remaining composite (`Softmax`, which is 6 sub-ops > 3) into its FIRST three pieces, obeying the
//! ≤3-fan-out rule:
//!   `Softmax → {RowMax, SubExp, Normalize}`   (max-shift → exp(x−m) → divide by row-sum)
//! `SubExp` and `Normalize` are themselves sub-composites — the NEXT IR (`FlatIR`) splits them
//! (`SubExp→{SubRowBcast,Exp}`, `Normalize→{RowSum,DivRowBcast}`), each ≤2. Every other `PrimOp` maps
//! 1:1. Each op's `eval` reproduces the matching `PrimOp`/`eval_node` arithmetic in the same f32 order.

use crate::macro_ir::seed_src;
use crate::prim_ir::{PrimIR, PrimOp, Region};
use scratchy_subtile::subtile_ir::{RopeForm, SubtileIR, eval_dag};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub rows: u32,
    pub cols: u32,
}

/// One SmIR1 op. Identical to `PrimOp` except `Softmax` is replaced by `{RowMax, SubExp, Normalize}`.
#[derive(Clone, Debug)]
pub enum SmOp {
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
    // softmax pieces:
    /// `out[r,0] = max_j in[r,j]`. ins: [x(R,C)].
    RowMax,
    /// `out[r,j] = exp(x[r,j] − m[r,0])`. ins: [x(R,C), m(R,1)].
    SubExp,
    /// `out[r,j] = e[r,j] / Σ_j' e[r,j']`. ins: [e(R,C)].
    Normalize,
}

#[derive(Clone, Debug)]
pub struct SmNode {
    pub op: SmOp,
    pub ins: Vec<usize>,
    pub out: usize,
}

#[derive(Clone, Debug)]
pub struct SmIR {
    pub tensors: Vec<Shape>,
    pub num_sources: u32,
    pub nodes: Vec<SmNode>,
    pub result: usize,
}

impl SmIR {
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
                SmOp::Matmul => {
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
                SmOp::Sum => {
                    let mut o = vec![0f32; or * oc];
                    for &inp in &n.ins {
                        for (d, v) in o.iter_mut().zip(&bufs[inp]) {
                            *d += *v;
                        }
                    }
                    o
                }
                SmOp::Mul => bufs[n.ins[0]]
                    .iter()
                    .zip(&bufs[n.ins[1]])
                    .map(|(&x, &y)| x * y)
                    .collect(),
                SmOp::Add => bufs[n.ins[0]]
                    .iter()
                    .zip(&bufs[n.ins[1]])
                    .map(|(&x, &y)| x + y)
                    .collect(),
                SmOp::Silu => bufs[n.ins[0]]
                    .iter()
                    .map(|&x| x / (1.0 + (-x).exp()))
                    .collect(),
                SmOp::ScalarMul { scale } => bufs[n.ins[0]].iter().map(|&x| x * *scale).collect(),
                SmOp::RotateHalf { head_dim } => {
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
                SmOp::Square => bufs[n.ins[0]].iter().map(|&x| x * x).collect(),
                SmOp::RowSum => {
                    let x = &bufs[n.ins[0]];
                    let d = self.tensors[n.ins[0]].cols as usize;
                    let mut o = vec![0f32; or];
                    for i in 0..or {
                        o[i] = x[i * d..(i + 1) * d].iter().copied().sum();
                    }
                    o
                }
                SmOp::RsqrtMeanEps { eps, n: nn } => bufs[n.ins[0]]
                    .iter()
                    .map(|&ss| 1.0 / (ss / *nn as f32 + *eps).sqrt())
                    .collect(),
                SmOp::MulRowBcast => {
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
                SmOp::MulColBcast => {
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
                SmOp::MulHeadBcast { head_dim } => {
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
                SmOp::Score {
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
                SmOp::WeightedValue {
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
                SmOp::RowMax => {
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
                SmOp::SubExp => {
                    let x = &bufs[n.ins[0]];
                    let m = &bufs[n.ins[1]]; // [R,1]
                    let mut o = vec![0f32; or * oc];
                    for i in 0..or {
                        for j in 0..oc {
                            o[i * oc + j] = (x[i * oc + j] - m[i]).exp();
                        }
                    }
                    o
                }
                SmOp::Normalize => {
                    let e = &bufs[n.ins[0]];
                    let mut o = vec![0f32; or * oc];
                    for i in 0..or {
                        let sum: f32 = e[i * oc..(i + 1) * oc].iter().copied().sum();
                        for j in 0..oc {
                            o[i * oc + j] = e[i * oc + j] / sum;
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

/// THE PASS: `PrimIR → SmIR1`. Splits `Softmax → {RowMax, SubExp, Normalize}` (fan-out 3); every other
/// op maps 1:1. Original tensor ids carry over; softmax intermediates (`m`, `e`) get fresh ids.
pub fn lower_prim_to_sm(p: &PrimIR) -> SmIR {
    let mut tensors: Vec<Shape> = p
        .tensors
        .iter()
        .map(|t| Shape {
            rows: t.rows,
            cols: t.cols,
        })
        .collect();
    let mut nodes: Vec<SmNode> = Vec::new();
    for node in &p.nodes {
        let out = node.out;
        let ins = node.ins.clone();
        let osh = tensors[out];
        macro_rules! one {
            ($op:expr) => {
                nodes.push(SmNode { op: $op, ins, out })
            };
        }
        match &node.op {
            PrimOp::Matmul => one!(SmOp::Matmul),
            PrimOp::Sum => one!(SmOp::Sum),
            PrimOp::Mul => one!(SmOp::Mul),
            PrimOp::Add => one!(SmOp::Add),
            PrimOp::Silu => one!(SmOp::Silu),
            PrimOp::ScalarMul { scale } => one!(SmOp::ScalarMul { scale: *scale }),
            PrimOp::RotateHalf { head_dim } => one!(SmOp::RotateHalf {
                head_dim: *head_dim
            }),
            PrimOp::Square => one!(SmOp::Square),
            PrimOp::RowSum => one!(SmOp::RowSum),
            PrimOp::RsqrtMeanEps { eps, n } => one!(SmOp::RsqrtMeanEps { eps: *eps, n: *n }),
            PrimOp::MulRowBcast => one!(SmOp::MulRowBcast),
            PrimOp::MulColBcast => one!(SmOp::MulColBcast),
            PrimOp::MulHeadBcast { head_dim } => one!(SmOp::MulHeadBcast {
                head_dim: *head_dim
            }),
            PrimOp::Score {
                num_q_heads,
                num_kv_heads,
                head_dim,
                scale,
                regions,
                out_cols_start,
            } => {
                one!(SmOp::Score {
                    num_q_heads: *num_q_heads,
                    num_kv_heads: *num_kv_heads,
                    head_dim: *head_dim,
                    scale: *scale,
                    regions: regions.clone(),
                    out_cols_start: *out_cols_start,
                })
            }
            PrimOp::WeightedValue {
                num_q_heads,
                num_kv_heads,
                head_dim,
                v_regions,
                out_cols_start,
            } => {
                one!(SmOp::WeightedValue {
                    num_q_heads: *num_q_heads,
                    num_kv_heads: *num_kv_heads,
                    head_dim: *head_dim,
                    v_regions: v_regions.clone(),
                    out_cols_start: *out_cols_start,
                })
            }
            PrimOp::Softmax => {
                // RowMax(x)→m[R,1]; SubExp(x,m)→e[R,C]; Normalize(e)→out  (fan-out 3)
                let x = ins[0];
                let m = {
                    tensors.push(Shape {
                        rows: osh.rows,
                        cols: 1,
                    });
                    tensors.len() - 1
                };
                nodes.push(SmNode {
                    op: SmOp::RowMax,
                    ins: vec![x],
                    out: m,
                });
                let e = {
                    tensors.push(osh);
                    tensors.len() - 1
                };
                nodes.push(SmNode {
                    op: SmOp::SubExp,
                    ins: vec![x, m],
                    out: e,
                });
                nodes.push(SmNode {
                    op: SmOp::Normalize,
                    ins: vec![e],
                    out,
                });
            }
        }
    }
    SmIR {
        tensors,
        num_sources: p.num_sources,
        nodes,
        result: p.result,
    }
}

/// VERIFY the `PrimIR → SmIR1` bridge against the golden `eval_dag`, per original tensor, relative-L2
/// ≤ 1e-2 (same metric as bridges 1–2). Panics (un-catchable `cargo build` failure) on divergence.
pub fn verify_against_eval_dag<F: RopeForm>(sm: &SmIR, subtile: &SubtileIR<F>) {
    const RELL2_TOL: f64 = 1e-2;
    const EPS: f64 = 1e-9;
    assert_eq!(
        sm.num_sources, subtile.num_sources,
        "SmIR bridge: source count {} != SubtileIR {}",
        sm.num_sources, subtile.num_sources
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
    let got = sm.eval(&refs);

    let n_orig = subtile.tensors.len();
    let mut worst_rell2 = 0f64;
    let mut worst_tid = 0usize;
    for tid in 0..n_orig {
        let g = &golden[tid];
        let o = &got[tid];
        assert_eq!(g.len(), o.len(), "SmIR bridge: tensor t{tid} size changed");
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
    let producing_op = sm
        .nodes
        .iter()
        .find(|n| n.out == worst_tid)
        .map(|n| format!("{:?}", n.op))
        .unwrap_or_else(|| "<source/leaf>".into());
    if std::env::var_os("SCRATCHY_SUPERDSC_DBG").is_some() {
        eprintln!(
            "[sdsc-stage] SmIR bridge: {n_orig} tensors, worst relative-L2 {worst_rell2:.2e} at \
             t{worst_tid} (op {producing_op}) [relL2 ≤ {RELL2_TOL} ⇒ faithful]"
        );
    }
    assert!(
        worst_rell2 <= RELL2_TOL,
        "SmIR bridge diverges at t{worst_tid} (op {producing_op}): relative-L2 {worst_rell2:.2e} > \
         {RELL2_TOL}. The PrimIR→SmIR1 softmax split changed the math — THIS bridge is the bug."
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prim_ir::{PrimIR, PrimNode, PrimOp, Shape as PShape};

    /// SmIR1.eval == PrimIR.eval on a hand-built softmax (RowMax/SubExp/Normalize == Softmax).
    #[test]
    fn sm_softmax_split_matches_prim() {
        let (r, c) = (3usize, 5usize);
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
        let sm = lower_prim_to_sm(&pir);
        let gs = sm.eval(&refs);
        for i in 0..(r * c) {
            assert!(
                (gp[1][i] - gs[1][i]).abs() < 1e-6,
                "softmax[{i}] prim={} sm={}",
                gp[1][i],
                gs[1][i]
            );
        }
        // each row sums to 1
        for i in 0..r {
            let s: f32 = gs[1][i * c..(i + 1) * c].iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "row {i} sums to {s}");
        }
    }
}
