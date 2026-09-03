// SPDX-License-Identifier: Apache-2.0
//! DeviceIR — the FIFTH distinct IR on the SDSC ladder, below `FlatIR`. It is the FIRST IR whose `eval`
//! runs over the **device address model** (`scratchy_subtile::sdsc_abstract::dev_off`): every tensor is
//! laid out in device memory with a stick axis (`[b/64, a, 64]` when sticking the last dim), and every
//! op reads/writes through that physical addressing. This is where a LAYOUT bug (the documented K-cache
//! producer/consumer mismatch: the score matmul reads K as a `[hd,cap]` cap-sticked Kᵀ KERNEL, but K is
//! produced natural `[seq,hd]`) becomes a wrong result — caught at build time vs `eval_dag`.
//!
//! Distinct from `FlatIR`: tensors carry a `DevLayout{rows,cols,stick_idx,base}` and the attention
//! `Score` consumes K through an explicit `RestickifyKt` (transpose to the `[hd,cap]` kernel) — modeling
//! the on-card restickify the emitter must perform. With the restickify present the bridge is GREEN and
//! IS the correct lowering to emit; omitting/mis-shaping it makes the bridge RED, localizing the bug.
//!
//! For tensors with consistent producer/consumer layouts (everything except the K kernel), `dev_off`
//! is a per-tensor bijection, so the eval reproduces `eval_dag` exactly — the verification is therefore
//! meaningful precisely at the layout boundaries (the K restickify), which is the on-card bug locus.

use crate::flat_ir::{FlatIR, FlatOp};
use crate::macro_ir::seed_src;
use crate::prim_ir::Region;
use scratchy_subtile::sdsc_abstract::dev_off;
use scratchy_subtile::subtile_ir::{RopeForm, SubtileIR, eval_dag};

/// Device layout of one tensor: logical `[rows, cols]`, the stick axis, and its base device address.
#[derive(Clone, Copy, Debug)]
pub struct DevLayout {
    pub rows: u32,
    pub cols: u32,
    pub stick_idx: u32,
    pub base: usize,
}
impl DevLayout {
    fn addr(&self, i: usize, j: usize) -> usize {
        self.base
            + dev_off(
                &[self.rows as usize, self.cols as usize],
                self.stick_idx as usize,
                &[i, j],
            )
    }
}

/// One DeviceIR op. Mirrors `FlatOp` plus `RestickifyKt` (transpose K→Kᵀ kernel for the score read).
#[derive(Clone, Debug)]
pub enum DevOp {
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
    SubRowBcast,
    Exp,
    DivRowBcast,
    /// Transpose a K segment's REGION `(r0,nr,c0,nc)` of the whole K tensor into `kv_count` Kᵀ kernels
    /// (each `[hd, nr]`), the layout the score matmul consumes: `o[kv·hd+d, s] = x[r0+s, c0+kv·hd+d]`.
    /// Mirrors `eval_node`/`FlatIR` gather (respects the prefix `valid_len` row slice + kv-head col
    /// offset) — using whole-tensor rows/cols here was the region bug the MacroIR Attn already hit.
    RestickifyKt {
        head_dim: u32,
        kv_count: u32,
        region: Region,
    },
}

#[derive(Clone, Debug)]
pub struct DevNode {
    pub op: DevOp,
    pub ins: Vec<usize>,
    pub out: usize,
}

#[derive(Clone, Debug)]
pub struct DeviceIR {
    pub tensors: Vec<DevLayout>,
    pub num_sources: u32,
    pub nodes: Vec<DevNode>,
    pub result: usize,
    /// Total device-memory footprint (elements) = the pass's final `next_base`. The eval allocates one
    /// flat `Vec<f32>` of this size and addresses it by `base + dev_off` — O(1), vs a HashMap (whose
    /// per-access hashing made the gemms ~50× slower and the build take >70 min).
    pub footprint: usize,
}

impl DeviceIR {
    /// Host interpreter over the device address model. Seeds sources at their device addresses, then
    /// runs each op reading/writing through `DevLayout::addr` (i.e. `dev_off`). Returns, per tensor, the
    /// logical row-major buffer (gathered back from device memory) so the result can be compared to
    /// `eval_dag` per tensor.
    pub fn eval(&self, sources: &[&[f32]]) -> Vec<Vec<f32>> {
        let mut mem = vec![0f32; self.footprint];
        // Seed sources at device addresses.
        for (s, src) in sources.iter().enumerate() {
            let l = self.tensors[s];
            let c = l.cols as usize;
            for (idx, &v) in src.iter().enumerate() {
                let (i, j) = (idx / c, idx % c);
                mem[l.addr(i, j)] = v;
            }
        }
        for n in &self.nodes {
            self.run_op(n, &mut mem);
        }
        // Gather every tensor back to a logical row-major buffer.
        self.tensors
            .iter()
            .map(|l| {
                let (r, c) = (l.rows as usize, l.cols as usize);
                let mut buf = vec![0f32; r * c];
                for i in 0..r {
                    for j in 0..c {
                        buf[i * c + j] = mem[l.addr(i, j)];
                    }
                }
                buf
            })
            .collect()
    }

    fn run_op(&self, n: &DevNode, mem: &mut [f32]) {
        let ol = self.tensors[n.out];
        let (or, oc) = (ol.rows as usize, ol.cols as usize);
        // Compute in f32 over a FLAT Vec (mem[base+dev_off]) — MATCHING eval_dag (and the other 4
        // bridges) bit-for-bit, O(1) per access (a HashMap made the gemms ~50× slower → 70-min builds).
        let rd = |mem: &[f32], t: usize, i: usize, j: usize| mem[self.tensors[t].addr(i, j)];
        match &n.op {
            DevOp::Matmul => {
                let al = self.tensors[n.ins[0]];
                let (m, k) = (al.rows as usize, al.cols as usize);
                for i in 0..m {
                    for j in 0..oc {
                        let mut acc = 0f32;
                        for l in 0..k {
                            acc += rd(mem, n.ins[0], i, l) * rd(mem, n.ins[1], l, j);
                        }
                        mem[ol.addr(i, j)] = acc;
                    }
                }
            }
            DevOp::Sum => {
                for i in 0..or {
                    for j in 0..oc {
                        let mut acc = 0f32;
                        for &inp in &n.ins {
                            acc += rd(mem, inp, i, j);
                        }
                        mem[ol.addr(i, j)] = acc;
                    }
                }
            }
            DevOp::Mul => self.ew2(n, mem, |a, b| a * b),
            DevOp::Add => self.ew2(n, mem, |a, b| a + b),
            DevOp::Silu => self.ew1(n, mem, |a| a / (1.0 + (-a).exp())),
            DevOp::ScalarMul { scale } => {
                let s = *scale;
                self.ew1(n, mem, move |a| a * s)
            }
            DevOp::Square => self.ew1(n, mem, |a| a * a),
            DevOp::Exp => self.ew1(n, mem, |a| a.exp()),
            DevOp::RotateHalf { head_dim } => {
                let hd = *head_dim as usize;
                let half = hd / 2;
                let heads = oc / hd;
                for r in 0..or {
                    for h in 0..heads {
                        let b = h * hd;
                        for d in 0..half {
                            let x_lo = rd(mem, n.ins[0], r, b + d);
                            let x_hi = rd(mem, n.ins[0], r, b + half + d);
                            mem[ol.addr(r, b + d)] = -x_hi;
                            mem[ol.addr(r, b + half + d)] = x_lo;
                        }
                    }
                }
            }
            DevOp::RowSum => {
                let c = self.tensors[n.ins[0]].cols as usize;
                for i in 0..or {
                    let mut acc = 0f32;
                    for j in 0..c {
                        acc += rd(mem, n.ins[0], i, j);
                    }
                    mem[ol.addr(i, 0)] = acc;
                }
            }
            DevOp::RowMax => {
                let c = self.tensors[n.ins[0]].cols as usize;
                for i in 0..or {
                    let mut acc = f32::NEG_INFINITY;
                    for j in 0..c {
                        acc = acc.max(rd(mem, n.ins[0], i, j));
                    }
                    mem[ol.addr(i, 0)] = acc;
                }
            }
            DevOp::RsqrtMeanEps { eps, n: nn } => {
                for i in 0..or {
                    let ss = rd(mem, n.ins[0], i, 0);
                    mem[ol.addr(i, 0)] = 1.0f32 / (ss / *nn as f32 + *eps).sqrt();
                }
            }
            DevOp::MulRowBcast => {
                for i in 0..or {
                    let b = rd(mem, n.ins[1], i, 0);
                    for j in 0..oc {
                        mem[ol.addr(i, j)] = rd(mem, n.ins[0], i, j) * b;
                    }
                }
            }
            DevOp::DivRowBcast => {
                for i in 0..or {
                    let b = rd(mem, n.ins[1], i, 0);
                    for j in 0..oc {
                        mem[ol.addr(i, j)] = rd(mem, n.ins[0], i, j) / b;
                    }
                }
            }
            DevOp::SubRowBcast => {
                for i in 0..or {
                    let b = rd(mem, n.ins[1], i, 0);
                    for j in 0..oc {
                        mem[ol.addr(i, j)] = rd(mem, n.ins[0], i, j) - b;
                    }
                }
            }
            DevOp::MulColBcast => {
                for i in 0..or {
                    for j in 0..oc {
                        mem[ol.addr(i, j)] = rd(mem, n.ins[0], i, j) * rd(mem, n.ins[1], 0, j);
                    }
                }
            }
            DevOp::MulHeadBcast { head_dim } => {
                let hd = *head_dim as usize;
                let heads = oc / hd;
                for r in 0..or {
                    for h in 0..heads {
                        let b = h * hd;
                        for d in 0..hd {
                            mem[ol.addr(r, b + d)] =
                                rd(mem, n.ins[0], r, b + d) * rd(mem, n.ins[1], 0, d);
                        }
                    }
                }
            }
            DevOp::RestickifyKt {
                head_dim,
                kv_count,
                region,
            } => {
                // region (r0,nr,c0,nc) of the whole K tensor → kv_count Kᵀ kernels [hd, nr]. `out` is
                // [kv_count·hd, nr]: row = kv·hd + d, col = s; reads x[r0+s, c0 + kv·hd + d].
                let hd = *head_dim as usize;
                let kvc = *kv_count as usize;
                let (r0, nr, c0) = (region.0 as usize, region.1 as usize, region.2 as usize);
                for kv in 0..kvc {
                    for d in 0..hd {
                        for s in 0..nr {
                            let v = rd(mem, n.ins[0], r0 + s, c0 + kv * hd + d);
                            mem[ol.addr(kv * hd + d, s)] = v;
                        }
                    }
                }
            }
            DevOp::Score {
                num_q_heads,
                num_kv_heads,
                head_dim,
                scale,
                regions,
                out_cols_start,
            } => {
                self.score(
                    n,
                    mem,
                    *num_q_heads,
                    *num_kv_heads,
                    *head_dim,
                    *scale,
                    regions,
                    *out_cols_start,
                );
            }
            DevOp::WeightedValue {
                num_q_heads,
                num_kv_heads,
                head_dim,
                v_regions,
                out_cols_start,
            } => {
                self.weighted_value(
                    n,
                    mem,
                    *num_q_heads,
                    *num_kv_heads,
                    *head_dim,
                    v_regions,
                    *out_cols_start,
                );
            }
        }
    }

    fn ew1(&self, n: &DevNode, mem: &mut [f32], f: impl Fn(f32) -> f32) {
        let ol = self.tensors[n.out];
        let (or, oc) = (ol.rows as usize, ol.cols as usize);
        for i in 0..or {
            for j in 0..oc {
                let v = mem[self.tensors[n.ins[0]].addr(i, j)];
                mem[ol.addr(i, j)] = f(v);
            }
        }
    }
    fn ew2(&self, n: &DevNode, mem: &mut [f32], f: impl Fn(f32, f32) -> f32) {
        let ol = self.tensors[n.out];
        let (or, oc) = (ol.rows as usize, ol.cols as usize);
        for i in 0..or {
            for j in 0..oc {
                let a = mem[self.tensors[n.ins[0]].addr(i, j)];
                let b = mem[self.tensors[n.ins[1]].addr(i, j)];
                mem[ol.addr(i, j)] = f(a, b);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn score(
        &self,
        n: &DevNode,
        mem: &mut [f32],
        nqh: u32,
        nkvh: u32,
        hd_u: u32,
        scale: f32,
        regions: &[Region],
        out_cols_start: u32,
    ) {
        // ins = [Q, Kt0, Kt1, …] — the K segments are ALREADY restickified to Kᵀ kernels [kv·hd, seq]
        // by RestickifyKt; `regions` here describe the ORIGINAL Q + K segments (rows.len = seq per seg,
        // cols.len/hd = kv_count). Score reads Q natural and Kt via its kernel layout.
        let hd = hd_u as usize;
        let gqa = (nqh / nkvh.max(1)) as usize;
        let ol = self.tensors[n.out];
        let (mq, qc) = (regions[0].1 as usize, regions[0].3 as usize);
        let qh_count = qc / hd;
        let qh_start = out_cols_start as usize / hd;
        let kvh_start = regions[1].2 as usize / hd;
        let q_t = n.ins[0];
        let q_r0 = regions[0].0 as usize;
        let q_c0 = regions[0].2 as usize;
        // total seq across K segments + per-segment (kt_tensor, kv_count, row_base_in_concat)
        let mut segs: Vec<(usize, usize, usize)> = Vec::new();
        let mut seq_len = 0usize;
        let mut kv_count = 0usize;
        for (si, kt) in n.ins[1..].iter().enumerate() {
            let seg_seq = regions[1 + si].1 as usize;
            kv_count = regions[1 + si].3 as usize / hd;
            segs.push((*kt, kv_count, seq_len));
            seq_len += seg_seq;
        }
        let _ = kv_count;
        for qi in 0..mq {
            for hl in 0..qh_count {
                let global_kv = (qh_start + hl) / gqa.max(1);
                let local_kv = global_kv - kvh_start;
                let row = qi * qh_count + hl;
                let causal_bound = seq_len + qi - mq;
                for &(kt, _kvc, base_s) in &segs {
                    let ktl = self.tensors[kt];
                    let seg_seq = ktl.cols as usize; // Kt is [kv·hd, seq] ⇒ cols = seq of this seg
                    for s_local in 0..seg_seq {
                        let s = base_s + s_local;
                        let val: f32 = if s > causal_bound {
                            f32::NEG_INFINITY
                        } else {
                            let mut dot = 0f32;
                            for d in 0..hd {
                                // Q[qi, hl*hd + d] (region-offset), Kt[local_kv*hd + d, s_local]
                                let q = mem[self.tensors[q_t].addr(q_r0 + qi, q_c0 + hl * hd + d)];
                                let kk = mem[ktl.addr(local_kv * hd + d, s_local)];
                                dot += q * kk;
                            }
                            dot * scale
                        };
                        mem[ol.addr(row, s)] = val;
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn weighted_value(
        &self,
        n: &DevNode,
        mem: &mut [f32],
        nqh: u32,
        nkvh: u32,
        hd_u: u32,
        v_regions: &[Region],
        out_cols_start: u32,
    ) {
        // ins = [P(scores), V0, V1, …]; V segments natural [seq, kv·hd] (V is fine natural).
        let hd = hd_u as usize;
        let gqa = (nqh / nkvh.max(1)) as usize;
        let ol = self.tensors[n.out];
        let (or, oc) = (ol.rows as usize, ol.cols as usize);
        let qh_count = oc / hd;
        let mq = or;
        let qh_start = out_cols_start as usize / hd;
        let kvh_start = v_regions[0].2 as usize / hd;
        let p_t = n.ins[0];
        // per V seg: (v_tensor, region_row_start, region_rows_len, region_col_start, base_s in concat).
        let mut segs: Vec<(usize, usize, usize, usize, usize)> = Vec::new();
        let mut base_s = 0usize;
        for (si, vt) in n.ins[1..].iter().enumerate() {
            let (r0, nr, c0) = (
                v_regions[si].0 as usize,
                v_regions[si].1 as usize,
                v_regions[si].2 as usize,
            );
            segs.push((*vt, r0, nr, c0, base_s));
            base_s += nr;
        }
        for qi in 0..mq {
            for hl in 0..qh_count {
                let global_kv = (qh_start + hl) / gqa.max(1);
                let local_kv = global_kv - kvh_start;
                let srow = qi * qh_count + hl;
                for d in 0..hd {
                    let mut acc = 0f32;
                    for &(vt, r0, nr, c0, bs) in &segs {
                        let vl = self.tensors[vt];
                        for s_local in 0..nr {
                            let p = mem[self.tensors[p_t].addr(srow, bs + s_local)];
                            let vv = mem[vl.addr(r0 + s_local, c0 + local_kv * hd + d)];
                            acc += p * vv;
                        }
                    }
                    mem[ol.addr(qi, hl * hd + d)] = acc;
                }
            }
        }
    }
}

/// THE PASS: `FlatIR → DeviceIR`. Assigns each tensor a stick-last device layout with a sequential base
/// (footprint padded to a multiple of 64 cols so sticks never overlap). For attention `Score`, inserts a
/// `RestickifyKt` per K segment (K[seq, kv·hd] → Kᵀ[kv·hd, seq] kernel) — the layout the score consumes.
pub fn lower_flat_to_device(f: &FlatIR) -> DeviceIR {
    // base allocator: each tensor gets rows * pad64(cols) cells (stick-last footprint upper bound).
    let mut next_base = 0usize;
    let mut alloc = |rows: u32, cols: u32| -> usize {
        let b = next_base;
        let pad_cols = ((cols as usize + 63) / 64) * 64;
        next_base += (rows as usize) * pad_cols.max(64);
        b
    };
    let mut tensors: Vec<DevLayout> = f
        .tensors
        .iter()
        .map(|t| DevLayout {
            rows: t.rows,
            cols: t.cols,
            stick_idx: 1,
            base: alloc(t.rows, t.cols),
        })
        .collect();
    let mut nodes: Vec<DevNode> = Vec::new();
    macro_rules! one {
        ($op:expr, $ins:expr, $out:expr) => {
            nodes.push(DevNode {
                op: $op,
                ins: $ins,
                out: $out,
            })
        };
    }
    for node in &f.nodes {
        let out = node.out;
        let ins = node.ins.clone();
        match &node.op {
            FlatOp::Matmul => one!(DevOp::Matmul, ins, out),
            FlatOp::Sum => one!(DevOp::Sum, ins, out),
            FlatOp::Mul => one!(DevOp::Mul, ins, out),
            FlatOp::Add => one!(DevOp::Add, ins, out),
            FlatOp::Silu => one!(DevOp::Silu, ins, out),
            FlatOp::ScalarMul { scale } => one!(DevOp::ScalarMul { scale: *scale }, ins, out),
            FlatOp::RotateHalf { head_dim } => one!(
                DevOp::RotateHalf {
                    head_dim: *head_dim
                },
                ins,
                out
            ),
            FlatOp::Square => one!(DevOp::Square, ins, out),
            FlatOp::RowSum => one!(DevOp::RowSum, ins, out),
            FlatOp::RsqrtMeanEps { eps, n } => {
                one!(DevOp::RsqrtMeanEps { eps: *eps, n: *n }, ins, out)
            }
            FlatOp::MulRowBcast => one!(DevOp::MulRowBcast, ins, out),
            FlatOp::MulColBcast => one!(DevOp::MulColBcast, ins, out),
            FlatOp::MulHeadBcast { head_dim } => one!(
                DevOp::MulHeadBcast {
                    head_dim: *head_dim
                },
                ins,
                out
            ),
            FlatOp::RowMax => one!(DevOp::RowMax, ins, out),
            FlatOp::SubRowBcast => one!(DevOp::SubRowBcast, ins, out),
            FlatOp::Exp => one!(DevOp::Exp, ins, out),
            FlatOp::DivRowBcast => one!(DevOp::DivRowBcast, ins, out),
            FlatOp::WeightedValue {
                num_q_heads,
                num_kv_heads,
                head_dim,
                v_regions,
                out_cols_start,
            } => {
                one!(
                    DevOp::WeightedValue {
                        num_q_heads: *num_q_heads,
                        num_kv_heads: *num_kv_heads,
                        head_dim: *head_dim,
                        v_regions: v_regions.clone(),
                        out_cols_start: *out_cols_start,
                    },
                    ins,
                    out
                );
            }
            FlatOp::Score {
                num_q_heads,
                num_kv_heads,
                head_dim,
                scale,
                regions,
                out_cols_start,
            } => {
                // Restickify each K segment (ins[1..]) → Kᵀ kernel [kv·hd, seq]; Score reads the kernels.
                let hd = *head_dim;
                let mut score_ins = vec![ins[0]];
                for (si, &k_in) in ins[1..].iter().enumerate() {
                    let reg = regions[1 + si];
                    let seg_seq = reg.1; // region rows.len (the valid_len slice)
                    let kv_count = reg.3 / hd; // region cols.len / hd
                    let kt = {
                        // Kᵀ kernel logical shape [kv_count*hd, seg_seq]; stick last (seq).
                        let base = alloc(kv_count * hd, seg_seq);
                        tensors.push(DevLayout {
                            rows: kv_count * hd,
                            cols: seg_seq,
                            stick_idx: 1,
                            base,
                        });
                        tensors.len() - 1
                    };
                    nodes.push(DevNode {
                        op: DevOp::RestickifyKt {
                            head_dim: hd,
                            kv_count,
                            region: reg,
                        },
                        ins: vec![k_in],
                        out: kt,
                    });
                    score_ins.push(kt);
                }
                nodes.push(DevNode {
                    op: DevOp::Score {
                        num_q_heads: *num_q_heads,
                        num_kv_heads: *num_kv_heads,
                        head_dim: *head_dim,
                        scale: *scale,
                        regions: regions.clone(),
                        out_cols_start: *out_cols_start,
                    },
                    ins: score_ins,
                    out,
                });
            }
        }
    }
    DeviceIR {
        tensors,
        num_sources: f.num_sources,
        nodes,
        result: f.result,
        footprint: next_base,
    }
}

/// VERIFY the `FlatIR → DeviceIR` bridge against the golden `eval_dag`, per original tensor, relative-L2
/// ≤ 1e-2. Panics (un-catchable `cargo build` failure) on divergence — a wrong device layout (e.g. a
/// missing/mis-shaped K restickify) makes the score read the wrong bytes and the bridge goes RED.
pub fn verify_against_eval_dag<F: RopeForm>(dev: &DeviceIR, subtile: &SubtileIR<F>) {
    const RELL2_TOL: f64 = 1e-2;
    const EPS: f64 = 1e-9;
    assert_eq!(
        dev.num_sources, subtile.num_sources,
        "DeviceIR bridge: source count {} != SubtileIR {}",
        dev.num_sources, subtile.num_sources
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
    let got = dev.eval(&refs);

    let n_orig = subtile.tensors.len();
    let mut worst_rell2 = 0f64;
    let mut worst_tid = 0usize;
    for tid in 0..n_orig {
        let g = &golden[tid];
        let o = &got[tid];
        assert_eq!(
            g.len(),
            o.len(),
            "DeviceIR bridge: tensor t{tid} size changed"
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
    let producing_op = dev
        .nodes
        .iter()
        .find(|n| n.out == worst_tid)
        .map(|n| format!("{:?}", n.op))
        .unwrap_or_else(|| "<source/leaf>".into());
    if std::env::var_os("SCRATCHY_SUPERDSC_DBG").is_some() {
        eprintln!(
            "[sdsc-stage] DeviceIR bridge: {n_orig} tensors, worst relative-L2 {worst_rell2:.2e} at \
             t{worst_tid} (op {producing_op}) [relL2 ≤ {RELL2_TOL} ⇒ device layout faithful]"
        );
    }
    assert!(
        worst_rell2 <= RELL2_TOL,
        "DeviceIR bridge diverges at t{worst_tid} (op {producing_op}): relative-L2 {worst_rell2:.2e} > \
         {RELL2_TOL}. The FlatIR→DeviceIR device-layout lowering reads/writes the wrong device addresses \
         at this op — THIS bridge is the layout bug (e.g. K-cache Kᵀ restickify wrong)."
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flat_ir::{FlatIR, FlatNode, FlatOp, Shape as FShape, lower_sm_to_flat};
    use crate::prim_ir::{PrimIR, PrimNode, PrimOp, Shape as PShape, lower_macro_to_prim};
    use crate::sm_ir::lower_prim_to_sm;

    /// dev_off sticking is a per-tensor bijection ⇒ a consistent-layout MLP reproduces eval exactly.
    #[test]
    fn device_mlp_matches_flat() {
        // hand-build a small matmul+silu+mul+add (the MLP shape), lower Flat→Device, compare.
        let (h, inter) = (4usize, 6usize);
        let tensors = vec![
            FShape {
                rows: 1,
                cols: h as u32,
            }, // 0 norm
            FShape {
                rows: h as u32,
                cols: inter as u32,
            }, // 1 Wg
            FShape {
                rows: h as u32,
                cols: inter as u32,
            }, // 2 Wu
            FShape {
                rows: inter as u32,
                cols: h as u32,
            }, // 3 Wd
            FShape {
                rows: 1,
                cols: h as u32,
            }, // 4 hidden
            FShape {
                rows: 1,
                cols: inter as u32,
            }, // 5 gate
            FShape {
                rows: 1,
                cols: inter as u32,
            }, // 6 up
            FShape {
                rows: 1,
                cols: inter as u32,
            }, // 7 silu
            FShape {
                rows: 1,
                cols: inter as u32,
            }, // 8 silu*up
            FShape {
                rows: 1,
                cols: h as u32,
            }, // 9 down
            FShape {
                rows: 1,
                cols: h as u32,
            }, // 10 out
        ];
        let nodes = vec![
            FlatNode {
                op: FlatOp::Matmul,
                ins: vec![0, 1],
                out: 5,
            },
            FlatNode {
                op: FlatOp::Matmul,
                ins: vec![0, 2],
                out: 6,
            },
            FlatNode {
                op: FlatOp::Silu,
                ins: vec![5],
                out: 7,
            },
            FlatNode {
                op: FlatOp::Mul,
                ins: vec![7, 6],
                out: 8,
            },
            FlatNode {
                op: FlatOp::Matmul,
                ins: vec![8, 3],
                out: 9,
            },
            FlatNode {
                op: FlatOp::Add,
                ins: vec![4, 9],
                out: 10,
            },
        ];
        let fir = FlatIR {
            tensors,
            num_sources: 5,
            nodes,
            result: 10,
        };
        let src: Vec<Vec<f32>> = (0..5)
            .map(|s| {
                (0..(fir.tensors[s].rows * fir.tensors[s].cols) as usize)
                    .map(|j| seed_src(s, j))
                    .collect()
            })
            .collect();
        let refs: Vec<&[f32]> = src.iter().map(|v| v.as_slice()).collect();
        let gf = fir.eval(&refs);
        let dev = lower_flat_to_device(&fir);
        let gd = dev.eval(&refs);
        for c in 0..h {
            assert!(
                (gf[10][c] - gd[10][c]).abs() < 1e-5,
                "out[{c}] flat={} dev={}",
                gf[10][c],
                gd[10][c]
            );
        }
    }

    /// A cap>64 K segment: the score read of the Kᵀ kernel must match the natural dot product —
    /// the documented K-cache layout case (cap=128>STK). Single q-head, single kv-head, hd=64.
    #[test]
    fn device_score_kt_cap_gt_stk() {
        let (hd, cap) = (64usize, 128usize); // cap > STK=64 is where natural≠Kᵀ
        // tensors: 0=Q[1,hd], 1=K[cap,hd], 2=V[cap,hd], 3=scores[1,cap], 4=out[1,hd]
        let tensors = vec![
            PShape {
                rows: 1,
                cols: hd as u32,
            },
            PShape {
                rows: cap as u32,
                cols: hd as u32,
            },
            PShape {
                rows: cap as u32,
                cols: hd as u32,
            },
            PShape {
                rows: 1,
                cols: cap as u32,
            },
            PShape {
                rows: 1,
                cols: hd as u32,
            },
        ];
        // Score (Q, K) → scores[1,cap]; WeightedValue (scores, V) → out[1,hd]. (No softmax: isolate layout.)
        let reg_q = (0u32, 1, 0, hd as u32);
        let reg_k = (0u32, cap as u32, 0, hd as u32);
        let nodes = vec![
            PrimNode {
                op: PrimOp::Score {
                    num_q_heads: 1,
                    num_kv_heads: 1,
                    head_dim: hd as u32,
                    scale: 1.0,
                    regions: vec![reg_q, reg_k],
                    out_cols_start: 0,
                },
                ins: vec![0, 1],
                out: 3,
            },
            PrimNode {
                op: PrimOp::WeightedValue {
                    num_q_heads: 1,
                    num_kv_heads: 1,
                    head_dim: hd as u32,
                    v_regions: vec![reg_k],
                    out_cols_start: 0,
                },
                ins: vec![3, 2],
                out: 4,
            },
        ];
        let pir = PrimIR {
            tensors,
            num_sources: 3,
            nodes,
            result: 4,
        };
        let src: Vec<Vec<f32>> = (0..3)
            .map(|s| {
                (0..(pir.tensors[s].rows * pir.tensors[s].cols) as usize)
                    .map(|j| seed_src(s, j))
                    .collect()
            })
            .collect();
        let refs: Vec<&[f32]> = src.iter().map(|v| v.as_slice()).collect();
        let gp = pir.eval(&refs);
        // chain Prim→Sm→Flat→Device (no softmax ops here, all pass through)
        let dev = lower_flat_to_device(&lower_sm_to_flat(&lower_prim_to_sm(&pir)));
        let gd = dev.eval(&refs);
        // Relative tolerance: PrimIR evals in f32, DeviceIR in f64, so a 128-term sum differs by f32
        // accumulation noise (~1e-6 relative). A WRONG Kᵀ kernel read would be O(1) relative.
        let close = |a: f32, b: f32| (a - b).abs() <= 1e-4 * a.abs().max(1.0);
        for j in 0..cap {
            assert!(
                close(gp[3][j], gd[3][j]),
                "scores[{j}] prim={} dev={} (Kᵀ kernel read wrong)",
                gp[3][j],
                gd[3][j]
            );
        }
        for d in 0..hd {
            assert!(
                close(gp[4][d], gd[4][d]),
                "out[{d}] prim={} dev={}",
                gp[4][d],
                gd[4][d]
            );
        }
    }

    /// REGION SLICE: K/V tensors are the full cache [cap,hd] but the score only attends to the first
    /// `valid` rows (region rows.len < tensor rows) — the prefix valid_len slice. RestickifyKt +
    /// WeightedValue must respect the region (using whole-tensor rows would diverge). cap=128>STK.
    #[test]
    fn device_score_region_row_slice() {
        let (hd, cap, valid) = (64usize, 128usize, 80usize); // attend to first 80 of 128 cache rows
        let tensors = vec![
            PShape {
                rows: 1,
                cols: hd as u32,
            }, // 0 Q
            PShape {
                rows: cap as u32,
                cols: hd as u32,
            }, // 1 K (full cache)
            PShape {
                rows: cap as u32,
                cols: hd as u32,
            }, // 2 V (full cache)
            PShape {
                rows: 1,
                cols: valid as u32,
            }, // 3 scores [1, valid]
            PShape {
                rows: 1,
                cols: hd as u32,
            }, // 4 out
        ];
        let reg_q = (0u32, 1, 0, hd as u32);
        let reg_k = (0u32, valid as u32, 0, hd as u32); // rows.len = valid (< cap) — the slice
        let nodes = vec![
            PrimNode {
                op: PrimOp::Score {
                    num_q_heads: 1,
                    num_kv_heads: 1,
                    head_dim: hd as u32,
                    scale: 1.0,
                    regions: vec![reg_q, reg_k],
                    out_cols_start: 0,
                },
                ins: vec![0, 1],
                out: 3,
            },
            PrimNode {
                op: PrimOp::WeightedValue {
                    num_q_heads: 1,
                    num_kv_heads: 1,
                    head_dim: hd as u32,
                    v_regions: vec![reg_k],
                    out_cols_start: 0,
                },
                ins: vec![3, 2],
                out: 4,
            },
        ];
        let pir = PrimIR {
            tensors,
            num_sources: 3,
            nodes,
            result: 4,
        };
        let src: Vec<Vec<f32>> = (0..3)
            .map(|s| {
                (0..(pir.tensors[s].rows * pir.tensors[s].cols) as usize)
                    .map(|j| seed_src(s, j))
                    .collect()
            })
            .collect();
        let refs: Vec<&[f32]> = src.iter().map(|v| v.as_slice()).collect();
        let gp = pir.eval(&refs);
        let dev = lower_flat_to_device(&lower_sm_to_flat(&lower_prim_to_sm(&pir)));
        let gd = dev.eval(&refs);
        let close = |a: f32, b: f32| (a - b).abs() <= 1e-4 * a.abs().max(1.0);
        for j in 0..valid {
            assert!(
                close(gp[3][j], gd[3][j]),
                "scores[{j}] prim={} dev={} (region row-slice wrong)",
                gp[3][j],
                gd[3][j]
            );
        }
        for d in 0..hd {
            assert!(
                close(gp[4][d], gd[4][d]),
                "out[{d}] prim={} dev={} (region row-slice wrong)",
                gp[4][d],
                gd[4][d]
            );
        }
    }
}
