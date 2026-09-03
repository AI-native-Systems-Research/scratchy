// SPDX-License-Identifier: Apache-2.0
//! MacroIR — the FIRST distinct intermediate IR below `SubtileIR` on the SDSC lowering ladder.
//!
//! Its OWN type, OWN tensor indexing (plain `usize`), OWN `eval`. The pass `lower_subtile_to_macro`
//! (lands next) expands each SubtileIR composite into ≤3 macro-steps:
//!   `RmsNorm → {InvRms, ApplyNorm}`, `SiluMul → {Silu, Mul}`, `RopeRotate/Append → {RotateHalf,
//!   RopeBlend}`; `Matmul`/`Elementwise`/`SumReduce` map 1:1; `AttnDecode → Attn` (split into
//!   Score/Softmax/WeightedValue by the NEXT IR). Every MacroOp's `eval` MIRRORS the matching
//!   `eval_node` arm in `scratchy_subtile` so this distinct IR cannot drift from the golden.
//!
//! Sources `0..num_sources` are aligned (by index) with the originating SubtileIR's sources, so the
//! same per-index seed feeds both — letting [`verify_against_eval_dag`] compare the forward result
//! buffers directly (via `scratchy_subtile::subtile_ir::result_buffer`, the public anchor).

use scratchy_subtile::subtile_ir::{EwKind, RopeForm, SubOp, SubtileIR, eval_dag};

/// Logical row-major shape (MacroIR's own, independent of SubtileIR's types).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub rows: u32,
    pub cols: u32,
}

/// One MacroIR op (whole-tensor; region tiling is a later IR). `ins`/`out` index `MacroIR::tensors`.
#[derive(Clone, Debug)]
pub enum MacroOp {
    /// `out[m,n] = Σ_k a[m,k]·w[k,n]`, w row-major `[K,N]`. ins: [a, w].
    Matmul,
    /// `out = Σ ins` (split-K combine / residual). ins: ≥1, all == out shape.
    Sum,
    /// `out = a·b`. ins: [a, b].
    Mul,
    /// `out = a+b`. ins: [a, b].
    Add,
    /// `out = silu(a)`. ins: [a].
    Silu,
    /// `out = a·scale` (unary scalar multiply — granite's embedding/residual/logits multipliers). ins: [a].
    ScalarMul { scale: f32 },
    /// `out[m,1] = 1/sqrt(mean(x[m,:]²)+eps)`. ins: [x].
    InvRms { eps: f32 },
    /// `out[m,d] = x·inv_rms[row]·gamma[col]`. ins: [x, inv_rms(m,1), gamma(1,d)].
    ApplyNorm,
    /// NeoX rotate-half per head: `rot[..,d]=-x[..,d+half]; rot[..,d+half]=x[..,d]`. ins: [x].
    RotateHalf { head_dim: u32 },
    /// `out = x·cos + rot·sin` (cos/sin `[1,hd]` per head, full table). ins: [x, rot, cos, sin].
    RopeBlend { head_dim: u32 },
    /// Decode attention (whole; split next IR). ins: [Q, K0,V0, K1,V1, …].
    ///
    /// Attention is the ONE MacroOp whose SubtileIR inputs carry NON-whole REGIONS: the prefix K/V
    /// segments are sliced to `valid_len−1` rows (the live cache window), and head-blocking can slice
    /// columns. `eval_node` reads them via `gather` (region-aware); so must this. `in_regions[i] =
    /// (rows.start, rows.len, cols.start, cols.len)` of `ins[i]` into the WHOLE tensor (gather geometry),
    /// and `out_cols_start` = the output region's first column (→ `qh_start`, the global first q-head).
    /// Without these, a whole-tensor read attends to the wrong key set (extra/garbage cache rows) and
    /// maps q→kv heads wrong — the t613 sign-flip the bridge caught.
    Attn {
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        scale: f32,
        in_regions: Vec<(u32, u32, u32, u32)>,
        out_cols_start: u32,
    },
}

#[derive(Clone, Debug)]
pub struct MacroNode {
    pub op: MacroOp,
    pub ins: Vec<usize>,
    pub out: usize,
}

/// A distinct lowering IR: a whole-tensor op graph. `tensors[0..num_sources]` are leaf sources
/// (index-aligned with the originating SubtileIR's sources); later ids are op outputs (incl. the
/// macro-step intermediates). `result` indexes the forward logits tensor.
#[derive(Clone, Debug)]
pub struct MacroIR {
    pub tensors: Vec<Shape>,
    pub num_sources: u32,
    pub nodes: Vec<MacroNode>,
    pub result: usize,
}

impl MacroIR {
    /// Host interpreter. `sources[s]` = row-major buffer for source `s`. Each op mirrors `eval_node`.
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
                MacroOp::Matmul => {
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
                MacroOp::Sum => {
                    let mut o = vec![0f32; or * oc];
                    for &inp in &n.ins {
                        for (d, v) in o.iter_mut().zip(&bufs[inp]) {
                            *d += *v;
                        }
                    }
                    o
                }
                MacroOp::Mul => bufs[n.ins[0]]
                    .iter()
                    .zip(&bufs[n.ins[1]])
                    .map(|(&x, &y)| x * y)
                    .collect(),
                MacroOp::Add => bufs[n.ins[0]]
                    .iter()
                    .zip(&bufs[n.ins[1]])
                    .map(|(&x, &y)| x + y)
                    .collect(),
                MacroOp::Silu => bufs[n.ins[0]]
                    .iter()
                    .map(|&x| x / (1.0 + (-x).exp()))
                    .collect(),
                MacroOp::ScalarMul { scale } => {
                    bufs[n.ins[0]].iter().map(|&x| x * *scale).collect()
                }
                MacroOp::InvRms { eps } => {
                    let x = &bufs[n.ins[0]];
                    let xsh = self.tensors[n.ins[0]];
                    let (m, d) = (xsh.rows as usize, xsh.cols as usize);
                    let mut o = vec![0f32; m];
                    for i in 0..m {
                        let row = &x[i * d..(i + 1) * d];
                        let ss: f32 = row.iter().map(|&v| v * v).sum();
                        o[i] = 1.0 / (ss / d as f32 + *eps).sqrt();
                    }
                    o
                }
                MacroOp::ApplyNorm => {
                    let x = &bufs[n.ins[0]];
                    let inv = &bufs[n.ins[1]];
                    let gamma = &bufs[n.ins[2]];
                    let (m, d) = (or, oc);
                    let mut o = vec![0f32; m * d];
                    for i in 0..m {
                        for j in 0..d {
                            o[i * d + j] = x[i * d + j] * inv[i] * gamma[j];
                        }
                    }
                    o
                }
                MacroOp::RotateHalf { head_dim } => {
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
                MacroOp::RopeBlend { head_dim } => {
                    let x = &bufs[n.ins[0]];
                    let rot = &bufs[n.ins[1]];
                    let cos = &bufs[n.ins[2]];
                    let sin = &bufs[n.ins[3]];
                    let hd = *head_dim as usize;
                    let heads = oc / hd;
                    let mut o = vec![0f32; or * oc];
                    for r in 0..or {
                        for h in 0..heads {
                            let base = r * oc + h * hd;
                            for d in 0..hd {
                                o[base + d] = x[base + d] * cos[d] + rot[base + d] * sin[d];
                            }
                        }
                    }
                    o
                }
                MacroOp::Attn {
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    scale,
                    in_regions,
                    out_cols_start,
                } => {
                    // Mirrors eval_node's AttnDecode arm EXACTLY, including region-aware `gather`: each
                    // input is sliced from its WHOLE tensor by its region (rows + cols). This is what
                    // makes the prefix-K/V `valid_len−1` slice + head-block (qh_start/kvh_start) correct.
                    let hd = *head_dim as usize;
                    let gqa = (*num_q_heads / (*num_kv_heads).max(1)) as usize;
                    // gather(ins[i]) by its region from the whole tensor (stride = tensor's full cols).
                    let gather = |idx: usize| -> (Vec<f32>, usize, usize) {
                        let tid = n.ins[idx];
                        let full_cols = self.tensors[tid].cols as usize;
                        let (r0, nr, c0, nc) = in_regions[idx];
                        let (r0, nr, c0, nc) = (r0 as usize, nr as usize, c0 as usize, nc as usize);
                        let src = &bufs[tid];
                        let mut g = Vec::with_capacity(nr * nc);
                        for i in 0..nr {
                            let base = (r0 + i) * full_cols + c0;
                            g.extend_from_slice(&src[base..base + nc]);
                        }
                        (g, nr, nc)
                    };
                    let (q, mq, qc) = gather(0);
                    let qh_count = qc / hd; // q-heads in this block
                    let qh_start = *out_cols_start as usize / hd; // global first q-head
                    let kvh_start = (in_regions[1].2 as usize) / hd; // K[0] region cols.start / hd
                    let mut k_all: Vec<f32> = Vec::new();
                    let mut v_all: Vec<f32> = Vec::new();
                    let mut kv_count = 0usize;
                    let mut i = 1;
                    while i < n.ins.len() {
                        let (k, _kr, kc) = gather(i);
                        let (v, _vr, _vc) = gather(i + 1);
                        kv_count = kc / hd;
                        k_all.extend_from_slice(&k);
                        v_all.extend_from_slice(&v);
                        i += 2;
                    }
                    let seg_w = kv_count * hd;
                    let seq_len = if seg_w > 0 { k_all.len() / seg_w } else { 0 };
                    let mut o = vec![0f32; mq * qh_count * hd];
                    for qi in 0..mq {
                        for hl in 0..qh_count {
                            let global_kv = (qh_start + hl) / gqa.max(1);
                            let local_kv = global_kv - kvh_start;
                            let q_off = qi * qh_count * hd + hl * hd;
                            let causal_bound = seq_len + qi - mq;
                            let mut scores = vec![0f32; seq_len];
                            for (s, score) in scores.iter_mut().enumerate() {
                                if s > causal_bound {
                                    *score = f32::NEG_INFINITY;
                                    continue;
                                }
                                let k_off = s * seg_w + local_kv * hd;
                                let mut dot = 0f32;
                                for d in 0..hd {
                                    dot += q[q_off + d] * k_all[k_off + d];
                                }
                                *score = dot * scale;
                            }
                            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            let mut sum = 0f32;
                            for sc in scores.iter_mut() {
                                *sc = (*sc - mx).exp();
                                sum += *sc;
                            }
                            for sc in scores.iter_mut() {
                                *sc /= sum;
                            }
                            for d in 0..hd {
                                let mut val = 0f32;
                                for (s, &w) in scores.iter().enumerate() {
                                    val += w * v_all[s * seg_w + local_kv * hd + d];
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

/// THE PASS: `SubtileIR → MacroIR`. The first production lowering step. Each composite expands into
/// AT MOST 3 macro-steps (`RmsNorm/SiluMul/Rope → 2`; everything else maps 1:1; `AttnDecode → Attn`,
/// split further by the next IR). Whole-tensor (the SDSC `krg` is built with `knb=8192` ⇒ gemms whole,
/// no col-chunking), so MacroOps mirror SubtileIR nodes directly. Sources stay index-aligned (tensor
/// ids `0..N` carry over; macro-step intermediates get fresh ids `≥N`) so `verify_against_eval_dag`
/// can seed both identically. Pair every call with `verify_against_eval_dag` (the build-time guard).
pub fn lower_subtile_to_macro<F: RopeForm>(subtile: &SubtileIR<F>) -> MacroIR {
    let mut tensors: Vec<Shape> = subtile
        .tensors
        .iter()
        .map(|t| Shape {
            rows: t.rows,
            cols: t.cols,
        })
        .collect();
    let mut nodes: Vec<MacroNode> = Vec::new();
    for node in &subtile.nodes {
        let out = node.output.tensor.index();
        let ins: Vec<usize> = node.inputs.iter().map(|i| i.tensor.index()).collect();
        let osh = tensors[out];
        match &node.op {
            SubOp::MatmulTile { .. } => nodes.push(MacroNode {
                op: MacroOp::Matmul,
                ins,
                out,
            }),
            SubOp::SumReduce => nodes.push(MacroNode {
                op: MacroOp::Sum,
                ins,
                out,
            }),
            SubOp::Elementwise(EwKind::Add) => nodes.push(MacroNode {
                op: MacroOp::Add,
                ins,
                out,
            }),
            SubOp::Elementwise(EwKind::Mul) => nodes.push(MacroNode {
                op: MacroOp::Mul,
                ins,
                out,
            }),
            SubOp::Elementwise(EwKind::Silu) => nodes.push(MacroNode {
                op: MacroOp::Silu,
                ins,
                out,
            }),
            // ⛔ DELIBERATELY NOT A `MacroOp::Gelu`. Minting one cascades into
            // `PrimOp`, both evaluators and the Kani harnesses that give this
            // tower its value — i.e. it would add a proof obligation for an op
            // no proof here currently reasons about. This crate is a
            // VERIFICATION tower, not a build dependency of the model codegen
            // (the proven pieces are canonical in `scratchy-subtile`), so a
            // gelu arch lowering through the SuperDSC path does not pass here.
            // Left loud so that whoever DOES need it adds the proofs too.
            // A reshape is a re-laying COPY on a stick-laid-out device, not a
            // macro op over equal-shaped buffers. Modelling it here would mean
            // teaching this tower a layout it does not otherwise reason about.
            SubOp::Reshape => unimplemented!(
                "sdsc macro IR has no Reshape — it is a restickify (the device layout \
                 depends on the row count), which this tower does not model"
            ),
            SubOp::Elementwise(EwKind::Gelu) => unimplemented!(
                "sdsc macro IR has no Gelu — adding one needs MacroOp + PrimOp arms and \
                 the Kani harnesses to match; the SuperDSC path emits the DDL's OpFunc::Gelu"
            ),
            SubOp::ScalarMul { scale } => nodes.push(MacroNode {
                op: MacroOp::ScalarMul { scale: *scale },
                ins,
                out,
            }),
            SubOp::SiluMul => {
                // silu(gate)→tmp; tmp·up→out  (fan-out 2)
                let tmp = tensors.len();
                tensors.push(osh);
                nodes.push(MacroNode {
                    op: MacroOp::Silu,
                    ins: vec![ins[0]],
                    out: tmp,
                });
                nodes.push(MacroNode {
                    op: MacroOp::Mul,
                    ins: vec![tmp, ins[1]],
                    out,
                });
            }
            // ⛔ SCALE ONLY: `ApplyNorm` multiplies by the STORED gain, so the
            // gemma-class (1 + w) convention is a different macro op. Matching the
            // variant in the pattern sends the other one to the enumerated
            // not-bridged arm rather than through this decomposition.
            SubOp::RmsNorm {
                eps,
                gain: scratchy_subtile::subtile_ir::GainConvention::Scale,
            } => {
                // InvRms(x)→inv[m,1]; ApplyNorm(x, inv, gamma)→out  (fan-out 2)
                let inv = tensors.len();
                tensors.push(Shape {
                    rows: osh.rows,
                    cols: 1,
                });
                nodes.push(MacroNode {
                    op: MacroOp::InvRms { eps: *eps },
                    ins: vec![ins[0]],
                    out: inv,
                });
                nodes.push(MacroNode {
                    op: MacroOp::ApplyNorm,
                    ins: vec![ins[0], inv, ins[1]],
                    out,
                });
            }
            SubOp::RmsNormReduce { eps } => nodes.push(MacroNode {
                op: MacroOp::InvRms { eps: *eps },
                ins,
                out,
            }),
            SubOp::RmsNormApply { .. } => nodes.push(MacroNode {
                op: MacroOp::ApplyNorm,
                ins,
                out,
            }),
            SubOp::RopeRotate { head_dim, .. } | SubOp::RopeAppend { head_dim, .. } => {
                // RotateHalf(x)→rot; RopeBlend(x, rot, cos, sin)→out  (fan-out 2)
                let rot = tensors.len();
                tensors.push(osh);
                nodes.push(MacroNode {
                    op: MacroOp::RotateHalf {
                        head_dim: head_dim.get(),
                    },
                    ins: vec![ins[0]],
                    out: rot,
                });
                nodes.push(MacroNode {
                    op: MacroOp::RopeBlend {
                        head_dim: head_dim.get(),
                    },
                    ins: vec![ins[0], rot, ins[1], ins[2]],
                    out,
                });
            }
            // ⛔ THE GEOMETRY ARRIVES AS ONE VALUE NOW, NOT THREE `u32`s, and this arm is why that
            // matters: it destructured `num_q_heads, num_kv_heads, head_dim` and passed them straight
            // into a `MacroOp::Attn` literal with the same three field names — two adjacent head counts
            // of the same type, transposable at the literal with nothing to notice. `871e1520` replaced
            // them with `ModelAttnGeometry`, whose `mint` spends the GQA divisibility proof once, and
            // this crate has not compiled since (so every Kani proof in it was unrunnable for two days —
            // the failure mode `kani-proofs-died-silently-in-the-rebuild` describes).
            SubOp::AttnDecode { geom, scale, .. } => {
                // Carry each input's REGION (the prefix-K/V `valid_len−1` row slice + any head-block
                // col slice) so Attn's eval gathers exactly what `eval_node` does. out_cols_start →
                // qh_start (global first q-head). Parallel to `ins`.
                let in_regions: Vec<(u32, u32, u32, u32)> = node
                    .inputs
                    .iter()
                    .map(|tr| {
                        (
                            tr.region.rows.start,
                            tr.region.rows.len,
                            tr.region.cols.start,
                            tr.region.cols.len,
                        )
                    })
                    .collect();
                nodes.push(MacroNode {
                    op: MacroOp::Attn {
                        // Read OUT of the geometry, in one place, rather than carried as three
                        // independent numbers. `MacroOp::Attn`'s own fields are still `u32` — the
                        // downstream ladder has not been converted — so this is the boundary where the
                        // typed value is spent, and it is a single expression per field.
                        num_q_heads: geom.nqh().get(),
                        num_kv_heads: geom.nkvh().get(),
                        head_dim: geom.hd().get(),
                        scale: *scale,
                        in_regions,
                        out_cols_start: node.output.region.cols.start,
                    },
                    ins,
                    out,
                });
            }
            // The rest of the arch vocabulary. The SHARED front end expresses every op
            // now, so they reach this bridge; each consumer answers for itself.
            // Enumerated, never `_`, so a new SubOp is E0004 here.
            SubOp::RmsNorm { .. }
            | SubOp::Elementwise(EwKind::QuickGelu | EwKind::GeluErf | EwKind::Sub)
            | SubOp::TanhSoftCap
            | SubOp::RmsNormUnit { .. }
            | SubOp::ScalarWeightMul
            | SubOp::GateSplit { .. }
            | SubOp::GateApply
            | SubOp::GateScale
            | SubOp::LoadPixels { .. }
            | SubOp::LoadPosEmbeds { .. }
            | SubOp::EmbeddingGather { .. }
            | SubOp::VisionRope
            | SubOp::VarlenAttention { .. }
            | SubOp::EncoderAttn { .. }
            | SubOp::GatedDeltaNet
            | SubOp::GemmaMoe { .. }
            | SubOp::Moe { .. }
            | SubOp::Mean => {
                unimplemented!("macro_ir: no MacroOp sequence for {:?}", node.op)
            }
        }
    }
    MacroIR {
        tensors,
        num_sources: subtile.num_sources,
        nodes,
        result: subtile.result.index(),
    }
}

/// Deterministic per-source seed (~[-2,2]); reproducible (no RNG/time). Shared across ALL bridges so
/// every IR's `eval` is fed the byte-identical sources — the precondition for comparing their outputs.
pub(crate) fn seed_src(tid: usize, j: usize) -> f32 {
    let mut h = (tid as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    ((h % 4001) as f32 / 1000.0) - 2.0
}

/// VERIFY the `SubtileIR → MacroIR` bridge PER ORIGINAL TENSOR: for every tensor the two graphs
/// share (the SubtileIR tids `0..N`, which `lower_subtile_to_macro` preserves), `MacroIR.eval` must
/// match `eval_dag` within an `allclose` tolerance (`|Δ| ≤ ATOL + RTOL·|golden|`).
///
/// Why per-tensor + allclose, not exact-on-the-logits: the macro-step decomposition is mathematically
/// equivalent but NOT bit-identical (a fused `x·inv·γ` vs split steps reassociates), and that f32 noise
/// COMPOUNDS over 30 layers (~1e-4 by the logits) — exact/abs-tol on the final logits false-fails.
/// Per-tensor catches a REAL bug (a wrong op/wiring makes ITS output tid diverge by O(magnitude) — far
/// above the tolerance) at the FIRST diverging tid, while genuine f32 reassociation (small, growing
/// gradually across tids) passes. Panics (un-catchable `cargo build` failure) on a real divergence.
pub fn verify_against_eval_dag<F: RopeForm>(macro_ir: &MacroIR, subtile: &SubtileIR<F>) {
    // Per-tensor RELATIVE-L2 equivalence: `relL2(t) = ‖golden_t − macro_t‖₂ / max(‖golden_t‖₂, ε)`.
    //
    // Why relative-L2 and not max-per-element-relative: the macro-step decomposition is mathematically
    // equivalent but reassociates f32 (fused `x·inv·γ` → split steps; matmul/softmax over ~30 layers),
    // so individual elements carry ~1% noise — and at a residual-add CANCELLATION point (large `a ≈ −b`,
    // small `a+b`) that sub-1% input noise amplifies into a double-digit per-element relative error at a
    // SINGLE element. That is a numeric artifact, NOT a lowering bug. A REAL bug (wrong op/wiring/region/
    // layout) corrupts the WHOLE op output — every element, or a structured per-head/per-row subset.
    // Relative-L2 averages over the tensor, so it is immune to a lone cancellation point (t636's 1/576 →
    // ~0.6%) yet still trips hard on any real bug: a single wrong head (64/576 elements, O(1) each) →
    // ‖Δ‖/‖g‖ ≈ √(64)/√(576) ≈ 0.33 = 33% ≫ the 1% gate. So 1% cleanly separates "faithful (f32 noise)"
    // from "broken", with NO false-fail on cancellation. The per-element diverge COUNT is still reported
    // (transparency: a real bug shows many diverging elements; cancellation shows 1).
    const RELL2_TOL: f64 = 1e-2;
    const EPS: f64 = 1e-9;
    assert_eq!(
        macro_ir.num_sources, subtile.num_sources,
        "MacroIR bridge: source count {} != SubtileIR {} (sources must be index-aligned)",
        macro_ir.num_sources, subtile.num_sources
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
    let got = macro_ir.eval(&refs);

    // Compare every ORIGINAL tensor (0..N); macro-step intermediates (ids ≥ N) have no counterpart.
    // Compute the GLOBAL worst relative-L2 FIRST (no early panic) so the report shows the true worst
    // tensor: a real bug is ONE tensor near 1.0 with many diverging elements; noise is small everywhere.
    let n_orig = subtile.tensors.len();
    let mut worst_rell2 = 0f64;
    let mut worst_tid = 0usize;
    for tid in 0..n_orig {
        let g = &golden[tid];
        let o = &got[tid];
        assert_eq!(
            g.len(),
            o.len(),
            "MacroIR bridge: tensor t{tid} size {} != golden {} — the pass changed a tensor's shape",
            o.len(),
            g.len()
        );
        let mut num = 0f64; // ‖Δ‖²
        let mut den = 0f64; // ‖g‖²
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
    // Characterize the worst tensor: which MacroOp produces it, the worst single element, and how many
    // elements diverge >1% relative. Whole-tensor / per-head divergence ⇒ a wiring/region/op bug; 1
    // element ⇒ a cancellation edge. This makes the bridge a DEBUGGER that names the broken op.
    let producing_op = macro_ir
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
            "[sdsc-stage] MacroIR bridge: {n_orig} tensors, worst relative-L2 {worst_rell2:.2e} at \
             t{worst_tid} (op {producing_op}); worst element eval_dag={wg} MacroIR={wo} (relΔ \
             {worst_rel:.2e}); {n_div}/{} elements >1% [relL2 ≤ {RELL2_TOL} ⇒ faithful]",
            g.len()
        );
    }
    assert!(
        worst_rell2 <= RELL2_TOL,
        "MacroIR bridge diverges at t{worst_tid} (op {producing_op}): relative-L2 {worst_rell2:.2e} > \
         {RELL2_TOL}, {n_div}/{} elements >1% (worst element eval_dag={wg} MacroIR={wo}). The \
         SubtileIR→MacroIR pass changed the math at this op — THIS bridge is the bug. (f32 reassociation \
         is ≪1% relative-L2 even with cancellation; a real op/layout/wiring error corrupts the whole \
         tensor → relative-L2 O(0.1..1).)",
        g.len()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// eval() of a hand-built MacroIR (matmul → silu → mul → +residual, the MLP shape) matches an
    /// independent f64 recomputation — proving the MacroOp evals are self-consistent before the
    /// SubtileIR→MacroIR pass + eval_dag anchor land.
    #[test]
    fn macro_eval_mlp_self_consistent() {
        let (hidden, inter) = (4usize, 6usize);
        // sources: 0=norm[1,hidden], 1=Wg[hidden,inter], 2=Wu[hidden,inter], 3=Wd[inter,hidden], 4=hidden[1,hidden]
        let tensors = vec![
            Shape {
                rows: 1,
                cols: hidden as u32,
            }, // 0 norm
            Shape {
                rows: hidden as u32,
                cols: inter as u32,
            }, // 1 Wg
            Shape {
                rows: hidden as u32,
                cols: inter as u32,
            }, // 2 Wu
            Shape {
                rows: inter as u32,
                cols: hidden as u32,
            }, // 3 Wd
            Shape {
                rows: 1,
                cols: hidden as u32,
            }, // 4 hidden(residual)
            Shape {
                rows: 1,
                cols: inter as u32,
            }, // 5 gate
            Shape {
                rows: 1,
                cols: inter as u32,
            }, // 6 up
            Shape {
                rows: 1,
                cols: inter as u32,
            }, // 7 silu
            Shape {
                rows: 1,
                cols: inter as u32,
            }, // 8 silu*up
            Shape {
                rows: 1,
                cols: hidden as u32,
            }, // 9 down
            Shape {
                rows: 1,
                cols: hidden as u32,
            }, // 10 out = hidden + down
        ];
        let nodes = vec![
            MacroNode {
                op: MacroOp::Matmul,
                ins: vec![0, 1],
                out: 5,
            },
            MacroNode {
                op: MacroOp::Matmul,
                ins: vec![0, 2],
                out: 6,
            },
            MacroNode {
                op: MacroOp::Silu,
                ins: vec![5],
                out: 7,
            },
            MacroNode {
                op: MacroOp::Mul,
                ins: vec![7, 6],
                out: 8,
            },
            MacroNode {
                op: MacroOp::Matmul,
                ins: vec![8, 3],
                out: 9,
            },
            MacroNode {
                op: MacroOp::Add,
                ins: vec![4, 9],
                out: 10,
            },
        ];
        let ir = MacroIR {
            tensors,
            num_sources: 5,
            nodes,
            result: 10,
        };
        let src: Vec<Vec<f32>> = (0..5)
            .map(|s| {
                (0..(ir.tensors[s].rows * ir.tensors[s].cols) as usize)
                    .map(|j| seed_src(s, j))
                    .collect()
            })
            .collect();
        let refs: Vec<&[f32]> = src.iter().map(|v| v.as_slice()).collect();
        let got = ir.eval(&refs);

        // Independent recompute.
        let nrm = &src[0];
        let (wg, wu, wd, hid) = (&src[1], &src[2], &src[3], &src[4]);
        let mut want = vec![0f32; hidden];
        let mut act = vec![0f32; inter];
        for i in 0..inter {
            let (mut g, mut u) = (0f32, 0f32);
            for d in 0..hidden {
                g += nrm[d] * wg[d * inter + i];
                u += nrm[d] * wu[d * inter + i];
            }
            act[i] = (g / (1.0 + (-g).exp())) * u;
        }
        for c in 0..hidden {
            let mut acc = hid[c];
            for i in 0..inter {
                acc += act[i] * wd[i * hidden + c];
            }
            want[c] = acc;
        }
        for c in 0..hidden {
            assert!(
                (got[10][c] - want[c]).abs() < 1e-6,
                "out[{c}] {} != {}",
                got[10][c],
                want[c]
            );
        }
    }
}
