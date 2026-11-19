// SPDX-License-Identifier: Apache-2.0
//! THE CANONICAL TAPE DUMP: every CUDA tape this build compiled, as values, for proving a change
//! leaves what runs unchanged.
//!
//! For each registered model (the build's scope: run it once per FP8 model, with the features
//! the `cuda_tapes_*_fp8` suites take) it writes, under `CARGO_TARGET_TMPDIR/tape-dump/<stem>/`:
//! - `m<bucket>.jsonl`, one tape: its header line (bucket, arena, regions, input, result), then
//!   one line per step in launch order;
//! - `weights.jsonl`, the weight table, one line per index;
//! - `entry.json`, the registry entry.
//!
//! It prints a sha256 per file, one per model and one over every model, on `TAPE_DUMP` lines.
//!
//! A step's line is the kernel it launches and every argument as a value, under this file's own
//! keys:
//! - dimensions as integers, offsets as bytes, weights as indices;
//! - float fields decoded (`{"f32": value, "bits": bits}`);
//! - stored choices (a router's matmul, a gate/up kernel, a routing program, a rope pairing, a gain
//!   convention) as what they compute;
//! - the kernel choices written into each step at emission, as `player`: a GEMV's split, a norm's
//!   row-or-tiled kernel, the decode attention's split, a GDN scan (at 99c60d50 the player made
//!   them at run time, by `gemv_splits`, `norm_row_takes`, `attn_decode_split` and
//!   `gdn_scan_is_recurrent` of the step's constants, under the same keys, so equal dumps prove
//!   the stored choices equal those functions').
//!
//! So a type-only change (a newtype, a split payload, a renamed variant or field) leaves the step
//! lines equal once this file maps the new types to the same keys, and a change to what runs does
//! not. The header lines and the weight table are the tape types' own `Serialize` output, keys
//! sorted, so they follow those types' names.
//!
//! `#[ignore]`d: it is a tool, not a check. No toolchain and no GPU:
//!
//! ```text
//! cargo test -p scratchy-target-cuda-models \
//!     --features scratchy-models/<stem>,scratchy-quantizations/fp8-dynamic-per-channel \
//!     --test tape_dump -- --ignored --nocapture
//! ```

// The models register through `inventory`; nothing here names one, so link the crate.
extern crate scratchy_target_cuda_models as _;

use std::fmt::Write as _;
use std::path::PathBuf;

use scratchy_target_cuda::registry::{self, CudaModelEntry};
use scratchy_target_cuda::step::{
    AddRmsNormBf16, AttnPrefillBf16, CudaStep, ExpertGemmBf16, ExpertGemvBf16, ExpertProjection,
    GainConvention, GateUpActBf16, GateUpKernel, GatedDeltaNetBf16, GatedOp, GdnScan, GemvSplit,
    Matmul, MatmulGroup, MatmulKernel, MoeWeight, NormKernel, PairRows, RmsNormBf16, RopePairing,
    RouteProgram, SiluMulBf16, attn_decode_split_scratch_bytes, gdn_bundle, gdn_conv_dim,
    gdn_scratch,
};
use scratchy_target_cuda::tape::{ArenaOffset, CudaTape, RotaryTable, WeightIx};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

/// An arena offset, in bytes.
fn off(o: ArenaOffset) -> Value {
    json!(o.0)
}

fn opt_off(o: Option<ArenaOffset>) -> Value {
    o.map_or(Value::Null, off)
}

/// A weight, by its index.
fn wix(w: WeightIx) -> Value {
    json!(w.0)
}

/// An f32 stored as its bits: the value and the bits.
fn f32_bits(bits: u32) -> Value {
    json!({"f32": f64::from(f32::from_bits(bits)), "bits": bits})
}

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .fold(String::new(), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

/// `v` as JSON with every object's keys sorted, whatever map `serde_json` was built with.
fn canon(v: &Value, out: &mut String) {
    match v {
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&Value::String(k.clone()).to_string());
                out.push(':');
                canon(&m[k], out);
            }
            out.push('}');
        }
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                canon(x, out);
            }
            out.push(']');
        }
        other => out.push_str(&other.to_string()),
    }
}

fn line(v: &Value) -> String {
    let mut s = String::new();
    canon(v, &mut s);
    s.push('\n');
    s
}

// ── the stored choices, as what they compute ──

fn gain(c: GainConvention) -> Value {
    match c {
        GainConvention::Scale => json!({"gain": "x̂ · w", "gain_offset": 0.0}),
        GainConvention::OnePlusScale => json!({"gain": "x̂ · (1 + w)", "gain_offset": 1.0}),
    }
}

fn pairing(p: RopePairing) -> Value {
    match p {
        RopePairing::Standard => json!("lane i rotates with lane i + rot_dim/2"),
        RopePairing::Proportional => json!("lane i rotates with lane i + head_dim/2"),
    }
}

fn act(a: GatedOp) -> Value {
    match a {
        GatedOp::Silu => json!({"act": "silu(gate) · up", "code": 0}),
        GatedOp::Gelu => json!({"act": "gelu(gate) · up", "code": 1}),
    }
}

fn moe_weight(w: MoeWeight) -> Value {
    match w {
        MoeWeight::Bundle => json!({"dtype": "bf16", "layout": "one bundle per layer"}),
        MoeWeight::Own => json!({"dtype": "bf16", "layout": "the tensor's own buffer"}),
        MoeWeight::Fp8Bundle => {
            json!({"dtype": "e4m3 + f32 per-row scale", "layout": "one bundle per layer"})
        }
        MoeWeight::Fp8Own => {
            json!({"dtype": "e4m3 + f32 per-row scale", "layout": "the tensor's own buffer"})
        }
    }
}

fn projection(p: ExpertProjection) -> Value {
    match p {
        ExpertProjection::Gate => json!("gate"),
        ExpertProjection::Up => json!("up"),
        ExpertProjection::Down => json!("down"),
    }
}

fn pair_rows(r: PairRows) -> Value {
    match r {
        PairRows::Token => json!("A row per token, shared by its picks"),
        PairRows::Pair => json!("A row per (token, pick) pair"),
    }
}

fn route(p: RouteProgram) -> Value {
    match p {
        RouteProgram::SoftmaxTopK { renorm } => json!({
            "program": "softmax over every expert, then the top-k picks",
            "renormalise_over_picks": renorm,
        }),
        RouteProgram::TopKScaledSoftmax { scale, .. } => json!({
            "program": "top-k of the raw logits, each times scale, softmax over the picks, then \
                        times the per-expert scale",
            "renormalise_over_picks": false,
            "scale": f32_bits(scale.0),
        }),
    }
}

fn matmul(k: MatmulKernel) -> Value {
    match k {
        MatmulKernel::Tiled => json!("gemm_bf16_abt"),
        MatmulKernel::Gemv(_) => json!("gemv_split_bf16_abt"),
        MatmulKernel::Col64 => json!("gemm_col64_bf16_abt"),
    }
}

fn gate_up_kernel(k: GateUpKernel) -> Value {
    match k {
        GateUpKernel::Gemm => json!({"launcher": "gate_up_gemm_bf16", "tile_cols": 128}),
        GateUpKernel::GemmCol64 => json!({"launcher": "gate_up_gemm_bf16", "tile_cols": 64}),
        GateUpKernel::Gemv(_) => json!({"launcher": "gate_up_gemv_bf16", "tile_cols": Value::Null}),
        GateUpKernel::GemmFp8 => json!({"launcher": "gate_up_gemm_fp8", "tile_cols": 128}),
        GateUpKernel::GemmCol64Fp8 => json!({"launcher": "gate_up_gemm_fp8", "tile_cols": 64}),
        GateUpKernel::GemvFp8(_) => {
            json!({"launcher": "gate_up_gemv_fp8", "tile_cols": Value::Null})
        }
    }
}

// ── the payloads, as values ──

/// A matmul and the residual it adds (a one-row GEMV's; `null` on every other step).
fn gemm((g, residual): (Matmul, Option<ArenaOffset>)) -> Value {
    json!({"a": off(g.a), "b": wix(g.b), "c": off(g.c), "m": g.m, "n": g.n, "k": g.k,
           "residual": opt_off(residual)})
}

/// A one-weight GEMM step's [`gemm`].
fn gemm_step(s: &CudaStep) -> Value {
    gemm(s.gemm_view().expect("a one-weight GEMM step"))
}

/// A grouped step's [`group`].
fn group_step(s: &CudaStep) -> Value {
    group(&s.group().expect("a grouped GEMM step").1)
}

fn group(g: &MatmulGroup) -> Value {
    let members: Vec<Value> = (g.members().iter())
        .map(|m| {
            m.map_or(
                Value::Null,
                |m| json!({"b": wix(m.b), "c": off(m.c), "n": m.n}),
            )
        })
        .collect();
    json!({"a": off(g.a), "m": g.m, "k": g.k, "members": members})
}

fn norm(n: &RmsNormBf16) -> Value {
    json!({"x": off(n.x), "gain_w": wix(n.gain), "y": off(n.y), "rows": n.rows, "cols": n.cols,
           "eps": f32_bits(n.eps.0), "convention": gain(n.convention)})
}

fn add_norm(f: &AddRmsNormBf16) -> Value {
    json!({"a": off(f.a), "b": off(f.b), "sum": off(f.sum), "norm": norm(&f.norm)})
}

fn silu_mul(s: &SiluMulBf16) -> Value {
    json!({"gate": off(s.gate), "up": off(s.up), "out": off(s.out), "rows": s.rows,
           "cols": s.cols})
}

fn gate_up(g: &GateUpActBf16) -> Value {
    json!({"gate": gemm((g.gate.matmul(), None)), "up": gemm((g.up.matmul(), None)), "out": off(g.out),
           "kernel": gate_up_kernel(g.kernel), "act": act(g.act)})
}

fn expert_gemv(g: &ExpertGemvBf16) -> Value {
    json!({"a": off(g.a), "rows": pair_rows(g.rows), "bundle": wix(g.bundle),
           "weight": moe_weight(g.weight), "proj": projection(g.proj), "topk_idx": off(g.topk_idx),
           "c": off(g.c), "m": g.m, "top_k": g.top_k, "experts": g.experts, "hidden": g.hidden,
           "inter": g.inter})
}

fn expert_gemm(g: &ExpertGemmBf16) -> Value {
    json!({"a": off(g.a), "bundle": wix(g.bundle), "weight": moe_weight(g.weight),
           "proj": projection(g.proj), "offsets": off(g.offsets),
           "tile_expert": off(g.tile_expert), "c": off(g.c), "max_tiles": g.max_tiles,
           "experts": g.experts, "hidden": g.hidden, "inter": g.inter, "n_tile": g.n_tile})
}

fn attn(a: &AttnPrefillBf16) -> Value {
    json!({"q": off(a.q), "block_table": off(a.block_table), "sequence": off(a.sequence),
           "out": off(a.out), "layer": a.layer, "rows": a.rows, "q_heads": a.q_heads,
           "kv_heads": a.kv_heads, "head_dim": a.head_dim, "scale": f32_bits(a.scale.0),
           "window": a.window})
}

fn gdn(g: &GatedDeltaNetBf16) -> Value {
    // The layouts the step carries are what the player derived from these keys at 99c60d50; the
    // dump checks them against that derivation instead of printing them.
    let (nk, hk, nv, hv) = (
        g.num_k_heads.get(),
        g.head_k.get(),
        g.num_v_heads.get(),
        g.head_v.get(),
    );
    let conv_dim = gdn_conv_dim(nk, hk, nv, hv).expect("a lowered GDN step's conv fits");
    assert_eq!(
        g.layout.conv_dim.get(),
        conv_dim,
        "the stored conv channels"
    );
    assert_eq!(
        Some(g.layout.scratch),
        gdn_scratch(g.rows.get(), conv_dim, nv, nv * hv),
        "the stored GDN scratch"
    );
    assert_eq!(
        Some(g.layout.bundle),
        gdn_bundle(conv_dim, g.conv_kernel.get(), nv, hv),
        "the stored GDN bundle"
    );
    json!({"qkv": off(g.qkv), "z": off(g.z), "a": off(g.a), "b": off(g.b),
           "bundle": wix(g.bundle), "out": off(g.out), "scratch": off(g.scratch),
           "sequence": off(g.sequence), "layer": g.layer, "rows": g.rows,
           "num_k_heads": g.num_k_heads, "num_v_heads": g.num_v_heads, "head_k": g.head_k,
           "head_v": g.head_v, "conv_kernel": g.conv_kernel, "eps": f32_bits(g.eps.0)})
}

/// A GEMV's stored K split, the player's choice.
fn splits(s: GemvSplit) -> Value {
    json!({"gemv_splits": s.splits})
}

/// A GEMV group's stored K split per member, the player's choice.
fn group_splits(splits: [Option<GemvSplit>; 3]) -> Value {
    let member_splits: Vec<Value> = (splits.iter())
        .map(|m| m.map_or(Value::Null, |m| json!(m.splits)))
        .collect();
    json!({"gemv_splits": member_splits})
}

/// The norm's stored row-or-tiled kernel, the player's choice.
fn norm_pick(n: &RmsNormBf16) -> Value {
    json!({"norm_row_takes": n.kernel == NormKernel::Row})
}

/// One step: the kernel(s) it launches, its arguments, and the player's run-time choices.
fn step_record(s: &CudaStep) -> Value {
    let (kernel, args, player): (&str, Value, Value) = match s {
        CudaStep::Embed(e) => (
            "embed_bf16",
            json!({"tokens": off(e.tokens), "table": wix(e.table), "out": off(e.out),
                   "rows": e.rows, "hidden": e.hidden, "vocab": e.vocab}),
            Value::Null,
        ),
        CudaStep::Gemm(_) => ("gemm_bf16_abt", gemm_step(s), Value::Null),
        CudaStep::GemmNarrow(_) => ("gemm_narrow_bf16_abt", gemm_step(s), Value::Null),
        CudaStep::Gemv(g) => (
            "gemv_bf16 (unsplit, split or residual by its arguments)",
            gemm_step(s),
            splits(g.split),
        ),
        CudaStep::GemmCol64(_) => ("gemm_col64_bf16_abt", gemm_step(s), Value::Null),
        CudaStep::GemmFp8(_) => ("gemm_fp8_abt (128-col tiles)", gemm_step(s), Value::Null),
        CudaStep::GemmCol64Fp8(_) => ("gemm_fp8_abt (64-col tiles)", gemm_step(s), Value::Null),
        CudaStep::GemmNarrowFp8(_) => ("gemm_narrow_fp8_abt", gemm_step(s), Value::Null),
        CudaStep::GemvFp8(g) => ("gemv_fp8_residual", gemm_step(s), splits(g.split)),
        CudaStep::GemvTcFp8(_) => ("gemv_fp8_b8", gemm_step(s), Value::Null),
        CudaStep::GemmGroup(_) => ("gemm_group_bf16_abt", group_step(s), Value::Null),
        CudaStep::GemmNarrowGroup(_) => ("gemm_narrow_group_bf16_abt", group_step(s), Value::Null),
        CudaStep::GemmGroupFp8(_) => ("gemm_group_fp8_abt", group_step(s), Value::Null),
        CudaStep::GemmNarrowGroupFp8(_) => {
            ("gemm_narrow_group_fp8_abt", group_step(s), Value::Null)
        }
        CudaStep::GemvGroup(g) => (
            "gemv_group_bf16_abt",
            group_step(s),
            group_splits(g.splits()),
        ),
        CudaStep::GemvGroupFp8(g) => (
            "gemv_group_fp8_abt",
            group_step(s),
            group_splits(g.splits()),
        ),
        CudaStep::GemvGroupTcFp8(_) => ("gemv_fp8_b8 (per member)", group_step(s), Value::Null),
        CudaStep::RmsNorm(n) => (
            "rms_norm_bf16, or rms_norm_quant_row's row kernel when norm_row_takes",
            norm(n),
            norm_pick(n),
        ),
        CudaStep::RouterNorm(n) => (
            "rms_norm_bf16, or rms_norm_quant_row's row kernel when norm_row_takes (a router's pre-norm)",
            norm(n),
            norm_pick(n),
        ),
        CudaStep::AddRmsNorm(f) => (
            "add_rms_norm_bf16, or rms_norm_quant_row's row kernel when norm_row_takes",
            add_norm(f),
            norm_pick(&f.norm),
        ),
        CudaStep::NormAdd(f) => (
            "norm_add_bf16, or rms_norm_quant_row's row kernel when norm_row_takes",
            json!({"norm": norm(&f.norm), "residual": off(f.residual)}),
            norm_pick(&f.norm),
        ),
        CudaStep::RmsNormQuantFp8(t) => (
            "rms_norm_quant_row",
            json!({"norm": norm(&t.norm), "q": off(t.q)}),
            Value::Null,
        ),
        CudaStep::AddRmsNormQuantFp8(t) => (
            "rms_norm_quant_row (with the add)",
            json!({"fold": add_norm(&t.fold), "q": off(t.q)}),
            Value::Null,
        ),
        CudaStep::RmsNormUnit(n) => (
            "rms_norm_unit_bf16",
            json!({"x": off(n.x), "y": off(n.y), "rows": n.rows, "cols": n.cols,
                   "eps": f32_bits(n.eps.0)}),
            Value::Null,
        ),
        CudaStep::Add(a) => (
            "add_bf16",
            json!({"a": off(a.a), "b": off(a.b), "out": off(a.out), "rows": a.rows,
                   "cols": a.cols}),
            Value::Null,
        ),
        CudaStep::SiluMul(m) => ("silu_mul_bf16", silu_mul(m), Value::Null),
        CudaStep::GeluMul(m) => ("gelu_mul_bf16", silu_mul(m), Value::Null),
        CudaStep::Rope(r) => (
            "rope_paired_bf16",
            json!({"x": off(r.x), "table": off(r.table), "out": off(r.out), "rows": r.rows,
                   "cols": r.cols, "head_dim": r.head_dim, "rot_dim": r.rot_dim,
                   "pairing": pairing(r.pairing)}),
            Value::Null,
        ),
        CudaStep::RopeAppend(r) => (
            "rope_append_fp8 (or rope_paired_bf16 + kv_append_fp8, by its arguments)",
            json!({"k": off(r.k), "v": off(r.v), "table": off(r.table),
                   "slot_mapping": off(r.slot_mapping), "k_out": off(r.k_out), "layer": r.layer,
                   "rows": r.rows, "cols": r.cols, "head_dim": r.head_dim, "rot_dim": r.rot_dim,
                   "pairing": pairing(r.pairing),
                   "q": r.q.map_or(Value::Null, |q| json!({"x": off(q.x), "out": off(q.out),
                                                           "cols": q.cols})),
                   "norms": r.norms.map_or(Value::Null, |n| json!({"q_gain": wix(n.q_gain),
                                                                   "k_gain": wix(n.k_gain),
                                                                   "eps": f32_bits(n.eps.0)}))}),
            Value::Null,
        ),
        CudaStep::GateSplit(g) => (
            "gate_split_bf16",
            json!({"qg": off(g.qg), "q": off(g.q), "gate": off(g.gate), "rows": g.rows,
                   "half_cols": g.half_cols, "head_dim": g.head_dim}),
            Value::Null,
        ),
        CudaStep::GateApply(g) => (
            "gate_apply_bf16",
            json!({"attn": off(g.attn), "gate": off(g.gate), "out": off(g.out), "rows": g.rows,
                   "cols": g.cols}),
            Value::Null,
        ),
        CudaStep::AttnPrefill(a) => (
            "attn_prefill_fp8kv (windowed, or d512, by its head_dim)",
            attn(a),
            Value::Null,
        ),
        CudaStep::AttnDecode(a) => (
            "attn_decode_fp8kv_split + attn_decode_merge",
            // The stored split scratch is what the player derived from the split at 99c60d50.
            {
                let at = attn_decode_split_scratch_bytes(
                    a.split.splits.get(),
                    a.attn.kv_heads.get(),
                    a.attn.head_dim.get(),
                );
                assert_eq!(
                    Some((a.scratch.ml_offset, a.scratch.bytes)),
                    at,
                    "the split scratch"
                );
                attn(&a.attn)
            },
            json!({"attn_decode_split": [a.split.splits, a.split.min_tiles]}),
        ),
        CudaStep::GatedDeltaNet(g) => (
            "gdn_conv1d + gdn_gating + (gdn_scan or gdn_chunk_scan) + gdn_rms_norm_gated",
            gdn(&g.gdn),
            json!({"gdn_scan_is_recurrent": g.scan == GdnScan::Recurrent}),
        ),
        CudaStep::GdnDecode(g) => ("gdn_decode", gdn(g), Value::Null),
        CudaStep::RouterLogits(r) => (
            "router logits (the stored matmul)",
            json!({"x": off(r.x), "bundle": wix(r.bundle), "weight": moe_weight(r.weight),
                   "logits": off(r.logits), "m": r.m, "experts": r.experts, "hidden": r.hidden,
                   "inter": r.inter, "matmul": matmul(r.kernel)}),
            match r.kernel {
                MatmulKernel::Gemv(split) => splits(split),
                MatmulKernel::Tiled | MatmulKernel::Col64 => Value::Null,
            },
        ),
        CudaStep::MoeRoute(r) => (
            "moe_route_bf16 (the stored program)",
            json!({"logits": off(r.logits), "topk_idx": off(r.topk_idx),
            "scores": off(r.scores), "m": r.m, "experts": r.experts, "top_k": r.top_k,
            "route": route(r.program),
            "expert_scale": match r.program {
                RouteProgram::TopKScaledSoftmax { expert_scale, .. } => wix(expert_scale),
                RouteProgram::SoftmaxTopK { .. } => Value::Null,
            }}),
            Value::Null,
        ),
        CudaStep::ExpertSort(e) | CudaStep::ExpertSortFp8(e) => {
            let kernel = match s {
                CudaStep::ExpertSort(_) => "moe_sort",
                _ => "moe_sort_fp8_m64",
            };
            (
                kernel,
                json!({"x": off(e.x), "topk_idx": off(e.topk_idx),
                       "chunk_hist": off(e.chunk_hist), "counts": off(e.counts),
                       "offsets": off(e.offsets), "tile_expert": off(e.tile_expert),
                       "pos": off(e.pos), "rows": off(e.rows), "m": e.m, "experts": e.experts,
                       "top_k": e.top_k, "hidden": e.hidden, "max_tiles": e.max_tiles}),
                Value::Null,
            )
        }
        CudaStep::ExpertActQuant(t) => (
            "expert_act_quant_fp8",
            json!({"act": silu_mul(&t.act), "op": act(t.op), "offsets": off(t.offsets),
                   "experts": t.experts, "q_act": off(t.q_act)}),
            Value::Null,
        ),
        CudaStep::ExpertGateUpGemmFp8(t) => (
            "expert_gate_up_gemm_fp8 + expert_quant_fp8",
            json!({"gate": expert_gemm(&t.gate), "up": expert_gemm(&t.up),
                   "act": json!({"act": silu_mul(&t.act.act), "op": act(t.act.op),
                                 "offsets": off(t.act.offsets), "experts": t.act.experts,
                                 "q_act": off(t.act.q_act)})}),
            Value::Null,
        ),
        CudaStep::ExpertGemm(g) => (
            "expert_gemm (tiled bf16 or fp8 m64, by its weight)",
            expert_gemm(g),
            Value::Null,
        ),
        CudaStep::ExpertGemv(g) => (
            "expert_gemv (split bf16 or fp8, by its weight)",
            expert_gemv(&g.gemv),
            splits(g.split),
        ),
        CudaStep::ExpertGemvTcFp8(g) => ("gemv_fp8_b8 (per pair)", expert_gemv(g), Value::Null),
        CudaStep::ExpertGateUpAct(g) => (
            "expert_gate_up_gemv (bf16 or fp8, by its weight)",
            json!({"gate": expert_gemv(&g.fold.gate), "up": expert_gemv(&g.fold.up),
                   "out": off(g.fold.out), "act": act(g.fold.act)}),
            splits(g.split),
        ),
        CudaStep::ExpertGateUpTcFp8(t) => (
            "act_quant_fp8 + gemv_fp8_b8 (per pair)",
            json!({"fold": json!({"gate": expert_gemv(&t.fold.gate),
                                  "up": expert_gemv(&t.fold.up), "out": off(t.fold.out),
                                  "act": act(t.fold.act)}),
                   "q_act": off(t.q_act)}),
            Value::Null,
        ),
        CudaStep::ExpertCombine(c) => (
            "moe_combine_bf16",
            json!({"down": off(c.down), "pos": opt_off(c.pos), "scores": off(c.scores),
                   "out": off(c.out), "m": c.m, "top_k": c.top_k, "hidden": c.hidden,
                   "down_rows": c.down_rows}),
            Value::Null,
        ),
        CudaStep::GateScale(g) => (
            "gate_scale_bf16",
            json!({"routed": off(g.routed), "shared": off(g.shared), "g": off(g.g),
                   "out": off(g.out), "rows": g.rows, "cols": g.cols}),
            Value::Null,
        ),
        CudaStep::Scale(x) => (
            "scale_bf16",
            json!({"x": off(x.x), "out": off(x.out), "rows": x.rows, "cols": x.cols,
                   "scale": f32_bits(x.scale.0)}),
            Value::Null,
        ),
        CudaStep::LayerScale(x) => (
            "layer_scale_bf16",
            json!({"x": off(x.x), "scale": wix(x.scale), "out": off(x.out), "rows": x.rows,
                   "cols": x.cols}),
            Value::Null,
        ),
        CudaStep::SoftCap(x) => (
            "soft_cap_bf16",
            json!({"x": off(x.x), "out": off(x.out), "rows": x.rows, "cols": x.cols,
                   "cap": f32_bits(x.cap.0)}),
            Value::Null,
        ),
        CudaStep::GatherLastRow(g) => (
            "gather_last_row_bf16",
            json!({"x": off(g.x), "sequence": off(g.sequence), "rows": g.rows, "cols": g.cols}),
            Value::Null,
        ),
        CudaStep::GateUpAct(g) => (
            "gate_up (the stored kernel)",
            gate_up(g),
            g.kernel.gemv_split().map_or(Value::Null, splits),
        ),
        CudaStep::GateUpTcFp8(t) => (
            "act_quant_fp8 + gemv_fp8_b8 (gate and up)",
            json!({"fold": gate_up(&t.fold), "q_act": off(t.q_act)}),
            Value::Null,
        ),
        CudaStep::QuantRowsFp8(q) => (
            "quant_rows_fp8 (or quant_row_fp8_one at one row)",
            json!({"x": off(q.x), "q": off(q.q), "rows": q.rows, "cols": q.cols}),
            Value::Null,
        ),
    };
    json!({"kernel": kernel, "args": args, "player": player})
}

/// One tape's header: everything but its steps, from its types' own `Serialize`.
fn tape_header(stem: &str, t: &CudaTape) -> Value {
    let mut v = serde_json::to_value(t).expect("the tape serializes");
    let m: &mut Map<String, Value> = v.as_object_mut().expect("a tape is an object");
    m.remove("steps");
    // The GDN scratch's stored regions are what `validate` derived from the header's own geometry
    // at 99c60d50: checked against that derivation, not printed.
    if let Some(g) = t.gdn {
        let (nk, hk, nv, hv) = (
            g.num_k_heads.get(),
            g.head_k.get(),
            g.num_v_heads.get(),
            g.head_v.get(),
        );
        let conv_dim = gdn_conv_dim(nk, hk, nv, hv).expect("the tape's GDN conv fits");
        let want = gdn_scratch(t.bucket_m.get(), conv_dim, nv, nv * hv);
        assert_eq!(Some(g.scratch_layout), want, "the tape's GDN scratch");
        (m.get_mut("gdn").and_then(Value::as_object_mut))
            .expect("a GDN tape is an object")
            .remove("scratch_layout");
    }
    // The rope tables by hand, the theta decoded: what the forward fills them with.
    let rotary = |r: Option<RotaryTable>| {
        r.map_or(Value::Null, |r| {
            json!({"span": {"offset": r.span.offset.0, "bytes": r.span.bytes},
                   "rot_dim": r.rot_dim, "freq_dim": r.freq_dim,
                   "theta": {"f64": r.theta.get(), "bits": r.theta.0}})
        })
    };
    m.insert("rotary".into(), rotary(t.rotary));
    m.insert("rotary_local".into(), rotary(t.rotary_local));
    m.insert("stem".into(), json!(stem));
    m.insert("steps_count".into(), json!(t.steps.len()));
    v
}

/// Every file of one model: (name, bytes).
fn model_files(e: &CudaModelEntry) -> Vec<(String, String)> {
    let mut files = Vec::new();
    for t in e.tapes {
        let mut text = line(&tape_header(e.stem, t));
        for (i, s) in t.steps.iter().enumerate() {
            let mut rec = step_record(s);
            rec.as_object_mut()
                .expect("a step record is an object")
                .insert("step".into(), json!(i));
            text.push_str(&line(&rec));
        }
        files.push((format!("m{}.jsonl", t.bucket_m), text));
    }
    let weights: String = (e.weights.iter().enumerate())
        .map(|(ix, w)| {
            line(&json!({"ix": ix, "weight": serde_json::to_value(w).expect("a weight entry")}))
        })
        .collect();
    files.push(("weights.jsonl".into(), weights));
    let buckets: Vec<u32> = e.tapes.iter().map(|t| t.bucket_m.get()).collect();
    files.push((
        "entry.json".into(),
        line(&json!({"stem": e.stem, "buckets": buckets, "weights": e.weights.len()})),
    ));
    files
}

#[test]
#[ignore = "a tool: the canonical dump of every CUDA tape this build compiled"]
fn dump_every_tape() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tape-dump");
    let mut entries: Vec<&CudaModelEntry> = registry::entries().collect();
    entries.sort_by_key(|e| e.stem);
    assert!(!entries.is_empty(), "no CUDA model in this build's scope");
    let mut all = String::new();
    for e in entries {
        let dir = root.join(e.stem);
        std::fs::create_dir_all(&dir).expect("the dump directory");
        let mut model = String::new();
        for (name, text) in model_files(e) {
            std::fs::write(dir.join(&name), &text).expect("write a dump file");
            let h = sha(text.as_bytes());
            println!("TAPE_DUMP {}/{name} sha256={h}", e.stem);
            let _ = writeln!(model, "{name} {h}");
        }
        let h = sha(model.as_bytes());
        println!("TAPE_DUMP_MODEL {} sha256={h} ({})", e.stem, dir.display());
        let _ = writeln!(all, "{} {h}", e.stem);
    }
    println!("TAPE_DUMP_ALL sha256={}", sha(all.as_bytes()));
}
