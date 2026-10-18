// SPDX-License-Identifier: Apache-2.0

//! THE PROMPT MoE KERNEL, through the ladder: `kernels/moe_prefill.py` (router + dense route +
//! every expert swept) from Python to the emulator, against an f64 model of the card's W8A8
//! numerics — and against the DECODE kernel's model on the same tokens.
//!
//! ```text
//!   triton_frontend      .py -> TTIR       (include_str!'d from the spyre target)
//!   triton_ktir          TTIR -> KTIR      (every slab a DIRECT load: no indirect tile)
//!   to_ktir_emit         KTIR -> IRFunction
//!   ktir_emulator        the executed output
//! ```
//!
//! Two claims, two tests:
//! 1. the executed kernel is the card's W8A8 prefill MoE (op by op, in expert-id order), with a
//!    control that sees which experts each token read;
//! 2. a token's prefill output equals its DECODE output (`moe.py`'s per-slot model) to f16
//!    accumulation order — the two kernels share every per-element grain, so the only
//!    difference is the association of the K-term f16 sum.

use std::collections::HashMap;

use ktir_core::arena::Arena;
use ktir_core::ir::IRFunction;
use ktir_core::opkind::OpKind;
use triton_frontend::codegen::{self, ArgSpec, KernelSpec};
use triton_frontend::opt;
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;

const SRC: &str = include_str!("../../../targets/spyre/kernels/moe_prefill.py");

/// The golden geometry (`triton-frontend`'s `common::moe_prefill`): 16 prompt rows.
const T: usize = 16;
const H: usize = 128;
const I: usize = 64;
const E: usize = 64;
const K: usize = 8;

const PTRS: &[(&str, &str)] = &[
    ("desc_xr", "*fp16"),
    ("desc_rg", "*fp16"),
    ("desc_rw", "*fp16"),
    ("desc_pes", "*fp16"),
    ("desc_xe", "*fp16"),
    ("desc_wg", "*fp8e4nv"),
    ("desc_sg", "*fp16"),
    ("desc_wu", "*fp8e4nv"),
    ("desc_su", "*fp16"),
    ("desc_wd", "*fp8e4nv"),
    ("desc_sd", "*fp16"),
    ("desc_o", "*fp16"),
    ("desc_dense", "*fp16"),
];

fn spec() -> KernelSpec {
    let mut signature = HashMap::new();
    for (k, v) in PTRS {
        signature.insert(k.to_string(), ArgSpec::parse(v).expect("pointer spec"));
    }
    let h = H as f64;
    let ces: [(&str, Val); 10] = [
        ("T", Val::Int(T as i128)),
        ("H", Val::Int(H as i128)),
        ("I", Val::Int(I as i128)),
        ("E", Val::Int(E as i128)),
        ("K", Val::Int(K as i128)),
        ("EPS", Val::Float(1e-6)),
        ("INV_H", Val::Float(1.0 / h)),
        ("TEMP", Val::Float(h.powf(-0.5))),
        ("INV_QMAX", Val::Float(1.0 / 448.0)),
        ("SCALE_FLOOR", Val::Float(1.0 / 448.0 / 448.0)),
    ];
    let mut constexprs = HashMap::new();
    for (k, v) in ces {
        signature.insert(
            k.to_string(),
            ArgSpec::parse("constexpr").expect("constexpr"),
        );
        constexprs.insert(k.to_string(), v);
    }
    KernelSpec {
        kernel: "moe_prefill_fwd".to_string(),
        signature,
        constexprs,
        file: "moe_prefill.py".to_string(),
    }
}

fn lowered() -> IRFunction<'static> {
    let mut tt = codegen::compile(SRC, &spec(), Target::spyre())
        .unwrap_or_else(|e| panic!("codegen refused the kernel: {e}"));
    opt::make_ttir(&mut tt).unwrap_or_else(|e| panic!("make_ttir refused the kernel: {e}"));
    let mut m = triton_ktir::from_ttir::convert(&tt)
        .unwrap_or_else(|e| panic!("from_ttir refused the kernel: {e}"));
    triton_ktir::make_ktir(&mut m, &[1])
        .unwrap_or_else(|e| panic!("make_ktir refused the kernel: {e}"));
    triton_ktir::passes::to_ktir::run(&mut m, &[1])
        .unwrap_or_else(|e| panic!("to_ktir refused the kernel: {e}"));
    triton_ktir::passes::to_ktir_emit::lower(&m, Arena::global())
        .unwrap_or_else(|e| panic!("to_ktir_emit refused the kernel: {e}"))
}

// ── the reference: `moe.rs`'s model, at T rows ─────────────────────────────────────────────

fn f16(v: f64) -> f64 {
    let b = ktir_core::codec::encode(&[v as f32], ktir_core::dtypes::DType::F16);
    ktir_core::codec::decode(&b, 1, ktir_core::dtypes::DType::F16)[0] as f64
}

fn e4m3(v: f64) -> f64 {
    ktir_core::codec::e4m3_to_f32(ktir_core::codec::f32_to_e4m3(v as f32)) as f64
}

fn fp8_scale(amax: f64) -> f64 {
    f16(f16(amax * f16(1.0 / 448.0))
        .max(f16(1.0 / 448.0 / 448.0))
        .min(448.0))
}

fn w8a8(codes: &[f64], s: f64, w: &[f64], w_scale: &[f64], n: usize) -> Vec<f64> {
    let k = codes.len();
    (0..n)
        .map(|j| {
            let dot: f64 = (0..k).map(|c| codes[c] * w[j * k + c]).sum();
            f16(f16(f16(dot) * s) * w_scale[j])
        })
        .collect()
}

fn gelu(g: f64) -> f64 {
    let g32 = g as f32;
    // The kernel's `1.5957691216057308` (= 2·√(2/π)), at the f32 the island computes in.
    let v = 1.595_769_2_f32 * (g32 + 0.044_715_f32 * g32 * g32 * g32);
    let e = f16((-v).exp() as f64);
    f16(g / f16(1.0 + e))
}

fn codes(row: &[f64]) -> (f64, Vec<f64>) {
    let s = fp8_scale(f16(row.iter().fold(0.0f64, |a, v| a.max(v.abs()))));
    let inv = f16(1.0 / s);
    let q = row
        .iter()
        .map(|v| e4m3(f16(v * inv).clamp(-448.0, 448.0)))
        .collect();
    (s, q)
}

struct Inputs {
    xr: Vec<f64>,
    rg: Vec<f64>,
    rw: Vec<f64>,
    pes: Vec<f64>,
    xe: Vec<f64>,
    wg: Vec<f64>,
    sg: Vec<f64>,
    wu: Vec<f64>,
    su: Vec<f64>,
    wd: Vec<f64>,
    sd: Vec<f64>,
}

/// `moe.rs`'s well-conditioned draw, at T rows: each token's router input is distinct, so the
/// T tokens route to DIFFERENT expert sets and the dense sweep sees many experts.
fn inputs() -> Inputs {
    struct Draw(u64);
    impl Draw {
        fn u(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 33) as f64) / ((1u64 << 31) as f64)
        }
        fn sym(&mut self, scale: f64) -> f64 {
            (2.0 * self.u() - 1.0) * scale
        }
    }
    let mut d = Draw(0x5eed_c0de);
    let xr: Vec<f64> = (0..T * H).map(|_| f16(d.sym(1.0))).collect();
    let rg: Vec<f64> = (0..H).map(|_| f16(1.0 + d.sym(0.25))).collect();
    let dir: Vec<f64> = (0..H).map(|_| d.sym(1.0)).collect();
    let rw: Vec<f64> = (0..E * H)
        .map(|i| {
            let (e, h) = (i / H, i % H);
            f16(0.02 * (e as f64 - E as f64 / 2.0) * dir[h] + d.sym(0.05))
        })
        .collect();
    let pes: Vec<f64> = (0..E).map(|_| f16(0.5 + d.u())).collect();
    let xe: Vec<f64> = (0..T * H).map(|_| f16(d.sym(1.0))).collect();
    let wg: Vec<f64> = (0..E * I * H).map(|_| e4m3(d.sym(2.0))).collect();
    let wu: Vec<f64> = (0..E * I * H).map(|_| e4m3(d.sym(2.0))).collect();
    let wd: Vec<f64> = (0..E * H * I).map(|_| e4m3(d.sym(2.0))).collect();
    let sg: Vec<f64> = (0..E * I).map(|_| f16(0.75 + 0.5 * d.u())).collect();
    let su: Vec<f64> = (0..E * I).map(|_| f16(0.75 + 0.5 * d.u())).collect();
    let sd: Vec<f64> = (0..E * H).map(|_| f16(0.75 + 0.5 * d.u())).collect();
    Inputs {
        xr,
        rg,
        rw,
        pes,
        xe,
        wg,
        sg,
        wu,
        su,
        wd,
        sd,
    }
}

/// Token t's routing: `moe.rs`'s `route` at row t.
fn route(x: &Inputs, t: usize) -> (Vec<usize>, Vec<f64>) {
    let row = &x.xr[t * H..(t + 1) * H];
    let ms = f16(f16(row.iter().map(|v| f16(v * v)).sum::<f64>()) * f16(1.0 / H as f64));
    let r = f16(1.0 / (ms + f16(1e-6)).sqrt());
    let xn: Vec<f64> = (0..H).map(|h| f16(f16(row[h] * r) * x.rg[h])).collect();
    let mut lg: Vec<f64> = (0..E)
        .map(|e| f16((0..H).map(|h| xn[h] * x.rw[e * H + h]).sum()))
        .collect();
    let mut ids = Vec::with_capacity(K);
    let mut vals = Vec::with_capacity(K);
    for _ in 0..K {
        let mx = lg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let id = (0..E).rev().find(|&e| lg[e] == mx).expect("a max lane");
        ids.push(id);
        vals.push(mx);
        lg[id] = f64::NEG_INFINITY;
    }
    let temp = f16((H as f64).powf(-0.5));
    let sv: Vec<f64> = vals.iter().map(|v| f16(v * temp)).collect();
    let mx = sv.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ex: Vec<f64> = sv.iter().map(|v| (v - mx).exp()).collect();
    let sum: f64 = ex.iter().sum();
    let w = (0..K)
        .map(|j| f16(f16(ex[j] / sum) * x.pes[ids[j]]))
        .collect();
    (ids, w)
}

/// Expert e's output for token t: gate/up at the token's scale, gelu·up, down at the
/// (token, e) row's own scale. Shared by both models below, so they differ ONLY in order.
fn expert(x: &Inputs, t: usize, e: usize) -> Vec<f64> {
    let (sx, qx) = codes(&x.xe[t * H..(t + 1) * H]);
    let slab = |b: &[f64]| b[e * I * H..(e + 1) * I * H].to_vec();
    let gate = w8a8(&qx, sx, &slab(&x.wg), &x.sg[e * I..(e + 1) * I], I);
    let up = w8a8(&qx, sx, &slab(&x.wu), &x.su[e * I..(e + 1) * I], I);
    let act: Vec<f64> = (0..I).map(|c| f16(gelu(gate[c]) * up[c])).collect();
    let (sa, qa) = codes(&act);
    w8a8(
        &qa,
        sa,
        &x.wd[e * H * I..(e + 1) * H * I],
        &x.sd[e * H..(e + 1) * H],
        H,
    )
}

/// The PREFILL model: every expert swept in id order, each weighted by the token's dense
/// route (0 off its top-K). `shift` moves every routed id to its neighbour (the control).
fn prefill_reference(x: &Inputs, shift: usize) -> Vec<f64> {
    let mut out = vec![0.0; T * H];
    for t in 0..T {
        let (ids, w) = route(x, t);
        let mut dense = vec![0.0f64; E];
        for (j, &e) in ids.iter().enumerate() {
            dense[(e + shift) % E] = w[j];
        }
        let mut acc = vec![0.0f64; H];
        for (e, &we) in dense.iter().enumerate() {
            let down = expert(x, t, e);
            for h in 0..H {
                acc[h] = f16(acc[h] + f16(down[h] * we));
            }
        }
        out[t * H..(t + 1) * H].copy_from_slice(&acc);
    }
    out
}

/// The DECODE model (`moe.py`'s): each token's K experts in SLOT order.
fn decode_reference(x: &Inputs) -> Vec<f64> {
    let mut out = vec![0.0; T * H];
    for t in 0..T {
        let (ids, w) = route(x, t);
        let mut acc = vec![0.0f64; H];
        for (j, &e) in ids.iter().enumerate() {
            let down = expert(x, t, e);
            for h in 0..H {
                acc[h] = f16(acc[h] + f16(down[h] * w[j]));
            }
        }
        out[t * H..(t + 1) * H].copy_from_slice(&acc);
    }
    out
}

fn worst(a: &[f64], b: &[f64], scale: f64) -> f64 {
    a.iter()
        .zip(b)
        .map(|(g, t)| (g - t).abs() / scale)
        .fold(0.0f64, f64::max)
}

fn executed() -> Vec<f64> {
    use ktir_core::dtypes::DType;
    use ktir_emulator::interpreter::Arg;
    use ktir_emulator::latency::HardwareConfig;
    let f = lowered();
    let x = inputs();
    let f16b = |v: &[f64]| {
        ktir_core::codec::encode(&v.iter().map(|&a| a as f32).collect::<Vec<_>>(), DType::F16)
    };
    let fp8b = |v: &[f64]| -> Vec<u8> {
        v.iter()
            .map(|&a| ktir_core::codec::f32_to_e4m3(a as f32))
            .collect()
    };
    let params: Vec<ktir_core::ir::Ssa> = f.arguments.iter().map(|(s, _)| *s).collect();
    assert_eq!(
        params.len(),
        PTRS.len(),
        "one argument per pointer parameter"
    );
    let bound: Vec<(Vec<u8>, Vec<usize>, DType)> = vec![
        (f16b(&x.xr), vec![T, H], DType::F16),
        (f16b(&x.rg), vec![H], DType::F16),
        (f16b(&x.rw), vec![E, H], DType::F16),
        (f16b(&x.pes), vec![E], DType::F16),
        (f16b(&x.xe), vec![T, H], DType::F16),
        (fp8b(&x.wg), vec![E, I, H], DType::Fp8E4m3),
        (f16b(&x.sg), vec![E, I], DType::F16),
        (fp8b(&x.wu), vec![E, I, H], DType::Fp8E4m3),
        (f16b(&x.su), vec![E, I], DType::F16),
        (fp8b(&x.wd), vec![E, H, I], DType::Fp8E4m3),
        (f16b(&x.sd), vec![E, H], DType::F16),
        (f16b(&[0.0; T * H]), vec![T, H], DType::F16),
        (f16b(&[0.0; T * E]), vec![T, E], DType::F16),
    ];
    let args: Vec<(ktir_core::ir::Ssa, Arg)> = params
        .iter()
        .zip(bound)
        .map(|(&ssa, (data, shape, dtype))| (ssa, Arg::TensorBytes { data, shape, dtype }))
        .collect();
    let (raw, _latency) = ktir_emulator::interpreter::execute_function_with_latency(
        &f,
        &args,
        HardwareConfig::default(),
    )
    .unwrap_or_else(|e| panic!("the emulator refused the kernel: {e}"));
    raw.get(&params[11])
        .expect("desc_o is written and read back")
        .data
        .iter()
        .map(|&v| v as f64)
        .collect()
}

/// THE EXECUTED KERNEL against the prefill model, with a CONTROL: every routed id moved to its
/// neighbour must be visibly different from what ran, or the comparison could not see which
/// experts each token read.
#[test]
fn the_executed_kernel_computes_the_cards_w8a8_prefill_moe() {
    let x = inputs();
    let got = executed();
    let truth = prefill_reference(&x, 0);
    let scale = truth.iter().fold(0.0f64, |a, v| a.max(v.abs())).max(1e-3);
    let err = worst(&got, &truth, scale);
    println!(
        "moe_prefill  n={}  worst err {err:.3e} (of the output's max {scale:.3})",
        T * H
    );
    assert!(
        err < 2e-2,
        "the executed prefill MoE differs from the W8A8 model by {err:.3e} of the output scale"
    );
    let control = worst(&got, &prefill_reference(&x, 1), scale);
    println!("moe_prefill  shifted-expert control err {control:.3e}");
    assert!(
        control > 1e-1,
        "the shifted-expert model agrees with the kernel to {control:.3e}, so this comparison \
         cannot see which experts each token read"
    );
    // The fixture must actually spread the tokens: a sweep that only ever met K experts would
    // not exercise the dense route's zeros.
    let mut seen = std::collections::BTreeSet::new();
    for t in 0..T {
        seen.extend(route(&x, t).0);
    }
    println!(
        "moe_prefill  {T} tokens route to {} distinct experts of {E}",
        seen.len()
    );
    assert!(
        seen.len() > 2 * K,
        "the {T} tokens route to only {} distinct experts — the dense sweep is not exercised",
        seen.len()
    );
}

/// PREFILL == DECODE for the same tokens, to f16 accumulation order. The two models share every
/// per-element grain (`expert`); they differ only in summing the K weighted terms in expert-id
/// order (prefill) or slot order (decode), with exact zeros for the unrouted experts in
/// prefill. So the executed prefill kernel must sit within a few f16 ulps of the decode model —
/// and the bar is set below the per-expert grain change it would see if the two kernels'
/// quantize grains ever drifted apart.
#[test]
fn a_tokens_prefill_output_is_its_decode_output() {
    let x = inputs();
    let got = executed();
    let dec = decode_reference(&x);
    let scale = dec.iter().fold(0.0f64, |a, v| a.max(v.abs())).max(1e-3);
    let err = worst(&got, &dec, scale);
    println!("moe_prefill vs decode  worst err {err:.3e}");
    assert!(
        err < 2e-2,
        "a token's prefill output differs from its decode output by {err:.3e} of the scale"
    );
}

/// Every slab is a DIRECT load: the prompt kernel has no data-dependent address at all.
#[test]
fn no_slab_is_an_indirect_tile() {
    let f = lowered();
    let n = f
        .ops_deep()
        .into_iter()
        .filter(|op| op.op_type == OpKind::KtdpConstructIndirectAccessTile)
        .count();
    assert_eq!(n, 0, "the dense sweep loads every expert's slab directly");
}
