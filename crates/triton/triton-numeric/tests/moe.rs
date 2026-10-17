// SPDX-License-Identifier: Apache-2.0

//! THE FUSED MoE KERNEL, through the ladder: `kernels/moe.py` (router + experts, one program)
//! from Python to the emulator, against an f64 model of the card's W8A8 numerics.
//!
//! ```text
//!   triton_frontend      .py -> TTIR       (include_str!'d from the spyre target: the file
//!                                           the splice splices, not a test-local copy)
//!   triton_ktir          TTIR -> KTIR      (the routed ids -> indirect slab tiles)
//!   to_ktir_emit         KTIR -> IRFunction
//!   ktir_emulator        the executed output
//! ```

use std::collections::HashMap;

use ktir_core::arena::Arena;
use ktir_core::ir::IRFunction;
use ktir_core::opkind::OpKind;
use triton_frontend::codegen::{self, ArgSpec, KernelSpec};
use triton_frontend::opt;
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;

const SRC: &str = include_str!("../../../targets/spyre/kernels/moe.py");

/// The golden geometry (`triton-frontend`'s `common::moe`): one token, 64 experts, top-8.
const M: usize = 1;
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
    ("desc_ids", "*i32"),
    ("desc_rs", "*fp16"),
    ("desc_act", "*fp16"),
];

fn spec() -> KernelSpec {
    let mut signature = HashMap::new();
    for (k, v) in PTRS {
        signature.insert(k.to_string(), ArgSpec::parse(v).expect("pointer spec"));
    }
    let h = H as f64;
    let ces: [(&str, Val); 10] = [
        ("M", Val::Int(M as i128)),
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
        kernel: "moe_fwd".to_string(),
        signature,
        constexprs,
        file: "moe.py".to_string(),
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

// ── the reference ────────────────────────────────────────────────────────────────────────

/// Round to f16 (RTNE), the card's compute width — through the emulator's own codec, so the
/// reference and the executor share one f16.
fn f16(v: f64) -> f64 {
    let b = ktir_core::codec::encode(&[v as f32], ktir_core::dtypes::DType::F16);
    ktir_core::codec::decode(&b, 1, ktir_core::dtypes::DType::F16)[0] as f64
}

/// Round to e4m3 (RTNE, saturating at ±448) — the card's `qfp8ch`.
fn e4m3(v: f64) -> f64 {
    ktir_core::codec::e4m3_to_f32(ktir_core::codec::f32_to_e4m3(v as f32)) as f64
}

/// `quantscalepertokenfp8`: `min(max(amax · 1/448, 1/448²), 448)` at f16.
fn fp8_scale(amax: f64) -> f64 {
    f16(f16(amax * f16(1.0 / 448.0))
        .max(f16(1.0 / 448.0 / 448.0))
        .min(448.0))
}

/// The W8A8 product of one row: `((codes · Wcodes) · s) · w_scale`, f16 at each step.
fn w8a8(codes: &[f64], s: f64, w: &[f64], w_scale: &[f64], n: usize) -> Vec<f64> {
    let k = codes.len();
    (0..n)
        .map(|j| {
            let dot: f64 = (0..k).map(|c| codes[c] * w[j * k + c]).sum();
            f16(f16(f16(dot) * s) * w_scale[j])
        })
        .collect()
}

/// The gelu-tanh the kernel spells: `g / (1 + exp(-1.5957691216057308 · (g + 0.044715 g³)))`,
/// the cubic and the exp in f32, the sum and divide in f16.
fn gelu(g: f64) -> f64 {
    let g32 = g as f32;
    // The kernel's `1.5957691216057308` (= 2·√(2/π)), at the f32 the island computes in.
    let v = 1.595_769_2_f32 * (g32 + 0.044_715_f32 * g32 * g32 * g32);
    let e = f16((-v).exp() as f64);
    f16(g / f16(1.0 + e))
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

/// A deterministic, well-conditioned draw. Every fp8 code is a REPRESENTABLE e4m3 value, so
/// the banks are exact; scales sit in [0.75, 1.25]; the router weights are spread so the
/// top-8 has clear gaps (a near-tie at the 8/9 boundary would make the reference's own f16
/// logit rounding decide a slot).
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
    let xr: Vec<f64> = (0..M * H).map(|_| f16(d.sym(1.0))).collect();
    let rg: Vec<f64> = (0..H).map(|_| f16(1.0 + d.sym(0.25))).collect();
    // Router rows: a per-expert bias along a fixed direction makes the logits well spread.
    let dir: Vec<f64> = (0..H).map(|_| d.sym(1.0)).collect();
    let rw: Vec<f64> = (0..E * H)
        .map(|i| {
            let (e, h) = (i / H, i % H);
            f16(0.02 * (e as f64 - E as f64 / 2.0) * dir[h] + d.sym(0.05))
        })
        .collect();
    let pes: Vec<f64> = (0..E).map(|_| f16(0.5 + d.u())).collect();
    let xe: Vec<f64> = (0..M * H).map(|_| f16(d.sym(1.0))).collect();
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

/// The routing the kernel computes: rmsnorm, f16 logits, top-K (descending; ties to the
/// higher id), softmax over the K at `H^-0.5`, × the selected per-expert scales.
fn route(x: &Inputs, m: usize) -> (Vec<usize>, Vec<f64>) {
    let row = &x.xr[m * H..(m + 1) * H];
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

/// `out[m]`, the card's W8A8 numerics op by op.
fn reference(x: &Inputs, ids_override: Option<&[usize]>) -> Vec<f64> {
    let mut out = vec![0.0; M * H];
    for m in 0..M {
        let (mut ids, w) = route(x, m);
        if let Some(o) = ids_override {
            ids = o.to_vec();
        }
        let xe = &x.xe[m * H..(m + 1) * H];
        let sx = fp8_scale(xe.iter().fold(0.0f64, |a, v| a.max(v.abs())));
        let inv = f16(1.0 / sx);
        let qx: Vec<f64> = xe
            .iter()
            .map(|v| e4m3(f16(v * inv).clamp(-448.0, 448.0)))
            .collect();
        let acts: Vec<Vec<f64>> = ids
            .iter()
            .map(|&e| {
                let slab = |b: &[f64]| b[e * I * H..(e + 1) * I * H].to_vec();
                let gate = w8a8(&qx, sx, &slab(&x.wg), &x.sg[e * I..(e + 1) * I], I);
                let up = w8a8(&qx, sx, &slab(&x.wu), &x.su[e * I..(e + 1) * I], I);
                (0..I).map(|c| f16(gelu(gate[c]) * up[c])).collect()
            })
            .collect();
        let amax = acts.iter().flatten().fold(0.0f64, |a, v| a.max(v.abs()));
        let sa = fp8_scale(f16(amax));
        let inv_a = f16(1.0 / sa);
        let mut acc = vec![0.0f64; H];
        for (j, &e) in ids.iter().enumerate() {
            let qa: Vec<f64> = acts[j]
                .iter()
                .map(|v| e4m3(f16(v * inv_a).clamp(-448.0, 448.0)))
                .collect();
            let down = w8a8(
                &qa,
                sa,
                &x.wd[e * H * I..(e + 1) * H * I],
                &x.sd[e * H..(e + 1) * H],
                H,
            );
            for h in 0..H {
                acc[h] = f16(acc[h] + f16(down[h] * w[j]));
            }
        }
        out[m * H..(m + 1) * H].copy_from_slice(&acc);
    }
    out
}

/// THE EXECUTED KERNEL against the reference, with a CONTROL: the same reference with every
/// routed id moved to its neighbour must be visibly different from what ran, or the
/// comparison could not see which experts the kernel read.
#[test]
fn the_executed_kernel_computes_the_cards_w8a8_moe() {
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
        (f16b(&x.xr), vec![M, H], DType::F16),
        (f16b(&x.rg), vec![H], DType::F16),
        (f16b(&x.rw), vec![E, H], DType::F16),
        (f16b(&x.pes), vec![E], DType::F16),
        (f16b(&x.xe), vec![M, H], DType::F16),
        (fp8b(&x.wg), vec![E, I, H], DType::Fp8E4m3),
        (f16b(&x.sg), vec![E, I], DType::F16),
        (fp8b(&x.wu), vec![E, I, H], DType::Fp8E4m3),
        (f16b(&x.su), vec![E, I], DType::F16),
        (fp8b(&x.wd), vec![E, H, I], DType::Fp8E4m3),
        (f16b(&x.sd), vec![E, H], DType::F16),
        (f16b(&[0.0; M * H]), vec![M, H], DType::F16),
        (vec![0u8; M * K * 4], vec![M, K], DType::I32),
        (f16b(&[0.0; M * K]), vec![M, K], DType::F16),
        (f16b(&[0.0; M * K * I]), vec![M, K * I], DType::F16),
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
    let read = |i: usize| -> Vec<f64> {
        raw.get(&params[i])
            .unwrap_or_else(|| panic!("parameter {i} is written and read back"))
            .data
            .iter()
            .map(|&v| v as f64)
            .collect()
    };
    let got = read(11);
    // The routing first: the ids the kernel stored must be the reference's.
    let (want_ids, _) = route(&x, 0);
    let got_ids: Vec<usize> = read(12).iter().map(|&v| v as usize).collect();
    assert_eq!(got_ids, want_ids, "the kernel routed to different experts");

    let truth = reference(&x, None);
    let scale = truth.iter().fold(0.0f64, |a, v| a.max(v.abs())).max(1e-3);
    let worst = got
        .iter()
        .zip(&truth)
        .map(|(g, t)| (g - t).abs() / scale)
        .fold(0.0f64, f64::max);
    println!(
        "moe  n={}  worst err {worst:.3e} (of the output's max {scale:.3})",
        M * H
    );
    assert!(
        worst < 2e-2,
        "the executed MoE differs from the W8A8 reference by {worst:.3e} of the output scale"
    );

    let shifted: Vec<usize> = want_ids.iter().map(|&e| (e + 1) % E).collect();
    let control = reference(&x, Some(&shifted));
    let control_err = got
        .iter()
        .zip(&control)
        .map(|(g, c)| (g - c).abs() / scale)
        .fold(0.0f64, f64::max);
    assert!(
        control_err > 1e-1,
        "the shifted-expert model agrees with the kernel to {control_err:.3e}, so this \
         comparison cannot see which experts the kernel read"
    );
}

/// The handoff census: every routed pair's three expert slabs and their three scale rows
/// reach the emulator as indirect tiles — `M · K · (3 + 3)` at pass 1's gate/up and pass 2's
/// down — so the numbers measure the gathers, not a folded-away direct read.
#[test]
fn every_routed_slab_reaches_the_emulator_as_an_indirect_tile() {
    let f = lowered();
    let n = f
        .ops_deep()
        .into_iter()
        .filter(|op| op.op_type == OpKind::KtdpConstructIndirectAccessTile)
        .count();
    assert_eq!(
        n,
        M * K * 6,
        "gate, up, down and their three scale rows, per pair"
    );
}
