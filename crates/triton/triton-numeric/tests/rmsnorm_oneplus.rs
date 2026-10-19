// SPDX-License-Identifier: Apache-2.0

//! THE `(1 + w)` RMSNORM BODIES, EXECUTED — the OnePlusScale entries' own numerics,
//! driven through the whole ladder to the emulator's numbers.
//!
//! ⚠️ WHY THIS FILE EXISTS — the door side of the row is gated at the descriptor
//! level (`triton_splice_golden.rs`'s oneplus door test censuses the `rmga_` op),
//! but the BODY's own two facts are only witnessed here:
//!
//! 1. **The census**: the gain leg's `arith.addf` — the ONE addf over the gamma
//!    load with the offset's constant splat — carries `Dtype = F16`. That is the
//!    f16-collapse law (SP-E2-03): `w + GAIN_OFFSET` promotes to f32 in Triton's
//!    semantic layer (an f16 tensor plus an f32 literal), and `LegalizeTypes`
//!    collapses the island back to f16 because the only f32-typed operand is the
//!    literal splat. THIS IS LOAD-BEARING FOR PROGRAM-DESCRIPTOR AGREEMENT, not a
//!    benign dtype choice: the door's `rmga_` add runs over f16 buffers, so a
//!    program whose gain add stayed f32 would compute a DIFFERENT `1 + w` than
//!    the descriptor the card runs — the exact divergence the door's
//!    same-formula law exists to prevent. If this census ever RED-stops (the
//!    collapse refuses), the fallback spelling is `tl.full([D], 1.0, tl.float16)`
//!    — an f16 splat from birth — and the census is what says so.
//! 2. **The numbers, against a host model that includes the `+1`**: the executed
//!    entry must reproduce `x · rsqrt(mean(x²) + eps) · (1 + w)` at the sandwich
//!    gate's own tolerance, and the NO-OFFSET control (the same model without
//!    the `+1`) must DIFFER far past it — the proof that this comparison can see
//!    a missing offset at all. The synthesized gamma spans `[-0.75, 0.75]`, so
//!    `1 + w` is up to 100% away from `w` — a dropped (or doubled) offset cannot
//!    hide.
//! 3. **The accumulator is the card's, not IEEE f16's**: the ladder's
//!    `LegalizeTypes` erases every f16→f32 widening (the C++ chain's own
//!    `rmsnorm_granite` oracle records the sum's `arith.addf : f16` combiner), so
//!    the reduce the emulator executes is an f16-TYPED tree-fold — but the
//!    accumulator that fold runs in on the card is DLF16 (`SEN169_FP16`: 6-bit
//!    exponent, 9-bit mantissa, no subnormals, magnitudes to ~8.58e9 — the
//!    vendor's FMA16 reduce templates), and the emulator models exactly that
//!    (`codec::round_dlf16` on the fma-class fold). That is not hypothetical:
//!    gemma-4-12b's sandwich norms at layers 18/22/48 (Σx² = 89,771 / 94,669 /
//!    91,950 at D=3840) sit past IEEE f16's 65504 ceiling — an IEEE-f16 fold
//!    turns them into inf → `rsqrt(inf)` = 0 → an ALL-ZERO norm branch — while
//!    DLF16 holds them five orders inside range. The kernels therefore state the
//!    sum PLAIN (`tl.sum(x·x, 1) · INV_D` — `rmsnorm.py`'s module doc owns the
//!    law): no pre-scale survives the model's 3e7 dynamic range of
//!    mean-of-squares, and none is needed. The ceiling test drives a row whose
//!    raw Σx² is ~24× past 65504 and demands the same model agreement as fact 2;
//!    the small-stream test pins the other end. The host model is f64 and never
//!    overflows, so it is the fixed side of both comparisons.
//!
//! Both entries of both kernels ship in the spyre target's own files — the same
//! text the splice's row splices — so a test-local copy cannot drift:
//! `rmsnorm.py`'s `rmsnorm_oneplus_fwd` and `windowedrmsnorm.py`'s
//! `windowedrmsnorm_oneplus_fwd`.

use std::collections::HashMap;

use ktir_core::arena::Arena;
use ktir_core::attrkey::AttrKey;
use ktir_core::dtypes::DType;
use ktir_core::ir::{Attr, IRFunction, Ssa};
use ktir_core::opkind::OpKind;
use ktir_emulator::interpreter::Arg;
use ktir_emulator::latency::HardwareConfig;
use triton_frontend::codegen::{self, ArgSpec, KernelSpec};
use triton_frontend::opt;
use triton_frontend::semantic::Val;
use triton_frontend::target::Target;

/// The kernels that ship — the splice's own rows splice these same files, so the
/// executed proof and the spliced program are one text, not two that agree.
const SRC: &str = include_str!("../../../targets/spyre/kernels/rmsnorm.py");
const WINDOWED_SRC: &str = include_str!("../../../targets/spyre/kernels/windowedrmsnorm.py");

/// The convention's own constants, stated once: gemma's epsilon and the offset
/// `GainConvention::OnePlusScale` defines (`from_offset` refuses every other value).
const EPS: f64 = 1e-6;
const GAIN_OFFSET: f64 = 1.0;

/// The plain entry's spec — the splice's own `kernel_spec` arm, mirrored: three
/// `*fp16` pointers in the kernel's declaration order, then the six constexprs.
fn plain_spec(m: usize, d: usize) -> KernelSpec {
    let mut signature = HashMap::new();
    for k in ["desc_x", "desc_w", "desc_o"] {
        signature.insert(
            k.to_string(),
            ArgSpec::parse("*fp16").expect("*fp16 parses"),
        );
    }
    for k in ["M", "D_MODEL", "BLOCK_M", "EPS", "INV_D", "GAIN_OFFSET"] {
        signature.insert(
            k.to_string(),
            ArgSpec::parse("constexpr").expect("constexpr parses"),
        );
    }
    let mut constexprs = HashMap::new();
    for (k, v) in [
        ("M", Val::Int(m as i128)),
        ("D_MODEL", Val::Int(d as i128)),
        ("BLOCK_M", Val::Int(m as i128)),
        ("EPS", Val::Float(EPS)),
        ("INV_D", Val::Float(1.0 / d as f64)),
        ("GAIN_OFFSET", Val::Float(GAIN_OFFSET)),
    ] {
        constexprs.insert(k.to_string(), v);
    }
    KernelSpec {
        kernel: "rmsnorm_oneplus_fwd".to_string(),
        signature,
        constexprs,
        file: "rmsnorm.py".to_string(),
    }
}

/// The windowed entry's spec — the sandwich mint's own constexpr set, mirrored.
fn windowed_spec(m: usize, heads: usize, d: usize) -> KernelSpec {
    let mut signature = HashMap::new();
    for k in ["desc_x", "desc_gamma", "desc_o"] {
        signature.insert(
            k.to_string(),
            ArgSpec::parse("*fp16").expect("*fp16 parses"),
        );
    }
    for k in ["M", "D", "HEADS", "EPS", "INV_D", "GAIN_OFFSET"] {
        signature.insert(
            k.to_string(),
            ArgSpec::parse("constexpr").expect("constexpr parses"),
        );
    }
    let mut constexprs = HashMap::new();
    for (k, v) in [
        ("M", Val::Int(m as i128)),
        ("D", Val::Int(d as i128)),
        ("HEADS", Val::Int(heads as i128)),
        ("EPS", Val::Float(EPS)),
        ("INV_D", Val::Float(1.0 / d as f64)),
        ("GAIN_OFFSET", Val::Float(GAIN_OFFSET)),
    ] {
        constexprs.insert(k.to_string(), v);
    }
    KernelSpec {
        kernel: "windowedrmsnorm_oneplus_fwd".to_string(),
        signature,
        constexprs,
        file: "windowedrmsnorm.py".to_string(),
    }
}

/// The whole chain to THEIR `IRFunction`, in the process-wide arena (the kernel
/// gates' own ladder, verbatim). Panics rather than returning an
/// error: a kernel that cannot reach the handoff is measuring nothing.
fn lowered(src: &str, spec: &KernelSpec) -> IRFunction<'static> {
    let mut tt = codegen::compile(src, spec, Target::spyre())
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

// ── THE CENSUS ──────────────────────────────────────────────────────────────────

/// THE GAIN ADDF'S LAW, as the door's own reader states it: over
/// `f.operations` (the door scans exactly this set — the census proves the door
/// can find what it needs, in the shape it accepts), exactly ONE `arith.addf`
/// has a `ktdp.load` operand; that addf carries exactly one constant-splat
/// operand, the splat is the constant 1.0, and the addf is F16-typed (the
/// collapse law — see the module doc for why the dtype is load-bearing).
fn census_gain_addf(f: &IRFunction<'static>, label: &str) {
    let def_of = |s: Ssa| f.operations.iter().find(|o| o.result == Some(s)).copied();
    let mut over_load = 0usize;
    for op in f
        .operations
        .iter()
        .filter(|o| o.op_type == OpKind::ArithAddf)
    {
        let has_load = op
            .operands
            .iter()
            .any(|&s| def_of(s).is_some_and(|d| d.op_type == OpKind::KtdpLoad));
        if !has_load {
            continue;
        }
        over_load += 1;
        // Exactly one splat operand, and it is the offset's constant.
        let splats: Vec<f64> = op
            .operands
            .iter()
            .filter_map(|&s| {
                let sp = def_of(s)?;
                if sp.op_type != OpKind::TensorSplat {
                    return None;
                }
                let c = def_of(*sp.operands.first()?)?;
                if c.op_type != OpKind::ArithConstant {
                    return None;
                }
                c.attributes.iter().find_map(|(kk, v)| match (kk, v) {
                    (AttrKey::Value, Attr::Float(x)) => Some(*x),
                    _ => None,
                })
            })
            .collect();
        assert!(
            splats.len() == 1 && splats[0] == GAIN_OFFSET,
            "{label}: the gain addf's splat operand must be the ONE offset constant \
             {GAIN_OFFSET}; got {splats:?}. The door reads this splat's value back off the \
             program to resolve its registry slot, so a different value is a registry desync"
        );
        assert!(
            matches!(op.attr(AttrKey::Dtype), Some(Attr::Dtype(DType::F16))),
            "{label}: the gain addf must carry Dtype = F16 (the f16-collapse law). Got {:?}. \
             The door's `rmga_` add runs over f16 buffers, so an f32-typed program add computes \
             a DIFFERENT `1 + w` than the descriptor the card runs — the fallback spelling is \
             `tl.full([D], 1.0, tl.float16)`, an f16 splat from birth",
            op.attr(AttrKey::Dtype)
        );
    }
    assert_eq!(
        over_load, 1,
        "{label}: exactly ONE addf over the gamma load — the shape the door's \
         `program_gain_offset` reader accepts. The epsilon's addf reads the reduce result, \
         not a load, so it never counts here"
    );
}

/// The census, on BOTH entries: the door finds the same one addf in the plain
/// body and in the windowed body's unrolled head loop (the gamma load is shared,
/// the add is stated ONCE outside the loop).
#[test]
fn the_gain_addf_is_the_one_f16_addf_over_the_load_in_both_entries() {
    census_gain_addf(&lowered(SRC, &plain_spec(7, 2048)), "rmsnorm_oneplus_fwd");
    census_gain_addf(
        &lowered(WINDOWED_SRC, &windowed_spec(7, 2, 256)),
        "windowedrmsnorm_oneplus_fwd",
    );
}

// ── THE NUMBERS ─────────────────────────────────────────────────────────────────

/// Deterministic source values, the sandwich gate's own recipe: `[-0.75, 0.75]`,
/// so the gamma row's `1 + w` sits in `[0.25, 1.75]` and a dropped offset is an
/// order-one error, not a rounding-scale one.
fn synth(id: usize, n: usize) -> Vec<f64> {
    (0..n)
        .map(|j| (((id * 131 + j * 7) % 197) as f64 / 197.0 - 0.5) * 3.0)
        .collect()
}

/// The host model of the plain body: `x · rsqrt(mean(x²) + eps) · (offset + w)`,
/// per row, in f64. `offset` is the control's knob — `0.0` is the no-offset
/// model the comparison must be able to refuse.
fn plain_model(x: &[f64], w: &[f64], m: usize, d: usize, offset: f64) -> Vec<f64> {
    let mut out = vec![0.0; m * d];
    for mi in 0..m {
        let mut ms = 0.0;
        for j in 0..d {
            ms += x[mi * d + j] * x[mi * d + j];
        }
        let r = 1.0 / (ms / d as f64 + EPS).sqrt();
        for j in 0..d {
            out[mi * d + j] = x[mi * d + j] * r * (offset + w[j]);
        }
    }
    out
}

/// The host model of the windowed body: the same chain per head window, over the
/// shared gamma row.
fn windowed_model(x: &[f64], g: &[f64], m: usize, heads: usize, d: usize, offset: f64) -> Vec<f64> {
    let full = heads * d;
    let mut out = vec![0.0; m * full];
    for mi in 0..m {
        for h in 0..heads {
            let base = mi * full + h * d;
            let mut ms = 0.0;
            for j in 0..d {
                ms += x[base + j] * x[base + j];
            }
            let r = 1.0 / (ms / d as f64 + EPS).sqrt();
            for j in 0..d {
                out[base + j] = x[base + j] * r * (offset + g[j]);
            }
        }
    }
    out
}

/// Run ONE entry's lowered program on the emulator with the given operand bytes
/// (positional binding: argument i is pointer parameter i, the crate's own law),
/// returning the output as f64s.
fn execute(
    src: &str,
    spec: &KernelSpec,
    bound: [(Vec<u8>, Vec<usize>); 3],
) -> (IRFunction<'static>, Vec<f64>) {
    let f = lowered(src, spec);
    let params: Vec<Ssa> = f.arguments.iter().map(|(s, _)| *s).collect();
    assert_eq!(
        params.len(),
        3,
        "desc_x, desc_gamma, desc_o — nothing else becomes an argument"
    );
    let args: Vec<(Ssa, Arg)> = params
        .iter()
        .zip(bound)
        .map(|(&ssa, (data, shape))| {
            (
                ssa,
                Arg::TensorBytes {
                    data,
                    shape,
                    dtype: DType::F16,
                },
            )
        })
        .collect();
    let (raw, _latency) = ktir_emulator::interpreter::execute_function_with_latency(
        &f,
        &args,
        HardwareConfig::default(),
    )
    .unwrap_or_else(|e| panic!("the emulator refused the kernel: {e}"));
    let out = raw
        .get(&params[2])
        .expect("desc_o is written by the stores and read back");
    (f, out.data.iter().map(|&v| v as f64).collect())
}

/// The f16 encoding of an f64 operand, the kernel gates' own hop.
fn encode_f16(v: &[f64]) -> Vec<u8> {
    ktir_core::codec::encode(&v.iter().map(|&a| a as f32).collect::<Vec<_>>(), DType::F16)
}

/// Compare against a model at the sandwich gate's own bar, and prove the
/// no-offset control exceeds it — the comparison can see a missing `+1`.
fn check(label: &str, got: &[f64], with_offset: &[f64], dropped: &[f64]) {
    assert_eq!(got.len(), with_offset.len(), "{label} width");
    let mut worst = 0.0f64;
    for (i, (&g, &wnt)) in got.iter().zip(with_offset).enumerate() {
        let err = (g - wnt).abs();
        let allowance = 2e-2 * wnt.abs().max(1.0);
        assert!(
            err <= allowance,
            "{label}[{i}]: got {g:+.6e}, want {wnt:+.6e} (err {err:.6e} > {allowance:.6e})"
        );
        worst = worst.max(err);
    }
    // THE CONTROL: the no-offset model must exceed the SAME allowance somewhere,
    // or a kernel that dropped the `+1` would read as green above.
    let mut control = 0.0f64;
    let mut sees = false;
    for (&wnt, &drp) in with_offset.iter().zip(dropped) {
        let diff = (wnt - drp).abs();
        let allowance = 2e-2 * wnt.abs().max(1.0);
        if diff > allowance {
            sees = true;
        }
        control = control.max(diff);
    }
    assert!(
        sees,
        "{label}: the no-offset control exceeds the allowance NOWHERE (max diff {control:.6e}), \
         so the comparison above cannot see a missing offset and its green result is vacuous"
    );
    println!("{label:34} max|err| = {worst:.6e}  no-offset control max|diff| = {control:.6e}");
}

/// The numbers, on BOTH entries: decode rows and a prefill-style rung, the
/// sandwich gate's own geometries (D=256 spans four sticks; H=4 makes the head
/// loop multi-trip before the unroll).
#[test]
fn the_executed_oneplus_bodies_compute_x_r_times_1_plus_w() {
    // ── the plain entry ──
    for &(m, d) in &[(1usize, 2048usize), (7, 2048)] {
        let x = synth(0, m * d);
        let w = synth(1, d);
        let bound = [
            (encode_f16(&x), vec![m, d]),
            (encode_f16(&w), vec![d]),
            (encode_f16(&vec![0.0; m * d]), vec![m, d]),
        ];
        let (_f, got) = execute(SRC, &plain_spec(m, d), bound);
        check(
            &format!("rmsnorm_oneplus m={m} d={d}"),
            &got,
            &plain_model(&x, &w, m, d, GAIN_OFFSET),
            &plain_model(&x, &w, m, d, 0.0),
        );
    }
    // ── the windowed entry ──
    for &(m, heads, d) in &[(7usize, 2usize, 256usize), (1, 2, 256), (5, 4, 128)] {
        let full = heads * d;
        let x = synth(0, m * full);
        let g = synth(1, d);
        let bound = [
            (encode_f16(&x), vec![m, full]),
            (encode_f16(&g), vec![d]),
            (encode_f16(&vec![0.0; m * full]), vec![m, full]),
        ];
        let (_f, got) = execute(WINDOWED_SRC, &windowed_spec(m, heads, d), bound);
        check(
            &format!("windowedrmsnorm_oneplus m={m} h={heads} d={d}"),
            &got,
            &windowed_model(&x, &g, m, heads, d, GAIN_OFFSET),
            &windowed_model(&x, &g, m, heads, d, 0.0),
        );
    }
}

/// ⭐ THE f16 CEILING — the same bodies on rows whose RAW Σx² sits far past 65504,
/// where an IEEE-f16 tree-fold would fold inf and emit an all-zero norm branch
/// (module doc, fact 3). The accumulator the card runs is DLF16 (~8.58e9 max) and
/// the emulator models it, so the PLAIN body must hold the same model agreement
/// the below-ceiling test demands; the host model is f64, so it is the fixed side
/// of the comparison. The scales are exact powers of two, so the f16-encoded
/// operands carry the synth recipe's own mantissas — the row differs from the
/// below-ceiling one in magnitude only.
///
/// The ceilings, for the record (the synth recipe is uniform on ±1.5, rms ≈ 0.87):
/// the plain entry at d=2048 overflows above rms ≈ 5.66 (Σx² > 65504) and the
/// windowed entry at d=256 above rms ≈ 16.0; the rows here run at rms ≈ 27.7 and
/// ≈ 55.4 — past the IEEE-f16 ceiling in both (24× and 12×), with 5400× and
/// 10900× of DLF16 range left above them.
#[test]
fn the_executed_bodies_survive_a_row_past_the_f16_sum_ceiling() {
    // ── the plain entry, scaled ×32: Σx² ≈ 1.57M at d=2048 ──
    for &(m, d) in &[(1usize, 2048usize), (7, 2048)] {
        let x: Vec<f64> = synth(0, m * d).iter().map(|v| v * 32.0).collect();
        let w = synth(1, d);
        let bound = [
            (encode_f16(&x), vec![m, d]),
            (encode_f16(&w), vec![d]),
            (encode_f16(&vec![0.0; m * d]), vec![m, d]),
        ];
        let (_f, got) = execute(SRC, &plain_spec(m, d), bound);
        check(
            &format!("rmsnorm_oneplus CEILING m={m} d={d}"),
            &got,
            &plain_model(&x, &w, m, d, GAIN_OFFSET),
            &plain_model(&x, &w, m, d, 0.0),
        );
    }
    // ── the windowed entry, scaled ×64: Σx² ≈ 786k per d=256 window ──
    for &(m, heads, d) in &[(7usize, 2usize, 256usize), (1, 2, 256)] {
        let full = heads * d;
        let x: Vec<f64> = synth(0, m * full).iter().map(|v| v * 64.0).collect();
        let g = synth(1, d);
        let bound = [
            (encode_f16(&x), vec![m, full]),
            (encode_f16(&g), vec![d]),
            (encode_f16(&vec![0.0; m * full]), vec![m, full]),
        ];
        let (_f, got) = execute(WINDOWED_SRC, &windowed_spec(m, heads, d), bound);
        check(
            &format!("windowedrmsnorm_oneplus CEILING m={m} h={heads} d={d}"),
            &got,
            &windowed_model(&x, &g, m, heads, d, GAIN_OFFSET),
            &windowed_model(&x, &g, m, heads, d, 0.0),
        );
    }
}

/// ⭐ THE SMALL-STREAM ROWS — the no-crush pin (module doc, fact 3). The rows run
/// at and below the model's smallest real stream — mean-of-squares 1.9e-5 (×0.005)
/// and 3.0e-6 (×0.002) against the 12b's measured floor 2.5e-5 (the sandwich
/// norms at layers 2-4) — and the PLAIN body must hold the SAME model agreement
/// the below-ceiling test demands at full magnitude. The accumulator is DLF16,
/// whose flush floor is per-element: squares below 2⁻³¹ flush to zero, i.e.
/// |x| < 2.2e-5 — seven orders under the smallest real square, so no real stream
/// crushes. (An IEEE-f16 accumulator would crush here too, one order sooner —
/// its subnormal quantum 2⁻²⁴ bites at |x| < 2.4e-4; the DLF16 model is what
/// moves the floor out of reach.) The host model is f64; the epsilon (1e-6) is
/// the model's own, and both sides add it, so the smallest row's eps-dominated
/// regime hides nothing. A future arch with smaller streams re-derives this pin.
#[test]
fn the_executed_bodies_stay_exact_on_small_stream_rows() {
    // ── the plain entry, at stream scale: mean-of-squares 1.9e-5 (×0.005) and
    // 3.0e-6 (×0.002) at d=2048 ──
    for &scale in &[0.005f64, 0.002] {
        for &(m, d) in &[(1usize, 2048usize), (7, 2048)] {
            let x: Vec<f64> = synth(0, m * d).iter().map(|v| v * scale).collect();
            let w = synth(1, d);
            let bound = [
                (encode_f16(&x), vec![m, d]),
                (encode_f16(&w), vec![d]),
                (encode_f16(&vec![0.0; m * d]), vec![m, d]),
            ];
            let (_f, got) = execute(SRC, &plain_spec(m, d), bound);
            check(
                &format!("rmsnorm_oneplus SMALL m={m} d={d} x{scale}"),
                &got,
                &plain_model(&x, &w, m, d, GAIN_OFFSET),
                &plain_model(&x, &w, m, d, 0.0),
            );
        }
    }
    // ── the windowed entry — the per-head windows the gate saw damaged first
    // (the layer-2 q/k staging), same scales ──
    for &scale in &[0.005f64, 0.002] {
        for &(m, heads, d) in &[(7usize, 2usize, 256usize), (1, 2, 256)] {
            let full = heads * d;
            let x: Vec<f64> = synth(0, m * full).iter().map(|v| v * scale).collect();
            let g = synth(1, d);
            let bound = [
                (encode_f16(&x), vec![m, full]),
                (encode_f16(&g), vec![d]),
                (encode_f16(&vec![0.0; m * full]), vec![m, full]),
            ];
            let (_f, got) = execute(WINDOWED_SRC, &windowed_spec(m, heads, d), bound);
            check(
                &format!("windowedrmsnorm_oneplus SMALL m={m} h={heads} d={d} x{scale}"),
                &got,
                &windowed_model(&x, &g, m, heads, d, GAIN_OFFSET),
                &windowed_model(&x, &g, m, heads, d, 0.0),
            );
        }
    }
}
