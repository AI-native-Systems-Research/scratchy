// SPDX-License-Identifier: Apache-2.0
//! A one-row dense GEMM with its pre-norm folded in (`gemv_normed_*`: a router's logits,
//! `MetalFusion::NormedRouter`) against the commands it replaces: the RMSNorm by the router scale,
//! then the plain GEMV over its row.
//!
//! Normalizing on load dots `x ⊙ gain` and scales each row by `1 / rms(x)`, so it skips the normed
//! row's rounding: its output must be as close to the exact `W · rmsnorm(x, gain)` as the unfused
//! norm then GEMV is — and the plain GEMV of the raw `x`, which the same bound must reject, shows
//! the bound sees a missing norm.
//!
//! GPU tests — run with `--test-threads=1` (standing rule).

use half::{bf16, f16};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize};
use scratchy_target_metal::aot::baked_build;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::mtl4_dispatch::Mtl4DispatchBatch;
use scratchy_target_metal::specialized_pipeline_cache::{
    ComputePipelineState, ConstantValue, PipelineKey, SpecializedPipelineCache,
};
use scratchy_target_metal::tape::ids::{BucketM, KDim, NDim, QSize, RmsNormEps};
use scratchy_target_metal::tape::kernel_constants::{
    NORM_THREADS, NormedGemvConstants, RmsNormConstants,
};
use scratchy_target_metal::tape::step::Eps;

type Device = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLDevice>>;
type Buffer = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>>;

#[derive(Clone, Copy, Debug)]
enum Dtype {
    F16,
    Bf16,
}

impl Dtype {
    fn bits(self, x: f32) -> u16 {
        match self {
            Dtype::F16 => f16::from_f32(x).to_bits(),
            Dtype::Bf16 => bf16::from_f32(x).to_bits(),
        }
    }

    fn value(self, b: u16) -> f64 {
        f64::from(match self {
            Dtype::F16 => f16::from_bits(b).to_f32(),
            Dtype::Bf16 => bf16::from_bits(b).to_f32(),
        })
    }

    /// The unit roundoff: the largest relative error of rounding to the type.
    fn roundoff(self) -> f64 {
        match self {
            Dtype::Bf16 => 2f64.powi(-8),
            Dtype::F16 => 2f64.powi(-11),
        }
    }

    fn tag(self) -> &'static str {
        match self {
            Dtype::F16 => "f16",
            Dtype::Bf16 => "bf16",
        }
    }
}

/// `n` logits of a `k`-wide row; activations and weights in `act`, the gain in `gain`.
#[derive(Clone, Copy, Debug)]
struct Case {
    act: Dtype,
    gain: Dtype,
    k: usize,
    n: usize,
}

const EPS: f32 = 1e-6;

struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn shared(device: &Device, data: &[u16]) -> Buffer {
    let bytes = std::mem::size_of_val(data);
    let buf = device
        .newBufferWithLength_options(bytes.max(16), MTLResourceOptions::StorageModeShared)
        .expect("newBuffer");
    // SAFETY: `buf` holds at least `bytes` bytes and does not overlap `data`.
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8,
            buf.contents().as_ptr() as *mut u8,
            bytes,
        );
    }
    buf
}

fn read(buf: &Buffer, n: usize) -> Vec<u16> {
    // SAFETY: every `buf` read here was made by `shared` from at least `n` u16s.
    unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const u16, n).to_vec() }
}

fn size((x, y, z): (usize, usize, usize)) -> MTLSize {
    MTLSize {
        width: x,
        height: y,
        depth: z,
    }
}

fn leak(name: String) -> &'static str {
    Box::leak(name.into_boxed_str())
}

struct Rig {
    device: Device,
    _cache: SpecializedPipelineCache,
    normed: ComputePipelineState,
    plain: ComputePipelineState,
    norm: ComputePipelineState,
}

fn rig(c: &Case) -> Option<Rig> {
    let device = detect_device()?.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("shaders");
    let (act, gain) = (c.act.tag(), c.gain.tag());
    let build = |library, name: &'static str, constants| {
        baked_build(&cache, &PipelineKey::new(library, name, constants)).expect(name)
    };
    let normed = NormedGemvConstants {
        n: NDim(c.n as u32),
        k: KDim(c.k as u32),
        eps: Eps(EPS),
    };
    let normed = build("gemm", leak(format!("gemv_normed_{act}_s_{gain}")), normed.into());
    let dims = vec![
        ConstantValue::uint(0, 1),
        ConstantValue::uint(1, c.n as u32),
        ConstantValue::uint(2, c.k as u32),
    ];
    let plain = build("gemm", leak(format!("gemv_{act}_specialized")), dims);
    let norm = RmsNormConstants {
        bucket_m: BucketM(1),
        q_size: QSize(c.k as u32),
        rms_norm_eps: RmsNormEps(EPS),
        weight_offset: 0.0,
    };
    let norm = build(
        "rmsnorm",
        leak(format!("rmsnorm_{act}_s_{gain}_specialized")),
        norm.into(),
    );
    Some(Rig {
        device,
        _cache: cache,
        normed,
        plain,
        norm,
    })
}

/// The weights `[n][k]`, the row and the gain, as stored.
struct Inputs {
    w: Vec<u16>,
    x: Vec<u16>,
    gain: Vec<u16>,
}

fn inputs(c: &Case, rng: &mut Lcg) -> Inputs {
    let w = (0..c.n * c.k).map(|_| c.act.bits(0.05 * rng.unit())).collect();
    let x = (0..c.k).map(|_| c.act.bits(6.0 * rng.unit())).collect();
    let gain = (0..c.k).map(|_| c.gain.bits(0.5 + rng.unit().abs())).collect();
    Inputs { w, x, gain }
}

/// `W · rmsnorm(x, gain)` in f64, each row with the bound on its error: the output's rounding,
/// plus twice what rounding each normed input to the activation type can move its dot.
fn exact(c: &Case, i: &Inputs) -> Vec<(f64, f64)> {
    let u = c.act.roundoff();
    let x: Vec<f64> = i.x.iter().map(|&b| c.act.value(b)).collect();
    let rms = (x.iter().map(|v| v * v).sum::<f64>() / c.k as f64 + f64::from(EPS)).sqrt();
    let xn: Vec<f64> = (x.iter().zip(&i.gain))
        .map(|(v, &g)| v / rms * c.gain.value(g))
        .collect();
    (0..c.n)
        .map(|r| {
            let row = &i.w[r * c.k..(r + 1) * c.k];
            let terms = row.iter().zip(&xn).map(|(&w, x)| c.act.value(w) * x);
            let (dot, scale) = terms.fold((0.0, 0.0), |(d, m), t| (d + t, m + f64::abs(t)));
            (dot, u * dot.abs() + 2.0 * u * scale)
        })
        .collect()
}

#[derive(Clone, Copy)]
enum How {
    Normed,
    NormThenPlain,
    Plain,
}

fn run(r: &Rig, c: &Case, i: &Inputs, how: How) -> Vec<u16> {
    let buf = |d: &[u16]| shared(&r.device, d);
    let (w, x, g) = (buf(&i.w), buf(&i.x), buf(&i.gain));
    let (y, normed_row) = (buf(&vec![0u16; c.n]), buf(&vec![0u16; c.k]));
    let mut batch = Mtl4DispatchBatch::begin(&r.device).expect("mtl4");
    let (groups, threads) = (size((c.n.div_ceil(4), 1, 1)), size((256, 1, 1)));
    match how {
        How::Normed => {
            let binds = [(&y, 0), (&x, 1), (&w, 2), (&g, 3)];
            batch.encode(&r.normed, &binds, &[], &[], &[], groups, threads);
        }
        How::Plain => {
            let binds = [(&y, 0), (&x, 1), (&w, 2)];
            batch.encode(&r.plain, &binds, &[], &[], &[], groups, threads);
        }
        How::NormThenPlain => {
            let binds = [(&normed_row, 0), (&x, 1), (&g, 2)];
            let one = size((1, 1, 1));
            let norm_threads = size((NORM_THREADS as usize, 1, 1));
            batch.encode(&r.norm, &binds, &[], &[], &[], one, norm_threads);
            batch.barrier();
            let binds = [(&y, 0), (&normed_row, 1), (&w, 2)];
            batch.encode(&r.plain, &binds, &[], &[], &[], groups, threads);
        }
    }
    batch.commit(true);
    read(&y, c.n)
}

/// Whether every row of `got` is within its bound of the exact; the first that is not.
fn within(c: &Case, got: &[u16], exact: &[(f64, f64)]) -> Result<(), String> {
    for (r, (&b, &(e, bound))) in got.iter().zip(exact).enumerate() {
        let v = c.act.value(b);
        if (v - e).abs() > bound {
            return Err(format!("row {r}: {v} vs exact {e} (bound {bound})"));
        }
    }
    Ok(())
}

fn cases() -> Vec<Case> {
    // Gemma 4 26B's router: 128 experts over a 2816-wide row (a partial last K block).
    let gemma4 = Case {
        act: Dtype::Bf16,
        gain: Dtype::Bf16,
        k: 2816,
        n: 128,
    };
    vec![
        gemma4,
        Case {
            gain: Dtype::F16,
            ..gemma4
        },
        Case {
            act: Dtype::F16,
            gain: Dtype::F16,
            k: 2048,
            n: 64,
        },
    ]
}

#[test]
fn a_normed_gemv_is_as_close_to_the_exact_normed_product_as_the_norm_then_gemv() {
    for c in cases() {
        let Some(r) = rig(&c) else {
            eprintln!("no Metal device; skipping");
            return;
        };
        let i = inputs(&c, &mut Lcg(c.k as u64 * 31 + c.n as u64));
        let want = exact(&c, &i);
        let check = |how, what| {
            within(&c, &run(&r, &c, &i, how), &want).map_err(|e| format!("{c:?}: {what}: {e}"))
        };
        check(How::NormThenPlain, "the norm, then the GEMV").unwrap_or_else(|e| panic!("{e}"));
        check(How::Normed, "the normed GEMV").unwrap_or_else(|e| panic!("{e}"));
        assert!(
            check(How::Plain, "the plain GEMV").is_err(),
            "{c:?}: the bound accepts the GEMV of the raw row"
        );
    }
}
