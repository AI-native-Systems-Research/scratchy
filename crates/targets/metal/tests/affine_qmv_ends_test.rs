// SPDX-License-Identifier: Apache-2.0
//! A one-row affine matvec's folded ends (`QmvEnds`: `MetalFusion::NormedQmv` / `ResidualQmv`)
//! against the commands they replace in a decode step: the RMSNorm before the matvec, the
//! residual add after it.
//!
//! - Normalizing on load dots `x ⊙ gain` and scales the row by `1 / rms(x)`, so it skips the
//!   normed row's rounding: its output must be as close to the exact `W · rmsnorm(x)` as the
//!   unfused `rmsnorm` then matvec is — and the plain matvec of the raw `x`, which the same bound
//!   must reject, shows the bound sees a missing norm.
//! - Adding into the residual must be the plain matvec's row added to the residual, to within the
//!   row's own rounding: the sum takes one rounding where the unfused add's took the row's too.
//!
//! GPU tests — run with `--test-threads=1` (standing rule).

use half::{bf16, f16};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize};
use scratchy_target_metal::aot::baked_build;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::mtl4_dispatch::Mtl4DispatchBatch;
use scratchy_target_metal::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};
use scratchy_target_metal::tape::ids::{BucketM, KDimI32, LayerId, NDimI32, QSize, RmsNormEps};
use scratchy_target_metal::tape::kernel_constants::{
    AffineCodes, AffineGatedQmvConstants, AffineQmvConstants, NORM_THREADS, RmsNormConstants,
};
use scratchy_target_metal::tape::quantized::{
    DequantDtype, QmvKernel, ScaleDtype, pick_qmv_kernel, qmv_dispatch_shape,
    qmv_kernel_static_name,
};
use scratchy_target_metal::tape::step::{Eps, GainOffset, GatedAct, QmvEnds, RowNorm};

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

    fn value(self, b: u16) -> f32 {
        match self {
            Dtype::F16 => f16::from_bits(b).to_f32(),
            Dtype::Bf16 => bf16::from_bits(b).to_f32(),
        }
    }

    fn tag(self) -> &'static str {
        match self {
            Dtype::F16 => "f16",
            Dtype::Bf16 => "bf16",
        }
    }

    fn dequant(self) -> DequantDtype {
        match self {
            Dtype::F16 => DequantDtype::F16,
            Dtype::Bf16 => DequantDtype::Bf16,
        }
    }

    fn scale(self) -> ScaleDtype {
        match self {
            Dtype::F16 => ScaleDtype::F16,
            Dtype::Bf16 => ScaleDtype::Bf16,
        }
    }
}

/// One matvec: `k` in, `n` out, 4-bit codes in groups of `group_size`; its norm's gain offset.
#[derive(Clone, Copy, Debug)]
struct Case {
    dtype: Dtype,
    group_size: usize,
    k: usize,
    n: usize,
    offset: f32,
}

impl Case {
    fn kernel(&self) -> QmvKernel {
        pick_qmv_kernel(self.n as u32, self.k as u32, 4)
    }
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 16
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 24) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

fn shared<T: Copy>(device: &Device, data: &[T]) -> Buffer {
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

fn read_u16(buf: &Buffer, n: usize) -> Vec<u16> {
    // SAFETY: every `buf` read here was made by `shared` from at least `n` u16s.
    unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const u16, n).to_vec() }
}

fn size((x, y, z): (u32, u32, u32)) -> MTLSize {
    MTLSize {
        width: x as usize,
        height: y as usize,
        depth: z as usize,
    }
}

/// An affine projection: its packed codes, scales and biases, and the weights they decode to.
struct Weights {
    w: Buffer,
    scales: Buffer,
    biases: Buffer,
    /// `[n][k]`, as the kernels decode them.
    decoded: Vec<f64>,
}

impl Weights {
    fn new(device: &Device, c: &Case, rng: &mut Lcg) -> Self {
        let words: Vec<u32> = (0..c.n * c.k / 8).map(|_| rng.next() as u32).collect();
        let groups = c.n * c.k / c.group_size;
        let (scales, biases): (Vec<u16>, Vec<u16>) = (0..groups)
            .map(|_| {
                let scale = (0.5 + rng.unit().abs()) * 0.02 / 15.0;
                let bias = -scale * 7.5 * (1.0 + 0.1 * rng.unit());
                (c.dtype.bits(scale), c.dtype.bits(bias))
            })
            .unzip();
        let decoded = (0..c.n * c.k)
            .map(|e| {
                let code = (words[e / 8] >> (4 * (e % 8))) & 0xF;
                let g = e / c.group_size;
                let (s, b) = (c.dtype.value(scales[g]), c.dtype.value(biases[g]));
                f64::from(s) * f64::from(code) + f64::from(b)
            })
            .collect();
        Self {
            w: shared(device, &words),
            scales: shared(device, &scales),
            biases: shared(device, &biases),
            decoded,
        }
    }

    /// Row `row`'s dot with `x`, and the sum of its terms' magnitudes: the scale its rounding
    /// errors take.
    fn dot(&self, k: usize, row: usize, x: &[f64]) -> (f64, f64) {
        let r = &self.decoded[row * k..(row + 1) * k];
        let terms = r.iter().zip(x).map(|(w, x)| w * x);
        terms.fold((0.0, 0.0), |(d, m), t| (d + t, m + t.abs()))
    }
}

/// The kernels of case `c` with the device and cache that built them.
struct Rig {
    device: Device,
    _cache: SpecializedPipelineCache,
    plain: scratchy_target_metal::specialized_pipeline_cache::ComputePipelineState,
    normed: scratchy_target_metal::specialized_pipeline_cache::ComputePipelineState,
    adding: scratchy_target_metal::specialized_pipeline_cache::ComputePipelineState,
    norm: scratchy_target_metal::specialized_pipeline_cache::ComputePipelineState,
}

const EPS: f32 = 1e-5;

fn ends(c: &Case, norm: bool, residual: bool) -> QmvEnds {
    let norm = norm.then_some(RowNorm {
        layer: LayerId(0),
        eps: Eps(EPS),
        offset: GainOffset(c.offset),
    });
    QmvEnds { norm, residual }
}

fn rig(c: &Case) -> Option<Rig> {
    let device = detect_device()?.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("shaders");
    let name = qmv_kernel_static_name(c.kernel(), c.dtype.dequant(), c.dtype.scale(), 4, c.group_size as u32);
    let qmv = |e: QmvEnds| {
        let mut v: Vec<ConstantValue> = AffineQmvConstants {
            k: KDimI32(c.k as i32),
            n: NDimI32(c.n as i32),
            codes: AffineCodes::AsWritten,
        }
        .into();
        v.extend(Vec::<ConstantValue>::from(e));
        baked_build(&cache, &PipelineKey::new("quantized_qmv", name, v)).expect(name)
    };
    let (plain, normed, adding) = (
        qmv(QmvEnds::default()),
        qmv(ends(c, true, false)),
        qmv(ends(c, false, true)),
    );
    let tag = c.dtype.tag();
    let norm_name: &'static str = Box::leak(format!("rmsnorm_{tag}_s_{tag}_specialized").into_boxed_str());
    let constants = RmsNormConstants {
        bucket_m: BucketM(1),
        q_size: QSize(c.k as u32),
        rms_norm_eps: RmsNormEps(EPS),
        weight_offset: c.offset,
    };
    let norm = baked_build(&cache, &PipelineKey::new("rmsnorm", norm_name, constants.into())).expect(norm_name);
    Some(Rig {
        device,
        _cache: cache,
        plain,
        normed,
        adding,
        norm,
    })
}

/// The residual row `x` and the norm's gain, as stored, and as their values.
fn inputs(c: &Case, rng: &mut Lcg) -> (Vec<u16>, Vec<u16>) {
    let x = (0..c.k).map(|_| c.dtype.bits(4.0 * rng.unit())).collect();
    let gain = (0..c.k)
        .map(|_| c.dtype.bits(1.0 - c.offset + 0.25 * rng.unit()))
        .collect();
    (x, gain)
}

/// `W · rmsnorm(x, gain + offset)`, exact, with each row's error scale ([`Weights::dot`]).
fn exact(c: &Case, w: &Weights, x: &[u16], gain: &[u16]) -> Vec<(f64, f64)> {
    let x: Vec<f64> = x.iter().map(|&b| f64::from(c.dtype.value(b))).collect();
    let ms = x.iter().map(|v| v * v).sum::<f64>() / c.k as f64;
    let inv = 1.0 / (ms + f64::from(EPS)).sqrt();
    let xn: Vec<f64> = (x.iter().zip(gain))
        .map(|(v, &g)| v * inv * (f64::from(c.dtype.value(g)) + f64::from(c.offset)))
        .collect();
    (0..c.n).map(|r| w.dot(c.k, r, &xn)).collect()
}

/// The matvec of case `c` over `x`: `pso` with the gain bound at 15 into a fresh row, or (`norm`)
/// the RMSNorm first and the plain matvec over its row; `y0` is the output row's initial bits.
fn matvec(r: &Rig, c: &Case, w: &Weights, x: &Buffer, gain: &Buffer, y0: &[u16], how: How) -> Vec<u16> {
    let y = shared(&r.device, y0);
    let normed_row = shared(&r.device, &vec![0u16; c.k]);
    let mut batch = Mtl4DispatchBatch::begin(&r.device).expect("mtl4");
    let (grid, threads) = qmv_dispatch_shape(c.kernel(), 1, c.n as u32, 1);
    let qmv = |batch: &mut Mtl4DispatchBatch, pso, input: &Buffer| {
        let binds = [(&w.w, 0), (&w.scales, 1), (&w.biases, 2), (input, 3), (&y, 4), (gain, 15)];
        batch.encode(pso, &binds, &[], &[], &[], size(grid), size(threads));
    };
    match how {
        How::Plain => qmv(&mut batch, &r.plain, x),
        How::Normed => qmv(&mut batch, &r.normed, x),
        How::Adding => qmv(&mut batch, &r.adding, x),
        How::NormThenPlain => {
            let binds = [(&normed_row, 0), (x, 1), (gain, 2)];
            let one = (1, 1, 1);
            batch.encode(&r.norm, &binds, &[], &[], &[], size(one), size((NORM_THREADS, 1, 1)));
            batch.barrier();
            qmv(&mut batch, &r.plain, &normed_row);
        }
    }
    batch.commit(true);
    read_u16(&y, c.n)
}

#[derive(Clone, Copy)]
enum How {
    Plain,
    Normed,
    Adding,
    NormThenPlain,
}

/// Whether every value of `got` is within the bound of its exact `(value, error scale)`: the
/// output's rounding, plus four times what rounding each normed input to the activation dtype can
/// move the dot.
fn within(c: &Case, got: &[u16], exact: &[(f64, f64)]) -> Result<(), String> {
    let ulp = match c.dtype {
        Dtype::Bf16 => 2f64.powi(-8),
        Dtype::F16 => 2f64.powi(-11),
    };
    for (i, (&g, &(e, scale))) in got.iter().zip(exact).enumerate() {
        let v = f64::from(c.dtype.value(g));
        let bound = ulp * e.abs() + 2.0 * ulp * scale;
        if (v - e).abs() > bound {
            return Err(format!("row {i}: {v} vs exact {e} (bound {bound})"));
        }
    }
    Ok(())
}

fn cases() -> Vec<Case> {
    let base = Case {
        dtype: Dtype::Bf16,
        group_size: 64,
        k: 3072,
        n: 1024,
        offset: 0.0,
    };
    vec![
        base,
        Case { offset: 1.0, ..base },
        Case { dtype: Dtype::F16, ..base },
        Case { k: 896, n: 136, ..base },
        Case { k: 128, n: 256, ..base },
    ]
}

#[test]
fn a_normalizing_matvec_is_as_close_to_the_exact_normed_product_as_the_norm_then_matvec() {
    for c in cases() {
        let Some(r) = rig(&c) else { return };
        let mut rng = Lcg(c.k as u64 * 7 + c.n as u64);
        let w = Weights::new(&r.device, &c, &mut rng);
        let (x, gain) = inputs(&c, &mut rng);
        let want = exact(&c, &w, &x, &gain);
        let (xb, gb) = (shared(&r.device, &x), shared(&r.device, &gain));
        let zero = vec![0u16; c.n];
        let run = |how| matvec(&r, &c, &w, &xb, &gb, &zero, how);
        within(&c, &run(How::NormThenPlain), &want)
            .unwrap_or_else(|e| panic!("{c:?}: the unfused norm then matvec: {e}"));
        within(&c, &run(How::Normed), &want)
            .unwrap_or_else(|e| panic!("{c:?}: the normalizing matvec: {e}"));
        assert!(
            within(&c, &run(How::Plain), &want).is_err(),
            "{c:?}: the bound accepts the raw matvec — it cannot see a missing norm"
        );
    }
}

#[test]
fn an_adding_matvec_is_the_plain_matvec_added_into_the_residual() {
    for c in cases() {
        let Some(r) = rig(&c) else { return };
        let mut rng = Lcg(c.k as u64 * 13 + c.n as u64);
        let w = Weights::new(&r.device, &c, &mut rng);
        let (x, gain) = inputs(&c, &mut rng);
        let residual: Vec<u16> = (0..c.n).map(|_| c.dtype.bits(8.0 * rng.unit())).collect();
        let (xb, gb) = (shared(&r.device, &x), shared(&r.device, &gain));
        let plain = matvec(&r, &c, &w, &xb, &gb, &vec![0u16; c.n], How::Plain);
        let added = matvec(&r, &c, &w, &xb, &gb, &residual, How::Adding);
        let ulp = match c.dtype {
            Dtype::Bf16 => 2f32.powi(-8),
            Dtype::F16 => 2f32.powi(-11),
        };
        for (i, ((&p, &res), &got)) in plain.iter().zip(&residual).zip(&added).enumerate() {
            let (p, got) = (c.dtype.value(p), c.dtype.value(got));
            let want = c.dtype.value(res) + p;
            assert!(
                (got - want).abs() <= ulp * (p.abs() + want.abs()),
                "{c:?} row {i}: {got} vs {want}"
            );
        }
        // The rows that cancel least keep the residual's scale: a missing add misses them all.
        assert_ne!(plain, added, "{c:?}: the adding matvec did not add");
    }
}

#[test]
fn a_normalizing_gated_matvec_is_as_close_as_the_norm_then_gated_matvec() {
    let c = cases()[0];
    let Some(r) = rig(&c) else { return };
    let cache = SpecializedPipelineCache::new(r.device.clone(), &[]).expect("shaders");
    let gated = |e: QmvEnds| {
        let mut v: Vec<ConstantValue> = AffineGatedQmvConstants {
            qmv: AffineQmvConstants {
                k: KDimI32(c.k as i32),
                n: NDimI32(c.n as i32),
                codes: AffineCodes::AsWritten,
            },
            act: GatedAct::Silu,
        }
        .into();
        v.extend(Vec::<ConstantValue>::from(e));
        baked_build(&cache, &PipelineKey::new("quantized_qmv", "affine_qmv_gated_fast_bf16_s_bf16_gs_64_b_4", v))
            .expect("gated")
    };
    let (plain, normed) = (gated(QmvEnds::default()), gated(ends(&c, true, false)));
    let mut rng = Lcg(99);
    let (gate, up) = (Weights::new(&r.device, &c, &mut rng), Weights::new(&r.device, &c, &mut rng));
    let (x, gain) = inputs(&c, &mut rng);
    let (g, u) = (exact(&c, &gate, &x, &gain), exact(&c, &up, &x, &gain));
    // `silu(g) · u`, its error scale carried through: |∂/∂g| ≤ 1.1·|u|, |∂/∂u| = |silu(g)|.
    let want: Vec<(f64, f64)> = (g.iter().zip(&u))
        .map(|(&(g, gs), &(u, us))| {
            let silu = g / (1.0 + (-g).exp());
            (silu * u, 1.1 * u.abs() * gs + silu.abs() * us)
        })
        .collect();
    let (xb, gb) = (shared(&r.device, &x), shared(&r.device, &gain));
    let normed_row = shared(&r.device, &vec![0u16; c.k]);
    let run = |pso, fused: bool| {
        let y = shared(&r.device, &vec![0u16; c.n]);
        let mut batch = Mtl4DispatchBatch::begin(&r.device).expect("mtl4");
        let input = if fused { &xb } else { &normed_row };
        if !fused {
            let binds = [(&normed_row, 0), (&xb, 1), (&gb, 2)];
            batch.encode(&r.norm, &binds, &[], &[], &[], size((1, 1, 1)), size((NORM_THREADS, 1, 1)));
            batch.barrier();
        }
        let binds = [
            (&gate.w, 0),
            (&gate.scales, 1),
            (&gate.biases, 2),
            (input, 3),
            (&y, 4),
            (&up.w, 5),
            (&up.scales, 6),
            (&up.biases, 7),
            (&gb, 15),
        ];
        batch.encode(pso, &binds, &[], &[], &[], size((1, c.n as u32 / 8, 1)), size((32, 4, 1)));
        batch.commit(true);
        read_u16(&y, c.n)
    };
    within(&c, &run(&plain, false), &want).expect("the norm then gated matvec");
    within(&c, &run(&normed, true), &want).expect("the normalizing gated matvec");
}
