// SPDX-License-Identifier: Apache-2.0
//! `affine_qmm_small_m_*` (NAX): the decode-batch MLX-affine 4-bit GEMM that
//! feeds the matrix unit the packed 4-bit codes directly — against a host f32
//! reference of `y = x · (s·q + b)ᵀ`, launched as the tape launches it (M
//! baked at the bucket, the grid scaled to the live rows), for every
//! activation / scale dtype and group size the lowering routes to it.
//!
//! GPU tests (NAX hardware only; skipped elsewhere) — run with
//! `--test-threads=1` (standing rule).

use half::{bf16, f16};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize};
use scratchy_target_metal::detect_device;
use scratchy_target_metal::mtl4_dispatch::Mtl4DispatchBatch;
use scratchy_target_metal::quantized::{
    DequantDtype, SMALL_M_TILE_COLS, ScaleDtype, SmallMTile, small_m_kernel_static_name,
};
use scratchy_target_metal::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};
use scratchy_target_metal::targets::is_nax_capable;

type Device = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLDevice>>;
type Buffer = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>>;

#[derive(Clone, Copy, PartialEq, Debug)]
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
    fn act(self) -> DequantDtype {
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
    fn eps(self) -> f32 {
        match self {
            Dtype::F16 => 1.0 / 1024.0,
            Dtype::Bf16 => 1.0 / 128.0,
        }
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

fn shared<T: Copy>(device: &Device, data: &[T]) -> Buffer {
    let bytes = std::mem::size_of_val(data);
    let buf = device
        .newBufferWithLength_options(bytes.max(16), MTLResourceOptions::StorageModeShared)
        .expect("newBuffer");
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8,
            buf.contents().as_ptr() as *mut u8,
            bytes,
        );
    }
    buf
}

/// One GEMM: `y[m, n] = Σ_k x[m, k] · (s[n, g]·q[n, k] + b[n, g])` over the
/// first `m` rows of a `bucket_m`-row activation.
struct Gemm {
    act: Dtype,
    scale: Dtype,
    group_size: usize,
    m: usize,
    bucket_m: usize,
    n: usize,
    k: usize,
    tile: SmallMTile,
}

fn run(device: &Device, cache: &SpecializedPipelineCache, g: &Gemm) {
    let (m, n, k, gs) = (g.m, g.n, g.k, g.group_size);
    let mut rng = Rng(0x5eed ^ (m * 7919 + n * 31 + k) as u64);
    // MLX affine: 8 codes per u32, the low nibble first.
    let codes: Vec<u8> = (0..n * k).map(|_| (rng.next() & 0xf) as u8).collect();
    let packed: Vec<u32> = codes
        .chunks(8)
        .map(|c| {
            c.iter()
                .enumerate()
                .fold(0u32, |w, (i, &q)| w | (q as u32) << (4 * i))
        })
        .collect();
    let groups = n * k / gs;
    let scales: Vec<f32> = (0..groups).map(|_| 0.01 + 0.04 * rng.unit()).collect();
    let biases: Vec<f32> = (0..groups).map(|_| rng.unit() - 0.5).collect();
    let x: Vec<f32> = (0..g.bucket_m * k)
        .map(|_| 2.0 * rng.unit() - 1.0)
        .collect();
    let round = |d: Dtype, v: &[f32]| -> Vec<u16> { v.iter().map(|&e| d.bits(e)).collect() };
    let (scales_h, biases_h, x_h) = (
        round(g.scale, &scales),
        round(g.scale, &biases),
        round(g.act, &x),
    );

    let kg = k / gs;
    let mut want = vec![0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0f64;
            for kk in 0..k {
                let grp = c * kg + kk / gs;
                let w = g.scale.value(scales_h[grp]) * codes[c * k + kk] as f32
                    + g.scale.value(biases_h[grp]);
                acc += (g.act.value(x_h[r * k + kk]) * w) as f64;
            }
            want[r * n + c] = acc as f32;
        }
    }

    let name = small_m_kernel_static_name(g.act.act(), g.scale.scale(), gs as u32, g.tile);
    let consts = vec![
        ConstantValue::int(0, k as i32),
        ConstantValue::int(1, n as i32),
        ConstantValue::int(2, g.bucket_m as i32),
    ];
    let pipeline = cache
        .get_or_build(&PipelineKey::new("quantized_qmm_nax", name, consts))
        .expect("small-M pipeline");
    let (w_buf, s_buf, b_buf, x_buf) = (
        shared(device, &packed),
        shared(device, &scales_h),
        shared(device, &biases_h),
        shared(device, &x_h),
    );
    let y_buf = shared(device, &vec![0u16; g.bucket_m * n]);
    let mut batch = Mtl4DispatchBatch::begin(device).expect("mtl4");
    batch.encode(
        &pipeline,
        &[
            (&w_buf, 0),
            (&s_buf, 1),
            (&b_buf, 2),
            (&x_buf, 3),
            (&y_buf, 4),
        ],
        &[],
        &[],
        &[],
        MTLSize {
            width: n / SMALL_M_TILE_COLS as usize,
            height: m.div_ceil(g.tile.rows() as usize),
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    batch.commit(true).expect("small-M gemm");
    let got: Vec<f32> =
        unsafe { std::slice::from_raw_parts(y_buf.contents().as_ptr() as *const u16, m * n) }
            .iter()
            .map(|&b| g.act.value(b))
            .collect();

    // f32 accumulation over K products of magnitude ≲ 1, then one rounding
    // to the activation dtype.
    let peak = want.iter().fold(0f32, |a, v| a.max(v.abs()));
    let (worst, at) = got
        .iter()
        .zip(&want)
        .enumerate()
        .map(|(i, (a, b))| ((a - b).abs(), i))
        .fold((0f32, 0), |w, e| if e.0 > w.0 { e } else { w });
    let tol = peak * g.act.eps() + 1e-3 * peak;
    assert!(
        worst <= tol,
        "{name} m={m} n={n} k={k}: |err| {worst} at {at} (got {}, want {}) > {tol}",
        got[at],
        want[at]
    );
}

fn with_nax(body: impl FnOnce(&Device, &SpecializedPipelineCache)) {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    if !is_nax_capable(di.profile.generation) {
        eprintln!("skipping: no NAX matrix unit");
        return;
    }
    let cache =
        SpecializedPipelineCache::with_standard_shaders(di.device.clone()).expect("shaders");
    body(&di.device, &cache);
}

/// The 8-row tile in the 8-token bucket, for every dtype pairing and group
/// size, at the batch sizes it serves.
#[test]
fn small_m_eight_row_tile_across_dtypes_and_group_sizes() {
    with_nax(|device, cache| {
        for (act, scale) in [
            (Dtype::Bf16, Dtype::Bf16),
            (Dtype::Bf16, Dtype::F16),
            (Dtype::F16, Dtype::F16),
            (Dtype::F16, Dtype::Bf16),
        ] {
            for group_size in [32, 64, 128] {
                for m in [5, 8] {
                    run(
                        device,
                        cache,
                        &Gemm {
                            act,
                            scale,
                            group_size,
                            m,
                            bucket_m: 8,
                            n: 256,
                            k: 1024,
                            tile: SmallMTile::Rows8,
                        },
                    );
                }
            }
        }
    });
}

/// The 16-row tile in the 64-token bucket: a partial tile and a full one
/// (the routed range), and two tiles.
#[test]
fn small_m_sixteen_row_tile() {
    with_nax(|device, cache| {
        for m in [9, 13, 16, 32] {
            run(
                device,
                cache,
                &Gemm {
                    act: Dtype::Bf16,
                    scale: Dtype::Bf16,
                    group_size: 64,
                    m,
                    bucket_m: 64,
                    n: 512,
                    k: 3072,
                    tile: SmallMTile::Rows16,
                },
            );
        }
    });
}
