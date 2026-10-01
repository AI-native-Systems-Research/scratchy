// SPDX-License-Identifier: Apache-2.0
//! The NAX MLX-affine 4-bit GEMMs that feed the matrix unit the offset-8
//! codes directly — `affine_qmm_small_m_*` (decode batches) and the
//! `affine_w4a8_quant_*` + `affine_qmm_w4a8_*` pair (int8 activations) —
//! against a host f32 reference of `y = x · (s·q + b)ᵀ`, launched as the
//! tape launches them (M baked at the bucket, the grid scaled to the live
//! rows), for every activation / scale dtype and group size the lowering
//! routes to them.
//!
//! GPU tests (NAX hardware only; skipped elsewhere) — run with
//! `--test-threads=1` (standing rule).

use half::{bf16, f16};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize};
use scratchy_target_metal::detect_device;
use scratchy_target_metal::mtl4_dispatch::Mtl4DispatchBatch;
use scratchy_target_metal::quantized::{
    DequantDtype, SMALL_M_TILE_COLS, ScaleDtype, SmallMTile, W4A8_TILE_ROWS, W4a8Rows, W4a8Tile,
    qmm_w4a8_static_name, small_m_kernel_static_name, w4a8_quant_static_name, w4a8_scratch_bytes,
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
    kernel: Kernel,
}

#[derive(Clone, Copy)]
enum Kernel {
    SmallM(SmallMTile),
    W4a8(W4a8Tile),
}

/// The activation the W4A8 pre-pass hands the GEMM: each (row, 64-chunk)
/// rounded to int8 at its own scale `amax / 127`, as the kernel computes it.
fn int8_chunks(x: &[f32]) -> Vec<f32> {
    x.chunks(64)
        .flat_map(|c| {
            let amax = c.iter().fold(0f32, |a, v| a.max(v.abs()));
            let (inv, a) = if amax > 0.0 {
                (127.0 / amax, amax / 127.0)
            } else {
                (0.0, 0.0)
            };
            c.iter().map(move |v| (v * inv).round_ties_even() * a)
        })
        .collect()
}

fn run(device: &Device, cache: &SpecializedPipelineCache, g: &Gemm) {
    let (m, n, k, gs) = (g.m, g.n, g.k, g.group_size);
    let mut rng = Rng(0x5eed ^ (m * 7919 + n * 31 + k) as u64);
    // MLX affine: 8 codes per u32, the low nibble first, stored offset-8
    // (q XOR 8, the signed q - 8) as the matrix unit reads them.
    let codes: Vec<u8> = (0..n * k).map(|_| (rng.next() & 0xf) as u8).collect();
    let packed: Vec<u32> = codes
        .chunks(8)
        .map(|c| {
            c.iter()
                .enumerate()
                .fold(0u32, |w, (i, &q)| w | ((q ^ 8) as u32) << (4 * i))
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

    let x_used: Vec<f32> = x_h.iter().map(|&b| g.act.value(b)).collect();
    let x_used = match g.kernel {
        Kernel::SmallM(_) => x_used,
        Kernel::W4a8(_) => int8_chunks(&x_used),
    };
    let kg = k / gs;
    let mut want = vec![0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0f64;
            for kk in 0..k {
                let grp = c * kg + kk / gs;
                let w = g.scale.value(scales_h[grp]) * codes[c * k + kk] as f32
                    + g.scale.value(biases_h[grp]);
                acc += (x_used[r * k + kk] * w) as f64;
            }
            want[r * n + c] = acc as f32;
        }
    }

    let consts = vec![
        ConstantValue::int(0, k as i32),
        ConstantValue::int(1, n as i32),
        ConstantValue::int(2, g.bucket_m as i32),
    ];
    let pipeline = |name: &'static str| {
        cache
            .get_or_build(&PipelineKey::new("quantized_qmm_nax", name, consts.clone()))
            .expect("pipeline")
    };
    let size = |width: usize, height: usize| MTLSize {
        width,
        height,
        depth: 1,
    };
    let (w_buf, s_buf, b_buf, x_buf) = (
        shared(device, &packed),
        shared(device, &scales_h),
        shared(device, &biases_h),
        shared(device, &x_h),
    );
    let y_buf = shared(device, &vec![0u16; g.bucket_m * n]);
    let mut batch = Mtl4DispatchBatch::begin(device).expect("mtl4");
    let name = match g.kernel {
        Kernel::SmallM(tile) => {
            let name = small_m_kernel_static_name(g.act.act(), g.scale.scale(), gs as u32, tile);
            batch.encode(
                &pipeline(name),
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
                size(
                    n / SMALL_M_TILE_COLS as usize,
                    m.div_ceil(tile.rows() as usize),
                ),
                size(
                    32 * scratchy_target_metal::quantized::SMALL_M_SIMDGROUPS as usize,
                    1,
                ),
            );
            name
        }
        Kernel::W4a8(tile) => {
            let scratch = shared(
                device,
                &vec![0u8; w4a8_scratch_bytes(g.bucket_m as u32, k as u32) as usize],
            );
            batch.encode(
                &pipeline(w4a8_quant_static_name(W4a8Rows::Dense, g.act.act())),
                &[(&x_buf, 0), (&scratch, 1)],
                &[],
                &[],
                &[],
                size((k / 64).div_ceil(16), m),
                size(128, 1),
            );
            batch.barrier();
            let name = qmm_w4a8_static_name(
                W4a8Rows::Dense,
                g.act.act(),
                g.scale.scale(),
                gs as u32,
                tile,
            );
            batch.encode(
                &pipeline(name),
                &[
                    (&w_buf, 0),
                    (&s_buf, 1),
                    (&b_buf, 2),
                    (&scratch, 3),
                    (&y_buf, 4),
                ],
                &[],
                &[],
                &[],
                size(
                    n / tile.cols() as usize,
                    m.div_ceil(W4A8_TILE_ROWS as usize),
                ),
                size(32 * tile.simdgroups() as usize, 1),
            );
            name
        }
    };
    batch.commit();
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
                for m in [4, 8] {
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
                            kernel: Kernel::SmallM(SmallMTile::Rows8),
                        },
                    );
                }
            }
        }
    });
}

/// The 16-row tile in the 64-token bucket: a partial tile and a full one
/// (the routed range), and two tiles. K = 896 (Qwen2.5-0.5B's hidden size)
/// is 14 groups of 64, which the tile's simdgroups split unevenly.
#[test]
fn small_m_sixteen_row_tile() {
    with_nax(|device, cache| {
        for (m, k) in [(9, 3072), (13, 3072), (16, 3072), (32, 3072), (13, 896)] {
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
                    k,
                    kernel: Kernel::SmallM(SmallMTile::Rows16),
                },
            );
        }
    });
}

/// W4A8 in the 64-token bucket, for every dtype pairing, both group sizes
/// and both tiles: a partial row tile, a full bucket, and a live batch
/// short of the bucket.
#[test]
fn w4a8_across_dtypes_group_sizes_and_tiles() {
    with_nax(|device, cache| {
        for (act, scale) in [
            (Dtype::Bf16, Dtype::Bf16),
            (Dtype::Bf16, Dtype::F16),
            (Dtype::F16, Dtype::F16),
            (Dtype::F16, Dtype::Bf16),
        ] {
            for group_size in [64, 128] {
                for (n, tile) in [(384, W4a8Tile::Cols128), (192, W4a8Tile::Cols64)] {
                    for m in [13, 45, 64] {
                        run(
                            device,
                            cache,
                            &Gemm {
                                act,
                                scale,
                                group_size,
                                m,
                                bucket_m: 64,
                                n,
                                k: 1024,
                                kernel: Kernel::W4a8(tile),
                            },
                        );
                    }
                }
            }
        }
    });
}
