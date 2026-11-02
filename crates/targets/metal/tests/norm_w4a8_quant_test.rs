// SPDX-License-Identifier: Apache-2.0
//! The quantizing norms (`rmsnorm_w4a8q_*`, `fused_add_rmsnorm_w4a8q_*`) against the pair they
//! replace — the plain norm, then the W4A8 pre-pass (`affine_w4a8_quant_*`) over its output —
//! byte for byte: the normed rows (and FusedAddRmsNorm's residual sum), the int8 rows and the
//! float2 chunk scales, for every activation / gain dtype pair, at Qwen3.8-27B's hidden 5120 and a
//! row narrower than the threadgroup, with the scratch's rows wider than the step's (the 8-row
//! bucket's 16-row layout).
//!
//! GPU tests (Metal 4 only; skipped elsewhere) — run with `--test-threads=1` (standing rule).

use half::{bf16, f16};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize};
use scratchy_target_metal::aot::baked_build;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::mtl4_dispatch::Mtl4DispatchBatch;
use scratchy_target_metal::quantized::{
    DequantDtype, W4a8Rows, w4a8_quant_static_name, w4a8_scratch_bytes,
};
use scratchy_target_metal::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};
use scratchy_target_metal::tape::ids::{BucketM, QSize, RmsNormEps};
use scratchy_target_metal::tape::kernel_constants::RmsNormConstants;

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
    fn tag(self) -> &'static str {
        match self {
            Dtype::F16 => "f16",
            Dtype::Bf16 => "bf16",
        }
    }
    fn act(self) -> DequantDtype {
        match self {
            Dtype::F16 => DequantDtype::F16,
            Dtype::Bf16 => DequantDtype::Bf16,
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

fn bytes(buf: &Buffer, len: usize) -> Vec<u8> {
    unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const u8, len) }.to_vec()
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Norm {
    RmsNorm,
    FusedAdd,
}

/// One case's rows: `m` live rows of a `bucket`-row norm over `k`-wide rows, the scratch laid out
/// for `scratch_rows`.
#[derive(Clone, Copy)]
struct Rows {
    k: usize,
    m: usize,
    bucket: usize,
    scratch_rows: usize,
}

/// One case: `rows` of `norm` at the `(act, scale)` dtypes.
fn case(
    device: &Device,
    cache: &SpecializedPipelineCache,
    norm: Norm,
    (act, scale): (Dtype, Dtype),
    rows: Rows,
) {
    let Rows {
        k,
        m,
        bucket,
        scratch_rows,
    } = rows;
    let mut rng = Rng(0x5eed ^ (k * 131 + m * 7 + bucket) as u64);
    // Rows of varied magnitude, a zero chunk (amax 0) in row 0.
    let x: Vec<u16> = (0..bucket * k)
        .map(|i| {
            if i < 64 {
                return act.bits(0.0);
            }
            let row = (i / k) as f32;
            act.bits((2.0 * rng.unit() - 1.0) * (0.5 + row))
        })
        .collect();
    let r: Vec<u16> = (0..bucket * k)
        .map(|i| {
            if i < 64 {
                return act.bits(0.0);
            }
            act.bits(2.0 * rng.unit() - 1.0)
        })
        .collect();
    let gain: Vec<u16> = (0..k).map(|_| scale.bits(0.5 + rng.unit())).collect();

    let lib = match norm {
        Norm::RmsNorm => "rmsnorm",
        Norm::FusedAdd => "fused_add_rmsnorm",
    };
    let suffix = format!("{}_s_{}_specialized", act.tag(), scale.tag());
    let plain_name = format!("{lib}_{suffix}");
    let fused_name = format!("{lib}_w4a8q_{suffix}");
    let leak = |s: String| -> &'static str { Box::leak(s.into_boxed_str()) };
    let norm_consts: Vec<ConstantValue> = RmsNormConstants {
        bucket_m: BucketM(bucket as u32),
        q_size: QSize(k as u32),
        rms_norm_eps: RmsNormEps(1e-6),
        weight_offset: 1.0,
    }
    .into();
    let threads = scratchy_target_metal::tape::kernel_constants::norm_threads(QSize(k as u32));
    let mut fused_consts = norm_consts.clone();
    fused_consts.push(ConstantValue::uint(6, scratch_rows as u32));
    let pipeline = |name: &'static str, consts: Vec<ConstantValue>| {
        baked_build(cache, &PipelineKey::new(lib, name, consts)).expect("pipeline")
    };
    let plain = pipeline(leak(plain_name.clone()), norm_consts);
    let fused = pipeline(leak(fused_name.clone()), fused_consts);
    let quant = baked_build(
        cache,
        &PipelineKey::new(
            "quantized_qmm_nax",
            w4a8_quant_static_name(W4a8Rows::Dense, act.act()),
            vec![
                ConstantValue::int(0, k as i32),
                ConstantValue::int(2, scratch_rows as i32),
            ],
        ),
    )
    .expect("quant pipeline");

    let size = |width: usize, height: usize| MTLSize {
        width,
        height,
        depth: 1,
    };
    let scratch_len = w4a8_scratch_bytes(scratch_rows as u32, k as u32) as usize;
    let act_len = bucket * k * 2;
    let gain_buf = shared(device, &gain);
    // Each side: (out/delta, in/residual, scratch), the scratch filled with garbage.
    let run = |fused_kernel: bool| -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let a = shared(device, &x);
        let b = shared(device, &r);
        let scratch = shared(device, &vec![0xa5u8; scratch_len]);
        let mut batch = Mtl4DispatchBatch::begin(device).expect("mtl4");
        // RmsNorm: out @0 (a), in @1 (b). FusedAdd: residual @0 (b), delta @1 (a).
        let (b0, b1) = match norm {
            Norm::RmsNorm => (&a, &b),
            Norm::FusedAdd => (&b, &a),
        };
        if fused_kernel {
            batch.encode(
                &fused,
                &[(b0, 0), (b1, 1), (&gain_buf, 2), (&scratch, 3)],
                &[],
                &[],
                &[],
                size(m, 1),
                size(threads as usize, 1),
            );
        } else {
            batch.encode(
                &plain,
                &[(b0, 0), (b1, 1), (&gain_buf, 2)],
                &[],
                &[],
                &[],
                size(m, 1),
                size(threads as usize, 1),
            );
            batch.barrier();
            // The pre-pass reads the normed rows: RmsNorm's output, FusedAdd's delta (both `a`).
            batch.encode(
                &quant,
                &[(&a, 0), (&scratch, 1)],
                &[],
                &[],
                &[],
                size((k / 64).div_ceil(16), m),
                size(128, 1),
            );
        }
        batch.commit(true);
        (
            bytes(&a, act_len),
            bytes(&b, act_len),
            bytes(&scratch, scratch_len),
        )
    };
    let want = run(false);
    let got = run(true);
    let what = format!("{fused_name} k={k} m={m} bucket={bucket} scratch rows={scratch_rows}");
    let first_diff = |a: &[u8], b: &[u8]| a.iter().zip(b).position(|(x, y)| x != y);
    assert_eq!(
        first_diff(&got.0, &want.0),
        None,
        "{what}: normed rows differ"
    );
    assert_eq!(
        first_diff(&got.1, &want.1),
        None,
        "{what}: input / residual rows differ"
    );
    let xq = scratch_rows * k;
    assert_eq!(
        first_diff(&got.2[..xq], &want.2[..xq]),
        None,
        "{what}: int8 rows differ"
    );
    assert_eq!(
        first_diff(&got.2[xq..], &want.2[xq..]),
        None,
        "{what}: chunk scales differ"
    );
    // The test reads something: the quantized rows are not the garbage fill.
    assert!(
        got.2[64..k].iter().any(|&b| b != 0xa5),
        "{what}: no int8 row"
    );
}

#[test]
fn quantizing_norms_match_the_norm_then_the_pre_pass() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let cache = SpecializedPipelineCache::new(di.device.clone(), &[]).expect("shaders");
    let device = &di.device;
    let dtypes = [
        (Dtype::Bf16, Dtype::Bf16),
        (Dtype::Bf16, Dtype::F16),
        (Dtype::F16, Dtype::F16),
        (Dtype::F16, Dtype::Bf16),
    ];
    for norm in [Norm::RmsNorm, Norm::FusedAdd] {
        for d in dtypes {
            // (k, live rows, bucket, scratch rows): the 64-row and 16-row buckets at hidden 5120,
            // the 8-row bucket's 16-row scratch, a 576-wide row (fewer threads than chunks x 8).
            for (k, m, bucket, rows) in [
                (5120, 16, 64, 64),
                (5120, 9, 16, 16),
                (5120, 5, 8, 16),
                (576, 3, 32, 32),
            ] {
                let rows = Rows {
                    k,
                    m,
                    bucket,
                    scratch_rows: rows,
                };
                case(device, &cache, norm, d, rows);
            }
        }
    }
}
