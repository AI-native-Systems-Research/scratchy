// SPDX-License-Identifier: Apache-2.0
//
// Parity test for `splitk_reduce_sum_<dtype>` (kernel ≡ CPU reference
// within bf16 cast noise). The kernel is the downstream half of the
// `affine_qmm_t_splitk` path: `qmm_t_splitk` writes a [split_k, M, N]
// intermediate, and this kernel sums along axis 0 to produce [M, N].
// The CPU reference here matches the in-kernel float accumulator, so
// the only loss is the per-output bf16 cast at the end.
//
// Dispatches on the production MTL4 path (see
// `common::dispatch_threadgroups`).
//
// Two tests:
//   - bf16, exercises the typical Llama-1B prefill split_k value
//   - f16, exercises the alternate dtype branch
//
// Test shapes mirror the qmm_t_splitk shape that fires for Llama-1B
// q_proj at M=64 N=2048 K=2048 gs=64 → split_k=4. We pick a smaller
// (M, N) to keep the test allocation modest while still exercising
// every threadgroup boundary.

mod common;

use objc2_metal::{MTLComputePipelineState, MTLSize};
use scratchy_target_metal::device::detect_device;
use scratchy_target_metal::quantized::{DequantDtype, splitk_reduce_kernel_static_name};
use scratchy_target_metal::shader_cache::ShaderCache;
use scratchy_target_metal::specialized_pipeline_cache::ConstantValue;

/// SplitMix64 — same deterministic PRNG used by the qmv/qmm tests.
struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_unit_f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

/// `splitk_reduce_sum_<dtype>` binding contract (replicated from
/// `MetalSplitKReduce::execute`): M/N/split_k are baked into the
/// pipeline as function constants 0/1/2; the only address bindings are
/// `output(0)` and `intermediate(1)`. The dispatch is 1D over M*N
/// output elements, `min(M*N, maxTotalThreadsPerThreadgroup)` threads
/// per group.
fn run_splitk_reduce_mtl4(
    device: &common::Device,
    intermediate: &common::Buffer,
    output: &common::Buffer,
    m: u32,
    n: u32,
    split_k: u32,
    dtype: DequantDtype,
) -> bool {
    let cache = ShaderCache::new(device.clone()).expect("MetalSplitKReduce shader cache");
    let constants = [
        ConstantValue::uint(0, m),
        ConstantValue::uint(1, n),
        ConstantValue::uint(2, split_k),
    ];
    let kernel_name = splitk_reduce_kernel_static_name(dtype);
    let pso = cache
        .get_pipeline_specialized(kernel_name, &constants)
        .expect("splitk_reduce pipeline");

    let nthreads = m as u64 * n as u64;
    let max_tpg = pso.maxTotalThreadsPerThreadgroup() as u64;
    let threads_per_threadgroup_x = nthreads.min(max_tpg);
    let threads_per_threadgroup = MTLSize {
        width: threads_per_threadgroup_x as usize,
        height: 1,
        depth: 1,
    };
    let threadgroups = MTLSize {
        width: (nthreads as usize).div_ceil(threads_per_threadgroup_x as usize),
        height: 1,
        depth: 1,
    };
    common::dispatch_threadgroups(
        device,
        &pso,
        &[output, intermediate],
        threadgroups,
        threads_per_threadgroup,
    )
}

#[test]
fn splitk_reduce_sum_bf16_matches_cpu_reference() {
    // Llama-1B q_proj prefill shape: split_k=4. Use a smaller (M, N) so
    // the test allocation is modest; the kernel is one-thread-per-output
    // so coverage is a function of M*N total, not per-shape behavior.
    let split_k: u32 = 4;
    let m: u32 = 64;
    let n: u32 = 256;

    // Per-element magnitudes match what `qmm_t_impl_inline` would have
    // accumulated into the splitk intermediate: roughly K-sum / split_k
    // worth of bf16-range float values. Random uniform in [-2, 2] per
    // partition keeps the per-output sum around magnitude 0 with std
    // ~sqrt(split_k)*1; well within bf16's representable range.
    let mut rng = SplitMix64(0xC0FFEE);
    let n_in = (split_k as usize) * (m as usize) * (n as usize);
    let intermediate: Vec<half::bf16> = (0..n_in)
        .map(|_| half::bf16::from_f32(4.0 * rng.next_unit_f32() - 2.0))
        .collect();

    // Compute the reference: sum-along-axis-0 in float, cast to bf16
    // at the end. This matches the kernel's float accumulator + final
    // T cast at the assignment site.
    let mut expected = vec![half::bf16::ZERO; (m * n) as usize];
    for k in 0..split_k as usize {
        for i in 0..m as usize {
            for j in 0..n as usize {
                let idx_in = k * (m as usize) * (n as usize) + i * (n as usize) + j;
                let idx_out = i * (n as usize) + j;
                let cur = expected[idx_out].to_f32();
                expected[idx_out] = half::bf16::from_f32(cur + intermediate[idx_in].to_f32());
            }
        }
    }

    // GPU dispatch.
    let Some(__dev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let device = __dev.device;

    let in_buf = common::shared_slice(&device, &intermediate);
    let out_buf = common::shared_zeroed(
        &device,
        (m as usize) * (n as usize) * std::mem::size_of::<half::bf16>(),
    );

    if !run_splitk_reduce_mtl4(
        &device,
        &in_buf,
        &out_buf,
        m,
        n,
        split_k,
        DequantDtype::Bf16,
    ) {
        return;
    }

    let actual: Vec<half::bf16> = common::read_slice(&out_buf, (m as usize) * (n as usize));

    // Tolerance: we sum split_k=4 bf16-cast partial values; each cast
    // is ~bf16_eps * |sum_so_far|. Final bf16 cast is ~bf16_eps * |out|.
    // The order of summation in CPU vs GPU can differ (CPU does sequential
    // bf16 cast per partition; GPU keeps a single float accumulator over
    // all split_k), so the noise budget is bounded by `split_k * bf16_eps
    // * max_partial_magnitude`. Empirically that's well under 4 * 0.01.
    let bf16_eps: f32 = 1.0 / 128.0;
    let mut worst: (usize, f32, f32, f32) = (0, 0.0, 0.0, 0.0);
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let af = a.to_f32();
        let ef = e.to_f32();
        let err = (af - ef).abs();
        if err > worst.3 {
            worst = (i, af, ef, err);
        }
    }
    let allowed = bf16_eps * (split_k as f32) * 4.0;
    assert!(
        worst.3 <= allowed,
        "splitk_reduce_sum bf16 mismatch at idx {}: actual={} expected={} err={} > allowed={}",
        worst.0,
        worst.1,
        worst.2,
        worst.3,
        allowed,
    );
}

#[test]
fn splitk_reduce_sum_f16_matches_cpu_reference() {
    // Same shape as bf16 — exercises the f16 instantiation. f16 has
    // narrower dynamic range than bf16 but our values stay in
    // [-(2*split_k), 2*split_k] which is well within f16 range.
    let split_k: u32 = 4;
    let m: u32 = 64;
    let n: u32 = 256;

    let mut rng = SplitMix64(0xBADCAFE);
    let n_in = (split_k as usize) * (m as usize) * (n as usize);
    let intermediate: Vec<half::f16> = (0..n_in)
        .map(|_| half::f16::from_f32(4.0 * rng.next_unit_f32() - 2.0))
        .collect();

    let mut expected = vec![half::f16::ZERO; (m * n) as usize];
    for k in 0..split_k as usize {
        for i in 0..m as usize {
            for j in 0..n as usize {
                let idx_in = k * (m as usize) * (n as usize) + i * (n as usize) + j;
                let idx_out = i * (n as usize) + j;
                let cur = expected[idx_out].to_f32();
                expected[idx_out] = half::f16::from_f32(cur + intermediate[idx_in].to_f32());
            }
        }
    }

    let Some(__dev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let device = __dev.device;

    let in_buf = common::shared_slice(&device, &intermediate);
    let out_buf = common::shared_zeroed(
        &device,
        (m as usize) * (n as usize) * std::mem::size_of::<half::f16>(),
    );

    if !run_splitk_reduce_mtl4(&device, &in_buf, &out_buf, m, n, split_k, DequantDtype::F16) {
        return;
    }

    let actual: Vec<half::f16> = common::read_slice(&out_buf, (m as usize) * (n as usize));

    // f16 eps = 1/1024 (vs bf16_eps = 1/128); but the per-partition
    // CPU cast still introduces bf16-magnitude rounding so the budget
    // structure is the same: split_k partitions * one cast each.
    let f16_eps: f32 = 1.0 / 1024.0;
    let mut worst: (usize, f32, f32, f32) = (0, 0.0, 0.0, 0.0);
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let af = a.to_f32();
        let ef = e.to_f32();
        let err = (af - ef).abs();
        if err > worst.3 {
            worst = (i, af, ef, err);
        }
    }
    // f16 tolerance: split_k partitions × per-cast eps × max-magnitude.
    // Max per-output magnitude ≤ split_k * 2 = 8.
    let allowed = f16_eps * (split_k as f32) * 8.0;
    assert!(
        worst.3 <= allowed,
        "splitk_reduce_sum f16 mismatch at idx {}: actual={} expected={} err={} > allowed={}",
        worst.0,
        worst.1,
        worst.2,
        worst.3,
        allowed,
    );
}
