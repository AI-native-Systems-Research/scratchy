// SPDX-License-Identifier: Apache-2.0
//
// `affine_qmv_*_<dtype>_gs_<gs>_b_4_*` parity test: kernel ≡ CPU reference.
//
// Mirrors the slow-reference math: dequantize the packed int4 weight per
// MLX `affine_dequantize` (`quantized.h:2536` — also covered by the
// pre-existing `quantized_dequantize_test`), then do a row-major matmul
// with f32 accumulator and bf16/f16 cast on the way out. That reference
// matches the qmv kernel's algebra: `result = scale * sum(x * nibble) +
// sum_of_x * bias`, which simplifies to `sum(x * (scale * nibble + bias))`.
//
// Three tests, one per dispatched kernel:
//   - qmv_quad:    K∈{64, 128} ∧ pow2 bits → tested with K=128 (Llama-style head_dim)
//   - qmv_fast:    N%8==0 ∧ K%512==0       → tested with N=64, K=512
//   - qmv generic: neither                  → tested with N=12, K=384

mod common;

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize};
use scratchy_target_metal::cpu_reference::affine_qmv_b4_bf16 as cpu_qmv_bf16;
use scratchy_target_metal::device::detect_device;
use scratchy_target_metal::quantized::{
    DequantDtype, QmvKernel, ScaleDtype, pick_qmv_kernel, qmv_dispatch_shape, qmv_kernel_name,
};
use scratchy_target_metal::shader_cache::ShaderCache;
use scratchy_target_metal::specialized_pipeline_cache::ConstantValue;

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Device = Retained<ProtocolObject<dyn MTLDevice>>;

/// Build the qmv pipeline for `(m, n, k, group_size, scale_dtype)` and
/// dispatch it on the production MTL4 path. Replicates the bindings of
/// `MetalAffineQmv::execute_with_kernel` (buffer 0=packed_w, 1=scales,
/// 2=biases, 3=x, 4=y; K/N ride as function constants 0/1, not buffers)
/// and the `qmv_dispatch_shape` grid. Returns `false` if the host has no
/// MTL4 queue. bits is fixed at 4 and B at 1 (decode-only), matching the
/// original test's `execute(.., 1 /* B */, group_size, 4, ..)` call.
#[allow(clippy::too_many_arguments)]
fn dispatch_qmv(
    device: &Device,
    packed_buf: &Buffer,
    scales_buf: &Buffer,
    biases_buf: &Buffer,
    x_buf: &Buffer,
    y_buf: &Buffer,
    m: usize,
    n: usize,
    k: usize,
    group_size: u32,
    scale_dtype: ScaleDtype,
) -> bool {
    let kernel = pick_qmv_kernel(n as u32, k as u32, 4);
    let kernel_name = qmv_kernel_name(
        kernel,
        DequantDtype::Bf16,
        scale_dtype,
        group_size,
        4,
        false,
    );
    let constants = [
        ConstantValue::int(0, k as i32),
        ConstantValue::int(1, n as i32),
    ];
    let shader_cache = ShaderCache::new(device.clone()).expect("ShaderCache");
    let pipeline = shader_cache
        .get_pipeline_specialized(&kernel_name, &constants)
        .expect("qmv pipeline");
    let (tg, tpg) = qmv_dispatch_shape(kernel, m as u32, n as u32, 1);
    common::dispatch_threadgroups(
        device,
        &pipeline,
        &[packed_buf, scales_buf, biases_buf, x_buf, y_buf],
        MTLSize {
            width: tg.0 as usize,
            height: tg.1 as usize,
            depth: tg.2 as usize,
        },
        MTLSize {
            width: tpg.0 as usize,
            height: tpg.1 as usize,
            depth: tpg.2 as usize,
        },
    )
}

// ─────────────────────────────────────────────────────────────────
// Test helpers
// ─────────────────────────────────────────────────────────────────

fn buffer_from_bytes(device: &Device, bytes: &[u8]) -> Buffer {
    unsafe {
        device
            .newBufferWithBytes_length_options(
                NonNull::new(bytes.as_ptr() as *mut c_void).unwrap(),
                bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
            .expect("newBufferWithBytes returned nil")
    }
}

fn zeroed_buffer(device: &Device, n_bytes: usize) -> Buffer {
    device
        .newBufferWithLength_options(n_bytes, MTLResourceOptions::StorageModeShared)
        .expect("newBufferWithLength returned nil")
}

fn read_buffer_bf16(buf: &Buffer, n_elements: usize) -> Vec<half::bf16> {
    let ptr = buf.contents().as_ptr() as *const half::bf16;
    unsafe { std::slice::from_raw_parts(ptr, n_elements) }.to_vec()
}

/// SplitMix64 — deterministic per-shape PRNG so failures reproduce.
struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_byte(&mut self) -> u8 {
        (self.next() & 0xff) as u8
    }
    fn next_unit_f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

/// Generate `(packed, scales, biases, x)` for a `(N, K)` weight + `(M, K)`
/// activation in bf16. Random nibbles, scales in [0.01, 0.05] (small to
/// keep the dequantized weights close to MLX's typical scale range), and
/// biases in [-0.5, 0.5].
fn make_inputs_bf16(
    seed: u64,
    n: usize,
    k: usize,
    m: usize,
    group_size: usize,
) -> (Vec<u8>, Vec<half::f16>, Vec<half::f16>, Vec<half::bf16>) {
    assert_eq!(k % group_size, 0);
    let n_bytes = n * k / 2; // pack_factor = 2 nibbles/byte
    let n_groups = n * k / group_size;

    let mut rng = SplitMix64(seed);
    let packed: Vec<u8> = (0..n_bytes).map(|_| rng.next_byte()).collect();
    // P10b: scales/biases ship F16 on disk (`T_scale = half`); the kernel
    // casts to T_act in-register.
    let scales: Vec<half::f16> = (0..n_groups)
        .map(|_| half::f16::from_f32(0.01 + 0.04 * rng.next_unit_f32()))
        .collect();
    let biases: Vec<half::f16> = (0..n_groups)
        .map(|_| half::f16::from_f32(rng.next_unit_f32() - 0.5))
        .collect();
    // Activations in [-1, 1) — typical residual-stream magnitude.
    let x: Vec<half::bf16> = (0..(m * k))
        .map(|_| half::bf16::from_f32(2.0 * rng.next_unit_f32() - 1.0))
        .collect();

    (packed, scales, biases, x)
}

#[allow(clippy::too_many_arguments)]
fn run_qmv_bf16(
    packed: &[u8],
    scales: &[half::f16],
    biases: &[half::f16],
    x: &[half::bf16],
    m: usize,
    n: usize,
    k: usize,
    group_size: u32,
) -> Option<Vec<half::bf16>> {
    let device = detect_device()?.device;

    let packed_buf = buffer_from_bytes(&device, packed);

    let scales_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(scales.as_ptr() as *const u8, std::mem::size_of_val(scales))
    };
    let biases_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(biases.as_ptr() as *const u8, std::mem::size_of_val(biases))
    };
    let x_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, std::mem::size_of_val(x)) };
    let scales_buf = buffer_from_bytes(&device, scales_bytes);
    let biases_buf = buffer_from_bytes(&device, biases_bytes);
    let x_buf = buffer_from_bytes(&device, x_bytes);

    let n_out = m * n;
    let y_buf = zeroed_buffer(&device, n_out * std::mem::size_of::<half::bf16>());

    if !dispatch_qmv(
        &device,
        &packed_buf,
        &scales_buf,
        &biases_buf,
        &x_buf,
        &y_buf,
        m,
        n,
        k,
        group_size,
        ScaleDtype::F16,
    ) {
        return Some(Vec::new());
    }

    Some(read_buffer_bf16(&y_buf, n_out))
}

/// Worst per-element noise in f32 space. We compare against the expected
/// bf16 accumulation noise floor: with `K` terms summed in f32 and cast
/// to bf16 at the end, the per-output noise std is bounded by
/// `sqrt(K) * eps_bf16 * max(|x|) * max(|w|)` plus a single-step bf16
/// rounding (~`|y| * 2^-7`). The test bounds the per-element absolute
/// error against that noise floor with a 4× safety factor — tighter
/// would catch a real bug, looser would miss systematic drift.
///
/// Returns `(index, metal_value, cpu_value, abs_err, allowed)` for the
/// worst element so failure messages point at the actual mismatch.
// `per_elem_magnitude` is a bound on `|x| * |w_dequant|` per element.
// With x ∈ [-1, 1) and w_dequant magnitude ≲ 0.5 (scale ≲ 0.05,
// |nibble * scale + bias| ≲ 1.25), per-element product magnitude ≲ ~0.5.
fn worst_abs_error_vs_noise_floor(
    metal: &[half::bf16],
    expected: &[half::bf16],
    k_dim: usize,
    per_elem_magnitude: f32,
) -> (usize, f32, f32, f32, f32) {
    // bf16 mantissa eps ≈ 2^-7 ≈ 7.8e-3 (relative). Per-element rounding
    // bounded by half-eps * magnitude. Sum-of-K-terms noise std grows by
    // sqrt(K). Final bf16 cast adds another ~|y| * 2^-7.
    let bf16_eps: f32 = 1.0 / 128.0;
    let sum_noise_std = (k_dim as f32).sqrt() * 0.5 * bf16_eps * per_elem_magnitude;
    let safety = 4.0;
    let mut worst = (0usize, 0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    for (i, (m, e)) in metal.iter().zip(expected.iter()).enumerate() {
        let mf = m.to_f32();
        let ef = e.to_f32();
        let abs_err = (mf - ef).abs();
        let allowed = safety * (sum_noise_std + ef.abs() * bf16_eps);
        if abs_err > worst.3 {
            worst = (i, mf, ef, abs_err, allowed);
        }
    }
    worst
}

// ─────────────────────────────────────────────────────────────────
// Kernel-pick smoke: confirm the dispatcher routes our test shapes
// to the variant we're trying to exercise.
// ─────────────────────────────────────────────────────────────────

#[test]
fn qmv_dispatcher_routes_test_shapes_correctly() {
    // qmv_quad sample: K=128
    assert!(matches!(
        pick_qmv_kernel(64, 128, 4),
        QmvKernel::Quad { d: 128 }
    ));
    // qmv_fast sample: K=512, N=64
    assert_eq!(pick_qmv_kernel(64, 512, 4), QmvKernel::Fast);
    // qmv generic sample: K=384, N=12
    assert_eq!(pick_qmv_kernel(12, 384, 4), QmvKernel::Generic);
}

// ─────────────────────────────────────────────────────────────────
// qmv_quad parity
// ─────────────────────────────────────────────────────────────────

#[test]
fn affine_qmv_quad_b4_bf16_matches_cpu_reference() {
    // K=128 hits the qmv_quad path. N=64 covers a full quadgroup row
    // (8 outputs/quadgroup × 8 quads/simd = 64 outputs/threadgroup).
    let m = 1;
    let n = 64;
    let k = 128;
    for &group_size in &[32usize, 64, 128] {
        let (packed, scales, biases, x) =
            make_inputs_bf16(0xCAFE_u64 ^ group_size as u64, n, k, m, group_size);
        let expected = cpu_qmv_bf16(&packed, &scales, &biases, &x, m, n, k, group_size);
        let Some(metal) = run_qmv_bf16(&packed, &scales, &biases, &x, m, n, k, group_size as u32)
        else {
            eprintln!("skipping: no Metal 4 GPU");
            return;
        };

        let (idx, mv, ev, abs_err, allowed) =
            worst_abs_error_vs_noise_floor(&metal, &expected, k, 0.5);
        assert!(
            abs_err <= allowed,
            "qmv_quad gs={group_size}: worst abs_err={abs_err:.5} at idx {idx} \
             (allowed {allowed:.5}; metal={mv}, cpu={ev})"
        );
    }
}

// ─────────────────────────────────────────────────────────────────
// qmv_fast parity
// ─────────────────────────────────────────────────────────────────

#[test]
fn affine_qmv_fast_b4_bf16_matches_cpu_reference() {
    // N%8==0 ∧ K%512==0 → qmv_fast. K=512 = 1 simd block.
    let m = 1;
    let n = 64;
    let k = 512;
    for &group_size in &[32usize, 64, 128] {
        let (packed, scales, biases, x) =
            make_inputs_bf16(0xBEEF_u64 ^ group_size as u64, n, k, m, group_size);
        let expected = cpu_qmv_bf16(&packed, &scales, &biases, &x, m, n, k, group_size);
        let Some(metal) = run_qmv_bf16(&packed, &scales, &biases, &x, m, n, k, group_size as u32)
        else {
            eprintln!("skipping: no Metal 4 GPU");
            return;
        };

        let (idx, mv, ev, abs_err, allowed) =
            worst_abs_error_vs_noise_floor(&metal, &expected, k, 0.5);
        assert!(
            abs_err <= allowed,
            "qmv_fast gs={group_size}: worst abs_err={abs_err:.5} at idx {idx} \
             (allowed {allowed:.5}; metal={mv}, cpu={ev})"
        );
    }
}

// ─────────────────────────────────────────────────────────────────
// Pins the qmv_fast kernel against an unaligned packed-weight buffer
// offset, mimicking what `PooledBufferAllocator::alloc_and_copy_host`'s
// mmap-alias short-circuit does for the live forward: the safetensors
// header for mlx-community/Llama-3.2-1B-Instruct-4bit is 41161 bytes
// (so the data section starts at file pos 41169 ≡ 1 mod 4), which
// makes every packed u32 weight tensor land at an offset whose
// remainder mod 4 is 1 within the mmap-backed Metal buffer. Binding
// that as `device const uint32_t* w [[buffer(0)]]` with an unaligned
// offset is UB per the Metal spec; whether Apple's driver silently
// tolerates it or returns wrong values determines whether the live
// divergence ("first decoded token correct, subsequent tokens
// repetitive garbage") is alignment-driven.
//
// The test allocates a buffer with an extra prefix of `ALIGN_PREFIX`
// bytes, places the packed weight at offset `ALIGN_PREFIX` (= 1 byte,
// i.e. mod-4 = 1), and binds via `setBuffer:offset:atIndex:` with
// that offset. If the kernel produces wrong values, alignment is the
// root cause.
//
// FAILS → alignment is the bug.
// PASSES → alignment is silently tolerated, look elsewhere.
// ─────────────────────────────────────────────────────────────────

/// Variant of `run_qmv_bf16` whose packed-weight buffer is allocated
/// with `prefix_bytes` of slack at the start, mimicking the live
/// forward's mmap-alias short-circuit that hands the kernel an
/// unaligned offset into a registered safetensors mmap. `MetalAffineQmv`
/// itself binds at offset 0 (its `execute` always passes `offset=0`),
/// so the test pre-creates a new sub-buffer view at the offset we
/// want by calling `newBufferWithBytesNoCopy` against the parent
/// buffer's contents+prefix pointer. The driver then sees a buffer
/// whose CONTENT-BASE is shifted by `prefix_bytes` from its parent
/// allocation, which is exactly the same address arithmetic the
/// mmap-alias path produces.
fn buffer_from_bytes_offset_into_parent(
    device: &Device,
    bytes: &[u8],
    prefix_bytes: usize,
) -> (Buffer, Buffer) {
    // Parent: prefix_bytes of slack + the actual packed bytes,
    // page-padded so newBufferWithBytesNoCopy doesn't reject.
    let page_size: usize = 16384; // Apple Silicon page size
    let total_payload = prefix_bytes + bytes.len();
    let parent_bytes = total_payload.div_ceil(page_size) * page_size;
    let parent = device
        .newBufferWithLength_options(parent_bytes, MTLResourceOptions::StorageModeShared)
        .expect("parent newBufferWithLength returned nil");
    unsafe {
        let dst = parent.contents().as_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst.add(prefix_bytes), bytes.len());
    }
    // Sub-buffer view starting at +prefix_bytes from parent's content
    // base. The driver will internally compute byte offsets from THIS
    // buffer's base — which is the parent's base + prefix_bytes — so
    // a kernel that binds with offset=0 against `child` is equivalent
    // to binding with offset=prefix_bytes against `parent`. That is
    // the exact byte-pattern the mmap-alias short-circuit produces.
    let child = unsafe {
        let parent_ptr = parent.contents().as_ptr() as *mut u8;
        let child_ptr = parent_ptr.add(prefix_bytes);
        let len = bytes.len();
        // Page-align the view length so newBufferWithBytesNoCopy
        // accepts it (the driver requires the bytes pointer + length
        // to be page-aligned; we already over-allocate the parent so
        // there's slack at the tail).
        let view_len = len.div_ceil(page_size) * page_size;
        device
            .newBufferWithBytesNoCopy_length_options_deallocator(
                NonNull::new(child_ptr as *mut c_void).expect("non-null child ptr"),
                view_len,
                MTLResourceOptions::StorageModeShared,
                None,
            )
            .expect("child newBufferWithBytesNoCopy returned nil")
    };
    (parent, child)
}

#[allow(clippy::too_many_arguments)]
fn run_qmv_bf16_with_packed_prefix(
    packed: &[u8],
    scales: &[half::f16],
    biases: &[half::f16],
    x: &[half::bf16],
    m: usize,
    n: usize,
    k: usize,
    group_size: u32,
    prefix_bytes: usize,
) -> Option<Vec<half::bf16>> {
    let device = detect_device()?.device;

    // Parent must stay live through dispatch; the child is the
    // buffer bound to the kernel. On the MTL4 path the child's
    // `gpuAddress()` (= parent base + prefix_bytes) is what gets bound,
    // so the kernel still sees the same unaligned address the classic
    // `setBuffer(child, offset=0)` produced.
    let (_parent_keepalive, packed_buf) =
        buffer_from_bytes_offset_into_parent(&device, packed, prefix_bytes);

    let scales_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(scales.as_ptr() as *const u8, std::mem::size_of_val(scales))
    };
    let biases_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(biases.as_ptr() as *const u8, std::mem::size_of_val(biases))
    };
    let x_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, std::mem::size_of_val(x)) };
    let scales_buf = buffer_from_bytes(&device, scales_bytes);
    let biases_buf = buffer_from_bytes(&device, biases_bytes);
    let x_buf = buffer_from_bytes(&device, x_bytes);

    let n_out = m * n;
    let y_buf = zeroed_buffer(&device, n_out * std::mem::size_of::<half::bf16>());

    if !dispatch_qmv(
        &device,
        &packed_buf,
        &scales_buf,
        &biases_buf,
        &x_buf,
        &y_buf,
        m,
        n,
        k,
        group_size,
        ScaleDtype::F16,
    ) {
        return Some(Vec::new());
    }

    Some(read_buffer_bf16(&y_buf, n_out))
}

/// Reproduces the live forward's unaligned-binding scenario: packed
/// u32 weight at offset %4 = 1 within a Metal buffer. We expect this
/// to PANIC on the `assert!(abs_err <= allowed)` because Apple's
/// M-series Metal driver returns wrong values from unaligned u32
/// bindings. The kernel itself can't defend against this; the fix
/// lives in `PooledBufferAllocator::aligned_mmap_offset`, which refuses the
/// mmap-alias short-circuit when the safetensors offset isn't 16-byte
/// aligned. If this test ever STOPS panicking (e.g. Apple silently
/// fixes the driver), revisit whether the allocator gate is still
/// needed — but until then it is load-bearing for int4 correctness.
#[test]
#[should_panic(expected = "qmv_fast UNALIGNED packed offset")]
fn affine_qmv_fast_b4_bf16_unaligned_packed_offset_1_byte_prefix() {
    let m = 1;
    let n = 2048;
    let k = 2048;
    let group_size = 64usize;
    let (packed, scales, biases, x) = make_inputs_bf16(0xA11A_u64, n, k, m, group_size);
    let expected = cpu_qmv_bf16(&packed, &scales, &biases, &x, m, n, k, group_size);
    let Some(metal) = run_qmv_bf16_with_packed_prefix(
        &packed,
        &scales,
        &biases,
        &x,
        m,
        n,
        k,
        group_size as u32,
        /*prefix_bytes=*/ 1,
    ) else {
        // No Metal 4 GPU (e.g. CI's paravirtual device): this #[should_panic]
        // test can't exercise the driver bug, so panic with the expected
        // message to keep it green (effectively skipped) rather than failing
        // "did not panic".
        panic!("qmv_fast UNALIGNED packed offset — skipped: no Metal 4 GPU");
    };

    let (idx, mv, ev, abs_err, allowed) = worst_abs_error_vs_noise_floor(&metal, &expected, k, 0.5);
    assert!(
        abs_err <= allowed,
        "qmv_fast UNALIGNED packed offset (prefix=1, q_proj shape M=1 N=2048 K=2048 gs=64): \
         worst abs_err={abs_err:.5} at idx {idx} (allowed {allowed:.5}; metal={mv}, cpu={ev}). \
         If this fails, the safetensors mmap-alias short-circuit in PooledBufferAllocator binds U32 \
         packed weights at an unaligned offset and Metal silently returns wrong values."
    );
}

// ─────────────────────────────────────────────────────────────────
// qmv_fast at the actual Llama-3.2-1B-4bit decode-time projection
// shapes. The standalone N=64/K=512 case passes but doesn't exercise
// the multi-simdgroup walk over a 2048-row out_vec, which is what
// the model actually hits. Pins the kernel against the live shapes.
// ─────────────────────────────────────────────────────────────────

#[test]
fn affine_qmv_fast_b4_bf16_llama_3_2_1b_q_proj_shape() {
    // q_proj / o_proj: hidden -> hidden = 2048 -> 2048, gs=64.
    let m = 1;
    let n = 2048;
    let k = 2048;
    let group_size = 64usize;
    let (packed, scales, biases, x) = make_inputs_bf16(0x11B_u64, n, k, m, group_size);
    let expected = cpu_qmv_bf16(&packed, &scales, &biases, &x, m, n, k, group_size);
    let Some(metal) = run_qmv_bf16(&packed, &scales, &biases, &x, m, n, k, group_size as u32)
    else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };

    let (idx, mv, ev, abs_err, allowed) = worst_abs_error_vs_noise_floor(&metal, &expected, k, 0.5);
    assert!(
        abs_err <= allowed,
        "qmv_fast Llama-1B q_proj (M=1, N=2048, K=2048, gs=64): \
         worst abs_err={abs_err:.5} at idx {idx} \
         (allowed {allowed:.5}; metal={mv}, cpu={ev})"
    );
}

#[test]
fn affine_qmv_fast_b4_bf16_llama_3_2_1b_kv_proj_shape() {
    // k_proj / v_proj: hidden -> 8 * head_dim = 2048 -> 512, gs=64.
    let m = 1;
    let n = 512;
    let k = 2048;
    let group_size = 64usize;
    let (packed, scales, biases, x) = make_inputs_bf16(0x11C_u64, n, k, m, group_size);
    let expected = cpu_qmv_bf16(&packed, &scales, &biases, &x, m, n, k, group_size);
    let Some(metal) = run_qmv_bf16(&packed, &scales, &biases, &x, m, n, k, group_size as u32)
    else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };

    let (idx, mv, ev, abs_err, allowed) = worst_abs_error_vs_noise_floor(&metal, &expected, k, 0.5);
    assert!(
        abs_err <= allowed,
        "qmv_fast Llama-1B kv_proj (M=1, N=512, K=2048, gs=64): \
         worst abs_err={abs_err:.5} at idx {idx} \
         (allowed {allowed:.5}; metal={mv}, cpu={ev})"
    );
}

#[test]
fn affine_qmv_fast_b4_bf16_llama_3_2_1b_gate_up_shape() {
    // gate_proj / up_proj: hidden -> intermediate = 2048 -> 8192, gs=64.
    let m = 1;
    let n = 8192;
    let k = 2048;
    let group_size = 64usize;
    let (packed, scales, biases, x) = make_inputs_bf16(0x11D_u64, n, k, m, group_size);
    let expected = cpu_qmv_bf16(&packed, &scales, &biases, &x, m, n, k, group_size);
    let Some(metal) = run_qmv_bf16(&packed, &scales, &biases, &x, m, n, k, group_size as u32)
    else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };

    let (idx, mv, ev, abs_err, allowed) = worst_abs_error_vs_noise_floor(&metal, &expected, k, 0.5);
    assert!(
        abs_err <= allowed,
        "qmv_fast Llama-1B gate/up_proj (M=1, N=8192, K=2048, gs=64): \
         worst abs_err={abs_err:.5} at idx {idx} \
         (allowed {allowed:.5}; metal={mv}, cpu={ev})"
    );
}

#[test]
fn affine_qmv_fast_b4_bf16_llama_3_2_1b_down_proj_shape() {
    // down_proj: intermediate -> hidden = 8192 -> 2048, gs=64.
    // Largest K in the model: 8192. K%512==0, N%8==0 → fast.
    let m = 1;
    let n = 2048;
    let k = 8192;
    let group_size = 64usize;
    let (packed, scales, biases, x) = make_inputs_bf16(0x11E_u64, n, k, m, group_size);
    let expected = cpu_qmv_bf16(&packed, &scales, &biases, &x, m, n, k, group_size);
    let Some(metal) = run_qmv_bf16(&packed, &scales, &biases, &x, m, n, k, group_size as u32)
    else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };

    let (idx, mv, ev, abs_err, allowed) = worst_abs_error_vs_noise_floor(&metal, &expected, k, 0.5);
    assert!(
        abs_err <= allowed,
        "qmv_fast Llama-1B down_proj (M=1, N=2048, K=8192, gs=64): \
         worst abs_err={abs_err:.5} at idx {idx} \
         (allowed {allowed:.5}; metal={mv}, cpu={ev})"
    );
}

// ─────────────────────────────────────────────────────────────────
// qmv generic parity (unaligned tail path)
// ─────────────────────────────────────────────────────────────────

#[test]
fn affine_qmv_generic_b4_bf16_matches_cpu_reference() {
    // N=12 (not a multiple of 8) ∧ K=384 (not a multiple of 512) →
    // qmv generic with the bounds-checked tail load. K must still be a
    // multiple of group_size; 384 / {32,64,128} all divide.
    let m = 1;
    let n = 12;
    let k = 384;
    for &group_size in &[32usize, 64, 128] {
        let (packed, scales, biases, x) =
            make_inputs_bf16(0xFACE_u64 ^ group_size as u64, n, k, m, group_size);
        let expected = cpu_qmv_bf16(&packed, &scales, &biases, &x, m, n, k, group_size);
        let Some(metal) = run_qmv_bf16(&packed, &scales, &biases, &x, m, n, k, group_size as u32)
        else {
            eprintln!("skipping: no Metal 4 GPU");
            return;
        };

        let (idx, mv, ev, abs_err, allowed) =
            worst_abs_error_vs_noise_floor(&metal, &expected, k, 0.5);
        assert!(
            abs_err <= allowed,
            "qmv generic gs={group_size}: worst abs_err={abs_err:.5} at idx {idx} \
             (allowed {allowed:.5}; metal={mv}, cpu={ev})"
        );
    }
}

// ─────────────────────────────────────────────────────────────────
// BF16 SCALES — Qwen3 mlx-community 4bit convention. Mirrors the
// three F16-scale tests above but loads scales/biases as `half::bf16`
// and dispatches the `_s_bf16_` kernel arms (instantiated via
// `INST_QMV_ALL(_, _, bf16, bfloat, …)` in `quantized_qmv.metal`).
// Validates `ScaleDtype::Bf16` end-to-end without needing a full
// Qwen3 model load (24 GiB-cap blocks Qwen3-30B-A3B end-to-end on
// this machine).
// ─────────────────────────────────────────────────────────────────

/// Same as `make_inputs_bf16` but with BF16 scales/biases.
fn make_inputs_bf16_s_bf16(
    seed: u64,
    n: usize,
    k: usize,
    m: usize,
    group_size: usize,
) -> (Vec<u8>, Vec<half::bf16>, Vec<half::bf16>, Vec<half::bf16>) {
    assert_eq!(k % group_size, 0);
    let n_bytes = n * k / 2;
    let n_groups = n * k / group_size;
    let mut rng = SplitMix64(seed);
    let packed: Vec<u8> = (0..n_bytes).map(|_| rng.next_byte()).collect();
    let scales: Vec<half::bf16> = (0..n_groups)
        .map(|_| half::bf16::from_f32(0.01 + 0.04 * rng.next_unit_f32()))
        .collect();
    let biases: Vec<half::bf16> = (0..n_groups)
        .map(|_| half::bf16::from_f32(rng.next_unit_f32() - 0.5))
        .collect();
    let x: Vec<half::bf16> = (0..(m * k))
        .map(|_| half::bf16::from_f32(2.0 * rng.next_unit_f32() - 1.0))
        .collect();
    (packed, scales, biases, x)
}

#[allow(clippy::too_many_arguments)]
fn run_qmv_bf16_s_bf16(
    packed: &[u8],
    scales: &[half::bf16],
    biases: &[half::bf16],
    x: &[half::bf16],
    m: usize,
    n: usize,
    k: usize,
    group_size: u32,
) -> Option<Vec<half::bf16>> {
    let device = detect_device()?.device;
    let packed_buf = buffer_from_bytes(&device, packed);
    let scales_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(scales.as_ptr() as *const u8, std::mem::size_of_val(scales))
    };
    let biases_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(biases.as_ptr() as *const u8, std::mem::size_of_val(biases))
    };
    let x_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, std::mem::size_of_val(x)) };
    let scales_buf = buffer_from_bytes(&device, scales_bytes);
    let biases_buf = buffer_from_bytes(&device, biases_bytes);
    let x_buf = buffer_from_bytes(&device, x_bytes);
    let n_out = m * n;
    let y_buf = zeroed_buffer(&device, n_out * std::mem::size_of::<half::bf16>());
    if !dispatch_qmv(
        &device,
        &packed_buf,
        &scales_buf,
        &biases_buf,
        &x_buf,
        &y_buf,
        m,
        n,
        k,
        group_size,
        ScaleDtype::Bf16,
    ) {
        return Some(Vec::new());
    }
    Some(read_buffer_bf16(&y_buf, n_out))
}

#[test]
fn affine_qmv_quad_b4_bf16_s_bf16_matches_cpu_reference() {
    // K=128 → qmv_quad with D=128.
    let m = 1;
    let n = 64;
    let k = 128;
    for &group_size in &[32usize, 64, 128] {
        let (packed, scales, biases, x) =
            make_inputs_bf16_s_bf16(0xC0DE_u64 ^ group_size as u64, n, k, m, group_size);
        let expected = scratchy_target_metal::cpu_reference::affine_qmv_b4_bf16_s_bf16(
            &packed, &scales, &biases, &x, m, n, k, group_size,
        );
        let Some(metal) =
            run_qmv_bf16_s_bf16(&packed, &scales, &biases, &x, m, n, k, group_size as u32)
        else {
            eprintln!("skipping: no Metal 4 GPU");
            return;
        };
        let (idx, mv, ev, abs_err, allowed) =
            worst_abs_error_vs_noise_floor(&metal, &expected, k, 0.5);
        assert!(
            abs_err <= allowed,
            "qmv_quad s_bf16 gs={group_size}: worst abs_err={abs_err:.5} at idx {idx} \
             (allowed {allowed:.5}; metal={mv}, cpu={ev})"
        );
    }
}

#[test]
fn affine_qmv_fast_b4_bf16_s_bf16_matches_cpu_reference() {
    let m = 1;
    let n = 64;
    let k = 512;
    for &group_size in &[32usize, 64, 128] {
        let (packed, scales, biases, x) =
            make_inputs_bf16_s_bf16(0xBEEF_u64 ^ group_size as u64, n, k, m, group_size);
        let expected = scratchy_target_metal::cpu_reference::affine_qmv_b4_bf16_s_bf16(
            &packed, &scales, &biases, &x, m, n, k, group_size,
        );
        let Some(metal) =
            run_qmv_bf16_s_bf16(&packed, &scales, &biases, &x, m, n, k, group_size as u32)
        else {
            eprintln!("skipping: no Metal 4 GPU");
            return;
        };
        let (idx, mv, ev, abs_err, allowed) =
            worst_abs_error_vs_noise_floor(&metal, &expected, k, 0.5);
        assert!(
            abs_err <= allowed,
            "qmv_fast s_bf16 gs={group_size}: worst abs_err={abs_err:.5} at idx {idx} \
             (allowed {allowed:.5}; metal={mv}, cpu={ev})"
        );
    }
}

#[test]
fn affine_qmv_generic_b4_bf16_s_bf16_matches_cpu_reference() {
    let m = 1;
    let n = 12;
    let k = 384;
    for &group_size in &[32usize, 64, 128] {
        let (packed, scales, biases, x) =
            make_inputs_bf16_s_bf16(0xFACE_u64 ^ group_size as u64, n, k, m, group_size);
        let expected = scratchy_target_metal::cpu_reference::affine_qmv_b4_bf16_s_bf16(
            &packed, &scales, &biases, &x, m, n, k, group_size,
        );
        let Some(metal) =
            run_qmv_bf16_s_bf16(&packed, &scales, &biases, &x, m, n, k, group_size as u32)
        else {
            eprintln!("skipping: no Metal 4 GPU");
            return;
        };
        let (idx, mv, ev, abs_err, allowed) =
            worst_abs_error_vs_noise_floor(&metal, &expected, k, 0.5);
        assert!(
            abs_err <= allowed,
            "qmv generic s_bf16 gs={group_size}: worst abs_err={abs_err:.5} at idx {idx} \
             (allowed {allowed:.5}; metal={mv}, cpu={ev})"
        );
    }
}
