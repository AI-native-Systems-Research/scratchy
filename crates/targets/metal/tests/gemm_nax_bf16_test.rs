//! Correctness + speed for the plain bf16 NAX (M5 matrix-accelerator) GEMM
//! `gemm_nax_bf16` (y = x @ w^T) added to quantized_qmm_nax.metal. This is the
//! GEMM the gemma4 hd512 unfused-attention path will use for QK^T / PV (MPS is
//! not usable inside the MTL4 forward). Measures TFLOP/s on the real attention
//! shapes so we KNOW the speedup is real before wiring.
//!
//! Run: cargo test -p scratchy-target-metal --release --test gemm_nax_bf16_test -- --ignored --nocapture

mod common;

use std::ffi::c_void;
use std::ptr::NonNull;
use std::time::Instant;

use half::bf16;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLDataType, MTLDevice, MTLFunctionConstantValues, MTLLibrary, MTLResourceOptions,
    MTLSize,
};
use scratchy_target_metal::detect_device;
use scratchy_target_metal::shader_cache::compile_nax_library_from_source;

type Buf = Retained<ProtocolObject<dyn MTLBuffer>>;

fn buf(dev: &ProtocolObject<dyn MTLDevice>, bytes: &[u8]) -> Buf {
    unsafe {
        dev.newBufferWithBytes_length_options(
            NonNull::new(bytes.as_ptr() as *mut c_void).unwrap(),
            bytes.len().max(1),
            MTLResourceOptions::StorageModeShared,
        )
        .expect("buf")
    }
}
fn as_bytes<T: Copy>(s: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

fn pipeline(
    dev: &ProtocolObject<dyn MTLDevice>,
    lib: &ProtocolObject<dyn MTLLibrary>,
    m: i32,
    n: i32,
    k: i32,
) -> Retained<ProtocolObject<dyn objc2_metal::MTLComputePipelineState>> {
    let fc = MTLFunctionConstantValues::new();
    // QMM_K=0, QMM_N=1, QMM_M=2  (int function constants)
    unsafe {
        for (val, idx) in [(k, 0usize), (n, 1), (m, 2)] {
            fc.setConstantValue_type_atIndex(
                NonNull::new(&val as *const i32 as *mut c_void).unwrap(),
                MTLDataType::Int,
                idx,
            );
        }
    }
    let func = lib
        .newFunctionWithName_constantValues_error(&NSString::from_str("gemm_nax_bf16"), &fc)
        .expect("specialize gemm_nax_bf16");
    dev.newComputePipelineStateWithFunction_error(&func)
        .expect("pipeline")
}

fn run_gemm(
    dev: &Retained<ProtocolObject<dyn MTLDevice>>,
    pipe: &ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    x: &Buf,
    w: &Buf,
    y: &Buf,
    m: u32,
    n: u32,
) {
    // Production MTL4 dispatch: kernel buffer-index order x(0), w(1), y(2);
    // M/N/K are baked into `pipe` as function constants, not buffer bindings.
    common::dispatch_threadgroups(
        dev,
        pipe,
        &[x, w, y],
        MTLSize {
            width: n.div_ceil(64) as usize,
            height: m.div_ceil(64) as usize,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 2,
            depth: 2,
        },
    );
}

#[test]
#[ignore]
fn gemm_nax_bf16_correct_and_fast() {
    let Some(device) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let dev = &device.device;
    let lib = compile_nax_library_from_source(dev).expect("compile NAX lib");

    // ---- Correctness: M=128, N=96 (unaligned), K=128 vs CPU A@B^T ----
    {
        let (m, n, k) = (128u32, 96u32, 128u32);
        let mut xf = vec![0f32; (m * k) as usize];
        let mut wf = vec![0f32; (n * k) as usize];
        for (i, x) in xf.iter_mut().enumerate() {
            *x = 0.05 * ((i % 17) as f32) - 0.4;
        }
        for (i, w) in wf.iter_mut().enumerate() {
            *w = 0.05 * ((i % 13) as f32) - 0.3;
        }
        let xb: Vec<u16> = xf.iter().map(|&v| bf16::from_f32(v).to_bits()).collect();
        let wb: Vec<u16> = wf.iter().map(|&v| bf16::from_f32(v).to_bits()).collect();
        let xbuf = buf(dev, as_bytes(&xb));
        let wbuf = buf(dev, as_bytes(&wb));
        let ybuf = buf(dev, &vec![0u8; (m * n * 2) as usize]);
        let pipe = pipeline(dev, &lib, m as i32, n as i32, k as i32);
        run_gemm(dev, &pipe, &xbuf, &wbuf, &ybuf, m, n);
        let yo: &[u16] = unsafe {
            std::slice::from_raw_parts(ybuf.contents().as_ptr() as *const u16, (m * n) as usize)
        };
        // decode bf16 inputs for the reference (same rounding the kernel reads)
        let xd: Vec<f32> = xb.iter().map(|&b| bf16::from_bits(b).to_f32()).collect();
        let wd: Vec<f32> = wb.iter().map(|&b| bf16::from_bits(b).to_f32()).collect();
        let mut max_err = 0f32;
        for mm in 0..m {
            for nn in 0..n {
                let mut acc = 0f32;
                for kk in 0..k {
                    acc += xd[(mm * k + kk) as usize] * wd[(nn * k + kk) as usize];
                }
                let got = bf16::from_bits(yo[(mm * n + nn) as usize]).to_f32();
                max_err = max_err.max((got - acc).abs());
            }
        }
        println!("gemm_nax_bf16 correctness: max abs err = {max_err:.4} (M={m} N={n} K={k})");
        assert!(max_err < 0.1, "NAX bf16 GEMM wrong: max err {max_err}");
    }

    // ---- Speed on the attention shapes ----
    let bench = |label: &str, m: u32, n: u32, k: u32| {
        let xbuf = buf(dev, &vec![0u8; (m as u64 * k as u64 * 2) as usize]);
        let wbuf = buf(dev, &vec![0u8; (n as u64 * k as u64 * 2) as usize]);
        let ybuf = buf(dev, &vec![0u8; (m as u64 * n as u64 * 2) as usize]);
        let pipe = pipeline(dev, &lib, m as i32, n as i32, k as i32);
        for _ in 0..2 {
            run_gemm(dev, &pipe, &xbuf, &wbuf, &ybuf, m, n);
        }
        let iters = 5;
        let t0 = Instant::now();
        for _ in 0..iters {
            run_gemm(dev, &pipe, &xbuf, &wbuf, &ybuf, m, n);
        }
        let dt = t0.elapsed().as_secs_f64() / iters as f64;
        let tflops = 2.0 * m as f64 * n as f64 * k as f64 / dt / 1e12;
        println!(
            "  {label}  M={m} N={n} K={k}: {:.2} ms  {:.2} TFLOP/s",
            dt * 1e3,
            tflops
        );
    };
    println!(
        "=== gemm_nax_bf16 throughput on attention shapes (vs MPS f16 14 TFLOP/s, mlx bf16 3.9) ==="
    );
    bench("QK^T @30k", 2048, 30000, 512);
    bench("PV   @30k", 2048, 512, 30000);
    bench("QK^T @8k ", 2048, 8192, 512);
}

/// Regression guard for the MTL4 threadgroup over-reservation fault.
///
/// Without `[[max_total_threads_per_threadgroup(WM*WN*32)]]`, the NAX matmul2d
/// (MXU) GEMM reserves its cooperative-tensor threadgroup for the device-max
/// thread count under the MTL4 compute encoder (~58 KiB) instead of the 128
/// threads it actually launches — which exceeds the MXU encoder limit and faults
/// at GPU exec with an opaque `MTLCommandBufferStatus(5)`. The attribute is the
/// fix (caps it to ~9 KiB). This value bug is decided by the Metal compiler, so
/// it cannot be a Rust `const`-assert; this source-invariant test ensures the fix
/// is never silently removed from any `gemm_nax_bf16*` kernel.
#[test]
fn gemm_nax_kernels_cap_threadgroup() {
    let src = include_str!("../shaders/quantized_qmm_nax.metal");
    let mut checked = 0usize;
    for line in src.lines() {
        if let Some(pos) = line.find("void gemm_nax_bf16") {
            // The kernel attribute is on the same line, before `void`.
            let prefix = &line[..pos];
            assert!(
                prefix.contains("max_total_threads_per_threadgroup"),
                "gemm_nax kernel missing the MXU threadgroup cap \
                 (reintroduces the MTL4 status-5 over-reservation fault): {line}"
            );
            checked += 1;
        }
    }
    assert!(
        checked >= 3,
        "expected >=3 gemm_nax_bf16* kernels, found {checked}"
    );
}
