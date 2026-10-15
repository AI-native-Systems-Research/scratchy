//! Correctness test for the steel (simdgroup) attention GEMMs
//! `gemm_bf16_qk` / `gemm_bf16_pv` in `gemm.metal` — the pre-M5 (non-NAX)
//! counterparts of `gemm_nax_bf16_qk/pv` for the gemma4 hd512 unfused path.
//!
//! Both compute C = A @ B^T. The dynamic dim (N for QKᵀ, K for PV) is read from
//! `seq_used[0]`. CRUCIALLY the PV case exercises a **weight row stride
//! (`w_ld` = GEMM_PV_W_LD = max_kv) LARGER than the contraction (`K` = kv_len)** with
//! the tail columns zeroed — the exact shape of the V^T dense buffer in the
//! model (gathered at static stride max_kv, contracted over only kv_len). This
//! guards the latent bug where the GEMM strode the weight by `kv_len` instead of
//! `max_kv` (correct only when they happen to be equal).

mod common;

use std::ffi::c_void;
use std::ptr::NonNull;

use half::bf16;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize};
use scratchy_target_metal::aot::baked_pipeline;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::specialized_pipeline_cache::ConstantValue;

fn shared(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
) -> objc2::rc::Retained<ProtocolObject<dyn MTLBuffer>> {
    unsafe {
        device
            .newBufferWithBytes_length_options(
                NonNull::new(bytes.as_ptr() as *mut c_void).unwrap(),
                bytes.len().max(1),
                MTLResourceOptions::StorageModeShared,
            )
            .expect("newBufferWithBytes")
    }
}

fn as_bytes<T: Copy>(s: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

fn run_and_check(qk: bool) {
    let Some(device) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let dev = &device.device;

    let m = 12u32; // Lq
    // (contraction kc, output N live, output N grid-max, weight row stride w_ld)
    let (kc, n_live, n_max, w_ld) = if qk {
        // QKᵀ: contraction K=hd=16 (static); N=kv_len live 20, grid baked at 24
        // (tiles past 20 early-return). kdense is contiguous → w_ld == K.
        (16u32, 20u32, 24u32, 16u32)
    } else {
        // PV: contraction K=kv_len=20 (from seq_used); N=hd=16 (static grid).
        // V^T weight row stride w_ld=24 > K=20 (the max_kv>kv_len bug shape).
        (20u32, 16u32, 16u32, 24u32)
    };

    // A [m, kc] (contraction stride). B [n_max, w_ld] (weight row stride); only
    // the first `kc` columns of each B row are live, the rest are zeroed.
    let a_f: Vec<f32> = (0..(m * kc))
        .map(|i| 0.05 * ((i % 17) as f32) - 0.4)
        .collect();
    let mut b_f = vec![0f32; (n_max * w_ld) as usize];
    for n in 0..n_max {
        for k in 0..kc {
            b_f[(n * w_ld + k) as usize] = 0.05 * (((n * w_ld + k) % 13) as f32) - 0.3;
        }
    }
    let a_bf: Vec<u16> = a_f.iter().map(|&x| bf16::from_f32(x).to_bits()).collect();
    let b_bf: Vec<u16> = b_f.iter().map(|&x| bf16::from_f32(x).to_bits()).collect();
    let a_buf = shared(dev, as_bytes(&a_bf));
    let b_buf = shared(dev, as_bytes(&b_bf));
    let out = vec![0u16; (m * n_max) as usize];
    let out_buf = shared(dev, as_bytes(&out));
    let seq_used = [if qk { n_live } else { kc }]; // QKᵀ→N, PV→K
    let seq_buf = shared(dev, as_bytes(&seq_used));

    let constants = match qk {
        // GEMM_M; GEMM_K = hd (== w_ld, contiguous kdense).
        true => vec![ConstantValue::uint(0, m), ConstantValue::uint(2, w_ld)],
        // GEMM_M; GEMM_N = hd; GEMM_PV_W_LD = max_kv (V^T row stride, set at load).
        false => vec![
            ConstantValue::uint(0, m),
            ConstantValue::uint(1, n_live),
            ConstantValue::uint(4, w_ld),
        ],
    };
    let fn_name = if qk { "gemm_bf16_qk" } else { "gemm_bf16_pv" };
    let pipeline = baked_pipeline(dev, "gemm", fn_name, constants)
        .unwrap_or_else(|e| panic!("{fn_name}: {e:?}"));

    // Production MTL4 dispatch: buffers bound at argument-table indices
    // matching the kernel's buffer(0..3) — out(0), A(1), B(2), seq(3).
    // Scalars (GEMM_M/N/K, the PV stride) are constants, so there are
    // no setBytes operands. dispatchThreads' exact extent becomes
    // ceil-divided threadgroups; the kernel bounds-checks so it is
    // bit-identical to the classic path.
    let dispatched = common::dispatch_threadgroups(
        dev,
        &pipeline,
        &[&out_buf, &a_buf, &b_buf, &seq_buf],
        MTLSize {
            width: n_max.div_ceil(32) as usize,
            height: m.div_ceil(32) as usize,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    if !dispatched {
        return;
    }

    let got: &[u16] = unsafe {
        std::slice::from_raw_parts(
            out_buf.contents().as_ptr() as *const u16,
            (m * n_max) as usize,
        )
    };

    // CPU ref: out[i][j] = sum_{k<kc} A[i*kc+k] * B[j*w_ld+k], output row stride n_live.
    for i in 0..m {
        for j in 0..n_live {
            let mut acc = 0f32;
            for k in 0..kc {
                acc += a_f[(i * kc + k) as usize] * b_f[(j * w_ld + k) as usize];
            }
            let g = bf16::from_bits(got[(i * n_live + j) as usize]).to_f32();
            assert!(
                (g - acc).abs() <= 5e-2 + 5e-2 * acc.abs(),
                "{fn_name} i={i} j={j}: got {g} want {acc}"
            );
        }
    }
}

#[test]
fn gemm_bf16_qk_matches_reference() {
    run_and_check(true);
}

#[test]
fn gemm_bf16_pv_strided_weight_matches_reference() {
    run_and_check(false);
}
