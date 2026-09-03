//! Correctness test for the fused scale + causal-mask + row softmax kernel
//! (`attention_causal_softmax.metal`), the gemma4 hd512 unfused-attention path.
//!
//! Validates the scale, the chunked-prefill causal mask (query i at absolute
//! pos (kv_len-Lq)+i attends keys [0, that]), the fp32-accumulated softmax, and
//! the zeroed masked tail — against a CPU reference.
//!
//! Dispatches on the production MTL4 path (see `common::dispatch_threadgroups`).

mod common;

use std::ffi::c_void;
use std::ptr::NonNull;

use half::{bf16, f16};
use objc2_foundation::NSString;
use objc2_metal::{MTLDataType, MTLDevice, MTLFunctionConstantValues, MTLLibrary, MTLSize};
use scratchy_target_metal::detect_device;

const Q_HEADS: u32 = 2;
const LQ: u32 = 3;
const KV_LEN: u32 = 5; // chunk_start = kv_len - Lq = 2
const SCALE: f32 = 0.5;

#[test]
fn causal_softmax_matches_reference() {
    let Some(__dev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let device = __dev.device;

    let src = include_str!("../shaders/attention_causal_softmax.metal");
    let opts = objc2_metal::MTLCompileOptions::new();
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(src), Some(&opts))
        .expect("compile attention_causal_softmax.metal");
    let func = library
        .newFunctionWithName(&NSString::from_str("causal_softmax_f16"))
        .expect("causal_softmax_f16");
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&func)
        .expect("pipeline");

    // Input scores [q_heads, Lq, kv_len], deterministic small fp16-exact values.
    let n = (Q_HEADS * LQ * KV_LEN) as usize;
    let mut raw = vec![0f32; n];
    for h in 0..Q_HEADS {
        for i in 0..LQ {
            for j in 0..KV_LEN {
                let idx = ((h * LQ + i) * KV_LEN + j) as usize;
                // Vary by all indices; keep small so f16 is exact.
                raw[idx] = ((h + 1) as f32) * 1.0 + (i as f32) * 0.5 - (j as f32) * 0.25;
            }
        }
    }
    let scores: Vec<u16> = raw.iter().map(|&x| f16::from_f32(x).to_bits()).collect();
    let scores_buf = common::shared_slice(&device, &scores);
    let params_buf = common::shared_slice(&device, &[LQ, KV_LEN]);
    let scale_buf = common::shared_f32(&device, SCALE);

    // buffer(0)=scores (in-place), buffer(1)=params[Lq,kv_len], buffer(2)=scale;
    // grid (Lq, q_heads, 1) threadgroups, one thread each.
    if !common::dispatch_threadgroups(
        &device,
        &pipeline,
        &[&scores_buf, &params_buf, &scale_buf],
        MTLSize {
            width: LQ as usize,
            height: Q_HEADS as usize,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    ) {
        return;
    }

    let out: Vec<u16> = common::read_slice(&scores_buf, n);

    // CPU reference.
    let chunk_start = KV_LEN - LQ;
    for h in 0..Q_HEADS {
        for i in 0..LQ {
            let valid = (chunk_start + i + 1) as usize;
            let base = ((h * LQ + i) * KV_LEN) as usize;
            // scaled scores
            let scaled: Vec<f32> = (0..valid).map(|j| raw[base + j] * SCALE).collect();
            let m = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scaled.iter().map(|&s| (s - m).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for j in 0..(KV_LEN as usize) {
                let got = f16::from_bits(out[base + j]).to_f32();
                let want = if j < valid { exps[j] / sum } else { 0.0 };
                assert!(
                    (got - want).abs() <= 2e-3 + 2e-3 * want.abs(),
                    "h={h} i={i} j={j}: got {got} want {want}"
                );
            }
        }
    }
}

/// Regression guard for the production variant `causal_softmax_prod_bf16`, which
/// derives the query count from `cu_seqlens_q` (NOT the dispatch grid / baked
/// bucket). The fault this guards: the prefill bucket (= dispatched rows) is
/// LARGER than `kv_len` for short contexts, so computing `valid = kv_len - rows`
/// underflows `uint` to ~4e9 → massive OOB read → GPU MTLCommandBufferStatus(5).
/// This test dispatches MORE rows (BUCKET_ROWS) than KV positions (KVP) with only
/// LQ_ACTUAL real queries — the exact shape that crashed before the fix. It must
/// run cleanly AND leave padding rows untouched.
#[test]
fn causal_softmax_prod_handles_bucket_larger_than_kv() {
    const BUCKET_ROWS: u32 = 8; // dispatched rows (over-sized bucket) > KVP
    const LQ_ACTUAL: u32 = 3; // real new-query tokens this forward
    const KVP: u32 = 5; // live kv_len; chunk_start = KVP - LQ_ACTUAL = 2
    const SCALE: f32 = 0.5;

    let Some(__dev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let device = __dev.device;

    let src = include_str!("../shaders/attention_causal_softmax.metal");
    let opts = objc2_metal::MTLCompileOptions::new();
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(src), Some(&opts))
        .expect("compile attention_causal_softmax.metal");

    // SOFT_SCALE is function_constant(1).
    let fcv = MTLFunctionConstantValues::new();
    let scale = SCALE;
    unsafe {
        fcv.setConstantValue_type_atIndex(
            NonNull::new(&scale as *const f32 as *mut c_void).unwrap(),
            MTLDataType::Float,
            1,
        );
    }
    let func = library
        .newFunctionWithName_constantValues_error(
            &NSString::from_str("causal_softmax_prod_bf16"),
            &fcv,
        )
        .expect("causal_softmax_prod_bf16");
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&func)
        .expect("pipeline");

    // scores [BUCKET_ROWS, KVP] bf16, row i at i*KVP (prod variant is single-head).
    let n = (BUCKET_ROWS * KVP) as usize;
    let mut raw = vec![0f32; n];
    for i in 0..BUCKET_ROWS {
        for j in 0..KVP {
            raw[(i * KVP + j) as usize] = 1.0 + (i as f32) * 0.5 - (j as f32) * 0.25;
        }
    }
    let scores: Vec<u16> = raw.iter().map(|&x| bf16::from_f32(x).to_bits()).collect();
    let scores_buf = common::shared_slice(&device, &scores);
    let seq_used_buf = common::shared_slice(&device, &[KVP]);
    let cu_buf = common::shared_slice(&device, &[0u32, LQ_ACTUAL]); // single-seq: lq_actual = cu[1]-cu[0]

    // buffer(0)=scores (in-place), buffer(1)=seq_used, buffer(2)=cu_seqlens;
    // dispatch the FULL bucket (more rows than KVP) — the crash shape.
    if !common::dispatch_threadgroups(
        &device,
        &pipeline,
        &[&scores_buf, &seq_used_buf, &cu_buf],
        MTLSize {
            width: BUCKET_ROWS as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    ) {
        return;
    }

    let out: Vec<u16> = common::read_slice(&scores_buf, n);

    // Real query rows [0, LQ_ACTUAL): correct causal softmax. Padding rows
    // [LQ_ACTUAL, BUCKET_ROWS): zeroed — the kernel clears them (rather than
    // leaving them untouched) so a garbage padding row can't poison a real
    // row through the NAX PV-gemm's unguarded contraction K-tail over-read.
    let chunk_start = KVP - LQ_ACTUAL;
    for i in 0..BUCKET_ROWS {
        let base = (i * KVP) as usize;
        if i >= LQ_ACTUAL {
            for j in 0..(KVP as usize) {
                let got = bf16::from_bits(out[base + j]).to_f32();
                assert!(
                    got.abs() <= 1e-2,
                    "padding row i={i} j={j} should be zeroed: got {got}"
                );
            }
            continue;
        }
        let valid = (chunk_start + i + 1) as usize;
        let scaled: Vec<f32> = (0..valid).map(|j| raw[base + j] * SCALE).collect();
        let m = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scaled.iter().map(|&s| (s - m).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for j in 0..(KVP as usize) {
            let got = bf16::from_bits(out[base + j]).to_f32();
            let want = if j < valid { exps[j] / sum } else { 0.0 };
            assert!(
                (got - want).abs() <= 1e-2 + 1e-2 * want.abs(),
                "i={i} j={j}: got {got} want {want}"
            );
        }
    }
}
