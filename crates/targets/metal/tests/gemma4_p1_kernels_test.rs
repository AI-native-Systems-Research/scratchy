// SPDX-License-Identifier: Apache-2.0
//! P1 kernel parity tests for the Gemma4 port:
//!   * `gelu_mul_f16` (decomposed GeGLU q-MLP tail) vs CPU gelu_tanh.
//!   * `tanh_soft_cap_f16_specialized` (final logit softcap, cap as
//!     function constant 1) vs CPU.
//!   * `ATTN_WINDOW` sliding-window mask in the paged SDPA prefill and
//!     decode kernels vs a CPU softmax-attention reference (window 0 ≡
//!     causal; window W masks `q_abs - k >= W`).
//!
//! GPU tests — run with `--test-threads=1` (standing rule).

mod common;

use half::f16;
use objc2_metal::{MTLBuffer, MTLComputePipelineState, MTLDevice, MTLResourceOptions, MTLSize};
use scratchy_target_metal::detect_device;
use scratchy_target_metal::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};

type Device = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLDevice>>;
type Buffer = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>>;

fn buf_f16(device: &Device, data: &[f32]) -> Buffer {
    let h: Vec<f16> = data.iter().map(|&v| f16::from_f32(v)).collect();
    let bytes = (h.len() * 2).max(4);
    let buf = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .expect("newBuffer");
    unsafe {
        std::ptr::copy_nonoverlapping(
            h.as_ptr() as *const u8,
            buf.contents().as_ptr() as *mut u8,
            h.len() * 2,
        );
    }
    buf
}

fn buf_u32(device: &Device, data: &[u32]) -> Buffer {
    let bytes = std::mem::size_of_val(data).max(4);
    let buf = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .expect("newBuffer");
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8,
            buf.contents().as_ptr() as *mut u8,
            std::mem::size_of_val(data),
        );
    }
    buf
}

fn buf_u64(device: &Device, data: &[u64]) -> Buffer {
    let bytes = std::mem::size_of_val(data).max(8);
    let buf = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .expect("newBuffer");
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8,
            buf.contents().as_ptr() as *mut u8,
            std::mem::size_of_val(data),
        );
    }
    buf
}

fn buf_zero(device: &Device, bytes: usize) -> Buffer {
    let buf = device
        .newBufferWithLength_options(bytes.max(4), MTLResourceOptions::StorageModeShared)
        .expect("newBuffer");
    unsafe { std::ptr::write_bytes(buf.contents().as_ptr() as *mut u8, 0, bytes) };
    buf
}

fn read_f16(buf: &Buffer, n: usize) -> Vec<f32> {
    unsafe {
        std::slice::from_raw_parts(buf.contents().as_ptr() as *const f16, n)
            .iter()
            .map(|h| h.to_f32())
            .collect()
    }
}

fn pseudo(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as u32 & 0x00FF_FFFF) as f32 / (1u32 << 23) as f32 - 1.0) * scale
        })
        .collect()
}

fn gelu_tanh_ref(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEFF: f32 = 0.044_715;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + COEFF * x * x * x)).tanh())
}

/// 1-D elementwise dispatch on the production MTL4 path (256-wide
/// threadgroups). `bufs[i]` binds at argument-table index `i`.
fn dispatch_1d(device: &Device, pipeline: &common::Pipeline, bufs: &[&Buffer], threads: usize) {
    let tg = 256usize;
    common::dispatch_threadgroups(
        device,
        pipeline,
        bufs,
        MTLSize {
            width: threads.div_ceil(tg),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );
}

#[test]
fn gelu_mul_f16_matches_cpu() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shaders");

    let n = 4096usize;
    let gate = pseudo(7, n, 4.0);
    let up = pseudo(11, n, 2.0);

    let key = PipelineKey::new(
        "silu_mul",
        "gelu_mul_f16",
        vec![ConstantValue::uint(0, n as u32)],
    );
    let pipeline = cache.get_or_build(&key).expect("gelu_mul pipeline");

    let gate_buf = buf_f16(&device, &gate);
    let up_buf = buf_f16(&device, &up);
    let out_buf = buf_zero(&device, n * 2);
    dispatch_1d(&device, &pipeline, &[&out_buf, &gate_buf, &up_buf], n);

    let got = read_f16(&out_buf, n);
    let mut max_err = 0f32;
    for i in 0..n {
        // reference uses the f16-rounded inputs the kernel actually read
        let g = f16::from_f32(gate[i]).to_f32();
        let u = f16::from_f32(up[i]).to_f32();
        let want = gelu_tanh_ref(g) * u;
        max_err = max_err.max((got[i] - want).abs());
    }
    assert!(max_err < 5e-3, "gelu_mul max_err {max_err}");
}

#[test]
fn tanh_soft_cap_f16_specialized_matches_cpu() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shaders");

    let n = 2048usize;
    let cap = 30.0f32;
    let x = pseudo(13, n, 80.0); // exercises the saturating tail

    let key = PipelineKey::new(
        "elementwise",
        "tanh_soft_cap_f16_specialized",
        // slot 1: slot 0 is BIAS_ADD_NUM_COLS (file-scoped indices).
        vec![ConstantValue::float(1, cap)],
    );
    let pipeline = cache.get_or_build(&key).expect("tanh_soft_cap pipeline");

    let in_buf = buf_f16(&device, &x);
    let out_buf = buf_zero(&device, n * 2);
    dispatch_1d(&device, &pipeline, &[&in_buf, &out_buf], n);

    let got = read_f16(&out_buf, n);
    let mut max_err = 0f32;
    for i in 0..n {
        let xi = f16::from_f32(x[i]).to_f32();
        let want = cap * (xi / cap).tanh();
        max_err = max_err.max((got[i] - want).abs());
    }
    assert!(max_err < 0.05, "tanh_soft_cap max_err {max_err}");
}

// ── Sliding-window attention parity ─────────────────────────────────

struct AttnCase {
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
    window: i32,
}

/// CPU reference: causal (+ optional window) softmax attention over a
/// contiguous K/V laid out `[block, kv_head, tok_in_block, head_dim]`
/// (the paged layout with an identity block table).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
fn attn_ref(
    q: &[f32], // [num_q, num_q_heads, head_dim] (one row per query token)
    k: &[f32], // paged layout, f16-rounded
    v: &[f32],
    case: &AttnCase,
    block_size: usize,
    q_positions: &[usize], // absolute K-axis position per query row
    scale: f32,
) -> Vec<f32> {
    let &AttnCase {
        num_q_heads,
        num_kv_heads,
        head_dim,
        kv_len,
        window,
    } = case;
    let group = num_q_heads / num_kv_heads;
    let kv_at = |tok: usize, kvh: usize, d: usize| -> f32 {
        let blk = tok / block_size;
        let tib = tok % block_size;
        let idx = ((blk * num_kv_heads + kvh) * block_size + tib) * head_dim + d;
        k[idx]
    };
    let vv_at = |tok: usize, kvh: usize, d: usize| -> f32 {
        let blk = tok / block_size;
        let tib = tok % block_size;
        let idx = ((blk * num_kv_heads + kvh) * block_size + tib) * head_dim + d;
        v[idx]
    };
    let mut out = vec![0f32; q_positions.len() * num_q_heads * head_dim];
    for (qi, &q_abs) in q_positions.iter().enumerate() {
        for h in 0..num_q_heads {
            let kvh = h / group;
            let qrow = &q[(qi * num_q_heads + h) * head_dim..][..head_dim];
            let mut scores = Vec::with_capacity(kv_len);
            for t in 0..kv_len {
                if t > q_abs {
                    scores.push(f32::NEG_INFINITY);
                    continue;
                }
                if window > 0 && (q_abs - t) as i64 >= window as i64 {
                    scores.push(f32::NEG_INFINITY);
                    continue;
                }
                let mut s = 0f32;
                for d in 0..head_dim {
                    s += qrow[d] * kv_at(t, kvh, d);
                }
                scores.push(s * scale);
            }
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|&s| (s - m).exp()).collect();
            let denom: f32 = exps.iter().sum();
            let orow = &mut out[(qi * num_q_heads + h) * head_dim..][..head_dim];
            for t in 0..kv_len {
                if exps[t] == 0.0 {
                    continue;
                }
                let w = exps[t] / denom;
                for d in 0..head_dim {
                    orow[d] += w * vv_at(t, kvh, d);
                }
            }
        }
    }
    out
}

fn attn_constants(case: &AttnCase, block_size: usize, max_blocks: usize) -> Vec<ConstantValue> {
    vec![
        ConstantValue::uint(0, case.head_dim as u32),
        ConstantValue::uint(1, case.num_q_heads as u32),
        ConstantValue::uint(2, case.num_kv_heads as u32),
        ConstantValue::float(3, 1.0 / (case.head_dim as f32).sqrt()),
        ConstantValue::uint(4, block_size as u32),
        ConstantValue::uint(5, max_blocks as u32),
        ConstantValue::uint(6, 0), // BPC=0 single-buffer fast path
        ConstantValue::int(7, case.window),
    ]
}

fn run_window_case(case: AttnCase) {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shaders");

    let block_size = 16usize;
    let num_blocks = case.kv_len.div_ceil(block_size);
    let kv_elems = num_blocks * case.num_kv_heads * block_size * case.head_dim;

    let k_host = pseudo(101, kv_elems, 1.0);
    let v_host = pseudo(103, kv_elems, 1.0);
    let k_buf = buf_f16(&device, &k_host);
    let v_buf = buf_f16(&device, &v_host);
    // f16-rounded copies for the CPU reference
    let k_r: Vec<f32> = k_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
    let v_r: Vec<f32> = v_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();

    // chunk-address tables (BPC=0: entry 0 = buffer base)
    let k_tab = buf_u64(&device, &[k_buf.gpuAddress()]);
    let v_tab = buf_u64(&device, &[v_buf.gpuAddress()]);

    let block_table: Vec<u32> = (0..num_blocks as u32).collect();
    let bt_buf = buf_u32(&device, &block_table);
    let scale = 1.0 / (case.head_dim as f32).sqrt();

    // ── decode: 1 query at q_abs = kv_len-1 ──
    {
        let q_host = pseudo(107, case.num_q_heads * case.head_dim, 1.0);
        let q_buf = buf_f16(&device, &q_host);
        let q_r: Vec<f32> = q_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
        let out_buf = buf_zero(&device, case.num_q_heads * case.head_dim * 2);
        let seq_used = buf_u32(&device, &[case.kv_len as u32]);

        let key = PipelineKey::new(
            "attention",
            "attention_via_cache_v2_f16_specialized",
            attn_constants(&case, block_size, num_blocks),
        );
        let pipeline = cache.get_or_build(&key).expect("decode pipeline");

        // K/V are reached via raw gpuAddress through the chunk table, so
        // they must be resident even though the kernel never binds them at
        // an argument index — append them to the dispatch slice (residency
        // set covers every buffer; the extra trailing bindings are unread).
        if !common::dispatch_threadgroups(
            &device,
            &pipeline,
            &[
                &out_buf, &q_buf, &seq_used, &bt_buf, &k_tab, &v_tab, &k_buf, &v_buf,
            ],
            MTLSize {
                width: 1,
                height: case.num_q_heads,
                depth: 1,
            },
            MTLSize {
                width: 1024,
                height: 1,
                depth: 1,
            },
        ) {
            return;
        }

        let got = read_f16(&out_buf, case.num_q_heads * case.head_dim);
        let want = attn_ref(
            &q_r,
            &k_r,
            &v_r,
            &case,
            block_size,
            &[case.kv_len - 1],
            scale,
        );
        let max_err = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs())
            .fold(0f32, f32::max);
        assert!(
            max_err < 2e-2,
            "decode window={} max_err {max_err}",
            case.window
        );
    }

    // ── prefill: all kv_len queries, q_abs = row index ──
    {
        let total_q = case.kv_len;
        let q_host = pseudo(109, total_q * case.num_q_heads * case.head_dim, 1.0);
        let q_buf = buf_f16(&device, &q_host);
        let q_r: Vec<f32> = q_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
        let out_buf = buf_zero(&device, total_q * case.num_q_heads * case.head_dim * 2);
        let cu_seqlens = buf_u32(&device, &[0, total_q as u32, 0, 0]);
        let seq_used = buf_u32(&device, &[case.kv_len as u32]);

        let key = PipelineKey::new(
            "attention",
            "attention_prefill_sdpa_v2_paged_f16_specialized",
            attn_constants(&case, block_size, num_blocks),
        );
        let pipeline = cache.get_or_build(&key).expect("prefill pipeline");

        if !common::dispatch_threadgroups(
            &device,
            &pipeline,
            &[
                &out_buf,
                &q_buf,
                &cu_seqlens,
                &seq_used,
                &bt_buf,
                &k_tab,
                &v_tab,
                &k_buf,
                &v_buf,
            ],
            MTLSize {
                width: case.num_q_heads,
                height: total_q,
                depth: 1,
            },
            MTLSize {
                width: 1024,
                height: 1,
                depth: 1,
            },
        ) {
            return;
        }

        let positions: Vec<usize> = (0..total_q).collect();
        let want = attn_ref(&q_r, &k_r, &v_r, &case, block_size, &positions, scale);
        let got = read_f16(&out_buf, total_q * case.num_q_heads * case.head_dim);
        let max_err = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs())
            .fold(0f32, f32::max);
        assert!(
            max_err < 2e-2,
            "prefill window={} max_err {max_err}",
            case.window
        );
    }
}

/// Direct correctness probe for the steel PAGED prefill kernel at BK=32,
/// head_dim 128 (multi-page K-tiles) vs the CPU reference. Fast (no model).
#[test]
fn steel_paged_bk32_bd128_matches_ref() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shaders");

    let case = AttnCase {
        num_q_heads: 8,
        num_kv_heads: 8,
        head_dim: 128,
        kv_len: 40, // partial tail (kv_rem=8): 2 BK=32 tiles, last partial
        window: 0,
    };
    let block_size = 16usize;
    let num_blocks = case.kv_len.div_ceil(block_size);
    let kv_elems = num_blocks * case.num_kv_heads * block_size * case.head_dim;

    let k_host = pseudo(101, kv_elems, 1.0);
    let v_host = pseudo(103, kv_elems, 1.0);
    let k_buf = buf_f16(&device, &k_host);
    let v_buf = buf_f16(&device, &v_host);
    let k_r: Vec<f32> = k_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
    let v_r: Vec<f32> = v_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
    let k_tab = buf_u64(&device, &[k_buf.gpuAddress()]);
    let v_tab = buf_u64(&device, &[v_buf.gpuAddress()]);
    let block_table: Vec<u32> = (0..num_blocks as u32).collect();
    let bt_buf = buf_u32(&device, &block_table);
    let scale = 1.0 / (case.head_dim as f32).sqrt();

    let total_q = case.kv_len;
    let q_host = pseudo(109, total_q * case.num_q_heads * case.head_dim, 1.0);
    let q_buf = buf_f16(&device, &q_host);
    let q_r: Vec<f32> = q_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
    let out_buf = buf_zero(&device, total_q * case.num_q_heads * case.head_dim * 2);
    let cu_seqlens = buf_u32(&device, &[0, total_q as u32, 0, 0]);
    let seq_used = buf_u32(&device, &[case.kv_len as u32]);

    let mut consts = attn_constants(&case, block_size, num_blocks);
    // Steel resolve() does chunk = physical / blocks_per_chunk; attn_constants
    // uses BPC=0 (SDPA single-buffer fast path) which would divide by zero.
    // BPC >= num_blocks keeps every block in chunk 0 (chunk_table has one entry).
    consts[6] = ConstantValue::uint(6, 128);
    consts.push(ConstantValue::uint(99, 0)); // DEBUG_MODE

    let key = PipelineKey::new(
        "attention_steel_paged",
        "attention_steel_paged_f16_bq32_bk32_bd128_wm4_wn1_bs16",
        consts,
    );
    let pipeline = cache.get_or_build(&key).expect("steel paged pipeline");

    let nq_blocks = total_q.div_ceil(32);
    if !common::dispatch_threadgroups(
        &device,
        &pipeline,
        &[
            &out_buf,
            &q_buf,
            &cu_seqlens,
            &seq_used,
            &bt_buf,
            &k_tab,
            &v_tab,
            &k_buf,
            &v_buf,
        ],
        MTLSize {
            width: nq_blocks,
            height: case.num_q_heads,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    ) {
        return;
    }

    let positions: Vec<usize> = (0..total_q).collect();
    let want = attn_ref(&q_r, &k_r, &v_r, &case, block_size, &positions, scale);
    let got = read_f16(&out_buf, total_q * case.num_q_heads * case.head_dim);
    let mut max_err = 0f32;
    let mut worst = 0usize;
    for i in 0..got.len() {
        let e = (got[i] - want[i]).abs();
        if e > max_err {
            max_err = e;
            worst = i;
        }
    }
    let hd = case.head_dim;
    let nqh = case.num_q_heads;
    eprintln!(
        "steel bk32: max_err={max_err} at idx={worst} (q={}, h={}, d={}) got={} want={}",
        worst / (nqh * hd),
        (worst / hd) % nqh,
        worst % hd,
        got[worst],
        want[worst]
    );
    assert!(max_err < 2e-2, "steel bk32 bd128 max_err {max_err}");
}

/// Correctness probe for the NAX (matrix-accelerator) PAGED attention
/// kernel at BQ=64, BK=32, head_dim 128 vs the CPU reference. M5-only
/// (the runtime `newLibraryWithSource` MPP compile only produces a
/// usable library on NAX hardware); skips with a note on non-NAX GPUs.
#[test]
fn steel_nax_paged_bd128_matches_ref() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shaders");

    let case = AttnCase {
        num_q_heads: 8,
        num_kv_heads: 8,
        head_dim: 128,
        // 70 keys: 3 BK=32 tiles (32,32,6) — exercises multi-tile +
        // partial-tail (kv_rem=6) + multi-page-per-tile (2 pages/tile).
        kv_len: 70,
        window: 0,
    };
    let block_size = 16usize;
    let num_blocks = case.kv_len.div_ceil(block_size);
    let kv_elems = num_blocks * case.num_kv_heads * block_size * case.head_dim;

    let k_host = pseudo(101, kv_elems, 1.0);
    let v_host = pseudo(103, kv_elems, 1.0);
    let k_buf = buf_f16(&device, &k_host);
    let v_buf = buf_f16(&device, &v_host);
    let k_r: Vec<f32> = k_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
    let v_r: Vec<f32> = v_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
    let k_tab = buf_u64(&device, &[k_buf.gpuAddress()]);
    let v_tab = buf_u64(&device, &[v_buf.gpuAddress()]);
    let block_table: Vec<u32> = (0..num_blocks as u32).collect();
    let bt_buf = buf_u32(&device, &block_table);
    let scale = 1.0 / (case.head_dim as f32).sqrt();

    let total_q = case.kv_len;
    let q_host = pseudo(109, total_q * case.num_q_heads * case.head_dim, 1.0);
    let q_buf = buf_f16(&device, &q_host);
    let q_r: Vec<f32> = q_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
    let out_buf = buf_zero(&device, total_q * case.num_q_heads * case.head_dim * 2);
    let cu_seqlens = buf_u32(&device, &[0, total_q as u32, 0, 0]);
    let seq_used = buf_u32(&device, &[case.kv_len as u32]);

    let mut consts = attn_constants(&case, block_size, num_blocks);
    consts[6] = ConstantValue::uint(6, 128); // BLOCKS_PER_CHUNK >= num_blocks
    consts.push(ConstantValue::uint(99, 0)); // (unused by nax kernel)

    let key = PipelineKey::new(
        "attention_steel_nax_paged",
        "attention_steel_nax_paged_f16_bq64_bk32_bd128_wm4_wn1_bs16",
        consts,
    );
    let pipeline = match cache.get_or_build(&key) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("skipping steel_nax_paged: pipeline build failed (non-NAX GPU?): {e:?}");
            return;
        }
    };

    let nq_blocks = total_q.div_ceil(64);
    if !common::dispatch_threadgroups(
        &device,
        &pipeline,
        &[
            &out_buf,
            &q_buf,
            &cu_seqlens,
            &seq_used,
            &bt_buf,
            &k_tab,
            &v_tab,
            &k_buf,
            &v_buf,
        ],
        MTLSize {
            width: nq_blocks,
            height: case.num_q_heads,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    ) {
        return;
    }

    let positions: Vec<usize> = (0..total_q).collect();
    let want = attn_ref(&q_r, &k_r, &v_r, &case, block_size, &positions, scale);
    let got = read_f16(&out_buf, total_q * case.num_q_heads * case.head_dim);
    let mut max_err = 0f32;
    let mut worst = 0usize;
    for i in 0..got.len() {
        let e = (got[i] - want[i]).abs();
        if e > max_err {
            max_err = e;
            worst = i;
        }
    }
    let hd = case.head_dim;
    let nqh = case.num_q_heads;
    eprintln!(
        "steel nax paged: max_err={max_err} at idx={worst} (q={}, h={}, d={}) got={} want={}",
        worst / (nqh * hd),
        (worst / hd) % nqh,
        worst % hd,
        got[worst],
        want[worst]
    );
    assert!(max_err < 2e-2, "steel nax paged bd128 max_err {max_err}");
}

/// Limiter probe for the NAX paged attention kernel at the same shape as
/// `steel_paged_limiter_bench` (24 q / 8 kv, head_dim 128, m=4096) so the
/// TFLOP/s is directly comparable to the simdgroup baseline (~3.3). Skips
/// on non-NAX GPUs.
#[test]
fn steel_nax_paged_limiter_bench() {
    let Some(detected) = detect_device() else {
        eprintln!("skip steel_nax_paged_limiter_bench: no metal device");
        return;
    };
    let device = detected.device.clone();
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("cache");

    let case = AttnCase {
        num_q_heads: 24,
        num_kv_heads: 8,
        head_dim: 128,
        kv_len: 4096,
        window: 0,
    };
    let block_size = 16usize;
    let num_blocks = case.kv_len.div_ceil(block_size);
    let kv_elems = num_blocks * case.num_kv_heads * block_size * case.head_dim;
    let k_buf = buf_f16(&device, &pseudo(101, kv_elems, 1.0));
    let v_buf = buf_f16(&device, &pseudo(103, kv_elems, 1.0));
    let k_tab = buf_u64(&device, &[k_buf.gpuAddress()]);
    let v_tab = buf_u64(&device, &[v_buf.gpuAddress()]);
    let block_table: Vec<u32> = (0..num_blocks as u32).collect();
    let bt_buf = buf_u32(&device, &block_table);
    let total_q = case.kv_len;
    let q_buf = buf_f16(
        &device,
        &pseudo(109, total_q * case.num_q_heads * case.head_dim, 1.0),
    );
    let out_buf = buf_zero(&device, total_q * case.num_q_heads * case.head_dim * 2);
    let cu_seqlens = buf_u32(&device, &[0, total_q as u32, 0, 0]);
    let seq_used = buf_u32(&device, &[case.kv_len as u32]);

    let mut consts = attn_constants(&case, block_size, num_blocks);
    consts[6] = ConstantValue::uint(6, 128);
    consts.push(ConstantValue::uint(99, 0));
    let key = PipelineKey::new(
        "attention_steel_nax_paged",
        "attention_steel_nax_paged_f16_bq64_bk32_bd128_wm4_wn1_bs16",
        consts,
    );
    let pipeline = match cache.get_or_build(&key) {
        Ok(p) => p,
        Err(e) => {
            eprintln!(
                "skip steel_nax_paged_limiter_bench: pipeline build failed (non-NAX?): {e:?}"
            );
            return;
        }
    };

    let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
    let tew = pipeline.threadExecutionWidth();
    let smem = pipeline.staticThreadgroupMemoryLength();
    eprintln!(
        "OCCUPANCY(nax): maxTotalThreadsPerThreadgroup={max_threads} threadExecutionWidth={tew} \
         staticThreadgroupMemoryLength={smem}B (kernel launches 128 threads / 4 simdgroups)"
    );

    let nq_blocks = total_q.div_ceil(64);
    let run = |iters: usize| -> bool {
        for _ in 0..iters {
            if !common::dispatch_threadgroups(
                &device,
                &pipeline,
                &[
                    &out_buf,
                    &q_buf,
                    &cu_seqlens,
                    &seq_used,
                    &bt_buf,
                    &k_tab,
                    &v_tab,
                    &k_buf,
                    &v_buf,
                ],
                MTLSize {
                    width: nq_blocks,
                    height: case.num_q_heads,
                    depth: 1,
                },
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            ) {
                return false;
            }
        }
        true
    };

    if !run(3) {
        return;
    } // warmup
    let iters = 50usize;
    let t = std::time::Instant::now();
    run(iters);
    let secs = t.elapsed().as_secs_f64();

    let m = case.kv_len as f64;
    let flop_per_call = 2.0 * 2.0 * (m * m / 2.0) * case.head_dim as f64 * case.num_q_heads as f64;
    let tflops = flop_per_call * iters as f64 / secs / 1e12;
    let ms = secs * 1000.0 / iters as f64;
    eprintln!(
        "BENCH steel NAX paged bd128 m={} {}q/{}kv: {ms:.3} ms/call, {tflops:.2} TFLOP/s  [simdgroup baseline ~3.3]",
        case.kv_len, case.num_q_heads, case.num_kv_heads
    );
}

// ── NAX (matrix-accelerator) rope-on-read parity ────────────────────
//
// Spans on the M5: hd128 prefill selects the NAX kernel
// (`attention_steel_nax_paged`), which runs Q@K^T / P@V on the matrix
// accelerator. Rope-on-read stages each K-tile in threadgroup memory and
// re-ropes it on the ALU (per-row unrotated-flag lookup) before the NAX
// K-frag load. This test stores K UNROTATED (block_table bit 31 set) and
// asserts the kernel output bit-matches a CPU reference that rotates K to
// each key's absolute position then does attention — the same correctness
// oracle the simdgroup steel kernel's `rope_on_read_steel_*` tests use.
struct NaxRopeCase {
    name: &'static str,
    head_dim: usize, // 128 (the only NAX instantiation)
    rot_dim: usize,
    pair_off: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    kv_len: usize,
}

/// Run the NAX paged attention kernel at hd128 with rope-on-read ON
/// (block_table bit 31 set, cos_sin bound) and check elementwise against a
/// CPU reference that rotates K to its absolute position. Returns the
/// observed max_err so the caller can assert. block_size is fixed at 16
/// (the bs16 NAX instantiation); BQ=64 → 128 threads / 4 simdgroups.
fn run_rope_on_read_nax_case(case: NaxRopeCase) {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shaders");

    let NaxRopeCase {
        name,
        head_dim,
        rot_dim,
        pair_off,
        num_q_heads,
        num_kv_heads,
        kv_len,
    } = case;
    let block_size = 16usize; // NAX bs16 instantiation
    let num_blocks = kv_len.div_ceil(block_size);
    let kv_elems = num_blocks * num_kv_heads * block_size * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let k_host = pseudo(511, kv_elems, 1.0);
    let v_host = pseudo(513, kv_elems, 1.0);
    let k_buf = buf_f16(&device, &k_host);
    let v_buf = buf_f16(&device, &v_host);
    let k_tab = buf_u64(&device, &[k_buf.gpuAddress()]);
    let v_tab = buf_u64(&device, &[v_buf.gpuAddress()]);
    let k_r: Vec<f32> = k_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
    let v_r: Vec<f32> = v_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();

    let cos_sin = build_cos_sin(kv_len, rot_dim);
    let cos_sin_buf = buf_f16(&device, &cos_sin);
    let cos_sin_r: Vec<f32> = cos_sin.iter().map(|&x| f16::from_f32(x).to_f32()).collect();

    // Every block flagged unrotated → bit 31 in block_table.
    let block_table: Vec<u32> = (0..num_blocks as u32).map(|b| b | 0x8000_0000).collect();
    let bt_buf = buf_u32(&device, &block_table);

    let total_q = kv_len;
    let q_host = pseudo(517, total_q * num_q_heads * head_dim, 1.0);
    let q_buf = buf_f16(&device, &q_host);
    let q_r: Vec<f32> = q_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
    let out_buf = buf_zero(&device, total_q * num_q_heads * head_dim * 2);
    let cu_seqlens = buf_u32(&device, &[0, total_q as u32, 0, 0]);
    let seq_used = buf_u32(&device, &[kv_len as u32]);

    // Rope-once scratch (dense by logical block, same per-block strides as the
    // cache). The rope-once kernel ropes the cache's K into this; the attention
    // then reads PRE-ROPED K from here at buffer(7) (no per-tile rotation).
    let scratch_buf = buf_zero(&device, kv_elems * 2);

    // BPC >= num_blocks → all blocks in chunk 0 (nax_resolve divides by BPC).
    let consts = vec![
        ConstantValue::uint(0, head_dim as u32),
        ConstantValue::uint(1, num_q_heads as u32),
        ConstantValue::uint(2, num_kv_heads as u32),
        ConstantValue::float(3, scale),
        ConstantValue::uint(4, block_size as u32),
        ConstantValue::uint(5, num_blocks as u32),
        ConstantValue::uint(6, 128),
        ConstantValue::int(7, 0),
        ConstantValue::uint(8, rot_dim as u32),
        ConstantValue::uint(9, pair_off as u32),
        ConstantValue::uint(10, 1),
        ConstantValue::uint(99, 0),
    ];
    // Rope-once kernel reads constants 1,2,5,6,8,9 + the cache; instantiated
    // per (dtype, head_dim) like the attention kernel.
    let rope_pipe = match cache.get_or_build(&PipelineKey::new(
        "attention_steel_nax_paged",
        "rope_once_nax_f16_bd128_bs16",
        consts.clone(),
    )) {
        Ok(p) => p,
        Err(e) => {
            eprintln!(
                "skipping nax rope-on-read {name}: rope-once build failed (non-NAX GPU?): {e:?}"
            );
            return;
        }
    };
    let key = PipelineKey::new(
        "attention_steel_nax_paged",
        "attention_steel_nax_paged_f16_bq64_bk32_bd128_wm4_wn1_bs16",
        consts,
    );
    let pipeline = match cache.get_or_build(&key) {
        Ok(p) => p,
        Err(e) => {
            eprintln!(
                "skipping nax rope-on-read {name}: pipeline build failed (non-NAX GPU?): {e:?}"
            );
            return;
        }
    };

    // ── Pass 1: rope K ONCE into the scratch (k_buf read via k_tab). ──
    // Grid: x = num_kv_heads * block_size * (rot_dim/2), y = num_blocks.
    if !common::dispatch_threadgroups(
        &device,
        &rope_pipe,
        &[
            &scratch_buf,
            &bt_buf,
            &k_tab,
            &seq_used,
            &cos_sin_buf,
            &k_buf,
        ],
        MTLSize {
            width: (num_kv_heads * block_size * (rot_dim / 2)).div_ceil(64),
            height: num_blocks,
            depth: 1,
        },
        MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    ) {
        return;
    }

    // ── Pass 2: attention reads PRE-ROPED K from the scratch (v_buf via
    // v_tab). A prior dispatch fully completes before the next, so the
    // roped scratch is visible to this pass.
    let nq_blocks = total_q.div_ceil(64);
    if !common::dispatch_threadgroups(
        &device,
        &pipeline,
        &[
            &out_buf,
            &q_buf,
            &cu_seqlens,
            &seq_used,
            &bt_buf,
            &k_tab,
            &v_tab,
            &scratch_buf,
            &v_buf,
        ],
        MTLSize {
            width: nq_blocks,
            height: num_q_heads,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    ) {
        return;
    }
    let got = read_f16(&out_buf, total_q * num_q_heads * head_dim);

    // CPU reference: rotate every K vector to its absolute position, then
    // run the same causal attention.
    let mut k_rot = k_r.clone();
    for blk in 0..num_blocks {
        for kvh in 0..num_kv_heads {
            for tib in 0..block_size {
                let tok = blk * block_size + tib;
                if tok >= kv_len {
                    continue;
                }
                let off = ((blk * num_kv_heads + kvh) * block_size + tib) * head_dim;
                cpu_rope_k(
                    &mut k_rot[off..off + head_dim],
                    rot_dim,
                    pair_off,
                    tok,
                    &cos_sin_r,
                );
            }
        }
    }
    let case_ref = AttnCase {
        num_q_heads,
        num_kv_heads,
        head_dim,
        kv_len,
        window: 0,
    };
    let positions: Vec<usize> = (0..total_q).collect();
    let want = attn_ref(&q_r, &k_rot, &v_r, &case_ref, block_size, &positions, scale);
    let max_err = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .fold(0f32, f32::max);
    eprintln!("rope-on-read NAX {name}: max_err={max_err}");
    assert!(max_err < 2e-2, "rope-on-read nax {name} max_err {max_err}");
}

#[test]
fn rope_on_read_nax_full_neox_hd128() {
    // Llama-3.x-class: hd128 full NeoX, num_q==num_kv (group 1). kv_len 70
    // → 3 BK=32 tiles (32,32,6): multi-tile + partial-tail + 2-pages/tile,
    // matching `steel_nax_paged_bd128_matches_ref`'s geometry.
    run_rope_on_read_nax_case(NaxRopeCase {
        name: "full-neox-hd128",
        head_dim: 128,
        rot_dim: 128,
        pair_off: 64,
        num_q_heads: 8,
        num_kv_heads: 8,
        kv_len: 70,
    });
}

#[test]
fn rope_on_read_nax_gqa4_hd128() {
    // Llama-3.x GQA (8 q / 2 kv = 4:1). Exercises kv_head_idx routing in
    // the staged-K resolve under rope-on-read.
    run_rope_on_read_nax_case(NaxRopeCase {
        name: "gqa4-hd128",
        head_dim: 128,
        rot_dim: 128,
        pair_off: 64,
        num_q_heads: 8,
        num_kv_heads: 2,
        kv_len: 70,
    });
}

/// Limiter probe for the steel paged prefill kernel at a realistic Llama-3.2-3B
/// shape (24 q / 8 kv heads, head_dim 128, m=4096). Prints achieved TFLOP/s
/// (vs the ~9 the 4-bit GEMM gets on the same GPU) plus the compiled pipeline's
/// occupancy facts (max threads, smem) so we can tell saturation from starvation.
#[test]
fn steel_paged_limiter_bench() {
    let Some(detected) = detect_device() else {
        eprintln!("skip steel_paged_limiter_bench: no metal device");
        return;
    };
    let device = detected.device.clone();
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("cache");

    let case = AttnCase {
        num_q_heads: 24,
        num_kv_heads: 8,
        head_dim: 128,
        kv_len: 4096,
        window: 0,
    };
    let block_size = 16usize;
    let num_blocks = case.kv_len.div_ceil(block_size);
    let kv_elems = num_blocks * case.num_kv_heads * block_size * case.head_dim;
    let k_buf = buf_f16(&device, &pseudo(101, kv_elems, 1.0));
    let v_buf = buf_f16(&device, &pseudo(103, kv_elems, 1.0));
    let k_tab = buf_u64(&device, &[k_buf.gpuAddress()]);
    let v_tab = buf_u64(&device, &[v_buf.gpuAddress()]);
    let block_table: Vec<u32> = (0..num_blocks as u32).collect();
    let bt_buf = buf_u32(&device, &block_table);
    let total_q = case.kv_len;
    let q_buf = buf_f16(
        &device,
        &pseudo(109, total_q * case.num_q_heads * case.head_dim, 1.0),
    );
    let out_buf = buf_zero(&device, total_q * case.num_q_heads * case.head_dim * 2);
    let cu_seqlens = buf_u32(&device, &[0, total_q as u32, 0, 0]);
    let seq_used = buf_u32(&device, &[case.kv_len as u32]);

    let mut consts = attn_constants(&case, block_size, num_blocks);
    consts[6] = ConstantValue::uint(6, 128);
    consts.push(ConstantValue::uint(99, 0));
    let key = PipelineKey::new(
        "attention_steel_paged",
        "attention_steel_paged_f16_bq32_bk32_bd128_wm4_wn1_bs16",
        consts,
    );
    let pipeline = cache.get_or_build(&key).expect("pipeline");

    let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
    let tew = pipeline.threadExecutionWidth();
    let smem = pipeline.staticThreadgroupMemoryLength();
    eprintln!(
        "OCCUPANCY: maxTotalThreadsPerThreadgroup={max_threads} threadExecutionWidth={tew} \
         staticThreadgroupMemoryLength={smem}B (kernel launches 128 threads / 4 simdgroups)"
    );

    let nq_blocks = total_q.div_ceil(32);
    let run = |iters: usize| -> bool {
        for _ in 0..iters {
            if !common::dispatch_threadgroups(
                &device,
                &pipeline,
                &[
                    &out_buf,
                    &q_buf,
                    &cu_seqlens,
                    &seq_used,
                    &bt_buf,
                    &k_tab,
                    &v_tab,
                    &k_buf,
                    &v_buf,
                ],
                MTLSize {
                    width: nq_blocks,
                    height: case.num_q_heads,
                    depth: 1,
                },
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            ) {
                return false;
            }
        }
        true
    };

    if !run(3) {
        return;
    } // warmup
    let iters = 50usize;
    let t = std::time::Instant::now();
    run(iters);
    let secs = t.elapsed().as_secs_f64();

    // causal self-attention, one layer, all heads:
    // QK^T and A·V each do (m*m/2) causal pairs * head_dim MACs; 2 matmuls, 2 flop/MAC.
    let m = case.kv_len as f64;
    let flop_per_call = 2.0 * 2.0 * (m * m / 2.0) * case.head_dim as f64 * case.num_q_heads as f64;
    let tflops = flop_per_call * iters as f64 / secs / 1e12;
    let ms = secs * 1000.0 / iters as f64;
    eprintln!(
        "BENCH steel bk32 bd128 m={} {}q/{}kv: {ms:.3} ms/call, {tflops:.2} TFLOP/s  [4-bit GEMM ~9 on this GPU]",
        case.kv_len, case.num_q_heads, case.num_kv_heads
    );
}

/// Steel prefill rope-on-read overhead: times the steel paged prefill at a
/// SmolLM-135M-like shape (head_dim 64, gqa 3) in two (+1 optional) configs —
///   (a) ROR OFF              : steel, no re-rope (the baseline)
///   (b) ROR ON (pipelined)   : steel re-ropes each staged K tile on read.
///                              For bd64 this takes the SOFTWARE-PIPELINED
///                              path (double-buffered K, next-block rope
///                              issued before the current block's PV so the
///                              cos_sin loads + rotate overlap the matmul).
///   (c) ROR ON, GQA-folded   : per-kv-head grid re-ropes each K tile ONCE
///                              and fans across the group. OPTIONAL — only
///                              built when STEEL_BENCH_FOLD=1 (the folded
///                              kernel variant is a separate, unbuilt feature;
///                              `get_or_build` on a missing symbol aborts).
/// The ratio (b)/(a) is the spans prefill TTFT overhead this change targets.
/// Every block is flagged unrotated so the full re-rope cost is exercised.
#[test]
fn steel_prefill_rope_on_read_fold_bench() {
    let Some(detected) = detect_device() else {
        eprintln!("skip steel_prefill_rope_on_read_fold_bench: no metal device");
        return;
    };
    let device = detected.device.clone();
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("cache");

    let head_dim = 64usize;
    let num_q_heads = 9usize;
    let num_kv_heads = 3usize;
    let gqa = num_q_heads / num_kv_heads;
    let kv_len = std::env::var("STEEL_BENCH_M")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096usize);
    let rot_dim = 64usize;
    let pair_off = 32usize;
    let block_size = 16usize;
    let num_blocks = kv_len.div_ceil(block_size);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_elems = num_blocks * num_kv_heads * block_size * head_dim;

    let k_buf = buf_f16(&device, &pseudo(401, kv_elems, 1.0));
    let v_buf = buf_f16(&device, &pseudo(403, kv_elems, 1.0));
    let k_tab = buf_u64(&device, &[k_buf.gpuAddress()]);
    let v_tab = buf_u64(&device, &[v_buf.gpuAddress()]);
    // Flagged (bit 31 set) → every block re-roped on read = the real span cost.
    let bt_flagged: Vec<u32> = (0..num_blocks as u32).map(|b| b | 0x8000_0000).collect();
    let bt_buf = buf_u32(&device, &bt_flagged);
    let total_q = kv_len;
    let q_buf = buf_f16(&device, &pseudo(409, total_q * num_q_heads * head_dim, 1.0));
    let out_buf = buf_zero(&device, total_q * num_q_heads * head_dim * 2);
    let cu_seqlens = buf_u32(&device, &[0, total_q as u32, 0, 0]);
    let seq_used = buf_u32(&device, &[kv_len as u32]);
    let cos_sin = buf_f16(&device, &build_cos_sin(kv_len, rot_dim));

    let base_consts = || {
        vec![
            ConstantValue::uint(0, head_dim as u32),
            ConstantValue::uint(1, num_q_heads as u32),
            ConstantValue::uint(2, num_kv_heads as u32),
            ConstantValue::float(3, scale),
            ConstantValue::uint(4, block_size as u32),
            ConstantValue::uint(5, num_blocks as u32),
            ConstantValue::uint(6, 128),
            ConstantValue::int(7, 0),
            ConstantValue::uint(99, 0),
        ]
    };
    let ror_consts = || {
        let mut c = base_consts();
        c.push(ConstantValue::uint(8, rot_dim as u32));
        c.push(ConstantValue::uint(9, pair_off as u32));
        c.push(ConstantValue::uint(10, 1));
        c
    };
    let off_pipe = cache
        .get_or_build(&PipelineKey::new(
            "attention_steel_paged",
            "attention_steel_paged_f16_bq32_bk16_bd64_wm4_wn1_bs16",
            base_consts(),
        ))
        .expect("off pipe");
    let on_unfolded_pipe = cache
        .get_or_build(&PipelineKey::new(
            "attention_steel_paged",
            "attention_steel_paged_f16_bq32_bk16_bd64_wm4_wn1_bs16",
            ror_consts(),
        ))
        .expect("on unfolded pipe");
    // GQA-folded variant is a separate, not-yet-built kernel; building a
    // missing symbol aborts inside Metal, so this is opt-in.
    let on_folded_pipe = if std::env::var("STEEL_BENCH_FOLD").is_ok() {
        Some(
            cache
                .get_or_build(&PipelineKey::new(
                    "attention_steel_paged",
                    "attention_steel_paged_ror_fold_f16_bq32_bk16_bd64_wm4_wn1_bs16",
                    ror_consts(),
                ))
                .expect("on folded pipe"),
        )
    } else {
        None
    };

    let nq_blocks = total_q.div_ceil(32);
    let run = |pipeline: &common::Pipeline, grid_y: usize, iters: usize| -> bool {
        for _ in 0..iters {
            if !common::dispatch_threadgroups(
                &device,
                pipeline,
                &[
                    &out_buf,
                    &q_buf,
                    &cu_seqlens,
                    &seq_used,
                    &bt_buf,
                    &k_tab,
                    &v_tab,
                    &cos_sin,
                    &k_buf,
                    &v_buf,
                ],
                MTLSize {
                    width: nq_blocks,
                    height: grid_y,
                    depth: 1,
                },
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            ) {
                return false;
            }
        }
        true
    };

    // Warmup.
    if !run(&off_pipe, num_q_heads, 3) {
        return;
    }
    run(&on_unfolded_pipe, num_q_heads, 3);
    if let Some(p) = &on_folded_pipe {
        run(p, num_kv_heads, 3);
    }

    let iters = 50usize;
    let time = |pipe: &common::Pipeline, grid_y: usize| {
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = std::time::Instant::now();
            run(pipe, grid_y, iters);
            best = best.min(t.elapsed().as_secs_f64() / iters as f64);
        }
        best
    };
    let off = time(&off_pipe, num_q_heads);
    let on_unf = time(&on_unfolded_pipe, num_q_heads);
    let on_fold = on_folded_pipe.as_ref().map(|p| time(p, num_kv_heads));
    eprintln!(
        "BENCH steel prefill ROR hd{head_dim} {num_q_heads}q/{num_kv_heads}kv (gqa{gqa}) m={kv_len}:\n\
         \x20  OFF (baseline)         {:.3} ms/call\n\
         \x20  ON pipelined           {:.3} ms/call  ratio {:.4}x\n\
         \x20  ON gqa-folded          {}",
        off * 1e3,
        on_unf * 1e3,
        on_unf / off,
        match on_fold {
            Some(f) => format!("{:.3} ms/call  ratio {:.4}x", f * 1e3, f / off),
            None => "(set STEEL_BENCH_FOLD=1; folded kernel not built)".to_string(),
        },
    );
}

#[test]
fn attention_window_disabled_matches_causal_ref() {
    run_window_case(AttnCase {
        num_q_heads: 4,
        num_kv_heads: 2,
        head_dim: 64,
        kv_len: 50,
        window: 0,
    });
}

#[test]
fn attention_window_masks_old_keys() {
    // window smaller than kv_len: the mask actively drops keys.
    run_window_case(AttnCase {
        num_q_heads: 4,
        num_kv_heads: 2,
        head_dim: 64,
        kv_len: 50,
        window: 8,
    });
}

#[test]
fn attention_window_gemma4_sliding_geometry() {
    // Gemma4 sliding-layer geometry: 16 q-heads, 8 kv-heads, head_dim
    // 256, window 1024 — with kv_len > window so the mask is active.
    run_window_case(AttnCase {
        num_q_heads: 16,
        num_kv_heads: 8,
        head_dim: 256,
        kv_len: 1100,
        window: 1024,
    });
}

#[test]
fn rmsnorm_unit_f16_matches_cpu() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shaders");

    // Gemma4 v_norm shapes: rows = T*kv_heads, width = head_dim.
    let (rows, width) = (24usize, 256usize);
    let eps = 1e-6f32;
    let x = pseudo(31, rows * width, 2.0);

    let key = PipelineKey::new(
        "rmsnorm",
        "rmsnorm_unit_f16_specialized",
        vec![
            ConstantValue::uint(0, rows as u32),
            ConstantValue::uint(1, width as u32),
            ConstantValue::float(2, eps),
        ],
    );
    let pipeline = cache.get_or_build(&key).expect("rmsnorm_unit pipeline");
    let in_buf = buf_f16(&device, &x);
    let out_buf = buf_zero(&device, rows * width * 2);

    if !common::dispatch_threadgroups(
        &device,
        &pipeline,
        &[&out_buf, &in_buf],
        MTLSize {
            width: rows,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    ) {
        return;
    }

    let got = read_f16(&out_buf, rows * width);
    let mut max_err = 0f32;
    for r in 0..rows {
        let row: Vec<f32> = (0..width)
            .map(|i| f16::from_f32(x[r * width + i]).to_f32())
            .collect();
        let ms = row.iter().map(|v| v * v).sum::<f32>() / width as f32;
        let rms = (ms + eps).sqrt();
        for i in 0..width {
            let want = row[i] / rms;
            max_err = max_err.max((got[r * width + i] - want).abs());
        }
    }
    assert!(max_err < 5e-3, "rmsnorm_unit max_err {max_err}");
}

#[test]
fn scalar_weight_mul_f16_matches_cpu() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shaders");

    let n = 3840usize;
    let scalar = 0.937f32; // a plausible layer_scalar value
    let x = pseudo(37, n, 3.0);
    let key = PipelineKey::new(
        "elementwise",
        "scalar_weight_mul_f16_specialized",
        Vec::new(),
    );
    let pipeline = cache
        .get_or_build(&key)
        .expect("scalar_weight_mul pipeline");
    let in_buf = buf_f16(&device, &x);
    let w_buf = buf_f16(&device, &[scalar]);
    let out_buf = buf_zero(&device, n * 2);
    dispatch_1d(&device, &pipeline, &[&out_buf, &in_buf, &w_buf], n);

    let got = read_f16(&out_buf, n);
    let w = f16::from_f32(scalar).to_f32();
    let mut max_err = 0f32;
    for i in 0..n {
        let want = f16::from_f32(x[i]).to_f32() * w;
        max_err = max_err.max((got[i] - want).abs());
    }
    assert!(max_err < 5e-3, "scalar_weight_mul max_err {max_err}");
}

// ── Rope-on-read (spans / position-independent KV) parity ────────────
//
// Span (relocatable) K is stored UNROTATED; the decode kernel re-ropes
// each cached K to the reader's own position on read. This proves the
// read-path rotation in the REAL kernel == a fresh CPU rope at the same
// position — the only correctness oracle the spans handoff trusts (a
// mocked rotation can't catch a fn-const / shuffle-pairing bug).
// Parameterized over the two gemma4 rope variants:
//   * full NeoX   (sliding hd256): pair_off = rot_dim/2, every dim rotates
//   * proportional (global hd512): pair_off = head_dim/2 > rot_dim/2, only
//     the first rot_dim/2 of each half rotate; the rest pass through.

struct RopeCase {
    name: &'static str,
    head_dim: usize,
    rot_dim: usize,
    pair_off: usize,
    num_heads: usize,
    kv_len: usize,
}

/// cos_sin table `[max_pos, rot_dim]` = `[cos(0..half); sin(0..half)]` per
/// row, matching `rope_append`'s layout (rope.metal). theta is arbitrary
/// here: the SAME table feeds both the kernel (buffer 7) and the CPU
/// reference, so the test isolates the rotation math/pairing/shuffle, not
/// the production cos_sin generation.
fn build_cos_sin(max_pos: usize, rot_dim: usize) -> Vec<f32> {
    let half = rot_dim / 2;
    let theta = 10000.0f32;
    let mut t = vec![0f32; max_pos * rot_dim];
    for pos in 0..max_pos {
        for d in 0..half {
            let inv_freq = 1.0 / theta.powf(2.0 * d as f32 / rot_dim as f32);
            let ang = pos as f32 * inv_freq;
            t[pos * rot_dim + d] = ang.cos();
            t[pos * rot_dim + half + d] = ang.sin();
        }
    }
    t
}

/// CPU NeoX rope on one K vector at `pos`, matching `rope_append` and the
/// kernel's `rope_on_read_k_slice` EXACTLY: pair `d` with `d + pair_off`,
/// rotary index `d` in `[0, half)`; dims outside `[0,half) ∪
/// [pair_off, pair_off+half)` pass through (proportional rope).
fn cpu_rope_k(k: &mut [f32], rot_dim: usize, pair_off: usize, pos: usize, cos_sin: &[f32]) {
    let half = rot_dim / 2;
    let base = pos * rot_dim;
    for d in 0..half {
        let c = cos_sin[base + d];
        let s = cos_sin[base + half + d];
        let x0 = k[d];
        let x1 = k[pair_off + d];
        k[d] = x0 * c - x1 * s;
        k[pair_off + d] = x1 * c + x0 * s;
    }
}

fn run_rope_on_read_case(case: RopeCase) {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shaders");

    let RopeCase {
        name,
        head_dim,
        rot_dim,
        pair_off,
        num_heads,
        kv_len,
    } = case;
    let block_size = 16usize;
    let num_blocks = kv_len.div_ceil(block_size);
    let kv_elems = num_blocks * num_heads * block_size * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Stored (UNROTATED) K + V, plus f16-rounded host copies for the ref.
    let k_host = pseudo(101, kv_elems, 1.0);
    let v_host = pseudo(103, kv_elems, 1.0);
    let k_buf = buf_f16(&device, &k_host);
    let v_buf = buf_f16(&device, &v_host);
    let k_tab = buf_u64(&device, &[k_buf.gpuAddress()]);
    let v_tab = buf_u64(&device, &[v_buf.gpuAddress()]);
    let k_r: Vec<f32> = k_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
    let v_r: Vec<f32> = v_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();

    // cos_sin (f16-rounded so the CPU rope matches the kernel's reads).
    let cos_sin = build_cos_sin(kv_len, rot_dim);
    let cos_sin_buf = buf_f16(&device, &cos_sin);
    let cos_sin_r: Vec<f32> = cos_sin.iter().map(|&x| f16::from_f32(x).to_f32()).collect();

    // Every block flagged unrotated → rope-on-read for every key. The
    // flag rides in block_table bit 31 (identity ids | 0x8000_0000).
    let block_table: Vec<u32> = (0..num_blocks as u32).map(|b| b | 0x8000_0000).collect();
    let bt_buf = buf_u32(&device, &block_table);

    let consts = vec![
        ConstantValue::uint(0, head_dim as u32),
        ConstantValue::uint(1, num_heads as u32),
        ConstantValue::uint(2, num_heads as u32), // group=1 (rotation is per-KV-head)
        ConstantValue::float(3, scale),
        ConstantValue::uint(4, block_size as u32),
        ConstantValue::uint(5, num_blocks as u32), // max_blocks_per_seq
        ConstantValue::uint(6, 0),                 // BPC=0 single-buffer fast path
        ConstantValue::int(7, 0),                  // window disabled
        ConstantValue::uint(8, rot_dim as u32),    // ATTN_ROT_DIM
        ConstantValue::uint(9, pair_off as u32),   // ATTN_PAIR_OFF
        ConstantValue::uint(10, 1),                // ATTN_ROPE_ON_READ
    ];
    let key = PipelineKey::new(
        "attention",
        "attention_via_cache_v2_f16_specialized",
        consts,
    );
    let pipeline = cache
        .get_or_build(&key)
        .expect("decode rope-on-read pipeline");

    // Decode: one query at q_abs = kv_len-1.
    let q_host = pseudo(107, num_heads * head_dim, 1.0);
    let q_buf = buf_f16(&device, &q_host);
    let q_r: Vec<f32> = q_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
    let out_buf = buf_zero(&device, num_heads * head_dim * 2);
    let seq_used = buf_u32(&device, &[kv_len as u32]);

    if !common::dispatch_threadgroups(
        &device,
        &pipeline,
        &[
            &out_buf,
            &q_buf,
            &seq_used,
            &bt_buf,
            &k_tab,
            &v_tab,
            &cos_sin_buf,
            &k_buf,
            &v_buf,
        ],
        MTLSize {
            width: 1,
            height: num_heads,
            depth: 1,
        },
        MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        },
    ) {
        return;
    }
    let got = read_f16(&out_buf, num_heads * head_dim);

    // Reference: rope each cached K at its position (= token index), then
    // run plain causal attention — the "fresh rope at the reuse position".
    let mut k_rot = k_r.clone();
    for blk in 0..num_blocks {
        for kvh in 0..num_heads {
            for tib in 0..block_size {
                let tok = blk * block_size + tib;
                if tok >= kv_len {
                    continue;
                }
                let off = ((blk * num_heads + kvh) * block_size + tib) * head_dim;
                cpu_rope_k(
                    &mut k_rot[off..off + head_dim],
                    rot_dim,
                    pair_off,
                    tok,
                    &cos_sin_r,
                );
            }
        }
    }
    let case_ref = AttnCase {
        num_q_heads: num_heads,
        num_kv_heads: num_heads,
        head_dim,
        kv_len,
        window: 0,
    };
    let want = attn_ref(
        &q_r,
        &k_rot,
        &v_r,
        &case_ref,
        block_size,
        &[kv_len - 1],
        scale,
    );

    let max_err = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .fold(0f32, f32::max);
    assert!(
        max_err < 2e-2,
        "rope-on-read {name} decode max_err {max_err}"
    );

    // ── prefill (register-path): all kv_len queries at q_abs = row ──
    // Same unrotated cache + flags; the register prefill kernel must
    // rope-on-read identically to decode.
    {
        let total_q = kv_len;
        let pf_consts = vec![
            ConstantValue::uint(0, head_dim as u32),
            ConstantValue::uint(1, num_heads as u32),
            ConstantValue::uint(2, num_heads as u32),
            ConstantValue::float(3, scale),
            ConstantValue::uint(4, block_size as u32),
            ConstantValue::uint(5, num_blocks as u32),
            ConstantValue::uint(6, 0),
            ConstantValue::int(7, 0),
            ConstantValue::uint(8, rot_dim as u32),
            ConstantValue::uint(9, pair_off as u32),
            ConstantValue::uint(10, 1),
        ];
        let pf_key = PipelineKey::new(
            "attention",
            "attention_prefill_sdpa_v2_paged_f16_specialized",
            pf_consts,
        );
        let pf_pipeline = cache
            .get_or_build(&pf_key)
            .expect("prefill rope-on-read pipeline");

        let q_pf = pseudo(109, total_q * num_heads * head_dim, 1.0);
        let q_pf_buf = buf_f16(&device, &q_pf);
        let q_pf_r: Vec<f32> = q_pf.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
        let out_pf = buf_zero(&device, total_q * num_heads * head_dim * 2);
        let cu_seqlens = buf_u32(&device, &[0, total_q as u32, 0, 0]);
        let seq_used_pf = buf_u32(&device, &[kv_len as u32]);
        // Per-token span labels at buffer(8): the kernel reads span_ids under
        // ATTN_ROR. All-zero = no span (attend everything); without it the
        // kernel reads whatever is bound there as labels and empties its
        // attention loop (all-zero output).
        let span_ids_pf = buf_u32(&device, &vec![0u32; kv_len]);

        if !common::dispatch_threadgroups(
            &device,
            &pf_pipeline,
            &[
                &out_pf,
                &q_pf_buf,
                &cu_seqlens,
                &seq_used_pf,
                &bt_buf,
                &k_tab,
                &v_tab,
                &cos_sin_buf,
                &span_ids_pf,
                &k_buf,
                &v_buf,
            ],
            MTLSize {
                width: num_heads,
                height: total_q,
                depth: 1,
            },
            MTLSize {
                width: 1024,
                height: 1,
                depth: 1,
            },
        ) {
            return;
        }
        let got_pf = read_f16(&out_pf, total_q * num_heads * head_dim);

        let positions: Vec<usize> = (0..total_q).collect();
        let want_pf = attn_ref(
            &q_pf_r, &k_rot, &v_r, &case_ref, block_size, &positions, scale,
        );
        let max_err_pf = got_pf
            .iter()
            .zip(&want_pf)
            .map(|(g, w)| (g - w).abs())
            .fold(0f32, f32::max);
        assert!(
            max_err_pf < 2e-2,
            "rope-on-read {name} prefill max_err {max_err_pf}"
        );
    }
}

#[test]
fn rope_on_read_decode_full_neox_hd256() {
    // gemma4 sliding-layer geometry: full NeoX, pair_off = rot_dim/2.
    run_rope_on_read_case(RopeCase {
        name: "full-neox-hd256",
        head_dim: 256,
        rot_dim: 256,
        pair_off: 128,
        num_heads: 2,
        kv_len: 40,
    });
}

#[test]
fn rope_on_read_decode_proportional_hd512() {
    // gemma4 global-layer geometry: proportional rope, pair_off =
    // head_dim/2 > rot_dim/2; dims [64,256) and [320,512) pass through.
    run_rope_on_read_case(RopeCase {
        name: "proportional-hd512",
        head_dim: 512,
        rot_dim: 128,
        pair_off: 256,
        num_heads: 2,
        kv_len: 40,
    });
}

#[test]
fn rope_on_read_decode_full_neox_hd128() {
    // Llama-class full rope; kv_len 33 spans 3 blocks (non-aligned tail).
    run_rope_on_read_case(RopeCase {
        name: "full-neox-hd128",
        head_dim: 128,
        rot_dim: 128,
        pair_off: 64,
        num_heads: 4,
        kv_len: 33,
    });
}

/// gqa_shared (global-prefill) rope-on-read: K is staged through smem and
/// rotated IN smem (not the register/shuffle path). Needs real GQA
/// (2 <= gqa <= 32). Proves the in-smem rotation == fresh CPU rope.
struct GqaRopeCase {
    name: &'static str,
    head_dim: usize,
    rot_dim: usize,
    pair_off: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    kv_len: usize,
}

fn run_rope_on_read_gqa_shared_case(case: GqaRopeCase) {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shaders");

    let GqaRopeCase {
        name,
        head_dim,
        rot_dim,
        pair_off,
        num_q_heads,
        num_kv_heads,
        kv_len,
    } = case;
    let block_size = 16usize; // gqa_shared: one sub-stage (block_size <= 16)
    let gqa = num_q_heads / num_kv_heads;
    assert!((2..=32).contains(&gqa), "gqa_shared needs 2..=32");
    let num_blocks = kv_len.div_ceil(block_size);
    let kv_elems = num_blocks * num_kv_heads * block_size * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let k_host = pseudo(211, kv_elems, 1.0);
    let v_host = pseudo(213, kv_elems, 1.0);
    let k_buf = buf_f16(&device, &k_host);
    let v_buf = buf_f16(&device, &v_host);
    let k_tab = buf_u64(&device, &[k_buf.gpuAddress()]);
    let v_tab = buf_u64(&device, &[v_buf.gpuAddress()]);
    let k_r: Vec<f32> = k_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
    let v_r: Vec<f32> = v_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();

    let cos_sin = build_cos_sin(kv_len, rot_dim);
    let cos_sin_buf = buf_f16(&device, &cos_sin);
    let cos_sin_r: Vec<f32> = cos_sin.iter().map(|&x| f16::from_f32(x).to_f32()).collect();

    // Every block flagged unrotated → flag in block_table bit 31.
    let block_table: Vec<u32> = (0..num_blocks as u32).map(|b| b | 0x8000_0000).collect();
    let bt_buf = buf_u32(&device, &block_table);

    let total_q = kv_len;
    let q_host = pseudo(217, total_q * num_q_heads * head_dim, 1.0);
    let q_buf = buf_f16(&device, &q_host);
    let q_r: Vec<f32> = q_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
    let out_buf = buf_zero(&device, total_q * num_q_heads * head_dim * 2);
    let cu_seqlens = buf_u32(&device, &[0, total_q as u32, 0, 0]);
    let seq_used = buf_u32(&device, &[kv_len as u32]);

    let consts = vec![
        ConstantValue::uint(0, head_dim as u32),
        ConstantValue::uint(1, num_q_heads as u32),
        ConstantValue::uint(2, num_kv_heads as u32),
        ConstantValue::float(3, scale),
        ConstantValue::uint(4, block_size as u32),
        ConstantValue::uint(5, num_blocks as u32),
        ConstantValue::uint(6, 0),
        ConstantValue::int(7, 0),
        ConstantValue::uint(8, rot_dim as u32),
        ConstantValue::uint(9, pair_off as u32),
        ConstantValue::uint(10, 1),
    ];
    let key = PipelineKey::new(
        "attention",
        "attention_prefill_sdpa_gqa_shared_f16_specialized",
        consts,
    );
    let pipeline = cache
        .get_or_build(&key)
        .expect("gqa_shared rope-on-read pipeline");

    if !common::dispatch_threadgroups(
        &device,
        &pipeline,
        &[
            &out_buf,
            &q_buf,
            &cu_seqlens,
            &seq_used,
            &bt_buf,
            &k_tab,
            &v_tab,
            &cos_sin_buf,
            &k_buf,
            &v_buf,
        ],
        MTLSize {
            width: num_kv_heads,
            height: total_q,
            depth: 1,
        },
        MTLSize {
            width: 32 * gqa,
            height: 1,
            depth: 1,
        },
    ) {
        return;
    }
    let got = read_f16(&out_buf, total_q * num_q_heads * head_dim);

    // Reference: rope each cached K (per kv-head) at its position.
    let mut k_rot = k_r.clone();
    for blk in 0..num_blocks {
        for kvh in 0..num_kv_heads {
            for tib in 0..block_size {
                let tok = blk * block_size + tib;
                if tok >= kv_len {
                    continue;
                }
                let off = ((blk * num_kv_heads + kvh) * block_size + tib) * head_dim;
                cpu_rope_k(
                    &mut k_rot[off..off + head_dim],
                    rot_dim,
                    pair_off,
                    tok,
                    &cos_sin_r,
                );
            }
        }
    }
    let case_ref = AttnCase {
        num_q_heads,
        num_kv_heads,
        head_dim,
        kv_len,
        window: 0,
    };
    let positions: Vec<usize> = (0..total_q).collect();
    let want = attn_ref(&q_r, &k_rot, &v_r, &case_ref, block_size, &positions, scale);
    let max_err = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .fold(0f32, f32::max);
    assert!(
        max_err < 2e-2,
        "rope-on-read gqa_shared {name} max_err {max_err}"
    );
}

#[test]
fn rope_on_read_gqa_shared_full_neox_hd256() {
    run_rope_on_read_gqa_shared_case(GqaRopeCase {
        name: "full-neox-hd256",
        head_dim: 256,
        rot_dim: 256,
        pair_off: 128,
        num_q_heads: 8,
        num_kv_heads: 4,
        kv_len: 40,
    });
}

#[test]
fn rope_on_read_gqa_shared_proportional_hd512() {
    // gemma4 global-prefill geometry: 16:1 GQA, proportional rope.
    run_rope_on_read_gqa_shared_case(GqaRopeCase {
        name: "proportional-hd512",
        head_dim: 512,
        rot_dim: 128,
        pair_off: 256,
        num_q_heads: 16,
        num_kv_heads: 1,
        kv_len: 40,
    });
}

/// gqa_shared ROPE-ONCE-TO-SCRATCH rope-on-read: K is roped ONCE into a dense
/// logical-block-indexed scratch by `rope_once_gqa_shared_*` (pass 1), then the
/// gqa_shared attention reads PRE-ROPED K from the scratch at buffer 7 with
/// ATTN_K_SCRATCH (slot 11) set and NO per-tile smem rotation (pass 2). This is
/// the gqa_shared twin of `run_rope_on_read_steel_gqa_case`. Proves the two-pass
/// flow == fresh CPU rope, at the gemma4 global geometry (hd 512, gqa 16,
/// proportional rope, page-unified block_size 32).
struct GqaScratchRopeCase {
    name: &'static str,
    head_dim: usize,
    rot_dim: usize,
    pair_off: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    block_size: usize,
    kv_len: usize,
}

fn run_rope_once_gqa_shared_case(case: GqaScratchRopeCase) {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shaders");

    let GqaScratchRopeCase {
        name,
        head_dim,
        rot_dim,
        pair_off,
        num_q_heads,
        num_kv_heads,
        block_size,
        kv_len,
    } = case;
    let gqa = num_q_heads / num_kv_heads;
    assert!((8..=32).contains(&gqa), "gqa_shared rope-once needs 8..=32");
    let num_blocks = kv_len.div_ceil(block_size);
    let kv_elems = num_blocks * num_kv_heads * block_size * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let k_host = pseudo(311, kv_elems, 1.0);
    let v_host = pseudo(313, kv_elems, 1.0);
    let k_buf = buf_f16(&device, &k_host);
    let v_buf = buf_f16(&device, &v_host);
    let k_tab = buf_u64(&device, &[k_buf.gpuAddress()]);
    let v_tab = buf_u64(&device, &[v_buf.gpuAddress()]);
    let k_r: Vec<f32> = k_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
    let v_r: Vec<f32> = v_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();

    let cos_sin = build_cos_sin(kv_len, rot_dim);
    let cos_sin_buf = buf_f16(&device, &cos_sin);
    let cos_sin_r: Vec<f32> = cos_sin.iter().map(|&x| f16::from_f32(x).to_f32()).collect();

    // Every block flagged unrotated → bit 31 in block_table.
    let block_table: Vec<u32> = (0..num_blocks as u32).map(|b| b | 0x8000_0000).collect();
    let bt_buf = buf_u32(&device, &block_table);

    let total_q = kv_len;
    let q_host = pseudo(317, total_q * num_q_heads * head_dim, 1.0);
    let q_buf = buf_f16(&device, &q_host);
    let q_r: Vec<f32> = q_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
    let out_buf = buf_zero(&device, total_q * num_q_heads * head_dim * 2);
    let cu_seqlens = buf_u32(&device, &[0, total_q as u32, 0, 0]);
    let seq_used = buf_u32(&device, &[kv_len as u32]);
    // Dense pre-roped K scratch, logical-block indexed (same per-block strides
    // as the cache). f16 → 2 B/elem.
    let scratch_buf = buf_zero(&device, kv_elems * 2);

    // BLOCKS_PER_CHUNK 0 (single chunk) — the kernel's `== 0` fast path.
    let consts = vec![
        ConstantValue::uint(0, head_dim as u32),
        ConstantValue::uint(1, num_q_heads as u32),
        ConstantValue::uint(2, num_kv_heads as u32),
        ConstantValue::float(3, scale),
        ConstantValue::uint(4, block_size as u32),
        ConstantValue::uint(5, num_blocks as u32),
        ConstantValue::uint(6, 0),
        ConstantValue::int(7, 0),
        ConstantValue::uint(8, rot_dim as u32),
        ConstantValue::uint(9, pair_off as u32),
        ConstantValue::uint(10, 1),
        ConstantValue::uint(11, 1), // ATTN_K_SCRATCH
    ];
    let rope_key = PipelineKey::new(
        "attention",
        "rope_once_gqa_shared_f16_specialized",
        consts.clone(),
    );
    let attn_key = PipelineKey::new(
        "attention",
        "attention_prefill_sdpa_gqa_shared_f16_specialized",
        consts,
    );
    let rope_pipe = cache.get_or_build(&rope_key).expect("rope_once_gqa_shared");
    let attn_pipe = cache
        .get_or_build(&attn_key)
        .expect("gqa_shared scratch attn");

    // ── Pass 1: rope K ONCE into the scratch (k_buf read via k_tab). ──
    // Grid: x = num_kv_heads * block_size * (rot_dim/2), y = num_blocks.
    if !common::dispatch_threadgroups(
        &device,
        &rope_pipe,
        &[
            &scratch_buf,
            &bt_buf,
            &k_tab,
            &seq_used,
            &cos_sin_buf,
            &k_buf,
        ],
        MTLSize {
            width: (num_kv_heads * block_size * (rot_dim / 2)).div_ceil(64),
            height: num_blocks,
            depth: 1,
        },
        MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    ) {
        return;
    }

    // ── Pass 2: gqa_shared attention reads PRE-ROPED K from the scratch
    // (v_buf via v_tab); pass 1 has fully completed.
    if !common::dispatch_threadgroups(
        &device,
        &attn_pipe,
        &[
            &out_buf,
            &q_buf,
            &cu_seqlens,
            &seq_used,
            &bt_buf,
            &k_tab,
            &v_tab,
            &scratch_buf,
            &v_buf,
        ],
        MTLSize {
            width: num_kv_heads,
            height: total_q,
            depth: 1,
        },
        MTLSize {
            width: 32 * gqa,
            height: 1,
            depth: 1,
        },
    ) {
        return;
    }
    let got = read_f16(&out_buf, total_q * num_q_heads * head_dim);

    // Reference: rope each cached K (per kv-head) at its position, then GQA attn.
    let mut k_rot = k_r.clone();
    for blk in 0..num_blocks {
        for kvh in 0..num_kv_heads {
            for tib in 0..block_size {
                let tok = blk * block_size + tib;
                if tok >= kv_len {
                    continue;
                }
                let off = ((blk * num_kv_heads + kvh) * block_size + tib) * head_dim;
                cpu_rope_k(
                    &mut k_rot[off..off + head_dim],
                    rot_dim,
                    pair_off,
                    tok,
                    &cos_sin_r,
                );
            }
        }
    }
    let case_ref = AttnCase {
        num_q_heads,
        num_kv_heads,
        head_dim,
        kv_len,
        window: 0,
    };
    let positions: Vec<usize> = (0..total_q).collect();
    let want = attn_ref(&q_r, &k_rot, &v_r, &case_ref, block_size, &positions, scale);
    let max_err = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .fold(0f32, f32::max);
    assert!(
        max_err < 2e-2,
        "rope-once gqa_shared {name} max_err {max_err}"
    );
}

#[test]
fn rope_once_gqa_shared_proportional_hd512_gemma4() {
    // gemma4 global-prefill geometry: 16:1 GQA, proportional rope (rot_dim
    // 128, pair_off head_dim/2 = 256), page-unified block_size 32.
    run_rope_once_gqa_shared_case(GqaScratchRopeCase {
        name: "proportional-hd512-bs32",
        head_dim: 512,
        rot_dim: 128,
        pair_off: 256,
        num_q_heads: 16,
        num_kv_heads: 1,
        block_size: 32,
        kv_len: 70, // spans 3 blocks of 32 (last partial) — exercises the tail
    });
}

#[test]
fn rope_once_gqa_shared_full_neox_hd256() {
    // Full NeoX (rot_dim == head_dim, pair_off = rot_dim/2), block_size 16.
    run_rope_once_gqa_shared_case(GqaScratchRopeCase {
        name: "full-neox-hd256-bs16",
        head_dim: 256,
        rot_dim: 256,
        pair_off: 128,
        num_q_heads: 8,
        num_kv_heads: 1,
        block_size: 16,
        kv_len: 40,
    });
}

/// steel (FA-2 tiled prefill) rope-on-read: K is staged into smem in a
/// transposed tile (K(r,d)=Ks[r+d*LDK]) before the MMA, and a BK-wide
/// tile can span >1 paged block (bk=32, block_size=16) — so the test
/// exercises the kernel's PER-ROW unrotated-flag lookup. block_size is
/// fixed at 16 (the bs16 instantiation). Full NeoX only (steel is not
/// instantiated for hd 512 / proportional).
struct SteelRopeCase {
    name: &'static str,
    head_dim: usize,
    bk: usize,
    rot_dim: usize,
    pair_off: usize,
    num_heads: usize,
    kv_len: usize,
}

fn run_rope_on_read_steel_case(case: SteelRopeCase) {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shaders");

    let SteelRopeCase {
        name,
        head_dim,
        bk,
        rot_dim,
        pair_off,
        num_heads,
        kv_len,
    } = case;
    let block_size = 16usize; // steel bs16 instantiation
    let num_blocks = kv_len.div_ceil(block_size);
    let kv_elems = num_blocks * num_heads * block_size * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let k_host = pseudo(311, kv_elems, 1.0);
    let v_host = pseudo(313, kv_elems, 1.0);
    let k_buf = buf_f16(&device, &k_host);
    let v_buf = buf_f16(&device, &v_host);
    let k_tab = buf_u64(&device, &[k_buf.gpuAddress()]);
    let v_tab = buf_u64(&device, &[v_buf.gpuAddress()]);
    let k_r: Vec<f32> = k_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
    let v_r: Vec<f32> = v_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();

    let cos_sin = build_cos_sin(kv_len, rot_dim);
    let cos_sin_buf = buf_f16(&device, &cos_sin);
    let cos_sin_r: Vec<f32> = cos_sin.iter().map(|&x| f16::from_f32(x).to_f32()).collect();

    // Every block flagged unrotated → flag in block_table bit 31.
    let block_table: Vec<u32> = (0..num_blocks as u32).map(|b| b | 0x8000_0000).collect();
    let bt_buf = buf_u32(&device, &block_table);

    let total_q = kv_len;
    let q_host = pseudo(317, total_q * num_heads * head_dim, 1.0);
    let q_buf = buf_f16(&device, &q_host);
    let q_r: Vec<f32> = q_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
    let out_buf = buf_zero(&device, total_q * num_heads * head_dim * 2);
    let cu_seqlens = buf_u32(&device, &[0, total_q as u32, 0, 0]);
    let seq_used = buf_u32(&device, &[kv_len as u32]);

    // Rope-once scratch (dense by logical block, same per-block strides as the
    // cache). `rope_once_steel` ropes the cache's K into this; the attention
    // then reads PRE-ROPED K from here at buffer(7) (no per-tile rotation).
    let scratch_buf = buf_zero(&device, kv_elems * 2);

    // BPC >= num_blocks → all blocks in chunk 0 (steel divides by BPC).
    let consts = vec![
        ConstantValue::uint(0, head_dim as u32),
        ConstantValue::uint(1, num_heads as u32),
        ConstantValue::uint(2, num_heads as u32),
        ConstantValue::float(3, scale),
        ConstantValue::uint(4, block_size as u32),
        ConstantValue::uint(5, num_blocks as u32),
        ConstantValue::uint(6, 128),
        ConstantValue::int(7, 0),
        ConstantValue::uint(8, rot_dim as u32),
        ConstantValue::uint(9, pair_off as u32),
        ConstantValue::uint(10, 1),
        ConstantValue::uint(99, 0), // DEBUG_MODE
    ];
    let symbol: &'static str = match (head_dim, bk) {
        (64, 16) => "attention_steel_paged_f16_bq32_bk16_bd64_wm4_wn1_bs16",
        (128, 32) => "attention_steel_paged_f16_bq32_bk32_bd128_wm4_wn1_bs16",
        (256, 16) => "attention_steel_paged_f16_bq32_bk16_bd256_wm4_wn1_bs16",
        _ => panic!("no steel instantiation for hd={head_dim} bk={bk}"),
    };
    // rope-once kernel (same library, head-dim-instantiated alongside the body).
    let rope_sym: &'static str = match head_dim {
        64 => "rope_once_steel_f16_bd64_bs16",
        96 => "rope_once_steel_f16_bd96_bs16",
        128 => "rope_once_steel_f16_bd128_bs16",
        256 => "rope_once_steel_f16_bd256_bs16",
        _ => panic!("no rope_once_steel instantiation for hd={head_dim}"),
    };
    let rope_pipe = cache
        .get_or_build(&PipelineKey::new(
            "attention_steel_paged",
            rope_sym,
            consts.clone(),
        ))
        .expect("rope_once_steel pipeline");
    let key = PipelineKey::new("attention_steel_paged", symbol, consts);
    let pipeline = cache
        .get_or_build(&key)
        .expect("steel rope-on-read pipeline");

    // ── Pass 1: rope K ONCE into the scratch (k_buf read via k_tab). ──
    // Grid: x = num_heads * block_size * (rot_dim/2), y = num_blocks.
    if !common::dispatch_threadgroups(
        &device,
        &rope_pipe,
        &[
            &scratch_buf,
            &bt_buf,
            &k_tab,
            &seq_used,
            &cos_sin_buf,
            &k_buf,
        ],
        MTLSize {
            width: (num_heads * block_size * (rot_dim / 2)).div_ceil(64),
            height: num_blocks,
            depth: 1,
        },
        MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    ) {
        return;
    }

    // ── Pass 2: attention reads PRE-ROPED K from the scratch (v_buf via
    // v_tab); pass 1 has fully completed.
    let nq_blocks = total_q.div_ceil(32);
    if !common::dispatch_threadgroups(
        &device,
        &pipeline,
        &[
            &out_buf,
            &q_buf,
            &cu_seqlens,
            &seq_used,
            &bt_buf,
            &k_tab,
            &v_tab,
            &scratch_buf,
            &v_buf,
        ],
        MTLSize {
            width: nq_blocks,
            height: num_heads,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    ) {
        return;
    }
    let got = read_f16(&out_buf, total_q * num_heads * head_dim);

    let mut k_rot = k_r.clone();
    for blk in 0..num_blocks {
        for kvh in 0..num_heads {
            for tib in 0..block_size {
                let tok = blk * block_size + tib;
                if tok >= kv_len {
                    continue;
                }
                let off = ((blk * num_heads + kvh) * block_size + tib) * head_dim;
                cpu_rope_k(
                    &mut k_rot[off..off + head_dim],
                    rot_dim,
                    pair_off,
                    tok,
                    &cos_sin_r,
                );
            }
        }
    }
    let case_ref = AttnCase {
        num_q_heads: num_heads,
        num_kv_heads: num_heads,
        head_dim,
        kv_len,
        window: 0,
    };
    let positions: Vec<usize> = (0..total_q).collect();
    let want = attn_ref(&q_r, &k_rot, &v_r, &case_ref, block_size, &positions, scale);
    let max_err = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .fold(0f32, f32::max);
    assert!(
        max_err < 2e-2,
        "rope-on-read steel {name} max_err {max_err}"
    );
}

#[test]
fn rope_on_read_steel_full_neox_hd128_bk32() {
    // bk=32 spans 2 paged blocks per tile → exercises per-row flag lookup.
    run_rope_on_read_steel_case(SteelRopeCase {
        name: "full-neox-hd128-bk32",
        head_dim: 128,
        bk: 32,
        rot_dim: 128,
        pair_off: 64,
        num_heads: 8,
        kv_len: 40,
    });
}

#[test]
fn rope_on_read_steel_full_neox_hd256_bk16() {
    // gemma4 sliding-prefill geometry (hd256 full NeoX), bk=16.
    run_rope_on_read_steel_case(SteelRopeCase {
        name: "full-neox-hd256-bk16",
        head_dim: 256,
        bk: 16,
        rot_dim: 256,
        pair_off: 128,
        num_heads: 8,
        kv_len: 40,
    });
}

#[test]
fn rope_on_read_steel_full_neox_hd64_bk16() {
    // SmolLM hd64 — the geometry whose double-buffered K footprint fits the
    // 32 KiB threadgroup budget, so this exercises the SOFTWARE-PIPELINED
    // rope path (Ks[2] + next-block rope overlapping the PV matmul).
    // kv_len 40 spans 3 blocks → multi-iteration pipeline + partial tail.
    run_rope_on_read_steel_case(SteelRopeCase {
        name: "full-neox-hd64-bk16",
        head_dim: 64,
        bk: 16,
        rot_dim: 64,
        pair_off: 32,
        num_heads: 3,
        kv_len: 40,
    });
}

/// steel GQA rope-on-read (rope-once-to-scratch): exercises REAL GQA
/// (2 <= gqa <= 8) through the production simdgroup steel path. With
/// rope-once-to-scratch the GQA fan-out is transparent — `rope_once_steel`
/// ropes K once PER KV-HEAD into the scratch, and the attention reads it for
/// every q-head in the group via `kv_head_idx = q_head_idx / gqa`. A gqa=1
/// test (the cases above) cannot catch a per-kv-head scratch-offset bug; this
/// validates the q→kv head mapping against a fresh CPU rope.
struct SteelGqaRopeCase {
    name: &'static str,
    head_dim: usize,
    bk: usize,
    rot_dim: usize,
    pair_off: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    kv_len: usize,
}

fn run_rope_on_read_steel_gqa_case(case: SteelGqaRopeCase) {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shaders");

    let SteelGqaRopeCase {
        name,
        head_dim,
        bk,
        rot_dim,
        pair_off,
        num_q_heads,
        num_kv_heads,
        kv_len,
    } = case;
    let block_size = 16usize; // steel bs16 instantiation
    let gqa = num_q_heads / num_kv_heads;
    assert!((2..=8).contains(&gqa), "steel fold needs 2..=8");
    let num_blocks = kv_len.div_ceil(block_size);
    let kv_elems = num_blocks * num_kv_heads * block_size * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let k_host = pseudo(331, kv_elems, 1.0);
    let v_host = pseudo(337, kv_elems, 1.0);
    let k_buf = buf_f16(&device, &k_host);
    let v_buf = buf_f16(&device, &v_host);
    let k_tab = buf_u64(&device, &[k_buf.gpuAddress()]);
    let v_tab = buf_u64(&device, &[v_buf.gpuAddress()]);
    let k_r: Vec<f32> = k_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
    let v_r: Vec<f32> = v_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();

    let cos_sin = build_cos_sin(kv_len, rot_dim);
    let cos_sin_buf = buf_f16(&device, &cos_sin);
    let cos_sin_r: Vec<f32> = cos_sin.iter().map(|&x| f16::from_f32(x).to_f32()).collect();

    // Every block flagged unrotated → flag in block_table bit 31.
    let block_table: Vec<u32> = (0..num_blocks as u32).map(|b| b | 0x8000_0000).collect();
    let bt_buf = buf_u32(&device, &block_table);

    let total_q = kv_len;
    let q_host = pseudo(339, total_q * num_q_heads * head_dim, 1.0);
    let q_buf = buf_f16(&device, &q_host);
    let q_r: Vec<f32> = q_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
    let out_buf = buf_zero(&device, total_q * num_q_heads * head_dim * 2);
    let cu_seqlens = buf_u32(&device, &[0, total_q as u32, 0, 0]);
    let seq_used = buf_u32(&device, &[kv_len as u32]);

    // Rope-once scratch (dense by logical block; mirrors the cache's per-block
    // strides INCLUDING the num_kv_heads dimension, so a q-head reads its
    // group's roped K via kv_head_idx = q_head_idx / gqa).
    let scratch_buf = buf_zero(&device, kv_elems * 2);

    // BPC >= num_blocks → all blocks in chunk 0 (steel divides by BPC).
    let consts = vec![
        ConstantValue::uint(0, head_dim as u32),
        ConstantValue::uint(1, num_q_heads as u32),
        ConstantValue::uint(2, num_kv_heads as u32),
        ConstantValue::float(3, scale),
        ConstantValue::uint(4, block_size as u32),
        ConstantValue::uint(5, num_blocks as u32),
        ConstantValue::uint(6, 128),
        ConstantValue::int(7, 0),
        ConstantValue::uint(8, rot_dim as u32),
        ConstantValue::uint(9, pair_off as u32),
        ConstantValue::uint(10, 1),
        ConstantValue::uint(99, 0), // DEBUG_MODE
    ];
    let symbol: &'static str = match (head_dim, bk) {
        (64, 16) => "attention_steel_paged_f16_bq32_bk16_bd64_wm4_wn1_bs16",
        (128, 32) => "attention_steel_paged_f16_bq32_bk32_bd128_wm4_wn1_bs16",
        (256, 16) => "attention_steel_paged_f16_bq32_bk16_bd256_wm4_wn1_bs16",
        _ => panic!("no steel instantiation for hd={head_dim} bk={bk}"),
    };
    let rope_sym: &'static str = match head_dim {
        64 => "rope_once_steel_f16_bd64_bs16",
        96 => "rope_once_steel_f16_bd96_bs16",
        128 => "rope_once_steel_f16_bd128_bs16",
        256 => "rope_once_steel_f16_bd256_bs16",
        _ => panic!("no rope_once_steel instantiation for hd={head_dim}"),
    };
    let rope_pipe = cache
        .get_or_build(&PipelineKey::new(
            "attention_steel_paged",
            rope_sym,
            consts.clone(),
        ))
        .expect("rope_once_steel pipeline");
    let key = PipelineKey::new("attention_steel_paged", symbol, consts);
    let pipeline = cache
        .get_or_build(&key)
        .expect("steel gqa rope-on-read pipeline");

    // ── Pass 1: rope K ONCE into the scratch (per kv-head; k_buf via k_tab). ──
    if !common::dispatch_threadgroups(
        &device,
        &rope_pipe,
        &[
            &scratch_buf,
            &bt_buf,
            &k_tab,
            &seq_used,
            &cos_sin_buf,
            &k_buf,
        ],
        MTLSize {
            width: (num_kv_heads * block_size * (rot_dim / 2)).div_ceil(64),
            height: num_blocks,
            depth: 1,
        },
        MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    ) {
        return;
    }

    // ── Pass 2: GQA attention reads PRE-ROPED K from the scratch (v_buf via
    // v_tab); pass 1 has fully completed. Steel grid Y axis is PER-Q-HEAD;
    // the kernel maps kv_head = q_head / gqa.
    let nq_blocks = total_q.div_ceil(32);
    if !common::dispatch_threadgroups(
        &device,
        &pipeline,
        &[
            &out_buf,
            &q_buf,
            &cu_seqlens,
            &seq_used,
            &bt_buf,
            &k_tab,
            &v_tab,
            &scratch_buf,
            &v_buf,
        ],
        MTLSize {
            width: nq_blocks,
            height: num_q_heads,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    ) {
        return;
    }
    let got = read_f16(&out_buf, total_q * num_q_heads * head_dim);

    // Reference: rope each cached K (per kv-head) at its position, then run
    // GQA causal softmax attention (each q_head h attends kv_head h/gqa).
    let mut k_rot = k_r.clone();
    for blk in 0..num_blocks {
        for kvh in 0..num_kv_heads {
            for tib in 0..block_size {
                let tok = blk * block_size + tib;
                if tok >= kv_len {
                    continue;
                }
                let off = ((blk * num_kv_heads + kvh) * block_size + tib) * head_dim;
                cpu_rope_k(
                    &mut k_rot[off..off + head_dim],
                    rot_dim,
                    pair_off,
                    tok,
                    &cos_sin_r,
                );
            }
        }
    }
    let case_ref = AttnCase {
        num_q_heads,
        num_kv_heads,
        head_dim,
        kv_len,
        window: 0,
    };
    let positions: Vec<usize> = (0..total_q).collect();
    let want = attn_ref(&q_r, &k_rot, &v_r, &case_ref, block_size, &positions, scale);
    let max_err = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .fold(0f32, f32::max);
    assert!(
        max_err < 2e-2,
        "rope-on-read steel-fold {name} max_err {max_err}"
    );
}

#[test]
fn rope_on_read_steel_gqa3_hd64_bk16() {
    // SmolLM-135M geometry: head_dim 64, gqa 3 (the actual spans target).
    // kv_len 40 spans 3 blocks → exercises the partial-tail K-tile too.
    run_rope_on_read_steel_gqa_case(SteelGqaRopeCase {
        name: "gqa3-hd64-bk16",
        head_dim: 64,
        bk: 16,
        rot_dim: 64,
        pair_off: 32,
        num_q_heads: 9,
        num_kv_heads: 3,
        kv_len: 40,
    });
}

#[test]
fn rope_on_read_steel_gqa4_hd128_bk32() {
    // Llama-3.x-class: head_dim 128, gqa 4, bk=32 (K-tile spans 2 blocks →
    // per-row unrotated-flag lookup inside the once-per-kb rope).
    run_rope_on_read_steel_gqa_case(SteelGqaRopeCase {
        name: "gqa4-hd128-bk32",
        head_dim: 128,
        bk: 32,
        rot_dim: 128,
        pair_off: 64,
        num_q_heads: 8,
        num_kv_heads: 2,
        kv_len: 40,
    });
}

// ── Write-path (rope_append store-unrotated) round-trip ──────────────
//
// The keystone the read-only tests leave open: prove that rope_append
// actually STORES relocatable K unrotated, and that feeding that exact
// GPU cache through the rope-on-read attention kernel reproduces a fresh
// rope at the reuse position. Also asserts the DEFAULT write path (no
// ROPE_ON_READ const) is byte-correct (still rotates) — the non-spans
// parity guard. Full NeoX, hd128, single sequence, identity slot/block.
#[test]
fn rope_append_store_unrotated_then_read_roundtrip() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shaders");

    let head_dim = 128usize;
    let rot_dim = 128usize;
    let pair_off = 64usize; // full NeoX
    let num_heads = 4usize; // num_q == num_kv (group 1) for rope_append + decode
    let block_size = 16usize;
    let kv_len = 33usize; // 3 blocks, non-aligned tail
    let num_blocks = kv_len.div_ceil(block_size);
    let bpc = 128u32; // chunk 0 (rope_append divides by BPC)

    // Raw (pre-rope) K + V, [token, kv_head, head_dim].
    let k_in = pseudo(401, kv_len * num_heads * head_dim, 1.0);
    let v_in = pseudo(403, kv_len * num_heads * head_dim, 1.0);
    let q_in = pseudo(405, kv_len * num_heads * head_dim, 1.0); // rope_append rotates this in place (unused by us)
    let k_in_r: Vec<f32> = k_in.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
    let v_in_r: Vec<f32> = v_in.iter().map(|&x| f16::from_f32(x).to_f32()).collect();

    let cos_sin = build_cos_sin(kv_len, rot_dim);
    let cos_sin_buf = buf_f16(&device, &cos_sin);
    let cos_sin_r: Vec<f32> = cos_sin.iter().map(|&x| f16::from_f32(x).to_f32()).collect();

    let positions: Vec<u32> = (0..kv_len as u32).collect();
    let pos_buf = buf_u32(&device, &positions);
    let slot_mapping: Vec<u32> = (0..kv_len as u32).collect(); // identity (plain)
    let slot_buf = buf_u32(&device, &slot_mapping);
    // Spans: slot_mapping with bit 31 set = store K unrotated (write path).
    let slot_flagged: Vec<u32> = (0..kv_len as u32).map(|s| s | 0x8000_0000).collect();
    let slot_buf_flagged = buf_u32(&device, &slot_flagged);

    let cache_elems = num_blocks * num_heads * block_size * head_dim;

    // Dispatch rope_append_f16 into a fresh KV cache. `ror`: Some(1) sets
    // the spans const (store unrotated for slot_mapping bit-31 slots);
    // None omits slot 9 (default rotate-on-write). `slot` carries the
    // bit-31 flags. Returns the stored K cache (f16->f32).
    let run_rope_append = |ror: Option<u32>, slot: &Buffer| -> Vec<f32> {
        let kv_k = buf_zero(&device, cache_elems * 2);
        let kv_v = buf_zero(&device, cache_elems * 2);
        let kv_k_tab = buf_u64(&device, &[kv_k.gpuAddress()]);
        let kv_v_tab = buf_u64(&device, &[kv_v.gpuAddress()]);
        let q_buf = buf_f16(&device, &q_in);
        let k_buf = buf_f16(&device, &k_in);
        let v_buf = buf_f16(&device, &v_in);

        let mut consts = vec![
            ConstantValue::uint(0, head_dim as u32),
            ConstantValue::uint(1, num_heads as u32),
            ConstantValue::uint(2, num_heads as u32),
            ConstantValue::uint(3, rot_dim as u32),
            ConstantValue::uint(4, block_size as u32),
            ConstantValue::uint(5, bpc),
            ConstantValue::uint(6, pair_off as u32),
        ];
        if let Some(v) = ror {
            consts.push(ConstantValue::uint(9, v)); // ROPE_ROPE_ON_READ
        }
        let key = PipelineKey::new("rope", "rope_append_f16_specialized", consts);
        let pipeline = cache.get_or_build(&key).expect("rope_append pipeline");

        // kv_k/kv_v are WRITTEN via raw gpuAddress through the chunk tables,
        // so they ride in the dispatch slice to be made resident.
        common::dispatch_threadgroups(
            &device,
            &pipeline,
            &[
                &q_buf,
                &k_buf,
                &v_buf,
                &cos_sin_buf,
                &pos_buf,
                slot,
                &kv_k_tab,
                &kv_v_tab,
                &kv_k,
                &kv_v,
            ],
            MTLSize {
                width: kv_len,
                height: num_heads,
                depth: 1,
            },
            MTLSize {
                width: head_dim,
                height: 1,
                depth: 1,
            },
        );
        read_f16(&kv_k, cache_elems)
    };

    // Paged cache index for (token, kv_head, d).
    let cache_idx = |tok: usize, kvh: usize, d: usize| -> usize {
        let blk = tok / block_size;
        let off = tok % block_size;
        ((blk * num_heads + kvh) * block_size + off) * head_dim + d
    };
    // Raw input index for (token, kv_head, d).
    let in_idx =
        |tok: usize, kvh: usize, d: usize| -> usize { (tok * num_heads + kvh) * head_dim + d };

    // (A) store-unrotated: stored cache K == raw input K (no rotation).
    let stored_unrot = run_rope_append(Some(1), &slot_buf_flagged);
    let mut max_a = 0f32;
    for tok in 0..kv_len {
        for kvh in 0..num_heads {
            for d in 0..head_dim {
                let got = stored_unrot[cache_idx(tok, kvh, d)];
                let want = k_in_r[in_idx(tok, kvh, d)];
                max_a = max_a.max((got - want).abs());
            }
        }
    }
    assert!(
        max_a < 1e-3,
        "store-unrotated: cache K != raw input K (max {max_a})"
    );

    // (C) parity: default write (no ROPE_ON_READ const) still ROTATES,
    // byte-correct vs CPU rope at the write position. Plain slot_mapping.
    let stored_rot = run_rope_append(None, &slot_buf);
    let mut max_c = 0f32;
    for tok in 0..kv_len {
        for kvh in 0..num_heads {
            let mut krow: Vec<f32> = (0..head_dim).map(|d| k_in_r[in_idx(tok, kvh, d)]).collect();
            cpu_rope_k(&mut krow, rot_dim, pair_off, tok, &cos_sin_r);
            for d in 0..head_dim {
                let got = stored_rot[cache_idx(tok, kvh, d)];
                max_c = max_c.max((got - krow[d]).abs());
            }
        }
    }
    assert!(
        max_c < 2e-2,
        "default rope_append drifted from rotate-on-write (max {max_c})"
    );

    // (B) round-trip: feed the UNROTATED cache through rope-on-read decode
    // attention; must match attn_ref over CPU-re-roped raw K at pos = key.
    let kv_k = buf_zero(&device, cache_elems * 2);
    let kv_v = buf_zero(&device, cache_elems * 2);
    let kv_k_tab = buf_u64(&device, &[kv_k.gpuAddress()]);
    let kv_v_tab = buf_u64(&device, &[kv_v.gpuAddress()]);
    {
        // Re-run rope_append (unrotated) into THIS cache (the closure used
        // its own buffers; here we need the persistent kv_k for attention).
        let q_buf = buf_f16(&device, &q_in);
        let k_buf = buf_f16(&device, &k_in);
        let v_buf = buf_f16(&device, &v_in);
        let consts = vec![
            ConstantValue::uint(0, head_dim as u32),
            ConstantValue::uint(1, num_heads as u32),
            ConstantValue::uint(2, num_heads as u32),
            ConstantValue::uint(3, rot_dim as u32),
            ConstantValue::uint(4, block_size as u32),
            ConstantValue::uint(5, bpc),
            ConstantValue::uint(6, pair_off as u32),
            ConstantValue::uint(9, 1),
        ];
        let key = PipelineKey::new("rope", "rope_append_f16_specialized", consts);
        let pipeline = cache.get_or_build(&key).expect("rope_append pipeline");
        if !common::dispatch_threadgroups(
            &device,
            &pipeline,
            &[
                &q_buf,
                &k_buf,
                &v_buf,
                &cos_sin_buf,
                &pos_buf,
                &slot_buf_flagged,
                &kv_k_tab,
                &kv_v_tab,
                &kv_k,
                &kv_v,
            ],
            MTLSize {
                width: kv_len,
                height: num_heads,
                depth: 1,
            },
            MTLSize {
                width: head_dim,
                height: 1,
                depth: 1,
            },
        ) {
            return;
        }
    }

    // Decode attention (rope-on-read) over the unrotated cache.
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q_dec = pseudo(407, num_heads * head_dim, 1.0);
    let q_dec_buf = buf_f16(&device, &q_dec);
    let q_dec_r: Vec<f32> = q_dec.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
    let out_buf = buf_zero(&device, num_heads * head_dim * 2);
    let seq_used = buf_u32(&device, &[kv_len as u32]);
    // Attention reads with bit-31 flag set (all blocks unrotated) so it
    // re-ropes the cache rope_append wrote unrotated.
    let block_table: Vec<u32> = (0..num_blocks as u32).map(|b| b | 0x8000_0000).collect();
    let bt_buf = buf_u32(&device, &block_table);
    let consts = vec![
        ConstantValue::uint(0, head_dim as u32),
        ConstantValue::uint(1, num_heads as u32),
        ConstantValue::uint(2, num_heads as u32),
        ConstantValue::float(3, scale),
        ConstantValue::uint(4, block_size as u32),
        ConstantValue::uint(5, num_blocks as u32),
        ConstantValue::uint(6, 0), // BPC=0 single-buffer fast path (attention)
        ConstantValue::int(7, 0),
        ConstantValue::uint(8, rot_dim as u32),
        ConstantValue::uint(9, pair_off as u32),
        ConstantValue::uint(10, 1),
    ];
    let key = PipelineKey::new(
        "attention",
        "attention_via_cache_v2_f16_specialized",
        consts,
    );
    let pipeline = cache.get_or_build(&key).expect("decode pipeline");
    // attention BPC=0 expects k_cache[0] = layer base; bind the cache buffer
    // directly via a 1-entry chunk table whose entry is the buffer base.
    // kv_k/kv_v are read via raw gpuAddress through the chunk tables → ride
    // in the dispatch slice for residency.
    if !common::dispatch_threadgroups(
        &device,
        &pipeline,
        &[
            &out_buf,
            &q_dec_buf,
            &seq_used,
            &bt_buf,
            &kv_k_tab,
            &kv_v_tab,
            &cos_sin_buf,
            &kv_k,
            &kv_v,
        ],
        MTLSize {
            width: 1,
            height: num_heads,
            depth: 1,
        },
        MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        },
    ) {
        return;
    }
    let got = read_f16(&out_buf, num_heads * head_dim);

    // Reference: paged K with each token's raw K roped at pos = token idx.
    let mut k_paged = vec![0f32; cache_elems];
    for tok in 0..kv_len {
        for kvh in 0..num_heads {
            let mut krow: Vec<f32> = (0..head_dim).map(|d| k_in_r[in_idx(tok, kvh, d)]).collect();
            cpu_rope_k(&mut krow, rot_dim, pair_off, tok, &cos_sin_r);
            for d in 0..head_dim {
                k_paged[cache_idx(tok, kvh, d)] = krow[d];
            }
        }
    }
    let mut v_paged = vec![0f32; cache_elems];
    for tok in 0..kv_len {
        for kvh in 0..num_heads {
            for d in 0..head_dim {
                v_paged[cache_idx(tok, kvh, d)] = v_in_r[in_idx(tok, kvh, d)];
            }
        }
    }
    let case_ref = AttnCase {
        num_q_heads: num_heads,
        num_kv_heads: num_heads,
        head_dim,
        kv_len,
        window: 0,
    };
    let want = attn_ref(
        &q_dec_r,
        &k_paged,
        &v_paged,
        &case_ref,
        block_size,
        &[kv_len - 1],
        scale,
    );
    let max_b = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .fold(0f32, f32::max);
    assert!(max_b < 2e-2, "write->read round-trip drifted (max {max_b})");
}

// ── Rope-on-read PARITY bench (general, model-independent) ───────────
//
// The perf guard the user mandated: enabling ROPE_ON_READ compiles the
// decode kernel WITH the per-key flag-load + branch in the K-loop, paid
// by EVERY spans-enabled model's decode even on non-span requests. This
// times the SAME kernel (attention_via_cache_v2) ROR-off vs ROR-on with
// an ALL-ZERO flag buffer (the non-span case) on the real device and
// asserts the rope-on-read specialization does not regress the hot path.
// Prints the ratio; fails on a gross (>15%) regression — the bar.
#[test]
fn rope_on_read_decode_parity_bench() {
    let Some(di) = detect_device() else {
        eprintln!("skip rope_on_read_decode_parity_bench: no metal device");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shaders");

    let head_dim = 128usize; // Llama/Qwen-class (general, not gemma4-specific)
    let num_heads = 32usize;
    let num_kv = 8usize;
    let block_size = 16usize;
    let kv_len = 8192usize; // long context → flag-load-per-key cost is visible
    let rot_dim = 128usize;
    let pair_off = 64usize;
    let num_blocks = kv_len.div_ceil(block_size);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_elems = num_blocks * num_kv * block_size * head_dim;

    let k_buf = buf_f16(&device, &pseudo(101, kv_elems, 1.0));
    let v_buf = buf_f16(&device, &pseudo(103, kv_elems, 1.0));
    let k_tab = buf_u64(&device, &[k_buf.gpuAddress()]);
    let v_tab = buf_u64(&device, &[v_buf.gpuAddress()]);
    let block_table: Vec<u32> = (0..num_blocks as u32).collect();
    let bt_buf = buf_u32(&device, &block_table);
    // Flagged block_table (every entry bit-31 set) → do_rot=true → EVERY cached
    // key is re-roped on read. This is the REAL span decode cost (launch-claude
    // re-ropes its whole cached span every token), vs the all-clear case which
    // only measures the bit-packing machinery.
    let bt_flagged: Vec<u32> = (0..num_blocks as u32).map(|b| b | 0x8000_0000).collect();
    let bt_buf_flagged = buf_u32(&device, &bt_flagged);
    let q_buf = buf_f16(&device, &pseudo(107, num_heads * head_dim, 1.0));
    let out_buf = buf_zero(&device, num_heads * head_dim * 2);
    let seq_used = buf_u32(&device, &[kv_len as u32]);
    // Non-span case: block_table has no bit-31 flags set (bt_buf above is
    // plain physical ids), so do_rot is always false on the ROR-on path.
    let cos_sin = buf_f16(&device, &build_cos_sin(kv_len, rot_dim));

    let base_consts = || {
        vec![
            ConstantValue::uint(0, head_dim as u32),
            ConstantValue::uint(1, num_heads as u32),
            ConstantValue::uint(2, num_kv as u32),
            ConstantValue::float(3, scale),
            ConstantValue::uint(4, block_size as u32),
            ConstantValue::uint(5, num_blocks as u32),
            ConstantValue::uint(6, 0),
            ConstantValue::int(7, 0),
        ]
    };
    let off_pipe = cache
        .get_or_build(&PipelineKey::new(
            "attention",
            "attention_via_cache_v2_f16_specialized",
            base_consts(),
        ))
        .expect("ROR-off pipeline");
    let mut on_consts = base_consts();
    on_consts.push(ConstantValue::uint(8, rot_dim as u32));
    on_consts.push(ConstantValue::uint(9, pair_off as u32));
    on_consts.push(ConstantValue::uint(10, 1));
    let on_pipe = cache
        .get_or_build(&PipelineKey::new(
            "attention",
            "attention_via_cache_v2_f16_specialized",
            on_consts,
        ))
        .expect("ROR-on pipeline");

    let run = |pipeline: &common::Pipeline, ror: bool, bt: &Buffer, iters: usize| -> bool {
        // ROR-on binds cos_sin at slot 6 (the flag rides in block_table bit
        // 31). K/V are reached via raw gpuAddress → ride in the slice for
        // residency.
        for _ in 0..iters {
            let ok = if ror {
                common::dispatch_threadgroups(
                    &device,
                    pipeline,
                    &[
                        &out_buf, &q_buf, &seq_used, bt, &k_tab, &v_tab, &cos_sin, &k_buf, &v_buf,
                    ],
                    MTLSize {
                        width: 1,
                        height: num_heads,
                        depth: 1,
                    },
                    MTLSize {
                        width: 1024,
                        height: 1,
                        depth: 1,
                    },
                )
            } else {
                common::dispatch_threadgroups(
                    &device,
                    pipeline,
                    &[
                        &out_buf, &q_buf, &seq_used, bt, &k_tab, &v_tab, &k_buf, &v_buf,
                    ],
                    MTLSize {
                        width: 1,
                        height: num_heads,
                        depth: 1,
                    },
                    MTLSize {
                        width: 1024,
                        height: 1,
                        depth: 1,
                    },
                )
            };
            if !ok {
                return false;
            }
        }
        true
    };

    let iters = 200usize;
    // Warm up + interleave to average out thermal/scheduling drift.
    if !run(&off_pipe, false, &bt_buf, 5) {
        return;
    }
    run(&on_pipe, true, &bt_buf, 5);
    run(&on_pipe, true, &bt_buf_flagged, 5);
    let mut off_secs = 0f64;
    let mut on_secs = 0f64;
    let mut rot_secs = 0f64;
    for _ in 0..3 {
        let t = std::time::Instant::now();
        run(&off_pipe, false, &bt_buf, iters);
        off_secs += t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        run(&on_pipe, true, &bt_buf, iters);
        on_secs += t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        run(&on_pipe, true, &bt_buf_flagged, iters);
        rot_secs += t.elapsed().as_secs_f64();
    }
    let off_ms = off_secs * 1000.0 / (3 * iters) as f64;
    let on_ms = on_secs * 1000.0 / (3 * iters) as f64;
    let rot_ms = rot_secs * 1000.0 / (3 * iters) as f64;
    let ratio = on_ms / off_ms;
    let rot_ratio = rot_ms / off_ms;
    eprintln!(
        "ROPE-ON-READ DECODE OVERHEAD (via_cache_v2, hd{head_dim} {num_heads}q/{num_kv}kv \
         kv_len={kv_len}):\n  baseline (rotate-on-write, read-as-is): {off_ms:.4} ms\n  \
         rope-on-read, NO keys rotate (machinery only): {on_ms:.4} ms  ({ratio:.3}x)\n  \
         rope-on-read, ALL keys re-roped (real span cost): {rot_ms:.4} ms  ({rot_ratio:.3}x)"
    );
    assert!(
        ratio < 1.15,
        "rope-on-read regressed the non-span decode hot path: {ratio:.3}x (bar 1.15x)"
    );
    // Re-roping every cached key is the feature's inherent decode cost; bar
    // catches a gross regression in the rotation path (simd_shuffle + rope).
    assert!(
        rot_ratio < 2.0,
        "rope-on-read rotation path regressed: {rot_ratio:.3}x (bar 2.0x)"
    );
}
