// SPDX-License-Identifier: Apache-2.0
//! gpt-oss attention sinks (`ATTN_SINKS`, const slot 21) across every
//! kernel family the arch reaches: the shared decode kernel
//! `attention_via_cache_v2` (plain AND sliding), the m<32 prefill
//! `attention_prefill_sdpa_v2_paged`, the gqa-cooperative
//! `attention_prefill_sdpa_gqa_shared`, the simdgroup steel paged kernel,
//! and the NAX matrix-accelerator twin — each vs a host f32
//! softmax-with-sink reference. Plus a sinks-off run pinning that the
//! const-fold is inert (the off pipeline's key carries no slot 21, so it
//! is byte-identical to the pre-sinks bake).
//!
//! The sink semantics under test (HF gpt_oss): the per-head sink logit is
//! an extra softmax column appended UNSCALED after qk·sm_scale (never
//! multiplied by it) and dropped before ·V — it contributes denominator
//! weight only.
//!
//! GPU tests — run with `--test-threads=1` (standing rule).

mod common;

use half::{bf16, f16};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize};
use scratchy_target_metal::aot::baked_build;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};

type Device = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLDevice>>;
type Buffer = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>>;

/// The element type a kernel stores: the bits round-trip helper.
trait Store: Copy {
    fn store(x: f32) -> Self;
    fn load(x: Self) -> f32;
    fn tag() -> &'static str;
}

impl Store for f16 {
    fn store(x: f32) -> Self {
        f16::from_f32(x)
    }
    fn load(x: Self) -> f32 {
        x.to_f32()
    }
    fn tag() -> &'static str {
        "f16"
    }
}

impl Store for bf16 {
    fn store(x: f32) -> Self {
        bf16::from_f32(x)
    }
    fn load(x: Self) -> f32 {
        x.to_f32()
    }
    fn tag() -> &'static str {
        "bf16"
    }
}

/// gpt-oss's attention geometry, shrunk for a kernel test: head_dim 64,
/// GQA 8 (8 query heads over 1 KV head), 16-token pages. `window` 0 = the
/// full-attention (odd) layers; >0 = the sliding (even) class.
struct Case {
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
    window: i32,
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

fn buf_t<T: Store>(device: &Device, data: &[f32]) -> Buffer {
    let h: Vec<T> = data.iter().map(|&v| T::store(v)).collect();
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

fn read_t<T: Store>(buf: &Buffer, n: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const T, n) }
        .iter()
        .map(|&h| T::load(h))
        .collect()
}

/// Host reference: causal (+ optional window) softmax attention over the
/// paged layout `[block, kv_head, tok_in_block, head_dim]` with an identity
/// block table, plus — `Some(sinks)` — the extra sink column: the RAW
/// per-head logit appended after qk·scale, dropped before ·V.
#[allow(clippy::too_many_arguments)]
fn sink_attn_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    case: &Case,
    block_size: usize,
    q_positions: &[usize],
    scale: f32,
    sinks: Option<&[f32]>,
) -> Vec<f32> {
    let &Case {
        num_q_heads,
        num_kv_heads,
        head_dim,
        kv_len,
        window,
    } = case;
    let group = num_q_heads / num_kv_heads;
    let kv_at = |tok: usize, kvh: usize, d: usize| -> f32 {
        k[((tok / block_size * num_kv_heads + kvh) * block_size + tok % block_size) * head_dim + d]
    };
    let vv_at = |tok: usize, kvh: usize, d: usize| -> f32 {
        v[((tok / block_size * num_kv_heads + kvh) * block_size + tok % block_size) * head_dim + d]
    };
    let mut out = vec![0f32; q_positions.len() * num_q_heads * head_dim];
    for (qi, &q_abs) in q_positions.iter().enumerate() {
        for h in 0..num_q_heads {
            let kvh = h / group;
            let qrow = &q[(qi * num_q_heads + h) * head_dim..][..head_dim];
            let mut scores = Vec::with_capacity(kv_len + 1);
            for t in 0..kv_len {
                let masked = t > q_abs || (window > 0 && (q_abs - t) as i64 >= window as i64);
                if masked {
                    scores.push(f32::NEG_INFINITY);
                    continue;
                }
                let mut s = 0f32;
                for (d, &qd) in qrow.iter().enumerate() {
                    s += qd * kv_at(t, kvh, d);
                }
                scores.push(s * scale);
            }
            // The sink column: UNSCALED, after every key.
            if let Some(sk) = sinks {
                scores.push(sk[h]);
            }
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|&s| (s - m).exp()).collect();
            let denom: f32 = exps.iter().sum();
            let orow = &mut out[(qi * num_q_heads + h) * head_dim..][..head_dim];
            // The sink column (the last entry) contributes denominator
            // weight only — iterate the key columns for ·V.
            for (t, &e) in exps[..kv_len].iter().enumerate() {
                if e == 0.0 {
                    continue;
                }
                let w = e / denom;
                for (d, o) in orow.iter_mut().enumerate() {
                    *o += w * vv_at(t, kvh, d);
                }
            }
        }
    }
    out
}

/// The shared pool every dispatch reads: K/V in the paged layout, an
/// identity block table, chunk tables pointing at the one buffer (BPC=0),
/// and the per-head sinks.
struct Pool {
    k_buf: Buffer,
    v_buf: Buffer,
    k_tab: Buffer,
    v_tab: Buffer,
    bt_buf: Buffer,
    sinks_buf: Buffer,
    k_r: Vec<f32>,
    v_r: Vec<f32>,
    sinks_r: Vec<f32>,
}

impl Pool {
    fn new<T: Store>(device: &Device, case: &Case, block_size: usize, seed: u64) -> Self {
        let num_blocks = case.kv_len.div_ceil(block_size);
        let kv_elems = num_blocks * case.num_kv_heads * block_size * case.head_dim;
        let k_host = pseudo(seed, kv_elems, 1.0);
        let v_host = pseudo(seed + 2, kv_elems, 1.0);
        let k_buf = buf_t::<T>(device, &k_host);
        let v_buf = buf_t::<T>(device, &v_host);
        let k_r: Vec<f32> = k_host.iter().map(|&x| T::load(T::store(x))).collect();
        let v_r: Vec<f32> = v_host.iter().map(|&x| T::load(T::store(x))).collect();
        let k_tab = buf_u64(device, &[k_buf.gpuAddress()]);
        let v_tab = buf_u64(device, &[v_buf.gpuAddress()]);
        let bt = buf_u32(device, &(0..num_blocks as u32).collect::<Vec<_>>());
        // Sink logits of gpt-oss's magnitude (learned biases, a few units):
        // many heads' sinks exceed every scaled score, exercising the
        // max-seeded-with-sink path, not just the tail term.
        let sinks_host = pseudo(seed + 4, case.num_q_heads, 3.0);
        let sinks_buf = buf_t::<T>(device, &sinks_host);
        let sinks_r: Vec<f32> = sinks_host.iter().map(|&x| T::load(T::store(x))).collect();
        Self {
            k_buf,
            v_buf,
            k_tab,
            v_tab,
            bt_buf: bt,
            sinks_buf,
            k_r,
            v_r,
            sinks_r,
        }
    }
}

fn base_constants(case: &Case, block_size: usize, max_blocks: usize) -> Vec<ConstantValue> {
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

fn max_err_of(got: &[f32], want: &[f32]) -> (f32, usize) {
    got.iter()
        .zip(want)
        .enumerate()
        .map(|(i, (g, w))| ((g - w).abs(), i))
        .fold((0f32, 0usize), |a, b| if b.0 > a.0 { b } else { a })
}

/// ── Decode: `attention_via_cache_v2` (plain + sliding + TurboQuant's
/// shared kernel), sinks at buffer 16 ──────────────────────────────────
///
/// `sinks`: `Some` runs the sinks-on pipeline (const slot 21 + the
/// buffer-16 binding) against the with-sink reference; `None` runs the
/// sinks-off pipeline — whose PipelineKey carries no slot 21, i.e. the
/// byte-identical pre-sinks bake — against the plain reference.
fn decode_case<T: Store>(name: &str, case: Case, sinks: bool) {
    let Some(di) = detect_device() else {
        eprintln!("{name}: no Metal device, skipped");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("shaders");
    let block_size = 16usize;
    let max_blocks = case.kv_len.div_ceil(block_size);
    let scale = 1.0 / (case.head_dim as f32).sqrt();
    let pool = Pool::new::<T>(&device, &case, block_size, 51);

    let q_host = pseudo(57, case.num_q_heads * case.head_dim, 2.0);
    let q_buf = buf_t::<T>(&device, &q_host);
    let q_r: Vec<f32> = q_host.iter().map(|&x| T::load(T::store(x))).collect();
    let out_buf = buf_zero(&device, case.num_q_heads * case.head_dim * 2);
    let seq_used = buf_u32(&device, &[case.kv_len as u32]);

    let mut consts = base_constants(&case, block_size, max_blocks);
    if sinks {
        consts.push(ConstantValue::uint(21, 1));
    }
    let key = PipelineKey::new(
        "attention",
        Box::leak(format!("attention_via_cache_v2_{}_specialized", T::tag()).into_boxed_str()),
        consts,
    );
    let pipeline = baked_build(&cache, &key).expect("decode pipeline");

    // K/V are reached via raw gpuAddress through the chunk table, so they
    // must be resident even where the kernel binds nothing — the slots
    // between the base six and the sinks binding (16) park them.
    let mut bufs = vec![
        &out_buf,
        &q_buf,
        &seq_used,
        &pool.bt_buf,
        &pool.k_tab,
        &pool.v_tab,
        &pool.k_buf,
        &pool.v_buf,
    ];
    if sinks {
        bufs.extend(std::iter::repeat_n(&pool.k_buf, 16 - 8));
        bufs.push(&pool.sinks_buf);
    }
    if !common::dispatch_threadgroups(
        &device,
        &pipeline,
        &bufs,
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

    let got = read_t::<T>(&out_buf, case.num_q_heads * case.head_dim);
    let want = sink_attn_ref(
        &q_r,
        &pool.k_r,
        &pool.v_r,
        &case,
        block_size,
        &[case.kv_len - 1],
        scale,
        sinks.then_some(&pool.sinks_r),
    );
    let (max_err, worst) = max_err_of(&got, &want);
    eprintln!("{name}: max_err={max_err} at {worst}");
    assert!(max_err < 2e-2, "{name} max_err {max_err}");
}

/// ── Prefill m<32: `attention_prefill_sdpa_v2_paged`, sinks at buffer 9 ──
fn prefill_sdpa_case<T: Store>(name: &str, case: Case) {
    let Some(di) = detect_device() else {
        eprintln!("{name}: no Metal device, skipped");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("shaders");
    let block_size = 16usize;
    let max_blocks = case.kv_len.div_ceil(block_size);
    let scale = 1.0 / (case.head_dim as f32).sqrt();
    let pool = Pool::new::<T>(&device, &case, block_size, 61);

    let total_q = case.kv_len;
    let q_host = pseudo(67, total_q * case.num_q_heads * case.head_dim, 2.0);
    let q_buf = buf_t::<T>(&device, &q_host);
    let q_r: Vec<f32> = q_host.iter().map(|&x| T::load(T::store(x))).collect();
    let out_buf = buf_zero(&device, total_q * case.num_q_heads * case.head_dim * 2);
    let cu_seqlens = buf_u32(&device, &[0, total_q as u32, 0, 0]);
    let seq_used = buf_u32(&device, &[case.kv_len as u32]);

    let mut consts = base_constants(&case, block_size, max_blocks);
    consts.push(ConstantValue::uint(21, 1));
    let key = PipelineKey::new(
        "attention",
        Box::leak(
            format!("attention_prefill_sdpa_v2_paged_{}_specialized", T::tag()).into_boxed_str(),
        ),
        consts,
    );
    let pipeline = baked_build(&cache, &key).expect("prefill sdpa pipeline");

    if !common::dispatch_threadgroups(
        &device,
        &pipeline,
        &[
            &out_buf,
            &q_buf,
            &cu_seqlens,
            &seq_used,
            &pool.bt_buf,
            &pool.k_tab,
            &pool.v_tab,
            &pool.k_buf,
            &pool.v_buf,
            &pool.sinks_buf,
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

    let got = read_t::<T>(&out_buf, total_q * case.num_q_heads * case.head_dim);
    let want = sink_attn_ref(
        &q_r,
        &pool.k_r,
        &pool.v_r,
        &case,
        block_size,
        &(0..total_q).collect::<Vec<_>>(),
        scale,
        Some(&pool.sinks_r),
    );
    let (max_err, worst) = max_err_of(&got, &want);
    eprintln!("{name}: max_err={max_err} at {worst}");
    assert!(max_err < 2e-2, "{name} max_err {max_err}");
}

/// ── Prefill gqa-cooperative: `attention_prefill_sdpa_gqa_shared`, sinks
/// at buffer 9 (slot 7 parks K — no rope-on-read on this path) ──────────
fn prefill_gqa_shared_case<T: Store>(name: &str, case: Case) {
    let Some(di) = detect_device() else {
        eprintln!("{name}: no Metal device, skipped");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("shaders");
    let block_size = 16usize; // gqa_shared: one sub-stage (block_size <= 16)
    let gqa = case.num_q_heads / case.num_kv_heads;
    assert!((2..=32).contains(&gqa), "gqa_shared needs 2..=32");
    let max_blocks = case.kv_len.div_ceil(block_size);
    let scale = 1.0 / (case.head_dim as f32).sqrt();
    let pool = Pool::new::<T>(&device, &case, block_size, 71);

    let total_q = case.kv_len;
    let q_host = pseudo(77, total_q * case.num_q_heads * case.head_dim, 2.0);
    let q_buf = buf_t::<T>(&device, &q_host);
    let q_r: Vec<f32> = q_host.iter().map(|&x| T::load(T::store(x))).collect();
    let out_buf = buf_zero(&device, total_q * case.num_q_heads * case.head_dim * 2);
    let cu_seqlens = buf_u32(&device, &[0, total_q as u32, 0, 0]);
    let seq_used = buf_u32(&device, &[case.kv_len as u32]);

    let mut consts = base_constants(&case, block_size, max_blocks);
    consts.push(ConstantValue::uint(21, 1));
    let key = PipelineKey::new(
        "attention",
        Box::leak(
            format!("attention_prefill_sdpa_gqa_shared_{}_specialized", T::tag()).into_boxed_str(),
        ),
        consts,
    );
    let pipeline = baked_build(&cache, &key).expect("gqa_shared pipeline");

    if !common::dispatch_threadgroups(
        &device,
        &pipeline,
        &[
            &out_buf,
            &q_buf,
            &cu_seqlens,
            &seq_used,
            &pool.bt_buf,
            &pool.k_tab,
            &pool.v_tab,
            &pool.k_buf,
            &pool.v_buf,
            &pool.sinks_buf,
        ],
        MTLSize {
            width: case.num_kv_heads,
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

    let got = read_t::<T>(&out_buf, total_q * case.num_q_heads * case.head_dim);
    let want = sink_attn_ref(
        &q_r,
        &pool.k_r,
        &pool.v_r,
        &case,
        block_size,
        &(0..total_q).collect::<Vec<_>>(),
        scale,
        Some(&pool.sinks_r),
    );
    let (max_err, worst) = max_err_of(&got, &want);
    eprintln!("{name}: max_err={max_err} at {worst}");
    assert!(max_err < 2e-2, "{name} max_err {max_err}");
}

/// ── Prefill steel family (simdgroup + NAX), sinks at buffer 9 ─────────
///
/// `symbol`/`threads`/`bq` name the instantiation (the generated
/// `steel_paged` table or `nax_paged_kernel`).
fn prefill_steel_case<T: Store>(
    name: &str,
    case: Case,
    library: &'static str,
    symbol: &'static str,
    threads: usize,
    bq: usize,
) {
    let Some(di) = detect_device() else {
        eprintln!("{name}: no Metal device, skipped");
        return;
    };
    let device = di.device.clone();
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("shaders");
    let block_size = 16usize;
    let max_blocks = case.kv_len.div_ceil(block_size);
    let scale = 1.0 / (case.head_dim as f32).sqrt();
    let pool = Pool::new::<T>(&device, &case, block_size, 81);

    let total_q = case.kv_len;
    let q_host = pseudo(87, total_q * case.num_q_heads * case.head_dim, 2.0);
    let q_buf = buf_t::<T>(&device, &q_host);
    let q_r: Vec<f32> = q_host.iter().map(|&x| T::load(T::store(x))).collect();
    let out_buf = buf_zero(&device, total_q * case.num_q_heads * case.head_dim * 2);
    let cu_seqlens = buf_u32(&device, &[0, total_q as u32, 0, 0]);
    let seq_used = buf_u32(&device, &[case.kv_len as u32]);

    let mut consts = base_constants(&case, block_size, max_blocks);
    // Steel resolve() does chunk = physical / blocks_per_chunk; BPC=0 would
    // divide by zero. BPC >= num_blocks keeps every block in chunk 0.
    consts[6] = ConstantValue::uint(6, 128);
    consts.push(ConstantValue::uint(99, 0)); // DEBUG_MODE (steel reads slot 99)
    consts.push(ConstantValue::uint(21, 1)); // ATTN_SINKS
    let key = PipelineKey::new(library, symbol, consts);
    let pipeline = baked_build(&cache, &key).expect("steel pipeline");

    let nq_blocks = total_q.div_ceil(bq);
    if !common::dispatch_threadgroups(
        &device,
        &pipeline,
        &[
            &out_buf,
            &q_buf,
            &cu_seqlens,
            &seq_used,
            &pool.bt_buf,
            &pool.k_tab,
            &pool.v_tab,
            &pool.k_buf,
            &pool.v_buf,
            &pool.sinks_buf,
        ],
        MTLSize {
            width: nq_blocks,
            height: case.num_q_heads,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    ) {
        return;
    }

    let got = read_t::<T>(&out_buf, total_q * case.num_q_heads * case.head_dim);
    let want = sink_attn_ref(
        &q_r,
        &pool.k_r,
        &pool.v_r,
        &case,
        block_size,
        &(0..total_q).collect::<Vec<_>>(),
        scale,
        Some(&pool.sinks_r),
    );
    let (max_err, worst) = max_err_of(&got, &want);
    eprintln!("{name}: max_err={max_err} at {worst}");
    assert!(max_err < 2e-2, "{name} max_err {max_err}");
}

/// gpt-oss's full-attention decode, bf16 (the arch's dtype): 3 pages, the
/// last partial (kv 48 = 2·16 + 16 — full pages; kv 40 would be partial,
/// covered by the sliding case below at 34).
#[test]
fn decode_sinks_bf16() {
    decode_case::<bf16>(
        "decode sinks bf16",
        Case {
            num_q_heads: 8,
            num_kv_heads: 1,
            head_dim: 64,
            kv_len: 48,
            window: 0,
        },
        true,
    );
}

#[test]
fn decode_sinks_f16() {
    decode_case::<f16>(
        "decode sinks f16",
        Case {
            num_q_heads: 8,
            num_kv_heads: 1,
            head_dim: 64,
            kv_len: 48,
            window: 0,
        },
        true,
    );
}

/// The even (sliding) layers: window 8 over a partial tail page
/// (kv 34 = 2·16 + 2).
#[test]
fn decode_sliding_sinks_bf16() {
    decode_case::<bf16>(
        "decode sliding sinks bf16",
        Case {
            num_q_heads: 8,
            num_kv_heads: 1,
            head_dim: 64,
            kv_len: 34,
            window: 8,
        },
        true,
    );
}

/// Sinks OFF: no slot-21 const (the PipelineKey is byte-identical to the
/// pre-sinks bake) and no sinks binding — the plain reference must match,
/// pinning that the const-fold left the off path inert.
#[test]
fn decode_sinks_off_is_plain_attention() {
    decode_case::<bf16>(
        "decode sinks-off bf16",
        Case {
            num_q_heads: 8,
            num_kv_heads: 1,
            head_dim: 64,
            kv_len: 48,
            window: 0,
        },
        false,
    );
}

/// Short-prompt prefill (m < BQ_STEEL=32) on the per-(q_head, query) sdpa
/// kernel — gpt-oss's f16 slot.
#[test]
fn prefill_sdpa_paged_sinks_f16() {
    prefill_sdpa_case::<f16>(
        "prefill sdpa sinks f16",
        Case {
            num_q_heads: 8,
            num_kv_heads: 1,
            head_dim: 64,
            kv_len: 24,
            window: 0,
        },
    );
}

/// …and its bf16 slot (gpt-oss's dtype).
#[test]
fn prefill_sdpa_paged_sinks_bf16() {
    prefill_sdpa_case::<bf16>(
        "prefill sdpa sinks bf16",
        Case {
            num_q_heads: 8,
            num_kv_heads: 1,
            head_dim: 64,
            kv_len: 24,
            window: 0,
        },
    );
}

/// The GQA-cooperative prefill kernel (gqa 8 ∈ 8..=32, full attention).
#[test]
fn prefill_gqa_shared_sinks_f16() {
    prefill_gqa_shared_case::<f16>(
        "prefill gqa_shared sinks f16",
        Case {
            num_q_heads: 8,
            num_kv_heads: 1,
            head_dim: 64,
            kv_len: 24,
            window: 0,
        },
    );
}

/// The simdgroup steel paged kernel at gpt-oss's head_dim 64 (BQ 32 — the
/// m ≥ 32 prefill bucket; 48 queries = 2 BQ tiles, partial tail 16).
#[test]
fn steel_paged_sinks_f16() {
    prefill_steel_case::<f16>(
        "steel paged sinks f16",
        Case {
            num_q_heads: 8,
            num_kv_heads: 1,
            head_dim: 64,
            kv_len: 48,
            window: 0,
        },
        "attention_steel_paged",
        "attention_steel_paged_f16_bq32_bk16_bd64_wm4_wn1_bs16",
        128,
        32,
    );
}

/// The NAX matrix-accelerator twin at head_dim 64 (BQ 64 — one tile).
/// Skips with a note on non-NAX GPUs.
#[test]
fn nax_paged_sinks_f16() {
    let Some(di) = detect_device() else {
        eprintln!("nax paged sinks: no Metal device, skipped");
        return;
    };
    if !scratchy_target_metal::targets::is_nax_capable(di.profile.generation) {
        eprintln!(
            "nax paged sinks: non-NAX GPU ({:?}), skipped",
            di.profile.generation
        );
        return;
    }
    let nax = scratchy_target_metal::steel_paged::nax_paged_kernel("f16", 64, 16)
        .expect("a hd-64 NAX instantiation");
    prefill_steel_case::<f16>(
        "nax paged sinks f16",
        Case {
            num_q_heads: 8,
            num_kv_heads: 1,
            head_dim: 64,
            kv_len: 48,
            window: 0,
        },
        "attention_steel_nax_paged",
        nax.symbol,
        nax.threads as usize,
        nax.bq as usize,
    );
}
