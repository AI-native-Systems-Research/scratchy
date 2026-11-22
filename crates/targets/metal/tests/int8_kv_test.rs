// SPDX-License-Identifier: Apache-2.0
//! The int8 KV store's kernels against the host codec (`scratchy_layers::kv_int8`): the writer's
//! encode (`kv_append`), the decode attention reading the store (`attention_decode_gqa_int8` and
//! its combine) over every head dim the codec takes and query-head groups below, at and past the
//! matrix units' 8 rows, and the prefill staging (`kv_stage`) that rotates cached rows back into
//! the dense scratch.
//!
//! GPU tests — run with `--test-threads=1` (standing rule).

mod common;

use half::bf16;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize};
use scratchy_layers::kv_int8::{decode_row, encode_row};
use scratchy_target_metal::aot::baked_build;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};

type Device = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLDevice>>;
type Buffer = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>>;

const BLOCK_SIZE: usize = 16;

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

fn round_bf16(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| bf16::from_f32(v).to_f32()).collect()
}

fn buf_bytes(device: &Device, bytes: &[u8]) -> Buffer {
    let buf = device
        .newBufferWithLength_options(bytes.len().max(16), MTLResourceOptions::StorageModeShared)
        .expect("newBuffer");
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            buf.contents().as_ptr() as *mut u8,
            bytes.len(),
        );
    }
    buf
}

fn bytes_of<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn buf_bf16(device: &Device, x: &[f32]) -> Buffer {
    let h: Vec<bf16> = x.iter().map(|&v| bf16::from_f32(v)).collect();
    buf_bytes(device, bytes_of(&h))
}

fn read<T: Copy>(buf: &Buffer, n: usize) -> Vec<T> {
    unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const T, n) }.to_vec()
}

fn read_bf16(buf: &Buffer, n: usize) -> Vec<f32> {
    read::<bf16>(buf, n).iter().map(|v| v.to_f32()).collect()
}

fn pipeline(
    device: &Device,
    library: &'static str,
    function: &str,
    c: Vec<ConstantValue>,
) -> objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>> {
    let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("shaders");
    let key = PipelineKey::new(library, Box::leak(function.to_owned().into_boxed_str()), c);
    baked_build(&cache, &key).expect("pipeline")
}

fn size(w: usize, h: usize, d: usize) -> MTLSize {
    MTLSize {
        width: w,
        height: h,
        depth: d,
    }
}

/// An int8 store of `kv_len` tokens' K and V rows, encoded on the host: the codes and scales at
/// row `token · num_kv + kv_head` (an identity block table), and the rows they decode to.
struct Store {
    codes_k: Buffer,
    codes_v: Buffer,
    scales_k: Buffer,
    scales_v: Buffer,
    k: Vec<f32>,
    v: Vec<f32>,
}

impl Store {
    fn new(device: &Device, kv_len: usize, num_kv: usize, d: usize, seed: u64) -> Self {
        let rows = kv_len.next_multiple_of(BLOCK_SIZE) * num_kv;
        let encode = |seed| {
            let x = pseudo(seed, rows * d, 1.5);
            let mut codes = Vec::with_capacity(rows * d);
            let mut scales = Vec::with_capacity(rows);
            let mut decoded = Vec::with_capacity(rows * d);
            for row in x.chunks(d) {
                let (c, s) = encode_row(row);
                decoded.extend(decode_row(&c, s));
                codes.extend(c);
                scales.push(s);
            }
            (codes, scales, decoded)
        };
        let (ck, sk, k) = encode(seed);
        let (cv, sv, v) = encode(seed + 1);
        Self {
            codes_k: buf_bytes(device, bytes_of(&ck)),
            codes_v: buf_bytes(device, bytes_of(&cv)),
            scales_k: buf_bytes(device, bytes_of(&sk)),
            scales_v: buf_bytes(device, bytes_of(&sv)),
            k,
            v,
        }
    }
}

/// Host attention of one query row's heads over the first `kv_len` tokens of `store`, the last
/// `window` of them when one is given.
fn reference(
    q: &[f32],
    store: &Store,
    kv_len: usize,
    window: Option<usize>,
    num_q: usize,
    num_kv: usize,
    d: usize,
) -> Vec<f32> {
    let group = num_q / num_kv;
    let scale = 1.0 / (d as f32).sqrt();
    let mut out = vec![0f32; num_q * d];
    for h in 0..num_q {
        let kvh = h / group;
        let row = |t: usize| (t * num_kv + kvh) * d;
        let qh = &q[h * d..][..d];
        let lo = window.map_or(0, |w| kv_len.saturating_sub(w));
        let scores: Vec<f32> = (0..kv_len)
            .map(|t| match t < lo {
                true => f32::NEG_INFINITY,
                false => {
                    qh.iter()
                        .zip(&store.k[row(t)..][..d])
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        * scale
                }
            })
            .collect();
        let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
        let sum: f32 = e.iter().sum();
        for (t, w) in e.iter().enumerate() {
            for (o, x) in out[h * d..][..d].iter_mut().zip(&store.v[row(t)..][..d]) {
                *o += w / sum * x;
            }
        }
    }
    out
}

/// The heads a decode threadgroup serves: the lowering's law (the largest divisor of the group
/// the fragment's 8 rows hold).
fn decode_heads(group: usize) -> usize {
    (1..=group.min(8))
        .filter(|r| group.is_multiple_of(*r))
        .max()
        .unwrap_or(1)
}

fn decode_case(
    num_q: usize,
    num_kv: usize,
    d: usize,
    kv_len: usize,
    splits: usize,
    window: Option<usize>,
) {
    decode_run(num_q, num_kv, d, kv_len, splits, window, None);
}

/// The decode at `key_group` simdgroups a key block (the law's when `None`): its output checked
/// against the host, and the seconds a call takes over `reps` more calls when there are any.
fn decode_run(
    num_q: usize,
    num_kv: usize,
    d: usize,
    kv_len: usize,
    splits: usize,
    window: Option<usize>,
    key_group: Option<(u32, usize)>,
) -> Option<f64> {
    let name = format!("decode q{num_q} kv{num_kv} d{d} len{kv_len} splits{splits} {window:?}");
    let Some(di) = detect_device() else {
        eprintln!("{name}: no Metal device, skipped");
        return None;
    };
    let device = di.device.clone();
    let store = Store::new(&device, kv_len, num_kv, d, 7 + d as u64);
    let (sg, reps) = key_group.unwrap_or((
        scratchy_target_metal::kv_int8::decode_key_group(d as u32),
        0,
    ));
    let max_blocks = kv_len.div_ceil(BLOCK_SIZE);
    let q = round_bf16(&pseudo(3, num_q * d, 2.0));
    let heads = decode_heads(num_q / num_kv);
    let base = vec![
        ConstantValue::uint(0, d as u32),
        ConstantValue::uint(1, num_q as u32),
        ConstantValue::uint(2, num_kv as u32),
        ConstantValue::float(3, 1.0 / (d as f32).sqrt()),
        ConstantValue::uint(4, BLOCK_SIZE as u32),
        ConstantValue::uint(5, max_blocks as u32),
        ConstantValue::uint(6, 0),
        ConstantValue::int(7, window.map_or(0, |w| w as i32)),
        ConstantValue::uint(13, 1),
        ConstantValue::uint(18, splits as u32),
        ConstantValue::uint(25, scratchy_target_metal::kv_int8::combine_groups(d as u32)),
    ];
    let split = pipeline(
        &device,
        "attention",
        "attention_decode_gqa_int8_bf16_specialized",
        [
            base.clone(),
            vec![
                ConstantValue::uint(16, heads as u32),
                ConstantValue::uint(24, sg),
            ],
        ]
        .concat(),
    );
    let combine = pipeline(
        &device,
        "attention",
        "attention_via_cache_v2_combine_bf16_specialized",
        base,
    );
    let q_buf = buf_bf16(&device, &q);
    let out = buf_bytes(&device, &vec![0u8; num_q * d * 2]);
    let seq_used = buf_bytes(&device, bytes_of(&[kv_len as u32]));
    let table: Vec<u32> = (0..max_blocks as u32).collect();
    let block_table = buf_bytes(&device, bytes_of(&table));
    // No own key: every key is read off the store.
    let slot_mapping = buf_bytes(&device, bytes_of(&[u32::MAX]));
    let partials = buf_bytes(&device, &vec![0u8; num_q * splits * (d + 2) * 4]);
    let unread = buf_bytes(&device, &[0u8; 64]);
    let mut bufs: Vec<&Buffer> = vec![&unread; 17];
    bufs[0] = &out;
    bufs[1] = &q_buf;
    bufs[2] = &seq_used;
    bufs[3] = &block_table;
    bufs[7] = &store.codes_k;
    bufs[8] = &store.codes_v;
    bufs[9] = &store.scales_k;
    bufs[10] = &store.scales_v;
    bufs[13] = &slot_mapping;
    bufs[16] = &partials;
    assert!(common::dispatch_threadgroups(
        &device,
        &split,
        &bufs,
        size(1, num_q / heads, splits),
        size(d, 1, 1),
    ));
    let mut cbufs: Vec<&Buffer> = vec![&unread; 17];
    cbufs[0] = &out;
    cbufs[16] = &partials;
    let combine_threads = 32 * scratchy_target_metal::kv_int8::combine_groups(d as u32) as usize;
    assert!(common::dispatch_threadgroups(
        &device,
        &combine,
        &cbufs,
        size(1, num_q, 1),
        size(combine_threads, 1, 1),
    ));
    let got = read_bf16(&out, num_q * d);
    let want = reference(&q, &store, kv_len, window, num_q, num_kv, d);
    let err = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (g - w).abs())
        .fold(0f32, f32::max);
    eprintln!("{name}: max err {err}");
    assert!(err < 2e-2, "{name}: max err {err}");
    // `reps` calls in one command buffer, each after the last (a decode step's order).
    let mut batch = scratchy_target_metal::mtl4_dispatch::Mtl4DispatchBatch::begin(&device)?;
    let binds: Vec<(&Buffer, usize)> = bufs.iter().enumerate().map(|(i, b)| (*b, i)).collect();
    let t = std::time::Instant::now();
    for _ in 0..reps {
        batch.encode(
            &split,
            &binds,
            &[],
            &[],
            &[],
            size(1, num_q / heads, splits),
            size(d, 1, 1),
        );
        batch.barrier();
    }
    batch.commit(true);
    (reps > 0).then(|| t.elapsed().as_secs_f64() / reps as f64)
}

/// The decode's key-group width, timed: an 8192-key context (16 query heads over 2, eight splits)
/// at each head dim, every key group the kernel takes, interleaved rounds. A probe, not a gate:
/// `--ignored --nocapture`.
#[test]
#[ignore = "a timing probe: --ignored --nocapture"]
fn decode_key_group_timings() {
    for d in [64usize, 128, 256, 512] {
        let groups: Vec<u32> = [1u32, 2, 4, 8]
            .into_iter()
            .filter(|sg| (d as u32 / 32).is_multiple_of(*sg))
            .collect();
        let mut times = vec![Vec::new(); groups.len()];
        for _ in 0..5 {
            for (i, &sg) in groups.iter().enumerate() {
                if let Some(t) = decode_run(16, 2, d, 8192, 8, None, Some((sg, 50))) {
                    times[i].push(t);
                }
            }
        }
        let line: Vec<String> = (groups.iter().zip(&mut times))
            .map(|(sg, t)| {
                t.sort_by(f64::total_cmp);
                format!("sg{sg} {:.1} us", t[t.len() / 2] * 1e6)
            })
            .collect();
        eprintln!("decode d{d}: {}", line.join(", "));
    }
}

/// The decode's split count, timed: an 8192-key context at each geometry (query heads, KV heads,
/// head dim, window), the law's key group, interleaved rounds. A probe, not a gate:
/// `--ignored --nocapture`.
#[test]
#[ignore = "a timing probe: --ignored --nocapture"]
fn decode_split_timings() {
    let shapes: [(usize, usize, usize, Option<usize>); 5] = [
        (16, 2, 512, None),
        (16, 2, 256, None),
        (32, 2, 128, None),
        (64, 8, 64, None),
        (16, 8, 256, Some(1024)),
    ];
    for (nq, nkv, d, window) in shapes {
        let splits = [4usize, 8, 16, 32, 64];
        let sg = scratchy_target_metal::kv_int8::decode_key_group(d as u32);
        let mut times = vec![Vec::new(); splits.len()];
        for _ in 0..5 {
            for (i, &s) in splits.iter().enumerate() {
                if let Some(t) = decode_run(nq, nkv, d, 8192, s, window, Some((sg, 50))) {
                    times[i].push(t);
                }
            }
        }
        let line: Vec<String> = (splits.iter().zip(&mut times))
            .map(|(s, t)| {
                t.sort_by(f64::total_cmp);
                format!("{s}: {:.1} us", t[t.len() / 2] * 1e6)
            })
            .collect();
        eprintln!(
            "decode {nq}q/{nkv}kv d{d} window {window:?}: {}",
            line.join(", ")
        );
    }
}

/// Every head dim the codec takes, its query-head group below the fragment's 8 rows (2), at it
/// (8), past it (16, two threadgroups a KV head), and one KV head a query head (1); one split and
/// several.
#[test]
fn the_decode_reads_the_store_as_the_host_decodes_it() {
    for d in [64, 128, 256, 512] {
        for (num_q, num_kv) in [(4, 2), (16, 2), (32, 2), (2, 2)] {
            for splits in [1, 3] {
                decode_case(num_q, num_kv, d, 300, splits, None);
            }
        }
    }
}

/// A sliding window: the keys before its first read as past the context, whichever split, or
/// block's key, they fall in — a window ending mid-block, and one wider than the context.
#[test]
fn the_decode_reads_a_sliding_window() {
    for d in [64, 256] {
        for splits in [1, 3] {
            for window in [100, 128, 1000] {
                decode_case(16, 2, d, 300, splits, Some(window));
            }
        }
    }
}

/// The writer's encode is the host's: `kv_append` with its int8 store (slot 10, buffers 16..19)
/// codes each row as `encode_row` does — within a code of it where the two round a tie apart —
/// and its scale to a few ulps.
#[test]
fn the_writer_encodes_as_the_host_does() {
    let Some(di) = detect_device() else {
        eprintln!("no Metal device, skipped");
        return;
    };
    let device = di.device.clone();
    for d in [64usize, 128, 256, 512] {
        let (num_kv, tokens) = (2usize, 5usize);
        let p = pipeline(
            &device,
            "kv_append",
            "kv_append_bf16",
            vec![
                ConstantValue::uint(0, d as u32),
                ConstantValue::uint(1, num_kv as u32),
                ConstantValue::uint(2, BLOCK_SIZE as u32),
                ConstantValue::uint(3, 1),
                ConstantValue::uint(10, 1),
            ],
        );
        let k = round_bf16(&pseudo(11, tokens * num_kv * d, 1.5));
        let v = round_bf16(&pseudo(13, tokens * num_kv * d, 0.5));
        let slots: Vec<u32> = (0..tokens as u32).map(|t| t * 3 + 1).collect();
        let rows = (3 * tokens + 1).next_multiple_of(BLOCK_SIZE) * num_kv;
        let pool = buf_bytes(&device, &vec![0u8; rows * d * 2]);
        let table = buf_bytes(&device, bytes_of(&[pool.gpuAddress()]));
        let (codes_k, codes_v) = (
            buf_bytes(&device, &vec![0u8; rows * d]),
            buf_bytes(&device, &vec![0u8; rows * d]),
        );
        let (scales_k, scales_v) = (
            buf_bytes(&device, &vec![0u8; rows * 4]),
            buf_bytes(&device, &vec![0u8; rows * 4]),
        );
        let (k_buf, v_buf) = (buf_bf16(&device, &k), buf_bf16(&device, &v));
        let slot_buf = buf_bytes(&device, bytes_of(&slots));
        let unread = buf_bytes(&device, &[0u8; 64]);
        let mut bufs: Vec<&Buffer> = vec![&unread; 20];
        bufs[0] = &k_buf;
        bufs[1] = &v_buf;
        bufs[2] = &slot_buf;
        bufs[3] = &table;
        bufs[4] = &table;
        // The pool is written through its chunk table's address: bound, so it is resident.
        bufs[5] = &pool;
        bufs[16] = &codes_k;
        bufs[17] = &scales_k;
        bufs[18] = &codes_v;
        bufs[19] = &scales_v;
        assert!(common::dispatch_threadgroups(
            &device,
            &p,
            &bufs,
            size(tokens, num_kv, 1),
            size(d, 1, 1),
        ));
        let (ck, cv) = (
            read::<i8>(&codes_k, rows * d),
            read::<i8>(&codes_v, rows * d),
        );
        let (sk, sv) = (read::<f32>(&scales_k, rows), read::<f32>(&scales_v, rows));
        for (t, &slot) in slots.iter().enumerate() {
            for h in 0..num_kv {
                let row = slot as usize * num_kv + h;
                for (x, codes, scales) in [(&k, &ck, &sk), (&v, &cv, &sv)] {
                    let (want, scale) = encode_row(&x[(t * num_kv + h) * d..][..d]);
                    let got = &codes[row * d..][..d];
                    let worst = (got.iter().zip(&want))
                        .map(|(a, b)| (i32::from(*a) - i32::from(*b)).abs())
                        .max();
                    assert!(
                        worst <= Some(1),
                        "d{d} token {t} head {h}: codes off by {worst:?}"
                    );
                    let rel = (scales[row] - scale).abs() / scale.max(1e-12);
                    assert!(
                        rel < 1e-5,
                        "d{d} token {t} head {h}: scale {} vs {scale}",
                        scales[row]
                    );
                }
            }
        }
    }
}

/// The staging pass writes each cached row of the step's sequences into the dense scratch,
/// decoded and rotated back, and leaves the step's new rows — the writer's — as they are.
#[test]
fn the_stage_writes_cached_rows_decoded_and_keeps_the_new() {
    let Some(di) = detect_device() else {
        eprintln!("no Metal device, skipped");
        return;
    };
    let device = di.device.clone();
    for d in [64usize, 128, 256, 512] {
        let (num_kv, kv_len, new) = (2usize, 40usize, 6usize);
        let store = Store::new(&device, kv_len, num_kv, d, 17);
        let max_blocks = kv_len.div_ceil(BLOCK_SIZE);
        let p = pipeline(
            &device,
            "attention",
            "kv_stage_bf16",
            vec![
                ConstantValue::uint(0, d as u32),
                ConstantValue::uint(2, num_kv as u32),
                ConstantValue::uint(4, BLOCK_SIZE as u32),
                ConstantValue::uint(5, max_blocks as u32),
                ConstantValue::uint(6, 0),
                ConstantValue::uint(13, 1),
            ],
        );
        // The scratch's new rows hold a marker the stage must not touch.
        let marker = 3.0f32;
        let elems = max_blocks * num_kv * BLOCK_SIZE * d;
        let scratch = buf_bf16(&device, &vec![marker; elems]);
        let table = buf_bytes(&device, bytes_of(&[scratch.gpuAddress()]));
        let bt: Vec<u32> = (0..max_blocks as u32).collect();
        let block_table = buf_bytes(&device, bytes_of(&bt));
        let seq_used = buf_bytes(&device, bytes_of(&[kv_len as u32]));
        let cu = buf_bytes(&device, bytes_of(&[0u32, new as u32]));
        let slots: Vec<u32> = ((kv_len - new) as u32..kv_len as u32).collect();
        let slot_mapping = buf_bytes(&device, bytes_of(&slots));
        let bufs: Vec<&Buffer> = vec![
            &table,
            &block_table,
            &seq_used,
            &cu,
            &slot_mapping,
            &store.codes_k,
            &store.scales_k,
            // The scratch is written through its chunk table's address: bound past the
            // kernel's slots, so it is resident.
            &scratch,
        ];
        assert!(common::dispatch_threadgroups(
            &device,
            &p,
            &bufs,
            size(max_blocks, num_kv, 1),
            size(32 * BLOCK_SIZE, 1, 1),
        ));
        let got = read_bf16(&scratch, elems);
        for t in 0..kv_len {
            for h in 0..num_kv {
                let at = ((t / BLOCK_SIZE * num_kv + h) * BLOCK_SIZE + t % BLOCK_SIZE) * d;
                let row = &got[at..][..d];
                if t >= kv_len - new {
                    assert!(
                        row.iter().all(|&x| x == marker),
                        "d{d}: new row {t} touched"
                    );
                    continue;
                }
                let want = &store.k[(t * num_kv + h) * d..][..d];
                let err = row
                    .iter()
                    .zip(want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    err < 2e-2,
                    "d{d} row {t} head {h}: max err {err}: got {:?} want {:?}",
                    &row[..6],
                    &want[..6]
                );
            }
        }
    }
}
