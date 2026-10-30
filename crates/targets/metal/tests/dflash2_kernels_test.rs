// SPDX-License-Identifier: Apache-2.0
//! Splash's DFlash2 draft kernels (`dflash2.metal`, `dflash2_select.metal`,
//! `dflash2_capture.metal`) against the CPU references of Splash's own tests (~/git/splash @
//! 1482f6bb `dev/tests/engine/draft_attention_metal_test.mm`, `draft_selector_metal_test.mm`,
//! `dflash_batch_control_metal_test.mm`), transcribed at Qwen3.6-35B-A3B's drafter geometry, with
//! their cases, their tolerances and their exactness claims. Where the port changed a kernel's
//! interface (a lane's ring is a pool's slot, a row's RoPE pair is computed at its position from
//! the baked rotary base, a prompt's commit finds its sequence by `cu_seqlens`), the reference
//! follows the port and the cases cover the change.
//!
//! GPU tests — run with `--test-threads=1` (standing rule).

use half::bf16;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize};
use scratchy_target_metal::aot::baked_build;
use scratchy_target_metal::detect_device;
use scratchy_target_metal::dflash2::{
    ATTENTION_SPLITS, BUDGET_LANES, CANDIDATES, DraftLaneParams, EDGE_THREADS, PROPOSALS,
    SAMPLING_SHARDS, SAMPLING_UNIFORMS, VERIFY_COST_SEED, VerifyCost, VerifyLane, VerifyRows,
};
use scratchy_target_metal::mtl4_dispatch::Mtl4DispatchBatch;
use scratchy_target_metal::specialized_pipeline_cache::{
    ComputePipelineState, PipelineKey, SpecializedPipelineCache,
};
use scratchy_target_metal::tape::constants::ConstantValue;
use scratchy_target_metal::tape::kernel_constants::{
    CaptureConstants, DraftConstants, DraftSelectConstants,
};

type Device = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLDevice>>;
type Buffer = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>>;

// Qwen3.6-35B-A3B's drafter (configs/dflash2/qwen3.6-35b-a3b-dflash2.json).
const ROWS: u32 = 8;
const QUERY_HEADS: u32 = 32;
const KV_HEADS: u32 = 8;
const HEAD_DIM: u32 = 128;
const WINDOW: u32 = 2048;
const HIDDEN: u32 = 2048;
const CONV_GROUP: u32 = 16;
const RANK: u32 = 256;
const RMS_EPS: f32 = 1e-6;
const SCALE: f32 = 0.088_388_35;
/// The drafter's rotary base (`rope_parameters.rope_theta`).
const THETA: f32 = 10_000_000.0;
const ATTENTION: u32 = QUERY_HEADS * HEAD_DIM;
const KV_WIDTH: u32 = KV_HEADS * HEAD_DIM;
const QKV: u32 = ATTENTION + 2 * KV_WIDTH;
const GROUP_ROWS: u32 = QUERY_HEADS / KV_HEADS * ROWS;
const PAIRS: usize = HEAD_DIM as usize / 2;
const DYNAMIC: u32 = 2 * 2 * (HIDDEN / CONV_GROUP);
const RING_LANE: usize = (KV_HEADS * WINDOW * HEAD_DIM) as usize;
const PARTIAL_FLOATS: usize = (GROUP_ROWS * HEAD_DIM + 2 * GROUP_ROWS) as usize;
/// Splash's batch width (SPLASH_MAXIMUM_BATCH_WIDTH), the most lanes its tests run.
const MAX_LANES: u32 = 4;
const NAN_BF16: u16 = 0x7FC0;

/// Splash's tests' LCG.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u32 {
        self.0 = (self.0)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next() & 0xFF_FFFF) as f32 / 8_388_608.0 - 1.0
    }
    fn bf16s(&mut self, n: usize, scale: f32) -> Vec<u16> {
        (0..n).map(|_| bits(self.unit() * scale)).collect()
    }
}

fn bits(x: f32) -> u16 {
    bf16::from_f32(x).to_bits()
}

fn value(b: u16) -> f32 {
    bf16::from_bits(b).to_f32()
}

/// The spacing of the bf16 binade holding `x` (Splash `ulpBf16`).
fn ulp_bf16(x: f32) -> f32 {
    let m = x.abs();
    if !m.is_normal() {
        return 2f32.powi(-133);
    }
    2f32.powi(m.log2().floor() as i32 - 7)
}

fn tg(width: usize, height: usize, depth: usize) -> MTLSize {
    MTLSize {
        width,
        height,
        depth,
    }
}

struct Gpu {
    device: Device,
    cache: SpecializedPipelineCache,
}

impl Gpu {
    fn new() -> Option<Self> {
        let Some(di) = detect_device() else {
            eprintln!("skipping: no Metal device");
            return None;
        };
        let device = di.device.clone();
        let cache = SpecializedPipelineCache::new(device.clone(), &[]).expect("shaders");
        Some(Self { device, cache })
    }

    fn pipeline(
        &self,
        library: &'static str,
        symbol: &'static str,
        constants: impl Into<Vec<ConstantValue>>,
    ) -> ComputePipelineState {
        let key = PipelineKey::new(library, symbol, constants.into());
        baked_build(&self.cache, &key).unwrap_or_else(|e| panic!("{symbol}: {e:?}"))
    }

    fn buffer<T: Copy>(&self, data: &[T]) -> Buffer {
        let bytes = std::mem::size_of_val(data);
        let buf = (self.device)
            .newBufferWithLength_options(bytes.max(16), MTLResourceOptions::StorageModeShared)
            .expect("newBuffer");
        // SAFETY: `buf` holds at least `bytes` bytes and does not overlap `data`.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                buf.contents().as_ptr() as *mut u8,
                bytes,
            );
        }
        buf
    }

    fn batch(&self) -> Mtl4DispatchBatch {
        Mtl4DispatchBatch::begin(&self.device).expect("an MTL4 queue")
    }
}

fn read<T: Copy>(buf: &Buffer, n: usize) -> Vec<T> {
    assert!(n * std::mem::size_of::<T>() <= buf.length());
    // SAFETY: `buf` holds at least `n` `T`s (checked above).
    unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const T, n).to_vec() }
}

fn draft_constants(finish: bool) -> DraftConstants {
    DraftConstants {
        rows: ROWS,
        query_heads: QUERY_HEADS,
        kv_heads: KV_HEADS,
        head_dim: HEAD_DIM,
        window: WINDOW,
        splits: ATTENTION_SPLITS,
        hidden: HIDDEN,
        conv_group: CONV_GROUP,
        conv_finish: u32::from(finish),
        rms_eps: RMS_EPS,
        scale: SCALE,
        rope_theta: THETA,
    }
}

/// The cosine and sine of every pair at `positions` positions, `[position][cos | sin]`, as Splash
/// computes them in fp32 (RopeScaling.hpp:46 `rotaryInverseFrequencies`, rope.metal:51-54): the
/// pair's inverse frequency `θ^(−pair / pairs)`, then its angle at the position. The reference
/// the kernels' own computation is checked against; no kernel reads a table.
fn rope_table(positions: u32) -> Vec<f32> {
    let mut table = vec![0.0f32; positions as usize * HEAD_DIM as usize];
    for p in 0..positions as usize {
        for d in 0..PAIRS {
            let angle = p as f32 * THETA.powf(-(d as f32) / PAIRS as f32);
            table[p * HEAD_DIM as usize + d] = angle.cos();
            table[p * HEAD_DIM as usize + PAIRS + d] = angle.sin();
        }
    }
    table
}

// ── draft_conv (draft_attention_metal_test.mm `surroundingPhases`) ─────────────────────────────

/// Both stages of the convolution, for one to four lanes of dyadic values: every output is EXACTLY
/// the bf16 rounding of the f32 arithmetic, the preceding row's tap skipped at each lane's first
/// row.
#[test]
fn draft_conv_is_exact_cpu_arithmetic() {
    let Some(gpu) = Gpu::new() else { return };
    let hidden = HIDDEN as usize;
    let groups = (HIDDEN / CONV_GROUP) as usize;
    let dyadic = |n: usize, multiplier: usize, modulus: usize| -> Vec<u16> {
        (0..n)
            .map(|i| bits(((i * multiplier) % modulus) as f32 / 8.0 - (modulus / 2) as f32 / 8.0))
            .collect()
    };
    for lanes in 1..=MAX_LANES {
        let rows = (lanes * ROWS) as usize;
        let input = dyadic(rows * hidden, 7, 23);
        let dynamic = dyadic(rows * DYNAMIC as usize, 5, 17);
        let base = dyadic(4 * hidden, 3, 13);
        let residual = dyadic(rows * hidden, 11, 19);
        for finish in [false, true] {
            let pso = gpu.pipeline("dflash2", "draft_conv_bf16", draft_constants(finish));
            let (b_in, b_dyn, b_base, b_res) = (
                gpu.buffer(&input),
                gpu.buffer(&dynamic),
                gpu.buffer(&base),
                gpu.buffer(&residual),
            );
            let out = gpu.buffer(&vec![0xFFFFu16; rows * hidden]);
            let mut batch = gpu.batch();
            batch.encode(
                &pso,
                &[
                    (&b_in, 0),
                    (&b_dyn, 1),
                    (&b_base, 2),
                    (&b_res, 3),
                    (&out, 4),
                ],
                &[],
                &[],
                &[],
                tg((ROWS * HIDDEN).div_ceil(256) as usize, lanes as usize, 1),
                tg(256, 1, 1),
            );
            batch.commit(true);
            let got: Vec<u16> = read(&out, rows * hidden);
            let kind = usize::from(finish);
            for row in 0..rows {
                for channel in 0..hidden {
                    let index = row * hidden + channel;
                    let group = channel / CONV_GROUP as usize;
                    let dyn_at = |tap: usize| {
                        value(dynamic[row * DYNAMIC as usize + (kind * 2 + tap) * groups + group])
                    };
                    let mut want = value(input[index])
                        * (value(base[kind * 2 * hidden + channel]) + dyn_at(0));
                    if row % ROWS as usize != 0 {
                        want += value(input[index - hidden])
                            * (value(base[(kind * 2 + 1) * hidden + channel]) + dyn_at(1));
                    }
                    if finish {
                        want += value(residual[index]);
                    }
                    assert_eq!(
                        got[index],
                        bits(want),
                        "lanes {lanes} finish {finish} row {row} channel {channel}"
                    );
                }
            }
        }
    }
}

// ── draft_attention_qkv + draft_attention_reorder (`surroundingPhases`) ───────────────────────

/// Every query and key head RMS-normalized (unit norm weights), rounded to bf16 and rotated by the
/// pair's fp32 angle at its row's position (`rope_table`) — within an ulp of the f64 rotation of
/// the bf16-rounded norms plus an ulp of the larger input — every value head copied, and the
/// reorder a pure permutation; the QKV input unwritten.
#[test]
fn draft_prepare_and_reorder_match_reference() {
    let Some(gpu) = Gpu::new() else { return };
    let starts = [0u32, 2044, 6130, 77];
    let rope = rope_table(6200);
    let ones = vec![bits(1.0); HEAD_DIM as usize];
    let (q_norm, k_norm) = (gpu.buffer(&ones), gpu.buffer(&ones));
    let prepare = gpu.pipeline(
        "dflash2",
        "draft_attention_qkv_bf16",
        draft_constants(true),
    );
    let reorder = gpu.pipeline(
        "dflash2",
        "draft_attention_reorder_bf16",
        draft_constants(true),
    );
    for lanes in 1..=MAX_LANES {
        let rows = (lanes * ROWS) as usize;
        let mut random = Random(0x5eed1234 + u64::from(HIDDEN + lanes));
        let qkv = random.bf16s(rows * QKV as usize, 1.0);
        let positions: Vec<u32> = (0..rows as u32)
            .map(|r| starts[(r / ROWS) as usize] + r % ROWS)
            .collect();
        let b_qkv = gpu.buffer(&qkv);
        let b_pos = gpu.buffer(&positions);
        let queries = gpu.buffer(&vec![0u16; rows * ATTENTION as usize]);
        let row_major = gpu.buffer(&vec![0u16; rows * ATTENTION as usize]);
        let query_keys = gpu.buffer(&vec![0u16; rows * KV_WIDTH as usize]);
        let query_values = gpu.buffer(&vec![0u16; rows * KV_WIDTH as usize]);
        let prepare_groups = (ROWS * KV_WIDTH)
            .div_ceil(256)
            .max(ROWS * (QUERY_HEADS + KV_HEADS));
        let mut batch = gpu.batch();
        batch.encode(
            &prepare,
            &[
                (&b_qkv, 0),
                (&queries, 1),
                (&q_norm, 2),
                (&k_norm, 3),
                (&b_pos, 5),
                (&query_keys, 6),
                (&query_values, 7),
            ],
            &[],
            &[],
            &[],
            tg(prepare_groups as usize, lanes as usize, 1),
            tg(256, 1, 1),
        );
        batch.barrier();
        batch.encode(
            &reorder,
            &[(&queries, 0), (&row_major, 1)],
            &[],
            &[],
            &[],
            tg((ROWS * ATTENTION).div_ceil(256) as usize, lanes as usize, 1),
            tg(256, 1, 1),
        );
        batch.commit(true);
        assert_eq!(
            read::<u16>(&b_qkv, qkv.len()),
            qkv,
            "prepare wrote its QKV input"
        );
        let q: Vec<u16> = read(&queries, rows * ATTENTION as usize);
        let k: Vec<u16> = read(&query_keys, rows * KV_WIDTH as usize);
        let v: Vec<u16> = read(&query_values, rows * KV_WIDTH as usize);
        let (hd, rows8) = (HEAD_DIM as usize, ROWS as usize);
        for lane in 0..lanes as usize {
            for row in 0..rows8 {
                let lane_row = lane * rows8 + row;
                let packed = &qkv[lane_row * QKV as usize..];
                let position = positions[lane_row] as usize;
                for head in 0..(QUERY_HEADS + KV_HEADS) as usize {
                    let query = head < QUERY_HEADS as usize;
                    let h = if query {
                        head
                    } else {
                        head - QUERY_HEADS as usize
                    };
                    let source = &packed[if query { 0 } else { ATTENTION as usize } + h * hd..];
                    let heads = if query { QUERY_HEADS } else { KV_HEADS } as usize;
                    let at = ((lane * heads + h) * rows8 + row) * hd;
                    let prepared = if query {
                        &q[at..at + hd]
                    } else {
                        &k[at..at + hd]
                    };
                    let squares: f64 = (0..hd)
                        .map(|d| f64::from(value(source[d])) * f64::from(value(source[d])))
                        .sum();
                    let inverse = 1.0 / (squares / hd as f64 + f64::from(RMS_EPS)).sqrt();
                    for d in 0..PAIRS {
                        let first =
                            f64::from(value(bits((f64::from(value(source[d])) * inverse) as f32)));
                        let second = f64::from(value(bits(
                            (f64::from(value(source[d + PAIRS])) * inverse) as f32,
                        )));
                        let c = f64::from(rope[position * hd + d]);
                        let s = f64::from(rope[position * hd + PAIRS + d]);
                        let rotated = [first * c - second * s, second * c + first * s];
                        for (half, &want) in rotated.iter().enumerate() {
                            let got = f64::from(value(prepared[d + half * PAIRS]));
                            let bound = f64::from(ulp_bf16(want as f32))
                                + f64::from(ulp_bf16(first.abs().max(second.abs()) as f32));
                            assert!(
                                (got - want).abs() <= bound,
                                "lanes {lanes} lane {lane} row {row} head {head} dim {}: {got} vs {want}",
                                d + half * PAIRS
                            );
                        }
                    }
                    if !query {
                        for d in 0..hd {
                            assert_eq!(
                                v[((lane * KV_HEADS as usize + h) * hd + d) * rows8 + row],
                                packed[(ATTENTION + KV_WIDTH) as usize + h * hd + d],
                                "value copy lane {lane} row {row} head {h} dim {d}"
                            );
                        }
                    }
                }
            }
        }
        let reordered: Vec<u16> = read(&row_major, rows * ATTENTION as usize);
        for lane in 0..lanes as usize {
            let lane_offset = lane * rows8 * ATTENTION as usize;
            for row in 0..rows8 {
                for head in 0..QUERY_HEADS as usize {
                    for dim in 0..hd {
                        assert_eq!(
                            reordered[lane_offset + row * ATTENTION as usize + head * hd + dim],
                            q[lane_offset + (head * rows8 + row) * hd + dim],
                            "reorder lane {lane} row {row} head {head} dim {dim}"
                        );
                    }
                }
            }
        }
    }
}

// ── draft_attention_split + draft_attention_reduce (`runCase`, `referenceRows`) ──────────────

/// One lane's and KV head's rows: each attends the ring window it can see plus all eight current
/// rows, exactly as the kernel's mask defines, in f64.
fn reference_rows(
    queries: &[u16],
    keys: &[u16],
    values: &[u16],
    query_keys: &[u16],
    query_values: &[u16],
    cache_length: u32,
) -> Vec<f32> {
    let (hd, rows, window) = (HEAD_DIM as usize, ROWS as usize, WINDOW as usize);
    let common_start = cache_length.saturating_sub(WINDOW - 1) as usize;
    let old_count = cache_length as usize - common_start;
    let mut output = vec![0.0f32; GROUP_ROWS as usize * hd];
    let mut scores = vec![0.0f64; old_count + rows];
    for row in 0..GROUP_ROWS as usize {
        let proposal = row % rows;
        let query_position = cache_length as usize + proposal;
        let row_start = query_position.saturating_sub(window - 1);
        let hidden_prefix = row_start - common_start;
        let query = &queries[row * hd..(row + 1) * hd];
        let dot = |key: &[u16]| -> f64 {
            (0..hd)
                .map(|d| f64::from(value(query[d])) * f64::from(value(key[d])))
                .sum()
        };
        let mut best = f64::NEG_INFINITY;
        for (key, score) in scores.iter_mut().enumerate() {
            *score = if key >= old_count {
                let current = key - old_count;
                dot(&query_keys[current * hd..]) * f64::from(SCALE)
            } else if key >= hidden_prefix {
                let slot = (common_start + key) % window;
                dot(&keys[slot * hd..]) * f64::from(SCALE)
            } else {
                f64::NEG_INFINITY
            };
            best = best.max(*score);
        }
        let mut sum = 0.0f64;
        let mut accumulated = vec![0.0f64; hd];
        for (key, &score) in scores.iter().enumerate() {
            if score == f64::NEG_INFINITY {
                continue;
            }
            let probability = (score - best).exp();
            sum += probability;
            for (d, acc) in accumulated.iter_mut().enumerate() {
                let v = if key >= old_count {
                    value(query_values[d * rows + key - old_count])
                } else {
                    value(values[d * window + (common_start + key) % window])
                };
                *acc += probability * f64::from(v);
            }
        }
        for d in 0..hd {
            output[row * hd + d] = (accumulated[d] / sum) as f32;
        }
    }
    output
}

/// The split attention and its fixed-order reduce for `lengths.len()` lanes, each lane's ring a
/// pool slot (assigned in reverse lane order) followed by a slot of NaN: a tile read past a ring
/// reaches its row as 0 x NaN, which the finiteness check rejects.
fn attention_case(gpu: &Gpu, lengths: &[u32]) {
    let lanes = lengths.len();
    let mut random = Random(0x5eed0000 + u64::from(lengths[0]));
    let rows = lanes * ROWS as usize;
    let queries = random.bf16s(rows * ATTENTION as usize, 1.0);
    let ring_slots: Vec<u32> = (0..lanes as u32)
        .map(|l| 2 * (lanes as u32 - 1 - l))
        .collect();
    let pool = |random: &mut Random| -> Vec<u16> {
        let mut pool = vec![NAN_BF16; 2 * lanes * RING_LANE];
        for &slot in &ring_slots {
            let at = slot as usize * RING_LANE;
            pool[at..at + RING_LANE].copy_from_slice(&random.bf16s(RING_LANE, 1.0));
        }
        pool
    };
    let (keys, values) = (pool(&mut random), pool(&mut random));
    let query_keys = random.bf16s(rows * KV_WIDTH as usize, 1.0);
    let query_values = random.bf16s(rows * KV_WIDTH as usize, 1.0);
    let positions: Vec<u32> = (0..rows as u32)
        .map(|r| lengths[(r / ROWS) as usize] + r % ROWS)
        .collect();
    let split = gpu.pipeline(
        "dflash2",
        "draft_attention_split_bf16",
        draft_constants(true),
    );
    let reduce = gpu.pipeline(
        "dflash2",
        "draft_attention_reduce_bf16",
        draft_constants(true),
    );
    let b_queries = gpu.buffer(&queries);
    let (b_keys, b_values) = (gpu.buffer(&keys), gpu.buffer(&values));
    let b_slots = gpu.buffer(&ring_slots);
    let b_pos = gpu.buffer(&positions);
    let (b_qk, b_qv) = (gpu.buffer(&query_keys), gpu.buffer(&query_values));
    let partials = gpu.buffer(&vec![
        0.0f32;
        lanes
            * (KV_HEADS * ATTENTION_SPLITS) as usize
            * PARTIAL_FLOATS
    ]);
    let mut batch = gpu.batch();
    batch.encode(
        &split,
        &[
            (&b_queries, 0),
            (&b_keys, 1),
            (&b_values, 2),
            (&b_slots, 3),
            (&b_pos, 4),
            (&b_qk, 5),
            (&b_qv, 6),
            (&partials, 7),
        ],
        &[],
        &[],
        &[],
        tg(KV_HEADS as usize, lanes, ATTENTION_SPLITS as usize),
        tg(256, 1, 1),
    );
    batch.barrier();
    batch.encode(
        &reduce,
        &[(&b_queries, 0), (&partials, 1)],
        &[],
        &[],
        &[],
        tg(KV_HEADS as usize, lanes, 1),
        tg(256, 1, 1),
    );
    batch.commit(true);
    let output: Vec<u16> = read(&b_queries, queries.len());
    let (hd, rows8, window) = (HEAD_DIM as usize, ROWS as usize, WINDOW as usize);
    for lane in 0..lanes {
        // The first and the final head, whose ring the NaN slot follows; all eight run above.
        for head in [0, KV_HEADS as usize - 1] {
            let query_offset = lane * rows8 * ATTENTION as usize + head * GROUP_ROWS as usize * hd;
            let ring = ring_slots[lane] as usize * RING_LANE + head * window * hd;
            let current = (lane * KV_HEADS as usize + head) * rows8 * hd;
            let reference = reference_rows(
                &queries[query_offset..],
                &keys[ring..],
                &values[ring..],
                &query_keys[current..],
                &query_values[current..],
                lengths[lane],
            );
            for (i, &want) in reference.iter().enumerate() {
                let got = value(output[query_offset + i]);
                assert!(
                    got.is_finite() && (got - want).abs() <= 0.02 + 0.02 * want.abs(),
                    "lane {lane} head {head} length {} row {} dim {}: {got} vs {want}",
                    lengths[lane],
                    i / hd,
                    i % hd
                );
            }
        }
    }
}

/// Splash's cases: an unaligned wrap, a wrap starting one slot before the ring's end, the short
/// prefix before any wrap, aligned wraps, windows filling exactly one or several splits, and the
/// ring's last slot — tile bounds, the window mask, empty splits and the empty-row guard.
#[test]
fn draft_attention_matches_reference() {
    let Some(gpu) = Gpu::new() else { return };
    for lengths in [
        &[0u32][..],
        &[2048, 2047],
        &[2100, 4094, 6143],
        &[262137, 4094, 500, 6143],
        &[512, 513, 1024, 1536],
        &[2046, 2049, 4095, 4096],
        &[8191, 262144],
    ] {
        attention_case(&gpu, lengths);
    }
}

// ── The context writers (`contextWriters`) ────────────────────────────────────────────────────

const CONTEXT_ROW: usize = 2 * KV_WIDTH as usize;
const UNTOUCHED: u16 = 0xC2C2;

/// Checks the ring of pool slot `slot` against `rows` rows of `kv` written at `positions`: a row's
/// key RMS-normalized, scaled by the key norm and rotated by its pair's fp32 angle at its position,
/// its value copied, into slot position % 2048 of each KV head's ring; every other slot keeps its
/// bits.
fn check_ring(
    keys: &[u16],
    values: &[u16],
    slot: usize,
    kv: &[u16],
    norm: &[u16],
    rope: &[f32],
    positions: &[u32],
) {
    let (hd, window) = (HEAD_DIM as usize, WINDOW as usize);
    let mut want_values = vec![UNTOUCHED; RING_LANE];
    let mut want_keys = vec![f32::NAN; RING_LANE];
    for (row, &position) in positions.iter().enumerate() {
        let ring_slot = position as usize % window;
        let (c, s) = (
            &rope[position as usize * hd..],
            &rope[position as usize * hd + PAIRS..],
        );
        for head in 0..KV_HEADS as usize {
            let key = &kv[row * CONTEXT_ROW + head * hd..];
            let val = &kv[row * CONTEXT_ROW + KV_WIDTH as usize + head * hd..];
            let square: f32 = (0..hd).map(|d| value(key[d]) * value(key[d])).sum();
            let inverse = 1.0 / (square / hd as f32 + RMS_EPS).sqrt();
            let normalized: Vec<f32> = (0..hd)
                .map(|d| value(bits(value(key[d]) * inverse * value(norm[d]))))
                .collect();
            let out = &mut want_keys[(head * window + ring_slot) * hd..];
            for d in 0..PAIRS {
                out[d] = normalized[d] * c[d] - normalized[d + PAIRS] * s[d];
                out[d + PAIRS] = normalized[d + PAIRS] * c[d] + normalized[d] * s[d];
            }
            for d in 0..hd {
                want_values[(head * hd + d) * window + ring_slot] = val[d];
            }
        }
    }
    let at = slot * RING_LANE;
    for i in 0..RING_LANE {
        let (got_key, got_value) = (keys[at + i], values[at + i]);
        assert_eq!(got_value, want_values[i], "slot {slot} value element {i}");
        let want = want_keys[i];
        if want.is_nan() {
            assert_eq!(
                got_key, UNTOUCHED,
                "slot {slot} key element {i} was written"
            );
        } else {
            let got = value(got_key);
            assert!(
                (got - want).abs() <= 0.02 + 0.02 * want.abs(),
                "slot {slot} key element {i}: {got} vs {want}"
            );
        }
    }
}

fn assert_untouched(ring: &[u16], slot: usize) {
    let at = slot * RING_LANE;
    assert!(
        ring[at..at + RING_LANE].iter().all(|&b| b == UNTOUCHED),
        "pool slot {slot} was written"
    );
}

/// A prompt's commit: two sequences of one step — 37 rows from position 6130 (slot 2034, across
/// the ring's end) and 20 rows from 0 — each into its pool slot by `cu_seqlens`, with the grid
/// rounded past the step's rows; the pool's third slot untouched.
#[test]
fn prefill_draft_context_writes_each_sequence_into_its_ring() {
    let Some(gpu) = Gpu::new() else { return };
    let mut random = Random(0xc0de0000 + u64::from(HIDDEN));
    let norm = random.bf16s(HEAD_DIM as usize, 1.0);
    let rope = rope_table(6200);
    let (first, second) = (37u32, 20u32);
    let tokens = first + second;
    let padded = 64u32;
    let kv = random.bf16s(padded as usize * CONTEXT_ROW, 1.0);
    let mut positions: Vec<u32> = (0..first).map(|r| 6130 + r).chain(0..second).collect();
    positions.resize(padded as usize, 0);
    let ring_slots = [2u32, 0];
    let cu_seqlens = [0u32, first, tokens];
    let pool = vec![UNTOUCHED; 3 * RING_LANE];
    let (b_keys, b_values) = (gpu.buffer(&pool), gpu.buffer(&pool));
    let pso = gpu.pipeline(
        "dflash2",
        "prefill_draft_context_kv_bf16",
        draft_constants(true),
    );
    let (b_kv, b_norm) = (gpu.buffer(&kv), gpu.buffer(&norm));
    let (b_pos, b_slots) = (gpu.buffer(&positions), gpu.buffer(&ring_slots));
    let (b_cu, b_n) = (gpu.buffer(&cu_seqlens), gpu.buffer(&[tokens]));
    let mut batch = gpu.batch();
    batch.encode(
        &pso,
        &[
            (&b_kv, 0),
            (&b_norm, 1),
            (&b_pos, 3),
            (&b_keys, 4),
            (&b_values, 5),
            (&b_slots, 6),
            (&b_cu, 7),
            (&b_n, 8),
        ],
        &[],
        &[],
        &[],
        tg((padded * KV_HEADS) as usize, 1, 1),
        tg(256, 1, 1),
    );
    batch.commit(true);
    let keys: Vec<u16> = read(&b_keys, pool.len());
    let values: Vec<u16> = read(&b_values, pool.len());
    check_ring(
        &keys,
        &values,
        2,
        &kv,
        &norm,
        &rope,
        &positions[..first as usize],
    );
    check_ring(
        &keys,
        &values,
        0,
        &kv[first as usize * CONTEXT_ROW..],
        &norm,
        &rope,
        &positions[first as usize..tokens as usize],
    );
    assert_untouched(&keys, 1);
    assert_untouched(&values, 1);
}

/// A cycle's commit: three lanes from positions 2044 (across the ring's end), 0 and 4101 verifying
/// 8, 3 and 5 rows of their blocks and retaining 8, 2 and 5 of them, each into its pool slot; the
/// pool's fourth slot untouched, and no row past the step's committed.
#[test]
fn draft_context_commit_writes_each_lanes_retained_rows() {
    let Some(gpu) = Gpu::new() else { return };
    let mut random = Random(0xc0de0001 + u64::from(HIDDEN));
    let norm = random.bf16s(HEAD_DIM as usize, 1.0);
    let rope = rope_table(6200);
    let starts = [2044u32, 0, 4101];
    let verify_rows = [8u32, 3, 5];
    let retained = [8u32, 2, 5];
    let ring_slots = [2u32, 0, 1];
    let lanes = starts.len();
    let cu: Vec<u32> = std::iter::once(0)
        .chain(verify_rows.iter().scan(0, |at, &n| {
            *at += n;
            Some(*at)
        }))
        .collect();
    let rows = cu[lanes] as usize;
    let kv = random.bf16s(rows * CONTEXT_ROW, 1.0);
    let positions: Vec<u32> = (0..lanes)
        .flat_map(|lane| (0..verify_rows[lane]).map(move |r| starts[lane] + r))
        .collect();
    let pool = vec![UNTOUCHED; 4 * RING_LANE];
    let (b_keys, b_values) = (gpu.buffer(&pool), gpu.buffer(&pool));
    let pso = gpu.pipeline(
        "dflash2",
        "draft_context_kv_commit_bf16",
        draft_constants(true),
    );
    let (b_kv, b_norm) = (gpu.buffer(&kv), gpu.buffer(&norm));
    let (b_pos, b_slots, b_retained) = (
        gpu.buffer(&positions),
        gpu.buffer(&ring_slots),
        gpu.buffer(&retained),
    );
    let (b_cu, b_n) = (gpu.buffer(&cu), gpu.buffer(&[rows as u32]));
    let mut batch = gpu.batch();
    batch.encode(
        &pso,
        &[
            (&b_kv, 0),
            (&b_norm, 1),
            (&b_pos, 3),
            (&b_keys, 4),
            (&b_values, 5),
            (&b_slots, 6),
            (&b_retained, 7),
            (&b_cu, 8),
            (&b_n, 9),
        ],
        &[],
        &[],
        &[],
        // The grid of a bucket of three whole blocks.
        tg(lanes * ROWS as usize * KV_HEADS as usize, 1, 1),
        tg(256, 1, 1),
    );
    batch.commit(true);
    let keys: Vec<u16> = read(&b_keys, pool.len());
    let values: Vec<u16> = read(&b_values, pool.len());
    for lane in 0..lanes {
        let kept = retained[lane] as usize;
        let first = cu[lane] as usize;
        check_ring(
            &keys,
            &values,
            ring_slots[lane] as usize,
            &kv[first * CONTEXT_ROW..],
            &norm,
            &rope,
            &positions[first..first + kept],
        );
    }
    assert_untouched(&keys, 3);
    assert_untouched(&values, 3);
}

// ── capture_rows (dflash_batch_control_metal_test.mm `testRowCopy`) ───────────────────────────

/// A target layer's 13 output rows (width 2048) into block 2 of 4 of the captured rows, bitwise,
/// with the grid rounded past the rows; every value outside the block keeps its poison.
#[test]
fn capture_rows_copies_into_its_block() {
    let Some(gpu) = Gpu::new() else { return };
    const POISON: u16 = 0xA5A5;
    let (rows, slots, slot) = (13usize, 4usize, 2usize);
    let hidden = HIDDEN as usize;
    let input: Vec<u16> = (0..rows * hidden)
        .map(|i| ((i as u64 * 2654435761) >> 16) as u16)
        .collect();
    let mut want = vec![POISON; rows * slots * hidden];
    for row in 0..rows {
        let at = (row * slots + slot) * hidden;
        want[at..at + hidden].copy_from_slice(&input[row * hidden..(row + 1) * hidden]);
    }
    for symbol in ["capture_rows_f16", "capture_rows_bf16"] {
        let constants = CaptureConstants {
            hidden: HIDDEN,
            slot: slot as u32,
            slots: slots as u32,
            elements: (rows * hidden) as u32,
        };
        let pso = gpu.pipeline("dflash2_capture", symbol, constants);
        let b_in = gpu.buffer(&input);
        let out = gpu.buffer(&vec![POISON; want.len()]);
        let mut batch = gpu.batch();
        batch.encode(
            &pso,
            &[(&b_in, 0), (&out, 1)],
            &[],
            &[],
            &[],
            tg((rows * hidden).div_ceil(256) + 1, 1, 1),
            tg(256, 1, 1),
        );
        batch.commit(true);
        assert_eq!(read::<u16>(&out, want.len()), want, "{symbol}");
    }
}

// ── The selector (draft_selector_metal_test.mm) ───────────────────────────────────────────────

const K: usize = CANDIDATES as usize;
const P: usize = PROPOSALS as usize;

fn select_constants(vocab: u32, lanes: u32, verify: VerifyRows) -> DraftSelectConstants {
    DraftSelectConstants {
        vocab,
        rows: ROWS,
        positions: PROPOSALS,
        candidates: CANDIDATES,
        rank: RANK,
        shards: SAMPLING_SHARDS,
        verify,
        lanes,
    }
}

/// Row patterns: peaked logits with a few spikes (the production shape), uniform noise, heavy ties
/// from a seven-value alphabet with -inf runs, and a row with ten finite tokens so -inf tokens
/// fill the tail by id.
#[derive(Clone, Copy)]
enum Pattern {
    Peaked,
    Uniform,
    Ties,
    Sparse,
}

fn fill_row(row: &mut [f32], pattern: Pattern, random: &mut Random) {
    let vocab = row.len();
    for v in row.iter_mut() {
        *v = match pattern {
            Pattern::Peaked => -8.0 + 3.0 * random.unit() * random.unit(),
            Pattern::Uniform => 4.0 * random.unit(),
            Pattern::Ties => {
                if random.next().is_multiple_of(11) {
                    f32::NEG_INFINITY
                } else {
                    (random.next() % 7) as f32 - 3.0
                }
            }
            Pattern::Sparse => f32::NEG_INFINITY,
        };
    }
    match pattern {
        Pattern::Peaked => {
            for _ in 0..40 {
                row[random.next() as usize % vocab] = 4.0 + 6.0 * random.unit();
            }
        }
        Pattern::Sparse => {
            for _ in 0..10 {
                row[random.next() as usize % vocab] = random.unit();
            }
        }
        Pattern::Uniform | Pattern::Ties => {}
    }
}

/// The row's sixteen largest tokens: value descending, id ascending on ties.
fn reference_top16(row: &[f32]) -> Vec<u32> {
    let beats = |a: &u32, b: &u32| {
        let (x, y) = (row[*a as usize], row[*b as usize]);
        y.partial_cmp(&x).expect("no NaN logits").then(a.cmp(b))
    };
    let mut order: Vec<u32> = (0..row.len() as u32).collect();
    let keep = K.min(row.len());
    if keep < order.len() {
        order.select_nth_unstable_by(keep, beats);
    }
    order.truncate(keep);
    order.sort_by(beats);
    order
}

/// The selector's buffers for `lanes` lanes over `vocab` tokens.
struct Selector {
    logits: Buffer,
    partial_ids: Buffer,
    partial_values: Buffer,
    candidates: Buffer,
    unary: Buffer,
    hidden: Buffer,
    predecessor: Buffer,
    successor: Buffer,
    input_ids: Buffer,
    num_tokens: Buffer,
    uniforms: Buffer,
    tokens: Buffer,
    q_probs: Buffer,
    confidence: Buffer,
    lane_params: Buffer,
}

impl Selector {
    fn run(&self, gpu: &Gpu, vocab: u32, lanes: u32) {
        let constants = || select_constants(vocab, lanes, VerifyRows::EveryRow);
        let top16 = gpu.pipeline(
            "dflash2_select",
            "draft_select_top16_sharded_f32",
            constants(),
        );
        let edges = gpu.pipeline("dflash2_select", "draft_select_edges_bf16", constants());
        let walk = gpu.pipeline("dflash2_select", "draft_select_dflash", constants());
        let positions = lanes as usize * P;
        let mut batch = gpu.batch();
        batch.encode(
            &top16,
            &[
                (&self.logits, 0),
                (&self.partial_ids, 1),
                (&self.partial_values, 2),
            ],
            &[],
            &[],
            &[],
            tg(positions * SAMPLING_SHARDS as usize, 1, 1),
            tg(256, 1, 1),
        );
        batch.barrier();
        batch.encode(
            &edges,
            &[
                (&self.partial_ids, 0),
                (&self.partial_values, 1),
                (&self.candidates, 2),
                (&self.unary, 3),
                (&self.hidden, 4),
                (&self.predecessor, 5),
                (&self.successor, 6),
                (&self.input_ids, 7),
                (&self.num_tokens, 8),
            ],
            &[],
            &[],
            &[],
            tg(positions, 1, 1),
            tg(EDGE_THREADS as usize, 1, 1),
        );
        batch.barrier();
        batch.encode(
            &walk,
            &[
                (&self.candidates, 0),
                (&self.unary, 1),
                (&self.partial_values, 2),
                (&self.uniforms, 3),
                (&self.tokens, 4),
                (&self.q_probs, 5),
                (&self.confidence, 6),
                (&self.lane_params, 7),
                (&self.num_tokens, 8),
            ],
            &[],
            &[],
            &[],
            tg(lanes as usize, 1, 1),
            tg(1, 1, 1),
        );
        batch.commit(true);
    }
}

/// A vocabulary's codebooks, shared by its cases.
struct Codebooks {
    predecessor: Vec<u16>,
    successor: Vec<u16>,
}

impl Codebooks {
    fn new(vocab: u32) -> Self {
        let mut random = Random(0xc0deb00c + u64::from(vocab));
        let n = vocab as usize * RANK as usize;
        Self {
            predecessor: random.bf16s(n, 0.1),
            successor: random.bf16s(n, 0.1),
        }
    }
}

fn selector_buffers(
    gpu: &Gpu,
    logits: &[f32],
    hidden: &[u16],
    codebooks: &Codebooks,
    anchors: &[u32],
    uniforms: &[f32],
    params: &[DraftLaneParams],
) -> Selector {
    let lanes = anchors.len();
    let candidates = lanes * P * K;
    let rows = lanes * ROWS as usize;
    let mut input_ids = vec![0u32; rows];
    for (lane, &anchor) in anchors.iter().enumerate() {
        input_ids[lane * ROWS as usize] = anchor;
    }
    Selector {
        logits: gpu.buffer(logits),
        partial_ids: gpu.buffer(&vec![0u32; candidates * SAMPLING_SHARDS as usize]),
        partial_values: gpu.buffer(&vec![0.0f32; candidates * (SAMPLING_SHARDS as usize + K)]),
        candidates: gpu.buffer(&vec![0u32; candidates]),
        unary: gpu.buffer(&vec![0.0f32; candidates]),
        hidden: gpu.buffer(hidden),
        predecessor: gpu.buffer(&codebooks.predecessor),
        successor: gpu.buffer(&codebooks.successor),
        input_ids: gpu.buffer(&input_ids),
        num_tokens: gpu.buffer(&[rows as u32]),
        uniforms: gpu.buffer(uniforms),
        tokens: gpu.buffer(&vec![0u32; lanes * P]),
        q_probs: gpu.buffer(&vec![0.0f32; candidates]),
        confidence: gpu.buffer(&vec![0.0f32; lanes * P]),
        lane_params: gpu.buffer(params),
    }
}

/// `runCase`: the sharded top-16 and its reduce return exactly the sixteen largest logits of every
/// proposal row in (value desc, id asc) order; the codebook walk picks the tokens a f64 evaluation
/// of the same scores picks; a sampled lane's probabilities are their softmax at its temperature
/// and its draw the reference's; each position's confidence is its candidates' largest softmax
/// probability at temperature 1.
fn select_case(gpu: &Gpu, codebooks: &Codebooks, vocab: u32, lanes: u32, sampling: bool) {
    let mut random =
        Random(0x5e1ec7 + u64::from(vocab) * 8 + u64::from(lanes) * 2 + u64::from(sampling));
    let rows = (lanes * ROWS) as usize;
    let v = vocab as usize;
    let patterns = [
        Pattern::Peaked,
        Pattern::Uniform,
        Pattern::Ties,
        Pattern::Sparse,
    ];
    let mut logits = vec![0.0f32; rows * v];
    for (row, chunk) in logits.chunks_mut(v).enumerate() {
        let pattern = patterns[(row / ROWS as usize + row % ROWS as usize) % patterns.len()];
        fill_row(chunk, pattern, &mut random);
    }
    let hidden = random.bf16s(rows * RANK as usize, 0.1);
    let uniforms: Vec<f32> = (0..lanes as usize * SAMPLING_UNIFORMS as usize)
        .map(|_| (random.unit() + 1.0) * 0.5)
        .collect();
    let anchors: Vec<u32> = (0..lanes).map(|_| random.next() % vocab).collect();
    let temperature = 0.8f32;
    let params: Vec<DraftLaneParams> = (0..lanes)
        .map(|_| DraftLaneParams {
            temperature: if sampling { temperature } else { 0.0 },
            sampling: u32::from(sampling),
            maximum: ROWS,
            remaining: ROWS,
        })
        .collect();
    let s = selector_buffers(
        gpu, &logits, &hidden, codebooks, &anchors, &uniforms, &params,
    );
    s.run(gpu, vocab, lanes);

    let n = lanes as usize * P * K;
    let candidates: Vec<u32> = read(&s.candidates, n);
    let unary: Vec<f32> = read(&s.unary, n);
    let tokens: Vec<u32> = read(&s.tokens, lanes as usize * P);
    let probabilities: Vec<f32> = read(&s.q_probs, n);
    let confidence: Vec<f32> = read(&s.confidence, lanes as usize * P);
    let rank = RANK as usize;
    for lane in 0..lanes as usize {
        let mut predecessor = anchors[lane] as usize;
        for position in 0..P {
            let global = lane * P + position;
            let row_index = lane * ROWS as usize + position + 1;
            let row = &logits[row_index * v..(row_index + 1) * v];
            let expected = reference_top16(row);
            for r in 0..K {
                let (id, value_at) = (candidates[global * K + r], unary[global * K + r]);
                if r < expected.len() {
                    assert!(
                        id == expected[r] && value_at == row[expected[r] as usize],
                        "vocab {vocab} lanes {lanes} lane {lane} position {position} rank {r}: \
                         ({id}, {value_at}) vs ({}, {})",
                        expected[r],
                        row[expected[r] as usize]
                    );
                } else {
                    assert!(
                        id == u32::MAX && value_at == f32::NEG_INFINITY,
                        "padding sentinel"
                    );
                }
            }
            // Scores as the kernel defines them, in f64.
            let scores: Vec<f64> = (0..K)
                .map(|r| {
                    let candidate = (candidates[global * K + r] as usize).min(v - 1);
                    let edge: f64 = (0..rank)
                        .map(|d| {
                            f64::from(value(codebooks.predecessor[predecessor * rank + d]))
                                * f64::from(value(hidden[row_index * rank + d]))
                                * f64::from(value(codebooks.successor[candidate * rank + d]))
                        })
                        .sum();
                    f64::from(unary[global * K + r]) + edge
                })
                .collect();
            let token = tokens[global];
            let selected = (0..K)
                .find(|&r| candidates[global * K + r] == token)
                .expect("the selector proposed a token outside its candidates");
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if sampling {
                let reference: Vec<f64> = scores
                    .iter()
                    .map(|s| ((s - maximum) / f64::from(temperature)).exp())
                    .collect();
                let sum: f64 = reference.iter().sum();
                let uniform = uniforms[lane * SAMPLING_UNIFORMS as usize + 1 + position];
                let mut cumulative = [0.0f64; K];
                let mut expected_selection = K - 1;
                for r in 0..K {
                    let probability = probabilities[global * K + r];
                    assert!(
                        (f64::from(probability) - reference[r] / sum).abs() < 1e-4,
                        "probabilities diverged from the softmax"
                    );
                    cumulative[r] =
                        if r > 0 { cumulative[r - 1] } else { 0.0 } + reference[r] / sum;
                    if expected_selection == K - 1 && cumulative[r] > f64::from(uniform) {
                        expected_selection = r;
                    }
                }
                // A draw within f32 rounding of the crossed boundary may go either way.
                let boundary =
                    (cumulative[selected.min(expected_selection)] - f64::from(uniform)).abs();
                assert!(
                    selected == expected_selection || boundary < 1e-5,
                    "lane {lane} position {position} drew {selected}, the reference \
                     {expected_selection}"
                );
            } else {
                let mut expected_selection = 0;
                for r in 1..K {
                    if scores[r] > scores[expected_selection] {
                        expected_selection = r;
                    }
                }
                assert!(
                    selected == expected_selection
                        || (scores[selected] - scores[expected_selection]).abs() < 1e-5,
                    "lane {lane} position {position} picked {selected}, the reference \
                     {expected_selection}"
                );
            }
            let sum: f64 = scores.iter().map(|s| (s - maximum).exp()).sum();
            assert!(
                (f64::from(confidence[global]) - 1.0 / sum).abs() < 1e-4,
                "confidence is not the largest candidate probability"
            );
            predecessor = token as usize;
        }
    }
}

/// The production vocabulary, an odd one that misaligns the vector loads and leaves shards a few
/// tokens, and one wider than a register chunk per thread; one to four lanes, greedy and sampled.
#[test]
fn draft_selector_matches_reference() {
    let Some(gpu) = Gpu::new() else { return };
    for vocab in [248_320u32, 1003, 270_005] {
        let codebooks = Codebooks::new(vocab);
        for lanes in 1..=MAX_LANES {
            for sampling in [false, true] {
                select_case(&gpu, &codebooks, vocab, lanes, sampling);
            }
        }
    }
}

/// `confidenceIgnoresDraws`: runs differing only in one position's proposal uniform write the same
/// confidences up to that position and the same proposals before it, while (over flat candidate
/// scores) drawing different proposals there — so whether a row is verified never depends on its
/// own draw.
#[test]
fn draft_confidence_ignores_draws() {
    let Some(gpu) = Gpu::new() else { return };
    let vocab = 1003u32;
    let mut random = Random(0xd2a35);
    let mut logits = vec![0.0f32; ROWS as usize * vocab as usize];
    for chunk in logits.chunks_mut(vocab as usize) {
        fill_row(chunk, Pattern::Uniform, &mut random);
    }
    let hidden = random.bf16s(ROWS as usize * RANK as usize, 0.1);
    let codebooks = Codebooks::new(vocab);
    let params = [DraftLaneParams {
        temperature: 1.0,
        sampling: 1,
        maximum: ROWS,
        remaining: ROWS,
    }];
    let run = |position: usize, uniform: f32| {
        let mut uniforms = vec![0.5f32; SAMPLING_UNIFORMS as usize];
        uniforms[1 + position] = uniform;
        let s = selector_buffers(&gpu, &logits, &hidden, &codebooks, &[7], &uniforms, &params);
        s.run(&gpu, vocab, 1);
        (read::<u32>(&s.tokens, P), read::<f32>(&s.confidence, P))
    };
    let mut redrawn = 0;
    for position in 0..P {
        let (low_tokens, low_confidence) = run(position, 0.05);
        let (high_tokens, high_confidence) = run(position, 0.95);
        assert_eq!(
            low_confidence[..=position],
            high_confidence[..=position],
            "a draw changed the confidence at or before its position"
        );
        assert_eq!(
            low_tokens[..position],
            high_tokens[..position],
            "a draw changed the proposals before its position"
        );
        redrawn += usize::from(low_tokens[position] != high_tokens[position]);
    }
    assert!(redrawn >= 5, "the uniforms redrew only {redrawn} proposals");
}

/// `verifyRowsChoice`: for random confidences (with exact 0 and 1), acceptance models, costs
/// (Splash's seed, and random nondecreasing shapes, flat among them) and lane maxima, a budget's
/// verify rows' expected tokens per cost, evaluated in f64, is the best of every assignment of 1
/// to its maximum rows per lane; with every row, each lane verifies its maximum.
#[test]
fn draft_verify_rows_choose_the_best_assignment() {
    let Some(gpu) = Gpu::new() else { return };
    let mut random = Random(0xb0d9e7);
    let lanes_max = MAX_LANES as usize;
    let mut models = vec![
        VerifyRows::EveryRow,
        VerifyRows::of_target(true),
    ];
    let flat = VerifyCost([[1.0; 4]; BUDGET_LANES]);
    let steep = VerifyCost([[0.1, 0.2, 0.4, 0.8]; BUDGET_LANES]);
    for cost in [flat, steep, VERIFY_COST_SEED] {
        models.push(VerifyRows::Budget {
            slope: 0.3 + 0.5 * (random.unit() + 1.0),
            intercepts: std::array::from_fn(|_| 1.5 * random.unit()),
            cost,
        });
    }
    for (m, &model) in models.iter().enumerate() {
        let pso = gpu.pipeline(
            "dflash2_select",
            "draft_verify_rows",
            select_constants(1003, MAX_LANES, model),
        );
        for trial in 0..40 {
            let lanes = 1 + trial % lanes_max;
            let q: Vec<f32> = (0..lanes * P)
                .map(|_| match random.next() % 8 {
                    0 => 0.0,
                    1 => 1.0,
                    _ => (random.unit() + 1.0) * 0.5,
                })
                .collect();
            let maxima: Vec<u32> = (0..lanes).map(|_| 1 + random.next() % ROWS).collect();
            let params: Vec<DraftLaneParams> = (maxima.iter())
                .map(|&maximum| DraftLaneParams {
                    temperature: 0.0,
                    sampling: 0,
                    maximum,
                    remaining: ROWS,
                })
                .collect();
            let (b_q, b_params) = (gpu.buffer(&q), gpu.buffer(&params));
            let b_rows = gpu.buffer(&vec![0u32; lanes]);
            let b_n = gpu.buffer(&[lanes as u32 * ROWS]);
            let mut batch = gpu.batch();
            batch.encode(
                &pso,
                &[(&b_q, 0), (&b_rows, 1), (&b_params, 2), (&b_n, 3)],
                &[],
                &[],
                &[],
                tg(1, 1, 1),
                tg(1, 1, 1),
            );
            batch.commit(true);
            let chosen: Vec<u32> = read(&b_rows, lanes);
            for (lane, (&c, &maximum)) in chosen.iter().zip(&maxima).enumerate() {
                assert!(
                    (1..=maximum).contains(&c),
                    "model {m} trial {trial} lane {lane}: {c} rows of at most {maximum}"
                );
            }
            let VerifyRows::Budget {
                slope,
                intercepts,
                cost,
            } = model
            else {
                assert_eq!(chosen, maxima, "every row did not verify the maxima");
                continue;
            };
            let survival: Vec<Vec<f64>> = (0..lanes)
                .map(|lane| {
                    let mut s = 1.0f64;
                    (0..P)
                        .map(|position| {
                            let c = f64::from(q[lane * P + position].clamp(1e-6, 1.0 - 1e-6));
                            let z = f64::from(slope) * (c / (1.0 - c)).ln()
                                + f64::from(intercepts[position]);
                            s /= 1.0 + (-z).exp();
                            s
                        })
                        .collect()
                })
                .collect();
            let value_of = |rows: &[u32]| -> f64 {
                let mut expected = 0.0;
                let mut total = 0u32;
                for (lane, &r) in rows.iter().enumerate() {
                    expected += 1.0 + survival[lane][..r as usize - 1].iter().sum::<f64>();
                    total += r;
                }
                expected / f64::from(cost.of(lanes as u32, total))
            };
            let mut rows = vec![1u32; lanes];
            let mut best = 0.0f64;
            loop {
                best = best.max(value_of(&rows));
                let Some(lane) = (0..lanes).find(|&l| rows[l] < maxima[l]) else {
                    break;
                };
                rows[lane] += 1;
                rows[..lane].fill(1);
            }
            let picked = value_of(&chosen);
            assert!(
                picked >= best * (1.0 - 1e-5),
                "model {m} trial {trial}: chose {picked} tokens per cost where {best} was possible"
            );
        }
    }
}

// ── dflash2_accept (dflash_batch_control_metal_test.mm `runWidth`) ─────────────────────────────

const MASK: u32 = 248_077;
const STOP: u32 = 248_044;

/// One acceptance's buffers: each lane's `VerifyLane`, its drafts (`[lanes, proposals]`), the
/// step's target tokens, and a sampled lane's candidates, their probabilities, its verify rows'
/// target probabilities of their drafts (`TargetVocabularyRow`'s last word) and its uniforms.
struct Acceptance {
    lanes: Vec<VerifyLane>,
    drafted: Vec<u32>,
    tokens: Vec<u32>,
    draft_ids: Vec<u32>,
    draft_probs: Vec<f32>,
    target_rows: Vec<[u32; 4]>,
    uniforms: Vec<f32>,
}

/// What an acceptance left: each lane's retained rows and accepted drafts, its next block's ids
/// and positions, and the step's tokens (a sampled lane's accepted rows its drafts).
struct Accepted {
    retained: Vec<u32>,
    accepted: Vec<u32>,
    ids: Vec<u32>,
    positions: Vec<u32>,
    tokens: Vec<u32>,
}

impl Acceptance {
    fn of(lanes: usize) -> Self {
        let candidates = lanes * P * CANDIDATES as usize;
        Self {
            lanes: Vec::new(),
            drafted: vec![0; lanes * P],
            tokens: Vec::new(),
            draft_ids: vec![u32::MAX; candidates],
            draft_probs: vec![0.0; candidates],
            target_rows: vec![[0; 4]; lanes * ROWS as usize],
            uniforms: vec![0.0; lanes * SAMPLING_UNIFORMS as usize],
        }
    }

    fn run(&self, gpu: &Gpu) -> Accepted {
        use scratchy_target_metal::tape::kernel_constants::DraftAcceptConstants;
        let lanes = self.lanes.len();
        let pso = gpu.pipeline(
            "dflash2_accept",
            "dflash2_accept",
            DraftAcceptConstants {
                rows: ROWS,
                mask: MASK,
            },
        );
        let b_lanes = gpu.buffer(&self.lanes);
        let (b_drafted, b_tokens) = (gpu.buffer(&self.drafted), gpu.buffer(&self.tokens));
        let zeros = vec![0u32; lanes];
        let (b_retained, b_accepted) = (gpu.buffer(&zeros), gpu.buffer(&zeros));
        let block = vec![0u32; lanes * ROWS as usize];
        let (b_ids, b_positions) = (gpu.buffer(&block), gpu.buffer(&block));
        let (b_draft_ids, b_draft_probs) =
            (gpu.buffer(&self.draft_ids), gpu.buffer(&self.draft_probs));
        let (b_rows, b_uniforms) = (gpu.buffer(&self.target_rows), gpu.buffer(&self.uniforms));
        let mut batch = gpu.batch();
        batch.encode(
            &pso,
            &[
                (&b_lanes, 0),
                (&b_drafted, 1),
                (&b_tokens, 2),
                (&b_retained, 3),
                (&b_accepted, 4),
                (&b_ids, 5),
                (&b_positions, 6),
                (&b_draft_ids, 7),
                (&b_draft_probs, 8),
                (&b_rows, 9),
                (&b_uniforms, 10),
            ],
            &[],
            &[],
            &[],
            tg(lanes, 1, 1),
            tg(1, 1, 1),
        );
        batch.commit(true);
        Accepted {
            retained: read(&b_retained, lanes),
            accepted: read(&b_accepted, lanes),
            ids: read(&b_ids, block.len()),
            positions: read(&b_positions, block.len()),
            tokens: read(&b_tokens, self.tokens.len()),
        }
    }
}

/// Greedy lane `lane` of `drafts` drafts, of which its target's tokens match the first `matched`,
/// its last verify row a token of its own.
fn greedy_lane(a: &mut Acceptance, lane: usize, drafts: u32, matched: u32, remaining: u32) {
    let first = a.tokens.len() as u32;
    for token in 0..drafts as usize {
        let proposal = 1000 + lane as u32 * 100 + token as u32;
        a.drafted[lane * P + token] = proposal;
        let matches = (token as u32) < matched;
        a.tokens.push(if matches { proposal } else { proposal + 50 });
    }
    a.tokens.push(9000 + lane as u32);
    a.lanes.push(VerifyLane {
        sampled: first,
        drafts,
        position: 100 * lane as u32 + 7,
        remaining,
        stop: [STOP, 248_046],
        ..VerifyLane::default()
    });
}

/// Each lane's next block: its last retained token, then the mask, from the position after its
/// retained rows.
fn assert_next_blocks(a: &Acceptance, got: &Accepted) {
    for (lane, params) in a.lanes.iter().enumerate() {
        let kept = got.retained[lane];
        let anchor = got.tokens[(params.sampled + kept - 1) as usize];
        let row = |r: usize| lane * ROWS as usize + r;
        assert_eq!(got.ids[row(0)], anchor, "lane {lane}'s next anchor");
        for r in 0..ROWS as usize {
            if r > 0 {
                assert_eq!(got.ids[row(r)], MASK, "lane {lane} row {r} embeds the mask");
            }
            let want = params.position + kept + r as u32;
            assert_eq!(got.positions[row(r)], want, "lane {lane} row {r}'s position");
        }
    }
}

/// One greedy acceptance of `lanes` lanes, lane `l` carrying `drafts[l]` drafts of which its
/// target's tokens match the first `matched[l]`; `stop`: a lane and verify row whose target token
/// is a stop token (a draft too when it is accepted). Each lane retains its accepted drafts and the
/// correction, at most `remaining` of them, cut after a stop token.
fn accept_case(
    gpu: &Gpu,
    drafts: &[u32],
    matched: &[u32],
    remaining: &[u32],
    stop: Option<(usize, usize)>,
) {
    let mut a = Acceptance::of(drafts.len());
    for lane in 0..drafts.len() {
        greedy_lane(&mut a, lane, drafts[lane], matched[lane], remaining[lane]);
        if let Some((_, row)) = stop.filter(|&(l, _)| l == lane) {
            a.tokens[a.lanes[lane].sampled as usize + row] = STOP;
            if (row as u32) < matched[lane] {
                a.drafted[lane * P + row] = STOP;
            }
        }
    }
    let got = a.run(gpu);
    for lane in 0..drafts.len() {
        let want_accepted = matched[lane].min(drafts[lane]);
        let mut want_retained = (want_accepted + 1).min(remaining[lane]);
        if let Some((l, row)) = stop
            && l == lane
            && (row as u32) < want_retained
        {
            want_retained = row as u32 + 1;
        }
        assert_eq!(got.accepted[lane], want_accepted, "lane {lane} accepted");
        assert_eq!(got.retained[lane], want_retained, "lane {lane} retained");
    }
    assert_eq!(got.tokens, a.tokens, "a greedy acceptance rewrote a target token");
    assert_next_blocks(&a, &got);
}

/// Splash's widths and counts: one to four lanes accepting none to three, then four to seven, of
/// seven drafts; output limits and a stop token shortening what lanes retain; lanes verifying
/// fewer drafts; and a lane that decoded without drafts retaining its one row.
#[test]
fn draft_acceptance_retains_what_splash_retains() {
    let Some(gpu) = Gpu::new() else { return };
    let full = [7u32; 4];
    for width in 1..=4 {
        accept_case(
            &gpu,
            &full[..width],
            &[0, 1, 2, 3][..width],
            &[8; 4][..width],
            None,
        );
        accept_case(
            &gpu,
            &full[..width],
            &[4, 5, 6, 7][..width],
            &[8; 4][..width],
            None,
        );
    }
    accept_case(&gpu, &full, &[7; 4], &[1, 2, 3, 8], Some((3, 3)));
    accept_case(&gpu, &[1, 2, 4, 0], &[7; 4], &[8; 4], None);
    accept_case(&gpu, &[1, 2, 4, 0], &[0, 1, 2, 3], &[1, 2, 3, 8], None);
}

/// `accept_sampled_lane`: a sampled lane accepts draft `j` while its acceptance uniform times the
/// draft's probability (looked up among its candidates; 0 for one not among them) is below its
/// row's target probability of it, keeps its target's draws past the accepted rows, and retains the
/// correction: a lane rejecting its fourth draft, one rejecting its first, one whose draft is not
/// among its candidates (accepted whenever its target gives it any probability), one accepting
/// every draft and keeping the bonus draw, and a greedy lane beside them.
#[test]
fn sampled_acceptance_tests_each_draft_against_its_target_probability() {
    let Some(gpu) = Gpu::new() else { return };
    // Per sampled lane, per draft: (q, p, u), accepted iff u · q < p.
    let lanes: [&[(f32, f32, f32)]; 4] = [
        &[(0.5, 0.4, 0.7), (0.9, 0.2, 0.1), (0.3, 0.3, 0.9), (0.6, 0.1, 0.5)],
        &[(0.8, 0.1, 0.5), (0.5, 0.5, 0.0)],
        &[(0.0, 0.2, 0.9), (0.4, 0.0, 0.1)],
        &[(0.2, 0.3, 0.9), (0.7, 0.6, 0.8), (0.1, 0.2, 0.0)],
    ];
    let want_accepted = [3u32, 0, 1, 3];
    let mut a = Acceptance::of(lanes.len() + 1);
    for (lane, draws) in lanes.iter().enumerate() {
        let first = a.tokens.len() as u32;
        let uniforms = lane * SAMPLING_UNIFORMS as usize;
        for (j, &(q, p, u)) in draws.iter().enumerate() {
            let draft = 2000 + 100 * lane as u32 + j as u32;
            a.drafted[lane * P + j] = draft;
            let candidates = (lane * P + j) * CANDIDATES as usize;
            for c in 0..CANDIDATES as usize {
                a.draft_ids[candidates + c] = 50_000 + c as u32;
                a.draft_probs[candidates + c] = 0.01;
            }
            if q > 0.0 {
                let at = candidates + (j * 5) % CANDIDATES as usize;
                a.draft_ids[at] = draft;
                a.draft_probs[at] = q;
            }
            a.target_rows[lane * ROWS as usize + j][3] = p.to_bits();
            // DRAFT_UNIFORM_ACCEPTANCE + j.
            a.uniforms[uniforms + 1 + P + j] = u;
            a.tokens.push(7000 + 100 * lane as u32 + j as u32);
        }
        a.tokens.push(7000 + 100 * lane as u32 + draws.len() as u32);
        a.lanes.push(VerifyLane {
            sampled: first,
            drafts: draws.len() as u32,
            position: 300 * lane as u32 + 11,
            remaining: 8,
            stop: [STOP, 248_046],
            sampling: 1,
            temperature: 1.0,
            top_p: 1.0,
            ..VerifyLane::default()
        });
    }
    let greedy = lanes.len();
    greedy_lane(&mut a, greedy, 5, 2, 8);
    let got = a.run(&gpu);
    for (lane, draws) in lanes.iter().enumerate() {
        let accepted = want_accepted[lane];
        assert_eq!(got.accepted[lane], accepted, "lane {lane} accepted");
        assert_eq!(got.retained[lane], accepted + 1, "lane {lane} retained");
        let first = a.lanes[lane].sampled as usize;
        for row in 0..=draws.len() {
            let want = match (row as u32) < accepted {
                true => a.drafted[lane * P + row],
                false => a.tokens[first + row],
            };
            assert_eq!(got.tokens[first + row], want, "lane {lane} row {row}'s token");
        }
    }
    assert_eq!(
        (got.accepted[greedy], got.retained[greedy]),
        (2, 3),
        "the greedy lane"
    );
    assert_next_blocks(&a, &got);
}
