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
    ATTENTION_SPLITS, BUDGET_LANES, CANDIDATES, DraftLaneParams, DraftSeed, EDGE_THREADS,
    PROPOSALS, SAMPLING_SHARDS, UNIFORM_ACCEPTANCE, UNIFORM_PROPOSALS, VERIFY_COST_SEED,
    VerifyCost, VerifyLane, VerifyRows, draft_uniform,
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
    let prepare = gpu.pipeline("dflash2", "draft_attention_qkv_bf16", draft_constants(true));
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
                (&self.tokens, 3),
                (&self.q_probs, 4),
                (&self.confidence, 5),
                (&self.lane_params, 6),
                (&self.num_tokens, 7),
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
    let anchors: Vec<u32> = (0..lanes).map(|_| random.next() % vocab).collect();
    let temperature = 0.8f32;
    let params: Vec<DraftLaneParams> = (0..lanes)
        .map(|_| DraftLaneParams {
            temperature: if sampling { temperature } else { 0.0 },
            sampling: u32::from(sampling),
            maximum: ROWS,
            remaining: ROWS,
            seed: DraftSeed::of(u64::from(random.next()) << 32 | u64::from(random.next())),
            position: random.next() % 100_000,
        })
        .collect();
    let s = selector_buffers(gpu, &logits, &hidden, codebooks, &anchors, &params);
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
                let draw = UNIFORM_PROPOSALS + position as u32;
                let uniform = draft_uniform(params[lane].seed, params[lane].position, draw);
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

/// `confidenceIgnoresDraws`: runs of other seeds — their proposals' uniforms all others
/// (`draft_uniform`) — that drew the same proposals before a position write the same confidences up
/// to it, while (over flat candidate scores) drawing different proposals — so whether a row is
/// verified never depends on its own draw. And one seed draws what it drew.
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
    let run = |seed: u64| {
        let params = [DraftLaneParams {
            temperature: 1.0,
            sampling: 1,
            maximum: ROWS,
            remaining: ROWS,
            seed: DraftSeed::of(seed),
            position: 4096,
        }];
        let s = selector_buffers(&gpu, &logits, &hidden, &codebooks, &[7], &params);
        s.run(&gpu, vocab, 1);
        (read::<u32>(&s.tokens, P), read::<f32>(&s.confidence, P))
    };
    let runs: Vec<_> = (0..12u64)
        .map(|seed| run(seed * 0x5851_f42d_4c95_7f2d))
        .collect();
    assert_eq!(
        run(0),
        runs[0],
        "a seed drew other proposals at the same position"
    );
    let mut redrawn = 0;
    for (a, (a_tokens, a_confidence)) in runs.iter().enumerate() {
        for (b_tokens, b_confidence) in &runs[a + 1..] {
            let agree = (0..P).find(|&p| a_tokens[p] != b_tokens[p]);
            let through = agree.map_or(P, |p| p + 1);
            assert_eq!(
                a_confidence[..through],
                b_confidence[..through],
                "a draw changed the confidence at or before its position"
            );
            redrawn += usize::from(agree.is_some());
        }
    }
    assert!(
        redrawn >= 30,
        "other seeds redrew only {redrawn} of 66 pairs' proposals"
    );
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
    let mut models = vec![VerifyRows::EveryRow, VerifyRows::of_target(true)];
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
                    maximum,
                    remaining: ROWS,
                    ..DraftLaneParams::default()
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
/// target probabilities of their drafts (`TargetVocabularyRow`'s last word).
struct Acceptance {
    lanes: Vec<VerifyLane>,
    drafted: Vec<u32>,
    tokens: Vec<u32>,
    draft_ids: Vec<u32>,
    draft_probs: Vec<f32>,
    target_rows: Vec<[u32; 4]>,
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
        let b_rows = gpu.buffer(&self.target_rows);
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
        a.tokens
            .push(if matches { proposal } else { proposal + 50 });
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
            assert_eq!(
                got.positions[row(r)],
                want,
                "lane {lane} row {r}'s position"
            );
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
    assert_eq!(
        got.tokens, a.tokens,
        "a greedy acceptance rewrote a target token"
    );
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
/// every draft and keeping the bonus draw, and a greedy lane beside them. Draft `j`'s uniform is
/// its step's draw `UNIFORM_ACCEPTANCE + j` of its seed and position (`draft_uniform`), and each
/// target probability sits a float either side of that uniform times the draft's probability: a
/// uniform other than Splash's draw flips verdicts.
#[test]
fn sampled_acceptance_tests_each_draft_against_its_target_probability() {
    let Some(gpu) = Gpu::new() else { return };
    // Per sampled lane, per draft: (q, accepted).
    let lanes: [&[(f32, bool)]; 4] = [
        &[(0.5, true), (0.9, true), (0.3, true), (0.6, false)],
        &[(0.8, false), (0.5, true)],
        &[(0.0, true), (0.4, false)],
        &[(0.2, true), (0.7, true), (0.1, true)],
    ];
    let want_accepted = [3u32, 0, 1, 3];
    let mut a = Acceptance::of(lanes.len() + 1);
    for (lane, draws) in lanes.iter().enumerate() {
        let first = a.tokens.len() as u32;
        let (seed, position) = (
            DraftSeed::of(0x9e37 + 77 * lane as u64),
            300 * lane as u32 + 11,
        );
        for (j, &(q, accept)) in draws.iter().enumerate() {
            let u = draft_uniform(seed, position, UNIFORM_ACCEPTANCE + j as u32);
            // The least target probability above u · q accepts; u · q itself rejects.
            let p = match accept {
                true => f32::from_bits((u * q).to_bits() + 1).max(f32::MIN_POSITIVE),
                false => u * q,
            };
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
            a.tokens.push(7000 + 100 * lane as u32 + j as u32);
        }
        a.tokens.push(7000 + 100 * lane as u32 + draws.len() as u32);
        a.lanes.push(VerifyLane {
            sampled: first,
            drafts: draws.len() as u32,
            position,
            remaining: 8,
            stop: [STOP, 248_046],
            sampling: 1,
            temperature: 1.0,
            top_p: 1.0,
            seed,
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
            assert_eq!(
                got.tokens[first + row],
                want,
                "lane {lane} row {row}'s token"
            );
        }
    }
    assert_eq!(
        (got.accepted[greedy], got.retained[greedy]),
        (2, 3),
        "the greedy lane"
    );
    assert_next_blocks(&a, &got);
}

// ── The target policy (Splash dev/tests/engine/target_sampling_metal_test.mm) ─────────────────
//
// Splash's target policy against a direct CPU reference, transcribed. A sampled verify row draws
// from its distribution over the whole vocabulary — the tokens min_p leaves, the top-k of those,
// then top-p of their mass — and its largest logit, its draft token's probability and its draw
// must match an evaluation of the same rules in f64: the draw falls where the reference
// distribution's cumulative sum in token order places the uniform. Data within float rounding of a
// min_p or top_p cut or of a draw's boundary are allowed either way, never passing by chance.
//
// Scratchy's differences, the tests' only ones: the logits are bf16; a row's uniforms are its
// lane's step's draws (`draft_uniform`), not uploaded, so where Splash pins an acceptance uniform
// the draft probability is set against the drawn one instead; a greedy lane's rows take scratchy's
// argmax and this policy leaves them; a grammar's mask arrives as -inf logits (`grammar_mask`);
// penalties wait for their port.

use scratchy_target_metal::dflash2::UNIFORM_CORRECTION;
use scratchy_target_metal::tape::kernel_constants::TargetSamplingConstants;

const TARGET_SHARDS: usize = 16;
const VOCABULARY_THREADS: usize = 1024;
const VOCABULARY_GROUPS: usize = 8;
const VOCABULARY_RANGES: usize = VOCABULARY_GROUPS * (VOCABULARY_THREADS / 32);
const SENTINEL: u32 = u32::MAX;
/// Splash's stop tokens, below every spike of `spiked_row`.
const STOPS: [u32; 2] = [1, 2];
/// How close a top_p cut may lie to a token's preceding mass, relative to the top-k mass, before
/// float sums over the vocabulary could keep or cut it either way (Splash `kTopPMargin`).
const TOP_P_MARGIN: f64 = 1e-5;
/// How close a logit may lie to the min_p cut, relative to the larger of the cut and the row's
/// maximum (Splash `kMinPMargin`).
const MIN_P_MARGIN: f64 = 1e-5;

/// A lane's policy (Splash `SamplingPolicy`): top-k (0: every token), temperature, top-p, min-p,
/// and whether it ignores end-of-sequence.
#[derive(Clone, Copy, Debug)]
struct Policy {
    top_k: u32,
    temperature: f32,
    top_p: f32,
    min_p: f32,
    excludes_stop: bool,
}

impl Policy {
    const fn new(top_k: u32, temperature: f32, top_p: f32) -> Self {
        Self {
            top_k,
            temperature,
            top_p,
            min_p: 0.0,
            excludes_stop: false,
        }
    }
    const fn min_p(self, min_p: f32) -> Self {
        Self { min_p, ..self }
    }
    const fn excluding_stop(self) -> Self {
        Self {
            excludes_stop: true,
            ..self
        }
    }
}

/// The tokens a row may select (Splash `Admission`): its mask's, less the stop tokens when its lane
/// ignores end-of-sequence.
#[derive(Clone, Copy, Default)]
struct Admission<'a> {
    mask: Option<&'a [bool]>,
    excludes_stop: bool,
    stops: [u32; 2],
}

impl Admission<'_> {
    fn admits(&self, token: u32) -> bool {
        self.mask.is_none_or(|m| m[token as usize])
            && !(self.excludes_stop && self.stops.contains(&token))
    }
}

/// A row's distribution (Splash `Distribution`): its admitted tokens in the order of the logits,
/// those min_p, top_k and top_p keep, each one's probability, and those within float rounding of a
/// cut (`ambiguous`, of `ambiguous_mass`).
#[derive(Clone, Default)]
struct Distribution {
    order: Vec<u32>,
    maximum: f32,
    mass: f64,
    margin: f64,
    ambiguous_mass: f64,
    probabilities: Vec<f64>,
    ambiguous: Vec<bool>,
}

impl Distribution {
    fn probability(&self, token: u32) -> f64 {
        self.probabilities
            .get(token as usize)
            .copied()
            .unwrap_or(0.0)
    }
}

/// Splash `referenceDistribution`.
fn reference_distribution(row: &[f32], policy: Policy, admits: Admission) -> Distribution {
    let vocabulary = row.len();
    let mut d = Distribution {
        margin: 1.0,
        ..Distribution::default()
    };
    d.order = (0..vocabulary as u32)
        .filter(|&t| admits.admits(t))
        .collect();
    let beats = |a: u32, b: u32| {
        let (x, y) = (row[a as usize], row[b as usize]);
        x > y || (x == y && a < b)
    };
    d.order.sort_by(|&a, &b| match (beats(a, b), beats(b, a)) {
        (true, _) => std::cmp::Ordering::Less,
        (_, true) => std::cmp::Ordering::Greater,
        _ => std::cmp::Ordering::Equal,
    });
    d.maximum = row[d.order[0] as usize];
    let maximum = f64::from(d.maximum);
    let temperature = f64::from(policy.temperature);
    let weight = |t: u32| ((f64::from(row[t as usize]) - maximum) / temperature).exp();
    d.ambiguous = vec![false; vocabulary];
    if policy.min_p > 0.0 {
        let cut = maximum + temperature * f64::from(policy.min_p).ln();
        let tolerance = MIN_P_MARGIN * 1f64.max(cut.abs()).max(maximum.abs());
        let mut kept = 0;
        let mut rank = 0;
        while rank < d.order.len() && f64::from(row[d.order[rank] as usize]) >= cut - tolerance {
            let token = d.order[rank];
            let value = f64::from(row[token as usize]);
            if value >= cut {
                kept = rank + 1;
            }
            if (value - cut).abs() <= tolerance
                && row[token as usize] != d.maximum
                && (policy.top_k == 0 || rank < policy.top_k as usize)
            {
                d.ambiguous[token as usize] = true;
                d.ambiguous_mass += weight(token);
            }
            rank += 1;
        }
        d.order.truncate(kept);
    }
    if policy.top_k != 0 && (policy.top_k as usize) < d.order.len() {
        d.order.truncate(policy.top_k as usize);
    }
    let top_mass: f64 = d.order.iter().map(|&t| weight(t)).sum();
    if policy.top_p < 1.0 {
        let slack = TOP_P_MARGIN + d.ambiguous_mass / top_mass;
        let target = f64::from(policy.top_p) * top_mass;
        let mut kept = d.order.len();
        let mut before = 0.0;
        for rank in 0..d.order.len() {
            let token = d.order[rank];
            let distance = (before - target).abs() / top_mass;
            d.margin = d.margin.min(distance);
            if distance <= slack {
                d.ambiguous[token as usize] = true;
                d.ambiguous_mass += weight(token);
            }
            if before > target && kept == d.order.len() {
                kept = rank;
            }
            if before > target * (1.0 + slack) + slack * top_mass {
                break;
            }
            before += weight(token);
        }
        d.order.truncate(kept);
    }
    d.probabilities = vec![0.0; vocabulary];
    d.mass = d.order.iter().map(|&t| weight(t)).sum();
    for &t in &d.order {
        d.probabilities[t as usize] = weight(t) / d.mass;
    }
    d.ambiguous_mass /= d.mass;
    d
}

/// The row's best `count` admitted tokens: value descending, id ascending (Splash
/// `referenceBest`).
fn reference_best(row: &[f32], count: usize, admits: Admission) -> Vec<u32> {
    let d = reference_distribution(row, Policy::new(0, 1.0, 1.0), admits);
    d.order.into_iter().take(count).collect()
}

/// The draw a sampled row must make (Splash `requireDraw`): the token whose interval of the
/// weights' cumulative sum, in token order, holds `uniform` times their total, within the float
/// rounding of the GPU's sums and `slack` of the total.
fn require_draw(token: u32, weights: &[f64], uniform: f64, slack: f64, label: &str) {
    let total: f64 = weights.iter().sum();
    let target = uniform * total;
    let tolerance = (1e-5 + slack) * total;
    assert!(
        (token as usize) < weights.len() && (weights[token as usize] > 0.0 || slack > 0.0),
        "{label}: drew token {token}, which has no weight"
    );
    let before: f64 = weights[..token as usize].iter().sum();
    assert!(
        before - tolerance <= target && target <= before + weights[token as usize] + tolerance,
        "{label}: drew token {token} whose interval [{}, {}) misses {uniform}",
        before / total,
        (before + weights[token as usize]) / total
    );
}

/// A rejected draft token's residual (Splash `residualWeights`): each token's probability less its
/// draft probability (a token's first entry), never below zero, or the distribution itself when
/// nothing remains.
fn residual_weights(target: &Distribution, candidates: &[u32], proposal: &[f32]) -> Vec<f64> {
    let mut weights = target.probabilities.clone();
    for (index, &token) in candidates.iter().enumerate() {
        if token as usize >= weights.len() || candidates[..index].contains(&token) {
            continue;
        }
        let w = &mut weights[token as usize];
        *w = (*w - f64::from(proposal[index])).max(0.0);
    }
    match weights.iter().sum::<f64>() <= 0.0 {
        true => target.probabilities.clone(),
        false => weights,
    }
}

/// Splash's `fillRow`: low noise with forty distinct spikes, every stop token below them.
fn spiked_row(row: &mut [f32], random: &mut Random) {
    let vocabulary = row.len() as u32;
    for x in row.iter_mut() {
        *x = -8.0 + 3.0 * random.unit() * random.unit();
    }
    for spike in 0..40 {
        let token = 3 + random.next() % (vocabulary - 3);
        row[token as usize] = 1.0 + 0.25 * spike as f32 + 0.1 * random.unit();
    }
}

#[derive(Clone, Copy)]
enum Shape {
    Peaked,
    Flat,
    Graded,
    Ties,
}

/// Splash's `fillShaped`.
fn fill_shaped(row: &mut [f32], shape: Shape, random: &mut Random) {
    let vocabulary = row.len() as u32;
    match shape {
        Shape::Peaked => return spiked_row(row, random),
        Shape::Ties => {
            for x in row.iter_mut() {
                *x = match random.next().is_multiple_of(11) {
                    true => f32::NEG_INFINITY,
                    false => (random.next() % 7) as f32 - 3.0,
                };
            }
            return;
        }
        Shape::Flat | Shape::Graded => {}
    }
    for x in row.iter_mut() {
        *x = match shape {
            Shape::Flat => random.unit(),
            _ => -6.0 + 2.0 * random.unit(),
        };
    }
    if let Shape::Graded = shape {
        for rank in 0..300u32 {
            let token = 3 + (rank * 7919 + 11) % (vocabulary - 3);
            row[token as usize] = 4.0 - 0.02 * rank as f32 + 0.005 * random.unit();
        }
    }
}

/// One step's target policy over `lanes` lanes of `ROWS` verify rows (`dflash2_target_sampling`):
/// the logits (bf16), each lane's `VerifyLane`, and its drafts with their candidates.
struct Targets {
    vocabulary: usize,
    logits: Vec<u16>,
    lanes: Vec<VerifyLane>,
    drafted: Vec<u32>,
    draft_ids: Vec<u32>,
    draft_probs: Vec<f32>,
}

/// What a policy run left: each row's token, `TargetVocabularyRow` and arrival count.
struct Drawn {
    tokens: Vec<u32>,
    records: Vec<[u32; 4]>,
    arrivals: Vec<u32>,
}

impl Drawn {
    fn maximum(&self, index: usize) -> f32 {
        f32::from_bits(self.records[index][0])
    }
    fn draft_probability(&self, index: usize) -> f32 {
        f32::from_bits(self.records[index][3])
    }
}

impl Targets {
    /// `policies`' lanes, every one drafting its block's proposals, at positions of their own.
    fn of(vocabulary: usize, policies: &[Policy], stops: [u32; 2], seed: u64) -> Self {
        let lanes = policies.len();
        let p = P;
        Self {
            vocabulary,
            logits: vec![0; lanes * ROWS as usize * vocabulary],
            lanes: (policies.iter().enumerate())
                .map(|(lane, policy)| VerifyLane {
                    sampled: lane as u32 * ROWS,
                    drafts: p as u32,
                    position: 1000 * lane as u32 + 17,
                    remaining: ROWS,
                    stop: stops,
                    sampling: u32::from(policy.temperature > 0.0),
                    top_k: policy.top_k,
                    temperature: policy.temperature,
                    top_p: policy.top_p,
                    min_p: policy.min_p,
                    exclude_stop: u32::from(policy.excludes_stop),
                    seed: DraftSeed::of(seed.wrapping_mul(0x9e37_79b9).wrapping_add(lane as u64)),
                })
                .collect(),
            drafted: vec![0; lanes * p],
            draft_ids: vec![u32::MAX; lanes * p * CANDIDATES as usize],
            draft_probs: vec![0.0; lanes * p * CANDIDATES as usize],
        }
    }

    fn rows(&self) -> usize {
        self.lanes.len() * ROWS as usize
    }

    fn set_row(&mut self, index: usize, values: &[f32]) {
        let v = self.vocabulary;
        for (to, &x) in self.logits[index * v..(index + 1) * v]
            .iter_mut()
            .zip(values)
        {
            *to = bits(x);
        }
    }

    /// Row `index` as the kernels read it: its bf16 logits, in f32.
    fn row(&self, index: usize) -> Vec<f32> {
        let v = self.vocabulary;
        self.logits[index * v..(index + 1) * v]
            .iter()
            .map(|&b| value(b))
            .collect()
    }

    /// The uniform a lane's correction draws with: its step's draw `UNIFORM_CORRECTION`.
    fn correction_uniform(&self, lane: usize) -> f32 {
        let l = &self.lanes[lane];
        draft_uniform(l.seed, l.position, UNIFORM_CORRECTION)
    }

    fn acceptance_uniform(&self, lane: usize, position: usize) -> f32 {
        let l = &self.lanes[lane];
        draft_uniform(l.seed, l.position, UNIFORM_ACCEPTANCE + position as u32)
    }

    /// Splash's `setDraft`: `candidates`, made up to sixteen with random ids, their probabilities
    /// random and normalized, and `draft` the drafted token.
    fn set_draft(
        &mut self,
        lane: usize,
        position: usize,
        mut candidates: Vec<u32>,
        draft: u32,
        random: &mut Random,
    ) {
        let at = (lane * P + position) * CANDIDATES as usize;
        while candidates.len() < CANDIDATES as usize {
            candidates.push(3 + random.next() % (self.vocabulary as u32 - 3));
        }
        let probabilities: Vec<f32> = (0..CANDIDATES)
            .map(|_| 0.05 + 0.5 * (random.unit() + 1.0))
            .collect();
        let total: f32 = probabilities.iter().sum();
        for (index, (&id, &q)) in candidates.iter().zip(&probabilities).enumerate() {
            self.draft_ids[at + index] = id;
            self.draft_probs[at + index] = q / total;
        }
        self.drafted[lane * P + position] = draft;
    }

    /// A one-candidate draft of `token` with probability `q` (the others none).
    fn set_one_hot_draft(&mut self, lane: usize, position: usize, token: u32, q: f32) {
        let at = (lane * P + position) * CANDIDATES as usize;
        self.draft_ids[at..at + CANDIDATES as usize].fill(u32::MAX);
        self.draft_probs[at..at + CANDIDATES as usize].fill(0.0);
        self.draft_ids[at] = token;
        self.draft_probs[at] = q;
        self.drafted[lane * P + position] = token;
    }

    /// Mass, search and draw over every row, the outputs poisoned first.
    fn run(&self, gpu: &Gpu) -> Drawn {
        let rows = self.rows();
        let constants = || TargetSamplingConstants {
            vocab: scratchy_target_metal::tape::ids::LogitsWidth(self.vocabulary as u32),
            rows: ROWS,
        };
        let pso = |symbol| gpu.pipeline("dflash2_target_sampling", symbol, constants());
        let (mass, search, draw) = (
            pso("dflash2_sample_mass_bf16"),
            pso("dflash2_sample_search_bf16"),
            pso("dflash2_sample_draw_bf16"),
        );
        let b_logits = gpu.buffer(&self.logits);
        let b_lanes = gpu.buffer(&self.lanes);
        let b_masses = gpu.buffer(&vec![0xA5A5_A5A5u32; rows * TARGET_SHARDS * 3]);
        let b_records = gpu.buffer(&vec![0xA5A5_A5A5u32; rows * 4]);
        let b_drafted = gpu.buffer(&self.drafted);
        let b_ids = gpu.buffer(&self.draft_ids);
        let b_probs = gpu.buffer(&self.draft_probs);
        let b_tokens = gpu.buffer(&vec![0xA5A5_A5A5u32; rows]);
        let b_ranges = gpu.buffer(&vec![0xA5A5_A5A5u32; rows * VOCABULARY_RANGES * 2]);
        let b_arrivals = gpu.buffer(&vec![0u32; rows]);
        let mut batch = gpu.batch();
        batch.encode(
            &mass,
            &[(&b_logits, 0), (&b_lanes, 1), (&b_masses, 2)],
            &[],
            &[],
            &[],
            tg(rows * TARGET_SHARDS, 1, 1),
            tg(256, 1, 1),
        );
        batch.barrier();
        batch.encode(
            &search,
            &[
                (&b_logits, 0),
                (&b_lanes, 1),
                (&b_masses, 2),
                (&b_records, 3),
            ],
            &[],
            &[],
            &[],
            tg(rows, 1, 1),
            tg(VOCABULARY_THREADS, 1, 1),
        );
        batch.barrier();
        batch.encode(
            &draw,
            &[
                (&b_logits, 0),
                (&b_lanes, 1),
                (&b_records, 2),
                (&b_drafted, 3),
                (&b_ids, 4),
                (&b_probs, 5),
                (&b_tokens, 6),
                (&b_ranges, 7),
                (&b_arrivals, 8),
            ],
            &[],
            &[],
            &[],
            tg(rows * VOCABULARY_GROUPS, 1, 1),
            tg(VOCABULARY_THREADS, 1, 1),
        );
        batch.commit(true);
        Drawn {
            tokens: read(&b_tokens, rows),
            records: (read::<u32>(&b_records, rows * 4).chunks(4))
                .map(|r| [r[0], r[1], r[2], r[3]])
                .collect(),
            arrivals: read(&b_arrivals, rows),
        }
    }

    /// Splash's `requireSampledRow`: row `index`'s largest logit, its draft token's probability
    /// and the correction a rejection takes, or, for the last row, the bonus token.
    fn require_sampled_row(&self, drawn: &Drawn, index: usize, target: &Distribution, label: &str) {
        let (lane, row) = (index / ROWS as usize, index % ROWS as usize);
        assert_eq!(
            drawn.arrivals[index], 0,
            "{label}: the draw left its arrival count"
        );
        assert_eq!(
            drawn.maximum(index),
            target.maximum,
            "{label}: the scan's largest logit differs"
        );
        let uniform = f64::from(self.correction_uniform(lane));
        let token = drawn.tokens[index];
        if row == P {
            return require_draw(
                token,
                &target.probabilities,
                uniform,
                target.ambiguous_mass,
                &format!("{label} bonus"),
            );
        }
        let draft = self.drafted[lane * P + row];
        let expected = target.probability(draft);
        let got = f64::from(drawn.draft_probability(index));
        assert!(
            target.ambiguous[draft as usize]
                || match expected == 0.0 {
                    true => got == 0.0,
                    false => (got - expected).abs() <= (1e-4 + target.ambiguous_mass) * expected,
                },
            "{label}: draft token probability {got} for {expected}"
        );
        let at = (lane * P + row) * CANDIDATES as usize;
        let residual = residual_weights(
            target,
            &self.draft_ids[at..at + CANDIDATES as usize],
            &self.draft_probs[at..at + CANDIDATES as usize],
        );
        require_draw(
            token,
            &residual,
            uniform,
            target.ambiguous_mass,
            &format!("{label} correction"),
        );
    }
}

/// Splash's `shardEdgeStopTokens`: stop tokens in the vocabulary's first shard and its last.
fn shard_edge_stops(vocabulary: usize) -> [u32; 2] {
    [1, vocabulary as u32 - 2]
}

/// Splash's `sampledRows`: sampled lanes with the default top-k of 20, larger ones, a disabled one
/// and one past the vocabulary, masked or ignoring end-of-sequence, over every row shape, beside
/// a greedy lane, which the policy leaves; with `min_p`, cutting by min_p first: alone, before a
/// top-k and a nucleus, at 1, and so low it drops nothing.
fn sampled_rows(gpu: &Gpu, vocabulary: usize, lanes: usize, min_p: bool) {
    let mut random =
        Random(0x7769_6465 + vocabulary as u64 * 8 + lanes as u64 + u64::from(min_p) * 64);
    let narrow = Policy::new(20, 0.9, 0.95);
    let disabled = Policy::new(0, 1.0, 0.95);
    let fifty = Policy::new(50, 0.8, 0.9);
    let thousand = Policy::new(1000, 1.3, 1.0).excluding_stop();
    let wide = Policy::new(2000, 1.0, 0.95);
    let beyond = Policy::new(vocabulary as u32 + 5, 0.6, 0.7);
    let greedy = Policy::new(1, 0.0, 1.0);
    let floor = Policy::new(0, 1.0, 1.0).min_p(0.05);
    let floor_nucleus = Policy::new(50, 0.8, 0.9).min_p(0.2);
    let heaviest = Policy::new(0, 0.9, 1.0).min_p(1.0);
    let floorless = Policy::new(20, 0.9, 0.95).excluding_stop().min_p(1e-30);
    let batches: [Vec<Policy>; 4] = match min_p {
        true => [
            vec![floor],
            vec![floor_nucleus, greedy],
            vec![heaviest, floorless, greedy],
            vec![floor, floor_nucleus, greedy, heaviest],
        ],
        false => [
            vec![disabled],
            vec![fifty, greedy],
            vec![beyond, narrow, greedy],
            vec![wide, thousand, greedy, fifty],
        ],
    };
    let policies = &batches[lanes - 1];
    // Splash's top-k-50 lanes (`fifty`, `floorNucleus`) are constrained: their mask arrives as -inf
    // logits.
    let masked = |lane: usize| policies[lane].top_k == 50;
    let mut t = Targets::of(
        vocabulary,
        policies,
        STOPS,
        vocabulary as u64 + lanes as u64,
    );
    let mut references = vec![Distribution::default(); t.rows()];
    let mut masks = vec![Vec::new(); t.rows()];
    for lane in 0..lanes {
        for row in 0..ROWS as usize {
            let index = lane * ROWS as usize + row;
            let mut values = vec![0.0f32; vocabulary];
            let shape = [Shape::Peaked, Shape::Flat, Shape::Graded][row % 3];
            fill_shaped(&mut values, shape, &mut random);
            if row % 2 == 1 {
                values[STOPS[0] as usize] = 13.0;
            }
            if masked(lane) {
                let mask: Vec<bool> = (0..vocabulary).map(|t| (t * 7 + row) % 5 != 0).collect();
                for (x, &keep) in values.iter_mut().zip(&mask) {
                    if !keep {
                        *x = f32::NEG_INFINITY;
                    }
                }
                masks[index] = mask;
            }
            t.set_row(index, &values);
            if policies[lane].temperature > 0.0 {
                let admits = Admission {
                    mask: masked(lane).then_some(masks[index].as_slice()),
                    excludes_stop: policies[lane].excludes_stop,
                    stops: STOPS,
                };
                references[index] = reference_distribution(&t.row(index), policies[lane], admits);
            }
        }
        // Draft tokens from the start and the middle of the distribution, and a random one.
        for position in 0..P {
            let kept = &references[lane * ROWS as usize + position].order;
            let mut candidates = Vec::new();
            if !kept.is_empty() {
                candidates = vec![kept[0], kept[kept.len() / 2]];
            }
            candidates.push(3 + random.next() % (vocabulary as u32 - 3));
            let draft = candidates[match candidates.len() == 3 {
                true => position % 3,
                false => 0,
            }];
            t.set_draft(lane, position, candidates, draft, &mut random);
        }
    }
    let drawn = t.run(gpu);
    let label = format!(
        "{} rows B{lanes} vocabulary {vocabulary}",
        ["sampled", "min_p"][usize::from(min_p)]
    );
    for (index, reference) in references.iter().enumerate() {
        let lane = index / ROWS as usize;
        let label = format!("{label} row {index}");
        match policies[lane].temperature > 0.0 {
            true => t.require_sampled_row(&drawn, index, reference, &label),
            false => assert_eq!(
                drawn.tokens[index], 0xA5A5_A5A5,
                "{label}: a greedy lane's row was drawn"
            ),
        }
    }
}

#[test]
fn target_policy_draws_sampled_rows_as_splash_does() {
    let Some(gpu) = Gpu::new() else { return };
    for vocabulary in [1003, 248_320] {
        for lanes in 1..=4 {
            sampled_rows(&gpu, vocabulary, lanes, false);
            sampled_rows(&gpu, vocabulary, lanes, true);
        }
    }
}

/// Splash's `mixedVerify`: a lane's rows select the same, bit for bit, whatever lanes share its
/// batch, under every mix of sampling lanes (a greedy lane leaves its rows).
#[test]
fn target_policy_selects_a_lane_alone_as_in_a_batch() {
    let Some(gpu) = Gpu::new() else { return };
    let vocabulary = 1003;
    let stops = shard_edge_stops(vocabulary);
    for lanes in 1..=4usize {
        for mask in 0..(1u32 << lanes) {
            let mut random = Random(9831 + lanes as u64);
            let policies: Vec<Policy> = (0..lanes)
                .map(|lane| {
                    let temperature = if mask >> lane & 1 == 1 { 0.7 } else { 0.0 };
                    Policy::new(8 + lane as u32, temperature, 0.9)
                })
                .collect();
            let mut t = Targets::of(vocabulary, &policies, stops, 77);
            for lane in 0..lanes {
                for row in 0..ROWS as usize {
                    let mut values = vec![0.0f32; vocabulary];
                    let shape = if row % 2 == 1 {
                        Shape::Ties
                    } else {
                        Shape::Peaked
                    };
                    fill_shaped(&mut values, shape, &mut random);
                    t.set_row(lane * ROWS as usize + row, &values);
                }
            }
            for entry in t.drafted.iter_mut() {
                *entry = random.next() % vocabulary as u32;
            }
            for (id, q) in t.draft_ids.iter_mut().zip(t.draft_probs.iter_mut()) {
                *id = random.next() % vocabulary as u32;
                *q = 0.5 * (random.unit() + 1.0) / CANDIDATES as f32;
            }
            let together = t.run(&gpu);
            for lane in 0..lanes {
                let mut alone = Targets::of(vocabulary, &policies[lane..=lane], stops, 0);
                alone.lanes[0] = VerifyLane {
                    sampled: 0,
                    ..t.lanes[lane]
                };
                let (r, v) = (ROWS as usize, vocabulary);
                alone.logits = t.logits[lane * r * v..(lane + 1) * r * v].to_vec();
                alone.drafted = t.drafted[lane * P..(lane + 1) * P].to_vec();
                let c = P * CANDIDATES as usize;
                alone.draft_ids = t.draft_ids[lane * c..(lane + 1) * c].to_vec();
                alone.draft_probs = t.draft_probs[lane * c..(lane + 1) * c].to_vec();
                let single = alone.run(&gpu);
                let rows = lane * r..(lane + 1) * r;
                assert_eq!(
                    together.tokens[rows.clone()],
                    single.tokens[..],
                    "B{lanes} mask {mask:b}: lane {lane}'s tokens"
                );
                if policies[lane].temperature > 0.0 {
                    assert_eq!(
                        together.records[rows],
                        single.records[..],
                        "B{lanes} mask {mask:b}: lane {lane}'s rows"
                    );
                }
            }
        }
    }
}

/// Splash's `ties`: top-k and top-p cuts inside thousands of tied logits keep exactly the ties the
/// reference keeps, ids ascending, over the 248,320-token vocabulary.
#[test]
fn target_policy_cuts_within_ties_as_splash_does() {
    let Some(gpu) = Gpu::new() else { return };
    let vocabulary = 248_320;
    for (name, tied, above, policy) in [
        (
            "top_k within 3000 ties",
            3000u32,
            10u32,
            Policy::new(1510, 1.0, 1.0),
        ),
        ("top_k within 600 ties", 600, 10, Policy::new(310, 0.7, 1.0)),
        (
            "top_k then top_p within 3000 ties",
            3000,
            10,
            Policy::new(1510, 2.0, 0.9),
        ),
        (
            "top_p within 3000 ties",
            3000,
            0,
            Policy::new(0, 1.0, 0.4999),
        ),
    ] {
        let mut random = Random(0x7469_6573 + u64::from(tied) + u64::from(policy.top_k));
        let tied: Vec<u32> = (0..tied).map(|i| 40 + i * 79).collect();
        let mut t = Targets::of(vocabulary, &[policy], STOPS, u64::from(policy.top_k));
        for row in 0..ROWS as usize {
            let mut values: Vec<f32> = (0..vocabulary).map(|_| -12.0 + random.unit()).collect();
            for &token in &tied {
                values[token as usize] = 2.0;
            }
            for index in 0..above {
                values[(5 + index * 13) as usize] = 5.0 + index as f32;
            }
            t.set_row(row, &values);
        }
        let target = reference_distribution(&t.row(0), policy, Admission::default());
        assert!(
            target.margin > TOP_P_MARGIN,
            "{name}: test data is ambiguous"
        );
        let kept = target.order.len() - above as usize;
        assert!(
            kept > 0 && kept < tied.len(),
            "{name}: the cut is not within the ties"
        );
        for position in 0..P {
            let draft = if position % 2 == 1 {
                tied[kept]
            } else {
                tied[kept - 1]
            };
            t.set_draft(0, position, vec![draft], draft, &mut random);
        }
        let drawn = t.run(&gpu);
        for row in 0..ROWS as usize {
            t.require_sampled_row(&drawn, row, &target, &format!("{name} row {row}"));
        }
    }
}

/// Splash's `minPCuts`: on rows whose weights are known — the heaviest 1, ten at 0.3, twenty at
/// 0.1, a hundred at 0.01 — min_p keeps the classes that weigh at least it, each tie whole; top_k
/// then counts and top_p measures within what it kept; the cut follows the temperature. A draft
/// token of the last class kept and one of the first dropped test it from both sides.
#[test]
fn target_policy_cuts_by_min_p_as_splash_does() {
    let Some(gpu) = Gpu::new() else { return };
    let vocabulary = 4096;
    let classes: [(u32, u32, f32); 4] = [
        (5, 1, 1.0),
        (100, 10, 0.3),
        (200, 20, 0.1),
        (300, 100, 0.01),
    ];
    let mut random = Random(0x6d69_6e70);
    let mut values = vec![-1000.0f32; vocabulary];
    for (first, count, weight) in classes {
        for index in 0..count {
            values[(first + index) as usize] = weight.ln();
        }
    }
    for (name, policy, kept, last, dropped) in [
        (
            "min_p 0.2",
            Policy::new(0, 1.0, 1.0).min_p(0.2),
            11,
            109u32,
            200u32,
        ),
        (
            "min_p 0.05",
            Policy::new(0, 1.0, 1.0).min_p(0.05),
            31,
            219,
            300,
        ),
        (
            "min_p 0.31 drops a whole tie",
            Policy::new(0, 1.0, 1.0).min_p(0.31),
            1,
            5,
            109,
        ),
        (
            "min_p 0.05 at temperature 0.5",
            Policy::new(0, 0.5, 1.0).min_p(0.05),
            11,
            109,
            200,
        ),
        (
            "min_p 0.2 then top_k 5",
            Policy::new(5, 1.0, 1.0).min_p(0.2),
            5,
            103,
            104,
        ),
        (
            "min_p 0.05 then top_p 0.5 of what it kept",
            Policy::new(0, 1.0, 0.5).min_p(0.05),
            8,
            106,
            107,
        ),
    ] {
        let mut t = Targets::of(vocabulary, &[policy], STOPS, kept as u64);
        for row in 0..ROWS as usize {
            t.set_row(row, &values);
        }
        let target = reference_distribution(&t.row(0), policy, Admission::default());
        assert!(
            target.order.len() == kept
                && target.ambiguous_mass == 0.0
                && target.probability(last) > 0.0
                && target.probability(dropped) == 0.0,
            "{name}: the reference keeps {} tokens",
            target.order.len()
        );
        for position in 0..P {
            let draft = if position % 2 == 1 { dropped } else { last };
            t.set_draft(0, position, vec![draft], draft, &mut random);
        }
        let drawn = t.run(&gpu);
        for row in 0..ROWS as usize {
            t.require_sampled_row(&drawn, row, &target, &format!("{name} row {row}"));
        }
    }
}

/// Splash's `targetTop1`: a sampled lane whose distribution keeps one token draws its masked
/// argmax at every verify row, and its draft token's probability is one-hot.
#[test]
fn target_policy_draws_a_one_token_distribution_as_its_argmax() {
    let Some(gpu) = Gpu::new() else { return };
    for vocabulary in [1003, 248_320] {
        let stops = shard_edge_stops(vocabulary);
        for lanes in 1..=4usize {
            let policies: Vec<Policy> = (0..lanes).map(|_| Policy::new(1, 0.8, 0.5)).collect();
            let mut t = Targets::of(vocabulary, &policies, stops, lanes as u64);
            let best: Vec<u32> = (0..t.rows())
                .map(|row| {
                    let values: Vec<f32> = (0..vocabulary)
                        .map(|token| match (token + row).is_multiple_of(17) {
                            true => {
                                ((token * 7 + row * 13) % 23) as f32 - 11.0
                                    + (token % 3) as f32 * 2f32.powi(-12)
                            }
                            false => f32::NEG_INFINITY,
                        })
                        .collect();
                    t.set_row(row, &values);
                    reference_best(&t.row(row), 1, Admission::default())[0]
                })
                .collect();
            for (row, &argmax) in best.iter().enumerate() {
                let (lane, r) = (row / ROWS as usize, row % ROWS as usize);
                if r < P {
                    let draft = if r % 2 == 1 {
                        argmax
                    } else {
                        (argmax + 1) % vocabulary as u32
                    };
                    t.set_one_hot_draft(lane, r, draft, 1.0);
                }
            }
            let drawn = t.run(&gpu);
            for (row, &argmax) in best.iter().enumerate() {
                assert_eq!(
                    drawn.tokens[row], argmax,
                    "B{lanes} row {row}: a sampled top-1 row drew other than its argmax"
                );
                let (lane, r) = (row / ROWS as usize, row % ROWS as usize);
                if r < P {
                    let one_hot = if t.drafted[lane * P + r] == argmax {
                        1.0
                    } else {
                        0.0
                    };
                    assert_eq!(
                        drawn.draft_probability(row),
                        one_hot,
                        "B{lanes} row {row}: not one-hot"
                    );
                }
            }
        }
    }
}

/// Splash's `nonFiniteRowsSelectTheSentinel`: a row with no selectable logit, every one NaN or
/// -inf, draws 0xFFFFFFFF at every verify row, and acceptance retains it as the next anchor.
#[test]
fn target_policy_draws_the_sentinel_from_a_row_with_no_logit() {
    let Some(gpu) = Gpu::new() else { return };
    let vocabulary = 1003;
    let policy = Policy::new(20, 1.0, 0.9);
    for logit in [f32::NAN, f32::NEG_INFINITY] {
        let mut random = Random(0x6e_616e);
        let mut t = Targets::of(vocabulary, &[policy], STOPS, 3);
        for row in 0..ROWS as usize {
            t.set_row(row, &vec![logit; vocabulary]);
        }
        for position in 0..P {
            let draft = 3 + random.next() % (vocabulary as u32 - 3);
            t.set_draft(0, position, vec![draft], draft, &mut random);
        }
        let drawn = t.run(&gpu);
        for row in 0..ROWS as usize {
            assert_eq!(
                drawn.tokens[row], SENTINEL,
                "{logit}: verify row {row} with no selectable logit selected {}",
                drawn.tokens[row]
            );
        }
        let accepted = t.accept(&gpu, &drawn);
        let kept = accepted.retained[0];
        assert!(
            (1..=ROWS).contains(&kept) && accepted.tokens[kept as usize - 1] == SENTINEL,
            "{logit}: acceptance did not retain the sentinel"
        );
    }
}

impl Targets {
    /// The acceptance after the policy (`dflash2_accept`): each lane's drafts against what the
    /// policy drew, its records' draft token probabilities.
    fn accept(&self, gpu: &Gpu, drawn: &Drawn) -> Accepted {
        Acceptance {
            lanes: self.lanes.clone(),
            drafted: self.drafted.clone(),
            tokens: drawn.tokens.clone(),
            draft_ids: self.draft_ids.clone(),
            draft_probs: self.draft_probs.clone(),
            target_rows: drawn.records.clone(),
        }
        .run(gpu)
    }

    /// Splash's acceptance uniform pinned through the draft probability instead: a one-hot draft of
    /// `token`, which the lane's acceptance test (u · q < p) keeps when `accept`, its probability
    /// half of what the target gives it over the lane's uniform; else at least the target's,
    /// rejected, and wholly out of the residual.
    fn set_tested_draft(&mut self, lane: usize, position: usize, token: u32, p: f64, accept: bool) {
        let u = f64::from(self.acceptance_uniform(lane, position)).max(1e-6);
        let q = match accept {
            true => 0.5 * p / u,
            false => (2.0 * p + 0.01).min(1.0) / u,
        };
        self.set_one_hot_draft(lane, position, token, q as f32);
    }
}

/// The correction a rejected row draws, or the bonus token (Splash's expectation in
/// `speculativeExactness`, `speculativeWholeVocabulary`): the target less the one-hot draft
/// token, at the lane's correction uniform, the data refused as ambiguous within `slack`.
fn expected_correction(
    target: &Distribution,
    draft: Option<u32>,
    uniform: f64,
    label: &str,
) -> u32 {
    let mut weights = target.probabilities.clone();
    if let Some(d) = draft {
        weights[d as usize] = 0.0;
    }
    let total: f64 = weights.iter().sum();
    let threshold = uniform * total;
    let slack = (1e-5 + target.ambiguous_mass) * total;
    let (mut cumulative, mut token) = (0.0, 0usize);
    while cumulative + weights[token] <= threshold {
        cumulative += weights[token];
        token += 1;
    }
    assert!(
        threshold - cumulative > slack
            && cumulative + weights[token] - threshold > slack
            && !target.ambiguous[token],
        "{label}: test data is ambiguous at the correction draw"
    );
    token as u32
}

/// Splash's `speculativeExactness` (its penalties wait for their port), over every sampling mask
/// of its two lanes as Splash's main loop runs it (target_sampling_metal_test.mm:2636):
/// acceptance over the policy's rows must accept and correct as a sequential decode would. Lane 0
/// drafts its first wrong token at position 3, lane 1 every token right, taking the bonus row. A
/// greedy lane's rows are scratchy's argmax, which writes every row's token before the policy
/// overwrites the sampled lanes': here each row's reference argmax.
#[test]
fn target_policy_and_acceptance_decode_as_a_sequential_decode_does() {
    let Some(gpu) = Gpu::new() else { return };
    for mask in 0..4u32 {
        exactness_over(&gpu, mask);
    }
}

fn exactness_over(gpu: &Gpu, mask: u32) {
    let vocabulary = 1003;
    let wrong_at = [3usize, P];
    let sampled = |lane: usize| mask >> lane & 1 == 1;
    let policies: Vec<Policy> = (0..2)
        .map(|lane| match sampled(lane) {
            true => Policy::new(32, 0.9, 0.9),
            false => Policy::new(1, 0.0, 0.9),
        })
        .collect();
    let mut random = Random(0x7370_6563 + u64::from(mask));
    let mut t = Targets::of(vocabulary, &policies, STOPS, 11 + u64::from(mask));
    let mut expected = [[0u32; ROWS as usize]; 2];
    let mut argmax = vec![0u32; t.rows()];
    for (lane, expected) in expected.iter_mut().enumerate() {
        let mut base = vec![0.0f32; vocabulary];
        spiked_row(&mut base, &mut random);
        let mut targets = Vec::new();
        for row in 0..ROWS as usize {
            let index = lane * ROWS as usize + row;
            let values: Vec<f32> = base.iter().map(|&b| b + 0.01 * random.unit()).collect();
            t.set_row(index, &values);
            let logits = t.row(index);
            let order = reference_best(&logits, 8, Admission::default());
            assert!(
                logits[order[0] as usize] - logits[order[1] as usize] > 1e-4,
                "test data is ambiguous at a greedy selection"
            );
            argmax[index] = order[0];
            targets.push(match sampled(lane) {
                true => reference_distribution(&logits, policies[lane], Admission::default()),
                false => Distribution::default(),
            });
            if row == P {
                break;
            }
            // The best token until the wrong position, then the eighth best.
            let right = row < wrong_at[lane];
            let draft = if right { order[0] } else { order[7] };
            match sampled(lane) {
                true => {
                    let p = targets[row].probability(draft);
                    let label = format!("mask {mask:02b}");
                    assert!(
                        !right || p > 1e-4,
                        "{label}: a sampled lane drafted a token its target cannot accept"
                    );
                    t.set_tested_draft(lane, row, draft, p, right);
                }
                false => t.set_one_hot_draft(lane, row, draft, 1.0),
            }
        }
        let accepted = wrong_at[lane];
        expected[..accepted].copy_from_slice(&t.drafted[lane * P..lane * P + accepted]);
        expected[accepted] = match sampled(lane) {
            true => {
                let draft = (accepted < P).then(|| t.drafted[lane * P + accepted]);
                let uniform = f64::from(t.correction_uniform(lane));
                let label = format!("mask {mask:02b} lane {lane}");
                expected_correction(&targets[accepted], draft, uniform, &label)
            }
            false => argmax[lane * ROWS as usize + accepted],
        };
    }
    let mut drawn = t.run(gpu);
    for (index, &token) in argmax.iter().enumerate() {
        if !sampled(index / ROWS as usize) {
            drawn.tokens[index] = token;
        }
    }
    let got = t.accept(gpu, &drawn);
    for (lane, expected) in expected.iter().enumerate() {
        let accepted = wrong_at[lane];
        let label = format!("mask {mask:02b} lane {lane}");
        assert_eq!(
            (got.accepted[lane], got.retained[lane]),
            (accepted as u32, accepted as u32 + 1),
            "{label} accepted"
        );
        let first = lane * ROWS as usize;
        assert_eq!(
            got.tokens[first..=first + accepted],
            expected[..=accepted],
            "{label}: output differs from a sequential decode"
        );
    }
}

/// Splash's `speculativeWholeVocabulary`: acceptance over broad distributions against a
/// sequential decode, beside a lane with the default top-k of 20 and one that min_p cuts.
#[test]
fn target_policy_and_acceptance_decode_broad_rows_as_a_sequential_decode_does() {
    let Some(gpu) = Gpu::new() else { return };
    let vocabulary = 5003;
    let wrong_at = [2usize, P, 4, 6];
    let policies = [
        Policy::new(0, 1.0, 1.0),
        Policy::new(2000, 0.8, 0.97),
        Policy::new(20, 0.9, 0.9),
        Policy::new(0, 1.0, 1.0).min_p(0.02),
    ];
    let mut random = Random(0x7768_6f6c);
    let mut t = Targets::of(vocabulary, &policies, STOPS, 23);
    let mut expected = [[0u32; ROWS as usize]; 4];
    for (lane, expected) in expected.iter_mut().enumerate() {
        for (row, expected_row) in expected.iter_mut().enumerate() {
            let mut values = vec![0.0f32; vocabulary];
            fill_shaped(
                &mut values,
                if row % 2 == 1 {
                    Shape::Flat
                } else {
                    Shape::Graded
                },
                &mut random,
            );
            t.set_row(lane * ROWS as usize + row, &values);
            let target = reference_distribution(
                &t.row(lane * ROWS as usize + row),
                policies[lane],
                Admission::default(),
            );
            if row < P {
                let right = row < wrong_at[lane];
                let draft = match right {
                    true => target.order[0],
                    false => target.order[5.min(target.order.len() - 1)],
                };
                let p = target.probability(draft);
                assert!(!right || p > 1e-6, "a draft token its target cannot accept");
                t.set_tested_draft(lane, row, draft, p, right);
                if right {
                    *expected_row = draft;
                }
            }
            if row == wrong_at[lane] {
                let draft = (row < P).then(|| t.drafted[lane * P + row]);
                let uniform = f64::from(t.correction_uniform(lane));
                *expected_row =
                    expected_correction(&target, draft, uniform, &format!("lane {lane}"));
            }
        }
    }
    let drawn = t.run(&gpu);
    let got = t.accept(&gpu, &drawn);
    for lane in 0..4 {
        assert_eq!(
            got.accepted[lane], wrong_at[lane] as u32,
            "acceptance lane {lane}"
        );
        let first = lane * ROWS as usize;
        assert_eq!(
            got.tokens[first..=first + wrong_at[lane]],
            expected[lane][..=wrong_at[lane]],
            "acceptance lane {lane}: output differs from a sequential decode"
        );
    }
}

/// A position whose correction draw is `want` units of 2^-24 for `seed`, and whose first
/// acceptance draw is at least `reject` (`overProposedResidual` pins both uniforms).
fn position_drawing(seed: DraftSeed, want: u32, reject: f32) -> u32 {
    (0..u32::MAX)
        .find(|&p| {
            draft_uniform(seed, p, UNIFORM_CORRECTION) * (1u64 << 24) as f32 == want as f32
                && draft_uniform(seed, p, UNIFORM_ACCEPTANCE) >= reject
        })
        .expect("a position drawing the uniform")
}

/// Splash's `overProposedResidual`: a rejected draft token's correction when the only weighted
/// tokens of one simdgroup's share of the vocabulary are draft candidates proposed at least as
/// likely as the target keeps them — nothing of that share remains in the residual, and the
/// correction comes from the others at every correction uniform down to 0. Acceptance takes the
/// correction as the next anchor.
#[test]
fn target_policy_corrects_from_an_over_proposed_residual() {
    let Some(gpu) = Gpu::new() else { return };
    let vocabulary = 4096;
    let policy = Policy::new(0, 1.0, 1.0);
    let mut values = vec![-1000.0f32; vocabulary];
    values[5] = 0.0;
    values[6] = (0.75 * 2f32.powi(-23)).ln();
    values[1000] = 0.5f32.ln();
    for index in 0..40 {
        values[1200 + 50 * index] = 0.01f32.ln() - 1e-4 * index as f32;
    }
    for want in [0u32, 1, 1 << 23, (1 << 24) - 1] {
        let label = format!("over-proposed residual, correction uniform {want}/2^24");
        let mut t = Targets::of(vocabulary, &[policy], STOPS, 31);
        for row in 0..ROWS as usize {
            t.set_row(row, &values);
        }
        let target = reference_distribution(&t.row(0), policy, Admission::default());
        // The first draft token is rejected: its acceptance draw times 0.9 exceeds its probability.
        let reject = (target.probability(5) / 0.9) as f32 + 1e-3;
        t.lanes[0].position = position_drawing(t.lanes[0].seed, want, reject);
        // Every position drafts token 5 with 0.9 and token 6 with 0.1, beside fourteen weightless.
        for position in 0..P {
            let at = position * CANDIDATES as usize;
            for index in 0..CANDIDATES as usize {
                t.draft_ids[at + index] = if index < 2 {
                    5 + index as u32
                } else {
                    6 + index as u32
                };
                t.draft_probs[at + index] = match index {
                    0 => 0.9,
                    1 => 0.1,
                    _ => 0.0,
                };
            }
            t.drafted[position] = 5;
        }
        let drawn = t.run(&gpu);
        for row in 0..ROWS as usize {
            t.require_sampled_row(&drawn, row, &target, &format!("{label} row {row}"));
        }
        let got = t.accept(&gpu, &drawn);
        assert_eq!(
            (got.accepted[0], got.retained[0]),
            (0, 1),
            "{label}: acceptance did not take the correction as its anchor"
        );
    }
}

/// Splash's `excludedStopTokens`: a lane that ignores end-of-sequence never draws a stop token at
/// a verify row, alone or beside lanes that keep them, and a stop token its draft proposes is
/// rejected; a lane that keeps them accepts it. The stop tokens lead every row.
#[test]
fn target_policy_never_draws_an_excluded_stop_token() {
    let Some(gpu) = Gpu::new() else { return };
    for vocabulary in [1003, 248_320] {
        let stops = shard_edge_stops(vocabulary);
        let best = |row: usize| 5 + (row * 7919) % (vocabulary - 8);
        let stop = |token: u32| stops.contains(&token);
        for lanes in 1..=4usize {
            for exclude in 0..(1u32 << lanes) {
                let policies: Vec<Policy> = (0..lanes)
                    .map(|lane| {
                        let p = Policy::new(32, 0.8, 1.0);
                        if exclude >> lane & 1 == 1 {
                            p.excluding_stop()
                        } else {
                            p
                        }
                    })
                    .collect();
                let mut t = Targets::of(vocabulary, &policies, stops, u64::from(exclude));
                for row in 0..t.rows() {
                    let mut values: Vec<f32> = (0..vocabulary)
                        .map(|token| -4.0 + (token % 7) as f32 * 0.01)
                        .collect();
                    values[best(row)] = 8.0;
                    values[stops[0] as usize] = 12.0;
                    values[stops[1] as usize] = 11.0;
                    t.set_row(row, &values);
                }
                // Drafts propose the first stop token, with all of their mass on it; a lane that keeps
                // the stop tokens accepts it (Splash's uniforms of 0.5, pinned here through q).
                for (lane, &policy) in policies.iter().enumerate() {
                    let admits = Admission {
                        mask: None,
                        excludes_stop: policy.excludes_stop,
                        stops,
                    };
                    let row = t.row(lane * ROWS as usize);
                    let p = reference_distribution(&row, policy, admits).probability(stops[0]);
                    t.set_tested_draft(lane, 0, stops[0], p, !policy.excludes_stop);
                }
                let drawn = t.run(&gpu);
                for row in 0..t.rows() {
                    let policy = policies[row / ROWS as usize];
                    let label =
                        format!("vocabulary {vocabulary} B{lanes} exclude {exclude:b} row {row}");
                    assert!(
                        row % ROWS as usize != 0
                            || (drawn.draft_probability(row) > 0.0) != policy.excludes_stop,
                        "{label}: a verify distribution mishandled the stop tokens"
                    );
                    assert!(
                        !policy.excludes_stop
                            || ((drawn.tokens[row] as usize) < vocabulary
                                && !stop(drawn.tokens[row])),
                        "{label}: an excluding verify row drew a stop token"
                    );
                }
                let got = t.accept(&gpu, &drawn);
                for (lane, policy) in policies.iter().enumerate() {
                    let output = &got.tokens[lane * ROWS as usize..];
                    let kept = got.retained[lane] as usize;
                    match policy.excludes_stop {
                        false => assert!(
                            kept == 1 && output[0] == stops[0],
                            "lane {lane}: the target did not accept the drafted stop token"
                        ),
                        true => {
                            assert!(
                                (1..=ROWS as usize).contains(&kept),
                                "lane {lane}: an excluding lane retained no token"
                            );
                            assert!(
                                output[..kept]
                                    .iter()
                                    .all(|&t| (t as usize) < vocabulary && !stop(t)),
                                "lane {lane}: an excluding lane accepted a stop token"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Splash's 4-bit context window (prefill/draft_context.metal `prefill_draft_context_store`,
/// `prefill_draft_context_load`): a group of 64 values keeps its fp16 minimum and its range over
/// 15 as its scale, and each value the nearest code, so a load reads back `minimum + code · scale`.
/// The values each group's store and load make, from the group's values.
fn window_round_trip(values: &[f32]) -> Vec<u16> {
    let low = values.iter().copied().fold(f32::INFINITY, f32::min);
    let high = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let scale = half::f16::from_f32((high - low) / 15.0);
    let minimum = half::f16::from_f32(low).to_f32();
    let code = |x: f32| match scale > half::f16::ZERO {
        true => ((x - minimum) / scale.to_f32())
            .round_ties_even()
            .clamp(0.0, 15.0),
        false => 0.0,
    };
    (values.iter())
        .map(|&x| bits(code(x).mul_add(scale.to_f32(), minimum)))
        .collect()
}

/// A prompt's commit stores each row's context into its sequence's window at its position, and a
/// rebuild loads back exactly what Splash's window reads: for two sequences in ring slots 2 and 0,
/// one at positions that wrap past the window's end, every value is the round trip of its group.
#[test]
fn a_context_window_reads_back_what_splash_stores() {
    let Some(gpu) = Gpu::new() else { return };
    let mut random = Random(0x51ab);
    let (h, group) = (HIDDEN as usize, 64usize);
    let cu: [u32; 3] = [0, 3, 7];
    let positions: Vec<u32> = (0..3)
        .map(|r| WINDOW * 2 + 2046 + r)
        .chain(10..14)
        .collect();
    let slots: [u32; 2] = [2, 0];
    let rows = positions.len();
    let context = random.bf16s(rows * h, 1.5);
    let window_bytes = (WINDOW as usize) * (h / 2) + (WINDOW as usize) * (h / group) * 4;
    let windows = gpu.buffer(&vec![0u8; 3 * window_bytes]);
    let (b_positions, b_slots) = (gpu.buffer(&positions), gpu.buffer(&slots));
    let (b_cu, b_n) = (gpu.buffer(&cu), gpu.buffer(&[rows as u32]));
    let b_context = gpu.buffer(&context);
    let loaded = gpu.buffer(&vec![0u16; rows * h]);
    let threadgroups = (rows * (h / group)).div_ceil(8);
    for (symbol, rows_buffer) in [
        ("prefill_draft_window_store_bf16", &b_context),
        ("draft_window_load_bf16", &loaded),
    ] {
        let pso = gpu.pipeline("dflash2", symbol, draft_constants(true));
        let mut batch = gpu.batch();
        batch.encode(
            &pso,
            &[
                (rows_buffer, 0),
                (&windows, 1),
                (&b_positions, 2),
                (&b_slots, 3),
                (&b_cu, 4),
                (&b_n, 5),
            ],
            &[],
            &[],
            &[],
            tg(threadgroups, 1, 1),
            tg(256, 1, 1),
        );
        batch.commit(true);
    }
    let got: Vec<u16> = read(&loaded, rows * h);
    for (row, position) in positions.iter().enumerate() {
        for g in 0..h / group {
            let at = row * h + g * group;
            let values: Vec<f32> = context[at..at + group].iter().map(|&b| value(b)).collect();
            assert_eq!(
                got[at..at + group],
                window_round_trip(&values)[..],
                "row {row} group {g} at position {position}"
            );
        }
    }
}

// ── The penalties (Splash target_sampling_metal_test.mm `penalties`, `requirePenalizedRows`) ────
//
// Splash's checks of its penalty kernel against the host's arithmetic, on scratchy's bf16 logits:
// each penalized sequence's rows rewritten from its own words row, counting at verify row r the
// drafts of input rows 1..r, rounded to bf16 as the kernel writes them; every other logit
// bitwise unchanged.

use scratchy_target_metal::dflash2::PenalizeEntry;

/// bf16's largest finite value, where the kernel saturates.
fn bf16_max() -> f32 {
    f32::from_bits(0x7f7f_0000)
}

/// Splash `referencePenalty`, on the target's sampler's counting: the logit of a token the prompt
/// and the output hold `count` times between them, as the kernel computes it before rounding.
fn reference_penalty(value: f32, count: u32, entry: &PenalizeEntry) -> f32 {
    if count == 0 {
        return value;
    }
    let value = value
        * match value > 0.0 {
            true => entry.repetition_inverse,
            false => entry.repetition,
        };
    (value - (entry.frequency * count as f32 + entry.presence)).clamp(-bf16_max(), bf16_max())
}

/// Splash `penaltyTable`: `rows` rows of words, each with small counts at a density of its own,
/// and a few counts at the 2^20-token limit.
fn penalty_table(vocabulary: usize, rows: usize) -> Vec<u32> {
    let mut table = vec![0u32; rows * vocabulary];
    for slot in 0..rows {
        let mut random = Random(0x0074_6162_6c65 + slot as u64);
        for token in 0..vocabulary {
            let draw = random.next() % 64;
            let mut word = u32::from(draw < 8 + 6 * slot as u32);
            if draw.is_multiple_of(3) {
                word += 1 + draw % (slot as u32 + 3);
            }
            if draw == 63 && token.is_multiple_of(7) {
                word = 1 << 20;
            }
            table[slot * vocabulary + token] = word;
        }
    }
    table
}

/// Two penalized sequences of one step's 16 logits rows — a verify block of 8 rows from row 0 on
/// words row 3 (every penalty), and a single row at row 11 on words row 1 (presence alone) —
/// rewrite exactly their own rows: each logit as Splash's host arithmetic computes it from its
/// words and its row's drafts, rounded to bf16, and every other row bitwise as it was. The drafts
/// repeat the block's best tokens, so the counts they add move its argmax.
#[test]
fn penalized_rows_are_rewritten_from_their_words_and_drafts() {
    let Some(gpu) = Gpu::new() else { return };
    let vocabulary = 1003usize;
    let mut random = Random(0x0070_656e);
    let rows = 16usize;
    let logits = random.bf16s(rows * vocabulary, 8.0);
    let table = penalty_table(vocabulary, 4);
    // The block's best tokens, by its first row.
    let mut order: Vec<usize> = (0..vocabulary).collect();
    order.sort_by(|&a, &b| value(logits[b]).total_cmp(&value(logits[a])));
    let best = |i: usize| order[i] as u32;
    let block = [
        best(9),
        best(0),
        best(1),
        best(0),
        best(2),
        best(3),
        best(1),
        best(4),
    ];
    let entries = [
        PenalizeEntry::new(0, &block, 3, [1.3, 0.25, 0.5]),
        PenalizeEntry::new(11, &[best(5)], 1, [1.0, 1.5, 0.0]),
    ];
    let b_logits = gpu.buffer(&logits);
    let b_words = gpu.buffer(&table);
    let b_entries = gpu.buffer(&entries);
    let constants = TargetSamplingConstants {
        vocab: scratchy_target_metal::tape::ids::LogitsWidth(vocabulary as u32),
        rows: ROWS,
    };
    let pso = gpu.pipeline(
        "dflash2_target_sampling",
        "dflash2_penalize_bf16",
        constants,
    );
    let mut batch = gpu.batch();
    batch.encode(
        &pso,
        &[(&b_logits, 0), (&b_words, 1), (&b_entries, 2)],
        &[],
        &[],
        &[],
        tg(entries.len() * vocabulary.div_ceil(256), 1, 1),
        tg(256, 1, 1),
    );
    batch.commit(true);
    let got: Vec<u16> = read(&b_logits, rows * vocabulary);

    let argmax = |v: &[u16]| {
        (0..vocabulary)
            .max_by(|&a, &b| value(v[a]).total_cmp(&value(v[b])))
            .expect("a vocabulary")
    };
    let mut moved = 0;
    for row in 0..rows {
        let entry = (entries.iter())
            .find(|e| (e.logits_row..e.logits_row + e.rows).contains(&(row as u32)));
        for token in 0..vocabulary {
            let at = row * vocabulary + token;
            let (before, after) = (logits[at], got[at]);
            let Some(entry) = entry else {
                assert_eq!(after, before, "row {row} token {token}: an unpenalized row");
                continue;
            };
            let r = (row as u32 - entry.logits_row) as usize;
            let word = table[entry.table_row as usize * vocabulary + token];
            let drafted = (1..=r).filter(|&i| entry.inputs[i] == token as u32).count() as u32;
            let count = word + drafted;
            let expected = reference_penalty(value(before), count, entry);
            // One product rounds alike; a fused subtraction may land a bf16 step either way.
            let exact = after == bits(expected);
            let within = (value(after) - expected).abs() <= ulp_bf16(expected);
            assert!(
                exact || (count > 0 && within),
                "row {row} token {token}: wrote {} for {} (count {count}), expected {expected}",
                value(after),
                value(before)
            );
        }
        if entry.is_some_and(|e| e.rows > 1) {
            let span = row * vocabulary..(row + 1) * vocabulary;
            moved += usize::from(argmax(&got[span.clone()]) != argmax(&logits[span]));
        }
    }
    assert!(moved > 0, "the penalties moved no verify row's argmax");
}
