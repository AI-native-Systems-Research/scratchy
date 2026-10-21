// SPDX-License-Identifier: Apache-2.0
//! The tape player on the H100: the GEMM-chain tapes of `common::chains` and the qwen3.5-0.8b MLP
//! block of `common::mlp` played through the arena and compared with an f64 CPU reference, one
//! arena allocation per worker, no allocation per forward, colour reuse, and the refusals.
//!
//! Built only under `nvcc` (`required-features`); no GPU fails rather than skips. Each stage is
//! checked against its own GPU input (`common::numerics`' bounds), so the bound never compounds
//! along a chain.

// `pub`: each test binary uses a different part of `common`, and nothing in it is dead code.
pub mod common {
    pub mod attn_gate;
    pub mod attn_layer;
    pub mod chains;
    pub mod mlp;
    pub mod numerics;
}

use common::attn_gate::{ATTN_GATE_M128, ATTN_GATE_M256, attn_gate};
use common::attn_layer::{ATTN_LAYER_M128, ATTN_LAYER_M256, attn_layer};
use common::chains::{
    HIDDEN, THREE_GEMM_M256, TWO_GEMM_M128, TWO_GEMM_M256, assert_tape_is_lowered, three_gemm,
    two_gemm,
};
use common::mlp::{EPS, MLP_M128, MLP_M256, mlp};
use common::numerics::{
    ATTN_SCORE_MAX, AttnCase, add_reference, attention_reference, bf16_inputs, centred_gains,
    compare, compare_attention, compare_rope, compare_rows, gate_apply_reference,
    gate_split_reference, gather_paged, mismatched_bits, rms_norm_reference, rope_reference,
    silu_mul_reference,
};
use scratchy_subtile::subtile_ir::GainConvention;
use scratchy_target_cuda::kernels::{gemm_bf16_abt, rms_norm_bf16, rope_bf16};
use scratchy_target_cuda::rotary::rotary_rows;
use scratchy_target_cuda::runtime::{
    CudaDevice, CudaWorker, DeviceBuffer, ForwardError, KvHalf, KvPoolConfig, KvSequence,
    WorkerError,
};
use scratchy_target_cuda::step::CudaStep;
use scratchy_target_cuda::tape::{
    ArenaOffset, ArenaSpan, CudaTape, KvTape, RotaryTable, TapeError, WeightIx,
};

/// One worker's buckets: `M = 128` and `M = 256` of the same two-GEMM graph.
static TWO_GEMM_BUCKETS: [CudaTape; 2] = [TWO_GEMM_M128, TWO_GEMM_M256];

const H: usize = HIDDEN as usize;
/// `W1`'s and `W3`'s `N`, and `W2`'s `K`.
const WIDE: usize = 1024;

/// The tapes played here are the ones CUDA's compiler lowers (as `tape_layout` checks too), so
/// what passes on the GPU is what a real lowering produces.
#[test]
fn the_played_tapes_are_the_lowered_ones() {
    assert_tape_is_lowered("two_gemm(128)", &TWO_GEMM_M128, &two_gemm(128));
    assert_tape_is_lowered("two_gemm(256)", &TWO_GEMM_M256, &two_gemm(256));
    assert_tape_is_lowered("three_gemm", &THREE_GEMM_M256, &three_gemm());
    assert_tape_is_lowered("mlp(128)", &MLP_M128, &mlp(128));
    assert_tape_is_lowered("mlp(256)", &MLP_M256, &mlp(256));
    assert_tape_is_lowered("attn_gate(128)", &ATTN_GATE_M128, &attn_gate(128));
    assert_tape_is_lowered("attn_gate(256)", &ATTN_GATE_M256, &attn_gate(256));
    assert_tape_is_lowered("attn_layer(128)", &ATTN_LAYER_M128, &attn_layer(128));
    assert_tape_is_lowered("attn_layer(256)", &ATTN_LAYER_M256, &attn_layer(256));
}

/// Device 0, which must be an H100; panics with the reason otherwise.
fn h100() -> CudaDevice {
    CudaDevice::open()
        .unwrap_or_else(|e| panic!("the `nvcc` feature's tests need an sm_90 GPU (H100): {e}"))
}

/// The chains' weights as stored on the device, `[N, K]`: `W1 [1024, 512]`, `W2 [512, 1024]`,
/// `W3 [1024, 512]`.
struct Weights {
    w1: Vec<u16>,
    w2: Vec<u16>,
    w3: Vec<u16>,
}

fn weights() -> Weights {
    Weights {
        w1: bf16_inputs(WIDE * H, 0x0057_0001),
        w2: bf16_inputs(H * WIDE, 0x0057_0002),
        w3: bf16_inputs(WIDE * H, 0x0057_0003),
    }
}

/// The embedded hidden of an `m`-row bucket.
fn input(m: usize) -> Vec<u16> {
    bf16_inputs(m * H, 0x0000_0E3B ^ m as u64)
}

/// Colour `c` of `tape`'s arena.
fn colour(tape: &CudaTape, c: usize) -> ArenaSpan {
    ArenaSpan {
        offset: ArenaOffset(tape.arena.colour_offsets[c]),
        bytes: tape.arena.colour_bytes[c],
    }
}

fn read(worker: &mut CudaWorker<'_>, span: ArenaSpan) -> Vec<u16> {
    let mut out = vec![0u16; span.bytes as usize / 2];
    worker.read_arena(span, &mut out).expect("read_arena");
    out
}

fn verdict(pass: bool) -> &'static str {
    if pass { "PASS" } else { "FAIL" }
}

/// Check `gpu` (`[m, n]`) against `a · bᵀ`; print the line; return whether it passed.
fn check_stage(
    what: &str,
    gpu: &[u16],
    a: &[u16],
    b: &[u16],
    (m, n, k): (usize, usize, usize),
) -> bool {
    let cmp = compare(gpu, a, b, m, n, k);
    eprintln!(
        "{what} m={m} n={n} k={k}: {} {}",
        cmp.line(),
        verdict(cmp.pass())
    );
    cmp.pass()
}

/// Both buckets of one worker: `y1` and `y2` each within the bound of its own GPU input.
#[test]
fn a_two_gemm_chain_matches_the_cpu_reference_in_each_bucket() {
    let device = h100();
    let w = weights();
    let table = [
        DeviceBuffer::empty(),
        DeviceBuffer::upload(&device, &w.w1).expect("upload W1"),
        DeviceBuffer::upload(&device, &w.w2).expect("upload W2"),
    ];
    let mut worker = CudaWorker::new(&device, &TWO_GEMM_BUCKETS, &table, None).expect("worker");
    let mut pass = true;
    for (bucket, tape) in TWO_GEMM_BUCKETS.iter().enumerate() {
        let m = tape.bucket_m as usize;
        let x = input(m);
        let mut y2 = vec![0u16; m * H];
        worker
            .forward(bucket, &x, &[], None, &mut y2)
            .expect("forward");
        let y1 = read(&mut worker, colour(tape, 1));
        pass &= check_stage("y1 = x·W1ᵀ", &y1, &x, &w.w1, (m, WIDE, H));
        pass &= check_stage("y2 = y1·W2ᵀ", &y2, &y1, &w.w2, (m, H, WIDE));
    }
    assert!(pass, "a stage of the two-GEMM chain is out of bound");
}

/// One arena of the larger bucket's bytes, three allocations in all, none per forward.
#[test]
fn the_worker_allocates_once_and_no_forward_allocates() {
    let device = h100();
    let w = weights();
    let before = device.allocations();
    let table = [
        DeviceBuffer::empty(),
        DeviceBuffer::upload(&device, &w.w1).expect("upload W1"),
        DeviceBuffer::upload(&device, &w.w2).expect("upload W2"),
    ];
    let mut worker = CudaWorker::new(&device, &TWO_GEMM_BUCKETS, &table, None).expect("worker");
    let built = device.allocations() - before;
    let arena = worker.arena_bytes();
    eprintln!(
        "arena_bytes={arena} (expect 1048576) allocations(weights + arena)={built} (expect 3) {}",
        verdict(arena == 1048576 && built == 3)
    );
    assert_eq!(arena, 1048576, "the arena is the larger bucket's total");
    assert_eq!(built, 3, "two weights and one arena");

    let inputs = [input(128), input(256)];
    let mut firsts: [Option<Vec<u16>>; 2] = [None, None];
    let at_forwards = device.allocations();
    let mut free_after_first = None;
    let mut identical = true;
    for (i, bucket) in [1usize, 0, 1, 0, 1].into_iter().enumerate() {
        let m = TWO_GEMM_BUCKETS[bucket].bucket_m as usize;
        let mut y2 = vec![0u16; m * H];
        worker
            .forward(bucket, &inputs[bucket], &[], None, &mut y2)
            .expect("forward");
        if i == 0 {
            free_after_first = Some(device.mem_info().expect("cudaMemGetInfo").0);
        }
        match &firsts[bucket] {
            Some(first) => identical &= *first == y2,
            None => firsts[bucket] = Some(y2),
        }
    }
    let free_after_first = free_after_first.expect("forward 1 ran");
    let free_after_last = device.mem_info().expect("cudaMemGetInfo").0;
    let during = device.allocations() - at_forwards;
    eprintln!(
        "five forwards (256, 128, 256, 128, 256): allocations={during} (expect 0) \
         free_after_forward_1={free_after_first} free_after_forward_5={free_after_last} \
         results_bit_identical={identical} {}",
        verdict(during == 0 && free_after_first == free_after_last && identical)
    );
    assert_eq!(during, 0, "a forward allocated device memory");
    assert_eq!(
        free_after_first, free_after_last,
        "device free memory moved"
    );
    assert!(identical, "a bucket's result changed between forwards");
}

/// `y3` is written over `y1`'s colour and is right; `y2` is the two-GEMM chain's result.
#[test]
fn the_three_gemm_chain_reuses_a_colour_and_stays_right() {
    let device = h100();
    let w = weights();
    let table = [
        DeviceBuffer::empty(),
        DeviceBuffer::upload(&device, &w.w1).expect("upload W1"),
        DeviceBuffer::upload(&device, &w.w2).expect("upload W2"),
        DeviceBuffer::upload(&device, &w.w3).expect("upload W3"),
    ];
    let x = input(256);

    let two = std::slice::from_ref(&TWO_GEMM_M256);
    let mut worker = CudaWorker::new(&device, two, &table[..3], None).expect("two-GEMM worker");
    let mut two_y2 = vec![0u16; 256 * H];
    worker
        .forward(0, &x, &[], None, &mut two_y2)
        .expect("two-GEMM forward");
    drop(worker);

    let three = std::slice::from_ref(&THREE_GEMM_M256);
    let mut worker = CudaWorker::new(&device, three, &table, None).expect("three-GEMM worker");
    let mut y3 = vec![0u16; 256 * WIDE];
    worker
        .forward(0, &x, &[], None, &mut y3)
        .expect("three-GEMM forward");
    let y2 = read(&mut worker, colour(&THREE_GEMM_M256, 2));
    let pass = check_stage(
        "y3 = y2·W3ᵀ (over y1's colour)",
        &y3,
        &y2,
        &w.w3,
        (256, WIDE, H),
    );
    let same = y2 == two_y2;
    eprintln!(
        "y2 of the three-GEMM chain == the two-GEMM result: {same} {}",
        verdict(same)
    );
    assert!(pass, "y3 is out of bound");
    assert!(same, "y2 differs from the two-GEMM chain's result");
}

/// A short input is refused before anything is enqueued, and the worker still plays after it; a
/// short weight is refused at construction.
#[test]
fn short_inputs_and_short_weights_are_refused() {
    let device = h100();
    let w = weights();
    let table = [
        DeviceBuffer::empty(),
        DeviceBuffer::upload(&device, &w.w1).expect("upload W1"),
        DeviceBuffer::upload(&device, &w.w2).expect("upload W2"),
    ];
    let mut worker = CudaWorker::new(&device, &TWO_GEMM_BUCKETS, &table, None).expect("worker");
    let x = input(256);
    let mut y2 = vec![0u16; 256 * H];
    let got = worker.forward(1, &x[..x.len() - 1], &[], None, &mut y2);
    let want = Err(ForwardError::InputSize {
        expected: 262144,
        got: 262142,
    });
    eprintln!("input one element short: {got:?} {}", verdict(got == want));
    assert_eq!(got, want);
    worker
        .forward(1, &x, &[], None, &mut y2)
        .expect("the next valid forward");
    let y1 = read(&mut worker, colour(&TWO_GEMM_M256, 1));
    let pass = check_stage(
        "after the refusal, y2 = y1·W2ᵀ",
        &y2,
        &y1,
        &w.w2,
        (256, H, WIDE),
    );
    assert!(pass, "the forward after a refusal is out of bound");
    drop(worker);

    let short = [
        DeviceBuffer::empty(),
        DeviceBuffer::upload(&device, &w.w1[..w.w1.len() - 1]).expect("upload short W1"),
        DeviceBuffer::upload(&device, &w.w2).expect("upload W2"),
    ];
    let got = CudaWorker::new(&device, &TWO_GEMM_BUCKETS, &short, None).map(|_| ());
    let want = Err(WorkerError::WeightTooSmall {
        ix: 1,
        need: 1048576,
        have: 1048574,
    });
    eprintln!("W1 one element short: {got:?} {}", verdict(got == want));
    assert_eq!(got, want);
}

// ── The qwen3.5-0.8b MLP block ─────────────────────────────────────

/// One worker's buckets: `M = 128` and `M = 256` of the MLP block.
static MLP_BUCKETS: [CudaTape; 2] = [MLP_M128, MLP_M256];

const MLP_H: usize = common::mlp::H as usize;
const MLP_I: usize = common::mlp::I as usize;

/// The block's weights as stored on the device: `g [H]` (zero-centred), `Wg [I, H]`, `Wu [I, H]`,
/// `Wd [H, I]`.
struct MlpWeights {
    g: Vec<u16>,
    wg: Vec<u16>,
    wu: Vec<u16>,
    wd: Vec<u16>,
}

fn mlp_weights() -> MlpWeights {
    MlpWeights {
        g: centred_gains(MLP_H, 0x0B10_0001),
        wg: bf16_inputs(MLP_I * MLP_H, 0x0B10_0002),
        wu: bf16_inputs(MLP_I * MLP_H, 0x0B10_0003),
        wd: bf16_inputs(MLP_H * MLP_I, 0x0B10_0004),
    }
}

/// The weight table, by source index: 0 is the embedded hidden, not a weight.
fn mlp_table(device: &CudaDevice, w: &MlpWeights) -> [DeviceBuffer; 5] {
    [
        DeviceBuffer::empty(),
        DeviceBuffer::upload(device, &w.g).expect("upload g"),
        DeviceBuffer::upload(device, &w.wg).expect("upload Wg"),
        DeviceBuffer::upload(device, &w.wu).expect("upload Wu"),
        DeviceBuffer::upload(device, &w.wd).expect("upload Wd"),
    ]
}

/// The embedded hidden of an `m`-row MLP bucket.
fn mlp_input(m: usize) -> Vec<u16> {
    bf16_inputs(m * MLP_H, 0x0000_0B1C ^ m as u64)
}

/// Check `gpu` against a row-kernel reference; print the line; return whether it passed.
fn check_rows(what: &str, gpu: &[u16], refs: &[f64], (rows, cols): (usize, usize)) -> bool {
    let cmp = compare_rows(gpu, refs, cols);
    eprintln!(
        "{what} rows={rows} cols={cols}: {} {}",
        cmp.line(),
        verdict(cmp.pass())
    );
    cmp.pass()
}

/// The tape's first step, its input norm, run on its own, on device copies of `x` and `g`, with
/// the tape's eps and convention: `n`, which a later step overwrites in the arena before the
/// forward ends (the MLP's `d`, the attention gate's `o`).
fn n_direct(device: &CudaDevice, tape: &CudaTape, x: &[u16], g: &[u16]) -> Vec<u16> {
    let CudaStep::RmsNorm(norm) = tape.steps[0] else {
        panic!("the tape's first step is its norm");
    };
    assert_eq!(
        (norm.eps(), norm.convention),
        (EPS, GainConvention::OnePlusScale)
    );
    let (xb, gb) = (
        DeviceBuffer::upload(device, x).expect("upload x"),
        DeviceBuffer::upload(device, g).expect("upload g"),
    );
    let out = DeviceBuffer::upload(device, &vec![0; x.len()]).expect("upload n");
    // SAFETY: cudaMalloc pointers are 256-byte aligned; `x` and `out` hold rows·cols elements and
    // `g` cols; `out` is its own allocation; null is the default stream.
    unsafe {
        rms_norm_bf16(
            xb.ptr(),
            gb.ptr(),
            out.ptr(),
            (norm.rows, norm.cols),
            norm.eps(),
            norm.convention.offset(),
            std::ptr::null_mut(),
        )
    }
    .expect("rms_norm_bf16 launch");
    out.download().expect("download n")
}

/// Both buckets of one worker, each stage against its own GPU input: `n_direct` against the norm
/// reference; `gate` and `up` against the GEMMs of `n_direct` (so a wrong norm step in the tape
/// shows up there, the norm kernel being deterministic); `a` against SiluMul of `gate` and `up`;
/// `d` against the GEMM of `a`; `y` bit-exactly against the add of `d` and `x`.
#[test]
fn the_mlp_block_matches_the_cpu_reference_in_each_bucket() {
    let device = h100();
    let w = mlp_weights();
    let table = mlp_table(&device, &w);
    let mut worker = CudaWorker::new(&device, &MLP_BUCKETS, &table, None).expect("worker");
    let mut pass = true;
    for (bucket, tape) in MLP_BUCKETS.iter().enumerate() {
        let m = tape.bucket_m as usize;
        let x = mlp_input(m);
        let mut y = vec![0u16; m * MLP_H];
        worker
            .forward(bucket, &x, &[], None, &mut y)
            .expect("forward");
        let n = n_direct(&device, tape, &x, &w.g);
        let n_ref = rms_norm_reference(&x, &w.g, (m, MLP_H), f64::from(EPS), 1.0);
        pass &= check_rows("n_direct = RmsNorm(x, g)", &n, &n_ref, (m, MLP_H));
        let gate = read(&mut worker, colour(tape, 2));
        let up = read(&mut worker, colour(tape, 3));
        let a = read(&mut worker, colour(tape, 5));
        let d = read(&mut worker, colour(tape, 1));
        pass &= check_stage("gate = n·Wgᵀ", &gate, &n, &w.wg, (m, MLP_I, MLP_H));
        pass &= check_stage("up = n·Wuᵀ", &up, &n, &w.wu, (m, MLP_I, MLP_H));
        let a_ref = silu_mul_reference(&gate, &up);
        pass &= check_rows("a = silu(gate)·up", &a, &a_ref, (m, MLP_I));
        pass &= check_stage("d = a·Wdᵀ", &d, &a, &w.wd, (m, MLP_H, MLP_I));
        let mismatched = mismatched_bits(&y, &add_reference(&d, &x));
        eprintln!(
            "y = d + x rows={m} cols={MLP_H}: mismatched_bits={mismatched} {}",
            verdict(mismatched == 0)
        );
        pass &= mismatched == 0;
    }
    assert!(pass, "a stage of the MLP block is out of bound");
}

/// One arena of the larger bucket's 8912896 bytes, five allocations in all (four weights and the
/// arena), none per forward, and every result bit-identical to its bucket's first.
#[test]
fn the_mlp_worker_allocates_once_and_no_forward_allocates() {
    let device = h100();
    let w = mlp_weights();
    let before = device.allocations();
    let table = mlp_table(&device, &w);
    let mut worker = CudaWorker::new(&device, &MLP_BUCKETS, &table, None).expect("worker");
    let built = device.allocations() - before;
    let arena = worker.arena_bytes();
    eprintln!(
        "mlp arena_bytes={arena} (expect 8912896) allocations(weights + arena)={built} \
         (expect 5) {}",
        verdict(arena == 8912896 && built == 5)
    );
    assert_eq!(arena, 8912896, "the arena is the larger bucket's total");
    assert_eq!(built, 5, "four weights and one arena");

    let inputs = [mlp_input(128), mlp_input(256)];
    let mut firsts: [Option<Vec<u16>>; 2] = [None, None];
    let at_forwards = device.allocations();
    let mut free_after_first = None;
    let mut identical = true;
    for (i, bucket) in [1usize, 0, 1, 0, 1].into_iter().enumerate() {
        let m = MLP_BUCKETS[bucket].bucket_m as usize;
        let mut y = vec![0u16; m * MLP_H];
        worker
            .forward(bucket, &inputs[bucket], &[], None, &mut y)
            .expect("forward");
        if i == 0 {
            free_after_first = Some(device.mem_info().expect("cudaMemGetInfo").0);
        }
        match &firsts[bucket] {
            Some(first) => identical &= *first == y,
            None => firsts[bucket] = Some(y),
        }
    }
    let free_after_first = free_after_first.expect("forward 1 ran");
    let free_after_last = device.mem_info().expect("cudaMemGetInfo").0;
    let during = device.allocations() - at_forwards;
    eprintln!(
        "mlp five forwards (256, 128, 256, 128, 256): allocations={during} (expect 0) \
         free_after_forward_1={free_after_first} free_after_forward_5={free_after_last} \
         results_bit_identical={identical} {}",
        verdict(during == 0 && free_after_first == free_after_last && identical)
    );
    assert_eq!(during, 0, "a forward allocated device memory");
    assert_eq!(
        free_after_first, free_after_last,
        "device free memory moved"
    );
    assert!(identical, "a bucket's result changed between forwards");
}

/// [`MLP_M256`]'s steps with its down GEMM (step 4) reading weight 9, which the tape does not
/// declare.
static BAD_WEIGHT_STEPS: [CudaStep; 6] = {
    let s = MLP_M256.steps;
    let down = match s[4] {
        CudaStep::Gemm(mut down) => {
            down.b = WeightIx(9);
            CudaStep::Gemm(down)
        }
        _ => panic!("MLP_M256's step 4 is its down GEMM"),
    };
    [s[0], s[1], s[2], s[3], down, s[5]]
};

static BAD_WEIGHT: [CudaTape; 1] = [CudaTape {
    steps: &BAD_WEIGHT_STEPS,
    ..MLP_M256
}];

/// A tape whose step reads a weight index it does not declare is refused by `CudaWorker::new`
/// before anything is allocated.
#[test]
fn a_tape_reading_an_undeclared_weight_is_refused_before_any_allocation() {
    let device = h100();
    let w = mlp_weights();
    let table = mlp_table(&device, &w);
    let before = device.allocations();
    let got = CudaWorker::new(&device, &BAD_WEIGHT, &table, None).map(|_| ());
    let allocated = device.allocations() - before;
    let want = Err(WorkerError::Tape {
        bucket: 0,
        error: TapeError::WeightIndex {
            step: 4,
            ix: 9,
            weights: 5,
        },
    });
    eprintln!(
        "MLP_M256 with its down GEMM reading weight 9: {got:?} allocations={allocated} \
         (expect 0) {}",
        verdict(got == want && allocated == 0)
    );
    assert_eq!(got, want);
    assert_eq!(allocated, 0, "a refused tape allocated");
}

// ── qwen3.5-0.8b's attention prelude and gate ──────────────────────

/// One worker's buckets: `M = 128` and `M = 256` of the attention prelude and gate.
static ATTN_BUCKETS: [CudaTape; 2] = [ATTN_GATE_M128, ATTN_GATE_M256];

const AG_H: usize = common::attn_gate::H as usize;
const AG_Q: usize = common::attn_gate::Q as usize;
const AG_HD: usize = common::attn_gate::HD as usize;
const AG_R: usize = common::attn_gate::R as usize;

/// The block's weights as stored on the device: `g_in [H]` and `q_norm [HD]` (zero-centred),
/// `Wq [2Q, H]`, `Wo [H, Q]`.
struct AttnWeights {
    g_in: Vec<u16>,
    wq: Vec<u16>,
    q_norm: Vec<u16>,
    wo: Vec<u16>,
}

fn attn_weights() -> AttnWeights {
    AttnWeights {
        g_in: centred_gains(AG_H, 0x0A77_0001),
        wq: bf16_inputs(2 * AG_Q * AG_H, 0x0A77_0002),
        q_norm: centred_gains(AG_HD, 0x0A77_0003),
        wo: bf16_inputs(AG_H * AG_Q, 0x0A77_0004),
    }
}

/// The weight table, by source index: 0 is the embedded hidden and 4, 5 the rotary `cos` and
/// `sin`, none of them weights.
fn attn_table(device: &CudaDevice, w: &AttnWeights) -> [DeviceBuffer; 7] {
    [
        DeviceBuffer::empty(),
        DeviceBuffer::upload(device, &w.g_in).expect("upload g_in"),
        DeviceBuffer::upload(device, &w.wq).expect("upload Wq"),
        DeviceBuffer::upload(device, &w.q_norm).expect("upload q_norm"),
        DeviceBuffer::empty(),
        DeviceBuffer::empty(),
        DeviceBuffer::upload(device, &w.wo).expect("upload Wo"),
    ]
}

/// The embedded hidden of an `m`-row attention bucket.
fn attn_input(m: usize) -> Vec<u16> {
    bf16_inputs(m * AG_H, 0x0000_0A7E ^ m as u64)
}

/// The positions of an `m`-row bucket: `7 + 3t`, with row 5 at 200000.
fn attn_positions(m: usize) -> Vec<u32> {
    (0..m as u32)
        .map(|t| if t == 5 { 200000 } else { 7 + 3 * t })
        .collect()
}

/// Both buckets of one worker, each stage against its own GPU input: `n_direct` against the norm
/// reference; `qg` against the GEMM of `n_direct`; `q` and `gate` bit-exactly against the per-head
/// split of `qg`; `qn` against the per-head norm of `q`; `qr` against the rope of `qn` under the
/// uploaded table, its pass-through lanes bit-exact; `ga` against the gate apply of `qr` and
/// `gate`; `o` against the GEMM of `ga`; `y` bit-exactly against the add of `o` and `x`.
#[test]
fn the_attention_gate_matches_the_cpu_reference_in_each_bucket() {
    let device = h100();
    let w = attn_weights();
    let table = attn_table(&device, &w);
    let mut worker = CudaWorker::new(&device, &ATTN_BUCKETS, &table, None).expect("worker");
    let mut pass = true;
    for (bucket, tape) in ATTN_BUCKETS.iter().enumerate() {
        let m = tape.bucket_m as usize;
        let x = attn_input(m);
        let positions = attn_positions(m);
        let mut y = vec![0u16; m * AG_H];
        worker
            .forward(bucket, &x, &positions, None, &mut y)
            .expect("forward");
        let n = n_direct(&device, tape, &x, &w.g_in);
        let n_ref = rms_norm_reference(&x, &w.g_in, (m, AG_H), f64::from(EPS), 1.0);
        pass &= check_rows("n_direct = RmsNorm(x, g_in)", &n, &n_ref, (m, AG_H));
        let qg = read(&mut worker, colour(tape, 2));
        pass &= check_stage("qg = n·Wqᵀ", &qg, &n, &w.wq, (m, 2 * AG_Q, AG_H));
        let q = read(&mut worker, colour(tape, 3));
        let gate = read(&mut worker, colour(tape, 8));
        let (want_q, want_gate) = gate_split_reference(&qg, (m, AG_Q), AG_HD);
        let split_bits = mismatched_bits(&q, &want_q) + mismatched_bits(&gate, &want_gate);
        eprintln!(
            "q, gate = GateSplit(qg) rows={m} half_cols={AG_Q}: mismatched_bits={split_bits} {}",
            verdict(split_bits == 0)
        );
        pass &= split_bits == 0;
        let qn = read(&mut worker, colour(tape, 4));
        let qn_ref = rms_norm_reference(&q, &w.q_norm, (m * 8, AG_HD), f64::from(EPS), 1.0);
        pass &= check_rows(
            "qn = RmsNorm(q viewed [8M, 256], q_norm)",
            &qn,
            &qn_ref,
            (m * 8, AG_HD),
        );
        let qr = read(&mut worker, colour(tape, 5));
        let rotary = tape.rotary.expect("the tape has a rotary table");
        let uploaded = read(&mut worker, rotary.span);
        let table_matches = uploaded == rotary_rows(&positions, rotary.rot_dim, rotary.theta());
        let refs = rope_reference(&qn, &uploaded, (m, AG_Q), (AG_HD, AG_R), AG_R / 2);
        let cmp = compare_rope(&qr, &qn, &refs, AG_Q);
        let ok = cmp.rotated.pass() && cmp.pass_through_mismatched == 0 && table_matches;
        eprintln!(
            "qr = RopeRotate(qn) rows={m} cols={AG_Q}: {} pass_through_mismatched_bits={} \
             table_is_the_positions'={table_matches} {}",
            cmp.rotated.line(),
            cmp.pass_through_mismatched,
            verdict(ok)
        );
        pass &= ok;
        let ga = read(&mut worker, colour(tape, 6));
        let ga_ref = gate_apply_reference(&qr, &gate);
        pass &= check_rows("ga = qr·sigmoid(gate)", &ga, &ga_ref, (m, AG_Q));
        let o = read(&mut worker, colour(tape, 1));
        pass &= check_stage(
            "o = ga·Woᵀ (over n's colour)",
            &o,
            &ga,
            &w.wo,
            (m, AG_H, AG_Q),
        );
        let mismatched = mismatched_bits(&y, &add_reference(&o, &x));
        eprintln!(
            "y = o + x rows={m} cols={AG_H}: mismatched_bits={mismatched} {}",
            verdict(mismatched == 0)
        );
        pass &= mismatched == 0;
    }
    assert!(pass, "a stage of the attention gate is out of bound");
}

/// One arena of the larger bucket's 8945664 bytes, five allocations in all (four weights and the
/// arena), none per forward, and every result bit-identical to its bucket's first.
#[test]
fn the_attention_gate_worker_allocates_once_and_no_forward_allocates() {
    let device = h100();
    let w = attn_weights();
    let before = device.allocations();
    let table = attn_table(&device, &w);
    let mut worker = CudaWorker::new(&device, &ATTN_BUCKETS, &table, None).expect("worker");
    let built = device.allocations() - before;
    let arena = worker.arena_bytes();
    eprintln!(
        "attention gate arena_bytes={arena} (expect 8945664) allocations(weights + arena)={built} \
         (expect 5) {}",
        verdict(arena == 8945664 && built == 5)
    );
    assert_eq!(arena, 8945664, "the arena is the larger bucket's total");
    assert_eq!(built, 5, "four weights and one arena");

    let inputs = [attn_input(128), attn_input(256)];
    let positions = [attn_positions(128), attn_positions(256)];
    let mut firsts: [Option<Vec<u16>>; 2] = [None, None];
    let at_forwards = device.allocations();
    let mut free_after_first = None;
    let mut identical = true;
    for (i, bucket) in [1usize, 0, 1, 0, 1].into_iter().enumerate() {
        let m = ATTN_BUCKETS[bucket].bucket_m as usize;
        let mut y = vec![0u16; m * AG_H];
        worker
            .forward(bucket, &inputs[bucket], &positions[bucket], None, &mut y)
            .expect("forward");
        if i == 0 {
            free_after_first = Some(device.mem_info().expect("cudaMemGetInfo").0);
        }
        match &firsts[bucket] {
            Some(first) => identical &= *first == y,
            None => firsts[bucket] = Some(y),
        }
    }
    let free_after_first = free_after_first.expect("forward 1 ran");
    let free_after_last = device.mem_info().expect("cudaMemGetInfo").0;
    let during = device.allocations() - at_forwards;
    eprintln!(
        "attention gate five forwards (256, 128, 256, 128, 256): allocations={during} (expect 0) \
         free_after_forward_1={free_after_first} free_after_forward_5={free_after_last} \
         results_bit_identical={identical} {}",
        verdict(during == 0 && free_after_first == free_after_last && identical)
    );
    assert_eq!(during, 0, "a forward allocated device memory");
    assert_eq!(
        free_after_first, free_after_last,
        "device free memory moved"
    );
    assert!(identical, "a bucket's result changed between forwards");
}

/// A forward given the wrong number of positions is refused before anything is enqueued, and a
/// tape with no rotary table takes none.
#[test]
fn a_forward_with_the_wrong_positions_is_refused() {
    let device = h100();
    let w = attn_weights();
    let table = attn_table(&device, &w);
    let mut worker = CudaWorker::new(&device, &ATTN_BUCKETS, &table, None).expect("worker");
    let x = attn_input(256);
    let mut y = vec![0u16; 256 * AG_H];
    let got = worker.forward(1, &x, &attn_positions(255), None, &mut y);
    let want = Err(ForwardError::Positions {
        expected: 256,
        got: 255,
    });
    let untouched = y.iter().all(|&b| b == 0);
    eprintln!(
        "attention gate with 255 positions: {got:?} result_untouched={untouched} {}",
        verdict(got == want && untouched)
    );
    assert_eq!(got, want);
    assert!(untouched, "a refused forward wrote its result");
    drop(worker);

    let w = mlp_weights();
    let table = mlp_table(&device, &w);
    let mut worker = CudaWorker::new(&device, &MLP_BUCKETS, &table, None).expect("MLP worker");
    let got = worker.forward(1, &mlp_input(256), &[7], None, &mut vec![0u16; 256 * MLP_H]);
    let want = Err(ForwardError::Positions {
        expected: 0,
        got: 1,
    });
    eprintln!("MLP with one position: {got:?} {}", verdict(got == want));
    assert_eq!(got, want);
}

/// [`ATTN_GATE_M256`] with its rotary table one row past the arena's end.
static BAD_ROTARY: [CudaTape; 1] = [CudaTape {
    rotary: Some(RotaryTable {
        span: ArenaSpan {
            offset: ArenaOffset(8912896 + 256),
            bytes: 32768,
        },
        rot_dim: 64,
        theta_bits: 0x416312D000000000,
    }),
    ..ATTN_GATE_M256
}];

/// A tape whose rotary table reaches past its arena is refused by `CudaWorker::new` before
/// anything is allocated.
#[test]
fn a_tape_whose_rotary_table_reaches_past_the_arena_is_refused() {
    assert_eq!(f64::from_bits(0x416312D000000000), 1e7);
    let device = h100();
    let w = attn_weights();
    let table = attn_table(&device, &w);
    let before = device.allocations();
    let got = CudaWorker::new(&device, &BAD_ROTARY, &table, None).map(|_| ());
    let allocated = device.allocations() - before;
    let want = Err(WorkerError::Tape {
        bucket: 0,
        error: TapeError::RotaryOutside {
            span: ArenaSpan {
                offset: ArenaOffset(8913152),
                bytes: 32768,
            },
            total: 8945664,
        },
    });
    eprintln!(
        "ATTN_GATE_M256 with its rotary table past the arena: {got:?} allocations={allocated} \
         (expect 0) {}",
        verdict(got == want && allocated == 0)
    );
    assert_eq!(got, want);
    assert_eq!(allocated, 0, "a refused tape allocated");
}

// ── qwen3.5-0.8b's full-attention layer ────────────────────────────

/// One worker's buckets: `M = 128` and `M = 256` of the full-attention layer.
static LAYER_BUCKETS: [CudaTape; 2] = [ATTN_LAYER_M128, ATTN_LAYER_M256];

const AL_H: usize = common::attn_layer::H as usize;
const AL_Q: usize = common::attn_layer::Q as usize;
const AL_KV: usize = common::attn_layer::KV as usize;
const AL_HD: usize = common::attn_layer::HD as usize;
const AL_R: usize = common::attn_layer::R as usize;
const AL_HEADS: usize = common::attn_layer::HEADS as usize;
const AL_KV_HEADS: usize = common::attn_layer::KV_HEADS as usize;
/// The KV pool's blocks in every attention-layer test.
const POOL_BLOCKS: u32 = 16;
/// One block of one layer's K (or V): `kv_heads · 64 · 256` bf16.
const BLOCK_ELEMS: usize = AL_KV_HEADS * 64 * AL_HD;

/// The layer's weights as stored on the device: the zero-centred gains `g_in [H]`, `q_norm` and
/// `k_norm [HD]`; `Wq [2Q, H]`, `Wk` and `Wv [KV, H]`, `Wo [H, Q]`.
struct LayerWeights {
    g_in: Vec<u16>,
    wq: Vec<u16>,
    q_norm: Vec<u16>,
    wk: Vec<u16>,
    k_norm: Vec<u16>,
    wv: Vec<u16>,
    wo: Vec<u16>,
}

fn layer_weights() -> LayerWeights {
    LayerWeights {
        g_in: centred_gains(AL_H, 0x0A11_0001),
        wq: bf16_inputs(2 * AL_Q * AL_H, 0x0A11_0002),
        q_norm: centred_gains(AL_HD, 0x0A11_0003),
        wk: bf16_inputs(AL_KV * AL_H, 0x0A11_0004),
        k_norm: centred_gains(AL_HD, 0x0A11_0005),
        wv: bf16_inputs(AL_KV * AL_H, 0x0A11_0006),
        wo: bf16_inputs(AL_H * AL_Q, 0x0A11_0007),
    }
}

/// The weight table, by source index (the fixture's): 0, the rotary pairs 7-10 and the prefix
/// caches 11, 12 are not weights.
fn layer_table(device: &CudaDevice, w: &LayerWeights) -> Vec<DeviceBuffer> {
    let up = |b: &[u16]| DeviceBuffer::upload(device, b).expect("upload a weight");
    let mut t: Vec<DeviceBuffer> = (0..14).map(|_| DeviceBuffer::empty()).collect();
    t[1] = up(&w.g_in);
    t[2] = up(&w.wq);
    t[3] = up(&w.q_norm);
    t[4] = up(&w.wk);
    t[5] = up(&w.k_norm);
    t[6] = up(&w.wv);
    t[13] = up(&w.wo);
    t
}

const POOL: Option<KvPoolConfig> = Some(KvPoolConfig {
    num_blocks: POOL_BLOCKS,
});

/// The embedded hidden of an `m`-row attention-layer bucket, under `seed`.
fn layer_input(m: usize, seed: u64) -> Vec<u16> {
    bf16_inputs(m * AL_H, seed ^ m as u64)
}

/// `m` positions of a sequence at `prefix_len`: `prefix_len + t`.
fn seq_positions(m: usize, prefix_len: u32) -> Vec<u32> {
    (0..m as u32).map(|t| prefix_len + t).collect()
}

/// Layer index 0's `half` of every block of the pool, in block order.
fn read_pool(worker: &mut CudaWorker<'_>, half: KvHalf) -> Vec<Vec<u16>> {
    (0..POOL_BLOCKS)
        .map(|b| {
            let mut out = vec![0u16; BLOCK_ELEMS];
            worker.read_kv(0, half, b, &mut out).expect("read_kv");
            out
        })
        .collect()
}

/// The sequence's first `tokens` rows of a half of the pool (`blocks`), `[tokens, KV]`.
fn gather(blocks: &[Vec<u16>], table: &[u32], tokens: usize) -> Vec<u16> {
    let flat: Vec<u16> = blocks.iter().flatten().copied().collect();
    gather_paged(&flat, table, tokens, (AL_KV_HEADS, 64, AL_HD))
}

/// Check the attention output `attn` (`[m, Q]`, its first `q_len` rows) against the f64
/// reference over the rotated query `qr` and the cache's K and V through `table`, `P` and `L`.
fn check_attention(
    what: &str,
    attn: &[u16],
    qr: &[u16],
    (k, v): (&[u16], &[u16]),
    (prefix_len, q_len): (usize, usize),
) -> bool {
    let (refs, score_max) = attention_reference(&AttnCase {
        q: qr,
        k,
        v,
        q_heads: AL_HEADS,
        kv_heads: AL_KV_HEADS,
        head_dim: AL_HD,
        prefix_len,
        q_len,
        scale: f64::from(common::attn_layer::SCALE),
    });
    let cmp = compare_attention(attn, &refs, AL_Q);
    let ok = cmp.pass() && score_max <= ATTN_SCORE_MAX;
    eprintln!(
        "{what} P={prefix_len} L={q_len}: {} score_max={score_max:.4} (≤ {ATTN_SCORE_MAX}) {}",
        cmp.line(),
        verdict(ok)
    );
    ok
}

/// The cache at each of the forward's real rows' slots is `kr`'s and `v`'s row bit for bit.
fn cache_mismatches(
    (k_blocks, v_blocks): (&[Vec<u16>], &[Vec<u16>]),
    (kr, v): (&[u16], &[u16]),
    table: &[u32],
    (prefix_len, q_len): (usize, usize),
) -> usize {
    let k_rows = gather(k_blocks, table, prefix_len + q_len);
    let v_rows = gather(v_blocks, table, prefix_len + q_len);
    let at = prefix_len * AL_KV;
    let n = q_len * AL_KV;
    mismatched_bits(&k_rows[at..][..n], &kr[..n]) + mismatched_bits(&v_rows[at..][..n], &v[..n])
}

/// Both buckets, P = 0 and L = M, table `[11, 3, 14, 0]` (its first `M / 64` entries), each stage
/// against its own GPU input; the cache bit-exact against `kr` and `v` at the mapped slots; the
/// attention against the f64 reference over `qr` and the cache contents.
#[test]
fn the_attention_layer_matches_the_cpu_reference_in_each_bucket() {
    let device = h100();
    let w = layer_weights();
    let table = layer_table(&device, &w);
    let mut worker = CudaWorker::new(&device, &LAYER_BUCKETS, &table, POOL).expect("worker");
    let mut pass = true;
    for (bucket, tape) in LAYER_BUCKETS.iter().enumerate() {
        let m = tape.bucket_m as usize;
        let block_table = &[11u32, 3, 14, 0][..m / 64];
        let x = layer_input(m, 0x0A1E);
        let positions = seq_positions(m, 0);
        let seq = KvSequence {
            block_table,
            prefix_len: 0,
            q_len: m as u32,
        };
        let mut y = vec![0u16; m * AL_H];
        worker
            .forward(bucket, &x, &positions, Some(&seq), &mut y)
            .expect("forward");
        // `n` (colour 1) holds `o` by the forward's end: the norm kernel is run alone.
        let n = n_direct(&device, tape, &x, &w.g_in);
        let n_ref = rms_norm_reference(&x, &w.g_in, (m, AL_H), f64::from(EPS), 1.0);
        pass &= check_rows("n_direct = RmsNorm(x, g_in)", &n, &n_ref, (m, AL_H));
        let qg = read(&mut worker, colour(tape, 2));
        pass &= check_stage("qg = n·Wqᵀ", &qg, &n, &w.wq, (m, 2 * AL_Q, AL_H));
        let q = read(&mut worker, colour(tape, 3));
        let gate = read(&mut worker, colour(tape, 11));
        let (want_q, want_gate) = gate_split_reference(&qg, (m, AL_Q), AL_HD);
        let split_bits = mismatched_bits(&q, &want_q) + mismatched_bits(&gate, &want_gate);
        eprintln!(
            "q, gate = GateSplit(qg) rows={m}: mismatched_bits={split_bits} {}",
            verdict(split_bits == 0)
        );
        pass &= split_bits == 0;
        let qn = read(&mut worker, colour(tape, 4));
        let qn_ref = rms_norm_reference(&q, &w.q_norm, (m * 8, AL_HD), f64::from(EPS), 1.0);
        pass &= check_rows("qn = RmsNorm(q, q_norm)", &qn, &qn_ref, (m * 8, AL_HD));
        // `k` (colour 5) holds `v` by the forward's end: the GEMM kernel is run alone on `n`.
        let k = k_direct(&device, tape, &n, &w.wk);
        pass &= check_stage("k = n·Wkᵀ", &k, &n, &w.wk, (m, AL_KV, AL_H));
        let kn = read(&mut worker, colour(tape, 6));
        let kn_ref = rms_norm_reference(&k, &w.k_norm, (m * 2, AL_HD), f64::from(EPS), 1.0);
        pass &= check_rows("kn = RmsNorm(k, k_norm)", &kn, &kn_ref, (m * 2, AL_HD));
        let v = read(&mut worker, colour(tape, 5));
        pass &= check_stage("v = n·Wvᵀ", &v, &n, &w.wv, (m, AL_KV, AL_H));
        let rotary = tape.rotary.expect("the tape has a rotary table");
        let uploaded = read(&mut worker, rotary.span);
        let table_ok = uploaded == rotary_rows(&positions, rotary.rot_dim, rotary.theta());
        let kr = read(&mut worker, colour(tape, 8));
        let refs = rope_reference(&kn, &uploaded, (m, AL_KV), (AL_HD, AL_R), AL_R / 2);
        let cmp = compare_rope(&kr, &kn, &refs, AL_KV);
        let ok = cmp.rotated.pass() && cmp.pass_through_mismatched == 0 && table_ok;
        eprintln!(
            "kr = RopeAppend's rope of kf rows={m}: {} pass_through_mismatched_bits={} \
             table_is_the_positions'={table_ok} {}",
            cmp.rotated.line(),
            cmp.pass_through_mismatched,
            verdict(ok)
        );
        pass &= ok;
        let kb = read_pool(&mut worker, KvHalf::K);
        let vb = read_pool(&mut worker, KvHalf::V);
        let cache_bits = cache_mismatches((&kb, &vb), (&kr, &v), block_table, (0, m));
        eprintln!(
            "cache[slots] = kr, v rows={m}: mismatched_bits={cache_bits} {}",
            verdict(cache_bits == 0)
        );
        pass &= cache_bits == 0;
        let attn = read(&mut worker, colour(tape, 9));
        let ga = read(&mut worker, colour(tape, 7));
        let ga_ref = gate_apply_reference(&attn, &gate);
        pass &= check_rows("ga = attn·sigmoid(gate)", &ga, &ga_ref, (m, AL_Q));
        let o = read(&mut worker, colour(tape, 1));
        pass &= check_stage("o = ga·Woᵀ", &o, &ga, &w.wo, (m, AL_H, AL_Q));
        let mismatched = mismatched_bits(&y, &add_reference(&o, &x));
        eprintln!(
            "y = o + x rows={m}: mismatched_bits={mismatched} {}",
            verdict(mismatched == 0)
        );
        pass &= mismatched == 0;
        // `qr` (colour 7) holds `ga` by the forward's end: the rope kernel is run alone on `qn`
        // and the uploaded table, and checked under the rope rule.
        let qr = qr_direct(&device, tape, &qn, &uploaded);
        let refs = rope_reference(&qn, &uploaded, (m, AL_Q), (AL_HD, AL_R), AL_R / 2);
        let cmp = compare_rope(&qr, &qn, &refs, AL_Q);
        let ok = cmp.rotated.pass() && cmp.pass_through_mismatched == 0;
        eprintln!(
            "qr = RopeRotate(qf) rows={m}: {} pass_through_mismatched_bits={} {}",
            cmp.rotated.line(),
            cmp.pass_through_mismatched,
            verdict(ok)
        );
        pass &= ok;
        let (k_rows, v_rows) = (gather(&kb, block_table, m), gather(&vb, block_table, m));
        pass &= check_attention(
            "attn = AttnPrefill(qr, cache)",
            &attn,
            &qr,
            (&k_rows, &v_rows),
            (0, m),
        );
        let CudaStep::AttnPrefill(step) = tape.steps[9] else {
            panic!("step 9 is the attention");
        };
        pass &= step.scale_bits == 0.0625f32.to_bits();
    }
    assert!(pass, "a stage of the attention layer is out of bound");
}

/// The tape's `k = n · Wkᵀ`, run by the GEMM kernel alone on device copies: the forward's `k` bit
/// for bit (the kernel is deterministic), which `v` overwrites in the arena.
fn k_direct(device: &CudaDevice, tape: &CudaTape, n: &[u16], wk: &[u16]) -> Vec<u16> {
    let CudaStep::Gemm(g) = tape.steps[4] else {
        panic!("step 4 is the K projection");
    };
    let (nb, wb) = (
        DeviceBuffer::upload(device, n).expect("upload n"),
        DeviceBuffer::upload(device, wk).expect("upload Wk"),
    );
    let out = DeviceBuffer::upload(device, &vec![0; (g.m * g.n) as usize]).expect("upload k");
    // SAFETY: cudaMalloc pointers are 256-byte aligned; `n` holds m·k, `Wk` n·k and `out` m·n
    // elements; `out` is its own allocation; null is the default stream.
    unsafe {
        gemm_bf16_abt(
            nb.ptr(),
            wb.ptr(),
            out.ptr(),
            g.m,
            g.n,
            g.k,
            std::ptr::null_mut(),
        )
    }
    .expect("gemm_bf16_abt launch");
    out.download().expect("download k")
}

/// The tape's rope of `qn`, run by the rope kernel alone on device copies, with the tape's table:
/// `qr`, which the gate apply overwrites in the arena before the forward ends. The kernel is
/// deterministic, so this is the forward's `qr` bit for bit.
fn qr_direct(device: &CudaDevice, tape: &CudaTape, qn: &[u16], table: &[u16]) -> Vec<u16> {
    let CudaStep::Rope(rope) = tape.steps[7] else {
        panic!("step 7 is the query's rope");
    };
    let (xb, tb) = (
        DeviceBuffer::upload(device, qn).expect("upload qn"),
        DeviceBuffer::upload(device, table).expect("upload the table"),
    );
    let out = DeviceBuffer::upload(device, &vec![0; qn.len()]).expect("upload qr");
    // SAFETY: cudaMalloc pointers are 256-byte aligned; `qn` and `out` hold rows·cols elements and
    // the table rows·rot_dim; `out` is its own allocation; null is the default stream.
    unsafe {
        rope_bf16(
            xb.ptr(),
            tb.ptr(),
            out.ptr(),
            (rope.rows, rope.cols),
            (rope.head_dim, rope.rot_dim),
            std::ptr::null_mut(),
        )
    }
    .expect("rope_bf16 launch");
    out.download().expect("download qr")
}

/// Forward A (P = 0, L = 256) then forward B on a new input (P = 256, L = 256) over the table
/// `[11, 3, 14, 0, 7, 9, 2, 5]`: B's attention against the f64 reference over 512 keys read back
/// from the cache, and A's four blocks bit-identical before and after B.
#[test]
fn a_second_chunk_reads_the_first_from_the_cache() {
    let device = h100();
    let w = layer_weights();
    let table = layer_table(&device, &w);
    let mut worker = CudaWorker::new(&device, &LAYER_BUCKETS, &table, POOL).expect("worker");
    let tape = &ATTN_LAYER_M256;
    let block_table = [11u32, 3, 14, 0, 7, 9, 2, 5];
    let mut y = vec![0u16; 256 * AL_H];
    let a = KvSequence {
        block_table: &block_table[..4],
        prefix_len: 0,
        q_len: 256,
    };
    worker
        .forward(
            1,
            &layer_input(256, 0xC4A),
            &seq_positions(256, 0),
            Some(&a),
            &mut y,
        )
        .expect("forward A");
    let (ka, va) = (
        read_pool(&mut worker, KvHalf::K),
        read_pool(&mut worker, KvHalf::V),
    );
    let b = KvSequence {
        block_table: &block_table,
        prefix_len: 256,
        q_len: 256,
    };
    let positions = seq_positions(256, 256);
    worker
        .forward(1, &layer_input(256, 0xC4B), &positions, Some(&b), &mut y)
        .expect("forward B");
    let (kb, vb) = (
        read_pool(&mut worker, KvHalf::K),
        read_pool(&mut worker, KvHalf::V),
    );
    let a_blocks_changed = (block_table[..4].iter())
        .map(|&blk| {
            let i = blk as usize;
            mismatched_bits(&ka[i], &kb[i]) + mismatched_bits(&va[i], &vb[i])
        })
        .sum::<usize>();
    eprintln!(
        "forward A's blocks after forward B: mismatched_bits={a_blocks_changed} {}",
        verdict(a_blocks_changed == 0)
    );
    let qn = read(&mut worker, colour(tape, 4));
    let rotary = tape.rotary.expect("the tape has a rotary table");
    let uploaded = read(&mut worker, rotary.span);
    let qr = qr_direct(&device, tape, &qn, &uploaded);
    let attn = read(&mut worker, colour(tape, 9));
    let (k, v) = (
        gather(&kb, &block_table, 512),
        gather(&vb, &block_table, 512),
    );
    let ok = check_attention(
        "attn of chunk B over 512 cached keys",
        &attn,
        &qr,
        (&k, &v),
        (256, 256),
    );
    assert_eq!(
        a_blocks_changed, 0,
        "forward B wrote over forward A's blocks"
    );
    assert!(ok, "chunk B's attention is out of bound");
}

/// M = 128 over `[11, 3, 14]`: forward 1 (P = 0, L = 128); forward 2, a new input at P = 0,
/// L = 37, whose padding rows write nothing (positions 37..127 keep forward 1's bytes); forward 3
/// (P = 37, L = 100), its real rows within bound against the cache, the rest of block 14
/// (positions 137..191) unchanged, its padding rows' `y` finite.
#[test]
fn a_short_chunk_pads() {
    let device = h100();
    let w = layer_weights();
    let table = layer_table(&device, &w);
    let mut worker = CudaWorker::new(&device, &LAYER_BUCKETS, &table, POOL).expect("worker");
    let tape = &ATTN_LAYER_M128;
    let block_table = [11u32, 3, 14];
    let mut y = vec![0u16; 128 * AL_H];
    let mut pass = true;
    let f1 = KvSequence {
        block_table: &block_table[..2],
        prefix_len: 0,
        q_len: 128,
    };
    worker
        .forward(
            0,
            &layer_input(128, 0x5401),
            &seq_positions(128, 0),
            Some(&f1),
            &mut y,
        )
        .expect("forward 1");
    let k1 = gather(&read_pool(&mut worker, KvHalf::K), &block_table, 192);
    let v1 = gather(&read_pool(&mut worker, KvHalf::V), &block_table, 192);

    let f2 = KvSequence {
        block_table: &block_table[..1],
        prefix_len: 0,
        q_len: 37,
    };
    worker
        .forward(
            0,
            &layer_input(128, 0x5402),
            &seq_positions(128, 0),
            Some(&f2),
            &mut y,
        )
        .expect("forward 2");
    let k2 = gather(&read_pool(&mut worker, KvHalf::K), &block_table, 192);
    let v2 = gather(&read_pool(&mut worker, KvHalf::V), &block_table, 192);
    let (lo, hi) = (37 * AL_KV, 128 * AL_KV);
    let kept =
        mismatched_bits(&k2[lo..hi], &k1[lo..hi]) + mismatched_bits(&v2[lo..hi], &v1[lo..hi]);
    eprintln!(
        "forward 2 (L = 37): positions 37..128 unchanged, mismatched_bits={kept} {}",
        verdict(kept == 0)
    );
    pass &= kept == 0;

    let f3 = KvSequence {
        block_table: &block_table,
        prefix_len: 37,
        q_len: 100,
    };
    let positions = seq_positions(128, 37);
    worker
        .forward(0, &layer_input(128, 0x5403), &positions, Some(&f3), &mut y)
        .expect("forward 3");
    let kb = read_pool(&mut worker, KvHalf::K);
    let vb = read_pool(&mut worker, KvHalf::V);
    let (k3, v3) = (
        gather(&kb, &block_table, 192),
        gather(&vb, &block_table, 192),
    );
    let (lo, hi) = (137 * AL_KV, 192 * AL_KV);
    let rest =
        mismatched_bits(&k3[lo..hi], &k2[lo..hi]) + mismatched_bits(&v3[lo..hi], &v2[lo..hi]);
    eprintln!(
        "forward 3 (P = 37, L = 100): positions 137..192 unchanged, mismatched_bits={rest} {}",
        verdict(rest == 0)
    );
    pass &= rest == 0;
    let qn = read(&mut worker, colour(tape, 4));
    let rotary = tape.rotary.expect("the tape has a rotary table");
    let uploaded = read(&mut worker, rotary.span);
    let qr = qr_direct(&device, tape, &qn, &uploaded);
    let attn = read(&mut worker, colour(tape, 9));
    pass &= check_attention(
        "attn of forward 3",
        &attn,
        &qr,
        (&k3[..137 * AL_KV], &v3[..137 * AL_KV]),
        (37, 100),
    );
    let padding_finite =
        (y[100 * AL_H..].iter()).all(|&b| half::bf16::from_bits(b).to_f32().is_finite());
    eprintln!(
        "forward 3 padding rows of y finite={padding_finite} {}",
        verdict(padding_finite)
    );
    pass &= padding_finite;
    assert!(pass, "a short chunk wrote padding slots or is out of bound");
}

/// 9 allocations at setup (7 weights, the arena, the pool), the arena 9733888 bytes and the pool
/// 2097152; 0 allocations over five forwards (256, 128, 256, 128, 256) with the same inputs and
/// sequence; free memory equal after forwards 1 and 5; every result bit-identical to its bucket's
/// first.
#[test]
fn the_attention_layer_worker_allocates_once_and_no_forward_allocates() {
    let device = h100();
    let w = layer_weights();
    let before = device.allocations();
    let table = layer_table(&device, &w);
    let mut worker = CudaWorker::new(&device, &LAYER_BUCKETS, &table, POOL).expect("worker");
    let built = device.allocations() - before;
    let arena = worker.arena_bytes();
    let pool = worker.kv_pool_bytes();
    eprintln!(
        "attention layer arena_bytes={arena} (expect 9733888) pool_bytes={pool:?} (expect \
         2097152) allocations(weights + arena + pool)={built} (expect 9) {}",
        verdict(arena == 9733888 && pool == Some(2097152) && built == 9)
    );
    assert_eq!(arena, 9733888, "the arena is the larger bucket's total");
    assert_eq!(
        pool,
        Some(2097152),
        "2 halves × 16 blocks × 2 heads × 64 × 256 bf16"
    );
    assert_eq!(built, 9, "seven weights, one arena, one pool");

    let inputs = [layer_input(128, 0xA110), layer_input(256, 0xA110)];
    let tables: [&[u32]; 2] = [&[11, 3], &[11, 3, 14, 0]];
    let mut firsts: [Option<Vec<u16>>; 2] = [None, None];
    let at_forwards = device.allocations();
    let mut free_after_first = None;
    let mut identical = true;
    for (i, bucket) in [1usize, 0, 1, 0, 1].into_iter().enumerate() {
        let m = LAYER_BUCKETS[bucket].bucket_m as usize;
        let seq = KvSequence {
            block_table: tables[bucket],
            prefix_len: 0,
            q_len: m as u32,
        };
        let mut y = vec![0u16; m * AL_H];
        worker
            .forward(
                bucket,
                &inputs[bucket],
                &seq_positions(m, 0),
                Some(&seq),
                &mut y,
            )
            .expect("forward");
        if i == 0 {
            free_after_first = Some(device.mem_info().expect("cudaMemGetInfo").0);
        }
        match &firsts[bucket] {
            Some(first) => identical &= *first == y,
            None => firsts[bucket] = Some(y),
        }
    }
    let free_after_first = free_after_first.expect("forward 1 ran");
    let free_after_last = device.mem_info().expect("cudaMemGetInfo").0;
    let during = device.allocations() - at_forwards;
    eprintln!(
        "attention layer five forwards (256, 128, 256, 128, 256): allocations={during} (expect 0) \
         free_after_forward_1={free_after_first} free_after_forward_5={free_after_last} \
         results_bit_identical={identical} {}",
        verdict(during == 0 && free_after_first == free_after_last && identical)
    );
    assert_eq!(during, 0, "a forward allocated device memory");
    assert_eq!(
        free_after_first, free_after_last,
        "device free memory moved"
    );
    assert!(identical, "a bucket's result changed between forwards");
}

/// Each of the seven KV `ForwardError`s, before any copy (the result untouched), and a good
/// forward after them still right; `KvUnexpected` on the attention-gate tape.
#[test]
fn a_bad_sequence_is_refused_before_any_copy() {
    let device = h100();
    let w = layer_weights();
    let table = layer_table(&device, &w);
    let mut worker = CudaWorker::new(&device, &LAYER_BUCKETS, &table, POOL).expect("worker");
    let x = layer_input(128, 0xBAD);
    let pos = seq_positions(128, 0);
    let good: &[u32] = &[11, 3];
    let seq = |block_table, prefix_len, q_len| KvSequence {
        block_table,
        prefix_len,
        q_len,
    };
    let long: Vec<u32> = (0..129).collect();
    let mut shifted = pos.clone();
    shifted[5] = 99;
    let cases: [(Option<KvSequence<'_>>, &[u32], ForwardError); 8] = [
        (None, &pos, ForwardError::KvMissing),
        (
            Some(seq(good, 0, 0)),
            &pos,
            ForwardError::KvQueryLength {
                q_len: 0,
                bucket_m: 128,
            },
        ),
        (
            Some(seq(good, 0, 129)),
            &pos,
            ForwardError::KvQueryLength {
                q_len: 129,
                bucket_m: 128,
            },
        ),
        (
            Some(seq(good, 8192, 1)),
            &pos,
            ForwardError::KvTooLong {
                tokens: 8193,
                max: 8192,
            },
        ),
        (
            Some(seq(&long, 0, 128)),
            &pos,
            ForwardError::KvBlockTable {
                got: 129,
                need: 2,
                max: 128,
            },
        ),
        (
            Some(seq(&[11, 16], 0, 128)),
            &pos,
            ForwardError::KvBlockOutsidePool {
                entry: 1,
                block: 16,
                num_blocks: 16,
            },
        ),
        (
            Some(seq(&[11, 11], 0, 128)),
            &pos,
            ForwardError::KvBlockRepeated {
                block: 11,
                first: 0,
                second: 1,
            },
        ),
        (
            Some(seq(good, 0, 128)),
            &shifted,
            ForwardError::PositionsDisagree {
                row: 5,
                got: 99,
                want: 5,
            },
        ),
    ];
    let mut pass = true;
    for (kv, positions, want) in &cases {
        let mut y = vec![0u16; 128 * AL_H];
        let got = worker.forward(0, &x, positions, kv.as_ref(), &mut y);
        let untouched = y.iter().all(|&b| b == 0);
        eprintln!(
            "refused {want:?}: got={got:?} result_untouched={untouched} {}",
            verdict(got.as_ref() == Err(want) && untouched)
        );
        pass &= got.as_ref() == Err(want) && untouched;
    }
    // Too few entries for the tokens.
    let mut y = vec![0u16; 128 * AL_H];
    let got = worker.forward(0, &x, &pos, Some(&seq(&[11], 0, 128)), &mut y);
    let want = Err(ForwardError::KvBlockTable {
        got: 1,
        need: 2,
        max: 128,
    });
    eprintln!(
        "refused a short block table: {got:?} {}",
        verdict(got == want)
    );
    pass &= got == want;
    // A good forward after them is the same as on a fresh worker.
    let mut after = vec![0u16; 128 * AL_H];
    worker
        .forward(0, &x, &pos, Some(&seq(good, 0, 128)), &mut after)
        .expect("a good forward after the refusals");
    let mut fresh_worker =
        CudaWorker::new(&device, &LAYER_BUCKETS, &table, POOL).expect("a fresh worker");
    let mut fresh = vec![0u16; 128 * AL_H];
    fresh_worker
        .forward(0, &x, &pos, Some(&seq(good, 0, 128)), &mut fresh)
        .expect("the fresh forward");
    let same = after == fresh;
    eprintln!(
        "a good forward after the refusals is the fresh one's: {same} {}",
        verdict(same)
    );
    pass &= same;
    drop(fresh_worker);
    drop(worker);

    // `KvUnexpected` on the attention-gate tape, which has no KV cache.
    let gw = attn_weights();
    let gt = attn_table(&device, &gw);
    let mut gate_worker = CudaWorker::new(&device, &ATTN_BUCKETS, &gt, None).expect("gate worker");
    let mut y = vec![0u16; 256 * AG_H];
    let got = gate_worker.forward(
        1,
        &attn_input(256),
        &attn_positions(256),
        Some(&seq(good, 0, 128)),
        &mut y,
    );
    eprintln!(
        "attention gate with a sequence: {got:?} {}",
        verdict(got == Err(ForwardError::KvUnexpected))
    );
    pass &= got == Err(ForwardError::KvUnexpected);
    assert!(pass, "a bad sequence was not refused before any copy");
}

/// A KV read of a layer index or a block outside the pool, or into a buffer of another size than
/// one block, is refused with nothing copied; so is any KV read on a worker with no pool.
#[test]
fn a_kv_read_outside_the_pool_is_refused() {
    let device = h100();
    let w = layer_weights();
    let table = layer_table(&device, &w);
    let mut worker = CudaWorker::new(&device, &LAYER_BUCKETS, &table, POOL).expect("worker");
    let mut pass = true;
    for (layer, block, len) in [
        (1usize, 0u32, BLOCK_ELEMS),
        (0, POOL_BLOCKS, BLOCK_ELEMS),
        (0, 0, BLOCK_ELEMS - 1),
    ] {
        let mut out = vec![SENTINEL_BF16; len];
        let got = worker.read_kv(layer, KvHalf::V, block, &mut out);
        let want = Err(ForwardError::KvRead {
            layer,
            block,
            expected: BLOCK_ELEMS,
            got: len,
        });
        let untouched = out.iter().all(|&b| b == SENTINEL_BF16);
        eprintln!(
            "read_kv(layer {layer}, block {block}, {len} elements): {got:?} \
             out_untouched={untouched} {}",
            verdict(got == want && untouched)
        );
        pass &= got == want && untouched;
    }
    drop(worker);
    let gw = attn_weights();
    let gt = attn_table(&device, &gw);
    let mut gate_worker = CudaWorker::new(&device, &ATTN_BUCKETS, &gt, None).expect("gate worker");
    let mut out = vec![0u16; BLOCK_ELEMS];
    let got = gate_worker.read_kv(0, KvHalf::K, 0, &mut out);
    let want = Err(ForwardError::KvRead {
        layer: 0,
        block: 0,
        expected: 0,
        got: BLOCK_ELEMS,
    });
    eprintln!(
        "read_kv on a worker with no pool: {got:?} {}",
        verdict(got == want)
    );
    pass &= got == want;
    assert!(pass, "a KV read outside the pool was not refused");
}

/// An element `read_kv` must not write.
const SENTINEL_BF16: u16 = 0xFFFF;

/// A worker's tapes, weights and pool.
type PoolCase<'w> = (
    &'static [CudaTape],
    &'w [DeviceBuffer],
    Option<KvPoolConfig>,
);

/// The worker's pool refusals, each before anything is allocated.
#[test]
fn the_kv_pool_is_given_exactly_when_a_tape_needs_it() {
    let device = h100();
    let w = layer_weights();
    let table = layer_table(&device, &w);
    let gw = attn_weights();
    let gt = attn_table(&device, &gw);
    let before = device.allocations();
    let cases: [(PoolCase<'_>, WorkerError); 4] = [
        ((&LAYER_BUCKETS, &table, None), WorkerError::KvPoolMissing),
        (
            (&LAYER_BUCKETS, &table, Some(KvPoolConfig { num_blocks: 0 })),
            WorkerError::KvPoolMissing,
        ),
        ((&ATTN_BUCKETS, &gt, POOL), WorkerError::KvPoolUnused),
        (
            (&OTHER_KV, &table, POOL),
            WorkerError::KvGeometry { bucket: 1 },
        ),
    ];
    let mut pass = true;
    for ((tapes, weights, pool), want) in cases {
        let got = CudaWorker::new(&device, tapes, weights, pool).map(|_| ());
        eprintln!(
            "{want:?}: got={got:?} {}",
            verdict(got == Err(want.clone()))
        );
        pass &= got == Err(want);
    }
    let allocated = device.allocations() - before;
    eprintln!(
        "allocations by the refused workers={allocated} (expect 0) {}",
        verdict(allocated == 0)
    );
    assert!(pass, "a pool refusal is wrong");
    assert_eq!(allocated, 0, "a refused worker allocated");
}

/// [`ATTN_LAYER_M128`] beside a copy of [`ATTN_LAYER_M256`] whose KV cache claims layer 5: one pool
/// cannot hold both.
static OTHER_KV: [CudaTape; 2] = [
    ATTN_LAYER_M128,
    CudaTape {
        kv: match ATTN_LAYER_M256.kv {
            Some(kv) => Some(KvTape { layers: &[5], ..kv }),
            None => None,
        },
        ..ATTN_LAYER_M256
    },
];
