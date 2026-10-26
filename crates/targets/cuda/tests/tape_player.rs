// SPDX-License-Identifier: Apache-2.0
//! The tape player on the H100: the GEMM-chain tapes of `common::chains` and the qwen3.5-0.8b MLP
//! block of `common::mlp` played through the arena and compared with an f64 CPU reference, one
//! arena allocation per worker, no allocation per forward, colour reuse, and the refusals; and the
//! attention layer's one-row (decode) tape over the cache its 128-row tape prefilled.
//!
//! Built only under `nvcc` (`required-features`); no GPU fails rather than skips. Each stage is
//! checked against its own GPU input (`common::numerics`' bounds), so the bound never compounds
//! along a chain.

// `pub`: each test binary uses a different part of `common`, and nothing in it is dead code.
pub mod common {
    pub mod attn_gate;
    pub mod attn_layer;
    pub mod chains;
    pub mod gdn_layer;
    pub mod gemma_attn;
    pub mod gemma_mlp;
    pub mod gemma_moe;
    pub mod gemma_rows;
    pub mod mlp;
    pub mod moe_block;
    pub mod numerics;
}

use common::attn_gate::{ATTN_GATE_M128, ATTN_GATE_M256, attn_gate};
use common::attn_layer::{ATTN_LAYER_M128, ATTN_LAYER_M256, attn_layer};
use common::chains::{
    HIDDEN, QWEN3_5_FACTS, QWEN3_5_GDN_FACTS, THREE_GEMM_M256, TWO_GEMM_M128, TWO_GEMM_M256,
    assert_tape_is_lowered, assert_tape_is_lowered_with, three_gemm, two_gemm,
};
use common::gdn_layer::{GDN_LAYER_M128, GDN_LAYER_M256, gdn_layer};
use common::mlp::{EPS, MLP_M128, MLP_M256, mlp};
use common::numerics::{
    ATTN_SCORE_MAX, AttnCase, add_reference, attention_reference, bf16_inputs, centred_gains,
    compare, compare_attention, compare_rope, compare_rows, gate_apply_reference,
    gate_split_reference, gather_paged, mismatched_bits, rms_norm_reference, rope_reference,
    silu_mul_reference,
};
use common::numerics::{
    GDN_SCAN_STEP, GdnDims, ScanInputs, ScanReference, ScanVariant, compare_f32, f32_from_halves,
    f32_inputs, f32_to_halves, gdn_conv_bound, gdn_conv_reference, gdn_gating_bound,
    gdn_gating_reference, gdn_norm_reference, gdn_scan_bound, gdn_scan_reference, widen,
};
use scratchy_subtile::gdn_state_layout::GdnStateLayout;
use scratchy_subtile::subtile_ir::GainConvention;
use scratchy_target_cuda::kernels::{gemm_bf16_abt, rms_norm_bf16, rope_bf16};
use scratchy_target_cuda::kernels::{
    layer_scale_bf16, rms_norm_unit_bf16, scale_bf16, soft_cap_bf16,
};
use scratchy_target_cuda::rotary::rotary_rows;
use scratchy_target_cuda::runtime::{
    CudaDevice, CudaWorker, DeviceBuffer, ForwardError, GdnPoolConfig, GdnSequence, Input, KvHalf,
    KvPoolConfig, KvSequence, WorkerError,
};
use scratchy_target_cuda::step::MatmulKernel;
use scratchy_target_cuda::step::{CudaStep, gdn_bundle, gdn_scratch};
use scratchy_target_cuda::tape::{
    ArenaLayout, ArenaOffset, ArenaSpan, CudaTape, GdnTape, KvTape, RotaryTable, TapeError,
    TapeInput, WeightIx,
};
use scratchy_target_cuda_compiler::canonical::{CanonicalAt, lower_canonical};
use scratchy_target_cuda_compiler::emit::static_tape;

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
    let mut worker =
        CudaWorker::new(&device, &TWO_GEMM_BUCKETS, &table, None, None).expect("worker");
    let mut pass = true;
    for (bucket, tape) in TWO_GEMM_BUCKETS.iter().enumerate() {
        let m = tape.bucket_m as usize;
        let x = input(m);
        let mut y2 = vec![0u16; m * H];
        worker
            .forward(bucket, Input::Hidden(&x), &[], None, None, &mut y2)
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
    let mut worker =
        CudaWorker::new(&device, &TWO_GEMM_BUCKETS, &table, None, None).expect("worker");
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
            .forward(
                bucket,
                Input::Hidden(&inputs[bucket]),
                &[],
                None,
                None,
                &mut y2,
            )
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
    let mut worker =
        CudaWorker::new(&device, two, &table[..3], None, None).expect("two-GEMM worker");
    let mut two_y2 = vec![0u16; 256 * H];
    worker
        .forward(0, Input::Hidden(&x), &[], None, None, &mut two_y2)
        .expect("two-GEMM forward");
    drop(worker);

    let three = std::slice::from_ref(&THREE_GEMM_M256);
    let mut worker =
        CudaWorker::new(&device, three, &table, None, None).expect("three-GEMM worker");
    let mut y3 = vec![0u16; 256 * WIDE];
    worker
        .forward(0, Input::Hidden(&x), &[], None, None, &mut y3)
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

/// A short input, and an input of the wrong kind, are refused before anything is enqueued, and
/// the worker still plays after them; a short weight is refused at construction.
#[test]
fn short_inputs_and_short_weights_are_refused() {
    let device = h100();
    let w = weights();
    let table = [
        DeviceBuffer::empty(),
        DeviceBuffer::upload(&device, &w.w1).expect("upload W1"),
        DeviceBuffer::upload(&device, &w.w2).expect("upload W2"),
    ];
    let mut worker =
        CudaWorker::new(&device, &TWO_GEMM_BUCKETS, &table, None, None).expect("worker");
    let x = input(256);
    let mut y2 = vec![0u16; 256 * H];
    let got = worker.forward(
        1,
        Input::Hidden(&x[..x.len() - 1]),
        &[],
        None,
        None,
        &mut y2,
    );
    let want = Err(ForwardError::InputSize {
        expected: 262144,
        got: 262142,
    });
    eprintln!("input one element short: {got:?} {}", verdict(got == want));
    assert_eq!(got, want);
    // Token ids for a tape whose input is the embedded hidden: refused before any copy.
    let ids = vec![0u32; 256];
    let got = worker.forward(1, Input::TokenIds(&ids), &[], None, None, &mut y2);
    let want = Err(ForwardError::InputKind {
        expected: "hidden rows",
        got: "token ids",
    });
    eprintln!(
        "token ids for a hidden tape: {got:?} {}",
        verdict(got == want)
    );
    assert_eq!(got, want);
    worker
        .forward(1, Input::Hidden(&x), &[], None, None, &mut y2)
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
    let got = CudaWorker::new(&device, &TWO_GEMM_BUCKETS, &short, None, None).map(|_| ());
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
    let mut worker = CudaWorker::new(&device, &MLP_BUCKETS, &table, None, None).expect("worker");
    let mut pass = true;
    for (bucket, tape) in MLP_BUCKETS.iter().enumerate() {
        let m = tape.bucket_m as usize;
        let x = mlp_input(m);
        let mut y = vec![0u16; m * MLP_H];
        worker
            .forward(bucket, Input::Hidden(&x), &[], None, None, &mut y)
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
    let mut worker = CudaWorker::new(&device, &MLP_BUCKETS, &table, None, None).expect("worker");
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
            .forward(
                bucket,
                Input::Hidden(&inputs[bucket]),
                &[],
                None,
                None,
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
    let got = CudaWorker::new(&device, &BAD_WEIGHT, &table, None, None).map(|_| ());
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
    let mut worker = CudaWorker::new(&device, &ATTN_BUCKETS, &table, None, None).expect("worker");
    let mut pass = true;
    for (bucket, tape) in ATTN_BUCKETS.iter().enumerate() {
        let m = tape.bucket_m as usize;
        let x = attn_input(m);
        let positions = attn_positions(m);
        let mut y = vec![0u16; m * AG_H];
        worker
            .forward(bucket, Input::Hidden(&x), &positions, None, None, &mut y)
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
    let mut worker = CudaWorker::new(&device, &ATTN_BUCKETS, &table, None, None).expect("worker");
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
            .forward(
                bucket,
                Input::Hidden(&inputs[bucket]),
                &positions[bucket],
                None,
                None,
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
    let mut worker = CudaWorker::new(&device, &ATTN_BUCKETS, &table, None, None).expect("worker");
    let x = attn_input(256);
    let mut y = vec![0u16; 256 * AG_H];
    let got = worker.forward(
        1,
        Input::Hidden(&x),
        &attn_positions(255),
        None,
        None,
        &mut y,
    );
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
    let mut worker =
        CudaWorker::new(&device, &MLP_BUCKETS, &table, None, None).expect("MLP worker");
    let got = worker.forward(
        1,
        Input::Hidden(&mlp_input(256)),
        &[7],
        None,
        None,
        &mut vec![0u16; 256 * MLP_H],
    );
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
        freq_dim: 64,
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
    let got = CudaWorker::new(&device, &BAD_ROTARY, &table, None, None).map(|_| ());
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
        window: None,
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
    let mut worker = CudaWorker::new(&device, &LAYER_BUCKETS, &table, POOL, None).expect("worker");
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
            .forward(
                bucket,
                Input::Hidden(&x),
                &positions,
                Some(&seq),
                None,
                &mut y,
            )
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
    let mut worker = CudaWorker::new(&device, &LAYER_BUCKETS, &table, POOL, None).expect("worker");
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
            Input::Hidden(&layer_input(256, 0xC4A)),
            &seq_positions(256, 0),
            Some(&a),
            None,
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
        .forward(
            1,
            Input::Hidden(&layer_input(256, 0xC4B)),
            &positions,
            Some(&b),
            None,
            &mut y,
        )
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
    let mut worker = CudaWorker::new(&device, &LAYER_BUCKETS, &table, POOL, None).expect("worker");
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
            Input::Hidden(&layer_input(128, 0x5401)),
            &seq_positions(128, 0),
            Some(&f1),
            None,
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
            Input::Hidden(&layer_input(128, 0x5402)),
            &seq_positions(128, 0),
            Some(&f2),
            None,
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
        .forward(
            0,
            Input::Hidden(&layer_input(128, 0x5403)),
            &positions,
            Some(&f3),
            None,
            &mut y,
        )
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
    let mut worker = CudaWorker::new(&device, &LAYER_BUCKETS, &table, POOL, None).expect("worker");
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
                Input::Hidden(&inputs[bucket]),
                &seq_positions(m, 0),
                Some(&seq),
                None,
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
    let mut worker = CudaWorker::new(&device, &LAYER_BUCKETS, &table, POOL, None).expect("worker");
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
        let got = worker.forward(0, Input::Hidden(&x), positions, kv.as_ref(), None, &mut y);
        let untouched = y.iter().all(|&b| b == 0);
        eprintln!(
            "refused {want:?}: got={got:?} result_untouched={untouched} {}",
            verdict(got.as_ref() == Err(want) && untouched)
        );
        pass &= got.as_ref() == Err(want) && untouched;
    }
    // Too few entries for the tokens.
    let mut y = vec![0u16; 128 * AL_H];
    let got = worker.forward(
        0,
        Input::Hidden(&x),
        &pos,
        Some(&seq(&[11], 0, 128)),
        None,
        &mut y,
    );
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
        .forward(
            0,
            Input::Hidden(&x),
            &pos,
            Some(&seq(good, 0, 128)),
            None,
            &mut after,
        )
        .expect("a good forward after the refusals");
    let mut fresh_worker =
        CudaWorker::new(&device, &LAYER_BUCKETS, &table, POOL, None).expect("a fresh worker");
    let mut fresh = vec![0u16; 128 * AL_H];
    fresh_worker
        .forward(
            0,
            Input::Hidden(&x),
            &pos,
            Some(&seq(good, 0, 128)),
            None,
            &mut fresh,
        )
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
    let mut gate_worker =
        CudaWorker::new(&device, &ATTN_BUCKETS, &gt, None, None).expect("gate worker");
    let mut y = vec![0u16; 256 * AG_H];
    let got = gate_worker.forward(
        1,
        Input::Hidden(&attn_input(256)),
        &attn_positions(256),
        Some(&seq(good, 0, 128)),
        None,
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
    let mut worker = CudaWorker::new(&device, &LAYER_BUCKETS, &table, POOL, None).expect("worker");
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
    let mut gate_worker =
        CudaWorker::new(&device, &ATTN_BUCKETS, &gt, None, None).expect("gate worker");
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
        let got = CudaWorker::new(&device, tapes, weights, pool, None).map(|_| ());
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

// ── qwen3.5-0.8b's linear-attention (GDN) layer ────────────────────

/// One worker's buckets: `M = 128` and `M = 256` of the GDN layer.
static GDN_BUCKETS: [CudaTape; 2] = [GDN_LAYER_M128, GDN_LAYER_M256];

const GL_H: usize = common::gdn_layer::H as usize;
const GL_NV: usize = common::gdn_layer::NV as usize;
const GL_VALUE: usize = common::gdn_layer::VALUE_DIM as usize;
const GL_CONV: usize = common::gdn_layer::CONV_DIM as usize;
/// The GDN pool's slots in every GDN-layer test.
const GDN_SLOTS: u32 = 4;
const GDN_POOL: Option<GdnPoolConfig> = Some(GdnPoolConfig {
    num_slots: GDN_SLOTS,
});
const G08: GdnDims = GdnDims {
    nk: 16,
    nv: 16,
    head_k: 128,
    head_v: 128,
    conv_kernel: 4,
};

#[test]
fn the_played_gdn_tapes_are_the_lowered_ones() {
    assert_tape_is_lowered_with(
        "gdn_layer(128)",
        &GDN_LAYER_M128,
        &gdn_layer(128),
        QWEN3_5_GDN_FACTS,
    );
    assert_tape_is_lowered_with(
        "gdn_layer(256)",
        &GDN_LAYER_M256,
        &gdn_layer(256),
        QWEN3_5_GDN_FACTS,
    );
}

/// The layer's weights: `g_in [H]` zero-centred, `Wqkv [CONV, H]`, `Wz [VALUE, H]`, `Wa` and `Wb
/// [NV, H]`, `Wo [H, VALUE]` (bf16); the packed f32 bundle's four parts.
struct GdnWeights {
    g_in: Vec<u16>,
    wqkv: Vec<u16>,
    wz: Vec<u16>,
    wa: Vec<u16>,
    wb: Vec<u16>,
    wo: Vec<u16>,
    conv1d: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    norm: Vec<f32>,
}

/// `len` bf16 in `[-scale, scale]`.
fn scaled(len: usize, seed: u64, scale: f32) -> Vec<u16> {
    bf16_inputs(len, seed)
        .into_iter()
        .map(|b| common::numerics::bits(half::bf16::from_bits(b).to_f32() * scale))
        .collect()
}

fn gdn_weights() -> GdnWeights {
    // Scaled so the projections of `n` (whose rows have unit RMS) land in about [-1, 1] for qkv
    // and z and [-2, 2] for a and b: the issue's test ranges.
    GdnWeights {
        g_in: centred_gains(GL_H, 0x6DA0_0001),
        wqkv: scaled(GL_CONV * GL_H, 0x6DA0_0002, 1.0 / 32.0),
        wz: scaled(GL_VALUE * GL_H, 0x6DA0_0003, 1.0 / 32.0),
        wa: scaled(GL_NV * GL_H, 0x6DA0_0004, 1.0 / 16.0),
        wb: scaled(GL_NV * GL_H, 0x6DA0_0005, 1.0 / 16.0),
        wo: scaled(GL_H * GL_VALUE, 0x6DA0_0006, 1.0 / 32.0),
        conv1d: f32_inputs(GL_CONV * 4, 0x6DA0_0007, (-0.5, 0.5)),
        a_log: f32_inputs(GL_NV, 0x6DA0_0008, (-1.0, 0.5)),
        dt_bias: f32_inputs(GL_NV, 0x6DA0_0009, (-0.5, 0.5)),
        norm: f32_inputs(128, 0x6DA0_000A, (0.5, 1.5)),
    }
}

/// The bundle packed as `gdn_bundle` lays it out, as the u16 pairs `DeviceBuffer` uploads.
fn pack_bundle(w: &GdnWeights) -> Vec<u16> {
    let layout = gdn_bundle(GL_CONV as u32, 4, GL_NV as u32, 128).expect("the bundle fits");
    let mut f = vec![0f32; layout.bytes as usize / 4];
    for (at, part) in [
        (layout.conv1d, &w.conv1d),
        (layout.a_log, &w.a_log),
        (layout.dt_bias, &w.dt_bias),
        (layout.norm, &w.norm),
    ] {
        f[at as usize / 4..][..part.len()].copy_from_slice(part);
    }
    f32_to_halves(&f)
}

/// The weight table, by source index (the fixture's): 0 is the embedded hidden.
fn gdn_table(device: &CudaDevice, w: &GdnWeights) -> Vec<DeviceBuffer> {
    let up = |b: &[u16]| DeviceBuffer::upload(device, b).expect("upload a weight");
    vec![
        DeviceBuffer::empty(),
        up(&w.g_in),
        up(&w.wqkv),
        up(&w.wz),
        up(&w.wa),
        up(&w.wb),
        up(&pack_bundle(w)),
        up(&w.wo),
    ]
}

/// The embedded hidden of an `m`-row GDN bucket, under `seed`.
fn gdn_input(m: usize, seed: u64) -> Vec<u16> {
    bf16_inputs(m * GL_H, seed ^ m as u64)
}

/// The GDN scratch's four regions after a forward, f32, read through `read_arena`.
struct Scratch {
    conv_out: Vec<f32>,
    g: Vec<f32>,
    beta: Vec<f32>,
    o: Vec<f32>,
}

fn read_scratch(worker: &mut CudaWorker<'_>, tape: &CudaTape) -> Scratch {
    let gdn = tape.gdn.expect("a GDN tape");
    let m = tape.bucket_m;
    let s = gdn_scratch(m, GL_CONV as u32, GL_NV as u32, GL_VALUE as u32).expect("fits");
    let base = gdn.scratch.offset.0;
    let mut part = |at: u64, elems: usize| {
        let span = ArenaSpan {
            offset: ArenaOffset(base + at),
            bytes: (elems * 4) as u64,
        };
        f32_from_halves(&read(worker, span))
    };
    let m = m as usize;
    Scratch {
        conv_out: part(s.conv_out, m * GL_CONV),
        g: part(s.g, m * GL_NV),
        beta: part(s.beta, m * GL_NV),
        o: part(s.o, m * GL_VALUE),
    }
}

/// Layer 0's slot `slot`: `(conv ring, ssm state)`.
fn read_gdn(worker: &mut CudaWorker<'_>, slot: u32) -> (Vec<f32>, Vec<f32>) {
    let l = GdnStateLayout::new(GDN_SLOTS, 16, 16, 128, 128, 4);
    let mut conv = vec![0f32; l.conv_elems_per_slot()];
    let mut ssm = vec![0f32; l.ssm_elems_per_slot()];
    worker
        .read_gdn_state(0, slot, &mut conv, &mut ssm)
        .expect("read_gdn_state");
    (conv, ssm)
}

/// Every slot's state, as bits.
fn gdn_snapshot(worker: &mut CudaWorker<'_>) -> Vec<Vec<u32>> {
    (0..GDN_SLOTS)
        .map(|s| {
            let (c, m) = read_gdn(worker, s);
            c.iter().chain(&m).map(|v| v.to_bits()).collect()
        })
        .collect()
}

fn to_f64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

/// Where a forward's scan starts: the state and running `Smax` before this chunk (`None` for a
/// fresh start, from zero), and its first token within the sequence.
struct ScanStart<'a> {
    state: Option<(&'a [f64], &'a [f64])>,
    t0: usize,
}

/// One forward's checks, every stage from its own GPU input: the norm (run alone, since `o`
/// overwrites `n`), the four projections, conv_out (from the ring `ring`, zeros when fresh), g and
/// β, o (from `start`), core, the output projection and the add. Rows past `q_len` of `core` must
/// be exactly `+0.0`. Returns whether all passed and the scan's reference, for a continuation.
fn check_gdn_forward(
    what: &str,
    (device, worker): (&CudaDevice, &mut CudaWorker<'_>),
    (tape, w): (&CudaTape, &GdnWeights),
    (x, y): (&[u16], &[u16]),
    (q_len, ring): (usize, &[f64]),
    start: &ScanStart<'_>,
) -> (bool, ScanReference) {
    let m = tape.bucket_m as usize;
    let mut pass = true;
    let n = n_direct(device, tape, x, &w.g_in);
    let n_ref = rms_norm_reference(x, &w.g_in, (m, GL_H), f64::from(EPS), 1.0);
    pass &= check_rows(
        &format!("{what} n = RmsNorm(x, g_in)"),
        &n,
        &n_ref,
        (m, GL_H),
    );
    let qkv = read(worker, colour(tape, 2));
    pass &= check_stage(
        &format!("{what} qkv = n·Wqkvᵀ"),
        &qkv,
        &n,
        &w.wqkv,
        (m, GL_CONV, GL_H),
    );
    let z = read(worker, colour(tape, 3));
    pass &= check_stage(
        &format!("{what} z = n·Wzᵀ"),
        &z,
        &n,
        &w.wz,
        (m, GL_VALUE, GL_H),
    );
    let a = read(worker, colour(tape, 4));
    pass &= check_stage(
        &format!("{what} a = n·Waᵀ (narrow)"),
        &a,
        &n,
        &w.wa,
        (m, GL_NV, GL_H),
    );
    let b = read(worker, colour(tape, 5));
    pass &= check_stage(
        &format!("{what} b = n·Wbᵀ (narrow)"),
        &b,
        &n,
        &w.wb,
        (m, GL_NV, GL_H),
    );
    let s = read_scratch(worker, tape);
    // conv_out from the real rows of qkv after the ring.
    let (refs, _) = gdn_conv_reference(
        &widen(&qkv[..q_len * GL_CONV]),
        &to_f64(&w.conv1d),
        ring,
        (q_len, GL_CONV, 4),
        false,
    );
    let r: Vec<f64> = refs.iter().map(|p| p.0).collect();
    let n_conv = q_len * GL_CONV;
    let cmp = compare_f32(
        &s.conv_out[..n_conv],
        &r,
        |i| gdn_conv_bound(refs[i].0, refs[i].1),
        |_| true,
    );
    eprintln!(
        "{what} conv_out L={q_len}: {} {}",
        cmp.line(),
        verdict(cmp.pass())
    );
    pass &= cmp.pass();
    let nh = q_len * GL_NV;
    let (g_ref, beta_ref) = gdn_gating_reference(
        &widen(&a[..nh]),
        &widen(&b[..nh]),
        &to_f64(&w.a_log),
        &to_f64(&w.dt_bias),
        GL_NV,
        true,
    );
    let gc = compare_f32(&s.g[..nh], &g_ref, |i| gdn_gating_bound(g_ref[i]), |_| true);
    let bc = compare_f32(
        &s.beta[..nh],
        &beta_ref,
        |i| gdn_gating_bound(beta_ref[i]),
        |_| true,
    );
    eprintln!(
        "{what} g, beta L={q_len}: g {} beta {} {}",
        gc.line(),
        bc.line(),
        verdict(gc.pass() && bc.pass())
    );
    pass &= gc.pass() && bc.pass();
    let zero_state = vec![0.0; GL_VALUE * 128];
    let (state, smax) = match start.state {
        Some((st, sm)) => (st, Some(sm)),
        None => (&zero_state[..], None),
    };
    let conv_out = to_f64(&s.conv_out[..n_conv]);
    let (gg, bb) = (to_f64(&s.g[..nh]), to_f64(&s.beta[..nh]));
    let scan = gdn_scan_reference(
        &ScanInputs {
            conv_out: &conv_out,
            g: &gg,
            beta: &bb,
            state,
            smax,
            t0: start.t0,
            q_len,
        },
        G08,
        ScanVariant::Real,
    );
    let o_ref: Vec<f64> = scan.o.iter().map(|p| p.0).collect();
    let oc = compare_f32(
        &s.o[..q_len * GL_VALUE],
        &o_ref,
        |i| gdn_scan_bound(scan.o[i].0, scan.o[i].1),
        |_| true,
    );
    eprintln!(
        "{what} o (scan) L={q_len} t0={}: {} {}",
        start.t0,
        oc.line(),
        verdict(oc.pass())
    );
    pass &= oc.pass();
    let core = read(worker, colour(tape, 6));
    let nv_rows = q_len * GL_VALUE;
    let core_ref = gdn_norm_reference(
        &to_f64(&s.o[..nv_rows]),
        &widen(&z[..nv_rows]),
        &to_f64(&w.norm),
        (q_len, GL_NV, 128),
        f64::from(EPS),
        true,
    );
    let cc = compare_rows(&core[..nv_rows], &core_ref, GL_VALUE);
    let not_zero = core[nv_rows..].iter().filter(|&&v| v != 0).count();
    eprintln!(
        "{what} core (gated norm) L={q_len}: {} padding_rows_not_plus_zero={not_zero} {}",
        cc.line(),
        verdict(cc.pass() && not_zero == 0)
    );
    pass &= cc.pass() && not_zero == 0;
    let o = read(worker, colour(tape, 1));
    pass &= check_stage(
        &format!("{what} o = core·Woᵀ"),
        &o,
        &core,
        &w.wo,
        (m, GL_H, GL_VALUE),
    );
    let mismatched = mismatched_bits(y, &add_reference(&o, x));
    eprintln!(
        "{what} y = o + x: mismatched_bits={mismatched} {}",
        verdict(mismatched == 0)
    );
    pass &= mismatched == 0;
    (pass, scan)
}

/// A fresh sequence's conv ring: zeros.
static FRESH_RING: [f64; GL_CONV * 3] = [0.0; GL_CONV * 3];

/// Both buckets, `q_len = M`, fresh in slot 1: every stage within bound of its own GPU input.
#[test]
fn the_gdn_layer_matches_the_cpu_reference_in_each_bucket() {
    let device = h100();
    let w = gdn_weights();
    let table = gdn_table(&device, &w);
    let mut worker =
        CudaWorker::new(&device, &GDN_BUCKETS, &table, None, GDN_POOL).expect("worker");
    let mut pass = true;
    for (bucket, tape) in GDN_BUCKETS.iter().enumerate() {
        let m = tape.bucket_m as usize;
        let x = gdn_input(m, 0x6D1);
        let seq = GdnSequence {
            slot: 1,
            q_len: m as u32,
            fresh: true,
        };
        let mut y = vec![0u16; m * GL_H];
        worker
            .forward(bucket, Input::Hidden(&x), &[], None, Some(&seq), &mut y)
            .expect("forward");
        let start = ScanStart { state: None, t0: 0 };
        let (ok, _) = check_gdn_forward(
            &format!("gdn M={m}"),
            (&device, &mut worker),
            (tape, &w),
            (&x, &y),
            (m, &FRESH_RING),
            &start,
        );
        pass &= ok;
    }
    assert!(pass, "a stage of the GDN layer is out of bound");
}

/// Bucket 256: chunk 1 (`q_len = 200`, fresh, slot 2), then chunk 2 (`q_len = 150`, continued).
/// Chunk 2's stages within bound, its scan against the reference from the state chunk 1 left; the
/// state after chunk 2 within the long-recurrence bound of a reference over all 350 tokens;
/// chunk 2's padding rows of `core` `+0.0`; slots 0, 1 and 3 bit-unchanged.
#[test]
fn the_gdn_layer_carries_its_state_across_two_chunks() {
    let device = h100();
    let w = gdn_weights();
    let table = gdn_table(&device, &w);
    let mut worker =
        CudaWorker::new(&device, &GDN_BUCKETS, &table, None, GDN_POOL).expect("worker");
    let tape = &GDN_LAYER_M256;
    let before = gdn_snapshot(&mut worker);
    let mut y = vec![0u16; 256 * GL_H];
    let x1 = gdn_input(256, 0xC41);
    let s1 = GdnSequence {
        slot: 2,
        q_len: 200,
        fresh: true,
    };
    worker
        .forward(1, Input::Hidden(&x1), &[], None, Some(&s1), &mut y)
        .expect("chunk 1");
    let fresh = ScanStart { state: None, t0: 0 };
    let (ok1, scan1) = check_gdn_forward(
        "gdn chunk 1",
        (&device, &mut worker),
        (tape, &w),
        (&x1, &y),
        (200, &FRESH_RING),
        &fresh,
    );
    let chunk1_scratch = read_scratch(&mut worker, tape);
    let (ring1, state1) = read_gdn(&mut worker, 2);
    let x2 = gdn_input(256, 0xC42);
    let s2 = GdnSequence {
        slot: 2,
        q_len: 150,
        fresh: false,
    };
    worker
        .forward(1, Input::Hidden(&x2), &[], None, Some(&s2), &mut y)
        .expect("chunk 2");
    let (st1, sm1) = (to_f64(&state1), scan1.smax.clone());
    let cont = ScanStart {
        state: Some((&st1, &sm1)),
        t0: 200,
    };
    let (ok2, _) = check_gdn_forward(
        "gdn chunk 2",
        (&device, &mut worker),
        (tape, &w),
        (&x2, &y),
        (150, &to_f64(&ring1)),
        &cont,
    );
    // The state after chunk 2 against one reference over all 350 tokens, from both chunks' GPU
    // conv_out, g and β.
    let chunk2_scratch = read_scratch(&mut worker, tape);
    let cat = |a: &[f32], b: &[f32], cols: usize, (l1, l2): (usize, usize)| {
        let mut v = to_f64(&a[..l1 * cols]);
        v.extend(to_f64(&b[..l2 * cols]));
        v
    };
    let lens = (200, 150);
    let conv_out = cat(
        &chunk1_scratch.conv_out,
        &chunk2_scratch.conv_out,
        GL_CONV,
        lens,
    );
    let g = cat(&chunk1_scratch.g, &chunk2_scratch.g, GL_NV, lens);
    let beta = cat(&chunk1_scratch.beta, &chunk2_scratch.beta, GL_NV, lens);
    let zero = vec![0.0; GL_VALUE * 128];
    let whole = gdn_scan_reference(
        &ScanInputs {
            conv_out: &conv_out,
            g: &g,
            beta: &beta,
            state: &zero,
            smax: None,
            t0: 0,
            q_len: 350,
        },
        G08,
        ScanVariant::Real,
    );
    let (_, state2) = read_gdn(&mut worker, 2);
    let sc = compare_f32(
        &state2,
        &whole.state,
        |i| GDN_SCAN_STEP * whole.state_scale[i / 128],
        |_| true,
    );
    let after = gdn_snapshot(&mut worker);
    let others: usize = [0usize, 1, 3]
        .iter()
        .map(|&s| {
            (before[s].iter().zip(&after[s]))
                .filter(|(a, b)| a != b)
                .count()
        })
        .sum();
    eprintln!(
        "gdn state after 350 tokens vs one reference: {} other_slots_mismatched_bits={others} {}",
        sc.line(),
        verdict(sc.pass() && others == 0)
    );
    assert!(ok1 && ok2, "a chunk's stage is out of bound");
    assert!(sc.pass(), "the state after two chunks is out of bound");
    assert_eq!(others, 0, "a chunk touched another slot");
}

/// Sequence A (two chunks in slot 2), then sequence B fresh in slot 2: B's result and state
/// bit-identical to B fresh in slot 0 of the same worker, and to B on a second worker.
#[test]
fn the_gdn_layer_does_not_leak_state_into_a_fresh_sequence() {
    let device = h100();
    let w = gdn_weights();
    let table = gdn_table(&device, &w);
    let mut worker =
        CudaWorker::new(&device, &GDN_BUCKETS, &table, None, GDN_POOL).expect("worker");
    let run = |worker: &mut CudaWorker<'_>, x: &[u16], seq: GdnSequence| {
        let mut y = vec![0u16; 256 * GL_H];
        worker
            .forward(1, Input::Hidden(x), &[], None, Some(&seq), &mut y)
            .expect("forward");
        y
    };
    let seq = |slot, q_len, fresh| GdnSequence { slot, q_len, fresh };
    run(&mut worker, &gdn_input(256, 0xA01), seq(2, 256, true));
    run(&mut worker, &gdn_input(256, 0xA02), seq(2, 100, false));
    let xb = gdn_input(256, 0xB01);
    let in_2 = run(&mut worker, &xb, seq(2, 180, true));
    let state_2 = read_gdn(&mut worker, 2);
    let in_0 = run(&mut worker, &xb, seq(0, 180, true));
    let state_0 = read_gdn(&mut worker, 0);
    let mut other = CudaWorker::new(&device, &GDN_BUCKETS, &table, None, GDN_POOL).expect("other");
    let on_other = run(&mut other, &xb, seq(3, 180, true));
    let state_other = read_gdn(&mut other, 3);
    let bits = |a: &(Vec<f32>, Vec<f32>), b: &(Vec<f32>, Vec<f32>)| {
        (a.0.iter().chain(&a.1).zip(b.0.iter().chain(&b.1)))
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count()
    };
    let (y0, s0) = (mismatched_bits(&in_2, &in_0), bits(&state_2, &state_0));
    let (y1, s1) = (
        mismatched_bits(&in_2, &on_other),
        bits(&state_2, &state_other),
    );
    eprintln!(
        "B fresh in slot 2 after A vs slot 0: y_mismatched={y0} state_mismatched_bits={s0}; vs a \
         second worker: y_mismatched={y1} state_mismatched_bits={s1} {}",
        verdict(y0 + s0 + y1 + s1 == 0)
    );
    assert_eq!(
        y0 + s0 + y1 + s1,
        0,
        "a fresh sequence saw a previous one's state"
    );
}

/// Arena 15253760 and pool 4489216 bytes; 9 allocations (7 weights, the arena, the pool), then 0
/// over five forwards (256, 128, 256, 128, 256; fresh and continued); free memory equal after
/// forwards 1 and 5; repeated identical fresh forwards bit-identical.
#[test]
fn the_gdn_layer_worker_allocates_once_and_no_forward_allocates() {
    let device = h100();
    let w = gdn_weights();
    let before = device.allocations();
    let table = gdn_table(&device, &w);
    let mut worker =
        CudaWorker::new(&device, &GDN_BUCKETS, &table, None, GDN_POOL).expect("worker");
    let built = device.allocations() - before;
    let arena = worker.arena_bytes();
    let pool = worker.gdn_pool_bytes();
    eprintln!(
        "gdn layer arena_bytes={arena} (expect 15253760) pool_bytes={pool:?} (expect 4489216) \
         allocations(weights + arena + pool)={built} (expect 9) {}",
        verdict(arena == 15253760 && pool == Some(4489216) && built == 9)
    );
    assert_eq!(arena, 15253760, "the arena is the larger bucket's total");
    assert_eq!(
        pool,
        Some(4489216),
        "4 slots: conv 294912 + ssm 4194304 bytes"
    );
    assert_eq!(built, 9, "seven weights, one arena, one pool");

    let inputs = [gdn_input(128, 0xA110), gdn_input(256, 0xA110)];
    let mut firsts: [Option<Vec<u16>>; 2] = [None, None];
    let at_forwards = device.allocations();
    let mut free_after_first = None;
    let mut identical = true;
    for (i, (bucket, fresh)) in [(1usize, true), (0, true), (1, false), (0, true), (1, true)]
        .into_iter()
        .enumerate()
    {
        let m = GDN_BUCKETS[bucket].bucket_m;
        let seq = GdnSequence {
            slot: bucket as u32,
            q_len: m,
            fresh,
        };
        let mut y = vec![0u16; m as usize * GL_H];
        worker
            .forward(
                bucket,
                Input::Hidden(&inputs[bucket]),
                &[],
                None,
                Some(&seq),
                &mut y,
            )
            .expect("forward");
        if i == 0 {
            free_after_first = Some(device.mem_info().expect("cudaMemGetInfo").0);
        }
        if fresh {
            match &firsts[bucket] {
                Some(first) => identical &= *first == y,
                None => firsts[bucket] = Some(y),
            }
        }
    }
    let free_after_first = free_after_first.expect("forward 1 ran");
    let free_after_last = device.mem_info().expect("cudaMemGetInfo").0;
    let during = device.allocations() - at_forwards;
    eprintln!(
        "gdn layer five forwards (256, 128, 256, 128, 256): allocations={during} (expect 0) \
         free_after_forward_1={free_after_first} free_after_forward_5={free_after_last} \
         fresh_results_bit_identical={identical} {}",
        verdict(during == 0 && free_after_first == free_after_last && identical)
    );
    assert_eq!(during, 0, "a forward allocated device memory");
    assert_eq!(
        free_after_first, free_after_last,
        "device free memory moved"
    );
    assert!(
        identical,
        "a fresh bucket's result changed between forwards"
    );
}

/// Each GDN `ForwardError`, before any copy: the result, the arena's GDN regions and every slot
/// of the pool unchanged; `GdnUnexpected` on the MLP tape. A bad state read is refused too.
#[test]
fn a_bad_gdn_sequence_is_refused_before_any_copy() {
    let device = h100();
    let w = gdn_weights();
    let table = gdn_table(&device, &w);
    let mut worker =
        CudaWorker::new(&device, &GDN_BUCKETS, &table, None, GDN_POOL).expect("worker");
    let x = gdn_input(128, 0xBAD);
    let mut y = vec![0u16; 128 * GL_H];
    worker
        .forward(
            0,
            Input::Hidden(&x),
            &[],
            None,
            Some(&GdnSequence {
                slot: 1,
                q_len: 128,
                fresh: true,
            }),
            &mut y,
        )
        .expect("a good forward first");
    let arena_before = read(
        &mut worker,
        ArenaSpan {
            offset: ArenaOffset(0),
            bytes: GDN_LAYER_M128.arena.total_bytes,
        },
    );
    let pool_before = gdn_snapshot(&mut worker);
    let seq = |slot, q_len| GdnSequence {
        slot,
        q_len,
        fresh: false,
    };
    let cases: [(Option<GdnSequence>, ForwardError); 4] = [
        (None, ForwardError::GdnMissing),
        (
            Some(seq(4, 128)),
            ForwardError::GdnSlotOutsidePool {
                slot: 4,
                num_slots: 4,
            },
        ),
        (
            Some(seq(1, 0)),
            ForwardError::GdnQueryLength {
                q_len: 0,
                bucket_m: 128,
            },
        ),
        (
            Some(seq(1, 129)),
            ForwardError::GdnQueryLength {
                q_len: 129,
                bucket_m: 128,
            },
        ),
    ];
    let mut pass = true;
    for (gdn, want) in &cases {
        let mut out = vec![0u16; 128 * GL_H];
        let got = worker.forward(0, Input::Hidden(&x), &[], None, gdn.as_ref(), &mut out);
        let untouched = out.iter().all(|&b| b == 0);
        let arena_same = read(
            &mut worker,
            ArenaSpan {
                offset: ArenaOffset(0),
                bytes: GDN_LAYER_M128.arena.total_bytes,
            },
        ) == arena_before;
        let pool_same = gdn_snapshot(&mut worker) == pool_before;
        let ok = got.as_ref() == Err(want) && untouched && arena_same && pool_same;
        eprintln!(
            "refused {want:?}: got={got:?} result_untouched={untouched} arena_unchanged=\
             {arena_same} pool_unchanged={pool_same} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    // A GDN sequence to a tape with no GDN layers.
    let mw = mlp_weights();
    let mt = mlp_table(&device, &mw);
    let mut mlp_worker = CudaWorker::new(&device, &MLP_BUCKETS, &mt, None, None).expect("mlp");
    let got = mlp_worker.forward(
        1,
        Input::Hidden(&mlp_input(256)),
        &[],
        None,
        Some(&seq(0, 256)),
        &mut vec![0u16; 256 * MLP_H],
    );
    eprintln!(
        "refused GdnUnexpected: {got:?} {}",
        verdict(got == Err(ForwardError::GdnUnexpected))
    );
    pass &= got == Err(ForwardError::GdnUnexpected);
    // A state read of a slot, a layer or a size the pool does not have.
    let l = GdnStateLayout::new(GDN_SLOTS, 16, 16, 128, 128, 4);
    let (ce, se) = (l.conv_elems_per_slot(), l.ssm_elems_per_slot());
    for (layer, slot, conv_len, ssm_len) in [(0, 4, ce, se), (1, 0, ce, se), (0, 0, ce - 1, se)] {
        let (mut c, mut s) = (vec![0f32; conv_len], vec![0f32; ssm_len]);
        let got = worker.read_gdn_state(layer, slot, &mut c, &mut s);
        let want = Err(ForwardError::GdnRead {
            layer,
            slot,
            conv: (ce, conv_len),
            ssm: (se, ssm_len),
        });
        eprintln!(
            "refused a GDN state read layer={layer} slot={slot}: {got:?} {}",
            verdict(got == want)
        );
        pass &= got == want;
    }
    assert!(pass, "a GDN refusal is wrong, or touched the device");
}

/// [`GDN_LAYER_M128`] beside a copy of [`GDN_LAYER_M256`] whose GDN state claims layer 4: one pool
/// cannot hold both.
static OTHER_GDN: [CudaTape; 2] = [
    GDN_LAYER_M128,
    CudaTape {
        gdn: match GDN_LAYER_M256.gdn {
            Some(g) => Some(GdnTape { layers: &[4], ..g }),
            None => None,
        },
        ..GDN_LAYER_M256
    },
];

/// A worker's tapes, weights and GDN pool.
type WorkerCase<'w> = (
    &'static [CudaTape],
    &'w [DeviceBuffer],
    Option<GdnPoolConfig>,
);

/// Each GDN `WorkerError`, before any allocation: no pool, a pool of 0 slots, a pool no tape uses,
/// two GDN geometries.
#[test]
fn the_gdn_pool_is_given_exactly_when_a_tape_needs_it() {
    let device = h100();
    let w = gdn_weights();
    let table = gdn_table(&device, &w);
    let mw = mlp_weights();
    let mt = mlp_table(&device, &mw);
    let before = device.allocations();
    let cases: [(WorkerCase<'_>, WorkerError); 4] = [
        ((&GDN_BUCKETS, &table, None), WorkerError::GdnPoolMissing),
        (
            (&GDN_BUCKETS, &table, Some(GdnPoolConfig { num_slots: 0 })),
            WorkerError::GdnPoolMissing,
        ),
        ((&MLP_BUCKETS, &mt, GDN_POOL), WorkerError::GdnPoolUnused),
        (
            (&OTHER_GDN, &table, GDN_POOL),
            WorkerError::GdnGeometry { bucket: 1 },
        ),
    ];
    let mut pass = true;
    for ((tapes, weights, pool), want) in cases {
        let got = CudaWorker::new(&device, tapes, weights, None, pool).map(|_| ());
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
    assert!(pass, "a GDN pool refusal is wrong");
    assert_eq!(allocated, 0, "a refused worker allocated");
}

// ── The one-row (decode) attention layer ───────────────────────────

/// The attention layer at one row as CUDA's compiler lowers it (its GEMMs GEMVs, its attention
/// the decode kernel's), beside the 128-row tape that prefills the same cache: one worker plays
/// both.
fn decode_layer_buckets() -> &'static [CudaTape] {
    let one = lower_canonical(
        &attn_layer(1),
        QWEN3_5_FACTS,
        CanonicalAt {
            stem: "attn_layer(1)",
            m: 1,
        },
    )
    .expect("the attention layer lowers at one row");
    Box::leak(Box::new([static_tape(&one), ATTN_LAYER_M128]))
}

/// A 128-row prefill of `P = 100` tokens over `[11, 3, 14]`, then decodes at positions 100..107
/// one token at a time through the one-row tape: each decode's attention against the f64
/// reference over the `P + 1` keys read back from the cache (the query its GEMV-projected,
/// normed and rotated row, read from the arena), each decode's K and V at its slot, and every
/// cache row below its position bit-unchanged by it.
#[test]
fn a_decode_step_attends_over_the_prefilled_cache() {
    let device = h100();
    let w = layer_weights();
    let table = layer_table(&device, &w);
    let buckets = decode_layer_buckets();
    let mut worker = CudaWorker::new(&device, buckets, &table, POOL, None).expect("worker");
    let one = &buckets[0];
    let (CudaStep::Rope(rope), CudaStep::RopeAppend(append), CudaStep::AttnDecode(decode)) =
        (one.steps[7], one.steps[8], one.steps[9])
    else {
        panic!("steps 7, 8 and 9 of the one-row tape are the rope, the append and the attention");
    };
    // One row of `cols` bf16 at `at`.
    let span_of = |at: ArenaOffset, cols: usize| ArenaSpan {
        offset: at,
        bytes: 2 * cols as u64,
    };
    let block_table = [11u32, 3, 14];
    let p0 = 100u32;
    let mut y = vec![0u16; 128 * AL_H];
    let prefill = KvSequence {
        block_table: &block_table[..2],
        prefix_len: 0,
        q_len: p0,
    };
    worker
        .forward(
            1,
            Input::Hidden(&layer_input(128, 0xDEC0)),
            &seq_positions(128, 0),
            Some(&prefill),
            None,
            &mut y,
        )
        .expect("the prefill");
    let mut pass = true;
    for step in 0..8u32 {
        let p = p0 + step;
        let (kb0, vb0) = (
            read_pool(&mut worker, KvHalf::K),
            read_pool(&mut worker, KvHalf::V),
        );
        let seq = KvSequence {
            block_table: &block_table[..(p as usize + 1).div_ceil(64)],
            prefix_len: p,
            q_len: 1,
        };
        let mut y1 = vec![0u16; AL_H];
        worker
            .forward(
                0,
                Input::Hidden(&layer_input(1, 0xDEC1 + u64::from(step))),
                &[p],
                Some(&seq),
                None,
                &mut y1,
            )
            .expect("a decode");
        let (kb, vb) = (
            read_pool(&mut worker, KvHalf::K),
            read_pool(&mut worker, KvHalf::V),
        );
        let tokens = p as usize + 1;
        let (k_before, v_before) = (
            gather(&kb0, &block_table, p as usize),
            gather(&vb0, &block_table, p as usize),
        );
        let (k_rows, v_rows) = (
            gather(&kb, &block_table, tokens),
            gather(&vb, &block_table, tokens),
        );
        let n = p as usize * AL_KV;
        let below =
            mismatched_bits(&k_rows[..n], &k_before) + mismatched_bits(&v_rows[..n], &v_before);
        let kr = read(&mut worker, span_of(append.k_out, AL_KV));
        let v = read(&mut worker, span_of(append.v, AL_KV));
        let appended = cache_mismatches((&kb, &vb), (&kr, &v), &block_table, (p as usize, 1));
        let finite = y1
            .iter()
            .all(|&b| half::bf16::from_bits(b).to_f32().is_finite());
        let qn = read(&mut worker, span_of(rope.x, AL_Q));
        let uploaded = read(&mut worker, one.rotary.expect("a rotary table").span);
        let qr = qr_direct(&device, one, &qn, &uploaded);
        let attn = read(&mut worker, span_of(decode.out, AL_Q));
        let ok = check_attention(
            &format!("decode {step}: attn = AttnDecode(qr, cache)"),
            &attn,
            &qr,
            (&k_rows, &v_rows),
            (p as usize, 1),
        );
        eprintln!(
            "decode {step} at position {p}: rows_below_changed_bits={below} \
             appended_mismatched_bits={appended} y_finite={finite} {}",
            verdict(below == 0 && appended == 0 && finite)
        );
        pass &= ok && below == 0 && appended == 0 && finite;
    }
    pass &= decode.rows == 1 && decode.scale_bits == 0.0625f32.to_bits();
    assert!(
        pass,
        "a decode step of the attention layer failed: see the lines above"
    );
}

// ── The qwen3-5-moe MoE block (#857) ───────────────────────────────

use common::moe_block::{
    MOE_FACTS, MUTATIONS, MoeWeights, Routing, moe_block, moe_weights, moe_x, references,
};

/// The MoE block's buckets: one gathered row, a sorted bucket, and the long run.
const MOE_MS: [u32; 3] = [1, 512, 4096];
/// #841(a)'s predicted MoE scratch bytes at each of [`MOE_MS`].
const MOE_SCRATCH: [u64; 3] = [58_368, 412_683_776, 737_939_712];
/// The tokens of an `m`-row bucket the f64 reference checks: about 64, odd and even.
fn moe_tokens(m: usize) -> Vec<usize> {
    (0..m).step_by((m / 64).max(1) | 1).collect()
}

/// The MoE block at each of [`MOE_MS`], as CUDA's compiler lowers it.
fn moe_buckets() -> &'static [CudaTape] {
    static TAPES: std::sync::OnceLock<Vec<CudaTape>> = std::sync::OnceLock::new();
    TAPES.get_or_init(|| {
        (MOE_MS.iter())
            .map(|&m| {
                let c = lower_canonical(
                    &moe_block(m),
                    MOE_FACTS,
                    CanonicalAt {
                        stem: "moe_block",
                        m: u64::from(m),
                    },
                )
                .unwrap_or_else(|e| panic!("the MoE block lowers at m={m}: {e:?}"));
                static_tape(&c)
            })
            .collect()
    })
}

fn moe_table(device: &CudaDevice, w: &MoeWeights) -> [DeviceBuffer; 7] {
    let up = |b: &[u16]| DeviceBuffer::upload(device, b).expect("upload a weight");
    [
        DeviceBuffer::empty(),
        up(&w.g),
        up(&w.bundle),
        up(&w.wsg),
        up(&w.wsu),
        up(&w.wgate),
        up(&w.wsd),
    ]
}

/// One forward of bucket `b`; its result.
fn moe_forward(worker: &mut CudaWorker<'_>, b: usize, x: &[u16]) -> Vec<u16> {
    let mut y = vec![0u16; x.len()];
    (worker.forward(b, Input::Hidden(x), &[], None, None, &mut y)).expect("the MoE forward");
    y
}

/// The routing's picks after a forward of bucket `b`, `[m, 8]` i32 read as u32.
fn moe_picks(worker: &mut CudaWorker<'_>, tape: &CudaTape) -> Vec<u32> {
    let span = tape.moe.expect("a MoE scratch").topk_idx;
    let h = read(worker, span);
    (h.as_chunks::<2>().0.iter())
        .map(|p| u32::from(p[0]) | u32::from(p[1]) << 16)
        .collect()
}

/// The lowered MoE tapes validate, and their MoE scratch is #841(a)'s prediction; the colours
/// and the scratch are printed.
#[test]
fn moe_block_tapes_lower_to_the_predicted_scratch() {
    let mut pass = true;
    for ((tape, &m), &want) in moe_buckets().iter().zip(&MOE_MS).zip(&MOE_SCRATCH) {
        let valid = tape.validate();
        let moe = tape.moe.expect("a MoE scratch");
        let bytes = moe.down.offset.0 + moe.down.bytes - moe.logits.offset.0;
        let ok = valid.is_ok() && bytes == want;
        eprintln!(
            "moe_block m={m}: arena_bytes={} colour_bytes={:?} moe_scratch_bytes={bytes} \
             (predicted {want}) max_tiles={} sorted={} validate={valid:?} {}",
            tape.arena.total_bytes,
            tape.arena.colour_bytes,
            moe.max_tiles,
            moe.sort.is_some(),
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "a MoE block tape is not the predicted one");
}

/// The MoE block at m = 1 (gathered), 512 (sorted) and 4,096 (the long run), on each routing
/// case, against the f64 block on about 64 tokens per case: CUDA's RMS and max error each at most
/// twice the bf16 emulation's (which rounds where the tape stores), the picks the emulation's on
/// every token that is not a near-tie, and each of the six mutated references out of bound on at
/// least half the tokens it changes. The near-tie and out-of-bound counts are printed.
#[test]
fn moe_block_matches_the_f64_block_in_each_bucket() {
    let device = h100();
    let w = moe_weights();
    let table = moe_table(&device, &w);
    let tapes = moe_buckets();
    let mut worker = CudaWorker::new(&device, tapes, &table, None, None).expect("worker");
    let mut pass = true;
    for (b, tape) in tapes.iter().enumerate() {
        let m = tape.bucket_m as usize;
        for routing in [Routing::Random, Routing::Skewed, Routing::Half] {
            let x = moe_x(routing, m, 0x0E_5EED ^ m as u64);
            let y = moe_forward(&mut worker, b, &x);
            let picks = moe_picks(&mut worker, tape);
            let tokens = moe_tokens(m);
            let refs = references(&w, &x, &tokens);
            let j = common::moe_block::judge(&refs, &tokens, &y, &picks);
            let ok = j.pass() && j.mutations_caught();
            eprintln!(
                "moe_block m={m} {routing:?}: tokens={} near_ties={} rms_err={:.4e} \
                 (emulation {:.4e}) max_err={:.4e} (emulation {:.4e}) out_of_bound={} \
                 nonfinite={} picks_differ={} {}",
                j.tokens,
                j.near_ties,
                j.gpu.0,
                j.emu.0,
                j.gpu.1,
                j.emu.1,
                j.out_of_bound,
                j.nonfinite,
                j.picks_differ,
                verdict(j.pass())
            );
            for (name, &(changed, out)) in MUTATIONS.iter().zip(&j.mutations) {
                eprintln!(
                    "moe_block m={m} {routing:?} self-check {name}: out_of_bound={out}/{changed} \
                     ({:.1}%) {}",
                    100.0 * out as f64 / changed.max(1) as f64,
                    verdict(changed > 0 && 2 * out >= changed)
                );
            }
            pass &= ok;
        }
    }
    assert!(
        pass,
        "the MoE block is out of bound, or a self-check fell short"
    );
}

/// A tape of no steps whose input is the whole `bytes`-byte arena: its forward fills every arena
/// byte, the MoE scratch included, with the input's bits.
fn whole_arena_tape(bytes: u64) -> CudaTape {
    let whole = ArenaSpan {
        offset: ArenaOffset(0),
        bytes,
    };
    CudaTape {
        bucket_m: 1,
        arena: ArenaLayout {
            colour_bytes: Box::leak(Box::new([bytes])),
            colour_offsets: Box::leak(Box::new([0])),
            total_bytes: bytes,
        },
        input: TapeInput::Hidden(whole),
        result: ArenaSpan {
            offset: ArenaOffset(0),
            bytes: 2,
        },
        weight_bytes: &[],
        steps: &[],
        rotary: None,
        kv: None,
        gdn: None,
        moe: None,
        rotary_local: None,
    }
}

/// The MoE block's buckets, then [`whole_arena_tape`] over the largest arena.
fn moe_buckets_and_fill() -> &'static [CudaTape] {
    static TAPES: std::sync::OnceLock<Vec<CudaTape>> = std::sync::OnceLock::new();
    TAPES.get_or_init(|| {
        let mut t = moe_buckets().to_vec();
        let bytes = (t.iter().map(|t| t.arena.total_bytes))
            .max()
            .expect("tapes");
        t.push(whole_arena_tape(bytes));
        t
    })
}

/// Fill the whole arena with `v` (bucket `fill` of [`moe_buckets_and_fill`]).
fn fill_arena(worker: &mut CudaWorker<'_>, fill: usize, bytes: u64, v: u16) {
    let mut one = [0u16; 1];
    let input = vec![v; bytes as usize / 2];
    (worker.forward(fill, Input::Hidden(&input), &[], None, None, &mut one)).expect("the fill");
}

/// The MoE block's invariants at m = 1 and 512 (random routing): a forward after the whole arena,
/// MoE scratch included, was poisoned with bf16 NaN (`0x7FC0`) gives finite output, bit-identical
/// to one after a zero-filled arena; a repeat is bit-identical; at m = 512, permuting the input
/// rows permutes the output rows bit for bit; and five forwards allocate nothing.
#[test]
fn moe_block_invariants_hold() {
    let device = h100();
    let w = moe_weights();
    let table = moe_table(&device, &w);
    let tapes = moe_buckets_and_fill();
    let mut worker = CudaWorker::new(&device, tapes, &table, None, None).expect("worker");
    let h = common::moe_block::H as usize;
    let fill = tapes.len() - 1;
    let bytes = tapes[fill].arena.total_bytes;
    let mut pass = true;
    for (b, tape) in tapes[..2].iter().enumerate() {
        let m = tape.bucket_m as usize;
        let x = moe_x(Routing::Random, m, 0x1A_0000 ^ m as u64);
        fill_arena(&mut worker, fill, bytes, 0);
        let after_zero = moe_forward(&mut worker, b, &x);
        fill_arena(&mut worker, fill, bytes, 0x7FC0);
        let after_nan = moe_forward(&mut worker, b, &x);
        let repeat = moe_forward(&mut worker, b, &x);
        let nonfinite = (after_nan.iter())
            .filter(|&&v| !half::bf16::from_bits(v).is_finite())
            .count();
        let poison = mismatched_bits(&after_nan, &after_zero);
        let rep = mismatched_bits(&repeat, &after_nan);
        let ok = nonfinite == 0 && poison == 0 && rep == 0;
        eprintln!(
            "moe_block m={m}: nan_poison_vs_zero_mismatched_bits={poison} nonfinite={nonfinite} \
             repeat_mismatched_bits={rep} {}",
            verdict(ok)
        );
        pass &= ok;
        if m > 1 {
            // Row t of the permuted input is row (37·t + 11) mod m of `x`: 37 is odd and m a
            // power of two, so it is a bijection of 0..m.
            let src = |t: usize| (37 * t + 11) % m;
            let mut xp = vec![0u16; x.len()];
            let mut want = vec![0u16; x.len()];
            for t in 0..m {
                xp[t * h..][..h].copy_from_slice(&x[src(t) * h..][..h]);
                want[t * h..][..h].copy_from_slice(&after_nan[src(t) * h..][..h]);
            }
            let perm = mismatched_bits(&moe_forward(&mut worker, b, &xp), &want);
            eprintln!(
                "moe_block m={m}: permuted_rows_mismatched_bits={perm} {}",
                verdict(perm == 0)
            );
            pass &= perm == 0;
        }
    }
    let before = device.allocations();
    let x = moe_x(Routing::Half, 512, 0x1A_0001);
    for _ in 0..5 {
        moe_forward(&mut worker, 1, &x);
    }
    let allocations = device.allocations() - before;
    eprintln!(
        "moe_block five forwards: allocations={allocations} (expect 0) {}",
        verdict(allocations == 0)
    );
    assert!(pass, "a MoE block invariant does not hold");
    assert_eq!(allocations, 0, "a MoE forward allocated");
}

// ── Gemma 4's steps are not launched yet (#885, 13a) ───────────────

use common::gemma_moe::{GEMMA_MOE_FACTS, gemma_moe};

/// The Gemma MoE block at m = 512, as CUDA's compiler lowers it: one bucket.
fn gemma_moe_buckets() -> &'static [CudaTape] {
    static TAPES: std::sync::OnceLock<Vec<CudaTape>> = std::sync::OnceLock::new();
    TAPES.get_or_init(|| {
        let at = CanonicalAt {
            stem: "gemma_moe",
            m: 512,
        };
        let c = lower_canonical(&gemma_moe(512), GEMMA_MOE_FACTS, at)
            .unwrap_or_else(|e| panic!("the Gemma MoE block lowers at m=512: {e}"));
        vec![static_tape(&c)]
    })
}

/// A Gemma tape is refused with `NotLaunched` at its first new step (the 64-column router GEMM, step 1),
/// before anything is uploaded or enqueued: the result colour keeps every byte it had, and the
/// host result buffer is untouched. Synthetic zero weights of the declared sizes.
#[test]
fn a_gemma_tape_is_refused_not_launched_before_anything_is_enqueued() {
    let device = h100();
    let tapes = gemma_moe_buckets();
    let tape = &tapes[0];
    let kinds: Vec<String> = (tape.steps.iter())
        .map(|s| format!("{s:?}").split('(').next().unwrap_or("").to_owned())
        .collect();
    eprintln!("gemma m=512 step kinds: {kinds:?}");
    assert!(
        matches!(tape.steps[1], CudaStep::RouterLogits(r) if r.kernel == MatmulKernel::Col64),
        "the first unlaunched step is the 64-column router GEMM: {:?}",
        tape.steps[1]
    );
    let table: Vec<DeviceBuffer> = (tape.weight_bytes.iter())
        .map(|&b| match b {
            0 => DeviceBuffer::empty(),
            b => DeviceBuffer::upload_bytes(&device, &vec![0u8; b as usize]).expect("upload"),
        })
        .collect();
    let table: &'static [DeviceBuffer] = Box::leak(table.into_boxed_slice());
    let mut worker = CudaWorker::new(&device, tapes, table, None, None).expect("worker");
    let m = tape.bucket_m as usize;
    let x = vec![0x3F80u16; m * common::gemma_moe::H as usize];
    let result = colour(tape, tape.arena.colour_offsets.len() - 1);
    let mut y = vec![0xBEEFu16; tape.result.bytes as usize / 2];
    let before = read(&mut worker, result);
    let e = worker
        .forward(0, Input::Hidden(&x), &[], None, None, &mut y)
        .expect_err("a Gemma step is not launched");
    eprintln!("gemma m=512: {e}");
    assert!(
        matches!(
            e,
            ForwardError::NotLaunched {
                step: 1,
                op: "RouterLogits"
            }
        ),
        "{e:?}"
    );
    assert!(y.iter().all(|&v| v == 0xBEEF), "the result is untouched");
    assert_eq!(read(&mut worker, result), before, "nothing was enqueued");
}

// ── Gemma 4's row steps, played (#921) ─────────────────────────────

use common::gemma_mlp::{GEMMA_MLP_FACTS, gemma_mlp};
use common::gemma_rows::{GEMMA_ROWS_FACTS, gemma_rows};

/// The Gemma row chain at m = 1 and 512, as CUDA's compiler lowers it, then
/// [`whole_arena_tape`] over the larger arena.
fn gemma_rows_buckets() -> &'static [CudaTape] {
    static TAPES: std::sync::OnceLock<Vec<CudaTape>> = std::sync::OnceLock::new();
    TAPES.get_or_init(|| {
        let mut t: Vec<CudaTape> = [1u32, 512]
            .into_iter()
            .map(|m| {
                let at = CanonicalAt {
                    stem: "gemma_rows",
                    m: u64::from(m),
                };
                let c = lower_canonical(&gemma_rows(m), GEMMA_ROWS_FACTS, at)
                    .unwrap_or_else(|e| panic!("the Gemma row chain lowers at m={m}: {e}"));
                static_tape(&c)
            })
            .collect();
        let bytes = (t.iter().map(|t| t.arena.total_bytes))
            .max()
            .expect("tapes");
        t.push(whole_arena_tape(bytes));
        t
    })
}

/// Play one row step of `tape` by its standalone launcher on a host copy of the arena: its input
/// region uploaded, the launch, its output region downloaded back into `arena`. `w` is the layer
/// scale's weight.
fn play_row_step(device: &CudaDevice, s: &CudaStep, arena: &mut [u16], w: &DeviceBuffer) {
    let region = |at: ArenaOffset, rows: u32, cols: u32| {
        let at = at.0 as usize / 2;
        at..at + rows as usize * cols as usize
    };
    let (x, out, rows, cols) = match *s {
        CudaStep::Scale(st) => (st.x, st.out, st.rows, st.cols),
        CudaStep::RmsNormUnit(st) => (st.x, st.y, st.rows, st.cols),
        CudaStep::LayerScale(st) => (st.x, st.out, st.rows, st.cols),
        CudaStep::SoftCap(st) => (st.x, st.out, st.rows, st.cols),
        ref other => panic!("the Gemma row chain holds only row maps: {other:?}"),
    };
    let xb = DeviceBuffer::upload(device, &arena[region(x, rows, cols)]).expect("upload x");
    let ob = DeviceBuffer::upload(device, &vec![0; rows as usize * cols as usize]).expect("out");
    let (dims, null) = ((rows, cols), std::ptr::null_mut());
    // SAFETY: cudaMalloc pointers are 256-byte aligned; `x` and `out` hold rows·cols bf16, `w`
    // one; `out` is its own allocation; null is the default stream, which the download waits for.
    unsafe {
        match *s {
            CudaStep::Scale(st) => scale_bf16(
                xb.ptr(),
                ob.ptr(),
                dims,
                f32::from_bits(st.scale_bits),
                null,
            ),
            CudaStep::RmsNormUnit(st) => {
                rms_norm_unit_bf16(xb.ptr(), ob.ptr(), dims, f32::from_bits(st.eps_bits), null)
            }
            CudaStep::LayerScale(_) => layer_scale_bf16(xb.ptr(), w.ptr(), ob.ptr(), dims, null),
            CudaStep::SoftCap(st) => {
                soft_cap_bf16(xb.ptr(), ob.ptr(), dims, f32::from_bits(st.cap_bits), null)
            }
            _ => unreachable!("matched above"),
        }
    }
    .expect("a standalone row launch");
    let got = ob.download().expect("download out");
    arena[region(out, rows, cols)].copy_from_slice(&got);
}

/// Gemma's row chain (`ScalarMul(√2816) → RmsNormUnit → ScalarWeightMul(w) → TanhSoftCap(30)`) at
/// m = 1 and 512: every colour of the arena after the forward equals what the standalone launchers
/// write when each plays its step on the previous one's downloaded output, bit for bit (so each
/// step is its kernel); the arena poisoned with bf16 NaN before the forward gives the bits of a
/// zero-filled one; a repeat is bit-identical; five forwards allocate nothing.
#[test]
fn gemma_row_chain_plays_each_step_as_its_kernel() {
    let device = h100();
    let tapes = gemma_rows_buckets();
    let w0 = bf16_inputs(1, 0x6E1D_3001)[0];
    let wb = DeviceBuffer::upload(&device, &[w0]).expect("upload w");
    let table = [DeviceBuffer::empty(), wb];
    let table: &'static [DeviceBuffer] = Box::leak(Box::new(table));
    let mut worker = CudaWorker::new(&device, tapes, table, None, None).expect("worker");
    let h = common::gemma_rows::H as usize;
    let fill = tapes.len() - 1;
    let bytes = tapes[fill].arena.total_bytes;
    let mut pass = true;
    for (b, tape) in tapes[..2].iter().enumerate() {
        let m = tape.bucket_m as usize;
        let kinds: Vec<String> = (tape.steps.iter())
            .map(|s| format!("{s:?}").split('(').next().unwrap_or("").to_owned())
            .collect();
        eprintln!("gemma_rows m={m} steps: {kinds:?}");
        let x = bf16_inputs(m * h, 0x6E1D_3000 ^ m as u64);
        let forward = |worker: &mut CudaWorker<'_>| {
            let mut y = vec![0u16; m * h];
            (worker.forward(b, Input::Hidden(&x), &[], None, None, &mut y)).expect("forward");
            y
        };
        fill_arena(&mut worker, fill, bytes, 0);
        let after_zero = forward(&mut worker);
        let colours: Vec<Vec<u16>> = (0..tape.arena.colour_offsets.len())
            .map(|c| read(&mut worker, colour(tape, c)))
            .collect();
        let mut host = vec![0u16; tape.arena.total_bytes as usize / 2];
        host[..m * h].copy_from_slice(&x);
        for s in tape.steps {
            play_row_step(&device, s, &mut host, &table[1]);
        }
        for (c, got) in colours.iter().enumerate() {
            let at = tape.arena.colour_offsets[c] as usize / 2;
            let want = &host[at..at + got.len()];
            let mismatched = mismatched_bits(got, want);
            eprintln!(
                "gemma_rows m={m} colour {c}: vs standalone launchers mismatched_bits={mismatched} \
                 {}",
                verdict(mismatched == 0)
            );
            pass &= mismatched == 0;
        }
        fill_arena(&mut worker, fill, bytes, 0x7FC0);
        let after_nan = forward(&mut worker);
        let repeat = forward(&mut worker);
        let poison = mismatched_bits(&after_nan, &after_zero);
        let rep = mismatched_bits(&repeat, &after_nan);
        let ok = poison == 0 && rep == 0;
        eprintln!(
            "gemma_rows m={m}: nan_poison_vs_zero_mismatched_bits={poison} \
             repeat_mismatched_bits={rep} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    let at_forwards = device.allocations();
    let x = bf16_inputs(512 * h, 0x6E1D_3005);
    for _ in 0..5 {
        let mut y = vec![0u16; 512 * h];
        (worker.forward(1, Input::Hidden(&x), &[], None, None, &mut y)).expect("forward");
    }
    let during = device.allocations() - at_forwards;
    eprintln!(
        "gemma_rows five forwards: allocations={during} (expect 0) {}",
        verdict(during == 0)
    );
    pass &= during == 0;
    assert!(
        pass,
        "a Gemma row step is not its kernel, or an invariant failed"
    );
}

/// The dense Gemma MLP at m = 1, as CUDA's compiler lowers it.
fn gemma_mlp_buckets() -> &'static [CudaTape] {
    static TAPES: std::sync::OnceLock<Vec<CudaTape>> = std::sync::OnceLock::new();
    TAPES.get_or_init(|| {
        let at = CanonicalAt {
            stem: "gemma_mlp",
            m: 1,
        };
        let c = lower_canonical(&gemma_mlp(1), GEMMA_MLP_FACTS, at)
            .unwrap_or_else(|e| panic!("the Gemma MLP lowers at m=1: {e}"));
        vec![static_tape(&c)]
    })
}

/// `bf16(v)` as f64: a value rounded where the tape stores it.
fn round_bf16(v: f64) -> f64 {
    f64::from(half::bf16::from_f64(v).to_f32())
}

/// One row of `x · Wᵀ`, `W` `[n, k]` row-major bf16 bits, in f64, each output through `rnd`.
fn gemv_f64(x: &[f64], w: &[u16], n: usize, rnd: fn(f64) -> f64) -> Vec<f64> {
    let k = x.len();
    let w = widen(w);
    (0..n)
        .map(|j| rnd(x.iter().zip(&w[j * k..][..k]).map(|(a, b)| a * b).sum()))
        .collect()
}

/// The dense MLP of one row in f64, `act` its activation, every stored value through `rnd` (the
/// identity for the reference, [`round_bf16`] for the emulation).
fn gemma_mlp_row(
    x: &[u16],
    (g, wg, wu, wd): (&[u16], &[u16], &[u16], &[u16]),
    act: fn(f64) -> f64,
    rnd: fn(f64) -> f64,
) -> Vec<f64> {
    let (h, i) = (common::gemma_mlp::H as usize, common::gemma_mlp::I as usize);
    let (xf, gf) = (widen(x), widen(g));
    let mean = xf.iter().map(|v| v * v).sum::<f64>() / h as f64;
    let inv = 1.0 / (mean + f64::from(common::gemma_mlp::EPS)).sqrt();
    let n: Vec<f64> = (xf.iter().zip(&gf))
        .map(|(v, w)| rnd(v * inv * w))
        .collect();
    let gate = gemv_f64(&n, wg, i, rnd);
    let up = gemv_f64(&n, wu, i, rnd);
    let a: Vec<f64> = (gate.iter().zip(&up))
        .map(|(g, u)| rnd(act(*g) * u))
        .collect();
    let d = gemv_f64(&a, wd, h, rnd);
    (d.iter().zip(&xf)).map(|(d, x)| rnd(d + x)).collect()
}

fn gelu_tanh(g: f64) -> f64 {
    let c = (2.0 / std::f64::consts::PI).sqrt();
    0.5 * g * (1.0 + (c * (g + 0.044715 * g * g * g)).tanh())
}

fn silu(g: f64) -> f64 {
    g / (1.0 + (-g).exp())
}

/// `(rms, max)` of `gpu − refs`.
fn rms_max(gpu: &[f64], refs: &[f64]) -> (f64, f64) {
    let (mut sq, mut mx) = (0.0f64, 0.0f64);
    for (g, r) in gpu.iter().zip(refs) {
        let e = (g - r).abs();
        sq += e * e;
        mx = mx.max(e);
    }
    ((sq / gpu.len() as f64).sqrt(), mx)
}

/// Gemma's dense MLP at the real geometry (H 2816, I 2112), m = 1 (its GEMMs are GEMVs): CUDA's
/// RMS and max error against the f64 block each at most 2× a host emulation's that rounds to bf16
/// where the tape stores. Self-check: the block with SiLU instead of GELU is out of that bound.
#[test]
fn gemma_dense_mlp_decode_block() {
    let device = h100();
    let tapes = gemma_mlp_buckets();
    let (h, i) = (common::gemma_mlp::H as usize, common::gemma_mlp::I as usize);
    // Weights scaled by 1/16, so the gates sit where GELU and SiLU differ.
    let small = |len, seed| -> Vec<u16> {
        (bf16_inputs(len, seed).into_iter())
            .map(|b| common::numerics::bits(half::bf16::from_bits(b).to_f32() / 16.0))
            .collect()
    };
    let g = bf16_inputs(h, 0x6E1D_4001);
    let (wg, wu, wd) = (
        small(i * h, 0x6E1D_4002),
        small(i * h, 0x6E1D_4003),
        small(h * i, 0x6E1D_4004),
    );
    let table = [
        DeviceBuffer::empty(),
        DeviceBuffer::upload(&device, &g).expect("upload g"),
        DeviceBuffer::upload(&device, &wg).expect("upload Wg"),
        DeviceBuffer::upload(&device, &wu).expect("upload Wu"),
        DeviceBuffer::upload(&device, &wd).expect("upload Wd"),
    ];
    let table: &'static [DeviceBuffer] = Box::leak(Box::new(table));
    let mut worker = CudaWorker::new(&device, tapes, table, None, None).expect("worker");
    let x = bf16_inputs(h, 0x6E1D_4000);
    let mut y = vec![0u16; h];
    (worker.forward(0, Input::Hidden(&x), &[], None, None, &mut y)).expect("forward");
    let gpu = widen(&y);
    let w = (&g[..], &wg[..], &wu[..], &wd[..]);
    let reference = gemma_mlp_row(&x, w, gelu_tanh, |v| v);
    let emulation = gemma_mlp_row(&x, w, gelu_tanh, round_bf16);
    let (cuda_rms, cuda_max) = rms_max(&gpu, &reference);
    let (emu_rms, emu_max) = rms_max(&emulation, &reference);
    let ok = cuda_rms <= 2.0 * emu_rms && cuda_max <= 2.0 * emu_max;
    eprintln!(
        "gemma dense mlp m=1: cuda_rms={cuda_rms:.4e} cuda_max={cuda_max:.4e} \
         emulation_rms={emu_rms:.4e} emulation_max={emu_max:.4e} rms_ratio={:.4} \
         max_ratio={:.4} {}",
        cuda_rms / (2.0 * emu_rms),
        cuda_max / (2.0 * emu_max),
        verdict(ok)
    );
    let mut pass = ok;
    let wrong = gemma_mlp_row(&x, w, silu, |v| v);
    let (silu_rms, silu_max) = rms_max(&gpu, &wrong);
    let caught = silu_rms > 2.0 * emu_rms || silu_max > 2.0 * emu_max;
    eprintln!(
        "gemma dense mlp self-check SiLU: rms={silu_rms:.4e} max={silu_max:.4e} out_of_bound={caught} \
         {}",
        verdict(caught)
    );
    pass &= caught;
    assert!(
        pass,
        "the Gemma dense MLP is out of bound, or the self-check fell short"
    );
}

// ── The Qwen routing at 128 experts is refused up front (#921) ────

/// [`gemma_moe`]'s block at m = 1 with its routing step rewritten to Qwen's program at its 128
/// experts (no per-expert scale): a tape the lowering accepts, since `op_abi::MOE_EXPERT_COUNTS`
/// holds 128 for either program, but whose routing the kernel has no instance for.
fn qwen_route_at_128() -> &'static [CudaTape] {
    static TAPES: std::sync::OnceLock<Vec<CudaTape>> = std::sync::OnceLock::new();
    TAPES.get_or_init(|| {
        let at = CanonicalAt {
            stem: "gemma_moe",
            m: 1,
        };
        let c = lower_canonical(&gemma_moe(1), GEMMA_MOE_FACTS, at)
            .unwrap_or_else(|e| panic!("the Gemma MoE block lowers at m=1: {e}"));
        let tape = static_tape(&c);
        let steps: Vec<CudaStep> = (tape.steps.iter())
            .map(|&s| match s {
                CudaStep::MoeRoute(r) => {
                    CudaStep::MoeRoute(scratchy_target_cuda::step::MoeRouteBf16 {
                        program: scratchy_target_cuda::step::RouteProgram::QwenSoftmax,
                        renorm: true,
                        expert_scale: None,
                        ..r
                    })
                }
                other => other,
            })
            .collect();
        vec![CudaTape {
            steps: Box::leak(steps.into_boxed_slice()),
            ..tape
        }]
    })
}

/// A Qwen routing at 128 experts (`step::MOE_ROUTE_EXPERTS` is 256, the Qwen program's one
/// instance) is refused with `NotLaunched` at that step before anything is uploaded or enqueued,
/// although the router norm and the one-row router GEMV before it now launch: the result colour
/// keeps every byte it had, and the host result is untouched.
#[test]
fn a_qwen_routing_at_128_experts_is_refused_before_anything_is_enqueued() {
    let device = h100();
    let tapes = qwen_route_at_128();
    let tape = &tapes[0];
    let route = (tape.steps.iter())
        .position(|s| matches!(s, CudaStep::MoeRoute(_)))
        .expect("the block routes");
    assert!(
        matches!(
            tape.steps[route],
            CudaStep::MoeRoute(r) if r.experts == 128
                && r.program == scratchy_target_cuda::step::RouteProgram::QwenSoftmax
        ),
        "{:?}",
        tape.steps[route]
    );
    let table: Vec<DeviceBuffer> = (tape.weight_bytes.iter())
        .map(|&b| match b {
            0 => DeviceBuffer::empty(),
            b => DeviceBuffer::upload_bytes(&device, &vec![0u8; b as usize]).expect("upload"),
        })
        .collect();
    let table: &'static [DeviceBuffer] = Box::leak(table.into_boxed_slice());
    let mut worker = CudaWorker::new(&device, tapes, table, None, None).expect("worker");
    let x = vec![0x3F80u16; common::gemma_moe::H as usize];
    let result = colour(tape, tape.arena.colour_offsets.len() - 1);
    let mut y = vec![0xBEEFu16; tape.result.bytes as usize / 2];
    let before = read(&mut worker, result);
    let e = worker
        .forward(0, Input::Hidden(&x), &[], None, None, &mut y)
        .expect_err("a Qwen routing at 128 experts is not launched");
    eprintln!("qwen route at 128, step {route}: {e}");
    assert!(
        matches!(e, ForwardError::NotLaunched { step, op: "MoeRoute" } if step == route),
        "{e:?}"
    );
    assert!(y.iter().all(|&v| v == 0xBEEF), "the result is untouched");
    assert_eq!(read(&mut worker, result), before, "nothing was enqueued");
}

// ── Gemma 4's sliding attention layer, played (#922) ───────────────

use common::gemma_attn::{GEMMA_ATTN_FACTS, gemma_attn};
use scratchy_target_cuda::step::AttnPrefillBf16;
use scratchy_target_cuda::tape::KvClass;

const GA_H: usize = common::gemma_attn::H as usize;
const GA_HD: usize = common::gemma_attn::HD as usize;
const GA_Q: usize = common::gemma_attn::Q as usize;
const GA_KV: usize = common::gemma_attn::KV as usize;
const GA_W: usize = common::gemma_attn::WINDOW as usize;
/// The pool's blocks: room for the 2012-token sequences and blocks to poison beside them.
const GA_BLOCKS: u32 = 48;
const GA_POOL: Option<KvPoolConfig> = Some(KvPoolConfig {
    num_blocks: GA_BLOCKS,
});
/// The sequence's block table: a permutation of the pool (7 is coprime with 48).
fn ga_table() -> Vec<u32> {
    (0..GA_BLOCKS).map(|b| (b * 7 + 5) % GA_BLOCKS).collect()
}

/// `t` with every attention step and KV class unwindowed: the same layer's full causal twin.
fn unwindowed(t: &CudaTape) -> CudaTape {
    let steps: Vec<CudaStep> = (t.steps.iter())
        .map(|s| match *s {
            CudaStep::AttnPrefill(a) => {
                CudaStep::AttnPrefill(AttnPrefillBf16 { window: None, ..a })
            }
            CudaStep::AttnDecode(a) => CudaStep::AttnDecode(AttnPrefillBf16 { window: None, ..a }),
            s => s,
        })
        .collect();
    let kv = t.kv.map(|kv| {
        let classes: Vec<KvClass> = (kv.classes.iter())
            .map(|c| KvClass { window: None, ..*c })
            .collect();
        KvTape {
            classes: Box::leak(classes.into_boxed_slice()),
            ..kv
        }
    });
    CudaTape {
        steps: Box::leak(steps.into_boxed_slice()),
        kv,
        ..*t
    }
}

/// The layer at m = 1 and 512 as CUDA's compiler lowers it, their unwindowed twins, then
/// [`whole_arena_tape`] over the largest arena: buckets 0, 1 (windowed), 2, 3 (twins), 4 (fill).
fn gemma_attn_buckets() -> &'static [CudaTape] {
    static TAPES: std::sync::OnceLock<Vec<CudaTape>> = std::sync::OnceLock::new();
    TAPES.get_or_init(|| {
        let lowered: Vec<CudaTape> = [1u32, 512]
            .iter()
            .map(|&m| {
                let at = CanonicalAt {
                    stem: "gemma_attn",
                    m: u64::from(m),
                };
                let c = lower_canonical(&gemma_attn(m), GEMMA_ATTN_FACTS, at)
                    .unwrap_or_else(|e| panic!("the Gemma sliding layer lowers at m={m}: {e}"));
                static_tape(&c)
            })
            .collect();
        let mut t = lowered.clone();
        t.extend(lowered.iter().map(unwindowed));
        let bytes = (t.iter().map(|t| t.arena.total_bytes))
            .max()
            .expect("tapes");
        t.push(whole_arena_tape(bytes));
        t
    })
}

/// One position's `(q, k, v)` in f64.
type GaToken = [Vec<f64>; 3];

/// The layer's weights as bits, and widened once for the host references.
struct GaWeights {
    bits: [Vec<u16>; 7],
    wide: [Vec<f64>; 7],
}

/// `g_in`, `Wq`, `q_norm`, `Wk`, `k_norm`, `Wv`, `Wo`: gains near 1 (the norms' near 1/4, so the
/// scores at scale 1 stay near unit size) and projections in [−1/16, 1/16].
fn ga_weights() -> GaWeights {
    let around = |len, seed, c: f32| -> Vec<u16> {
        (bf16_inputs(len, seed).into_iter())
            .map(|b| common::numerics::bits(c * (1.0 + half::bf16::from_bits(b).to_f32() / 8.0)))
            .collect()
    };
    let small = |len, seed| -> Vec<u16> {
        (bf16_inputs(len, seed).into_iter())
            .map(|b| common::numerics::bits(half::bf16::from_bits(b).to_f32() / 16.0))
            .collect()
    };
    let bits = [
        around(GA_H, 0x0922_A001, 1.0),
        small(GA_Q * GA_H, 0x0922_A002),
        around(GA_HD, 0x0922_A003, 0.25),
        small(GA_KV * GA_H, 0x0922_A004),
        around(GA_HD, 0x0922_A005, 0.25),
        small(GA_KV * GA_H, 0x0922_A006),
        small(GA_H * GA_Q, 0x0922_A007),
    ];
    let wide = bits.clone().map(|b| widen(&b));
    GaWeights { bits, wide }
}

/// The worker's weight table: the sources' order, `x`, the rotary pairs, the caches and the
/// unread global table empty.
fn ga_table_buffers(device: &CudaDevice, w: &GaWeights) -> &'static [DeviceBuffer] {
    let up = |b: &[u16]| DeviceBuffer::upload(device, b).expect("upload a weight");
    let e = DeviceBuffer::empty;
    let [g, wq, qn, wk, kn, wv, wo] = &w.bits;
    let table = vec![
        e(),
        up(g),
        up(wq),
        up(qn),
        up(wk),
        up(kn),
        up(wv),
        e(),
        e(),
        e(),
        e(),
        e(),
        e(),
        up(wo),
        e(),
    ];
    Box::leak(table.into_boxed_slice())
}

/// The input row of position `p`: the same at every forward that holds it.
fn ga_x(p: usize) -> Vec<u16> {
    bf16_inputs(GA_H, 0x0922_B000 + p as u64)
}

/// `[n] = x · Wᵀ`, `W` `[n, k]` widened, each output through `rnd`.
fn ga_gemv(x: &[f64], w: &[f64], n: usize, rnd: fn(f64) -> f64) -> Vec<f64> {
    let k = x.len();
    (0..n)
        .map(|j| rnd(x.iter().zip(&w[j * k..][..k]).map(|(a, b)| a * b).sum()))
        .collect()
}

/// Each `cols`-wide row of `x` RMS-normed (eps 1e-6) by `gain` (`None`: the unit norm).
fn ga_norm(x: &[f64], cols: usize, gain: Option<&[f64]>, rnd: fn(f64) -> f64) -> Vec<f64> {
    let eps = f64::from(common::gemma_attn::EPS);
    (x.chunks_exact(cols))
        .flat_map(|row| {
            let inv = 1.0 / (row.iter().map(|v| v * v).sum::<f64>() / cols as f64 + eps).sqrt();
            (row.iter().enumerate()).map(move |(d, v)| rnd(v * inv * gain.map_or(1.0, |g| g[d])))
        })
        .collect()
}

/// Each 256-lane head of `x` rotated in full at position `p`: the f64 angle's cos and sin for the
/// reference, the bf16 table's for the emulation.
fn ga_rope(x: &[f64], p: u32, emulate: bool, rnd: fn(f64) -> f64) -> Vec<f64> {
    let half = GA_HD / 2;
    let (c, s): (Vec<f64>, Vec<f64>) = if emulate {
        let t = widen(&rotary_rows(&[p], GA_HD as u32, common::gemma_attn::THETA));
        (t[..half].to_vec(), t[half..].to_vec())
    } else {
        (0..half)
            .map(|i| {
                let a =
                    f64::from(p) / common::gemma_attn::THETA.powf(2.0 * i as f64 / GA_HD as f64);
                (a.cos(), a.sin())
            })
            .unzip()
    };
    let mut out = x.to_vec();
    for head in out.as_chunks_mut::<GA_HD>().0 {
        let src = head.to_vec();
        for d in 0..half {
            head[d] = rnd(src[d] * c[d] - src[d + half] * s[d]);
            head[d + half] = rnd(src[d + half] * c[d] + src[d] * s[d]);
        }
    }
    out
}

/// Position `p`'s `(q, k, v)` through `rnd`: q and k projected, per-head normed and rotated, v
/// projected and unit-normed per head. q only when `with_q`.
fn ga_token(p: usize, w: &GaWeights, emulate: bool, with_q: bool) -> GaToken {
    let rnd: fn(f64) -> f64 = if emulate { round_bf16 } else { |v| v };
    let [g, wq, qn, wk, kn, wv, _] = &w.wide;
    let n = ga_norm(&widen(&ga_x(p)), GA_H, Some(g), rnd);
    let q = if with_q {
        let q = ga_norm(&ga_gemv(&n, wq, GA_Q, rnd), GA_HD, Some(qn), rnd);
        ga_rope(&q, p as u32, emulate, rnd)
    } else {
        Vec::new()
    };
    let k = ga_norm(&ga_gemv(&n, wk, GA_KV, rnd), GA_HD, Some(kn), rnd);
    let k = ga_rope(&k, p as u32, emulate, rnd);
    let v = ga_norm(&ga_gemv(&n, wv, GA_KV, rnd), GA_HD, None, rnd);
    [q, k, v]
}

/// The k and v of positions `0 .. tokens` (both flavours), across the host's threads.
fn ga_keys(w: &GaWeights, tokens: usize, emulate: bool) -> Vec<GaToken> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let per = tokens.div_ceil(threads);
    std::thread::scope(|scope| {
        let parts: Vec<_> = (0..threads)
            .map(|t| {
                scope.spawn(move || {
                    (t * per..((t + 1) * per).min(tokens))
                        .map(|p| ga_token(p, w, emulate, false))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        (parts.into_iter())
            .flat_map(|h| h.join().expect("a reference thread"))
            .collect()
    })
}

/// The layer's output row at position `p` (16 heads over 8 kv heads, the window's keys), every
/// stored value through `rnd`; the emulation also rounds each unnormalised `p` to bf16 before
/// P·V, as the kernel does.
fn ga_row(p: usize, keys: &[GaToken], w: &GaWeights, emulate: bool) -> Vec<f64> {
    let rnd: fn(f64) -> f64 = if emulate { round_bf16 } else { |v| v };
    let [q, ..] = ga_token(p, w, emulate, true);
    let heads = GA_Q / GA_HD;
    let first = (p + 1).saturating_sub(GA_W);
    let mut attn = vec![0.0; GA_Q];
    for h in 0..heads {
        let kh = h / (heads / (GA_KV / GA_HD));
        let qh = &q[h * GA_HD..][..GA_HD];
        let s: Vec<f64> = (first..=p)
            .map(|j| {
                let kj = &keys[j][1][kh * GA_HD..][..GA_HD];
                f64::from(common::gemma_attn::SCALE)
                    * qh.iter().zip(kj).map(|(a, b)| a * b).sum::<f64>()
            })
            .collect();
        let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
        let l: f64 = e.iter().sum();
        for d in 0..GA_HD {
            let o: f64 = (first..=p)
                .zip(&e)
                .map(|(j, &ej)| rnd(ej) * keys[j][2][kh * GA_HD + d])
                .sum();
            attn[h * GA_HD + d] = rnd(o / l);
        }
    }
    ga_gemv(&attn, &w.wide[6], GA_H, rnd)
}

/// One forward of bucket `b` over positions `p .. p + len`, the sequence's blocks `table`; the
/// output's real rows. Each row's input is [`ga_x`] of its position (or `x` when given).
fn ga_forward(
    worker: &mut CudaWorker<'_>,
    b: usize,
    (p, len): (usize, usize),
    table: &[u32],
    x: Option<u16>,
) -> Vec<u16> {
    let m = gemma_attn_buckets()[b].bucket_m as usize;
    let mut input = vec![0u16; m * GA_H];
    for i in 0..len {
        let row = match x {
            Some(v) => vec![v; GA_H],
            None => ga_x(p + i),
        };
        input[i * GA_H..][..GA_H].copy_from_slice(&row);
    }
    let seq = KvSequence {
        block_table: &table[..(p + len).div_ceil(64)],
        prefix_len: p as u32,
        q_len: len as u32,
    };
    let mut y = vec![0u16; m * GA_H];
    (worker.forward(
        b,
        Input::Hidden(&input),
        &seq_positions(m, p as u32),
        Some(&seq),
        None,
        &mut y,
    ))
    .expect("a Gemma sliding layer forward");
    y.truncate(len * GA_H);
    y
}

/// Prefill positions `0 .. tokens` in 512-row chunks through bucket 1.
fn ga_prefill(worker: &mut CudaWorker<'_>, tokens: usize, table: &[u32]) {
    for p in (0..tokens).step_by(512) {
        ga_forward(worker, 1, (p, 512.min(tokens - p)), table, None);
    }
}

/// CUDA's rows at `positions` (row `i` of `y` is position `positions[i]`) against the f64 layer,
/// RMS and max error each at most 2× the bf16 emulation's.
fn ga_check(
    what: &str,
    y: &[(usize, &[u16])],
    (keys, emu_keys): (&[GaToken], &[GaToken]),
    w: &GaWeights,
) -> bool {
    let (mut gpu, mut refs, mut emu) = (Vec::new(), Vec::new(), Vec::new());
    std::thread::scope(|scope| {
        let rows: Vec<_> = (y.iter())
            .map(|&(p, _)| {
                scope.spawn(move || (ga_row(p, keys, w, false), ga_row(p, emu_keys, w, true)))
            })
            .collect();
        for (&(_, row), h) in y.iter().zip(rows) {
            let (r, e) = h.join().expect("a reference thread");
            gpu.extend(widen(row));
            refs.extend(r);
            emu.extend(e);
        }
    });
    let finite = gpu.iter().all(|v| v.is_finite());
    let (cuda_rms, cuda_max) = rms_max(&gpu, &refs);
    let (emu_rms, emu_max) = rms_max(&emu, &refs);
    let ok = finite && cuda_rms <= 2.0 * emu_rms && cuda_max <= 2.0 * emu_max;
    eprintln!(
        "gemma sliding layer {what} ({} rows): cuda_rms={cuda_rms:.4e} cuda_max={cuda_max:.4e} \
         emulation_rms={emu_rms:.4e} emulation_max={emu_max:.4e} rms_ratio={:.4} max_ratio={:.4} \
         finite={finite} {}",
        y.len(),
        cuda_rms / (2.0 * emu_rms),
        cuda_max / (2.0 * emu_max),
        verdict(ok)
    );
    ok
}

/// Gemma 4's sliding attention layer at the real geometry (H 2816, 16 q heads over 8 kv heads of
/// 256, window 1024), at m = 512 and m = 1, against the f64 layer (CUDA's RMS and max error each at
/// most 2× a host emulation's that rounds to bf16 where the tape stores): a fresh 512-row prefill
/// (P = 0, no key outside any window), bit-identical to the same layer unwindowed; a chunk at
/// P = 1500, L = 512 (the window drops keys); a decode chain from P = 1020 to 1030 across the
/// window's first bite. Every eighth row of a prefill is checked.
#[test]
fn gemma_attn_sliding_layer_matches_the_f64_layer() {
    let device = h100();
    let tapes = gemma_attn_buckets();
    for (b, t) in tapes[..2].iter().enumerate() {
        let kinds: Vec<String> = (t.steps.iter())
            .map(|s| format!("{s:?}").split('(').next().unwrap_or("").to_owned())
            .collect();
        eprintln!("gemma sliding layer bucket {b} m={}: {kinds:?}", t.bucket_m);
    }
    let w = ga_weights();
    let weights = ga_table_buffers(&device, &w);
    let mut worker = CudaWorker::new(&device, tapes, weights, GA_POOL, None).expect("worker");
    let table = ga_table();
    let tokens = 1500 + 512;
    let (keys, emu_keys) = (ga_keys(&w, tokens, false), ga_keys(&w, tokens, true));
    let kv = (&keys[..], &emu_keys[..]);
    let rows_of = |y: &[u16], p: usize, len: usize| -> Vec<(usize, Vec<u16>)> {
        (0..len)
            .step_by(8)
            .map(|i| (p + i, y[i * GA_H..][..GA_H].to_vec()))
            .collect()
    };
    let check = |what: &str, rows: &[(usize, Vec<u16>)]| {
        let y: Vec<(usize, &[u16])> = rows.iter().map(|(p, r)| (*p, &r[..])).collect();
        ga_check(what, &y, kv, &w)
    };
    let mut pass = true;

    let fresh = ga_forward(&mut worker, 1, (0, 512), &table, None);
    pass &= check("fresh prefill P=0 L=512", &rows_of(&fresh, 0, 512));
    let causal = ga_forward(&mut worker, 3, (0, 512), &table, None);
    let mismatched = mismatched_bits(&fresh, &causal);
    eprintln!(
        "gemma sliding layer fresh prefill vs the unwindowed layer: mismatched_bits={mismatched} {}",
        verdict(mismatched == 0)
    );
    pass &= mismatched == 0;

    ga_prefill(&mut worker, 1500, &table);
    let chunk = ga_forward(&mut worker, 1, (1500, 512), &table, None);
    pass &= check("chunk P=1500 L=512", &rows_of(&chunk, 1500, 512));

    ga_prefill(&mut worker, 1020, &table);
    for p in 1020..=1030 {
        let y = ga_forward(&mut worker, 0, (p, 1), &table, None);
        pass &= check(&format!("decode P={p}"), &[(p, y)]);
    }
    assert!(
        pass,
        "the Gemma sliding layer is out of bound: see the lines above"
    );
}

/// The layer's invariants: a forward after the arena was poisoned with bf16 NaN and the pool's
/// blocks wholly below the window were overwritten with NaN K and V (a forward of NaN rows over
/// them) is finite and bit-identical to one after a zero-filled arena, at m = 512 (P = 1500) and
/// m = 1 (P = 1100); five forwards allocate nothing; repeats are bit-identical.
#[test]
fn gemma_attn_sliding_layer_invariants() {
    let device = h100();
    let tapes = gemma_attn_buckets();
    let w = ga_weights();
    let weights = ga_table_buffers(&device, &w);
    let mut worker = CudaWorker::new(&device, tapes, weights, GA_POOL, None).expect("worker");
    let table = ga_table();
    let fill = tapes.len() - 1;
    let bytes = tapes[fill].arena.total_bytes;
    let mut pass = true;
    for (b, p, len) in [(1usize, 1500usize, 512usize), (0, 1100, 1)] {
        ga_prefill(&mut worker, p, &table);
        fill_arena(&mut worker, fill, bytes, 0);
        let clean = ga_forward(&mut worker, b, (p, len), &table, None);
        // The blocks wholly below the lowest window floor of the forward.
        let below = (p + 1).saturating_sub(GA_W) / 64;
        ga_forward(&mut worker, 1, (0, below * 64), &table, Some(0x7FC0));
        fill_arena(&mut worker, fill, bytes, 0x7FC0);
        let poisoned = ga_forward(&mut worker, b, (p, len), &table, None);
        let repeat = ga_forward(&mut worker, b, (p, len), &table, None);
        let finite = all_finite_bf16(&poisoned);
        let (poison, rep) = (
            mismatched_bits(&poisoned, &clean),
            mismatched_bits(&repeat, &poisoned),
        );
        let ok = finite && poison == 0 && rep == 0 && below > 0;
        eprintln!(
            "gemma sliding layer m={} P={p} L={len}: {below} blocks below the window NaN, arena NaN: \
             nan_poison_vs_zero_mismatched_bits={poison} finite={finite} \
             repeat_mismatched_bits={rep} {}",
            tapes[b].bucket_m,
            verdict(ok)
        );
        pass &= ok;
    }
    let at = device.allocations();
    let first = [
        ga_forward(&mut worker, 1, (1500, 512), &table, None),
        ga_forward(&mut worker, 0, (1100, 1), &table, None),
    ];
    let mut identical = true;
    for b in [1usize, 0, 1, 0, 1] {
        let y = match b {
            1 => ga_forward(&mut worker, 1, (1500, 512), &table, None),
            _ => ga_forward(&mut worker, 0, (1100, 1), &table, None),
        };
        identical &= y == first[usize::from(b == 0)];
    }
    let during = device.allocations() - at;
    eprintln!(
        "gemma sliding layer five forwards (512, 1, 512, 1, 512): allocations={during} (expect 0) \
         results_bit_identical={identical} {}",
        verdict(during == 0 && identical)
    );
    pass &= during == 0 && identical;
    assert!(
        pass,
        "a Gemma sliding layer invariant failed: see the lines above"
    );
}

/// Each bf16 of `y` is finite.
fn all_finite_bf16(y: &[u16]) -> bool {
    y.iter()
        .all(|&b| half::bf16::from_bits(b).to_f32().is_finite())
}
