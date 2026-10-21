// SPDX-License-Identifier: Apache-2.0
//! The tape player on the H100: the GEMM-chain tapes of `common::chains` played through the arena
//! and compared with an f64 CPU reference, one arena allocation per worker, no allocation per
//! forward, colour reuse, and the refusals.
//!
//! Built only under `nvcc` (`required-features`); no GPU fails rather than skips. Each stage is
//! checked against its own GPU input (`common::numerics`' bound), so the bound never compounds
//! along a chain.

mod common {
    pub mod chains;
    pub mod numerics;
}

use common::chains::{
    HIDDEN, THREE_GEMM_M256, TWO_GEMM_M128, TWO_GEMM_M256, assert_tape_is_lowered, three_gemm,
    two_gemm,
};
use common::numerics::{bf16_inputs, compare};
use scratchy_target_cuda::runtime::{
    CudaDevice, CudaWorker, DeviceBuffer, ForwardError, WorkerError,
};
use scratchy_target_cuda::tape::{ArenaOffset, ArenaSpan, CudaTape};

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
    let mut worker = CudaWorker::new(&device, &TWO_GEMM_BUCKETS, &table).expect("worker");
    let mut pass = true;
    for (bucket, tape) in TWO_GEMM_BUCKETS.iter().enumerate() {
        let m = tape.bucket_m as usize;
        let x = input(m);
        let mut y2 = vec![0u16; m * H];
        worker.forward(bucket, &x, &mut y2).expect("forward");
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
    let mut worker = CudaWorker::new(&device, &TWO_GEMM_BUCKETS, &table).expect("worker");
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
            .forward(bucket, &inputs[bucket], &mut y2)
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
    let mut worker = CudaWorker::new(&device, two, &table[..3]).expect("two-GEMM worker");
    let mut two_y2 = vec![0u16; 256 * H];
    worker
        .forward(0, &x, &mut two_y2)
        .expect("two-GEMM forward");
    drop(worker);

    let three = std::slice::from_ref(&THREE_GEMM_M256);
    let mut worker = CudaWorker::new(&device, three, &table).expect("three-GEMM worker");
    let mut y3 = vec![0u16; 256 * WIDE];
    worker.forward(0, &x, &mut y3).expect("three-GEMM forward");
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
    let mut worker = CudaWorker::new(&device, &TWO_GEMM_BUCKETS, &table).expect("worker");
    let x = input(256);
    let mut y2 = vec![0u16; 256 * H];
    let got = worker.forward(1, &x[..x.len() - 1], &mut y2);
    let want = Err(ForwardError::InputSize {
        expected: 262144,
        got: 262142,
    });
    eprintln!("input one element short: {got:?} {}", verdict(got == want));
    assert_eq!(got, want);
    worker
        .forward(1, &x, &mut y2)
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
    let got = CudaWorker::new(&device, &TWO_GEMM_BUCKETS, &short).map(|_| ());
    let want = Err(WorkerError::WeightTooSmall {
        ix: 1,
        need: 1048576,
        have: 1048574,
    });
    eprintln!("W1 one element short: {got:?} {}", verdict(got == want));
    assert_eq!(got, want);
}
