// SPDX-License-Identifier: Apache-2.0
//! Whole-model prefill and decode forwards of qwen3.5-0.8b's emitted CUDA tapes on the H100.
//!
//! The tapes are `CUDA_TAPES`, exactly as `#[forward]` emitted them; the weights are synthetic,
//! each buffer sized from the tapes' `weight_bytes` and filled by what reads it: a GEMM's `[N, K]`
//! matrix uniform in ±1/√K, a norm's zero-centred gain in ±0.1, a GDN layer's packed f32 bundle
//! in the ranges the GDN kernel tests use, the embedding table's rows uniform in ±1/32 (which the
//! tied lm_head reads too, a `[vocab, hidden]` matrix, so it is filled as a matrix of `k =
//! hidden`: ±1/√1024). The input is token ids, uniform over the vocabulary and seeded; padding rows are id
//! 0. Checks:
//!
//! - (a) the worker takes every tape, and its arena is the largest tape's;
//! - (b) a fresh 300-token forward in the m=512 bucket gives finite logits on every real row;
//! - (c) the lm_head GEMM, read back from the arena, against an f64 reference on six rows over
//!   every column, under the GEMM bound of `numerics.rs`;
//! - (d) repeating (b) in other KV blocks and another GDN slot gives the same logits, bit for
//!   bit;
//! - (e) the same 300 rows in the m=1024 bucket give the same real-row logits, bit for bit;
//! - (f) a 512-token sequence in one forward, and as 300 fresh tokens then 212 continuing, agree
//!   bit for bit on rows 300..512;
//!
//! Each compared pair of (d), (e) and (f) runs its second sequence in KV blocks and a GDN slot
//! disjoint from the first's, so a run that failed to write its cache or state cannot pass by
//! reading the other's.
//! - (g) a KV sequence and a GDN sequence that disagree on fresh are refused before any copy;
//! - (h) the worker allocates the weights, the arena and the two pools, and a forward nothing;
//! - (i) a token id outside the vocabulary, a count other than the bucket's and hidden rows are
//!   refused before any copy;
//! - (j) the worker takes all five tapes (the m=1 decode tape among them), its arena the largest's,
//!   266 + 3 allocations at setup and none across one prefill and 8 decodes;
//! - (k) 300 tokens prefilled, then decodes of positions 300..307, each against the same row of one
//!   308-token forward: finite, cosine ≥ 0.999 (a bound, not bits: the decode runs the GEMV and
//!   the decode attention, the prefill the tiled GEMM and the prefill attention); the last
//!   decode's lm_head GEMV within the GEMM bound of f64 over every column;
//! - (l) the prefill and 8 decodes repeated in disjoint blocks and slot are bit-identical;
//! - (m) the 8 decodes leave the cache at positions 0..300 bit-unchanged;
//! - (n) decodes at P = 319 and P = 320 (a new block) are finite and within (k)'s cosine;
//! - (o) at m=1, a wrong position, a short block table, fresh at P > 0 and q_len = 2 are refused
//!   before any copy.
//!
//! Built only under `nvcc` (`required-features`), and needs an sm_90 GPU: no GPU fails rather
//! than skips. Every check prints its line before a test fails, so one run reports every number.

// `pub`: this binary uses part of `common`, and nothing in it is dead code.
#[path = "../../../targets/cuda/tests/common/numerics.rs"]
pub mod numerics;

use std::sync::OnceLock;

use half::bf16;
use numerics::{bf16_inputs, compare, f32_inputs, f32_to_halves};
use scratchy_models::qwen3_5::qwen3_5_0_8b::CUDA_TAPES;
use scratchy_target_cuda::runtime::{
    CudaDevice, CudaWorker, DeviceBuffer, ForwardError, GdnPoolConfig, GdnSequence, Input, KvHalf,
    KvPoolConfig, KvSequence,
};
use scratchy_target_cuda::step::{
    CudaStep, EmbedBf16, GatedDeltaNetBf16, GemmBf16, gdn_bundle, gdn_conv_dim,
};
use scratchy_target_cuda::tape::{ArenaSpan, CudaTape};

const VOCAB: usize = 248_320;
const HIDDEN: usize = 1024;
/// The real rows of the one-bucket forwards.
const Q: u32 = 300;
/// The rows (c) checks the lm_head on.
const TAIL_ROWS: [usize; 6] = [0, 37, 128, 255, 298, 299];
const KV_POOL: KvPoolConfig = KvPoolConfig { num_blocks: 16 };
const GDN_POOL: GdnPoolConfig = GdnPoolConfig { num_slots: 2 };

fn verdict(pass: bool) -> &'static str {
    if pass { "PASS" } else { "FAIL" }
}

/// The bucket index of `m` in [`CUDA_TAPES`].
fn bucket(m: u32) -> usize {
    (CUDA_TAPES.iter().position(|t| t.bucket_m == m)).unwrap_or_else(|| panic!("no m={m} tape"))
}

/// What reads each weight source, which decides how it is filled.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Role {
    /// A GEMM's `[n, k]` bf16 matrix.
    Matrix { k: u32 },
    /// An RmsNorm's bf16 gain.
    Gain,
    /// A GDN layer's packed f32 bundle.
    Bundle {
        conv_dim: u32,
        conv_kernel: u32,
        nv: u32,
        head_v: u32,
    },
}

/// Every weight source's role, read off the tapes' steps; each source has one.
fn roles() -> Vec<Option<Role>> {
    let n = CUDA_TAPES[0].weight_bytes.len();
    let mut roles: Vec<Option<Role>> = vec![None; n];
    let mut set = |ix: u32, role: Role| {
        let slot = &mut roles[ix as usize];
        assert!(
            slot.is_none_or(|r| r == role),
            "weight {ix}: {slot:?} and {role:?}"
        );
        *slot = Some(role);
    };
    for tape in CUDA_TAPES {
        for step in tape.steps {
            match *step {
                // The table the tied lm_head reads too: a `[vocab, hidden]` matrix.
                CudaStep::Embed(EmbedBf16 { table, hidden, .. }) => {
                    set(table.0, Role::Matrix { k: hidden })
                }
                CudaStep::Gemm(GemmBf16 { b, k, .. })
                | CudaStep::GemmNarrow(GemmBf16 { b, k, .. })
                | CudaStep::Gemv(GemmBf16 { b, k, .. }) => set(b.0, Role::Matrix { k }),
                CudaStep::RmsNorm(n) => set(n.gain.0, Role::Gain),
                CudaStep::GatedDeltaNet(GatedDeltaNetBf16 {
                    bundle,
                    num_k_heads,
                    num_v_heads,
                    head_k,
                    head_v,
                    conv_kernel,
                    ..
                }) => set(
                    bundle.0,
                    Role::Bundle {
                        conv_dim: gdn_conv_dim(num_k_heads, head_k, num_v_heads, head_v)
                            .expect("conv_dim fits"),
                        conv_kernel,
                        nv: num_v_heads,
                        head_v,
                    },
                ),
                _ => {}
            }
        }
    }
    roles
}

/// `len` bf16 uniform in `[-scale, scale]`.
fn scaled(len: usize, seed: u64, scale: f32) -> Vec<u16> {
    bf16_inputs(len, seed)
        .into_iter()
        .map(|b| numerics::bits(bf16::from_bits(b).to_f32() * scale))
        .collect()
}

/// The host bytes of weight `ix`, as the u16 halves `DeviceBuffer` uploads: `bytes` long.
fn weight(ix: usize, role: Role, bytes: u64) -> Vec<u16> {
    let len = bytes as usize / 2;
    let seed = 0x0812_0000 ^ ix as u64;
    match role {
        Role::Matrix { k } => scaled(len, seed, 1.0 / (k as f32).sqrt()),
        Role::Gain => scaled(len, seed, 0.1),
        Role::Bundle {
            conv_dim,
            conv_kernel,
            nv,
            head_v,
        } => {
            let l = gdn_bundle(conv_dim, conv_kernel, nv, head_v).expect("the bundle fits");
            assert_eq!(l.bytes, bytes, "weight {ix}: the bundle's bytes");
            let mut f = vec![0f32; l.bytes as usize / 4];
            let parts = [
                (l.conv1d, (conv_dim * conv_kernel) as usize, (-0.5, 0.5)),
                (l.a_log, nv as usize, (-1.0, 0.5)),
                (l.dt_bias, nv as usize, (-0.5, 0.5)),
                (l.norm, head_v as usize, (0.5, 1.5)),
            ];
            for (i, (at, n, range)) in parts.into_iter().enumerate() {
                let part = f32_inputs(n, seed ^ ((i as u64 + 1) << 32), range);
                f[at as usize / 4..][..n].copy_from_slice(&part);
            }
            f32_to_halves(&f)
        }
    }
}

/// The synthetic weights on the host, built once: per source, the largest any tape declares;
/// `None` for a source that is not a weight.
fn host_weights() -> &'static [Option<Vec<u16>>] {
    static HOST: OnceLock<Vec<Option<Vec<u16>>>> = OnceLock::new();
    HOST.get_or_init(|| {
        let roles = roles();
        (0..roles.len())
            .map(|ix| {
                let bytes = (CUDA_TAPES.iter())
                    .map(|t| t.weight_bytes[ix])
                    .max()
                    .unwrap_or(0);
                match (bytes, roles[ix]) {
                    (0, _) => None,
                    (_, Some(role)) => Some(weight(ix, role, bytes)),
                    (_, None) => panic!("weight {ix} declares {bytes} bytes and no step reads it"),
                }
            })
            .collect()
    })
}

/// One test's device and weight table, and the device's allocation count before the weights.
struct Rig {
    device: CudaDevice,
    table: Vec<DeviceBuffer>,
    before_weights: u64,
}

fn rig() -> Rig {
    let device = CudaDevice::open()
        .unwrap_or_else(|e| panic!("the `nvcc` feature's tests need an sm_90 GPU (H100): {e}"));
    let before_weights = device.allocations();
    let table = (host_weights().iter().enumerate())
        .map(|(ix, w)| match w {
            None => DeviceBuffer::empty(),
            Some(w) => DeviceBuffer::upload(&device, w)
                .unwrap_or_else(|e| panic!("upload weight {ix}: {e}")),
        })
        .collect();
    Rig {
        device,
        table,
        before_weights,
    }
}

impl Rig {
    /// A worker over every tape, with the KV and GDN pools.
    fn worker(&self) -> CudaWorker<'_> {
        CudaWorker::new(
            &self.device,
            CUDA_TAPES,
            &self.table,
            Some(KV_POOL),
            Some(GDN_POOL),
        )
        .unwrap_or_else(|e| panic!("CudaWorker::new: {e}"))
    }
}

/// The token ids of a `tokens`-token sequence, uniform over the vocabulary, seeded.
fn sequence_rows(tokens: usize) -> Vec<u32> {
    let mut state = 0x0816_E3B0u64;
    (0..tokens)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((state >> 33) % VOCAB as u64) as u32
        })
        .collect()
}

/// `ids` padded to a `tape`'s bucket: the padding rows are id 0.
fn padded(tape: &CudaTape, ids: &[u32]) -> Vec<u32> {
    let mut x = vec![0u32; tape.bucket_m as usize];
    x[..ids.len()].copy_from_slice(ids);
    x
}

/// Positions `from..from + q`, then `from + q..` for the padding rows: one per bucket row.
fn positions(tape: &CudaTape, from: u32) -> Vec<u32> {
    (from..from + tape.bucket_m).collect()
}

/// A sequence's place in the pools: its block table and its GDN slot. The two halves of the
/// 16-block pool and the two slots are disjoint, so the second run of a compared pair (d, e, f)
/// cannot read what the first wrote: a chunk whose K/V or state went missing shows.
#[derive(Clone, Copy)]
struct Seat {
    blocks: &'static [u32; 8],
    slot: u32,
}

/// Blocks 0..8 and slot 0; each is enough for 512 tokens of 64.
const SEAT_A: Seat = Seat {
    blocks: &[0, 1, 2, 3, 4, 5, 6, 7],
    slot: 0,
};
/// Blocks 8..16 and slot 1: no block or slot of [`SEAT_A`]'s.
const SEAT_B: Seat = Seat {
    blocks: &[8, 9, 10, 11, 12, 13, 14, 15],
    slot: 1,
};

/// One forward of `rows` (`q` real rows of token ids) at `prefix_len`, in `seat`: the full
/// result.
fn forward(
    worker: &mut CudaWorker<'_>,
    m: u32,
    rows: &[u32],
    prefix_len: u32,
    seat: Seat,
) -> Result<Vec<u16>, ForwardError> {
    let b = bucket(m);
    let tape = &CUDA_TAPES[b];
    let q = rows.len() as u32;
    let slot = seat.slot;
    let kv = KvSequence {
        block_table: seat.blocks,
        prefix_len,
        q_len: q,
    };
    let gdn = GdnSequence {
        slot,
        q_len: q,
        fresh: prefix_len == 0,
    };
    let mut out = vec![0u16; tape.result.bytes as usize / 2];
    worker.forward(
        b,
        Input::TokenIds(&padded(tape, rows)),
        &positions(tape, prefix_len),
        Some(&kv),
        Some(&gdn),
        &mut out,
    )?;
    Ok(out)
}

/// The first `q` rows of a logits result.
fn real(logits: &[u16], q: usize) -> &[u16] {
    &logits[..q * VOCAB]
}

/// How two logits results (`[rows, VOCAB]`) differ: the elements that differ in any bit, the
/// first such, the row with the most, and the largest |difference| (as f32).
struct Difference {
    mismatched: usize,
    rows: usize,
    first: (usize, usize),
    worst_row: (usize, usize),
    widest: f32,
}

fn difference(a: &[u16], b: &[u16]) -> Option<Difference> {
    let first = a.iter().zip(b).position(|(x, y)| x != y)?;
    let per_row: Vec<usize> = (a.chunks(VOCAB).zip(b.chunks(VOCAB)))
        .map(|(x, y)| x.iter().zip(y).filter(|(p, q)| p != q).count())
        .collect();
    let (row, &count) = (per_row.iter().enumerate())
        .max_by_key(|&(_, c)| *c)
        .expect("a non-empty result");
    let widest = (a.iter().zip(b))
        .map(|(&x, &y)| (bf16::from_bits(x).to_f32() - bf16::from_bits(y).to_f32()).abs())
        .fold(0f32, f32::max);
    Some(Difference {
        mismatched: per_row.iter().sum(),
        rows: per_row.iter().filter(|&&c| c > 0).count(),
        first: (first / VOCAB, first % VOCAB),
        worst_row: (row, count),
        widest,
    })
}

fn bit_identical(what: &str, a: &[u16], b: &[u16]) -> bool {
    match difference(a, b) {
        None => {
            eprintln!("{what}: bit-identical {}", verdict(true));
            true
        }
        Some(d) => {
            eprintln!(
                "{what}: mismatched_bits={}/{} in {} of {} rows, first at row {} col {}, worst \
                 row {} ({} elements), largest |difference| {:e} {}",
                d.mismatched,
                a.len(),
                d.rows,
                a.len() / VOCAB,
                d.first.0,
                d.first.1,
                d.worst_row.0,
                d.worst_row.1,
                d.widest,
                verdict(false)
            );
            false
        }
    }
}

/// (b): the fresh 300-token forward in the m=512 bucket, in `seat`.
fn fresh_300(worker: &mut CudaWorker<'_>, seat: Seat) -> Vec<u16> {
    let rows = sequence_rows(Q as usize);
    forward(worker, 512, &rows, 0, seat).expect("the 300-token forward")
}

#[test]
fn a_the_worker_takes_every_tape() {
    let r = rig();
    let w = r.worker();
    let arena = w.arena_bytes();
    let largest = CUDA_TAPES.iter().map(|t| t.arena.total_bytes).max();
    eprintln!(
        "(a) tapes={} arena_bytes={arena} (expect 2585035776, the largest tape's) kv_pool={:?} \
         gdn_pool={:?} {}",
        CUDA_TAPES.len(),
        w.kv_pool_bytes(),
        w.gdn_pool_bytes(),
        verdict(arena == 2_585_035_776 && Some(arena) == largest)
    );
    assert_eq!(arena, 2_585_035_776, "the worker's arena");
}

#[test]
fn b_a_fresh_forward_gives_finite_logits() {
    let r = rig();
    let logits = fresh_300(&mut r.worker(), SEAT_A);
    let real = real(&logits, Q as usize);
    let non_finite = real
        .iter()
        .filter(|&&b| !bf16::from_bits(b).to_f32().is_finite())
        .count();
    let largest = (real.iter())
        .map(|&b| bf16::from_bits(b).to_f32().abs())
        .fold(0f32, f32::max);
    eprintln!(
        "(b) m=512 q_len=300: non-finite logits on real rows={non_finite} (expect 0), largest \
         |logit|={largest:e} (stop and report above 1e4) {}",
        verdict(non_finite == 0 && largest <= 1e4)
    );
    assert_eq!(non_finite, 0, "non-finite logits on a real row");
    assert!(
        largest <= 1e4,
        "the largest |logit| is {largest:e}, above 1e4"
    );
}

#[test]
fn c_the_lm_head_against_an_f64_reference() {
    let r = rig();
    let mut worker = r.worker();
    let logits = fresh_300(&mut worker, SEAT_A);
    let tape = &CUDA_TAPES[bucket(512)];
    // The last step is the lm_head GEMM; its `a` is the final norm's output, still in the arena.
    let Some(CudaStep::Gemm(GemmBf16 { a, b, c, m, n, k })) = tape.steps.last().copied() else {
        panic!("the tape does not end with a Gemm");
    };
    assert_eq!(
        (m, n, k),
        (512, VOCAB as u32, HIDDEN as u32),
        "the lm_head's shape"
    );
    assert_eq!(c, tape.result.offset, "the lm_head writes the result");
    let mut normed = vec![0u16; m as usize * HIDDEN];
    worker
        .read_arena(
            ArenaSpan {
                offset: a,
                bytes: (normed.len() * 2) as u64,
            },
            &mut normed,
        )
        .expect("read the final norm's output");
    let w = host_weights()[b.0 as usize]
        .as_ref()
        .expect("the lm_head's weight");
    let rows: Vec<u16> = (TAIL_ROWS.iter())
        .flat_map(|&r| normed[r * HIDDEN..][..HIDDEN].iter().copied())
        .collect();
    let gpu: Vec<u16> = (TAIL_ROWS.iter())
        .flat_map(|&r| logits[r * VOCAB..][..VOCAB].iter().copied())
        .collect();
    let cmp = compare(&gpu, &rows, w, TAIL_ROWS.len(), VOCAB, HIDDEN);
    let (wi, ..) = cmp.worst;
    eprintln!(
        "(c) lm_head rows {TAIL_ROWS:?} x {VOCAB} columns: {} (worst row is {}) {}",
        cmp.line(),
        TAIL_ROWS[wi],
        verdict(cmp.pass())
    );
    assert!(
        cmp.pass(),
        "{} lm_head elements out of bound",
        cmp.out_of_bound
    );
}

#[test]
fn d_repeating_the_forward_is_bit_identical() {
    let r = rig();
    let mut worker = r.worker();
    let first = fresh_300(&mut worker, SEAT_A);
    let again = fresh_300(&mut worker, SEAT_B);
    let ok = bit_identical(
        "(d) the 300-token forward repeated in disjoint blocks and slot",
        real(&first, Q as usize),
        real(&again, Q as usize),
    );
    assert!(ok, "a repeated forward differs");
}

#[test]
fn e_the_bucket_does_not_change_the_real_rows() {
    let r = rig();
    let mut worker = r.worker();
    let in_512 = fresh_300(&mut worker, SEAT_A);
    let rows = sequence_rows(Q as usize);
    let in_1024 = forward(&mut worker, 1024, &rows, 0, SEAT_B).expect("the m=1024 forward");
    let ok = bit_identical(
        "(e) 300 rows in m=1024 against m=512",
        real(&in_1024, Q as usize),
        real(&in_512, Q as usize),
    );
    assert!(
        ok,
        "the m=1024 bucket's real rows differ from the m=512 bucket's"
    );
}

#[test]
fn f_a_chunked_sequence_matches_one_shot() {
    let r = rig();
    let mut w = r.worker();
    let rows = sequence_rows(512);
    let one_shot = forward(&mut w, 512, &rows, 0, SEAT_A).expect("512 tokens in one forward");
    // The chunked run in the other blocks and slot: its continuation reads only what its own
    // first chunk wrote.
    forward(&mut w, 512, &rows[..Q as usize], 0, SEAT_B).expect("the first 300 tokens");
    let rest = forward(&mut w, 512, &rows[Q as usize..], Q, SEAT_B).expect("the last 212");
    let tail = 512 - Q as usize;
    let ok = bit_identical(
        "(f) rows 300..512: 300 fresh then 212 continuing, against one 512-token forward",
        &rest[..tail * VOCAB],
        &one_shot[Q as usize * VOCAB..512 * VOCAB],
    );
    assert!(ok, "the chunked sequence differs from the one-shot one");
}

#[test]
fn g_kv_and_gdn_must_agree_on_fresh() {
    let r = rig();
    let mut worker = r.worker();
    let tape = &CUDA_TAPES[bucket(512)];
    let x = padded(tape, &sequence_rows(Q as usize));
    let mut pass = true;
    for (prefix_len, fresh) in [(0, false), (64, true)] {
        let kv = KvSequence {
            block_table: SEAT_A.blocks,
            prefix_len,
            q_len: Q,
        };
        let gdn = GdnSequence {
            slot: 0,
            q_len: Q,
            fresh,
        };
        let mut out = vec![0u16; tape.result.bytes as usize / 2];
        let got = worker.forward(
            bucket(512),
            Input::TokenIds(&x),
            &positions(tape, prefix_len),
            Some(&kv),
            Some(&gdn),
            &mut out,
        );
        let want = Err(ForwardError::FreshDisagrees { prefix_len, fresh });
        let untouched = out.iter().all(|&b| b == 0);
        let ok = got == want && untouched;
        eprintln!(
            "(g) prefix_len={prefix_len} fresh={fresh}: {got:?} result_untouched={untouched} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(
        pass,
        "a KV/GDN disagreement was not refused before any copy"
    );
}

#[test]
fn h_the_worker_allocates_once() {
    let r = rig();
    let after_weights = r.device.allocations();
    let mut worker = r.worker();
    let after_worker = r.device.allocations();
    let weights = after_weights - r.before_weights;
    let worker_allocs = after_worker - after_weights;
    for _ in 0..5 {
        let _ = fresh_300(&mut worker, SEAT_A);
    }
    let forwards = r.device.allocations() - after_worker;
    let ok = weights == 266 && worker_allocs == 3 && forwards == 0;
    eprintln!(
        "(h) allocations: weights={weights} (expect 266) worker={worker_allocs} (expect 3: \
         arena, KV pool, GDN pool) after five forwards={forwards} (expect 0) {}",
        verdict(ok)
    );
    assert!(ok, "the allocation counts are wrong");
}

/// The forward's token ids are checked before any copy: an id at or above the vocabulary, a count
/// other than the bucket's rows, and hidden rows for a tape that takes ids are refused, and the
/// result is untouched.
#[test]
fn i_token_ids_are_checked_before_any_copy() {
    let r = rig();
    let mut worker = r.worker();
    let tape = &CUDA_TAPES[bucket(512)];
    let mut ids = padded(tape, &sequence_rows(Q as usize));
    ids[17] = VOCAB as u32;
    let kv = KvSequence {
        block_table: SEAT_A.blocks,
        prefix_len: 0,
        q_len: Q,
    };
    let gdn = GdnSequence {
        slot: 0,
        q_len: Q,
        fresh: true,
    };
    let hidden = vec![0u16; 512 * HIDDEN];
    let short = &ids[..511];
    let cases: [(&str, Input<'_>, ForwardError); 3] = [
        (
            "id = vocab",
            Input::TokenIds(&ids),
            ForwardError::TokenOutOfVocab {
                row: 17,
                id: VOCAB as u32,
                vocab: VOCAB as u32,
            },
        ),
        (
            "511 ids",
            Input::TokenIds(short),
            ForwardError::InputSize {
                expected: 2048,
                got: 2044,
            },
        ),
        (
            "hidden rows",
            Input::Hidden(&hidden),
            ForwardError::InputKind {
                expected: "token ids",
                got: "hidden rows",
            },
        ),
    ];
    let mut pass = true;
    for (what, input, want) in cases {
        let mut out = vec![0u16; tape.result.bytes as usize / 2];
        let got = worker.forward(
            bucket(512),
            input,
            &positions(tape, 0),
            Some(&kv),
            Some(&gdn),
            &mut out,
        );
        let untouched = out.iter().all(|&b| b == 0);
        let ok = got == Err(want.clone()) && untouched;
        eprintln!(
            "(i) {what}: {got:?} result_untouched={untouched} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "a bad token-id input was not refused before any copy");
}

// ── Decode: one token at a time through the m=1 tape ───────────────

/// The decodes (k) through (n) run after their prefill.
const DECODES: u32 = 8;

/// The cosine of two bf16 rows, in f64.
fn cosine(a: &[u16], b: &[u16]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (
            f64::from(bf16::from_bits(x).to_f32()),
            f64::from(bf16::from_bits(y).to_f32()),
        );
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    ab / (aa.sqrt() * bb.sqrt())
}

/// The largest |a − b| over two bf16 rows, as f32.
fn max_diff(a: &[u16], b: &[u16]) -> f32 {
    (a.iter().zip(b))
        .map(|(&x, &y)| (bf16::from_bits(x).to_f32() - bf16::from_bits(y).to_f32()).abs())
        .fold(0f32, f32::max)
}

fn all_finite(row: &[u16]) -> bool {
    row.iter().all(|&b| bf16::from_bits(b).to_f32().is_finite())
}

/// A fresh `prefill`-token forward in the m=512 bucket, then one decode per token of `decodes`
/// (one row, the m=1 tape) at positions `prefill, prefill + 1, …`, in `seat`: each decode's
/// logits, `[VOCAB]`.
fn prefill_then_decode(
    worker: &mut CudaWorker<'_>,
    (prefill, decodes): (&[u32], &[u32]),
    seat: Seat,
) -> Vec<Vec<u16>> {
    forward(worker, 512, prefill, 0, seat).expect("the prefill");
    (decodes.iter().enumerate())
        .map(|(i, &token)| {
            let p = (prefill.len() + i) as u32;
            forward(worker, 1, &[token], p, seat).unwrap_or_else(|e| panic!("decode at {p}: {e}"))
        })
        .collect()
}

/// Every KV layer's K and V of `seat`'s first `tokens` positions, read back from the pool:
/// per layer and half, `[tokens, kv_heads · head_dim]` bf16.
fn cached_rows(worker: &mut CudaWorker<'_>, seat: Seat, tokens: usize) -> Vec<Vec<u16>> {
    let kv = CUDA_TAPES[bucket(1)].kv.expect("a KV cache");
    let (heads, bs, hd) = (
        kv.kv_heads as usize,
        kv.block_size as usize,
        kv.head_dim as usize,
    );
    let mut out = Vec::new();
    for layer in 0..kv.layers.len() {
        for half in [KvHalf::K, KvHalf::V] {
            let blocks: Vec<Vec<u16>> = (seat.blocks[..tokens.div_ceil(bs)].iter())
                .map(|&b| {
                    let mut block = vec![0u16; heads * bs * hd];
                    worker.read_kv(layer, half, b, &mut block).expect("read_kv");
                    block
                })
                .collect();
            let mut rows = Vec::with_capacity(tokens * heads * hd);
            for t in 0..tokens {
                for h in 0..heads {
                    rows.extend_from_slice(&blocks[t / bs][(h * bs + t % bs) * hd..][..hd]);
                }
            }
            out.push(rows);
        }
    }
    out
}

/// (j): the worker takes all five tapes (the m=1 decode tape among them) in the arena of the
/// largest; the weights, the arena and the two pools are the setup's 266 + 3 allocations, and one
/// prefill and eight decodes allocate nothing.
#[test]
fn j_the_decode_tape_allocates_nothing() {
    let r = rig();
    let after_weights = r.device.allocations();
    let mut worker = r.worker();
    let after_worker = r.device.allocations();
    let (weights, setup) = (
        after_weights - r.before_weights,
        after_worker - after_weights,
    );
    let ms: Vec<u32> = CUDA_TAPES.iter().map(|t| t.bucket_m).collect();
    let arena = worker.arena_bytes();
    let rows = sequence_rows((Q + DECODES) as usize);
    let (prefill, decodes) = rows.split_at(Q as usize);
    let _ = prefill_then_decode(&mut worker, (prefill, decodes), SEAT_A);
    let during = r.device.allocations() - after_worker;
    let ok = ms == [1, 512, 1024, 2048, 4096]
        && arena == 2_585_035_776
        && weights == 266
        && setup == 3
        && during == 0;
    eprintln!(
        "(j) tapes {ms:?} arena_bytes={arena} (expect 2585035776) allocations weights={weights} \
         (expect 266) worker={setup} (expect 3) across one prefill and {DECODES} decodes={during} \
         (expect 0) {}",
        verdict(ok)
    );
    assert!(
        ok,
        "the decode worker's tapes, arena or allocations are wrong"
    );
}

/// (k): 300 fresh tokens prefilled in m=512, then the decodes of positions 300..307 one at a
/// time, against rows 300..307 of one 308-token m=512 forward in disjoint blocks and slot: two
/// different kernel paths (the GEMV and the decode attention against the tiled GEMM and the
/// prefill attention), so a bound, not bits: every logit finite, cosine ≥ 0.999 per row, the max
/// |difference| printed. And the last decode's lm_head GEMV, read back from the arena, within
/// #761's bound of f64 over every column.
#[test]
fn k_decodes_agree_with_one_shot_rows() {
    let r = rig();
    let mut worker = r.worker();
    let rows = sequence_rows((Q + DECODES) as usize);
    let (prefill, decodes) = rows.split_at(Q as usize);
    let decoded = prefill_then_decode(&mut worker, (prefill, decodes), SEAT_A);
    // The last decode's lm_head: its `a` is the final norm's one row, still in the arena.
    let tape = &CUDA_TAPES[bucket(1)];
    let Some(CudaStep::Gemv(GemmBf16 { a, b, c, m, n, k })) = tape.steps.last().copied() else {
        panic!("the m=1 tape does not end with a Gemv");
    };
    assert_eq!((m, n, k), (1, VOCAB as u32, HIDDEN as u32), "the lm_head");
    assert_eq!(c, tape.result.offset, "the lm_head writes the result");
    let mut normed = vec![0u16; HIDDEN];
    worker
        .read_arena(
            ArenaSpan {
                offset: a,
                bytes: (HIDDEN * 2) as u64,
            },
            &mut normed,
        )
        .expect("read the final norm's row");
    let w = host_weights()[b.0 as usize]
        .as_ref()
        .expect("the lm_head's weight");
    let last = decoded.last().expect("eight decodes");
    let cmp = compare(last, &normed, w, 1, VOCAB, HIDDEN);
    let mut pass = cmp.pass();
    eprintln!(
        "(k) the last decode's lm_head GEMV over {VOCAB} columns: {} {}",
        cmp.line(),
        verdict(cmp.pass())
    );
    let one_shot = forward(&mut worker, 512, &rows, 0, SEAT_B).expect("308 tokens in one forward");
    for (i, row) in decoded.iter().enumerate() {
        let at = Q as usize + i;
        let want = &one_shot[at * VOCAB..][..VOCAB];
        let (cos, diff, finite) = (cosine(row, want), max_diff(row, want), all_finite(row));
        let ok = finite && cos >= 0.999;
        eprintln!(
            "(k) decode at position {at} against row {at} of one 308-token forward: \
             finite={finite} cosine={cos:.6} (need >= 0.999) max|diff|={diff:e} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(
        pass,
        "a decode is out of agreement with the one-shot row: see the lines above"
    );
}

/// (l): the prefill and eight decodes, repeated in disjoint KV blocks and GDN slot, give the same
/// eight rows of logits, bit for bit.
#[test]
fn l_repeated_decodes_are_bit_identical() {
    let r = rig();
    let mut worker = r.worker();
    let rows = sequence_rows((Q + DECODES) as usize);
    let (prefill, decodes) = rows.split_at(Q as usize);
    let first = prefill_then_decode(&mut worker, (prefill, decodes), SEAT_A);
    let again = prefill_then_decode(&mut worker, (prefill, decodes), SEAT_B);
    let ok = bit_identical(
        "(l) a prefill and 8 decodes repeated in disjoint blocks and slot",
        &first.concat(),
        &again.concat(),
    );
    assert!(ok, "repeated decodes differ");
}

/// (m): the eight decodes leave every cache row at positions 0..300, K and V of every KV layer,
/// bit-unchanged.
#[test]
fn m_decodes_leave_the_prefix_cache_unchanged() {
    let r = rig();
    let mut worker = r.worker();
    let rows = sequence_rows((Q + DECODES) as usize);
    let (prefill, decodes) = rows.split_at(Q as usize);
    forward(&mut worker, 512, prefill, 0, SEAT_A).expect("the prefill");
    let before = cached_rows(&mut worker, SEAT_A, Q as usize);
    for (i, &token) in decodes.iter().enumerate() {
        let p = Q + i as u32;
        forward(&mut worker, 1, &[token], p, SEAT_A).unwrap_or_else(|e| panic!("decode {p}: {e}"));
    }
    let after = cached_rows(&mut worker, SEAT_A, Q as usize);
    let changed: usize = (before.iter().zip(&after))
        .map(|(a, b)| a.iter().zip(b).filter(|(x, y)| x != y).count())
        .sum();
    let ok = changed == 0;
    eprintln!(
        "(m) the cache at positions 0..{Q} (K and V of {} layers) after {DECODES} decodes: \
         changed elements={changed} {}",
        before.len() / 2,
        verdict(ok)
    );
    assert!(ok, "a decode wrote over the prefix's cache");
}

/// (n): a decode at P = 319 (kv_len 320, the last slot of block 4) and at P = 320 (kv_len 321, the
/// first slot of a new block), each finite and within (k)'s cosine of the same row of one
/// 321-token forward in disjoint blocks and slot.
#[test]
fn n_decodes_at_a_block_boundary() {
    let r = rig();
    let mut worker = r.worker();
    let rows = sequence_rows(321);
    let decoded = prefill_then_decode(&mut worker, (&rows[..319], &rows[319..]), SEAT_A);
    let one_shot = forward(&mut worker, 512, &rows, 0, SEAT_B).expect("321 tokens in one forward");
    let mut pass = true;
    for (i, row) in decoded.iter().enumerate() {
        let at = 319 + i;
        let want = &one_shot[at * VOCAB..][..VOCAB];
        let (cos, diff, finite) = (cosine(row, want), max_diff(row, want), all_finite(row));
        let ok = finite && cos >= 0.999;
        eprintln!(
            "(n) decode at P={at} (kv_len {}) against row {at} of one 321-token forward: \
             finite={finite} cosine={cos:.6} (need >= 0.999) max|diff|={diff:e} {}",
            at + 1,
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "a decode at the block boundary is out of agreement");
}

/// (o): at m=1, a position other than `prefix_len`, a block table too short for `P + 1`, a GDN
/// sequence called fresh at `P > 0`, and `q_len = 2` are each refused before any copy, the
/// result untouched.
#[test]
fn o_a_bad_decode_is_refused_before_any_copy() {
    let r = rig();
    let mut worker = r.worker();
    let b = bucket(1);
    let blocks = SEAT_A.blocks;
    let p = 64u32;
    let kv = |block_table, q_len| KvSequence {
        block_table,
        prefix_len: p,
        q_len,
    };
    let gdn = |fresh, q_len| GdnSequence {
        slot: SEAT_A.slot,
        q_len,
        fresh,
    };
    let cases: [(&str, KvSequence<'_>, GdnSequence, u32, ForwardError); 4] = [
        (
            "position P + 1",
            kv(blocks, 1),
            gdn(false, 1),
            p + 1,
            ForwardError::PositionsDisagree {
                row: 0,
                got: p + 1,
                want: u64::from(p),
            },
        ),
        (
            "a one-block table for 65 tokens",
            kv(&blocks[..1], 1),
            gdn(false, 1),
            p,
            ForwardError::KvBlockTable {
                got: 1,
                need: 2,
                max: 128,
            },
        ),
        (
            "fresh at P = 64",
            kv(blocks, 1),
            gdn(true, 1),
            p,
            ForwardError::FreshDisagrees {
                prefix_len: p,
                fresh: true,
            },
        ),
        (
            "q_len = 2",
            kv(blocks, 2),
            gdn(false, 2),
            p,
            ForwardError::KvQueryLength {
                q_len: 2,
                bucket_m: 1,
            },
        ),
    ];
    let mut pass = true;
    for (what, kv, gdn, position, want) in cases {
        let mut out = vec![0u16; VOCAB];
        let got = worker.forward(
            b,
            Input::TokenIds(&[7]),
            &[position],
            Some(&kv),
            Some(&gdn),
            &mut out,
        );
        let untouched = out.iter().all(|&x| x == 0);
        let ok = got == Err(want.clone()) && untouched;
        eprintln!(
            "(o) m=1 {what}: {got:?} result_untouched={untouched} {}",
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(pass, "a bad decode was not refused before any copy");
}

// ── (k) diagnostic: where the decode chain leaves the one-shot trajectory ──

/// Layer `layer`'s K and V of position `pos` in `seat`'s blocks: `[kv_heads · head_dim]` bf16
/// each, for every KV layer.
fn kv_at(worker: &mut CudaWorker<'_>, seat: Seat, pos: usize) -> Vec<[Vec<u16>; 2]> {
    let kv = CUDA_TAPES[bucket(1)].kv.expect("a KV cache");
    let (heads, bs, hd) = (
        kv.kv_heads as usize,
        kv.block_size as usize,
        kv.head_dim as usize,
    );
    let block = seat.blocks[pos / bs];
    (0..kv.layers.len())
        .map(|layer| {
            [KvHalf::K, KvHalf::V].map(|half| {
                let mut b = vec![0u16; heads * bs * hd];
                worker.read_kv(layer, half, block, &mut b).expect("read_kv");
                (0..heads)
                    .flat_map(|h| b[(h * bs + pos % bs) * hd..][..hd].to_vec())
                    .collect()
            })
        })
        .collect()
}

/// Every GDN layer's conv ring and recurrent state in `slot`: `[conv, ssm]` f32 each.
fn gdn_at(worker: &mut CudaWorker<'_>, slot: u32) -> Vec<[Vec<f32>; 2]> {
    let g = CUDA_TAPES[bucket(1)].gdn.expect("GDN layers");
    let conv_dim = gdn_conv_dim(g.num_k_heads, g.head_k, g.num_v_heads, g.head_v).expect("fits");
    let conv = conv_dim as usize * (g.conv_kernel as usize - 1);
    let ssm = (g.num_v_heads * g.head_v * g.head_k) as usize;
    (0..g.layers.len())
        .map(|layer| {
            let (mut c, mut s) = (vec![0f32; conv], vec![0f32; ssm]);
            worker
                .read_gdn_state(layer, slot, &mut c, &mut s)
                .expect("read_gdn_state");
            [c, s]
        })
        .collect()
}

/// `‖a − b‖₂ / ‖b‖₂` and `max |a − b|`, in f64; infinite when either holds a non-finite value.
fn rel(a: impl Iterator<Item = f64> + Clone, b: impl Iterator<Item = f64> + Clone) -> (f64, f64) {
    let (mut d2, mut b2, mut max) = (0f64, 0f64, 0f64);
    for (x, y) in a.zip(b) {
        if !(x.is_finite() && y.is_finite()) {
            return (f64::INFINITY, f64::INFINITY);
        }
        d2 += (x - y) * (x - y);
        b2 += y * y;
        max = max.max((x - y).abs());
    }
    ((d2 / b2.max(f64::MIN_POSITIVE)).sqrt(), max)
}

fn rel_bf16(a: &[u16], b: &[u16]) -> (f64, f64) {
    let w = |v: &[u16]| {
        v.iter()
            .map(|&x| f64::from(bf16::from_bits(x).to_f32()))
            .collect::<Vec<_>>()
    };
    let (a, b) = (w(a), w(b));
    rel(a.iter().copied(), b.iter().copied())
}

fn rel_f32(a: &[f32], b: &[f32]) -> (f64, f64) {
    rel(
        a.iter().map(|&x| f64::from(x)),
        b.iter().map(|&x| f64::from(x)),
    )
}

/// One trajectory's state after a decode at one position: every KV layer's `[K, V]` at that
/// position, every GDN layer's `[conv ring, recurrent state]`, and the logits.
struct Snapshot {
    kv: Vec<[Vec<u16>; 2]>,
    gdn: Vec<[Vec<f32>; 2]>,
    logits: Vec<u16>,
}

fn snapshot(worker: &mut CudaWorker<'_>, seat: Seat, pos: usize, logits: Vec<u16>) -> Snapshot {
    Snapshot {
        kv: kv_at(worker, seat, pos),
        gdn: gdn_at(worker, seat.slot),
        logits,
    }
}

/// One trajectory's difference from the one-shot's at one step: per KV layer K and V at the
/// step's position, per GDN layer the conv ring and the recurrent state, as relative L2 `‖Δ‖/‖ref‖`
/// (the largest, and its layer, printed with every layer's), and the logits' cosine and max
/// |difference|.
fn report(what: &str, step: usize, got: &Snapshot, want: &Snapshot) -> f64 {
    let line = |kind: &str, v: Vec<(f64, f64)>| {
        let (worst, at) = (v.iter().enumerate())
            .map(|(i, &(r, _))| (r, i))
            .fold((0f64, 0usize), |a, b| if b.0 > a.0 { b } else { a });
        let all: Vec<String> = v.iter().map(|(r, _)| format!("{r:.1e}")).collect();
        eprintln!(
            "(k-diag) {what} step {step} {kind}: worst rel L2 {worst:.3e} at layer index {at}, \
             per layer [{}]",
            all.join(" ")
        );
    };
    for (half, name) in [(0, "K"), (1, "V")] {
        line(
            name,
            (got.kv.iter().zip(&want.kv))
                .map(|(g, w)| rel_bf16(&g[half], &w[half]))
                .collect(),
        );
    }
    for (part, name) in [(0, "GDN conv ring"), (1, "GDN recurrent state")] {
        line(
            name,
            (got.gdn.iter().zip(&want.gdn))
                .map(|(g, w)| rel_f32(&g[part], &w[part]))
                .collect(),
        );
    }
    let (g, w) = (&got.logits, &want.logits);
    let cos = cosine(g, w);
    eprintln!(
        "(k-diag) {what} step {step} logits: cosine={cos:.6} max|diff|={:e} rel L2 {:.3e}",
        max_diff(g, w),
        rel_bf16(g, w).0
    );
    cos
}

/// (k)'s diagnostic. The chain (300 prefilled, then decodes at 300..307, each reading the K/V and
/// GDN state earlier decodes wrote) and a teacher-forced variant (for each i, 300 + i tokens
/// prefilled fresh, then one decode at 300 + i: one step off the one-shot trajectory), each
/// compared after its decode at 300 + i with a fresh one-shot forward of 301 + i tokens (in
/// disjoint blocks and slot): the K and V the decode wrote at 300 + i in every KV layer, every
/// GDN layer's conv ring and recurrent state, and the logits. The one-shot of 301 + i tokens is
/// checked bit-identical, on its last row, to row 300 + i of one 308-token forward, so it is the
/// one-shot trajectory (k) compares against. Asserted: every value finite, the references
/// bit-identical, and every teacher-forced decode's logits at cosine ≥ 0.999 (one step's
/// difference, which is (k)'s criterion for one step); the chain's numbers are printed.
#[test]
fn k_diag_decode_drift_against_one_shot() {
    let r = rig();
    let mut worker = r.worker();
    let rows = sequence_rows((Q + DECODES) as usize);
    let one_shot = forward(&mut worker, 512, &rows, 0, SEAT_B).expect("308 tokens in one forward");
    let reference = |worker: &mut CudaWorker<'_>, i: usize| {
        let n = Q as usize + i + 1;
        let out = forward(worker, 512, &rows[..n], 0, SEAT_B).expect("the one-shot reference");
        let last = out[(n - 1) * VOCAB..][..VOCAB].to_vec();
        snapshot(worker, SEAT_B, n - 1, last)
    };
    let mut pass = true;
    let mut refs_identical = true;
    // The chain.
    forward(&mut worker, 512, &rows[..Q as usize], 0, SEAT_A).expect("the prefill");
    let mut chain_cos = Vec::new();
    for i in 0..DECODES as usize {
        let p = Q as usize + i;
        let logits = forward(&mut worker, 1, &[rows[p]], p as u32, SEAT_A).expect("a decode");
        pass &= all_finite(&logits);
        let got = snapshot(&mut worker, SEAT_A, p, logits);
        let want = reference(&mut worker, i);
        refs_identical &= want.logits[..] == one_shot[p * VOCAB..][..VOCAB];
        chain_cos.push(report("chain", i, &got, &want));
    }
    // Teacher-forced: each decode one step off the one-shot trajectory.
    let mut forced_cos = Vec::new();
    for i in 0..DECODES as usize {
        let p = Q as usize + i;
        forward(&mut worker, 512, &rows[..p], 0, SEAT_A).expect("the teacher-forced prefill");
        let logits = forward(&mut worker, 1, &[rows[p]], p as u32, SEAT_A).expect("a decode");
        pass &= all_finite(&logits);
        let got = snapshot(&mut worker, SEAT_A, p, logits);
        let want = reference(&mut worker, i);
        refs_identical &= want.logits[..] == one_shot[p * VOCAB..][..VOCAB];
        forced_cos.push(report("teacher-forced", i, &got, &want));
    }
    let forced_ok = forced_cos.iter().all(|&c| c >= 0.999);
    eprintln!(
        "(k-diag) logits cosine per step: chain {chain_cos:.6?} teacher-forced {forced_cos:.6?}; \
         references bit-identical to the 308-token forward's rows={refs_identical} \
         teacher-forced all >= 0.999={forced_ok} {}",
        verdict(pass && refs_identical && forced_ok)
    );
    assert!(
        refs_identical,
        "a one-shot reference differs from the 308-token forward's row"
    );
    assert!(pass, "a decode gave a non-finite logit");
    assert!(
        forced_ok,
        "a teacher-forced decode is below cosine 0.999: one step is wrong"
    );
}
