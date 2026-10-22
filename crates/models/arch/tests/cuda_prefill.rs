// SPDX-License-Identifier: Apache-2.0
//! One whole-model prefill forward of qwen3.5-0.8b's emitted CUDA tapes on the H100.
//!
//! The tapes are `CUDA_TAPES`, exactly as `#[forward]` emitted them; the weights are synthetic,
//! each buffer sized from the tapes' `weight_bytes` and filled by what reads it: a GEMM's `[N, K]`
//! matrix uniform in ±1/√K, a norm's zero-centred gain in ±0.1, a GDN layer's packed f32 bundle
//! in the ranges the GDN kernel tests use. The input is the embedded hidden, rows uniform in ±1
//! (embedding is not a tape step yet). Checks:
//!
//! - (a) the worker takes all four tapes, and its arena is the largest tape's;
//! - (b) a fresh 300-token forward in the m=512 bucket gives finite logits on every real row;
//! - (c) the lm_head GEMM, read back from the arena, against an f64 reference on six rows over
//!   every column, under the GEMM bound of `numerics.rs`;
//! - (d) repeating (b) on the same slot gives the same logits, bit for bit;
//! - (e) the same 300 rows in the m=1024 bucket give the same real-row logits, bit for bit;
//! - (f) a 512-token sequence in one forward, and as 300 fresh tokens then 212 continuing, agree
//!   bit for bit on rows 300..512;
//! - (g) a KV sequence and a GDN sequence that disagree on fresh are refused before any copy;
//! - (h) the worker allocates the weights, the arena and the two pools, and a forward nothing.
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
    CudaDevice, CudaWorker, DeviceBuffer, ForwardError, GdnPoolConfig, GdnSequence, KvPoolConfig,
    KvSequence,
};
use scratchy_target_cuda::step::{CudaStep, GatedDeltaNetBf16, GemmBf16, gdn_bundle, gdn_conv_dim};
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
                CudaStep::Gemm(GemmBf16 { b, k, .. })
                | CudaStep::GemmNarrow(GemmBf16 { b, k, .. }) => set(b.0, Role::Matrix { k }),
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

/// The embedded hidden of a `tokens`-token sequence, rows uniform in ±1.
fn sequence_rows(tokens: usize) -> Vec<u16> {
    bf16_inputs(tokens * HIDDEN, 0x0812_E3B0)
}

/// `rows` of hidden padded to a `tape`'s bucket: the padding rows are zero.
fn padded(tape: &CudaTape, rows: &[u16]) -> Vec<u16> {
    let mut x = vec![0u16; tape.input.bytes as usize / 2];
    x[..rows.len()].copy_from_slice(rows);
    x
}

/// Positions `from..from + q`, then `from + q..` for the padding rows: one per bucket row.
fn positions(tape: &CudaTape, from: u32) -> Vec<u32> {
    (from..from + tape.bucket_m).collect()
}

/// The block table every forward here uses: blocks 0..8, enough for 512 tokens of 64.
const BLOCKS: [u32; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

/// One forward of `rows` (`q` real rows) at `prefix_len`, on GDN slot `slot`: the full result.
fn forward(
    worker: &mut CudaWorker<'_>,
    m: u32,
    rows: &[u16],
    prefix_len: u32,
    slot: u32,
) -> Result<Vec<u16>, ForwardError> {
    let b = bucket(m);
    let tape = &CUDA_TAPES[b];
    let q = (rows.len() / HIDDEN) as u32;
    let kv = KvSequence {
        block_table: &BLOCKS,
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
        &padded(tape, rows),
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

/// The first element at which `a` and `b` differ, and the largest |difference| (as f32).
fn first_difference(a: &[u16], b: &[u16]) -> Option<(usize, usize, f32)> {
    let first = a.iter().zip(b).position(|(x, y)| x != y)?;
    let widest = (a.iter().zip(b))
        .map(|(&x, &y)| (bf16::from_bits(x).to_f32() - bf16::from_bits(y).to_f32()).abs())
        .fold(0f32, f32::max);
    Some((first / VOCAB, first % VOCAB, widest))
}

fn bit_identical(what: &str, a: &[u16], b: &[u16]) -> bool {
    match first_difference(a, b) {
        None => {
            eprintln!("{what}: bit-identical {}", verdict(true));
            true
        }
        Some((row, col, widest)) => {
            eprintln!(
                "{what}: first differs at row {row} col {col}, largest |difference| {widest:e} {}",
                verdict(false)
            );
            false
        }
    }
}

/// (b): the fresh 300-token forward in the m=512 bucket, on GDN slot 0.
fn fresh_300(worker: &mut CudaWorker<'_>) -> Vec<u16> {
    let rows = sequence_rows(Q as usize);
    forward(worker, 512, &rows, 0, 0).expect("the 300-token forward")
}

#[test]
fn a_the_worker_takes_every_tape() {
    let r = rig();
    let w = r.worker();
    let arena = w.arena_bytes();
    let largest = CUDA_TAPES.iter().map(|t| t.arena.total_bytes).max();
    eprintln!(
        "(a) tapes={} arena_bytes={arena} (expect 2585019392, the largest tape's) kv_pool={:?} \
         gdn_pool={:?} {}",
        CUDA_TAPES.len(),
        w.kv_pool_bytes(),
        w.gdn_pool_bytes(),
        verdict(arena == 2_585_019_392 && Some(arena) == largest)
    );
    assert_eq!(arena, 2_585_019_392, "the worker's arena");
}

#[test]
fn b_a_fresh_forward_gives_finite_logits() {
    let r = rig();
    let logits = fresh_300(&mut r.worker());
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
    let logits = fresh_300(&mut worker);
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
    let first = fresh_300(&mut worker);
    let again = fresh_300(&mut worker);
    let ok = bit_identical(
        "(d) the 300-token forward repeated on slot 0",
        real(&first, Q as usize),
        real(&again, Q as usize),
    );
    assert!(ok, "a repeated forward differs");
}

#[test]
fn e_the_bucket_does_not_change_the_real_rows() {
    let r = rig();
    let mut worker = r.worker();
    let in_512 = fresh_300(&mut worker);
    let rows = sequence_rows(Q as usize);
    let in_1024 = forward(&mut worker, 1024, &rows, 0, 0).expect("the m=1024 forward");
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
    let one_shot = forward(&mut w, 512, &rows, 0, 0).expect("512 tokens in one forward");
    forward(&mut w, 512, &rows[..Q as usize * HIDDEN], 0, 1).expect("the first 300 tokens");
    let rest = forward(&mut w, 512, &rows[Q as usize * HIDDEN..], Q, 1).expect("the last 212");
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
            block_table: &BLOCKS,
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
            &x,
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
    let nonzero = (0..CUDA_TAPES[0].weight_bytes.len())
        .filter(|&ix| CUDA_TAPES.iter().any(|t| t.weight_bytes[ix] > 0))
        .count() as u64;
    for _ in 0..5 {
        let _ = fresh_300(&mut worker);
    }
    let forwards = r.device.allocations() - after_worker;
    let ok = weights == nonzero && worker_allocs == 3 && forwards == 0;
    eprintln!(
        "(h) allocations: weights={weights} (expect {nonzero}) worker={worker_allocs} (expect 3: \
         arena, KV pool, GDN pool) after five forwards={forwards} (expect 0) {}",
        verdict(ok)
    );
    assert!(ok, "the allocation counts are wrong");
}
