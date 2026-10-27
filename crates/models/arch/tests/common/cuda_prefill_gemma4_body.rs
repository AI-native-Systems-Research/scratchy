// SPDX-License-Identifier: Apache-2.0
// The body of Gemma 4 26B-A4B's `cuda_prefill_gemma4` binary (AI-native-Systems-Research/
// scratchy#925): the qwen body's checks (a) to (o) where they apply, at the qwen body's thresholds
// unless a check says otherwise, plus the sliding window's (p) to (r). Not the qwen body
// (`cuda_prefill_body.rs`): that one hard-codes qwen's vocabulary, a GDN pool and a 16-block KV
// pool the window never bites in, and editing it would change the qwen binaries. Included with
// `include!` at the binary's root, not as a module, so every test keeps its name. The including
// binary supplies the model: its `CUDA_TAPES`, the `numerics` module
// (`targets/cuda/tests/common/numerics.rs`), and the constants `HIDDEN`, `VOCAB`, `WEIGHTS` and
// `WEIGHT_BYTES` (the weight buffers the rig uploads), `ARENA` (the largest tape's arena bytes),
// `KV_POOL_BYTES`, `LAYERS`, `WINDOW`, `SLIDING_LAYERS` and `CAP` (the final soft-cap).
//
// Decode against one-shot rows ((k), (n), (r)) is printed and gated only on finiteness: #859's
// decision on decode's reduction order past 64 keys, or a floor for MoE models, is the user's.

use std::collections::BTreeSet;
use std::sync::OnceLock;

use half::bf16;
use numerics::{Comparison, bf16_in, bf16_inputs, reference, soft_cap_bound, widen};
use scratchy_target_cuda::runtime::{
    CudaDevice, CudaWorker, DeviceBuffer, ForwardError, GdnSequence, Input, KvHalf, KvPoolConfig,
    KvSequence,
};
use scratchy_target_cuda::step::{
    AttnPrefillBf16, CudaStep, EmbedBf16, ExpertGemmBf16, ExpertGemvBf16, GainConvention, GemmBf16,
    LayerScaleBf16, MoeRouteBf16, MoeWeight, RmsNormBf16, RouterLogitsBf16, expert_bank,
};
use scratchy_target_cuda::tape::{ArenaSpan, CudaTape, KvClass, KvTape};

/// The real rows of the one-bucket forwards.
const Q: u32 = 300;
/// The rows (c) checks the lm_head on.
const TAIL_ROWS: [usize; 6] = [0, 37, 128, 255, 298, 299];
/// 72 blocks of 64 tokens: two seats of 2,304 tokens, enough for (p)'s 2,048 and (r)'s 1,033.
const KV_POOL: KvPoolConfig = KvPoolConfig { num_blocks: 72 };
/// Each seat's blocks.
const SEAT_BLOCKS: usize = 36;

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
    /// A GEMM's `[n, k]` bf16 matrix: every projection, the router's, and the embedding table
    /// (the tied lm_head reads it as one too).
    Matrix { k: u32 },
    /// An RMSNorm's stored bf16 gain (`GainConvention::Scale`).
    Gain,
    /// The router pre-norm's gain, `router.scale`.
    RouterScale,
    /// The routing's `router.per_expert_scale`, `[experts]`.
    PerExpertScale,
    /// A layer's `layer_scalar`, `[1]`.
    LayerScalar,
    /// A layer's expert bank (`step::expert_bank`): every expert's fused gate and up, then every
    /// expert's down.
    ExpertBank {
        experts: u32,
        hidden: u32,
        inter: u32,
    },
}

/// Every weight source's role, read off the tapes' steps; each source has one. Every step kind is
/// named: a Qwen bundle, a zero-centred gain or a GDN layer in a Gemma tape panics.
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
                CudaStep::Embed(EmbedBf16 { table, hidden, .. }) => {
                    set(table.0, Role::Matrix { k: hidden })
                }
                CudaStep::Gemm(GemmBf16 { b, k, .. })
                | CudaStep::GemmNarrow(GemmBf16 { b, k, .. })
                | CudaStep::GemmCol64(GemmBf16 { b, k, .. })
                | CudaStep::Gemv(GemmBf16 { b, k, .. }) => set(b.0, Role::Matrix { k }),
                CudaStep::RmsNorm(RmsNormBf16 {
                    gain, convention, ..
                }) => match convention {
                    GainConvention::Scale => set(gain.0, Role::Gain),
                    GainConvention::OnePlusScale => {
                        panic!("weight {}: a zero-centred gain", gain.0)
                    }
                },
                CudaStep::RouterNorm(RmsNormBf16 { gain, .. }) => set(gain.0, Role::RouterScale),
                CudaStep::LayerScale(LayerScaleBf16 { scale, .. }) => {
                    set(scale.0, Role::LayerScalar)
                }
                CudaStep::MoeRoute(MoeRouteBf16 { expert_scale, .. }) => match expert_scale {
                    Some(ix) => set(ix.0, Role::PerExpertScale),
                    None => panic!("a Qwen routing in a Gemma tape"),
                },
                CudaStep::RouterLogits(RouterLogitsBf16 {
                    bundle,
                    weight,
                    hidden,
                    ..
                }) => match weight {
                    MoeWeight::Own => set(bundle.0, Role::Matrix { k: hidden }),
                    MoeWeight::Bundle => panic!("weight {}: a Qwen MoE bundle", bundle.0),
                },
                CudaStep::ExpertGemm(ExpertGemmBf16 {
                    bundle,
                    weight,
                    experts,
                    hidden,
                    inter,
                    ..
                })
                | CudaStep::ExpertGemv(ExpertGemvBf16 {
                    bundle,
                    weight,
                    experts,
                    hidden,
                    inter,
                    ..
                }) => match weight {
                    MoeWeight::Own => set(
                        bundle.0,
                        Role::ExpertBank {
                            experts,
                            hidden,
                            inter,
                        },
                    ),
                    MoeWeight::Bundle => panic!("weight {}: a Qwen MoE bundle", bundle.0),
                },
                CudaStep::GatedDeltaNet(_) => panic!("a GDN layer in a Gemma tape"),
                // The steps that read no weight.
                CudaStep::Add(_)
                | CudaStep::SiluMul(_)
                | CudaStep::Rope(_)
                | CudaStep::GateSplit(_)
                | CudaStep::GateApply(_)
                | CudaStep::RopeAppend(_)
                | CudaStep::AttnPrefill(_)
                | CudaStep::AttnDecode(_)
                | CudaStep::ExpertSort(_)
                | CudaStep::ExpertCombine(_)
                | CudaStep::GateScale(_)
                | CudaStep::Scale(_)
                | CudaStep::RmsNormUnit(_)
                | CudaStep::GeluMul(_)
                | CudaStep::SoftCap(_) => {}
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
    let seed = 0x0925_0000 ^ ix as u64;
    match role {
        Role::Matrix { k } => scaled(len, seed, 1.0 / (k as f32).sqrt()),
        Role::Gain | Role::RouterScale => bf16_in(len, seed, (0.9, 1.1)),
        Role::PerExpertScale => bf16_in(len, seed, (0.5, 2.0)),
        Role::LayerScalar => bf16_in(len, seed, (0.5, 1.0)),
        // The gate and up rows uniform in ±1/√hidden, the down rows in ±1/√inter, each part from
        // its own seed.
        Role::ExpertBank {
            experts,
            hidden,
            inter,
        } => {
            let l = expert_bank(experts, hidden, inter).expect("the expert bank fits");
            assert_eq!(l.bytes, bytes, "weight {ix}: the expert bank's bytes");
            let parts = [(l.down - l.gate_up, hidden), (l.bytes - l.down, inter)];
            let mut out = Vec::with_capacity(len);
            for (i, (part, k)) in parts.into_iter().enumerate() {
                let seed = seed ^ ((i as u64 + 1) << 32);
                out.extend(scaled(part as usize / 2, seed, 1.0 / (k as f32).sqrt()));
            }
            out
        }
    }
}

/// The synthetic weights on the host, built once, on up to 4 threads (each weight from its own
/// seed, so the bytes do not depend on the split): per source, the largest any tape declares;
/// `None` for a source that is not a weight.
fn host_weights() -> &'static [Option<Vec<u16>>] {
    static HOST: OnceLock<Vec<Option<Vec<u16>>>> = OnceLock::new();
    HOST.get_or_init(|| {
        let roles = roles();
        let bytes: Vec<u64> = (0..roles.len())
            .map(|ix| {
                (CUDA_TAPES.iter())
                    .map(|t| t.weight_bytes[ix])
                    .max()
                    .unwrap_or(0)
            })
            .collect();
        let mut out: Vec<Option<Vec<u16>>> = vec![None; roles.len()];
        let threads = std::thread::available_parallelism().map_or(1, |p| p.get().min(4));
        let per = out.len().div_ceil(threads);
        std::thread::scope(|s| {
            for (c, chunk) in out.chunks_mut(per).enumerate() {
                let (roles, bytes) = (&roles, &bytes);
                s.spawn(move || {
                    for (i, slot) in chunk.iter_mut().enumerate() {
                        let ix = c * per + i;
                        *slot = match (bytes[ix], roles[ix]) {
                            (0, _) => None,
                            (b, Some(role)) => Some(weight(ix, role, b)),
                            (b, None) => {
                                panic!("weight {ix} declares {b} bytes and no step reads it")
                            }
                        };
                    }
                });
            }
        });
        out
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
    /// A worker over every tape, with the KV pool and no GDN pool (Gemma has no GDN layer).
    fn worker(&self) -> CudaWorker<'_> {
        self.worker_over(CUDA_TAPES)
    }

    /// A worker over `tapes` (the window-off copies, or the tapes and their cuts), on the same
    /// weight buffers.
    fn worker_over(&self, tapes: &'static [CudaTape]) -> CudaWorker<'_> {
        CudaWorker::new(&self.device, tapes, &self.table, Some(KV_POOL), None)
            .unwrap_or_else(|e| panic!("CudaWorker::new: {e}"))
    }
}

/// The token ids of a `tokens`-token sequence, uniform over the vocabulary, seeded.
fn sequence_rows(tokens: usize) -> Vec<u32> {
    let mut state = 0x0925_E3B0u64;
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

/// A sequence's place in the KV pool: its block table. The two halves of the 72-block pool are
/// disjoint, so the second run of a compared pair (d, e, f) cannot read what the first wrote: a
/// chunk whose K/V went missing shows.
#[derive(Clone, Copy)]
struct Seat {
    blocks: &'static [u32],
}

const fn seat_blocks(from: u32) -> [u32; SEAT_BLOCKS] {
    let mut b = [0; SEAT_BLOCKS];
    let mut i = 0;
    while i < SEAT_BLOCKS {
        b[i] = from + i as u32;
        i += 1;
    }
    b
}

const BLOCKS_A: [u32; SEAT_BLOCKS] = seat_blocks(0);
const BLOCKS_B: [u32; SEAT_BLOCKS] = seat_blocks(SEAT_BLOCKS as u32);
/// Blocks 0..36: 2,304 tokens of 64.
const SEAT_A: Seat = Seat { blocks: &BLOCKS_A };
/// Blocks 36..72: no block of [`SEAT_A`]'s.
const SEAT_B: Seat = Seat { blocks: &BLOCKS_B };

/// One forward of `rows` (token ids, `rows.len()` real rows) at `prefix_len`, in `seat`, through
/// tape `b` of `tapes` (the worker's own list): the full result.
fn run(
    worker: &mut CudaWorker<'_>,
    (tapes, b): (&[CudaTape], usize),
    rows: &[u32],
    prefix_len: u32,
    seat: Seat,
) -> Result<Vec<u16>, ForwardError> {
    let tape = &tapes[b];
    let kv = KvSequence {
        block_table: seat.blocks,
        prefix_len,
        q_len: rows.len() as u32,
    };
    let mut out = vec![0u16; tape.result.bytes as usize / 2];
    worker.forward(
        b,
        Input::TokenIds(&padded(tape, rows)),
        &positions(tape, prefix_len),
        Some(&kv),
        None,
        &mut out,
    )?;
    Ok(out)
}

/// [`run`] through the m-row tape of [`CUDA_TAPES`], which every worker here lists first, in order.
fn forward(
    worker: &mut CudaWorker<'_>,
    m: u32,
    rows: &[u32],
    prefix_len: u32,
    seat: Seat,
) -> Result<Vec<u16>, ForwardError> {
    run(worker, (CUDA_TAPES, bucket(m)), rows, prefix_len, seat)
}

/// The first `q` rows of a logits result.
fn real(logits: &[u16], q: usize) -> &[u16] {
    &logits[..q * VOCAB]
}

/// How two results of `width`-wide rows differ: the elements that differ in any bit, the rows
/// that do, the first such element, the row with the most, and the largest |difference| (as f32).
struct Difference {
    mismatched: usize,
    rows: usize,
    first: (usize, usize),
    worst_row: (usize, usize),
    widest: f32,
}

fn difference(a: &[u16], b: &[u16], width: usize) -> Option<Difference> {
    let first = a.iter().zip(b).position(|(x, y)| x != y)?;
    let per_row: Vec<usize> = (a.chunks(width).zip(b.chunks(width)))
        .map(|(x, y)| x.iter().zip(y).filter(|(p, q)| p != q).count())
        .collect();
    let (row, &count) = (per_row.iter().enumerate())
        .max_by_key(|&(_, c)| *c)
        .expect("a non-empty result");
    Some(Difference {
        mismatched: per_row.iter().sum(),
        rows: per_row.iter().filter(|&&c| c > 0).count(),
        first: (first / width, first % width),
        worst_row: (row, count),
        widest: max_diff(a, b),
    })
}

fn bit_identical(what: &str, a: &[u16], b: &[u16]) -> bool {
    match difference(a, b, VOCAB) {
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

/// The non-finite values of a result and its largest |value|.
fn finite_and_largest(v: &[u16]) -> (usize, f32) {
    let non_finite = (v.iter())
        .filter(|&&b| !bf16::from_bits(b).to_f32().is_finite())
        .count();
    let largest = (v.iter())
        .map(|&b| bf16::from_bits(b).to_f32().abs())
        .fold(0f32, f32::max);
    (non_finite, largest)
}

#[test]
fn a_the_worker_takes_every_tape() {
    let r = rig();
    let host_bytes: u64 = (host_weights().iter().flatten())
        .map(|w| w.len() as u64 * 2)
        .sum();
    let buffers = host_weights().iter().flatten().count() as u64;
    let w = r.worker();
    let arena = w.arena_bytes();
    let largest = CUDA_TAPES.iter().map(|t| t.arena.total_bytes).max();
    let (kv, gdn) = (w.kv_pool_bytes(), w.gdn_pool_bytes());
    let ok = arena == ARENA
        && Some(arena) == largest
        && kv == Some(KV_POOL_BYTES)
        && gdn.is_none()
        && (buffers, host_bytes) == (WEIGHTS, WEIGHT_BYTES);
    eprintln!(
        "(a) tapes={} arena_bytes={arena} (expect {ARENA}, the largest tape's) kv_pool={kv:?} \
         (expect {KV_POOL_BYTES}) gdn_pool={gdn:?} (expect None) weights={buffers} buffers of \
         {host_bytes} bytes (expect {WEIGHTS} of {WEIGHT_BYTES}) {}",
        CUDA_TAPES.len(),
        verdict(ok)
    );
    let (free, total) = r.device.mem_info().expect("cudaMemGetInfo");
    let used = (total - free) as u64;
    let known = WEIGHT_BYTES + ARENA + KV_POOL_BYTES;
    eprintln!(
        "(a) device memory: used={used} of total={total} bytes ({:.2} of {:.2} GiB); weights + \
         arena + KV pool = {known} bytes; the rest (context and fatbin) = {} bytes ({:.3} GiB, \
         #925 reads about 0.6 GiB); printed, not gated",
        used as f64 / f64::from(1u32 << 30),
        total as f64 / f64::from(1u32 << 30),
        used as i64 - known as i64,
        (used as f64 - known as f64) / f64::from(1u32 << 30)
    );
    assert_eq!(arena, ARENA, "the worker's arena");
    assert!(
        ok,
        "the worker's arena, pools or weights are not the predicted ones"
    );
}

#[test]
fn b_a_fresh_forward_gives_finite_capped_logits() {
    let r = rig();
    let logits = fresh_300(&mut r.worker(), SEAT_A);
    let (non_finite, largest) = finite_and_largest(real(&logits, Q as usize));
    let ok = non_finite == 0 && f64::from(largest) <= CAP;
    eprintln!(
        "(b) m=512 q_len=300: non-finite logits on real rows={non_finite} (expect 0), largest \
         |logit|={largest:e} (at most the soft-cap, {CAP}) {}",
        verdict(ok)
    );
    assert_eq!(non_finite, 0, "non-finite logits on a real row");
    assert!(
        f64::from(largest) <= CAP,
        "the largest |logit| is {largest:e}, above the soft-cap {CAP}"
    );
}

/// A tape's lm_head GEMM (or GEMV), the tape's second-last step, and the soft-cap after it, which
/// writes over the GEMM's output, the result, with [`CAP`]. The GEMM's `a` is the final norm's
/// output, still in the arena after the forward: the soft-cap reads and writes only the logits.
fn lm_head(tape: &CudaTape) -> GemmBf16 {
    let [.., head, cap] = tape.steps else {
        panic!("m={}: fewer than two steps", tape.bucket_m);
    };
    let (CudaStep::Gemm(g) | CudaStep::Gemv(g)) = *head else {
        panic!(
            "m={}: the second-last step is not the lm_head",
            tape.bucket_m
        );
    };
    let CudaStep::SoftCap(s) = *cap else {
        panic!("m={}: the last step is not the soft-cap", tape.bucket_m);
    };
    assert_eq!(
        (g.b.0, g.n, g.k),
        (0, VOCAB as u32, HIDDEN as u32),
        "the tied lm_head"
    );
    assert_eq!(
        (s.x, s.out, s.rows, s.cols),
        (g.c, tape.result.offset, g.m, g.n),
        "the soft-cap writes over the lm_head's logits, the result"
    );
    assert_eq!(f64::from(f32::from_bits(s.cap_bits)), CAP, "the cap");
    g
}

/// Compare `gpu` (capped logits, `[m, n]` bf16 bits) with the f64 lm_head `a · bᵀ`, capped in
/// f64: the GEMM's bound interval `[r − e, r + e]` (`numerics::compare`'s `e = |r|/128 + S/16384`)
/// is mapped through `cap · tanh(·/cap)` at both ends, and each end widened by the soft-cap
/// kernel's own bound (`numerics::soft_cap_bound`). The worst element's ratio is its distance
/// from `cap(r)` over the half-width on its side.
fn compare_capped(
    gpu: &[u16],
    a: &[u16],
    b: &[u16],
    (m, n, k): (usize, usize, usize),
) -> Comparison {
    let refs = reference(&widen(a), &widen(b), m, n, k);
    let cap = |x: f64| CAP * (x / CAP).tanh();
    let (mut max_err, mut sum_err, mut bad) = (0.0f64, 0.0f64, 0usize);
    let (mut worst, mut worst_ratio) = ((0, 0, 0.0f64, 0.0f64, 0.0f64), -1.0f64);
    for (idx, (&bits, &(r, s))) in gpu.iter().zip(&refs).enumerate() {
        let g = f64::from(bf16::from_bits(bits).to_f32());
        let e = r.abs() / 128.0 + s / 16384.0;
        let (lo, hi) = (cap(r - e), cap(r + e));
        let (lo, hi) = (lo - soft_cap_bound(lo), hi + soft_cap_bound(hi));
        let c = cap(r);
        let bound = if g >= c { hi - c } else { c - lo };
        let err = (g - c).abs();
        if !(g.is_finite() && lo <= g && g <= hi) {
            bad += 1;
        }
        let ratio = if g.is_finite() {
            err / bound.max(f64::MIN_POSITIVE)
        } else {
            f64::INFINITY
        };
        if ratio > worst_ratio {
            worst_ratio = ratio;
            worst = (idx / n, idx % n, g, c, bound);
        }
        if g.is_finite() {
            max_err = max_err.max(err);
            sum_err += err;
        }
    }
    Comparison {
        max_abs_err: max_err,
        mean_abs_err: sum_err / (m * n) as f64,
        worst,
        out_of_bound: bad,
    }
}

#[test]
fn c_the_lm_head_against_an_f64_reference() {
    let r = rig();
    let mut worker = r.worker();
    let logits = fresh_300(&mut worker, SEAT_A);
    let tape = &CUDA_TAPES[bucket(512)];
    let GemmBf16 { a, b, m, .. } = lm_head(tape);
    assert_eq!(m, 512, "the lm_head's rows");
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
    let cmp = compare_capped(&gpu, &rows, w, (TAIL_ROWS.len(), VOCAB, HIDDEN));
    let (wi, ..) = cmp.worst;
    eprintln!(
        "(c) lm_head and soft-cap, rows {TAIL_ROWS:?} x {VOCAB} columns, the GEMM bound capped \
         in f64 at both ends: {} (worst row is {}) {}",
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
        "(d) the 300-token forward repeated in disjoint blocks",
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

// ── Cut tapes: a layer's output, or its expert picks, read inside the forward ──

/// `tape` cut just after step `last`, its result `out`: a `'static` copy, as
/// `cuda_prefill_35b`'s `cut` (#859). Every span stays in the same arena, so it validates as the
/// full tape does.
fn cut(tape: &CudaTape, last: usize, out: ArenaSpan) -> CudaTape {
    CudaTape {
        steps: Box::leak(tape.steps[..=last].to_vec().into_boxed_slice()),
        result: out,
        ..*tape
    }
}

/// Per layer, `tape` cut just after the layer's last step, its layer scale: the result is the
/// layer's output, `[m, hidden]`.
fn layer_cuts(tape: &CudaTape) -> Vec<CudaTape> {
    let rows = u64::from(tape.bucket_m);
    (tape.steps.iter().enumerate())
        .filter_map(|(i, s)| match *s {
            CudaStep::LayerScale(l) => Some(cut(
                tape,
                i,
                ArenaSpan {
                    offset: l.out,
                    bytes: rows * u64::from(l.cols) * 2,
                },
            )),
            _ => None,
        })
        .collect()
}

/// Per layer, `tape` cut just after the layer's routing: the result is its picks, `topk_idx`,
/// `[m, top_k]` i32.
fn route_cuts(tape: &CudaTape) -> Vec<CudaTape> {
    let moe = tape.moe.expect("a MoE scratch");
    (tape.steps.iter().enumerate())
        .filter(|(_, s)| matches!(s, CudaStep::MoeRoute(_)))
        .map(|(i, _)| cut(tape, i, moe.topk_idx))
        .collect()
}

/// [`CUDA_TAPES`], then each of `ms`'s tapes' `cuts`, [`LAYERS`] per bucket: cut `l` of `ms[i]`
/// is tape `CUDA_TAPES.len() + i · LAYERS + l`.
fn with_cuts(ms: &[u32], cuts: fn(&CudaTape) -> Vec<CudaTape>) -> &'static [CudaTape] {
    let mut list = CUDA_TAPES.to_vec();
    for &m in ms {
        let c = cuts(&CUDA_TAPES[bucket(m)]);
        assert_eq!(c.len(), LAYERS, "m={m}: one cut per layer");
        list.extend(c);
    }
    Box::leak(list.into_boxed_slice())
}

/// The tape index of cut `layer` of the `i`-th bucket a [`with_cuts`] list was made for.
fn cut_ix(i: usize, layer: usize) -> usize {
    CUDA_TAPES.len() + i * LAYERS + layer
}

/// (f)'s and (q)'s diagnostic, run only when a chunked prefill differs from its one-shot: the
/// first layer whose output rows differ. The one-shot is `tokens` fresh at `one_shot_m`; the
/// chunks are `(m, start, end)` in order, the last compared with the one-shot's rows `start..end`.
/// The earlier chunks run once on the full tapes; then, per layer, the one-shot and the last chunk
/// run cut after that layer (a cut forward rewrites the positions it runs with what the full one
/// writes there).
fn first_differing_layer(
    r: &Rig,
    what: &str,
    (one_shot_m, tokens): (u32, &[u32]),
    chunks: &[(u32, usize, usize)],
) -> Option<usize> {
    let (&(last_m, start, end), before) = chunks.split_last().expect("a chunk");
    let tapes = with_cuts(&[one_shot_m, last_m], layer_cuts);
    let mut w = r.worker_over(tapes);
    for &(m, s, e) in before {
        forward(&mut w, m, &tokens[s..e], s as u32, SEAT_A).expect("an earlier chunk");
    }
    for layer in 0..LAYERS {
        let one =
            run(&mut w, (tapes, cut_ix(0, layer)), tokens, 0, SEAT_B).expect("the one-shot, cut");
        let got = run(
            &mut w,
            (tapes, cut_ix(1, layer)),
            &tokens[start..end],
            start as u32,
            SEAT_A,
        )
        .expect("the last chunk, cut");
        let want = &one[start * HIDDEN..end * HIDDEN];
        let got = &got[..(end - start) * HIDDEN];
        if let Some(d) = difference(got, want, HIDDEN) {
            eprintln!(
                "{what} diagnostic: layer {layer}'s output rows {start}..{end} differ first: \
                 mismatched_bits={} in {} rows, first at row {} col {}, largest |difference| {:e}",
                d.mismatched,
                d.rows,
                start + d.first.0,
                d.first.1,
                d.widest
            );
            return Some(layer);
        }
    }
    eprintln!("{what} diagnostic: every layer's output rows {start}..{end} are bit-identical");
    None
}

#[test]
fn f_a_chunked_sequence_matches_one_shot() {
    let r = rig();
    let mut w = r.worker();
    let rows = sequence_rows(512);
    let one_shot = forward(&mut w, 512, &rows, 0, SEAT_A).expect("512 tokens in one forward");
    // The chunked run in the other blocks: its continuation reads only what its own first chunk
    // wrote.
    forward(&mut w, 512, &rows[..Q as usize], 0, SEAT_B).expect("the first 300 tokens");
    let rest = forward(&mut w, 512, &rows[Q as usize..], Q, SEAT_B).expect("the last 212");
    let tail = 512 - Q as usize;
    let ok = bit_identical(
        "(f) rows 300..512: 300 fresh then 212 continuing, against one 512-token forward",
        &rest[..tail * VOCAB],
        &one_shot[Q as usize * VOCAB..512 * VOCAB],
    );
    if !ok {
        drop(w);
        let chunks = [(512, 0, Q as usize), (512, Q as usize, 512)];
        first_differing_layer(&r, "(f)", (512, &rows), &chunks);
    }
    assert!(
        ok,
        "the chunked sequence differs from the one-shot one: stop and report"
    );
}

/// (g) does not apply: Gemma has no GDN layer, so a KV sequence and a GDN sequence cannot
/// disagree on `fresh`. What does: no tape has GDN layers, the worker has no GDN pool, and a
/// forward given a GDN sequence is refused before any copy.
#[test]
fn g_kv_and_gdn_must_agree_on_fresh() {
    let r = rig();
    let mut worker = r.worker();
    eprintln!("(g) n/a: no GDN layer (no KV/GDN fresh disagreement to refuse)");
    let tape = &CUDA_TAPES[bucket(512)];
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
    let mut out = vec![0u16; tape.result.bytes as usize / 2];
    let got = worker.forward(
        bucket(512),
        Input::TokenIds(&padded(tape, &sequence_rows(Q as usize))),
        &positions(tape, 0),
        Some(&kv),
        Some(&gdn),
        &mut out,
    );
    let untouched = out.iter().all(|&b| b == 0);
    let no_gdn = CUDA_TAPES.iter().all(|t| t.gdn.is_none()) && worker.gdn_pool_bytes().is_none();
    let ok = no_gdn && got == Err(ForwardError::GdnUnexpected) && untouched;
    eprintln!(
        "(g) no tape has GDN layers and the worker no GDN pool={no_gdn}; a GDN sequence: {got:?} \
         result_untouched={untouched} {}",
        verdict(ok)
    );
    assert!(
        ok,
        "a GDN sequence for a model with no GDN layer was not refused before any copy"
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
    let ok = weights == WEIGHTS && worker_allocs == 2 && forwards == 0;
    eprintln!(
        "(h) allocations: weights={weights} (expect {WEIGHTS}) worker={worker_allocs} (expect 2: \
         arena, KV pool) after five forwards={forwards} (expect 0) {}",
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
            None,
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

/// The decodes (k) through (m) run after their prefill.
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

/// A fresh `prefill`-token forward in the m-row bucket `m`, then one decode per token of `decodes`
/// (one row, the m=1 tape) at positions `prefill, prefill + 1, …`, in `seat`: each decode's
/// logits, `[VOCAB]`.
fn prefill_then_decode(
    worker: &mut CudaWorker<'_>,
    m: u32,
    (prefill, decodes): (&[u32], &[u32]),
    seat: Seat,
) -> Vec<Vec<u16>> {
    forward(worker, m, prefill, 0, seat).expect("the prefill");
    (decodes.iter().enumerate())
        .map(|(i, &token)| {
            let p = (prefill.len() + i) as u32;
            forward(worker, 1, &[token], p, seat).unwrap_or_else(|e| panic!("decode at {p}: {e}"))
        })
        .collect()
}

/// Every KV layer's K and V of `seat`'s first `tokens` positions, read back from the pool, each
/// layer at its own class's layout: per layer and half, `[tokens, kv_heads · head_dim]` bf16.
fn cached_rows(worker: &mut CudaWorker<'_>, seat: Seat, tokens: usize) -> Vec<Vec<u16>> {
    let kv = CUDA_TAPES[bucket(1)].kv.expect("a KV cache");
    let bs = kv.block_size as usize;
    let mut out = Vec::new();
    for layer in 0..kv.layers.len() {
        let class = kv.classes[usize::from(kv.layer_class[layer])];
        let (heads, hd) = (class.kv_heads as usize, class.head_dim as usize);
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
/// largest; the weights, the arena and the KV pool are the setup's `WEIGHTS` + 2 allocations, and
/// one prefill and eight decodes allocate nothing.
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
    let _ = prefill_then_decode(&mut worker, 512, (prefill, decodes), SEAT_A);
    let during = r.device.allocations() - after_worker;
    let ok = ms == [1, 512, 1024, 2048, 4096]
        && arena == ARENA
        && weights == WEIGHTS
        && setup == 2
        && during == 0;
    eprintln!(
        "(j) tapes {ms:?} arena_bytes={arena} (expect {ARENA}) allocations weights={weights} \
         (expect {WEIGHTS}) worker={setup} (expect 2) across one prefill and {DECODES} \
         decodes={during} (expect 0) {}",
        verdict(ok)
    );
    assert!(
        ok,
        "the decode worker's tapes, arena or allocations are wrong"
    );
}

/// The index of `row`'s largest value (the lowest on a tie) and its margin over the second
/// largest, as f32.
fn top1_margin(row: &[u16]) -> (usize, f32) {
    let v = |i: usize| bf16::from_bits(row[i]).to_f32();
    let (mut first, mut second) = (0usize, None::<usize>);
    for i in 1..row.len() {
        if v(i) > v(first) {
            second = Some(first);
            first = i;
        } else if second.is_none_or(|s| v(i) > v(s)) {
            second = Some(i);
        }
    }
    (first, second.map_or(f32::INFINITY, |s| v(first) - v(s)))
}

/// One decode's logits against the one-shot row of the same position, printed: finiteness,
/// greedy tokens, the one-shot row's top-1 − top-2 margin and whether it is a near-tie (at most
/// twice the max |difference|, #816's P3 rule), the cosine and the max |difference|. Returns
/// `(finite, near-tie)`; only finiteness gates.
fn decode_line(what: &str, row: &[u16], want: &[u16]) -> (bool, bool) {
    let (cos, diff, finite) = (cosine(row, want), max_diff(row, want), all_finite(row));
    let (got_token, _) = top1_margin(row);
    let (want_token, margin) = top1_margin(want);
    let tie = margin <= 2.0 * diff;
    eprintln!(
        "{what}: finite={finite} token={got_token} one-shot token={want_token} same={} one-shot \
         margin={margin:e} (near-tie when <= 2 x max|diff| = {:e}: {tie}) cosine={cos:.6} \
         max|diff|={diff:e} (printed, not gated, pending #859; finiteness gated) {}",
        got_token == want_token,
        2.0 * diff,
        verdict(finite)
    );
    (finite, tie)
}

/// i32 words from the u16 halves a result holds.
fn words(halves: &[u16]) -> Vec<i32> {
    (halves.as_chunks::<2>().0.iter())
        .map(|&[lo, hi]| (u32::from(lo) | u32::from(hi) << 16) as i32)
        .collect()
}

/// Row `row`'s picks, as a set, of a `[rows, top_k]` result.
fn picks(idx: &[i32], row: usize, top_k: usize) -> BTreeSet<i32> {
    idx[row * top_k..][..top_k].iter().copied().collect()
}

/// (k)'s pick scan: the route cuts of the m=512 and m=1 tapes (`with_cuts(&[512, 1],
/// route_cuts)`), the one-shot's picks per layer, and the top-k.
struct PickScan {
    tapes: &'static [CudaTape],
    one_shot: Vec<Vec<i32>>,
    top_k: usize,
}

impl PickScan {
    /// The one-shot picks of `rows` fresh at m=512 in `seat`, per layer, from the route cuts (a
    /// cut forward rewrites the positions it runs with what the full one writes there).
    fn new(
        worker: &mut CudaWorker<'_>,
        tapes: &'static [CudaTape],
        rows: &[u32],
        seat: Seat,
    ) -> Self {
        let one_shot = (0..LAYERS)
            .map(|l| {
                words(&run(worker, (tapes, cut_ix(0, l)), rows, 0, seat).expect("a one-shot cut"))
            })
            .collect();
        let top_k = CUDA_TAPES[bucket(1)].moe.expect("a MoE scratch").top_k as usize;
        Self {
            tapes,
            one_shot,
            top_k,
        }
    }

    /// A decode of `token` at `p` in `seat`, on the cache as it stands, cut after each layer's
    /// routing: the first layer whose picks are not the one-shot row `p`'s, and how many layers'
    /// are not. A cut decode writes position `p` of the layers it runs with what the full decode
    /// writes there, so the scan leaves the cache as it found it but for position `p`.
    fn flips(
        &self,
        worker: &mut CudaWorker<'_>,
        token: u32,
        p: usize,
        seat: Seat,
    ) -> (Option<usize>, usize) {
        let (mut first, mut count) = (None, 0);
        for l in 0..LAYERS {
            let got = run(worker, (self.tapes, cut_ix(1, l)), &[token], p as u32, seat)
                .expect("a cut decode");
            if picks(&words(&got), 0, self.top_k) != picks(&self.one_shot[l], p, self.top_k) {
                first = first.or(Some(l));
                count += 1;
            }
        }
        (first, count)
    }
}

/// (k): 300 fresh tokens prefilled in m=512, then a chain of decodes of positions 300..307 one at a
/// time, each against row 300 + i of one 308-token m=512 forward in disjoint blocks. Per step
/// printed: the decode line ([`decode_line`]) and the first layer whose expert picks are not the
/// one-shot's, with how many are not. Gated: every logit finite, and the last decode's lm_head
/// GEMV and soft-cap, read back from the arena, within (c)'s bound over every column.
///
/// The cosine is printed, not gated at 0.999 or any floor: #859 found a decode equal to the
/// prefill row bit for bit only up to 64 keys (the D = 256 decode's even/odd tile split), and on
/// synthetic MoE weights a near-equal router logit then flips a top-8 pick. Whether decode's
/// reduction order becomes the prefill's or MoE models get a floor is the user's open #859
/// decision.
#[test]
fn k_chained_decodes_against_one_shot_rows() {
    let r = rig();
    let tapes = with_cuts(&[512, 1], route_cuts);
    let mut worker = r.worker_over(tapes);
    let rows = sequence_rows((Q + DECODES) as usize);
    let scan = PickScan::new(&mut worker, tapes, &rows, SEAT_B);
    let one_shot = forward(&mut worker, 512, &rows, 0, SEAT_B).expect("308 tokens in one forward");
    forward(&mut worker, 512, &rows[..Q as usize], 0, SEAT_A).expect("the prefill");
    let mut decoded = Vec::new();
    let mut pass = true;
    let mut near_ties = 0usize;
    for i in 0..DECODES as usize {
        let p = Q as usize + i;
        let (first, count) = scan.flips(&mut worker, rows[p], p, SEAT_A);
        let row = forward(&mut worker, 1, &[rows[p]], p as u32, SEAT_A).expect("a decode");
        let (finite, tie) = decode_line(
            &format!("(k) chained decode at position {p} against row {p} of one 308-token forward"),
            &row,
            &one_shot[p * VOCAB..][..VOCAB],
        );
        eprintln!(
            "(k) chained decode at position {p}: first layer whose picks differ from the one-shot \
             row's={first:?}, layers whose picks differ={count} of {LAYERS} (printed)"
        );
        near_ties += usize::from(tie);
        pass &= finite;
        decoded.push(row);
    }
    eprintln!("(k) near-tie steps {near_ties} of {DECODES} (printed)");
    // The last decode's lm_head: its `a` is the final norm's one row, still in the arena.
    let GemmBf16 { a, b, m, .. } = lm_head(&CUDA_TAPES[bucket(1)]);
    assert_eq!(m, 1, "the decode's lm_head");
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
    let cmp = compare_capped(last, &normed, w, (1, VOCAB, HIDDEN));
    eprintln!(
        "(k) the last decode's lm_head GEMV and soft-cap over {VOCAB} columns: {} {}",
        cmp.line(),
        verdict(cmp.pass())
    );
    assert!(cmp.pass(), "the decode's lm_head is out of bound");
    assert!(pass, "a chained decode gave a non-finite logit");
}

/// (k), teacher-forced: for each of the 8 positions 300 + i, a fresh prefill of 300 + i tokens,
/// then one decode at 300 + i, against row 300 + i of one 308-token forward in disjoint blocks.
/// Each decode is one step off the one-shot trajectory. Printed as (k), with the pick scan; gated
/// only on finiteness.
#[test]
fn k_teacher_forced_decodes_against_one_shot_rows() {
    let r = rig();
    let tapes = with_cuts(&[512, 1], route_cuts);
    let mut worker = r.worker_over(tapes);
    let rows = sequence_rows((Q + DECODES) as usize);
    let scan = PickScan::new(&mut worker, tapes, &rows, SEAT_B);
    let one_shot = forward(&mut worker, 512, &rows, 0, SEAT_B).expect("308 tokens in one forward");
    let mut pass = true;
    let mut near_ties = 0usize;
    for i in 0..DECODES as usize {
        let p = Q as usize + i;
        forward(&mut worker, 512, &rows[..p], 0, SEAT_A).expect("the prefill");
        let (first, count) = scan.flips(&mut worker, rows[p], p, SEAT_A);
        let row = forward(&mut worker, 1, &[rows[p]], p as u32, SEAT_A).expect("the decode");
        let (finite, tie) = decode_line(
            &format!(
                "(k teacher-forced) {p} tokens prefilled, decode at {p}, against row {p} of one \
                 308-token forward"
            ),
            &row,
            &one_shot[p * VOCAB..][..VOCAB],
        );
        eprintln!(
            "(k teacher-forced) decode at {p}: first layer whose picks differ from the one-shot \
             row's={first:?}, layers whose picks differ={count} of {LAYERS} (printed)"
        );
        near_ties += usize::from(tie);
        pass &= finite;
    }
    eprintln!("(k teacher-forced) near-tie steps {near_ties} of {DECODES} (printed)");
    assert!(pass, "a teacher-forced decode gave a non-finite logit");
}

/// (l): the prefill and eight decodes, repeated in disjoint KV blocks, give the same eight rows of
/// logits, bit for bit.
#[test]
fn l_repeated_decodes_are_bit_identical() {
    let r = rig();
    let mut worker = r.worker();
    let rows = sequence_rows((Q + DECODES) as usize);
    let (prefill, decodes) = rows.split_at(Q as usize);
    let first = prefill_then_decode(&mut worker, 512, (prefill, decodes), SEAT_A);
    let again = prefill_then_decode(&mut worker, 512, (prefill, decodes), SEAT_B);
    let ok = bit_identical(
        "(l) a prefill and 8 decodes repeated in disjoint blocks",
        &first.concat(),
        &again.concat(),
    );
    assert!(ok, "repeated decodes differ");
}

/// (m): the eight decodes leave every cache row at positions 0..300, K and V of all 30 layers
/// (each at its own class's layout), bit-unchanged.
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
    let ok = changed == 0 && before.len() == 2 * LAYERS;
    eprintln!(
        "(m) the cache at positions 0..{Q} (K and V of {} layers) after {DECODES} decodes: \
         changed elements={changed} {}",
        before.len() / 2,
        verdict(ok)
    );
    assert!(ok, "a decode wrote over the prefix's cache");
}

/// (n): a decode at P = 319 (kv_len 320, the last slot of block 4) and at P = 320 (kv_len 321, the
/// first slot of a new block), each against the same row of one 321-token forward in disjoint
/// blocks. Printed as (k); gated only on finiteness.
#[test]
fn n_decodes_at_a_block_boundary() {
    let r = rig();
    let mut worker = r.worker();
    let rows = sequence_rows(321);
    let decoded = prefill_then_decode(&mut worker, 512, (&rows[..319], &rows[319..]), SEAT_A);
    let one_shot = forward(&mut worker, 512, &rows, 0, SEAT_B).expect("321 tokens in one forward");
    let mut pass = true;
    for (i, row) in decoded.iter().enumerate() {
        let at = 319 + i;
        let (finite, _) = decode_line(
            &format!(
                "(n) decode at P={at} (kv_len {}) against row {at} of one 321-token forward",
                at + 1
            ),
            row,
            &one_shot[at * VOCAB..][..VOCAB],
        );
        pass &= finite;
    }
    assert!(
        pass,
        "a decode at the block boundary gave a non-finite logit"
    );
}

/// (o): at m=1, a position other than `prefix_len`, a block table too short for `P + 1`, a token
/// id out of the vocabulary, and `q_len = 2` are each refused before any copy, the result
/// untouched. qwen's GDN case (a GDN sequence called fresh at `P > 0`) does not apply.
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
    let cases: [(&str, KvSequence<'_>, u32, u32, ForwardError); 4] = [
        (
            "position P + 1",
            kv(blocks, 1),
            7,
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
            7,
            p,
            ForwardError::KvBlockTable {
                got: 1,
                need: 2,
                max: 128,
            },
        ),
        (
            "id = vocab",
            kv(blocks, 1),
            VOCAB as u32,
            p,
            ForwardError::TokenOutOfVocab {
                row: 0,
                id: VOCAB as u32,
                vocab: VOCAB as u32,
            },
        ),
        (
            "q_len = 2",
            kv(blocks, 2),
            7,
            p,
            ForwardError::KvQueryLength {
                q_len: 2,
                bucket_m: 1,
            },
        ),
    ];
    let mut pass = true;
    for (what, kv, id, position, want) in cases {
        let mut out = vec![0u16; VOCAB];
        let got = worker.forward(
            b,
            Input::TokenIds(&[id]),
            &[position],
            Some(&kv),
            None,
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
    eprintln!("(o) m=1 fresh at P = 64: n/a: no GDN layer");
    assert!(pass, "a bad decode was not refused before any copy");
}

// ── The sliding window: (p) one shot, (q) chunked, (r) decode, past 1024 positions ──

/// `t` with the window off: every attention step and KV class unwindowed (`tape_player`'s
/// `unwindowed`, #922), so the sliding layers attend causally to every key. The KV layout is the
/// window-free one's, so one pool fits both (`KvTape::same_pool`).
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

/// The attention steps of `t` with a window.
fn windowed_steps(t: &CudaTape) -> usize {
    (t.steps.iter())
        .filter(|s| {
            matches!(
                s,
                CudaStep::AttnPrefill(AttnPrefillBf16 {
                    window: Some(_),
                    ..
                }) | CudaStep::AttnDecode(AttnPrefillBf16 {
                    window: Some(_),
                    ..
                })
            )
        })
        .count()
}

/// [`CUDA_TAPES`] with the window off, in order, each checked: it validates (a class with no
/// window is the qwen case), and its windowed attention steps went from [`SLIDING_LAYERS`] to 0.
fn unwindowed_tapes(what: &str) -> &'static [CudaTape] {
    static TAPES: OnceLock<Vec<CudaTape>> = OnceLock::new();
    let tapes = TAPES.get_or_init(|| CUDA_TAPES.iter().map(unwindowed).collect());
    for (t, w) in CUDA_TAPES.iter().zip(tapes) {
        let valid = w.validate();
        let (before, after) = (windowed_steps(t), windowed_steps(w));
        let ok = valid.is_ok() && (before, after) == (SLIDING_LAYERS, 0);
        eprintln!(
            "{what} the window-off copy of m={}: validate={valid:?} windowed attention steps \
             {before} -> {after} (expect {SLIDING_LAYERS} -> 0) {}",
            t.bucket_m,
            verdict(ok)
        );
        assert!(
            ok,
            "the window-off copy of m={} is wrong: stop and report",
            t.bucket_m
        );
    }
    tapes
}

/// The tokens (p) and (q) run: 2,048, rows 1024..2047 past the window.
const PAST: usize = 2048;

/// (p): a 2,048-token fresh forward at m=2048 is finite and capped; its rows 0..1023 equal rows
/// 0..1023 of a 1,024-token forward at m=1024 bit for bit (below the window every sliding layer
/// sees the same keys, and the global layers are causal).
#[test]
fn p_window_one_shot_past_the_window() {
    let r = rig();
    let mut w = r.worker();
    let rows = sequence_rows(PAST);
    let long = forward(&mut w, 2048, &rows, 0, SEAT_A).expect("2048 tokens in one forward");
    let (non_finite, largest) = finite_and_largest(&long);
    let finite = non_finite == 0 && f64::from(largest) <= CAP;
    eprintln!(
        "(p) m=2048 q_len=2048: non-finite logits={non_finite} (expect 0), largest \
         |logit|={largest:e} (at most {CAP}) {}",
        verdict(finite)
    );
    let short = forward(&mut w, 1024, &rows[..WINDOW], 0, SEAT_B).expect("the 1024-token forward");
    let below = bit_identical(
        "(p) rows 0..1024 of the 2048-token forward against one 1024-token forward",
        &long[..WINDOW * VOCAB],
        &short,
    );
    assert!(
        finite,
        "the 2048-token forward is not finite or past the cap"
    );
    assert!(
        below,
        "below the window the 2048-token forward differs from the 1024-token one"
    );
}

/// (p)'s self-check: the same 2,048-token forward with the window off (a second worker over the
/// same weights, on [`unwindowed_tapes`]) differs on at least 90% of rows 1024..2047, where the
/// sliding layers drop keys. Its rows 0..1023 against the window-off ones are printed, not gated.
#[test]
fn p_window_self_check_the_window_drops_keys() {
    let r = rig();
    let rows = sequence_rows(PAST);
    let on = {
        let mut w = r.worker();
        forward(&mut w, 2048, &rows, 0, SEAT_A).expect("the windowed forward")
    };
    let off = {
        let mut w = r.worker_over(unwindowed_tapes("(p)"));
        forward(&mut w, 2048, &rows, 0, SEAT_A).expect("the window-off forward")
    };
    let differ = (WINDOW..PAST)
        .filter(|&i| on[i * VOCAB..][..VOCAB] != off[i * VOCAB..][..VOCAB])
        .count();
    let share = differ as f64 / (PAST - WINDOW) as f64;
    let bites = share >= 0.9;
    eprintln!(
        "(p) self-check: rows {WINDOW}..{PAST} that differ with the window off={differ} of {} \
         ({:.1}%, need >= 90%) {}",
        PAST - WINDOW,
        100.0 * share,
        verdict(bites)
    );
    bit_identical(
        "(p) self-check (printed, not gated): rows 0..1024 with the window on against off",
        &on[..WINDOW * VOCAB],
        &off[..WINDOW * VOCAB],
    );
    assert!(
        bites,
        "the window changes under 90% of the rows past it: stop and report"
    );
}

/// (q): the same 2,048 tokens prefilled as four 512-row chunks at m=512; the last chunk's rows
/// (1536..2047) equal the one-shot m=2048 forward's bit for bit: a chunk's windowed attention reads
/// the same keys in the same tile order as the one-shot. The three earlier chunks' rows are
/// printed against the one-shot's, not gated. On a difference, the first differing layer is found
/// by cutting the tapes after each layer.
#[test]
fn q_window_chunked_across_the_window() {
    let r = rig();
    let mut w = r.worker();
    let rows = sequence_rows(PAST);
    let one_shot = forward(&mut w, 2048, &rows, 0, SEAT_B).expect("2048 tokens in one forward");
    let chunks: Vec<(u32, usize, usize)> = (0..PAST / 512)
        .map(|c| (512, c * 512, c * 512 + 512))
        .collect();
    let mut pass = true;
    for &(m, s, e) in &chunks {
        let got = forward(&mut w, m, &rows[s..e], s as u32, SEAT_A).expect("a chunk");
        let gated = e == PAST;
        let same = bit_identical(
            &format!(
                "(q) rows {s}..{e}: a 512-row chunk at prefix {s}, against the one-shot m=2048 \
                 forward ({})",
                if gated { "gated" } else { "printed, not gated" }
            ),
            &got,
            &one_shot[s * VOCAB..e * VOCAB],
        );
        pass &= same || !gated;
    }
    if !pass {
        drop(w);
        first_differing_layer(&r, "(q)", (2048, &rows), &chunks);
    }
    assert!(
        pass,
        "a chunked prefill across the window differs from the one-shot: stop and report"
    );
}

/// The tokens (r) prefills, and the positions it decodes: 1,020 … 1,032, across the window's first
/// bite at 1,024.
const R_PREFILL: usize = 1020;
const R_LAST: usize = 1032;

/// (r): 1,020 tokens prefilled at m=1024, then decodes at positions 1,020 … 1,032, each against row
/// `p` of one 1,033-token forward at m=2048. Printed as (k); gated only on finiteness.
#[test]
fn r_window_decodes_across_the_window() {
    let r = rig();
    let mut w = r.worker();
    let rows = sequence_rows(R_LAST + 1);
    let decoded = prefill_then_decode(
        &mut w,
        1024,
        (&rows[..R_PREFILL], &rows[R_PREFILL..]),
        SEAT_A,
    );
    let one_shot = forward(&mut w, 2048, &rows, 0, SEAT_B).expect("1033 tokens in one forward");
    let mut pass = true;
    let mut near_ties = 0usize;
    for (i, row) in decoded.iter().enumerate() {
        let p = R_PREFILL + i;
        let (finite, tie) = decode_line(
            &format!(
                "(r) decode at P={p} (window floor {}) against row {p} of one 1033-token forward",
                p.saturating_sub(WINDOW - 1)
            ),
            row,
            &one_shot[p * VOCAB..][..VOCAB],
        );
        near_ties += usize::from(tie);
        pass &= finite;
    }
    eprintln!(
        "(r) near-tie steps {near_ties} of {} (printed)",
        decoded.len()
    );
    assert!(pass, "a decode across the window gave a non-finite logit");
}

/// (r)'s self-check: the same prefill and decode chain with the window off (a second worker over
/// the same weights) differs at every p ≥ 1,024, where the window drops key 0 and on, and is bit
/// for bit the windowed chain's at p < 1,024.
#[test]
fn r_window_self_check_the_window_drops_key_0() {
    let r = rig();
    let rows = sequence_rows(R_LAST + 1);
    let split = (&rows[..R_PREFILL], &rows[R_PREFILL..]);
    let on = {
        let mut w = r.worker();
        prefill_then_decode(&mut w, 1024, split, SEAT_A)
    };
    let off = {
        let mut w = r.worker_over(unwindowed_tapes("(r)"));
        prefill_then_decode(&mut w, 1024, split, SEAT_A)
    };
    let mut pass = true;
    for (i, (a, b)) in on.iter().zip(&off).enumerate() {
        let p = R_PREFILL + i;
        let mismatched = a.iter().zip(b).filter(|(x, y)| x != y).count();
        let ok = if p < WINDOW {
            mismatched == 0
        } else {
            mismatched > 0
        };
        eprintln!(
            "(r) self-check: decode at P={p} with the window on against off: \
             mismatched_bits={mismatched} (expect {}) {}",
            if p < WINDOW {
                "0, below the window"
            } else {
                "> 0, past it"
            },
            verdict(ok)
        );
        pass &= ok;
    }
    assert!(
        pass,
        "the window-off decode chain does not differ exactly from P = 1024: stop and report"
    );
}

// ── (k) diagnostics: layer 0 step by step, and the attention regime (#925) ──

use numerics::f32_from_halves;
use scratchy_target_cuda::tape::ArenaOffset;

/// The decode position the layer-0 diagnostic follows: (k)'s first step.
const DIAG_P: usize = 300;

/// A step's kind, and the kind of its m=1 counterpart: the m=512 tape's GEMMs are the m=1 tape's
/// GEMVs, its prefill attention the decode's, its grouped expert GEMM the gathered GEMV.
fn kinds(s: &CudaStep) -> (&'static str, &'static str) {
    match s {
        CudaStep::Embed(_) => ("Embed", "Embed"),
        CudaStep::Gemm(_) => ("Gemm", "Gemv"),
        CudaStep::GemmNarrow(_) => ("GemmNarrow", "GemmNarrow"),
        CudaStep::GemmCol64(_) => ("GemmCol64", "Gemv"),
        CudaStep::Gemv(_) => ("Gemv", "Gemv"),
        CudaStep::RmsNorm(_) => ("RmsNorm", "RmsNorm"),
        CudaStep::Add(_) => ("Add", "Add"),
        CudaStep::SiluMul(_) => ("SiluMul", "SiluMul"),
        CudaStep::Rope(_) => ("Rope", "Rope"),
        CudaStep::GateSplit(_) => ("GateSplit", "GateSplit"),
        CudaStep::GateApply(_) => ("GateApply", "GateApply"),
        CudaStep::RopeAppend(_) => ("RopeAppend", "RopeAppend"),
        CudaStep::AttnPrefill(_) => ("AttnPrefill", "AttnDecode"),
        CudaStep::AttnDecode(_) => ("AttnDecode", "AttnDecode"),
        CudaStep::GatedDeltaNet(_) => ("GatedDeltaNet", "GatedDeltaNet"),
        CudaStep::RouterLogits(_) => ("RouterLogits", "RouterLogits"),
        CudaStep::MoeRoute(_) => ("MoeRoute", "MoeRoute"),
        CudaStep::ExpertSort(_) => ("ExpertSort", "ExpertSort"),
        CudaStep::ExpertGemm(_) => ("ExpertGemm", "ExpertGemv"),
        CudaStep::ExpertGemv(_) => ("ExpertGemv", "ExpertGemv"),
        CudaStep::ExpertCombine(_) => ("ExpertCombine", "ExpertCombine"),
        CudaStep::GateScale(_) => ("GateScale", "GateScale"),
        CudaStep::Scale(_) => ("Scale", "Scale"),
        CudaStep::RmsNormUnit(_) => ("RmsNormUnit", "RmsNormUnit"),
        CudaStep::GeluMul(_) => ("GeluMul", "GeluMul"),
        CudaStep::LayerScale(_) => ("LayerScale", "LayerScale"),
        CudaStep::SoftCap(_) => ("SoftCap", "SoftCap"),
        CudaStep::RouterNorm(_) => ("RouterNorm", "RouterNorm"),
    }
}

/// Layer 0's steps of `tape`, from the embed to the layer's scale, the sort left out (the m=1
/// tape gathers its pairs instead): their indices.
fn layer0_steps(tape: &CudaTape) -> Vec<usize> {
    let end = (tape.steps.iter())
        .position(|s| matches!(s, CudaStep::LayerScale(_)))
        .expect("a layer scale");
    (0..=end)
        .filter(|&i| !matches!(tape.steps[i], CudaStep::ExpertSort(_)))
        .collect()
}

/// Row `p`'s part of an m-row tape's span `many` when it holds `m` times the m=1 span `one`'s
/// bytes (one row per token); `None` otherwise (the sorted or padded expert rows, the KV
/// regions), and when either is absent.
fn row_of(
    one: Option<ArenaSpan>,
    many: Option<ArenaSpan>,
    m: u32,
    p: usize,
) -> Option<(ArenaSpan, ArenaSpan)> {
    let (one, many) = (one?, many?);
    (many.bytes == u64::from(m) * one.bytes).then_some((
        one,
        ArenaSpan {
            offset: ArenaOffset(many.offset.0 + p as u64 * one.bytes),
            bytes: one.bytes,
        },
    ))
}

/// An arena span's bytes after a forward.
fn read_span(worker: &mut CudaWorker<'_>, span: ArenaSpan) -> Vec<u16> {
    let mut v = vec![0u16; span.bytes as usize / 2];
    worker.read_arena(span, &mut v).expect("read_arena");
    v
}

/// `‖a − b‖₂ / ‖b‖₂` of two bf16 rows, in f64; infinite when either holds a non-finite value.
fn rel_l2(a: &[u16], b: &[u16]) -> f64 {
    rel_l2_f64(a, &widen(b))
}

/// `‖a − r‖₂ / ‖r‖₂` of a bf16 row against an f64 one; infinite on a non-finite value.
fn rel_l2_f64(a: &[u16], r: &[f64]) -> f64 {
    let (mut d2, mut r2) = (0f64, 0f64);
    for (x, &y) in widen(a).into_iter().zip(r) {
        if !(x.is_finite() && y.is_finite()) {
            return f64::INFINITY;
        }
        d2 += (x - y) * (x - y);
        r2 += y * y;
    }
    (d2 / r2.max(f64::MIN_POSITIVE)).sqrt()
}

/// One region's drift, decode against one-shot, as printed, and its rel L2 (0 or infinite for
/// a region of words, by whether any differs); `n/a` when the region is not one row per token.
fn drift(name: &str, pair: Option<(Vec<u16>, Vec<u16>)>, words: bool) -> (String, Option<f64>) {
    let Some((got, want)) = pair else {
        return (format!("{name}=n/a (not one row per token)"), None);
    };
    let mismatched = got.iter().zip(&want).filter(|(x, y)| x != y).count();
    if words {
        let rel = if mismatched == 0 { 0.0 } else { f64::INFINITY };
        return (
            format!("{name} words mismatched_halves={mismatched}/{}", got.len()),
            Some(rel),
        );
    }
    let rel = rel_l2(&got, &want);
    (
        format!(
            "{name} max|diff|={:e} rel L2={rel:.3e} mismatched_bits={mismatched}/{}",
            max_diff(&got, &want),
            got.len()
        ),
        Some(rel),
    )
}

/// One KV layer's K and V of `seat`'s first `tokens` positions, at the layer's own class
/// layout: `[tokens, kv_heads · head_dim]` bf16 each.
fn layer_kv(worker: &mut CudaWorker<'_>, seat: Seat, layer: usize, tokens: usize) -> [Vec<u16>; 2] {
    let kv = CUDA_TAPES[bucket(1)].kv.expect("a KV cache");
    let bs = kv.block_size as usize;
    let class = kv.classes[usize::from(kv.layer_class[layer])];
    let (heads, hd) = (class.kv_heads as usize, class.head_dim as usize);
    [KvHalf::K, KvHalf::V].map(|half| {
        let mut rows = vec![0u16; tokens * heads * hd];
        for b in 0..tokens.div_ceil(bs) {
            let mut block = vec![0u16; heads * bs * hd];
            (worker.read_kv(layer, half, seat.blocks[b], &mut block)).expect("read_kv");
            for t in b * bs..((b + 1) * bs).min(tokens) {
                for h in 0..heads {
                    rows[(t * heads + h) * hd..][..hd]
                        .copy_from_slice(&block[(h * bs + t % bs) * hd..][..hd]);
                }
            }
        }
        rows
    })
}

/// Row `p`'s attention in f64 from its query row `q` (`[q_heads, head_dim]`) and a cache's keys
/// and values (`[tokens, kv_heads · head_dim]`) at `a`'s scale and window: the output
/// `[q_heads · head_dim]` and, per head, the top-1 minus top-2 score and the scores' standard
/// deviation.
fn attention_row(
    q: &[u16],
    [k, v]: &[Vec<u16>; 2],
    a: AttnPrefillBf16,
    p: usize,
) -> (Vec<f64>, Vec<(f64, f64)>) {
    let (qh, kh, hd) = (a.q_heads as usize, a.kv_heads as usize, a.head_dim as usize);
    let lo = a.window.map_or(0, |w| (p + 1).saturating_sub(w as usize));
    let (q, k, v) = (widen(q), widen(k), widen(v));
    let scale = f64::from(a.scale());
    let mut out = vec![0f64; qh * hd];
    let mut stats = Vec::with_capacity(qh);
    for h in 0..qh {
        let g = h / (qh / kh);
        let qr = &q[h * hd..][..hd];
        let s: Vec<f64> = (lo..=p)
            .map(|t| {
                scale
                    * qr.iter()
                        .zip(&k[(t * kh + g) * hd..][..hd])
                        .map(|(x, y)| x * y)
                        .sum::<f64>()
            })
            .collect();
        let max = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = s.iter().map(|x| (x - max).exp()).collect();
        let z: f64 = e.iter().sum();
        for (j, t) in (lo..=p).enumerate() {
            for d in 0..hd {
                out[h * hd + d] += e[j] / z * v[(t * kh + g) * hd + d];
            }
        }
        let mut sorted = s.clone();
        sorted.sort_by(|x, y| y.total_cmp(x));
        let mean = s.iter().sum::<f64>() / s.len() as f64;
        let sd = (s.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / s.len() as f64).sqrt();
        stats.push((
            sorted[0] - sorted.get(1).copied().unwrap_or(f64::NEG_INFINITY),
            sd,
        ));
    }
    (out, stats)
}

/// The median of `v`.
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// (k)'s diagnostic, layer 0 step by step (#925): 300 tokens prefilled at m=512 in seat A, then per
/// step of layer 0 the m=1 tape cut after it, decoding token 300 at P = 300 in seat A, against row
/// 300 of the m=512 tape cut after the matching step, 301 tokens fresh in seat B. Per step it
/// prints the kind, the input's (read 0) and the output's max |difference| and rel L2, decode
/// against one-shot; a region that is not one row per token (the sorted and padded expert rows) is
/// `n/a`. At the routing, the picks and scores, and the one-shot logits' rank-8 minus rank-9 gap.
/// At the attention, each path's output against an f64 attention of its own query and cache, the
/// cache rows of the two seats, and the per-head top-1 minus top-2 score gap and score spread.
/// Then the first step whose output drifts (rel L2 above 1e-2 and 10 times its input's): where a
/// real defect would show.
///
/// Gated, the decode check that holds on these synthetic weights (#925's GKE diagnosis: the
/// whole-model cosine is the near-hard softmax's, not a defect): (i) every step up to and
/// including the `RopeAppend` bit-identical, input and output, and the K and V cache rows of both
/// seats (0..=300) too; (ii) no step drifts; (iii) the decode attention's rel L2 against the f64 of
/// its own query and cache at most twice the prefill attention's against its own; (iv) the same
/// expert picks at the routing.
#[test]
fn k_diag_layer0_step_by_step_decode_against_one_shot() {
    let r = rig();
    let (one, many) = (&CUDA_TAPES[bucket(1)], &CUDA_TAPES[bucket(512)]);
    let (s1, s512) = (layer0_steps(one), layer0_steps(many));
    let k1: Vec<&str> = s1.iter().map(|&i| kinds(&one.steps[i]).0).collect();
    let k512: Vec<&str> = s512.iter().map(|&i| kinds(&many.steps[i]).1).collect();
    eprintln!("(k-diag0) layer 0's m=1 steps {k1:?}");
    assert_eq!(
        k1, k512,
        "layer 0's m=512 steps, as their m=1 kinds, pair one to one"
    );
    let extents = |t: &CudaTape, i: usize| t.steps[i].extents().expect("the step's extents");
    let mut list = CUDA_TAPES.to_vec();
    list.extend(s1.iter().map(|&i| cut(one, i, extents(one, i).writes)));
    list.extend(s512.iter().map(|&i| cut(many, i, extents(many, i).writes)));
    let tapes: &'static [CudaTape] = Box::leak(list.into_boxed_slice());
    let mut w = r.worker_over(tapes);
    let rows = sequence_rows(DIAG_P + 1);
    let p = DIAG_P;
    forward(&mut w, 512, &rows[..p], 0, SEAT_A).expect("the prefill");
    let top_k = one.moe.expect("a MoE scratch").top_k as usize;
    let mut first = None;
    // The gates: (i) through the K/V write, (i) the cache rows, (iii) the attention, (iv) the
    // picks; the last three `None` until their step ran.
    let append = (s1.iter())
        .position(|&i| matches!(one.steps[i], CudaStep::RopeAppend(_)))
        .expect("layer 0's RopeAppend");
    let mut through_append = true;
    let (mut cache_same, mut attention_ok, mut picks_same) = (None, None, None);
    for (j, (&i1, &i5)) in s1.iter().zip(&s512).enumerate() {
        let (a, b) = (one.steps[i1], many.steps[i5]);
        let (ea, eb) = (extents(one, i1), extents(many, i5));
        let input = row_of(ea.reads[0], eb.reads[0], 512, p);
        let output = row_of(Some(ea.writes), Some(eb.writes), 512, p);
        let second = row_of(ea.second_write, eb.second_write, 512, p);
        let spans = [input, output, second];
        run(
            &mut w,
            (tapes, CUDA_TAPES.len() + j),
            &rows[p..=p],
            p as u32,
            SEAT_A,
        )
        .expect("the m=1 cut");
        let got: Vec<Option<Vec<u16>>> = spans
            .iter()
            .map(|s| s.map(|(x, _)| read_span(&mut w, x)))
            .collect();
        run(
            &mut w,
            (tapes, CUDA_TAPES.len() + s1.len() + j),
            &rows[..=p],
            0,
            SEAT_B,
        )
        .expect("the m=512 cut");
        let want: Vec<Option<Vec<u16>>> = spans
            .iter()
            .map(|s| s.map(|(_, y)| read_span(&mut w, y)))
            .collect();
        let pair = |k: usize| got[k].clone().zip(want[k].clone());
        let route = matches!(a, CudaStep::MoeRoute(_));
        let (in_line, in_rel) = match ea.in_place {
            true => (
                "in=in place (read after the step: the output)".to_string(),
                None,
            ),
            false => drift("in", pair(0), matches!(a, CudaStep::Embed(_))),
        };
        let (out_line, out_rel) = drift("out", pair(1), route);
        if j <= append {
            // Both regions present and equal: a region that is not one row per token, or an
            // in-place step's input, is not proven bit-identical.
            let same = |k: usize| pair(k).is_some_and(|(x, y)| x == y);
            through_append &= same(1) && (ea.in_place || same(0));
        }
        eprintln!(
            "(k-diag0) step {j} (m=1 step {i1} {}, m=512 step {i5} {}): {in_line}; {out_line}",
            kinds(&a).0,
            kinds(&b).0
        );
        if route
            && let (Some((gi, wi)), Some((gs, ws)), Some((_, wl))) = (pair(1), pair(2), pair(0))
        {
            let (gi, wi) = (words(&gi), words(&wi));
            let (gs, ws) = (f32_from_halves(&gs), f32_from_halves(&ws));
            let mut sorted: Vec<f32> = wl.iter().map(|&x| bf16::from_bits(x).to_f32()).collect();
            sorted.sort_by(|x, y| y.total_cmp(x));
            let score = (gs.iter().zip(&ws))
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            let same = picks(&gi, 0, top_k) == picks(&wi, 0, top_k);
            picks_same = Some(picks_same.unwrap_or(true) && same);
            eprintln!(
                "(k-diag0) step {j} routing: picks same={same} decode {:?} one-shot {:?}; scores \
                 max|diff|={score:e}; one-shot logits rank-8 minus rank-9 gap={:e}",
                picks(&gi, 0, top_k),
                picks(&wi, 0, top_k),
                sorted[top_k - 1] - sorted[top_k]
            );
        }
        if let (CudaStep::AttnDecode(at), Some((gq, wq)), Some((go, wo))) = (a, pair(0), pair(1)) {
            let layer = at.layer as usize;
            let (ca, cb) = (
                layer_kv(&mut w, SEAT_A, layer, p + 1),
                layer_kv(&mut w, SEAT_B, layer, p + 1),
            );
            let width = ca[0].len() / (p + 1);
            for (h, name) in [(0, "K"), (1, "V")] {
                let (x, y) = (&ca[h], &cb[h]);
                let prefix = x[..p * width]
                    .iter()
                    .zip(&y[..p * width])
                    .filter(|(u, v)| u != v)
                    .count();
                let row_same = x[p * width..] == y[p * width..];
                cache_same = Some(cache_same.unwrap_or(true) && prefix == 0 && row_same);
                eprintln!(
                    "(k-diag0) step {j} cache layer {layer} {name}: seat A against seat B rows \
                     0..{p} mismatched_bits={prefix}; row {p} {}",
                    drift(
                        "row",
                        Some((x[p * width..].to_vec(), y[p * width..].to_vec())),
                        false
                    )
                    .0
                );
            }
            let (rd, _) = attention_row(&gq, &ca, at, p);
            let (ro, stats) = attention_row(&wq, &cb, at, p);
            let gaps: Vec<f64> = stats.iter().map(|s| s.0).collect();
            let sds: Vec<f64> = stats.iter().map(|s| s.1).collect();
            let (dec, pre) = (rel_l2_f64(&go, &rd), rel_l2_f64(&wo, &ro));
            attention_ok = Some(attention_ok.unwrap_or(true) && dec <= 2.0 * pre);
            eprintln!(
                "(k-diag0) step {j} attention: decode against the f64 of its own query and cache \
                 rel L2={:.3e}; one-shot against the f64 of its own rel L2={:.3e}; the two f64 \
                 references' rel L2={:.3e}; per-head top-1 minus top-2 score gap min={:e} \
                 median={:e}; score sd median={:e} (scale {})",
                dec,
                pre,
                rel_l2_f64(&wo, &rd),
                gaps.iter().copied().fold(f64::INFINITY, f64::min),
                median(gaps.clone()),
                median(sds),
                at.scale()
            );
        }
        if first.is_none()
            && let Some(o) = out_rel
            && o > 1e-2
            && o > 10.0 * in_rel.unwrap_or(0.0)
        {
            first = Some((j, kinds(&a).0));
        }
    }
    eprintln!(
        "(k-diag0) first layer-0 step whose output drifts (rel L2 > 1e-2 and > 10x its input's): \
         {first:?} (gated: expect None) {}",
        verdict(first.is_none())
    );
    let gates = [
        (
            "(i) steps 0..={append} (through the RopeAppend) bit-identical, input and output",
            Some(through_append),
        ),
        (
            "(i) the K and V cache rows 0..=300 of both seats bit-identical",
            cache_same,
        ),
        ("(ii) no step drifts", Some(first.is_none())),
        (
            "(iii) the decode attention's rel L2 against its own f64 at most 2x the prefill's",
            attention_ok,
        ),
        ("(iv) the same expert picks at the routing", picks_same),
    ];
    let mut pass = true;
    for (what, ok) in gates {
        eprintln!(
            "(k-diag0) gate {what}: {} {}",
            ok.map_or("not reached".to_string(), |b| b.to_string()),
            verdict(ok == Some(true))
        );
        pass &= ok == Some(true);
    }
    assert!(
        pass,
        "layer 0's decode is out of agreement with the prefill's row: see the gate lines above"
    );
}

/// The q_norm and k_norm gains: every `RmsNorm` whose row is one attention head (256 or 512
/// columns), 2 per layer.
fn qk_norm_gains() -> BTreeSet<usize> {
    (CUDA_TAPES[bucket(1)].steps.iter())
        .filter_map(|s| match *s {
            CudaStep::RmsNorm(n) if n.cols == 256 || n.cols == 512 => Some(n.gain.0 as usize),
            _ => None,
        })
        .collect()
}

/// (k)'s regime test (#925): the teacher-forced (k) at P = 300..307 and (n) at P = 319, 320, rerun
/// with ONLY the q_norm and k_norm gains scaled by 1/4 (exact in bf16), as #922's layer test did,
/// so the scores at Gemma's scale 1 stay near unit size; every other weight is the body's. Per step
/// it prints the decode line ([`decode_line`]) and the first layer whose expert picks are not the
/// one-shot's. If the cosine returns to 0.95 or more, the decode has no defect and (k)'s cosines
/// near 0 are the synthetic weights' near-hard softmax. Asserted: only finiteness.
#[test]
fn k_regime_quarter_qk_norm_gains_decode_against_one_shot() {
    let mut r = rig();
    let qk = qk_norm_gains();
    assert_eq!(qk.len(), 2 * LAYERS, "a q_norm and a k_norm gain per layer");
    for &ix in &qk {
        let w = host_weights()[ix].as_ref().expect("a gain");
        let quarter: Vec<u16> = (w.iter())
            .map(|&b| numerics::bits(bf16::from_bits(b).to_f32() * 0.25))
            .collect();
        r.table[ix] = DeviceBuffer::upload(&r.device, &quarter).expect("upload a quartered gain");
    }
    eprintln!("(k-regime) q_norm and k_norm gains scaled by 1/4: weights {qk:?}");
    let tapes = with_cuts(&[512, 1], route_cuts);
    let mut w = r.worker_over(tapes);
    let mut pass = true;
    let mut cosines = Vec::new();
    // The teacher-forced (k).
    let rows = sequence_rows((Q + DECODES) as usize);
    let scan = PickScan::new(&mut w, tapes, &rows, SEAT_B);
    let one_shot = forward(&mut w, 512, &rows, 0, SEAT_B).expect("308 tokens in one forward");
    for p in Q as usize..(Q + DECODES) as usize {
        forward(&mut w, 512, &rows[..p], 0, SEAT_A).expect("the prefill");
        let (first, count) = scan.flips(&mut w, rows[p], p, SEAT_A);
        let row = forward(&mut w, 1, &[rows[p]], p as u32, SEAT_A).expect("the decode");
        let want = &one_shot[p * VOCAB..][..VOCAB];
        let (finite, _) = decode_line(
            &format!(
                "(k-regime teacher-forced) {p} tokens prefilled, decode at {p}, against row {p} of one 308-token forward"
            ),
            &row,
            want,
        );
        eprintln!(
            "(k-regime teacher-forced) decode at {p}: first layer whose picks differ from the \
             one-shot row's={first:?}, layers whose picks differ={count} of {LAYERS} (printed)"
        );
        cosines.push(cosine(&row, want));
        pass &= finite;
    }
    // (n): 319 prefilled, then decodes at 319 and 320.
    let rows = sequence_rows(321);
    let scan = PickScan::new(&mut w, tapes, &rows, SEAT_B);
    let one_shot = forward(&mut w, 512, &rows, 0, SEAT_B).expect("321 tokens in one forward");
    forward(&mut w, 512, &rows[..319], 0, SEAT_A).expect("the prefill");
    for p in 319..321 {
        let (first, count) = scan.flips(&mut w, rows[p], p, SEAT_A);
        let row = forward(&mut w, 1, &[rows[p]], p as u32, SEAT_A).expect("the decode");
        let want = &one_shot[p * VOCAB..][..VOCAB];
        let (finite, _) = decode_line(
            &format!(
                "(k-regime n) decode at P={p} (kv_len {}) against row {p} of one 321-token forward",
                p + 1
            ),
            &row,
            want,
        );
        eprintln!(
            "(k-regime n) decode at P={p}: first layer whose picks differ from the one-shot \
             row's={first:?}, layers whose picks differ={count} of {LAYERS} (printed)"
        );
        cosines.push(cosine(&row, want));
        pass &= finite;
    }
    let worst = cosines.iter().copied().fold(f64::INFINITY, f64::min);
    eprintln!(
        "(k-regime) worst cosine over the 10 decodes={worst:.6} (>= 0.95: {}; printed, not gated)",
        worst >= 0.95
    );
    assert!(pass, "a decode gave a non-finite logit");
}
