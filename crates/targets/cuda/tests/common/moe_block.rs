// SPDX-License-Identifier: Apache-2.0
//! One Qwen3.6-35B-A3B MoE block, `y = x + gate_scale(routed(n), shared(n), n · w_g)` with
//! `n = RmsNorm(x, g)`: the canonical the MoE block tape test lowers (#857), written in the order
//! `#[forward]` emits the `qwen3-5-moe` DSL's block (`crates/models/arch/dsl/qwen3-5-moe.py`; the
//! ops and their operands are the ones the 35B canonical hands to CUDA's seam):
//!
//! ```text
//!  0 n      = RmsNorm(x, g)                          [M, 2048]
//!  1 logits = RouterLogits(n, bundle)                [M, 256]    the router at the bundle's byte 0
//!  2 p      = RouteSoftmax(logits)
//!  3 sorted = RouteArgsort(p)
//!  4 idx    = RouteTopK(sorted)                      [M, 8]      2-4 fold into one MoeRoute
//!  5 s      = RouteGatherScores(p, idx)
//!  6 w      = RouteRenorm(s)                         [M, 8]
//!  7 rows   = ExpertSort(n, idx)                     the pair rows (sorted at M ≥ 8)
//!  8 gate   = ExpertMatmul(Gate, rows, rows, bundle) [P, 512]
//!  9 up     = ExpertMatmul(Up, rows, rows, bundle)   [P, 512]
//! 10 act    = ExpertGatedAct(Silu, gate, up)
//! 11 down   = ExpertMatmul(Down, act, rows, bundle)  [P, 2048]
//! 12 back   = ExpertUnsort(down, rows)
//! 13 routed = ExpertCombine(back, w)                 [M, 2048]
//! 14 sg     = n · Wsg                                [M, 512]    the shared expert
//! 15 su     = n · Wsu                                [M, 512]
//! 16 gl     = n · Wgate                              [M, 1]      the narrow GEMM at n = 1
//! 17 ss     = Silu(sg)
//! 18 sa     = ss · su                                            17-18 fold into one SiluMul
//! 19 sd     = sa · Wsd                               [M, 2048]
//! 20 mix    = GateScale(routed, sd, gl)              [M, 2048]
//! 21 y      = mix + x                                [M, 2048]   the result
//! ```
//!
//! Geometry is the 35B's: hidden `H = 2048`, 256 experts, top-8, expert and shared intermediate
//! `I = 512`, `rms_norm_eps` 1e-6, the zero-centred gain. Ops 1-13 are one MoE expansion, as the
//! front end marks them. Sources: 0 `x` (the embedded hidden), 1 `g`, 2 the bundle (a `[1, 1]`
//! source; on the device the dense `moe_bundle`: router, fused gate and up, down), 3 `Wsg`, 4
//! `Wsu`, 5 `Wgate`, 6 `Wsd`. Every dense weight is declared in the IR as `[K, N]`; its bytes on
//! the device are `[N, K]`.

use scratchy_subtile::handoff::{Expansion, ExpansionId, LoweredDecode};
use scratchy_subtile::lower::{ExpertWeight, GemmWeight, InputRef, LoweringInput, OpDesc};
use scratchy_subtile::subtile_ir::{
    EwKind, ExpertBundle, ExpertProj, GainConvention, GatedAct, NumExperts, RouterBundle,
    SharedExpertBound, SourceShape, SubOp, TopK,
};
use scratchy_target_cuda_compiler::canonical::{CudaModelFacts, SharedExpertOwner};

use super::chains::{QWEN3_5_FACTS, with_hidden};

/// The 35B's hidden size.
pub const H: u32 = 2048;
/// Its experts, picks per token, and expert (and shared-expert) intermediate.
pub const EXPERTS: u32 = 256;
pub const TOP_K: u32 = 8;
pub const I: u32 = 512;
/// qwen3.5's `rms_norm_eps`.
pub const EPS: f32 = 1e-6;

/// The 35B's model facts for the block: qwen3.5's, with the shared expert the DSL's own ops.
pub const MOE_FACTS: CudaModelFacts = CudaModelFacts {
    shared_expert: SharedExpertOwner::Dsl,
    ..QWEN3_5_FACTS
};

/// The block's ops at `m` rows, as the table above.
pub fn moe_input(m: u32) -> LoweringInput {
    let nz = |n| std::num::NonZeroU32::new(n).expect("non-zero");
    let (experts, k, bundle) = (
        NumExperts::new(nz(EXPERTS)),
        TopK::new(nz(TOP_K)),
        ExpertBundle::SharedFused,
    );
    let op = |op, inputs| OpDesc { op, m, inputs };
    let expert = |proj, n, inputs| {
        op(
            SubOp::ExpertMatmul {
                proj,
                n,
                k,
                weight: ExpertWeight::Dense,
                bundle,
            },
            inputs,
        )
    };
    let matmul = |n, act, w| {
        op(
            SubOp::MatmulTile {
                n,
                weight: GemmWeight::Dense,
            },
            vec![act, InputRef::Ext(w)],
        )
    };
    use InputRef::{Ext, Op};
    LoweringInput {
        exported: None,
        sources: vec![
            SourceShape { rows: m, cols: H },
            SourceShape { rows: 1, cols: H },
            SourceShape { rows: 1, cols: 1 },
            SourceShape { rows: H, cols: I },
            SourceShape { rows: H, cols: I },
            SourceShape { rows: H, cols: 1 },
            SourceShape { rows: I, cols: H },
        ],
        ops: vec![
            op(
                SubOp::RmsNorm {
                    eps: EPS,
                    gain: GainConvention::OnePlusScale,
                },
                vec![Ext(0), Ext(1)],
            ),
            op(
                SubOp::RouterLogits {
                    experts,
                    router: RouterBundle::SharedFused,
                },
                vec![Op(0), Ext(2)],
            ),
            op(SubOp::RouteSoftmax, vec![Op(1)]),
            op(SubOp::RouteArgsort, vec![Op(2)]),
            op(SubOp::RouteTopK { k }, vec![Op(3)]),
            op(SubOp::RouteGatherScores, vec![Op(2), Op(4)]),
            op(SubOp::RouteRenorm, vec![Op(5)]),
            op(SubOp::ExpertSort { experts, k, bundle }, vec![Op(0), Op(4)]),
            expert(ExpertProj::Gate, I, vec![Op(7), Op(7), Ext(2)]),
            expert(ExpertProj::Up, I, vec![Op(7), Op(7), Ext(2)]),
            op(
                SubOp::ExpertGatedAct {
                    act: GatedAct::Silu,
                },
                vec![Op(8), Op(9)],
            ),
            expert(ExpertProj::Down, H, vec![Op(10), Op(7), Ext(2)]),
            op(SubOp::ExpertUnsort, vec![Op(11), Op(7)]),
            op(
                SubOp::ExpertCombine {
                    hidden: H,
                    shared: SharedExpertBound(Some(nz(I))),
                },
                vec![Op(12), Op(6)],
            ),
            matmul(I, Op(0), 3),
            matmul(I, Op(0), 4),
            matmul(1, Op(0), 5),
            op(SubOp::Elementwise(EwKind::Silu), vec![Op(14)]),
            op(SubOp::Elementwise(EwKind::Mul), vec![Op(17), Op(15)]),
            matmul(H, Op(18), 6),
            op(SubOp::GateScale, vec![Op(13), Op(19), Op(16)]),
            op(SubOp::Elementwise(EwKind::Add), vec![Op(20), Ext(0)]),
        ],
        result: 21,
    }
}

/// The block at `m` rows as the front end hands it over: source 0 the embedded hidden, ops 1-13
/// one MoE expansion.
pub fn moe_block(m: u32) -> LoweredDecode {
    let mut l = with_hidden(moe_input(m));
    for (i, e) in l.op_expansion.iter_mut().enumerate() {
        if (1..=13).contains(&i) {
            *e = Some(Expansion {
                id: ExpansionId(0),
                guard: None,
            });
        }
    }
    l
}

// ── Synthetic weights and routings ─────────────────────────────────
//
// Each router row is a scaled row of a Sylvester Hadamard matrix (`h_{e+1}`, entries ±1, every two
// rows orthogonal), and each token row `x` is `κ · Σ_k L[k] · h_{e_k + 1}` plus a little uniform
// noise, over the 9 experts `e_k` its routing case names. After the norm, expert `e_k`'s logit is
// `L[k]` and every other expert's about 0 (σ ≈ 0.05): the picks are the case's, 0.3 apart, so a
// near-tie is rare, and the 8 picks hold about a third of the softmax mass, so the renorm matters.
// The three cases differ in `x` only, so one weight upload serves them all.

/// The logits a token's 9 named experts get, best first.
pub const LOGITS: [f64; 9] = [3.6, 3.3, 3.0, 2.7, 2.4, 2.1, 1.8, 1.5, 0.8];
/// The routing signal's scale in `x`, and the noise's amplitude.
const KAPPA: f64 = 0.05;
const ETA: f64 = 0.2;
/// The 8 experts every token of a skewed routing picks.
pub const SKEWED: [u32; 8] = [17, 40, 77, 101, 150, 199, 230, 251];

/// How a bucket's tokens are routed (#841's three routing cases).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Routing {
    /// Each token to its own 9 random experts.
    Random,
    /// Every token to [`SKEWED`] (in its own order), its 9th random.
    Skewed,
    /// Even tokens as [`Routing::Skewed`], odd ones as [`Routing::Random`].
    Half,
}

/// The block's weights, as the device holds them (`[N, K]` bf16): the gain, the MoE bundle
/// (router `[256, H]`, fused gate and up `[256, 2I, H]`, down `[256, H, I]`), and the shared
/// expert's gate `[I, H]`, up `[I, H]`, scalar gate `[1, H]` and down `[H, I]`.
pub struct MoeWeights {
    pub g: Vec<u16>,
    pub bundle: Vec<u16>,
    pub wsg: Vec<u16>,
    pub wsu: Vec<u16>,
    pub wgate: Vec<u16>,
    pub wsd: Vec<u16>,
}

const HU: usize = H as usize;
const IU: usize = I as usize;
const EU: usize = EXPERTS as usize;
const GATE_UP: usize = EU * HU;
const DOWN: usize = GATE_UP + EU * 2 * IU * HU;

impl MoeWeights {
    fn router(&self, e: usize) -> &[u16] {
        &self.bundle[e * HU..][..HU]
    }
    fn gate(&self, e: usize, j: usize) -> &[u16] {
        &self.bundle[GATE_UP + (e * 2 * IU + j) * HU..][..HU]
    }
    fn up(&self, e: usize, j: usize) -> &[u16] {
        &self.bundle[GATE_UP + (e * 2 * IU + IU + j) * HU..][..HU]
    }
    fn down(&self, e: usize, j: usize) -> &[u16] {
        &self.bundle[DOWN + (e * HU + j) * IU..][..IU]
    }
}

/// splitmix64 from `seed`.
fn mix(seed: u64) -> impl FnMut() -> u64 {
    let mut state = seed;
    move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// A uniform draw in `[-1, 1)`.
fn unit(r: &mut impl FnMut() -> u64) -> f64 {
    (r() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

fn bits(v: f64) -> u16 {
    half::bf16::from_f32(v as f32).to_bits()
}

/// `len` bf16 uniform in `[-a, a)`, filled by every core in chunks of their own seeds.
fn uniform(out: &mut [u16], seed: u64, a: f64) {
    const CHUNK: usize = 1 << 20;
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let chunks: Vec<(usize, &mut [u16])> = out.chunks_mut(CHUNK).enumerate().collect();
    let per = chunks.len().div_ceil(threads).max(1);
    let mut chunks = chunks;
    std::thread::scope(|s| {
        while !chunks.is_empty() {
            let take = chunks.len().min(per);
            let mine: Vec<_> = chunks.drain(..take).collect();
            s.spawn(move || {
                for (c, part) in mine {
                    let mut r = mix(seed ^ (c as u64).wrapping_mul(0x5851_F42D_4C95_7F2D));
                    for v in part.iter_mut() {
                        *v = bits(a * unit(&mut r));
                    }
                }
            });
        }
    });
}

/// Hadamard row `r`'s element `i`.
fn hadamard(r: usize, i: usize) -> f64 {
    if (r & i).count_ones().is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}

/// The norm's rms of a token row: the signal's and the noise's.
fn x_rms() -> f64 {
    let l2: f64 = LOGITS.iter().map(|l| l * l).sum();
    (KAPPA * KAPPA * l2 + ETA * ETA / 3.0).sqrt()
}

/// The block's synthetic weights (about 1.6 GB of bf16).
pub fn moe_weights() -> MoeWeights {
    let mut g = vec![0u16; HU];
    uniform(&mut g, 0x0E_0001, 0.05);
    let mut bundle = vec![0u16; DOWN + EU * HU * IU];
    let sigma_r = x_rms() / (H as f64 * KAPPA);
    for e in 0..EU {
        for i in 0..HU {
            bundle[e * HU + i] = bits(sigma_r * hadamard(e + 1, i));
        }
    }
    uniform(&mut bundle[GATE_UP..DOWN], 0x0E_0002, 0.04);
    uniform(&mut bundle[DOWN..], 0x0E_0003, 0.1);
    let mut wsg = vec![0u16; IU * HU];
    let mut wsu = vec![0u16; IU * HU];
    let mut wgate = vec![0u16; HU];
    let mut wsd = vec![0u16; HU * IU];
    uniform(&mut wsg, 0x0E_0004, 0.04);
    uniform(&mut wsu, 0x0E_0005, 0.04);
    uniform(&mut wgate, 0x0E_0006, 0.077);
    uniform(&mut wsd, 0x0E_0007, 0.1);
    MoeWeights {
        g,
        bundle,
        wsg,
        wsu,
        wgate,
        wsd,
    }
}

/// The 9 experts token `t` is routed to under `routing`, best first.
fn named_experts(routing: Routing, t: usize, seed: u64) -> [u32; 9] {
    let mut r = mix(seed ^ (t as u64).wrapping_mul(0x2545_F491_4F6C_DD1D));
    let skewed = match routing {
        Routing::Random => false,
        Routing::Skewed => true,
        Routing::Half => t.is_multiple_of(2),
    };
    let mut picked: Vec<u32> = Vec::with_capacity(9);
    if skewed {
        let mut s = SKEWED.to_vec();
        while !s.is_empty() {
            picked.push(s.remove(r() as usize % s.len()));
        }
    }
    while picked.len() < 9 {
        let e = (r() % EXPERTS as u64) as u32;
        if !picked.contains(&e) && !(skewed && SKEWED.contains(&e)) {
            picked.push(e);
        }
    }
    picked.try_into().expect("9 experts")
}

/// The embedded hidden of an `m`-row bucket under `routing`.
pub fn moe_x(routing: Routing, m: usize, seed: u64) -> Vec<u16> {
    let mut x = Vec::with_capacity(m * HU);
    let mut noise = mix(seed ^ 0x4E01_5E00);
    for t in 0..m {
        let named = named_experts(routing, t, seed);
        for i in 0..HU {
            let signal: f64 = (named.iter().zip(LOGITS))
                .map(|(&e, l)| l * hadamard(e as usize + 1, i))
                .sum();
            x.push(bits(KAPPA * signal + ETA * unit(&mut noise)));
        }
    }
    x
}

// ── The f64 reference and the bf16 emulation ───────────────────────

fn bf(b: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(b) << 16))
}

/// `v` rounded as a kernel's store rounds its f32 result to bf16.
fn rb(v: f64) -> f64 {
    bf(bits(v))
}

fn dot(a: &[f64], w: &[u16]) -> f64 {
    a.iter().zip(w).map(|(x, &b)| x * bf(b)).sum()
}

fn silu(v: f64) -> f64 {
    v / (1.0 + (-v).exp())
}

fn sigmoid(v: f64) -> f64 {
    1.0 / (1.0 + (-v).exp())
}

/// One token's references: the f64 result, the emulation's, the emulation's picks, whether the
/// emulated logits hold a near-tie among the 9 best, and the six mutated f64 results.
pub struct TokenRef {
    pub y: Vec<f64>,
    pub emu: Vec<u16>,
    /// Ascending, as the routing kernel stores them: slot 7 the best.
    pub picks: [u32; 8],
    pub near_tie: bool,
    pub mutated: [Vec<f64>; 6],
}

/// The six mutations of the reference, in #857's order.
pub const MUTATIONS: [&str; 6] = [
    "(i) the 9th expert instead of the 8th",
    "(ii) expert index e+1 mod 256",
    "(iii) no renorm",
    "(iv) gate and up halves swapped",
    "(v) no sigmoid on the shared gate",
    "(vi) uniform 1/8 combine weights",
];

/// The norm, rounded when `round`.
fn norm(w: &MoeWeights, x: &[u16], round: bool) -> Vec<f64> {
    let ms = x.iter().map(|&v| bf(v) * bf(v)).sum::<f64>() / HU as f64;
    let inv = 1.0 / (ms + f64::from(EPS)).sqrt();
    let r = |v: f64| if round { rb(v) } else { v };
    (x.iter().zip(&w.g))
        .map(|(&v, &g)| r(bf(v) * inv * (1.0 + bf(g))))
        .collect()
}

/// The 9 best experts by logit (equal logits by the higher index), best first, and every
/// expert's softmax probability.
fn route(logits: &[f64]) -> ([usize; 9], Vec<f64>) {
    let mut order: Vec<usize> = (0..EU).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(b.cmp(&a)));
    let mx = logits[order[0]];
    let e: Vec<f64> = logits.iter().map(|l| (l - mx).exp()).collect();
    let s: f64 = e.iter().sum();
    (
        order[..9].try_into().expect("9"),
        e.iter().map(|v| v / s).collect(),
    )
}

/// One expert's gate and up rows of `n`.
fn gate_up(w: &MoeWeights, n: &[f64], e: usize, round: bool) -> (Vec<f64>, Vec<f64>) {
    let r = |v: f64| if round { rb(v) } else { v };
    (
        (0..IU).map(|j| r(dot(n, w.gate(e, j)))).collect(),
        (0..IU).map(|j| r(dot(n, w.up(e, j)))).collect(),
    )
}

fn act(g: &[f64], u: &[f64], round: bool) -> Vec<f64> {
    (g.iter().zip(u))
        .map(|(&g, &u)| {
            let v = silu(g) * u;
            if round { rb(v) } else { v }
        })
        .collect()
}

fn down(w: &MoeWeights, a: &[f64], e: usize, round: bool) -> Vec<f64> {
    (0..HU)
        .map(|j| {
            let v = dot(a, w.down(e, j));
            if round { rb(v) } else { v }
        })
        .collect()
}

/// The shared expert's output and its scalar gate's logit.
fn shared(w: &MoeWeights, n: &[f64], round: bool) -> (Vec<f64>, f64) {
    let r = |v: f64| if round { rb(v) } else { v };
    let sg: Vec<f64> = (0..IU).map(|j| r(dot(n, &w.wsg[j * HU..][..HU]))).collect();
    let su: Vec<f64> = (0..IU).map(|j| r(dot(n, &w.wsu[j * HU..][..HU]))).collect();
    let sa = act(&sg, &su, round);
    let sd = (0..HU)
        .map(|j| r(dot(&sa, &w.wsd[j * IU..][..IU])))
        .collect();
    (sd, r(dot(n, &w.wgate)))
}

/// `Σ_j weights[j] · down rows[j]`.
fn combine(rows: &[Vec<f64>], weights: &[f64]) -> Vec<f64> {
    (0..HU)
        .map(|i| rows.iter().zip(weights).map(|(d, s)| s * d[i]).sum())
        .collect()
}

/// `routed + sd · gate`, plus `x`.
fn finish(routed: &[f64], sd: &[f64], gate: f64, x: &[u16]) -> Vec<f64> {
    (0..HU)
        .map(|i| routed[i] + sd[i] * gate + bf(x[i]))
        .collect()
}

/// A bf16 value's spacing at `v`.
fn ulp(v: f64) -> f64 {
    let a = v.abs().max(f64::MIN_POSITIVE);
    2f64.powi(a.log2().floor() as i32 - 7)
}

/// The references of one token row `x`.
pub fn token_reference(w: &MoeWeights, x: &[u16]) -> TokenRef {
    // The emulation: every value the tape stores rounded to bf16 there, the scores f32.
    let n = norm(w, x, true);
    let logits: Vec<f64> = (0..EU).map(|e| rb(dot(&n, w.router(e)))).collect();
    let (best, p) = route(&logits);
    let near_tie = best.windows(2).any(|b| {
        let (a, c) = (logits[b[0]], logits[b[1]]);
        a - c <= ulp(a.abs().max(c.abs()))
    });
    let top: f64 = best[..8].iter().map(|&e| p[e]).sum();
    let mut rows = Vec::with_capacity(8);
    let mut scores = Vec::with_capacity(8);
    for &e in &best[..8] {
        let (g, u) = gate_up(w, &n, e, true);
        rows.push(down(w, &act(&g, &u, true), e, true));
        scores.push(f64::from((p[e] / top) as f32));
    }
    let routed: Vec<f64> = combine(&rows, &scores).into_iter().map(rb).collect();
    let (sd, gl) = shared(w, &n, true);
    let emu = (0..HU)
        .map(|i| {
            let mix = rb(routed[i] + sd[i] * sigmoid(gl));
            half::bf16::from_f32(mix as f32 + bf(x[i]) as f32).to_bits()
        })
        .collect();
    let mut picks = [0u32; 8];
    for (j, &e) in best[..8].iter().enumerate() {
        picks[7 - j] = e as u32;
    }

    // The f64 reference, with its own routing, and the mutations of it.
    let n = norm(w, x, false);
    let logits: Vec<f64> = (0..EU).map(|e| dot(&n, w.router(e))).collect();
    let (best, p) = route(&logits);
    let mut cache: std::collections::HashMap<usize, (Vec<f64>, Vec<f64>)> = Default::default();
    let mut gu = |e: usize| {
        cache
            .entry(e)
            .or_insert_with(|| gate_up(w, &n, e, false))
            .clone()
    };
    let routed_by = |experts: &[usize], weights: &[f64], gu: &mut dyn FnMut(usize) -> _| {
        let rows: Vec<Vec<f64>> = (experts.iter())
            .map(|&e| {
                let (g, u): (Vec<f64>, Vec<f64>) = gu(e);
                down(w, &act(&g, &u, false), e, false)
            })
            .collect();
        combine(&rows, weights)
    };
    let pick8 = &best[..8];
    let top: f64 = pick8.iter().map(|&e| p[e]).sum();
    let renorm: Vec<f64> = pick8.iter().map(|&e| p[e] / top).collect();
    let routed = routed_by(pick8, &renorm, &mut gu);
    let (sd, gl) = shared(w, &n, false);
    let y = finish(&routed, &sd, sigmoid(gl), x);
    let ninth: Vec<usize> = best[..7].iter().copied().chain([best[8]]).collect();
    let top9: f64 = ninth.iter().map(|&e| p[e]).sum();
    let w9: Vec<f64> = ninth.iter().map(|&e| p[e] / top9).collect();
    let shifted: Vec<usize> = pick8.iter().map(|&e| (e + 1) % EU).collect();
    let raw: Vec<f64> = pick8.iter().map(|&e| p[e]).collect();
    let swapped: Vec<Vec<f64>> = (pick8.iter())
        .map(|&e| {
            let (g, u) = gu(e);
            down(w, &act(&u, &g, false), e, false)
        })
        .collect();
    let mutated = [
        finish(&routed_by(&ninth, &w9, &mut gu), &sd, sigmoid(gl), x),
        finish(&routed_by(&shifted, &renorm, &mut gu), &sd, sigmoid(gl), x),
        finish(&routed_by(pick8, &raw, &mut gu), &sd, sigmoid(gl), x),
        finish(&combine(&swapped, &renorm), &sd, sigmoid(gl), x),
        finish(&routed, &sd, gl, x),
        finish(&routed_by(pick8, &[0.125; 8], &mut gu), &sd, sigmoid(gl), x),
    ];
    TokenRef {
        y,
        emu,
        picks,
        near_tie,
        mutated,
    }
}

/// [`token_reference`] of each token `t` of `tokens` (rows of `x`), on every core.
pub fn references(w: &MoeWeights, x: &[u16], tokens: &[usize]) -> Vec<TokenRef> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let per = tokens.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        let parts: Vec<_> = (tokens.chunks(per))
            .map(|part| {
                s.spawn(move || {
                    (part.iter())
                        .map(|&t| token_reference(w, &x[t * HU..][..HU]))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        (parts.into_iter())
            .flat_map(|p| p.join().expect("a reference thread"))
            .collect()
    })
}

/// One case's verdict: the GPU's and the emulation's errors against f64 over the tokens that are
/// not near-ties, the picks that differ from the emulation's, and each mutation's fraction.
pub struct Judgement {
    pub tokens: usize,
    pub near_ties: usize,
    pub gpu: (f64, f64),
    pub emu: (f64, f64),
    pub out_of_bound: usize,
    pub nonfinite: usize,
    pub picks_differ: usize,
    /// Per mutation: the tokens it changes, and those of them out of bound.
    pub mutations: [(usize, usize); 6],
}

impl Judgement {
    /// The measured-noise rule: CUDA's RMS and max error each at most twice the emulation's, the
    /// picks the emulation's, every output finite.
    pub fn pass(&self) -> bool {
        self.gpu.0 <= 2.0 * self.emu.0
            && self.gpu.1 <= 2.0 * self.emu.1
            && self.picks_differ == 0
            && self.nonfinite == 0
            && self.tokens > self.near_ties
    }

    /// Whether every mutation is out of bound on at least half the tokens it changes.
    pub fn mutations_caught(&self) -> bool {
        (self.mutations.iter()).all(|&(changed, out)| changed > 0 && 2 * out >= changed)
    }
}

/// Judge the GPU's result `y` (`[m, H]`) and picks (`[m, 8]`) on the referenced `tokens`. An
/// element is out of bound past twice the emulation's max error, the bound a mutation is held to
/// too.
pub fn judge(refs: &[TokenRef], tokens: &[usize], y: &[u16], picks: &[u32]) -> Judgement {
    let (mut sq, mut mx, mut esq, mut emx, mut n) = (0.0, 0.0f64, 0.0, 0.0f64, 0usize);
    let (mut nonfinite, mut picks_differ, mut near_ties) = (0, 0, 0);
    for (r, &t) in refs.iter().zip(tokens) {
        if r.near_tie {
            near_ties += 1;
            continue;
        }
        picks_differ += usize::from(picks[t * 8..][..8] != r.picks);
        for i in 0..HU {
            let g = bf(y[t * HU + i]);
            nonfinite += usize::from(!g.is_finite());
            let (e, ee) = ((g - r.y[i]).abs(), (bf(r.emu[i]) - r.y[i]).abs());
            sq += e * e;
            esq += ee * ee;
            mx = mx.max(e);
            emx = emx.max(ee);
            n += 1;
        }
    }
    let bound = 2.0 * emx;
    let mut out_of_bound = 0;
    let mut mutations = [(0usize, 0usize); 6];
    for (r, &t) in refs.iter().zip(tokens) {
        if r.near_tie {
            continue;
        }
        out_of_bound += (0..HU)
            .filter(|&i| (bf(y[t * HU + i]) - r.y[i]).abs() > bound)
            .count();
        for (k, m) in r.mutated.iter().enumerate() {
            let worst = (m.iter().zip(&r.y))
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            if worst > 0.0 {
                mutations[k].0 += 1;
                mutations[k].1 += usize::from(worst > bound);
            }
        }
    }
    let rms = |s: f64| (s / n.max(1) as f64).sqrt();
    Judgement {
        tokens: tokens.len(),
        near_ties,
        gpu: (rms(sq), mx),
        emu: (rms(esq), emx),
        out_of_bound,
        nonfinite,
        picks_differ,
        mutations,
    }
}
