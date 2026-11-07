// SPDX-License-Identifier: Apache-2.0
//! Gemma 4 26B-A4B's MoE block at its real geometry (#924), the canonical the Gemma MoE block tape
//! test lowers: the sparse block of [`super::gemma_moe`] (the router's own pre-norm, its 128-expert
//! projection, the Gemma routing, the dense SwitchGLU bank's GeGLU experts and the combine) with
//! the dense GeGLU MLP beside it and the residual add, `y = routed(n2) + mlp(n) + x`:
//!
//! ```text
//!  0 n      = RmsNorm(x, g)                        [M, 2816]   GainConvention::Scale
//!  1 gate   = n · Wg                               [M, 2112]   the 64-column GEMM at M ≥ 2
//!  2 up     = n · Wu                               [M, 2112]
//!  3 s      = Gelu(gate)                                       3-4 fold into one GeluMul
//!  4 a      = s · up
//!  5 dense  = a · Wd                               [M, 2816]
//!  6 n2     = RmsNorm(x, g2)                       [M, 2816]   the experts' own pre-norm
//!  7 xr     = RouterNorm(x, router.scale)          [M, 2816]   into the MoE scratch
//!  8 logits = RouterLogits(xr, router.proj)        [M, 128]    the 64-column GEMM at M ≥ 2
//!  9-14       the Gemma route: top-8 of the raw logits, × hidden^-0.5, softmax, × per_expert_scale
//! 15 rows   = ExpertSort(n2, idx)                  sorted at M ≥ 8
//! 16 eg     = ExpertMatmul(Gate, rows, bank)       [P, 704]    n_tile 64
//! 17 eu     = ExpertMatmul(Up, rows, bank)         [P, 704]
//! 18 act    = ExpertGatedAct(Gelu, eg, eu)
//! 19 down   = ExpertMatmul(Down, act, bank)        [P, 2816]   n_tile 256
//! 20 back   = ExpertUnsort(down, rows)
//! 21 routed = ExpertCombine(back, w)               [M, 2816]
//! 22 sum    = routed + dense
//! 23 y      = sum + x                              the result
//! ```
//!
//! The shapes are `gemma-4-26b-a4b-it`'s: hidden 2816, 128 experts, top-8, expert width 704, the
//! dense MLP's 2112, `rms_norm_eps` 1e-6. The routing is HF's and `golden_gen_carrier.py`'s
//! `gemma_moe`: `xr = x · rsqrt(mean(x²) + eps) · router.scale`, `logits = xr · router.projᵀ`,
//! the top-8 of the raw logits, `softmax(top · hidden^-0.5) · per_expert_scale[idx]`, GELU (tanh)
//! experts, the weighted sum, as the DSL wires it (`gemma4-moe.py`: the router reads the residual,
//! the experts `pre_feedforward_layernorm_2`'s rows, the dense MLP `pre_feedforward_layernorm`'s).
//! The real layer's post-norms and layer scalar are #921's row steps, not this block's. Ops 7-21
//! are one MoE expansion. Sources: 0 `x` (the embedded hidden), 1 `g`, 2 `Wg`, 3 `Wu`, 4 `Wd`,
//! 5 `g2`, then the router's and the bank's bundle tensors: 6 `router.scale` `[1, H]`,
//! 7 `router.proj` `[H, E]`, 8 `per_expert_scale` `[1, E]`, 9 the gate `[E·I, H]`, 10 the up
//! `[E·I, H]`, 11 the down `[E·H, I]`. Every dense weight is declared `[K, N]`; its device bytes
//! are `[N, K]`, and the bank is `step::expert_bank`'s bytes.
//!
//! The reference and the emulation also play [`super::gemma_moe`]'s small block ([`SMALL`]: H 256,
//! I 128, the experts on `x` itself, the result the combine).

use scratchy_subtile::handoff::{
    BundleTensor, Expansion, ExpansionId, LoweredDecode, SourceBinding, UnrollIndex,
};
use scratchy_subtile::lower::{ExpertWeight, GemmWeight, InputRef, LoweringInput, OpDesc};
use scratchy_subtile::subtile_ir::{
    EwKind, ExpertBundle, ExpertProj, GainConvention, GatedAct, NumExperts, RouterBundle,
    SharedExpertBound, SourceShape, SubOp, TopK,
};
use scratchy_target_cuda_compiler::canonical::CudaModelFacts;

use super::chains::{GEMMA4_FACTS, with_hidden};
use super::moe_block::Routing;

/// Gemma 4 26B-A4B's hidden size, expert count, picks, expert width and dense MLP width.
pub const H: u32 = 2816;
pub const EXPERTS: u32 = 128;
pub const TOP_K: u32 = 8;
pub const I: u32 = 704;
pub const DENSE_I: u32 = 2112;
/// Its `rms_norm_eps`.
pub const EPS: f32 = 1e-6;
/// The router's and the experts' bundle ids, and the block's layer.
pub const ROUTER: u32 = 20;
pub const BANK: u32 = 21;
pub const LAYER: u64 = 0;

/// The block lowers with Gemma 4's facts (the 64-column tile).
pub const GEMMA_MOE_BLOCK_FACTS: CudaModelFacts = GEMMA4_FACTS;

/// The block's ops at `m` rows, as the table above.
pub fn gemma_moe_block_input(m: u32) -> LoweringInput {
    let nz = |n| std::num::NonZeroU32::new(n).expect("non-zero");
    let (experts, k, bundle) = (
        NumExperts::new(nz(EXPERTS)),
        TopK::new(nz(TOP_K)),
        ExpertBundle::SwitchGlu,
    );
    let gemma = RouterBundle::Gemma;
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
    let row = |cols| SourceShape { rows: 1, cols };
    LoweringInput {
        exported: None,
        sources: vec![
            SourceShape { rows: m, cols: H },
            row(H),
            SourceShape {
                rows: H,
                cols: DENSE_I,
            },
            SourceShape {
                rows: H,
                cols: DENSE_I,
            },
            SourceShape {
                rows: DENSE_I,
                cols: H,
            },
            row(H),
            row(H),
            SourceShape {
                rows: H,
                cols: EXPERTS,
            },
            row(EXPERTS),
            SourceShape {
                rows: EXPERTS * I,
                cols: H,
            },
            SourceShape {
                rows: EXPERTS * I,
                cols: H,
            },
            SourceShape {
                rows: EXPERTS * H,
                cols: I,
            },
        ],
        ops: vec![
            op(
                SubOp::RmsNorm {
                    eps: EPS,
                    gain: GainConvention::Scale,
                },
                vec![Ext(0), Ext(1)],
            ),
            matmul(DENSE_I, Op(0), 2),
            matmul(DENSE_I, Op(0), 3),
            op(SubOp::Elementwise(EwKind::Gelu), vec![Op(1)]),
            op(SubOp::Elementwise(EwKind::Mul), vec![Op(3), Op(2)]),
            matmul(H, Op(4), 4),
            op(
                SubOp::RmsNorm {
                    eps: EPS,
                    gain: GainConvention::Scale,
                },
                vec![Ext(0), Ext(5)],
            ),
            op(
                SubOp::RouterNorm {
                    eps: EPS,
                    router: gemma,
                },
                vec![Ext(0), Ext(6)],
            ),
            op(
                SubOp::RouterLogits {
                    experts,
                    router: gemma,
                },
                vec![Op(7), Ext(7)],
            ),
            op(SubOp::RouteArgsort, vec![Op(8)]),
            op(SubOp::RouteTopK { k }, vec![Op(9)]),
            op(SubOp::RouteGatherScores, vec![Op(8), Op(10)]),
            op(
                SubOp::RouteScale {
                    scale: (H as f32).powf(-0.5),
                },
                vec![Op(11)],
            ),
            op(SubOp::RouteSoftmax, vec![Op(12)]),
            op(
                SubOp::RouteExpertScale { router: gemma },
                vec![Op(13), Op(10), Ext(8)],
            ),
            op(
                SubOp::ExpertSort { experts, k, bundle },
                vec![Op(6), Op(10)],
            ),
            expert(ExpertProj::Gate, I, vec![Op(15), Op(15), Ext(9)]),
            expert(ExpertProj::Up, I, vec![Op(15), Op(15), Ext(10)]),
            op(
                SubOp::ExpertGatedAct {
                    act: GatedAct::Gelu,
                },
                vec![Op(16), Op(17)],
            ),
            expert(ExpertProj::Down, H, vec![Op(18), Op(15), Ext(11)]),
            op(SubOp::ExpertUnsort, vec![Op(19), Op(15)]),
            op(
                SubOp::ExpertCombine {
                    hidden: H,
                    shared: SharedExpertBound(None),
                },
                vec![Op(20), Op(14)],
            ),
            op(SubOp::Elementwise(EwKind::Add), vec![Op(21), Op(5)]),
            op(SubOp::Elementwise(EwKind::Add), vec![Op(22), Ext(0)]),
        ],
        result: 23,
    }
}

/// The bundle tensor each weight source is, by source index.
pub const TENSORS: [(usize, u32, BundleTensor); 6] = [
    (6, ROUTER, BundleTensor::RouterScale),
    (7, ROUTER, BundleTensor::RouterGate),
    (8, ROUTER, BundleTensor::RouterPerExpertScale),
    (9, BANK, BundleTensor::ExpertGateW),
    (10, BANK, BundleTensor::ExpertUpW),
    (11, BANK, BundleTensor::ExpertDownW),
];

/// The block at `m` rows as the front end hands it over: source 0 the embedded hidden, the router's
/// and the bank's weights one bundle tensor each of layer 0's bundles, ops 7-21 one MoE expansion.
pub fn gemma_moe_block(m: u32) -> LoweredDecode {
    let mut l = with_hidden(gemma_moe_block_input(m));
    for (ix, id, tensor) in TENSORS {
        l.bindings[ix] = SourceBinding::BundleTensor {
            id,
            index: Some(UnrollIndex(LAYER)),
            tensor,
        };
    }
    for (i, e) in l.op_expansion.iter_mut().enumerate() {
        if (7..=21).contains(&i) {
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
// The weights are #924's: the router uniform in ±1/√H, each expert's gate and up rows in ±1/√H and
// its down rows in ±1/√I, `router.scale` and `per_expert_scale` uniform in [0.5, 2]; the dense MLP's
// gain in [0.5, 2], its gate and up rows in ±1/√H and its down rows in ±1/√2112. Each token row is
// `Σ_k L[k] · r̂_{e_k}` (`r̂_e` the unit direction of router row `e`) plus uniform noise, over the
// 9 experts `e_k` its routing case names: after the router norm the 8 best named experts' logits
// sit well above every unnamed expert's, and the 9th named expert's below the 8th's.

/// The weights of the 8 picks and the 9th, best first: the picks close together, the 9th apart.
const LOGITS: [f64; 9] = [3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 1.0];
/// The noise's amplitude in `x`.
const ETA: f64 = 0.1;
/// The 8 experts every token of a skewed routing picks.
pub const SKEWED: [u32; 8] = [5, 19, 33, 60, 71, 90, 101, 126];

/// A block's geometry: the real one ([`REAL`]) or [`super::gemma_moe`]'s small one ([`SMALL`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub h: usize,
    pub i: usize,
    /// The dense MLP's width, `None` for the small block (no MLP, no residual, the experts on `x`).
    pub dense: Option<usize>,
}

/// The real block's geometry.
pub const REAL: Geometry = Geometry {
    h: H as usize,
    i: I as usize,
    dense: Some(DENSE_I as usize),
};
/// [`super::gemma_moe`]'s small block.
pub const SMALL: Geometry = Geometry {
    h: super::gemma_moe::H as usize,
    i: super::gemma_moe::I as usize,
    dense: None,
};

const EU: usize = EXPERTS as usize;
const KU: usize = TOP_K as usize;

/// A block's weights, as the device holds them (`[N, K]` bf16): the dense MLP's gain, gate, up and
/// down and the experts' pre-norm gain `g2` (empty for [`SMALL`]), the router's scale `[H]`,
/// projection `[128, H]` and per-expert scale `[128]`, and the expert bank (`step::expert_bank`'s
/// layout).
pub struct GemmaMoeWeights {
    pub geometry: Geometry,
    pub g: Vec<u16>,
    pub wg: Vec<u16>,
    pub wu: Vec<u16>,
    pub wd: Vec<u16>,
    pub g2: Vec<u16>,
    pub rscale: Vec<u16>,
    pub router: Vec<u16>,
    pub pes: Vec<u16>,
    pub bank: Vec<u16>,
}

impl GemmaMoeWeights {
    fn router_row(&self, e: usize) -> &[u16] {
        let h = self.geometry.h;
        &self.router[e * h..][..h]
    }
    fn gate(&self, e: usize, j: usize) -> &[u16] {
        let Geometry { h, i, .. } = self.geometry;
        &self.bank[(e * 2 * i + j) * h..][..h]
    }
    fn up(&self, e: usize, j: usize) -> &[u16] {
        let Geometry { h, i, .. } = self.geometry;
        &self.bank[(e * 2 * i + i + j) * h..][..h]
    }
    fn down(&self, e: usize, j: usize) -> &[u16] {
        let Geometry { h, i, .. } = self.geometry;
        &self.bank[EU * 2 * i * h + (e * h + j) * i..][..i]
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

/// A uniform draw in `[0, 1)`.
fn unit01(r: &mut impl FnMut() -> u64) -> f64 {
    (r() >> 11) as f64 / (1u64 << 53) as f64
}

fn bits(v: f64) -> u16 {
    half::bf16::from_f32(v as f32).to_bits()
}

/// `out` bf16 uniform in `[lo, hi)`, filled by every core in chunks of their own seeds.
fn uniform(out: &mut [u16], seed: u64, (lo, hi): (f64, f64)) {
    const CHUNK: usize = 1 << 20;
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut chunks: Vec<(usize, &mut [u16])> = out.chunks_mut(CHUNK).enumerate().collect();
    let per = chunks.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        while !chunks.is_empty() {
            let take = chunks.len().min(per);
            let mine: Vec<_> = chunks.drain(..take).collect();
            s.spawn(move || {
                for (c, part) in mine {
                    let mut r = mix(seed ^ (c as u64).wrapping_mul(0x5851_F42D_4C95_7F2D));
                    for v in part.iter_mut() {
                        *v = bits(lo + (hi - lo) * unit01(&mut r));
                    }
                }
            });
        }
    });
}

fn filled(len: usize, seed: u64, range: (f64, f64)) -> Vec<u16> {
    let mut v = vec![0u16; len];
    uniform(&mut v, seed, range);
    v
}

/// A block's synthetic weights (about 1.6 GB of bf16 at [`REAL`]).
pub fn gemma_moe_weights(geometry: Geometry) -> GemmaMoeWeights {
    let Geometry { h, i, dense } = geometry;
    let (ah, ai) = (1.0 / (h as f64).sqrt(), 1.0 / (i as f64).sqrt());
    let scales = (0.5, 2.0);
    let seed = 0x6E4B_0000 ^ h as u64;
    let gate_up = EU * 2 * i * h;
    let mut bank = vec![0u16; gate_up + EU * h * i];
    uniform(&mut bank[..gate_up], seed ^ 1, (-ah, ah));
    uniform(&mut bank[gate_up..], seed ^ 2, (-ai, ai));
    let (g, wg, wu, wd, g2) = match dense {
        Some(di) => {
            let ad = 1.0 / (di as f64).sqrt();
            (
                filled(h, seed ^ 3, scales),
                filled(di * h, seed ^ 4, (-ah, ah)),
                filled(di * h, seed ^ 5, (-ah, ah)),
                filled(h * di, seed ^ 6, (-ad, ad)),
                filled(h, seed ^ 10, scales),
            )
        }
        None => Default::default(),
    };
    GemmaMoeWeights {
        geometry,
        g,
        wg,
        wu,
        wd,
        g2,
        rscale: filled(h, seed ^ 7, scales),
        router: filled(EU * h, seed ^ 8, (-ah, ah)),
        pes: filled(EU, seed ^ 9, scales),
        bank,
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
        let e = (r() % EU as u64) as u32;
        if !picked.contains(&e) && !(skewed && SKEWED.contains(&e)) {
            picked.push(e);
        }
    }
    picked.try_into().expect("9 experts")
}

/// The embedded hidden of an `m`-row bucket under `routing`.
pub fn gemma_moe_x(w: &GemmaMoeWeights, routing: Routing, m: usize, seed: u64) -> Vec<u16> {
    let h = w.geometry.h;
    let unit_rows: Vec<Vec<f64>> = (0..EU)
        .map(|e| {
            let r: Vec<f64> = w.router_row(e).iter().map(|&b| bf(b)).collect();
            let norm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            r.into_iter().map(|v| v / norm).collect()
        })
        .collect();
    let mut x = Vec::with_capacity(m * h);
    let mut noise = mix(seed ^ 0x4E01_5E00);
    for t in 0..m {
        let named = named_experts(routing, t, seed);
        let row: Vec<u16> = (0..h)
            .map(|i| {
                let signal: f64 = (named.iter().zip(LOGITS))
                    .map(|(&e, l)| l * unit_rows[e as usize][i])
                    .sum();
                bits(signal + ETA * (2.0 * unit01(&mut noise) - 1.0))
            })
            .collect();
        x.extend(row);
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

fn gelu(v: f64) -> f64 {
    let c = (2.0 / std::f64::consts::PI).sqrt();
    0.5 * v * (1.0 + (c * (v + 0.044715 * v * v * v)).tanh())
}

fn silu(v: f64) -> f64 {
    v / (1.0 + (-v).exp())
}

/// One token's references: the f64 result, the emulation's, the f64 and emulated logits, the
/// emulation's picks (ascending by index), and the seven mutated f64 results.
pub struct TokenRef {
    pub y: Vec<f64>,
    pub emu: Vec<u16>,
    pub logits: Vec<f64>,
    pub emu_logits: Vec<f64>,
    pub picks: [u32; 8],
    pub mutated: [Vec<f64>; 7],
}

/// The seven self-checks of #924, in its order.
pub const GEMMA_MUTATIONS: [&str; 7] = [
    "(1) the 9th expert instead of the 8th",
    "(2) expert e+1 mod 128",
    "(3) no x hidden^-0.5",
    "(4) no per-expert scale",
    "(5) gate and up halves swapped",
    "(6) SiLU instead of GELU",
    "(7) uniform 1/8 combine weights",
];

/// `x̂ · gain` of one row, rounded when `round`.
fn rms(x: &[u16], gain: &[u16], round: bool) -> Vec<f64> {
    let ms = x.iter().map(|&v| bf(v) * bf(v)).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ms + f64::from(EPS)).sqrt();
    let r = |v: f64| if round { rb(v) } else { v };
    (x.iter().zip(gain))
        .map(|(&v, &g)| r(bf(v) * inv * bf(g)))
        .collect()
}

/// The 9 best experts by logit (equal logits by the higher index), best first.
fn best9(logits: &[f64]) -> [usize; 9] {
    let mut order: Vec<usize> = (0..EU).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(b.cmp(&a)));
    order[..9].try_into().expect("9")
}

/// The Gemma weights of `picks`: `softmax(logits · scale)` over them, times the per-expert scale
/// when `pes`.
fn route_weights(
    w: &GemmaMoeWeights,
    logits: &[f64],
    picks: &[usize],
    scale: f64,
    pes: bool,
) -> Vec<f64> {
    let z: Vec<f64> = picks.iter().map(|&e| logits[e] * scale).collect();
    let mx = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let ez: Vec<f64> = z.iter().map(|v| (v - mx).exp()).collect();
    let s: f64 = ez.iter().sum();
    (ez.iter().zip(picks))
        .map(|(v, &e)| v / s * if pes { bf(w.pes[e]) } else { 1.0 })
        .collect()
}

/// One expert's down row of `n` through its GeGLU (`act` the activation; `swap` the gate and up
/// exchanged), every stored value rounded when `round`.
fn expert(
    w: &GemmaMoeWeights,
    n: &[f64],
    e: usize,
    (act, swap, round): (fn(f64) -> f64, bool, bool),
) -> Vec<f64> {
    let Geometry { h, i, .. } = w.geometry;
    let r = |v: f64| if round { rb(v) } else { v };
    let a: Vec<f64> = (0..i)
        .map(|j| {
            let (g, u) = (r(dot(n, w.gate(e, j))), r(dot(n, w.up(e, j))));
            let (g, u) = if swap { (u, g) } else { (g, u) };
            r(act(g) * u)
        })
        .collect();
    (0..h).map(|j| r(dot(&a, w.down(e, j)))).collect()
}

/// The dense MLP's output of `n`, rounded where the tape stores when `round`.
fn dense_mlp(w: &GemmaMoeWeights, n: &[f64], di: usize, round: bool) -> Vec<f64> {
    let h = w.geometry.h;
    let r = |v: f64| if round { rb(v) } else { v };
    let a: Vec<f64> = (0..di)
        .map(|j| {
            let g = r(dot(n, &w.wg[j * h..][..h]));
            let u = r(dot(n, &w.wu[j * h..][..h]));
            r(gelu(g) * u)
        })
        .collect();
    (0..h).map(|j| r(dot(&a, &w.wd[j * di..][..di]))).collect()
}

/// `Σ_j weights[j] · rows[j]`.
fn combine(rows: &[Vec<f64>], weights: &[f64], h: usize) -> Vec<f64> {
    (0..h)
        .map(|c| rows.iter().zip(weights).map(|(d, s)| s * d[c]).sum())
        .collect()
}

/// The references of one token row `x`.
pub fn token_reference(w: &GemmaMoeWeights, x: &[u16]) -> TokenRef {
    let Geometry { h, dense, .. } = w.geometry;
    let scale = (h as f64).powf(-0.5);
    let xf: Vec<f64> = x.iter().map(|&v| bf(v)).collect();
    let gelu_act = (gelu as fn(f64) -> f64, false);

    // The emulation: every value the tape stores rounded to bf16 there, the route weights f32.
    let xr = rms(x, &w.rscale, true);
    let emu_logits: Vec<f64> = (0..EU).map(|e| rb(dot(&xr, w.router_row(e)))).collect();
    let best = best9(&emu_logits);
    let weights: Vec<f64> =
        route_weights(w, &emu_logits, &best[..8], f64::from(scale as f32), true)
            .into_iter()
            .map(|v| f64::from(v as f32))
            .collect();
    let (n_emu, n2_emu) = match dense {
        Some(_) => (rms(x, &w.g, true), rms(x, &w.g2, true)),
        None => (xf.clone(), xf.clone()),
    };
    let rows: Vec<Vec<f64>> = (best[..8].iter())
        .map(|&e| expert(w, &n2_emu, e, (gelu_act.0, gelu_act.1, true)))
        .collect();
    let routed: Vec<f64> = combine(&rows, &weights, h).into_iter().map(rb).collect();
    let emu = match dense {
        Some(di) => {
            let d = dense_mlp(w, &n_emu, di, true);
            (0..h).map(|c| bits(rb(routed[c] + d[c]) + xf[c])).collect()
        }
        None => routed.iter().map(|&v| bits(v)).collect(),
    };
    let mut picks: [u32; 8] = std::array::from_fn(|j| best[j] as u32);
    picks.sort_unstable();

    // The f64 reference, with its own routing, and the mutations of it.
    let xr = rms(x, &w.rscale, false);
    let logits: Vec<f64> = (0..EU).map(|e| dot(&xr, w.router_row(e))).collect();
    let best = best9(&logits);
    let (n, n2) = match dense {
        Some(_) => (rms(x, &w.g, false), rms(x, &w.g2, false)),
        None => (xf.clone(), xf.clone()),
    };
    let mut cache: std::collections::HashMap<usize, Vec<f64>> = Default::default();
    let mut down = |e: usize| {
        (cache.entry(e))
            .or_insert_with(|| expert(w, &n2, e, (gelu, false, false)))
            .clone()
    };
    let finish = |routed: Vec<f64>| -> Vec<f64> {
        match dense {
            Some(di) => {
                let d = dense_mlp(w, &n, di, false);
                (0..h).map(|c| routed[c] + d[c] + xf[c]).collect()
            }
            None => routed,
        }
    };
    let pick8: Vec<usize> = best[..8].to_vec();
    let wts = route_weights(w, &logits, &pick8, scale, true);
    let rows8: Vec<Vec<f64>> = pick8.iter().map(|&e| down(e)).collect();
    let routed = combine(&rows8, &wts, h);
    // The dense MLP and the residual are the same in every mutation: computed once.
    let tail: Vec<f64> = finish(vec![0.0; h]);
    let with_tail = |r: Vec<f64>| -> Vec<f64> { r.iter().zip(&tail).map(|(a, b)| a + b).collect() };
    let y = with_tail(routed);
    let ninth: Vec<usize> = best[..7].iter().copied().chain([best[8]]).collect();
    let w9 = route_weights(w, &logits, &ninth, scale, true);
    let rows9: Vec<Vec<f64>> = ninth.iter().map(|&e| down(e)).collect();
    let shifted: Vec<usize> = pick8.iter().map(|&e| (e + 1) % EU).collect();
    let rows_shifted: Vec<Vec<f64>> = shifted.iter().map(|&e| down(e)).collect();
    let swapped: Vec<Vec<f64>> = (pick8.iter())
        .map(|&e| expert(w, &n2, e, (gelu, true, false)))
        .collect();
    let silued: Vec<Vec<f64>> = (pick8.iter())
        .map(|&e| expert(w, &n2, e, (silu, false, false)))
        .collect();
    let mutated = [
        with_tail(combine(&rows9, &w9, h)),
        with_tail(combine(&rows_shifted, &wts, h)),
        with_tail(combine(
            &rows8,
            &route_weights(w, &logits, &pick8, 1.0, true),
            h,
        )),
        with_tail(combine(
            &rows8,
            &route_weights(w, &logits, &pick8, scale, false),
            h,
        )),
        with_tail(combine(&swapped, &wts, h)),
        with_tail(combine(&silued, &wts, h)),
        with_tail(combine(&rows8, &[0.125; 8], h)),
    ];
    TokenRef {
        y,
        emu,
        logits,
        emu_logits,
        picks,
        mutated,
    }
}

/// [`token_reference`] of each token `t` of `tokens` (rows of `x`), on every core.
pub fn gemma_references(w: &GemmaMoeWeights, x: &[u16], tokens: &[usize]) -> Vec<TokenRef> {
    let h = w.geometry.h;
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let per = tokens.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        let parts: Vec<_> = (tokens.chunks(per))
            .map(|part| {
                s.spawn(move || {
                    (part.iter())
                        .map(|&t| token_reference(w, &x[t * h..][..h]))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        (parts.into_iter())
            .flat_map(|p| p.join().expect("a reference thread"))
            .collect()
    })
}

/// One case's verdict: the measured routing noise, the GPU's and the emulation's errors against
/// f64 over the tokens that are not router near-ties, the picks that differ from the emulation's,
/// and each mutation's fraction.
pub struct GemmaJudgement {
    pub tokens: usize,
    /// The largest `|emulated − f64|` logit over the referenced tokens.
    pub routing_noise: f64,
    pub near_ties: usize,
    pub gpu: (f64, f64),
    pub emu: (f64, f64),
    pub out_of_bound: usize,
    pub nonfinite: usize,
    pub picks_differ: usize,
    /// Per mutation: the tokens it changes, and those of them out of bound.
    pub mutations: [(usize, usize); 7],
}

impl GemmaJudgement {
    /// The measured-noise rule: CUDA's RMS and max error each at most twice the emulation's, the
    /// picks the emulation's off the near-ties, every output finite, and some token not a near-tie.
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

/// Judge the GPU's result `y` (`[m, h]`) and picks (`[m, 8]`) on the referenced `tokens`. A token
/// is a router near-tie (#859's P9 rule) when its f64 8th and 9th logits are within twice the
/// measured routing noise; its picks and its result are not judged. An element is out of bound
/// past twice the emulation's max error, the bound a mutation is held to too.
pub fn gemma_judge(
    refs: &[TokenRef],
    tokens: &[usize],
    h: usize,
    y: &[u16],
    picks: &[u32],
) -> GemmaJudgement {
    let routing_noise = (refs.iter())
        .flat_map(|r| {
            r.logits
                .iter()
                .zip(&r.emu_logits)
                .map(|(a, b)| (a - b).abs())
        })
        .fold(0.0, f64::max);
    let near = |r: &TokenRef| {
        let b = best9(&r.logits);
        r.logits[b[7]] - r.logits[b[8]] <= 2.0 * routing_noise
    };
    let (mut sq, mut mx, mut esq, mut emx, mut n) = (0.0, 0.0f64, 0.0, 0.0f64, 0usize);
    let (mut nonfinite, mut picks_differ, mut near_ties) = (0, 0, 0);
    for (r, &t) in refs.iter().zip(tokens) {
        if near(r) {
            near_ties += 1;
            continue;
        }
        let mut got: [u32; 8] = picks[t * KU..][..KU].try_into().expect("8 picks");
        got.sort_unstable();
        picks_differ += usize::from(got != r.picks);
        for c in 0..h {
            let g = bf(y[t * h + c]);
            nonfinite += usize::from(!g.is_finite());
            let (e, ee) = ((g - r.y[c]).abs(), (bf(r.emu[c]) - r.y[c]).abs());
            sq += e * e;
            esq += ee * ee;
            mx = mx.max(e);
            emx = emx.max(ee);
            n += 1;
        }
    }
    let bound = 2.0 * emx;
    let mut out_of_bound = 0;
    let mut mutations = [(0usize, 0usize); 7];
    for (r, &t) in refs.iter().zip(tokens) {
        if near(r) {
            continue;
        }
        out_of_bound += (0..h)
            .filter(|&c| (bf(y[t * h + c]) - r.y[c]).abs() > bound)
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
    GemmaJudgement {
        tokens: tokens.len(),
        routing_noise,
        near_ties,
        gpu: (rms(sq), mx),
        emu: (rms(esq), emx),
        out_of_bound,
        nonfinite,
        picks_differ,
        mutations,
    }
}
