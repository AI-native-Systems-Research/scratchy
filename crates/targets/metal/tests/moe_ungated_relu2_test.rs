// Nemotron-H's MoE block on the gathered expert kernels, against a CPU reference: the
// sigmoid + correction-bias routing (`moe_route.metal`, renormalized and scaled by the routed
// scaling factor AFTER the renorm), the UNGATED relu² expert body (`affine_gather_qmv_gated` at
// `GATED_ACT = 3`: `fc1` alone, relu² of its rows) and the weighted combine
// (`affine_gather_qmv_combine` over `fc2`), with `fc1` and `fc2` at their own MLX-affine widths
// — 4/2 (Nemotron-3-Super's layer 8), 2/2, 4/4 — on both matvec flavours (`_fast` and not).
// The fused command must also give the split chain's bits (`affine_gather_qmv`, then `relu2`),
// and routing the token itself (`MetalFusion::MoeRouted`) the routing command's. The gated and
// combine kernels take their pair / token along X (the plain gather along Z).
mod common;

use half::bf16;
use objc2_metal::MTLSize;
use scratchy_subtile::subtile_ir::{ExpertBody, UngatedAct};
use scratchy_target_metal::aot::{BakedPipeline, baked_pipeline};
use scratchy_target_metal::device::detect_device;
use scratchy_target_metal::tape::constants::ConstantValue;
use scratchy_target_metal::tape::ids::{ElementCount, KDimI32, NDimI32, NumExperts, TopK};
use scratchy_target_metal::tape::kernel_constants::{
    AffineCodes, AffineCombineQmvConstants, AffineGatedQmvConstants, AffineGatherQmvConstants,
    AffineQmvConstants, GatherRows, GeluConstants, MoeRouteConstants, RoutedConstants,
};
use scratchy_target_metal::tape::step::{LayerId, RoutePost, RoutePre, RouteProgram, Scale};

const GS: usize = 64;

fn size(w: usize, h: usize, d: usize) -> MTLSize {
    MTLSize {
        width: w,
        height: h,
        depth: d,
    }
}

/// Deterministic values in [0, 1).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }

    fn bf16s(&mut self, n: usize, lo: f32, hi: f32) -> Vec<bf16> {
        (0..n)
            .map(|_| bf16::from_f32(lo + (hi - lo) * self.next()))
            .collect()
    }
}

/// One projection's experts at `bits`: packed codes, bf16 scales and biases, `[experts, n_out,
/// k_in]`, and the same weights dequantized on the host (`q · scale + bias`, MLX's affine law).
struct Experts {
    w: common::Buffer,
    s: common::Buffer,
    b: common::Buffer,
    dequant: Vec<f32>,
    n_out: usize,
    k_in: usize,
    bits: u32,
}

impl Experts {
    fn new(
        d: &common::Device,
        rng: &mut Lcg,
        experts: usize,
        n_out: usize,
        k_in: usize,
        bits: u32,
    ) -> Self {
        let rows = experts * n_out;
        let words_per_row = k_in * bits as usize / 32;
        let words: Vec<u32> = (0..rows * words_per_row)
            .map(|_| (rng.next() * 65536.0) as u32 | (((rng.next() * 65536.0) as u32) << 16))
            .collect();
        let groups = rows * k_in / GS;
        // Ranges so a 2-bit code's step stays a few times the bias spread.
        let scales = rng.bf16s(groups, 0.01, 0.04);
        let biases = rng.bf16s(groups, -0.05, 0.05);
        let mut dequant = vec![0f32; rows * k_in];
        for r in 0..rows {
            for i in 0..k_in {
                let bit = i * bits as usize;
                let word = words[r * words_per_row + bit / 32];
                let code = (word >> (bit % 32)) & ((1 << bits) - 1);
                let g = r * (k_in / GS) + i / GS;
                dequant[r * k_in + i] = code as f32 * scales[g].to_f32() + biases[g].to_f32();
            }
        }
        Self {
            w: common::shared_slice(d, &words),
            s: common::shared_slice(d, &scales),
            b: common::shared_slice(d, &biases),
            dequant,
            n_out,
            k_in,
            bits,
        }
    }

    fn qmv(&self) -> AffineQmvConstants {
        AffineQmvConstants {
            k: KDimI32(self.k_in as i32),
            n: NDimI32(self.n_out as i32),
            codes: AffineCodes::AsWritten,
        }
    }

    /// `kernel`'s symbol for this projection's shape and width (`_fast` as the lowering picks it).
    fn pipeline(&self, d: &common::Device, kernel: &str, c: Vec<ConstantValue>) -> BakedPipeline {
        let fast = scratchy_target_metal::tape::quantized::qmv_fast_covers(
            self.n_out as u32,
            self.k_in as u32,
        );
        let fast = if fast { "_fast" } else { "" };
        let sym = format!("{kernel}{fast}_bf16_s_bf16_gs_64_b_{}", self.bits);
        baked_pipeline(d, "quantized_qmv", &sym, c).expect(kernel)
    }

    fn bindings(&self) -> Vec<(&common::Buffer, usize)> {
        vec![(&self.w, 0), (&self.s, 1), (&self.b, 2)]
    }

    /// Row `o` of expert `e` times `x`, accumulated in f32.
    fn dot(&self, e: usize, o: usize, x: &[f32]) -> f32 {
        let row = &self.dequant[(e * self.n_out + o) * self.k_in..][..self.k_in];
        row.iter().zip(x).map(|(w, x)| w * x).sum()
    }
}

/// Encodes `dispatches` in one command buffer, a barrier after each, and waits.
/// One dispatch: its pipeline, its `(buffer, index)` bindings, threadgroups, threads.
type Dispatch<'a> = (
    &'a BakedPipeline,
    Vec<(&'a common::Buffer, usize)>,
    MTLSize,
    MTLSize,
);

fn run(d: &common::Device, dispatches: &[Dispatch<'_>]) {
    let mut batch = common::Mtl4DispatchBatch::begin(d).expect("an MTL4 queue");
    for (pso, buffers, groups, threads) in dispatches {
        batch.encode(pso, buffers, &[], &[], &[], *groups, *threads);
        batch.barrier();
    }
    batch.commit(true);
}

fn bits16(buf: &common::Buffer, n: usize) -> Vec<u16> {
    common::read_slice::<u16>(buf, n)
}

fn f32s(buf: &common::Buffer, n: usize) -> Vec<f32> {
    bits16(buf, n)
        .into_iter()
        .map(|b| bf16::from_bits(b).to_f32())
        .collect()
}

/// Nemotron-H's routing program: F32 sigmoid + correction bias picks, the unbiased sigmoids
/// renormalized and scaled by the routed scaling factor last.
fn nemotron_route(scale: f32) -> RouteProgram {
    RouteProgram {
        pre: RoutePre::SigmoidBias(LayerId(0)),
        scale: None,
        post: RoutePost::Renorm,
        expert_scale: None,
        post_scale: Some(Scale(scale)),
    }
}

/// One MoE block: `tokens` rows, `experts` experts of which each token takes `k`, `fc1` [E,
/// inter, hidden] and `fc2` [E, hidden, inter] at their own widths.
struct Block {
    tokens: usize,
    experts: usize,
    k: usize,
    hidden: usize,
    inter: usize,
    fc1: Experts,
    fc2: Experts,
    x: common::Buffer,
    logits: common::Buffer,
    bias: common::Buffer,
    program: RouteProgram,
}

/// A block's geometry: `tokens` rows, `experts` experts of which each token takes `k`, and
/// `fc1` / `fc2` at their own widths.
#[derive(Clone, Copy)]
struct Shape {
    bits: (u32, u32),
    hidden: usize,
    inter: usize,
    experts: usize,
    k: usize,
    tokens: usize,
}

impl Shape {
    fn what(&self) -> String {
        let Shape {
            bits: (b1, b2),
            hidden,
            experts,
            k,
            tokens,
            ..
        } = *self;
        format!("fc1 {b1}-bit / fc2 {b2}-bit, hidden {hidden}, {k} of {experts}, {tokens} tokens")
    }
}

impl Block {
    fn new(d: &common::Device, shape: Shape, seed: u64) -> Self {
        let Shape {
            bits: (b1, b2),
            hidden,
            inter,
            experts,
            k,
            tokens,
        } = shape;
        let mut rng = Lcg(seed);
        let fc1 = Experts::new(d, &mut rng, experts, inter, hidden, b1);
        let fc2 = Experts::new(d, &mut rng, experts, hidden, inter, b2);
        let bias: Vec<f32> = (0..experts).map(|_| 0.2 * (rng.next() - 0.5)).collect();
        Self {
            tokens,
            experts,
            k,
            hidden,
            inter,
            x: common::shared_slice(d, &rng.bf16s(tokens * hidden, -2.0, 2.0)),
            logits: common::shared_slice(d, &rng.bf16s(tokens * experts, -3.0, 3.0)),
            bias: common::shared_slice(d, &bias),
            fc1,
            fc2,
            program: nemotron_route(2.5),
        }
    }

    fn route_pipeline(&self, d: &common::Device) -> BakedPipeline {
        let c = MoeRouteConstants {
            experts: NumExperts(self.experts as u32),
            top_k: TopK(self.k as u32),
            program: self.program,
        };
        let bn = if self.experts > 128 { 64 } else { 32 };
        baked_pipeline(
            d,
            "moe_route",
            &format!("moe_route_bfloat16_bn{bn}"),
            c.into(),
        )
        .expect("route")
    }

    fn rows(&self) -> GatherRows {
        GatherRows::Tokens(TopK(self.k as u32))
    }

    /// The fused ungated command's constants; `routed`: it routes the token itself.
    fn up_act_constants(&self, routed: bool) -> Vec<ConstantValue> {
        let qmv = AffineGatherQmvConstants {
            qmv: self.fc1.qmv(),
            rows: self.rows(),
        };
        let mut c: Vec<ConstantValue> = AffineGatedQmvConstants {
            qmv,
            act: ExpertBody::Ungated(UngatedAct::Relu2),
        }
        .into();
        if routed {
            c.extend(Vec::<ConstantValue>::from(RoutedConstants {
                experts: NumExperts(self.experts as u32),
                program: self.program,
            }));
        }
        c
    }

    fn combine_pipeline(&self, d: &common::Device) -> BakedPipeline {
        let c = AffineCombineQmvConstants {
            qmv: self.fc2.qmv(),
            top_k: TopK(self.k as u32),
            gate_scale: false,
            residual: false,
        };
        self.fc2.pipeline(d, "affine_gather_qmv_combine", c.into())
    }

    fn pairs(&self) -> usize {
        self.tokens * self.k
    }

    /// The CPU reference over the picks and scores the routing stored: per pair, fc1 · x rounded
    /// to bf16 as the matvec stores it, relu² rounded once, fc2 · that rounded to bf16, then
    /// each token's pair rows summed by their scores in f32 and rounded.
    fn reference(&self, inds: &[u32], scores: &[f32]) -> Vec<f32> {
        let x = f32s(&self.x, self.tokens * self.hidden);
        let round = |v: f32| bf16::from_f32(v).to_f32();
        let mut out = vec![0f32; self.tokens * self.hidden];
        for t in 0..self.tokens {
            let xt = &x[t * self.hidden..][..self.hidden];
            let mut acc = vec![0f32; self.hidden];
            for j in 0..self.k {
                let e = inds[t * self.k + j] as usize;
                let h: Vec<f32> = (0..self.inter)
                    .map(|o| {
                        let r = round(self.fc1.dot(e, o, xt)).max(0.0);
                        round(r * r)
                    })
                    .collect();
                for (o, a) in acc.iter_mut().enumerate() {
                    *a += round(self.fc2.dot(e, o, &h)) * scores[t * self.k + j];
                }
            }
            for (o, a) in acc.into_iter().enumerate() {
                out[t * self.hidden + o] = round(a);
            }
        }
        out
    }

    /// The CPU routing: the picks order on each expert's F32 sigmoid plus its correction bias
    /// (ties by index ascending), the scores are the picks' unbiased sigmoids renormalized and
    /// times the routed scaling factor — mlx-lm's `group_expert_select` at `n_group = 1`.
    fn reference_route(&self) -> (Vec<u32>, Vec<f32>) {
        let logits = f32s(&self.logits, self.tokens * self.experts);
        let bias: Vec<f32> = common::read_slice(&self.bias, self.experts);
        let Some(Scale(scale)) = self.program.post_scale else {
            unreachable!("nemotron's program scales last")
        };
        let sigmoid = |x: f32| 1.0f32 / (1.0 + (-x).exp());
        let (mut inds, mut scores) = (Vec::new(), Vec::new());
        for t in 0..self.tokens {
            let row = &logits[t * self.experts..][..self.experts];
            let mut order: Vec<usize> = (0..self.experts).collect();
            order.sort_by(|&a, &b| {
                (sigmoid(row[b]) + bias[b])
                    .total_cmp(&(sigmoid(row[a]) + bias[a]))
                    .then(a.cmp(&b))
            });
            let picks: Vec<usize> = order[..self.k].iter().rev().copied().collect();
            let s: Vec<f32> = picks.iter().map(|&e| sigmoid(row[e])).collect();
            let sum: f32 = s.iter().sum();
            inds.extend(picks.iter().map(|&e| e as u32));
            scores.extend(s.iter().map(|v| v / sum * scale));
        }
        (inds, scores)
    }
}

/// Max |a - b| over max |b|, and the cosine.
fn compare(got: &[f32], want: &[f32]) -> (f64, f64) {
    let peak = want.iter().fold(0f64, |m, v| m.max(f64::from(v.abs())));
    let diff = got
        .iter()
        .zip(want)
        .fold(0f64, |m, (a, b)| m.max(f64::from((a - b).abs())));
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (a, b) in got.iter().zip(want) {
        let (a, b) = (f64::from(*a), f64::from(*b));
        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    (diff / peak, dot / (na.sqrt() * nb.sqrt()))
}

/// Nemotron-3-Super's 4/2 split bundle and its 2/2 and 4/4 ones (and the 2/4 converse), at
/// widths the `_fast` matvec covers and not.
const CASES: &[Shape] = &[
    Shape {
        bits: (4, 2),
        hidden: 256,
        inter: 128,
        experts: 64,
        k: 6,
        tokens: 3,
    },
    Shape {
        bits: (2, 2),
        hidden: 128,
        inter: 64,
        experts: 16,
        k: 2,
        tokens: 1,
    },
    Shape {
        bits: (4, 4),
        hidden: 128,
        inter: 64,
        experts: 8,
        k: 2,
        tokens: 4,
    },
    Shape {
        bits: (2, 4),
        hidden: 192,
        inter: 128,
        experts: 32,
        k: 4,
        tokens: 2,
    },
];

#[test]
fn ungated_relu2_moe_matches_the_cpu_reference() {
    let Some(dev) = detect_device() else { return };
    let d = dev.device;
    for (n, shape) in CASES.iter().enumerate() {
        let Shape {
            hidden,
            inter,
            experts,
            k,
            tokens,
            ..
        } = *shape;
        let what = shape.what();
        let blk = Block::new(&d, *shape, 0x5eed + n as u64);
        let route = blk.route_pipeline(&d);
        let up_act = blk
            .fc1
            .pipeline(&d, "affine_gather_qmv_gated", blk.up_act_constants(false));
        let combine = blk.combine_pipeline(&d);
        let pairs = blk.pairs();
        let (inds, scores) = (
            common::shared_zeroed(&d, pairs * 4),
            common::shared_zeroed(&d, pairs * 2),
        );
        let (up_y, down_y, out) = (
            common::shared_zeroed(&d, pairs * inter * 2),
            common::shared_zeroed(&d, pairs * hidden * 2),
            common::shared_zeroed(&d, tokens * hidden * 2),
        );
        let bn = if experts > 128 { 64 } else { 32 };
        let mut up = blk.fc1.bindings();
        up.extend([(&blk.x, 3), (&inds, 4), (&up_y, 5)]);
        let mut comb = blk.fc2.bindings();
        comb.extend([
            (&up_y, 3),
            (&inds, 4),
            (&down_y, 5),
            (&scores, 6),
            (&out, 7),
        ]);
        run(
            &d,
            &[
                (
                    &route,
                    vec![(&blk.logits, 0), (&inds, 1), (&scores, 2), (&blk.bias, 4)],
                    size(1, tokens, 1),
                    size(bn, 1, 1),
                ),
                (
                    &up_act,
                    up,
                    size(pairs, inter.div_ceil(8), 1),
                    size(32, 4, 1),
                ),
                (
                    &combine,
                    comb,
                    size(tokens, hidden.div_ceil(4), 1),
                    size(32, k, 1),
                ),
            ],
        );
        // The routing, against the CPU's picks and its scores rounded as the kernel stores them.
        let (want_inds, want_scores) = blk.reference_route();
        let got_inds = common::read_slice::<u32>(&inds, pairs);
        assert_eq!(got_inds, want_inds, "{what}: picks");
        let got_scores = f32s(&scores, pairs);
        let (score_diff, _) = compare(&got_scores, &want_scores);
        assert!(
            score_diff < 1e-2,
            "{what}: scores off by {score_diff} of the peak"
        );
        // The block, against the CPU over the stored picks and scores.
        let want = blk.reference(&got_inds, &got_scores);
        let got = f32s(&out, tokens * hidden);
        assert!(got.iter().any(|&v| v != 0.0), "{what}: no output");
        let (diff, cos) = compare(&got, &want);
        assert!(
            cos > 0.9999 && diff < 2e-2,
            "{what}: cosine {cos}, max diff {diff} of the peak"
        );
    }
}

/// The fused ungated command gives the split chain's bits — `affine_gather_qmv` over `fc1`,
/// then `relu2` over its rows — and, routing the token itself, the routing command's picks and
/// scores (the last scale included) and the same rows.
#[test]
fn fused_ungated_command_matches_the_split_chain_and_the_routing_command() {
    let Some(dev) = detect_device() else { return };
    let d = dev.device;
    for (n, shape) in CASES.iter().enumerate() {
        let Shape {
            inter,
            experts,
            tokens,
            ..
        } = *shape;
        let what = shape.what();
        let blk = Block::new(&d, *shape, 0xbeef + n as u64);
        let pairs = blk.pairs();
        let route = blk.route_pipeline(&d);
        let bn = if experts > 128 { 64 } else { 32 };
        let zero = |n: usize| common::shared_zeroed(&d, n);
        let (inds, scores) = (zero(pairs * 4), zero(pairs * 2));
        run(
            &d,
            &[(
                &route,
                vec![(&blk.logits, 0), (&inds, 1), (&scores, 2), (&blk.bias, 4)],
                size(1, tokens, 1),
                size(bn, 1, 1),
            )],
        );
        // Split: the plain gather-qmv, then relu2 in place.
        let plain = blk.fc1.pipeline(
            &d,
            "affine_gather_qmv",
            AffineGatherQmvConstants {
                qmv: blk.fc1.qmv(),
                rows: blk.rows(),
            }
            .into(),
        );
        let n_act = pairs * inter;
        let relu2 = baked_pipeline(
            &d,
            "activation",
            "relu2_bf16",
            GeluConstants {
                elements: ElementCount(n_act as u32),
            }
            .into(),
        )
        .expect("relu2");
        let split_y = zero(n_act * 2);
        let mut b = blk.fc1.bindings();
        b.extend([(&blk.x, 3), (&inds, 4), (&split_y, 5)]);
        run(
            &d,
            &[
                (&plain, b, size(1, inter.div_ceil(8), pairs), size(32, 2, 1)),
                (
                    &relu2,
                    vec![(&split_y, 0), (&split_y, 1)],
                    size(n_act.div_ceil(256), 1, 1),
                    size(256, 1, 1),
                ),
            ],
        );
        // Fused, over the routing command's picks.
        let fused = blk
            .fc1
            .pipeline(&d, "affine_gather_qmv_gated", blk.up_act_constants(false));
        let fused_y = zero(n_act * 2);
        let mut b = blk.fc1.bindings();
        b.extend([(&blk.x, 3), (&inds, 4), (&fused_y, 5)]);
        run(
            &d,
            &[(&fused, b, size(pairs, inter.div_ceil(8), 1), size(32, 4, 1))],
        );
        assert_eq!(
            bits16(&fused_y, n_act),
            bits16(&split_y, n_act),
            "{what}: fused rows"
        );
        // Routed: the fused command picks and scores the token itself (one token per run, as
        // the one-row fold dispatches it).
        if tokens == 1 {
            let routed =
                blk.fc1
                    .pipeline(&d, "affine_gather_qmv_gated", blk.up_act_constants(true));
            let (r_inds, r_scores, r_y) = (zero(pairs * 4), zero(pairs * 2), zero(n_act * 2));
            let mut b = blk.fc1.bindings();
            b.extend([(&blk.x, 3), (&r_inds, 4), (&r_y, 5)]);
            b.extend([(&blk.logits, 10), (&r_scores, 11), (&blk.bias, 13)]);
            run(
                &d,
                &[(
                    &routed,
                    b,
                    size(pairs, inter.div_ceil(8), 1),
                    size(32, 4, 1),
                )],
            );
            assert_eq!(
                common::read_slice::<u32>(&r_inds, pairs),
                common::read_slice::<u32>(&inds, pairs),
                "{what}: routed picks"
            );
            assert_eq!(
                bits16(&r_scores, pairs),
                bits16(&scores, pairs),
                "{what}: routed scores"
            );
            assert_eq!(
                bits16(&r_y, n_act),
                bits16(&split_y, n_act),
                "{what}: routed rows"
            );
        }
    }
}
