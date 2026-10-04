// The MoE routing kernel (`moe_route.metal`) against the kernels it replaces, for each router's
// program: it must give the same top-k indices and scores, bit for bit. `bench_moe_route` times
// Gemma-4-26B-A4B's routing both ways (`BENCH` lines).
mod common;

use std::time::Instant;

use half::bf16;
use objc2_metal::MTLSize;
use scratchy_target_metal::aot::{BakedPipeline, baked_pipeline};
use scratchy_target_metal::device::detect_device;
use scratchy_target_metal::tape::constants::{ConstSlot, ConstantValue};
use scratchy_target_metal::tape::ids::{ElementCount, NumExperts, TopK};
use scratchy_target_metal::tape::kernel_constants::{
    ArgsortConstants, MoeRouteConstants, MoeTopKConstants, ScalarMulConstants, ScoresRow,
    SoftmaxConstants,
};
use scratchy_target_metal::tape::step::{LayerId, RoutePost, RouteProgram, Scale};

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

/// One dispatch, encoded with a barrier after it.
struct Dispatch<'a> {
    pso: &'a BakedPipeline,
    buffers: Vec<(&'a common::Buffer, usize)>,
    groups: MTLSize,
    threads: MTLSize,
}

/// Runs `chain` `n` times in one command buffer, a barrier after every dispatch; the best wall
/// time per chain over `rounds`, in µs.
fn run<'a>(
    device: &common::Device,
    n: usize,
    rounds: usize,
    chain: impl Fn() -> Vec<Dispatch<'a>>,
) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..rounds {
        let mut batch = common::Mtl4DispatchBatch::begin(device).expect("an MTL4 queue");
        for _ in 0..n {
            for d in chain() {
                batch.encode(d.pso, &d.buffers, &[], &[], &[], d.groups, d.threads);
                batch.barrier();
            }
        }
        let t = Instant::now();
        batch.commit(true);
        best = best.min(t.elapsed().as_secs_f64());
    }
    best / n as f64 * 1e6
}

/// A router's routing over `tokens` rows of `experts` logits, both ways: the kernels in
/// sequence, and the routing kernel.
struct Routing {
    program: RouteProgram,
    experts: usize,
    top_k: usize,
    tokens: usize,
    // Today's chain, in order.
    chain: Vec<(BakedPipeline, Stage)>,
    route: BakedPipeline,
    expert_scale: common::Buffer,
    // Each way's buffers: logits, sorted experts, top-k indices, top-k scores.
    split: [common::Buffer; 4],
    fused: [common::Buffer; 4],
}

/// What one of today's kernels runs over.
#[derive(Clone, Copy)]
enum Stage {
    SoftmaxLogits,
    Argsort,
    TopK,
    Gather,
    Scale,
    SoftmaxScores,
    Renorm,
    ExpertScale,
}

impl Routing {
    /// `ties`: logits from a handful of values, -0 and +0 among them, so equal scores straddle
    /// every top-k boundary.
    fn new(
        device: &common::Device,
        program: RouteProgram,
        (e, k, n): (usize, usize, usize),
        ties: bool,
    ) -> Self {
        let mut rng = Lcg(0x5eed ^ (e * 31 + k) as u64);
        let logits = match ties {
            false => rng.bf16s(n * e, -4.0, 4.0),
            true => {
                let values = [-1.0f32, -0.0, 0.0, 0.5, 1.0, 2.0];
                let pick = |_| bf16::from_f32(values[(rng.next() * 6.0) as usize % 6]);
                (0..n * e).map(pick).collect()
            }
        };
        let pairs = n * k;
        let (experts, top_k) = (NumExperts(e as u32), TopK(k as u32));
        let bn = if e > 128 { 64 } else { 32 };
        let pipe = |lib: &'static str, sym: &str, c: Vec<ConstantValue>| {
            baked_pipeline(device, lib, sym, c).expect(lib)
        };
        let top_k_consts = || MoeTopKConstants { experts, top_k }.into();
        let softmax = |row| SoftmaxConstants { row }.into();
        let mut chain = Vec::new();
        if program.pre_softmax {
            let c = softmax(ScoresRow::Experts(experts));
            let p = pipe("softmax", "block_softmax_precise_bfloat16", c);
            chain.push((p, Stage::SoftmaxLogits));
        }
        let sort = format!("c_arg_block_sort_bfloat16_uint32_bn{bn}_tn4");
        let c = ArgsortConstants { experts }.into();
        chain.push((pipe("argpartition", &sort, c), Stage::Argsort));
        let p = pipe(
            "slice_trailing_cols",
            "slice_trailing_cols_u32",
            top_k_consts(),
        );
        chain.push((p, Stage::TopK));
        let p = pipe(
            "take_along_axis",
            "take_along_axis_2d_contig_bfloat16",
            top_k_consts(),
        );
        chain.push((p, Stage::Gather));
        if let Some(Scale(scale)) = program.scale {
            let elements = ElementCount(pairs as u32);
            let c = ScalarMulConstants { scale, elements }.into();
            let p = pipe("elementwise", "scalar_mul_bf16_specialized", c);
            chain.push((p, Stage::Scale));
        }
        match program.post {
            RoutePost::None => {}
            RoutePost::Softmax => {
                let c = softmax(ScoresRow::TopK(top_k));
                let p = pipe("softmax", "block_softmax_precise_bfloat16", c);
                chain.push((p, Stage::SoftmaxScores));
            }
            RoutePost::Renorm => {
                let c = softmax(ScoresRow::TopK(top_k));
                chain.push((pipe("softmax", "topk_renorm_bfloat16", c), Stage::Renorm));
            }
        }
        if program.expert_scale.is_some() {
            let c = vec![ConstantValue::uint(ConstSlot(0), pairs as u32)];
            let p = pipe("moe_per_expert_scale", "moe_per_expert_scale_bfloat16", c);
            chain.push((p, Stage::ExpertScale));
        }
        let route = MoeRouteConstants {
            experts,
            top_k,
            program,
        };
        let sym = format!("moe_route_bfloat16_bn{bn}");
        let buffers = || {
            [
                common::shared_slice(device, &logits),
                common::shared_zeroed(device, n * e * 4),
                common::shared_zeroed(device, pairs * 4),
                common::shared_zeroed(device, pairs * 2),
            ]
        };
        Self {
            program,
            experts: e,
            top_k: k,
            tokens: n,
            chain,
            route: pipe("moe_route", &sym, route.into()),
            expert_scale: common::shared_slice(device, &rng.bf16s(e, 0.5, 1.5)),
            split: buffers(),
            fused: buffers(),
        }
    }

    fn chain(&self) -> Vec<Dispatch<'_>> {
        let [lg, sorted, inds, scores] = &self.split;
        let (n, k, pairs) = (self.tokens, self.top_k, self.tokens * self.top_k);
        let bn = if self.experts > 128 { 64 } else { 32 };
        let per_k = (size(k.div_ceil(k.min(32)), n, 1), size(k.min(32), 1, 1));
        let per_row = (size(n, 1, 1), size(256, 1, 1));
        let per_pair = (size(pairs.div_ceil(256), 1, 1), size(256, 1, 1));
        self.chain
            .iter()
            .map(|(pso, stage)| {
                let (buffers, (groups, threads)) = match stage {
                    Stage::SoftmaxLogits => (vec![(lg, 0), (lg, 1)], per_row),
                    Stage::Argsort => (vec![(lg, 0), (sorted, 1)], (size(1, n, 1), size(bn, 1, 1))),
                    Stage::TopK => (vec![(sorted, 0), (inds, 1)], per_k),
                    Stage::Gather => (vec![(lg, 0), (inds, 1), (scores, 2)], per_k),
                    Stage::Scale => (vec![(scores, 0), (scores, 1)], per_pair),
                    Stage::SoftmaxScores | Stage::Renorm => {
                        (vec![(scores, 0), (scores, 1)], per_row)
                    }
                    Stage::ExpertScale => (
                        vec![(scores, 0), (inds, 1), (&self.expert_scale, 2)],
                        per_pair,
                    ),
                };
                Dispatch {
                    pso,
                    buffers,
                    groups,
                    threads,
                }
            })
            .collect()
    }

    fn fused(&self) -> Vec<Dispatch<'_>> {
        let [lg, _, inds, scores] = &self.fused;
        let bn = if self.experts > 128 { 64 } else { 32 };
        let mut buffers = vec![(lg, 0), (inds, 1), (scores, 2)];
        if self.program.expert_scale.is_some() {
            buffers.push((&self.expert_scale, 3));
        }
        vec![Dispatch {
            pso: &self.route,
            buffers,
            groups: size(1, self.tokens, 1),
            threads: size(bn, 1, 1),
        }]
    }

    /// Each way's top-k indices and scores, as raw bits.
    fn outputs(&self, way: &[common::Buffer; 4]) -> (Vec<u32>, Vec<u16>) {
        let pairs = self.tokens * self.top_k;
        (
            common::read_slice::<u32>(&way[2], pairs),
            common::read_slice::<u16>(&way[3], pairs),
        )
    }
}

const GEMMA: RouteProgram = RouteProgram {
    pre_softmax: false,
    scale: Some(Scale(0.018_844_6)),
    post: RoutePost::Softmax,
    expert_scale: Some(LayerId(0)),
};
const TOP_K_SOFTMAX: RouteProgram = RouteProgram {
    pre_softmax: false,
    scale: None,
    post: RoutePost::Softmax,
    expert_scale: None,
};
const SHARED: RouteProgram = RouteProgram {
    pre_softmax: true,
    scale: None,
    post: RoutePost::Renorm,
    expert_scale: None,
};
const SHARED_UNNORMED: RouteProgram = RouteProgram {
    pre_softmax: true,
    scale: None,
    post: RoutePost::None,
    expert_scale: None,
};

#[test]
fn the_routing_kernel_matches_the_kernels_it_replaces() {
    let Some(d) = detect_device() else { return };
    let device = d.device;
    // (program, experts, top-k): Gemma-4, Qwen3-MoE, Mixtral, Qwen3.5-MoE, Qwen1.5-MoE.
    let cases = [
        ("gemma", GEMMA, 128, 8),
        ("qwen3-moe", TOP_K_SOFTMAX, 128, 8),
        ("mixtral", TOP_K_SOFTMAX, 8, 2),
        ("qwen3.5-moe", SHARED, 256, 8),
        ("qwen1.5-moe", SHARED_UNNORMED, 60, 4),
    ];
    for (name, program, e, k) in cases {
        for (tokens, ties) in [(1, false), (3, false), (3, true)] {
            let r = Routing::new(&device, program, (e, k, tokens), ties);
            run(&device, 1, 1, || r.chain());
            run(&device, 1, 1, || r.fused());
            let (split, fused) = (r.outputs(&r.split), r.outputs(&r.fused));
            assert!(split.1.iter().any(|&v| v != 0), "{name}: no scores");
            let what = format!("{name} tokens={tokens} ties={ties}");
            assert_eq!(split.0, fused.0, "{what}: top-k indices");
            assert_eq!(split.1, fused.1, "{what}: top-k scores");
        }
    }
}

#[test]
#[ignore]
fn bench_moe_route() {
    let Some(d) = detect_device() else { return };
    let device = d.device;
    let r = Routing::new(&device, GEMMA, (128, 8, 1), false);
    let launches = r.chain.len();
    run(&device, 500, 2, || r.chain());
    for round in 0..3 {
        let chain = run(&device, 1000, 4, || r.chain());
        let fused = run(&device, 1000, 4, || r.fused());
        println!(
            "BENCH moe routing round {round} (Gemma-4, 128 experts, top 8): today {launches} \
             launches {chain:.2} us, routing kernel {fused:.2} us ({:+.1}%)",
            (fused / chain - 1.0) * 100.0
        );
    }
}
