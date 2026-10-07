// The MoE block's fused expert kernels against the kernels they replace, on Gemma-4-26B-A4B's
// decode shapes (128 experts, top 8, hidden 2816, expert width 704, 4-bit g64, bf16):
//   affine_gather_qmv_gated   = gate gather-qmv, up gather-qmv, gelu_mul / silu_mul
//   affine_gather_qmv_combine = down gather-qmv, moe_weighted_sum
// and each routing its token itself (`MetalFusion::MoeRouted`) = the routing command, then it.
// Each must give the same bits. `bench_moe_fused` times both chains (`BENCH` lines).
mod common;

use std::time::Instant;

use half::bf16;
use objc2_metal::MTLSize;
use scratchy_target_metal::aot::{BakedPipeline, baked_pipeline};
use scratchy_target_metal::device::detect_device;
use scratchy_target_metal::tape::constants::{ConstSlot, ConstantValue};
use scratchy_target_metal::tape::ids::{KDimI32, NDimI32, NumExperts, TopK};
use scratchy_target_metal::tape::kernel_constants::{
    AffineCodes, AffineGatherQmvConstants, AffineQmvConstants, GatherRows, MoeRouteConstants,
    RoutedConstants,
};
use scratchy_target_metal::tape::step::{LayerId, RoutePost, RouteProgram, Scale};

const EXPERTS: usize = 128;
const TOP_K: usize = 8;
const HIDDEN: usize = 2816;
const INTER: usize = 704;
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

/// One projection's experts: 4-bit codes, bf16 scales and biases, `[experts, n_out, k_in]`.
struct Experts {
    w: common::Buffer,
    s: common::Buffer,
    b: common::Buffer,
    n_out: usize,
    k_in: usize,
}

impl Experts {
    fn new(
        device: &common::Device,
        rng: &mut Lcg,
        experts: usize,
        n_out: usize,
        k_in: usize,
    ) -> Self {
        let codes: Vec<u8> = (0..experts * n_out * k_in / 2)
            .map(|_| (rng.next() * 256.0) as u8)
            .collect();
        let groups = experts * n_out * k_in / GS;
        Self {
            w: common::shared_slice(device, &codes),
            s: common::shared_slice(device, &rng.bf16s(groups, 0.002, 0.02)),
            b: common::shared_slice(device, &rng.bf16s(groups, -0.08, 0.08)),
            n_out,
            k_in,
        }
    }

    fn constants(&self, rows: GatherRows) -> Vec<ConstantValue> {
        AffineGatherQmvConstants {
            qmv: AffineQmvConstants {
                k: KDimI32(self.k_in as i32),
                n: NDimI32(self.n_out as i32),
                codes: AffineCodes::AsWritten,
            },
            rows,
        }
        .into()
    }

    /// `kernel`'s symbol for this projection's shape (`_fast` as `affine_gather_qmv_symbol`).
    fn symbol(&self, kernel: &str) -> String {
        let fast = self.n_out.is_multiple_of(8) && self.k_in.is_multiple_of(512);
        let fast = if fast { "_fast" } else { "" };
        format!("{kernel}{fast}_bf16_s_bf16_gs_64_b_4")
    }

    fn pipeline(&self, d: &common::Device, kernel: &str, c: Vec<ConstantValue>) -> BakedPipeline {
        baked_pipeline(d, "quantized_qmv", &self.symbol(kernel), c).expect(kernel)
    }

    fn bindings(&self) -> [(&common::Buffer, usize); 3] {
        [(&self.w, 0), (&self.s, 1), (&self.b, 2)]
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
    chain: impl Fn(usize) -> Vec<Dispatch<'a>>,
) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..rounds {
        let mut batch = common::Mtl4DispatchBatch::begin(device).expect("an MTL4 queue");
        for i in 0..n {
            for d in chain(i) {
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

/// Gemma's MoE block on `tokens` rows: the expert weights, the activations, and both chains'
/// pipelines and outputs.
struct Block {
    gate: Experts,
    up: Experts,
    down: Experts,
    tokens: usize,
    x: common::Buffer,
    /// Disjoint expert sets (each `[tokens, TOP_K]`), cycled when timing.
    indices: Vec<common::Buffer>,
    scores: common::Buffer,
    // Unfused: gate, up, act, down, weighted sum.
    gate_qmv: BakedPipeline,
    act: BakedPipeline,
    down_qmv: BakedPipeline,
    weighted_sum: BakedPipeline,
    gate_y: common::Buffer,
    up_y: common::Buffer,
    down_y: common::Buffer,
    out: common::Buffer,
    // Fused: gated, combine.
    gated: BakedPipeline,
    combine: BakedPipeline,
    fused_gate_y: common::Buffer,
    fused_up_y: common::Buffer,
    fused_down_y: common::Buffer,
    fused_out: common::Buffer,
}

impl Block {
    /// `gelu` picks GELU (tanh) over SiLU as the gated activation; `experts` is the bank size.
    fn new(device: &common::Device, tokens: usize, gelu: bool, experts: usize) -> Self {
        let mut rng = Lcg(0x5eed);
        let gate = Experts::new(device, &mut rng, experts, INTER, HIDDEN);
        let up = Experts::new(device, &mut rng, experts, INTER, HIDDEN);
        let down = Experts::new(device, &mut rng, experts, HIDDEN, INTER);
        let pairs = tokens * TOP_K;
        // Set s, token t, slot j: expert (s + 16 * (j + t)) % experts — distinct within a token.
        let indices = (0..16u32)
            .map(|s| {
                let set: Vec<u32> = (0..pairs as u32)
                    .map(|p| (s + 16 * (p % TOP_K as u32 + p / TOP_K as u32)) % experts as u32)
                    .collect();
                common::shared_slice(device, &set)
            })
            .collect();
        let rows = GatherRows::Tokens(TopK(TOP_K as u32));
        let act_code = ConstantValue::int(ConstSlot(3), i32::from(gelu));
        let mut gated_constants = gate.constants(rows);
        gated_constants.push(act_code);
        let act_symbol = if gelu {
            "gelu_mul_bf16"
        } else {
            "silu_mul_bf16"
        };
        let act_n = vec![ConstantValue::uint(ConstSlot(0), (pairs * INTER) as u32)];
        let wsum_constants = vec![
            ConstantValue::int(ConstSlot(0), TOP_K as i32),
            ConstantValue::int(ConstSlot(1), HIDDEN as i32),
        ];
        let zeroed = |n: usize| common::shared_zeroed(device, n * 2);
        Self {
            gate_qmv: gate.pipeline(device, "affine_gather_qmv", gate.constants(rows)),
            act: baked_pipeline(device, "silu_mul", act_symbol, act_n).expect("act"),
            down_qmv: down.pipeline(
                device,
                "affine_gather_qmv",
                down.constants(GatherRows::Pairs),
            ),
            weighted_sum: baked_pipeline(
                device,
                "moe_weighted_sum",
                "moe_weighted_sum_bfloat16",
                wsum_constants,
            )
            .expect("weighted sum"),
            gated: gate.pipeline(device, "affine_gather_qmv_gated", gated_constants),
            combine: down.pipeline(device, "affine_gather_qmv_combine", down.constants(rows)),
            x: common::shared_slice(device, &rng.bf16s(tokens * HIDDEN, -1.0, 1.0)),
            scores: common::shared_slice(device, &rng.bf16s(pairs, 0.0, 0.3)),
            indices,
            gate_y: zeroed(pairs * INTER),
            up_y: zeroed(pairs * INTER),
            down_y: zeroed(pairs * HIDDEN),
            out: zeroed(tokens * HIDDEN),
            fused_gate_y: zeroed(pairs * INTER),
            fused_up_y: zeroed(pairs * INTER),
            fused_down_y: zeroed(pairs * HIDDEN),
            fused_out: zeroed(tokens * HIDDEN),
            gate,
            up,
            down,
            tokens,
        }
    }

    fn gather<'a>(
        &'a self,
        pso: &'a BakedPipeline,
        e: &'a Experts,
        x: &'a common::Buffer,
        set: usize,
        y: &'a common::Buffer,
    ) -> Dispatch<'a> {
        let mut buffers = e.bindings().to_vec();
        buffers.extend([(x, 3), (&self.indices[set], 4), (y, 5)]);
        Dispatch {
            pso,
            buffers,
            groups: size(1, e.n_out.div_ceil(8), self.tokens * TOP_K),
            threads: size(32, 2, 1),
        }
    }

    /// gate, up, act (over the gate rows), down, weighted sum.
    fn unfused(&self, set: usize) -> Vec<Dispatch<'_>> {
        let n = self.tokens * TOP_K * INTER;
        vec![
            self.gather(&self.gate_qmv, &self.gate, &self.x, set, &self.gate_y),
            self.gather(&self.gate_qmv, &self.up, &self.x, set, &self.up_y),
            Dispatch {
                pso: &self.act,
                buffers: vec![(&self.gate_y, 0), (&self.gate_y, 1), (&self.up_y, 2)],
                groups: size(n.div_ceil(256), 1, 1),
                threads: size(256, 1, 1),
            },
            self.gather(&self.down_qmv, &self.down, &self.gate_y, set, &self.down_y),
            Dispatch {
                pso: &self.weighted_sum,
                buffers: vec![(&self.down_y, 0), (&self.scores, 1), (&self.out, 2)],
                groups: size(HIDDEN.div_ceil(64), self.tokens, 1),
                threads: size(64, 1, 1),
            },
        ]
    }

    /// gated (gate, up, act), combine (down, weighted sum).
    fn fused(&self, set: usize) -> Vec<Dispatch<'_>> {
        let mut gated = self.gate.bindings().to_vec();
        gated.extend([
            (&self.x, 3),
            (&self.indices[set], 4),
            (&self.fused_gate_y, 5),
        ]);
        gated.extend([(&self.up.w, 6), (&self.up.s, 7), (&self.up.b, 8)]);
        gated.push((&self.fused_up_y, 9));
        let mut combine = self.down.bindings().to_vec();
        combine.extend([(&self.fused_gate_y, 3), (&self.indices[set], 4)]);
        combine.extend([
            (&self.fused_down_y, 5),
            (&self.scores, 6),
            (&self.fused_out, 7),
        ]);
        vec![
            Dispatch {
                pso: &self.gated,
                buffers: gated,
                groups: size(1, INTER.div_ceil(8), self.tokens * TOP_K),
                threads: size(32, 4, 1),
            },
            Dispatch {
                pso: &self.combine,
                buffers: combine,
                groups: size(1, HIDDEN.div_ceil(4), self.tokens),
                threads: size(32, TOP_K, 1),
            },
        ]
    }
}

fn bits(buf: &common::Buffer, n: usize) -> Vec<u16> {
    common::read_slice::<u16>(buf, n)
}

#[test]
fn fused_moe_kernels_match_the_unfused_chain() {
    let Some(d) = detect_device() else { return };
    let device = d.device;
    for gelu in [true, false] {
        for tokens in [1, 3] {
            let block = Block::new(&device, tokens, gelu, EXPERTS);
            run(&device, 1, 1, |_| block.unfused(0));
            run(&device, 1, 1, |_| block.fused(0));
            let pairs = tokens * TOP_K;
            let act = bits(&block.gate_y, pairs * INTER);
            let out = bits(&block.out, tokens * HIDDEN);
            assert!(act.iter().any(|&v| v != 0) && out.iter().any(|&v| v != 0));
            let what = format!("gelu={gelu} tokens={tokens}");
            assert_eq!(
                act,
                bits(&block.fused_gate_y, pairs * INTER),
                "{what}: act(gate) * up"
            );
            assert_eq!(
                out,
                bits(&block.fused_out, tokens * HIDDEN),
                "{what}: combined rows"
            );
        }
    }
}

/// The sorted gathered path (`MG_BM=1`): the `moe_group` sort, then the fused gated kernel and
/// the split down over the SORTED rows, then the unsort — must give the same bits as the fused
/// token-order chain. This is the layout a `Sorted` bake dispatches (pairs 64–127: bucket-8
/// decode), so it pins that the sort's `indices_pad`/`x_pad` views feed the gather kernels
/// exactly as the router's `topk_inds`/token rows do.
#[test]
fn sorted_gathered_moe_matches_the_token_order_chain() {
    let Some(d) = detect_device() else { return };
    let device = d.device;
    // Gemma-4's 128 experts and Qwen3.5/3.6-MoE's 256; the bucket full, and a short step's 5 live
    // rows in the bucket of 8 — the sort's offsets count every static row, the scatter, unsort
    // and sum run the live ones, the matvecs the full grid over stale rows.
    let tokens = 8usize;
    for (gelu, experts, live) in [true, false]
        .into_iter()
        .flat_map(|g| [EXPERTS, 256].map(|e| (g, e)))
        .flat_map(|(g, e)| [tokens, 5].map(|l| (g, e, l)))
    {
        let block = Block::new(&device, tokens, gelu, experts);
        run(&device, 1, 1, |_| block.fused(0));
        let live_pairs = live * TOP_K;
        let reference_act = bits(&block.fused_gate_y, tokens * TOP_K * INTER);
        let reference_out = bits(&block.fused_out, tokens * HIDDEN);
        assert!(reference_act.iter().any(|&v| v != 0));

        let pairs = tokens * TOP_K;
        // The sort's own buffers: count/offset/total/fill, pos, indices_pad, x_pad. The u32
        // index buffers are sized in bytes (`shared_zeroed`); the bf16 row buffers start
        // all-NaN, the scratch a previous step left behind.
        let u32_buf = |n: usize| common::shared_zeroed(&device, n * 4);
        let e_buf = |n: usize| common::shared_slice(&device, &vec![0xffffu16; n]);
        let (count, offset, total) = (u32_buf(experts), u32_buf(experts), u32_buf(1));
        let (fill, pos) = (u32_buf(experts), u32_buf(pairs));
        let indices_pad = u32_buf(pairs);
        let x_pad = e_buf(pairs * HIDDEN);
        // The unsorted row buffer the down output gathers back into.
        let sorted_down_y = e_buf(pairs * HIDDEN);
        let token_rows = e_buf(pairs * HIDDEN);

        let int = |slot: u16, v: i32| ConstantValue::int(ConstSlot(slot), v);
        let offsets_pso = baked_pipeline(
            &device,
            "moe_group",
            "moe_group_offsets",
            vec![
                int(0, pairs as i32),
                int(1, experts as i32),
                int(6, 1), // MG_BM = 1: no padding — a sorted bake.
            ],
        )
        .expect("offsets");
        let init_pso = baked_pipeline(
            &device,
            "moe_group",
            "moe_group_init",
            vec![
                int(1, experts as i32),
                int(2, pairs as i32), // MG_BM=1 ⇒ mpad_max = MG_M.
                int(7, 0),            // MG_SENTINEL = 0: the gather matvecs have no skip guard.
            ],
        )
        .expect("init");
        let scatter_pso = baked_pipeline(
            &device,
            "moe_group",
            "moe_group_scatter_bfloat16",
            vec![
                int(0, pairs as i32),
                int(1, experts as i32),
                int(2, pairs as i32),
                int(3, TOP_K as i32),
                int(4, HIDDEN as i32),
            ],
        )
        .expect("scatter");
        let gather_pso = baked_pipeline(
            &device,
            "moe_group",
            "moe_group_gather_bfloat16",
            vec![int(5, HIDDEN as i32)],
        )
        .expect("gather");

        // The sorted chain: gated over the sorted rows (its own output buffers), then the
        // plain down gather-qmv (GatherRows::Pairs) over the sorted act rows, then the
        // unsort and the weighted sum over the token-order rows.
        let rows = GatherRows::Pairs;
        let act_code = int(3, i32::from(gelu));
        let mut gated_constants = block.gate.constants(rows);
        gated_constants.push(act_code);
        let gated_pso = block
            .gate
            .pipeline(&device, "affine_gather_qmv_gated", gated_constants);
        let down_pso =
            block
                .down
                .pipeline(&device, "affine_gather_qmv", block.down.constants(rows));
        let sorted_gate_y = e_buf(pairs * INTER);
        let sorted_up_y = e_buf(pairs * INTER);
        let sorted_out = e_buf(tokens * HIDDEN);

        run(&device, 1, 1, |_| {
            vec![
                Dispatch {
                    pso: &offsets_pso,
                    buffers: vec![
                        (&block.indices[0], 0),
                        (&count, 1),
                        (&offset, 2),
                        (&total, 3),
                    ],
                    groups: size(1, 1, 1),
                    threads: size(256, 1, 1),
                },
                Dispatch {
                    pso: &init_pso,
                    buffers: vec![(&indices_pad, 0), (&fill, 1)],
                    groups: size(pairs.div_ceil(256), 1, 1),
                    threads: size(256, 1, 1),
                },
                Dispatch {
                    pso: &scatter_pso,
                    buffers: vec![
                        (&block.indices[0], 0),
                        (&offset, 1),
                        (&block.x, 2),
                        (&fill, 3),
                        (&pos, 4),
                        (&indices_pad, 5),
                        (&x_pad, 6),
                    ],
                    groups: size(1, live_pairs, 1),
                    threads: size(HIDDEN.min(256), 1, 1),
                },
                Dispatch {
                    pso: &gated_pso,
                    buffers: vec![
                        (&block.gate.w, 0),
                        (&block.gate.s, 1),
                        (&block.gate.b, 2),
                        (&x_pad, 3),
                        (&indices_pad, 4),
                        (&sorted_gate_y, 5),
                        (&block.up.w, 6),
                        (&block.up.s, 7),
                        (&block.up.b, 8),
                        (&sorted_up_y, 9),
                    ],
                    groups: size(1, INTER.div_ceil(8), pairs),
                    threads: size(32, 4, 1),
                },
                Dispatch {
                    pso: &down_pso,
                    buffers: vec![
                        (&block.down.w, 0),
                        (&block.down.s, 1),
                        (&block.down.b, 2),
                        (&sorted_gate_y, 3),
                        (&indices_pad, 4),
                        (&sorted_down_y, 5),
                    ],
                    groups: size(1, HIDDEN.div_ceil(8), pairs),
                    threads: size(32, 2, 1),
                },
                Dispatch {
                    pso: &gather_pso,
                    buffers: vec![(&sorted_down_y, 0), (&pos, 1), (&token_rows, 2)],
                    groups: size(1, live_pairs, 1),
                    threads: size(HIDDEN.min(256), 1, 1),
                },
                Dispatch {
                    pso: &block.weighted_sum,
                    buffers: vec![(&token_rows, 0), (&block.scores, 1), (&sorted_out, 2)],
                    groups: size(HIDDEN.div_ceil(64), live, 1),
                    threads: size(64, 1, 1),
                },
            ]
        });
        let what = format!("sorted gelu={gelu} experts={experts} live={live}/{tokens}");
        // The unsort restores token order, so the act rows compare after applying pos, and
        // the combined output compares directly.
        let pos_v = common::read_slice::<u32>(&pos, live_pairs);
        let act_ref = bits(&block.fused_gate_y, pairs * INTER);
        let act_sorted = bits(&sorted_gate_y, pairs * INTER);
        for (p, &sorted_p) in pos_v.iter().enumerate() {
            for c in 0..INTER {
                let at = |rows: &[u16], r: usize| rows[r * INTER + c];
                assert_eq!(
                    at(&act_ref, p),
                    at(&act_sorted, sorted_p as usize),
                    "{what}: act row {p} (sorted row {sorted_p})"
                );
            }
        }
        assert_eq!(
            reference_out[..live * HIDDEN],
            bits(&sorted_out, live * HIDDEN),
            "{what}: combined rows"
        );
    }
}

/// The routing of one token's `logits` by `program`: the routing command's picks and scores, then
/// the fused kernels over them, against the gated kernel routing the token itself and storing the
/// picks and scores the combine then reads. `time`: the chains' best times per run, in µs.
fn routed_matches_the_routing_command(
    device: &common::Device,
    block: &Block,
    program: RouteProgram,
    time: bool,
) -> Result<Option<(f64, f64)>, String> {
    let mut rng = Lcg(0xfeed);
    let logits = rng.bf16s(EXPERTS, -3.0, 3.0);
    let expert_scale = common::shared_slice(device, &rng.bf16s(EXPERTS, 0.5, 1.5));
    let experts = NumExperts(EXPERTS as u32);
    let top_k = TopK(TOP_K as u32);
    let route = MoeRouteConstants {
        experts,
        top_k,
        program,
    };
    let route = baked_pipeline(device, "moe_route", "moe_route_bfloat16_bn32", route.into())
        .expect("route");
    // The command softmaxes its logits in place: the routed kernel reads its own copy.
    let (in_place, read) = (
        common::shared_slice(device, &logits),
        common::shared_slice(device, &logits),
    );
    // Each chain's picks and scores; the routed chain's start zero, its gated kernel's to store.
    let picks = || {
        (
            common::shared_zeroed(device, TOP_K * 4),
            common::shared_zeroed(device, TOP_K * 2),
        )
    };
    let ((inds, scores), (routed_inds, routed_scores)) = (picks(), picks());
    let rows = GatherRows::Tokens(top_k);
    let mut gated_c = block.gate.constants(rows);
    gated_c.push(ConstantValue::int(ConstSlot(3), 1));
    gated_c.extend(Vec::<ConstantValue>::from(RoutedConstants {
        experts,
        program,
    }));
    let gated = block
        .gate
        .pipeline(device, "affine_gather_qmv_gated", gated_c);
    let zeroed = |n: usize| common::shared_zeroed(device, n * 2);
    let (gate_y, up_y, down_y, out) = (
        zeroed(TOP_K * INTER),
        zeroed(TOP_K * INTER),
        zeroed(TOP_K * HIDDEN),
        zeroed(HIDDEN),
    );
    let chain = |routed: bool| {
        let (gy, uy, dy, o) = match routed {
            false => (
                &block.fused_gate_y,
                &block.fused_up_y,
                &block.fused_down_y,
                &block.fused_out,
            ),
            true => (&gate_y, &up_y, &down_y, &out),
        };
        let (i, sc) = match routed {
            false => (&inds, &scores),
            true => (&routed_inds, &routed_scores),
        };
        let mut g = block.gate.bindings().to_vec();
        g.extend([(&block.x, 3), (i, 4), (gy, 5)]);
        g.extend([
            (&block.up.w, 6),
            (&block.up.s, 7),
            (&block.up.b, 8),
            (uy, 9),
        ]);
        let mut c = block.down.bindings().to_vec();
        c.extend([(gy, 3), (i, 4), (dy, 5), (sc, 6), (o, 7)]);
        let mut chain = Vec::new();
        match routed {
            false => chain.push(Dispatch {
                pso: &route,
                buffers: vec![(&in_place, 0), (&inds, 1), (&scores, 2), (&expert_scale, 3)],
                groups: size(1, 1, 1),
                threads: size(32, 1, 1),
            }),
            true => g.extend([(&read, 10), (sc, 11), (&expert_scale, 12)]),
        }
        let gated = if routed { &gated } else { &block.gated };
        chain.push(Dispatch {
            pso: gated,
            buffers: g,
            groups: size(1, INTER.div_ceil(8), TOP_K),
            threads: size(32, 4, 1),
        });
        chain.push(Dispatch {
            pso: &block.combine,
            buffers: c,
            groups: size(1, HIDDEN.div_ceil(4), 1),
            threads: size(32, TOP_K, 1),
        });
        chain
    };
    run(device, 1, 1, |_| chain(false));
    run(device, 1, 1, |_| chain(true));
    let picked = common::read_slice::<u32>(&inds, TOP_K);
    let mut distinct = picked.clone();
    distinct.sort_unstable();
    distinct.dedup();
    if distinct.len() != TOP_K || picked.iter().any(|&e| e as usize >= EXPERTS) {
        return Err(format!("the routing command's picks {picked:?}"));
    }
    if picked != common::read_slice::<u32>(&routed_inds, TOP_K) {
        return Err("the stored picks differ".into());
    }
    if bits(&scores, TOP_K) != bits(&routed_scores, TOP_K) {
        return Err("the stored scores differ".into());
    }
    let act = bits(&block.fused_gate_y, TOP_K * INTER);
    if act != bits(&gate_y, TOP_K * INTER) {
        return Err("act(gate) * up differs".into());
    }
    let combined = bits(&block.fused_out, HIDDEN);
    if combined.iter().all(|&v| v == 0) || combined != bits(&out, HIDDEN) {
        return Err("the combined rows differ".into());
    }
    // Timed after the checks: the routing command's softmax runs in place.
    Ok(time.then(|| {
        (
            run(device, 300, 4, |_| chain(false)),
            run(device, 300, 4, |_| chain(true)),
        )
    }))
}

#[test]
fn routed_moe_kernels_match_the_routing_command_then_the_fused_kernels() {
    let Some(d) = detect_device() else { return };
    let device = d.device;
    let block = Block::new(&device, 1, true, EXPERTS);
    // Gemma-4's program, and the shared-expert router's (a softmax over every expert first).
    let gemma = RouteProgram {
        pre_softmax: false,
        scale: Some(Scale(0.018_844_6)),
        post: RoutePost::Softmax,
        expert_scale: Some(LayerId(0)),
    };
    let shared = RouteProgram {
        pre_softmax: true,
        scale: None,
        post: RoutePost::Renorm,
        expert_scale: None,
    };
    for (what, program) in [("gemma", gemma), ("shared", shared)] {
        routed_matches_the_routing_command(&device, &block, program, false)
            .unwrap_or_else(|e| panic!("{what}: {e}"));
    }
}

/// The sort's histogram + padded scan at Qwen3.5/3.6-MoE's 256 experts — twice Gemma-4's — with
/// pairs routed to every expert, the ones past 128 included: each expert's count and padded
/// offset, and the padded total, must match the host's. Dispatched as a grouped bake does it
/// (`MG_BM = 64`, one threadgroup of 256 threads, a 4096-token bucket at top 8).
#[test]
fn moe_group_offsets_count_every_expert_at_256() {
    let Some(d) = detect_device() else { return };
    let device = d.device;
    const QWEN_EXPERTS: usize = 256;
    const BM: usize = 64;
    let pairs = 4096 * TOP_K;
    // Skewed: expert e takes a share growing with e, so the high experts carry the most rows.
    let mut rng = Lcg(7);
    let inds: Vec<u32> = (0..pairs)
        .map(|_| (rng.next().sqrt() * QWEN_EXPERTS as f32) as u32)
        .collect();
    let mut want_count = vec![0u32; QWEN_EXPERTS];
    for &e in &inds {
        want_count[e as usize] += 1;
    }
    assert!(want_count[QWEN_EXPERTS / 2..].iter().all(|&c| c > 0));
    let mut want_offset = Vec::with_capacity(QWEN_EXPERTS);
    let mut acc = 0u32;
    for &c in &want_count {
        want_offset.push(acc);
        acc += c.div_ceil(BM as u32) * BM as u32;
    }

    let u32_buf = |n: usize| common::shared_zeroed(&device, n * 4);
    let (count, offset, total) = (u32_buf(QWEN_EXPERTS), u32_buf(QWEN_EXPERTS), u32_buf(1));
    let topk_inds = common::shared_slice(&device, &inds);
    let int = |slot: u16, v: i32| ConstantValue::int(ConstSlot(slot), v);
    let offsets_pso = baked_pipeline(
        &device,
        "moe_group",
        "moe_group_offsets",
        vec![
            int(0, pairs as i32),
            int(1, QWEN_EXPERTS as i32),
            int(6, BM as i32),
        ],
    )
    .expect("offsets");
    run(&device, 1, 1, |_| {
        vec![Dispatch {
            pso: &offsets_pso,
            buffers: vec![(&topk_inds, 0), (&count, 1), (&offset, 2), (&total, 3)],
            groups: size(1, 1, 1),
            threads: size(256, 1, 1),
        }]
    });
    assert_eq!(common::read_slice::<u32>(&count, QWEN_EXPERTS), want_count);
    assert_eq!(
        common::read_slice::<u32>(&offset, QWEN_EXPERTS),
        want_offset
    );
    assert_eq!(common::read_slice::<u32>(&total, 1), vec![acc]);
}

/// `n_out` rows over `k_in` of a plain (dense, not gathered) 4-bit matvec, read from DRAM:
/// 16 distinct weight matrices, cycled, so no dispatch finds its weights in cache.
fn plain_qmv_us(device: &common::Device, n_out: usize, k_in: usize) -> f64 {
    let mut rng = Lcg(7);
    let mats: Vec<[common::Buffer; 3]> = (0..16)
        .map(|_| {
            let codes: Vec<u8> = (0..n_out * k_in / 2)
                .map(|_| (rng.next() * 256.0) as u8)
                .collect();
            let groups = n_out * k_in / GS;
            [
                common::shared_slice(device, &codes),
                common::shared_slice(device, &rng.bf16s(groups, 0.002, 0.02)),
                common::shared_slice(device, &rng.bf16s(groups, -0.08, 0.08)),
            ]
        })
        .collect();
    let x = common::shared_slice(device, &rng.bf16s(k_in, -1.0, 1.0));
    let y = common::shared_zeroed(device, n_out * 2);
    let fast = n_out.is_multiple_of(8) && k_in.is_multiple_of(512);
    let fast = if fast { "_fast" } else { "" };
    let name = format!("affine_qmv{fast}_bf16_s_bf16_gs_64_b_4_batch_0");
    let constants = AffineQmvConstants {
        k: KDimI32(k_in as i32),
        n: NDimI32(n_out as i32),
        codes: AffineCodes::AsWritten,
    };
    let pso = baked_pipeline(device, "quantized_qmv", &name, constants.into()).expect("qmv");
    run(device, 300, 4, |i| {
        let [w, s, b] = &mats[i % mats.len()];
        vec![Dispatch {
            pso: &pso,
            buffers: vec![(w, 0), (s, 1), (b, 2), (&x, 3), (&y, 4)],
            groups: size(1, n_out.div_ceil(8), 1),
            threads: size(32, 2, 1),
        }]
    })
}

#[test]
#[ignore]
fn bench_moe_fused() {
    let Some(d) = detect_device() else { return };
    let device = d.device;
    let block = Block::new(&device, 1, true, EXPERTS);
    let sets = block.indices.len();
    let pct = |new: f64, old: f64| (new / old - 1.0) * 100.0;
    run(&device, 200, 2, |i| block.unfused(i % sets));
    for round in 0..3 {
        let unfused = run(&device, 300, 4, |i| block.unfused(i % sets));
        let fused = run(&device, 300, 4, |i| block.fused(i % sets));
        let gated_3 = run(&device, 300, 4, |i| {
            block.unfused(i % sets).into_iter().take(3).collect()
        });
        let gated_1 = run(&device, 300, 4, |i| {
            block.fused(i % sets).into_iter().take(1).collect()
        });
        let combine_2 = run(&device, 300, 4, |i| {
            block.unfused(i % sets).into_iter().skip(3).collect()
        });
        let combine_1 = run(&device, 300, 4, |i| {
            block.fused(i % sets).into_iter().skip(1).collect()
        });
        println!(
            "BENCH moe round {round}: whole expert half: today 5 launches {unfused:.1} us, \
             fused 2 launches {fused:.1} us ({:+.1}%)",
            pct(fused, unfused)
        );
        println!(
            "BENCH moe round {round}: gate+up+act: today 3 launches {gated_3:.1} us, \
             fused 1 launch {gated_1:.1} us ({:+.1}%)",
            pct(gated_1, gated_3)
        );
        println!(
            "BENCH moe round {round}: down+combine: today 2 launches {combine_2:.1} us, \
             fused 1 launch {combine_1:.1} us ({:+.1}%)",
            pct(combine_1, combine_2)
        );
    }
    // The routing as its own command, then the fused kernels; the gated kernel routing itself.
    let gemma = RouteProgram {
        pre_softmax: false,
        scale: Some(Scale(0.018_844_6)),
        post: RoutePost::Softmax,
        expert_scale: Some(LayerId(0)),
    };
    for round in 0..3 {
        let timed = routed_matches_the_routing_command(&device, &block, gemma, true);
        let (command, routed) = timed.expect("routed").expect("timed");
        println!(
            "BENCH moe round {round}: routing command + 2 launches {command:.1} us, \
             routed in the gated launch {routed:.1} us ({:+.1}%)",
            pct(routed, command)
        );
    }
    // One gathered projection alone (today's kernel), then the same bytes as one expert
    // projection (8.9 MB) and as the gate+up pair (17.8 MB) as a plain matvec from DRAM: what a
    // launch of that size reaches without the expert gather.
    let bytes = |n: usize, k: usize| (n * k / 2 + 2 * 2 * n * k / GS) as f64;
    for (label, skip) in [("gate", 0), ("down", 3)] {
        let us = run(&device, 300, 4, |i| {
            block
                .unfused(i % sets)
                .into_iter()
                .skip(skip)
                .take(1)
                .collect()
        });
        let b = TOP_K as f64 * bytes(INTER, HIDDEN);
        println!(
            "BENCH gathered matvec alone, {label} (8 experts, {:.1} MB): {us:.1} us, {:.0} GB/s",
            b / 1e6,
            b / (us * 1e-6) / 1e9
        );
    }
    for n_out in [TOP_K * INTER, 2 * TOP_K * INTER] {
        let us = plain_qmv_us(&device, n_out, HIDDEN);
        let b = bytes(n_out, HIDDEN);
        println!(
            "BENCH plain matvec from DRAM, {n_out}x{HIDDEN} ({:.1} MB): {us:.1} us, {:.0} GB/s",
            b / 1e6,
            b / (us * 1e-6) / 1e9
        );
    }
}

/// The combine computing a shared expert's gated rows and the residual add as it stores each row
/// (`MetalFusion::CombineEpilogue`) leaves the same residual as the combine, `gate_scale`, then
/// `residual_add`, bit for bit; the same combine without the gate scale leaves another.
#[test]
fn combine_ends_match_gate_scale_then_the_residual_add() {
    let Some(d) = detect_device() else { return };
    let device = d.device;
    for tokens in [1, 3] {
        let block = Block::new(&device, tokens, false, EXPERTS);
        let mut rng = Lcg(0xc0de);
        let n = tokens * HIDDEN;
        let shared = common::shared_slice(&device, &rng.bf16s(n, -1.0, 1.0));
        let g = common::shared_slice(&device, &rng.bf16s(tokens, -3.0, 3.0));
        let h = rng.bf16s(n, -4.0, 4.0);
        let residual = common::shared_slice(&device, &h);
        let gated_rows = common::shared_zeroed(&device, n * 2);
        // The reference: the fused chain into `fused_out`, then the two kernels.
        let elementwise = |d: &common::Device, symbol: (&'static str, &'static str), c| {
            baked_pipeline(d, symbol.0, symbol.1, c).expect(symbol.1)
        };
        let gate_scale = elementwise(
            &device,
            ("gate_scale", "gate_scale_bf16"),
            vec![
                ConstantValue::uint(ConstSlot(0), n as u32),
                ConstantValue::uint(ConstSlot(1), HIDDEN as u32),
            ],
        );
        let add = elementwise(
            &device,
            ("elementwise", "residual_add_bf16_specialized"),
            vec![],
        );
        run(&device, 1, 1, |_| {
            let mut chain = block.fused(0);
            chain.extend([
                Dispatch {
                    pso: &gate_scale,
                    buffers: vec![
                        (&gated_rows, 0),
                        (&block.fused_out, 1),
                        (&shared, 2),
                        (&g, 3),
                    ],
                    groups: size(n.div_ceil(256), 1, 1),
                    threads: size(256, 1, 1),
                },
                Dispatch {
                    pso: &add,
                    buffers: vec![(&residual, 0), (&gated_rows, 1)],
                    groups: size(n / 256, 1, 1),
                    threads: size(256, 1, 1),
                },
            ]);
            chain
        });
        let want = bits(&residual, n);
        assert_ne!(
            want,
            bits(&common::shared_slice(&device, &h), n),
            "tokens={tokens}: no add"
        );
        // The combine with its ends (and with the residual add alone), into copies of `h`.
        let rows = GatherRows::Tokens(TopK(TOP_K as u32));
        for gate_scaled in [true, false] {
            let mut c = block.down.constants(rows);
            c.extend(gate_scaled.then(|| ConstantValue::boolean(ConstSlot(18), true)));
            c.push(ConstantValue::boolean(ConstSlot(19), true));
            let ends = block.down.pipeline(&device, "affine_gather_qmv_combine", c);
            let into = common::shared_slice(&device, &h);
            run(&device, 1, 1, |_| {
                let mut chain = block.fused(0);
                let combine = chain.last_mut().expect("the combine");
                combine.pso = &ends;
                for b in combine.buffers.iter_mut().filter(|(_, i)| *i == 7) {
                    b.0 = &into;
                }
                combine.buffers.extend([(&shared, 8), (&g, 9)]);
                chain
            });
            let got = bits(&into, n);
            match gate_scaled {
                true => assert_eq!(got, want, "tokens={tokens}: gate scale + residual add"),
                false => assert_ne!(got, want, "tokens={tokens}: the gate scale matters"),
            }
        }
    }
}

/// The gated kernel normalizing its token rows as it loads them (`MetalFusion::NormedQmv`: the
/// MoE block's input norm folded into its gathered experts) skips the normed row's rounding: its up
/// rows must be as close to the exact `up · rmsnorm(x, gain)` as the norm, then the kernel, are —
/// and the kernel over the raw rows, which the same bound must reject, shows the bound sees a
/// missing norm.
#[test]
fn a_normed_gated_kernel_is_as_close_to_the_exact_normed_rows_as_the_norm_then_the_kernel() {
    use half::bf16;
    use scratchy_target_metal::tape::ids::{BucketM, QSize, RmsNormEps};
    use scratchy_target_metal::tape::kernel_constants::{NORM_THREADS, RmsNormConstants};
    use scratchy_target_metal::tape::step::{Eps, GainOffset, RowNorm};
    #[derive(Clone, Copy, Debug)]
    enum How {
        NormThenKernel,
        Normed,
        Raw,
    }
    let Some(d) = detect_device() else { return };
    let device = d.device;
    const EPS: f32 = 1e-6;
    let block = Block::new(&device, 1, false, EXPERTS);
    let gain = Lcg(0x6a1).bf16s(HIDDEN, 0.5, 1.5);
    let gain_buf = common::shared_slice(&device, &gain);
    let rows = GatherRows::Tokens(TopK(TOP_K as u32));
    let mut c = block.gate.constants(rows);
    c.push(ConstantValue::int(ConstSlot(3), 0));
    let norm = RowNorm {
        layer: LayerId(0),
        eps: Eps(EPS),
        offset: GainOffset(0.0),
    };
    c.extend(Vec::<ConstantValue>::from(norm));
    let normed = block.gate.pipeline(&device, "affine_gather_qmv_gated", c);
    let rmsnorm = RmsNormConstants {
        bucket_m: BucketM(1),
        q_size: QSize(HIDDEN as u32),
        rms_norm_eps: RmsNormEps(EPS),
        weight_offset: 0.0,
    };
    let rmsnorm = baked_pipeline(
        &device,
        "rmsnorm",
        "rmsnorm_bf16_s_bf16_specialized",
        rmsnorm.into(),
    )
    .expect("rmsnorm");
    let pairs = TOP_K;
    // The SiLU gated kernel's up rows over the token's row, as `how` reads it.
    let up_rows = |how: How| {
        let up_y = common::shared_zeroed(&device, pairs * INTER * 2);
        let gate_y = common::shared_zeroed(&device, pairs * INTER * 2);
        let normed_x = common::shared_zeroed(&device, HIDDEN * 2);
        run(&device, 1, 1, |_| {
            let mut chain = Vec::new();
            let (pso, x) = match how {
                How::NormThenKernel => {
                    chain.push(Dispatch {
                        pso: &rmsnorm,
                        buffers: vec![(&normed_x, 0), (&block.x, 1), (&gain_buf, 2)],
                        groups: size(1, 1, 1),
                        threads: size(NORM_THREADS as usize, 1, 1),
                    });
                    (&block.gated, &normed_x)
                }
                How::Normed => (&normed, &block.x),
                How::Raw => (&block.gated, &block.x),
            };
            let mut gated = block.gate.bindings().to_vec();
            gated.extend([(x, 3), (&block.indices[0], 4), (&gate_y, 5)]);
            gated.extend([(&block.up.w, 6), (&block.up.s, 7), (&block.up.b, 8)]);
            gated.push((&up_y, 9));
            if let How::Normed = how {
                gated.push((&gain_buf, 15));
            }
            chain.push(Dispatch {
                pso,
                buffers: gated,
                groups: size(1, INTER.div_ceil(8), pairs),
                threads: size(32, 4, 1),
            });
            chain
        });
        common::read_slice::<bf16>(&up_y, pairs * INTER)
    };
    // The exact up rows, each with its bound: the output's rounding, plus twice what rounding each
    // normed input to bf16 can move its dot.
    let x: Vec<f64> = common::read_slice::<bf16>(&block.x, HIDDEN)
        .iter()
        .map(|v| f64::from(v.to_f32()))
        .collect();
    let rms = (x.iter().map(|v| v * v).sum::<f64>() / HIDDEN as f64 + f64::from(EPS)).sqrt();
    let xn: Vec<f64> = (x.iter().zip(&gain))
        .map(|(v, g)| v / rms * f64::from(g.to_f32()))
        .collect();
    let codes = common::read_slice::<u8>(&block.up.w, EXPERTS * INTER * HIDDEN / 2);
    let scales = common::read_slice::<bf16>(&block.up.s, EXPERTS * INTER * HIDDEN / GS);
    let biases = common::read_slice::<bf16>(&block.up.b, EXPERTS * INTER * HIDDEN / GS);
    let indices = common::read_slice::<u32>(&block.indices[0], pairs);
    let u = 2f64.powi(-8);
    let exact: Vec<(f64, f64)> = (0..pairs * INTER)
        .map(|at| {
            let (e, r) = (indices[at / INTER] as usize, at % INTER);
            let terms = (0..HIDDEN).map(|k| {
                let i = (e * INTER + r) * HIDDEN + k;
                let code = f64::from((codes[i / 2] >> (4 * (i % 2))) & 15);
                let (s, b) = (scales[i / GS].to_f32(), biases[i / GS].to_f32());
                (f64::from(s) * code + f64::from(b)) * xn[k]
            });
            let (dot, sum) = terms.fold((0.0, 0.0), |(d, m), t| (d + t, m + t.abs()));
            (dot, u * dot.abs() + 2.0 * u * sum)
        })
        .collect();
    let outside = |how| {
        let got = up_rows(how);
        got.iter()
            .zip(&exact)
            .position(|(g, &(e, bound))| (f64::from(g.to_f32()) - e).abs() > bound)
    };
    assert_eq!(
        outside(How::NormThenKernel),
        None,
        "the norm, then the gated kernel"
    );
    assert_eq!(outside(How::Normed), None, "the normed gated kernel");
    assert!(
        outside(How::Raw).is_some(),
        "the bound accepts the gated kernel over the raw rows"
    );
}
