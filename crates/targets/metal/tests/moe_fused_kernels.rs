// The MoE block's fused expert kernels against the kernels they replace, on Gemma-4-26B-A4B's
// decode shapes (128 experts, top 8, hidden 2816, expert width 704, 4-bit g64, bf16):
//   affine_gather_qmv_gated   = gate gather-qmv, up gather-qmv, gelu_mul / silu_mul
//   affine_gather_qmv_combine = down gather-qmv, moe_weighted_sum
// Each must give the same bits. `bench_moe_fused` times both chains (`BENCH` lines).
mod common;

use std::time::Instant;

use half::bf16;
use objc2_metal::MTLSize;
use scratchy_target_metal::aot::{BakedPipeline, baked_pipeline};
use scratchy_target_metal::device::detect_device;
use scratchy_target_metal::tape::constants::{ConstSlot, ConstantValue};
use scratchy_target_metal::tape::ids::{KDimI32, NDimI32, TopK};
use scratchy_target_metal::tape::kernel_constants::{
    AffineCodes, AffineGatherQmvConstants, AffineQmvConstants, GatherRows,
};

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
    fn new(device: &common::Device, rng: &mut Lcg, n_out: usize, k_in: usize) -> Self {
        let codes: Vec<u8> = (0..EXPERTS * n_out * k_in / 2)
            .map(|_| (rng.next() * 256.0) as u8)
            .collect();
        let groups = EXPERTS * n_out * k_in / GS;
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
    /// `gelu` picks GELU (tanh) over SiLU as the gated activation.
    fn new(device: &common::Device, tokens: usize, gelu: bool) -> Self {
        let mut rng = Lcg(0x5eed);
        let gate = Experts::new(device, &mut rng, INTER, HIDDEN);
        let up = Experts::new(device, &mut rng, INTER, HIDDEN);
        let down = Experts::new(device, &mut rng, HIDDEN, INTER);
        let pairs = tokens * TOP_K;
        // Set s, token t, slot j: expert (s + 16 * (j + t)) % 128 — distinct within a token.
        let indices = (0..16u32)
            .map(|s| {
                let set: Vec<u32> = (0..pairs as u32)
                    .map(|p| (s + 16 * (p % TOP_K as u32 + p / TOP_K as u32)) % EXPERTS as u32)
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
            let block = Block::new(&device, tokens, gelu);
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
    let block = Block::new(&device, 1, true);
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
