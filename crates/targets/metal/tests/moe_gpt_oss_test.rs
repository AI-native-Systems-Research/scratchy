// The gpt-oss MoE block's expert kernels at the 120b's decode shapes — 128 experts, top 4,
// hidden 2880, intermediate 2880, 2-bit g64, bf16/bf16, SwiGLU-OAI, per-expert LINEAR biases:
//   affine_gather_qmv_gated   gate/up linear biases at buffer(14), switch at slot 4
//   affine_gather_qmv_combine down linear bias at buffer(10), switch at slot 6
//   the gated kernel routing its token itself (PRE 3: F32 logit + linear bias, softmax over
//   the top-4's biased values)
// The routed kernel must give the routing command's picks and scores and the unrouted chain's
// bits; and the whole block must match the CPU reference (`cpu_reference.rs`'s b2 matvec, the
// bias adds, `swiglu_oai`, the weighted combine) — the only independent oracle for the
// activation's ±7 clamps and the biases' positions.
mod common;

use half::bf16;
use objc2_metal::MTLSize;
use scratchy_target_metal::aot::{BakedPipeline, baked_pipeline};
use scratchy_target_metal::cpu_reference::affine_qmv_b2_bf16_s_bf16;
use scratchy_target_metal::device::detect_device;
use scratchy_target_metal::tape::constants::{ConstSlot, ConstantValue};
use scratchy_target_metal::tape::ids::{KDimI32, NDimI32, NumExperts, TopK};
use scratchy_target_metal::tape::kernel_constants::{
    AffineCodes, AffineGatherQmvConstants, AffineQmvConstants, GatherRows, MoeRouteConstants,
    RoutedConstants,
};
use scratchy_target_metal::tape::step::{LayerId, RoutePost, RoutePre, RouteProgram};

const EXPERTS: usize = 128;
const TOP_K: usize = 4;
// Small, law-true shapes — K a multiple of GS (64) and of the b2 pack (4),
// N wide enough to carry the linear biases past the ±7 clamps. The 120b's
// real 2880×2880 expert slabs made this fixture generate ~1G values per
// build; the b2 kernels' real-width behavior is pinned by quantized_b2_test
// and affine_gather_qmv_test at N,K ∈ {2880, 4096} — THIS test's pins are
// the composition: PRE 3 routing, SwiGLU-OAI, the linear-bias slots, and
// routed-vs-unrouted equality.
const HIDDEN: usize = 192;
const INTER: usize = 192;
const GS: usize = 64;
/// gpt-oss's routing program: the F32 linear bias orders the top-4, the biased values are read
/// back as the scores, softmaxed over the top-4.
const PROGRAM: RouteProgram = RouteProgram {
    pre: RoutePre::Bias(LayerId(0)),
    scale: None,
    post: RoutePost::Softmax,
    expert_scale: None,
};

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

/// `swiglu_oai_mul_f` (`gated_act.h`) in f32: `y = (clamp(u, ±7)+1) · min(g, 7) ·
/// sigmoid(1.702·min(g, 7))` — the ±7 limit is gpt-oss's `swiglu_limit`, baked into the kernel.
fn swiglu_oai(g: f32, u: f32) -> f32 {
    const LIMIT: f32 = 7.0;
    let gate = g.min(LIMIT);
    let up = u.clamp(-LIMIT, LIMIT);
    (up + 1.0) * gate / (1.0 + (-1.702 * gate).exp())
}

/// One projection's experts: 2-bit codes (4 per byte, LSB-first — the bitstream
/// `affine_qmv_b2` reads), bf16 scales and biases, `[experts, n_out, k_in]` with each row
/// packed to `k_in / 4` bytes. The host copies stay for the CPU reference.
struct Experts {
    w: common::Buffer,
    s: common::Buffer,
    b: common::Buffer,
    codes: Vec<u8>,
    scales: Vec<bf16>,
    biases: Vec<bf16>,
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
        // ONE expert's slab, replicated across the pool with a per-expert XOR
        // of the codes: the fixture costs one expert's generation, not
        // `experts`' (CI runs this suite in debug), while every expert still
        // reads distinct bits, so a mis-gathered expert computes the wrong
        // rows. The XOR is a bijection on the slab's bytes, so the code
        // histogram — codes spanning the scale's 4 steps — is unchanged.
        // 2-bit codes: 4 per byte, so a row packs k_in / 4 bytes.
        let slab: Vec<u8> = (0..n_out * k_in / 4)
            .map(|_| (rng.next() * 256.0) as u8)
            .collect();
        let codes: Vec<u8> = (0..experts)
            .flat_map(|e| slab.iter().map(move |&c| c ^ e as u8))
            .collect();
        // Centred weights of ~±0.03: codes span 4 steps of the scale, the bias takes off half.
        let steps = 3.0f32;
        let groups = n_out * k_in / GS;
        let (slab_scales, slab_biases): (Vec<bf16>, Vec<bf16>) = (0..groups)
            .map(|_| {
                let scale = (0.5 + rng.next()) * 0.06 / steps;
                let bias = -scale * steps / 2.0 * (1.0 + 0.1 * (rng.next() * 2.0 - 1.0));
                (bf16::from_f32(scale), bf16::from_f32(bias))
            })
            .unzip();
        let scales = slab_scales.repeat(experts);
        let biases = slab_biases.repeat(experts);
        Self {
            w: common::shared_slice(device, &codes),
            s: common::shared_slice(device, &scales),
            b: common::shared_slice(device, &biases),
            codes,
            scales,
            biases,
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

    /// `kernel`'s symbol for this projection's shape, as the lowering picks it (`_fast` only
    /// over the fast rule's shapes — 2880 is not a multiple of 512, so the generic kernel).
    fn symbol(&self, kernel: &str) -> String {
        let fast = scratchy_target_metal::tape::quantized::qmv_fast_covers(
            self.n_out as u32,
            self.k_in as u32,
        );
        let fast = if fast { "_fast" } else { "" };
        format!("{kernel}{fast}_bf16_s_bf16_gs_{GS}_b_2")
    }

    fn pipeline(&self, d: &common::Device, kernel: &str, c: Vec<ConstantValue>) -> BakedPipeline {
        baked_pipeline(d, "quantized_qmv", &self.symbol(kernel), c).expect(kernel)
    }

    /// The gated kernel's constants: the gather matvec's, the SwiGLU-OAI act (slot 3), and the
    /// linear-bias switch (slot 4).
    fn gated_constants(&self, rows: GatherRows) -> Vec<ConstantValue> {
        let mut c = self.constants(rows);
        c.push(ConstantValue::int(ConstSlot(3), 2));
        c.push(ConstantValue::boolean(ConstSlot(4), true));
        c
    }

    /// The combine kernel's constants: the down matvec's, its top-k (slot 2), and the
    /// linear-bias switch (slot 6).
    fn combine_constants(&self, rows: GatherRows) -> Vec<ConstantValue> {
        let mut c = self.constants(rows);
        c.push(ConstantValue::boolean(ConstSlot(6), true));
        c
    }

    fn bindings(&self) -> [(&common::Buffer, usize); 3] {
        [(&self.w, 0), (&self.s, 1), (&self.b, 2)]
    }
}

/// A per-expert linear bias the loader packs: `[E, rows]` bf16.
struct LinearBias {
    buf: common::Buffer,
    host: Vec<bf16>,
}

impl LinearBias {
    /// The gate/up concat `[E, 2·inter]` — expert e's gate rows then its up rows — wide enough
    /// to push past the ±7 clamps on its own: the reference must catch a mis-clamp, not just a
    /// mis-add.
    fn gate_up(device: &common::Device, rng: &mut Lcg, experts: usize) -> Self {
        let host = rng.bf16s(experts * 2 * INTER, -9.0, 9.0);
        Self {
            buf: common::shared_slice(device, &host),
            host,
        }
    }

    fn down(device: &common::Device, rng: &mut Lcg, experts: usize) -> Self {
        let host = rng.bf16s(experts * HIDDEN, -2.0, 2.0);
        Self {
            buf: common::shared_slice(device, &host),
            host,
        }
    }
}

/// One dispatch, encoded with a barrier after it.
struct Dispatch<'a> {
    pso: &'a BakedPipeline,
    buffers: Vec<(&'a common::Buffer, usize)>,
    groups: MTLSize,
    threads: MTLSize,
}

/// Runs `chain` once.
fn run(device: &common::Device, chain: Vec<Dispatch<'_>>) {
    let mut batch = common::Mtl4DispatchBatch::begin(device).expect("an MTL4 queue");
    for d in &chain {
        batch.encode(d.pso, &d.buffers, &[], &[], &[], d.groups, d.threads);
        batch.barrier();
    }
    batch.commit(true);
}

fn bits(buf: &common::Buffer, n: usize) -> Vec<u16> {
    common::read_slice::<u16>(buf, n)
}

/// The gpt-oss MoE block on `tokens` rows: the experts, the linear and router biases, and both
/// chains' pipelines and buffers.
struct Block {
    gate: Experts,
    up: Experts,
    down: Experts,
    tokens: usize,
    x: common::Buffer,
    x_host: Vec<bf16>,
    /// The unrouted chain's picks and scores — the routing command writes them.
    indices: common::Buffer,
    scores: common::Buffer,
    /// The routed kernel's own picks and scores — it stores them as it routes.
    routed_indices: common::Buffer,
    routed_scores: common::Buffer,
    logits: common::Buffer,
    gate_up_bias: LinearBias,
    down_bias: LinearBias,
    router_bias: common::Buffer,
    route: BakedPipeline,
    gated: BakedPipeline,
    routed_gated: BakedPipeline,
    combine: BakedPipeline,
    // The unrouted chain's outputs.
    gate_y: common::Buffer,
    down_y: common::Buffer,
    out: common::Buffer,
    // The routed chain's.
    routed_gate_y: common::Buffer,
    routed_down_y: common::Buffer,
    routed_out: common::Buffer,
}

impl Block {
    fn new(device: &common::Device, tokens: usize) -> Self {
        let mut rng = Lcg(0x900d);
        let (gate, up, down) = (
            Experts::new(device, &mut rng, EXPERTS, INTER, HIDDEN),
            Experts::new(device, &mut rng, EXPERTS, INTER, HIDDEN),
            Experts::new(device, &mut rng, EXPERTS, HIDDEN, INTER),
        );
        let pairs = tokens * TOP_K;
        // Distinct experts per token, mixed across the bank.
        let indices_host: Vec<u32> = (0..pairs as u32)
            .map(|p| (17 + 31 * (p % TOP_K as u32 + p / TOP_K as u32)) % EXPERTS as u32)
            .collect();
        let scores_host = rng.bf16s(pairs, 0.05, 0.45);
        let x_host = rng.bf16s(tokens * HIDDEN, -1.0, 1.0);
        let router_bias_host: Vec<f32> = (0..EXPERTS).map(|_| -0.5 + rng.next()).collect();
        // A 4th-vs-5th tie in the BIASED logits is outside the two routers'
        // contract — the route command and the in-kernel router may order an
        // exact tie differently — so the fixture re-rolls until it draws
        // none. The fixture's draw count must not be what decides this.
        let mut logits_host = rng.bf16s(tokens * EXPERTS, -3.0, 3.0);
        while (0..tokens).any(|t| {
            let mut biased: Vec<f32> = (0..EXPERTS)
                .map(|e| logits_host[t * EXPERTS + e].to_f32() + router_bias_host[e])
                .collect();
            biased.sort_by(|a, b| b.total_cmp(a));
            biased[TOP_K - 1] == biased[TOP_K]
        }) {
            logits_host = rng.bf16s(tokens * EXPERTS, -3.0, 3.0);
        }
        let (gate_up_bias, down_bias) = (
            LinearBias::gate_up(device, &mut rng, EXPERTS),
            LinearBias::down(device, &mut rng, EXPERTS),
        );
        let (rows, experts) = (
            GatherRows::Tokens(TopK(TOP_K as u32)),
            NumExperts(EXPERTS as u32),
        );
        let route = MoeRouteConstants {
            experts,
            top_k: TopK(TOP_K as u32),
            program: PROGRAM,
        };
        let mut routed_gated_c = gate.gated_constants(rows);
        routed_gated_c.extend(Vec::<ConstantValue>::from(RoutedConstants {
            experts,
            program: PROGRAM,
        }));
        let zeroed = |n: usize| common::shared_zeroed(device, n * 2);
        Self {
            route: baked_pipeline(device, "moe_route", "moe_route_bfloat16_bn32", route.into())
                .expect("route"),
            gated: gate.pipeline(
                device,
                "affine_gather_qmv_gated",
                gate.gated_constants(rows),
            ),
            routed_gated: gate.pipeline(device, "affine_gather_qmv_gated", routed_gated_c),
            combine: down.pipeline(
                device,
                "affine_gather_qmv_combine",
                down.combine_constants(rows),
            ),
            gate,
            up,
            down,
            tokens,
            x: common::shared_slice(device, &x_host),
            x_host,
            indices: common::shared_slice(device, &indices_host),
            scores: common::shared_slice(device, &scores_host),
            routed_indices: common::shared_zeroed(device, pairs * 4),
            routed_scores: common::shared_zeroed(device, pairs * 2),
            logits: common::shared_slice(device, &logits_host),
            gate_up_bias,
            down_bias,
            router_bias: common::shared_slice(device, &router_bias_host),
            gate_y: zeroed(pairs * INTER),
            down_y: zeroed(pairs * HIDDEN),
            out: zeroed(tokens * HIDDEN),
            routed_gate_y: zeroed(pairs * INTER),
            routed_down_y: zeroed(pairs * HIDDEN),
            routed_out: zeroed(tokens * HIDDEN),
        }
    }

    /// The gated kernel over `indices`' picks, its activation into `gate_y`: buffers 0-2 the
    /// gate projection, 3-5 x / indices / gate_y, 6-9 the up projection and its rows, 14 the
    /// gate/up linear biases.
    fn gated<'a>(
        &'a self,
        pso: &'a BakedPipeline,
        indices: &'a common::Buffer,
        gate_y: &'a common::Buffer,
        routed: Option<(&'a common::Buffer, &'a common::Buffer, &'a common::Buffer)>,
    ) -> Dispatch<'a> {
        let mut buffers = self.gate.bindings().to_vec();
        buffers.extend([(&self.x, 3), (indices, 4), (gate_y, 5)]);
        buffers.extend([(&self.up.w, 6), (&self.up.s, 7), (&self.up.b, 8)]);
        buffers.push((&self.gate_up_bias.buf, 14));
        if let Some((logits, scores, router_bias)) = routed {
            buffers.extend([(logits, 10), (scores, 11), (router_bias, 13)]);
        }
        Dispatch {
            pso,
            buffers,
            groups: size(self.tokens * TOP_K, INTER.div_ceil(8), 1),
            threads: size(32, 4, 1),
        }
    }

    /// The combine over `indices`' picks and `scores`, reading the gated rows `gate_y` into
    /// `out`: buffers 0-2 the down projection, 3-7 the rows / indices / scratch / scores / out,
    /// 10 the down linear bias.
    fn combine<'a>(
        &'a self,
        indices: &'a common::Buffer,
        scores: &'a common::Buffer,
        gate_y: &'a common::Buffer,
        down_y: &'a common::Buffer,
        out: &'a common::Buffer,
    ) -> Dispatch<'a> {
        let mut buffers = self.down.bindings().to_vec();
        buffers.extend([(gate_y, 3), (indices, 4), (down_y, 5)]);
        buffers.extend([(scores, 6), (out, 7), (&self.down_bias.buf, 10)]);
        Dispatch {
            pso: &self.combine,
            buffers,
            groups: size(self.tokens, HIDDEN.div_ceil(4), 1),
            threads: size(32, TOP_K, 1),
        }
    }

    /// The unrouted chain: the routing command, then the fused kernels reading its picks.
    fn unrouted(&self) -> Vec<Dispatch<'_>> {
        vec![
            Dispatch {
                pso: &self.route,
                buffers: vec![
                    (&self.logits, 0),
                    (&self.indices, 1),
                    (&self.scores, 2),
                    (&self.router_bias, 4),
                ],
                groups: size(1, self.tokens, 1),
                threads: size(32, 1, 1),
            },
            self.gated(&self.gated, &self.indices, &self.gate_y, None),
            self.combine(
                &self.indices,
                &self.scores,
                &self.gate_y,
                &self.down_y,
                &self.out,
            ),
        ]
    }

    /// The routed chain: the gated kernel routes its token itself and stores its picks, then
    /// the combine reads them.
    fn routed(&self) -> Vec<Dispatch<'_>> {
        vec![
            self.gated(
                &self.routed_gated,
                &self.routed_indices,
                &self.routed_gate_y,
                Some((&self.logits, &self.routed_scores, &self.router_bias)),
            ),
            self.combine(
                &self.routed_indices,
                &self.routed_scores,
                &self.routed_gate_y,
                &self.routed_down_y,
                &self.routed_out,
            ),
        ]
    }
}

/// The routed kernel's picks, scores and outputs must be the routing command's and the
/// unrouted chain's, bit for bit.
#[test]
fn routed_gpt_oss_moe_matches_the_routing_command_then_the_unrouted_chain() {
    let Some(d) = detect_device() else { return };
    let device = d.device;
    let block = Block::new(&device, 2);
    run(&device, block.unrouted());
    run(&device, block.routed());
    let pairs = 2 * TOP_K;
    let picked = common::read_slice::<u32>(&block.indices, pairs);
    let mut distinct = picked.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(
        distinct.len(),
        pairs,
        "the routing command's picks {picked:?}"
    );
    assert_eq!(
        picked,
        common::read_slice::<u32>(&block.routed_indices, pairs),
        "the stored picks differ"
    );
    assert_eq!(
        bits(&block.scores, pairs),
        bits(&block.routed_scores, pairs),
        "the stored scores differ"
    );
    assert_eq!(
        bits(&block.gate_y, pairs * INTER),
        bits(&block.routed_gate_y, pairs * INTER),
        "act(gate) * up differs"
    );
    let out = bits(&block.out, 2 * HIDDEN);
    assert!(
        out.iter().any(|&v| v != 0),
        "the combined rows computed nothing"
    );
    assert_eq!(
        out,
        bits(&block.routed_out, 2 * HIDDEN),
        "the combined rows differ"
    );
}

/// The whole block against the CPU reference: the b2 matvecs (`affine_qmv_b2`), the linear
/// biases added to the rounded matvec outputs, `swiglu_oai`, the down matvec, its bias, and the
/// f32 weighted combine in slot order. The matvec's simdgroup accumulation order differs from
/// the reference's sequential one, so the comparison carries the gather tests' tolerance.
#[test]
fn gpt_oss_linear_biases_match_the_reference() {
    let Some(d) = detect_device() else { return };
    let device = d.device;
    let tokens = 2;
    let block = Block::new(&device, tokens);
    run(&device, block.unrouted());
    let pairs = tokens * TOP_K;
    let indices = common::read_slice::<u32>(&block.indices, pairs);
    let scores = common::read_slice::<bf16>(&block.scores, pairs);
    // The gated kernel's rows: expert e's gate and up matvecs, the linear biases, the act.
    fn slab(proj: &Experts, e: usize) -> (&[u8], &[bf16], &[bf16]) {
        let (w, sb) = (proj.n_out * proj.k_in / 4, proj.n_out * proj.k_in / GS);
        (
            &proj.codes[e * w..(e + 1) * w],
            &proj.scales[e * sb..(e + 1) * sb],
            &proj.biases[e * sb..(e + 1) * sb],
        )
    }
    let mut act: Vec<bf16> = Vec::with_capacity(pairs * INTER);
    for (p, &e) in indices.iter().enumerate() {
        let (n, e) = (p / TOP_K, e as usize);
        let matvec = |proj: &Experts| {
            let (w, s, b) = slab(proj, e);
            affine_qmv_b2_bf16_s_bf16(
                w,
                s,
                b,
                &block.x_host[n * HIDDEN..(n + 1) * HIDDEN],
                1,
                proj.n_out,
                proj.k_in,
                GS,
            )
        };
        let (g_raw, u_raw) = (matvec(&block.gate), matvec(&block.up));
        let bias = &block.gate_up_bias.host[e * 2 * INTER..(e + 1) * 2 * INTER];
        for j in 0..INTER {
            let g = g_raw[j].to_f32() + bias[j].to_f32();
            let u = u_raw[j].to_f32() + bias[INTER + j].to_f32();
            act.push(bf16::from_f32(swiglu_oai(g, u)));
        }
    }
    let got_act = bits(&block.gate_y, pairs * INTER);
    let mut max_act_err = 0.0f32;
    for p in 0..pairs {
        for j in 0..INTER {
            let (g, w) = (
                bf16::from_bits(got_act[p * INTER + j]).to_f32(),
                act[p * INTER + j].to_f32(),
            );
            let err = (g - w).abs();
            max_act_err = max_act_err.max(err);
            assert!(
                err < 0.3 || err / w.abs().max(1e-3) < 0.2,
                "act pair {p} (expert {}) col {j}: got {g} want {w} err {err}",
                indices[p],
            );
        }
    }
    eprintln!("gpt-oss act max_err={max_act_err:.3e}");
    // The combine: each pair's down matvec, its bias, the f32 weighted sum in slot order.
    let mut max_out_err = 0.0f32;
    let got_out = bits(&block.out, tokens * HIDDEN);
    for n in 0..tokens {
        for col in 0..HIDDEN {
            let mut acc = 0.0f32;
            for k in 0..TOP_K {
                let p = n * TOP_K + k;
                let e = indices[p] as usize;
                let (w, s, b) = slab(&block.down, e);
                let row = affine_qmv_b2_bf16_s_bf16(
                    w,
                    s,
                    b,
                    &act[p * INTER..(p + 1) * INTER],
                    1,
                    HIDDEN,
                    INTER,
                    GS,
                );
                let r = row[col].to_f32() + block.down_bias.host[e * HIDDEN + col].to_f32();
                acc = f32::mul_add(r, scores[p].to_f32(), acc);
            }
            let (g, w) = (bf16::from_bits(got_out[n * HIDDEN + col]).to_f32(), acc);
            let err = (g - w).abs();
            max_out_err = max_out_err.max(err);
            assert!(
                err < 0.3 || err / w.abs().max(1e-3) < 0.2,
                "out token {n} col {col}: got {g} want {w} err {err}",
            );
        }
    }
    eprintln!("gpt-oss combined max_err={max_out_err:.3e}");
}
