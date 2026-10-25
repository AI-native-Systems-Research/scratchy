// SPDX-License-Identifier: Apache-2.0
//! Gemma 4 26B-A4B's emitted CUDA tapes, checked with no toolchain (AI-native-Systems-Research/
//! scratchy#885, part 13a: lowering and expansion; the kernels are 13b and 13c).
//!
//! `#[forward]` hands the model (arch `gemma4-moe`, bf16) to CUDA's compiler through the
//! target-emitter seam, and its module holds what came back: `CUDA_TAPES`, `CUDA_UNLOWERED` and
//! `CUDA_WEIGHTS`. The checks, each one of #885's predictions:
//!
//! - every emitted tape is, field by field, what CUDA's compiler lowers the same canonicals to
//!   (`lower_model`), each validates, and the emitted `CUDA_WEIGHTS` is its weight table;
//! - the buckets are [1, 512, 1024, 2048, 4096], and [2, 4, 8, 16, 32, 64] are left out;
//! - per bucket, the steps of #885's table: one embed scale and one soft-cap; per layer a unit
//!   norm, two GeGLUs (dense and expert), a layer scale, the router norm, logits and routing; the
//!   experts gathered (GEMV) at m = 1 and sorted (sort, grouped GEMM) above; the dense gate and up
//!   on the 64-column GEMM above one row;
//! - two KV classes (sliding: 25 layers of 8 × 256 with a window of 1024; global: layers 5, 11,
//!   17, 23 and 29 of 2 × 512) and two rotary tables;
//! - the MoE scratch and the arena per bucket, printed;
//! - the weight table: 657 checkpoint tensors, the tied lm_head at the embedding table's index,
//!   50,466,283,580 device bytes;
//! - the model is in the CUDA registry once, under its stem.
//!
//! Two differences from #885's table, both of representation or size, not of what runs:
//! - the router's 128-column GEMM at m ≥ 512 is the `RouterLogits` step on the 64-column kernel
//!   (`MatmulKernel::Col64`), not a `GemmCol64` step, so `GemmCol64` counts the 60 dense gate and
//!   up GEMMs only;
//! - the m = 4096 arena is 5,660,823,040 bytes, not about 3.3 GB: the soft-cap writes a colour of
//!   its own (`op_abi`'s fresh rule), so the lm_head's logits and the capped logits are two
//!   `[4096, 262144]` bf16 buffers, 2,147,483,648 bytes each.

#[path = "../../../targets/cuda/tests/common/chains.rs"]
pub mod chains;

use std::collections::BTreeMap;
use std::path::PathBuf;

use scratchy_models::gemma4_moe::gemma_4_26b_a4b_it::{CUDA_TAPES, CUDA_UNLOWERED, CUDA_WEIGHTS};
use scratchy_subtile::handoff::TargetModelInput;
use scratchy_target_cuda::step::{
    AttnPrefillBf16, CudaStep, GemmBf16, MatmulKernel, MoeWeight, RopePairing, RouteProgram,
};
use scratchy_target_cuda::tape::{ArenaSpan, CudaTape, KvClass};
use scratchy_target_cuda::weights::CudaWeight;
use scratchy_target_cuda_compiler::emit::lower_model;

/// The model the seam hands over, and the Cargo feature that selects it.
const STEM: &str = "gemma-4-26b-a4b-it";
const FEATURE: &str = "CARGO_FEATURE_GEMMA_4_26B_A4B_IT";

/// Every lowered bucket, ascending: decode's one row, then the prefill buckets.
const TAPE_BUCKETS: [u32; 5] = [1, 512, 1024, 2048, 4096];
/// The DSL's workload rungs between one row and the GEMM's 128-row tile.
const LEFT_OUT: [u32; 6] = [2, 4, 8, 16, 32, 64];
const VOCAB: u32 = 262_144;
const HIDDEN: u32 = 2816;
const LAYERS: u32 = 30;
/// The global (full-attention) layers; every other layer is a sliding one.
const GLOBAL_LAYERS: [u32; 5] = [5, 11, 17, 23, 29];
const QUERY_HEADS: u32 = 16;
/// The two attention classes: sliding (8 kv heads of 256, a window of 1024), global (2 of 512).
const SLIDING: KvClass = KvClass {
    kv_heads: 8,
    head_dim: 256,
    window: Some(1024),
};
const GLOBAL: KvClass = KvClass {
    kv_heads: 2,
    head_dim: 512,
    window: None,
};
/// The MoE block: 128 experts, top-8, an expert width of 704; the dense MLP's width.
const EXPERTS: u32 = 128;
const TOP_K: u32 = 8;
const INTER: u32 = 704;
const DENSE_INTER: u32 = 2112;
/// The soft-cap's cap (`final_logit_softcapping`).
const CAP: f32 = 30.0;

/// The steps every bucket shares, whatever its kernels.
const COMMON_STEPS: [(&str, usize); 16] = [
    ("Add", 90),
    ("Embed", 1),
    ("ExpertCombine", 30),
    ("GeluMul", 60),
    ("LayerScale", 30),
    ("MoeRoute", 30),
    ("RmsNorm", 271),
    ("RmsNormUnit", 30),
    ("Rope", 30),
    ("RopeAppend", 30),
    ("RouterLogits", 30),
    ("RouterNorm", 30),
    ("Scale", 1),
    ("SoftCap", 1),
    ("AttnSteps", 30),
    ("ExpertSteps", 90),
];
/// The decode tape's own steps: its GEMMs are GEMVs, its experts gathered.
const DECODE_STEPS: [(&str, usize); 3] = [("AttnDecode", 30), ("ExpertGemv", 90), ("Gemv", 206)];
/// Every prefill tape's own steps: its experts sorted, then the grouped GEMM; the dense gate and
/// up on the 64-column GEMM.
const PREFILL_STEPS: [(&str, usize); 5] = [
    ("AttnPrefill", 30),
    ("ExpertGemm", 90),
    ("ExpertSort", 30),
    ("Gemm", 146),
    ("GemmCol64", 60),
];
const DECODE_STEP_COUNT: usize = 990;
const PREFILL_STEP_COUNT: usize = 1020;
const COLOURS: usize = 24;

/// Per bucket: the arena's `total_bytes` (with the token ids), the MoE scratch's offset and bytes
/// (from `logits` to the end of the router-norm rows), and its max tiles.
const ARENAS: [(u32, u64, u64, u64, u32); 5] = [
    (1, 1_277_696, 1_192_192, 85_248, 0),
    (512, 927_906_560, 609_618_688, 318_285_824, 159),
    (1024, 1_604_037_376, 1_219_236_608, 384_796_672, 191),
    (2048, 2_956_299_264, 2_438_472_448, 517_818_624, 255),
    (4096, 5_660_823_040, 4_876_944_128, 783_862_528, 383),
];

/// The weight table: 627 entries naming the checkpoint's 657 text tensors (each expert bank names
/// two), in a table of 756 indices, 50,466,283,580 device bytes; 30 expert banks.
const WEIGHTS: usize = 627;
const TABLE_LEN: usize = 756;
const BANKS: usize = 30;
const TENSORS: usize = 657;
const WEIGHT_BYTES: u64 = 50_466_283_580;
/// Layer 0's expert bank: `gate_up [128, 1408, 2816]` then `down [128, 2816, 704]`, bf16.
const LAYER_0_BANK: CudaWeight = CudaWeight::ExpertBank {
    gate_up: "language_model.model.layers.0.experts.gate_up_proj",
    down: "language_model.model.layers.0.experts.down_proj",
    experts: EXPERTS,
    hidden: HIDDEN,
    inter: INTER,
};
const BANK_BYTES: u64 = 128 * (1408 + 704) * 2816 * 2;
const SHAPES: [(&str, &[u64]); 8] = [
    (
        "language_model.model.layers.0.router.proj.weight",
        &[128, 2816],
    ),
    ("language_model.model.layers.0.router.scale", &[2816]),
    (
        "language_model.model.layers.0.router.per_expert_scale",
        &[128],
    ),
    ("language_model.model.layers.0.layer_scalar", &[1]),
    (
        "language_model.model.layers.0.mlp.gate_proj.weight",
        &[2112, 2816],
    ),
    (
        "language_model.model.layers.0.self_attn.v_proj.weight",
        &[2048, 2816],
    ),
    (
        "language_model.model.layers.5.self_attn.k_proj.weight",
        &[1024, 2816],
    ),
    (
        "language_model.model.layers.5.self_attn.q_norm.weight",
        &[512],
    ),
];

/// The model's canonicals, as `#[forward]` handed them to the seam: selected, as the build script
/// that expanded `CUDA_TAPES` did, by `CARGO_FEATURE_<STEM>` and an empty `SCRATCHY_QUANTS`, set
/// once here, before the pipeline reads them; no other test in this binary reads the environment.
fn input() -> TargetModelInput {
    static INPUT: std::sync::OnceLock<TargetModelInput> = std::sync::OnceLock::new();
    INPUT.get_or_init(capture).clone()
}

fn capture() -> TargetModelInput {
    // SAFETY: set before the pipeline's own threads start, and no other thread of this binary
    // reads or writes the environment.
    unsafe {
        std::env::set_var(FEATURE, "1");
        std::env::set_var("SCRATCHY_QUANTS", "");
    }
    let models = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dsl = std::fs::read_to_string(models.join("dsl/gemma4-moe.py")).expect("read the DSL");
    let carrier = scratchy_forward_compiler_macro::parse_python_file(&dsl).expect("parse the DSL");
    let inputs = scratchy_forward_compiler_macro::target_model_inputs(
        carrier,
        &models.join("configs/gemma4-moe"),
    )
    .expect("the forward pipeline");
    let mut inputs: Vec<_> = (inputs.into_iter()).filter(|i| i.stem == STEM).collect();
    assert_eq!(inputs.len(), 1, "one {STEM} model handed to the seam");
    inputs.remove(0)
}

/// The lowered tape of bucket `m`, looked up by its `bucket_m`, never by position.
fn tape_of(m: u32) -> &'static CudaTape {
    (CUDA_TAPES.iter())
        .find(|t| t.bucket_m == m)
        .unwrap_or_else(|| panic!("no m={m} tape"))
}

/// One tape's steps, by kind; plus `AttnSteps` and `ExpertSteps`, every attention and every
/// expert projection whatever its kernel.
fn histogram(tape: &CudaTape) -> BTreeMap<&'static str, usize> {
    let mut h = BTreeMap::new();
    for step in tape.steps {
        let kind = match step {
            CudaStep::Embed(_) => "Embed",
            CudaStep::Gemm(_) => "Gemm",
            CudaStep::GemmNarrow(_) => "GemmNarrow",
            CudaStep::Gemv(_) => "Gemv",
            CudaStep::RmsNorm(_) => "RmsNorm",
            CudaStep::Add(_) => "Add",
            CudaStep::SiluMul(_) => "SiluMul",
            CudaStep::Rope(_) => "Rope",
            CudaStep::GateSplit(_) => "GateSplit",
            CudaStep::GateApply(_) => "GateApply",
            CudaStep::RopeAppend(_) => "RopeAppend",
            CudaStep::AttnPrefill(_) => "AttnPrefill",
            CudaStep::AttnDecode(_) => "AttnDecode",
            CudaStep::GatedDeltaNet(_) => "GatedDeltaNet",
            CudaStep::RouterLogits(_) => "RouterLogits",
            CudaStep::MoeRoute(_) => "MoeRoute",
            CudaStep::ExpertSort(_) => "ExpertSort",
            CudaStep::ExpertGemm(_) => "ExpertGemm",
            CudaStep::ExpertGemv(_) => "ExpertGemv",
            CudaStep::ExpertCombine(_) => "ExpertCombine",
            CudaStep::GateScale(_) => "GateScale",
            CudaStep::Scale(_) => "Scale",
            CudaStep::RmsNormUnit(_) => "RmsNormUnit",
            CudaStep::GeluMul(_) => "GeluMul",
            CudaStep::LayerScale(_) => "LayerScale",
            CudaStep::SoftCap(_) => "SoftCap",
            CudaStep::RouterNorm(_) => "RouterNorm",
            CudaStep::GemmCol64(_) => "GemmCol64",
        };
        *h.entry(kind).or_default() += 1;
        match step {
            CudaStep::AttnPrefill(_) | CudaStep::AttnDecode(_) => {
                *h.entry("AttnSteps").or_default() += 1
            }
            CudaStep::ExpertGemm(_) | CudaStep::ExpertGemv(_) => {
                *h.entry("ExpertSteps").or_default() += 1
            }
            _ => {}
        }
    }
    h
}

const fn end(s: ArenaSpan) -> u64 {
    s.offset.0 + s.bytes
}

#[test]
fn every_emitted_tape_is_what_cuda_lowers_from_the_same_canonical() {
    let input = input();
    assert_eq!(input.arch, "gemma4-moe", "the arch the seam names");
    let model = lower_model(&input).unwrap_or_else(|e| panic!("lower_model: {e}"));
    assert_eq!(model.tapes.len(), CUDA_TAPES.len(), "tapes");
    for (tape, lowered) in CUDA_TAPES.iter().zip(&model.tapes) {
        let name = format!("{STEM} m={}", tape.bucket_m);
        chains::assert_tape_is(&name, tape, lowered);
    }
    assert_eq!(CUDA_WEIGHTS, model.weights.as_slice(), "CUDA_WEIGHTS");
}

#[test]
fn the_tapes_are_the_decode_and_prefill_buckets_and_validate() {
    let ms: Vec<u32> = CUDA_TAPES.iter().map(|t| t.bucket_m).collect();
    assert_eq!(ms, TAPE_BUCKETS, "CUDA_TAPES' buckets");
    for tape in CUDA_TAPES {
        tape.validate()
            .unwrap_or_else(|e| panic!("m={}: {e}", tape.bucket_m));
    }
    let left: Vec<u32> = CUDA_UNLOWERED.iter().map(|&(m, _)| m).collect();
    assert_eq!(left, LEFT_OUT, "CUDA_UNLOWERED's buckets");
    for &(m, why) in CUDA_UNLOWERED {
        let want = format!("m = {m} must be a non-zero multiple of 128, the bf16 GEMM's tile");
        assert!(why.contains(&want), "m={m}: {why}");
    }
}

/// The step histogram of every bucket (#885's table), the step and colour counts, and the
/// 64-column GEMMs: the dense gate and up (n = 2112) above one row, and the router's 128 columns
/// on the 64-column kernel there.
#[test]
fn every_tape_has_the_predicted_steps() {
    for tape in CUDA_TAPES {
        let m = tape.bucket_m;
        let own: &[(&str, usize)] = if m == 1 {
            &DECODE_STEPS
        } else {
            &PREFILL_STEPS
        };
        let want: BTreeMap<&str, usize> = (COMMON_STEPS.iter().chain(own).copied()).collect();
        assert_eq!(histogram(tape), want, "m={m}: step histogram");
        let count = if m == 1 {
            DECODE_STEP_COUNT
        } else {
            PREFILL_STEP_COUNT
        };
        assert_eq!(tape.steps.len(), count, "m={m}: steps");
        assert_eq!(tape.arena.colour_bytes.len(), COLOURS, "m={m}: colours");
        for step in tape.steps {
            match *step {
                CudaStep::GemmCol64(GemmBf16 { n, k, .. }) => {
                    assert_eq!((n, k), (DENSE_INTER, HIDDEN), "m={m}: a dense gate or up")
                }
                CudaStep::RouterLogits(r) => {
                    let kernel = match m {
                        1 => MatmulKernel::Gemv,
                        _ => MatmulKernel::Col64,
                    };
                    assert_eq!((r.kernel, r.weight), (kernel, MoeWeight::Own), "m={m}");
                    assert_eq!(
                        r.x,
                        tape.moe
                            .expect("a scratch")
                            .router_norm
                            .expect("rows")
                            .offset
                    );
                }
                CudaStep::MoeRoute(r) => {
                    let scale = (HIDDEN as f32).powf(-0.5).to_bits();
                    assert_eq!(
                        r.program,
                        RouteProgram::GemmaScaledSoftmax { scale_bits: scale },
                        "m={m}"
                    );
                    assert!(r.expert_scale.is_some() && !r.renorm, "m={m}");
                }
                CudaStep::ExpertGemm(g) => {
                    let tile = match g.n_k().0 {
                        INTER => 64,
                        _ => 256,
                    };
                    assert_eq!((g.weight, g.n_tile), (MoeWeight::Own, tile), "m={m}");
                }
                CudaStep::SoftCap(s) => assert_eq!(
                    (s.rows, s.cols, f32::from_bits(s.cap_bits)),
                    (m, VOCAB, CAP),
                    "m={m}: the soft-cap"
                ),
                CudaStep::Scale(s) => assert_eq!(
                    f32::from_bits(s.scale_bits),
                    (HIDDEN as f32).sqrt(),
                    "m={m}: the embedding scale"
                ),
                _ => {}
            }
        }
    }
}

/// Two KV classes, each layer's class, every attention step at its layer's class (25 windowed at
/// 8 × 256, 5 at 2 × 512, a GQA group of 2 and 8), and two rotary tables: the global one 128 of
/// 512 lanes at theta 1e6 paired at half the head, the local one 256 lanes at theta 1e4.
#[test]
fn the_attention_classes_and_rotary_tables_are_the_models() {
    for tape in CUDA_TAPES {
        let m = tape.bucket_m;
        let kv = tape.kv.expect("a KV cache");
        let layers: Vec<u32> = (0..LAYERS).collect();
        assert_eq!(kv.layers, layers.as_slice(), "m={m}: KV layers");
        assert_eq!(kv.classes, [SLIDING, GLOBAL], "m={m}: the classes");
        let want: Vec<u8> = (0..LAYERS)
            .map(|l| u8::from(GLOBAL_LAYERS.contains(&l)))
            .collect();
        assert_eq!(kv.layer_class, want.as_slice(), "m={m}: each layer's class");
        assert_eq!(
            (kv.kv_heads, kv.head_dim),
            (8, 256),
            "m={m}: the pool's geometry"
        );
        let mut by_class: BTreeMap<u8, usize> = BTreeMap::new();
        for (i, step) in tape.steps.iter().enumerate() {
            let (CudaStep::AttnDecode(a) | CudaStep::AttnPrefill(a)) = *step else {
                continue;
            };
            let AttnPrefillBf16 {
                layer,
                q_heads,
                kv_heads,
                head_dim,
                window,
                scale_bits,
                ..
            } = a;
            let class = kv.layer_class[layer as usize];
            let c = kv.classes[usize::from(class)];
            assert_eq!(
                (q_heads, kv_heads, head_dim, window),
                (QUERY_HEADS, c.kv_heads, c.head_dim, c.window),
                "m={m}: step {i}"
            );
            assert_eq!(f32::from_bits(scale_bits), 1.0, "m={m}: Gemma's scale is 1");
            *by_class.entry(class).or_default() += 1;
        }
        assert_eq!(by_class, [(0, 25), (1, 5)].into_iter().collect(), "m={m}");
        let global = tape.rotary.expect("the global table");
        let local = tape.rotary_local.expect("the local table");
        assert_eq!(
            (global.rot_dim, global.theta()),
            (128, 1e6),
            "m={m}: global"
        );
        assert_eq!((local.rot_dim, local.theta()), (256, 1e4), "m={m}: local");
        for step in tape.steps {
            let (table, head_dim, rot_dim, pairing) = match *step {
                CudaStep::Rope(r) => (r.table, r.head_dim, r.rot_dim, r.pairing),
                CudaStep::RopeAppend(r) => (r.table, r.head_dim, r.rot_dim, r.pairing),
                _ => continue,
            };
            let want = match head_dim {
                512 => (global.span.offset, 128, RopePairing::HalfHead),
                _ => (local.span.offset, 256, RopePairing::HalfRotated),
            };
            assert_eq!(
                (table, rot_dim, pairing),
                want,
                "m={m}: head_dim {head_dim}"
            );
        }
    }
}

/// The MoE scratch and the arena of every bucket, printed: the geometry, the regions in order,
/// each 256-aligned and inside the arena, the sort's regions exactly at the sorted buckets, the
/// router-norm rows last at `[m, 2816]` bf16, and the totals.
#[test]
fn the_moe_scratch_and_arena_are_the_measured_layout() {
    for &(m, total, at, bytes, max_tiles) in &ARENAS {
        let tape = tape_of(m);
        let moe = tape.moe.expect("a MoE scratch");
        eprintln!(
            "{STEM} m={m}: arena {} bytes, MoE scratch {bytes} bytes at {at}, {max_tiles} max \
             tiles, colours {:?}",
            tape.arena.total_bytes, tape.arena.colour_bytes
        );
        assert_eq!(tape.arena.total_bytes, total, "m={m}: arena total_bytes");
        assert_eq!(
            (moe.experts, moe.top_k, moe.hidden, moe.inter, moe.tile_rows),
            (EXPERTS, TOP_K, HIDDEN, INTER, 128),
            "m={m}: MoE geometry"
        );
        assert_eq!(moe.max_tiles, max_tiles, "m={m}: max tiles");
        assert_eq!(
            moe.sort.is_some(),
            m > 1,
            "m={m}: sorted exactly above one row"
        );
        let norm = moe.router_norm.expect("the router-norm rows");
        assert_eq!(
            norm.bytes,
            u64::from(m) * u64::from(HIDDEN) * 2,
            "m={m}: router norm"
        );
        let mut regions = vec![moe.logits, moe.topk_idx, moe.scores];
        if let Some(s) = moe.sort {
            regions.extend([
                s.chunk_hist,
                s.counts,
                s.offsets,
                s.tile_expert,
                s.pos,
                s.rows,
            ]);
        }
        regions.extend([moe.gate, moe.up, moe.act, moe.down, norm]);
        for (i, r) in regions.iter().enumerate() {
            assert_eq!(r.offset.0 % 256, 0, "m={m}: region {i} is 256-aligned");
            assert!(end(*r) <= total, "m={m}: region {i} is in the arena");
        }
        for w in regions.windows(2) {
            assert!(end(w[0]) <= w[1].offset.0, "m={m}: regions in order, apart");
        }
        let span = moe.span();
        assert_eq!(
            (span.offset.0, span.bytes),
            (at, bytes),
            "m={m}: the MoE scratch"
        );
        let rows = match m {
            1 => u64::from(TOP_K),
            _ => 128 * u64::from(max_tiles),
        };
        assert_eq!(moe.down.bytes, rows * u64::from(HIDDEN) * 2, "m={m}: down");
        assert_eq!(moe.act.bytes, rows * u64::from(INTER) * 2, "m={m}: act");
    }
    // #885's m = 4096 numbers: 383 tiles (R = 49,024 rows), the down rows 276,103,168 bytes, and
    // the logits (twice: the lm_head's and the soft-cap's) 2,147,483,648 bytes each.
    let tape = tape_of(4096);
    let moe = tape.moe.expect("a MoE scratch");
    assert_eq!((moe.max_tiles, moe.down.bytes), (383, 276_103_168));
    let logits = (tape.arena.colour_bytes.iter()).filter(|&&b| b == 2_147_483_648);
    assert_eq!(logits.count(), 2, "the lm_head's and the soft-cap's logits");
}

/// The weight table: 627 entries, 30 expert banks, 657 distinct checkpoint tensors none named
/// twice, 50,466,283,580 device bytes; the router's three tensors and the layer scalar; layer 0's
/// bank; and the tied lm_head reading the embedding table's index, the last GEMM of every tape.
#[test]
fn the_weight_table_names_the_checkpoints_tensors() {
    assert_eq!(CUDA_WEIGHTS.len(), TABLE_LEN, "the table's indices");
    let entries: Vec<(usize, CudaWeight)> = (CUDA_WEIGHTS.iter().enumerate())
        .filter_map(|(ix, w)| w.map(|w| (ix, w)))
        .collect();
    assert_eq!(entries.len(), WEIGHTS, "Some entries");
    let banks = (entries.iter()).filter(|(_, w)| matches!(w, CudaWeight::ExpertBank { .. }));
    assert_eq!(banks.count(), BANKS, "expert banks");
    let mut names = std::collections::BTreeSet::new();
    for (ix, w) in &entries {
        let tensors = match *w {
            CudaWeight::Bf16 { tensor, .. } => vec![tensor],
            CudaWeight::ExpertBank { gate_up, down, .. } => vec![gate_up, down],
            CudaWeight::GdnBundle { .. } | CudaWeight::MoeBundle { .. } => {
                panic!("weight {ix}: Gemma has no Qwen bundle")
            }
        };
        for t in tensors {
            assert!(names.insert(t), "weight {ix}: {t} named twice");
        }
        for tape in CUDA_TAPES {
            assert_eq!(
                Some(tape.weight_bytes[*ix]),
                w.packed_bytes(),
                "m={}: weight {ix}'s declared bytes",
                tape.bucket_m
            );
        }
    }
    assert_eq!(names.len(), TENSORS, "distinct tensors");
    let bytes: u64 = (entries.iter())
        .map(|(_, w)| w.packed_bytes().expect("fits"))
        .sum();
    assert_eq!(bytes, WEIGHT_BYTES, "the table's bytes");
    assert!(
        entries.iter().any(|(_, w)| *w == LAYER_0_BANK),
        "layer 0's expert bank"
    );
    assert_eq!(
        LAYER_0_BANK.packed_bytes(),
        Some(BANK_BYTES),
        "a bank's bytes"
    );
    let by_name = |name: &str| {
        (entries.iter())
            .find(|(_, w)| matches!(w, CudaWeight::Bf16 { tensor, .. } if *tensor == name))
            .copied()
            .unwrap_or_else(|| panic!("no entry for {name}"))
    };
    let shape = |w: CudaWeight| match w {
        CudaWeight::Bf16 { shape, .. } => shape,
        _ => panic!("not one tensor"),
    };
    let (e, embed) = by_name("language_model.model.embed_tokens.weight");
    assert_eq!(e, 0, "the embedding table is the embedded hidden's index");
    assert_eq!(
        shape(embed),
        [u64::from(VOCAB), u64::from(HIDDEN)],
        "the embedding table"
    );
    for (name, want) in SHAPES {
        assert_eq!(shape(by_name(name).1), want, "{name}");
    }
    for tape in CUDA_TAPES {
        let m = tape.bucket_m;
        let head = (tape.steps.iter().rev()).find_map(|s| match *s {
            CudaStep::Gemm(g) | CudaStep::Gemv(g) | CudaStep::GemmCol64(g) => Some(g),
            _ => None,
        });
        let Some(GemmBf16 { b, n, k, .. }) = head else {
            panic!("m={m}: no GEMM");
        };
        assert_eq!((b.0, n, k), (0, VOCAB, HIDDEN), "m={m}: the tied lm_head");
        assert!(
            matches!(tape.steps.last(), Some(CudaStep::SoftCap(_))),
            "m={m}: the tape ends with the soft-cap"
        );
    }
}

/// The model is in the CUDA registry exactly once, under its stem, holding the emitted tapes and
/// weight table themselves.
#[test]
fn the_model_is_registered_under_its_stem() {
    let entries: Vec<_> = scratchy_target_cuda::registry::entries().collect();
    let stems: Vec<&str> = entries.iter().map(|e| e.stem).collect();
    assert_eq!(stems, [STEM], "the registry");
    assert!(
        std::ptr::eq(entries[0].tapes, CUDA_TAPES),
        "the entry's tapes"
    );
    assert!(
        std::ptr::eq(entries[0].weights, CUDA_WEIGHTS),
        "the entry's weight table"
    );
}
