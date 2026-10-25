// SPDX-License-Identifier: Apache-2.0
//! Qwen3.6-35B-A3B's emitted CUDA tapes, checked with no toolchain (AI-native-Systems-Research/
//! scratchy#841, scope (a): lowering and expansion; the MoE kernels are #857).
//!
//! `#[forward]` hands the model (arch `qwen3-5-moe`) to CUDA's compiler through the target-emitter
//! seam, and its module holds what came back: `CUDA_TAPES`, `CUDA_UNLOWERED` and `CUDA_WEIGHTS`.
//! The checks, each a prediction of #841's "Whole-model expansion" section:
//!
//! - every emitted tape is, field by field, what CUDA's compiler lowers the same canonicals to
//!   (`lower_model`), each validates, and the emitted `CUDA_WEIGHTS` is its weight table;
//! - the buckets are [1, 512, 1024, 2048, 4096], and [2, 4, 8, 64] are left out, as for qwen3.5;
//! - per bucket, the MoE block's steps: 40 each of RouterLogits, MoeRoute, ExpertCombine and
//!   GateScale; ExpertSort ×40 and ExpertGemm ×120 at m ≥ 512, or ExpertGemv ×120 and no sort at
//!   m = 1; SiluMul ×80 (routed and shared); GemmNarrow ×100 (40 `shared_expert_gate`s at n = 1,
//!   60 GDN a/b projections at n = 32);
//! - 10 KV layers at 16 q / 2 kv heads (a GQA group of 8), 30 GDN layers at 16/32 heads × 128;
//! - the MoE scratch per bucket, whose totals are #841's table's (58,368 / 412,683,776 /
//!   459,148,800 / 552,079,104 / 737,939,712 bytes), every region 256-aligned and in the arena;
//! - the weight table: 523 entries, 30 GDN bundles and 40 MoE bundles, 693 distinct checkpoint
//!   tensors, 69,323,206,656 device bytes, and the untied `lm_head.weight` at its own index;
//! - the model is in the CUDA registry once, under its stem.
//!
//! The arenas and MoE scratch are printed per bucket. The qwen3.5 tests' shared body
//! (`common/cuda_tapes_body.rs`) reads the `qwen3-5` DSL and config directory, so this binary is its
//! own.

#[path = "../../../targets/cuda/tests/common/chains.rs"]
pub mod chains;

use std::collections::BTreeMap;
use std::path::PathBuf;

use scratchy_models::qwen3_5_moe::qwen3_6_35b_a3b::{CUDA_TAPES, CUDA_UNLOWERED, CUDA_WEIGHTS};
use scratchy_subtile::handoff::TargetModelInput;
use scratchy_target_cuda::step::{AttnPrefillBf16, CudaStep, GemmBf16};
use scratchy_target_cuda::tape::{ArenaSpan, CudaTape};
use scratchy_target_cuda::weights::CudaWeight;
use scratchy_target_cuda_compiler::emit::lower_model;

/// The model the seam hands over, and the Cargo feature that selects it.
const STEM: &str = "qwen3.6-35b-a3b";
const FEATURE: &str = "CARGO_FEATURE_QWEN3.6_35B_A3B";

/// Every lowered bucket, ascending: decode's one row, then the prefill buckets.
const TAPE_BUCKETS: [u32; 5] = [1, 512, 1024, 2048, 4096];
const VOCAB: u32 = 248_320;
const HIDDEN: u32 = 2048;
/// Full attention's query and KV heads.
const HEADS: (u32, u32) = (16, 2);
/// GDN's key and value heads, and their dims.
const GDN_HEADS: (u32, u32) = (16, 32);
const LAYERS: u32 = 40;
const KV_LAYERS: &[u32] = &[3, 7, 11, 15, 19, 23, 27, 31, 35, 39];
/// The MoE block's geometry: experts, top-k, the expert's and the shared expert's intermediate.
const EXPERTS: u32 = 256;
const TOP_K: u32 = 8;
const INTER: u32 = 512;

/// The steps every bucket shares, whatever its kernels.
const COMMON_STEPS: [(&str, usize); 14] = [
    ("Add", 80),
    ("Embed", 1),
    ("ExpertCombine", 40),
    ("GateApply", 10),
    ("GateScale", 40),
    ("GateSplit", 10),
    ("GatedDeltaNet", 30),
    ("GemmNarrow", 100),
    ("MoeRoute", 40),
    ("RmsNorm", 101),
    ("Rope", 10),
    ("RopeAppend", 10),
    ("RouterLogits", 40),
    ("SiluMul", 80),
];
/// The decode tape's own steps: its GEMMs are GEMVs, its experts gathered.
const DECODE_STEPS: [(&str, usize); 3] = [("AttnDecode", 10), ("ExpertGemv", 120), ("Gemv", 251)];
/// Every prefill tape's own steps: its experts sorted, then the grouped GEMM.
const PREFILL_STEPS: [(&str, usize); 4] = [
    ("AttnPrefill", 10),
    ("ExpertGemm", 120),
    ("ExpertSort", 40),
    ("Gemm", 251),
];
const DECODE_STEP_COUNT: usize = 973;
const PREFILL_STEP_COUNT: usize = 1013;
const COLOURS: usize = 29;

/// Per bucket: the arena's `total_bytes` (with the token ids), the MoE scratch's offset and bytes
/// (from `logits` to the end of `down`), and its max tiles.
const ARENAS: [(u32, u64, u64, u64, u32); 5] = [
    (1, 763_904, 705_280, 58_368, 0),
    (512, 772_613_632, 359_927_808, 412_683_776, 286),
    (1024, 1_179_007_488, 719_854_592, 459_148_800, 318),
    (2048, 1_991_795_456, 1_439_708_160, 552_079_104, 382),
    (4096, 3_617_371_392, 2_879_415_296, 737_939_712, 510),
];

/// The weight table: 523 entries from the checkpoint's 693 decoder tensors, in a table of 547
/// indices.
const WEIGHTS: usize = 523;
const TABLE_LEN: usize = 547;
const GDN_BUNDLES: usize = 30;
const MOE_BUNDLES: usize = 40;
const TENSORS: usize = 693;
const WEIGHT_BYTES: u64 = 69_323_206_656;
/// One MoE bundle: the router `[256, 2048]`, gate_up `[256, 1024, 2048]` and down `[256, 2048,
/// 512]` bf16, back to back.
const MOE_BUNDLE_BYTES: u64 = 1_611_661_312;
const LAYER_0_MOE: CudaWeight = CudaWeight::MoeBundle {
    router: "language_model.model.layers.0.mlp.gate.weight",
    gate_up: "language_model.model.layers.0.mlp.experts.gate_up_proj",
    down: "language_model.model.layers.0.mlp.experts.down_proj",
    experts: EXPERTS,
    hidden: HIDDEN,
    inter: INTER,
};
const SHAPES: [(&str, &[u64]); 4] = [
    (
        "language_model.model.layers.0.mlp.shared_expert_gate.weight",
        &[1, 2048],
    ),
    (
        "language_model.model.layers.0.mlp.shared_expert.down_proj.weight",
        &[2048, 512],
    ),
    (
        "language_model.model.layers.3.self_attn.k_proj.weight",
        &[512, 2048],
    ),
    (
        "language_model.model.layers.0.linear_attn.in_proj_a.weight",
        &[32, 2048],
    ),
];
/// `tie_word_embeddings` is false: the lm_head is its own tensor, at the table's last index.
const LM_HEAD: (u32, &str) = (546, "lm_head.weight");

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
    let dsl = std::fs::read_to_string(models.join("dsl/qwen3-5-moe.py")).expect("read the DSL");
    let carrier = scratchy_forward_compiler_macro::parse_python_file(&dsl).expect("parse the DSL");
    let inputs = scratchy_forward_compiler_macro::target_model_inputs(
        carrier,
        &models.join("configs/qwen3-5-moe"),
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

/// One tape's steps, by kind.
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
    }
    h
}

const fn end(s: ArenaSpan) -> u64 {
    s.offset.0 + s.bytes
}

#[test]
fn every_emitted_tape_is_what_cuda_lowers_from_the_same_canonical() {
    let input = input();
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
    assert_eq!(left, [2, 4, 8, 64], "CUDA_UNLOWERED's buckets");
    for &(m, why) in CUDA_UNLOWERED {
        let want = format!("m = {m} must be a non-zero multiple of 128, the bf16 GEMM's tile");
        assert!(why.contains(&want), "m={m}: {why}");
    }
}

/// The step histogram of every bucket: the MoE block's steps 40 per layer (the router, routing,
/// combine and gate scale), the experts gathered (GEMV) at m = 1 and sorted (sort, grouped GEMM)
/// at every prefill bucket, SiluMul for both the routed and the shared experts, and the narrow
/// GEMM for the shared expert's gate at n = 1 and the GDN a/b projections at n = 32.
#[test]
fn every_tape_has_the_moe_blocks_steps() {
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
        let mut narrow: BTreeMap<u32, usize> = BTreeMap::new();
        for step in tape.steps {
            if let CudaStep::GemmNarrow(GemmBf16 { n, .. }) = *step {
                *narrow.entry(n).or_default() += 1;
            }
        }
        assert_eq!(
            narrow,
            [(1, 40), (32, 60)].into_iter().collect(),
            "m={m}: the narrow GEMMs by n"
        );
    }
}

/// 10 KV layers at 16 q / 2 kv heads of 256 (a GQA group of 8), every attention step at that
/// geometry, and 30 GDN layers at 16 key / 32 value heads of 128.
#[test]
fn the_attention_and_gdn_geometry_is_the_models() {
    let gdn_layers: Vec<u32> = (0..LAYERS).filter(|l| !KV_LAYERS.contains(l)).collect();
    for tape in CUDA_TAPES {
        let m = tape.bucket_m;
        let kv = tape.kv.expect("a KV cache");
        assert_eq!(kv.layers, KV_LAYERS, "m={m}: KV layers");
        assert_eq!(
            (kv.kv_heads, kv.head_dim),
            (HEADS.1, 256),
            "m={m}: KV geometry"
        );
        for (i, step) in tape.steps.iter().enumerate() {
            if let CudaStep::AttnDecode(AttnPrefillBf16 {
                q_heads,
                kv_heads,
                head_dim,
                ..
            })
            | CudaStep::AttnPrefill(AttnPrefillBf16 {
                q_heads,
                kv_heads,
                head_dim,
                ..
            }) = *step
            {
                assert_eq!(
                    (q_heads, kv_heads, head_dim),
                    (HEADS.0, HEADS.1, 256),
                    "m={m}: step {i}"
                );
            }
        }
        let gdn = tape.gdn.expect("GDN layers");
        assert_eq!(gdn.layers, gdn_layers.as_slice(), "m={m}: GDN layers");
        assert_eq!(
            (gdn.num_k_heads, gdn.num_v_heads, gdn.head_k, gdn.head_v),
            (GDN_HEADS.0, GDN_HEADS.1, 128, 128),
            "m={m}: GDN geometry"
        );
    }
}

/// The MoE scratch of every bucket, printed: the geometry, the regions contiguous from `logits` to
/// the end of `down`, each 256-aligned, inside the arena and after the KV and GDN regions, the sort's
/// regions exactly at the sorted buckets, and #841's predicted totals.
#[test]
fn the_moe_scratch_is_the_predicted_layout() {
    for &(m, total, at, bytes, max_tiles) in &ARENAS {
        let tape = tape_of(m);
        let moe = tape.moe.expect("a MoE scratch");
        eprintln!(
            "{STEM} m={m}: arena {} bytes, MoE scratch {bytes} bytes at {at}, {max_tiles} max \
             tiles: {moe:?}",
            tape.arena.total_bytes
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
        regions.extend([moe.gate, moe.up, moe.act, moe.down]);
        for (i, r) in regions.iter().enumerate() {
            assert_eq!(r.offset.0 % 256, 0, "m={m}: region {i} is 256-aligned");
            assert!(end(*r) <= total, "m={m}: region {i} is in the arena");
        }
        for w in regions.windows(2) {
            assert!(end(w[0]) <= w[1].offset.0, "m={m}: regions in order, apart");
        }
        let first = regions[0].offset.0;
        let last = end(*regions.last().expect("regions"));
        assert_eq!(
            (first, last - first),
            (at, bytes),
            "m={m}: MoE scratch offset and bytes"
        );
        let kv = tape.kv.expect("a KV cache");
        let gdn = tape.gdn.expect("GDN layers");
        assert!(
            end(kv.sequence) <= first && end(gdn.scratch) <= first,
            "m={m}: the MoE scratch follows the KV and GDN regions"
        );
        // The rows the experts compute: every pair gathered at m = 1, 128 · max_tiles sorted.
        let rows = if m == 1 {
            u64::from(TOP_K)
        } else {
            128 * u64::from(max_tiles)
        };
        assert_eq!(moe.down.bytes, rows * u64::from(HIDDEN) * 2, "m={m}: down");
        assert_eq!(moe.act.bytes, rows * u64::from(INTER) * 2, "m={m}: act");
    }
}

/// The weight table: 523 entries, 30 GDN bundles and 40 MoE bundles, 693 distinct checkpoint
/// tensors none named twice, 69,323,206,656 device bytes; the MoE bundle's three tensors and bytes;
/// and the untied lm_head at its own index, read by every tape's last GEMM.
#[test]
fn the_weight_table_names_the_checkpoints_tensors() {
    assert_eq!(CUDA_WEIGHTS.len(), TABLE_LEN, "the table's indices");
    let entries: Vec<(usize, CudaWeight)> = (CUDA_WEIGHTS.iter().enumerate())
        .filter_map(|(ix, w)| w.map(|w| (ix, w)))
        .collect();
    assert_eq!(entries.len(), WEIGHTS, "Some entries");
    let count = |f: fn(&CudaWeight) -> bool| entries.iter().filter(|(_, w)| f(w)).count();
    assert_eq!(
        count(|w| matches!(w, CudaWeight::GdnBundle { .. })),
        GDN_BUNDLES,
        "GDN bundles"
    );
    assert_eq!(
        count(|w| matches!(w, CudaWeight::MoeBundle { .. })),
        MOE_BUNDLES,
        "MoE bundles"
    );
    let mut names = std::collections::BTreeSet::new();
    for (ix, w) in &entries {
        let tensors = match *w {
            CudaWeight::Bf16 { tensor, .. } => vec![tensor],
            CudaWeight::GdnBundle {
                conv1d,
                a_log,
                dt_bias,
                norm,
                ..
            } => vec![conv1d, a_log, dt_bias, norm],
            CudaWeight::MoeBundle {
                router,
                gate_up,
                down,
                ..
            } => vec![router, gate_up, down],
            CudaWeight::ExpertBank { gate_up, down, .. } => vec![gate_up, down],
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
    let moe = (entries.iter())
        .find_map(|(_, w)| match *w {
            b @ CudaWeight::MoeBundle {
                router: "language_model.model.layers.0.mlp.gate.weight",
                ..
            } => Some(b),
            _ => None,
        })
        .expect("layer 0's MoE bundle");
    assert_eq!(moe, LAYER_0_MOE, "layer 0's MoE bundle");
    assert_eq!(
        moe.packed_bytes(),
        Some(MOE_BUNDLE_BYTES),
        "a MoE bundle's bytes"
    );
    let by_name = |name: &str| {
        (entries.iter())
            .find(|(_, w)| matches!(w, CudaWeight::Bf16 { tensor, .. } if *tensor == name))
            .copied()
            .unwrap_or_else(|| panic!("no entry for {name}"))
    };
    let shape = |w: CudaWeight| match w {
        CudaWeight::Bf16 { shape, .. } => shape,
        CudaWeight::GdnBundle { .. } | CudaWeight::MoeBundle { .. } => panic!("a bundle"),
        CudaWeight::ExpertBank { .. } => panic!("an expert bank"),
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
    let (at, lm_head) = by_name(LM_HEAD.1);
    assert_eq!(at, LM_HEAD.0 as usize, "the lm_head's index");
    assert_eq!(
        shape(lm_head),
        [u64::from(VOCAB), u64::from(HIDDEN)],
        "the lm_head"
    );
    for tape in CUDA_TAPES {
        let m = tape.bucket_m;
        let (Some(&CudaStep::Gemm(GemmBf16 { b, n, k, .. }))
        | Some(&CudaStep::Gemv(GemmBf16 { b, n, k, .. }))) = tape.steps.last()
        else {
            panic!("m={m}: the tape ends with the lm_head GEMM");
        };
        assert_eq!(
            (b.0, n, k),
            (LM_HEAD.0, VOCAB, HIDDEN),
            "m={m}: the lm_head"
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
