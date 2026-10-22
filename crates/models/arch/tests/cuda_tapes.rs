// SPDX-License-Identifier: Apache-2.0
//! qwen3.5-0.8b's emitted CUDA tapes, checked with no toolchain.
//!
//! `#[forward]` hands every model to CUDA's compiler through the target-emitter seam, and the
//! model's module holds what came back: `CUDA_TAPES`, one `static` tape per bucket (the one-row
//! decode bucket and the four prefill buckets), and `CUDA_UNLOWERED`, the buckets left out. The
//! checks:
//!
//! - every emitted tape is, field by field, what CUDA's compiler lowers the same canonicals to
//!   (`lower_model`: `lower_canonical` per bucket, then the weight table's tie rewrite), and the
//!   emitted `CUDA_WEIGHTS` is its weight table;
//! - the tapes carry the numbers measured for this model: the step histogram, the colours, the
//!   arena, input and result spans per bucket, the weights, and the KV and GDN layers; the
//!   one-row tape the layout #830 predicted, its GEMMs GEMVs and its attention the decode kernel;
//! - the weight table names the checkpoint's tensors: 266 entries, 18 of them GDN bundles, 320
//!   distinct tensors, none named twice, and the tied lm_head reads the embedding table.

#[path = "../../../targets/cuda/tests/common/chains.rs"]
pub mod chains;

use std::collections::BTreeMap;
use std::path::PathBuf;

use scratchy_models::qwen3_5::qwen3_5_0_8b::{CUDA_TAPES, CUDA_UNLOWERED, CUDA_WEIGHTS};
use scratchy_subtile::handoff::TargetModelInput;
use scratchy_target_cuda::step::{AttnPrefillBf16, CudaStep, EmbedBf16, GemmBf16};
use scratchy_target_cuda::tape::{ArenaOffset, ArenaSpan, CudaTape, TapeInput, WeightIx};
use scratchy_target_cuda::weights::CudaWeight;
use scratchy_target_cuda_compiler::emit::{lower_model, model_facts};

/// The 0.8b's canonicals, as `#[forward]` handed them to the seam.
///
/// The pipeline selects models the way the build script that expanded `CUDA_TAPES` did: from
/// Cargo's `CARGO_FEATURE_<STEM>` and the quant scope in `SCRATCHY_QUANTS`, which Cargo and the
/// build script set for the build script only. This build's scope is the one model and no quant
/// preset, so the same two are set here, once, before the pipeline reads them; no other test in
/// this binary reads the environment.
fn input() -> TargetModelInput {
    static INPUT: std::sync::OnceLock<TargetModelInput> = std::sync::OnceLock::new();
    INPUT.get_or_init(capture).clone()
}

fn capture() -> TargetModelInput {
    // SAFETY: set before the pipeline's own threads start, and no other thread of this binary
    // reads or writes the environment.
    unsafe {
        std::env::set_var("CARGO_FEATURE_QWEN3.5_0.8B", "1");
        std::env::set_var("SCRATCHY_QUANTS", "");
    }
    let models = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dsl = std::fs::read_to_string(models.join("dsl/qwen3-5.py")).expect("read the DSL");
    let carrier = scratchy_forward_compiler_macro::parse_python_file(&dsl).expect("parse the DSL");
    let inputs = scratchy_forward_compiler_macro::target_model_inputs(
        carrier,
        &models.join("configs/qwen3-5"),
    )
    .expect("the forward pipeline");
    let mut inputs: Vec<_> = (inputs.into_iter())
        .filter(|i| i.stem == "qwen3.5-0.8b")
        .collect();
    assert_eq!(inputs.len(), 1, "one qwen3.5-0.8b model handed to the seam");
    inputs.remove(0)
}

/// The prefill buckets.
const BUCKETS: [u32; 4] = [512, 1024, 2048, 4096];
/// Every lowered bucket, ascending: decode's one row, then the prefill buckets.
const TAPE_BUCKETS: [u32; 5] = [1, 512, 1024, 2048, 4096];

const fn span(offset: u64, bytes: u64) -> ArenaSpan {
    ArenaSpan {
        offset: ArenaOffset(offset),
        bytes,
    }
}

/// Per bucket: the arena's `total_bytes` before the token ids (#812's), colour 0 (the embedded
/// hidden), the result span. The token ids, `4·bucket_m` bytes, sit at that total, and the arena
/// grows by exactly them.
const ARENAS: [(u32, u64, ArenaSpan, ArenaSpan); 4] = [
    (
        512,
        323_128_320,
        span(0, 1_048_576),
        span(39_354_368, 254_279_680),
    ),
    (
        1024,
        646_255_616,
        span(0, 2_097_152),
        span(78_708_736, 508_559_360),
    ),
    (
        2048,
        1_292_510_208,
        span(0, 4_194_304),
        span(157_417_472, 1_017_118_720),
    ),
    (
        4096,
        2_585_019_392,
        span(0, 8_388_608),
        span(314_834_944, 2_034_237_440),
    ),
];

/// Per bucket, the arena's `total_bytes` with the token ids.
const TOTALS: [u64; 4] = [323_130_368, 646_259_712, 1_292_518_400, 2_585_035_776];

const VOCAB: u32 = 248_320;
const HIDDEN: u32 = 1024;

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
        };
        *h.entry(kind).or_default() += 1;
    }
    h
}

#[test]
fn every_emitted_tape_is_what_cuda_lowers_from_the_same_canonical() {
    let input = input();
    let facts = model_facts(&input.facts);
    assert_eq!(
        (facts.head_dim, facts.rot_dim, facts.rope_theta),
        (256, 64, 1e7),
        "the 0.8b's head_dim, rot_dim and rope_theta"
    );
    let gdn = facts.gdn.expect("the 0.8b has GDN layers");
    assert_eq!(
        (gdn.num_k_heads, gdn.head_k, gdn.conv_kernel, gdn.eps),
        (16, 128, 4, 1e-6),
        "the 0.8b's GDN facts"
    );
    assert_eq!(
        facts.embed.map(|e| e.vocab),
        Some(VOCAB),
        "the 0.8b's embedding table"
    );
    // The whole model as the seam lowers it: every bucket, every sk (one tape per m, which
    // `lower_model` checks), then the weight table.
    let model = lower_model(&input).unwrap_or_else(|e| panic!("lower_model: {e}"));
    assert_eq!(model.tapes.len(), CUDA_TAPES.len(), "tapes");
    for (tape, lowered) in CUDA_TAPES.iter().zip(&model.tapes) {
        let name = format!("qwen3.5-0.8b m={}", tape.bucket_m);
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
}

/// The buckets above one row and below the GEMM's tile are left out, each by its first GEMM's
/// `m`; the one-row bucket is not among them, it lowers (the GEMV).
#[test]
fn the_buckets_below_the_gemm_tile_are_left_out_and_named() {
    let ms: Vec<u32> = CUDA_UNLOWERED.iter().map(|&(m, _)| m).collect();
    assert_eq!(ms, [2, 4, 8, 64], "CUDA_UNLOWERED's buckets");
    for &(m, why) in CUDA_UNLOWERED {
        let want = format!("m = {m} must be a non-zero multiple of 128, the bf16 GEMM's tile");
        assert!(why.contains(&want), "m={m}: {why}");
    }
}

/// The one-row decode tape: the layout #830 predicted from the shared colouring and arena passes.
/// Every GEMM whose `n` the narrow GEMM does not take is a GEMV, the attentions are the decode
/// kernel's, and the steps, colours and regions sit where the prefill buckets' rules put them.
#[test]
fn the_decode_tape_has_the_predicted_layout() {
    let tape = tape_of(1);
    let want_steps: BTreeMap<&str, usize> = [
        ("Add", 48),
        ("AttnDecode", 6),
        ("Embed", 1),
        ("GateApply", 6),
        ("GateSplit", 6),
        ("GatedDeltaNet", 18),
        ("GemmNarrow", 36),
        ("Gemv", 151),
        ("RmsNorm", 61),
        ("Rope", 6),
        ("RopeAppend", 6),
        ("SiluMul", 24),
    ]
    .into_iter()
    .collect();
    assert_eq!(tape.steps.len(), 369, "steps");
    assert_eq!(histogram(tape), want_steps, "step histogram");
    assert_eq!(
        tape.arena.colour_bytes,
        [
            2048, 2048, 12288, 4096, 32, 32, 4096, 2048, 7168, 7168, 7168, 7168, 2048, 8192, 1024,
            1024, 1024, 4096, 4096, 496640, 4096, 4096, 4096, 4096, 4096, 4096
        ],
        "colour bytes"
    );
    assert_eq!(tape.arena.total_bytes, 633_600, "arena total_bytes");
    assert_eq!(
        tape.result,
        span(77_312, 496_640),
        "the logits, [1, 248320] bf16"
    );
    let ids = span(633_344, 4);
    assert_eq!(
        tape.input,
        TapeInput::TokenIds {
            span: ids,
            vocab: VOCAB
        },
        "input"
    );
    assert_eq!(
        tape.steps[0],
        CudaStep::Embed(EmbedBf16 {
            tokens: ids.offset,
            table: WeightIx(0),
            out: ArenaOffset(0),
            rows: 1,
            hidden: HIDDEN,
            vocab: VOCAB,
        }),
        "the embed"
    );
    let rotary = tape.rotary.expect("a rotary table");
    assert_eq!(
        (rotary.span, rotary.rot_dim),
        (span(598_528, 128), 64),
        "rotary"
    );
    let kv = tape.kv.expect("a KV cache");
    assert_eq!(
        (kv.block_table, kv.slot_mapping, kv.sequence),
        (span(598_784, 512), span(599_296, 4), span(599_552, 8)),
        "KV regions"
    );
    let gdn = tape.gdn.expect("GDN layers");
    assert_eq!(
        (gdn.sequence, gdn.scratch),
        (span(599_808, 16), span(600_064, 33_280)),
        "GDN regions"
    );
    for (i, step) in tape.steps.iter().enumerate() {
        match *step {
            CudaStep::Gemv(GemmBf16 { m, .. }) | CudaStep::GemmNarrow(GemmBf16 { m, .. }) => {
                assert_eq!(m, 1, "step {i}");
            }
            CudaStep::AttnDecode(AttnPrefillBf16 {
                rows,
                q_heads,
                kv_heads,
                head_dim,
                block_table,
                sequence,
                ..
            }) => {
                assert_eq!(
                    (rows, q_heads, kv_heads, head_dim),
                    (1, 8, 2, 256),
                    "step {i}"
                );
                assert_eq!(
                    (block_table, sequence),
                    (kv.block_table.offset, kv.sequence.offset)
                );
            }
            CudaStep::Gemm(_) | CudaStep::AttnPrefill(_) => {
                panic!("step {i}: a prefill kernel in the one-row tape: {step:?}")
            }
            _ => {}
        }
    }
}

/// The four prefill tapes, each looked up by its bucket.
#[test]
fn every_prefill_tape_has_the_measured_layout() {
    let want_steps: BTreeMap<&str, usize> = [
        ("Add", 48),
        ("Embed", 1),
        ("AttnPrefill", 6),
        ("GateApply", 6),
        ("GateSplit", 6),
        ("GatedDeltaNet", 18),
        ("Gemm", 151),
        ("GemmNarrow", 36),
        ("RmsNorm", 61),
        ("Rope", 6),
        ("RopeAppend", 6),
        ("SiluMul", 24),
    ]
    .into_iter()
    .collect();
    let kv_layers = [3, 7, 11, 15, 19, 23];
    let gdn_layers: Vec<u32> = (0..24).filter(|l| !kv_layers.contains(l)).collect();
    for &(m, total, input, result) in &ARENAS {
        let tape = tape_of(m);
        assert_eq!(tape.steps.len(), 369, "m={m}: steps");
        assert_eq!(histogram(tape), want_steps, "m={m}: step histogram");
        assert_eq!(tape.arena.colour_bytes.len(), 26, "m={m}: colours");
        let with_ids = TOTALS[BUCKETS.iter().position(|&b| b == m).expect("a bucket")];
        assert_eq!(tape.arena.total_bytes, with_ids, "m={m}: arena total_bytes");
        assert_eq!(
            with_ids,
            total + 4 * u64::from(m),
            "m={m}: the arena grows by 4·m"
        );
        let ids = span(total, 4 * u64::from(m));
        assert_eq!(
            tape.input,
            TapeInput::TokenIds {
                span: ids,
                vocab: VOCAB
            },
            "m={m}: input"
        );
        assert_eq!(
            tape.steps[0],
            CudaStep::Embed(EmbedBf16 {
                tokens: ids.offset,
                table: WeightIx(0),
                out: input.offset,
                rows: m,
                hidden: HIDDEN,
                vocab: VOCAB,
            }),
            "m={m}: the embed writes colour 0, {input:?}"
        );
        assert_eq!(tape.arena.colour_bytes[0], input.bytes, "m={m}: colour 0");
        assert_eq!(tape.result, result, "m={m}: result span");
        // The result is the full logits, `[m, 248320]` bf16.
        assert_eq!(result.bytes, u64::from(m) * 248_320 * 2, "m={m}: logits");
        let weights: Vec<u64> = (tape.weight_bytes.iter().copied())
            .filter(|&b| b > 0)
            .collect();
        assert_eq!(weights.len(), 266, "m={m}: nonzero weights");
        assert_eq!(
            weights.iter().sum::<u64>(),
            1_505_683_456,
            "m={m}: weight bytes"
        );
        let kv = tape.kv.expect("a KV cache");
        assert_eq!(kv.layers, kv_layers, "m={m}: KV layers");
        assert_eq!((kv.kv_heads, kv.head_dim), (2, 256), "m={m}: KV geometry");
        let gdn = tape.gdn.expect("GDN layers");
        assert_eq!(gdn.layers, gdn_layers.as_slice(), "m={m}: GDN layers");
        assert_eq!(
            (
                gdn.num_k_heads,
                gdn.num_v_heads,
                gdn.head_k,
                gdn.head_v,
                gdn.conv_kernel
            ),
            (16, 16, 128, 128, 4),
            "m={m}: GDN geometry"
        );
    }
}

/// The weight table: one entry per weight index the tapes read, 266 of them, from the
/// checkpoint's 320 decoder tensors (247 bf16 tensors, 18 GDN bundles of four, the embedding
/// table); none named twice, because the tied lm_head reads the embedding table's index.
#[test]
fn the_weight_table_names_the_checkpoints_tensors() {
    assert_eq!(
        CUDA_WEIGHTS.len(),
        CUDA_TAPES[0].weight_bytes.len(),
        "one entry per index"
    );
    let entries: Vec<(usize, CudaWeight)> = (CUDA_WEIGHTS.iter().enumerate())
        .filter_map(|(ix, w)| w.map(|w| (ix, w)))
        .collect();
    assert_eq!(entries.len(), 266, "Some entries");
    let bundles = (entries.iter())
        .filter(|(_, w)| matches!(w, CudaWeight::GdnBundle { .. }))
        .count();
    assert_eq!(bundles, 18, "GDN bundles");
    let mut names = std::collections::BTreeSet::new();
    let mut leads = std::collections::BTreeSet::new();
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
        };
        assert!(
            leads.insert(tensors[0]),
            "weight {ix}: {} named twice",
            tensors[0]
        );
        names.extend(tensors);
        for tape in CUDA_TAPES {
            assert_eq!(
                Some(tape.weight_bytes[*ix]),
                w.packed_bytes(),
                "m={}: weight {ix}'s declared bytes",
                tape.bucket_m
            );
        }
    }
    assert_eq!(names.len(), 320, "distinct tensors");
    for (ix, w) in CUDA_WEIGHTS.iter().enumerate() {
        if w.is_none() {
            for tape in CUDA_TAPES {
                assert_eq!(tape.weight_bytes[ix], 0, "m={}: weight {ix}", tape.bucket_m);
            }
        }
    }
    let bytes: u64 = (entries.iter())
        .map(|(_, w)| w.packed_bytes().expect("fits"))
        .sum();
    assert_eq!(bytes, 1_505_683_456, "the table's bytes");
    let by_name = |name: &str| {
        (entries.iter())
            .find(|(_, w)| matches!(w, CudaWeight::Bf16 { tensor, .. } if *tensor == name))
            .copied()
            .unwrap_or_else(|| panic!("no entry for {name}"))
    };
    let shape = |w: CudaWeight| match w {
        CudaWeight::Bf16 { shape, .. } => shape,
        CudaWeight::GdnBundle { .. } => panic!("a bundle"),
    };
    let (e, embed) = by_name("language_model.model.embed_tokens.weight");
    assert_eq!(e, 0, "the embedding table is the embedded hidden's index");
    assert_eq!(shape(embed), [248_320, 1024], "the embedding table");
    for (name, want) in [
        (
            "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
            &[6144, 1024][..],
        ),
        (
            "language_model.model.layers.3.self_attn.q_proj.weight",
            &[4096, 1024][..],
        ),
        ("language_model.model.norm.weight", &[1024][..]),
    ] {
        assert_eq!(shape(by_name(name).1), want, "{name}");
    }
    let bundle = (entries.iter())
        .find_map(|(_, w)| match *w {
            b @ CudaWeight::GdnBundle {
                conv1d: "language_model.model.layers.0.linear_attn.conv1d.weight",
                ..
            } => Some(b),
            _ => None,
        })
        .expect("layer 0's bundle");
    assert_eq!(
        bundle,
        CudaWeight::GdnBundle {
            conv1d: "language_model.model.layers.0.linear_attn.conv1d.weight",
            a_log: "language_model.model.layers.0.linear_attn.A_log",
            dt_bias: "language_model.model.layers.0.linear_attn.dt_bias",
            norm: "language_model.model.layers.0.linear_attn.norm.weight",
            conv_dim: 6144,
            conv_kernel: 4,
            nv: 16,
            head_v: 128,
        }
    );
    assert_eq!(bundle.packed_bytes(), Some(99_328), "a bundle's bytes");
    let layout = scratchy_target_cuda::step::gdn_bundle(6144, 4, 16, 128).expect("fits");
    assert_eq!(
        (
            layout.conv1d,
            layout.a_log,
            layout.dt_bias,
            layout.norm,
            layout.bytes
        ),
        (0, 98_304, 98_560, 98_816, 99_328),
        "the bundle's regions"
    );
    // The tied lm_head: the last step reads the embedding table's index, and its own is empty.
    // The decode tape's is the GEMV, every prefill tape's the tiled GEMM.
    for tape in CUDA_TAPES {
        let m = tape.bucket_m;
        let (Some(&CudaStep::Gemm(GemmBf16 { b, n, k, .. }))
        | Some(&CudaStep::Gemv(GemmBf16 { b, n, k, .. }))) = tape.steps.last()
        else {
            panic!("m={m}: the tape ends with the lm_head GEMM");
        };
        assert_eq!(
            matches!(tape.steps.last(), Some(CudaStep::Gemv(_))),
            m == 1,
            "m={m}: the lm_head's kernel"
        );
        assert_eq!(
            (b, n, k),
            (WeightIx(e as u32), VOCAB, HIDDEN),
            "m={m}: the lm_head"
        );
    }
}
