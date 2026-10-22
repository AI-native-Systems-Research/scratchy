// SPDX-License-Identifier: Apache-2.0
//! qwen3.5-0.8b's emitted CUDA tapes, checked with no toolchain.
//!
//! `#[forward]` hands every model to CUDA's compiler through the target-emitter seam, and the
//! model's module holds what came back: `CUDA_TAPES`, one `static` tape per prefill bucket, and
//! `CUDA_UNLOWERED`, the buckets left out. Two checks:
//!
//! - every emitted tape is, field by field, what `lower_canonical` gives for the same canonical
//!   (the canonical the seam was handed, recaptured here through the same pipeline);
//! - the tapes carry the numbers measured for this model: the step histogram, the colours, the
//!   arena, input and result spans per bucket, the weights, and the KV and GDN layers.

#[path = "../../../targets/cuda/tests/common/chains.rs"]
pub mod chains;

use std::collections::BTreeMap;
use std::path::PathBuf;

use scratchy_models::qwen3_5::qwen3_5_0_8b::{CUDA_TAPES, CUDA_UNLOWERED};
use scratchy_subtile::handoff::TargetModelInput;
use scratchy_target_cuda::step::CudaStep;
use scratchy_target_cuda::tape::{ArenaOffset, ArenaSpan, CudaTape};
use scratchy_target_cuda_compiler::emit::model_facts;

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

const BUCKETS: [u32; 4] = [512, 1024, 2048, 4096];

const fn span(offset: u64, bytes: u64) -> ArenaSpan {
    ArenaSpan {
        offset: ArenaOffset(offset),
        bytes,
    }
}

/// Per bucket: the arena's `total_bytes`, the input span (colour 0), the result span.
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

/// One tape's steps, by kind.
fn histogram(tape: &CudaTape) -> BTreeMap<&'static str, usize> {
    let mut h = BTreeMap::new();
    for step in tape.steps {
        let kind = match step {
            CudaStep::Gemm(_) => "Gemm",
            CudaStep::GemmNarrow(_) => "GemmNarrow",
            CudaStep::RmsNorm(_) => "RmsNorm",
            CudaStep::Add(_) => "Add",
            CudaStep::SiluMul(_) => "SiluMul",
            CudaStep::Rope(_) => "Rope",
            CudaStep::GateSplit(_) => "GateSplit",
            CudaStep::GateApply(_) => "GateApply",
            CudaStep::RopeAppend(_) => "RopeAppend",
            CudaStep::AttnPrefill(_) => "AttnPrefill",
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
    for tape in CUDA_TAPES {
        let m = tape.bucket_m;
        // Every sk of the bucket: the tape is one per m because it reads no sk.
        let canonicals: Vec<_> = (input.buckets.iter()).filter(|b| b.point.m == m).collect();
        assert!(!canonicals.is_empty(), "m={m}: no canonical");
        for b in canonicals {
            let name = format!("qwen3.5-0.8b m={m} sk={}", b.point.sk);
            chains::assert_tape_is_lowered_with(&name, tape, &b.lowered, facts);
        }
    }
}

#[test]
fn the_tapes_are_the_four_prefill_buckets_and_validate() {
    let ms: Vec<u32> = CUDA_TAPES.iter().map(|t| t.bucket_m).collect();
    assert_eq!(ms, BUCKETS, "CUDA_TAPES' buckets");
    for tape in CUDA_TAPES {
        tape.validate()
            .unwrap_or_else(|e| panic!("m={}: {e}", tape.bucket_m));
    }
}

#[test]
fn the_buckets_below_the_gemm_tile_are_left_out_and_named() {
    let ms: Vec<u32> = CUDA_UNLOWERED.iter().map(|&(m, _)| m).collect();
    assert_eq!(ms, [1, 2, 4, 8, 64], "CUDA_UNLOWERED's buckets");
    for &(m, why) in CUDA_UNLOWERED {
        let want = if m == 1 {
            "m = 1 is a decode GEMV, and CUDA has no decode GEMV kernel yet".to_string()
        } else {
            format!("m = {m} must be a non-zero multiple of 128, the bf16 GEMM's tile")
        };
        assert!(why.contains(&want), "m={m}: {why}");
    }
}

#[test]
fn every_tape_has_the_measured_layout() {
    let want_steps: BTreeMap<&str, usize> = [
        ("Add", 48),
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
    for (tape, &(m, total, input, result)) in CUDA_TAPES.iter().zip(&ARENAS) {
        assert_eq!(tape.bucket_m, m);
        assert_eq!(tape.steps.len(), 368, "m={m}: steps");
        assert_eq!(histogram(tape), want_steps, "m={m}: step histogram");
        assert_eq!(tape.arena.colour_bytes.len(), 26, "m={m}: colours");
        assert_eq!(tape.arena.total_bytes, total, "m={m}: arena total_bytes");
        assert_eq!(tape.input, input, "m={m}: input span");
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
