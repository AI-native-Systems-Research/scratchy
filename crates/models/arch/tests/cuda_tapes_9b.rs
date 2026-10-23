// SPDX-License-Identifier: Apache-2.0
//! qwen3.5-9b's emitted CUDA tapes, checked with no toolchain.
//!
//! The same checks as qwen3.5-0.8b's `cuda_tapes.rs` (the tests are `common/cuda_tapes_body.rs`),
//! at the 9b's numbers, measured by the toolchain-free expansion (AI-native-Systems-Research/
//! scratchy#839):
//!
//! - every emitted tape is, field by field, what CUDA's compiler lowers the same canonicals to,
//!   and the emitted `CUDA_WEIGHTS` is its weight table;
//! - the tapes carry the 9b's step histogram (491 steps), 25 colours, the arena, input and result
//!   spans per bucket, the weights, and the KV and GDN layers; the one-row tape's GEMMs are GEMVs
//!   and its attention the decode kernel;
//! - the weight table names the checkpoint's tensors: 355 entries, 24 of them GDN bundles, 427
//!   distinct tensors, none named twice. The 9b's lm_head is untied: `lm_head.weight` at its own
//!   index 374, and nothing but the embed reads the embedding table.
//!
//! The 9b builds as `qwen3.5-9b-text-only` (this binary's required feature). The bare
//! `qwen3.5-9b` feature selects the vision architecture `qwen3-5-vl`, whose config is the same
//! file; under `cuda` it fails the build with "no models selected", because no `qwen3-5` config
//! is then selected.

#[path = "../../../targets/cuda/tests/common/chains.rs"]
pub mod chains;

use scratchy_models::qwen3_5::qwen3_5_9b::{CUDA_TAPES, CUDA_UNLOWERED, CUDA_WEIGHTS};

include!("common/cuda_tapes_body.rs");

/// The model the seam hands over, and the Cargo feature that selects it.
const STEM: &str = "qwen3.5-9b";
const FEATURE: &str = "CARGO_FEATURE_QWEN3.5_9B_TEXT_ONLY";

const VOCAB: u32 = 248_320;
const HIDDEN: u32 = 4096;
/// Full attention's query and KV heads.
const HEADS: (u32, u32) = (16, 4);
/// GDN's key and value heads: value head `v` reads key head `v / 2`.
const GDN_HEADS: (u32, u32) = (16, 32);
const LAYERS: u32 = 32;
const KV_LAYERS: &[u32] = &[3, 7, 11, 15, 19, 23, 27, 31];

/// Steps per tape, every bucket: 1 embed, 24 GDN layers of 14 steps, 8 attention layers of 19,
/// the final norm and the lm_head.
const STEPS: usize = 491;
const DECODE_STEPS: [(&str, usize); 12] = [
    ("Add", 64),
    ("AttnDecode", 8),
    ("Embed", 1),
    ("GateApply", 8),
    ("GateSplit", 8),
    ("GatedDeltaNet", 24),
    ("GemmNarrow", 48),
    ("Gemv", 201),
    ("RmsNorm", 81),
    ("Rope", 8),
    ("RopeAppend", 8),
    ("SiluMul", 32),
];
const DECODE_COLOUR_BYTES: &[u64] = &[
    8192, 8192, 16384, 8192, 64, 64, 24576, 24576, 24576, 24576, 8192, 2048, 2048, 2048, 8192,
    8192, 496640, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
];
const DECODE_TOTAL: u64 = 784_128;
const DECODE_RESULT: ArenaSpan = span(170_496, 496_640);
const DECODE_IDS: ArenaSpan = span(783_872, 4);
const DECODE_ROTARY: ArenaSpan = span(732_672, 128);
/// The block table, the slot mapping, the sequence.
const DECODE_KV: [ArenaSpan; 3] = [span(732_928, 512), span(733_440, 4), span(733_696, 8)];
/// The GDN sequence, the GDN scratch.
const DECODE_GDN: [ArenaSpan; 2] = [span(733_952, 16), span(734_208, 49_664)];

const PREFILL_STEPS: [(&str, usize); 12] = [
    ("Add", 64),
    ("Embed", 1),
    ("AttnPrefill", 8),
    ("GateApply", 8),
    ("GateSplit", 8),
    ("GatedDeltaNet", 24),
    ("Gemm", 201),
    ("GemmNarrow", 48),
    ("RmsNorm", 81),
    ("Rope", 8),
    ("RopeAppend", 8),
    ("SiluMul", 32),
];
const COLOURS: usize = 25;

/// Per bucket: the arena's `total_bytes` before the token ids, colour 0 (the embedded hidden),
/// the result span. The token ids, `4·bucket_m` bytes, sit at that total, and the arena grows by
/// exactly them.
const ARENAS: [(u32, u64, ArenaSpan, ArenaSpan); 4] = [
    (
        512,
        400_296_960,
        span(0, 4_194_304),
        span(87_097_344, 254_279_680),
    ),
    (
        1024,
        800_592_896,
        span(0, 8_388_608),
        span(174_194_688, 508_559_360),
    ),
    (
        2048,
        1_601_184_768,
        span(0, 16_777_216),
        span(348_389_376, 1_017_118_720),
    ),
    (
        4096,
        3_202_368_512,
        span(0, 33_554_432),
        span(696_778_752, 2_034_237_440),
    ),
];

/// Per bucket, the arena's `total_bytes` with the token ids.
const TOTALS: [u64; 4] = [400_299_008, 800_596_992, 1_601_192_960, 3_202_384_896];

/// The weight table: 355 entries from the checkpoint's 427 decoder tensors (330 bf16 tensors, 24
/// GDN bundles of four, the embedding table), in a table of 375 indices.
const WEIGHTS: usize = 355;
const WEIGHT_BYTES: u64 = 17_909_194_752;
const BUNDLES: usize = 24;
const TENSORS: usize = 427;
const SHAPES: [(&str, &[u64]); 5] = [
    (
        "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
        &[8192, 4096],
    ),
    (
        "language_model.model.layers.0.linear_attn.in_proj_a.weight",
        &[32, 4096],
    ),
    (
        "language_model.model.layers.3.self_attn.q_proj.weight",
        &[8192, 4096],
    ),
    (
        "language_model.model.layers.3.mlp.down_proj.weight",
        &[4096, 12288],
    ),
    ("language_model.model.norm.weight", &[4096]),
];
const LAYER_0_BUNDLE: CudaWeight = CudaWeight::GdnBundle {
    conv1d: "language_model.model.layers.0.linear_attn.conv1d.weight",
    a_log: "language_model.model.layers.0.linear_attn.A_log",
    dt_bias: "language_model.model.layers.0.linear_attn.dt_bias",
    norm: "language_model.model.layers.0.linear_attn.norm.weight",
    conv_dim: 8192,
    conv_kernel: 4,
    nv: 32,
    head_v: 128,
};
const BUNDLE_BYTES: u64 = 132_096;
/// The bundle's conv1d, a_log, dt_bias and norm offsets, and its bytes.
const BUNDLE_REGIONS: (u64, u64, u64, u64, u64) = (0, 131_072, 131_328, 131_584, 132_096);
/// `tie_word_embeddings` is false: the lm_head is its own tensor, at the index after the final
/// norm's.
const UNTIED_LM_HEAD: Option<(u32, &str)> = Some((374, "lm_head.weight"));
