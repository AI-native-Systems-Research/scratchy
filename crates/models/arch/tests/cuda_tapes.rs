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
//!   distinct tensors, none named twice, and the tied lm_head reads the embedding table;
//! - the model is in the CUDA registry once, under its stem, with these very tapes and table.
//!
//! The tests are `common/cuda_tapes_body.rs`, shared with qwen3.5-9b's `cuda_tapes_9b.rs`; this
//! file is the 0.8b's module and its numbers.

#[path = "../../../targets/cuda/tests/common/chains.rs"]
pub mod chains;

use scratchy_models::qwen3_5::qwen3_5_0_8b::{CUDA_TAPES, CUDA_UNLOWERED, CUDA_WEIGHTS};

include!("common/cuda_tapes_body.rs");

/// The model the seam hands over, and the Cargo feature that selects it.
const STEM: &str = "qwen3.5-0.8b";
const FEATURE: &str = "CARGO_FEATURE_QWEN3.5_0.8B";

const VOCAB: u32 = 248_320;
const HIDDEN: u32 = 1024;
/// Full attention's query and KV heads.
const HEADS: (u32, u32) = (8, 2);
/// GDN's key and value heads.
const GDN_HEADS: (u32, u32) = (16, 16);
const LAYERS: u32 = 24;
const KV_LAYERS: &[u32] = &[3, 7, 11, 15, 19, 23];

/// Steps per tape, every bucket.
const STEPS: usize = 369;
const DECODE_STEPS: [(&str, usize); 12] = [
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
];
const DECODE_COLOUR_BYTES: &[u64] = &[
    2048, 2048, 12288, 4096, 32, 32, 4096, 2048, 7168, 7168, 7168, 7168, 2048, 8192, 1024, 1024,
    1024, 4096, 4096, 496640, 4096, 4096, 4096, 4096, 4096, 4096,
];
const DECODE_TOTAL: u64 = 633_600;
const DECODE_RESULT: ArenaSpan = span(77_312, 496_640);
const DECODE_IDS: ArenaSpan = span(633_344, 4);
const DECODE_ROTARY: ArenaSpan = span(598_528, 128);
/// The block table, the slot mapping, the sequence.
const DECODE_KV: [ArenaSpan; 3] = [span(598_784, 512), span(599_296, 4), span(599_552, 8)];
/// The GDN sequence, the GDN scratch.
const DECODE_GDN: [ArenaSpan; 2] = [span(599_808, 16), span(600_064, 33_280)];

const PREFILL_STEPS: [(&str, usize); 12] = [
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
];
const COLOURS: usize = 26;

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

/// The weight table: 266 entries from the checkpoint's 320 decoder tensors (247 bf16 tensors, 18
/// GDN bundles of four, the embedding table).
const WEIGHTS: usize = 266;
const WEIGHT_BYTES: u64 = 1_505_683_456;
const BUNDLES: usize = 18;
const TENSORS: usize = 320;
const SHAPES: [(&str, &[u64]); 3] = [
    (
        "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
        &[6144, 1024],
    ),
    (
        "language_model.model.layers.3.self_attn.q_proj.weight",
        &[4096, 1024],
    ),
    ("language_model.model.norm.weight", &[1024]),
];
const LAYER_0_BUNDLE: CudaWeight = CudaWeight::GdnBundle {
    conv1d: "language_model.model.layers.0.linear_attn.conv1d.weight",
    a_log: "language_model.model.layers.0.linear_attn.A_log",
    dt_bias: "language_model.model.layers.0.linear_attn.dt_bias",
    norm: "language_model.model.layers.0.linear_attn.norm.weight",
    conv_dim: 6144,
    conv_kernel: 4,
    nv: 16,
    head_v: 128,
};
const BUNDLE_BYTES: u64 = 99_328;
/// The bundle's conv1d, a_log, dt_bias and norm offsets, and its bytes.
const BUNDLE_REGIONS: (u64, u64, u64, u64, u64) = (0, 98_304, 98_560, 98_816, 99_328);
/// `tie_word_embeddings`: the lm_head reads the embedding table's index.
const UNTIED_LM_HEAD: Option<(u32, &str)> = None;
