// SPDX-License-Identifier: Apache-2.0
//! Qwen3.8-27B's emitted CUDA tapes under `fp8-dynamic-per-channel` (FP8 S2), checked with no
//! toolchain. The tests are `common/cuda_tapes_fp8_body.rs`; this binary gives them the model.
//!
//! The checkpoint (`axteam/Qwen3.8-27B-FP8-dynamic`) quantizes every decoder projection, the GDN
//! ones included, and leaves the lm_head bf16. So, per prefill bucket: 64 fp8 groups (48 GDN
//! qkv/z, 16 attention q/k/v), 48 fp8 narrow groups (GDN a + b, 48 + 48 columns), 128 fp8 GEMMs
//! (48 GDN out, 16 attention o, 64 MLP downs) and 64 fp8 MLP gate/up pairs, each after its own
//! quant write (304). The decode tape has the same projections as fp8 GEMVs and no quant write.
//!
//! The 27B builds as `qwen3.8-27b-text-only`, as the 9b does (`cuda_tapes_9b.rs`).

#[path = "../../../targets/cuda/tests/common/chains.rs"]
pub mod chains;

use scratchy_models::qwen3_5::qwen3_8_27b_fp8_dynamic_per_channel::{
    CUDA_TAPES, CUDA_UNLOWERED, CUDA_WEIGHTS,
};

include!("common/cuda_tapes_fp8_body.rs");

const STEM: &str = "qwen3.8-27b";
const FEATURE: &str = "CARGO_FEATURE_QWEN3.8_27B_TEXT_ONLY";
const ARCH: &str = "qwen3-5";

const DECODE_STEPS: [(&str, usize); 11] = [
    ("AttnDecode", 16),
    ("Embed", 1),
    ("GateApply", 16),
    ("GateSplit", 16),
    ("GateUpAct", 64),
    ("GatedDeltaNet", 48),
    ("Gemv", 1),
    ("GemvFp8", 128),
    ("GemvGroupFp8", 112),
    ("RmsNorm", 161),
    ("RopeAppend", 16),
];
const PREFILL_STEPS: [(&str, usize); 15] = [
    ("AddRmsNorm", 128),
    ("AttnPrefill", 16),
    ("Embed", 1),
    ("GateApply", 16),
    ("GateSplit", 16),
    ("GateUpAct", 64),
    ("GatedDeltaNet", 48),
    ("Gemm", 1),
    ("GemmFp8", 128),
    ("GemmGroupFp8", 64),
    ("GemmNarrowGroupFp8", 48),
    ("QuantRowsFp8", 304),
    ("RmsNorm", 33),
    ("Rope", 16),
    ("RopeAppend", 16),
];
const QUANTS: usize = 304;
/// The quant writes' readers: one per write (no experts).
const QUANT_READERS: usize = 304;
/// The `k`s quantized: the hidden (5120), GDN's and attention's out (6144), the MLP's down
/// (17408).
const QUANT_COLS: [u32; 3] = [5120, 6144, 17408];
const QUANT_SCRATCH: [(u32, u64, u64); 4] = [
    (512, 537_760_768, 8_914_944),
    (1024, 1_075_520_512, 17_829_888),
    (2048, 2_151_040_000, 35_659_776),
    (4096, 4_302_078_976, 71_319_552),
];
/// No MoE.
const EXPERT_WEIGHT: Option<MoeWeight> = None;
const TABLE_LEN: usize = 1239;
const WEIGHT_KINDS: [(&str, usize); 3] = [("Bf16", 163), ("Fp8", 496), ("GdnBundle", 48)];
const HEADERS: &str = "fp8_headers_qwen3.8-27b.json";
