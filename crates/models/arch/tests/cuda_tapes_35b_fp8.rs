// SPDX-License-Identifier: Apache-2.0
//! Qwen3.6-35B-A3B's emitted CUDA tapes under `fp8-dynamic-per-channel` (FP8 S2), checked with no
//! toolchain. The tests are `common/cuda_tapes_fp8_body.rs`; this binary gives them the model.
//!
//! The checkpoint (`RedHatAI/Qwen3.6-35B-A3B-FP8-dynamic`) quantizes the attention projections,
//! the shared expert and every routed expert, and leaves the GDN projections, the routers, the
//! shared expert's gate and the lm_head bf16. So, per prefill bucket: 10 fp8 q/k/v groups, 50 fp8
//! GEMMs (10 attention o projections, 40 shared-expert downs) and 40 fp8 shared-expert gate/up
//! pairs, each after its own quant write (100); the experts read the fp8 bundle. The decode tape
//! has the same projections as fp8 GEMVs and no quant write.

#[path = "../../../targets/cuda/tests/common/chains.rs"]
pub mod chains;

use scratchy_models::qwen3_5_moe::qwen3_6_35b_a3b_fp8_dynamic_per_channel::{
    CUDA_TAPES, CUDA_UNLOWERED, CUDA_WEIGHTS,
};

include!("common/cuda_tapes_fp8_body.rs");

const STEM: &str = "qwen3.6-35b-a3b";
const FEATURE: &str = "CARGO_FEATURE_QWEN3.6_35B_A3B";
const ARCH: &str = "qwen3-5-moe";

const DECODE_STEPS: [(&str, usize); 22] = [
    ("AddRmsNorm", 80),
    ("AttnDecode", 10),
    ("Embed", 1),
    ("ExpertCombine", 40),
    ("ExpertGateUpAct", 40),
    ("ExpertGemv", 40),
    ("GateApply", 10),
    ("GateScale", 40),
    ("GateSplit", 10),
    ("GateUpAct", 40),
    ("GdnDecode", 30),
    ("GemmNarrow", 40),
    ("GemmNarrowGroup", 30),
    ("Gemv", 31),
    ("GemvFp8", 50),
    ("GemvGroup", 30),
    ("GemvGroupFp8", 10),
    ("MoeRoute", 40),
    ("RmsNorm", 21),
    ("Rope", 10),
    ("RopeAppend", 10),
    ("RouterLogits", 40),
];
const PREFILL_STEPS: [(&str, usize); 24] = [
    ("AddRmsNorm", 80),
    ("AttnPrefill", 10),
    ("Embed", 1),
    ("ExpertCombine", 40),
    ("ExpertGemm", 120),
    ("ExpertSort", 40),
    ("GateApply", 10),
    ("GateScale", 40),
    ("GateSplit", 10),
    ("GateUpAct", 40),
    ("GatedDeltaNet", 30),
    ("Gemm", 31),
    ("GemmFp8", 50),
    ("GemmGroup", 30),
    ("GemmGroupFp8", 10),
    ("GemmNarrow", 40),
    ("GemmNarrowGroup", 30),
    ("MoeRoute", 40),
    ("QuantRowsFp8", 100),
    ("RmsNorm", 21),
    ("Rope", 10),
    ("RopeAppend", 10),
    ("RouterLogits", 40),
    ("SiluMul", 40),
];
/// The fp8 consumers of a prefill tape, each after its own quant write.
const QUANTS: usize = 100;
/// The `k`s quantized: the shared expert's down (512), the hidden (2048), attention's o (4096).
const QUANT_COLS: [u32; 3] = [512, 2048, 4096];
/// Per prefill bucket: the quant scratch's offset (the bf16 tape's arena total) and bytes (`[m,
/// 4096]` e4m3 then `[m]` f32).
const QUANT_SCRATCH: [(u32, u64, u64); 4] = [
    (512, 772_613_632, 2_099_200),
    (1024, 1_179_007_488, 4_198_400),
    (2048, 1_991_795_456, 8_396_800),
    (4096, 3_617_371_392, 16_793_600),
];
const EXPERT_WEIGHT: Option<MoeWeight> = Some(MoeWeight::Fp8Bundle);
const TABLE_LEN: usize = 707;
const WEIGHT_KINDS: [(&str, usize); 4] = [
    ("Bf16", 293),
    ("Fp8", 160),
    ("Fp8MoeBundle", 40),
    ("GdnBundle", 30),
];
const HEADERS: &str = "fp8_headers_qwen3.6-35b-a3b.json";
