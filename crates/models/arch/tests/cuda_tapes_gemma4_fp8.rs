// SPDX-License-Identifier: Apache-2.0
//! Gemma 4 26B-A4B's emitted CUDA tapes under `fp8-dynamic-per-channel` (FP8 S2), checked with no
//! toolchain. The tests are `common/cuda_tapes_fp8_body.rs`; this binary gives them the model.
//!
//! The checkpoint (`RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic`) quantizes the attention projections,
//! the dense MLP and every routed expert, and leaves the router (`router.proj`, its own bf16 entry
//! beside the fp8 expert bank) and the lm_head bf16. So, per prefill bucket: 30 fp8 q/k/v groups,
//! 60 fp8 GEMMs (30 attention o projections, 30 MLP downs) and 30 fp8 MLP gate/up pairs, each after
//! its own quant write, and the 90 routed-expert GEMMs over the fp8 bank, which read all their
//! sorted or activation rows quantized, the up reusing its gate's write (180). The decode tape has
//! the same projections as fp8 GEMVs, and the decode tape's fp8 GEMVs the shape rule takes run on
//! the tensor cores, each after a one-row quant write (or reading one a step before it wrote).

#[path = "../../../targets/cuda/tests/common/chains.rs"]
pub mod chains;

use scratchy_models::gemma4_moe::gemma_4_26b_a4b_it_fp8_dynamic_per_channel::{
    CUDA_TAPES, CUDA_UNLOWERED, CUDA_WEIGHTS,
};

include!("common/cuda_tapes_fp8_body.rs");

const STEM: &str = "gemma-4-26b-a4b-it";
const FEATURE: &str = "GEMMA_4_26B_A4B_IT";
const ARCH: &str = "gemma4-moe";
/// The decode tape's greedy argmax width (W7): the model's vocab, its whole logits row.
const GREEDY_WIDTH: u32 = 262_144;

const DECODE_STEPS: [(&str, usize); 21] = [
    ("AttnDecode", 30),
    ("Embed", 1),
    ("ExpertCombine", 30),
    ("ExpertGateUpTcFp8", 30),
    ("ExpertGemvTcFp8", 30),
    ("GateUpTcFp8", 30),
    ("Gemv", 1),
    ("GemvGroupTcFp8", 30),
    ("GemvTcFp8", 60),
    ("LayerScale", 30),
    ("MoeRoute", 30),
    ("NormAdd", 90),
    ("QuantRowsFp8", 90),
    ("RmsNorm", 101),
    ("RmsNormQuantFp8", 30),
    ("RmsNormUnit", 5),
    ("RopeAppend", 30),
    ("RouterLogits", 30),
    ("RouterNorm", 30),
    ("Scale", 1),
    ("SoftCap", 1),
];
/// The decode tape's quant writes and their readers (FP8 S8 T2).
const DECODE_QUANTS: usize = 90;
const DECODE_QUANT_READERS: usize = 180;
const PREFILL_STEPS: [(&str, usize); 23] = [
    ("AttnPrefill", 30),
    ("Embed", 1),
    ("ExpertCombine", 30),
    ("ExpertGateUpGemmFp8", 30),
    ("ExpertGemm", 30),
    ("ExpertSortFp8", 30),
    ("GateUpAct", 30),
    ("GatherLastRow", 1),
    ("Gemm", 1),
    ("GemmFp8", 60),
    ("GemmGroupFp8", 30),
    ("LayerScale", 30),
    ("MoeRoute", 30),
    ("NormAdd", 90),
    ("QuantRowsFp8", 150),
    ("RmsNorm", 181),
    ("RmsNormUnit", 30),
    ("Rope", 30),
    ("RopeAppend", 30),
    ("RouterLogits", 30),
    ("RouterNorm", 30),
    ("Scale", 1),
    ("SoftCap", 1),
];
const QUANTS: usize = 150;
/// The quant writes' readers: every fp8 consumer, the expert up GEMM included.
const QUANT_READERS: usize = 180;
/// The `k`s quantized: the routed experts' down (704), the MLP's down (2112), the hidden (2816),
/// the sliding and the global layers' o (4096, 8192).
const QUANT_COLS: [u32; 4] = [2112, 2816, 4096, 8192];
const QUANT_SCRATCH: [(u32, u64, u64); 4] = [
    (512, 535_476_992, 4_196_352),
    (1024, 946_056_192, 8_392_704),
    (2048, 1_767_214_592, 16_785_408),
    (4096, 3_409_531_392, 33_570_816),
];
const EXPERT_WEIGHT: Option<MoeWeight> = Some(MoeWeight::Fp8Own);
const TABLE_LEN: usize = 1051;
const WEIGHT_KINDS: [(&str, usize); 3] = [("Bf16", 392), ("Fp8", 205), ("Fp8ExpertBank", 30)];
const HEADERS: &str = "fp8_headers_gemma-4-26b-a4b-it.json";
