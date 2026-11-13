// SPDX-License-Identifier: Apache-2.0
//! Qwen3.6-35B-A3B's emitted CUDA tapes under `fp8-dynamic-per-channel` (FP8 S2), checked with no
//! toolchain. The tests are `common/cuda_tapes_fp8_body.rs`; this binary gives them the model.
//!
//! The checkpoint (`RedHatAI/Qwen3.6-35B-A3B-FP8-dynamic`) quantizes the attention projections, the
//! shared expert and every routed expert, and leaves the GDN projections, the routers, the shared
//! expert's gate and the lm_head bf16. So, per prefill bucket: 10 fp8 q/k/v groups, 50 fp8 GEMMs
//! (10 attention o projections, 40 shared-expert downs) and 40 fp8 shared-expert gate/up pairs,
//! each after its own quant write, and the 120 routed-expert GEMMs over the fp8 bundle, which read
//! all their sorted or activation rows quantized, the up reusing its gate's write (180). The decode
//! tape has the same projections as fp8 GEMVs, and the decode tape's fp8 GEMVs the shape rule takes
//! run on the tensor cores, each after a one-row quant write (or reading one a step before it
//! wrote).

#[path = "../../../targets/cuda/tests/common/chains.rs"]
pub mod chains;

use scratchy_models::qwen3_5_moe::qwen3_6_35b_a3b_fp8_dynamic_per_channel::{
    CUDA_TAPES, CUDA_UNLOWERED, CUDA_WEIGHTS,
};

include!("common/cuda_tapes_fp8_body.rs");

const STEM: &str = "qwen3.6-35b-a3b";
const FEATURE: &str = "QWEN3.6_35B_A3B";
const ARCH: &str = "qwen3-5-moe";
/// The decode tape's greedy argmax width (W7): the model's vocab, its whole logits row.
const GREEDY_WIDTH: u32 = 248_320;

const DECODE_STEPS: [(&str, usize); 22] = [
    ("AddRmsNorm", 30),
    ("AddRmsNormQuantFp8", 10),
    ("AttnDecode", 10),
    ("Embed", 1),
    ("ExpertCombine", 40),
    ("ExpertGateUpTcFp8", 40),
    ("ExpertGemvTcFp8", 40),
    ("GateApply", 10),
    ("GateScale", 40),
    ("GateSplit", 10),
    ("GateUpTcFp8", 40),
    ("GdnDecode", 30),
    ("GemmNarrow", 40),
    ("Gemv", 31),
    ("GemvGroup", 60),
    ("GemvGroupTcFp8", 10),
    ("GemvTcFp8", 50),
    ("MoeRoute", 40),
    ("QuantRowsFp8", 50),
    ("RmsNorm", 61),
    ("RopeAppend", 10),
    ("RouterLogits", 40),
];
/// The decode tape's quant writes and their readers (FP8 S8 T2).
const DECODE_QUANTS: usize = 50;
const DECODE_QUANT_READERS: usize = 180;
const PREFILL_STEPS: [(&str, usize); 25] = [
    ("AddRmsNorm", 80),
    ("AttnPrefill", 10),
    ("Embed", 1),
    ("ExpertCombine", 40),
    ("ExpertGateUpGemmFp8", 40),
    ("ExpertGemm", 40),
    ("ExpertSortFp8", 40),
    ("GateApply", 10),
    ("GateScale", 40),
    ("GateSplit", 10),
    ("GateUpAct", 40),
    ("GatedDeltaNet", 30),
    ("GatherLastRow", 1),
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
];
/// The quant writes of a prefill tape: one per fp8 consumer, the expert up GEMM's shared with its
/// gate's.
const QUANTS: usize = 100;
/// The quant writes' readers: every fp8 consumer, the expert up GEMM included.
const QUANT_READERS: usize = 180;
/// The `k`s quantized: the shared expert's down (512), the hidden (2048), attention's o (4096),
/// and the routed experts' (512 down, 2048 gate/up).
const QUANT_COLS: [u32; 3] = [512, 2048, 4096];
/// Per prefill bucket: the quant scratch's offset (the bf16 tape's arena total) and bytes (the
/// largest expert write, `[max_tiles·128, 2048]` e4m3 then its f32 scales).
const QUANT_SCRATCH: [(u32, u64, u64); 4] = [
    (512, 588_064_256, 2_099_200),
    (1024, 994_458_368, 4_198_400),
    (2048, 1_807_246_592, 8_396_800),
    (4096, 3_432_823_040, 16_793_600),
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

/// The shared expert's gate/up reads the routed experts' quantization of the same normed row
/// (FP8 S8 T2): no quant write between the two, so the one write serves both, in every MoE layer.
#[test]
fn the_shared_gate_up_reuses_the_routed_quant_write() {
    let steps = tape_of(1).steps;
    let mut layers = 0;
    for (i, s) in steps.iter().enumerate() {
        let CudaStep::ExpertGateUpTcFp8(_) = s else {
            continue;
        };
        let j = (i + 1..steps.len())
            .find(|&j| matches!(steps[j], CudaStep::GateUpTcFp8(_)))
            .unwrap_or_else(|| panic!("m=1 step {i}: no shared gate/up after it"));
        assert_eq!(
            steps[j].quantized_input(),
            s.quantized_input(),
            "m=1 step {j}: the routed experts' row"
        );
        assert!(
            !(steps[i..j].iter()).any(|s| matches!(s, CudaStep::QuantRowsFp8(_))),
            "m=1 steps {i}..{j}: no quant write between"
        );
        layers += 1;
    }
    assert_eq!(layers, 40, "m=1: every MoE layer");
}
