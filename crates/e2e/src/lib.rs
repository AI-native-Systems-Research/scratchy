// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! End-to-end test infrastructure for the vLLM Rust inference engine.
//!
//! Provides [`TestServer`] for starting the server in-process,
//! [`Client`] for sending HTTP requests, and assertion helpers for
//! validating OpenAI-compatible responses.

pub mod assertions;
pub mod client;
pub mod server;

pub use client::Client;
pub use server::TestServer;

/// Whether the GPU backend the e2e suite needs is actually usable on this host.
///
/// On `metal` builds this requires a Metal 4 device: GitHub's hosted macOS
/// runners expose a virtualized "Apple Paravirtual device" that has no Metal 4
/// compiler, so the metal worker cannot initialize and every model-booting test
/// would fail at startup. GPU-only tests call [`skip_if_no_gpu`] (or check this
/// directly) and return early there. Non-metal builds always return `true`.
#[must_use]
pub fn gpu_backend_available() -> bool {
    #[cfg(feature = "metal")]
    {
        scratchy_target_metal::metal4_available()
    }
    #[cfg(not(feature = "metal"))]
    {
        true
    }
}

/// Early-return a test with a skip message when [`gpu_backend_available`] is
/// false. Use at the top of any e2e test that boots a real GPU worker.
#[macro_export]
macro_rules! skip_if_no_gpu {
    () => {
        if !$crate::gpu_backend_available() {
            eprintln!("skipping: no usable GPU backend (e.g. no Metal 4 on hosted CI runner)");
            return;
        }
    };
}

/// Test models by architecture (smallest available for CI).
///
/// # Backend-portable constants
///
/// Constants like `SMOLLM` resolve to the right model for the active backend
/// (MLX 4-bit on `metal`).
///
/// Backend-specific constants (`*_4BIT`, etc.) are still available for tests
/// that target a single backend.
pub struct TestModels;

impl TestModels {
    // -----------------------------------------------------------------------
    // Backend-portable models — use these in cross-platform tests
    // -----------------------------------------------------------------------

    #[cfg(feature = "metal")]
    pub const SMOLLM: &str = "mlx-community/SmolLM-135M-Instruct-4bit";

    #[cfg(feature = "metal")]
    pub const QWEN2: &str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit";

    #[cfg(feature = "metal")]
    pub const QWEN3: &str = "mlx-community/Qwen3-0.6B-4bit";

    #[cfg(feature = "metal")]
    pub const GEMMA2: &str = "mlx-community/gemma-2-2b-it-4bit";

    #[cfg(feature = "metal")]
    pub const DEEPSEEK_V2_LITE: &str = "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx";

    #[cfg(feature = "metal")]
    pub const GRANITE: &str = "mlx-community/granite-3.3-2b-instruct-4bit";

    #[cfg(feature = "metal")]
    pub const LLAMA_3_2: &str = "mlx-community/Llama-3.2-1B-Instruct-4bit";

    #[cfg(feature = "metal")]
    pub const GEMMA3: &str = "mlx-community/gemma-3-270m-it-qat-4bit";

    // Gemma4 — the launch-claude target. Two layer classes: sliding (hd256,
    // 8 kv-heads, window 1024, steel prefill) + global (hd512, proportional
    // rope, gqa_shared prefill — head_dim 512 has no steel instantiation).
    // GEMMA4 = the 12B dense variant (6.3 GB on disk, num_global_kv_heads=1,
    // gqa=16). GEMMA4_MOE = the 26B-a4b MoE variant (15 GB, the SWA-validated
    // checkpoint). Both exercise the gqa_shared global-prefill rope-once path.
    #[cfg(feature = "metal")]
    pub const GEMMA4: &str = "mlx-community/gemma-4-12B-it-4bit";
    #[cfg(feature = "metal")]
    pub const GEMMA4_MOE: &str = "mlx-community/gemma-4-26b-a4b-it-4bit";

    #[cfg(feature = "metal")]
    pub const PHI3_5: &str = "mlx-community/Phi-3.5-mini-instruct-4bit";

    #[cfg(feature = "metal")]
    pub const PHI4: &str = "mlx-community/Unsloth-Phi-4-mini-instruct-4bit";

    #[cfg(feature = "metal")]
    pub const MISTRAL: &str = "mlx-community/Mistral-7B-Instruct-v0.3-4bit";

    #[cfg(feature = "metal")]
    pub const GEMMA3_VLM: &str = "mlx-community/gemma-3-4b-it-qat-3bit";

    #[cfg(feature = "metal")]
    pub const QWEN3_MOE: &str =
        "justneedsomeavailableusername/Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder-V1.2-mlx-4Bit";

    // -----------------------------------------------------------------------
    // MLX-only models (metal backend)
    // -----------------------------------------------------------------------

    // Tier 1: Tiny (<500 MB) — run on every PR
    pub const SMOLLM_135M_4BIT: &str = "mlx-community/SmolLM-135M-Instruct-4bit";
    pub const QWEN2_0_5B_4BIT: &str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit";
    pub const QWEN3_0_6B_4BIT: &str = "mlx-community/Qwen3-0.6B-4bit";

    // Tier 2: Small (<1 GB) — run on every PR
    pub const LLAMA_3_2_1B_4BIT: &str = "mlx-community/Llama-3.2-1B-Instruct-4bit";

    // Tier 2: Small (<1 GB) — run on every PR
    pub const GEMMA3_270M_4BIT: &str = "mlx-community/gemma-3-270m-it-qat-4bit";

    // Tier 3: Medium (1–3 GB) — nightly only
    pub const GEMMA2_2B_4BIT: &str = "mlx-community/gemma-2-2b-it-4bit";
    pub const PHI3_5_MINI_4BIT: &str = "mlx-community/Phi-3.5-mini-instruct-4bit";
    pub const PHI4_MINI_4BIT: &str = "mlx-community/Unsloth-Phi-4-mini-instruct-4bit";

    // Tier 4: Large (3+ GB) — weekly/manual only
    pub const MISTRAL_7B_4BIT: &str = "mlx-community/Mistral-7B-Instruct-v0.3-4bit";
    pub const DEEPSEEK_V2_LITE_4BIT: &str =
        "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx";

    // MoE models
    pub const QWEN3_MOE_4X06B_4BIT: &str =
        "justneedsomeavailableusername/Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder-V1.2-mlx-4Bit";

    // Float16 variants for non-quantized testing
    pub const SMOLLM_135M_F16: &str = "mlx-community/SmolLM2-135M-Instruct";

    // -----------------------------------------------------------------------
    // GPTQ / AWQ / BNB quantized models
    // -----------------------------------------------------------------------

    // GPTQ quantized models (CPU, not MLX)
    pub const QWEN2_0_5B_GPTQ_INT4: &str = "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4";

    // AWQ quantized models (CPU, not MLX)
    pub const QWEN2_0_5B_AWQ: &str = "Qwen/Qwen2.5-0.5B-Instruct-AWQ";
    // AWQ Llama-3.2-1B — exercises scratchy-forward-compiler's MarlinGemm +
    // MarlinFusedQkvRope* + MarlinFusedGateUpSiluMul impls. The repo's
    // `quantization_config` matches the `llama-3.2-1b` + `awq-gemm`
    // overlay variant synthesized from `crates/models/arch/configs/llama/`.
    pub const LLAMA_3_2_1B_AWQ: &str = "AMead10/Llama-3.2-1B-Instruct-AWQ";

    // Gemma2 GPTQ quantized models (ungated)
    pub const GEMMA2_2B_GPTQ_INT4: &str = "qilowoq/gemma-2-2B-it-4Bit-GPTQ";
    // Gemma2 AWQ — dolphin fine-tune (instruct-formatted) of gemma-2-2b.
    // solidrust's repo materializes both `embed_tokens.weight` and
    // `lm_head.weight` explicitly, unlike RichardErkhov's
    // `google_-_gemma-2-2b-it-awq` which ships only `lm_head.weight`
    // and trips the fingerprint's `embed_tokens.weight` gate. Covers
    // the AWQ × Gemma2 cell (alt sliding/full attention + softcap +
    // GELU MLP) that parity.csv lists as supported but had no e2e
    // coverage before.
    pub const GEMMA2_2B_AWQ: &str = "solidrust/dolphin-2.9.4-gemma2-2b-AWQ";

    // GPTQ with desc_act (activation ordering) — tests g_idx sort + perm pipeline
    pub const TINYLLAMA_1B_GPTQ_DESC_ACT: &str = "TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ";
    // Compressed-tensors INT4 — Neural Magic / RedHatAI pack-quantized
    // layout. Same uint4b8 bits as GPTQ; loader sniffs `.weight_packed`
    // + `.weight_scale` and transposes before the shared Marlin repack.
    pub const TINYLLAMA_1B_W4A16_CT: &str = "nm-testing/TinyLlama-1.1B-Chat-v1.0-W4A16-e2e";

    // BitsAndBytes quantized models (MLX dequant-at-load or CPU)
    pub const LLAMA_3_2_1B_BNB_4BIT: &str = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit";
    pub const TINYLLAMA_1B_BNB_8BIT: &str = "Jiqing/TinyLlama-1.1B-Chat-v1.0-bnb-8bit";
    // BNB NF4 (4-bit, double-quant) — exercises scratchy-forward-compiler's
    // Bnb4bitLinear::load{,_concat} + Bnb4GemmImpl / fused gate-up
    // silu / fused QKV-rope path on Qwen3 (QK-norm variant). Unsloth
    // repo matches parity.csv's verified BNB4 checkpoint.
    pub const QWEN3_0_6B_BNB_4BIT: &str = "unsloth/Qwen3-0.6B-bnb-4bit";
    pub const QWEN2_0_5B_BNB_4BIT: &str = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit";
    pub const GEMMA2_2B_W4A16_CT: &str = "RedHatAI/gemma-2-2b-it-quantized.w4a16";
    pub const GRANITE_3_1_2B_GPTQ: &str = "sroecker/granite-3.1-2b-instruct-gptq";
    pub const GRANITE_3_2B_BNB_4BIT: &str = "unsloth/granite-3.2-2b-instruct-bnb-4bit";

    // Granite (IBM) — MLX 4-bit quantized
    pub const GRANITE_3_3_2B_4BIT: &str = "mlx-community/granite-3.3-2b-instruct-4bit";
    // Multimodal (vision-language) models
    // Tier 3: ~2.8 GB — nightly only (QAT = quantization-aware training)
    pub const GEMMA3_4B_IT_QAT_3BIT: &str = "mlx-community/gemma-3-4b-it-qat-3bit";
    // Tier 4: ~8 GB BF16 SafeTensors — Gemma3ForConditionalGeneration (Candle path)
    pub const GEMMA3_4B_IT: &str = "google/gemma-3-4b-it";

    // ModernBERT — encoder-only (bidirectional attention, RoPE, GeGLU)
    // ~430 MB safetensors, hidden_size=768, 22 layers
    pub const MODERNBERT_BASE: &str = "answerdotai/ModernBERT-base";

    // Qwen2-VL multimodal (vision-language) models
    // Tier 3: ~4.4 GB BF16 SafeTensors — qwen2 path.
    pub const QWEN2_VL_2B_INSTRUCT: &str = "Qwen/Qwen2-VL-2B-Instruct";
    // Tier 4: ~4.6 GB 4-bit quantized — MLX path
    pub const QWEN2_VL_7B_4BIT: &str = "mlx-community/Qwen2-VL-7B-4bit";

    // Qwen2.5-VL multimodal — pure-DSL qwen2-5-vl arch.
    // Vision tower diffs from Qwen2-VL: RMSNorm vision blocks, SwiGLU
    // MLP (vs QuickGELU FC1/FC2), per-layer attention switching
    // (full at indices [7,15,23,31], windowed otherwise), entry-side
    // window permutation gather + reverse permutation post-merger.
    pub const QWEN2_5_VL_3B_INSTRUCT: &str = "Qwen/Qwen2.5-VL-3B-Instruct";
}
