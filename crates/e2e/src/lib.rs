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
/// Constants like `SMOLLM` resolve to the right model for the active backend:
/// MLX 4-bit on `metal`, safetensors BF16 on `cuda`. Use these in tests that
/// should run on **both** backends.
///
/// Backend-specific constants (`*_4BIT`, `*_CUDA`, etc.) are still available
/// for tests that target a single backend.
pub struct TestModels;

impl TestModels {
    // -----------------------------------------------------------------------
    // Backend-portable models — use these in cross-platform tests
    // -----------------------------------------------------------------------

    #[cfg(feature = "metal")]
    pub const SMOLLM: &str = "mlx-community/SmolLM-135M-Instruct-4bit";
    #[cfg(feature = "cuda")]
    pub const SMOLLM: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";

    #[cfg(feature = "metal")]
    pub const QWEN2: &str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit";
    #[cfg(feature = "cuda")]
    pub const QWEN2: &str = "Qwen/Qwen2.5-0.5B";

    #[cfg(feature = "metal")]
    pub const QWEN3: &str = "mlx-community/Qwen3-0.6B-4bit";
    #[cfg(feature = "cuda")]
    pub const QWEN3: &str = "Qwen/Qwen3-0.6B";

    #[cfg(feature = "metal")]
    pub const GEMMA2: &str = "mlx-community/gemma-2-2b-it-4bit";
    #[cfg(feature = "cuda")]
    pub const GEMMA2: &str = "unsloth/gemma-2-2b-it";

    #[cfg(feature = "metal")]
    pub const DEEPSEEK_V2_LITE: &str = "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx";
    #[cfg(feature = "cuda")]
    pub const DEEPSEEK_V2_LITE: &str = "deepseek-ai/DeepSeek-V2-Lite";

    #[cfg(feature = "metal")]
    pub const GRANITE: &str = "mlx-community/granite-3.3-2b-instruct-4bit";
    #[cfg(feature = "cuda")]
    pub const GRANITE: &str = "ibm-granite/granite-3.3-2b-instruct";

    #[cfg(feature = "metal")]
    pub const LLAMA_3_2: &str = "mlx-community/Llama-3.2-1B-Instruct-4bit";
    #[cfg(feature = "cuda")]
    pub const LLAMA_3_2: &str = "unsloth/Llama-3.2-1B-Instruct";

    #[cfg(feature = "metal")]
    pub const GEMMA3: &str = "mlx-community/gemma-3-270m-it-qat-4bit";
    #[cfg(feature = "cuda")]
    pub const GEMMA3: &str = "unsloth/gemma-3-270m-it";

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
    #[cfg(feature = "cuda")]
    pub const PHI3_5: &str = "unsloth/Phi-3.5-mini-instruct";

    // Phi-3-mini-4k-instruct — `Phi3ForCausalLM`, dense bf16, MHA
    // (32 q-heads == 32 kv-heads), no LongRoPE (`rope_scaling: null`,
    // max_position_embeddings=4096). First arch with packed
    // on-disk weights (`self_attn.qkv_proj.weight` + `mlp.gate_up_proj.weight`);
    // the loader slices them back into the five logical tensors the
    // DSL body references.
    #[cfg(feature = "cuda")]
    pub const PHI3_MINI_4K_CUDA: &str = "microsoft/Phi-3-mini-4k-instruct";

    // Phi-3-medium-4k-instruct — `Phi3ForCausalLM`, dense bf16, GQA
    // (40 q-heads, 10 kv-heads, head_dim=128, hidden=5120,
    // intermediate=17920, 40 layers). `rope_scaling: null`,
    // max_position_embeddings=4096 — no LongRoPE. On-disk packed
    // `self_attn.qkv_proj.weight` (`[5120+2*1280, 5120]`) is a GQA
    // split, which the manifest-driven `__packed_splits__` prelude
    // handles by emitting per-slice row counts from the manifest.
    #[cfg(feature = "cuda")]
    pub const PHI3_MEDIUM_4K_CUDA: &str = "microsoft/Phi-3-medium-4k-instruct";

    // Phi-3.5-mini-instruct — `Phi3ForCausalLM`, dense bf16, MHA
    // (same shapes as Phi-3-mini-4k: hidden=3072, heads=32, kv=32,
    // head_dim=96, 32 layers, intermediate=8192) BUT
    // max_position_embeddings=131072 with LongRoPE (su-scaling):
    // `rope_scaling: { type: "longrope", short_factor: [48 floats],
    // long_factor: [48 floats], original_max_position_embeddings=4096 }`.
    // Points at microsoft's official repo — the unsloth mirror that
    // the pre-existing `PHI3_5` const uses has been re-exported as
    // `LlamaForCausalLM` with LongRoPE stripped, which is useless for
    // testing this path.
    #[cfg(feature = "cuda")]
    pub const PHI3_5_MINI_CUDA: &str = "microsoft/Phi-3.5-mini-instruct";

    #[cfg(feature = "metal")]
    pub const PHI4: &str = "mlx-community/Unsloth-Phi-4-mini-instruct-4bit";
    #[cfg(feature = "cuda")]
    pub const PHI4: &str = "unsloth/Phi-4-mini-instruct";

    // Phi-4-mini-instruct — `Phi3ForCausalLM`, dense bf16, GQA
    // (hidden=3072, heads=24, kv=8, head_dim=128, 32 layers,
    // intermediate=8192), `partial_rotary_factor=0.75` ⇒ rotary_dim=96
    // with LongRoPE (su-scaling; factor vectors are length
    // rotary_dim/2 = 48, not head_dim/2 = 64). `tie_word_embeddings=true`
    // (first in the Phi-3 family). Points at microsoft's official
    // repo to mirror PHI3_MINI_4K_CUDA / PHI3_5_MINI_CUDA; `PHI4`
    // above points at unsloth (still Phi3ForCausalLM + longrope +
    // partial_rotary, verified).
    #[cfg(feature = "cuda")]
    pub const PHI4_MINI_CUDA: &str = "microsoft/Phi-4-mini-instruct";

    // Phi-3-mini-128k-instruct — same shapes as mini-4k (MHA, no
    // partial rotary) but max_pos=131072 with LongRoPE. Config-drop
    // on top of the phi3.5-mini-style rotary path.
    #[cfg(feature = "cuda")]
    pub const PHI3_MINI_128K_CUDA: &str = "microsoft/Phi-3-mini-128k-instruct";

    // Phi-3-medium-128k-instruct — GQA (40 q, 10 kv, head_dim=128,
    // hidden=5120) + LongRoPE. 14B bf16; won't fit on L4, golden
    // must be generated on an A100 / H100.
    #[cfg(feature = "cuda")]
    pub const PHI3_MEDIUM_128K_CUDA: &str = "microsoft/Phi-3-medium-128k-instruct";

    // Phi-4 (the full 14B, not mini) — GQA 40/10, head_dim=128,
    // hidden=5120, no rope_scaling, no partial rotary. 14B bf16;
    // won't fit on L4.
    #[cfg(feature = "cuda")]
    pub const PHI4_FULL_CUDA: &str = "microsoft/Phi-4";

    // Phi-4-reasoning — same shape as Phi-4 but
    // `partial_rotary_factor=1.0` (normalized to full rotary in the
    // codegen) and max_pos=32768. 14B bf16; A100-class required.
    #[cfg(feature = "cuda")]
    pub const PHI4_REASONING_CUDA: &str = "microsoft/Phi-4-reasoning";

    // Phi-4-reasoning-plus — same as reasoning (identical shapes +
    // config discriminators). Dedup may collapse the forward fn with
    // Phi-4-reasoning's.
    #[cfg(feature = "cuda")]
    pub const PHI4_REASONING_PLUS_CUDA: &str = "microsoft/Phi-4-reasoning-plus";

    // Phi-4-mini-reasoning — identical config shape to
    // Phi-4-mini-instruct (GQA 24/8, partial=0.75, longrope, tied
    // lm_head). Fits on L4.
    #[cfg(feature = "cuda")]
    pub const PHI4_MINI_REASONING_CUDA: &str = "microsoft/Phi-4-mini-reasoning";

    #[cfg(feature = "metal")]
    pub const MISTRAL: &str = "mlx-community/Mistral-7B-Instruct-v0.3-4bit";
    #[cfg(feature = "cuda")]
    pub const MISTRAL: &str = "unsloth/mistral-7b-instruct-v0.3";

    #[cfg(feature = "metal")]
    pub const GEMMA3_VLM: &str = "mlx-community/gemma-3-4b-it-qat-3bit";
    #[cfg(feature = "cuda")]
    pub const GEMMA3_VLM: &str = "unsloth/gemma-3-4b-it";

    #[cfg(feature = "metal")]
    pub const QWEN3_MOE: &str =
        "justneedsomeavailableusername/Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder-V1.2-mlx-4Bit";
    #[cfg(feature = "cuda")]
    pub const QWEN3_MOE: &str = "TroyDoesAI/Qwen3-MoE-3B";

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
    // GPTQ / AWQ / BNB / GGUF quantized models
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

    // GGUF quantized models (CPU, not MLX)
    pub const GEMMA3_270M_GGUF: &str = "unsloth/gemma-3-270m-it-qat-GGUF";
    pub const GEMMA3_1B_GGUF: &str = "unsloth/gemma-3-1b-it-GGUF";
    pub const QWEN2_0_5B_GGUF: &str = "Qwen/Qwen2.5-0.5B-Instruct-GGUF";
    pub const QWEN3_0_6B_GGUF: &str = "unsloth/Qwen3-0.6B-GGUF";
    pub const QWEN3_NEXT_0_8B_GGUF: &str = "unsloth/Qwen3.5-0.8B-GGUF";
    pub const LLAMA_3_2_1B_IQ1M_GGUF: &str = "unsloth/Llama-3.2-1B-Instruct-GGUF";
    pub const LLAMA_3_2_1B_IQ1M_FILE: &str = "Llama-3.2-1B-Instruct-UD-IQ1_M.gguf";
    // Q4_K_M variant of the same unsloth repo; file co-located with IQ1_M above.
    pub const LLAMA_3_2_1B_Q4KM_GGUF: &str = "unsloth/Llama-3.2-1B-Instruct-GGUF";
    pub const LLAMA_3_2_1B_Q4KM_FILE: &str = "Llama-3.2-1B-Instruct-Q4_K_M.gguf";
    // Remaining IQ quants of the same unsloth repo. Together with IQ1_M above,
    // these cover 5 of the 7 IQ types wired in `GgmlDType::from_gguf`.
    // IQ2_S and IQ3_S aren't shipped by unsloth — tested separately on Mistral.
    pub const LLAMA_3_2_1B_IQ1S_FILE: &str = "Llama-3.2-1B-Instruct-UD-IQ1_S.gguf";
    pub const LLAMA_3_2_1B_IQ2XXS_FILE: &str = "Llama-3.2-1B-Instruct-UD-IQ2_XXS.gguf";
    pub const LLAMA_3_2_1B_IQ4NL_FILE: &str = "Llama-3.2-1B-Instruct-IQ4_NL.gguf";
    pub const LLAMA_3_2_1B_IQ4XS_FILE: &str = "Llama-3.2-1B-Instruct-IQ4_XS.gguf";
    // Mistral-7B-Instruct-v0.3 — only bartowski repo found to ship IQ2_S / IQ3_S
    // as a Llama/Mistral/Qwen2-compatible GGUF. 7B is larger than ideal but
    // these two IQ types aren't available at 1B scale anywhere we checked.
    pub const MISTRAL_7B_V03_GGUF: &str = "bartowski/Mistral-7B-Instruct-v0.3-GGUF";
    pub const MISTRAL_7B_V03_IQ2S_FILE: &str = "Mistral-7B-Instruct-v0.3-IQ2_S.gguf";
    pub const MISTRAL_7B_V03_IQ3S_FILE: &str = "Mistral-7B-Instruct-v0.3-IQ3_S.gguf";

    // -----------------------------------------------------------------------
    // CUDA-only models (safetensors BF16)
    // -----------------------------------------------------------------------

    pub const SMOLLM_135M_CUDA: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";
    pub const QWEN2_0_5B_CUDA: &str = "Qwen/Qwen2.5-0.5B";
    // CommandR (CohereForCausalLM) — single-layer trim of the real
    // `CohereForAI/c4ai-command-r-v01` 35B checkpoint by Citaman
    // (mergekit slice). Full v01 dims preserved: hidden_size=8192,
    // head_dim=128, 64 q-heads / 64 kv-heads (no GQA), no QK norm,
    // rope_theta=8e6, vocab_size=256000, logit_scale=0.0625, tied
    // embeddings. ~5GB bf16 — fits on a single L4. Trained weights →
    // logprobs have real signal (not uniform noise), so the token-
    // equivalence test catches actual math bugs. The full 35B doesn't
    // fit on L4 and Cohere's smaller official checkpoints are gated.
    pub const COMMAND_R_1L_CUDA: &str = "Citaman/command-r-1-layer";
    // MoE models for CUDA — safetensors BF16.
    //
    // Qwen2-MoE / Qwen1.5-MoE A2.7B-Chat slimmed to 2 layers
    // (`Qwen2MoeForCausalLM`, BF16/FP16, full hidden=2048, 60 routed
    // experts × top-4, shared expert intermediate=5632, ~3GB FP16).
    // The single coherent L4-fitting Qwen2-MoE fixture: real trained
    // weights from Qwen1.5-MoE-A2.7B-Chat with all but 2 decoder
    // layers pruned (mergekit-style trim, mirrors the
    // `Citaman/command-r-1-layer` pattern). Used to exercise the
    // `SharedFusedMoELayer::load` + forward path with
    // `shared_expert_intermediate_size > 0` (Qwen3-MoE-Instruct
    // ships shared_inter=0 fleet-wide; this is the only available
    // path for the routed+shared MoE branch). Ships no tokenizer —
    // see `ensure_slimed_qwen_tokenizer` in e_correctness for
    // borrowing Qwen1.5-MoE-A2.7B-Chat's vocab files.
    pub const QWEN2_MOE_SLIMED_CUDA: &str = "JacobAndersson/slimed-qwen-3";
    // Mixtral 8x248M DPO-tuned — `MixtralForCausalLM` (BF16, 8 experts,
    // top-2, 12 layers, hidden=1024, intermediate=4096, ~2B total
    // params, ~4GB BF16). Real DPO-tuned chat fine-tune (oasst2 +
    // Intel orca DPO pairs) → produces coherent English on simple
    // prompts. The Mixtral-arch checkpoint that fits L4 AND produces
    // coherent output AND ships its own tokenizer. (An earlier
    // `if001/small_mixtral_ja_llm_jp_tk` 0.8B checkpoint was used
    // until 2026-05-04; dropped because it ships FP32 weights with no
    // tokenizer and its near-uniform output flips argmax across
    // independent server processes — unsuitable for golden parity.)
    pub const MIXTRAL_TINY_DPO_CUDA: &str =
        "NickyNicky/Mixtral-TinyMistral-8x248M-Instruct_oasst2_chatML_Intel_orca_dpo_pairs_DPO_V1";
    // Qwen2 MoE: ~14.3B total params (~29GB BF16), Qwen2MoeForCausalLM — fits on L40S (48GB)
    pub const QWEN2_MOE_A2_7B_CUDA: &str = "Qwen/Qwen1.5-MoE-A2.7B-Chat";

    // Qwen3 — safetensors BF16 for cuda-backend (~1.2GB)
    pub const QWEN3_0_6B_CUDA: &str = "Qwen/Qwen3-0.6B";

    // Qwen3.5-9B — Gated-DeltaNet hybrid text decoder (~19.3GB BF16, 4
    // safetensors shards). The full repo is `Qwen3_5ForConditionalGeneration`
    // (multimodal); the text path our `qwen3-5` arch
    // compiles exercises 32 layers (24 linear-attn / 8 full-attn,
    // 1-in-4 pattern), head_dim=256, partial_rotary_factor=0.25,
    // mrope_section=[11,11,10]. Fits on a single H100 (80GB).
    pub const QWEN3_5_9B_CUDA: &str = "Qwen/Qwen3.5-9B";

    // Gemma2 — safetensors BF16 for cuda-backend (~5GB, Gemma2ForCausalLM)
    pub const GEMMA2_2B_IT_CUDA: &str = "unsloth/gemma-2-2b-it";

    // Gemma3 — safetensors BF16 for cuda-backend (~2GB, Gemma3ForCausalLM)
    pub const GEMMA3_1B_IT_CUDA: &str = "unsloth/gemma-3-1b-it";
    // Gemma3 — safetensors BF16 for TP testing (~8GB, 8 kv_heads → TP=2 safe)
    pub const GEMMA3_4B_IT_CUDA: &str = "unsloth/gemma-3-4b-it";

    // DeepSeek V2 — safetensors BF16 for TP testing (2x L40S)
    pub const DEEPSEEK_V2_LITE_CUDA: &str = "deepseek-ai/DeepSeek-V2-Lite";
    // DeepSeek V3 — 4-layer synthetic tiny model (q_lora_rank path + sigmoid MoE routing).
    // Generated by scripts/make_tiny_deepseek_v3.py. Must pre-exist on the test host.
    pub const DEEPSEEK_V3_TINY_CUDA: &str = "/tmp/deepseek-v3-tiny";
    // DeepSeek V3 — ByteDance-Seed/academic-ds-9B: real trained 9B MoE model using full
    // V3 architecture (hidden=2048, heads=16, q_lora_rank=1024, 16 layers, 64 experts).
    // Trained from scratch on 350B+ English tokens; produces coherent output.
    pub const DEEPSEEK_V3_ACADEMIC_9B_CUDA: &str = "ByteDance-Seed/academic-ds-9B";
    // Moonlight-16B-A3B-Instruct — `DeepseekV3ForCausalLM`, 16B MoE BF16.
    // The only public DeepseekV3ForCausalLM checkpoint with K2-style flat routing
    // (q_lora_rank=null, n_group=1, topk_group=1, sigmoid+noaux_tc, routed_scaling_factor=2.446)
    // that fits a single H100 (32 GB BF16, 27 layers, 64 routed + 2 shared experts).
    // Routes through the `deepseek-v3-flat` arch (a sibling of deepseek-v3, direct q_proj DSL).
    pub const MOONLIGHT_16B_A3B_INSTRUCT_CUDA: &str = "moonshotai/Moonlight-16B-A3B-Instruct";

    // Granite (IBM) — MLX 4-bit quantized
    pub const GRANITE_3_3_2B_4BIT: &str = "mlx-community/granite-3.3-2b-instruct-4bit";
    // Granite (IBM) — safetensors BF16, CUDA
    pub const GRANITE_3_3_2B_INSTRUCT: &str = "ibm-granite/granite-3.3-2b-instruct";
    // Granite GGUF — quantized, CUDA
    pub const GRANITE_3_3_2B_INSTRUCT_GGUF: &str = "ibm-granite/granite-3.3-2b-instruct-GGUF";

    // FP8 quantized models (CUDA-backend, SM89+)
    pub const QWEN2_0_5B_FP8: &str = "RedHatAI/Qwen2.5-0.5B-FP8-dynamic";
    pub const LLAMA_3_1_8B_FP8: &str = "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8";
    // FP8 dynamic-per-tensor (W8A8, weights channel-strategy,
    // activations token-strategy dynamic) — the class covered by
    // Slice 1 of the scratchy FP8 rollout.
    pub const LLAMA_3_2_1B_FP8: &str = "RedHatAI/Llama-3.2-1B-Instruct-FP8-dynamic";
    pub const QWEN3_0_6B_FP8: &str = "RedHatAI/Qwen3-0.6B-FP8-dynamic";
    pub const GEMMA2_2B_FP8: &str = "espressor/google.gemma-2-2b-it_W8A8_FP8";
    pub const GEMMA3_1B_FP8: &str = "RedHatAI/gemma-3-1b-it-FP8-dynamic";
    pub const GRANITE_3_1_2B_FP8: &str = "RedHatAI/granite-3.1-2b-instruct-FP8-dynamic";
    pub const MISTRAL_7B_V03_FP8: &str = "nm-testing/Mistral-7B-Instruct-v0.3-FP8-Dynamic";

    // FP8 static-per-tensor (Slice 2) — pre-calibrated per-tensor
    // `input_scale` baked into the checkpoint. Scratchy's
    // `Fp8Linear::forward` branches on `input_scale` presence to
    // select the static CUTLASS epilogue. Four arches covered by
    // available HF repos; others (qwen3-0.6b, gemma3-1b, granite-3.1-2b)
    // lack small-size static-FP8 checkpoints upstream.
    pub const QWEN2_1_5B_FP8_STATIC: &str = "RedHatAI/Qwen2-1.5B-Instruct-FP8";
    pub const LLAMA_3_2_1B_FP8_STATIC: &str = "RedHatAI/Llama-3.2-1B-Instruct-FP8";
    pub const GEMMA2_2B_FP8_STATIC: &str = "RedHatAI/gemma-2-2b-it-FP8";
    pub const MISTRAL_7B_V03_FP8_STATIC: &str = "RedHatAI/Mistral-7B-Instruct-v0.3-FP8";

    // FP8 blockwise-per-128×128 (Slice 3) — DeepSeek-V3-style block
    // quantization: weights FP8 `[N, K]` with 2-D scale tensor
    // `[ceil(N/128), ceil(K/128)]`. Runtime path uses
    // `Fp8BlockLinear::forward` (dequant to BF16 then cuBLAS GEMM —
    // a native block-scaled FP8 GEMM kernel is a perf follow-up).
    pub const QWEN3_0_6B_FP8_BLOCK: &str = "RedHatAI/Qwen3-0.6B-FP8-BLOCK";
    // DeepSeek V3 academic-9B re-quantized to FP8-block-128×128 (the
    // canonical V3/K2 storage layout). Same MLA topology as
    // `DEEPSEEK_V3_ACADEMIC_9B_CUDA` BF16 with `q_a_proj`/`q_b_proj`/
    // `kv_a_proj_with_mqa`/`kv_b_proj`/`o_proj` carrying FP8 E4M3
    // weights + 2-D `[N/128, K/128]` scales, and the 64-routed +
    // 2-shared MoE storing block-scaled experts. Routes through
    // `Fp8GemmImpl` (dense Linears) + `DeepSeekFp8BlockMoeImpl` (MoE).
    pub const DEEPSEEK_V3_ACADEMIC_9B_FP8_BLOCK_CUDA: &str = "starpit/academic-ds-9b-fp8-block";

    // Moonlight-16B-A3B-Instruct re-quantized to FP8-block-128×128.
    // Flat-Q variant (q_lora_rank=null → direct q_proj, no q_a/q_b split),
    // K2-style sigmoid+noaux_tc routing, routed_scaling_factor=2.446.
    // Routes through the `deepseek-v3-flat` arch + `DeepSeekFp8BlockMoeImpl`.
    // Upload with: huggingface-cli upload starpit/moonlight-16b-a3b-instruct-fp8-block <dir> .
    #[cfg(feature = "cuda")]
    pub const MOONLIGHT_16B_A3B_INSTRUCT_FP8_BLOCK_CUDA: &str =
        "starpit/moonlight-16b-a3b-instruct-fp8-block";

    // FP8 MoE models (CUDA-backend, SM89+)
    // 2-layer Mixtral 8x7B FP8 (~3GB) — small enough for single L40S
    pub const MIXTRAL_8X7B_FP8_2L: &str = "fxmarty/Mixtral-8x7B-Instruct-v0.1-FP8-KV-2-layers";

    // Multimodal (vision-language) models
    // Tier 3: ~2.8 GB — nightly only (QAT = quantization-aware training)
    pub const GEMMA3_4B_IT_QAT_3BIT: &str = "mlx-community/gemma-3-4b-it-qat-3bit";
    // Tier 4: ~8 GB BF16 SafeTensors — Gemma3ForConditionalGeneration (Candle path)
    pub const GEMMA3_4B_IT: &str = "google/gemma-3-4b-it";

    // ModernBERT — encoder-only (bidirectional attention, RoPE, GeGLU)
    // ~430 MB safetensors, hidden_size=768, 22 layers
    pub const MODERNBERT_BASE: &str = "answerdotai/ModernBERT-base";

    // Qwen2-VL multimodal (vision-language) models
    // Tier 3: ~4.4 GB BF16 SafeTensors — qwen2 path. The
    // unsloth single-file mirror does not match the loader's shard probe
    // and falls through to the cuda-backend match (no Qwen2-VL arm).
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
